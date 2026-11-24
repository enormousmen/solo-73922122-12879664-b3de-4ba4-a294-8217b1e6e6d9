"""模型服务：单作业内存状态、训练/预测/扩列/保存/重载、逐批验证。

进程内只保留一个活动模型（单作业）。训练、扩列、保存、重载
通过同一把异步锁串行化；预测与验证集维护不与锁互斥
（预测只读取活动 bundle；验证集只做不可变快照替换），
因此试训/验证进行中仍可预测、也可替换验证集。

逐批验证语义：
- 验证集未设置时 `/train` 照常直接续训一步；
- 已设置时先在深拷贝上试训一步，用试训前后的模型给同一组
  验证读数分类：只要有样本从“判对”变为“判错”，整批拒绝，
  原模型与 Adam 状态保持不变；
- 试训开始后验证集只要被设置/追加/清空过（revision 变化，
  即使换回原内容），该批报冲突并丢弃试训结果；
- 只有验证通过且验证集未变，试训后的模型与 Adam 状态
  才作为整体一次性生效。
"""
from __future__ import annotations

import asyncio
import copy
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from .checkpoint import (
    CheckpointBundle,
    expand_bundle,
    load_checkpoint,
    save_checkpoint_atomic,
)
from .data import align_labels, align_rows

TRAIN_APPLIED = "applied"
TRAIN_REJECTED = "rejected"
TRAIN_CONFLICT = "conflict"


@dataclass(frozen=True)
class ValidationSet:
    """不可变验证集快照：revision 每次设置/追加/清空都会递增。"""

    revision: int
    items: tuple[tuple[dict[str, float], int], ...] = field(default_factory=tuple)

    def as_payload(self) -> list[dict[str, Any]]:
        return [{"reading": dict(reading), "label": label} for reading, label in self.items]


class ModelService:
    def __init__(self, bundle: CheckpointBundle):
        self._bundle = bundle
        self._lock = asyncio.Lock()
        self._validation_lock = asyncio.Lock()
        self._validation: ValidationSet | None = None
        self._validation_revision = 0

    # ---- 基础信息 -------------------------------------------------
    @property
    def bundle(self) -> CheckpointBundle:
        return self._bundle

    @property
    def lock(self) -> asyncio.Lock:
        return self._lock

    def info(self) -> dict:
        b = self._bundle
        return {
            "feature_names": list(b.feature_names),
            "input_dim": len(b.feature_names),
            "hidden_dim": b.hidden_dim,
            "num_classes": b.num_classes,
            "dtype": str(b.dtype).replace("torch.", ""),
            "optimizer_step": _read_step(b.optimizer),
            "source_path": b.source_path,
        }

    # ---- 验证集（本次运行内生效，不随检查点保存） ------------------
    def validation_state(self) -> dict:
        v = self._validation
        return {
            "configured": v is not None,
            "revision": self._validation_revision,
            "count": len(v.items) if v else 0,
            "items": v.as_payload() if v else [],
        }

    def _snapshot(self) -> ValidationSet | None:
        return self._validation

    async def set_validation(self, items: list[dict]) -> dict:
        """整体设置或替换验证集（逐条提交也可用 append 接口）。"""
        async with self._validation_lock:
            self._validation = ValidationSet(
                revision=self._validation_revision + 1,
                items=self._normalize_validation_items(items),
            )
            self._validation_revision += 1
            return self.validation_state()

    async def append_validation(self, items: list[dict]) -> dict:
        """逐条追加拆检读数与确认类别。"""
        async with self._validation_lock:
            existing = list(self._validation.items) if self._validation else []
            merged = existing + list(self._normalize_validation_items(items))
            self._validation_revision += 1
            self._validation = ValidationSet(
                revision=self._validation_revision, items=tuple(merged)
            )
            return self.validation_state()

    async def clear_validation(self) -> dict:
        """清空验证集：之后 `/train` 恢复为照常续训。"""
        async with self._validation_lock:
            self._validation_revision += 1
            self._validation = None
            return self.validation_state()

    def _normalize_validation_items(
        self, items: list[dict]
    ) -> tuple[tuple[dict[str, float], int], ...]:
        num_classes = self._bundle.num_classes
        normalized: list[tuple[dict[str, float], int]] = []
        for i, item in enumerate(items):
            try:
                reading = item["reading"]
                label = item["label"]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"第 {i} 条验证样本需包含 reading 与 label") from exc
            if not isinstance(reading, dict) or not reading:
                raise ValueError(f"第 {i} 条验证样本的 reading 必须是非空对象")
            reading_copy = {str(k): float(v) for k, v in reading.items()}
            if not isinstance(label, int) or isinstance(label, bool):
                raise ValueError(f"第 {i} 条验证样本的 label 必须是整数")
            if label < 0 or label >= num_classes:
                raise ValueError(
                    f"第 {i} 条验证样本的 label={label} 超出 [0, {num_classes - 1}]"
                )
            normalized.append((reading_copy, label))
        return tuple(normalized)

    # ---- 预测 -----------------------------------------------------
    @torch.no_grad()
    def predict(self, instances: list[dict]) -> list[dict]:
        b = self._bundle
        x, missing = align_rows(instances, b.feature_names, b.dtype)
        logits = b.model(x)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
        return [
            {
                "label": int(labels[i]),
                "score": float(probs[i, labels[i]]),
                "probabilities": [float(p) for p in probs[i]],
                "zero_filled_columns": missing[i],
            }
            for i in range(len(instances))
        ]

    # ---- 训练（有验证集时先试训，再决定是否整批生效） --------------
    async def train_step(
        self, instances: list[dict], labels: list[int]
    ) -> tuple[dict, int]:
        """返回 (响应体, HTTP 状态码)。

        - 200 已生效；409 被拒（验证集回归）或冲突（试训期间验证集被替换）。
        """
        if len(instances) != len(labels):
            raise ValueError(
                f"instances 数量({len(instances)})与 labels 数量({len(labels)})不一致"
            )
        b = self._bundle
        # 先按活动模型做输入校验，避免无谓的深拷贝
        align_rows(instances, b.feature_names, b.dtype)
        align_labels(labels, b.num_classes)

        validation = self._snapshot()
        if validation is None:
            # 未设置验证集：照常续训一步
            result = _train_once(self._bundle, instances, labels)
            return {
                "status": TRAIN_APPLIED,
                "applied": True,
                "step": result["step"],
                "trial_step": result["step"],
                "loss_before": result["loss_before"],
                "loss_after": result["loss_after"],
                "num_samples": result["num_samples"],
                "feature_names": result["feature_names"],
                "validation": None,
            }, 200

        start_revision = validation.revision
        # 深拷贝放到工作线程：既让 /predict 在复制期间不被阻塞，
        # 又保证试训起点就是当前活动状态（train 持锁，bundle 不会被替换）。
        trial = await asyncio.to_thread(copy.deepcopy, self._bundle)
        outcome = await asyncio.to_thread(
            _run_trial, trial, validation, instances, labels
        )

        # 校验 revision 与提交之间没有 await，二者对验证集替换是原子的
        if self._validation_revision != start_revision:
            return _unchanged_body(TRAIN_CONFLICT, self._bundle, instances, outcome), 409

        if outcome["report"]["regressed_indices"]:
            return _unchanged_body(TRAIN_REJECTED, self._bundle, instances, outcome), 409

        # 验证通过：模型与 Adam 状态作为同一 bundle 一次性替换生效
        self._bundle = trial
        return {
            "status": TRAIN_APPLIED,
            "applied": True,
            "step": outcome["step"],
            "trial_step": outcome["step"],
            "loss_before": outcome["loss_before"],
            "loss_after": outcome["loss_after"],
            "num_samples": len(instances),
            "feature_names": list(trial.feature_names),
            "validation": outcome["report"],
        }, 200

    # ---- 扩列 -----------------------------------------------------
    def expand(self, new_feature_names: list[str]) -> dict:
        names = list(dict.fromkeys(new_feature_names))  # 去重但保序
        if len(names) != len(new_feature_names):
            raise ValueError("new_feature_names 内部存在重复列名")
        new_bundle = expand_bundle(self._bundle, names)
        self._bundle = new_bundle  # 整体替换，旧引用若正在预测仍可安全读完
        return {
            "feature_names": list(new_bundle.feature_names),
            "input_dim": len(new_bundle.feature_names),
            "added": names,
        }

    # ---- 保存（原子） ---------------------------------------------
    def save(self, path: str | None) -> str:
        target = os.path.abspath(path) if path else self._bundle.source_path
        save_checkpoint_atomic(self._bundle, target)
        return target

    # ---- 从磁盘重载（重启/回退用） --------------------------------
    def reload(self, path: str | None = None) -> dict:
        target = os.path.abspath(path) if path else self._bundle.source_path
        b = load_checkpoint(target, dtype=self._bundle.dtype)
        self._bundle = b
        return {
            "path": b.source_path,
            "feature_names": list(b.feature_names),
            "input_dim": len(b.feature_names),
            "optimizer_step": _read_step(b.optimizer),
        }


def _read_step(optimizer: torch.optim.Optimizer) -> int:
    """兼容不同 PyTorch 版本读取 Adam 的 step。"""
    state = optimizer.state
    if not state:
        return 0
    step = next(iter(state.values())).get("step", 0)
    if torch.is_tensor(step):
        return int(step.item())
    return int(step)


def _train_once(
    bundle: CheckpointBundle, instances: list[dict], labels: list[int]
) -> dict:
    """在给定 bundle 上训练一步（调用方保证独占该 bundle）。"""
    b = bundle
    x, _ = align_rows(instances, b.feature_names, b.dtype)
    y = align_labels(labels, b.num_classes)

    criterion = torch.nn.CrossEntropyLoss()
    b.model.train()

    with torch.no_grad():
        loss_before = float(criterion(b.model(x), y))

    b.optimizer.zero_grad()
    logits = b.model(x)
    loss = criterion(logits, y)
    loss.backward()
    b.optimizer.step()

    with torch.no_grad():
        loss_after = float(criterion(b.model(x), y))

    return {
        "step": _read_step(b.optimizer),
        "loss_before": loss_before,
        "loss_after": loss_after,
        "num_samples": len(instances),
        "feature_names": list(b.feature_names),
    }


def _classify(
    bundle: CheckpointBundle, rows: list[dict]
) -> tuple[list[int], list[list[str]]]:
    x, missing = align_rows(rows, bundle.feature_names, bundle.dtype)
    was_training = bundle.model.training
    bundle.model.eval()
    try:
        with torch.no_grad():
            labels = torch.argmax(bundle.model(x), dim=1)
    finally:
        if was_training:
            bundle.model.train()
    return [int(v) for v in labels], missing


def _unchanged_body(
    status: str,
    active: CheckpointBundle,
    instances: list[dict],
    outcome: dict,
) -> dict:
    """被拒/冲突响应：正式模型未变，step/feature_names 取自活动 bundle。"""
    return {
        "status": status,
        "applied": False,
        "step": _read_step(active.optimizer),
        "trial_step": outcome["step"],
        "loss_before": outcome["loss_before"],
        "loss_after": outcome["loss_after"],
        "num_samples": len(instances),
        "feature_names": list(active.feature_names),
        "validation": outcome["report"],
    }


def _run_trial(
    trial: CheckpointBundle,
    validation: ValidationSet,
    instances: list[dict],
    labels: list[int],
) -> dict:
    """在副本上试训一步，并用试训前后模型给同一组验证读数分类。

    验证读数沿用与训练/预测完全相同的按列名对齐、缺列补 0 规则。
    模型处于 train 模式时前向结果无随机性（无 dropout/batch-norm），
    但仍显式切到 eval 再还原，保证与 `/predict` 的分类口径一致。
    """
    rows = [reading for reading, _ in validation.items]
    expected = [label for _, label in validation.items]

    before_labels, missing = _classify(trial, rows)
    result = _train_once(trial, instances, labels)
    after_labels, _ = _classify(trial, rows)

    samples = []
    regressed: list[int] = []
    before_correct = 0
    after_correct = 0
    for i, (expected_label, before, after) in enumerate(
        zip(expected, before_labels, after_labels)
    ):
        ok_before = before == expected_label
        ok_after = after == expected_label
        before_correct += int(ok_before)
        after_correct += int(ok_after)
        # “判对变成判错”才拒绝；原本就判错的样本不构成回归
        is_regression = ok_before and not ok_after
        if is_regression:
            regressed.append(i)
        samples.append(
            {
                "index": i,
                "label": expected_label,
                "before_label": before,
                "after_label": after,
                "before_correct": ok_before,
                "after_correct": ok_after,
                "regressed": is_regression,
                "zero_filled_columns": missing[i],
            }
        )

    return {
        **result,
        "report": {
            "revision": validation.revision,
            "num_samples": len(rows),
            "before_correct": before_correct,
            "after_correct": after_correct,
            "regressed_indices": regressed,
            "samples": samples,
        },
    }
