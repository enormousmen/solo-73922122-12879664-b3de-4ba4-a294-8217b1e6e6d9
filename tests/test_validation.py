"""逐批验证（/validation + /train 门禁）测试。

覆盖：
1. 验证集设置/追加/替换/清空与请求校验；
2. 有回归样本时整批拒绝，返回回归样本序号，模型与 Adam 状态不变；
3. 拒绝后可继续用原模型续训（步数从原值 +1）；
4. 验证集在试训期间被替换（即使换回原内容）→ 冲突，试训结果丢弃；
5. 试训/验证期间 /predict 仍可用已生效模型预测；
6. 原本就判错的样本不算“由对变错”；
7. 试训异常时模型与 Adam 状态保持不变；
8. 验证读数沿用按列名对齐、缺列补 0 规则；
9. 未设置验证集时照常续训。
"""
from __future__ import annotations

import asyncio
import importlib
import os
import threading

import pytest
import torch
from fastapi.testclient import TestClient

from app.checkpoint import load_checkpoint
from app.service import ModelService

torch.set_num_threads(1)

# 在初始 model.pt 上确认过的确定性数据：
# 这条错误标签样本会让验证集第 4 条（index=3）由判对(2)变为判错(0)。
BAD_BATCH = {
    "instances": [{"temp": -0.251, "pressure": 0.158, "rpm": 1.271, "current": 0.364}],
    "labels": [0],
}
GOOD_BATCH = {
    "instances": [{"temp": 0.647, "pressure": -0.504, "rpm": -0.06, "current": -1.654}],
    "labels": [0],
}
VALIDATION_ITEMS = [
    {"reading": {"temp": 0.5, "pressure": 3.062, "rpm": 1.654, "current": 0.096}, "label": 2},
    {"reading": {"temp": 0.082, "pressure": 0.353, "rpm": 0.684, "current": 0.239}, "label": 1},
    {"reading": {"temp": -0.569, "pressure": -0.501, "rpm": -0.131, "current": 0.78}, "label": 1},
    {"reading": {"temp": 0.663, "pressure": -0.202, "rpm": 2.609, "current": 0.205}, "label": 2},
]

CHECKPOINT = os.path.join("checkpoints", "model.pt")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PATH", os.path.abspath(CHECKPOINT))
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    import app.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c, main


def _state_fingerprint(bundle) -> tuple[list[torch.Tensor], dict, int]:
    params = [p.detach().clone() for p in bundle.model.parameters()]
    opt_state = []
    for p in bundle.model.parameters():
        s = bundle.optimizer.state.get(p, {})
        opt_state.append(
            {
                k: (v.detach().clone() if torch.is_tensor(v) else v)
                for k, v in s.items()
            }
        )
    return params, opt_state, _adam_step(bundle)


def _adam_step(bundle) -> int:
    for s in bundle.optimizer.state.values():
        step = s.get("step", 0)
        return int(step.item()) if torch.is_tensor(step) else int(step)
    return 0


def _assert_same_state(bundle, fingerprint) -> None:
    params, opt_state, step = fingerprint
    for before, after in zip(params, bundle.model.parameters()):
        assert torch.equal(before, after.detach()), "模型参数在不应生效时发生了变化"
    for before_state, param in zip(opt_state, bundle.model.parameters()):
        now = bundle.optimizer.state.get(param, {})
        assert set(before_state) == set(now)
        for k, v in before_state.items():
            cur = now[k]
            if torch.is_tensor(v):
                assert torch.equal(v, cur), f"Adam 状态 {k} 在不应生效时发生了变化"
            else:
                assert v == cur
    assert _adam_step(bundle) == step


# ---------------------------------------------------------------- API 流程
def test_validation_set_append_replace_clear(client):
    c, _ = client

    assert c.get("/validation").json()["configured"] is False
    resp = c.put("/validation", json={"items": VALIDATION_ITEMS})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["configured"] is True and body["count"] == 4 and body["revision"] == 1
    assert body["items"] == VALIDATION_ITEMS

    # 逐条追加
    extra = {"reading": {"temp": 0.1, "pressure": 0.2, "rpm": 0.3, "current": 0.4}, "label": 0}
    body = c.post("/validation", json={"items": [extra]}).json()
    assert body["count"] == 5 and body["revision"] == 2
    assert body["items"][-1] == extra

    # 整体替换（内容相同也算替换，revision 递增）
    body = c.put("/validation", json={"items": VALIDATION_ITEMS}).json()
    assert body["count"] == 4 and body["revision"] == 3

    # 清空
    body = c.delete("/validation").json()
    assert body["configured"] is False and body["revision"] == 4 and body["count"] == 0

    # 请求校验
    assert c.put("/validation", json={"items": []}).status_code == 422
    assert (
        c.put(
            "/validation",
            json={"items": [{"reading": {"temp": 0.1}, "label": 9}]},
        ).status_code
        == 422
    )
    assert (
        c.put(
            "/validation",
            json={"items": [{"reading": {"temp": 0.1}, "label": 0, "extra": 1}]},
        ).status_code
        == 422
    )


def test_rejected_batch_keeps_model_and_adam_then_continue(client):
    c, main = client
    svc = main.service

    assert c.put("/validation", json={"items": VALIDATION_ITEMS}).status_code == 200
    before = _state_fingerprint(svc.bundle)
    pred_before = c.post(
        "/predict", json={"instances": [it["reading"] for it in VALIDATION_ITEMS]}
    ).json()

    resp = c.post("/train", json=BAD_BATCH)
    assert resp.status_code == 409, resp.text
    body = resp.json()
    assert body["applied"] is False and body["status"] == "rejected"
    assert body["step"] == 200 and body["trial_step"] == 201
    assert body["validation"]["regressed_indices"] == [3]
    sample = body["validation"]["samples"][3]
    assert sample["before_correct"] is True and sample["after_correct"] is False
    assert sample["before_label"] == 2 and sample["after_label"] == 0

    # 原模型与 Adam 状态逐位保留，正式步数不变
    _assert_same_state(svc.bundle, before)
    assert c.get("/model").json()["optimizer_step"] == 200
    pred_after = c.post(
        "/predict", json={"instances": [it["reading"] for it in VALIDATION_ITEMS]}
    ).json()
    assert pred_after == pred_before

    # 被拒后继续续训：下一批通过，步数从 200 -> 201
    resp = c.post("/train", json=GOOD_BATCH)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "applied" and body["applied"] is True
    assert body["step"] == 201 and body["validation"]["regressed_indices"] == []
    assert c.get("/model").json()["optimizer_step"] == 201


def test_unset_validation_trains_normally(client):
    c, _ = client
    assert c.get("/validation").json()["configured"] is False
    resp = c.post("/train", json=GOOD_BATCH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "applied" and body["validation"] is None
    assert body["step"] == 201


def test_validation_alignment_and_zero_fill_after_expand(client):
    c, main = client
    assert c.post("/expand", json={"new_feature_names": ["vibration"]}).status_code == 200
    assert c.put("/validation", json={"items": VALIDATION_ITEMS}).status_code == 200

    preds = c.post(
        "/predict", json={"instances": [it["reading"] for it in VALIDATION_ITEMS]}
    ).json()["predictions"]

    resp = c.post("/train", json=GOOD_BATCH)
    assert resp.status_code == 200, resp.text
    report = resp.json()["validation"]
    # 扩列后验证读数缺 vibration：按补 0 规则对齐，分类前的标签与 /predict 一致
    for item, pred in zip(report["samples"], preds):
        assert item["zero_filled_columns"] == ["vibration"]
        assert item["before_label"] == pred["label"]


def test_preexisting_misclassification_is_not_a_regression(client):
    c, main = client
    svc = main.service
    preds = svc.predict([it["reading"] for it in VALIDATION_ITEMS])
    # 故意给第 1 条一个与当前预测不同的确认类别：它本来就判错，
    # 试训后即使仍判错，也不构成“判对 -> 判错”。
    wrong = [dict(it) for it in VALIDATION_ITEMS]
    wrong[0]["label"] = (preds[0]["label"] + 1) % 3
    assert c.put("/validation", json={"items": wrong}).status_code == 200

    resp = c.post("/train", json=GOOD_BATCH)
    assert resp.status_code == 200, resp.text
    report = resp.json()["validation"]
    assert report["regressed_indices"] == []
    assert report["samples"][0]["before_correct"] is False
    assert report["before_correct"] == 3


# ------------------------------------------------------- 并发：试训期间
def _make_service() -> ModelService:
    return ModelService(load_checkpoint(os.path.abspath(CHECKPOINT)))


def _slow_trial_factory(monkeypatch, service_module):
    """让试训卡在主线程可控的位置，便于并发地替换验证集/预测。"""
    original = service_module._run_trial
    started = threading.Event()
    release = threading.Event()

    def slow_trial(*args, **kwargs):
        started.set()
        assert release.wait(10), "测试未放行试训线程"
        return original(*args, **kwargs)

    monkeypatch.setattr(service_module, "_run_trial", slow_trial)
    return started, release


@pytest.mark.parametrize("same_content", [False, True])
def test_conflict_when_validation_replaced_during_trial(monkeypatch, same_content):
    import app.service as service_module

    svc = _make_service()
    started, release = _slow_trial_factory(monkeypatch, service_module)
    before = _state_fingerprint(svc.bundle)

    async def scenario():
        await svc.set_validation(VALIDATION_ITEMS)
        active_revision = svc.validation_state()["revision"]
        task = asyncio.create_task(svc.train_step(**BAD_BATCH))

        assert await asyncio.to_thread(started.wait, 10)
        # 试训进行中：/predict 仍用已生效模型，结果与试训前一致
        preds = svc.predict([it["reading"] for it in VALIDATION_ITEMS])
        assert [p["label"] for p in preds] == [2, 1, 1, 2]
        assert _adam_step(svc.bundle) == 200

        # 试训进行中替换验证集（same_content 时内容与原来完全相同）
        new_items = list(VALIDATION_ITEMS)
        if not same_content:
            new_items = new_items[:-1]
        await svc.set_validation(new_items)
        assert svc.validation_state()["revision"] == active_revision + 1

        release.set()
        return await task

    body, status_code = asyncio.run(scenario())

    assert status_code == 409
    assert body["status"] == "conflict" and body["applied"] is False
    assert body["trial_step"] == 201  # 试训确实跑过，但结果被丢弃
    assert _adam_step(svc.bundle) == 200
    _assert_same_state(svc.bundle, before)

    # 冲突后可继续：按当前（替换后）验证集重新提交同一批
    async def retry():
        return await svc.train_step(**GOOD_BATCH)

    body, status_code = asyncio.run(retry())
    assert status_code == 200 and body["applied"] is True and body["step"] == 201


def test_trial_failure_keeps_model_and_adam(monkeypatch):
    import app.service as service_module

    svc = _make_service()
    before = _state_fingerprint(svc.bundle)

    async def setup():
        await svc.set_validation(VALIDATION_ITEMS)

    asyncio.run(setup())

    def boom(*args, **kwargs):
        raise RuntimeError("模拟试训失败")

    monkeypatch.setattr(service_module, "_run_trial", boom)

    async def scenario():
        with pytest.raises(RuntimeError):
            await svc.train_step(**BAD_BATCH)

    asyncio.run(scenario())
    _assert_same_state(svc.bundle, before)
    assert _adam_step(svc.bundle) == 200

    # 失败不影响后续正常提交
    monkeypatch.undo()
    body, status_code = asyncio.run(svc.train_step(**GOOD_BATCH))
    assert status_code == 200 and body["step"] == 201


# ------------------------------------------------ 完整 HTTP 栈并发验证
def test_http_predict_and_validation_put_during_trial(monkeypatch):
    """试训阻塞期间，HTTP 的 /predict 与 PUT /validation 都能正常完成。"""
    import httpx

    monkeypatch.setenv("CHECKPOINT_PATH", os.path.abspath(CHECKPOINT))
    monkeypatch.setenv("TORCH_NUM_THREADS", "1")
    import app.main as main
    import app.service as service_module

    importlib.reload(main)
    started, release = _slow_trial_factory(monkeypatch, service_module)
    bad_body = {**BAD_BATCH}
    old_revision = {"value": 0}

    async def scenario():
        async with main.lifespan(main.app):
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as ac:
                resp = await ac.put("/validation", json={"items": VALIDATION_ITEMS})
                assert resp.status_code == 200
                old_revision["value"] = resp.json()["revision"]

                train_task = asyncio.create_task(ac.post("/train", json=bad_body))
                assert await asyncio.to_thread(started.wait, 10)

                # 试训中预测：用的是尚未提交的已生效模型
                pred = await ac.post(
                    "/predict",
                    json={"instances": [it["reading"] for it in VALIDATION_ITEMS]},
                )
                assert pred.status_code == 200
                assert [p["label"] for p in pred.json()["predictions"]] == [2, 1, 1, 2]

                # 试训中替换验证集：内容相同，revision 也要变化
                put = await ac.put("/validation", json={"items": VALIDATION_ITEMS})
                assert put.status_code == 200
                assert put.json()["revision"] == old_revision["value"] + 1

                release.set()
                return await train_task

    resp = asyncio.run(scenario())
    assert resp.status_code == 409
    assert resp.json()["status"] == "conflict"

    # 校验 409 响应体与文档模型一致
    from app.schemas import TrainResponse, ValidationStateResponse

    TrainResponse.model_validate(resp.json())
    ValidationStateResponse.model_validate(
        {"configured": True, "revision": old_revision["value"] + 1, "count": 4,
         "items": VALIDATION_ITEMS}
    )

