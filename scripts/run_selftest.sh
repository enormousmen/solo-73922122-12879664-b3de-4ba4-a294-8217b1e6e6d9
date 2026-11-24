#!/usr/bin/env bash
# 交付前关键自测：数值对照 + 全量 pytest。
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python}"

echo "== 1. 两处硬性数值对照（容差 1e-6）=="
"$PYTHON" - <<'EOF'
import asyncio
import torch
from app.checkpoint import load_checkpoint, expand_bundle
from app.data import align_rows
from app.service import ModelService

torch.set_num_threads(1)
OLD = ["temp", "pressure", "rpm", "current"]
g = torch.Generator().manual_seed(7)
rows = [dict(zip(OLD, torch.randn(4, generator=g).tolist())) for _ in range(12)]
labels = [i % 3 for i in range(12)]

base = load_checkpoint("checkpoints/model.pt")
exp = expand_bundle(load_checkpoint("checkpoints/model.pt"), ["vibration"])

def scores(b, rs):
    x, _ = align_rows(rs, b.feature_names, b.dtype)
    with torch.no_grad():
        return torch.softmax(b.model(x), dim=1)

e_score = (scores(base, rows) - scores(exp, rows)).abs().max().item()

plain = load_checkpoint("checkpoints/model.pt")
exp2 = expand_bundle(load_checkpoint("checkpoints/model.pt"), ["vibration"])
asyncio.run(ModelService(plain).train_step(rows, labels))
asyncio.run(ModelService(exp2).train_step(rows, labels))

max_e = e_score
for (n, p), (_, p2) in zip(plain.model.named_parameters(), exp2.model.named_parameters()):
    b = p2.detach()[:, :-1] if n == "fc1.weight" else p2.detach()
    max_e = max(max_e, (p.detach() - b).abs().max().item())

pp, pe = list(plain.model.parameters()), list(exp2.model.parameters())
idx = {n: i for i, (n, _) in enumerate(exp2.model.named_parameters())}
pidx = {n: i for i, (n, _) in enumerate(plain.model.named_parameters())}
for n in pidx:
    sp, se = plain.optimizer.state[pp[pidx[n]]], exp2.optimizer.state[pe[idx[n]]]
    for k in sp:
        a, b2 = sp[k], se[k]
        if n == "fc1.weight" and torch.is_tensor(a) and a.dim() == 2:
            b2 = b2[:, :-1]
        max_e = max(max_e, (a - b2).abs().max().item() if torch.is_tensor(a) else abs(a - b2))

print(f"分类分数误差: {e_score:.3e}；全部对照最大绝对误差: {max_e:.3e}（限值 1e-6）")
assert max_e <= 1e-6, "数值对照超限"
print("数值对照通过")
EOF

echo
echo "== 2. 逐批验证门禁（拒绝/冲突/未设置验证集）=="
"$PYTHON" -I - <<'EOF'
import asyncio
import sys

sys.path.insert(0, ".")
import torch
from app.checkpoint import load_checkpoint
from app.service import ModelService

torch.set_num_threads(1)
VAL = [
    {"reading": {"temp": 0.5, "pressure": 3.062, "rpm": 1.654, "current": 0.096}, "label": 2},
    {"reading": {"temp": 0.082, "pressure": 0.353, "rpm": 0.684, "current": 0.239}, "label": 1},
    {"reading": {"temp": -0.569, "pressure": -0.501, "rpm": -0.131, "current": 0.78}, "label": 1},
    {"reading": {"temp": 0.663, "pressure": -0.202, "rpm": 2.609, "current": 0.205}, "label": 2},
]
BAD_INST = [{"temp": -0.251, "pressure": 0.158, "rpm": 1.271, "current": 0.364}]
BAD_LABELS = [0]
GOOD_INST = [{"temp": 0.647, "pressure": -0.504, "rpm": -0.06, "current": -1.654}]
GOOD_LABELS = [0]

async def main():
    svc = ModelService(load_checkpoint("checkpoints/model.pt"))
    await svc.set_validation(VAL)
    body, code = await svc.train_step(BAD_INST, BAD_LABELS)
    assert code == 409 and body["status"] == "rejected", (code, body)
    assert body["validation"]["regressed_indices"] == [3], body["validation"]
    assert body["step"] == 200, body["step"]
    print(f"拒绝批次：HTTP {code}，回归样本序号 {body['validation']['regressed_indices']}，正式步数仍为 {body['step']}")

    body, code = await svc.train_step(GOOD_INST, GOOD_LABELS)
    assert code == 200 and body["step"] == 201, (code, body)
    print(f"通过批次：HTTP {code}，正式步数 {body['step']}")

    await svc.clear_validation()
    body, code = await svc.train_step(GOOD_INST, GOOD_LABELS)
    assert code == 200 and body["validation"] is None, (code, body)
    print(f"未设置验证集：HTTP {code}，照常续训到 {body['step']}")

asyncio.run(main())
print("逐批验证门禁通过")
EOF

echo
echo "== 3. 全量自动化测试 =="
"$PYTHON" -m pytest tests/ -v
