#!/usr/bin/env bash
# 逐批验证 API 示例：设置验证集 -> 提交一批被拒（判对变判错）-> 保留原模型 ->
# 再提交一批通过 -> 被拒后继续续训。
#
# 前提：docker compose up -d --build 已启动服务（默认 8000 端口），
# 且模型处于初始状态（temp, pressure, rpm, current 四列，Adam step = 200）。
# 如需重复执行，先 `docker compose restart classifier-api` 回到初始状态。
#
# 依赖：curl、python3（仅用于格式化/断言 JSON）。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
# 容器内路径；在宿主机直接跑服务时可用 SAVE_PATH=./checkpoints/model_validated.pt
SAVE_PATH="${SAVE_PATH:-/app/checkpoints/model_validated.pt}"

# 用 python3 做“打印 + 断言”的小工具，避免引入 jq 依赖
show() { python3 -m json.tool; }

# 验证集：质检员逐条提交 4 条拆检读数与确认类别
VALIDATION_JSON='{
  "items": [
    {"reading": {"temp": 0.5,    "pressure": 3.062, "rpm": 1.654, "current": 0.096}, "label": 2},
    {"reading": {"temp": 0.082,  "pressure": 0.353, "rpm": 0.684, "current": 0.239}, "label": 1},
    {"reading": {"temp": -0.569, "pressure": -0.501, "rpm": -0.131, "current": 0.78},  "label": 1},
    {"reading": {"temp": 0.663,  "pressure": -0.202, "rpm": 2.609, "current": 0.205}, "label": 2}
  ]
}'

# 会被拒绝的一批：错误标签让验证集第 4 条由判对(2)变成判错(0)
REJECT_BATCH='{
  "instances": [{"temp": -0.251, "pressure": 0.158, "rpm": 1.271, "current": 0.364}],
  "labels": [0]
}'

# 会通过的一批：验证集全部仍判对
ACCEPT_BATCH='{
  "instances": [{"temp": 0.647, "pressure": -0.504, "rpm": -0.06, "current": -1.654}],
  "labels": [0]
}'

step_of() { python3 -c 'import json,sys; print(json.load(sys.stdin)["optimizer_step"])'; }

echo "== 1. 初始模型（Adam step 应为 200）=="
STEP0=$(curl -s "${BASE_URL}/model" | tee /dev/stderr | step_of)
echo

echo "== 2. 整批设置验证集（PUT /validation）=="
curl -s -X PUT "${BASE_URL}/validation" \
    -H "Content-Type: application/json" \
    -d "${VALIDATION_JSON}" | show

echo
echo "   再逐条追加一条拆检记录（POST /validation）"
curl -s -X POST "${BASE_URL}/validation" \
    -H "Content-Type: application/json" \
    -d '{"items": [{"reading": {"temp": 0.12, "pressure": -0.44, "rpm": 0.91, "current": -0.37}, "label": 1}]}' \
    | python3 -c 'import json,sys; b=json.load(sys.stdin); print("count =", b["count"], "revision =", b["revision"])'

echo
echo "   恢复为 4 条验证样本（替换时 revision 仍会递增）"
curl -s -X PUT "${BASE_URL}/validation" \
    -H "Content-Type: application/json" \
    -d "${VALIDATION_JSON}" \
    | python3 -c 'import json,sys; b=json.load(sys.stdin); print("count =", b["count"], "revision =", b["revision"])'

echo
echo "== 3. 提交一批“会让验证样本判对变判错”的样本 =="
echo "预期：409，status=rejected，regressed_indices 含回归样本序号，applied=false"
REJECT_STATUS=$(curl -s -o /tmp/validation_reject.json -w '%{http_code}' \
    -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d "${REJECT_BATCH}")
cat /tmp/validation_reject.json | show
echo "HTTP ${REJECT_STATUS}"
test "${REJECT_STATUS}" = "409"
python3 - <<'PY'
import json
body = json.load(open("/tmp/validation_reject.json"))
assert body["status"] == "rejected" and body["applied"] is False, body
assert body["validation"]["regressed_indices"], body["validation"]
print("已被拒，回归样本序号:", body["validation"]["regressed_indices"])
PY

echo
echo "== 4. 被拒后模型与 Adam 状态保持原样（step 仍为 ${STEP0}）=="
STEP1=$(curl -s "${BASE_URL}/model" | step_of)
echo "Adam step: ${STEP1}"
test "${STEP1}" = "${STEP0}"

echo
echo "== 5. 被拒后继续续训：提交一批可通过验证的样本 =="
ACCEPT_STATUS=$(curl -s -o /tmp/validation_accept.json -w '%{http_code}' \
    -X POST "${BASE_URL}/train" \
    -H "Content-Type: application/json" \
    -d "${ACCEPT_BATCH}")
cat /tmp/validation_accept.json | show
echo "HTTP ${ACCEPT_STATUS}"
test "${ACCEPT_STATUS}" = "200"
python3 - <<'PY'
import json
body = json.load(open("/tmp/validation_accept.json"))
assert body["status"] == "applied" and body["applied"] is True, body
assert body["validation"]["regressed_indices"] == [], body["validation"]
print("验证通过，模型步数:", body["step"])
PY

echo
echo "== 6. 再次确认步数只前进了 1（被拒批次不计入）=="
STEP2=$(curl -s "${BASE_URL}/model" | step_of)
echo "Adam step: ${STEP2}（期望 $((${STEP0} + 1))）"
test "${STEP2}" = "$((${STEP0} + 1))"

echo
echo "== 7. 保存通过验证后的检查点 =="
curl -s -X POST "${BASE_URL}/save" \
    -H "Content-Type: application/json" \
    -d "{\"path\": \"${SAVE_PATH}\"}" | show

echo
echo "示例完成：被拒批次未生效，随后一批通过验证并继续续训。"
echo "如需再跑一次，执行：docker compose restart classifier-api"
