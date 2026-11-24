# 设备振动测点扩列续训 API

产线加装振动测点后，分类模型可在**不停机、不重训**的前提下追加输入列并继续训练；
未改造设备仍用原有字段预测，缺失的新列自动补 0。基于 **FastAPI + CPU 版 PyTorch**
实现的单作业（single-job）API，附带已训练的两层分类模型及**含 Adam 状态**的检查点。

质检员可为本次运行设置验证集，之后 `/train` 改为**逐批验证**：先在副本上试训一步，
用试训前后的模型对同一组拆检读数分类；只要有样本从判对变成判错，就拒绝整批并返回
这些样本的序号，原模型与 Adam 状态保持不变。详见第 4 节。

## 1. 目录结构

```
.
├── app/
│   ├── model.py          # 两层分类器与扩列（零权重追加列）
│   ├── checkpoint.py     # 检查点加载/扩列/原子保存（含 Adam 状态迁移）
│   ├── data.py           # 按列名对齐、缺列补 0、多余列忽略
│   ├── schemas.py        # 请求/响应模型
│   ├── service.py        # 单作业状态、训练一步、预测、逐批验证门禁、保存、重载
│   └── main.py           # FastAPI 路由与启动加载
├── scripts/
│   └── train_initial.py  # 生成初始检查点（合成数据，200 步 Adam）
├── tests/
│   ├── test_core.py      # 数值对照 / 扩列 / Adam 状态 / 保存失败等 9 个用例
│   ├── test_api.py       # HTTP 端到端 7 个用例
│   └── test_validation.py # 逐批验证门禁 9 个用例（共 25 项）
├── checkpoints/
│   └── model.pt          # 已训练检查点（4 输入 → 8 隐藏 → 3 类，含 Adam 状态）
├── examples/
│   ├── expand_api.sh     # 扩列 API 示例
│   ├── retrain_api.sh    # 续训 API 示例
│   └── validation_api.sh # 验证集 + 被拒后继续续训的完整示例
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 2. 模型与检查点

- 模型：`Linear(4→8) → ReLU → Linear(8→3)`，原始字段
  `temp, pressure, rpm, current`，3 个分类。
- 初始检查点由 `scripts/train_initial.py` 在合成三分类数据上训练 200 步得到
  （训练集准确率约 99%），文件内同时保存：
  - 模型 `state_dict`
  - **Adam 优化器完整状态**（`exp_avg`、`exp_avg_sq`、`step`、超参）
  - 特征列名、隐藏维度、类别数、dtype、格式版本
- 检查点格式版本字段 `version=1`，加载入口 `app.checkpoint.load_checkpoint`。

## 3. 扩列与续训的一致性设计

| 需求 | 实现方式 |
| --- | --- |
| 按列名对齐 | `app/data.py` 以模型 `feature_names` 为准建立列索引，请求中的列名映射到固定位置 |
| 缺新列补 0 | 对齐张量先以 0 初始化，缺失列保持 0；响应中通过 `zero_filled_columns` 回显 |
| 多余列忽略 | 不在 `feature_names` 中的键直接丢弃 |
| 扩列后旧分数不变 | 仅在 `fc1.weight` **尾部追加零初始化列**，偏置与 `fc2` 不变；零列对补 0 的旧样本前向无贡献 |
| 续训一步后原参数/Adam 一致 | 旧参数及其 `exp_avg`/`exp_avg_sq`/`step` 原样迁移；新列动量同样以 0 追加，参数顺序不变 |
| 新读数更新新权重 | 带 `vibration` 的样本使新列产生非零梯度，Adam 一步即可更新新增权重 |
| 确定性 | 单线程 CPU（`TORCH_NUM_THREADS=1`）+ `torch.use_deterministic_algorithms(True)` |

## 4. API 一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| GET | `/model` | 当前列名、维度、dtype、Adam step |
| POST | `/predict` | 预测；缺列补 0，返回标签、各类概率、补 0 列 |
| POST | `/train` | 用一批样本**训练一步**；已设验证集时先试训并通过门禁才生效 |
| PUT | `/validation` | **整体设置/替换**本次运行的验证集 |
| POST | `/validation` | **逐条追加**拆检读数与质检员确认的故障类别 |
| GET | `/validation` | 查看当前验证集、条数与 revision |
| DELETE | `/validation` | 清空验证集，`/train` 恢复为照常续训 |
| POST | `/expand` | 追加一个或多个测点列（零权重 + Adam 零状态） |
| POST | `/save` | 原子保存检查点（先写临时文件再 `os.replace`） |
| POST | `/reload` | 从磁盘重载（模拟重启/回退），支持 query 参数 `path` |

训练、扩列、保存、重载由同一把异步锁串行化；预测为只读，不阻塞。

### 逐批验证（质检员门禁）

质检员可在本次运行内为 `/train` 设置验证集，之后每一批训练都要先过验证：

1. **设置/替换验证集**：`PUT /validation` 整体设置；`POST /validation` 逐条提交
   （每条 `{"reading": {列名: 读数}, "label": 已确认类别}`）。每次调用都会让
   内部 `revision` 递增，即使内容相同。
2. **试训一步**：提交 `/train` 时，服务先把当前模型与 Adam 状态**深拷贝**，
   在副本上按原有规则（按列名对齐、缺列补 0、多余列忽略）训练一步。
3. **前后分类对照**：用试训前、试训后的副本给**同一组验证读数**分别分类。
   只要有一条样本从“判对”变为“判错”，整批拒绝，响应 `409`，
   `validation.regressed_indices` 给出这些样本在验证集中的序号（从 0 开始）。
   原本就判错的样本不算回归。
4. **冲突**：试训开始后验证集只要被设置/追加/清空过（`revision` 变化，
   **换回原内容也算**），该批报 `409 / status=conflict`，试训结果丢弃。
5. **生效方式**：只有验证通过且验证集未变，试训后的模型与 Adam 状态才
   作为同一份状态**一次性整体生效**。
6. **不生效的三种情况**（拒绝 / 冲突 / 试训内部失败）都保留原模型与 Adam 状态，
   响应中的 `step` 仍是原步数，`trial_step` 是试训临时走到的步数，仅供排查。
7. **试训期间**：`/predict` 继续用**已生效**的模型预测；`/validation` 也可随时替换。
8. **未设置验证集**：`/train` 照常续训一步，行为与之前一致（`validation: null`）。

验证集只在本次运行内存中生效，不写入检查点，重启后需重新提交。

`/train` 响应要点：

| 字段 | 含义 |
| --- | --- |
| `status` | `applied` / `rejected`（有样本判对变判错）/ `conflict`（验证集被换过） |
| `applied` | 模型与 Adam 状态是否已生效 |
| `step` | 生效后（或未生效时当前的）Adam 步数 |
| `trial_step` | 试训副本走到的步数，未生效时可对照 |
| `validation` | 验证报告：`before_correct`、`after_correct`、`regressed_indices`、逐条 `samples` |

HTTP 状态码：`200` 通过；`409` 被拒或冲突；`422` 请求非法。

### 保存与回退语义

- 保存走“临时文件 + 原子 rename”，**保存失败时原检查点与内存模型均不受影响**，服务继续可用。
- 保存新版本（如 `model_v2.pt`）后，旧文件 `model.pt` 保持不变，仍可通过 `/reload` 切回。
- 重启容器后从 `CHECKPOINT_PATH` 加载；把环境变量指向新版本即恢复扩列后的模型，
  旧字段预测与按新列续训均可用。

## 5. 本地运行（不使用 Docker）

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python scripts/train_initial.py          # （可选）重新生成 checkpoints/model.pt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 6. Docker Compose 启动

```bash
docker compose up -d --build
docker compose logs -f classifier-api
curl http://127.0.0.1:8000/model
```

（本节命令需在有 Docker 的环境执行；本次交付环境无 Docker，未实际构建镜像。）

`./checkpoints` 挂载进容器，容器内保存的新版本在宿主机同目录可见。
镜像以单 uvicorn worker 运行，保证进程内只有一个活动模型。

逐批验证示例（推荐先跑，脚本自带断言）：

```bash
bash examples/validation_api.sh
```

### 扩列示例

```bash
bash examples/expand_api.sh
```

核心请求：

```bash
curl -X POST http://127.0.0.1:8000/expand \
  -H "Content-Type: application/json" \
  -d '{"new_feature_names": ["vibration"]}'

# 未改造设备：只传原字段，vibration 自动补 0
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1}]}'
```

### 续训示例

```bash
bash examples/retrain_api.sh
```

核心请求（带新测点读数的样本）：

```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {"temp": 0.5, "pressure": -0.3, "rpm": 1.2, "current": 0.1, "vibration": 0.25},
      {"temp": -1.1, "pressure": 0.8, "rpm": -0.4, "current": 0.9, "vibration": 0.51}
    ],
    "labels": [0, 1]
  }'
```

### 逐批验证示例（含被拒后继续续训）

```bash
bash examples/validation_api.sh
```

脚本会依次演示：设置验证集 → 提交一批会让验证样本判对变判错的读数（预期 409、
`regressed_indices` 列出回归样本序号）→ 确认 Adam step 未变 → 提交可通过的一批
（200，step +1）→ 保存。脚本自带断言，任何一步不符合预期都会以非零码退出。

核心请求：

```bash
# 1) 质检员逐条提交拆检读数与确认类别（也可 PUT 整体替换）
curl -X POST http://127.0.0.1:8000/validation \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"reading": {"temp": 0.5, "pressure": 3.062, "rpm": 1.654, "current": 0.096}, "label": 2},
      {"reading": {"temp": 0.663, "pressure": -0.202, "rpm": 2.609, "current": 0.205}, "label": 2}
    ]
  }'

# 2) 提交一批训练样本：先试训，用前后模型给上面同一组读数分类
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"temp": -0.251, "pressure": 0.158, "rpm": 1.271, "current": 0.364}],
       "labels": [0]}'

# 若有样本由判对变判错：HTTP 409，本批未生效
# {"status":"rejected","applied":false,"step":200,"trial_step":201,
#  "validation":{"regressed_indices":[1], ...}}

# 3) 直接继续提交下一批；原模型与 Adam 状态未被污染
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{"instances": [{"temp": 0.647, "pressure": -0.504, "rpm": -0.06, "current": -1.654}],
       "labels": [0]}'
# {"status":"applied","applied":true,"step":201,"validation":{"regressed_indices":[], ...}}
```

保存与重启恢复：

```bash
curl -X POST http://127.0.0.1:8000/save \
  -H "Content-Type: application/json" \
  -d '{"path": "/app/checkpoints/model_v2.pt"}'

# 重启后将 CHECKPOINT_PATH 指向 model_v2.pt，或显式 reload
curl -X POST "http://127.0.0.1:8000/reload?path=/app/checkpoints/model_v2.pt"
```

## 7. 自测与实测结果

```bash
python -m pytest tests/ -v
```

**实测环境**：Python 3.11 / torch 2.15.0+cpu / FastAPI 0.165，float32，单线程。
（本机无 Docker，容器相关步骤未实测，见第 8 节。）

需求中的两处硬性数值对照（12 条随机旧样本，全量 25 个测试通过）：

| 对照项 | 限值 | 实测最大绝对误差 |
| --- | --- | --- |
| 扩列后、续训前旧样本分类分数（softmax 概率） | 1e-6 | **0.000e+00** |
| 同批旧样本各训一步后的原参数（4 个张量） | 1e-6 | **0.000e+00** |
| 同上，对应 Adam 的 `exp_avg`/`exp_avg_sq`/`step` | 1e-6 | **0.000e+00** |

其他实测结论：

- 新列零初始化；仅喂旧字段训练一步后新列权重仍为 0；带 `vibration` 读数训练后新列权重变为非零。
- 保存到非法路径返回 500，原 `model.pt` 可重新加载、服务预测不受影响。
- 保存 → 重载（等价重启）→ 旧字段预测正常（补 0 列正确回显）→ 按新列续训，Adam step 正确递增（201→202→203）。
- 重名列、空扩列、越界标签、样本/标签数量不一致均返回 422。
- 逐批验证：确定性数据下，错误标签批次使验证集第 4 条由 2 变为 0、返回 409
  `regressed_indices=[3]`，Adam step 保持 200；随后一批通过验证，step 变为 201。
- 试训期间替换验证集（内容相同或不同）均返回 `status=conflict`，试训走到的
  `trial_step=201` 被丢弃，正式 step 与全部参数/Adam 状态逐位不变。
- 试训期间 `/predict` 仍用已生效模型，返回标签与试训前一致。
- 试训内部抛异常时返回 500，模型与 Adam 状态逐位不变，之后可正常提交。

## 8. 已实现功能与未完成项

**已实现**

- 单作业 FastAPI 服务，CPU PyTorch，启动即加载含 Adam 状态的检查点
- 两层分类模型初始训练脚本与随附检查点
- 按列名对齐、缺列补 0、多余列忽略，预测响应回显补 0 列
- 追加测点列（支持一次多列、重名拒绝），零权重 + Adam 零状态迁移
- 扩列前后旧样本分数、续训一步后原参数与 Adam 状态逐位一致（误差 0 ≤ 1e-6）
- 带新读数样本可更新新增权重；检查点原子保存、失败保护、重载恢复
- **逐批验证门禁**：验证集设置/逐条追加/替换/清空；试训一步后用前后模型
  对同一组读数分类，任一“判对变判错”即整批拒绝并返回样本序号；试训期间
  验证集被替换（含换回原内容）报冲突；拒绝/冲突/失败均保留原模型与 Adam 状态，
  通过才整体生效；未设验证集照常续训；试训期间 `/predict` 与验证集替换均可用
- Dockerfile + docker-compose（单 worker、检查点卷挂载、健康检查）
- 扩列/续训/逐批验证 API shell 示例与 25 项自动化测试

**未完成 / 边界说明**

- 当前环境未安装 Docker（无 `docker` 二进制、无守护进程、无 root/user namespace），
  **镜像与 Compose 未在本机实际构建运行**。已做静态校验：`docker-compose.yml` 可被
  YAML 解析、镜像按 CPU wheel + 单 worker 编写；实际启动仍需在有 Docker 的环境执行
  `docker compose up -d --build`。文中 Docker 相关结论均标注为“待实测”。
- 验证集只保存在本次运行内存中，不随 `/save` 写入检查点，进程重启即丢失。
- 每批训练仍固定为“每请求一步”；验证也是“一步之后”的即时对照，没有多步累积
  阈值（例如连续 N 批或多 epoch 的综合评估）。
- 试训在事件循环的工作线程中运行，已让 `/predict` 与验证集替换不被阻塞；但同一
  时刻只允许一个 `/train`（进程内锁串行化），并发训练请求会排队。
- 未接入鉴权/限流与多作业并发；服务按单作业设计。
- 合成数据仅用于演示与数值核对；接入真实产线数据时需替换
  `scripts/train_initial.py` 的数据来源并重新标定类别。
