"""单作业分类模型 API（FastAPI + CPU PyTorch）。

启动时从 CHECKPOINT_PATH 加载含 Adam 状态的检查点。
环境变量：
    CHECKPOINT_PATH   检查点文件（默认 checkpoints/model.pt）
    MODEL_DTYPE       float32（默认）或 float64
    TORCH_NUM_THREADS CPU 线程数（默认 1，保证可复现的逐位一致行为）
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

# 必须在 import torch 前设置线程数，保证 GEMM 路径稳定可复现
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "1"))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("TORCH_NUM_THREADS", "1"))

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from .checkpoint import load_checkpoint
from .schemas import (
    ExpandRequest,
    ExpandResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
    SaveRequest,
    SaveResponse,
    SetValidationRequest,
    TrainRequest,
    TrainResponse,
    ValidationStateResponse,
)
from .service import ModelService

CHECKPOINT_PATH = os.path.abspath(
    os.environ.get("CHECKPOINT_PATH", "checkpoints/model.pt")
)
MODEL_DTYPE = "float64" if os.environ.get("MODEL_DTYPE", "float32") == "float64" else "float32"

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
torch.use_deterministic_algorithms(True)

_DTYPE = {"float32": torch.float32, "float64": torch.float64}[MODEL_DTYPE]

service: ModelService | None = None


@asynccontextmanager
async def lifespan(app: "FastAPI"):
    global service
    try:
        bundle = load_checkpoint(CHECKPOINT_PATH, dtype=_DTYPE)
    except Exception as exc:  # 启动期失败直接暴露，避免空服务
        raise RuntimeError(f"加载检查点失败 {CHECKPOINT_PATH}: {exc}") from exc
    service = ModelService(bundle)
    yield


app = FastAPI(
    title="设备振动测点扩列续训 API",
    version="1.0.0",
    description="两层分类模型：支持按列名扩列（缺列补 0）、续训与原子保存。",
    lifespan=lifespan,
)


def _svc() -> ModelService:
    if service is None:
        raise HTTPException(status_code=503, detail="服务尚未完成初始化")
    return service


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model", response_model=ModelInfoResponse)
def model_info() -> dict:
    return _svc().info()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> dict:
    svc = _svc()
    try:
        preds = svc.predict(req.instances)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"feature_names": svc.bundle.feature_names, "predictions": preds}


@app.post(
    "/train",
    response_model=TrainResponse,
    responses={
        409: {
            "model": TrainResponse,
            "description": "status=rejected（验证样本判对变判错）或 "
            "status=conflict（试训期间验证集被替换），本批未生效",
        }
    },
)
async def train(req: TrainRequest) -> JSONResponse:
    svc = _svc()
    async with svc.lock:
        try:
            body, status_code = await svc.train_step(req.instances, req.labels)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # 试训/校验内部失败：活动模型未提交，服务可继续
            raise HTTPException(status_code=500, detail=f"训练失败（本批未生效）: {exc}") from exc
    # 被拒/冲突时返回 409，响应体结构与成功时一致，便于调用方固定解析
    return JSONResponse(status_code=status_code, content=jsonable_encoder(body))


# ---- 验证集管理（逐条提交、整体替换、清空） -------------------------
@app.put("/validation", response_model=ValidationStateResponse)
async def set_validation(req: SetValidationRequest) -> dict:
    """整体设置或替换本次运行的验证集。"""
    svc = _svc()
    try:
        return await svc.set_validation([item.model_dump() for item in req.items])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/validation", response_model=ValidationStateResponse)
async def append_validation(req: SetValidationRequest) -> dict:
    """逐条追加拆检读数与质检员确认的故障类别。"""
    svc = _svc()
    try:
        return await svc.append_validation([item.model_dump() for item in req.items])
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/validation", response_model=ValidationStateResponse)
def validation_state() -> dict:
    return _svc().validation_state()


@app.delete("/validation", response_model=ValidationStateResponse)
async def clear_validation() -> dict:
    """清空验证集；之后 /train 恢复为“照常续训”。"""
    return await _svc().clear_validation()


@app.post("/expand", response_model=ExpandResponse)
async def expand(req: ExpandRequest) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            return svc.expand(req.new_feature_names)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/save", response_model=SaveResponse)
async def save(req: SaveRequest) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            path = svc.save(req.path)
        except OSError as exc:
            # 保存失败：内存模型与原检查点均不受影响
            raise HTTPException(status_code=500, detail=f"保存失败: {exc}") from exc
    return {"path": path, "ok": True}


@app.post("/reload", response_model=ReloadResponse)
async def reload(path: str | None = None) -> dict:
    svc = _svc()
    async with svc.lock:
        try:
            return svc.reload(path)
        except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError) as exc:
            raise HTTPException(status_code=500, detail=f"重载失败: {exc}") from exc
