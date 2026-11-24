"""FastAPI 请求/响应模型。"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    # 每条样本为 {列名: 数值}；模型需要但未提供的列自动补 0
    instances: list[dict[str, float]] = Field(min_length=1)


class PredictedItem(BaseModel):
    label: int
    score: float
    probabilities: list[float]
    zero_filled_columns: list[str]


class PredictResponse(BaseModel):
    feature_names: list[str]
    predictions: list[PredictedItem]


class TrainRequest(BaseModel):
    instances: list[dict[str, float]] = Field(min_length=1)
    labels: list[int] = Field(min_length=1)


class ValidationItem(BaseModel):
    """一条已确认的拆检记录：读数列 + 质检员确认的故障类别。"""

    model_config = ConfigDict(extra="forbid")

    reading: dict[str, float]
    label: int


class SetValidationRequest(BaseModel):
    """整体设置/替换验证集。"""

    items: list[ValidationItem] = Field(min_length=1)


class ValidationStateResponse(BaseModel):
    configured: bool
    revision: int
    count: int
    items: list[ValidationItem]


class ValidationSampleResult(BaseModel):
    index: int
    label: int
    before_label: int
    after_label: int
    before_correct: bool
    after_correct: bool
    regressed: bool
    zero_filled_columns: list[str]


class ValidationReport(BaseModel):
    revision: int
    num_samples: int
    before_correct: int
    after_correct: int
    regressed_indices: list[int]
    samples: list[ValidationSampleResult]


class TrainResponse(BaseModel):
    # status=applied 表示模型与 Adam 状态已一并生效；
    # status=rejected 表示验证集出现“判对 -> 判错”，整批未生效。
    status: str
    applied: bool
    step: int
    trial_step: int | None = None
    loss_before: float
    loss_after: float
    num_samples: int
    feature_names: list[str]
    validation: ValidationReport | None = None


class ExpandRequest(BaseModel):
    new_feature_names: list[str] = Field(min_length=1)


class ExpandResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    added: list[str]


class ModelInfoResponse(BaseModel):
    feature_names: list[str]
    input_dim: int
    hidden_dim: int
    num_classes: int
    dtype: str
    optimizer_step: int
    source_path: str


class SaveRequest(BaseModel):
    path: str | None = None  # 为空则覆盖当前检查点路径


class SaveResponse(BaseModel):
    path: str
    ok: bool


class ReloadResponse(BaseModel):
    path: str
    feature_names: list[str]
    input_dim: int
    optimizer_step: int


class ErrorMessage(BaseModel):
    detail: Any
