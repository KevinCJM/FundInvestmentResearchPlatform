"""Strict research contracts: users configure studies, never submit fitted betas."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Frequency = Literal["daily", "weekly", "monthly", "quarterly"]
Stage = Literal["product", "event_macro", "macro_market"]
Role = Literal["driver", "macro", "market"]
Unit = Literal["return", "bp", "pp", "points"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class ProductRef(Contract):
    kind: Literal["etf", "fund"]
    product_id: str = Field(min_length=1, max_length=80)
    name: str = Field(default="", max_length=120)


class ModelFields(Contract):
    name: str = Field(min_length=1, max_length=100)
    stage: Stage = "product"
    method: Literal["ols"] = "ols"
    inputs: list[str] = Field(min_length=1, max_length=8)
    outputs: list[str] = Field(default_factory=list, max_length=8)
    targets: list[ProductRef] = Field(default_factory=list, max_length=64)
    frequency: Frequency = "monthly"
    start_date: date
    end_date: date
    validation_start: date
    as_of: date
    lags: int = Field(default=0, ge=0, le=6, strict=True)
    min_train: int = Field(default=36, ge=30, le=10000, strict=True)
    min_validation: int = Field(default=12, ge=10, le=2000, strict=True)
    minimum_validation_r2: float = Field(default=0.0, ge=-1.0, le=0.99)
    refit_after_validation: bool = True

    @model_validator(mode="after")
    def relationships(self):
        if not self.start_date < self.validation_start <= self.end_date <= self.as_of:
            raise ValueError("日期须满足：开始日 < 验证开始日 ≤ 数据结束日 ≤ 研究截止日")
        if len(set(self.inputs)) != len(self.inputs) or len(set(self.outputs)) != len(self.outputs):
            raise ValueError("输入或输出不能重复")
        if set(self.inputs) & set(self.outputs):
            raise ValueError("同一变量不能同时充当模型输入和输出")
        if len(self.inputs) * (self.lags + 1) > 24:
            raise ValueError("输入数量乘以滞后阶数加一不能超过 24")
        if self.stage == "product":
            if not self.targets or self.outputs or self.lags:
                raise ValueError("产品模型须选择产品、使用同期暴露，不填写宏观输出或滞后")
            keys = [(item.kind, item.product_id) for item in self.targets]
            if len(keys) != len(set(keys)):
                raise ValueError("研究产品不能重复")
        elif self.targets or not self.outputs or self.frequency not in {"monthly", "quarterly"}:
            raise ValueError("宏观传导须选择输出变量，使用月频或季频，不直接选择产品")
        return self


class PublishRequest(Contract):
    definition: ModelFields
    preview_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    note: str = Field(default="", max_length=1000)
    effective_from: date | None = None
    valid_days: int = Field(default=30, ge=1, le=365, strict=True)
    acknowledge_limitations: Literal[True]


class RetireRequest(Contract):
    note: str = Field(default="", max_length=1000)


class SeriesImport(Contract):
    name: str = Field(min_length=1, max_length=100)
    roles: list[Role] = Field(min_length=1, max_length=3)
    unit: Unit
    frequency: Frequency
    transform: Literal["price_return", "percent_rate_change", "difference", "identity"]
    source_label: str = Field(min_length=1, max_length=300)
    csv_text: str = Field(min_length=1, max_length=2_000_000)
    category: Literal["activity", "inflation", "policy", "energy", "credit", "equity", "currency", "other"] = "other"

    @model_validator(mode="after")
    def units(self):
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("变量角色不能重复")
        if self.transform == "price_return" and self.unit != "return":
            raise ValueError("价格转换的输出必须是简单收益率")
        if self.transform == "percent_rate_change" and self.unit != "bp":
            raise ValueError("百分数利率的变动输出必须是 bp")
        if self.transform == "difference" and self.unit == "return":
            raise ValueError("收益率请使用价格转换或直接输入，不能用价格差替代收益率")
        return self


class Cashflow(Contract):
    years: float = Field(gt=0, le=100)
    amount: float = Field(gt=0, le=1e12)


class CashflowStudy(Contract):
    name: str = Field(min_length=1, max_length=100)
    product_id: str = Field(min_length=1, max_length=80, pattern=r"^[\w.:-]+$")
    as_of: date
    yield_factor_id: str = Field(min_length=1, max_length=120)
    yield_percent: float = Field(ge=-50, le=100)
    compounding: Literal[1, 2, 4, 12] = 2
    cashflows: list[Cashflow] = Field(min_length=1, max_length=600)
    source_label: str = Field(min_length=1, max_length=300)

    @field_validator("cashflows")
    @classmethod
    def ordered(cls, flows):
        if any(flows[index].years >= flows[index + 1].years for index in range(len(flows) - 1)):
            raise ValueError("现金流期限须严格递增；同一期现金流请合并")
        return flows


class CashflowPublishRequest(Contract):
    study: CashflowStudy
    preview_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    note: str = Field(default="", max_length=1000)
    effective_from: date | None = None
    valid_days: int = Field(default=30, ge=1, le=365, strict=True)
    acknowledge_limitations: Literal[True]
