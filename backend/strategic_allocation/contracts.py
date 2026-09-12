"""Versioned, same-currency, long-only research inputs; no client performance."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.tactical_allocation.contracts import AssetLimit, GroupLimit

Number = Annotated[float, Field(strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=120)]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class MandateRequest(Contract):
    name: Identifier
    as_of: date
    review_date: date
    currency: Currency = "CNY"
    horizon_years: int = Field(default=10, ge=1, le=30, strict=True)
    target_return: Number = Field(default=0.0, ge=-0.5, le=1)
    max_volatility: Number = Field(default=0.15, gt=0, le=2)
    min_liquid_weight: Number = Field(default=0, ge=0, le=1)
    max_illiquid_weight: Number = Field(default=0, ge=0, le=1)
    max_tracking_error: Number = Field(default=0.10, gt=0, le=1)
    risk_aversion: Number = Field(default=5, gt=0, le=1000)
    rebalance_policy: Literal["monthly", "quarterly", "annually", "threshold"] = "quarterly"
    rebalance_note: str = Field(default="", max_length=1000)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def dates(self):
        if self.as_of > date.today() or self.review_date <= self.as_of:
            raise ValueError("研究日不能在未来，政策复核日须晚于研究日。")
        if self.rebalance_policy == "threshold" and not self.rebalance_note:
            raise ValueError("阈值再平衡须说明触发和恢复规则；本页只记录政策，不自动交易。")
        return self


class RiskReferenceRequest(Contract):
    alloc_name: Identifier
    as_of: date
    start_date: date
    end_date: date
    shrinkage: Number = Field(default=0.1, ge=0, le=1)
    periods_per_year: int = Field(default=252, ge=1, le=366, strict=True)

    @model_validator(mode="after")
    def dates(self):
        if not self.start_date < self.end_date <= self.as_of <= date.today():
            raise ValueError("日期须满足：样本开始 < 样本结束 ≤ 研究日 ≤ 今天。")
        return self


class AssetAssumption(Contract):
    id: Identifier
    role: Literal["growth", "rates", "inflation", "credit", "liquidity", "diversifier"]
    liquidity: Literal["liquid", "illiquid"]
    rationale: str = Field(min_length=3, max_length=1000)
    annual_return: Number = Field(ge=-0.5, le=2)
    annual_volatility: Number = Field(gt=0, le=3)
    mean_uncertainty: Number = Field(ge=0, le=1)


class CmaRequest(Contract):
    name: Identifier
    alloc_name: Identifier
    as_of: date
    currency: Currency = "CNY"
    horizon_years: int = Field(default=10, ge=1, le=30, strict=True)
    return_basis: Literal["annual_arithmetic_total_return"] = "annual_arithmetic_total_return"
    source: str = Field(min_length=3, max_length=2000)
    basis_confirmed: Literal[True]
    assets: list[AssetAssumption] = Field(min_length=1, max_length=30)
    correlation: list[list[Number]] = Field(min_length=1, max_length=30)
    risk_origin: Literal["manual", "historical_reference"] = "manual"
    risk_reference: RiskReferenceRequest | None = None
    risk_reference_hash: Fingerprint | None = None

    @model_validator(mode="after")
    def shape(self):
        if self.as_of > date.today():
            raise ValueError("长期假设的研究日不能位于未来。")
        count = len(self.assets)
        if len({a.id for a in self.assets}) != count:
            raise ValueError("每个资产类别只能有一条假设。")
        if len(self.correlation) != count or any(len(row) != count for row in self.correlation):
            raise ValueError("相关矩阵行列须与资产列表完全一致。")
        if self.risk_origin == "historical_reference":
            if not self.risk_reference or not self.risk_reference_hash:
                raise ValueError("历史风险参考须保留样本区间和来源校验。")
            if self.risk_reference.alloc_name != self.alloc_name or self.risk_reference.as_of != self.as_of:
                raise ValueError("风险参考须属于同一分类与研究日。")
        elif self.risk_reference is not None or self.risk_reference_hash is not None:
            raise ValueError("人工风险假设不能保留旧历史参考认证，请清除引用。")
        return self


class PublishCmaRequest(Contract):
    request: CmaRequest
    preview_hash: Fingerprint


class PolicyRequest(Contract):
    mandate_id: Identifier
    cma_id: Identifier
    constraints: dict[str, AssetLimit] = Field(default_factory=dict)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=28)
    uncertainty_penalty: Number = Field(default=1, ge=0, le=5)
    candidate_count: int = Field(default=2000, ge=200, le=5000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)


class PublishPolicyRequest(Contract):
    request: PolicyRequest
    preview_hash: Fingerprint
    candidate_id: Literal["minimum-risk", "nominal-utility", "robust-utility", "maximum-return"]
    name: Identifier
    reason: str = Field(min_length=5, max_length=2000)
