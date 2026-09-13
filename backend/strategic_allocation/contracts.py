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


class FundingFlow(Contract):
    name: Identifier
    kind: Literal["contribution", "withdrawal"]
    amount: Number = Field(gt=0, le=1e12)
    first_month: int = Field(ge=1, le=360, strict=True)
    last_month: int = Field(ge=1, le=360, strict=True)
    every_months: Literal[1, 3, 12] = 1

    @model_validator(mode="after")
    def interval(self):
        if self.last_month < self.first_month:
            raise ValueError("现金流结束月不能早于开始月；单次支付请将两者设为同一月。")
        return self


class FundingPlan(Contract):
    total_capital: Number = Field(gt=0, le=1e12)
    outside_reserve: Number = Field(default=0, ge=0, le=1e12)
    terminal_target: Number = Field(ge=0, le=1e13)
    amount_basis: Literal["nominal", "real"] = "nominal"
    inflation: Number = Field(default=0, ge=-0.05, le=0.20)
    annual_fee: Number = Field(default=0, ge=0, le=0.10)
    required_probability: Number = Field(ge=0.5, le=0.99)
    liquidity_months: int = Field(default=12, ge=1, le=36, strict=True)
    contribution_stress_ratio: Number = Field(default=0.5, ge=0, le=1)
    drawdown_alert: Number = Field(default=0.2, gt=0, le=1)
    flows: list[FundingFlow] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def capital(self):
        if self.outside_reserve >= self.total_capital:
            raise ValueError("组合外储备必须小于总资金，须保留正的可投资本金。")
        if self.terminal_target == 0 and not any(flow.kind == "withdrawal" for flow in self.flows):
            raise ValueError("期末目标为0时，须至少定义一笔必要支付，不能把空目标评为成功。")
        return self


class BenchmarkPolicy(Contract):
    name: Identifier
    alloc_name: Identifier
    weights: dict[Identifier, Number] = Field(min_length=1, max_length=30)
    target_excess_return: Number = Field(ge=-0.5, le=1)
    max_tracking_error: Number = Field(ge=0, le=1)

    @model_validator(mode="after")
    def full_investment(self):
        if any(value < 0 or value > 1 for value in self.weights.values()) or abs(sum(self.weights.values()) - 1) > 1e-8:
            raise ValueError("基准大类权重须非负且合计为100%。")
        return self


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
    max_tracking_error: Number = Field(default=0.10, ge=0, le=1)
    risk_aversion: Number = Field(default=5, gt=0, le=1000)
    rebalance_policy: Literal["monthly", "quarterly", "annually", "threshold"] = "quarterly"
    rebalance_note: str = Field(default="", max_length=1000)
    note: str = Field(default="", max_length=2000)
    objective_kind: Literal["absolute_return", "funding_goal", "benchmark_relative"] = "absolute_return"
    funding_plan: FundingPlan | None = None
    benchmark: BenchmarkPolicy | None = None
    boundary_reason: str = Field(default="", max_length=2000)
    allocation_scope: Identifier | None = None
    asset_limits: dict[str, AssetLimit] = Field(default_factory=dict, max_length=30)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def dates(self):
        if self.as_of > date.today() or self.review_date <= self.as_of:
            raise ValueError("研究日不能在未来，政策复核日须晚于研究日。")
        if (self.objective_kind == "funding_goal") != (self.funding_plan is not None):
            raise ValueError("金额目标须提供资金计划；其他目标不能残留金额计划。")
        if (self.objective_kind == "benchmark_relative") != (self.benchmark is not None):
            raise ValueError("相对目标须提供真实基准权重；其他目标不能残留基准设置。")
        if self.objective_kind != "absolute_return" and self.target_return != 0:
            raise ValueError("非绝对收益目标不使用最低算术收益字段，请清零；所需复合收益另行计算。")
        if self.funding_plan:
            months = self.horizon_years * 12
            if self.funding_plan.liquidity_months > months or any(flow.last_month > months for flow in self.funding_plan.flows):
                raise ValueError("现金流或流动性窗口超出了投资期限；不能静默截断支付计划。")
        if self.rebalance_policy == "threshold" and not self.rebalance_note:
            raise ValueError("阈值再平衡须说明触发和恢复规则；本页只记录政策，不自动交易。")
        if (self.asset_limits or self.group_limits) and not self.allocation_scope:
            raise ValueError("资产或分组授权必须绑定所属大类方案，不能仅按资产名称复用。")
        if self.benchmark and self.allocation_scope and self.benchmark.alloc_name != self.allocation_scope:
            raise ValueError("相对基准与资产授权必须属于同一个大类方案。")
        seen_groups = set()
        for group in self.group_limits:
            if (group.id in seen_groups or not group.assets or len(set(group.assets)) != len(group.assets)
                    or group.lo > group.hi):
                raise ValueError("投资授权分组必须有唯一名称、非空不重复成员及有效上下界。")
            seen_groups.add(group.id)
        return self


class MandateStudyRequest(Contract):
    definition: MandateRequest
    cma_id: Identifier | None = None
    simulation_paths: int = Field(default=2000, ge=500, le=10000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    uncertainty_penalty: Number = Field(default=1, ge=0, le=5)


class ConfirmMandateRequest(Contract):
    request: MandateStudyRequest
    preview_hash: Fingerprint
    acknowledge_limits: Literal[True]


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
