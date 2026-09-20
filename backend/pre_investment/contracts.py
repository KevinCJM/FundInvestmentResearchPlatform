"""Explicit research assumptions; absent evidence is never inferred as zero."""

from datetime import date
from datetime import date as CalendarDate
from typing import Literal
from pydantic import Field, model_validator
from backend.strategic_allocation.common_contracts import (
    Contract,
    Number,
    Identifier,
    Fingerprint,
)


class AllocationSource(Contract):
    kind: Literal["saa_policy", "taa_decision"]
    id: Identifier
    content_hash: Fingerprint
    implementation_mapping_id: Identifier | None = None


class ProductAllocation(Contract):
    kind: Literal["etf", "fund", "cash"]
    product_id: Identifier
    asset_class_id: Identifier
    weight: Number = Field(ge=0, le=1)
    max_weight: Number = Field(default=1, gt=0, le=1)
    current_value: Number = Field(default=0, ge=0, le=1e12)
    buy_rate: Number | None = Field(default=None, ge=0, le=0.2)
    sell_rate: Number | None = Field(default=None, ge=0, le=0.2)
    fee_source: str = Field(default="", max_length=2000)
    fee_valid_until: date | None = None
    fee_basis: Literal["each_side_notional", "unknown"] = "unknown"
    holding_days: int | None = Field(default=None, ge=0, le=50000)
    channel: str = Field(default="", max_length=200)
    nav_includes_management_fee: bool | None = None
    settlement_date: date | None = None
    same_month_settlement_confirmed: bool = False
    settlement_terms_source: str = Field(default="", max_length=2000)


class CashEvent(Contract):
    id: Identifier
    date: CalendarDate | None = None
    kind: Literal["receipt", "payment"]
    amount: Number = Field(gt=0, le=1e12)
    evidence: str = Field(min_length=3, max_length=2000)


class Reconciliation(Contract):
    occurrence_id: Fingerprint
    status: Literal["paid", "partial", "unpaid", "unknown"]
    paid_amount: Number = Field(ge=0, le=1e12)
    evidence: str = Field(min_length=3, max_length=2000)


class FundingState(Contract):
    valuation_at: date
    knowledge_cutoff: date
    confirmed_investable_value: Number = Field(ge=0, le=1e12)
    settled_cash: Number = Field(ge=0, le=1e12)
    restricted_cash: Number = Field(default=0, ge=0, le=1e12)
    receivables: Number = Field(default=0, ge=0, le=1e12)
    payables: Number = Field(default=0, ge=0, le=1e12)
    elapsed_months: int = Field(default=0, ge=0, le=360, strict=True)
    cutoff_phase: Literal["after_model_month_end", "calendar_adapter_required"] = (
        "after_model_month_end"
    )
    transition_cost_in_balance: bool = False
    evidence: str = Field(min_length=3, max_length=2000)
    reconciliation: list[Reconciliation] = Field(default_factory=list, max_length=8640)
    cash_events: list[CashEvent] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def clocks(self):
        if (
            self.valuation_at > self.knowledge_cutoff
            or self.knowledge_cutoff > date.today()
        ):
            raise ValueError("余额日不得晚于知识截止日，知识截止日不得晚于今天。")
        if len({x.occurrence_id for x in self.reconciliation}) != len(
            self.reconciliation
        ):
            raise ValueError("同一支付发生额只能核对一次。")
        if len({x.id for x in self.cash_events}) != len(self.cash_events):
            raise ValueError("现金事件标识不得重复。")
        return self


class ImplementationCandidate(Contract):
    name: Identifier
    source: AllocationSource
    as_of: date
    start_date: date
    products: list[ProductAllocation] = Field(min_length=1, max_length=50)
    state: FundingState
    annual_additional_fee: Number = Field(default=0, ge=0, le=0.1)
    annual_fee_source: str = Field(default="", max_length=2000)
    return_basis_confirmed: bool = False
    horizon_stationarity_acknowledged: bool = False
    future_weight_rule: Literal[
        "frozen_scalar_proxy", "buy_and_hold", "monthly_rebalance"
    ] = "frozen_scalar_proxy"
    future_fee_assumption: Literal["constant_declared_rates_sensitivity", "unknown"] = (
        "unknown"
    )
    paths: int = Field(default=2000, ge=200, le=10000, strict=True)
    search_seed: int = Field(default=42, ge=0, le=2147483647, strict=True)
    validation_seed: int = Field(default=314159, ge=0, le=2147483647, strict=True)
    min_validation_r2: Number = Field(default=0, ge=-1, le=1)
    scenario_release_ids: list[Identifier] = Field(default_factory=list, max_length=5)
    scenario_exposure_release_id: Identifier | None = None

    @model_validator(mode="after")
    def axes(self):
        if not self.start_date < self.as_of <= date.today():
            raise ValueError("历史开始日须早于研究日，研究日不能晚于今天。")
        if self.state.knowledge_cutoff != self.as_of:
            raise ValueError("资金状态和研究包必须使用同一知识截止日。")
        if self.search_seed == self.validation_seed:
            raise ValueError("搜索与独立验证必须采用不同随机流。")
        keys = [(p.kind, p.product_id.upper()) for p in self.products]
        if (
            len(set(keys)) != len(keys)
            or sum(p.kind == "cash" for p in self.products) > 1
        ):
            raise ValueError("产品不可重复，现金最多一个独立项目。")
        return self


class PackageWrite(Contract):
    candidate: ImplementationCandidate
    expected_revision: int = Field(default=0, ge=0, strict=True)
    idempotency_key: str = Field(min_length=8, max_length=160)
    copied_from_id: Identifier | None = None


class PackageAction(Contract):
    expected_revision: int = Field(ge=1, strict=True)
    candidate_hash: Fingerprint
    idempotency_key: str = Field(min_length=8, max_length=160)


class FinalizePackage(PackageAction):
    validation_report_hash: Fingerprint
    reviewer: str = Field(min_length=2, max_length=120)
    reason: str = Field(min_length=5, max_length=2000)
    review_due_at: date
    accept_research_limits: Literal[True]


class RegisterAttempt(Contract):
    hypothesis_family: str = Field(min_length=3, max_length=200)
    candidate_hash: Fingerprint
    status: Literal["started", "succeeded", "failed", "canceled"]
    logical_attempt_id: Identifier
    reason: str = Field(min_length=3, max_length=2000)
    idempotency_key: str = Field(min_length=8, max_length=160)
