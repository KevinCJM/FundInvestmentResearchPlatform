"""Explicit research contracts; no client-provided performance or PIT waivers."""
from __future__ import annotations

from datetime import date
from typing import Literal, Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AssetLimit(Contract):
    min_weight: float = Field(default=0, ge=0, le=1)
    max_weight: float = Field(default=1, ge=0, le=1)
    max_abs_tilt: float = Field(default=.1, ge=0, le=1)


class GroupLimit(Contract):
    id: str = Field(min_length=1, max_length=120)
    assets: list[str] = Field(min_length=1, max_length=30)
    lo: float = Field(default=0, ge=0, le=1)
    hi: float = Field(default=1, ge=0, le=1)


class BaselineRequest(Contract):
    alloc_name: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=120)
    as_of: date
    weights: dict[str, float]
    constraints: dict[str, AssetLimit] = Field(default_factory=dict)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=30)


class WalkForwardConfig(Contract):
    window_mode: Literal["rolling", "expanding"] = "rolling"
    training_periods: int = Field(default=126, ge=20, le=2500, strict=True)
    validation_periods: int = Field(default=63, ge=20, le=1000, strict=True)


class DecisionPolicy(Contract):
    mode: Literal["scheduled"] = "scheduled"
    decision_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "monthly"
    execution_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = "daily"
    cost_basis: Literal["half_turnover", "gross_traded_weight"] = "half_turnover"
    execution_lag: int = Field(default=1, ge=1, le=250, strict=True)
    min_holding_periods: int = Field(default=0, ge=0, le=2500, strict=True)
    deviation_threshold: float = Field(default=0, ge=0, le=1, strict=True)


class DatedSignal(Contract):
    observed_on: date
    available_on: date
    expires_on: date
    values: dict[str, Annotated[float, Field(strict=True)]]

    @model_validator(mode="after")
    def dated(self):
        if not self.observed_on <= self.available_on <= self.expires_on:
            raise ValueError("信号日期须满足观察日 ≤ 可得日 ≤ 到期日。")
        if not self.values or any(isinstance(v, bool) or not -1 <= v <= 1 for v in self.values.values()):
            raise ValueError("标准化信号须完整且位于 [-1, 1]，0 表示中性。")
        return self


class SignalComponent(Contract):
    id: str = Field(min_length=1, max_length=80)
    kind: Literal["momentum", "value", "carry", "macro", "risk_sentiment"]
    weight: float = Field(ge=0, le=1, strict=True)
    lookback: int = Field(default=60, ge=2, le=1000, strict=True)
    max_age_days: int = Field(default=31, ge=1, le=3650, strict=True)
    source: str = Field(min_length=1, max_length=500)
    methodology: str = Field(min_length=1, max_length=2000)
    unit: Literal["standardized_score_minus1_plus1"] = "standardized_score_minus1_plus1"
    observations: list[DatedSignal] = Field(default_factory=list, max_length=2000)

    @model_validator(mode="after")
    def documented(self):
        if not self.source.strip() or not self.methodology.strip():
            raise ValueError("每个信号须填写来源和标准化研究方法。")
        if self.kind != "momentum" and not self.observations:
            raise ValueError("外部信号必须提供真实已研究的日期化标准值。")
        if self.kind == "momentum" and self.observations:
            raise ValueError("动量窗口不接受外部观测值。")
        if len({row.available_on for row in self.observations}) != len(self.observations):
            raise ValueError("同一信号不能有重复可得日。")
        return self


class PreviewRequest(Contract):
    baseline_id: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    as_of: date
    train_end_date: date
    signal_mode: Literal["momentum", "manual", "regime", "composite"] = "momentum"
    lookback: int = Field(default=60, ge=2, le=1000)
    manual_tilts: dict[str, float] = Field(default_factory=dict)
    regime_run_id: str | None = Field(default=None, max_length=120)
    state_tilts: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_abs_tilt: float = Field(default=.1, ge=0, le=1)
    transaction_cost_bps: float = Field(default=10, ge=0, le=1000)
    risk_penalty: float = Field(default=3, ge=0, le=1000)
    max_tracking_error: float = Field(default=.1, ge=0, le=10)
    max_turnover: float = Field(default=1, gt=0, le=1)
    confidence_floor: float = Field(default=.6, ge=0, le=1)
    max_signal_age_days: int = Field(default=31, ge=1, le=3650)
    search: bool = True
    selected_candidate_id: str | None = Field(default=None, max_length=80)
    objective: Literal["active_utility", "excess_return", "min_drawdown"] = "active_utility"
    current_weights: dict[str, float] | None = None
    review_days: int = Field(default=30, ge=1, le=365)
    note: str = Field(default="", max_length=2000)
    walk_forward: WalkForwardConfig | None = None
    decision_policy: DecisionPolicy | None = None
    signal_components: list[SignalComponent] = Field(default_factory=list, max_length=8)
    current_weights_as_of: date | None = None
    last_execution_date: date | None = None

    @model_validator(mode="after")
    def ordered_dates(self):
        if not self.start_date < self.train_end_date < self.end_date <= self.as_of:
            raise ValueError("日期须满足：开始日 < 训练截止日 < 回测截止日 ≤ 研究时点。")
        if self.as_of > date.today():
            raise ValueError("研究时点不能位于未来。")
        if self.signal_mode == "regime" and not self.regime_run_id:
            raise ValueError("请先选择已发布的实时市场状态。")
        if self.signal_mode == "composite":
            if not self.signal_components or abs(sum(c.weight for c in self.signal_components) - 1) > 1e-8:
                raise ValueError("组合信号权重须明确且合计为 1。")
            if len({c.id for c in self.signal_components}) != len(self.signal_components):
                raise ValueError("组合信号 ID 不能重复。")
        if self.current_weights_as_of and self.current_weights_as_of != self.as_of:
            raise ValueError("实际持仓时点须与本次研究日一致。")
        if self.last_execution_date and self.last_execution_date > self.as_of:
            raise ValueError("最近执行日不能晚于研究日。")
        return self


class Scenario(Contract):
    kind: Literal["shock", "historical"]
    name: str = Field(default="情景比较", min_length=1, max_length=120)
    shocks: dict[str, float] = Field(default_factory=dict)
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def named(self):
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("请给情景填写名称。")
        return self


class ScenarioRequest(Contract):
    preview_request: PreviewRequest
    candidate_id: str | None = None
    scenario: Scenario


class SaveDecisionRequest(Contract):
    request: PreviewRequest
    preview_hash: str = Field(min_length=64, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=2000)
    scenarios: list[Scenario] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def unique_scenarios(self):
        if len({item.name for item in self.scenarios}) != len(self.scenarios):
            raise ValueError("情景名称不能重复，请用名称区分不同实验。")
        return self
