"""Explicit research contracts; no client-provided performance or PIT waivers."""
from __future__ import annotations

from datetime import date
from typing import Literal

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


class PreviewRequest(Contract):
    baseline_id: str = Field(min_length=1, max_length=120)
    start_date: date
    end_date: date
    as_of: date
    train_end_date: date
    signal_mode: Literal["momentum", "manual", "regime"] = "momentum"
    lookback: int = Field(default=60, ge=2, le=1000)
    manual_tilts: dict[str, float] = Field(default_factory=dict)
    regime_run_id: str | None = Field(default=None, max_length=120)
    state_tilts: dict[str, dict[str, float]] = Field(default_factory=dict)
    max_abs_tilt: float = Field(default=.1, ge=0, le=1)
    transaction_cost_bps: float = Field(default=10, ge=0, le=1000)
    risk_penalty: float = Field(default=3, ge=0, le=1000)
    max_tracking_error: float = Field(default=.1, gt=0, le=10)
    max_turnover: float = Field(default=1, gt=0, le=1)
    confidence_floor: float = Field(default=.6, ge=0, le=1)
    max_signal_age_days: int = Field(default=31, ge=1, le=3650)
    search: bool = True
    selected_candidate_id: str | None = Field(default=None, max_length=80)
    objective: Literal["active_utility", "excess_return", "min_drawdown"] = "active_utility"
    current_weights: dict[str, float] | None = None
    review_days: int = Field(default=30, ge=1, le=365)
    note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def ordered_dates(self):
        if not self.start_date < self.train_end_date < self.end_date <= self.as_of:
            raise ValueError("日期须满足：开始日 < 训练截止日 < 回测截止日 ≤ 研究时点。")
        if self.as_of > date.today():
            raise ValueError("研究时点不能位于未来。")
        if self.signal_mode == "regime" and not self.regime_run_id:
            raise ValueError("请先选择已发布的实时市场状态。")
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
