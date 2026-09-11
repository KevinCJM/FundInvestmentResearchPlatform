"""Contracts for reusable factor scenarios and applications of published models."""
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from backend.sensitivity.contracts import Contract, Frequency


class ScenarioFields(Contract):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    entry: Literal["event", "macro", "market"] = "market"
    frequency: Frequency = "monthly"
    event_template: Literal["energy", "policy", "credit", "custom"] = "custom"
    event_model_release_id: str | None = Field(default=None, max_length=120)
    macro_model_release_id: str | None = Field(default=None, max_length=120)
    input_ids: list[str] = Field(default_factory=list, max_length=8)
    # Conventional display units: percentages for returns; bp/pp/points otherwise.
    rows: list[list[float]] = Field(min_length=1, max_length=1200)
    shock_basis: Literal["period_change"] = "period_change"

    @model_validator(mode="after")
    def chain(self):
        if any(len(row) < 1 or len(row) > 8 for row in self.rows):
            raise ValueError("每期须包含 1 至 8 个输入变化")
        if self.entry == "market":
            if not self.input_ids or self.event_model_release_id or self.macro_model_release_id:
                raise ValueError("直接市场冲击须选择因子，不引用宏观模型")
        elif self.input_ids or not self.macro_model_release_id:
            raise ValueError("宏观入口由已发布模型决定输入变量，必须选择宏观到市场模型")
        if (self.entry == "event") != bool(self.event_model_release_id):
            raise ValueError("只有宏观事件入口需要事件到宏观模型")
        if len(set(self.input_ids)) != len(self.input_ids):
            raise ValueError("风险因子不能重复")
        return self


class ScenarioPublish(Contract):
    definition: ScenarioFields
    preview_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    note: str = Field(default="", max_length=1000)
    valid_days: int = Field(default=90, ge=1, le=365, strict=True)
    acknowledge_limitations: Literal[True]


class ImpactTarget(Contract):
    kind: Literal["product", "portfolio_run"]
    product_key: str | None = Field(default=None, max_length=120)
    portfolio_run_id: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def identity(self):
        if self.kind == "product" and (not self.product_key or self.portfolio_run_id):
            raise ValueError("产品压测须选择产品，不选择组合快照")
        if self.kind == "portfolio_run" and (not self.portfolio_run_id or self.product_key):
            raise ValueError("组合压测须选择不可变组合运行，不直接填产品")
        return self


class ImpactRequest(Contract):
    scenario_release_id: str = Field(min_length=1, max_length=120)
    exposure_release_id: str = Field(min_length=1, max_length=120)
    target: ImpactTarget
    as_of: date
    holding_policy: Literal["buy_and_hold", "constant_weights_zero_cost"] = "buy_and_hold"
    hold_other_factors_constant: Literal[True]
    notional: float = Field(default=1.0, gt=0, le=1e15)
    usage: Literal["research"] = "research"
