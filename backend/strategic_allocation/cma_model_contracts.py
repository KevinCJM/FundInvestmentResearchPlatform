"""Opt-in CMA models; no changes to the existing manual CMA contract."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .common_contracts import Contract, Currency, Identifier, Number

MatrixRow = Annotated[list[Number], Field(min_length=1, max_length=30)]
Matrix = Annotated[list[MatrixRow], Field(min_length=1, max_length=30)]
Source = Annotated[str, Field(min_length=3, max_length=2000)]


def _matrix_axis(matrix: list[list[float]], count: int) -> None:
    if len(matrix) != count or any(len(row) != count for row in matrix):
        raise ValueError("CMA_MODEL_MATRIX_AXIS: 协方差行列须按完整资产轴排列。")


class CmaModelContext(Contract):
    asset_ids: list[Identifier] = Field(min_length=1, max_length=30)
    as_of: date
    currency: Currency
    return_basis: Literal["annual_arithmetic_total_return"] = "annual_arithmetic_total_return"
    source: Source

    @model_validator(mode="after")
    def context(self):
        if len(set(self.asset_ids)) != len(self.asset_ids):
            raise ValueError("CMA_MODEL_AXIS: 资产标识不得重复。")
        if self.as_of > date.today():
            raise ValueError("CMA_MODEL_DATE: 研究日不能位于未来。")
        return self


class BlackLittermanView(Contract):
    kind: Literal["absolute", "relative"]
    asset_id: Identifier
    relative_to: Identifier | None = None
    annual_return: Number
    view_std: Number = Field(gt=0)
    observed_on: date
    available_on: date
    source: Source

    @model_validator(mode="after")
    def view(self):
        if self.kind == "absolute":
            if self.relative_to is not None or not -0.5 <= self.annual_return <= 2:
                raise ValueError("CMA_BL_ABSOLUTE: 绝对总收益须在 -50% 至 200%，且无比较资产。")
        elif (not self.relative_to or self.relative_to == self.asset_id
              or not -2.5 <= self.annual_return <= 2.5):
            raise ValueError("CMA_BL_RELATIVE: 相对观点须选择另一资产，收益差在 ±250 个百分点内。")
        if self.observed_on > self.available_on:
            raise ValueError("CMA_BL_VIEW_DATE: 观察日不能晚于可得日。")
        return self


class BlackLittermanRequest(CmaModelContext):
    method: Literal["black_litterman"]
    covariance: Matrix
    risk_covariance_basis: Literal["input_covariance"]
    market_weights: dict[Identifier, Number] = Field(min_length=1, max_length=30)
    market_weight_source: Source
    delta: Number = Field(gt=0)
    tau: Number = Field(gt=0)
    risk_free_rate: Number = Field(ge=-0.5, le=2)
    views: list[BlackLittermanView] = Field(default_factory=list, max_length=60)

    @model_validator(mode="after")
    def inputs(self):
        _matrix_axis(self.covariance, len(self.asset_ids))
        if (set(self.market_weights) != set(self.asset_ids)
                or any(w < 0 or w > 1 for w in self.market_weights.values())
                or abs(sum(self.market_weights.values()) - 1) > 1e-8):
            raise ValueError("CMA_BL_MARKET_WEIGHTS: 明确市场权重须完整、非负并合计100%。")
        for view in self.views:
            if view.asset_id not in self.asset_ids or (view.relative_to and view.relative_to not in self.asset_ids):
                raise ValueError("CMA_BL_VIEW_AXIS: 观点引用了未知资产。")
            if view.available_on > self.as_of:
                raise ValueError("CMA_BL_VIEW_UNAVAILABLE: 观点在研究日尚不可得。")
        return self


class CmaScenario(Contract):
    id: Identifier
    probability: Number = Field(ge=0, le=1)
    annual_returns: dict[Identifier, Number] = Field(min_length=1, max_length=30)
    covariance: Matrix | None = None
    source: Source


class ScenarioMixtureRequest(CmaModelContext):
    method: Literal["scenario_mixture"]
    risk_mode: Literal["shared", "scenario_specific"]
    shared_covariance: Matrix | None = None
    scenarios: list[CmaScenario] = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def inputs(self):
        if len({s.id for s in self.scenarios}) != len(self.scenarios):
            raise ValueError("CMA_SCENARIO_ID: 情景标识不得重复。")
        if abs(sum(s.probability for s in self.scenarios) - 1) > 1e-8:
            raise ValueError("CMA_SCENARIO_PROBABILITY: 显式情景概率须合计100%，不自动归一化。")
        if self.risk_mode == "shared":
            if self.shared_covariance is None or any(s.covariance is not None for s in self.scenarios):
                raise ValueError("CMA_SCENARIO_RISK: 共用风险须显式提供一个矩阵，不能混入逐情景风险。")
            _matrix_axis(self.shared_covariance, len(self.asset_ids))
        elif self.shared_covariance is not None or any(s.covariance is None for s in self.scenarios):
            raise ValueError("CMA_SCENARIO_RISK: 逐情景风险须全部提供，不能隐式共用。")
        for scenario in self.scenarios:
            if set(scenario.annual_returns) != set(self.asset_ids):
                raise ValueError("CMA_SCENARIO_AXIS: 每个情景须完整覆盖同一资产轴。")
            if any(not -0.5 <= r <= 2 for r in scenario.annual_returns.values()):
                raise ValueError("CMA_MODEL_RETURN_RANGE: 年化总收益须在 -50% 至 200%。")
            if scenario.covariance is not None:
                _matrix_axis(scenario.covariance, len(self.asset_ids))
        return self


CmaModelRequest = Annotated[BlackLittermanRequest | ScenarioMixtureRequest, Field(discriminator="method")]
CMA_MODEL_ADAPTER = TypeAdapter(CmaModelRequest)
