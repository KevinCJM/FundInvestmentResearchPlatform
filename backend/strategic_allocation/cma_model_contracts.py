"""Opt-in CMA models; no changes to the existing manual CMA contract."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .common_contracts import Contract, Currency, Fingerprint, Identifier, Number
from .reference_contracts import ReferenceInputRequest

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


class BlackLittermanViewLeg(Contract):
    asset_id: Identifier
    coefficient: Number = Field(ge=-4, le=4)


class BlackLittermanBasketView(Contract):
    """A normalized total-return basket, not a leveraged portfolio order."""
    kind: Literal["basket"]
    basis: Literal["absolute", "relative"]
    legs: list[BlackLittermanViewLeg] = Field(min_length=1, max_length=30)
    annual_return: Number
    view_std: Number = Field(gt=0)
    observed_on: date
    available_on: date
    source: Source

    @model_validator(mode="after")
    def basket(self):
        if len({leg.asset_id for leg in self.legs}) != len(self.legs):
            raise ValueError("CMA_BL_VIEW_AXIS: 篮子资产不得重复。")
        gross = sum(abs(leg.coefficient) for leg in self.legs)
        total = sum(leg.coefficient for leg in self.legs)
        if not 1e-12 <= gross <= 4 or abs(total - (1 if self.basis == "absolute" else 0)) > 1e-10:
            raise ValueError("CMA_BL_VIEW_PICK: 篮子不能全零，绝对观点系数合计为1、相对观点为0，绝对值合计不超过4。")
        if not (-0.5 <= self.annual_return <= 2 if self.basis == "absolute" else -2.5 <= self.annual_return <= 2.5):
            raise ValueError("CMA_BL_VIEW_RETURN: 篮子观点收益超出所选口径范围。")
        if self.observed_on > self.available_on:
            raise ValueError("CMA_BL_VIEW_DATE: 观察日不能晚于可得日。")
        return self


BlackLittermanViewRequest = Annotated[
    BlackLittermanView | BlackLittermanBasketView, Field(discriminator="kind")]


class BlackLittermanRequest(CmaModelContext):
    method: Literal["black_litterman"]
    covariance: Matrix
    risk_covariance_basis: Literal["input_covariance"]
    market_weights: dict[Identifier, Number] = Field(min_length=1, max_length=30)
    market_weight_source: Source
    delta: Number = Field(gt=0)
    tau: Number = Field(gt=0)
    risk_free_rate: Number = Field(ge=-0.5, le=2)
    views: list[BlackLittermanViewRequest] = Field(default_factory=list, max_length=60)

    @model_validator(mode="after")
    def inputs(self):
        _matrix_axis(self.covariance, len(self.asset_ids))
        if (set(self.market_weights) != set(self.asset_ids)
                or any(w < 0 or w > 1 for w in self.market_weights.values())
                or abs(sum(self.market_weights.values()) - 1) > 1e-8):
            raise ValueError("CMA_BL_MARKET_WEIGHTS: 明确市场权重须完整、非负并合计100%。")
        for view in self.views:
            referenced = ([leg.asset_id for leg in view.legs] if view.kind == "basket"
                          else [view.asset_id, *([view.relative_to] if view.relative_to else [])])
            if any(asset not in self.asset_ids for asset in referenced):
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


class CmaWindow(Contract):
    kind: Literal["1Y", "2Y", "3Y", "5Y", "10Y", "common_since_inception", "custom"] = "5Y"
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def interval(self):
        if self.kind == "custom":
            if self.start_date is None or self.end_date is None or self.start_date >= self.end_date:
                raise ValueError("自定义历史窗口须提供有效的开始和结束日期。")
        elif self.start_date is not None or self.end_date is not None:
            raise ValueError("相对历史窗口不同时填写自定义日期。")
        return self


class CmaVersionRef(Contract):
    id: Identifier
    content_hash: Fingerprint


class StatisticalCmaContext(CmaModelContext):
    window: CmaWindow = Field(default_factory=CmaWindow)
    observation_frequency: Literal["daily"] = "daily"
    periods_per_year: Literal[252] = 252
    proxy_inputs: ReferenceInputRequest | None = None

    @model_validator(mode="after")
    def evidence_context(self):
        if self.currency != "CNY":
            raise ValueError("历史证据目前支持 CNY/SSE 日频，不自动转换币种。")
        if self.window.end_date is not None and self.window.end_date > self.as_of:
            raise ValueError("历史窗口结束日不能晚于研究日。")
        if self.proxy_inputs is not None and (self.proxy_inputs.as_of != self.as_of
                or self.proxy_inputs.currency != self.currency
                or [a.id for a in self.proxy_inputs.assets] != self.asset_ids):
            raise ValueError("研究代理须与 CMA 使用同一资产轴、日期和币种。")
        return self


class HistoricalCmaRequest(StatisticalCmaContext):
    method: Literal["historical_statistics"]
    shrinkage: Number = Field(default=0.1, ge=0, le=1)


class BayesianCmaRequest(StatisticalCmaContext):
    method: Literal["bayesian_niw"]
    prior_ref: CmaVersionRef
    prior_mode: Literal["recenter", "continue"] = "recenter"
    mean_prior_observations: Number | None = Field(default=None, gt=0, le=100000)
    covariance_prior_observations: Number | None = Field(default=None, gt=0, le=100000)
    data_reuse_acknowledged: bool = False

    @model_validator(mode="after")
    def prior_strength(self):
        supplied = (self.mean_prior_observations, self.covariance_prior_observations)
        if self.prior_mode == "recenter" and any(x is None for x in supplied):
            raise ValueError("新建 NIW 先验须明确均值和风险的日频等效观察数。")
        if self.prior_mode == "continue" and any(x is not None for x in supplied):
            raise ValueError("后验续更继承原信息量，不重复提供新先验强度。")
        return self


class RegimeCmaRequest(StatisticalCmaContext):
    method: Literal["historical_regime_occupancy"]
    run_ref: CmaVersionRef
    probabilities: dict[Identifier, Number] | None = Field(default=None, min_length=1, max_length=60)
    probability_reason: str = Field(default="", max_length=2000)
    shrinkage: Number = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def probability_contract(self):
        if self.probabilities is not None:
            if any(x < 0 or x > 1 for x in self.probabilities.values()) or abs(sum(self.probabilities.values()) - 1) > 1e-8:
                raise ValueError("应用概率须非负且合计 100%，不会自动归一化。")
            if len(self.probability_reason.strip()) < 5:
                raise ValueError("覆盖历史占用率须填写原因。")
        return self


CmaModelRequest = Annotated[
    BlackLittermanRequest | ScenarioMixtureRequest | HistoricalCmaRequest | BayesianCmaRequest | RegimeCmaRequest,
    Field(discriminator="method"),
]
CMA_MODEL_ADAPTER = TypeAdapter(CmaModelRequest)
