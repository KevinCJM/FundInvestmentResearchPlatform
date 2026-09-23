"""Request-only contracts for the custom indicator research workflow.

Importing this module must stay free of business side effects: it never
constructs the indicator service or touches a data directory.  Route modules
re-export these names so existing import paths keep working, while agent-side
modules can import pure request schemas without initialising business stores.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, StrictFloat, model_validator

from custom_indicators.series_service import MAX_SERIES_INSTANCES
from custom_indicators.service import (
    MAX_EVALUATION_INDICATORS,
    MAX_PLAN_TARGETS,
    SUPPORTED_PERIODS,
)
from cal_indicators.typed_operators import TYPED_DSL_VERSION


class StableValidationRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request: Request):
            try:
                return await original(request)
            except RequestValidationError as exc:
                diagnostics = []
                for error in exc.errors():
                    location = [str(item) for item in error.get("loc", []) if item != "body"]
                    diagnostics.append(
                        {
                            "code": str(error.get("type", "invalid_request")),
                            "message": str(error.get("msg", "请求参数无效。")),
                            "field": ".".join(location) or None,
                        }
                    )
                field = diagnostics[0]["field"] if diagnostics else None
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": {
                            "code": "REQUEST_VALIDATION_ERROR",
                            "message": "请求参数无效。",
                            "field": field,
                            "diagnostics": diagnostics,
                        }
                    },
                )

        return handler


Direction = Literal["higher_better", "lower_better"]


def _all_supported_periods() -> list[str]:
    return list(SUPPORTED_PERIODS)


InstrumentKind = Literal["etf", "fund"]
ContextKind = Literal["single_product", "portfolio"]
IndicatorType = Literal[
    "return",
    "risk",
    "risk_adjusted",
    "path",
    "market_liquidity",
    "technical",
    "other",
]


class SeriesParameterDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = Field(min_length=1, max_length=80)
    type: Literal["integer", "number"]
    default: StrictFloat
    minimum: StrictFloat
    maximum: StrictFloat
    exclusive_minimum: bool = False
    exclusive_maximum: bool = False
    step: StrictFloat = Field(default=1.0, gt=0)
    description: str = Field(default="", max_length=300)


class FixedSeriesParameter(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = Field(min_length=1, max_length=80)
    type: Literal["integer", "number"]
    value: float
    source: str = Field(default="definition", max_length=80)


class IndependentRequest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def reject_retired_results(cls, values):
        if isinstance(values, dict) and {"scalar_outputs", "output_id"}.intersection(values):
            raise ValueError("请使用独立指标ID；旧多结果定义或子结果引用需要先迁移。")
        return values


class RollingSourceDefinition(IndependentRequest):
    """Version-locked scalar source for a generated rolling series.

    ``transform_version`` / ``definition_hash`` are the canonical public
    fields.  The older ``version`` / ``source_definition_hash`` names remain
    optional so previously generated drafts can still be validated and saved.
    """

    kind: Literal["rolling_scalar"] = "rolling_scalar"
    transform_version: Optional[Literal["1.0.0", "2.0.0", "3.0.0"]] = None
    version: Optional[Literal["1.0.0", "2.0.0", "3.0.0"]] = None
    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: int = Field(ge=1)
    indicator_name: str = Field(default="", max_length=80)
    definition_hash: Optional[str] = Field(default=None, min_length=64, max_length=64)
    source_definition_hash: Optional[str] = Field(default=None, min_length=64, max_length=64)
    source_dsl_version: str = Field(default="", max_length=40)
    window_observations: int = Field(ge=1, le=5_000)
    minimum_observations: Optional[int] = Field(default=None, ge=1, le=5_000)
    detached: bool = False


class SeriesOutputDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = Field(min_length=1, max_length=80)
    expression: str = Field(min_length=1, max_length=4000)
    unit: str = Field(default="", max_length=20)
    display_format: Literal["number", "percent"] = "number"
    precision: int = Field(default=4, ge=0, le=8)
    output_measure: str = Field(default="dimensionless", min_length=1, max_length=120)


class ParameterContract(IndependentRequest):
    parameter_contract_version: Optional[Literal["1.0"]] = None


class IndicatorDraft(ParameterContract):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    expression: str = Field(default="", max_length=4000)
    periods: Optional[list[str]] = None
    period_policy: Literal["all_supported"] = "all_supported"
    unit: str = Field(default="", max_length=20)
    display_format: Literal["number", "percent", "date"] = "number"
    precision: int = Field(default=2, ge=0, le=8)
    direction: Literal["neutral", "higher_better", "lower_better"] = "neutral"
    indicator_type: IndicatorType = "other"
    annual_risk_free_rate_percent: float = Field(default=0.0, ge=-100.0, le=100.0)
    dsl_version: str = TYPED_DSL_VERSION
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None
    variable_registry_version: Optional[str] = None
    data_contract_version: Optional[str] = None
    context_schema_version: Optional[str] = None
    context_kind: ContextKind = "single_product"
    result_kind: Literal["scalar", "time_series"] = "scalar"
    output_contract: Literal["scalar", "series_bundle"] = "scalar"
    output_measure: Optional[str] = None
    parameter_schema: list[SeriesParameterDefinition] = Field(default_factory=list, max_length=16)
    fixed_parameters: list[FixedSeriesParameter] = Field(default_factory=list, max_length=16)
    series_outputs: list[SeriesOutputDefinition] = Field(default_factory=list, max_length=8)
    axis_anchor: Optional[str] = Field(default=None, max_length=80)
    history_policy: Optional[Literal["lookback", "full_history"]] = None
    lookback_parameter: Optional[str] = Field(default=None, max_length=80)
    minimum_observations: int = Field(default=1, ge=1, le=20_000)
    methodology: str = Field(default="", max_length=500)
    data_basis: str = Field(default="", max_length=500)
    template_origin: Any = None
    rolling_source: Optional[RollingSourceDefinition] = None
    rolling_transform: Any = None


class IndicatorUpdate(IndicatorDraft):
    revision: int = Field(ge=1)


class ValidateRequest(IndicatorDraft):
    name: str = Field(default="未保存指标", max_length=80)


class ParameterInspectRequest(BaseModel):
    definition: ValidateRequest


class ParameterBindRequest(ParameterInspectRequest):
    candidate_id: Optional[str] = Field(default=None, max_length=200)
    parameter_id: Optional[str] = Field(default=None, max_length=64)
    fixed_parameter_id: Optional[str] = Field(default=None, max_length=64)


class DeriveRollingSeriesRequest(IndependentRequest):
    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: int = Field(ge=1)
    window_observations: int = Field(ge=1, le=5_000)
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    description: Optional[str] = Field(default=None, max_length=500)


class RollingScalarDraftRequest(IndependentRequest):
    """Compatibility request for the canonical rolling-series derivation."""

    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    window_observations: int = Field(ge=1, le=5_000)
    min_periods: Optional[int] = Field(default=None, ge=1, le=5_000)
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)


class ComposeArgument(BaseModel):
    parameter: str = Field(min_length=1, max_length=80)
    source: Literal["variable", "constant", "expression"]
    value: str | float


class ComposeRequest(IndependentRequest):
    operator_id: Optional[str] = None
    template_id: Optional[str] = None
    indicator_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    arguments: list[ComposeArgument] = Field(default_factory=list, max_length=16)
    context: ContextKind = "single_product"
    dsl_version: Optional[str] = None
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None
    variable_registry_version: Optional[str] = None
    data_contract_version: Optional[str] = None
    context_schema_version: Optional[str] = None
    parameter_schema: list[SeriesParameterDefinition] = Field(default_factory=list, max_length=16)


class InferRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=4000)
    context: ContextKind = "single_product"
    dsl_version: str = TYPED_DSL_VERSION
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None
    parameter_schema: list[SeriesParameterDefinition] = Field(default_factory=list, max_length=16)


class EvaluationTarget(BaseModel):
    kind: InstrumentKind
    product_id: str = Field(min_length=1, max_length=100)


class AvailabilityRequest(BaseModel):
    kind: Optional[InstrumentKind] = None
    product_id: Optional[str] = Field(default=None, min_length=1, max_length=100)
    targets: list[EvaluationTarget] = Field(default_factory=list, max_length=10)
    variable_ids: list[str] = Field(default_factory=list, max_length=64)
    period: str = "1Y"
    as_of: Optional[str] = None


class IndicatorReference(IndependentRequest):
    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)


class PrepareEvaluationRequest(IndependentRequest):
    indicator_ids: list[str] = Field(default_factory=list, max_length=MAX_EVALUATION_INDICATORS)
    indicator_refs: list[IndicatorReference] = Field(default_factory=list, max_length=MAX_EVALUATION_INDICATORS)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)


class EvaluateRequest(IndependentRequest):
    indicator_ids: list[str] = Field(default_factory=list, max_length=MAX_EVALUATION_INDICATORS)
    indicator_refs: list[IndicatorReference] = Field(default_factory=list, max_length=MAX_EVALUATION_INDICATORS)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)
    targets: list[EvaluationTarget] = Field(min_length=1, max_length=50)
    period: str
    as_of: Optional[str] = None
    include_series: bool = False
    # Applies to inline_definition only; a saved indicator carries its values
    # on its own indicator_refs entry.
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)


class SeriesIndicatorInstance(BaseModel):
    indicator_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)
    # Caller-owned name echoed on the matching result. The same indicator may be
    # requested several times with different parameters, so indicator_id alone
    # no longer identifies which result belongs to which instance.
    instance_key: Optional[str] = Field(default=None, min_length=1, max_length=64)


class EvaluateSeriesRequest(BaseModel):
    indicator_instances: list[SeriesIndicatorInstance] = Field(min_length=1, max_length=MAX_SERIES_INSTANCES)
    target: EvaluationTarget
    period: str
    as_of: Optional[str] = None
    max_points: int = Field(default=5000, ge=1, le=5000)


class ExportExcelRequest(IndependentRequest):
    indicator_ids: list[str] = Field(default_factory=list, max_length=1)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)
    targets: list[EvaluationTarget] = Field(min_length=1, max_length=10)
    period: str
    as_of: Optional[str] = None
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)


class SnapshotIndicatorItem(IndependentRequest):
    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: int = Field(ge=1)
    period: str = Field(min_length=1, max_length=12)
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)
    channel_id: Optional[str] = Field(default=None, min_length=1, max_length=80)
    reducer: Optional[Literal["last_finite"]] = None


class SnapshotIndicatorConfigUpdate(BaseModel):
    revision: int = Field(ge=1)
    items: list[SnapshotIndicatorItem] = Field(default_factory=list, max_length=30)


class EvaluatePortfolioRequest(IndependentRequest):
    run_id: str = Field(min_length=1, max_length=100)
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)


class PlanIndicatorInput(IndependentRequest):
    indicator_id: str = Field(min_length=1)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    period: str
    weight: float = Field(ge=0)
    direction: Optional[Direction] = None
    # Saved once with the plan and never overridden at run time, so the same
    # plan revision reproduces the same numbers.
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)


class PlanProductSelectionFilters(BaseModel):
    fund_type: list[str] = Field(default_factory=list, max_length=100)
    invest_type: list[str] = Field(default_factory=list, max_length=100)
    qdii_type: list[str] = Field(default_factory=list, max_length=100)
    market: list[str] = Field(default_factory=list, max_length=100)
    status: list[str] = Field(default_factory=list, max_length=100)
    management: list[str] = Field(default_factory=list, max_length=100)
    custodian: list[str] = Field(default_factory=list, max_length=100)


class PlanProductCondition(BaseModel):
    field: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_]+$")
    operator: Literal["gte", "lte", "gt", "lt", "eq"]
    value: str = Field(min_length=1, max_length=100)


class PlanProductSelection(BaseModel):
    query: str = Field(default="", max_length=100)
    filters: PlanProductSelectionFilters = Field(default_factory=PlanProductSelectionFilters)
    conditions: list[PlanProductCondition] = Field(default_factory=list, max_length=20)
    selection_mode: Literal["manual", "all_matching"] = "manual"


class EvaluationPlanDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    product_kind: Optional[Literal["etf", "fund"]] = None
    indicators: list[PlanIndicatorInput] = Field(min_length=1, max_length=10)
    targets: list[EvaluationTarget] = Field(min_length=1, max_length=MAX_PLAN_TARGETS)
    product_selection: Optional[PlanProductSelection] = None
    missing_policy: Literal["strict"] = "strict"


class EvaluationPlanUpdate(EvaluationPlanDraft):
    revision: int = Field(ge=1)


class PlanRunRequest(BaseModel):
    as_of: Optional[str] = None


__all__ = [
    "AvailabilityRequest",
    "ComposeArgument",
    "ComposeRequest",
    "ContextKind",
    "DeriveRollingSeriesRequest",
    "Direction",
    "EvaluatePortfolioRequest",
    "EvaluateRequest",
    "EvaluateSeriesRequest",
    "EvaluationPlanDraft",
    "EvaluationPlanUpdate",
    "EvaluationTarget",
    "ExportExcelRequest",
    "FixedSeriesParameter",
    "IndependentRequest",
    "IndicatorDraft",
    "IndicatorReference",
    "IndicatorType",
    "IndicatorUpdate",
    "InferRequest",
    "InstrumentKind",
    "ParameterBindRequest",
    "ParameterContract",
    "ParameterInspectRequest",
    "PlanIndicatorInput",
    "PlanProductCondition",
    "PlanProductSelection",
    "PlanProductSelectionFilters",
    "PlanRunRequest",
    "PrepareEvaluationRequest",
    "RollingScalarDraftRequest",
    "RollingSourceDefinition",
    "SeriesIndicatorInstance",
    "SeriesOutputDefinition",
    "SeriesParameterDefinition",
    "SnapshotIndicatorConfigUpdate",
    "SnapshotIndicatorItem",
    "StableValidationRoute",
    "ValidateRequest",
]
