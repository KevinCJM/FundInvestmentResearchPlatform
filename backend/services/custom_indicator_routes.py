"""FastAPI routes for the custom indicator research workflow."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field, StrictFloat, model_validator
from starlette.background import BackgroundTask

from custom_indicators.errors import IndicatorDomainError
from custom_indicators.graph_contracts import EditorStateUpdate, GraphResolveRequest
from custom_indicators.graph_service import IndicatorGraphService
from custom_indicators.series_parameters import inspect_parameter_inputs, bind_parameter_input
from custom_indicators.service import CustomIndicatorService, MAX_PLAN_TARGETS, SUPPORTED_PERIODS
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


router = APIRouter(tags=["custom-indicators"], route_class=StableValidationRoute)
indicator_service = CustomIndicatorService()


Direction = Literal["higher_better", "lower_better"]
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


def _all_supported_periods() -> list[str]:
    return list(SUPPORTED_PERIODS)


class SeriesParameterDefinition(BaseModel):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    label: str = Field(min_length=1, max_length=80)
    type: Literal["integer", "number"]
    default: StrictFloat
    minimum: StrictFloat
    maximum: StrictFloat
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
    transform_version: Optional[Literal["1.0.0"]] = None
    version: Optional[Literal["1.0.0"]] = None
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


class PrepareEvaluationRequest(IndependentRequest):
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)
    indicator_refs: list[IndicatorReference] = Field(default_factory=list, max_length=10)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)


class EvaluateRequest(IndependentRequest):
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)
    indicator_refs: list[IndicatorReference] = Field(default_factory=list, max_length=10)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)
    targets: list[EvaluationTarget] = Field(min_length=1, max_length=50)
    period: str
    as_of: Optional[str] = None
    include_series: bool = False


class SeriesIndicatorInstance(BaseModel):
    indicator_id: Optional[str] = Field(default=None, min_length=1, max_length=120)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    inline_definition: Optional[IndicatorDraft] = None
    compile_token: Optional[str] = Field(default=None, min_length=64, max_length=64)
    parameters: dict[str, StrictFloat] = Field(default_factory=dict)


class EvaluateSeriesRequest(BaseModel):
    indicator_instances: list[SeriesIndicatorInstance] = Field(min_length=1, max_length=10)
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


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except IndicatorDomainError as exc:
        headers = {"Retry-After": "1"} if exc.code == "INDICATOR_ENGINE_BUSY" else None
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail(),
            headers=headers,
        ) from exc


@router.get("/api/custom-indicators/meta")
def custom_indicator_meta():
    return _call(indicator_service.meta)


@router.post("/api/custom-indicators/validate")
def validate_custom_indicator(request: ValidateRequest):
    return _call(indicator_service.validate, request.model_dump())


@router.post("/api/custom-indicators/parameters/inspect")
def inspect_custom_indicator_parameters(request: ParameterInspectRequest):
    return _call(inspect_parameter_inputs, request.definition.model_dump())


@router.post("/api/custom-indicators/parameters/bind")
def bind_custom_indicator_parameter(request: ParameterBindRequest):
    definition = _call(bind_parameter_input, request.definition.model_dump(),
                       candidate_id=request.candidate_id, parameter_id=request.parameter_id,
                       fixed_parameter_id=request.fixed_parameter_id)
    return {"definition": definition, **_call(inspect_parameter_inputs, definition)}


@router.post("/api/custom-indicators/compose")
def compose_custom_indicator(request: ComposeRequest):
    return _call(indicator_service.compose, request.model_dump())


@router.post("/api/custom-indicators/infer")
def infer_custom_indicator(request: InferRequest):
    return _call(indicator_service.infer, request.model_dump())


@router.post("/api/custom-indicators/graph/resolve")
def resolve_indicator_graph(request: GraphResolveRequest):
    return _call(IndicatorGraphService(indicator_service).resolve, request)


@router.get("/api/custom-indicators/{indicator_id}/editor-state")
def get_indicator_editor_state(indicator_id: str, revision: int = Query(ge=1)):
    return _call(IndicatorGraphService(indicator_service).read_state, indicator_id, revision)


@router.put("/api/custom-indicators/{indicator_id}/editor-state")
def put_indicator_editor_state(indicator_id: str, request: EditorStateUpdate, revision: int = Query(ge=1)):
    return _call(IndicatorGraphService(indicator_service).save_state, indicator_id, revision, request)


@router.post("/api/custom-indicators/derive-rolling-series")
def derive_rolling_custom_indicator(request: DeriveRollingSeriesRequest):
    return _call(
        indicator_service.derive_rolling_series,
        indicator_id=request.indicator_id,
        indicator_revision=request.indicator_revision,
        window_observations=request.window_observations,
        name=request.name,
        description=request.description,
    )


@router.post("/api/custom-indicators/rolling-scalar-draft")
def build_rolling_scalar_draft(request: RollingScalarDraftRequest):
    return _call(
        indicator_service.build_rolling_scalar_draft,
        request.indicator_id,
        request.indicator_revision,
        request.window_observations,
        request.min_periods,
        request.name,
    )


@router.post("/api/custom-indicators/availability", include_in_schema=False)
@router.post("/api/custom-indicators/variables/availability")
def custom_indicator_availability(request: AvailabilityRequest):
    return _call(
        indicator_service.availability,
        kind=request.kind,
        product_id=request.product_id,
        targets=[target.model_dump() for target in request.targets] or None,
        variable_ids=request.variable_ids or None,
        period=request.period,
        as_of=request.as_of,
    )


@router.post("/api/custom-indicators/prepare")
def prepare_custom_indicator_evaluation(request: PrepareEvaluationRequest):
    return _call(
        indicator_service.prepare_evaluation,
        indicator_ids=request.indicator_ids,
        indicator_refs=[item.model_dump() for item in request.indicator_refs] or None,
        inline_definition=request.inline_definition.model_dump() if request.inline_definition else None,
        compile_token=request.compile_token,
    )


@router.post("/api/custom-indicators/evaluate")
def evaluate_custom_indicators(request: EvaluateRequest):
    inline = request.inline_definition.model_dump() if request.inline_definition else None
    return _call(
        indicator_service.evaluate,
        indicator_ids=request.indicator_ids,
        indicator_refs=[item.model_dump() for item in request.indicator_refs] or None,
        inline_definition=inline,
        targets=[target.model_dump() for target in request.targets],
        period=request.period,
        as_of=request.as_of,
        include_series=request.include_series,
        compile_token=request.compile_token,
    )


@router.post("/api/custom-indicators/evaluate-series")
def evaluate_custom_indicator_series(request: EvaluateSeriesRequest):
    return _call(
        indicator_service.evaluate_series,
        indicator_instances=[
            {
                **item.model_dump(exclude={"inline_definition"}),
                "inline_definition": (
                    item.inline_definition.model_dump()
                    if item.inline_definition is not None
                    else None
                ),
            }
            for item in request.indicator_instances
        ],
        target=request.target.model_dump(),
        period=request.period,
        as_of=request.as_of,
        max_points=request.max_points,
    )


@router.post("/api/custom-indicators/export-excel")
def export_custom_indicator_excel(request: ExportExcelRequest):
    inline = request.inline_definition.model_dump() if request.inline_definition else None
    artifact = _call(
        indicator_service.export_excel,
        indicator_ids=request.indicator_ids,
        inline_definition=inline,
        targets=[target.model_dump() for target in request.targets],
        period=request.period,
        as_of=request.as_of,
        compile_token=request.compile_token,
        parameters=request.parameters,
    )
    return FileResponse(
        path=artifact.path,
        media_type=artifact.media_type,
        filename=artifact.filename,
        headers={"Cache-Control": "no-store"},
        background=BackgroundTask(artifact.cleanup),
    )


@router.post("/api/custom-indicators/evaluate-portfolio")
def evaluate_portfolio_custom_indicators(request: EvaluatePortfolioRequest):
    inline = request.inline_definition.model_dump() if request.inline_definition else None
    return _call(
        indicator_service.evaluate_portfolio,
        run_id=request.run_id,
        indicator_ids=request.indicator_ids,
        inline_definition=inline,
        compile_token=request.compile_token,
    )


@router.get("/api/custom-indicators")
def list_custom_indicators(
    context_kind: Optional[ContextKind] = None,
    product_kind: Optional[Literal["etf", "fund", "portfolio"]] = None,
    source: Optional[Literal["built_in", "custom"]] = None,
    indicator_type: Optional[IndicatorType] = None,
    category: Optional[str] = Query(default=None, max_length=80),
    include_compatibility: bool = False,
):
    return _call(
        indicator_service.list_indicators,
        context_kind=context_kind,
        product_kind=product_kind,
        source=source,
        category=indicator_type or category,
        include_compatibility=include_compatibility,
    )


@router.post("/api/custom-indicators", status_code=status.HTTP_201_CREATED)
def create_custom_indicator(request: IndicatorDraft):
    return _call(indicator_service.create_indicator, request.model_dump())


@router.get("/api/custom-indicators/snapshot-config")
def get_snapshot_indicator_config():
    return _call(indicator_service.get_snapshot_config)


@router.put("/api/custom-indicators/snapshot-config")
def update_snapshot_indicator_config(request: SnapshotIndicatorConfigUpdate):
    return _call(
        indicator_service.update_snapshot_config,
        request.revision,
        [item.model_dump() for item in request.items],
    )


@router.get("/api/custom-indicators/{indicator_id}")
def get_custom_indicator(indicator_id: str):
    return _call(indicator_service.get_indicator, indicator_id)


@router.put("/api/custom-indicators/{indicator_id}")
def update_custom_indicator(indicator_id: str, request: IndicatorUpdate):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(indicator_service.update_indicator, indicator_id, revision, payload)


@router.delete("/api/custom-indicators/{indicator_id}")
def delete_custom_indicator(indicator_id: str, revision: int = Query(ge=1)):
    _call(indicator_service.delete_indicator, indicator_id, revision)
    return {"deleted_id": indicator_id}


@router.get("/api/evaluation-plans")
def list_evaluation_plans(kind: Optional[Literal["etf", "fund"]] = None):
    return _call(indicator_service.list_plans, kind)


@router.get("/api/evaluation-plan-runs/{result_id}")
def get_evaluation_plan_run(
    result_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=500),
):
    return _call(
        indicator_service.get_plan_run_result,
        result_id,
        page=page,
        page_size=page_size,
    )


@router.post("/api/evaluation-plans", status_code=status.HTTP_201_CREATED)
def create_evaluation_plan(request: EvaluationPlanDraft):
    return _call(indicator_service.create_plan, request.model_dump())


@router.get("/api/evaluation-plans/{plan_id}")
def get_evaluation_plan(plan_id: str):
    return _call(indicator_service.get_plan, plan_id)


@router.put("/api/evaluation-plans/{plan_id}")
def update_evaluation_plan(plan_id: str, request: EvaluationPlanUpdate):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(indicator_service.update_plan, plan_id, revision, payload)


@router.delete("/api/evaluation-plans/{plan_id}")
def delete_evaluation_plan(plan_id: str, revision: int = Query(ge=1)):
    _call(indicator_service.delete_plan, plan_id, revision)
    return {"deleted_id": plan_id}


@router.post("/api/evaluation-plans/{plan_id}/run")
def run_evaluation_plan(plan_id: str, request: Optional[PlanRunRequest] = None):
    return _call(indicator_service.run_plan, plan_id, request.as_of if request else None)
