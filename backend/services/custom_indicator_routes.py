"""FastAPI routes for the custom indicator research workflow."""

from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from custom_indicators.errors import IndicatorDomainError
from custom_indicators.graph_contracts import EditorStateUpdate, GraphResolveRequest
from custom_indicators.graph_service import IndicatorGraphService
from custom_indicators.series_parameters import inspect_parameter_inputs, bind_parameter_input
from custom_indicators.service import CustomIndicatorService
from pit.context import resolve_request_context
from services.custom_indicator_contracts import (
    AvailabilityRequest,
    ComposeRequest,
    ContextKind,
    DeriveRollingSeriesRequest,
    Direction,
    EvaluatePortfolioRequest,
    EvaluateRequest,
    EvaluateSeriesRequest,
    EvaluationPlanDraft,
    EvaluationPlanUpdate,
    EvaluationTarget,
    ExportExcelRequest,
    FixedSeriesParameter,
    IndependentRequest,
    IndicatorDraft,
    IndicatorReference,
    IndicatorType,
    IndicatorUpdate,
    InferRequest,
    InstrumentKind,
    ParameterBindRequest,
    ParameterContract,
    ParameterInspectRequest,
    PlanIndicatorInput,
    PlanProductCondition,
    PlanProductSelection,
    PlanProductSelectionFilters,
    PlanRunRequest,
    PrepareEvaluationRequest,
    RollingScalarDraftRequest,
    RollingSourceDefinition,
    SeriesIndicatorInstance,
    SeriesOutputDefinition,
    SeriesParameterDefinition,
    SnapshotIndicatorConfigUpdate,
    SnapshotIndicatorItem,
    StableValidationRoute,
    SUPPORTED_PERIODS,
    ValidateRequest,
)


router = APIRouter(tags=["custom-indicators"], route_class=StableValidationRoute)
indicator_service = CustomIndicatorService()


def pit_as_of(stated: Optional[str]) -> Optional[str]:
    """The研究日 an indicator run actually computes under.

    A stated 截止日 still wins — that box is the per-run override, and a
    backtest sweeping `as_of` must not be clamped. Saying nothing now inherits
    the platform口径 instead of quietly reading to the last row on disk, which
    is what made 产品研究 show 2026 numbers under a 2014 research day.
    """

    # Read against the dir the market data itself comes from, so the口径 and
    # the rows can never be from two different places.
    return resolve_request_context(indicator_service.market_data_dir, stated).as_of


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
        as_of=pit_as_of(request.as_of),
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
        as_of=pit_as_of(request.as_of),
        include_series=request.include_series,
        compile_token=request.compile_token,
        parameters=request.parameters,
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
        as_of=pit_as_of(request.as_of),
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
        as_of=pit_as_of(request.as_of),
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
    return _call(indicator_service.run_plan, plan_id, pit_as_of(request.as_of if request else None))
