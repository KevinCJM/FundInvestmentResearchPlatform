"""FastAPI routes for the custom indicator research workflow."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from custom_indicators.errors import IndicatorDomainError
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
    "other",
]


def _all_supported_periods() -> list[str]:
    return list(SUPPORTED_PERIODS)


class IndicatorDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    expression: str = Field(min_length=1, max_length=1000)
    periods: Optional[list[str]] = None
    period_policy: Literal["all_supported"] = "all_supported"
    unit: str = Field(default="", max_length=20)
    display_format: Literal["number", "percent"] = "number"
    precision: int = Field(default=2, ge=0, le=8)
    direction: Direction = "higher_better"
    indicator_type: IndicatorType = "other"
    annual_risk_free_rate_percent: float = Field(default=0.0, ge=-100.0, le=100.0)
    dsl_version: str = TYPED_DSL_VERSION
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None
    variable_registry_version: Optional[str] = None
    data_contract_version: Optional[str] = None
    context_schema_version: Optional[str] = None
    context_kind: ContextKind = "single_product"
    output_contract: Literal["scalar"] = "scalar"
    output_measure: Optional[str] = None
    template_origin: Any = None


class IndicatorUpdate(IndicatorDraft):
    revision: int = Field(ge=1)


class ValidateRequest(BaseModel):
    name: str = Field(default="未保存指标", max_length=80)
    expression: str = Field(default="", max_length=1000)
    periods: Optional[list[str]] = None
    period_policy: Literal["all_supported"] = "all_supported"
    annual_risk_free_rate_percent: float = Field(default=0.0, ge=-100.0, le=100.0)
    dsl_version: str = TYPED_DSL_VERSION
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None
    variable_registry_version: Optional[str] = None
    data_contract_version: Optional[str] = None
    context_schema_version: Optional[str] = None
    context_kind: ContextKind = "single_product"
    output_contract: Literal["scalar"] = "scalar"
    output_measure: Optional[str] = None


class ComposeArgument(BaseModel):
    parameter: str = Field(min_length=1, max_length=80)
    source: Literal["variable", "constant", "expression"]
    value: str | float


class ComposeRequest(BaseModel):
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


class InferRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=1000)
    context: ContextKind = "single_product"
    dsl_version: str = TYPED_DSL_VERSION
    operator_registry_version: Optional[str] = None
    numeric_kernel_version: Optional[str] = None


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


class EvaluateRequest(BaseModel):
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)
    inline_definition: Optional[IndicatorDraft] = None
    targets: list[EvaluationTarget] = Field(min_length=1, max_length=50)
    period: str
    as_of: Optional[str] = None
    include_series: bool = False


class SnapshotIndicatorItem(BaseModel):
    indicator_id: str = Field(min_length=1, max_length=120)
    indicator_revision: int = Field(ge=1)
    period: str = Field(min_length=1, max_length=12)


class SnapshotIndicatorConfigUpdate(BaseModel):
    revision: int = Field(ge=1)
    items: list[SnapshotIndicatorItem] = Field(default_factory=list, max_length=30)


class EvaluatePortfolioRequest(BaseModel):
    run_id: str = Field(min_length=1, max_length=100)
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)
    inline_definition: Optional[IndicatorDraft] = None


class PlanIndicatorInput(BaseModel):
    indicator_id: str = Field(min_length=1)
    indicator_revision: Optional[int] = Field(default=None, ge=1)
    period: str
    weight: float = Field(ge=0)
    direction: Optional[Direction] = None


class PlanProductSelectionFilters(BaseModel):
    fund_type: list[str] = Field(default_factory=list, max_length=100)
    invest_type: list[str] = Field(default_factory=list, max_length=100)
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


@router.post("/api/custom-indicators/compose")
def compose_custom_indicator(request: ComposeRequest):
    return _call(indicator_service.compose, request.model_dump())


@router.post("/api/custom-indicators/infer")
def infer_custom_indicator(request: InferRequest):
    return _call(indicator_service.infer, request.model_dump())


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


@router.post("/api/custom-indicators/evaluate")
def evaluate_custom_indicators(request: EvaluateRequest):
    inline = request.inline_definition.model_dump() if request.inline_definition else None
    return _call(
        indicator_service.evaluate,
        indicator_ids=request.indicator_ids,
        inline_definition=inline,
        targets=[target.model_dump() for target in request.targets],
        period=request.period,
        as_of=request.as_of,
        include_series=request.include_series,
    )


@router.post("/api/custom-indicators/evaluate-portfolio")
def evaluate_portfolio_custom_indicators(request: EvaluatePortfolioRequest):
    inline = request.inline_definition.model_dump() if request.inline_definition else None
    return _call(
        indicator_service.evaluate_portfolio,
        run_id=request.run_id,
        indicator_ids=request.indicator_ids,
        inline_definition=inline,
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
