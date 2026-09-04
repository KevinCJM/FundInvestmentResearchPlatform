"""FastAPI routes for historical regime identification research."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from custom_indicators.errors import ConflictError, IndicatorDomainError, NotFoundError
from historical_regimes.service import HistoricalRegimeService
from historical_regimes.v2_service import RegimeGraphV2Service
from services.custom_indicator_routes import StableValidationRoute, indicator_service


router = APIRouter(tags=["historical-regimes"], route_class=StableValidationRoute)
historical_regime_service = HistoricalRegimeService(
    indicator_service=indicator_service,
)
regime_graph_v2_service = RegimeGraphV2Service(
    indicator_service=indicator_service,
)


class DefinitionDraft(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    template_id: Optional[str] = None
    template: Optional[str] = None
    target: Optional[dict[str, Any]] = None
    data: Optional[dict[str, Any]] = None
    features: dict[str, Any] = Field(default_factory=dict)
    algorithm: dict[str, Any]
    states: list[dict[str, Any]] = Field(default_factory=list, max_length=12)
    validation: dict[str, Any] = Field(default_factory=dict)
    usage_intent: str = "research_display"


class DefinitionUpdate(DefinitionDraft):
    revision: int = Field(ge=1)


class RunRequest(BaseModel):
    definition: dict[str, Any]
    mode: Literal["realtime", "retrospective"] = "realtime"
    as_of: Optional[str] = None
    compile_token: Optional[str] = None


class FormulaPrepareRequest(BaseModel):
    definition: dict[str, Any]
    mode: Literal["realtime", "retrospective"] = "realtime"
    as_of: Optional[str] = None


class PublishRequest(BaseModel):
    usage: str | list[str]
    note: str = Field(default="", max_length=500)


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=8)
    reference_run_id: Optional[str] = None


class TAABacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_returns: list[dict[str, Any]] = Field(min_length=1, max_length=20000)
    base_weights: dict[str, float] = Field(min_length=1, max_length=100)
    state_tilts: dict[str, dict[str, float]] = Field(min_length=1, max_length=12)
    limits: dict[str, float] = Field(default_factory=dict)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0, le=1000.0)
    confidence_floor: float = Field(default=0.0, ge=0.0, le=1.0)
    periods_per_year: int = Field(default=252, ge=1, le=3660)
    max_signal_age_days: int = Field(default=31, ge=1, le=3650)


class RegimeGraphDefinitionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: dict[str, Any]


class RegimeGraphDefinitionUpdateRequest(RegimeGraphDefinitionRequest):
    revision: int = Field(ge=1)


class RegimeGraphPreviewRequest(RegimeGraphDefinitionRequest):
    compile_token: str
    mode: Literal["realtime", "retrospective"] = "realtime"
    as_of: Optional[str] = None
    ttl_seconds: int = Field(default=1800, ge=1, le=86400)


class RegimeGraphAssetCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["template", "subgraph"]
    asset: dict[str, Any]


class RegimeGraphAssetUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1)
    asset: dict[str, Any]


class RegimeExperimentDimensionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=64)
    parameter: str = Field(min_length=1, max_length=100)
    values: list[Any] = Field(min_length=1, max_length=12)


class RegimeExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    definition: dict[str, Any]
    compile_token: str = Field(min_length=1)
    mode: Literal["realtime", "retrospective"] = "realtime"
    as_of: Optional[str] = None
    parameter_grid: list[RegimeExperimentDimensionRequest] = Field(
        min_length=1,
        max_length=6,
    )
    ranking_metric: Literal[
        "agreement",
        "classified_ratio",
        "low_flip_rate",
        "boundary_distance",
    ] = "agreement"


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except IndicatorDomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


def _raise_v1_read_only() -> None:
    raise ConflictError(
        "V1_DEFINITION_READ_ONLY",
        "v1 历史情景定义已进入只读兼容；请复制为 v2 图谱后编辑。",
        field="definition",
    )


def _raise_v1_run_read_only() -> None:
    raise ConflictError(
        "V1_RUN_READ_ONLY",
        "v1 历史情景运行已进入只读兼容；请先复制为 v2 图谱，再预热并运行。",
        field="definition",
    )


def _raise_v1_publication_read_only() -> None:
    raise ConflictError(
        "V1_RUN_PUBLICATION_READ_ONLY",
        "v1 历史运行及其发布记录保持只读，不能新增发布；可继续查看和用于已有绑定。",
        field="run_id",
    )


def _run_stores_are_shared() -> bool:
    try:
        left = historical_regime_service.runs.store.path.resolve()
        right = regime_graph_v2_service.runs.store.path.resolve()
    except (AttributeError, OSError):
        return False
    return left == right


@router.get("/api/historical-regimes/meta")
def get_historical_regime_meta():
    return _call(historical_regime_service.meta)


@router.get("/api/historical-regimes/nodes")
def get_regime_graph_node_catalog():
    return _call(regime_graph_v2_service.catalog)


@router.get("/api/historical-regimes/templates/v2")
def list_regime_graph_templates():
    return _call(regime_graph_v2_service.templates)


@router.post("/api/historical-regimes/templates/{template_id}/instantiate")
def instantiate_regime_graph_template(template_id: str):
    return _call(regime_graph_v2_service.instantiate_template, template_id)


@router.post("/api/historical-regimes/infer")
def infer_regime_graph(request: RegimeGraphDefinitionRequest):
    return _call(regime_graph_v2_service.infer, request.definition)


@router.post("/api/historical-regimes/prepare")
def prepare_regime_graph(request: RegimeGraphDefinitionRequest):
    return _call(regime_graph_v2_service.prepare, request.definition)


@router.post(
    "/api/historical-regimes/preview-runs",
    status_code=status.HTTP_202_ACCEPTED,
)
def create_regime_graph_preview(request: RegimeGraphPreviewRequest):
    return _call(
        regime_graph_v2_service.create_preview,
        request.definition,
        compile_token=request.compile_token,
        mode=request.mode,
        as_of=request.as_of,
        ttl_seconds=request.ttl_seconds,
    )


@router.get("/api/historical-regimes/preview-runs/{preview_id}")
def get_regime_graph_preview(preview_id: str):
    return _call(regime_graph_v2_service.get_preview, preview_id)


@router.delete("/api/historical-regimes/preview-runs/{preview_id}")
def cancel_regime_graph_preview(preview_id: str):
    return _call(regime_graph_v2_service.cancel_preview, preview_id)


@router.get("/api/historical-regimes/preview-runs/{preview_id}/series")
def get_regime_graph_preview_series(
    preview_id: str,
    node_id: Optional[str] = None,
    port: Optional[str] = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=1000, ge=1, le=5000),
):
    return _call(
        regime_graph_v2_service.preview_series,
        preview_id,
        node_id=node_id,
        port=port,
        offset=offset,
        limit=limit,
    )


@router.get("/api/historical-regimes/v2/definitions")
def list_regime_graph_definitions():
    return {"items": _call(regime_graph_v2_service.list_definitions)}


@router.post(
    "/api/historical-regimes/v2/definitions",
    status_code=status.HTTP_201_CREATED,
)
def create_regime_graph_definition(request: RegimeGraphDefinitionRequest):
    return _call(regime_graph_v2_service.create_definition, request.definition)


@router.get("/api/historical-regimes/v2/definitions/{definition_id}")
def get_regime_graph_definition(
    definition_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    return _call(regime_graph_v2_service.get_definition, definition_id, revision)


@router.put("/api/historical-regimes/v2/definitions/{definition_id}")
def update_regime_graph_definition(
    definition_id: str,
    request: RegimeGraphDefinitionUpdateRequest,
):
    return _call(
        regime_graph_v2_service.update_definition,
        definition_id,
        request.revision,
        request.definition,
    )


@router.get("/api/historical-regimes/v2/graph-assets")
def list_regime_graph_assets(
    kind: Optional[Literal["template", "subgraph"]] = None,
):
    return {"items": _call(regime_graph_v2_service.list_graph_assets, kind)}


@router.post(
    "/api/historical-regimes/v2/graph-assets",
    status_code=status.HTTP_201_CREATED,
)
def create_regime_graph_asset(request: RegimeGraphAssetCreateRequest):
    return _call(
        regime_graph_v2_service.create_graph_asset,
        request.kind,
        request.asset,
    )


@router.get("/api/historical-regimes/v2/graph-assets/{asset_id}")
def get_regime_graph_asset(
    asset_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    return _call(regime_graph_v2_service.get_graph_asset, asset_id, revision)


@router.put("/api/historical-regimes/v2/graph-assets/{asset_id}")
def update_regime_graph_asset(
    asset_id: str,
    request: RegimeGraphAssetUpdateRequest,
):
    return _call(
        regime_graph_v2_service.update_graph_asset,
        asset_id,
        request.revision,
        request.asset,
    )


@router.post("/api/historical-regimes/v2/graph-assets/{asset_id}/instantiate")
def instantiate_user_regime_graph_asset(
    asset_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    return _call(regime_graph_v2_service.instantiate_graph_asset, asset_id, revision)


@router.post(
    "/api/historical-regimes/v2/experiments",
    status_code=status.HTTP_201_CREATED,
)
def create_regime_graph_experiment(request: RegimeExperimentRequest):
    return _call(
        regime_graph_v2_service.run_batch_experiment,
        request.definition,
        [item.model_dump() for item in request.parameter_grid],
        compile_token=request.compile_token,
        mode=request.mode,
        as_of=request.as_of,
        ranking_metric=request.ranking_metric,
    )


@router.get("/api/historical-regimes/v2/experiments")
def list_regime_graph_experiments(definition_id: Optional[str] = None):
    return {"items": _call(regime_graph_v2_service.list_experiments, definition_id)}


@router.get("/api/historical-regimes/v2/experiments/{experiment_id}")
def get_regime_graph_experiment(experiment_id: str):
    return _call(regime_graph_v2_service.get_experiment, experiment_id)


@router.get("/api/historical-regimes/definitions")
def list_historical_regime_definitions():
    return {"items": _call(historical_regime_service.list_definitions)}


@router.post("/api/historical-regimes/definitions", status_code=status.HTTP_201_CREATED)
def create_historical_regime_definition(request: DefinitionDraft):
    del request
    return _call(_raise_v1_read_only)


@router.get("/api/historical-regimes/definitions/{definition_id}")
def get_historical_regime_definition(
    definition_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    return _call(historical_regime_service.get_definition, definition_id, revision)


@router.put("/api/historical-regimes/definitions/{definition_id}")
def update_historical_regime_definition(definition_id: str, request: DefinitionUpdate):
    del definition_id, request
    return _call(_raise_v1_read_only)


@router.post("/api/historical-regimes/definitions/{definition_id}/copy-to-v2")
def copy_historical_regime_definition_to_v2(
    definition_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    source = _call(historical_regime_service.get_definition, definition_id, revision)
    return _call(regime_graph_v2_service.copy_v1_definition, source)


@router.post("/api/historical-regimes/run", status_code=status.HTTP_201_CREATED)
def run_historical_regime(request: RunRequest):
    if str(request.definition.get("schema_version") or "") == "2.0":
        return _call(
            regime_graph_v2_service.run_saved,
            request.definition,
            request.mode,
            request.as_of,
            request.compile_token,
        )
    return _call(_raise_v1_run_read_only)


@router.post("/api/historical-regimes/formulas/prepare")
def prepare_historical_regime_formula(request: FormulaPrepareRequest):
    return _call(
        historical_regime_service.prepare_formula,
        request.definition,
        request.mode,
        request.as_of,
    )


@router.get("/api/historical-regimes/runs")
def list_historical_regime_runs(definition_id: Optional[str] = None):
    if _run_stores_are_shared():
        return {"items": _call(regime_graph_v2_service.list_runs, definition_id)}
    classic = _call(historical_regime_service.list_runs, definition_id)
    graph = _call(regime_graph_v2_service.list_runs, definition_id)
    merged = {str(item.get("id")): item for item in [*classic, *graph]}
    return {"items": list(merged.values())}


@router.get("/api/historical-regimes/runs/{run_id}")
def get_historical_regime_run(run_id: str):
    if _run_stores_are_shared():
        return _call(regime_graph_v2_service.get_run, run_id)
    try:
        return regime_graph_v2_service.get_run(run_id)
    except NotFoundError:
        return _call(historical_regime_service.get_run, run_id)


@router.post("/api/historical-regimes/runs/{run_id}/publish")
def publish_historical_regime_run(run_id: str, request: PublishRequest):
    try:
        run = regime_graph_v2_service.get_run(run_id)
    except NotFoundError:
        run = None
    if isinstance(run, dict) and str(run.get("schema_version") or "") == "2.0":
        return _call(regime_graph_v2_service.publish, run_id, request.usage, request.note)
    if run is None:
        _call(historical_regime_service.get_run, run_id)
    return _call(_raise_v1_publication_read_only)


@router.post("/api/historical-regimes/runs/{run_id}/taa-backtest")
def backtest_historical_regime_taa(run_id: str, request: TAABacktestRequest):
    try:
        run = regime_graph_v2_service.get_run(run_id)
    except NotFoundError:
        run = None
    if isinstance(run, dict) and str(run.get("schema_version") or "") == "2.0":
        return _call(
            regime_graph_v2_service.taa_backtest,
            run_id,
            request.model_dump(),
        )
    return _call(
        historical_regime_service.taa_backtest,
        run_id,
        request.model_dump(),
    )


@router.post("/api/historical-regimes/compare")
def compare_historical_regime_runs(request: CompareRequest):
    if _run_stores_are_shared():
        return _call(regime_graph_v2_service.compare, request.run_ids, request.reference_run_id)
    runs: list[dict[str, Any]] = []
    for run_id in request.run_ids:
        try:
            runs.append(regime_graph_v2_service.get_run(run_id))
        except NotFoundError:
            runs.append(_call(historical_regime_service.get_run, run_id))
    return _call(regime_graph_v2_service.compare_snapshots, runs, request.reference_run_id)
