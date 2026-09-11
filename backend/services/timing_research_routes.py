"""Product timing authoring, bounded jobs and immutable research references."""
from pathlib import Path
import os
from fastapi import APIRouter, HTTPException, Query
from backend.custom_indicators.errors import IndicatorDomainError as BackendDomainError
from custom_indicators.errors import IndicatorDomainError
from services.custom_indicator_routes import StableValidationRoute, indicator_service
from backend.timing_research.contracts import (
    Definition, DefinitionUpdate, PrepareRequest, RunRequest, CompareRequest, ReleaseRequest, BindingRequest,
)
from backend.timing_research.service import TimingResearchService

router = APIRouter(prefix="/api/timing-research", tags=["timing-research"], route_class=StableValidationRoute)
timing_service = TimingResearchService(
    Path(os.getenv("TIMING_RESEARCH_DATA_DIR", str(indicator_service.workspace_data_dir))),
    market_data_dir=indicator_service.market_data_dir, indicator_service=indicator_service,
)


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (IndicatorDomainError, BackendDomainError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


@router.get("/catalog")
def catalog():
    return _call(timing_service.catalog)


@router.get("/definitions")
def definitions():
    return {"items": _call(timing_service.repository.list_definitions)}


@router.post("/definitions", status_code=201)
def create_definition(body: Definition):
    return _call(timing_service.save_definition, body)


@router.put("/definitions/{identifier}")
def update_definition(identifier: str, body: DefinitionUpdate):
    return _call(timing_service.save_definition, Definition.model_validate(body.model_dump(exclude={"revision"})), identifier, body.revision)


@router.post("/prepare")
def prepare(body: PrepareRequest):
    return _call(timing_service.prepare, body.definition)


@router.post("/runs", status_code=202)
def start_run(body: RunRequest):
    return _call(timing_service.submit, body)


@router.get("/jobs/{identifier}")
def job(identifier: str):
    return _call(timing_service.job, identifier)


@router.get("/runs")
def runs(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    return _call(timing_service.repository.list_runs, offset=offset, limit=limit)


@router.get("/runs/{identifier}")
def run(identifier: str):
    return _call(timing_service.repository.get_run, identifier)


@router.get("/runs/{identifier}/products/{code}")
def product(identifier: str, code: str, offset: int = Query(0, ge=0), limit: int = Query(1200, ge=1, le=12000)):
    return _call(timing_service.repository.get_product, identifier, code, offset=offset, limit=limit)


@router.post("/compare")
def compare(body: CompareRequest):
    return _call(timing_service.compare, body.run_ids)


@router.post("/releases", status_code=201)
def release(body: ReleaseRequest):
    run = _call(timing_service.repository.get_run, body.run_id, limit=1)
    if not run.get("successful_products"):
        raise HTTPException(422, detail={"code": "TIMING_RELEASE_EMPTY", "message": "没有成功产品的研究不能冻结为研究版本。"})
    return _call(timing_service.repository.create_release, body.run_id, run["name"], body.note)


@router.get("/releases")
def releases():
    return {"items": _call(timing_service.repository.list_releases)}


@router.post("/bindings", status_code=201)
def bind(body: BindingRequest):
    return _call(timing_service.repository.create_binding, body.release_id, body.context, "workspace", body.note)


@router.get("/bindings")
def bindings():
    return {"items": _call(timing_service.repository.list_bindings)}
