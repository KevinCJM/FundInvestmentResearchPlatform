"""FastAPI routes for scenario simulation and portfolio stress testing."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from compute_policy import validate_execution_audit
from custom_indicators.errors import IndicatorDomainError
from scenario_stress.contracts import summarize_portfolio_weights
from scenario_stress.numba_kernels import scenario_stress_numba_status
from scenario_stress.service import ScenarioStressService
from services.custom_indicator_routes import StableValidationRoute


router = APIRouter(tags=["scenario-stress"], route_class=StableValidationRoute)
scenario_stress_service = ScenarioStressService()


class DefinitionDraft(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    method: Optional[str] = None
    type: Optional[str] = None
    horizon: int = Field(default=1, ge=1, le=1200)
    initial_nav: float = Field(default=1.0, gt=0)
    factors: list[dict[str, Any]] = Field(default_factory=list, max_length=32)
    assets: list[dict[str, Any]] = Field(min_length=1, max_length=64)
    portfolios: list[dict[str, Any]] = Field(min_length=1, max_length=32)
    mapping: dict[str, Any] = Field(default_factory=dict)
    scenario: dict[str, Any]
    limits: list[dict[str, Any]] = Field(default_factory=list, max_length=32)
    usage_intent: str = "research_display"


class DefinitionUpdate(DefinitionDraft):
    revision: int = Field(ge=1)


class RunRequest(BaseModel):
    definition: dict[str, Any]


class PublishRequest(BaseModel):
    usage: str | list[str]
    note: str = Field(default="", max_length=500)


class CompareRequest(BaseModel):
    run_ids: list[str] = Field(min_length=2, max_length=8)
    reference_run_id: Optional[str] = None


class WeightSummaryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    weights: dict[str, float] = Field(min_length=1, max_length=64)


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except IndicatorDomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


@router.get("/api/scenario-stress/meta")
def get_scenario_stress_meta():
    return _call(scenario_stress_service.meta)


@router.post("/api/scenario-stress/weight-summary")
def summarize_scenario_stress_weights(request: WeightSummaryRequest):
    summary = summarize_portfolio_weights(request.weights)
    return {
        **summary,
        "execution": validate_execution_audit(scenario_stress_numba_status()),
    }


@router.get("/api/scenario-stress/definitions")
def list_scenario_stress_definitions(include_archived: bool = False):
    return {"items": _call(scenario_stress_service.list_definitions, include_archived)}


@router.post("/api/scenario-stress/definitions", status_code=status.HTTP_201_CREATED)
def create_scenario_stress_definition(request: DefinitionDraft):
    return _call(scenario_stress_service.create_definition, request.model_dump(exclude_none=True))


@router.get("/api/scenario-stress/definitions/{definition_id}")
def get_scenario_stress_definition(
    definition_id: str,
    revision: Optional[int] = Query(default=None, ge=1),
):
    return _call(scenario_stress_service.get_definition, definition_id, revision)


@router.put("/api/scenario-stress/definitions/{definition_id}")
def update_scenario_stress_definition(definition_id: str, request: DefinitionUpdate):
    payload = request.model_dump(exclude_none=True)
    revision = int(payload.pop("revision"))
    return _call(scenario_stress_service.update_definition, definition_id, revision, payload)


@router.delete("/api/scenario-stress/definitions/{definition_id}")
def archive_scenario_stress_definition(
    definition_id: str,
    revision: int = Query(ge=1),
):
    return _call(scenario_stress_service.archive_definition, definition_id, revision)


@router.post("/api/scenario-stress/run", status_code=status.HTTP_201_CREATED)
def run_scenario_stress(request: RunRequest):
    return _call(scenario_stress_service.run, request.definition)


@router.post("/api/scenario-stress/batch-run", status_code=status.HTTP_201_CREATED)
def batch_run_scenario_stress(request: RunRequest):
    return _call(scenario_stress_service.batch_run, request.definition)


@router.get("/api/scenario-stress/runs")
def list_scenario_stress_runs(definition_id: Optional[str] = None):
    return {"items": _call(scenario_stress_service.list_runs, definition_id)}


@router.get("/api/scenario-stress/runs/{run_id}")
def get_scenario_stress_run(run_id: str):
    return _call(scenario_stress_service.get_run, run_id)


@router.post("/api/scenario-stress/runs/{run_id}/publish")
def publish_scenario_stress_run(run_id: str, request: PublishRequest):
    return _call(scenario_stress_service.publish, run_id, request.usage, request.note)


@router.post("/api/scenario-stress/compare")
def compare_scenario_stress_runs(request: CompareRequest):
    return _call(scenario_stress_service.compare, request.run_ids, request.reference_run_id)
