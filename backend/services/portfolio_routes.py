"""FastAPI routes for versioned portfolio research and immutable runs."""

from __future__ import annotations

import io
from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from custom_indicators.errors import IndicatorDomainError
from custom_indicators.portfolio_service import PortfolioResearchService
from portfolio_regime import PublishedRegimeBacktestReference
from services.custom_indicator_routes import StableValidationRoute


router = APIRouter(tags=["portfolio-research"], route_class=StableValidationRoute)
portfolio_service = PortfolioResearchService()


class ResearchTargetDraft(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    kind: Literal["portfolio"] = "portfolio"
    definition: dict[str, Any]
    source: str = Field(default="workspace", max_length=80)


class ResearchTargetUpdate(ResearchTargetDraft):
    revision: int = Field(ge=1)


class RunRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    as_of: Optional[str] = None
    start_date: Optional[str] = None
    historical_regime: Optional[PublishedRegimeBacktestReference] = Field(
        default=None,
        validation_alias=AliasChoices("historical_regime", "regime"),
    )


class DiagnoseRequest(BaseModel):
    indicator_ids: list[str] = Field(default_factory=list, max_length=10)


class ScenarioRequest(BaseModel):
    name: str = Field(default="历史情景", min_length=1, max_length=80)
    start_date: str
    end_date: str


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except IndicatorDomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


@router.get("/api/research-targets")
def list_research_targets(kind: Literal["portfolio"] = "portfolio"):
    del kind
    return {"items": _call(portfolio_service.list_targets)}


@router.post("/api/research-targets", status_code=status.HTTP_201_CREATED)
def create_research_target(request: ResearchTargetDraft):
    return _call(portfolio_service.create_target, request.model_dump())


@router.get("/api/research-targets/{target_id}")
def get_research_target(target_id: str, revision: Optional[int] = Query(default=None, ge=1)):
    return _call(portfolio_service.get_target, target_id, revision)


@router.put("/api/research-targets/{target_id}")
def update_research_target(target_id: str, request: ResearchTargetUpdate):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(portfolio_service.update_target, target_id, revision, payload)


@router.delete("/api/research-targets/{target_id}")
def delete_research_target(target_id: str, revision: int = Query(ge=1)):
    _call(portfolio_service.delete_target, target_id, revision)
    return {"deleted_id": target_id}


@router.post("/api/research-targets/{target_id}/run", status_code=status.HTTP_201_CREATED)
def run_research_target(target_id: str, request: Optional[RunRequest] = None):
    body = request or RunRequest()
    return _call(
        portfolio_service.run_target,
        target_id,
        as_of=body.as_of,
        start_date=body.start_date,
        historical_regime=body.historical_regime,
    )


@router.get("/api/portfolio-runs")
def list_portfolio_runs(target_id: Optional[str] = None):
    return {"items": _call(portfolio_service.list_runs, target_id)}


@router.get("/api/portfolio-runs/{run_id}")
def get_portfolio_run(run_id: str):
    return _call(portfolio_service.get_run, run_id)


@router.post("/api/portfolio-runs/{run_id}/diagnose")
def diagnose_portfolio_run(run_id: str, request: Optional[DiagnoseRequest] = None):
    return _call(
        portfolio_service.diagnose,
        run_id,
        (request or DiagnoseRequest()).indicator_ids,
    )


@router.post("/api/portfolio-runs/{run_id}/scenario")
def run_portfolio_scenario(run_id: str, request: ScenarioRequest):
    result = _call(
        portfolio_service.scenario,
        run_id,
        start_date=request.start_date,
        end_date=request.end_date,
    )
    return {"name": request.name, **result}


@router.get("/api/portfolio-runs/{run_id}/export")
def export_portfolio_run(
    run_id: str,
    format: Literal["csv", "zip"] = "zip",
    table: Literal[
        "summary",
        "components",
        "daily-contributions",
        "weight-path",
        "covariance",
        "correlation",
        "risk-contributions",
        "scenario-summary",
        "scenario-series",
    ] = "summary",
    scenario_start: Optional[str] = None,
    scenario_end: Optional[str] = None,
):
    try:
        filename, payload, media_type = portfolio_service.export(
            run_id,
            archive=format == "zip",
            table=table,
            scenario_start=scenario_start,
            scenario_end=scenario_end,
        )
    except IndicatorDomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(io.BytesIO(payload), media_type=media_type, headers=headers)
