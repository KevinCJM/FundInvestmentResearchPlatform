"""Read-only API for the research data laboratory."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

try:
    from backend.research_series.numba_kernels import warm_research_series_numba_kernels
    from backend.research_series.service import ResearchSeriesError, ResearchSeriesService
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from research_series.numba_kernels import warm_research_series_numba_kernels
    from research_series.service import ResearchSeriesError, ResearchSeriesService


router = APIRouter(prefix="/api/research-series", tags=["research-series"])
research_series_service = ResearchSeriesService()

# Importing the router is part of FastAPI startup. Fail before serving requests
# if any fixed numerical signature is unavailable.
RESEARCH_SERIES_NUMBA_READINESS = warm_research_series_numba_kernels()


class ResearchSeriesProfileRequest(BaseModel):
    series_id: str = Field(default="upload:time_series", min_length=3, max_length=240)
    field: str | None = Field(default=None, min_length=1, max_length=120)
    inline_rows: list[dict[str, Any]] | None = Field(
        default=None,
        max_length=20_000,
    )
    rows: list[dict[str, Any]] | None = Field(default=None, max_length=20_000)
    name: str | None = Field(default=None, min_length=1, max_length=160)
    frequency: Literal[
        "daily",
        "weekly",
        "monthly",
        "quarterly",
        "annual",
        "irregular",
    ] = "daily"
    availability_mode: Literal["point_in_time", "latest"] = "point_in_time"
    register_artifact: bool = True
    artifact_id: str | None = Field(default=None, min_length=1, max_length=96)
    checksum: str | None = Field(default=None, min_length=1, max_length=96)
    start_date: str | None = None
    end_date: str | None = None
    as_of: str | None = None
    vintage: str | None = Field(default=None, min_length=1, max_length=160)
    rolling_window: int = Field(default=20, ge=2, le=1_260)
    sample_limit: int = Field(default=500, ge=2, le=2_000)


class ResearchSeriesCompareSource(ResearchSeriesProfileRequest):
    id: str | None = Field(default=None, min_length=1, max_length=120)
    label: str | None = Field(default=None, min_length=1, max_length=160)


class ResearchSeriesCompareRequest(BaseModel):
    sources: list[ResearchSeriesCompareSource] = Field(min_length=2, max_length=4)
    sample_limit: int = Field(default=500, ge=2, le=2_000)


def _call(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except ResearchSeriesError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


@router.get("/catalog")
def get_research_series_catalog(
    kind: Literal["index", "macro", "indicator", "upload"] | None = None,
    status: Literal["available", "not_downloaded"] | None = None,
    q: str | None = Query(default=None, max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=1_000),
):
    return _call(
        research_series_service.catalog,
        kind=kind,
        status=status,
        query=q,
        offset=offset,
        limit=limit,
    )


@router.post("/profile")
def profile_research_series(request: ResearchSeriesProfileRequest):
    return _call(
        research_series_service.profile,
        **request.model_dump(),
    )


@router.post("/compare")
def compare_research_series(request: ResearchSeriesCompareRequest):
    return _call(
        research_series_service.compare,
        sources=[source.model_dump() for source in request.sources],
        sample_limit=request.sample_limit,
    )


__all__ = [
    "RESEARCH_SERIES_NUMBA_READINESS",
    "ResearchSeriesCompareRequest",
    "ResearchSeriesCompareSource",
    "ResearchSeriesProfileRequest",
    "get_research_series_catalog",
    "compare_research_series",
    "profile_research_series",
    "router",
]
