"""Read-only API for the research data laboratory."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from services.custom_indicator_routes import indicator_service

try:
    from backend.research_series.numba_kernels import warm_research_series_numba_kernels
    from backend.research_series.service import ResearchSeriesError, ResearchSeriesService
    from backend.research_series.file_import import MAX_FILE_BYTES, parse_series_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from research_series.numba_kernels import warm_research_series_numba_kernels
    from research_series.service import ResearchSeriesError, ResearchSeriesService
    from research_series.file_import import MAX_FILE_BYTES, parse_series_file


router = APIRouter(prefix="/api/research-series", tags=["research-series"])
def _indicator_versions():
    # Reuse the exact registry used for execution, including built-ins and history.
    return [
        indicator_service.get_indicator(item["id"], int(item["revision"]))
        for item in indicator_service.indicators.list_all_versions()
    ]


research_series_service = ResearchSeriesService(indicator_versions=_indicator_versions)

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
    kind: Literal["index", "etf", "fund", "macro", "indicator", "upload"] | None = None,
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


@router.get("/uploads")
def list_uploaded_research_series(
    q: str = Query(default="", max_length=100),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
):
    return _call(research_series_service.uploaded_series, query=q, offset=offset, limit=limit)


@router.post("/parse-file")
async def parse_research_series_file(
    request: Request,
    filename: str = Query(min_length=1, max_length=240),
    sheet: str | None = Query(default=None, max_length=100),
):
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_FILE_BYTES:
            raise HTTPException(413, detail="文件大小不能超过 8 MB。")
    try:
        return await run_in_threadpool(parse_series_file, bytes(content), filename, sheet)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc)) from exc


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
