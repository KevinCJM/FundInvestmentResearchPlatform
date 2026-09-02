"""Read-only index catalog and coverage endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Query

from .index_data import index_summary, list_indices


router = APIRouter(prefix="/api/indices", tags=["indices"])


@router.get("/summary")
def get_index_summary():
    return index_summary()


@router.get("")
def get_indices(
    q: str | None = Query(default=None, max_length=100),
    source: str | None = Query(default=None, max_length=50),
    market: str | None = Query(default=None, max_length=50),
    category: str | None = Query(default=None, max_length=100),
    active: str | None = Query(default=None, pattern="^(active|inactive)$"),
    coverage: str | None = Query(default=None, pattern="^(ready|stale|missing)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    return list_indices(
        query=q,
        source=source,
        market=market,
        category=category,
        active=active,
        coverage=coverage,
        page=page,
        page_size=page_size,
    )
