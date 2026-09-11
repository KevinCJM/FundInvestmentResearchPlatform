"""Read-only API for the platform-owned data model catalog."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

try:
    from backend.data_model.catalog import CatalogScope, get_data_model_catalog, get_table_definition
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from data_model.catalog import CatalogScope, get_data_model_catalog, get_table_definition


router = APIRouter(prefix="/api/data-model", tags=["data-model"])


@router.get("/catalog")
def data_model_catalog(scope: CatalogScope = "external"):
    """Return import targets by default; internal/all requires explicit opt-in."""

    return get_data_model_catalog(scope=scope)


@router.get("/tables/{table_id}")
def data_model_table(table_id: str, scope: CatalogScope = "external"):
    """Return a table in the requested scope, never an implicit internal target."""

    table = get_table_definition(table_id, scope=scope)
    if table is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DATA_MODEL_TABLE_NOT_FOUND",
                "message": f"当前数据字典范围内不存在表 {table_id}。",
            },
        )
    return table
