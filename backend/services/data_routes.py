"""Administrative routes for the local Tushare data refresh job."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, Field, SecretStr

from services.data_refresh import (
    data_refresh_enabled,
    full_refresh_enabled,
    remove_local_tushare_token,
    refresh_manager,
    save_local_tushare_token,
    tushare_token_configuration_enabled,
    tushare_token_configured,
)


router = APIRouter(prefix="/api/data", tags=["data-refresh"])


IndexScope = Literal[
    "catalog", "domestic", "industry", "concept", "global", "futures", "valuation", "constituents"
]


class DataRefreshRequest(BaseModel):
    modules: list[Literal["base", "etf", "fund", "index"]] = Field(
        default_factory=lambda: ["base", "etf", "fund", "index"]
    )
    mode: Literal["incremental", "full"] = "incremental"
    index_scopes: list[IndexScope] | None = None


class AnalyticsRebuildRequest(BaseModel):
    candidate: bool = False


class TushareTokenRequest(BaseModel):
    token: SecretStr


@router.get("/refresh/status")
def refresh_status(response: Response):
    response.headers["Cache-Control"] = "no-store"
    return refresh_manager.snapshot()


def _ensure_token_configuration_enabled() -> None:
    if not data_refresh_enabled() or not tushare_token_configuration_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前环境未开启前端 Token 配置。",
        )


@router.put("/token")
def configure_tushare_token(request: TushareTokenRequest):
    """Store a frontend-supplied token locally without ever returning it."""

    _ensure_token_configuration_enabled()
    try:
        return save_local_tushare_token(request.token.get_secret_value())
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法保存 Tushare Token。",
        ) from exc


@router.delete("/token")
def clear_tushare_token():
    """Remove the local token without affecting data already downloaded."""

    _ensure_token_configuration_enabled()
    try:
        return remove_local_tushare_token()
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="无法清除 Tushare Token。",
        ) from exc


@router.post("/refresh", status_code=status.HTTP_202_ACCEPTED)
def start_refresh(request: DataRefreshRequest | None = None):
    request = request or DataRefreshRequest()
    if not data_refresh_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前环境未开启数据刷新；请设置 DATA_REFRESH_ENABLED=true。",
        )
    if not tushare_token_configured():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="尚未配置 Tushare Token，请先在主界面的数据管理中保存。",
        )
    if request.mode == "full" and not full_refresh_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="全量更新未开启；请设置 DATA_FULL_REFRESH_ENABLED=true。",
        )
    try:
        modules = list(request.modules)
        if "index" in modules:
            return refresh_manager.start(
                modules, request.mode, list(request.index_scopes or []) or None
            )
        return refresh_manager.start(modules, request.mode)
    except (ValueError, PermissionError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/analytics/rebuild")
def rebuild_analytics_snapshot(request: AnalyticsRebuildRequest | None = None):
    """Rebuild local analytics artifacts without making any Tushare request."""

    if not data_refresh_enabled():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前环境未开启本地数据管理；请设置 DATA_REFRESH_ENABLED=true。",
        )
    try:
        if request and request.candidate:
            return refresh_manager.rebuild_analytics(candidate=True)
        return refresh_manager.rebuild_analytics()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="分析快照重建失败。") from exc
