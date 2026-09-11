"""Workspace localization API. There is deliberately no system-translation write route."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.routing import APIRoute
from pydantic import ValidationError

from localization.contracts import BusinessUpdate, I18nError, ImportApply, ImportValidate, LanguagesUpdate, Locale, PreferencesUpdate, RestoreRequest, Scope
from localization.service import LocalizationService

MAX_BODY_BYTES = 1024 * 1024


class LocalizationRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def bounded(request: Request):
            try:
                if request.method in {"POST", "PUT", "PATCH"}:
                    chunks, size = [], 0
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_BODY_BYTES:
                            return JSONResponse(status_code=413, content={"detail": {"code": "I18N_BODY_TOO_LARGE", "message": "翻译请求不能超过 1 MB。"}})
                        chunks.append(chunk)
                    request._body = b"".join(chunks)
                return await handler(request)
            except I18nError as exc:
                return JSONResponse(status_code=exc.status, content={"detail": exc.detail()})
            except (RequestValidationError, ValidationError) as exc:
                # No rejected values, stack traces, or storage paths in responses.
                fields = [".".join(str(part) for part in error.get("loc", ()) if part != "body") for error in exc.errors()]
                return JSONResponse(status_code=422, content={"detail": {"code": "I18N_INVALID_ENTRY", "message_key": "errors.I18N_INVALID_ENTRY", "message": "请求字段或翻译包格式无效。", "field": fields[0] if fields else None}})
        return bounded


@lru_cache(maxsize=4)
def _service_at(path: str) -> LocalizationService:
    return LocalizationService(Path(path))


def localization_service() -> LocalizationService:
    default = Path(__file__).resolve().parents[2] / "data"
    path = os.environ.get("I18N_DATA_DIR") or os.environ.get("CUSTOM_INDICATOR_DATA_DIR") or str(default)
    return _service_at(str(Path(path).expanduser().resolve()))


router = APIRouter(prefix="/api/i18n", tags=["localization"], route_class=LocalizationRoute)


@router.get("/settings")
def settings(service: LocalizationService = Depends(localization_service)):
    return service.state()


@router.put("/preferences")
def preferences(request: PreferencesUpdate, service: LocalizationService = Depends(localization_service)):
    return service.set_preferences(request)


@router.put("/languages")
def languages(request: LanguagesUpdate, service: LocalizationService = Depends(localization_service)):
    return service.set_languages(request)


@router.get("/matrix")
def matrix(scope: Scope = "system", q: str = Query(default="", max_length=240), module: str = Query(default="", max_length=80), status: Literal["all", "customized", "missing"] = "all", sort_by: str = Query(default="code", min_length=1, max_length=35), sort_dir: Literal["asc", "desc"] = "asc", page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=200), service: LocalizationService = Depends(localization_service)):
    return service.matrix(scope, q, module, status, page, page_size, sort_by, sort_dir)


@router.get("/catalog")
def catalog(scope: Scope = "system", locale: Locale = "zh-CN", q: str = Query(default="", max_length=240), module: str = Query(default="", max_length=80), status: Literal["all", "customized", "missing"] = "all", page: int = Query(default=1, ge=1), page_size: int = Query(default=50, ge=1, le=200), service: LocalizationService = Depends(localization_service)):
    return service.list_entries(scope, locale, q, module, status, page, page_size)


@router.get("/bundle")
def bundle(request: Request, locale: Locale = "zh-CN", service: LocalizationService = Depends(localization_service)):
    payload = service.bundle(locale)
    etag = f'"{payload["catalog_version"]}:{payload["revision"]}:{payload["preferences_revision"]}:{payload["locale"]}:{payload["default_locale"]}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


@router.patch("/business")
def update_business(request: BusinessUpdate, service: LocalizationService = Depends(localization_service)):
    return service.update(request)


@router.get("/business/export")
def export_business(service: LocalizationService = Depends(localization_service)):
    return JSONResponse(service.export(), headers={"Content-Disposition": 'attachment; filename="business-translations.json"', "Cache-Control": "no-store"})


@router.post("/business/import/validate")
def validate_import(request: ImportValidate, service: LocalizationService = Depends(localization_service)):
    return service.validate_import(request)


@router.post("/business/import/apply")
def apply_import(request: ImportApply, service: LocalizationService = Depends(localization_service)):
    return service.apply_import(request)


@router.get("/business/history")
def history(service: LocalizationService = Depends(localization_service)):
    return service.history()


@router.post("/business/restore")
def restore(request: RestoreRequest, service: LocalizationService = Depends(localization_service)):
    return service.restore(request)
