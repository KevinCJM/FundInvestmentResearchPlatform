"""Routes for the PIT capability sheet, data releases and research context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from pit.audit import audit_all, clear_cache
from pit.catalog import GRADE_DESCRIPTIONS, GRADE_LABELS, RUN_MODES, STRICT_COVERAGE_FLOOR
from pit.context import (
    PitContextError,
    build_context,
    parse_view_override,
    reset_view_override,
    resolve,
    set_view_override,
)
from pit.release import RELEASE_STORE, DataReleaseError, DataReleaseRepository
from pit.settings import PitSettingsRepository


DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

router = APIRouter(prefix="/api/pit", tags=["pit"])


def _repository() -> DataReleaseRepository:
    return DataReleaseRepository(DATA_DIR / RELEASE_STORE)


def _settings() -> PitSettingsRepository:
    return PitSettingsRepository(DATA_DIR)


class PitViewOverrideMiddleware:
    """Turns the per-tab PIT headers into the viewing口径 for one request.

    The system setting is the default every page displays against; this is how a
    reader temporarily looks at another vintage — or at the raw disk — without
    changing what anyone else sees, and without every endpoint growing three
    parameters it would eventually forget to pass.

    Plain ASGI rather than `BaseHTTPMiddleware` on purpose: the context variable
    has to be set in the same task the endpoint runs in, and `BaseHTTPMiddleware`
    hands the downstream app to a child task.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            override = parse_view_override(DATA_DIR, Headers(scope=scope))
        except (PitContextError, DataReleaseError) as exc:
            # Fail loudly. Silently ignoring a口径 the user picked is exactly how
            # a number gets read under a vintage nobody meant.
            response = JSONResponse(
                status_code=400, content={"detail": f"临时 PIT 口径无效：{exc}"}
            )
            await response(scope, receive, send)
            return
        if override is None:
            await self.app(scope, receive, send)
            return
        token = set_view_override(override)
        try:
            await self.app(scope, receive, send)
        finally:
            reset_view_override(token)


class ReleaseCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    note: str = ""


class SettingsRequest(BaseModel):
    activeReleaseId: Optional[str] = None
    runMode: str = "RESEARCH"
    note: str = ""


class ContextRequest(BaseModel):
    asOf: Optional[str] = None
    runMode: str = "RESEARCH"
    dataReleaseId: Optional[str] = None


@router.get("/meta")
def pit_meta() -> dict[str, Any]:
    """Static vocabulary the UI renders: grades and run modes."""

    return {
        "grades": [
            {"id": key, "label": label, "description": GRADE_DESCRIPTIONS[key]}
            for key, label in GRADE_LABELS.items()
        ],
        "run_modes": [{"id": key, "label": label} for key, label in RUN_MODES.items()],
        "strict_coverage_floor": STRICT_COVERAGE_FLOOR,
    }


@router.get("/audit")
def pit_audit(refresh: bool = False) -> dict[str, Any]:
    """Measured PIT capability of every declared dataset."""

    if refresh:
        clear_cache()
    payload = audit_all(DATA_DIR)
    latest = _repository().latest()
    payload["latest_release"] = (
        {
            "id": latest["id"],
            "name": latest["name"],
            "created_at": latest["created_at"],
            "release_fingerprint": latest["release_fingerprint"],
        }
        if latest
        else None
    )
    return payload


@router.get("/releases")
def list_releases() -> dict[str, Any]:
    return {"releases": _repository().list_releases()}


@router.post("/releases")
def create_release(req: ReleaseCreateRequest):
    try:
        return _repository().create(DATA_DIR, req.name, req.note)
    except DataReleaseError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@router.get("/releases/{release_id}")
def get_release(release_id: str):
    try:
        return _repository().get(release_id)
    except DataReleaseError as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})


@router.get("/settings")
def get_settings():
    """The system-level PIT口径 every page displays against."""

    try:
        return _settings().describe()
    except PitContextError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@router.put("/settings")
def put_settings(req: SettingsRequest):
    """Apply one data release as the platform口径, or clear it back to no PIT."""

    try:
        return _settings().update(req.activeReleaseId, req.runMode, req.note)
    except (PitContextError, DataReleaseError) as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@router.post("/context/resolve")
def resolve_context(req: ContextRequest):
    """Answer 'what can I use under this context?' before anything runs."""

    try:
        context = build_context(req.asOf, req.runMode, req.dataReleaseId)
    except PitContextError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    payload = resolve(DATA_DIR, context)
    if context.data_release_id:
        try:
            release = _repository().get(context.data_release_id)
            payload["release"] = {
                "id": release["id"],
                "name": release["name"],
                "created_at": release["created_at"],
                "release_fingerprint": release["release_fingerprint"],
            }
        except DataReleaseError as exc:
            payload["usable"] = False
            payload["release"] = None
            payload["release_error"] = str(exc)
    return payload


__all__ = ["router"]
