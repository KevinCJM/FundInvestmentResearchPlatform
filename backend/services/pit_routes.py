"""Routes for the PIT capability sheet, data releases and research context."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from pit.audit import audit_all, clear_cache, start_scan
from pit.catalog import GRADE_DESCRIPTIONS, GRADE_LABELS, RUN_MODES, STRICT_COVERAGE_FLOOR
from pit.context import (
    PitContextError,
    ResearchContext,
    build_context,
    parse_view_override,
    reset_view_override,
    resolve,
    resolve_request_context,
    set_view_override,
)
from pit.universe import KIND_LABELS, KINDS, universe_as_of
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
    # A version is the whole口径, so it is defined with the day it stands on.
    asOf: Optional[str] = None
    runMode: str = "RESEARCH"


class ReleaseUpdateRequest(BaseModel):
    """The口径 half of a version. Its fingerprints are not editable."""

    name: str = Field(min_length=1, max_length=120)
    note: str = ""
    asOf: Optional[str] = None
    runMode: str = "RESEARCH"


class SettingsRequest(BaseModel):
    activeReleaseId: Optional[str] = None
    # Both null when a version is applied: the version answers. They stay for a
    # bare research day with nothing sealed yet.
    asOf: Optional[str] = None
    runMode: Optional[str] = None
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
    """Measured PIT capability of every declared dataset.

    Answers from what is already measured and starts the scan behind the
    request. Reading the clock columns of every declared file is minutes on a
    cold cache, and a settings page that hangs for minutes is a broken page —
    the payload says which datasets are still pending so the client can poll.
    """

    if refresh:
        clear_cache()
    payload = audit_all(DATA_DIR, scan=False)
    if payload["summary"]["pending"]:
        payload["scan"] = {**start_scan(DATA_DIR), "pending": payload["scan"]["pending"]}
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


@router.get("/universe")
def pit_universe(kind: str = "fund", as_of: Optional[str] = None, sample: int = 12):
    """Which products a screen run on the research day was allowed to choose from.

    The number that matters is the contrast: today's table against the day's.
    A universe that shrinks from 1,760 to 224 is not a rounding difference — it
    is every fund launched since, which no 2019 screen could have picked.
    """

    if kind not in KINDS:
        return JSONResponse(status_code=400, content={"detail": f"不支持的产品域：{kind}"})
    try:
        context = resolve_request_context(DATA_DIR, as_of)
        view = universe_as_of(DATA_DIR, context, kind=kind)
        today = universe_as_of(DATA_DIR, ResearchContext(), kind=kind)
    except PitContextError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    code_field = KINDS[kind][1]
    name_field = "name" if "name" in view.detail.columns else code_field
    members = [
        {"code": str(row[code_field]), "name": str(row.get(name_field, ""))}
        for _, row in view.detail.head(max(0, int(sample))).iterrows()
    ] if not view.detail.empty else []
    return {
        "kind": kind,
        "kind_label": KIND_LABELS[kind],
        "as_of": view.as_of,
        "run_mode": context.run_mode,
        "coverage": view.coverage,
        "replayable": view.replayable,
        "history_begins_at": view.history_begins_at,
        "member_count": len(view.codes),
        "latest_member_count": len(today.codes),
        "excluded_by_replay": max(0, len(today.codes) - len(view.codes)),
        "sample": members,
        "warnings": list(view.warnings),
    }


@router.get("/releases")
def list_releases() -> dict[str, Any]:
    return {"releases": _repository().list_releases()}


@router.post("/releases")
def create_release(req: ReleaseCreateRequest):
    try:
        return _repository().create(DATA_DIR, req.name, req.note, req.asOf, req.runMode)
    except DataReleaseError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@router.get("/releases/{release_id}")
def get_release(release_id: str):
    try:
        return _repository().get(release_id)
    except DataReleaseError as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})


@router.put("/releases/{release_id}")
def update_release(release_id: str, req: ReleaseUpdateRequest):
    """Fix a version's口径 without re-sealing its vintage."""

    try:
        return _repository().update(
            release_id, name=req.name, note=req.note, as_of=req.asOf, run_mode=req.runMode
        )
    except DataReleaseError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


@router.delete("/releases/{release_id}")
def delete_release(release_id: str):
    """Delete a version, unless the platform is standing on it right now.

    Deleting the applied one would leave every page silently back on no-PIT,
    so the switch is the user's to make first — this is a guard, not a nag.
    """

    try:
        active = _settings().raw().get("active_release_id")
        if str(active or "") == str(release_id or "").strip():
            name = _repository().get(release_id)["name"]
            return JSONResponse(
                status_code=409,
                content={
                    "detail": f"版本「{name}」正在被全平台使用；请先切到别的版本或「不用 PIT」并应用，然后再删除。"
                },
            )
        return {"deleted_id": _repository().delete(release_id)}
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
    """Set the platform口径: which day to stand on, which vintage, how strict."""

    try:
        return _settings().update(req.activeReleaseId, req.runMode, req.note, req.asOf)
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
