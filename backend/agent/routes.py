"""Agent business APIs plus durable run control and resumable event delivery."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from custom_indicators.errors import IndicatorDomainError
from services.custom_indicator_contracts import StableValidationRoute
from . import commit as commit_flow
from . import harness, memory
from .catalog import build_catalog
from .contracts import (AgentError, AgentMessageRequest, AgentScopeRequest, AgentSessionCreate,
                        AgentCancelRequest, AgentInvalidateRequest, CommitPreviewRequest, CommitRequest, MemoryRequest, MemoryRevokeRequest)
from .llm import build_client, LLMUnavailableError
from .llm_settings import LlmSettingsStore
from .sessions import AgentSessionStore, append_event, assert_mutable, public_run, stable_json, utc_now
from .scopes import PAGE_SCOPES, SCOPE_LABELS, SCOPE_TOOLS, scope_for_page, validate_page_context, validate_scope_for_page


@asynccontextmanager
async def agent_lifespan(app):
    yield
    controller = getattr(app.state, "agent_controller", None)
    if controller:
        await controller.close()


router = APIRouter(tags=["agent"], route_class=StableValidationRoute, lifespan=agent_lifespan)


def _call(fn):
    try:
        return fn()
    except LLMUnavailableError as exc:
        detail = {"code": exc.code, "message": str(exc)}
        if exc.upstream_status is not None:
            detail["upstream_status"] = exc.upstream_status
        raise HTTPException(502, detail=detail) from exc
    except (AgentError, IndicatorDomainError) as exc:
        headers = {"Retry-After": "1"} if exc.code in {"INDICATOR_ENGINE_BUSY", "AGENT_SERVICE_BUSY"} else None
        raise HTTPException(exc.status_code, detail=exc.detail(), headers=headers) from exc


def resolve_service(request: Request):
    service = getattr(request.app.state, "agent_indicator_service", None)
    if service is None:
        raise AgentError("AGENT_SERVICE_UNAVAILABLE", "智能体业务服务尚未挂载。", status_code=503)
    return service


def session_store():
    return AgentSessionStore()


def controller(request):
    harness.require_controller_loop()
    instance = getattr(request.app.state, "agent_controller", None)
    if instance is None:
        callbacks = getattr(request.app.state, 'agent_page_services', {})
        instance = harness.RunController(session_store(), page_services=callbacks)
        request.app.state.agent_controller = instance
    instance.reconcile()
    return instance


def _llm_client(session_id):
    client = build_client(LlmSettingsStore().read(), session_id=session_id)
    if client is None:
        raise AgentError("AGENT_NOT_CONFIGURED", "尚未配置 LLM API，请在设置中保存后重试。", status_code=503, field="llm")
    return client


@router.get("/api/agent/meta")
def agent_meta(request: Request):
    service = _call(lambda: resolve_service(request))
    settings = LlmSettingsStore().public()
    try:
        version = build_catalog(service)["version"]
    except Exception:
        version = ""
    return {"configured": settings["configured"], "provider": settings["provider"] if settings["configured"] else None,
            "model": settings["model"] if settings["configured"] else None, "catalog_version": version,
            "schema_version": 2, "execution_policy": {"no_progress_detection": True, "recoverable_runs": True},
            "scopes": [{"id": scope, "label": SCOPE_LABELS[scope], "pages": sorted(p for p, s in PAGE_SCOPES.items() if s == scope), "tools": list(SCOPE_TOOLS[scope])} for scope in SCOPE_TOOLS],
            "limits": harness.execution_limits()}


@router.post("/api/agent/sessions", status_code=201)
def create_agent_session(request: AgentSessionCreate):
    def run():
        scope = request.scope or scope_for_page(request.page_context.page)
        validate_scope_for_page(scope, request.page_context.page)
        state = session_store().create(page_context=request.page_context, scope=scope)
        return {"session_id": state["session_id"], "session_revision": 0, "scope": scope, "state": "idle", "page_context": state["page_context"]}
    return _call(run)


@router.get("/api/agent/sessions")
def list_agent_sessions(limit: int = Query(default=20, ge=1, le=100)):
    return _call(lambda: {"items": session_store().list_sessions(limit)})


@router.get("/api/agent/sessions/{session_id}")
async def get_agent_session(session_id: str, request: Request, event_from: int = Query(default=1, ge=1, le=2**63-1)):
    return _call(lambda: controller(request).store.public(session_id, event_from=event_from))


@router.get("/api/agent/sessions/{session_id}/previews/{preview_id}")
async def get_agent_preview(session_id: str, preview_id: str, request: Request, historical: bool = False):
    # Controller ownership stays on the loop; only persisted preview I/O uses a worker.
    store = _call(lambda: controller(request).store)
    return await run_in_threadpool(_call, lambda: store.preview(session_id, preview_id, historical=historical))


@router.post("/api/agent/sessions/{session_id}/scope")
def set_agent_scope(session_id: str, request: AgentScopeRequest):
    def run():
        store = session_store()
        with store.locked(session_id) as state:
            assert_mutable(state)
            validate_scope_for_page(request.scope, state["page_context"]["page"])
            if state["scope"] != request.scope:
                state.update(scope=request.scope, session_revision=state["session_revision"]+1, pending_confirmation=None)
                append_event(state, {"type": "scope", "speaker": "human", "scope": request.scope})
                store.write(state)
            return {"session_id": session_id, "scope": state["scope"], "session_revision": state["session_revision"], "updated_at": state.get("updated_at", utc_now())}
    return _call(run)


@router.post("/api/agent/sessions/{session_id}/messages")
async def post_agent_message(session_id: str, request: AgentMessageRequest, http_request: Request,
                             response_mode: Literal["sync", "async"] = "sync"):
    def start():
        validate_page_context(request.page_context, pit_off=str(http_request.headers.get("x-pit-off", "")).lower() in {"1", "true", "yes"}, allow_authoring=True)
        instance = controller(http_request)
        if instance.store.find_message(session_id, request.message_id):
            existing, replayed = instance.store.accept(session_id, request, {})
            return instance, existing, replayed
        llm = _llm_client(session_id)
        return instance, *instance.submit(session_id, request, llm, resolve_service(http_request))
    instance, run, replayed = _call(start)
    if response_mode == "async":
        return JSONResponse({**public_run(run), "replayed": replayed}, status_code=202)
    run = await instance.wait(run)
    if run.get("error"):
        return JSONResponse({"detail": {**run["error"], "run_id": run["run_id"]}}, status_code=502)
    return {**(run.get("response") or public_run(run)), "replayed": replayed}


@router.get("/api/agent/sessions/{session_id}/runs/{run_id}")
async def get_agent_run(session_id: str, run_id: str, request: Request):
    return _call(lambda: public_run(controller(request).store.get_run(session_id, run_id)))


@router.post("/api/agent/sessions/{session_id}/runs/{run_id}/cancel")
async def cancel_agent_run(session_id: str, run_id: str, body: AgentCancelRequest, request: Request):
    run = _call(lambda: controller(request).cancel(session_id, run_id))
    return JSONResponse(public_run(run), status_code=202 if run["status"] == "stopping" else 200)


@router.post("/api/agent/sessions/{session_id}/runs/{run_id}/invalidate-context")
async def invalidate_agent_context(session_id: str, run_id: str, body: AgentInvalidateRequest, request: Request):
    run = _call(lambda: controller(request).cancel(session_id, run_id, reason="context_changed", context=body.page_context))
    return public_run(run)


@router.get("/api/agent/sessions/{session_id}/events")
async def agent_events(session_id: str, request: Request, after_seq: int = Query(default=0, ge=0, le=2**63-1),
                       limit: int = Query(default=200, ge=1, le=200), stream: bool = False,
                       kind: Literal["all", "messages"] = "all", before_seq: int | None = Query(default=None, ge=1, le=2**63-1)):
    store = _call(lambda: controller(request).store)
    _call(lambda: store.read(session_id))
    if kind == "messages":
        if stream:
            raise HTTPException(422, detail="历史消息分页不使用流式模式。")
        return _call(lambda: store.message_page(session_id, before_seq=before_seq, limit=limit))
    if not stream:
        return _call(lambda: store.events(session_id, after_seq=after_seq, limit=limit))
    header = request.headers.get("last-event-id")
    if header:
        try:
            parsed_cursor = int(header)
            if not 0 <= parsed_cursor <= 2**63-1:
                raise ValueError
            after_seq = max(after_seq, parsed_cursor)
        except ValueError:
            raise HTTPException(422, detail={"code": "AGENT_EVENT_CURSOR_INVALID", "message": "事件游标无效。"}) from None
    async def generate():
        cursor = after_seq
        ticks = 0
        while not await request.is_disconnected():
            page = store.events(session_id, after_seq=cursor, limit=limit)
            for event in page["items"]:
                cursor = event["seq"]
                yield f"id: {cursor}\nevent: agent\ndata: {stable_json(event)}\n\n"
            if page["has_more"]:
                continue
            state = store.read(session_id)
            if not state.get("active_run_id") and not state.get("execution_blocked_by"):
                break
            if ticks % 50 == 0:
                yield ": heartbeat\n\n"
            ticks += 1
            await asyncio.sleep(.3)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.post("/api/agent/sessions/{session_id}/commit-preview")
def agent_commit_preview(session_id: str, request: CommitPreviewRequest, http_request: Request):
    return _call(lambda: commit_flow.preview(store=session_store(), session_id=session_id, request=request, service=resolve_service(http_request)))


@router.post("/api/agent/sessions/{session_id}/commit")
def agent_commit(session_id: str, request: CommitRequest, http_request: Request):
    return _call(lambda: commit_flow.commit(store=session_store(), session_id=session_id, request=request, service=resolve_service(http_request)))


@router.post("/api/agent/sessions/{session_id}/memory")
def agent_memory(session_id: str, request: MemoryRequest):
    return _call(lambda: memory.resolve(store=session_store(), session_id=session_id, request=request))


@router.get("/api/agent/sessions/{session_id}/memory")
def agent_memory_sources(session_id: str):
    def read():
        store = session_store()
        state = store.read(session_id)
        targets = ((state.get('page_context') or {}).get('calculation') or {}).get('targets', [])
        return {'items': memory.recall(store=store, session_id=session_id,
                                      object_ids=[target['product_id'] for target in targets]),
                'legacy_unavailable': any(item.get('decision') == 'accept' and not item.get('memory_id')
                                          for item in (state.get('memory') or {}).values() if isinstance(item, dict))}
    return _call(read)


@router.post("/api/agent/sessions/{session_id}/memory/revoke")
def agent_memory_revoke(session_id: str, request: MemoryRevokeRequest):
    return _call(lambda: memory.revoke(store=session_store(), session_id=session_id, request=request))
