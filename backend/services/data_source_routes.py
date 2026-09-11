"""Bounded, local-only source-center writes; never echo credentials or input errors."""
from __future__ import annotations
import json
import os
from ipaddress import ip_address
from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from backend.data_storage import StorageError

try:
    from backend.data_sources import service
    from backend.data_sources.mapping import preview, validate_mapping
    from backend.data_sources.models import CenterError
    from backend.data_sources.store import SourceStore
    from backend.data_sources import resolution_store, sync_jobs
    from backend.data_sources.resolution import resolve_records
    from backend.data_sources.resolution_models import ResolutionConfig
except ModuleNotFoundError:
    from data_sources import service
    from data_sources.mapping import preview, validate_mapping
    from data_sources.models import CenterError
    from data_sources.store import SourceStore
    from data_sources import resolution_store, sync_jobs
    from data_sources.resolution import resolve_records
    from data_sources.resolution_models import ResolutionConfig

router = APIRouter(prefix="/api/data-sources", tags=["data-source-center"])


@lru_cache(maxsize=1)
def get_store() -> SourceStore:
    store = SourceStore()
    store.seed()
    return store


def require_same_origin(request: Request) -> None:
    origin = request.headers.get("origin")
    if not origin:
        if request.headers.get("sec-fetch-site") == "cross-site":
            raise HTTPException(403, detail={"code": "CROSS_SITE_WRITE", "message": "不允许跨站修改数据源。"})
        return
    parsed = urlsplit(origin)
    trusted = {value.strip() for value in os.getenv("DATA_SOURCE_ALLOWED_ORIGINS", "").split(",") if value.strip()}
    own = str(request.base_url).rstrip("/")
    loopback = {"localhost", "127.0.0.1", "::1"}
    local_dev = parsed.hostname in loopback and request.url.hostname in loopback and request.client is not None and request.client.host in loopback
    if origin != own and origin not in trusted and not local_dev:
        raise HTTPException(403, detail={"code": "CROSS_SITE_WRITE", "message": "不允许跨站修改数据源。"})


async def body(request: Request) -> dict:
    # Origin is CSRF protection, not authentication. CLI clients can omit it.
    # Only the socket peer is trusted; forwarded headers never grant access.
    try:
        peer = ip_address(request.client.host) if request.client else None
        if peer is not None and peer.version == 6 and peer.ipv4_mapped is not None:
            peer = peer.ipv4_mapped
        peer_is_local = peer is not None and peer.is_loopback
    except ValueError:
        peer_is_local = False
    if not peer_is_local or request.url.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise HTTPException(403, detail={
            'code': 'SOURCE_LOCAL_ONLY',
            'message': '数据源配置和任务操作仅允许后端本机访问，请使用 localhost 或 127.0.0.1。',
        })
    require_same_origin(request)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 1048576:
            raise HTTPException(413, detail={"code": "BODY_TOO_LARGE", "message": "配置或样本不能超过 1MB。"})
        chunks.append(chunk)
    try:
        result = json.loads(b"".join(chunks))
    except (ValueError, RecursionError):
        raise HTTPException(422, detail={"code": "INVALID_JSON", "message": "请求必须为有效 JSON 对象。"}) from None
    if not isinstance(result, dict):
        raise HTTPException(422, detail={"code": "INVALID_JSON", "message": "请求必须为 JSON 对象。"})
    return result


def revision(payload: dict) -> int:
    value = payload.get("expected_revision")
    if type(value) is not int or value < 0:
        raise HTTPException(422, detail={"code": "REVISION_REQUIRED", "message": "请提供有效配置修订号。"})
    return value


async def invoke(operation, *args):
    try:
        return await run_in_threadpool(operation, *args)
    except (CenterError, StorageError) as exc:
        raise HTTPException(exc.status, detail={"code": exc.code, "message": exc.message}) from None
    except (ValueError, TypeError):
        raise HTTPException(422, detail={"code": "INVALID_CONFIGURATION", "message": "配置格式、约束或样本无效。"}) from None
    except Exception:
        raise HTTPException(500, detail={"code": "SOURCE_CENTER_FAILED", "message": "数据源操作未完成，请检查服务状态。"}) from None


@router.get("/catalog")
async def get_catalog():
    return await invoke(service.catalog, get_store())


@router.put("/config/{kind}")
async def save_config(kind: Literal["source", "interface"], request: Request):
    payload = await body(request)
    return await invoke(service.save, get_store(), kind, payload.get("config"), revision(payload))


@router.delete("/config/{kind}/{identifier}")
async def delete_config(kind: Literal["source", "interface"], identifier: str, request: Request):
    payload = await body(request)
    def remove():
        with service.mutation_lock(get_store()):
            get_store().delete(kind, identifier, revision(payload))
        return {"deleted": True}
    return await invoke(remove)


@router.put("/credentials/{source_id}")
async def update_credential(source_id: str, request: Request):
    payload = await body(request)
    value = payload.get("value")
    if value is not None and not isinstance(value, str):
        raise HTTPException(422, detail={"code": "INVALID_CREDENTIAL", "message": "凭据必须是文本。"})
    return await invoke(service.set_credential, get_store(), source_id, value)


@router.post("/validate")
async def validate_config(request: Request):
    payload = await body(request)
    def validate():
        return validate_mapping(service.parse_config("interface", payload.get("config")))
    return await invoke(validate)


@router.post("/preview")
async def preview_mapping(request: Request):
    payload = await body(request)
    def calculate_preview():
        return preview(service.parse_config("interface", payload.get("config")), payload.get("sample"))
    return await invoke(calculate_preview)


@router.post("/interfaces/{identifier}/sample")
async def sample_interface(identifier: str, request: Request):
    payload = await body(request)
    if payload.get("confirm") is not True:
        raise HTTPException(422, detail={"code": "CONFIRM_SAMPLE", "message": "采样会使用真实接口配额，请明确确认。"})
    params = payload.get("params", {})
    if not isinstance(params, dict):
        raise HTTPException(422, detail={"code": "INVALID_PARAMS", "message": "接口参数须为对象。"})
    return await invoke(service.sample, get_store(), identifier, params, revision(payload))


@router.get("/resolution/config")
async def resolution_config():
    return await invoke(resolution_store.get_policy, get_store())


@router.put("/resolution/config")
async def save_resolution_config(request: Request):
    payload = await body(request)
    def save():
        with service.mutation_lock(get_store()):
            return resolution_store.save_policy(get_store(), payload.get("config"), revision(payload))
    return await invoke(save)


@router.post("/resolution/preview")
async def preview_resolution(request: Request):
    payload = await body(request)
    def preview_rules():
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) > 1000 or any(not isinstance(row, dict) for row in rows):
            raise CenterError("PREVIEW_ROWS_INVALID", "离线预览须提供不超过 1000 条标准记录。")
        config = ResolutionConfig.model_validate(payload.get("config"))
        result = resolve_records(str(payload.get("table_id") or ""), rows, config, as_of=payload.get("as_of"))
        result["preview_only"] = True
        return result
    return await invoke(preview_rules)


@router.post("/resolution/run")
async def run_resolution(request: Request):
    payload = await body(request)
    def resolve():
        with service.mutation_lock(get_store()):
            return resolution_store.resolve_saved(get_store(), str(payload.get("table_id") or ""), revision(payload), payload.get("start_date"), payload.get("end_date"), payload.get("as_of"))
    return await invoke(resolve)


@router.get("/sync/jobs")
async def get_sync_jobs():
    return await invoke(sync_jobs.list_jobs, get_store())


@router.post("/interfaces/{identifier}/sync")
async def sync_interface(identifier: str, request: Request):
    payload = await body(request)
    if payload.get("confirm") is not True or not isinstance(payload.get("params", {}), dict):
        raise HTTPException(422, detail={"code": "CONFIRM_SYNC", "message": "请确认下载范围；此操作会访问外部数据源。"})
    return await invoke(sync_jobs.start_sync, get_store(), identifier, revision(payload), payload.get("params", {}), str(payload.get("mode") or "incremental"))
