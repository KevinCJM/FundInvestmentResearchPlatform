"""Local-only storage management independent of the possibly offline data database."""
from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from backend.data_storage import StorageError, StorageManager
from backend.data_sources.service import enabled
from .data_source_routes import body, require_same_origin

router = APIRouter(prefix='/api/data-storage', tags=['data-storage'])
manager = StorageManager()


def local(request, *, write=False):
    if not request.client or request.client.host not in {'127.0.0.1', '::1', 'testclient'}:
        raise HTTPException(403, detail={'code': 'STORAGE_LOCAL_ONLY', 'message': '存储管理仅允许后端本机访问。'})
    if request.client.host != 'testclient' and request.url.hostname not in {'localhost', '127.0.0.1', '::1'}:
        raise HTTPException(403, detail={'code': 'STORAGE_LOCAL_ONLY', 'message': '请使用 localhost 或 127.0.0.1 打开存储管理。'})
    require_same_origin(request)
    if write and not enabled():
        raise HTTPException(403, detail={'code': 'STORAGE_READ_ONLY', 'message': '当前环境禁止修改数据存储。'})


async def invoke(operation, *args):
    try:
        return await run_in_threadpool(operation, *args)
    except StorageError as exc:
        raise HTTPException(exc.status, detail={'code': exc.code, 'message': exc.message}) from None
    except OSError:
        raise HTTPException(503, detail={'code': 'STORAGE_IO_ERROR', 'message': '磁盘操作未完成，请检查连接和目录权限。'}) from None


@router.get('')
async def status(request: Request):
    local(request)
    return {**await invoke(manager.status), 'editing_enabled': enabled()}


@router.post('/probe')
async def probe(request: Request):
    local(request, write=True)
    payload = await body(request)
    return await invoke(manager.probe, payload.get('path'))


@router.put('/plan')
async def save(request: Request):
    local(request, write=True)
    payload = await body(request)
    if payload.get('confirm') is not True:
        raise HTTPException(422, detail={'code': 'STORAGE_CONFIRM_REQUIRED', 'message': '请确认只保存待迁移计划。'})
    return {**await invoke(manager.save_plan, payload.get('path'), payload.get('expected_revision')), 'editing_enabled': enabled()}


@router.post('/existing/probe')
async def probe_existing(request: Request):
    local(request, write=True)
    payload = await body(request)
    return await invoke(manager.probe_existing, payload.get('path'))


@router.put('/existing/plan')
async def attach_existing(request: Request):
    local(request, write=True)
    payload = await body(request)
    if payload.get('confirm') is not True:
        raise HTTPException(422, detail={'code': 'STORAGE_CONFIRM_REQUIRED', 'message': '请确认共用整个数据区及其配置、记录和凭据。'})
    return {**await invoke(manager.save_attachment, payload.get('path'), payload.get('expected_revision'),
                           payload.get('expected_id')), 'editing_enabled': enabled()}


@router.delete('/plan')
async def cancel(request: Request):
    local(request, write=True)
    payload = await body(request)
    return {**await invoke(manager.cancel_plan, payload.get('expected_revision')), 'editing_enabled': enabled()}
