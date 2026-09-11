"""ETL control endpoints, sharing source-center write and request boundaries."""
from __future__ import annotations

import re
from typing import Literal
from fastapi import APIRouter, HTTPException, Request

from .data_source_routes import body, get_store, invoke, revision
from . import etl_recovery
try:
    from backend.data_sources import etl_service
    from backend.data_sources.etl_store import EtlStore, public_run
    from backend.data_sources.etl_executor import observe
    from backend.data_sources.etl_templates import tushare_fund_workflow, template_catalog
    from backend.data_sources.etl_dependencies import plan_dependencies as plan_task_dependencies
    from backend.data_sources.task_catalog import task_catalog as registered_tasks
    from backend.data_sources.models import CenterError, ID_PATTERN
except ModuleNotFoundError:
    from data_sources import etl_service
    from data_sources.etl_store import EtlStore, public_run
    from data_sources.etl_executor import observe
    from data_sources.etl_templates import tushare_fund_workflow, template_catalog
    from data_sources.etl_dependencies import plan_dependencies as plan_task_dependencies
    from data_sources.task_catalog import task_catalog as registered_tasks
    from data_sources.models import CenterError, ID_PATTERN

router = APIRouter(prefix="/api/data-sources/etl", tags=["etl-workflows"])


@router.get("/templates/tushare-funds")
async def tushare_template():
    def load():
        definition = tushare_fund_workflow(get_store())
        etl_service.inspect_plan(get_store(), definition)
        return {"definition": definition.model_dump(mode="json"), "expected_revision": 0}
    return await invoke(load)


@router.get('/tasks')
async def task_catalog():
    def load():
        return registered_tasks(get_store())
    return await invoke(load)


@router.get('/templates')
async def templates():
    def load():
        return template_catalog(get_store())
    return await invoke(load)


@router.get("/workflows")
async def workflows():
    return await invoke(lambda: EtlStore(get_store()).workflows())


@router.put("/workflows/{identifier}")
async def save_workflow(identifier: str, request: Request):
    payload = await body(request)
    if not re.fullmatch(ID_PATTERN, identifier):
        raise HTTPException(422, detail={"code": "ETL_ID_INVALID", "message": "流程 ID 格式无效。"})
    return await invoke(etl_service.save_workflow, get_store(), identifier, payload.get("definition"), revision(payload))


@router.delete("/workflows/{identifier}")
async def delete_workflow(identifier: str, request: Request):
    payload = await body(request)
    def remove():
        etl_service._writable()
        EtlStore(get_store()).delete_workflow(identifier, revision(payload))
        return {"deleted": True}
    return await invoke(remove)


@router.post("/validate")
async def validate(request: Request):
    payload = await body(request)
    return await invoke(etl_service.validate, get_store(), payload.get("definition"), payload.get("options"))


@router.post('/dependencies/plan')
async def plan_dependencies(request: Request):
    payload = await body(request)
    def plan():
        definition = plan_task_dependencies(etl_service.parse_definition(payload.get('definition')))
        etl_service.inspect_plan(get_store(), definition)
        return {'definition': definition.model_dump(mode='json'), 'published': False}
    return await invoke(plan)


@router.post("/runs")
async def start(request: Request):
    return await invoke(etl_service.start, get_store(), await body(request))


@router.get("/runs")
async def runs(view: Literal['all', 'current'] = 'all'):
    def load():
        store = get_store()
        journal = EtlStore(store)
        if view == 'current':
            from .etl_run_view import current_run_views, read_run_records
            records = read_run_records(store)
            groups = current_run_views(records)
            items = [journal.interrupted(run) for run, _, _ in groups]
        else:
            groups = []
            items = [journal.interrupted(run) for run in journal.runs()]
        successors = etl_recovery.successor_map(store, records=records if view == 'current' else None)
        fingerprint = etl_service.execution_fingerprint() if any(run['status'] in {'FAILED', 'CANCELLED', 'INTERRUPTED'} for run in items) else None
        values = [_run_with_recovery(store, run, detail=False, fingerprint=fingerprint, successors=successors) for run in items]
        for value, (_, metadata, chain) in zip(values, groups):
            value['history'] = metadata
            # During migration the stopped successor may exist before resume;
            # keep the original recovery job visible in the same card.
            if value['status'] not in {'RUNNING', 'SUCCEEDED'}:
                for ancestor in chain:
                    job = etl_recovery.job_status(store, ancestor['run_id'])
                    if job and job['status'] in etl_recovery.ACTIVE:
                        value['recovery'] = {**value.get('recovery', {}), 'job': job}
                        break
        return values
    return await invoke(load)


@router.get("/runs/{identifier}")
async def get_run(identifier: str):
    def load():
        store = get_store()
        journal = EtlStore(store)
        return _run_with_recovery(store, journal.interrupted(journal.get_run(identifier)))
    return await invoke(load)


def _run_with_recovery(store, run, *, detail=True, fingerprint=None, successors=None):
    value = public_run(observe(EtlStore(store), run), detail=detail)
    if run['status'] in {'FAILED', 'CANCELLED', 'INTERRUPTED'}:
        value['recovery'] = etl_recovery.describe(store, run, etl_service.recovery_status(store, run, fingerprint), successors)
    if run['status'] == 'INTERRUPTED' and run.get('error') == '服务已退出。已完成步骤保留，可继续未完成步骤。':
        value['error'] = '调度服务已中断，已完成步骤保留；是否可继续以按钮上方的恢复检查为准。'
    return value


@router.post("/runs/{identifier}/cancel")
async def cancel(identifier: str, request: Request):
    await body(request)
    return await invoke(etl_service.cancel, get_store(), identifier)


@router.post("/runs/{identifier}/resume")
async def resume(identifier: str, request: Request):
    payload = await body(request)
    def perform():
        store = get_store()
        etl_recovery.guard_successor(store, identifier)
        return etl_service.resume(store, identifier, payload.get('confirm') is True)
    return await invoke(perform)


@router.post('/runs/{identifier}/recovery', status_code=202)
async def recover(identifier: str, request: Request):
    return await invoke(etl_recovery.start, get_store(), identifier, await body(request))


@router.get('/runs/{identifier}/recovery')
async def recovery_progress(identifier: str):
    return await invoke(etl_recovery.job_status, get_store(), identifier)
