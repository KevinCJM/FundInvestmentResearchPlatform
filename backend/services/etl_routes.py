"""ETL control endpoints, sharing source-center write and request boundaries."""
from __future__ import annotations

import re
from fastapi import APIRouter, HTTPException, Request

from .data_source_routes import body, get_store, invoke, revision
try:
    from backend.data_sources import etl_service
    from backend.data_sources.etl_store import EtlStore, public_run
    from backend.data_sources.etl_templates import tushare_fund_workflow
    from backend.data_sources.models import CenterError, ID_PATTERN
except ModuleNotFoundError:
    from data_sources import etl_service
    from data_sources.etl_store import EtlStore, public_run
    from data_sources.etl_templates import tushare_fund_workflow
    from data_sources.models import CenterError, ID_PATTERN

router = APIRouter(prefix="/api/data-sources/etl", tags=["etl-workflows"])


@router.get("/templates/tushare-funds")
async def tushare_template():
    def load():
        definition = tushare_fund_workflow(get_store())
        etl_service.inspect_plan(get_store(), definition)
        return {"definition": definition.model_dump(mode="json"), "expected_revision": 0}
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


@router.post("/runs")
async def start(request: Request):
    return await invoke(etl_service.start, get_store(), await body(request))


@router.get("/runs")
async def runs():
    def load():
        journal = EtlStore(get_store())
        return [public_run(journal.interrupted(run), detail=False) for run in journal.runs()]
    return await invoke(load)


@router.get("/runs/{identifier}")
async def get_run(identifier: str):
    def load():
        journal = EtlStore(get_store())
        return public_run(journal.interrupted(journal.get_run(identifier)))
    return await invoke(load)


@router.post("/runs/{identifier}/cancel")
async def cancel(identifier: str, request: Request):
    await body(request)
    return await invoke(etl_service.cancel, get_store(), identifier)


@router.post("/runs/{identifier}/resume")
async def resume(identifier: str, request: Request):
    payload = await body(request)
    return await invoke(etl_service.resume, get_store(), identifier, payload.get("confirm") is True)
