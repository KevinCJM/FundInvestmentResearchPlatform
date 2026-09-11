"""Factor research API. Router factory also enables network-free integration tests."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import ValidationError as ContractError
from backend.custom_indicators.errors import IndicatorDomainError
from backend.factor_research.contracts import (
    AttributionFields, BindingFields, DatasetFields, FactorFields, FactorUpdate,
    PortfolioProfile, ReleaseFields, RunRequest, StudyFields, StudyUpdate,
)
from backend.factor_research.service import FactorResearchService, ROOT_DATA
from backend.factor_research.return_contracts import FF3SourceFields, ReturnDatasetFields, ReturnPlanFields, ReturnPlanUpdate


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except IndicatorDomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc
    except ContractError as exc:
        raise HTTPException(status_code=422, detail={"code": "FACTOR_INVALID_REQUEST", "message": str(exc)}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=422, detail={"code": "FACTOR_DATA_MISSING", "message": "缺少所需数据文件，请在数据同步中补齐。"}) from exc


def build_router(service: FactorResearchService):
    router = APIRouter(prefix="/api/factor-research", tags=["factor-research"])

    @router.get("/catalog")
    def catalog():
        return call(service.catalog)

    @router.get("/products")
    def products(kind: Literal["etf", "fund", "stock"] = "etf", query: str = Query("", max_length=80), limit: int = Query(50, ge=1, le=120)):
        return call(service.products, kind, query, limit)

    @router.get("/factors")
    def factors():
        return {"items": service.factors.list()}

    @router.post("/factors", status_code=201)
    def create_factor(request: FactorFields):
        return call(service.save_factor, request.model_dump(mode="json"))

    @router.put("/factors/{object_id}")
    def update_factor(object_id: str, request: FactorUpdate):
        return call(service.save_factor, request.model_dump(mode="json", exclude={"revision"}), object_id, request.revision)

    @router.get("/studies")
    def studies():
        return {"items": service.studies.list()}

    @router.post("/studies", status_code=201)
    def create_study(request: StudyFields):
        return call(service.save_study, request.model_dump(mode="json"))

    @router.put("/studies/{object_id}")
    def update_study(object_id: str, request: StudyUpdate):
        return call(service.save_study, request.model_dump(mode="json", exclude={"revision"}), object_id, request.revision)

    @router.post("/studies/{object_id}/runs", status_code=201)
    def run_study(object_id: str, request: RunRequest):
        return call(service.run_study, object_id, request.revision)

    @router.get("/runs")
    def runs():
        return service.list_runs()

    @router.get("/runs/{object_id}")
    def run(object_id: str):
        value = call(service.artifacts.get, object_id)
        if value["kind"] not in ("run", "attribution"):
            raise HTTPException(422, detail={"code": "FACTOR_RUN_KIND", "message": "请选择特征研究或归因运行。"})
        return value

    @router.get("/datasets")
    def datasets():
        return call(service.datasets)

    @router.post("/datasets", status_code=201)
    def dataset(request: DatasetFields):
        return call(service.add_dataset, request.model_dump(mode="json"))

    @router.get("/return-catalog")
    def return_catalog():
        return {**service.return_research.catalog(), "ready": service._ready}

    @router.get("/return-plans")
    def return_plans():
        return {"items": service.return_research.plans.list()}

    @router.post("/return-plans", status_code=201)
    def create_return_plan(request: ReturnPlanFields):
        return call(service.return_research.save_plan, request.model_dump(mode="json"))

    @router.put("/return-plans/{object_id}")
    def update_return_plan(object_id: str, request: ReturnPlanUpdate):
        return call(service.return_research.save_plan, request.model_dump(mode="json", exclude={"revision"}), object_id, request.revision)

    @router.post("/return-plans/{object_id}/runs", status_code=201)
    def run_return_plan(object_id: str, request: RunRequest):
        return call(service.return_research.run, object_id, request.revision)

    @router.get("/return-sources")
    def return_sources():
        return call(service.return_research.sources)

    @router.post("/return-sources", status_code=201)
    def create_return_source(request: FF3SourceFields):
        return call(service.return_research.import_source, request.model_dump(mode="json"))

    @router.post("/return-datasets", status_code=201)
    def create_return_dataset(request: ReturnDatasetFields):
        return call(service.return_research.import_dataset, request.model_dump(mode="json"))

    @router.get("/return-datasets/{object_id}")
    def return_dataset(object_id: str):
        return call(service.return_research.dataset, object_id)

    @router.get("/return-datasets/{object_id}/export")
    def export_return_dataset(object_id: str):
        text = call(service.return_research.export_csv, object_id)
        return Response(text, media_type="text/csv", headers={
            "Content-Disposition": f'attachment; filename="{object_id}.csv"',
        })

    @router.get("/attributions")
    def attributions():
        return {"items": service.artifacts.list("attribution")}

    @router.post("/attributions", status_code=201)
    def attribution(request: AttributionFields):
        return call(service.run_attribution, request.model_dump(mode="json"))

    @router.get("/releases")
    def releases(product_id: str | None = Query(None, max_length=32)):
        return call(service.releases, product_id)

    @router.post("/releases", status_code=201)
    def publish(request: ReleaseFields):
        return call(service.publish, request.model_dump(mode="json"))

    @router.get("/releases/{object_id}")
    def release(object_id: str):
        return call(service.release, object_id)

    @router.post("/releases/{object_id}/retire")
    def retire(object_id: str):
        return call(service.retire, object_id)

    @router.get("/releases/{object_id}/monitor")
    def monitor(object_id: str):
        return call(service.monitor, object_id)

    @router.post("/releases/{object_id}/portfolio-profile", status_code=201)
    def portfolio_profile(object_id: str, request: PortfolioProfile):
        return call(service.portfolio_profile, object_id, request.model_dump(mode="json"))

    @router.get("/bindings")
    def bindings(context_type: str | None = Query(None, max_length=40), context_id: str | None = Query(None, max_length=120)):
        return call(service.get_bindings, context_type, context_id)

    @router.post("/bindings", status_code=201)
    def bind(request: BindingFields):
        return call(service.bind, request.model_dump(mode="json"))

    return router


factor_service = FactorResearchService(
    workspace_data_dir=Path(os.getenv("CUSTOM_INDICATOR_DATA_DIR") or ROOT_DATA),
    market_data_dir=ROOT_DATA,
)
router = build_router(factor_service)
