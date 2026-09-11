"""Domain-separated research APIs. Preview is ephemeral; publish persists."""
from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError as ContractError

from backend.custom_indicators.errors import IndicatorDomainError
from backend.data_sources.models import CenterError
from backend.factor_research.data import product_catalog
from backend.market_data import MarketDataManifestError, resolve_tushare_data_dir
from backend.sensitivity.contracts import (
    CashflowPublishRequest, CashflowStudy, ModelFields, PublishRequest, RetireRequest, SeriesImport,
)
from backend.sensitivity.service import ModelResearchService

try:
    from custom_indicators.errors import IndicatorDomainError as DirectDomainError
except ImportError:
    DirectDomainError = IndicatorDomainError


def call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except (IndicatorDomainError, DirectDomainError) as exc:
        raise HTTPException(exc.status_code, detail=exc.detail()) from exc
    except CenterError as exc:
        raise HTTPException(
            getattr(exc, "status_code", getattr(exc, "status", 503)),
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except ContractError as exc:
        raise HTTPException(
            422,
            detail={"code": "RESEARCH_CONTRACT_INVALID", "message": "输入未通过校验，请检查日期、必填项和参数范围。"},
        ) from exc
    except (FileNotFoundError, MarketDataManifestError) as exc:
        raise HTTPException(
            422,
            detail={
                "code": "RESEARCH_DATA_MISSING",
                "message": "所需数据或磁盘快照不可用，请在数据管理中检查；不会使用示例数据替代。",
            },
        ) from exc


def build_router(service: ModelResearchService):
    prefix = "/api/risk-models" if service.domain == "product" else "/api/scenario-transmission"
    router = APIRouter(prefix=prefix, tags=[service.domain + "-sensitivity"])

    @router.get("/catalog")
    def catalog():
        return call(service.catalog)

    @router.get("/products")
    def products(kind: Literal["etf", "fund"] = "etf", query: str = Query("", max_length=80)):
        return {
            "items": call(
                product_catalog,
                call(resolve_tushare_data_dir, service.data_dir, strict=True),
                kind,
                query,
                50,
            )
        }

    @router.post("/variables", status_code=201)
    def import_variable(request: SeriesImport):
        return call(service.variables.import_series, request.model_dump(mode="json"))

    @router.post("/previews")
    def preview_model(request: ModelFields):
        return call(service.preview, request.model_dump(mode="json"))

    @router.get("/runs")
    def runs():
        return call(service.runs)

    @router.get("/runs/{identifier}")
    def run(identifier: str):
        return call(service.get_run, identifier)

    @router.post("/releases", status_code=201)
    def publish(request: PublishRequest):
        return call(service.publish, request.model_dump(mode="json"))

    @router.get("/releases")
    def releases(
        as_of: date | None = None,
        product_key: str | None = Query(None, max_length=120),
        include_unavailable: bool = True,
    ):
        return call(service.releases, as_of, product_key, include_unavailable)

    @router.post("/releases/{identifier}/retire")
    def retire(identifier: str, request: RetireRequest):
        return call(service.retire, identifier, request.model_dump(mode="json"))

    if service.domain == "product":
        @router.post("/cashflow-previews")
        def cashflow_preview(request: CashflowStudy):
            return call(service.cashflow_preview, request.model_dump(mode="json"))

        @router.post("/cashflow-releases", status_code=201)
        def cashflow_publish(request: CashflowPublishRequest):
            return call(service.publish_cashflow, request.model_dump(mode="json"))

    return router


risk_model_service = ModelResearchService()
transmission_service = ModelResearchService(domain="transmission")
risk_model_router = build_router(risk_model_service)
transmission_router = build_router(transmission_service)
