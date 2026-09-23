"""HTTP validation and registration for the shared product operations."""
from __future__ import annotations

from typing import Literal, Optional
from fastapi import APIRouter, Query
from . import instrument_service as service
from .instrument_service import ProductAnalysisRequest, ProductCompareAnalysisRequest

router = APIRouter(prefix="/api/instruments", tags=["instruments"])


@router.get("/analytics")
def instrument_analytics(
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    return service.instrument_analytics(
        kind=kind,
        fund_type=fund_type,
        invest_type=invest_type,
        status=status,
        management=management,
        market=market,
    )


@router.get("/analytics/trend")
def instrument_analytics_trend(
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    dimension: Literal["all", "fund_type", "invest_type", "management"] = Query(default="all"),
    values: Optional[list[str]] = Query(default=None),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    return service.instrument_analytics_trend(
        kind=kind,
        dimension=dimension,
        values=values,
        fund_type=fund_type,
        invest_type=invest_type,
        status=status,
        management=management,
        market=market,
    )


@router.get("/analytics/rankings")
def instrument_analytics_rankings(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    metric: str = Query(default="return_1y"),
    sort_dir: Literal["asc", "desc"] = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    active_only: bool = Query(default=True),
    fund_type: Optional[list[str]] = Query(default=None),
    invest_type: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
):
    return service.instrument_analytics_rankings(
        kind=kind,
        metric=metric,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
        active_only=active_only,
        fund_type=fund_type,
        invest_type=invest_type,
        status=status,
        management=management,
        market=market,
    )


@router.get("/search")
def instrument_search(
    q: str = Query(default=""),
    kind: Literal["all", "etf", "fund"] = Query(default="all"),
    sort_by: str = Query(default="name"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=200),
):
    return service.instrument_search(
        q=q,
        kind=kind,
        sort_by=sort_by,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )


@router.get("/products")
def instrument_products(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    q: str = Query(default=""),
    fund_type: Optional[list[str]] = Query(default=None),
    fund_category: Optional[list[str]] = Query(default=None, alias="type"),
    invest_type: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    custodian: Optional[list[str]] = Query(default=None),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=200),
    sort_by: str = Query(default="issue_amount"),
    sort_dir: Literal["asc", "desc"] = Query(default="desc"),
    conditions: Optional[list[str]] = Query(default=None, alias="condition"),
    snapshot_metrics: Optional[list[str]] = Query(default=None, alias="snapshot_metric"),
    qdii_type: Optional[list[str]] = Query(default=None),
):
    return service.instrument_products(
        kind=kind,
        q=q,
        fund_type=fund_type,
        fund_category=fund_category,
        invest_type=invest_type,
        market=market,
        status=status,
        management=management,
        custodian=custodian,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_dir=sort_dir,
        conditions=conditions,
        snapshot_metrics=snapshot_metrics,
        qdii_type=qdii_type,
    )


@router.get("/products/selection")
def instrument_product_selection(
    kind: Literal["etf", "fund"] = Query(default="etf"),
    q: str = Query(default=""),
    fund_type: Optional[list[str]] = Query(default=None),
    fund_category: Optional[list[str]] = Query(default=None, alias="type"),
    invest_type: Optional[list[str]] = Query(default=None),
    market: Optional[list[str]] = Query(default=None),
    status: Optional[list[str]] = Query(default=None),
    management: Optional[list[str]] = Query(default=None),
    custodian: Optional[list[str]] = Query(default=None),
    sort_by: str = Query(default="name"),
    sort_dir: Literal["asc", "desc"] = Query(default="asc"),
    conditions: Optional[list[str]] = Query(default=None, alias="condition"),
    qdii_type: Optional[list[str]] = Query(default=None),
):
    return service.instrument_product_selection(
        kind=kind,
        q=q,
        fund_type=fund_type,
        fund_category=fund_category,
        invest_type=invest_type,
        market=market,
        status=status,
        management=management,
        custodian=custodian,
        sort_by=sort_by,
        sort_dir=sort_dir,
        conditions=conditions,
        qdii_type=qdii_type,
    )


@router.post("/products/{product_id}/analysis")
def instrument_product_analysis(
    product_id: str,
    request: ProductAnalysisRequest,
    kind: Literal["etf", "fund"] = Query(default="etf"),
):
    return service.instrument_product_analysis(product_id=product_id, request=request, kind=kind)


@router.post("/products/{product_id}/compare-analysis")
def instrument_product_compare_analysis(
    product_id: str,
    request: ProductCompareAnalysisRequest,
    kind: Literal["etf", "fund"] = Query(default="etf"),
):
    return service.instrument_product_compare_analysis(product_id=product_id, request=request, kind=kind)


@router.get("/products/{product_id}")
def instrument_product_detail(
    product_id: str,
    kind: Literal["etf", "fund"] = Query(default="etf"),
    include_timeseries: bool = Query(default=True),
):
    return service.instrument_product_detail(
        product_id=product_id,
        kind=kind,
        include_timeseries=include_timeseries,
    )


@router.get("/products/{product_id}/price-series")
def instrument_product_price_series(
    product_id: str,
    kind: Literal["etf", "fund"] = Query(default="etf"),
    basis: str = Query(default="raw_kline"),
):
    return service.instrument_product_price_series(product_id=product_id, kind=kind, basis=basis)
