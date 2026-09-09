"""FastAPI routes for product-pool research and investable-universe snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from custom_indicators.errors import IndicatorDomainError
from backend.custom_indicators.errors import IndicatorDomainError as FactorDomainError
from product_pools.errors import ProductPoolError
from product_pools.repository import ProductPoolRepository
from product_pools.service import ProductPoolService, version_data_as_of
from services.custom_indicator_routes import indicator_service
from services.product_pool_review import ProductPoolReviewDataService
from pit.context import PitContextError, resolve_request_context
from pit.guard import check_universe, universe_lineage

router = APIRouter(tags=["product-pools"])

DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()


def _with_pit(version: dict[str, Any]) -> dict[str, Any]:
    """Label a pool version with the research day its member list actually knew.

    A frozen member list is reproducible, not causal: read under a research day
    earlier than the data it was screened on, it is future knowledge wearing a
    historical date. The version is still returned — refusing it would break
    every existing screen — but never without saying so.
    """

    if not isinstance(version, dict):
        return version
    snapshot = version.get("snapshot") or {}
    established = version_data_as_of(version)
    try:
        context = resolve_request_context(DATA_DIR)
    except PitContextError:
        return version
    label = (
        f"产品池「{version.get('pool_name') or snapshot.get('name') or version.get('pool_id')}」"
    )
    findings = check_universe(context, established_at=established, label=label)
    lineage = universe_lineage(
        findings, established_at=established, source="product_pool_version"
    )
    plans = [
        str(item.get("plan_id"))
        for item in (version.get("evaluation_plans") or snapshot.get("evaluation_plans") or [])
        if item.get("plan_id")
    ]
    # Replay is possible exactly when the selection has reproducible inputs.
    lineage["replayable"] = bool(plans)
    lineage["replay_plan_ids"] = plans
    version["pit"] = lineage
    return version


class IndicatorEvaluationGateway:
    """Adapter that keeps the product-pool domain independent of indicator internals."""

    def get_plan(self, plan_id: str) -> dict[str, Any]:
        if plan_id.startswith("factor-release-"):
            from services.factor_research_routes import factor_service
            return factor_service.evaluation_plan(plan_id)
        return indicator_service.get_plan(plan_id)

    def run_plan(self, plan_id: str, as_of: str | None = None) -> dict[str, Any]:
        if plan_id.startswith("factor-release-"):
            from services.factor_research_routes import factor_service
            return factor_service.evaluation_run(plan_id, as_of)
        return indicator_service.run_plan(plan_id, as_of)

    def get_run_page(
        self,
        result_id: str,
        *,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        if result_id.startswith("factor-run-"):
            from services.factor_research_routes import factor_service
            return factor_service.evaluation_page(result_id, page, page_size)
        return indicator_service.get_plan_run_result(
            result_id,
            page=page,
            page_size=page_size,
        )


_workspace_dir = Path(indicator_service.workspace_data_dir)
_market_data_root = Path(__file__).resolve().parents[2] / "data"
_product_pool_repository = ProductPoolRepository(_workspace_dir / "product_pools.json")
product_pool_service = ProductPoolService(
    _product_pool_repository,
    IndicatorEvaluationGateway(),
)
product_pool_review_service = ProductPoolReviewDataService(
    _product_pool_repository,
    _market_data_root,
)


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except ProductPoolError as exc:
        detail: dict[str, Any] = {"code": exc.code, "message": exc.message}
        if exc.field:
            detail["field"] = exc.field
        raise HTTPException(status_code=exc.status_code, detail=detail) from exc
    except (IndicatorDomainError, FactorDomainError) as exc:
        # Evaluation is an upstream domain dependency of product-pool attach.
        # Preserve its stable business error instead of leaking an ASGI 500.
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


class ProductPoolCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    purpose: str = Field(default="", max_length=200)
    owner: str = Field(default="", max_length=80)


class ProductPoolUpdate(ProductPoolCreate):
    revision: int = Field(ge=1)


class EvaluationPlanAttach(BaseModel):
    revision: int = Field(ge=1)
    plan_id: str = Field(min_length=1, max_length=120)
    as_of: str | None = None
    selection_mode: Literal["all_ranked", "top_n", "top_percent"] = "all_ranked"
    selection_value: float | None = None


class ManualMemberCreate(BaseModel):
    revision: int = Field(ge=1)
    plan_id: str = Field(min_length=1, max_length=120)
    kind: Literal["etf", "fund"] | None = None
    product_id: str = Field(min_length=1, max_length=100)
    code: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=160)
    reason: str = Field(min_length=1, max_length=300)


class ProductPoolMemberFields(BaseModel):
    research_status: Literal["pending", "approved", "watch", "rejected"]
    usage_status: Literal["normal", "limited", "no_new", "hold_only", "unavailable"]
    primary_plan_id: str = Field(min_length=1, max_length=120)
    max_weight: float | None = None
    reasons: list[str] = Field(default_factory=list, max_length=20)
    owner: str = Field(default="", max_length=80)
    review_due_date: str | None = None
    valid_until: str | None = None
    substitute_group: str = Field(default="", max_length=80)


class ProductPoolMemberUpdate(ProductPoolMemberFields):
    revision: int = Field(ge=1)


class ProductPoolMemberBatchItem(ProductPoolMemberFields):
    kind: Literal["etf", "fund"]
    product_id: str = Field(min_length=1, max_length=100)


class ProductPoolMemberBatchUpdate(BaseModel):
    revision: int = Field(ge=1)
    items: list[ProductPoolMemberBatchItem] = Field(min_length=1, max_length=1000)


class ProductPoolPublish(BaseModel):
    revision: int = Field(ge=1)
    effective_from: str
    effective_to: str | None = None
    publication_note: str = Field(default="", max_length=500)


class InvestableUniverseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    research_date: str
    version_ids: list[str] = Field(min_length=1)
    excluded_product_keys: list[str] = Field(default_factory=list)


@router.get("/api/product-pools")
def list_product_pools():
    return _call(product_pool_service.list_pools)


@router.post("/api/product-pools", status_code=status.HTTP_201_CREATED)
def create_product_pool(request: ProductPoolCreate):
    return _call(product_pool_service.create_pool, request.model_dump())


@router.get("/api/product-pools/{pool_id}")
def get_product_pool(pool_id: str):
    return _call(product_pool_service.get_pool, pool_id)


@router.get("/api/product-pools/{pool_id}/review-data")
def get_product_pool_review_data(
    pool_id: str,
    basic_fields: list[str] | None = Query(default=None, alias="basic_field"),
    snapshot_metrics: list[str] | None = Query(default=None, alias="snapshot_metric"),
):
    return _call(
        product_pool_review_service.get_review_data,
        pool_id,
        basic_fields=basic_fields,
        snapshot_metrics=snapshot_metrics,
    )


@router.put("/api/product-pools/{pool_id}")
def update_product_pool(pool_id: str, request: ProductPoolUpdate):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(product_pool_service.update_pool, pool_id, revision, payload)


@router.delete("/api/product-pools/{pool_id}")
def archive_product_pool(pool_id: str, revision: int = Query(ge=1)):
    return _call(product_pool_service.archive_pool, pool_id, revision)


@router.post("/api/product-pools/{pool_id}/evaluation-plans")
def attach_evaluation_plan(pool_id: str, request: EvaluationPlanAttach):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(
        product_pool_service.attach_evaluation_plan,
        pool_id,
        revision,
        payload,
    )


@router.delete("/api/product-pools/{pool_id}/evaluation-plans/{plan_id}")
def remove_evaluation_plan(
    pool_id: str,
    plan_id: str,
    revision: int = Query(ge=1),
):
    return _call(
        product_pool_service.remove_evaluation_plan,
        pool_id,
        revision,
        plan_id,
    )


@router.post("/api/product-pools/{pool_id}/members")
def add_manual_member(pool_id: str, request: ManualMemberCreate):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(product_pool_service.add_manual_member, pool_id, revision, payload)


@router.put("/api/product-pools/{pool_id}/members/batch")
def batch_update_product_pool_members(
    pool_id: str,
    request: ProductPoolMemberBatchUpdate,
):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(
        product_pool_service.update_members,
        pool_id,
        revision,
        payload["items"],
    )


@router.put("/api/product-pools/{pool_id}/members/{kind}/{product_id}")
def update_product_pool_member(
    pool_id: str,
    kind: Literal["etf", "fund"],
    product_id: str,
    request: ProductPoolMemberUpdate,
):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(
        product_pool_service.update_member,
        pool_id,
        revision,
        kind,
        product_id,
        payload,
    )


@router.post("/api/product-pools/{pool_id}/publish")
def publish_product_pool(pool_id: str, request: ProductPoolPublish):
    payload = request.model_dump()
    revision = payload.pop("revision")
    return _call(product_pool_service.publish_pool, pool_id, revision, payload)


@router.get("/api/product-pool-versions")
def list_product_pool_versions(
    pool_id: str | None = None,
    active_on: str | None = None,
):
    return _call(
        product_pool_service.list_versions,
        pool_id=pool_id,
        active_on=active_on,
    )


@router.get("/api/product-pool-versions/{version_id}")
def get_product_pool_version(version_id: str):
    return _with_pit(_call(product_pool_service.get_version, version_id))


@router.get("/api/product-pool-versions/{version_id}/replay")
def replay_product_pool_version(version_id: str, as_of: str = Query(min_length=1)):
    """Who this pool would have contained standing on `as_of`."""

    return _call(product_pool_service.replay_version, version_id, as_of)


@router.get("/api/product-pool-versions/{version_id}/diff")
def diff_product_pool_versions(
    version_id: str,
    against: str = Query(min_length=1),
):
    return _call(product_pool_service.diff_versions, version_id, against)


@router.post(
    "/api/investable-universe-snapshots",
    status_code=status.HTTP_201_CREATED,
)
def create_investable_universe_snapshot(request: InvestableUniverseCreate):
    return _call(product_pool_service.create_universe_snapshot, request.model_dump())


@router.get("/api/investable-universes/{snapshot_id}/products")
def search_investable_universe_products(
    snapshot_id: str,
    q: str = Query(default="", max_length=200),
    kind: Literal["etf", "fund"] | None = None,
    eligible_only: bool = True,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    return _call(
        product_pool_service.search_universe_products,
        snapshot_id,
        query=q,
        kind=kind,
        eligible_only=eligible_only,
        page=page,
        page_size=page_size,
    )


@router.get("/api/investable-universe-snapshots/{snapshot_id}")
def get_investable_universe_snapshot(snapshot_id: str):
    return _call(product_pool_service.get_universe_snapshot, snapshot_id)
