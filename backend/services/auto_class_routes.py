"""Routes for the pre-investment SAA node 「自动构建大类」."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from product_pools.constants import UNIVERSE_SNAPSHOT_STORE
from product_pools.errors import ProductPoolDomainError
from product_pools.membership import InvestableUniverseMembership
from product_pools.repository import InvestableUniverseRepository
from auto_asset_class import (
    AutoClassError,
    AutoClassRequestSpec,
    auto_classification_meta,
    run_auto_classification,
)


DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

router = APIRouter(prefix="/api/asset-classes/auto", tags=["auto-asset-class"])


class PoolProduct(BaseModel):
    code: str
    name: str = ""
    kind: Literal["etf", "fund"]


class AutoClassPreviewRequest(BaseModel):
    universe_snapshot_id: str = Field(min_length=1, max_length=120)
    products: list[PoolProduct] = Field(default_factory=list)
    startDate: str = "2020-01-01"
    algorithm: Literal["rule", "hierarchical", "kmedoids", "kmeans"] = "hierarchical"
    features: Literal["correlation", "metrics", "pca", "blend"] = "correlation"
    linkage: Literal["average", "complete", "ward"] = "average"
    k: Optional[int] = None
    sizeMin: int = 2
    sizeMax: int = 8
    unassignedPolicy: Literal["park", "force"] = "park"
    weightMode: Literal["equal", "inv_vol", "inv_var", "affinity"] = "inv_vol"
    # Contract-taxonomy level used to name the classes.
    taxonomyLevel: Literal["asset_class", "category", "detail"] = "asset_class"
    # Taxonomy level the statistical clustering may not cross.
    blockBy: Literal["none", "asset_class", "category", "detail"] = "none"
    seed: int = 20260101


@router.get("/meta")
def auto_class_meta() -> dict[str, Any]:
    return auto_classification_meta()


def _validate_universe_products(
    req: AutoClassPreviewRequest,
) -> tuple[dict[str, Any], dict[str, float]]:
    validator = InvestableUniverseMembership(
        InvestableUniverseRepository(DATA_DIR / UNIVERSE_SNAPSHOT_STORE)
    )
    result = validator.validate(
        req.universe_snapshot_id,
        [
            {"kind": item.kind, "product_id": item.code}
            for item in req.products
        ],
    )
    # The pool's own per-product weight restrictions travel with the resolved
    # members; without them the classifier would happily hand a limited product
    # a weight the pool forbids.
    limits: dict[str, float] = {}
    for member in result.members:
        product_id = str(member.get("product_id") or "").strip()
        raw = member.get("max_weight")
        if not product_id or raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if 0.0 < value <= 1.0:
            limits[product_id] = min(limits.get(product_id, value), value)
    return result.reference, limits


@router.post("/preview")
def auto_class_preview(req: AutoClassPreviewRequest):
    """Run one classification draft inside a locked investable universe."""

    try:
        universe_reference, max_weights = _validate_universe_products(req)
    except ProductPoolDomainError as exc:
        diagnostics = exc.diagnostics or []
        detail = exc.message
        if diagnostics:
            items = [
                f"{item.get('product_id')}: {item.get('reason')}"
                for item in diagnostics
                if item.get("product_id")
            ]
            if items:
                detail = f"{detail} {'；'.join(items)}"
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    spec = AutoClassRequestSpec(
        codes=[item.code for item in req.products],
        names=[item.name for item in req.products],
        start_date=req.startDate,
        algorithm=req.algorithm,
        features=req.features,
        linkage=req.linkage,
        k=req.k,
        size_min=req.sizeMin,
        size_max=req.sizeMax,
        unassigned_policy=req.unassignedPolicy,
        weight_mode=req.weightMode,
        taxonomy_level=req.taxonomyLevel,
        block_by=req.blockBy,
        seed=req.seed,
        max_weights=max_weights,
    )
    try:
        result = run_auto_classification(DATA_DIR, spec)
        result["universe_snapshot"] = universe_reference
        return result
    except AutoClassError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


__all__ = ["router", "auto_class_meta", "auto_class_preview"]
