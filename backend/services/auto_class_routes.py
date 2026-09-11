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
    ALGORITHMS,
    BLOCK_MODES,
    FEATURE_SETS,
    LINKAGE_METHODS,
    WEIGHT_MODES,
    AutoClassError,
    AutoClassRequestSpec,
    auto_classification_meta,
    run_auto_classification,
)
from fund_taxonomy import TAXONOMY_LEVELS
from pit.catalog import RUN_MODES
from pit.context import PitContextError, resolve_request_context
from pit.guard import assert_no_universe_lookahead, universe_lineage


DATA_DIR = (Path(__file__).resolve().parents[2] / "data").resolve()

router = APIRouter(prefix="/api/asset-classes/auto", tags=["auto-asset-class"])


# Bound to the engine registries rather than retyped: a new algorithm or feature
# set must not be able to reach production as a 422 the UI cannot even render.
_ALGORITHM = Literal[*tuple(ALGORITHMS)]
_FEATURES = Literal[*tuple(FEATURE_SETS)]
_LINKAGE = Literal[*tuple(LINKAGE_METHODS)]
_WEIGHT_MODE = Literal[*tuple(WEIGHT_MODES)]
_TAXONOMY_LEVEL = Literal[*TAXONOMY_LEVELS]
_BLOCK_MODE = Literal[*tuple(BLOCK_MODES)]
_RUN_MODE = Literal[*tuple(RUN_MODES)]


class PoolProduct(BaseModel):
    code: str
    name: str = ""
    kind: Literal["etf", "fund"]


class AutoClassPreviewRequest(BaseModel):
    universe_snapshot_id: str = Field(min_length=1, max_length=120)
    products: list[PoolProduct] = Field(default_factory=list)
    startDate: str = "2020-01-01"
    algorithm: _ALGORITHM = "hierarchical"
    features: _FEATURES = "correlation"
    linkage: _LINKAGE = "average"
    k: Optional[int] = None
    sizeMin: int = 2
    sizeMax: int = 8
    unassignedPolicy: Literal["park", "force"] = "park"
    weightMode: _WEIGHT_MODE = "inv_vol"
    # Contract-taxonomy level used to name the classes.
    taxonomyLevel: _TAXONOMY_LEVEL = "asset_class"
    # Taxonomy level the statistical clustering may not cross.
    blockBy: _BLOCK_MODE = "none"
    seed: int = 20260101
    # Research context: which day the run pretends to stand on, how strictly,
    # and against which data vintage.
    # All three default to None on purpose: an unstated field must be
    # distinguishable from an explicit choice, otherwise the system-level PIT
    # setting could never apply.
    asOf: Optional[str] = None
    runMode: Optional[_RUN_MODE] = None
    dataReleaseId: Optional[str] = None


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

    try:
        context = resolve_request_context(DATA_DIR, req.asOf, req.runMode, req.dataReleaseId)
        # The candidate set is the one look-ahead no per-formula check can see: a
        # pool screened on 2026 numbers and replayed over 2018 is perfectly causal
        # in every individual computation and wrong as a whole. The guard already
        # lived in pit.guard and the snapshot's own research_date is resolved just
        # above -- this is the wiring that was missing.
        universe_findings = assert_no_universe_lookahead(
            context,
            established_at=universe_reference.get("research_date"),
            label=f"可投资域「{universe_reference.get('name') or universe_reference.get('id')}」",
        )
    except PitContextError as exc:
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
        # A request that states nothing inherits the system-level PIT setting,
        # so the口径 is right even when a page forgets to send it.
        as_of=context.as_of,
        run_mode=context.run_mode,
        data_release_id=context.data_release_id,
        max_weights=max_weights,
    )
    try:
        result = run_auto_classification(DATA_DIR, spec)
        result["universe_snapshot"] = universe_reference
        result["pit"]["universe"] = universe_lineage(
            universe_findings,
            established_at=universe_reference.get("research_date"),
            source="investable_universe_snapshot",
        )
        return result
    except AutoClassError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except FileNotFoundError as exc:
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})


__all__ = ["router", "auto_class_meta", "auto_class_preview"]
