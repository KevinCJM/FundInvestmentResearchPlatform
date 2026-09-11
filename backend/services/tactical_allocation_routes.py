"""Human-oriented TAA workbench; all computations are transient until save."""
from pathlib import Path
import os

from fastapi import APIRouter, HTTPException

from backend.custom_indicators.errors import IndicatorDomainError as BackendDomainError
from custom_indicators.errors import IndicatorDomainError
from services.custom_indicator_routes import StableValidationRoute, indicator_service
from services.historical_regime_routes import regime_graph_v2_service
from backend.tactical_allocation.contracts import BaselineRequest, PreviewRequest, ScenarioRequest, SaveDecisionRequest
from backend.tactical_allocation.service import TacticalAllocationService


router = APIRouter(prefix="/api/tactical-allocation", tags=["tactical-allocation"], route_class=StableValidationRoute)
tactical_service = TacticalAllocationService(
    Path(os.getenv("TACTICAL_ALLOCATION_DATA_DIR", str(indicator_service.workspace_data_dir))),
    indicator_service.market_data_dir, regime_resolver=regime_graph_v2_service.resolve_taa_run,
    universe_dir=indicator_service.workspace_data_dir,
)


def _call(fn, *args):
    try:
        return fn(*args)
    except (IndicatorDomainError, BackendDomainError) as exc:
        raise HTTPException(exc.status_code, detail=exc.detail()) from exc
    except ValueError as exc:
        translations = {
            "Training and untouched validation each require at least 20 observations.": "训练区与留出验证区各至少需要 20 个共同收益观察值，请调整日期。",
            "Selected candidate violates training constraints.": "该候选在训练区不满足风险或换手约束，请选择可行候选。",
            "Unknown candidate selection.": "候选已失效，请重新比较。",
            "Invalid current recommendation inputs.": "当前建议输入不完整，请检查资产、偏离与权重约束。",
        }
        message = translations.get(str(exc), "计算输入不符合约束，请检查权重合计、偏离、冲击范围和样本窗口。")
        raise HTTPException(422, detail={"code": "TAA_INPUT_INVALID", "message": message}) from exc


@router.get("/catalog")
def catalog():
    return _call(tactical_service.catalog)


@router.post("/baselines", status_code=201)
def create_baseline(body: BaselineRequest):
    return _call(tactical_service.create_baseline, body.model_dump(mode="json"))


@router.get("/baselines/{identifier}")
def get_baseline(identifier: str):
    return _call(tactical_service.repository.get_baseline, identifier)


@router.post("/preview")
def preview(body: PreviewRequest):
    return _call(tactical_service.preview, body)


@router.post("/preflight")
def preflight(body: PreviewRequest):
    return _call(tactical_service.preflight, body)


@router.post("/scenarios")
def scenario(body: ScenarioRequest):
    return _call(tactical_service.scenario, body)


@router.post("/decisions", status_code=201)
def save_decision(body: SaveDecisionRequest):
    return _call(tactical_service.save_decision, body)


@router.get("/decisions/{identifier}")
def get_decision(identifier: str):
    return _call(tactical_service.repository.get_decision, identifier)


@router.post("/decisions/{identifier}/product-allocation")
def product_allocation(identifier: str):
    return _call(tactical_service.product_allocation, identifier)
