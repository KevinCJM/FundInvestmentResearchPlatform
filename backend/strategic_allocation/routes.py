"""Dependency-injected API; tests never initialize the production workspace."""
from fastapi import APIRouter, HTTPException
from numpy.linalg import LinAlgError

from backend.custom_indicators.errors import IndicatorDomainError
from .contracts import (CmaRequest, PolicyRequest, PublishCmaRequest,
                        PublishPolicyRequest, RiskReferenceRequest,
                        MandateStudyRequest, ConfirmMandateRequest)
from .service import StrategicAllocationService
from .universe_contracts import UniverseRequest, ConfirmUniverseRequest, ImplementationMapRequest, ConfirmImplementationMapRequest

MESSAGES = {
    "CMA_MATRIX_SHAPE": "相关矩阵的大小与资产数量不匹配。",
    "CMA_VOLATILITY": "年化波动率须为正的有限数值。",
    "CMA_CORRELATION_DIAGONAL": "每个资产与自身的相关系数须为 1。",
    "CMA_CORRELATION_RANGE": "相关系数须位于 -1 至 1。",
    "CMA_CORRELATION_SYMMETRY": "相关矩阵须对称，请核对上下三角。",
    "CMA_CORRELATION_PSD": "相关矩阵不是半正定矩阵，这些相关性无法同时成立；系统不会偷偷修正。",
    "RISK_SAMPLE_INVALID": "历史风险参考至少需要 20 条完整、有效的共同收益观察值。",
    "RISK_ZERO_VARIANCE": "样本存在零波动资产，无法估计有效相关矩阵。",
    "POLICY_INPUT_INVALID": "政策权重、收益或风险参数不符合数值契约。",
    "POLICY_SEARCH_SHAPE": "资产或约束维度不一致，或候选数量超出预算。",
}


def _call(function, *args):
    try:
        return function(*args)
    except IndicatorDomainError as exc:
        raise HTTPException(exc.status_code, detail=exc.detail()) from exc
    except RuntimeError as exc:
        if "预热" not in str(exc) and "CMA_MODEL_NOT_READY" not in str(exc):
            raise
        raise HTTPException(503, detail={"code": "SAA_NOT_READY",
            "message": "本进程尚未完成配置计算预热，请稍后重试。"}) from exc
    except (ValueError, LinAlgError) as exc:
        raise HTTPException(422, detail={"code": "SAA_INPUT_INVALID",
            "message": MESSAGES.get(str(exc), "输入不符合资产配置契约，请核对数据、矩阵和约束。")}) from exc


def build_router(service: StrategicAllocationService) -> APIRouter:
    router = APIRouter(prefix="/api/strategic-allocation", tags=["strategic-allocation"])

    @router.get("/catalog")
    def catalog():
        return _call(service.catalog)

    @router.post("/universes/preview")
    def preview_universe(body: UniverseRequest):
        return _call(service.scopes.preview_universe, body)

    @router.post("/universes/confirm", status_code=201)
    def confirm_universe(body: ConfirmUniverseRequest):
        return _call(service.scopes.confirm_universe, body)

    @router.get("/universes/{identifier}")
    def get_universe(identifier: str):
        return _call(service.scopes.get_universe, identifier)

    @router.post("/implementation-maps/preview")
    def preview_mapping(body: ImplementationMapRequest):
        return _call(service.scopes.preview_mapping, body)

    @router.post("/implementation-maps/confirm", status_code=201)
    def confirm_mapping(body: ConfirmImplementationMapRequest):
        return _call(service.scopes.confirm_mapping, body)

    @router.get("/implementation-maps/{identifier}")
    def get_mapping(identifier: str):
        return _call(service.scopes.get_mapping, identifier)

    # Research-only input diagnostics and version storage; no order execution.
    @router.post("/mandates/preview")
    def preview_mandate(body: MandateStudyRequest):
        return _call(service.preview_mandate, body)

    # Deterministic cash-flow echo while the objective is still being typed; no storage.
    @router.post("/mandates/funding")
    def mandate_funding(body: MandateStudyRequest):
        return _call(service.mandate_funding, body)

    @router.post("/mandates/confirm", status_code=201)
    def confirm_mandate(body: ConfirmMandateRequest):
        return _call(service.confirm_mandate, body)

    @router.get("/mandates/{identifier}")
    def get_mandate(identifier: str):
        return _call(service.get_mandate, identifier)

    @router.delete("/mandates/{identifier}")
    def retire_mandate(identifier: str):
        return _call(service.retire_mandate, identifier)

    @router.post("/risk-reference")
    def risk_reference(body: RiskReferenceRequest):
        return _call(service.risk_reference, body)

    @router.post("/cma/preview")
    def preview_cma(body: CmaRequest):
        return _call(service.preview_cma, body)

    @router.post("/cma", status_code=201)
    def publish_cma(body: PublishCmaRequest):
        return _call(service.publish_cma, body)

    @router.get("/cma/{identifier}")
    def get_cma(identifier: str):
        return _call(service.get_cma, identifier)

    @router.post("/policy/preview")
    def preview_policy(body: PolicyRequest):
        return _call(service.preview_policy, body)

    @router.post("/policies", status_code=201)
    def publish_policy(body: PublishPolicyRequest):
        return _call(service.publish_policy, body)

    return router
