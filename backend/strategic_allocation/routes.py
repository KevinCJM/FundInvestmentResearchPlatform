"""Dependency-injected API; tests never initialize the production workspace."""
from fastapi import APIRouter, HTTPException

from backend.custom_indicators.errors import IndicatorDomainError
from .contracts import (CmaRequest, MandateRequest, PolicyRequest, PublishCmaRequest,
                        PublishPolicyRequest, RiskReferenceRequest)
from .service import StrategicAllocationService

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
    except ValueError as exc:
        raise HTTPException(422, detail={"code": "SAA_INPUT_INVALID",
            "message": MESSAGES.get(str(exc), "输入不符合资产配置契约，请核对数据、矩阵和约束。")}) from exc


def build_router(service: StrategicAllocationService) -> APIRouter:
    router = APIRouter(prefix="/api/strategic-allocation", tags=["strategic-allocation"])

    @router.get("/catalog")
    def catalog():
        return _call(service.catalog)

    @router.post("/mandates", status_code=201)
    def save_mandate(body: MandateRequest):
        return _call(service.save_mandate, body)

    @router.get("/mandates/{identifier}")
    def get_mandate(identifier: str):
        return _call(service.get_mandate, identifier)

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
