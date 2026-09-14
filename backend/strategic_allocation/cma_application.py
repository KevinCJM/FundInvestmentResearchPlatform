"""Apply explicit CMA models once; consume frozen results without recomputation."""
from __future__ import annotations

import copy
import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from .cma_models import evaluate_cma_model


def apply_model(request, asset_ids: list[str]) -> tuple[dict, dict]:
    """Input/output boundary only. Numerical transformations live in model kernels."""
    try:
        result = evaluate_cma_model(request.model, asset_ids=asset_ids,
                                    as_of=request.as_of, currency=request.currency)
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValidationError("SAA_CMA_MODEL_INVALID",
            "模型输入无法形成有效长期假设，请核对资产顺序、日期、币种、观点及风险矩阵；不会自动修补矩阵。") from exc
    if result.model_audit["return_basis"] != request.return_basis:
        raise ValidationError("SAA_CMA_MODEL_BASIS", "模型与长期假设的收益口径必须完全一致。")
    effective = request.model_dump(mode="json")
    audit = result.model_audit
    for i, asset in enumerate(effective["assets"]):
        asset["annual_return"] = float(result.effective_returns[i])
        asset["annual_volatility"] = audit["effective_volatility"][i]
    effective["correlation"] = audit["effective_correlation"]
    payload = result.to_payload()
    payload["content_hash"] = digest_json(payload)
    arrays = {"covariance": result.effective_covariance,
              "effective_returns": result.effective_returns,
              "mean_uncertainty": np.asarray([a.mean_uncertainty for a in request.assets], dtype=np.float64)}
    if result.posterior_mean_covariance is not None:
        arrays["posterior_mean_covariance"] = result.posterior_mean_covariance
    return {"effective_assumptions": effective, "effective_returns": payload["effective_returns"],
            "effective_covariance": payload["effective_covariance"], "model_result": payload}, arrays


def frozen_assumptions(cma: dict) -> dict:
    """Old artifacts retain their exact manual interpretation, with no writes."""
    raw = cma["definition"]
    if raw.get("model") is None:
        return raw
    effective = cma.get("effective_assumptions")
    result = cma.get("model_result")
    if not effective or not result:
        raise ValidationError("SAA_CMA_MODEL_LINEAGE", "模型版本缺少冻结的有效假设，请建立新版本；历史记录不能补算。")
    hashed = {key: value for key, value in result.items() if key != "content_hash"}
    if result.get("content_hash") != digest_json(hashed):
        raise ValidationError("SAA_CMA_MODEL_LINEAGE", "冻结模型结果校验不一致，请重新读取完整版本。")
    audit = result.get("model_audit", {})
    if (result.get("asset_ids") != [a["id"] for a in raw["assets"]]
            or result.get("definition") != raw["model"]
            or [a.get("annual_return") for a in effective["assets"]] != result.get("effective_returns")
            or [a.get("annual_volatility") for a in effective["assets"]] != audit.get("effective_volatility")
            or [a.get("mean_uncertainty") for a in effective["assets"]] != [a["mean_uncertainty"] for a in raw["assets"]]
            or effective.get("correlation") != audit.get("effective_correlation")
            or cma.get("covariance") != result.get("effective_covariance")
            or cma.get("effective_returns") != result.get("effective_returns")
            or cma.get("effective_covariance") != result.get("effective_covariance")):
        raise ValidationError("SAA_CMA_MODEL_LINEAGE", "原始模型、有效假设与冻结风险不一致，已停止使用。")
    return effective


def frozen_policy_assumptions(policy: dict) -> dict:
    assumptions = policy["assumptions"]
    if assumptions.get("model") is None:
        if policy.get("raw_assumptions", {}).get("model") is not None:
            raise ValidationError("SAA_CMA_MODEL_LINEAGE", "模型政策不能丢失有效假设标识。")
        return assumptions
    if not policy.get("raw_assumptions"):
        raise ValidationError("SAA_CMA_MODEL_LINEAGE", "模型政策缺少原始模型血缘；不能补算历史。")
    return frozen_assumptions({**policy, "definition": policy["raw_assumptions"],
                               "effective_assumptions": assumptions})


def frozen_numeric_inputs(cma: dict, repository):
    """One checked mmap boundary; consumers keep the mapped owners alive."""
    definition = frozen_assumptions(cma)
    if definition.get("model") is not None:
        arrays = repository.arrays(cma["id"], ("effective_returns", "covariance", "mean_uncertainty"))
        return arrays["effective_returns"], arrays["covariance"], arrays["mean_uncertainty"]
    return (np.asarray([a["annual_return"] for a in definition["assets"]], dtype=np.float64),
            repository.arrays(cma["id"], ("covariance",))["covariance"],
            np.asarray([a["mean_uncertainty"] for a in definition["assets"]], dtype=np.float64))


def frozen_model_lineage(cma: dict) -> dict:
    if cma["definition"].get("model") is None:
        return {}
    frozen_assumptions(cma)
    return copy.deepcopy({"raw_assumptions": cma["definition"],
                          "model_result": cma["model_result"],
                          "effective_returns": cma["effective_returns"],
                          "effective_covariance": cma["effective_covariance"]})
