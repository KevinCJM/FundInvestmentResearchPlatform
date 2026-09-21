"""Apply explicit CMA models once; consume frozen results without recomputation."""
from __future__ import annotations

import copy
import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from .cma_models import evaluate_cma_model


def apply_model(request, asset_ids: list[str], *, evidence: dict | None = None) -> tuple[dict, dict]:
    """Input/output boundary only. Numerical transformations live in model kernels."""
    try:
        result = evaluate_cma_model(request.model, asset_ids=asset_ids,
                                    as_of=request.as_of, currency=request.currency, evidence=evidence)
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
        if result.mean_uncertainty is not None:
            asset["mean_uncertainty"] = float(result.mean_uncertainty[i])
    effective["correlation"] = audit["effective_correlation"]
    payload = result.to_payload()
    payload["content_hash"] = digest_json(payload)
    arrays = {"covariance": result.effective_covariance,
              "effective_returns": result.effective_returns,
              "mean_uncertainty": result.mean_uncertainty if result.mean_uncertainty is not None
                  else np.asarray([a.mean_uncertainty for a in request.assets], dtype=np.float64)}
    if result.posterior_mean_covariance is not None:
        arrays["posterior_mean_covariance"] = result.posterior_mean_covariance
    if result.mean_estimation_covariance is not None:
        arrays["mean_estimation_covariance"] = result.mean_estimation_covariance
    return {"effective_assumptions": effective, "effective_returns": payload["effective_returns"],
            "effective_covariance": payload["effective_covariance"], "model_result": payload}, arrays


def frozen_assumptions(cma: dict) -> dict:
    """Old artifacts retain their exact manual interpretation, with no writes."""
    if "multi_cma" in cma:
        from .multi_cma import validate_frozen
        assumptions = validate_frozen(cma["multi_cma"])
        if (cma.get("definition") != assumptions or cma.get("covariance") != cma["multi_cma"]["effective_covariance"]
                or cma.get("content_hash") != cma["multi_cma"]["content_hash"]):
            raise ValidationError("SAA_MULTI_CMA_LINEAGE", "派生参数与冻结融合结果不一致。")
        return assumptions
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
    expected_uncertainty = (result.get("mean_uncertainty", [a["mean_uncertainty"] for a in raw["assets"]])
                            if raw.get("schema_version") == "2.0" else [a["mean_uncertainty"] for a in raw["assets"]])
    if (result.get("asset_ids") != [a["id"] for a in raw["assets"]]
            or result.get("definition") != raw["model"]
            or [a.get("annual_return") for a in effective["assets"]] != result.get("effective_returns")
            or [a.get("annual_volatility") for a in effective["assets"]] != audit.get("effective_volatility")
            or [a.get("mean_uncertainty") for a in effective["assets"]] != expected_uncertainty
            or effective.get("correlation") != audit.get("effective_correlation")
            or cma.get("covariance") != result.get("effective_covariance")
            or cma.get("effective_returns") != result.get("effective_returns")
            or cma.get("effective_covariance") != result.get("effective_covariance")):
        raise ValidationError("SAA_CMA_MODEL_LINEAGE", "原始模型、有效假设与冻结风险不一致，已停止使用。")
    return effective


def frozen_policy_assumptions(policy: dict) -> dict:
    from .uncertainty import validate_frozen_uncertainty
    validate_frozen_uncertainty(policy)
    mode = policy.get("mode", "single")
    if mode not in ("single", "parameter_average", "compatible_all_models"):
        raise ValidationError("SAA_MULTI_CMA_MODE", "未知的政策模式，不能按单模型继续计算。")
    if mode != "single" or "multi_cma" in policy:
        from .multi_cma import validate_frozen
        if (mode == "single" or policy.get("schema_version") != "2.0" or policy.get("cma_id") is not None
                or mode != (policy.get("multi_cma") or {}).get("mode")):
            raise ValidationError("SAA_MULTI_CMA_MODE", "融合政策的模式或来源身份不完整。")
        assumptions = validate_frozen(policy.get("multi_cma"))
        if mode == "compatible_all_models" and (policy.get("compatibility", {}).get("gate") != "all_frozen_models"
                or policy.get("selection_request", {}).get("mode") != mode
                or policy.get("selection", {}).get("all_models_pass") is not True):
            raise ValidationError("SAA_MULTI_CMA_LINEAGE", "共同约束政策缺少全部模型通过的确认依据。")
        if (policy.get("assumptions") != assumptions or policy.get("covariance") != policy["multi_cma"]["effective_covariance"]
                or policy.get("cma_hash") != policy["multi_cma"]["content_hash"]):
            raise ValidationError("SAA_MULTI_CMA_LINEAGE", "政策使用的参数与冻结融合结果不一致。")
        return assumptions
    if policy.get("schema_version", "1.0") != "1.0" or not policy.get("cma_id"):
        raise ValidationError("SAA_MULTI_CMA_MODE", "政策版本缺少支持的完整风险来源。")
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
    if "multi_cma" in cma:
        multi = cma["multi_cma"]
        values = tuple(np.asarray(multi[key], dtype=np.float64) for key in
                       ("effective_returns", "effective_covariance", "effective_mean_uncertainty"))
        for value in values:
            value.flags.writeable = False
        return values
    if definition.get("model") is not None:
        arrays = repository.arrays(cma["id"], ("effective_returns", "covariance", "mean_uncertainty"))
        return arrays["effective_returns"], arrays["covariance"], arrays["mean_uncertainty"]
    return (np.asarray([a["annual_return"] for a in definition["assets"]], dtype=np.float64),
            repository.arrays(cma["id"], ("covariance",))["covariance"],
            np.asarray([a["mean_uncertainty"] for a in definition["assets"]], dtype=np.float64))


def frozen_mean_covariance(cma: dict, repository):
    """Read the model's mean covariance, never substitute return covariance or refit history."""
    frozen_assumptions(cma)
    model = cma.get("model_result", {})
    key = ("mean_estimation_covariance" if model.get("method") == "historical_statistics"
           else "posterior_mean_covariance")
    expected = model.get(key)
    if expected is None or key not in cma.get("arrays", {}):
        raise ValidationError("SAA_UNCERTAINTY_SET_UNAVAILABLE",
            "所选版本没有冻结的均值协方差数组，不能使用椭球稳健化；请创建新版本或显式选择区间模式。")
    matrix = repository.arrays(cma["id"], (key,))[key]
    # JSON is already resident metadata; comparison does not copy the mapped matrix.
    if matrix.tolist() != expected:
        raise ValidationError("SAA_UNCERTAINTY_LINEAGE", "均值协方差数组与冻结模型结果不一致。")
    return matrix, key


def frozen_model_lineage(cma: dict) -> dict:
    if "multi_cma" in cma:
        frozen_assumptions(cma)
        return {"multi_cma": copy.deepcopy(cma["multi_cma"])}
    if cma["definition"].get("model") is None:
        return {}
    frozen_assumptions(cma)
    return copy.deepcopy({"raw_assumptions": cma["definition"],
                          "model_result": cma["model_result"],
                          "effective_returns": cma["effective_returns"],
                          "effective_covariance": cma["effective_covariance"]})
