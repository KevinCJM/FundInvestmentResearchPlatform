"""Resolve and freeze an opt-in single-CMA ellipsoid without changing box semantics."""
from __future__ import annotations

from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from .cma_application import frozen_mean_covariance
from .kernels import mean_covariance_diagnostics_kernel
from . import uncertainty_kernels as numeric

CONFIDENCE = {"68": .68, "90": .90, "95": .95}


def resolve_uncertainty(request, cma, repository):
    if request.uncertainty_set == "box":
        return None, None
    numeric.require_ready()
    if request.mode != "single":
        raise ValidationError("SAA_UNCERTAINTY_SET_UNAVAILABLE", "多 CMA 椭球均值误差尚未定义。")
    matrix, key = frozen_mean_covariance(cma, repository)
    try:
        standard_error, dimension, positive_definite = mean_covariance_diagnostics_kernel(matrix)
    except ValueError as exc:
        raise ValidationError("SAA_UNCERTAINTY_COVARIANCE_INVALID", "冻结均值协方差不符合有限、对称、半正定要求。") from exc
    result = cma["model_result"]
    audit = result["model_audit"]
    method = result["method"]
    warnings = ["覆盖水平以所选模型前提为条件，不是未来收益保证；资产风险与均值估计风险分别计算。"]
    code, information = 0, 0.0
    if method == "black_litterman":
        calibration = "conditional_gaussian_mean_ellipsoid"
    elif method == "bayesian_niw":
        if not positive_definite:
            raise ValidationError("SAA_UNCERTAINTY_COVARIANCE_INVALID", "NIW 后验均值协方差须为正定。")
        posterior = audit.get("niw_posterior", {})
        information = float(posterior.get("nu", float("nan"))) - len(result["asset_ids"]) + 1.0
        code, calibration = 1, "conditional_niw_multivariate_t_covariance"
    elif method == "historical_statistics":
        information = float(audit["observations"])
        if audit.get("shrinkage") == 0 and (positive_definite or dimension == 0) and information > dimension:
            code, calibration = 2, "iid_normal_hotelling_sample_mean"
            warnings.append("Hotelling 覆盖依赖独立正态样本假设，不表示真实收益已通过正态性检验。")
        else:
            if not request.uncertainty_approximation_acknowledged:
                raise ValidationError("SAA_UNCERTAINTY_APPROXIMATION_REQUIRED",
                    "收缩或奇异样本均值协方差只支持高斯插件近似；请明确确认近似，或使用可适用的未收缩样本版本。")
            calibration = "gaussian_plugin_not_exact_confidence"
            warnings.append("当前为样本／收缩均值协方差的高斯插件近似，名义覆盖水平不是精确频率学覆盖率。")
    else:
        raise ValidationError("SAA_UNCERTAINTY_SET_UNAVAILABLE", "当前模型没有可校准的均值协方差。")
    probability = CONFIDENCE[request.uncertainty_confidence]
    try:
        kappa = numeric.uncertainty_radius_kernel(probability, int(dimension), code, information)
    except ValueError as exc:
        raise ValidationError("SAA_UNCERTAINTY_CALIBRATION", "均值不确定性维度或模型信息量不足，不能生成覆盖半径。") from exc
    if not positive_definite and dimension:
        warnings.append("使用非零方差坐标数校准高斯半径；协方差秩更低时是保守覆盖，不按数值特征值阈值删减均值方向。")
    payload = {"schema_version":"1.0", "set":"ellipsoidal", "source_cma_id":cma["id"],
               "source_cma_hash":cma["content_hash"], "covariance_field":key,
               "mean_covariance":matrix.tolist(), "mean_covariance_hash":digest_json(matrix.tolist()),
               "standard_error":standard_error.tolist(), "dimension":int(dimension),
               "dimension_basis":"nonzero_variance_coordinates", "confidence":request.uncertainty_confidence,
               "kappa":float(kappa), "calibration":calibration,
               "approximation_acknowledged":request.uncertainty_approximation_acknowledged,
               "warnings":warnings, "execution":numeric.execution_audit()}
    payload["content_hash"] = digest_json(payload)
    return matrix, payload


def validate_frozen_uncertainty(policy):
    """Structural/hash verification only. Never refit, recalibrate or rewrite a saved policy."""
    request = policy.get("selection_request", {})
    mode = request.get("uncertainty_set", "box")
    evidence = policy.get("uncertainty_model")
    if mode == "box" and evidence is None:
        return
    if mode != "ellipsoidal" or not isinstance(evidence, dict) or policy.get("mode", "single") != "single":
        raise ValidationError("SAA_UNCERTAINTY_LINEAGE", "冻结政策的均值不确定集或证据不完整，不能降级为区间模式。")
    result = policy.get("model_result", {})
    matrix = evidence.get("mean_covariance")
    key = evidence.get("covariance_field")
    if (evidence.get("schema_version") != "1.0" or evidence.get("set") != "ellipsoidal"
            or evidence.get("content_hash") != digest_json({k:v for k,v in evidence.items() if k != "content_hash"})
            or evidence.get("source_cma_id") != policy.get("cma_id")
            or evidence.get("source_cma_hash") != policy.get("cma_hash")
            or evidence.get("confidence") != request.get("uncertainty_confidence")
            or evidence.get("approximation_acknowledged") != request.get("uncertainty_approximation_acknowledged", False)
            or key not in {"mean_estimation_covariance", "posterior_mean_covariance"}
            or matrix is None or result.get(key) != matrix or evidence.get("mean_covariance_hash") != digest_json(matrix)):
        raise ValidationError("SAA_UNCERTAINTY_LINEAGE", "冻结均值协方差、来源或覆盖设置不一致，历史记录不能补算。")
