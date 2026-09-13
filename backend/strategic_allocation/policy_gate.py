"""Shared preview/application gate for an adopted strategic research policy."""
from datetime import date

import numpy as np

from backend.custom_indicators.errors import ValidationError
from . import kernels


def check_policy(baseline: dict, weights: dict, tracking_error_limit: float, as_of: str) -> dict | None:
    policy = baseline.get("policy")
    if policy is None:
        return None  # Historical baselines do not acquire fabricated policy evidence.
    kernels.require_ready()
    mandate = policy["mandate"]
    assumptions = policy["assumptions"]
    names = [asset["id"] for asset in baseline["assets"]]
    if set(weights) != set(names) or [a["id"] for a in assumptions["assets"]] != names:
        raise ValidationError("SAA_POLICY_AXIS", "目标权重与冻结政策的资产轴不一致。")
    values = np.asarray([weights[name] for name in names], dtype=np.float64)
    means = np.asarray([a["annual_return"] for a in assumptions["assets"]], dtype=np.float64)
    uncertainty = np.asarray([a["mean_uncertainty"] for a in assumptions["assets"]], dtype=np.float64)
    covariance = np.asarray(policy["covariance"], dtype=np.float64)
    metrics, _ = kernels.portfolio_moments_kernel(values, means, covariance,
                                                uncertainty, float(mandate["risk_aversion"]), 1.0)
    policy_weights = np.asarray([asset["base_weight"] for asset in baseline["assets"]], dtype=np.float64)
    expected_tracking_error = kernels.expected_active_risk_kernel(values, policy_weights, covariance)
    violations = []
    if not np.isfinite(tracking_error_limit) or tracking_error_limit > mandate["max_tracking_error"] + 1e-10:
        violations.append("战术主动风险上限不得超过投资目标的政策预算。")
    if np.isfinite(tracking_error_limit) and expected_tracking_error > tracking_error_limit + 1e-10:
        violations.append("当前目标在冻结 CMA 下的预期主动风险超过本次战术主动风险上限。")
    if expected_tracking_error > mandate["max_tracking_error"] + 1e-10:
        violations.append("当前目标在冻结 CMA 下的预期主动风险超过投资目标的政策预算。")
    if metrics[1] > mandate["max_volatility"] + 1e-10:
        violations.append("当前目标在冻结 CMA 下的预期波动超过投资目标上限。")
    if mandate.get("objective_kind", "absolute_return") == "absolute_return" and metrics[0] < mandate["target_return"] - 1e-10:
        violations.append("当前目标在冻结CMA下的预期收益低于投资授权下限。")
    benchmark_check = None
    if mandate.get("benchmark"):
        benchmark = mandate["benchmark"]
        if set(benchmark["weights"]) != set(names):
            raise ValidationError("SAA_POLICY_AXIS", "冻结基准的资产轴不一致。")
        benchmark_weights = np.asarray([benchmark["weights"][name] for name in names], dtype=np.float64)
        benchmark_te = kernels.expected_active_risk_kernel(values, benchmark_weights, covariance)
        excess = kernels.expected_excess_return_kernel(values, benchmark_weights, means)
        benchmark_check = {"name": benchmark["name"], "tracking_error": float(benchmark_te),
                           "max_tracking_error": benchmark["max_tracking_error"],
                           "expected_excess_return": float(excess), "target_excess_return": benchmark["target_excess_return"]}
        if excess < benchmark["target_excess_return"] - 1e-10:
            violations.append("当前目标在冻结CMA下相对授权基准的预期超额低于目标。")
        if benchmark_te > benchmark["max_tracking_error"] + 1e-10:
            violations.append("当前目标相对投资授权基准的主动风险超过上限。")
    expires = policy["expires_on"]
    if as_of < baseline["as_of"] or as_of >= expires:
        violations.append("政策尚未适用于本研究日或已到复核日期，请重新确认长期政策。")
    return {"within_limits": not violations, "violations": violations,
            "current_application_eligible": str(date.today()) < expires and not violations,
            "benchmark_check": benchmark_check,
            "goal_diagnostic_scope": "strategic_plan_only_not_tactical_probability_guarantee" if mandate.get("funding_plan") else None,
            "expected_volatility": float(metrics[1]), "max_volatility": mandate["max_volatility"],
            "expected_tracking_error": float(expected_tracking_error),
            "requested_tracking_error_limit": float(tracking_error_limit) if np.isfinite(tracking_error_limit) else None,
            "max_tracking_error": mandate["max_tracking_error"], "expires_on": expires,
            "cma_id": policy["cma_id"], "mandate_id": policy["mandate_id"], "execution": kernels.execution_audit()}


def require_policy_application(baseline: dict, weights: dict, tracking_error_limit: float, as_of: str) -> None:
    check = check_policy(baseline, weights, tracking_error_limit, as_of)
    if check and not check["within_limits"]:
        raise ValidationError("SAA_POLICY_LIMIT", "；".join(check["violations"]))
    if check and str(date.today()) >= check["expires_on"]:
        raise ValidationError("SAA_POLICY_EXPIRED", "历史研究可以保留，但政策已到复核日，不能直接用于当前产品应用。")
