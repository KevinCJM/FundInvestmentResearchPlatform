"""Orchestration for deterministic funding and CMA-based diagnostics, never writes."""
from __future__ import annotations

import numpy as np
from . import goal_kernels as goals
from backend.custom_indicators.errors import ValidationError
from .mandate_inputs import effective_funding_plan, cash_success_required

FUNDING_METRICS = (
    "success_probability", "probability_lower", "probability_upper", "payment_failure_probability",
    "terminal_p05", "terminal_median", "terminal_p95", "expected_terminal_shortfall",
    "expected_unpaid_payments", "market_drawdown_p95", "drawdown_alert_probability",
    "required_initial_capital", "additional_initial_capital", "success_with_10pct_more_capital",
    "success_with_10pct_lower_target", "gate_required_initial_capital", "gate_additional_initial_capital",
    "gate_success_probability", "gate_probability_lower", "gate_probability_upper",
)
LIMITATIONS = [
    "月度独立对数正态组合代理，仅匹配CMA年度简单收益均值和波动；不是厚尾、状态转换或各资产再平衡仿真。",
    "模拟区间只表示采样误差，不包括CMA和模型误差；历史研究日不等于已证明当时可得或已部署。",
    "流动性预算基于现金流和人工liquid标签，不认证逐产品赎回、结算或市场冲击。",
    "费用为年有效财富扣减，现金流为月末先投入后支付；未支付金额未折现。",
    "比较的是既有SAA的四类代表组合，不是成功概率全局优化；未通过不证明所有可行组合都无解。",
]


def funding_inputs(definition: dict):
    plan = effective_funding_plan(definition)
    if plan is None:
        return None
    goals.require_ready()
    months = definition["horizon_years"] * 12
    periods = np.asarray([[row["first_month"], row["last_month"], row["every_months"]]
                          for row in plan["flows"]], dtype=np.int64).reshape(-1, 3)
    amounts = np.asarray([row["amount"] if row["kind"] == "contribution" else -row["amount"]
                          for row in plan["flows"]], dtype=np.float64)
    inflows, outflows = goals.funding_schedule_kernel(periods, amounts, months, plan["inflation"],
                                                     int(plan["amount_basis"] == "real"))
    summary, status = goals.funding_summary_kernel(
        plan["total_capital"], plan["outside_reserve"], plan["terminal_target"], months,
        plan["inflation"], plan["annual_fee"], int(plan.get("target_amount_basis", plan["amount_basis"]) == "real"),
        inflows, outflows, plan["liquidity_months"], plan["contribution_stress_ratio"])
    names = ("investable_capital", "nominal_terminal_target", "total_contributions", "total_withdrawals",
             "required_liquid_capital", "required_liquid_weight", "required_effective_return",
             "liquidity_payment_buffer", "liquidity_payment_buffer_ratio", "liquidity_shortfall_capital")
    result = dict(zip(names, [float(v) if np.isfinite(v) else None for v in summary], strict=True))
    # The deterministic return the cash flows demand stays available even without a
    # funding success condition; it is a constraint, never a success probability.
    result.update({"cashflow_required_return": result["required_effective_return"],
                   "cashflow_required_return_status": ("solved", "at_lower_bound", "above_search_bound")[status],
                   "root_status": ("solved", "at_lower_bound", "above_search_bound")[status],
                   "return_search_bounds": [-0.99, 5.0], "currency": definition["currency"],
                   "fee_included": True, "required_return_basis": "annual_effective_gross_of_model_fee",
                   "cashflow_timing": "equal_model_month_end_contribution_then_payment",
                   "liquidity_months": plan["liquidity_months"],
                   "monthly_cashflows": [{"month": i + 1, "contribution": float(inflows[i]),
                                           "withdrawal": float(outflows[i])} for i in range(months)]})
    if not cash_success_required(definition):
        result.update(required_effective_return=None, root_status="not_applicable",
                      nominal_terminal_target=None)
    inflows.flags.writeable = False
    outflows.flags.writeable = False
    return result, inflows, outflows


def _funding_metrics(values: np.ndarray) -> dict:
    metrics = dict(zip(FUNDING_METRICS,
        [float(v) if np.isfinite(v) else None for v in values], strict=True))
    metrics["capital_gate_status"] = "solved" if metrics["gate_required_initial_capital"] is not None else "insufficient_paths"
    return metrics


def require_goal_checks(definition: dict, candidates: list[dict]) -> None:
    """Validate diagnostic evidence; missing data is never a passing goal."""
    plan = effective_funding_plan(definition)
    if not cash_success_required(definition):
        return
    threshold = plan["required_probability"]
    for candidate in candidates:
        check = candidate.get("goal_check")
        central = check.get("central") if isinstance(check, dict) else None
        if not isinstance(central, dict):
            raise ValidationError("MANDATE_DIAGNOSTIC_INCOMPLETE", "资金目标缺少完整的概率诊断，不能判定通过或采纳。")
        probability = [central.get(name) for name in ("probability_lower", "success_probability", "probability_upper")]
        if (any(type(v) not in (int, float) or not np.isfinite(v) or not 0 <= v <= 1 for v in probability)
                or not probability[0] <= probability[1] <= probability[2]
                or type(check.get("within_limits")) is not bool
                or check.get("threshold") != threshold
                or check.get("gate_basis") != "wilson_95pct_lower_bound"
                or check["within_limits"] != (probability[0] >= threshold)):
            raise ValidationError("MANDATE_DIAGNOSTIC_INCOMPLETE", "资金目标的概率、区间或采纳门槛不一致，须重新诊断。")


def diagnose_funding(definition: dict, candidates: list[dict], *, paths: int, seed: int) -> dict:
    prepared = funding_inputs(definition)
    if prepared is None or not cash_success_required(definition):
        return {"kind": definition.get("objective_kind", "absolute_return"),
                "funding": prepared[0] if prepared else None, "candidates": candidates, "limitations": []}
    summary, inflows, outflows = prepared
    plan = effective_funding_plan(definition)
    draws, _ = goals.seeded_factor_draws_kernel(inflows.size, paths, 1, seed, 0, 5.)
    draws.flags.writeable = False
    for candidate in candidates:
        metrics = candidate["metrics"]
        values, fan = goals.funding_paths_kernel(
            draws, metrics["expected_return"], metrics["volatility"], summary["investable_capital"],
            inflows, outflows, summary["nominal_terminal_target"], plan["annual_fee"],
            plan["required_probability"], plan["drawdown_alert"], 1.0)
        central = _funding_metrics(values)
        if plan.get("drawdown_alert_enabled") is False:
            central["drawdown_alert_probability"] = None
        central["annual_fan"] = [{"year": i, "p05": float(row[0]), "median": float(row[1]), "p95": float(row[2])}
                                 for i, row in enumerate(fan)]
        conservative = None
        if metrics["conservative_return"] > -1:
            stressed, _ = goals.funding_paths_kernel(
                draws, metrics["conservative_return"], metrics["volatility"], summary["investable_capital"],
                inflows, outflows, summary["nominal_terminal_target"], plan["annual_fee"],
                plan["required_probability"], plan["drawdown_alert"], plan["contribution_stress_ratio"])
            conservative = _funding_metrics(stressed)
            if plan.get("drawdown_alert_enabled") is False:
                conservative["drawdown_alert_probability"] = None
        candidate["goal_check"] = {"within_limits": central["probability_lower"] >= plan["required_probability"],
            "threshold": plan["required_probability"], "gate_basis": "wilson_95pct_lower_bound",
            "central": central, "conservative": conservative,
            "stress_contribution_ratio": plan["contribution_stress_ratio"],
            "stress_is_sensitivity_not_probability": True,
            "capital_gate_basis": "wilson_95pct_lower_bound_same_paths"}
    require_goal_checks(definition, candidates)
    return {"kind": "funding_goal", "funding": summary, "candidates": candidates,
            "model": {"version": goals.VERSION, "paths": paths, "seed": seed,
                      "frequency": "monthly", "common_random_numbers": True},
            "limitations": LIMITATIONS, "execution": goals.execution_audit()}
