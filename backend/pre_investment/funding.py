"""Confirmed balances and occurrence reconciliation adapt to the one funding core."""

import calendar
from datetime import date
import numpy as np
from numba import njit, types
from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import digest_json
from backend.strategic_allocation import goal_kernels as goals
from backend.strategic_allocation.mandate_inputs import (
    effective_funding_plan,
    cash_success_required,
)
from backend.strategic_allocation.planning import _funding_metrics

V = types.Array(types.float64, 1, "A", readonly=True)
INT_VECTOR = types.Array(types.int64, 1, "A", readonly=True)


@njit(
    (
        V,
        INT_VECTOR,
        INT_VECTOR,
        V,
        types.int64,
        types.int64,
        types.float64,
        types.int64,
    ),
    cache=True,
    nogil=True,
)
def remaining_flows_kernel(
    amounts, months, directions, paid, elapsed, horizon, inflation, real
):
    inflows, outflows = np.zeros(horizon - elapsed), np.zeros(horizon - elapsed)
    nominal = np.empty(amounts.size)
    missed, outstanding = False, 0.0
    for i in range(amounts.size):
        nominal[i] = amounts[i] * (
            (1.0 + inflation) ** (months[i] / 12.0) if real else 1.0
        )
        if paid[i] < 0 or paid[i] > nominal[i] + 1e-8:
            raise ValueError("FUNDING_RECONCILIATION_AMOUNT")
        due = nominal[i] - paid[i]
        if months[i] <= elapsed and directions[i] < 0 and due > 1e-8:
            missed = True
            outstanding += due
        if due > 1e-8 and elapsed < horizon:
            period = max(0, months[i] - elapsed - 1)
            if directions[i] > 0:
                # Past missing contributions are not invented as future income.
                if months[i] > elapsed:
                    inflows[period] += due
            else:
                outflows[period] += due
    return inflows, outflows, nominal, missed, outstanding


@njit(
    (types.float64, types.float64, types.float64, types.int64), cache=True, nogil=True
)
def terminal_target_kernel(target, inflation, horizon, real):
    return target * ((1.0 + inflation) ** (horizon / 12.0) if real else 1.0)


for _kernel in (remaining_flows_kernel, terminal_target_kernel):
    _kernel.disable_compile()


def month_date(origin, month):
    source = date.fromisoformat(str(origin))
    year, m = divmod(source.year * 12 + source.month - 1 + month, 12)
    return date(year, m + 1, min(source.day, calendar.monthrange(year, m + 1)[1]))


def occurrences(mandate):
    plan = effective_funding_plan(mandate)
    if plan is None:
        return [], None
    budget = mandate.get("cash_budget") or plan
    budget_hash = digest_json(budget)
    origin = budget.get("balance_as_of", mandate["as_of"])
    result = []
    for index, row in enumerate(plan["flows"]):
        for month in range(
            row["first_month"], row["last_month"] + 1, row["every_months"]
        ):
            result.append(
                {
                    "occurrence_id": digest_json([budget_hash, index, month]),
                    "flow_index": index,
                    "name": row["name"],
                    "month": month,
                    "kind": row["kind"],
                    "amount": row["amount"],
                    "nominal_amount": float(
                        terminal_target_kernel(
                            float(row["amount"]),
                            float(plan["inflation"]),
                            float(month),
                            int(plan["amount_basis"] == "real"),
                        )
                    ),
                    "due_at": str(month_date(origin, month)),
                }
            )
    return result, {"budget_hash": budget_hash, "origin": origin, "plan": plan}


def prepare(mandate, state):
    goals.require_ready()
    rows, context = occurrences(mandate)
    if context is None:
        return None
    plan = context["plan"]
    horizon = mandate["horizon_years"] * 12
    if state.elapsed_months > horizon:
        raise ValidationError(
            "FUNDING_HORIZON", "已过期数不能超过原计划期限，请确认新目标版本。"
        )
    if (
        state.cutoff_phase != "after_model_month_end"
        or month_date(context["origin"], state.elapsed_months) != state.valuation_at
    ):
        raise ValidationError(
            "CASHFLOW_CALENDAR_ADAPTER_REQUIRED",
            "余额日不在原计划模型月边界；须先提供日期适配，不能取整剩余年数。",
        )
    entered = {x.occurrence_id: x for x in state.reconciliation}
    if set(entered) - {x["occurrence_id"] for x in rows}:
        raise ValidationError(
            "FUNDING_OCCURRENCE_UNKNOWN", "核对记录不属于冻结的原预算。"
        )
    paid = []
    for row in rows:
        item = entered.get(row["occurrence_id"])
        if row["month"] <= state.elapsed_months and (
            item is None or item.status == "unknown"
        ):
            raise ValidationError(
                "FUNDING_RECONCILIATION_REQUIRED",
                "请核对每笔已到期现金流；日期已过不代表已支付。",
            )
        if row["month"] > state.elapsed_months and item is not None:
            raise ValidationError(
                "FUNDING_FUTURE_RECONCILIATION",
                "提前支付需调整原预算版本，不能在月末适配中隐式移期。",
            )
        paid.append(item.paid_amount if item else 0.0)
    inflows, outflows, nominal, missed, outstanding = remaining_flows_kernel(
        np.asarray([x["amount"] for x in rows], dtype=np.float64),
        np.asarray([x["month"] for x in rows], dtype=np.int64),
        np.asarray(
            [1 if x["kind"] == "contribution" else -1 for x in rows], dtype=np.int64
        ),
        np.asarray(paid, dtype=np.float64),
        state.elapsed_months,
        horizon,
        float(plan["inflation"]),
        int(plan["amount_basis"] == "real"),
    )
    for i, row in enumerate(rows):
        item = entered.get(row["occurrence_id"])
        if item and (
            (item.status == "paid" and abs(item.paid_amount - nominal[i]) > 1e-8)
            or (item.status == "unpaid" and item.paid_amount != 0)
            or (item.status == "partial" and not 0 < item.paid_amount < nominal[i])
        ):
            raise ValidationError(
                "FUNDING_STATUS_AMOUNT", "已付、部分支付或未付状态与金额不一致。"
            )
        row["nominal_amount"] = float(nominal[i])
    target = terminal_target_kernel(
        float(plan["terminal_target"]),
        float(plan["inflation"]),
        float(horizon),
        int(plan.get("target_amount_basis", plan["amount_basis"]) == "real"),
    )
    inflows.flags.writeable = False
    outflows.flags.writeable = False
    return {
        "inflows": inflows,
        "outflows": outflows,
        "target": float(target),
        "plan": plan,
        "probability_required": cash_success_required(mandate),
        "remaining_months": horizon - state.elapsed_months,
        "original_plan_missed_payment": bool(missed),
        "overdue_payment_amount": float(outstanding),
        "occurrences": rows,
        "original_budget_hash": context["budget_hash"],
        "original_price_base": str(context["origin"]),
    }


def diagnose(prepared, mean, volatility, initial, fee, paths, seed):
    goals.require_ready()
    months = prepared["remaining_months"]
    if initial < 0:
        raise ValidationError(
            "FUNDING_NEGATIVE_CAPITAL", "费用后可投资价值为负，不支持融资续算。"
        )
    if months == 0:
        passed = (
            initial >= prepared["target"] and prepared["overdue_payment_amount"] <= 1e-8
        )
        return {
            "status": "passed" if passed else "failed",
            "deterministic": True,
            "terminal_value": initial,
            "remaining_months": 0,
            "future_conditional_success_probability": None,
            "original_plan_missed_payment": prepared["original_plan_missed_payment"],
        }
    drift, scale = goals.funding_monthly_parameters_kernel(
        float(mean), float(volatility), float(fee), 0, 1
    )
    draws, _ = goals.seeded_factor_draws_kernel(months, paths, 1, seed, 0, 5.0)
    draws.flags.writeable = False
    probability = (
        float(prepared["plan"]["required_probability"])
        if prepared["probability_required"]
        else 0.8
    )
    metrics, fan = goals.funding_paths_from_monthly_kernel(
        draws,
        drift,
        scale,
        initial,
        prepared["inflows"],
        prepared["outflows"],
        prepared["target"],
        probability,
        1.0,
        1.0,
    )
    result = _funding_metrics(metrics)
    checkpoints = [0, *range(12, months + 1, 12)]
    if checkpoints[-1] != months:
        checkpoints.append(months)
    return {
        "status": (
            ("passed" if result["probability_lower"] >= probability else "failed")
            if prepared["probability_required"]
            else "not_applicable"
        ),
        "metrics": result,
        "future_conditional_success_probability": result["success_probability"],
        "original_plan_missed_payment": prepared["original_plan_missed_payment"],
        "remaining_months": months,
        "threshold": probability if prepared["probability_required"] else None,
        "confidence": "per_model_wilson_95_not_joint",
        "seed": seed,
        "paths": paths,
        "scope": "monthly_scalar_moment_proxy",
        "target": prepared["target"],
        "fan": [
            {"month": m, "p05": float(v[0]), "median": float(v[1]), "p95": float(v[2])}
            for m, v in zip(checkpoints, fan, strict=True)
        ],
    }


def warm():
    goals.warm_goal_kernels()
    remaining_flows_kernel(
        np.array([10.0]),
        np.array([1], dtype=np.int64),
        np.array([-1], dtype=np.int64),
        np.zeros(1),
        0,
        13,
        0.0,
        0,
    )
    terminal_target_kernel(100.0, 0.02, 13.0, 1)
