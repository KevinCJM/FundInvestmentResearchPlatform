"""Bounded forward product paths with explicit model-month settlement scope."""

import numpy as np
from backend.custom_indicators.errors import ValidationError
from backend.strategic_allocation.goal_kernels import seeded_factor_draws_kernel
from .costs import fee_arrays
from . import path_kernels as kernels

METRICS = (
    "success_probability",
    "probability_lower",
    "probability_upper",
    "payment_failure_probability",
    "terminal_p05",
    "terminal_median",
    "terminal_p95",
    "expected_unpaid_payments",
    "expected_path_cost",
    "first_payment_gap_month",
)


def diagnose(candidate, prepared, trade, models, arrays, fee, *, validation):
    if not kernels.audit()["complete"]:
        raise RuntimeError("产品路径计算尚未完成本进程预热。")
    if candidate.future_fee_assumption != "constant_declared_rates_sensitivity":
        raise ValidationError(
            "PRODUCT_PATH_FEE_ASSUMPTION",
            "请确认未来使用声明费率不变的费用敏感性假设；当前费率不自动适用于整个未来期限。",
        )
    if any(p.kind == "fund" for p in candidate.products):
        raise ValidationError(
            "PRODUCT_PATH_FUND_SCHEDULE",
            "场外基金的未来申赎依赖持有期费率表；当前单一费率仅用于本次调整和历史敏感性回放。",
        )
    if any(
        p.kind != "cash"
        and (
            not p.same_month_settlement_confirmed or len(p.settlement_terms_source) < 3
        )
        for p in candidate.products
    ):
        raise ValidationError(
            "PRODUCT_PATH_SETTLEMENT",
            "产品路径需逐项确认模型月末可同步结算及依据；有延迟或未知条款时不能假定即时到账。",
        )
    if sum(p.kind == "cash" for p in candidate.products) != 1:
        raise ValidationError(
            "PRODUCT_PATH_CASH",
            "逐产品付款路径需要一个明确的现金项目，不能用总市值代替现金。",
        )
    months = prepared["remaining_months"]
    if not months:
        return {
            "status": "not_applicable",
            "results": [],
            "reason": "剩余 0 期，只核对终值及未付事实。",
        }
    if months * candidate.paths * len(candidate.products) * 8 > 64_000_000:
        raise ValidationError(
            "PRODUCT_PATH_BUDGET", "产品路径数组超过 64MB，请减少路径数或缩短研究期限。"
        )
    if (
        months * candidate.paths * len(candidate.products) ** 2 * len(models)
        > 80_000_000
    ):
        raise ValidationError(
            "PRODUCT_PATH_WORK_BUDGET",
            "逐模型产品路径超过计算预算，请减少路径数、产品数或剩余期限。",
        )
    seed = candidate.validation_seed if validation else candidate.search_seed
    draws, _ = seeded_factor_draws_kernel(
        months, candidate.paths, len(candidate.products), seed, 0, 5.0
    )
    draws.flags.writeable = False
    buy, sell, cash = fee_arrays(candidate)
    initial = np.asarray(trade["post_cost_holdings"], dtype=np.float64)
    target = np.asarray([p.weight for p in candidate.products], dtype=np.float64)
    results = []
    for i, model in enumerate(models):
        drift, loading = kernels.monthly_joint_lognormal_kernel(
            arrays[f"product_means_{i}"], arrays[f"product_covariance_{i}"]
        )
        metrics, _ = kernels.product_funding_paths_kernel(
            draws,
            drift,
            loading,
            initial,
            target,
            buy,
            sell,
            cash,
            prepared["inflows"],
            prepared["outflows"],
            prepared["target"],
            float(fee),
            int(candidate.future_weight_rule == "monthly_rebalance"),
        )
        threshold = (
            float(prepared["plan"]["required_probability"])
            if prepared["probability_required"]
            else None
        )
        passed = metrics[1] >= threshold if threshold is not None else metrics[3] == 0
        results.append(
            {
                "model_id": model["model_id"],
                "name": model["name"],
                "enforced": model["enforced"],
                "status": (
                    ("passed" if passed else "failed")
                    if threshold is not None
                    else "not_applicable"
                ),
                "metrics": dict(zip(METRICS, metrics.tolist(), strict=True)),
                "threshold": threshold,
                "confidence": "per_model_wilson_95_not_joint",
                "seed": seed,
                "paths": candidate.paths,
                "original_plan_missed_payment": prepared[
                    "original_plan_missed_payment"
                ],
            }
        )
    return {
        "status": (
            "not_applicable"
            if not prepared["probability_required"]
            else (
                "passed"
                if all(r["status"] == "passed" for r in results if r["enforced"])
                else "failed"
            )
        ),
        "results": results,
        "scope": "conditional_same_model_month_end_settlement",
        "weight_rule": candidate.future_weight_rule,
        "assumptions": [
            "联合对数正态年度矩匹配，月度独立增量。",
            "交易、入金、支付按模型月末先到账、再调仓、再支付。",
            "未来交易费按当前声明费率不变作敏感性研究，不是全期限费率已获确认。",
            "买入持有不自动卖出补足现金；月度再平衡仍可能出现支付现金不足。",
            "逐产品已确认同月结算是模型适用前提，不认证盘中成交、延期或暂停情景。",
        ],
    }
