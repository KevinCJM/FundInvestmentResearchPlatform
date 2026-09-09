"""Array orchestration and serialization for factor-based return attribution."""
from __future__ import annotations

import numpy as np

from backend.custom_indicators.errors import ValidationError
from . import attribution_kernels as kernels
from .repository import clean

ATTRIBUTION_VERSION = "factor-contributions-njit-1.0.0"
DAILY_REASONS = {0: None, 1: "拟合有效样本不足", 2: "约束求解未收敛", 3: "解释变量秩亏或无有效变化",
                 4: "滚动窗口预热", 5: "产品收益缺失或无效", 6: "因子收益或RF缺失", 7: "数值计算或对账失败"}
SUMMARY_KEYS = ("days", "valid_days", "actual_days", "total_return", "contribution_sum",
                "reconciliation_error", "model_r2", "residual_volatility", "coverage", "status_code")


def validate_attribution_size(request, days, assets, factors):
    if days < 2:
        raise ValidationError("ATTRIBUTION_DATES", "归因区间至少需要两个交易日。")
    if days * assets * (3 * factors + 10) > 2_000_000:
        raise ValidationError("ATTRIBUTION_RESULT_LIMIT", "贡献结果超过200万单元，请减少产品或缩短区间。")
    if request["exposure_mode"] == "rolling":
        window = request["rolling_window"]
        if days <= window:
            raise ValidationError("ATTRIBUTION_WINDOW", "研究区间必须长于滚动窗口，预热期不生成贡献。")
        if request["min_observations"] < max(30, 5 * factors):
            raise ValidationError("ATTRIBUTION_MINIMUM", "最少有效样本不能低于30或因子数的5倍。")
        fits = ((days - window - 1) // request["refit_step"] + 1) * assets
        if fits > 2500:
            raise ValidationError("ATTRIBUTION_FIT_LIMIT", "滚动拟合超过2500次，请减少产品或区间，或提高重估间隔。")


def build_contribution_analysis(request, data, dates, start, split, returns, factors, rf,
                                coefficients, stats, names, dependent_return):
    """Keep numerical operations in the kernels; Python handles dates and records."""
    n, assets = returns.shape
    k = len(names)
    mode = int(request["exposure_mode"] == "rolling")
    validate_attribution_size(request, n, assets, k)
    available = kernels.return_availability_kernel(data["available"])[start:].copy()
    days = data["days"][start:].copy()
    betas, fit_meta, last_beta, last_stats = kernels.exposure_path_kernel(
        returns, factors, rf, available, days, coefficients, stats, split,
        request["rolling_window"], request["min_observations"], request["refit_step"],
        mode, 0 if request["model"] == "rbsa" else 1)
    components, checks, statuses = kernels.daily_contributions_kernel(returns, factors, rf, betas, fit_meta)
    evaluation_start = request["rolling_window"] if mode else 0
    labels = dates.strftime("%Y-%m-%d").tolist()
    selected = list(range(k)) + ([k] if dependent_return == "excess" else []) + [k + 1, k + 2]
    definitions = [{"id": f"factor_{j}", "label": name, "kind": "factor"} for j, name in enumerate(names)]
    if dependent_return == "excess":
        definitions.append({"id": "rf", "label": "无风险收益", "kind": "risk_free"})
    definitions += [{"id": "intercept", "label": "模型截距", "kind": "intercept"},
                    {"id": "residual", "label": "未解释残差", "kind": "residual"}]
    products = []
    for a, identity in enumerate(data["identities"]):
        daily = []
        for t, label in enumerate(labels):
            fit_start, fit_end = int(fit_meta[t, a, 2]), int(fit_meta[t, a, 3])
            code = int(statuses[t, a])
            daily.append({
                "date": label, "sample": "in_sample" if t < split else "out_of_sample",
                "status": "ok" if code == 0 else "warmup" if code == 4 else "unavailable",
                "reason": DAILY_REASONS[code], "actual_return": returns[t, a],
                "exposures": betas[t, a, :k], "factor_returns": factors[t],
                "contributions": components[t, a, selected],
                "contribution_sum": checks[t, a, 0], "reconciliation_error": checks[t, a, 1],
                "fit_start": labels[fit_start] if fit_start >= 0 else None,
                "fit_end": labels[fit_end] if fit_end >= 0 else None,
                "fit_observations": fit_meta[t, a, 1], "fit_r2": fit_meta[t, a, 4],
                "exposure_status": "ok" if fit_meta[t, a, 0] == 0 else "unavailable",
                "exposure_basis": "prior_window" if mode else "retrospective_fit" if t < split else "fixed_training_fit",
            })
        products.append({**identity, "daily": daily, "summaries": {}, "curves": {}})
    ranges = [("all", evaluation_start, n),
              ("in_sample", evaluation_start, max(evaluation_start, split)),
              ("out_of_sample", max(evaluation_start, split), n)]
    months = {}
    for t in range(evaluation_start, n):
        months.setdefault(labels[t][:7], []).append(t)
    ranges.extend((month, positions[0], positions[-1] + 1) for month, positions in months.items())
    m = components.shape[2]
    for name, lo, hi in ranges:
        curves, summary = kernels.link_contributions_kernel(returns, components, statuses, lo, hi)
        for a, product in enumerate(products):
            fields = dict(zip(SUMMARY_KEYS, summary[a, m:]))
            status = int(fields.pop("status_code"))
            product["summaries"][name] = {
                **fields, "start_date": labels[lo] if hi > lo else None,
                "end_date": labels[hi - 1] if hi > lo else None,
                "status": ("complete", "incomplete", "empty", "numerical_error")[status],
                "contributions": summary[a, selected],
            }
            if name in {"all", "in_sample", "out_of_sample"}:
                product["curves"][name] = [{"date": labels[t], "contributions": curves[t - lo, a, selected],
                    "total_return": curves[t - lo, a, m], "contribution_sum": curves[t - lo, a, m + 1],
                    "reconciliation_error": curves[t - lo, a, m + 2]} for t in range(lo, hi)]
    if mode:
        # Legacy overview fields describe the final scheduled fit and walk-forward OOS.
        # Arithmetic annualization is already computed by the shared NJIT fitter.
        for a, product in enumerate(products):
            oos = product["summaries"]["out_of_sample"]
            last_stats[a, 1] = oos["valid_days"]
            last_stats[a, 3] = oos["model_r2"]
            last_stats[a, 6] = oos["residual_volatility"]
    analysis = {
        "schema_version": 1, "engine_version": ATTRIBUTION_VERSION,
        "mode": request["exposure_mode"], "dependent_return": dependent_return,
        "linking_method": "beginning_wealth_weighted", "units": "decimal_return_contribution",
        "warmup_days": evaluation_start, "evaluation_start": labels[evaluation_start],
        "summary_basis": "last_scheduled_fit_and_walk_forward_oos" if mode else "fixed_in_sample_fit",
        "components": definitions, "products": products,
        "notes": [
            "对账正确仅表示分解恒等式成立，不证明模型有效；请同时检查样本外R²、残差和覆盖率。",
            "贡献为百分点而非独立策略收益；每个统计区间独立从财富1串联，月度贡献不能直接相加。",
            "区间缺口不跳过、不补零；完整贡献不可用时仍可查看有数据的逐日记录和其他独立月份。",
            "滚动模式仅用此前收益日期及已公告净值；因子数据缺乏完整历史版本/发布时间，结果不是实时PIT信号。",
            "固定模式样本内暴露为事后拟合；滚动样本外为逐步前推，并非始终冻结训练集。",
        ],
    }
    return clean(analysis), last_beta, last_stats, {
        "return_available": available, "exposure_path": betas, "fit_metadata": fit_meta,
        "daily_contributions": components, "daily_status": statuses,
    }
