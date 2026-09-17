"""Clock-driven segmented validation using the existing TAA simulation core.

Folds are independently funded experiments, not a stitched investable strategy.
They never replace the primary study's declared train/holdout selection.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
from numba import int64, njit, types, uint8

from backend.custom_indicators.errors import ValidationError
from . import numeric

IM = types.Array(int64, 2, "A", readonly=True)
U = types.Array(uint8, 1, "A", readonly=True)
_WARMED_PID: int | None = None


@njit(types.UniTuple(int64, 4)(IM, U, int64, int64, int64), cache=True, nogil=True)
def mature_training_kernel(available, flags, start, stop, cutoff):
    """Trim only immature trailing labels; interior unknowns remain blockers."""
    if not 0 <= start <= stop <= available.shape[0] or flags.size != available.shape[0]:
        raise ValueError("WALK_FORWARD_AXIS")
    end = stop
    while end > start:
        known = True
        for asset in range(available.shape[1]):
            stamp = available[end - 1, asset]
            if stamp < 0 or stamp > cutoff:
                known = False
        if known:
            break
        end -= 1
    unknown, future, signals = 0, 0, 0
    for row in range(start, end):
        signals += int(flags[row] != 0)
        for asset in range(available.shape[1]):
            if available[row, asset] < 0:
                unknown += 1
            elif available[row, asset] > cutoff:
                future += 1
    return end, unknown, future, signals


def warm_walk_forward_kernels() -> dict:
    global _WARMED_PID
    _WARMED_PID = None
    mature_training_kernel.disable_compile()
    mature_training_kernel(np.zeros((40, 2), dtype=np.int64), np.ones(40, dtype=np.uint8), 0, 20, 1)
    complete = len(mature_training_kernel.signatures) == 1 and bool(mature_training_kernel.nopython_signatures)
    if not complete:
        raise RuntimeError("多段样本外验证内核预热失败。")
    _WARMED_PID = os.getpid()
    return execution_audit()


def execution_audit() -> dict:
    complete = _WARMED_PID == os.getpid() and len(mature_training_kernel.signatures) == 1
    return {"backend": "numba_njit_fixed_signature", "engine": "taa-segmented-validation/1.0.0",
            "complete": complete, "nopython": complete, "object_mode": 0, "python_fallback": 0,
            "request_time_compilation": 0,
            "kernel_signatures": {"mature_training_kernel": [str(s) for s in mature_training_kernel.signatures]}}


def evaluate_walk_forward(request, data, signals, base, lower, upper, caps, group_args) -> dict:
    if _WARMED_PID != os.getpid():
        raise RuntimeError("多段样本外验证内核未完成启动预热。")
    config = request.walk_forward
    if config is None:
        raise ValueError("WALK_FORWARD_CONFIG_REQUIRED")
    returns, probabilities, flags = data["returns"], signals["probabilities"], signals["use_signal"]
    size = len(data["dates"])
    if size - config.training_periods < 20:
        raise ValidationError("TAA_WALK_FORWARD_SHORT", "当前样本不足以形成一段训练和至少 20 期验证；请缩短训练窗口或扩大区间。")
    boundaries = [(start, min(size, start + config.validation_periods))
                  for start in range(config.training_periods, size, config.validation_periods)
                  if size - start >= 20]
    if len(boundaries) > 40:
        raise ValidationError("TAA_WALK_FORWARD_BUDGET", "单次最多验证 40 段，请增大每段验证期数或缩短研究区间。")
    plan = None
    if request.decision_policy:
        from .clocks import simulation_clock, decision_signal_flags_kernel
        plan = simulation_clock(request, data, signals)
        selection_flags = decision_signal_flags_kernel(flags, plan["decisions"])
    else:
        selection_flags = flags
    folds = []
    for number, (validation_start, end) in enumerate(boundaries, start=1):
        start = 0 if config.window_mode == "expanding" else validation_start - config.training_periods
        cutoff = (date.fromisoformat(data["period_starts"][validation_start]) - date(1970, 1, 1)).days
        mature_end, unknown, future, signal_count = mature_training_kernel(
            data["available_at"], selection_flags, start, validation_start, cutoff)
        fold = {"fold": number, "train_start": data["period_starts"][start],
                "train_end": data["dates"][mature_end - 1] if mature_end > start else None,
                "validation_start": data["dates"][validation_start], "validation_end": data["dates"][end - 1],
                "decision_cutoff": data["period_starts"][validation_start],
                "training_observations": mature_end - start, "validation_observations": end - validation_start,
                "purged_training_periods": validation_start - mature_end,
                "unknown_training_values": unknown, "future_training_values": future,
                "effective_training_signals": signal_count}
        reasons = []
        if mature_end - start < 20:
            reasons.append("剔除尚未成熟的训练尾部后，不足 20 个观察期。")
        if unknown or future:
            reasons.append("训练区间内部存在未知或尚未可得收益，不能用于当时的选优。")
        if request.search and not signal_count:
            reasons.append("训练期没有有效信号，不能比较偏离强度。")
        if reasons:
            folds.append({**fold, "status": "blocked", "reasons": reasons})
            continue
        # Every slice shares the original history. The explicit gap prevents
        # immature training labels from either selecting strength or earning P&L.
        result = numeric.evaluate_candidates(
            returns[start:end], probabilities[start:end], flags[start:end], base, signals["state_tilts"],
            lower, upper, caps, mature_end - start,
            strengths=np.asarray(numeric.DEFAULT_STRENGTHS if request.search else (0., 1.), dtype=np.float64),
            cost=request.transaction_cost_bps, risk_penalty=request.risk_penalty,
            max_tracking_error=request.max_tracking_error, max_turnover=request.max_turnover,
            objective=request.objective, selected_candidate_id=None if request.search else "scale-1",
            validation_start_index=validation_start - start,
            allow_infeasible_selected=not request.search,
            decision_policy=request.decision_policy,
            clock=None if plan is None else {k: v[start:end] for k, v in plan.items()},
            direct_tilts=None if signals.get("direct_tilts") is None else signals["direct_tilts"][start:end], **group_args)
        selected = next(item for item in result["candidates"] if item["id"] == result["selected_id"])
        if not selected["feasible"]:
            folds.append({**fold, "status": "blocked",
                          "reasons": ["固定假设在该段训练区超过主动风险或换手限制；保留该假设，不改选其他候选。"],
                          "selected_id": selected["id"], "strength": selected["strength"],
                          "training": selected["train"], "validation_feasible": selected["validation_feasible"]})
            continue
        folds.append({**fold, "status": "complete", "reasons": [], "selected_id": selected["id"],
                      "strength": selected["strength"], "training": selected["train"],
                      "validation": selected["validation"], "validation_feasible": selected["validation_feasible"],
                      "baseline_fallback_exemption": selected["baseline_fallback_exemption"]})
    return {"config": config.model_dump(mode="json"), "folds": folds,
            "completed_folds": len([f for f in folds if f["status"] == "complete"]),
            "blocked_folds": len([f for f in folds if f["status"] == "blocked"]),
            "excluded_tail_observations": size - boundaries[-1][1], "research_only": True,
            "primary_selection_changed": False, "holdout_used_for_selection": False,
            "independently_funded_intervals": True, "execution": execution_audit(),
            "warnings": ["每段独立从 SAA 起步，不能拼接成连续可交易净值。",
                         "已知数据不等于当时已部署模型；多次查看结果后改规则仍可能过拟合。",
                         "本诊断不替换主研究的训练/留出选择；下一段训练允许使用当时已经成熟的历史验证收益。"]}
