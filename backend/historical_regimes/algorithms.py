"""Regime algorithms with explicit causal and retrospective execution paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    from backend.instrument_analytics_numba import count_true_kernel, finite_mask_kernel
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from instrument_analytics_numba import count_true_kernel, finite_mask_kernel

from custom_indicators.errors import ValidationError

from .numba_kernels import (
    apply_standardization_kernel,
    change_point_kernel,
    component_order_kernel,
    conflict_rate_kernel,
    ensemble_consensus_kernel,
    execution_audit,
    feature_pipeline_kernel,
    gmm_fit_kernel,
    gmm_posterior_kernel,
    hmm_filtered_posterior_kernel,
    hmm_fit_kernel,
    hmm_smoothed_posterior_kernel,
    initialize_gaussian_means_kernel,
    merrill_clock_kernel,
    posterior_assignment_kernel,
    require_historical_regime_kernels_ready,
    standardize_fit_kernel,
    trend_state_kernel,
    turning_point_kernel,
    valid_rows_kernel,
)


UNKNOWN_STATE = "unclassified"


def _finite_count(values: np.ndarray) -> int:
    return int(
        count_true_kernel(
            finite_mask_kernel(np.ascontiguousarray(values, dtype=np.float64))
        )
    )


@dataclass
class AlgorithmOutput:
    labels: list[str]
    filtered: np.ndarray
    scores: np.ndarray
    probabilities: list[Optional[dict[str, float]]]
    confidence: np.ndarray
    reasons: list[list[str]]
    recognition_index: np.ndarray
    features: dict[str, np.ndarray] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    causality: dict[str, Any] = field(default_factory=dict)


def _state_by_role(states: list[dict[str, Any]], role: str, fallback: int) -> str:
    for state in states:
        if state.get("role") == role:
            return str(state["id"])
    ordered = sorted(states, key=lambda item: int(item.get("order", 0)))
    return str(ordered[min(max(fallback, 0), len(ordered) - 1)]["id"])


def _three_states(states: list[dict[str, Any]]) -> tuple[str, str, str]:
    return (
        _state_by_role(states, "positive", 0),
        _state_by_role(states, "neutral", min(1, len(states) - 1)),
        _state_by_role(states, "negative", len(states) - 1),
    )


def _coerce_number(value: Any, default: float, field: str) -> float:
    try:
        result = float(default if value is None else value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("INVALID_PARAMETER", f"{field} 必须是数值。", field) from exc
    if not np.isfinite(result):
        raise ValidationError("INVALID_PARAMETER", f"{field} 必须是有限数值。", field)
    return result


def _coerce_int(value: Any, default: int, field: str, minimum: int = 1) -> int:
    try:
        result = int(value if value is not None else default)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError("INVALID_PARAMETER", f"{field} 必须是整数。", field) from exc
    if result < minimum:
        raise ValidationError("INVALID_PARAMETER", f"{field} 不能小于 {minimum}。", field)
    return result


def _numeric_column(frame: pd.DataFrame, field: str) -> np.ndarray:
    return np.ascontiguousarray(
        pd.to_numeric(frame[field], errors="coerce").to_numpy(dtype=np.float64),
    )


def _feature_contract(
    features: dict[str, Any],
    *,
    allow_transform: bool = True,
) -> dict[str, Any]:
    method = str(features.get("filter") or "ema").lower()
    filter_codes = {
        "ema": 0,
        "sma": 1,
        "kalman": 2,
        "zero_phase": 3,
        "zero_phase_butterworth": 3,
        "filtfilt": 3,
    }
    if method not in filter_codes:
        raise ValidationError("UNSUPPORTED_FILTER", "不支持的滤波方法。", "features.filter")
    transform = str(features.get("transform") or "identity").lower()
    transform_codes = {
        "identity": 0,
        "none": 0,
        "level": 0,
        "log": 1,
        "return": 2,
        "zscore": 3,
    }
    if not allow_transform:
        transform = "identity"
    if transform not in transform_codes:
        raise ValidationError(
            "UNSUPPORTED_TRANSFORM",
            "features.transform 仅支持 identity、log、return 或 zscore。",
            "features.transform",
        )
    process_variance = _coerce_number(
        features.get("process_variance"),
        1e-5,
        "features.process_variance",
    )
    measurement_variance = _coerce_number(
        features.get("measurement_variance"),
        1e-3,
        "features.measurement_variance",
    )
    if process_variance < 0.0 or measurement_variance <= 0.0:
        raise ValidationError(
            "INVALID_FILTER_VARIANCE",
            "卡尔曼滤波过程方差须不小于 0，观测方差须大于 0。",
            "features",
        )
    window = _coerce_int(features.get("window"), 20, "features.window", 2)
    order = min(_coerce_int(features.get("order"), 2, "features.order"), 5)
    return {
        "method": (
            "zero_phase_njit_lowpass" if filter_codes[method] == 3 else method
        ),
        "filter_code": filter_codes[method],
        "transform": transform,
        "transform_code": transform_codes[transform],
        "window": window,
        "process_variance": process_variance,
        "measurement_variance": measurement_variance,
        "order": order,
        "causal": filter_codes[method] != 3,
    }


def _trend_features(
    frame: pd.DataFrame,
    features: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, str]:
    values = _numeric_column(frame, "value")
    contract = _feature_contract(features)
    if (
        contract["filter_code"] == 3
        and _finite_count(values) < max(12, contract["window"])
    ):
        raise ValidationError(
            "INSUFFICIENT_FILTER_DATA",
            "零相位滤波需要更多有效观测。",
            "features.window",
        )
    slope_window = _coerce_int(
        features.get("slope_window"),
        5,
        "features.slope_window",
    )
    volatility_window = _coerce_int(
        features.get("volatility_window"),
        20,
        "features.volatility_window",
        2,
    )
    _, filtered, slope, volatility = feature_pipeline_kernel(
        values,
        int(contract["transform_code"]),
        int(contract["filter_code"]),
        int(contract["window"]),
        slope_window,
        volatility_window,
        float(contract["process_variance"]),
        float(contract["measurement_variance"]),
        int(contract["order"]),
    )
    return (
        filtered,
        slope,
        volatility,
        bool(contract["causal"]),
        str(contract["method"]),
    )


def _format_confirmed_codes(
    labels_codes: np.ndarray,
    reason_codes: np.ndarray,
    pending_counts: np.ndarray,
    transition_from: np.ndarray,
    transition_to: np.ndarray,
    state_ids: list[str],
    confirmation: int,
) -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
    labels = [
        state_ids[int(code)] if int(code) >= 0 else UNKNOWN_STATE
        for code in labels_codes
    ]
    reasons: list[list[str]] = []
    evidence: list[dict[str, Any]] = []
    for index, code in enumerate(reason_codes):
        if int(code) == 0:
            reasons.append(["输入或派生特征缺失，未用 0 替代"])
        elif int(code) == 1:
            previous = state_ids[int(transition_from[index])]
            current = state_ids[int(transition_to[index])]
            reasons.append(
                [
                    f"连续 {confirmation} 期满足状态切换条件",
                    f"{previous} → {current}",
                ]
            )
            evidence.append(
                {
                    "kind": "state_transition",
                    "candidate_start_index": max(index - confirmation + 1, 0),
                    "recognized_index": index,
                    "from_state": previous,
                    "to_state": current,
                    "confirmation_observations": confirmation,
                }
            )
        elif int(code) == 2:
            reasons.append(
                [
                    "候选状态确认中"
                    f"（{int(pending_counts[index])}/{confirmation}）"
                ]
            )
        else:
            reasons.append([f"维持状态 {labels[index]}"])
    return labels, reasons, evidence


def _run_trend_rule(
    frame: pd.DataFrame,
    states: list[dict[str, Any]],
    features: dict[str, Any],
    params: dict[str, Any],
    *,
    relative: bool,
) -> AlgorithmOutput:
    positive, neutral, negative = _three_states(states)
    filtered, slope, volatility, causal, filter_method = _trend_features(frame, features)
    raw_values = _numeric_column(frame, "value")
    upper = _coerce_number(params.get("upper", params.get("bull_enter", 0.001)), 0.001, "algorithm.parameters.upper")
    lower = _coerce_number(params.get("lower", params.get("bear_enter", -0.001)), -0.001, "algorithm.parameters.lower")
    positive_exit = _coerce_number(params.get("positive_exit", params.get("bull_exit", 0.0)), 0.0, "algorithm.parameters.positive_exit")
    negative_exit = _coerce_number(params.get("negative_exit", params.get("bear_exit", 0.0)), 0.0, "algorithm.parameters.negative_exit")
    if lower >= upper:
        raise ValidationError("INVALID_THRESHOLDS", "下阈值必须小于上阈值。", "algorithm.parameters")
    confirmation = _coerce_int(params.get("confirmation"), 3, "algorithm.parameters.confirmation")
    min_duration = _coerce_int(params.get("min_duration"), 1, "algorithm.parameters.min_duration")
    (
        label_codes,
        confidence,
        recognition,
        _,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = trend_state_kernel(
        raw_values,
        np.ascontiguousarray(slope),
        upper,
        lower,
        positive_exit,
        negative_exit,
        confirmation,
        min_duration,
    )
    labels, reasons, evidence = _format_confirmed_codes(
        label_codes,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
        [positive, neutral, negative],
        confirmation,
    )
    raw_valid = np.isfinite(raw_values)
    filtered_values = filtered.copy()
    slope_values = slope.copy()
    volatility_values = volatility.copy()
    filtered_values[~raw_valid] = np.nan
    slope_values[~raw_valid] = np.nan
    volatility_values[~raw_valid] = np.nan
    classification = "causal" if causal else "repainting"
    family = "relative_strength" if relative else "causal_filter"
    return AlgorithmOutput(
        labels=labels,
        filtered=filtered_values,
        scores=slope_values,
        probabilities=[None] * len(frame),
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        features={"trend_slope": slope_values, "rolling_volatility": volatility_values},
        evidence=evidence,
        diagnostics={
            "filter": filter_method,
            "thresholds": {"upper": upper, "lower": lower},
            "relative": relative,
            "execution_audit": execution_audit(
                family,
                ["feature_pipeline", "trend_state"],
            ),
        },
        causality={
            "classification": classification,
            "is_causal": causal,
            "uses_future_data": not causal,
            "repaints": not causal,
            "blockers": [] if causal else ["零相位滤波使用观测点之后的数据，历史边界会重绘。"],
            "warnings": [],
        },
    )


def _run_turning_point(
    frame: pd.DataFrame,
    states: list[dict[str, Any]],
    params: dict[str, Any],
) -> AlgorithmOutput:
    positive, neutral, negative = _three_states(states)
    values = _numeric_column(frame, "value")
    window = _coerce_int(params.get("window"), 20, "algorithm.parameters.window", 2)
    min_move = _coerce_number(params.get("min_move"), 0.08, "algorithm.parameters.min_move")
    if len(values) < 2 * window + 3:
        raise ValidationError("INSUFFICIENT_TURNING_POINT_DATA", "峰谷识别样本不足，需覆盖至少两个对称窗口。", "target")
    label_codes, recognition, confidence, segment_move, extrema = turning_point_kernel(
        values,
        window,
        min_move,
    )
    state_ids = [positive, neutral, negative]
    labels = [
        state_ids[int(code)] if int(code) >= 0 else UNKNOWN_STATE
        for code in label_codes
    ]
    reasons = [
        [
            "由事后确认的峰谷区间划分",
            f"区间变动 {float(segment_move[index]):.2%}",
        ]
        if int(label_codes[index]) >= 0
        else ["尚未形成完整峰谷区间"]
        for index in range(len(values))
    ]
    evidence: list[dict[str, Any]] = []
    for index, kind_code in enumerate(extrema):
        if int(kind_code) == 0:
            continue
        evidence.append(
            {
                "kind": "peak" if int(kind_code) == 1 else "trough",
                "index": index,
                "recognized_index": min(index + window, len(values) - 1),
                "value": float(values[index]),
            }
        )
    return AlgorithmOutput(
        labels=labels,
        filtered=values.copy(),
        scores=np.full(len(values), np.nan),
        probabilities=[None] * len(values),
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        evidence=evidence,
        diagnostics={
            "symmetric_window": window,
            "minimum_phase_move": min_move,
            "turning_points": len(evidence),
            "execution_audit": execution_audit(
                "turning_point",
                ["turning_point"],
            ),
        },
        causality={
            "classification": "non_causal",
            "is_causal": False,
            "uses_future_data": True,
            "repaints": True,
            "blockers": ["峰谷需要未来对称窗口确认，不能作为当时可交易信号。"],
            "warnings": ["可用于历史叙事与条件统计，不可直接用于正式回测或 TAA。"],
        },
    )


def _run_merrill_clock(
    frame: pd.DataFrame,
    states: list[dict[str, Any]],
    features: dict[str, Any],
    params: dict[str, Any],
) -> AlgorithmOutput:
    growth_field = str(params.get("growth_field") or "growth")
    inflation_field = str(params.get("inflation_field") or "inflation")
    if growth_field not in frame.columns or inflation_field not in frame.columns:
        raise ValidationError("MISSING_CLOCK_FEATURE", "美林时钟需要增长与通胀字段。", "target.rows")
    window = _coerce_int(features.get("window"), 3, "features.window", 2)
    slope_window = _coerce_int(features.get("slope_window"), 1, "features.slope_window")
    growth = _numeric_column(frame, growth_field)
    inflation = _numeric_column(frame, inflation_field)
    contract = _feature_contract({**features, "window": window}, allow_transform=False)
    if contract["filter_code"] == 3 and (
        _finite_count(growth) < max(12, window)
        or _finite_count(inflation) < max(12, window)
    ):
        raise ValidationError(
            "INSUFFICIENT_FILTER_DATA",
            "零相位滤波需要更多有效观测。",
            "features.window",
        )
    role_ids = [
        _state_by_role(states, "growth_up_inflation_down", 0),
        _state_by_role(states, "growth_up_inflation_up", 1),
        _state_by_role(states, "growth_down_inflation_up", 2),
        _state_by_role(states, "growth_down_inflation_down", 3),
    ]
    confirmation = _coerce_int(params.get("confirmation"), 2, "algorithm.parameters.confirmation")
    (
        growth_filtered,
        growth_direction,
        inflation_direction,
        label_codes,
        confidence,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = merrill_clock_kernel(
        growth,
        inflation,
        int(contract["filter_code"]),
        window,
        slope_window,
        float(contract["process_variance"]),
        float(contract["measurement_variance"]),
        int(contract["order"]),
        confirmation,
    )
    labels, reasons, evidence = _format_confirmed_codes(
        label_codes,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
        role_ids,
        confirmation,
    )
    causal = bool(contract["causal"])
    return AlgorithmOutput(
        labels=labels,
        filtered=growth_filtered,
        scores=growth_direction,
        probabilities=[None] * len(frame),
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        features={
            "growth_level": growth,
            "inflation_level": inflation,
            "growth_direction": growth_direction,
            "inflation_direction": inflation_direction,
        },
        evidence=evidence,
        diagnostics={
            "growth_field": growth_field,
            "inflation_field": inflation_field,
            "filters": [contract["method"], contract["method"]],
            "release_date_aware": True,
            "execution_audit": execution_audit(
                "merrill_clock",
                ["merrill_clock"],
            ),
        },
        causality={
            "classification": "causal" if causal else "repainting",
            "is_causal": causal,
            "uses_future_data": not causal,
            "repaints": not causal,
            "blockers": [] if causal else ["宏观时钟使用双边滤波，历史状态会重绘。"],
            "warnings": ["宏观数据的实时可靠性取决于 available_at 与 vintage 完整性。"],
        },
    )


def _model_features(
    frame: pd.DataFrame,
    params: dict[str, Any],
    features: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    requested = params.get("feature_fields")
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",") if item.strip()]
    if isinstance(requested, list) and requested:
        fields = [str(item) for item in requested]
        missing = [field for field in fields if field not in frame.columns]
        if missing:
            raise ValidationError("MISSING_MODEL_FEATURE", f"缺少模型特征：{', '.join(missing)}。", "algorithm.parameters.feature_fields")
        matrix = np.ascontiguousarray(
            np.column_stack([_numeric_column(frame, field) for field in fields]),
            dtype=np.float64,
        )
        displayed = matrix[:, 0].copy()
    else:
        model_features = dict(features)
        if "volatility_window" not in model_features and params.get("volatility_window") is not None:
            model_features["volatility_window"] = params["volatility_window"]
        filtered, slope, volatility, _, _ = _trend_features(frame, model_features)
        matrix = np.ascontiguousarray(
            np.column_stack([slope, volatility]),
            dtype=np.float64,
        )
        displayed = filtered.copy()
        fields = ["trend_slope", "rolling_volatility"]
    valid = valid_rows_kernel(matrix) != 0
    return matrix, valid, fields, displayed


def _component_state_map(means: np.ndarray, states: list[dict[str, Any]]) -> dict[int, str]:
    ordered_components = component_order_kernel(np.ascontiguousarray(means))
    ordered_states = sorted(states, key=lambda state: int(state.get("order", 0)), reverse=True)
    if len(states) == 3:
        positive, neutral, negative = _three_states(states)
        ordered_state_ids = [negative, neutral, positive]
    else:
        ordered_state_ids = [str(state["id"]) for state in ordered_states]
    return {int(component): ordered_state_ids[min(index, len(ordered_state_ids) - 1)] for index, component in enumerate(ordered_components)}


def _run_latent_model(
    frame: pd.DataFrame,
    states: list[dict[str, Any]],
    params: dict[str, Any],
    features: dict[str, Any],
    mode: str,
    family: str,
) -> AlgorithmOutput:
    matrix, valid, feature_names, displayed = _model_features(frame, params, features)
    valid_indices = np.flatnonzero(valid)
    components = _coerce_int(params.get("states"), min(3, len(states)), "algorithm.parameters.states", 2)
    components = min(components, len(states))
    initial_train_size = _coerce_int(params.get("initial_train_size"), max(30, components * 10), "algorithm.parameters.initial_train_size", components * 5)
    iterations = min(_coerce_int(params.get("iterations"), 60, "algorithm.parameters.iterations"), 200)
    if len(valid_indices) < max(initial_train_size + (1 if mode == "realtime" else 0), components * 5):
        raise ValidationError("INSUFFICIENT_MODEL_DATA", "模型有效样本不足，请缩短初始训练窗或补充数据。", "target")
    training_count = initial_train_size if mode == "realtime" else len(valid_indices)
    training_indices = valid_indices[:training_count]
    training_raw = np.ascontiguousarray(matrix[training_indices], dtype=np.float64)
    training, mean, scale = standardize_fit_kernel(training_raw)
    all_standardized = apply_standardization_kernel(matrix, mean, scale)
    initialized_means = initialize_gaussian_means_kernel(
        np.ascontiguousarray(training, dtype=np.float64),
        np.int64(components),
        np.int64(0),
        np.int64(0),
        np.empty((0, 0), dtype=np.float64),
    )
    labels = [UNKNOWN_STATE] * len(frame)
    probabilities: list[Optional[dict[str, float]]] = [None] * len(frame)
    confidence = np.full(len(frame), np.nan, dtype=float)
    reasons: list[list[str]] = [["模型特征尚不可用"] for _ in range(len(frame))]
    recognition = np.arange(len(frame), dtype=int)

    if family == "gmm":
        weights, means, variances, completed = gmm_fit_kernel(
            np.ascontiguousarray(training),
            components,
            iterations,
            initialized_means,
        )
        classify_indices = valid_indices[training_count:] if mode == "realtime" else valid_indices
        posterior = gmm_posterior_kernel(
            np.ascontiguousarray(all_standardized[classify_indices]),
            np.ascontiguousarray(weights),
            np.ascontiguousarray(means),
            np.ascontiguousarray(variances),
        )
        model_details: dict[str, Any] = {"weights": weights.tolist(), "means": means.tolist(), "variances": variances.tolist()}
    else:
        initial, transition, means, variances, completed = hmm_fit_kernel(
            np.ascontiguousarray(training),
            components,
            iterations,
            initialized_means,
        )
        if mode == "retrospective":
            classify_indices = valid_indices
            observations = np.ascontiguousarray(
                all_standardized[classify_indices],
                dtype=np.float64,
            )
            posterior = hmm_smoothed_posterior_kernel(
                observations,
                np.ascontiguousarray(initial),
                np.ascontiguousarray(transition),
                np.ascontiguousarray(means),
                np.ascontiguousarray(variances),
            )
            probability_type = "smoothed"
        else:
            classify_indices = valid_indices[training_count:]
            observations = np.ascontiguousarray(
                all_standardized[classify_indices],
                dtype=np.float64,
            )
            posterior = hmm_filtered_posterior_kernel(
                np.ascontiguousarray(training),
                observations,
                np.ascontiguousarray(initial),
                np.ascontiguousarray(transition),
                np.ascontiguousarray(means),
                np.ascontiguousarray(variances),
            )
            probability_type = "filtered"
        model_details = {"initial": initial.tolist(), "transition": transition.tolist(), "means": means.tolist(), "variances": variances.tolist(), "probability_type": probability_type}

    mapping = _component_state_map(means, states)
    assignments, assigned_confidence = posterior_assignment_kernel(
        np.ascontiguousarray(posterior),
    )
    for row_index, point_index in enumerate(classify_indices):
        component = int(assignments[row_index])
        state_id = mapping[component]
        labels[point_index] = state_id
        probabilities[point_index] = {mapping[index]: float(posterior[row_index, index]) for index in range(components)}
        confidence[point_index] = float(assigned_confidence[row_index])
        reasons[point_index] = [f"{family.upper()} 最大后验状态 {state_id}", f"置信度 {confidence[point_index]:.1%}"]
    if mode == "realtime":
        for point_index in training_indices:
            reasons[point_index] = ["初始训练窗，仅用于估计参数，不产生可交易状态"]
    classification = "point_in_time_trained" if mode == "realtime" else "non_causal"
    blockers = [] if mode == "realtime" else ["模型使用全样本拟合；事后状态会随新增数据变化。"]
    if mode == "retrospective" and family != "gmm":
        blockers.append("HMM 使用 smoothed 概率，包含未来观测信息。")
    realtime_start = int(classify_indices[0]) if len(classify_indices) else None
    kernel_ids = [
        "valid_rows",
        "standardize_fit",
        "apply_standardization",
        "initialize_gaussian_means",
        "component_order",
        "posterior_assignment",
    ]
    if not params.get("feature_fields"):
        kernel_ids.insert(0, "feature_pipeline")
    if family == "gmm":
        kernel_ids.extend(["gmm_fit", "gmm_posterior"])
    else:
        kernel_ids.extend(
            [
                "hmm_fit",
                (
                    "hmm_filtered_posterior"
                    if mode == "realtime"
                    else "hmm_smoothed_posterior"
                ),
            ]
        )
    return AlgorithmOutput(
        labels=labels,
        filtered=displayed,
        scores=matrix[:, 0],
        probabilities=probabilities,
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        features={name: matrix[:, index] for index, name in enumerate(feature_names)},
        diagnostics={
            "model": family,
            "feature_fields": feature_names,
            "feature_pipeline": {
                "transform": features.get("transform", "identity"),
                "filter": features.get("filter", "ema"),
                "window": features.get("window", 20),
                "slope_window": features.get("slope_window", 5),
                "volatility_window": features.get("volatility_window", params.get("volatility_window", 10)),
                "source": "uploaded_fields" if params.get("feature_fields") else "unified_trend_pipeline",
            },
            "iterations": completed,
            "training_observations": training_count,
            "realtime_start_index": realtime_start,
            "parameters": model_details,
            "execution_audit": execution_audit(family, kernel_ids),
        },
        causality={
            "classification": classification,
            "is_causal": mode == "realtime",
            "uses_future_data": mode != "realtime",
            "repaints": mode != "realtime",
            "blockers": blockers,
            "warnings": ["实时路径仅在初始训练窗结束后输出状态。"] if mode == "realtime" else [],
        },
    )


def _run_change_point(
    frame: pd.DataFrame,
    states: list[dict[str, Any]],
    params: dict[str, Any],
    mode: str,
) -> AlgorithmOutput:
    positive, neutral, negative = _three_states(states)
    values = _numeric_column(frame, "value")
    window = _coerce_int(params.get("window"), 20, "algorithm.parameters.window", 4)
    threshold = _coerce_number(params.get("threshold"), 1.5, "algorithm.parameters.threshold")
    confirmation = _coerce_int(params.get("confirmation"), 2, "algorithm.parameters.confirmation")
    uses_future = mode != "realtime"
    (
        score,
        label_codes,
        confidence,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = change_point_kernel(
        values,
        window,
        threshold,
        confirmation,
        uses_future,
    )
    labels, reasons, evidence = _format_confirmed_codes(
        label_codes,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
        [positive, neutral, negative],
        confirmation,
    )
    return AlgorithmOutput(
        labels=labels,
        filtered=values.copy(),
        scores=score,
        probabilities=[None] * len(frame),
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        features={"change_score": score.copy()},
        evidence=evidence,
        diagnostics={
            "detector": (
                "online_two_window"
                if mode == "realtime"
                else "centered_two_window"
            ),
            "window": window,
            "threshold": threshold,
            "execution_audit": execution_audit(
                "change_point",
                ["change_point"],
            ),
        },
        causality={
            "classification": "causal" if not uses_future else "non_causal",
            "is_causal": not uses_future,
            "uses_future_data": uses_future,
            "repaints": uses_future,
            "blockers": [] if not uses_future else ["事后结构突变检测使用未来窗口。"],
            "warnings": [],
        },
    )


def _run_ensemble(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    mode: str,
) -> AlgorithmOutput:
    params = definition["algorithm"].get("parameters", {})
    members = params.get("members")
    if not isinstance(members, list) or not 2 <= len(members) <= 8:
        raise ValidationError("INVALID_ENSEMBLE_MEMBERS", "集成模型必须配置 2 至 8 个候选算法。", "algorithm.parameters.members")
    outputs: list[AlgorithmOutput] = []
    weights: list[float] = []
    member_meta: list[dict[str, Any]] = []
    for index, member in enumerate(members):
        if not isinstance(member, dict):
            raise ValidationError("INVALID_ENSEMBLE_MEMBER", "每个集成成员必须是对象。", "algorithm.parameters.members")
        family = str(member.get("family") or "")
        if not family or family == "ensemble":
            raise ValidationError("RECURSIVE_ENSEMBLE_BLOCKED", "集成成员不能再次引用 ensemble。", "algorithm.parameters.members")
        weight = _coerce_number(member.get("weight"), 1.0, f"algorithm.parameters.members.{index}.weight")
        if weight <= 0:
            raise ValidationError("INVALID_ENSEMBLE_WEIGHT", "集成权重必须大于 0。", "algorithm.parameters.members")
        member_definition = {
            **definition,
            "features": {**definition.get("features", {}), **(member.get("features") if isinstance(member.get("features"), dict) else {})},
            "algorithm": {
                "family": family,
                "parameters": member.get("parameters") if isinstance(member.get("parameters"), dict) else {},
            },
        }
        output = run_algorithm(frame, member_definition, mode)
        outputs.append(output)
        weights.append(weight)
        member_meta.append(
            {
                "family": family,
                "weight": weight,
                "causality_class": output.causality.get("classification"),
                "is_causal": bool(output.causality.get("is_causal")),
                "diagnostics": output.diagnostics,
            }
        )
    consensus_threshold = _coerce_number(params.get("consensus_threshold"), 0.6, "algorithm.parameters.consensus_threshold")
    if not 0.5 <= consensus_threshold <= 1.0:
        raise ValidationError("INVALID_CONSENSUS_THRESHOLD", "共识阈值必须在 0.5 至 1 之间。", "algorithm.parameters.consensus_threshold")
    state_ids = [str(state["id"]) for state in definition["states"]]
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    member_labels = np.full((len(outputs), len(frame)), -1, dtype=np.int64)
    for member_index, output in enumerate(outputs):
        member_labels[member_index] = np.asarray(
            [state_index.get(state_id, -1) for state_id in output.labels],
            dtype=np.int64,
        )
    (
        label_codes,
        probability_matrix,
        confidence,
        filtered,
        scores,
        recognition,
        conflicts,
    ) = ensemble_consensus_kernel(
        np.ascontiguousarray(member_labels),
        np.ascontiguousarray(weights, dtype=np.float64),
        np.ascontiguousarray(
            np.vstack([output.filtered for output in outputs]),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            np.vstack([output.scores for output in outputs]),
            dtype=np.float64,
        ),
        np.ascontiguousarray(
            np.vstack([output.recognition_index for output in outputs]),
            dtype=np.int64,
        ),
        len(state_ids),
        consensus_threshold,
    )
    labels = [
        state_ids[int(code)] if int(code) >= 0 else UNKNOWN_STATE
        for code in label_codes
    ]
    probabilities: list[Optional[dict[str, float]]] = [
        {
            state_id: float(probability_matrix[point_index, state_position])
            for state_position, state_id in enumerate(state_ids)
        }
        for point_index in range(len(frame))
    ]
    reasons: list[list[str]] = []
    for point_index in range(len(frame)):
        member_votes = [
            f"{member['family']}={output.labels[point_index]}"
            for member, output in zip(member_meta, outputs)
        ]
        winner_confidence = max(probabilities[point_index].values())
        if int(label_codes[point_index]) >= 0:
            reasons.append([f"加权共识 {winner_confidence:.1%}", "；".join(member_votes)])
        else:
            reasons.append([f"最高共识 {winner_confidence:.1%} 低于拒判阈值 {consensus_threshold:.1%}", "；".join(member_votes)])
    uses_future = any(output.causality.get("uses_future_data") for output in outputs)
    repaints = any(output.causality.get("repaints") for output in outputs)
    causal = all(output.causality.get("is_causal") for output in outputs) and not uses_future and not repaints
    blockers = [
        f"{member['family']}：{blocker}"
        for member, output in zip(member_meta, outputs)
        for blocker in output.causality.get("blockers", [])
    ]
    return AlgorithmOutput(
        labels=labels,
        filtered=filtered,
        scores=scores,
        probabilities=probabilities,
        confidence=confidence,
        reasons=reasons,
        recognition_index=recognition,
        features={"ensemble_consensus": confidence.copy()},
        evidence=[{"kind": "ensemble_configuration", "members": member_meta, "consensus_threshold": consensus_threshold}],
        diagnostics={
            "model": "ensemble",
            "members": member_meta,
            "consensus_threshold": consensus_threshold,
            "conflict_rejections": conflicts,
            "conflict_rate": float(
                conflict_rate_kernel(conflicts, len(frame))
            ),
            "execution_audit": execution_audit(
                "ensemble",
                ["ensemble_consensus", "conflict_rate"],
            ),
        },
        causality={
            "classification": "causal" if causal else "non_causal",
            "is_causal": causal,
            "uses_future_data": uses_future,
            "repaints": repaints,
            "blockers": blockers,
            "warnings": [f"{conflicts} 个观测因候选算法未达共识而拒判。"] if conflicts else [],
        },
    )


def run_algorithm(
    frame: pd.DataFrame,
    definition: dict[str, Any],
    mode: str,
) -> AlgorithmOutput:
    """Dispatch an algorithm while preserving its causal contract."""

    require_historical_regime_kernels_ready()
    family = str(definition["algorithm"]["family"])
    params = definition["algorithm"].get("parameters", {})
    features = definition.get("features", {})
    states = definition["states"]
    if mode == "realtime" and str(features.get("filter") or "").lower() in {"zero_phase", "zero_phase_butterworth", "filtfilt"}:
        raise ValidationError(
            "NON_CAUSAL_REALTIME_REQUEST",
            "零相位滤波会使用未来数据，不能以实时模式运行。请选择 retrospective。",
            "features.filter",
        )
    if family == "causal_filter":
        return _run_trend_rule(frame, states, features, params, relative=False)
    if family == "relative_strength":
        return _run_trend_rule(frame, states, features, params, relative=True)
    if family == "turning_point":
        if mode == "realtime":
            raise ValidationError("NON_CAUSAL_REALTIME_REQUEST", "峰谷算法需要未来窗口，只能以 retrospective 模式运行。", "mode")
        return _run_turning_point(frame, states, params)
    if family == "merrill_clock":
        return _run_merrill_clock(frame, states, features, params)
    if family in {"hmm", "markov", "gmm"}:
        return _run_latent_model(frame, states, params, features, mode, family)
    if family == "change_point":
        return _run_change_point(frame, states, params, mode)
    if family == "ensemble":
        return _run_ensemble(frame, definition, mode)
    raise ValidationError("UNSUPPORTED_ALGORITHM", "不支持的历史情景识别算法。", "algorithm.family")
