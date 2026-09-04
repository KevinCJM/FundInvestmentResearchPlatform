"""Explainability, validation and comparison for regime runs."""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .algorithms import AlgorithmOutput, UNKNOWN_STATE
from .numba_kernels import (
    comparison_pair_kernel,
    conditional_statistics_kernel,
    execution_audit,
    finite_mean_kernel,
    integer_sum_kernel,
    label_summary_kernel,
    path_metrics_kernel,
    prefix_stability_kernel,
    row_disagreement_kernel,
    state_counts_kernel,
    transition_matrix_kernel,
    validation_windows_kernel,
)


def _number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _date(value: Any) -> str:
    return pd.Timestamp(value).date().isoformat()


def serialise_series(
    frame: pd.DataFrame,
    output: AlgorithmOutput,
    states: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    labels = {str(state["id"]): str(state["label"]) for state in states}
    available = pd.to_datetime(frame["available_at"])
    observation_dates = pd.to_datetime(frame["observation_date"])
    rows: list[dict[str, Any]] = []
    for index, (_, source) in enumerate(frame.iterrows()):
        recognized_index = min(max(int(output.recognition_index[index]), index), len(frame) - 1)
        recognized_at = max(pd.Timestamp(available.iloc[index]), pd.Timestamp(available.iloc[recognized_index]))
        effective_at = next(
            (
                max(pd.Timestamp(available.iloc[next_index]), pd.Timestamp(observation_dates.iloc[next_index]))
                for next_index in range(index + 1, len(frame))
                if max(pd.Timestamp(available.iloc[next_index]), pd.Timestamp(observation_dates.iloc[next_index])) > recognized_at
            ),
            None,
        )
        observation_date = _date(source["observation_date"])
        recognized_date = _date(recognized_at)
        feature_values = {
            key: _number(values[index])
            for key, values in output.features.items()
        }
        state_id = output.labels[index]
        probabilities = output.probabilities[index]
        probability_source = "model"
        if probabilities is None and state_id != UNKNOWN_STATE:
            probabilities = {state_id: 1.0}
            probability_source = "deterministic_state"
        elif probabilities is None:
            probability_source = "unavailable"
        rows.append(
            {
                "observation_date": observation_date,
                "date": observation_date,
                "data_available_at": _date(source["available_at"]),
                "recognized_at": recognized_date,
                "signal_date": recognized_date,
                "effective_date": _date(effective_at) if effective_at is not None else None,
                "executable": effective_at is not None,
                "value": _number(source.get("value")),
                "filtered_value": _number(output.filtered[index]),
                "score": _number(output.scores[index]),
                "state_id": state_id,
                "state_label": labels.get(state_id, "未分类"),
                "probabilities": probabilities or {},
                "probability_source": probability_source,
                "confidence": _number(output.confidence[index]),
                "features": feature_values,
                "reasons": output.reasons[index],
                "revision": source.get("revision"),
                "vintage": source.get("vintage"),
                "is_final": bool(source.get("is_final", True)),
            }
        )
    return rows


def _periods_per_year(frequency: str) -> int:
    return {"daily": 252, "weekly": 52, "monthly": 12, "quarterly": 4, "annual": 1}.get(frequency, 252)


def analytics_execution_audit() -> dict[str, Any]:
    return execution_audit(
        "analytics",
        [
            "path_metrics",
            "finite_mean",
            "conditional_statistics",
            "transition_matrix",
            "prefix_stability",
            "state_counts",
            "label_summary",
            "validation_windows",
            "integer_sum",
            "predecessor_count",
        ],
    )


def _path_metrics(values: np.ndarray, periods_per_year: int) -> dict[str, Any]:
    metrics = path_metrics_kernel(
        np.ascontiguousarray(values, dtype=np.float64),
        periods_per_year,
    )
    return {
        "return": _number(metrics[0]),
        "annualized_return": _number(metrics[1]),
        "volatility": _number(metrics[2]),
        "max_drawdown": _number(metrics[3]),
    }


def build_segments(
    series: list[dict[str, Any]],
    frequency: str,
) -> list[dict[str, Any]]:
    periods = _periods_per_year(frequency)
    segments: list[dict[str, Any]] = []
    start = 0
    while start < len(series):
        state = series[start]["state_id"]
        end = start
        while end + 1 < len(series) and series[end + 1]["state_id"] == state:
            end += 1
        if state != UNKNOWN_STATE:
            subset = series[start : end + 1]
            values = np.array([item["value"] if item["value"] is not None else np.nan for item in subset], dtype=float)
            confidences = np.asarray(
                [
                    float(item["confidence"])
                    if item.get("confidence") is not None
                    else np.nan
                    for item in subset
                ],
                dtype=np.float64,
            )
            segment = {
                "state_id": state,
                "state_label": series[start]["state_label"],
                "start_date": series[start]["observation_date"],
                "end_date": series[end]["observation_date"],
                "effective_start": series[start]["effective_date"],
                "recognized_at": series[start]["recognized_at"],
                "duration_observations": len(subset),
                "confidence": _number(
                    finite_mean_kernel(np.ascontiguousarray(confidences)),
                ),
                "reasons": series[start].get("reasons", []),
                **_path_metrics(values, periods),
            }
            segments.append(segment)
        start = end + 1
    return segments


def conditional_statistics(
    series: list[dict[str, Any]],
    states: list[dict[str, Any]],
    frequency: str,
) -> list[dict[str, Any]]:
    periods = _periods_per_year(frequency)
    values = np.array([item["value"] if item["value"] is not None else np.nan for item in series], dtype=float)
    state_ids = [str(state["id"]) for state in states]
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    label_codes = np.asarray(
        [state_index.get(str(item["state_id"]), -1) for item in series],
        dtype=np.int64,
    )
    metrics = conditional_statistics_kernel(
        np.ascontiguousarray(values, dtype=np.float64),
        np.ascontiguousarray(label_codes),
        len(states),
        periods,
    )
    result: list[dict[str, Any]] = []
    for state_position, state in enumerate(states):
        state_id = str(state["id"])
        row = metrics[state_position]
        result.append(
            {
                "state_id": state_id,
                "state_label": state["label"],
                "observations": int(row[0]),
                "return_observations": int(row[1]),
                "mean_period_return": _number(row[2]),
                "return": _number(row[2]),
                "annualized_return": _number(row[3]),
                "volatility": _number(row[4]),
                "max_drawdown": _number(row[5]),
                "sharpe": _number(row[6]),
                "positive_rate": _number(row[7]),
                "win_rate": _number(row[7]),
                "return_alignment": "forward_1_period_from_signal",
            }
        )
    return result


def transition_matrix(series: list[dict[str, Any]], states: list[dict[str, Any]]) -> dict[str, Any]:
    state_ids = [str(state["id"]) for state in states]
    index = {state_id: offset for offset, state_id in enumerate(state_ids)}
    label_codes = np.asarray(
        [index.get(str(item["state_id"]), -1) for item in series],
        dtype=np.int64,
    )
    counts, probabilities = transition_matrix_kernel(
        np.ascontiguousarray(label_codes),
        len(state_ids),
    )
    return {"states": state_ids, "counts": counts.tolist(), "probabilities": probabilities.tolist()}


def causality_report(output: AlgorithmOutput, mode: str, series: list[dict[str, Any]]) -> dict[str, Any]:
    base = dict(output.causality)
    temporal_violations = []
    for index, item in enumerate(series):
        dates = [pd.Timestamp(item[key]) for key in ("observation_date", "data_available_at", "recognized_at")]
        if item.get("effective_date"):
            dates.append(pd.Timestamp(item["effective_date"]))
        if dates != sorted(dates):
            temporal_violations.append(index)
    is_causal = bool(base.get("is_causal")) and not temporal_violations
    eligible = ["research_display", "product_research"]
    if mode == "realtime" and is_causal and not base.get("repaints"):
        eligible.extend(["formal_backtest", "taa"])
    blockers = list(base.get("blockers", []))
    if temporal_violations:
        blockers.append("存在观测日、可得日、识别日、生效日倒序。")
    return {
        **base,
        "is_causal": is_causal,
        "realtime_eligible": mode == "realtime" and is_causal,
        "publish_eligible_usages": eligible,
        "blockers": blockers,
        "checks": [
            {"id": "temporal_order", "passed": not temporal_violations, "violations": temporal_violations[:20]},
            {"id": "future_data", "passed": not bool(base.get("uses_future_data"))},
            {"id": "historical_repaint", "passed": not bool(base.get("repaints"))},
            {"id": "realtime_mode_claim", "passed": mode != "realtime" or is_causal},
            {"id": "last_point_not_executable", "passed": not bool(series[-1].get("executable")) if series else True},
        ],
    }


def prefix_stability(
    base: AlgorithmOutput,
    prefix: Optional[AlgorithmOutput],
    prefix_length: int,
) -> dict[str, Any]:
    if prefix is None or prefix_length <= 0:
        return {"status": "unavailable", "prefix_observations": 0, "agreement": None, "revisions": None}
    state_ids = sorted(
        {
            label
            for label in (*base.labels[:prefix_length], *prefix.labels[:prefix_length])
            if label != UNKNOWN_STATE
        }
    )
    state_index = {state_id: index for index, state_id in enumerate(state_ids)}
    base_codes = np.asarray(
        [state_index.get(label, -1) for label in base.labels[:prefix_length]],
        dtype=np.int64,
    )
    prefix_codes = np.asarray(
        [state_index.get(label, -1) for label in prefix.labels[:prefix_length]],
        dtype=np.int64,
    )
    metrics = prefix_stability_kernel(
        np.ascontiguousarray(base_codes),
        np.ascontiguousarray(prefix_codes),
    )
    comparable = int(metrics[0])
    revisions = int(metrics[1])
    return {
        "status": "passed" if revisions == 0 else "revisions_detected",
        "prefix_observations": int(prefix_length),
        "comparable_observations": comparable,
        "agreement": _number(metrics[2]),
        "revisions": revisions,
        "revision_rate": _number(metrics[3]),
    }


def walk_forward_report(
    frame: pd.DataFrame,
    output: AlgorithmOutput,
    folds: int,
    runner: Callable[[pd.DataFrame], AlgorithmOutput],
) -> dict[str, Any]:
    if not output.causality.get("is_causal"):
        return {"status": "blocked", "reason": "非因果算法不能伪装为 walk-forward 验证。", "folds": []}
    observations = len(frame)
    _, windows = validation_windows_kernel(
        np.int64(observations),
        np.int64(folds),
    )
    reports: list[dict[str, Any]] = []
    classified_by_fold: list[int] = []
    for fold in range(folds):
        train_end = int(windows[fold, 0])
        test_end = int(windows[fold, 1])
        if test_end <= train_end:
            continue
        try:
            prefix_output = runner(frame.iloc[:test_end].copy())
        except Exception as exc:  # domain validation is reported per fold, never hidden
            reports.append({"fold": fold + 1, "status": "insufficient", "train_end_index": int(windows[fold, 2]), "test_end_index": int(windows[fold, 4]), "reason": str(exc)})
            continue
        labels = prefix_output.labels[train_end:test_end]
        usable = [label for label in labels if label != UNKNOWN_STATE]
        state_ids = sorted(set(usable))
        state_index = {state_id: index for index, state_id in enumerate(state_ids)}
        label_codes = np.asarray(
            [state_index.get(label, -1) for label in labels],
            dtype=np.int64,
        )
        state_counts = state_counts_kernel(
            np.ascontiguousarray(label_codes),
            len(state_ids),
        )
        classified_observations = int(
            label_summary_kernel(np.ascontiguousarray(label_codes))[0]
        )
        classified_by_fold.append(classified_observations)
        reports.append(
            {
                "fold": fold + 1,
                "status": "ok",
                "train_end_index": int(windows[fold, 2]),
                "test_start_index": int(windows[fold, 3]),
                "test_end_index": int(windows[fold, 4]),
                "test_observations": int(windows[fold, 5]),
                "classified_observations": classified_observations,
                "state_distribution": {
                    state_id: int(state_counts[index])
                    for index, state_id in enumerate(state_ids)
                },
            }
        )
    combined_predictions = int(
        integer_sum_kernel(
            np.ascontiguousarray(classified_by_fold, dtype=np.int64)
        )
    )
    return {
        "status": "completed" if reports and combined_predictions else "insufficient",
        "method": "expanding_prefix",
        "fold_count": len(reports),
        "classified_observations": combined_predictions,
        "folds": reports,
    }


def compare_runs(runs: list[dict[str, Any]], reference_run_id: Optional[str] = None) -> dict[str, Any]:
    if not runs:
        return {"run_ids": [], "reference_run_id": None, "agreement_rate": None, "runs": [], "pairwise": [], "disagreement_periods": []}
    reference = next((run for run in runs if run["id"] == reference_run_id), runs[0])
    summaries = []
    pairwise = []
    for run in runs:
        summaries.append(
            {
                "run_id": run["id"],
                "name": run.get("name"),
                "mode": run.get("mode"),
                "algorithm_family": run.get("algorithm", {}).get("family"),
                "segments": len(run.get("segments", [])),
                "causality_class": run.get("causality", {}).get("classification"),
                "publish_eligible_usages": run.get("causality", {}).get("publish_eligible_usages", []),
            }
        )
    label_maps = {
        run["id"]: {item["observation_date"]: item["state_id"] for item in run.get("series", [])}
        for run in runs
    }
    pair_agreements: list[float] = []
    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1 :]:
            left_labels = label_maps[left["id"]]
            right_labels = label_maps[right["id"]]
            common = sorted(set(left_labels) & set(right_labels))
            state_ids = sorted(
                {
                    state_id
                    for date in common
                    for state_id in (left_labels[date], right_labels[date])
                    if state_id != UNKNOWN_STATE
                }
            )
            state_index = {
                state_id: position for position, state_id in enumerate(state_ids)
            }
            left_codes = np.asarray(
                [state_index.get(left_labels[date], -1) for date in common],
                dtype=np.int64,
            )
            right_codes = np.asarray(
                [state_index.get(right_labels[date], -1) for date in common],
                dtype=np.int64,
            )
            metrics = comparison_pair_kernel(
                np.ascontiguousarray(left_codes),
                np.ascontiguousarray(right_codes),
            )
            usable_count = int(metrics[0])
            agreement = float(metrics[2])
            if np.isfinite(agreement):
                pair_agreements.append(agreement)
            pairwise.append(
                {
                    "left_run_id": left["id"],
                    "right_run_id": right["id"],
                    "common_observations": usable_count,
                    "agreement_rate": _number(agreement),
                    "boundary_distance": _number(metrics[3]),
                }
            )

    common_dates = sorted(set.intersection(*(set(labels) for labels in label_maps.values()))) if label_maps else []
    all_state_ids = sorted(
        {
            state_id
            for date in common_dates
            for labels in label_maps.values()
            for state_id in (labels[date],)
            if state_id != UNKNOWN_STATE
        }
    )
    all_state_index = {
        state_id: position for position, state_id in enumerate(all_state_ids)
    }
    comparison_codes = np.full(
        (len(common_dates), len(label_maps)),
        -1,
        dtype=np.int64,
    )
    run_ids = list(label_maps)
    for date_index, date in enumerate(common_dates):
        for run_index, run_id in enumerate(run_ids):
            comparison_codes[date_index, run_index] = all_state_index.get(
                label_maps[run_id][date],
                -1,
            )
    disagreement_mask = row_disagreement_kernel(
        np.ascontiguousarray(comparison_codes)
    )
    disagreements: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for date_index, date in enumerate(common_dates):
        states_at_date = {run_id: labels[date] for run_id, labels in label_maps.items()}
        disagrees = int(disagreement_mask[date_index]) == 1
        if disagrees and current is None:
            current = {"start_date": date, "end_date": date, "states": states_at_date}
        elif disagrees and current is not None and current["states"] == states_at_date:
            current["end_date"] = date
        elif disagrees and current is not None:
            disagreements.append(current)
            current = {"start_date": date, "end_date": date, "states": states_at_date}
        elif current is not None:
            disagreements.append(current)
            current = None
    if current is not None:
        disagreements.append(current)
    return {
        "run_ids": [run["id"] for run in runs],
        "reference_run_id": reference["id"],
        "agreement_rate": (
            _number(
                finite_mean_kernel(
                    np.ascontiguousarray(pair_agreements, dtype=np.float64),
                )
            )
            if pair_agreements
            else None
        ),
        "runs": summaries,
        "pairwise": pairwise,
        "disagreement_periods": disagreements,
        "execution": execution_audit(
            "comparison",
            [
                "comparison_pair",
                "finite_mean",
                "row_disagreement",
            ],
        ),
    }
