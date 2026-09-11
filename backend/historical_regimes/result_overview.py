"""Immutable observation-date summaries for completed Regime Graph runs.

The graph is never evaluated here. Python handles frozen metadata and row
serialization; the fixed-signature kernel owns segmentation and sample counts.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import numpy as np
from numba import int64, njit, types

_I64 = int64[::1]
_I64_2D = int64[:, ::1]
_SEGMENTS_RESULT = types.Tuple((_I64_2D, _I64, int64, int64, int64))


@njit(_SEGMENTS_RESULT(_I64, int64), cache=True)
def overview_segments_kernel(
    state_codes: np.ndarray, state_count: int,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """Return [start, end, code, count] runs and classified sample statistics."""
    size = state_codes.shape[0]
    boundaries = np.empty((size, 4), dtype=np.int64)
    counts = np.zeros(state_count, dtype=np.int64)
    if size == 0:
        return boundaries, counts, 0, 0, 0
    segment_count = 0
    switches = 0
    classified = 0
    start = 0
    previous = -1
    for index in range(size):
        code = state_codes[index]
        if code < 0 or code >= state_count:
            code = -1
        if code >= 0:
            counts[code] += 1
            classified += 1
        if index > 0 and code != previous:
            boundaries[segment_count, 0] = start
            boundaries[segment_count, 1] = index - 1
            boundaries[segment_count, 2] = previous
            boundaries[segment_count, 3] = index - start
            segment_count += 1
            start = index
            if previous >= 0 and code >= 0:
                switches += 1
        previous = code
    boundaries[segment_count, 0] = start
    boundaries[segment_count, 1] = size - 1
    boundaries[segment_count, 2] = previous
    boundaries[segment_count, 3] = size - start
    segment_count += 1
    return boundaries[:segment_count], counts, switches, classified, size - classified


# Eager signature compilation is performed during module import at startup.
overview_segments_kernel.disable_compile()
_WARMED = False


def warm_result_overview_kernel() -> dict[str, Any]:
    global _WARMED
    overview_segments_kernel(np.array([0, -1, 1], dtype=np.int64), np.int64(2))
    overview_segments_kernel(np.empty(0, dtype=np.int64), np.int64(0))
    _WARMED = (
        bool(overview_segments_kernel.nopython_signatures)
        and len(overview_segments_kernel.signatures) == len(overview_segments_kernel.nopython_signatures)
        and overview_segments_kernel._can_compile is False
    )
    if not _WARMED:
        raise RuntimeError("Historical regime overview NJIT warmup failed")
    return {
        "complete": True,
        "python_fallback": 0,
        "request_time_compilation": 0,
        "signature_count": len(overview_segments_kernel.signatures),
    }


def build_result_overview(
    *,
    run_kind: str,
    run_id: str,
    definition: Mapping[str, Any],
    series: Sequence[Mapping[str, Any]],
    definition_hash: str,
    graph_hash: str,
    mode: str,
    as_of: str | None,
    data_snapshots: Mapping[str, Any] | None = None,
    definition_id: str | None = None,
    revision: int | None = None,
    frequency: str | None = None,
    calendar: str | None = None,
    series_endpoint: str,
    series_artifact: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if not _WARMED:
        raise RuntimeError("Historical regime overview NJIT runtime is not ready")
    result = result or {}
    states = copy.deepcopy(list(definition.get("states") or []))
    lookup = {state["id"]: index for index, state in enumerate(states)}
    codes = np.fromiter(
        (lookup.get(row.get("state_id"), -1) for row in series),
        dtype=np.int64,
        count=len(series),
    )
    boundaries, counts, switch_count, classified, unknown = overview_segments_kernel(codes, np.int64(len(states)))
    segments: list[dict[str, Any]] = []
    unknown_intervals: list[dict[str, Any]] = []
    for start, end, code, count in boundaries:
        start_index, end_index, state_index = int(start), int(end), int(code)
        first, last = series[start_index], series[end_index]
        state = states[state_index] if state_index >= 0 else None
        item = {
            "id": f"{run_id}:{start_index}:{end_index}",
            "state_id": state["id"] if state else "unclassified",
            "label": state["label"] if state else "未分类",
            "start_date": first.get("observation_date"),
            "end_date": last.get("observation_date"),
            "start_index": start_index,
            "end_index": end_index,
            "observations": int(count),
            "confirmed_at": first.get("recognized_at"),
            "effective_start": first.get("effective_date"),
            "reasons": copy.deepcopy(first.get("reasons") or []),
        }
        if state is None:
            item["reason"] = "unknown"
            unknown_intervals.append(item)
        else:
            segments.append(item)
    # Counts are returned by the NJIT kernel; no dataframe groupby is needed.
    state_counts = {state["id"]: int(counts[index]) for index, state in enumerate(states)}
    date_range = {
        "start": series[0].get("observation_date") if series else None,
        "end": series[-1].get("observation_date") if series else None,
    }
    display_source_id = result.get("display_source")
    primary_target = next(
        (target for target in definition.get("evaluation_targets", [])
         if target.get("id") == display_source_id),
        None,
    )
    display_node = next(
        (node for node in definition.get("graph", {}).get("nodes", [])
         if node.get("id") == display_source_id),
        None,
    )
    source = (primary_target or {}).get("source") or (display_node or {}).get("parameters") or {}
    label = (primary_target or {}).get("name") or (display_node or {}).get("label") or display_source_id or "识别主对照走势"
    return {
        "schema_version": "2.0",
        "result_kind": str(result.get("result_kind") or "regime_states"),
        "manual_events": copy.deepcopy(result.get("manual_events") or []),
        "manual_event_summary": copy.deepcopy(result.get("manual_event_summary") or {}),
        "temporal_capability": copy.deepcopy(result.get("temporal_capability")),
        "run_kind": run_kind,
        "run_id": run_id,
        "definition_id": definition_id,
        "revision": revision,
        "definition_revision": revision,
        "definition_hash": definition_hash,
        "graph_hash": graph_hash,
        "mode": mode,
        "as_of": as_of,
        "data_snapshots": copy.deepcopy(dict(data_snapshots or {})),
        "frequency": frequency,
        "calendar": calendar,
        "time_basis": "observation",
        "date_range": date_range,
        "complete": True,
        "created_at": created_at,
        "evaluation_results": copy.deepcopy(result.get("evaluation_results") or {}),
        "numeric_channels": copy.deepcopy({key: value for key, value in (definition.get("graph", {}).get("channel_metadata") or {}).items()
            if key not in {"state", "probabilities", "confidence", "recognition_index", "effective_index", "reason_code"}}),
        "interval_convention": "inclusive_observation_indices",
        "states": states,
        "unknown_state": {"id": "unclassified", "label": "未分类", "color": "#94a3b8", "reason": "原因未记录，可能为预热、缺失或规则未命中。"},
        "segments": segments,
        "unknown_intervals": unknown_intervals,
        "summary": {
            "total": len(series),
            "classified": int(classified),
            "unknown": int(unknown),
            "state_counts": state_counts,
            "switch_count": int(switch_count),
            "denominator": "all_observations",
            "switch_count_basis": "adjacent_classified_observations",
        },
        "primary_series": {
            "source_kind": "final_series",
            "run_id": run_id,
            "node_id": None,
            "port": "state",
            "value_column": "value",
            "date_column": "observation_date",
            "unit": source.get("unit"),
            "price_basis": source.get("price_basis"),
            "display_source_id": display_source_id,
            "endpoint": series_endpoint,
            "response_field": "items" if run_kind == "preview" else "series",
            "label": label,
            "date_range": copy.deepcopy(date_range),
            "total": len(series),
            "artifact": copy.deepcopy(series_artifact),
        },
        "capabilities": {
            "observation": {"available": True},
            "effective": {
                "available": False,
                "reason": "未提供完整生效时间轴，当前色带按观测日期展示。",
            },
            "probabilities": {"available": "probabilities" in (definition.get("graph", {}).get("outputs") or {})},
            "confidence": {"available": "confidence" in (definition.get("graph", {}).get("outputs") or {})},
            "evidence": {"available": bool(series)},
        },
    }
