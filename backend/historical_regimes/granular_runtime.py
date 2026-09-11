"""Dispatch domain primitives over fixed arrays; no data access or mathematics in Python."""
from __future__ import annotations

import numpy as np

from .condition_numba import (
    COMPARISON_OPCODES, condition_compare_kernel, condition_valid_kernel,
    condition_logic_kernel, select_state_kernel,
)
from .peak_trough_numba import ps_filter_pivots_kernel, peak_trough_sideways_kernel
from .segment_numba import phase_direction_kernel, boundary_line_kernel
from .v2_numba import state_probabilities_kernel, state_confidence_kernel

EMPTY_FLOAT = np.empty(0, dtype=np.float64)
EMPTY_INT = np.empty(0, dtype=np.int64)


def execute_granular_node(node_type, parameters, inputs, state_count):
    def floating(name, default=EMPTY_FLOAT):
        return np.ascontiguousarray(inputs.get(name, default), dtype=np.float64)

    def integer(name, default=EMPTY_INT):
        return np.ascontiguousarray(inputs.get(name, default), dtype=np.int64)

    if node_type == "condition.compare":
        result = condition_compare_kernel(floating("value"), floating("bound"),
            np.float64(parameters.get("threshold", 0.0)), np.int64(COMPARISON_OPCODES[parameters.get("operator", "ge")]))
        return {"condition": result}
    if node_type == "condition.valid":
        return {"condition": condition_valid_kernel(floating("value"))}
    if node_type in {"condition.all", "condition.any", "condition.not"}:
        unary = node_type == "condition.not"
        return {"condition": condition_logic_kernel(integer("condition" if unary else "left"), integer("right"),
            np.int64({"condition.all": 0, "condition.any": 1, "condition.not": 2}[node_type]))}
    if node_type == "state.select":
        return {"state": select_state_kernel(integer("condition"), integer("when_true"), integer("when_false"),
            np.int64(parameters.get("true_code", 0)), np.int64(parameters.get("false_code", 1)))}
    if node_type == "state.encode":
        states = integer("state")
        return {"probabilities": state_probabilities_kernel(states, np.int64(state_count)),
                "confidence": state_confidence_kernel(states)}
    if node_type == "pivot.ps_filter":
        pivots, prices = ps_filter_pivots_kernel(floating("value"), floating("pivot"),
            np.int64(parameters.get("min_phase", 4)), np.int64(parameters.get("min_cycle", 16)),
            np.float64(parameters.get("amplitude_exception", .2)))
        return {"pivot": pivots, "marker": pivots, "pivot_price": prices}
    if node_type == "segment.phase_direction":
        return {"phase": phase_direction_kernel(floating("pivot"), integer("start"), integer("end"))}
    if node_type == "segment.boundary_line":
        return {"value": boundary_line_kernel(floating("value"), integer("start"), integer("end"))}
    if node_type == "post.peak_sideways":
        result = peak_trough_sideways_kernel(floating("value"), integer("phase"), integer("start"), integer("end"),
            np.int64(parameters.get("sideways_enabled", False)), np.int64(2 if state_count == 3 else 1),
            np.int64(1 if state_count == 3 else -1), np.float64(parameters.get("small_swing_threshold", .03)),
            np.float64(parameters.get("sideways_max_range", .06)), np.float64(parameters.get("sideways_max_efficiency", .25)),
            np.int64(parameters.get("sideways_min_duration", 20)))
        return dict(zip(("state", "sideways_range", "sideways_efficiency", "sideways_start_index", "sideways_end_index", "sideways_swing_count"), result))
    raise ValueError(f"Unregistered granular node: {node_type}")
