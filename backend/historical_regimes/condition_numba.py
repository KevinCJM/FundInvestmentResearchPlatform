"""Nominal conditions: 1 true, 0 false, -1 unavailable.

Comparisons reuse Indicator Center mathematics. Empty optional arrays select
scalar parameters; they never stand for unavailable observations.
"""
import numpy as np
from numba import float64, int64, njit, types
from cal_indicators.typed_numba_kernels import _comparison_value

F = float64[::1]
I = int64[::1]
RF = types.Array(float64, 1, "A", readonly=True)
RI = types.Array(int64, 1, "A", readonly=True)
COMPARISON_OPCODES = {"eq": 40, "ne": 41, "lt": 42, "le": 43, "gt": 44, "ge": 45}


@njit(I(RF, RF, float64, int64), cache=True)
def condition_compare_kernel(values, bounds, threshold, opcode):
    if bounds.size != 0 and bounds.size != values.size:
        raise ValueError("Condition bounds must have the same length as values.")
    if opcode < 40 or opcode > 45:
        raise ValueError("Invalid comparison opcode.")
    result = np.full(values.size, -1, dtype=np.int64)
    for t in range(values.size):
        bound = threshold if bounds.size == 0 else bounds[t]
        if np.isfinite(values[t]) and np.isfinite(bound):
            result[t] = int(_comparison_value(opcode, values[t], bound))
    return result


@njit(I(RF), cache=True)
def condition_valid_kernel(values):
    result = np.empty(values.size, dtype=np.int64)
    for t in range(values.size):
        result[t] = 1 if np.isfinite(values[t]) else 0
    return result


@njit(I(RI, RI, int64), cache=True)
def condition_logic_kernel(left, right, opcode):
    """0 AND, 1 OR, 2 NOT; binary conditions strictly propagate missingness."""
    if opcode < 0 or opcode > 2 or (opcode != 2 and right.size != left.size):
        raise ValueError("Invalid condition operation or axis.")
    result = np.full(left.size, -1, dtype=np.int64)
    for t in range(left.size):
        a = left[t]
        if a != 0 and a != 1:
            continue
        if opcode == 2:
            result[t] = 1 - a
            continue
        b = right[t]
        if b != 0 and b != 1:
            continue
        result[t] = int((a == 1 and b == 1) if opcode == 0 else (a == 1 or b == 1))
    return result


@njit(I(RI, RI, RI, int64, int64), cache=True)
def select_state_kernel(condition, when_true, when_false, true_code, false_code):
    """Select only one branch; unavailable conditions never become neutral."""
    n = condition.size
    if (when_true.size != 0 and when_true.size != n) or (when_false.size != 0 and when_false.size != n):
        raise ValueError("State branches must have the same length as conditions.")
    result = np.full(n, -1, dtype=np.int64)
    for t in range(n):
        if condition[t] == 1:
            result[t] = true_code if when_true.size == 0 else when_true[t]
        elif condition[t] == 0:
            result[t] = false_code if when_false.size == 0 else when_false[t]
    return result


CONDITION_KERNELS = {
    "condition_compare": condition_compare_kernel,
    "condition_valid": condition_valid_kernel,
    "condition_logic": condition_logic_kernel,
    "select_state": select_state_kernel,
}
for dispatcher in CONDITION_KERNELS.values():
    dispatcher.disable_compile()
