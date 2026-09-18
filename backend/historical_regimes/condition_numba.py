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


@njit(types.Tuple((I, F, F))(RI, RI, RF, int64, int64), cache=True)
def continuous_state_kernel(candidates, initial, observed, confirmation, state_count):
    """Hold a modeled state between confirmed candidates; never invent missing prices.

    Evidence: 0 initial estimate, 1 confirmed candidate, 2 held state, 3 pending switch.
    -1 in candidates means no new proposal, not missing market observations.
    """
    size = candidates.size
    if initial.size != size or observed.size != size or not 1 <= confirmation <= 252 or not 2 <= state_count <= 12:
        raise ValueError("INVALID_CONTINUOUS_STATE_CONTRACT")
    states = np.empty(size, dtype=np.int64)
    evidence = np.empty(size)
    pending = np.zeros(size)
    active = proposed = -1
    count = 0
    confirmed = False
    for t in range(size):
        if not np.isfinite(observed[t]) or observed[t] <= 0:
            raise ValueError("CONTINUOUS_STATE_OBSERVATION_MISSING")
        candidate = candidates[t]
        if candidate < -1 or candidate >= state_count or initial[t] < -1 or initial[t] >= state_count:
            raise ValueError("CONTINUOUS_STATE_CODE_INVALID")
        if t == 0:
            if initial[t] < 0:
                raise ValueError("CONTINUOUS_STATE_INITIAL_REQUIRED")
            active = initial[t]
        basis = 2.0 if confirmed else 0.0
        if candidate < 0:
            proposed, count = -1, 0
        elif candidate == active:
            proposed, count = -1, 0
            confirmed, basis = True, 1.0
        else:
            count = count + 1 if candidate == proposed else 1
            proposed = candidate
            if count >= confirmation:
                active = proposed
                proposed, count = -1, 0
                confirmed, basis = True, 1.0
            else:
                basis = 3.0
        states[t], evidence[t], pending[t] = active, basis, count
    return states, evidence, pending


CONDITION_KERNELS = {
    "continuous_state": continuous_state_kernel,
    "condition_compare": condition_compare_kernel,
    "condition_valid": condition_valid_kernel,
    "condition_logic": condition_logic_kernel,
    "select_state": select_state_kernel,
}
for dispatcher in CONDITION_KERNELS.values():
    dispatcher.disable_compile()
