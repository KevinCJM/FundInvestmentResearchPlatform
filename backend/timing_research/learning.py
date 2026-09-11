"""Frozen, train-only state/action learning for the ETF adaptation.

These are independent numerical stages, not replicas of the WF stock learners:
explicit conditions -> states -> mature trade utility -> frozen action -> signal.
All inputs are read-only arbitrary-stride views; no input is copied or mutated.
The grouped Welford accumulator fuses domain-specific maturity/state selection
with statistics so no per-state trade arrays or masks need to be materialized.
"""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

from backend.compute_policy import validate_execution_audit
from backend.timing_research.numeric import REASON_STOP, TRADE_COLUMNS

I = types.Array(int64, 1, "A", readonly=True)
IM = types.Array(int64, 2, "A", readonly=True)
FM = types.Array(float64, 2, "A", readonly=True)
IO = int64[::1]
FO = float64[:, ::1]

LEARNING_VERSION = "etf-state-action-njit/1.0.0"
SCORE_COLUMNS = ("count", "mean", "win_rate", "utility", "stop_rate")
MAX_STATE_CONDITIONS = 3
MAX_STATES = 12  # Three condition bits, four quarters or twelve calendar months.
TRADE_WIDTH = len(TRADE_COLUMNS)


@njit(IO(IM), cache=True, nogil=True)
def encode_states_kernel(conditions):
    """Encode condition k as bit 2**k; any unknown keeps the state unknown.

    Zero conditions explicitly define one unconditional state (0). This is not
    a fallback for an unknown condition or an unsupported training state.
    """
    if conditions.shape[0] > MAX_STATE_CONDITIONS:
        raise ValueError("At most three explicit conditions can define a state.")
    output = np.empty(conditions.shape[1], dtype=np.int64)
    for t in range(conditions.shape[1]):
        code = 0
        unknown = False
        for k in range(conditions.shape[0]):
            condition = conditions[k, t]
            if condition < -1 or condition > 1:
                raise ValueError("Conditions must use -1, 0 or 1.")
            if condition == -1:
                unknown = True
            elif condition == 1:
                code |= 1 << k
        output[t] = -1 if unknown else code
    return output


@njit(FO(FM, int64, I, int64, int64, int64, int64, float64, float64), cache=True, nogil=True)
def score_trades_kernel(trades, count, states, start, split, n_states,
                        min_trades, confidence, risk_penalty):
    """Score mature net trade outcomes in each signal-date state.

    signal >= start, entry < split and exit < split are required. An exit at
    split-1 close is available for a model frozen after that close, before the
    split open. Non-finite required columns and open trades are not labels.
    MAE/MFE are deliberately not required: daily stop fills may leave them NaN.
    Utility is mean - confidence * sample_std / sqrt(n) - penalty * stop_rate.
    n=1 cannot estimate sample uncertainty, hence has unavailable utility.
    """
    if (trades.shape[1] != TRADE_WIDTH or count < 0 or count > trades.shape[0]
            or start < 0 or split < start or split > states.size
            or n_states < 1 or n_states > MAX_STATES or min_trades < 1):
        raise ValueError("Invalid trade, state, training interval or sample contract.")
    if (not np.isfinite(confidence) or confidence < 0.0
            or not np.isfinite(risk_penalty) or risk_penalty < 0.0):
        raise ValueError("Confidence and risk penalty must be finite and nonnegative.")
    for t in range(start, split):
        if states[t] < -1 or states[t] >= n_states:
            raise ValueError("State code is outside the declared training states.")

    output = np.full((n_states, len(SCORE_COLUMNS)), np.nan)
    m2 = np.zeros(n_states)
    wins = np.zeros(n_states, dtype=np.int64)
    stops = np.zeros(n_states, dtype=np.int64)
    for state in range(n_states):
        output[state, 0] = 0.0
    for row in range(count):
        valid = True
        for column in range(8):
            if not np.isfinite(trades[row, column]):
                valid = False
                break
        if not valid:
            continue
        signal_value, entry_value, exit_value = trades[row, 0], trades[row, 1], trades[row, 2]
        if signal_value < start or entry_value >= split or exit_value >= split or exit_value < 0:
            continue
        signal, entry, exit_index = int(signal_value), int(entry_value), int(exit_value)
        reason = int(trades[row, 7])
        if (signal != signal_value or entry != entry_value or exit_index != exit_value
                or signal < 0 or signal >= states.size or entry <= signal
                or exit_index <= entry or exit_index >= states.size
                or trades[row, 3] <= 0.0 or trades[row, 4] <= 0.0
                or trades[row, 6] != exit_index - entry
                or trades[row, 7] != reason or reason < 1 or reason > 4):
            raise ValueError("Mature trade indices, prices, holding period or reason are invalid.")
        state = states[signal]
        if state == -1:
            continue
        outcome = trades[row, 5]
        size = output[state, 0] + 1.0
        previous_mean = output[state, 1] if size > 1.0 else 0.0
        delta = outcome - previous_mean
        mean = previous_mean + delta / size
        m2[state] += delta * (outcome - mean)
        output[state, 0], output[state, 1] = size, mean
        wins[state] += int(outcome > 0.0)
        stops[state] += int(reason == REASON_STOP)
    for state in range(n_states):
        size = output[state, 0]
        if size > 0.0:
            output[state, 2] = wins[state] / size
            output[state, 4] = stops[state] / size
            if size >= min_trades and size > 1.0:
                uncertainty = np.sqrt(max(0.0, m2[state]) / (size - 1.0) / size)
                utility = output[state, 1] - confidence * uncertainty - risk_penalty * output[state, 4]
                if np.isfinite(utility):
                    output[state, 3] = utility
    return output


@njit(IO(FM, float64, int64), cache=True, nogil=True)
def choose_actions_kernel(scores, min_utility, cash_index):
    """Select a supported action strictly above the floor; ties keep first.

    cash_index=-1 means no action; a designated cash column is never a risky
    competitor. When present it must be represented by an all-zero signal lane.
    No other state's score or global average substitutes for missing support.
    """
    if (scores.shape[0] < 1 or scores.shape[0] > MAX_STATES
            or cash_index < -1 or cash_index >= scores.shape[1]
            or not np.isfinite(min_utility)):
        raise ValueError("Invalid utility matrix, action index or selection floor.")
    selected = np.full(scores.shape[0], cash_index, dtype=np.int64)
    for state in range(scores.shape[0]):
        best = min_utility
        for action in range(scores.shape[1]):
            if action == cash_index:
                continue
            utility = scores[state, action]
            if np.isfinite(utility) and utility > best:
                best, selected[state] = utility, action
    return selected


@njit(IO(IM, I, I, int64), cache=True, nogil=True)
def route_actions_kernel(signals, states, selected, split):
    """Apply a frozen state/action map, starting with split-1 close signals.

    The caller freezes training at split-1 close, so that day's current features
    may trigger the first OOS open. Earlier rows remain unknown (-1). Unknown
    states remain unknown; selected=-1 means explicitly abstain (0), not missing.
    This routes candidate signals only; actual holding/exit rules are separate.
    """
    if (signals.shape[1] != states.size or selected.size < 1 or selected.size > MAX_STATES
            or split < 0 or split > states.size):
        raise ValueError("Invalid signal/state axis, selected states or freeze split.")
    for state in range(selected.size):
        if selected[state] < -1 or selected[state] >= signals.shape[0]:
            raise ValueError("Selected action is outside the signal lanes.")
    output = np.full(states.size, -1, dtype=np.int64)
    for t in range(states.size):
        state = states[t]
        if state < -1 or state >= selected.size:
            raise ValueError("State code is outside the frozen policy.")
        for action in range(signals.shape[0]):
            if signals[action, t] < -1 or signals[action, t] > 1:
                raise ValueError("Action signals must use -1, 0 or 1.")
        if t < split - 1 or state == -1:
            continue
        action = selected[state]
        output[t] = 0 if action == -1 else signals[action, t]
    return output


@njit(IO(I, I, I, int64, int64, int64), cache=True, nogil=True)
def priority_quota_kernel(core, supplement, month_ids, max_core_before_supp,
                          max_supp_per_month, cooldown):
    """Emit core first; supplement only under causal month-to-date quotas.

    Monthly counters track emitted candidates, not later executed trades, and
    reset when the ordered month ID changes. Supplement cooldown spans months.
    A known core signal wins even if supplement is unknown; otherwise any
    unknown input stays unknown and does not increment an emitted count.
    The counters and last-emission index form one inseparable streaming state.
    """
    if (core.size != supplement.size or core.size != month_ids.size
            or max_core_before_supp < 0 or max_supp_per_month < 0 or cooldown < 0):
        raise ValueError("Invalid priority/quota axis, quota or cooldown.")
    output = np.zeros(core.size, dtype=np.int64)
    current_month = -1
    core_count = 0
    supplement_count = 0
    last_supplement = -1
    for t in range(core.size):
        month = month_ids[t]
        if month < 0 or month < current_month:
            raise ValueError("Month IDs must be nonnegative and ordered.")
        if (core[t] < -1 or core[t] > 1 or supplement[t] < -1 or supplement[t] > 1):
            raise ValueError("Priority/quota conditions must use -1, 0 or 1.")
        if month != current_month:
            current_month, core_count, supplement_count = month, 0, 0
        if core[t] == 1:
            output[t] = 1
            core_count += 1
        elif core[t] == -1 or supplement[t] == -1:
            output[t] = -1
        elif (supplement[t] == 1 and core_count < max_core_before_supp
              and supplement_count < max_supp_per_month
              and (last_supplement < 0 or t - last_supplement > cooldown)):
            output[t] = 1
            supplement_count += 1
            last_supplement = t
    return output


KERNELS = {"encode_states": encode_states_kernel, "score_trades": score_trades_kernel,
           "choose_actions": choose_actions_kernel, "route_actions": route_actions_kernel,
           "priority_quota": priority_quota_kernel}
for _kernel in KERNELS.values():
    _kernel.disable_compile()


def learning_execution_audit():
    """Fail closed unless every production learning dispatcher is frozen NJIT."""
    for kernel in KERNELS.values():
        if (not kernel.nopython_signatures or len(kernel.signatures) != len(kernel.nopython_signatures)
                or kernel._can_compile):
            raise RuntimeError("择时训练内核尚未完成固定签名预热。")
    return {"complete": True, **validate_execution_audit({
        "execution_backend": "numba_njit_fixed_signature", "engine_version": LEARNING_VERSION,
        "nopython": True, "python_fallback": 0, "python_operator_calls": 0,
        "request_time_compilation": 0,
        "kernel_signatures": {name: [str(sig) for sig in kernel.signatures] for name, kernel in KERNELS.items()},
    })}


def warm_learning_kernels():
    """Startup verifies eager fixed-signature preparation; never compiles on use."""
    return learning_execution_audit()
