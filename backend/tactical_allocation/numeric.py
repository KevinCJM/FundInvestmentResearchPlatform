"""TAA research primitives delegating every portfolio path to one current core.

Inputs are immutable arbitrary-stride arrays. Candidate evaluation reuses those
arrays and allocates only candidate outputs; no simulation engine is duplicated.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
from numba import float64, int64, njit, types, uint8

from backend.historical_regimes.taa import (
    _performance_kernel,
    _taa_output_validation_kernel,
    _taa_path_kernel,
    _weight_vector_validation_kernel,
    taa_numba_execution_audit,
    warm_taa_numba_kernels,
)

F = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
I = types.Array(int64, 1, "A", readonly=True)
IM = types.Array(int64, 2, "A", readonly=True)
U = types.Array(uint8, 1, "A", readonly=True)
UM = types.Array(uint8, 2, "A", readonly=True)
FO, MO, UO = float64[::1], float64[:, ::1], uint8[::1]
DEFAULT_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
_WARMED_PID: int | None = None
_WARMING_PID: int | None = None
METRIC_NAMES = (
    "total_return", "baseline_return", "excess_return", "annual_volatility",
    "max_drawdown", "tracking_error", "information_ratio", "turnover",
    "average_turnover", "max_turnover", "cost", "baseline_turnover",
    "baseline_cost", "score", "observations", "total_return_difference",
)


@njit(int64(M, M, U, F, M, F, F, F), cache=True, nogil=True)
def input_status_kernel(returns, probabilities, use_signal, base, tilts, lower, upper, caps):
    """Validate shared axes, finite returns and probability/weight budgets."""
    n, a = returns.shape
    s = tilts.shape[0]
    if (a == 0 or n == 0 or s == 0 or base.size != a or tilts.shape[1] != a
            or probabilities.shape != (n, s) or use_signal.size != n
            or lower.size != a or upper.size != a or caps.size != a):
        return 1
    _, status = _weight_vector_validation_kernel(base, 1.0, 1e-6, 0.0, 1.0)
    if status != 0:
        return 2
    for j in range(a):
        if (not np.isfinite(lower[j]) or not np.isfinite(upper[j])
                or not np.isfinite(caps[j]) or lower[j] < 0.0 or upper[j] > 1.0
                or lower[j] > base[j] + 1e-10 or upper[j] < base[j] - 1e-10
                or caps[j] < 0.0 or caps[j] > 1.0):
            return 3
    for k in range(s):
        _, status = _weight_vector_validation_kernel(tilts[k], 0.0, 1e-6, -1.0, 1.0)
        if status != 0:
            return 4
    for t in range(n):
        for j in range(a):
            if not np.isfinite(returns[t, j]) or returns[t, j] <= -1.0:
                return 5
        if use_signal[t] > 1:
            return 6
        total = 0.0
        for k in range(s):
            p = probabilities[t, k]
            if not np.isfinite(p) or p < 0.0 or p > 1.0:
                return 6
            total += p
        if use_signal[t] == 1 and abs(total - 1.0) > 1e-6:
            return 6
    return 0


@njit(types.Tuple((MO, FO))(F, M, F, F, F, float64, UM, F, F), cache=True, nogil=True)
def constrained_tilts_kernel(base, tilts, lower, upper, caps, strength, groups, group_lower, group_upper):
    """Preserve each zero-sum direction with one joint feasible state scale."""
    result = np.empty(tilts.shape, dtype=np.float64)
    scales = np.ones(tilts.shape[0], dtype=np.float64)
    if groups.shape[1] != base.size or groups.shape[0] != group_lower.size or group_lower.size != group_upper.size:
        raise ValueError("Group constraint axes do not match assets.")
    for s in range(tilts.shape[0]):
        scale = 1.0
        for a in range(tilts.shape[1]):
            delta = strength * tilts[s, a]
            if abs(delta) > 1e-14:
                scale = min(scale, caps[a] / abs(delta))
                if delta > 0:
                    scale = min(scale, (upper[a] - base[a]) / delta)
                else:
                    scale = min(scale, (lower[a] - base[a]) / delta)
        for g in range(groups.shape[0]):
            baseline_group, delta_group = 0.0, 0.0
            for a in range(base.size):
                if groups[g, a] > 1:
                    raise ValueError("Group membership must be zero or one.")
                if groups[g, a] == 1:
                    baseline_group += base[a]
                    delta_group += strength * tilts[s, a]
            if (not np.isfinite(group_lower[g]) or not np.isfinite(group_upper[g])
                    or group_lower[g] < 0.0 or group_upper[g] > 1.0
                    or group_lower[g] > baseline_group + 1e-10
                    or group_upper[g] < baseline_group - 1e-10):
                raise ValueError("SAA baseline violates the requested group constraints.")
            if delta_group > 1e-14:
                scale = min(scale, (group_upper[g] - baseline_group) / delta_group)
            elif delta_group < -1e-14:
                scale = min(scale, (group_lower[g] - baseline_group) / delta_group)
        scale = max(0.0, scale)
        scales[s] = scale
        for a in range(tilts.shape[1]):
            result[s, a] = strength * scale * tilts[s, a]
    return result, scales


@njit(FO(M, int64, float64, float64, int64), cache=True, nogil=True)
def path_metrics_kernel(path, asset_count, periods_per_year, penalty, objective):
    """Net performance and active risk on one independently funded interval."""
    o = asset_count * 2
    n = path.shape[0]
    if n < 1 or periods_per_year <= 0.0:
        raise ValueError("Empty evaluation interval.")
    taa = _performance_kernel(path[:, o + 7], path[:, o + 9], periods_per_year)
    baseline = _performance_kernel(path[:, o + 5], path[:, o + 8], periods_per_year)
    mean = 0.0
    turnover = 0.0
    maximum_turnover = 0.0
    cost = 0.0
    baseline_turnover = 0.0
    baseline_cost = 0.0
    for t in range(n):
        mean += path[t, o + 7] - path[t, o + 5]
        turnover += path[t, o + 1]
        maximum_turnover = max(maximum_turnover, path[t, o + 1])
        cost += path[t, o + 3]
        baseline_turnover += path[t, o + 10]
        baseline_cost += path[t, o + 12]
    mean /= n
    variance = 0.0
    for t in range(n):
        difference = path[t, o + 7] - path[t, o + 5] - mean
        variance += difference * difference
    variance = variance / (n - 1) * periods_per_year if n > 1 else 0.0
    tracking_error = np.sqrt(variance)
    relative = (1.0 + taa[0]) / (1.0 + baseline[0]) - 1.0
    score = mean * periods_per_year - penalty * variance
    if objective == 1:
        score = relative
    elif objective == 2:
        score = taa[4]
    result = np.empty(len(METRIC_NAMES), dtype=np.float64)
    result[0], result[1], result[2], result[3], result[4] = taa[0], baseline[0], relative, taa[2], taa[4]
    result[5] = tracking_error
    result[6] = mean * periods_per_year / tracking_error if tracking_error > 1e-14 else np.nan
    result[7] = turnover
    result[8] = turnover / n
    result[9] = maximum_turnover
    result[10] = cost
    result[11] = baseline_turnover
    result[12] = baseline_cost
    result[13] = score
    result[14] = n
    result[15] = taa[0] - baseline[0]
    return result


@njit(int64(M, U), cache=True, nogil=True)
def select_candidate_kernel(metrics, feasible):
    """Select only from training scores; deterministic ties prefer zero tilt."""
    winner = 0
    score = metrics[0, 13]
    for k in range(1, metrics.shape[0]):
        if feasible[k] == 1 and metrics[k, 13] > score + 1e-12:
            winner, score = k, metrics[k, 13]
    return winner


@njit(uint8(F, float64, float64), cache=True, nogil=True)
def candidate_feasibility_kernel(metrics, maximum_tracking_error, maximum_turnover):
    return np.uint8(metrics[5] <= maximum_tracking_error + 1e-12
                    and metrics[9] <= maximum_turnover + 1e-12)


@njit(types.Array(int64, 2, "C")(IM), cache=True, nogil=True)
def returns_availability_kernel(nav_available):
    """A period return requires both endpoint NAVs to have become available."""
    if nav_available.shape[0] < 1:
        raise ValueError("NAV availability requires at least one date.")
    result = np.empty((nav_available.shape[0] - 1, nav_available.shape[1]), dtype=np.int64)
    for t in range(result.shape[0]):
        for a in range(result.shape[1]):
            first, second = nav_available[t, a], nav_available[t + 1, a]
            result[t, a] = -1 if first < 0 or second < 0 else max(first, second)
    return result


@njit(types.Tuple((int64, int64))(IM, int64, int64, int64), cache=True, nogil=True)
def knowledge_window_status_kernel(available, cutoff, start, end):
    """Count immature and unknown outcome cells before training can rank them."""
    if start < 0 or end < start or end > available.shape[0] or cutoff < 0:
        raise ValueError("Invalid outcome knowledge cutoff or interval.")
    future, unknown = 0, 0
    for t in range(start, end):
        for a in range(available.shape[1]):
            if available[t, a] < 0:
                unknown += 1
            elif available[t, a] > cutoff:
                future += 1
    return future, unknown


MOMENTUM_ALGORITHM_VERSION = "available-window-relative-momentum/2.0.0"
IO = types.Array(int64, 2, "C")


@njit(IO(IM, I, I, int64, int64, int64), cache=True, nogil=True)
def momentum_windows_kernel(available, starts, ends, lookback, as_of, max_age):
    """Latest contiguous, fully known window at each cutoff; never skip rows.

    A rolling maximum checks endpoint publication times. A ready-time min-heap
    releases completed windows in O(n log n), including out-of-order releases.
    Output columns: start, exclusive end, available day, calendar age, status.
    Status: 0 warmup, 1 no known window, 2 expired, 3 available.
    """
    n, a = available.shape
    if n == 0 or a == 0 or starts.size != n or ends.size != n or lookback < 1 or lookback > 5000 or max_age < 0:
        raise ValueError("Invalid momentum window axes or limits.")
    for t in range(n):
        if starts[t] < 0 or ends[t] <= starts[t] or (t > 0 and starts[t] < ends[t - 1]):
            raise ValueError("Momentum requires ordered non-overlapping dated returns.")
    if as_of < ends[-1]:
        raise ValueError("Current cutoff precedes an observed return.")
    latest_publication = np.full(n, -1, dtype=np.int64)
    unknown_rows = np.zeros(n, dtype=np.uint8)
    for t in range(n):
        for j in range(a):
            latest_publication[t] = max(latest_publication[t], available[t, j])
            if available[t, j] < 0:
                unknown_rows[t] = 1
    output = np.full((n + 1, 5), -1, dtype=np.int64)
    queue = np.empty(lookback, dtype=np.int64)
    heap_days, heap_ends = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    head, size, unknown, heap_size, chosen, chosen_ready = 0, 0, 0, 0, -1, -1
    for t in range(n + 1):
        cutoff = starts[t] if t < n else as_of
        output[t, 4] = 0 if t < lookback else 1
        if t == 0:
            continue
        added = t - 1
        if t > lookback:
            expired = t - lookback - 1
            unknown -= unknown_rows[expired]
            while size > 0 and queue[head] <= expired:
                head = (head + 1) % lookback
                size -= 1
        unknown += unknown_rows[added]
        while size > 0 and latest_publication[queue[(head + size - 1) % lookback]] <= latest_publication[added]:
            size -= 1
        queue[(head + size) % lookback] = added
        size += 1
        if t >= lookback and unknown == 0:
            ready = max(latest_publication[queue[head]], ends[added])
            pos = heap_size
            heap_size += 1
            while pos > 0:
                parent = (pos - 1) // 2
                if heap_days[parent] <= ready:
                    break
                heap_days[pos], heap_ends[pos] = heap_days[parent], heap_ends[parent]
                pos = parent
            heap_days[pos], heap_ends[pos] = ready, t
        while heap_size > 0 and heap_days[0] <= cutoff:
            if heap_ends[0] > chosen:
                chosen, chosen_ready = heap_ends[0], heap_days[0]
            heap_size -= 1
            if heap_size > 0:
                replacement_day, replacement_end = heap_days[heap_size], heap_ends[heap_size]
                pos = 0
                while pos * 2 + 1 < heap_size:
                    child = pos * 2 + 1
                    if child + 1 < heap_size and heap_days[child + 1] < heap_days[child]:
                        child += 1
                    if heap_days[child] >= replacement_day:
                        break
                    heap_days[pos], heap_ends[pos] = heap_days[child], heap_ends[child]
                    pos = child
                heap_days[pos], heap_ends[pos] = replacement_day, replacement_end
        if chosen >= lookback:
            age = cutoff - ends[chosen - 1]
            output[t, 0], output[t, 1] = chosen - lookback, chosen
            output[t, 2], output[t, 3] = chosen_ready, age
            output[t, 4] = 2 if age > max_age else 3
    return output


_MOMENTUM_RESULT = types.Tuple((MO, UO, MO, FO, int64, FO, UO, int64, IO))


@njit(_MOMENTUM_RESULT(M, int64, float64, IM, I, I, int64, int64), cache=True, nogil=True)
def momentum_signals_kernel(returns, lookback, max_tilt, available, starts, ends, as_of, max_age):
    """Relative return strength of the latest mature contiguous window."""
    n, a = returns.shape
    if available.shape != returns.shape or not np.isfinite(max_tilt) or max_tilt < 0.0 or max_tilt > 1.0:
        raise ValueError("Invalid momentum returns, knowledge axes or tilt.")
    for t in range(n):
        for j in range(a):
            if not np.isfinite(returns[t, j]) or returns[t, j] <= -1.0:
                raise ValueError("Momentum returns must be finite and greater than -100%.")
    windows = momentum_windows_kernel(available, starts, ends, lookback, as_of, max_age)
    probabilities = np.zeros((n, a), dtype=np.float64)
    use_signal, knowledge = np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.uint8)
    tilts = np.zeros((a, a), dtype=np.float64)
    for s in range(a):
        for j in range(a):
            if a > 1:
                tilts[s, j] = max_tilt if j == s else -max_tilt / (a - 1)
    log_total = np.zeros(a, dtype=np.float64)
    current_probabilities = np.zeros(a, dtype=np.float64)
    momentum = np.full(a, np.nan, dtype=np.float64)
    current_use, current_known, cursor = 0, 0, 0
    # Chosen window ends can only advance. Reuse one rolling accumulator, with
    # no per-window materialization or copy of shared input arrays.
    for t in range(n + 1):
        chosen = windows[t, 1]
        if chosen < 0:
            continue
        while cursor < chosen:
            for j in range(a):
                log_total[j] += np.log1p(returns[cursor, j])
                if cursor >= lookback:
                    log_total[j] -= np.log1p(returns[cursor - lookback, j])
            cursor += 1
        highest, lowest = -np.inf, np.inf
        for j in range(a):
            value = np.expm1(log_total[j])
            if not np.isfinite(value):
                raise ValueError("Momentum accumulation overflowed.")
            if t == n:
                momentum[j] = value
            highest, lowest = max(highest, value), min(lowest, value)
        fresh = windows[t, 4] == 3
        valid = fresh and highest - lowest > 1e-12 and a > 1
        ties = 0
        if valid:
            for j in range(a):
                if abs(np.expm1(log_total[j]) - highest) <= 1e-12:
                    ties += 1
        for j in range(a):
            probability = 1.0 / ties if valid and abs(np.expm1(log_total[j]) - highest) <= 1e-12 else 0.0
            if t < n:
                probabilities[t, j] = probability
            else:
                current_probabilities[j] = probability
        if t < n:
            use_signal[t], knowledge[t] = np.uint8(valid), np.uint8(fresh)
        else:
            current_use, current_known = int(valid), int(fresh)
    return probabilities, use_signal, tilts, current_probabilities, current_use, momentum, knowledge, current_known, windows


@njit(types.Tuple((int64, int64, int64, int64))(U, U, int64), cache=True, nogil=True)
def signal_counts_kernel(active, known, split):
    if active.size != known.size or split < 0 or split > active.size:
        raise ValueError("Invalid signal summary axes.")
    train_active, validation_active, train_known, validation_known = 0, 0, 0, 0
    for t in range(active.size):
        if t < split:
            train_active += active[t]
            train_known += known[t]
        else:
            validation_active += active[t]
            validation_known += known[t]
    return train_active, validation_active, train_known, validation_known


@njit(FO(F, I, F), cache=True, nogil=True)
def compose_product_weights_kernel(class_weights, class_indices, within_weights):
    """Compose asset-class budgets with explicit within-class product weights."""
    if class_indices.size != within_weights.size or class_weights.size == 0:
        raise ValueError("Product membership axes do not match.")
    _, status = _weight_vector_validation_kernel(class_weights, 1.0, 1e-6, 0.0, 1.0)
    if status != 0:
        raise ValueError("Class weights must form a finite budget of one.")
    sums = np.zeros(class_weights.size, dtype=np.float64)
    result = np.empty(within_weights.size, dtype=np.float64)
    for p in range(within_weights.size):
        c = class_indices[p]
        if c < 0 or c >= class_weights.size:
            raise ValueError("Unknown product asset-class index.")
        if not np.isfinite(within_weights[p]) or within_weights[p] < 0.0 or within_weights[p] > 1.0:
            raise ValueError("Invalid within-class product weight.")
        sums[c] += within_weights[p]
        result[p] = class_weights[c] * within_weights[p]
    for c in range(class_weights.size):
        if abs(sums[c] - 1.0) > 1e-6:
            raise ValueError("Every class requires a complete within-class budget of one.")
    return result


@njit(types.Tuple((FO, FO, FO))(M, M, F), cache=True, nogil=True)
def scenario_contributions_kernel(returns, path, base):
    """Link each asset's period P&L to terminal wealth; costs remain explicit."""
    baseline = np.zeros(base.size, dtype=np.float64)
    target = np.zeros(base.size, dtype=np.float64)
    o = base.size * 2
    base_wealth, taa_wealth = 1.0, 1.0
    for t in range(returns.shape[0]):
        for a in range(base.size):
            baseline[a] += base_wealth * (1.0 - path[t, o + 11]) * base[a] * returns[t, a]
            target[a] += taa_wealth * (1.0 - path[t, o + 2]) * path[t, a] * returns[t, a]
        base_wealth, taa_wealth = path[t, o + 8], path[t, o + 9]
    excess = np.empty(base.size, dtype=np.float64)
    for a in range(base.size):
        excess[a] = target[a] - baseline[a]
    return baseline, target, excess


@njit(MO(F, F), cache=True, nogil=True)
def target_tilt_kernel(base, target):
    if base.size != target.size:
        raise ValueError("Target and baseline axes do not match.")
    _, status = _weight_vector_validation_kernel(target, 1.0, 1e-6, 0.0, 1.0)
    if status != 0:
        raise ValueError("Target weights must form a finite budget of one.")
    result = np.empty((1, base.size), dtype=np.float64)
    for a in range(base.size):
        result[0, a] = target[a] - base[a]
    return result


@njit(types.Tuple((FO, FO, float64, int64))(F, F, F), cache=True, nogil=True)
def recommendation_details_kernel(base, target, current):
    if base.size != target.size or current.size != base.size:
        raise ValueError("Current holding and target axes do not match.")
    _, status = _weight_vector_validation_kernel(current, 1.0, 1e-6, 0.0, 1.0)
    if status:
        raise ValueError("Current holdings must form a finite budget of one.")
    tilts, trades = np.empty(base.size), np.empty(base.size)
    turnover = 0.0
    deviation_norm = 0.0
    for a in range(base.size):
        tilts[a] = target[a] - base[a]
        trades[a] = target[a] - current[a]
        turnover += abs(trades[a]) * 0.5
        deviation_norm += abs(tilts[a])
    return tilts, trades, turnover, int(deviation_norm > 1e-10)


@njit(FO(F, I, int64), cache=True, nogil=True)
def aggregate_class_weights_kernel(weights, class_indices, class_count):
    if class_count < 1 or class_indices.size != weights.size:
        raise ValueError("Invalid class membership axes.")
    result = np.zeros(class_count, dtype=np.float64)
    for p in range(weights.size):
        c = class_indices[p]
        if c < 0 or c >= class_count or not np.isfinite(weights[p]) or weights[p] < 0.0:
            raise ValueError("Invalid product weight or class index.")
        result[c] += weights[p]
    return result


@njit(FO(F, M, int64, float64), cache=True, nogil=True)
def current_signal_tilt_kernel(probabilities, state_tilts, use_signal, strength):
    output = np.zeros(state_tilts.shape[1], dtype=np.float64)
    if probabilities.size != state_tilts.shape[0]:
        raise ValueError("Current signal axes do not match.")
    if use_signal:
        for a in range(output.size):
            for state in range(probabilities.size):
                output[a] += probabilities[state] * state_tilts[state, a] * strength
    return output


current_signal_tilt_kernel.disable_compile()

KERNELS = (input_status_kernel, constrained_tilts_kernel, path_metrics_kernel,
           select_candidate_kernel, candidate_feasibility_kernel, returns_availability_kernel,
           momentum_windows_kernel, momentum_signals_kernel, signal_counts_kernel, compose_product_weights_kernel,
           scenario_contributions_kernel, target_tilt_kernel,
           recommendation_details_kernel, aggregate_class_weights_kernel,
           knowledge_window_status_kernel, current_signal_tilt_kernel)


def execution_audit() -> dict[str, Any]:
    inherited = taa_numba_execution_audit()
    complete = all(len(k.signatures) == 1 and k.nopython_signatures
                   and not any(v.objectmode for v in k.overloads.values()) for k in KERNELS)
    warmed = _WARMED_PID == os.getpid()
    return {
        "engine": "tactical-allocation-njit/1.1.0", "backend": "numba_njit_fixed_signature",
        "fully_warmed": bool(complete and inherited["complete"] and warmed),
        "complete": bool(complete and inherited["complete"] and warmed),
        "nopython": bool(complete), "python_fallback": 0, "object_mode": 0,
        "request_time_compilation": 0, "shared_portfolio_engine": inherited["engine"],
        "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in KERNELS},
    }


def _require_ready():
    if _WARMED_PID != os.getpid() and _WARMING_PID != os.getpid():
        raise RuntimeError("Tactical allocation kernels were not warmed in this worker.")


def _array(value: Any, dtype: Any, ndim: int) -> np.ndarray:
    if dtype in (np.int64, np.uint8):
        raw = np.asarray(value)
        if raw.dtype.kind not in ("i", "u", "b"):
            raise ValueError("Integer metadata cannot be inferred from fractional values.")
        if dtype == np.int64 and raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError("Integer metadata exceeds the signed int64 range.")
        if dtype == np.uint8 and (np.any(raw < 0) or np.any(raw > 1)):
            raise ValueError("Signal validity flags must be zero or one.")
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise ValueError(f"Expected a {ndim}-dimensional numeric array.")
    return result


def _metrics(values: np.ndarray) -> dict[str, Any]:
    return {name: (int(values[i]) if name == "observations" else float(values[i])) if np.isfinite(values[i]) else None
            for i, name in enumerate(METRIC_NAMES)}


def _groups(asset_count, membership, lower, upper):
    if membership is None:
        if lower is not None or upper is not None:
            raise ValueError("Group bounds require explicit group membership.")
        return np.empty((0, asset_count), dtype=np.uint8), np.empty(0), np.empty(0)
    if lower is None or upper is None:
        raise ValueError("Every group requires explicit lower and upper bounds.")
    return _array(membership, np.uint8, 2), _array(lower, np.float64, 1), _array(upper, np.float64, 1)


def _checked_path(returns, probabilities, flags, tilts, base, cost):
    path, _, _, _ = _taa_path_kernel(returns, probabilities, flags, tilts, base, 0.0, 1.0, 1.0, cost)
    if _taa_output_validation_kernel(path, base.size, 1e-6) != 0:
        raise ValueError("Non-finite or invalid TAA portfolio path.")
    return path


def evaluate_candidates(
    returns, probabilities, use_signal, base, state_tilts, min_weights, max_weights,
    max_abs_tilts, train_end_index, strengths=DEFAULT_STRENGTHS, cost=0.0,
    periods_per_year=252, risk_penalty=3.0, max_tracking_error=1.0,
    max_turnover=1.0, objective="active_utility", selected_candidate_id=None,
    group_membership=None, group_min=None, group_max=None,
):
    """Freeze a training-only selected strength, then evaluate its holdout path."""
    _require_ready()
    returns, probabilities = _array(returns, np.float64, 2), _array(probabilities, np.float64, 2)
    if returns.shape[0] > 20000 or returns.shape[1] > 100:
        raise ValueError("TAA research exceeds the bounded 20000-period / 100-asset workspace.")
    flags = _array(use_signal, np.uint8, 1)
    base, tilts = _array(base, np.float64, 1), _array(state_tilts, np.float64, 2)
    lower, upper, caps = (_array(x, np.float64, 1) for x in (min_weights, max_weights, max_abs_tilts))
    groups, group_lower, group_upper = _groups(base.size, group_membership, group_min, group_max)
    status = input_status_kernel(returns, probabilities, flags, base, tilts, lower, upper, caps)
    if status:
        raise ValueError(f"Invalid TAA inputs (contract {status}).")
    if isinstance(train_end_index, bool) or int(train_end_index) != train_end_index:
        raise ValueError("Training split must be an integer observation index.")
    split = int(train_end_index)
    if split < 20 or returns.shape[0] - split < 20:
        raise ValueError("Training and untouched validation each require at least 20 observations.")
    strength_values = _array(strengths, np.float64, 1)
    if tuple(strength_values) not in (DEFAULT_STRENGTHS, (0.0, 1.0)):
        raise ValueError("Candidate strengths must be the predeclared grid or baseline/manual pair.")
    for value, minimum, maximum, label in (
        (cost, 0, 1000, "cost"), (periods_per_year, 1, 3660, "periods_per_year"),
        (risk_penalty, 0, 1000, "risk_penalty"), (max_tracking_error, 0, 10, "max_tracking_error"),
        (max_turnover, 0, 1, "max_turnover"),
    ):
        if not np.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid {label}.")
    if isinstance(periods_per_year, bool) or int(periods_per_year) != periods_per_year:
        raise ValueError("Annualization requires an integer number of periods.")
    objectives = {"active_utility": 0, "net_excess": 1, "excess_return": 1, "min_drawdown": 2}
    if objective not in objectives:
        raise ValueError("Unknown training objective.")
    train_metrics = np.empty((strength_values.size, len(METRIC_NAMES)), dtype=np.float64)
    feasible = np.zeros(strength_values.size, dtype=np.uint8)
    bounded, scales = [], []
    # Each half is a view. Candidate-specific tilts and returned paths are outputs.
    for index, strength in enumerate(strength_values):
        effective, scale = constrained_tilts_kernel(base, tilts, lower, upper, caps, float(strength), groups, group_lower, group_upper)
        bounded.append(effective)
        scales.append(scale)
        path = _checked_path(returns[:split], probabilities[:split], flags[:split], effective, base, float(cost))
        train_metrics[index] = path_metrics_kernel(path, base.size, float(periods_per_year), float(risk_penalty), objectives[objective])
        feasible[index] = candidate_feasibility_kernel(train_metrics[index], float(max_tracking_error), float(max_turnover))
    # SAA is always an available fallback, even if its drift rebalancing exceeds
    # a requested tactical turnover cap; this exemption is explicitly reported.
    feasible[0] = 1
    auto_selected = int(select_candidate_kernel(train_metrics, feasible))
    selected = auto_selected
    if selected_candidate_id is not None:
        ids = [f"scale-{index}" for index in range(strength_values.size)]
        if selected_candidate_id not in ids:
            raise ValueError("Unknown candidate selection.")
        selected = ids.index(selected_candidate_id)
        if not feasible[selected]:
            raise ValueError("Selected candidate violates training constraints.")
    selected_train_path = _checked_path(returns[:split], probabilities[:split], flags[:split], bounded[selected], base, float(cost))
    selected_train_path.setflags(write=False)
    candidates, selected_path = [], None
    for index, strength in enumerate(strength_values):
        path = _checked_path(returns[split:], probabilities[split:], flags[split:], bounded[index], base, float(cost))
        validation = path_metrics_kernel(path, base.size, float(periods_per_year), float(risk_penalty), objectives[objective])
        candidates.append({
            "id": f"scale-{index}", "strength": float(strength), "feasible": bool(feasible[index]),
            "validation_feasible": bool(candidate_feasibility_kernel(validation, float(max_tracking_error), float(max_turnover))),
            "constraint_scales": scales[index].tolist(), "train": _metrics(train_metrics[index]),
            "validation": _metrics(validation), "baseline_fallback_exemption": index == 0,
        })
        if index == selected:
            selected_path = path
    assert selected_path is not None
    selected_path.setflags(write=False)
    selected_tilts = bounded[selected]
    selected_tilts.setflags(write=False)
    offset = base.size * 2
    return {
        "candidates": candidates, "selected_id": f"scale-{selected}", "selected_index": selected,
        "auto_selected_id": f"scale-{auto_selected}", "auto_selected_index": auto_selected,
        "selected_train_path": selected_train_path, "selected_validation_path": selected_path,
        "train_baseline_nav": selected_train_path[:, offset + 8],
        "train_selected_nav": selected_train_path[:, offset + 9],
        "train_selected_weights": selected_train_path[:, :base.size],
        "selected_path": selected_path, "baseline_nav": selected_path[:, offset + 8],
        "selected_nav": selected_path[:, offset + 9], "selected_weights": selected_path[:, :base.size],
        "selected_state_tilts": selected_tilts, "execution": execution_audit(),
        "selection_policy": {"scope": "predeclared_direction_and_strength_grid", "objective": objective,
                             "training_observations": split, "validation_observations": returns.shape[0] - split,
                             "holdout_used_for_selection": False, "independently_funded_intervals": True,
                             "future_optimality_claim": False, "baseline_fallback_exemption": True},
    }


def build_momentum_signals(returns, lookback, max_tilt, available_at=None, period_starts=None, as_of_day=None,
                           period_ends=None, max_signal_age_days=31):
    _require_ready()
    values = _array(returns, np.float64, 2)
    if isinstance(lookback, bool) or int(lookback) != lookback:
        raise ValueError("Momentum lookback must be an integer.")
    available = np.full(values.shape, -1, dtype=np.int64) if available_at is None else _array(available_at, np.int64, 2)
    if period_starts is None or period_ends is None or as_of_day is None:
        raise ValueError("Momentum requires explicit return dates and decision cutoff.")
    starts, ends = _array(period_starts, np.int64, 1), _array(period_ends, np.int64, 1)
    if isinstance(max_signal_age_days, bool) or int(max_signal_age_days) != max_signal_age_days:
        raise ValueError("Momentum signal age must be an integer.")
    result = momentum_signals_kernel(values, int(lookback), float(max_tilt), available, starts, ends, int(as_of_day), int(max_signal_age_days))
    names = ("probabilities", "use_signal", "state_tilts", "current_probabilities", "current_use_signal",
             "current_momentum", "knowledge_verified", "current_knowledge_verified", "windows")
    return dict(zip(names, result))


def compose_product_weights(class_weights, class_indices, within_weights):
    _require_ready()
    return compose_product_weights_kernel(_array(class_weights, np.float64, 1),
                                          _array(class_indices, np.int64, 1),
                                          _array(within_weights, np.float64, 1))


def aggregate_class_weights(weights, class_indices, class_count):
    _require_ready()
    if isinstance(class_count, bool) or int(class_count) != class_count:
        raise ValueError("class_count must be an integer.")
    return aggregate_class_weights_kernel(_array(weights, np.float64, 1),
                                          _array(class_indices, np.int64, 1), int(class_count))


def recommend_weights(probabilities, use_signal, base, state_tilts, lo, hi, max_tilt, strength, current_weights=None,
                      group_membership=None, group_min=None, group_max=None):
    _require_ready()
    base, tilts = _array(base, np.float64, 1), _array(state_tilts, np.float64, 2)
    lower, upper, caps = (_array(v, np.float64, 1) for v in (lo, hi, max_tilt))
    probabilities = _array(probabilities, np.float64, 1).reshape(1, -1)
    if use_signal not in (0, 1) or not np.isfinite(strength) or not 0 <= strength <= 1.5:
        raise ValueError("Invalid current signal or strength.")
    flags = np.array([use_signal], dtype=np.uint8)
    # A zero-return path obtains precisely the same constrained target as the
    # historical engine; this feature calculation is never added to its sample.
    returns = np.zeros((1, base.size))
    if input_status_kernel(returns, probabilities, flags, base, tilts, lower, upper, caps):
        raise ValueError("Invalid current recommendation inputs.")
    groups, group_lower, group_upper = _groups(base.size, group_membership, group_min, group_max)
    effective, scales = constrained_tilts_kernel(base, tilts, lower, upper, caps, float(strength), groups, group_lower, group_upper)
    path = _checked_path(returns, probabilities, flags, effective, base, 0.0)
    weights = path[0, :base.size]
    current = base if current_weights is None else _array(current_weights, np.float64, 1)
    deviations, trades, turnover, has_deviation = recommendation_details_kernel(base, weights, current)
    path.setflags(write=False)
    weights.setflags(write=False)
    return {"weights": weights, "tilts": deviations, "trade_deltas": trades,
            "turnover": float(turnover), "constraint_scales": scales,
            "state_tilts": effective, "fallback_to_saa": not bool(use_signal),
            "has_deviation": bool(has_deviation), "is_saa": not bool(has_deviation)}


def stress_compare(returns, base, target, cost=0.0, periods_per_year=252):
    _require_ready()
    values = _array(returns, np.float64, 2)
    base, target = _array(base, np.float64, 1), _array(target, np.float64, 1)
    if values.shape[0] == 0 or not np.isfinite(cost) or not 0 <= cost <= 1000:
        raise ValueError("Invalid scenario interval or costs.")
    if not np.isfinite(periods_per_year) or not 1 <= periods_per_year <= 3660:
        raise ValueError("Invalid scenario annualization.")
    if isinstance(periods_per_year, bool) or int(periods_per_year) != periods_per_year:
        raise ValueError("Annualization requires an integer number of periods.")
    tilts = target_tilt_kernel(base, target)
    probabilities, flags = np.ones((values.shape[0], 1)), np.ones(values.shape[0], dtype=np.uint8)
    if input_status_kernel(values, probabilities, flags, base, tilts, np.zeros(base.size), np.ones(base.size), np.ones(base.size)):
        raise ValueError("Invalid scenario returns or weights.")
    path = _checked_path(values, probabilities, flags, tilts, base, float(cost))
    metrics = path_metrics_kernel(path, base.size, float(periods_per_year), 0.0, 0)
    baseline_contribution, target_contribution, excess_contribution = scenario_contributions_kernel(values, path, base)
    baseline_performance = _performance_kernel(path[:, base.size * 2 + 5], path[:, base.size * 2 + 8], float(periods_per_year))
    path.setflags(write=False)
    return {
        "baseline": {"total_return": float(metrics[1]), "annual_volatility": float(baseline_performance[2]) if np.isfinite(baseline_performance[2]) else None,
                     "max_drawdown": float(baseline_performance[4]), "turnover": float(metrics[11]), "cost": float(metrics[12])},
        "target": _metrics(metrics), "excess_return": float(metrics[2]),
        "baseline_nav": path[:, base.size * 2 + 8], "target_nav": path[:, base.size * 2 + 9],
        "baseline_contributions": baseline_contribution, "target_contributions": target_contribution,
        "excess_contributions": excess_contribution,
        "total_return_difference": float(metrics[15]),
        "relative_excess_return": float(metrics[2]),
        "baseline_cost": float(metrics[12]), "target_cost": float(metrics[10]),
        "holding_policy": "rebalance_to_target_each_observation", "execution": execution_audit(),
    }


def knowledge_window_status(available_at, cutoff, start=0, end=None):
    _require_ready()
    available = _array(available_at, np.int64, 2)
    stop = available.shape[0] if end is None else end
    if any(isinstance(v, bool) or int(v) != v for v in (cutoff, start, stop)):
        raise ValueError("Knowledge dates and interval bounds must be integers.")
    future, unknown = knowledge_window_status_kernel(available, int(cutoff), int(start), int(stop))
    return {"future_cells": int(future), "unknown_cells": int(unknown),
            "verified": future == 0 and unknown == 0}


def warm_tactical_allocation_kernels():
    global _WARMED_PID, _WARMING_PID
    _WARMED_PID = None
    _WARMING_PID = os.getpid()
    try:
        from backend.research_input_checks import warm_research_input_checks
        warm_research_input_checks()
        warm_taa_numba_kernels()
        returns = np.zeros((40, 2), dtype=np.float64)
        starts = np.arange(40, dtype=np.int64)
        available = np.broadcast_to((starts + 1)[:, None], returns.shape).copy()
        base = np.array([0.5, 0.5])
        signals = build_momentum_signals(returns, 2, 0.1, available, starts, 40, starts + 1)
        signal_counts_kernel(signals["use_signal"], signals["knowledge_verified"], 20)
        current_signal_tilt_kernel(signals["current_probabilities"], signals["state_tilts"], int(signals["current_use_signal"]), 1.0)
        evaluate_candidates(returns, signals["probabilities"], signals["use_signal"], base,
                            signals["state_tilts"], np.zeros(2), np.ones(2), np.ones(2), 20)
        returns_availability_kernel(available)
        compose_product_weights(base, np.array([0, 1], dtype=np.int64), np.ones(2))
        aggregate_class_weights(base, np.array([0, 1], dtype=np.int64), 2)
        recommend_weights(signals["current_probabilities"], signals["current_use_signal"], base,
                          signals["state_tilts"], np.zeros(2), np.ones(2), np.ones(2), 1.0)
        stress_compare(returns[:2], base, base)
        knowledge_window_status(available, 40)
        _WARMED_PID = os.getpid()
        audit = execution_audit()
        if not audit["fully_warmed"]:
            _WARMED_PID = None
            raise RuntimeError("Tactical allocation NJIT warmup incomplete.")
        return audit
    finally:
        _WARMING_PID = None
