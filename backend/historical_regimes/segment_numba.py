"""Independent retrospective turning points and interval statistics; no I/O."""
import numpy as np
from numba import float64, int64, njit, types

F = float64[::1]
I = int64[::1]
READONLY_F = types.Array(float64, 1, "A", readonly=True)
READONLY_I = types.Array(int64, 1, "A", readonly=True)


@njit(types.Tuple((F, F))(F, int64, int64, int64, int64), cache=True)
def local_extrema_kernel(prices, left_window, right_window, head_window, tail_window):
    pivots = np.zeros(prices.size)
    pivot_prices = np.full(prices.size, np.nan)
    start = 0
    while start < prices.size:
        if not np.isfinite(prices[start]) or prices[start] <= 0:
            pivots[start] = np.nan
            start += 1
            continue
        stop = start + 1
        while stop < prices.size and np.isfinite(prices[stop]) and prices[stop] > 0:
            stop += 1
        previous = -1
        for t in range(start + max(left_window, head_window), stop - max(right_window, tail_window)):
            peak, trough = True, True
            for j in range(t - left_window, t):
                peak = peak and prices[t] > prices[j]
                trough = trough and prices[t] < prices[j]
            for j in range(t + 1, t + right_window + 1):
                peak = peak and prices[t] >= prices[j]
                trough = trough and prices[t] <= prices[j]
            if not peak and not trough:
                continue
            kind = 1.0 if peak else -1.0
            if previous >= 0:
                if pivots[previous] == kind:
                    if kind * (prices[t] - prices[previous]) <= 0:
                        continue
                    pivots[previous], pivot_prices[previous] = 0.0, np.nan
                elif kind * (prices[t] - prices[previous]) <= 0:
                    continue
            pivots[t], pivot_prices[t], previous = kind, prices[t], t
        start = stop
    return pivots, pivot_prices


@njit(types.Tuple((I, I))(F), cache=True)
def between_pivots_kernel(pivots):
    starts = np.full(pivots.size, -1, dtype=np.int64)
    ends = np.full(pivots.size, -1, dtype=np.int64)
    previous = -1
    for t in range(pivots.size):
        kind = pivots[t]
        if not np.isfinite(kind):
            previous = -1
        elif kind == 1 or kind == -1:
            if previous >= 0 and kind != pivots[previous]:
                for j in range(previous, t):
                    starts[j], ends[j] = previous, t
            previous = t
        elif kind != 0:
            previous = -1
    return starts, ends


@njit(F(F, I, I, int64, int64), cache=True)
def interval_statistic_kernel(prices, starts, ends, statistic, ddof):
    """0 change, 1 amplitude, 2 simple-return std, 3 duration, 4 efficiency.

    Input boundaries are a single segmentation's T-aligned arrays. Each path
    is inspected once and only complete, coherent intervals are broadcast.
    """
    n = prices.size
    output = np.full(n, np.nan)
    if starts.size != n or ends.size != n or statistic < 0 or statistic > 4 or ddof < 0:
        return output
    t = 0
    while t < n:
        left, right = starts[t], ends[t]
        if left != t or right <= left or right >= n:
            t += 1
            continue
        valid = np.isfinite(prices[left]) and prices[left] > 0
        low, high = prices[left], prices[left]
        path, mean, m2 = 0.0, 0.0, 0.0
        count = 0
        for j in range(left, right):
            valid = valid and starts[j] == left and ends[j] == right
        for j in range(left + 1, right + 1):
            if not np.isfinite(prices[j]) or prices[j] <= 0:
                valid = False
                break
            if not valid:
                break
            low, high = min(low, prices[j]), max(high, prices[j])
            path += abs(prices[j] - prices[j - 1])
            change = prices[j] / prices[j - 1] - 1.0
            count += 1
            delta = change - mean
            mean += delta / count
            m2 += delta * (change - mean)
        value = np.nan
        if valid:
            if statistic == 0:
                value = prices[right] / prices[left] - 1.0
            elif statistic == 1:
                value = high / low - 1.0
            elif statistic == 2 and count > ddof:
                value = np.sqrt(max(0.0, m2 / (count - ddof)))
            elif statistic == 3:
                value = float(right - left)
            elif statistic == 4:
                value = abs(prices[right] - prices[left]) / path if path > 0 else 0.0
        if np.isfinite(value):
            for j in range(left, right):
                output[j] = value
        t = right
    return output


@njit(types.Tuple((I, int64))(F, F, F), cache=True)
def range_threshold_kernel(values, upper, lower):
    states = np.full(values.size, -1, dtype=np.int64)
    invalid = 0
    if upper.size != values.size or lower.size != values.size:
        return states, 1
    for t in range(values.size):
        if not np.isfinite(upper[t]) or not np.isfinite(lower[t]):
            continue
        if lower[t] >= upper[t]:
            invalid += 1
            continue
        if np.isfinite(values[t]):
            states[t] = 0 if values[t] > upper[t] else (2 if values[t] < lower[t] else 1)
    return states, invalid


@njit(I(F, I, I), cache=True)
def phase_direction_kernel(pivots, starts, ends):
    """Complete-wave direction: 0 trough-to-peak, 1 peak-to-trough, -1 missing."""
    result = np.full(pivots.size, -1, dtype=np.int64)
    for t in range(pivots.size):
        left, right = starts[t], ends[t]
        if 0 <= left <= t < right < pivots.size:
            if pivots[left] == -1.0 and pivots[right] == 1.0:
                result[t] = 0
            elif pivots[left] == 1.0 and pivots[right] == -1.0:
                result[t] = 1
    return result


@njit(I(READONLY_I, READONLY_F, READONLY_I, READONLY_I, float64), cache=True)
def drawdown_cycle_reference_kernel(phases, changes, starts, ends, stress_drawdown):
    """Retrospective normal/recovery/stress labels over complete pivot segments.

    The kernel does not discover pivots or recompute interval returns. It only
    carries one piece of cross-segment state: an up-phase is Recovery iff the
    immediately preceding down-phase breached the configured drawdown.
    Open head/tail regions stay unclassified.
    """
    n = phases.size
    result = np.full(n, -1, dtype=np.int64)
    if changes.size != n or starts.size != n or ends.size != n or not 0.0 < stress_drawdown < 1.0:
        return result
    prior_stress = False
    t = 0
    while t < n:
        left, right = starts[t], ends[t]
        if left != t or right <= left or right >= n:
            t += 1
            continue
        coherent = True
        for j in range(left, right):
            coherent = coherent and starts[j] == left and ends[j] == right
        phase = phases[t]
        change = changes[t]
        if not coherent or phase not in (0, 1) or not np.isfinite(change):
            prior_stress = False
            t = right
            continue
        # A shared pivot keeps the completed preceding segment's label. Only
        # the first classified segment has no preceding environment to retain.
        if result[left] == -1:
            result[left] = 0
        for j in range(left + 1, right + 1):
            result[j] = 0
        if phase == 1:
            prior_stress = change <= -stress_drawdown
            if prior_stress:
                # The peak itself belongs to the preceding environment; the
                # decline becomes Stress from the first month after the peak.
                for j in range(left + 1, right + 1):
                    result[j] = 2
        else:
            if prior_stress:
                # The shared trough belongs to the completed Stress decline;
                # Recovery begins only with the first observation after it.
                result[left] = 2
                for j in range(left + 1, right + 1):
                    result[j] = 1
            prior_stress = False
        t = right
    return result


@njit(I(READONLY_F, int64, float64, float64, float64), cache=True)
def drawdown_cycle_realtime_kernel(prices, lookback, stress_drawdown, recovery_rebound, recovery_exit_drawdown):
    """Causal monthly drawdown state machine: Normal=0, Recovery=1, Stress=2."""
    n = prices.size
    result = np.full(n, -1, dtype=np.int64)
    if lookback < 2 or not 0.0 <= recovery_exit_drawdown < stress_drawdown < 1.0 or not 0.0 < recovery_rebound < 1.0:
        return result
    state = 0
    trough = np.nan
    for i in range(n):
        price = prices[i]
        if not np.isfinite(price) or price <= 0.0:
            result[i] = -1
            state = -1
            trough = np.nan
            continue
        start = max(0, i - lookback + 1)
        high = price
        valid = True
        for j in range(start, i + 1):
            if not np.isfinite(prices[j]) or prices[j] <= 0.0:
                valid = False
                break
            high = max(high, prices[j])
        if not valid or high <= 0.0:
            result[i] = -1
            state = -1
            trough = np.nan
            continue
        drawdown = price / high - 1.0
        if state == -1:
            # After a gap the hysteresis band cannot establish an environment.
            # Wait for an observed entry/exit boundary instead of inventing Normal.
            if drawdown <= -stress_drawdown:
                state = 2
                trough = price
            elif drawdown >= -recovery_exit_drawdown:
                state = 0
                trough = price
        elif state == 0:
            if drawdown <= -stress_drawdown:
                state = 2
                trough = price
        elif state == 2:
            if not np.isfinite(trough) or price < trough:
                trough = price
            elif price / trough - 1.0 >= recovery_rebound:
                state = 1
        else:
            if not np.isfinite(trough) or price < trough:
                trough = price
                state = 2
            elif drawdown >= -recovery_exit_drawdown:
                state = 0
                trough = price
        result[i] = state
    return result


@njit(F(F, I, I), cache=True)
def boundary_line_kernel(prices, starts, ends):
    """Interpolate complete boundaries, including the terminal boundary price."""
    result = np.full(prices.size, np.nan)
    t = 0
    while t < prices.size:
        left, right = starts[t], ends[t]
        if left != t or right <= left or right >= prices.size:
            t += 1
            continue
        valid = np.isfinite(prices[left]) and np.isfinite(prices[right])
        for j in range(left, right):
            valid = valid and starts[j] == left and ends[j] == right
        if valid:
            for j in range(left, right):
                result[j] = prices[left] + (prices[right] - prices[left]) * (j - left) / (right - left)
            result[right] = prices[right]
        t = right
    return result


SEGMENT_KERNELS = {
    "phase_direction": phase_direction_kernel,
    "boundary_line": boundary_line_kernel,
    "local_extrema": local_extrema_kernel,
    "between_pivots": between_pivots_kernel,
    "interval_statistic": interval_statistic_kernel,
    "range_threshold": range_threshold_kernel,
    "drawdown_cycle_reference": drawdown_cycle_reference_kernel,
    "drawdown_cycle_realtime": drawdown_cycle_realtime_kernel,
}
for _kernel in SEGMENT_KERNELS.values():
    _kernel.disable_compile()
