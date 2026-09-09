"""Independent retrospective turning points and interval statistics; no I/O."""
import numpy as np
from numba import float64, int64, njit, types

F = float64[::1]
I = int64[::1]


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


SEGMENT_KERNELS = {
    "local_extrema": local_extrema_kernel,
    "between_pivots": between_pivots_kernel,
    "interval_statistic": interval_statistic_kernel,
    "range_threshold": range_threshold_kernel,
}
for _kernel in SEGMENT_KERNELS.values():
    _kernel.disable_compile()
