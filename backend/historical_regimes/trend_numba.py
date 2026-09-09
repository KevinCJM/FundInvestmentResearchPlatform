"""Causal trend features and explicit retrospective denoising, fixed signatures."""

import numpy as np
from numba import float64, int64, njit, types

F = float64[::1]
I = int64[::1]
FEATURES = types.UniTuple(F, 8)
STATES = types.Tuple((I, I, F, F))


@njit(F(F, int64), cache=True)
def super_smoother_kernel(values, period):
    result = np.full(values.size, np.nan)
    theta = np.sqrt(2.0) * np.pi / period
    a = np.exp(-theta)
    c2, c3 = 2.0 * a * np.cos(theta), -a * a
    c1 = 1.0 - c2 - c3
    count = 0
    previous_input = previous = older = 0.0
    for t in range(values.size):
        x = values[t]
        if not np.isfinite(x):
            count = 0
            continue
        current = x if count < 2 else c1 * (x + previous_input) / 2.0 + c2 * previous + c3 * older
        if not np.isfinite(current):
            count = 0
            continue
        older, previous, previous_input = previous, current, x
        count += 1
        if count >= period:
            result[t] = current
    return result


@njit(F(F, int64, int64, int64), cache=True)
def kama_kernel(values, window, fast, slow):
    result = np.full(values.size, np.nan)
    count = 0
    previous = 0.0
    fast_gain, slow_gain = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    for t in range(values.size):
        if not np.isfinite(values[t]):
            count = 0
            continue
        if count == 0:
            previous = values[t]
        count += 1
        if count <= window:
            continue
        noise = 0.0
        for j in range(t - window + 1, t + 1):
            noise += abs(values[j] - values[j - 1])
        efficiency = abs(values[t] - values[t - window]) / noise if noise > 0.0 else 0.0
        gain = (efficiency * (fast_gain - slow_gain) + slow_gain) ** 2
        previous += gain * (values[t] - previous)
        result[t] = previous
    return result


@njit(FEATURES(F, F, int64, int64, int64, float64, int64, float64, float64), cache=True)
def trend_features_kernel(values, trend, volatility_window, slope_window, efficiency_window,
                          scale_floor, shock_window, drawdown_alert, shock_alert):
    distance = np.full(values.size, np.nan)
    slope = np.full(values.size, np.nan)
    efficiency = np.full(values.size, np.nan)
    scale = np.full(values.size, np.nan)
    drawdown = np.full(values.size, np.nan)
    risk = np.full(values.size, np.nan)
    index_value = np.full(values.size, np.nan)
    filtered_index = np.full(values.size, np.nan)
    gain = 2.0 / (volatility_window + 1.0)
    raw_count = trend_count = returns_count = 0
    variance = previous = peak = 0.0
    for t in range(values.size):
        x = values[t]
        if not np.isfinite(x):
            raw_count = trend_count = returns_count = 0
            variance = 0.0
            continue
        if raw_count == 0:
            peak = x
        peak = max(peak, x)
        raw_count += 1
        if x < 709.0:
            index_value[t] = np.exp(x)
        if np.isfinite(trend[t]) and trend[t] < 709.0:
            filtered_index[t] = np.exp(trend[t])
        trend_count = trend_count + 1 if np.isfinite(trend[t]) else 0
        drawdown[t] = np.expm1(x - peak)
        risk[t] = 1.0 if drawdown[t] <= -drawdown_alert else 0.0
        if raw_count > shock_window and x - values[t - shock_window] <= np.log1p(-shock_alert):
            risk[t] = 1.0
        # Use the previous period's RMS: today's drop must not widen its own band.
        if returns_count >= volatility_window:
            scale[t] = max(scale_floor, np.sqrt(variance))
            if np.isfinite(trend[t]):
                distance[t] = (x - trend[t]) / scale[t]
            if trend_count > slope_window:
                slope[t] = (trend[t] - trend[t - slope_window]) / (slope_window * scale[t])
        if raw_count > efficiency_window:
            noise = 0.0
            for j in range(t - efficiency_window + 1, t + 1):
                noise += abs(values[j] - values[j - 1])
            efficiency[t] = abs(x - values[t - efficiency_window]) / noise if noise > 0.0 else 0.0
        if raw_count > 1:
            squared = (x - previous) ** 2
            variance = squared if returns_count == 0 else gain * squared + (1.0 - gain) * variance
            returns_count += 1
        previous = x
    return distance, slope, efficiency, scale, drawdown, risk, index_value, filtered_index


@njit(STATES(F, F, F, float64, float64, float64, float64, int64), cache=True)
def trend_regime_kernel(distance, slope, efficiency, band, trend_enter, flat_threshold,
                        efficiency_ceiling, confirmation):
    states = np.full(distance.size, -1, dtype=np.int64)
    candidates = np.full(distance.size, -1, dtype=np.int64)
    pending = np.zeros(distance.size)
    phase = np.full(distance.size, np.nan)
    active = candidate = -1
    count = 0
    for t in range(distance.size):
        d, s, er = distance[t], slope[t], efficiency[t]
        if not (np.isfinite(d) and np.isfinite(s) and np.isfinite(er)):
            candidate, count = -1, 0
            continue
        proposed = -1
        if d >= band and s >= trend_enter:
            proposed = 0
        elif d <= -band and s <= -trend_enter:
            proposed = 2
        elif abs(s) <= flat_threshold and er <= efficiency_ceiling:
            proposed = 1
        candidates[t] = proposed
        if proposed < 0 or proposed == active:
            candidate, count = -1, 0
        else:
            count = count + 1 if proposed == candidate else 1
            candidate = proposed
            if count >= confirmation:
                active = proposed
                candidate, count = -1, 0
        states[t], pending[t] = active, count
        if active == 0:
            phase[t] = 1.0 if d < 0 else 0.0  # bull correction / advance
        elif active == 2:
            phase[t] = 3.0 if d > 0 else 2.0  # bear rally / decline
        elif active == 1:
            phase[t] = 4.0
    return states, candidates, pending, phase


@njit(I(I, F, int64, float64, int64), cache=True)
def merge_short_regimes_kernel(states, prices, max_duration, max_move, following_confirmation):
    """One left-to-right pass over ORIGINAL runs; never merge unknowns or the tail."""
    result = states.copy()
    starts = np.empty(states.size, dtype=np.int64)
    ends = np.empty(states.size, dtype=np.int64)
    runs = 0
    for t in range(states.size):
        if t == 0 or states[t] != states[t - 1]:
            if runs:
                ends[runs - 1] = t
            starts[runs] = t
            runs += 1
    if runs:
        ends[runs - 1] = states.size
    for r in range(1, runs - 1):
        first, stop = starts[r], ends[r]
        parent, middle = states[starts[r - 1]], states[first]
        # A removed run cannot anchor a second, contradictory merge.
        if result[starts[r - 1]] != parent:
            continue
        if not ((parent == 0 and middle == 2) or (parent == 2 and middle == 0)):
            continue
        if states[starts[r + 1]] != parent or stop - first > max_duration:
            continue
        if ends[r + 1] - starts[r + 1] < following_confirmation:
            continue
        extreme = prices[starts[r - 1]]
        valid = np.isfinite(extreme) and extreme > 0.0
        for j in range(stop, stop + following_confirmation):
            if not np.isfinite(prices[j]) or prices[j] <= 0.0:
                valid = False
        # Include the preceding phase's extreme; endpoint returns can hide a crash.
        for j in range(starts[r - 1], stop):
            x = prices[j]
            if not np.isfinite(x) or x <= 0.0:
                valid = False
                break
            if parent == 0:
                extreme = max(extreme, x)
                if j >= first and 1.0 - x / extreme >= max_move:
                    valid = False
            else:
                extreme = min(extreme, x)
                if j >= first and x / extreme - 1.0 >= max_move:
                    valid = False
        if valid:
            for j in range(first, stop):
                result[j] = parent
    return result


TREND_KERNELS = {
    "super_smoother": super_smoother_kernel,
    "kama": kama_kernel,
    "trend_features": trend_features_kernel,
    "trend_regime": trend_regime_kernel,
    "merge_short_regimes": merge_short_regimes_kernel,
}

for dispatcher in TREND_KERNELS.values():
    dispatcher.disable_compile()
