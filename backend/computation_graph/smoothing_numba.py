"""Shared smoothing kernels; observation-spaced input, never Python fallback.

Derivations and boundary contracts: docs/regimes/smoothing.md.
Inputs are borrowed read-only strided views; only outputs and workspaces allocate.
"""
import numpy as np
from numba import float64, int64, njit, types

READ = types.Array(float64, 1, "A", readonly=True)
OUT = float64[::1]


@njit(OUT(READ, int64), cache=True)
def butterworth_zero_phase_kernel(values, period):
    """Second-order Butterworth, forward/backward, odd padding of nine samples."""
    if period < 3 or period > 5000:
        raise ValueError("INVALID_SMOOTHING_PARAMETERS")
    result = np.full(values.size, np.nan)
    work = np.empty(values.size + 18)
    k = np.tan(np.pi / period)
    norm = 1.0 / (1.0 + np.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    b1, b2 = 2.0 * b0, b0
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - np.sqrt(2.0) * k + k * k) * norm
    start = 0
    while start < values.size:
        if not np.isfinite(values[start]):
            start += 1
            continue
        end = start + 1
        scale = abs(values[start])
        while end < values.size and np.isfinite(values[end]):
            scale = max(scale, abs(values[end]))
            end += 1
        size = end - start
        if size > 9:
            if scale == 0.0:
                scale = 1.0
            # Linear homogeneity permits bounded arithmetic even for finite 1e308 inputs.
            for j in range(9):
                work[j] = 2.0 * (values[start] / scale) - values[start + 9 - j] / scale
                work[9 + size + j] = 2.0 * (values[end - 1] / scale) - values[end - 2 - j] / scale
            for j in range(size):
                work[9 + j] = values[start + j] / scale
            # Direct form II transposed, steady-state initialization in each direction.
            for direction in range(2):
                first = 0 if direction == 0 else size + 17
                z1 = (1.0 - b0) * work[first]
                z2 = (b2 - a2) * work[first]
                for j in range(size + 18):
                    pos = j if direction == 0 else size + 17 - j
                    x = work[pos]
                    y = b0 * x + z1
                    z1 = b1 * x - a1 * y + z2
                    z2 = b2 * x - a2 * y
                    work[pos] = y
            for j in range(size):
                value = work[9 + j] * scale
                if np.isfinite(value):
                    result[start + j] = value
        start = end
    return result


@njit(OUT(READ, int64, int64), cache=True)
def savitzky_golay_centered_kernel(values, window, polyorder):
    """Centered polynomial projection; incomplete/nonfinite windows stay NaN."""
    if window < 3 or window > 501 or window % 2 != 1 or polyorder < 0 or polyorder > 5 or polyorder >= window:
        raise ValueError("INVALID_SMOOTHING_PARAMETERS")
    result = np.full(values.size, np.nan)
    if values.size < window:
        return result
    half = window // 2
    # Twice-reorthogonalized MGS on a scaled Vandermonde basis; no normal equations.
    q = np.empty((polyorder + 1, window))
    weights = np.zeros(window)
    for degree in range(polyorder + 1):
        for j in range(window):
            q[degree, j] = ((j - half) / half) ** degree
        for repeat in range(2):
            for previous in range(degree):
                dot = 0.0
                for j in range(window):
                    dot += q[degree, j] * q[previous, j]
                for j in range(window):
                    q[degree, j] -= dot * q[previous, j]
        norm = 0.0
        for j in range(window):
            norm += q[degree, j] ** 2
        norm = np.sqrt(norm)
        for j in range(window):
            q[degree, j] /= norm
        for j in range(window):
            weights[j] += q[degree, half] * q[degree, j]
    count = 0
    for right in range(values.size):
        count = count + 1 if np.isfinite(values[right]) else 0
        if count >= window:
            total = 0.0
            for j in range(window):
                total += weights[j] * values[right - window + 1 + j]
            if np.isfinite(total):
                result[right - half] = total
    return result


@njit(OUT(READ, int64, int64), cache=True)
def ehlers_error_correcting_kernel(values, period, gain_limit):
    """Ehlers/Way discrete error correction; gain_limit is in tenths, inclusive."""
    if period < 2 or period > 5000 or gain_limit < 0 or gain_limit > 100:
        raise ValueError("INVALID_SMOOTHING_PARAMETERS")
    result = np.full(values.size, np.nan)
    alpha = 2.0 / (period + 1.0)
    ema = ec = 0.0
    count = 0
    for t in range(values.size):
        x = values[t]
        if not np.isfinite(x):
            count = 0
            continue
        if count == 0:
            ema = ec = x
        else:
            ema = alpha * x + (1.0 - alpha) * ema
            best, error = np.nan, np.inf
            for gain_step in range(-gain_limit, gain_limit + 1):
                candidate = alpha * (ema + gain_step / 10.0 * (x - ec)) + (1.0 - alpha) * ec
                loss = abs(x - candidate)
                if np.isfinite(candidate) and loss < error:
                    best, error = candidate, loss
            ec = best
        if not np.isfinite(ema) or not np.isfinite(ec):
            count = 0
            continue
        count += 1
        if count >= period:
            result[t] = ec
    return result


@njit(int64[::1](types.Array(int64, 1, "A", readonly=True), int64), cache=True)
def smoothing_available_kernel(available, retrospective):
    result = np.empty(available.size, dtype=np.int64)
    latest = np.iinfo(np.int64).min
    for t in range(available.size):
        latest = max(latest, available[t])
        result[t] = latest
    if retrospective:
        for t in range(available.size):
            result[t] = latest
    return result


SMOOTHING_KERNELS = {
    "smoothing_available": smoothing_available_kernel,
    "butterworth_zero_phase": butterworth_zero_phase_kernel,
    "savitzky_golay_centered": savitzky_golay_centered_kernel,
    "ehlers_error_correcting": ehlers_error_correcting_kernel,
}
for dispatcher in SMOOTHING_KERNELS.values():
    dispatcher.disable_compile()
