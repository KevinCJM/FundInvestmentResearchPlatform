from __future__ import annotations

"""Fixed-signature NJIT kernel for the legacy synthetic product price path."""

import hashlib

import numpy as np
from numba import float64, int64, njit, types, uint64


SYNTHETIC_SERIES_ENGINE_VERSION = "synthetic-product-series-njit-1.0.0"
SYNTHETIC_SERIES_KERNEL_VERSION = "ohlcv-xorshift64-1"

_RNG_RESULT = types.Tuple((uint64, float64))


@njit(_RNG_RESULT(uint64), cache=False, nogil=True, inline="always")
def _next_uniform(state: np.uint64) -> tuple[np.uint64, float]:
    """Advance xorshift64* and return a value strictly inside (0, 1)."""

    state ^= state >> np.uint64(12)
    state ^= state << np.uint64(25)
    state ^= state >> np.uint64(27)
    raw = state * np.uint64(2685821657736338717)
    value = (float(raw >> np.uint64(11)) + 0.5) * (1.0 / 9007199254740992.0)
    return state, value


@njit(_RNG_RESULT(uint64), cache=False, nogil=True, inline="always")
def _next_normal(state: np.uint64) -> tuple[np.uint64, float]:
    state, first = _next_uniform(state)
    state, second = _next_uniform(state)
    value = np.sqrt(-2.0 * np.log(first)) * np.cos(2.0 * np.pi * second)
    return state, value


@njit(float64[:, ::1](int64, float64, int64), cache=False, nogil=True)
def synthetic_ohlcv_kernel(seed: int, issue_amount: float, periods: int) -> np.ndarray:
    """Return columns open/close/high/low/volume for a deterministic path."""

    if periods <= 0:
        raise ValueError("periods must be positive")

    state = np.uint64(seed)
    if state == np.uint64(0):
        state = np.uint64(0x9E3779B97F4A7C15)

    base_price = 1.0
    if np.isfinite(issue_amount):
        base_price = issue_amount / 5000.0
        if base_price < 0.8:
            base_price = 0.8
        elif base_price > 8.0:
            base_price = 8.0

    volume_base = 80000.0
    if np.isfinite(issue_amount):
        volume_base = issue_amount * 120.0
        if volume_base < 20000.0:
            volume_base = 20000.0
        elif volume_base > 5000000.0:
            volume_base = 5000000.0

    output = np.empty((periods, 5), dtype=np.float64)
    close_value = base_price if base_price > 0.5 else 0.5
    state, open_shock = _next_normal(state)
    open_value = close_value * (1.0 + 0.004 * open_shock)

    for index in range(periods):
        if index > 0:
            previous_close = close_value
            state, close_shock = _next_normal(state)
            close_value = previous_close * (1.0006 + 0.018 * close_shock)
            if close_value < 0.2:
                close_value = 0.2
            state, open_shock = _next_normal(state)
            open_value = previous_close * (1.0 + 0.006 * open_shock)

        state, high_shock = _next_uniform(state)
        state, low_shock = _next_uniform(state)
        state, volume_shock = _next_uniform(state)
        upper = open_value if open_value > close_value else close_value
        lower = open_value if open_value < close_value else close_value
        high_value = upper * (1.002 + 0.018 * high_shock)
        low_value = lower * (0.998 - 0.018 * low_shock)
        if low_value < 0.1:
            low_value = 0.1
        if high_value < low_value:
            high_value = low_value

        output[index, 0] = open_value
        output[index, 1] = close_value
        output[index, 2] = high_value
        output[index, 3] = low_value
        output[index, 4] = volume_base * (0.5 + 1.1 * volume_shock)

    return output


def synthetic_series_execution_audit() -> dict[str, object]:
    kernels = (_next_uniform, _next_normal, synthetic_ohlcv_kernel)
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in kernels
    }
    material = "|".join(
        [SYNTHETIC_SERIES_ENGINE_VERSION, SYNTHETIC_SERIES_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return {
        "engine": SYNTHETIC_SERIES_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": SYNTHETIC_SERIES_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(bool(kernel.nopython_signatures) for kernel in kernels),
        "python_fallback": 0,
    }


def warm_synthetic_series_numba_kernel() -> dict[str, object]:
    try:
        from backend.compute_policy import validate_execution_audit
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from compute_policy import validate_execution_audit

    values = synthetic_ohlcv_kernel(17, 100.0, 3)
    if values.shape != (3, 5) or not np.isfinite(values).all():
        raise RuntimeError("产品演示序列 NJIT 内核预热失败")
    return validate_execution_audit(synthetic_series_execution_audit())


__all__ = [
    "synthetic_ohlcv_kernel",
    "synthetic_series_execution_audit",
    "warm_synthetic_series_numba_kernel",
]
