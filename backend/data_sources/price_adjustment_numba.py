"""Fixed-signature numerical kernels for back-adjusted prices.

Every kernel expects the rows already sorted by (product, date), so one product
occupies one contiguous block and a block boundary is `codes[i] != codes[i-1]`.
No kernel divides by an unchecked denominator: a non-positive or non-finite one
voids the value instead of producing inf under the numpy error model.

cache=False on purpose: this package is imported both as `data_sources.*` and as
`backend.data_sources.*`, and a numba on-disk cache entry pickles the module name
it was written under, so the second name fails to load the first one's entry.
"""

from __future__ import annotations

import hashlib

import numpy as np
from numba import float64, int64, njit, types

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


PRICE_ADJUSTMENT_ENGINE_VERSION = "price-adjustment-njit-1.0.0"
PRICE_ADJUSTMENT_KERNEL_VERSION = "back-adjust-factor-1"

ORIGIN_NONE = 0
ORIGIN_SOURCE = 1
ORIGIN_PRE_CLOSE = 2

_F1 = float64[::1]
_I1 = int64[::1]
_FACTOR_ORIGIN = types.Tuple((_F1, _I1))
_DIVERGENCE = types.Tuple((int64, float64))
_SUMMARY = types.Tuple((_I1, int64, int64))


@njit(_F1(_I1, _F1), cache=False, nogil=True)
def source_factor_kernel(codes, raw):
    """Normalise a vendor factor to each product's first trading day.

    A product whose factor is missing on any row is voided whole: a partially
    covered series cannot be normalised against a first day it does not have.
    """
    total = raw.size
    out = np.full(total, np.nan)
    if codes.size != total:
        return out
    start = 0
    while start < total:
        stop = start + 1
        while stop < total and codes[stop] == codes[start]:
            stop += 1
        complete = True
        for i in range(start, stop):
            if not np.isfinite(raw[i]) or raw[i] <= 0.0:
                complete = False
                break
        if complete:
            base = raw[start]
            for i in range(start, stop):
                out[i] = raw[i] / base
        start = stop
    return out


@njit(_F1(_I1, _F1, _F1), cache=False, nogil=True)
def pre_close_factor_kernel(codes, close, pre_close):
    """F[0]=1, F[t]=F[t-1]·close[t-1]/pre_close[t] per product.

    pre_close is the ex-rights previous close, so close[t]/pre_close[t] is the
    real dividend-inclusive return and the running product is the back-adjust
    factor. One unusable ratio voids the product from that day on: carrying the
    accumulation across a gap would pass an unadjusted ratio off as adjusted.
    """
    total = close.size
    out = np.full(total, np.nan)
    if codes.size != total or pre_close.size != total:
        return out
    start = 0
    while start < total:
        stop = start + 1
        while stop < total and codes[stop] == codes[start]:
            stop += 1
        factor = 1.0
        out[start] = 1.0
        for i in range(start + 1, stop):
            divisor = pre_close[i]
            if not np.isfinite(divisor) or divisor <= 0.0:
                break
            ratio = close[i - 1] / divisor
            if not np.isfinite(ratio) or ratio <= 0.0:
                break
            factor *= ratio
            out[i] = factor
        start = stop
    return out


@njit(_FACTOR_ORIGIN(_F1, _F1), cache=False, nogil=True)
def combine_factor_kernel(from_source, from_pre_close):
    """The vendor factor wins; the derivation only fills what it leaves empty."""
    total = from_source.size
    factor = np.full(total, np.nan)
    origin = np.zeros(total, dtype=np.int64)
    if from_pre_close.size != total:
        return factor, origin
    for i in range(total):
        if np.isfinite(from_source[i]):
            factor[i] = from_source[i]
            origin[i] = ORIGIN_SOURCE
        elif np.isfinite(from_pre_close[i]):
            factor[i] = from_pre_close[i]
            origin[i] = ORIGIN_PRE_CLOSE
    return factor, origin


@njit(_F1(_F1, _F1), cache=False, nogil=True)
def scale_kernel(values, factor):
    """Raw price × factor. No factor means no adjusted price, never the raw one."""
    total = values.size
    out = np.full(total, np.nan)
    if factor.size != total:
        return out
    for i in range(total):
        out[i] = values[i] * factor[i]
    return out


@njit(_DIVERGENCE(_F1, _F1), cache=False, nogil=True)
def divergence_kernel(left, right):
    """Rows where both factor paths exist, and their largest relative gap."""
    compared = np.int64(0)
    worst = np.nan
    if left.size != right.size:
        return compared, worst
    for i in range(left.size):
        if not np.isfinite(left[i]) or not np.isfinite(right[i]):
            continue
        compared += 1
        if right[i] == 0.0:
            continue
        relative = abs(left[i] - right[i]) / abs(right[i])
        if np.isfinite(relative) and (np.isnan(worst) or relative > worst):
            worst = relative
    return compared, worst


@njit(_SUMMARY(_I1, _I1, _F1), cache=False, nogil=True)
def factor_summary_kernel(codes, origin, factor):
    """Per-product origin counts, products carrying an event, and empty rows.

    A product is counted under the first origin it actually declares, and counts
    as having an adjustment event when its last usable factor is not 1.
    """
    counts = np.zeros(3, dtype=np.int64)
    events = np.int64(0)
    missing = np.int64(0)
    total = factor.size
    if codes.size != total or origin.size != total:
        return counts, events, missing
    for i in range(total):
        if not np.isfinite(factor[i]):
            missing += 1
    start = 0
    while start < total:
        stop = start + 1
        while stop < total and codes[stop] == codes[start]:
            stop += 1
        first = ORIGIN_NONE
        for i in range(start, stop):
            if origin[i] != ORIGIN_NONE:
                first = origin[i]
                break
        counts[first] += 1
        for i in range(stop - 1, start - 1, -1):
            if np.isfinite(factor[i]):
                if round(factor[i], 10) != 1.0:
                    events += 1
                break
        start = stop
    return counts, events, missing


_DISPATCHERS = (
    source_factor_kernel,
    pre_close_factor_kernel,
    combine_factor_kernel,
    scale_kernel,
    divergence_kernel,
    factor_summary_kernel,
)

for _dispatcher in _DISPATCHERS:
    _dispatcher.disable_compile()


def price_adjustment_execution_audit() -> dict[str, object]:
    signatures = {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.signatures]
        for dispatcher in _DISPATCHERS
    }
    material = "|".join(
        [PRICE_ADJUSTMENT_ENGINE_VERSION, PRICE_ADJUSTMENT_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit(
        {
            "engine": PRICE_ADJUSTMENT_ENGINE_VERSION,
            "backend": "numba_njit_fixed_signature",
            "kernel_version": PRICE_ADJUSTMENT_KERNEL_VERSION,
            "kernel_signatures": signatures,
            "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
            "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in _DISPATCHERS),
            "object_mode": 0,
            "python_fallback": 0,
            "request_time_compilation": 0,
            "fully_warmed": all(bool(dispatcher.signatures) for dispatcher in _DISPATCHERS),
        }
    )


def warm_price_adjustment_kernels() -> dict[str, object]:
    """Call every kernel once on a two-product sample and check it answered."""

    codes = np.ascontiguousarray(np.array([0, 0, 0, 1, 1], dtype=np.int64))
    close = np.ascontiguousarray(np.array([10.0, 10.2, 9.9, 3.0, 3.1], dtype=np.float64))
    previous = np.ascontiguousarray(np.array([10.0, 10.0, 9.9, 3.0, 3.0], dtype=np.float64))
    vendor = np.ascontiguousarray(np.array([2.0, 2.0, 2.06, np.nan, 2.0], dtype=np.float64))

    from_source = source_factor_kernel(codes, vendor)
    from_pre_close = pre_close_factor_kernel(codes, close, previous)
    factor, origin = combine_factor_kernel(from_source, from_pre_close)
    adjusted = scale_kernel(close, factor)
    compared, worst = divergence_kernel(from_source, from_pre_close)
    counts, events, missing = factor_summary_kernel(codes, origin, factor)

    if (
        from_source.size != codes.size
        or from_pre_close.size != codes.size
        or factor.size != codes.size
        or origin.size != codes.size
        or adjusted.size != codes.size
        or counts.size != 3
        or compared < 0
        or missing < 0
        or events < 0
        or not np.isfinite(worst)
    ):
        raise RuntimeError("复权价格 NJIT 内核预热失败")
    return price_adjustment_execution_audit()


__all__ = [
    "ORIGIN_NONE",
    "ORIGIN_PRE_CLOSE",
    "ORIGIN_SOURCE",
    "PRICE_ADJUSTMENT_ENGINE_VERSION",
    "PRICE_ADJUSTMENT_KERNEL_VERSION",
    "combine_factor_kernel",
    "divergence_kernel",
    "factor_summary_kernel",
    "pre_close_factor_kernel",
    "price_adjustment_execution_audit",
    "scale_kernel",
    "source_factor_kernel",
    "warm_price_adjustment_kernels",
]
