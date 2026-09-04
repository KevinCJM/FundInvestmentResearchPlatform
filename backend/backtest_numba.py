from __future__ import annotations

"""Fixed-signature Numba kernels for the shared portfolio backtest engine.

Only array-contract validation, date alignment and JSON serialization stay in
Python.  Every numerical path in this module is compiled eagerly in nopython
mode; callers must fail closed instead of substituting a Python calculation.
"""

import hashlib

import numpy as np
from numba import float64, int64, njit, types

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


BACKTEST_ENGINE_VERSION = "portfolio-backtest-njit-1.0.0"
BACKTEST_KERNEL_VERSION = "portfolio-path-metrics-3"


_PATH_RESULT = types.Tuple((float64[:], float64[:]))
_ANNUAL_RESULT = types.Tuple((int64[::1], float64[:, ::1]))


@njit(float64[:](int64), cache=False)
def uniform_weights_kernel(asset_count: int) -> np.ndarray:
    if asset_count <= 0:
        raise ValueError("asset_count must be positive")
    output = np.empty(asset_count, dtype=np.float64)
    weight = 1.0 / asset_count
    for index in range(asset_count):
        output[index] = weight
    return output


@njit(float64[:](float64[:, ::1]), cache=False)
def cumulative_returns_kernel(nav: np.ndarray) -> np.ndarray:
    """Return first-to-last finite NAV return for every series."""

    rows, columns = nav.shape
    output = np.full(columns, np.nan, dtype=np.float64)
    for column in range(columns):
        first = np.nan
        last = np.nan
        for row in range(rows):
            value = nav[row, column]
            if np.isfinite(value):
                if not np.isfinite(first):
                    first = value
                last = value
        if np.isfinite(first) and first != 0.0 and np.isfinite(last):
            output[column] = last / first - 1.0
    return output


@njit(_ANNUAL_RESULT(float64[:, ::1], int64[::1], float64), cache=False)
def annual_metrics_kernel(
    nav: np.ndarray,
    years: np.ndarray,
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-calendar-year metrics for each NAV column.

    The metrics are flattened by ``series * 7`` in this order: cumulative,
    daily volatility, annual return, annual volatility, Sharpe, max drawdown,
    Calmar.  Year boundaries never borrow the prior year's final observation.
    """

    rows, columns = nav.shape
    if years.size != rows:
        raise ValueError("year labels must match NAV rows")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")
    if rows == 0 or columns == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, columns * 7), dtype=np.float64)

    minimum_year = years[0]
    maximum_year = years[0]
    for year in years:
        if year < minimum_year:
            minimum_year = year
        if year > maximum_year:
            maximum_year = year
    if minimum_year < 0 or maximum_year < minimum_year:
        raise ValueError("year labels must be non-negative")
    year_count = maximum_year - minimum_year + 1
    output_years = np.empty(year_count, dtype=np.int64)
    output = np.full((year_count, columns * 7), np.nan, dtype=np.float64)

    for year_index in range(year_count):
        year = minimum_year + year_index
        output_years[year_index] = year
        for column in range(columns):
            first = np.nan
            last = np.nan
            previous = np.nan
            observation_count = 0
            return_count = 0
            return_sum = 0.0
            return_square_sum = 0.0
            peak = np.nan
            max_drawdown = 0.0

            for row in range(rows):
                if years[row] != year:
                    continue
                value = nav[row, column]
                if not np.isfinite(value):
                    continue
                if observation_count == 0:
                    first = value
                    peak = value
                last = value
                observation_count += 1
                if value > peak:
                    peak = value
                if peak != 0.0:
                    drawdown = value / peak - 1.0
                    if drawdown < max_drawdown:
                        max_drawdown = drawdown
                if np.isfinite(previous) and previous != 0.0:
                    period_return = value / previous - 1.0
                    if np.isfinite(period_return):
                        return_sum += period_return
                        return_square_sum += period_return * period_return
                        return_count += 1
                previous = value

            base = column * 7
            if observation_count == 0:
                continue
            output[year_index, base + 5] = max_drawdown
            if observation_count == 1:
                output[year_index, base] = 0.0
                output[year_index, base + 2] = 0.0
                continue
            if first != 0.0 and np.isfinite(last):
                cumulative = last / first - 1.0
                output[year_index, base] = cumulative
                ratio = last / first
                if ratio > 0.0:
                    output[year_index, base + 2] = (
                        ratio ** (periods_per_year / max(return_count, 1)) - 1.0
                    )
            if return_count >= 2:
                mean = return_sum / return_count
                variance = (
                    return_square_sum - return_count * mean * mean
                ) / (return_count - 1)
                if variance < 0.0 and variance > -1e-15:
                    variance = 0.0
                if variance >= 0.0:
                    daily_volatility = np.sqrt(variance)
                    annual_volatility = daily_volatility * np.sqrt(periods_per_year)
                    output[year_index, base + 1] = daily_volatility
                    output[year_index, base + 3] = annual_volatility
                    if annual_volatility > 1e-12:
                        output[year_index, base + 4] = (
                            mean * periods_per_year / annual_volatility
                        )
            annual_return = output[year_index, base + 2]
            if np.isfinite(annual_return) and abs(max_drawdown) > 1e-12:
                output[year_index, base + 6] = annual_return / abs(max_drawdown)

    return output_years, output


@njit(
    _PATH_RESULT(float64[:, ::1], float64[::1], int64, int64, float64, float64),
    cache=False,
)
def portfolio_segment_path_kernel(
    nav: np.ndarray,
    requested_weights: np.ndarray,
    start_idx: int,
    end_idx: int,
    initial_value: float,
    empty_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a buy-and-hold segment and the actually investable weights."""

    row_count, asset_count = nav.shape
    if requested_weights.size != asset_count:
        raise ValueError("weight count does not match asset count")
    if start_idx < 0 or end_idx < start_idx or end_idx >= row_count:
        raise ValueError("invalid portfolio segment bounds")

    normalized = np.zeros(asset_count, dtype=np.float64)
    total = 0.0
    for asset_idx in range(asset_count):
        base = nav[start_idx, asset_idx]
        weight = requested_weights[asset_idx]
        if np.isfinite(base) and base != 0.0 and np.isfinite(weight):
            normalized[asset_idx] = weight
            total += weight

    segment_len = end_idx - start_idx + 1
    path = np.empty(segment_len, dtype=np.float64)
    if total <= 0.0:
        raise ValueError("portfolio weights have no positive investable total")

    for asset_idx in range(asset_count):
        normalized[asset_idx] /= total

    for offset in range(segment_len):
        row_idx = start_idx + offset
        relative_value = 0.0
        for asset_idx in range(asset_count):
            weight = normalized[asset_idx]
            if weight == 0.0:
                continue
            value = nav[row_idx, asset_idx]
            base = nav[start_idx, asset_idx]
            if np.isfinite(value) and base != 0.0:
                relative_value += weight * value / base
        path[offset] = initial_value * relative_value
    return path, normalized


@njit(float64[:](float64[::1], float64), cache=False)
def portfolio_metrics_kernel(nav: np.ndarray, periods_per_year: float) -> np.ndarray:
    """Return annual return/vol, Sharpe, 99% VaR/ES, MDD and Calmar."""

    result = np.empty(7, dtype=np.float64)
    for idx in range(result.size):
        result[idx] = np.nan
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")

    finite_count = 0
    for value in nav:
        if np.isfinite(value):
            finite_count += 1
    if finite_count < 2:
        return result

    compact = np.empty(finite_count, dtype=np.float64)
    pos = 0
    for value in nav:
        if np.isfinite(value):
            compact[pos] = value
            pos += 1

    raw_returns = np.empty(finite_count - 1, dtype=np.float64)
    valid_return_count = 0
    for idx in range(1, finite_count):
        previous = compact[idx - 1]
        current = compact[idx]
        if previous != 0.0:
            value = current / previous - 1.0
            if np.isfinite(value):
                raw_returns[valid_return_count] = value
                valid_return_count += 1
    if valid_return_count == 0:
        return result

    returns = raw_returns[:valid_return_count]
    total = 0.0
    for value in returns:
        total += value
    mean = total / valid_return_count
    annual_return = mean * periods_per_year

    annual_vol = np.nan
    if valid_return_count > 1:
        variance_sum = 0.0
        for value in returns:
            delta = value - mean
            variance_sum += delta * delta
        annual_vol = np.sqrt(variance_sum / (valid_return_count - 1)) * np.sqrt(periods_per_year)

    ordered = np.sort(returns.copy())
    position = 0.01 * (valid_return_count - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    quantile = ordered[lower]
    if upper != lower:
        fraction = position - lower
        quantile = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
    var99 = -quantile
    tail_sum = 0.0
    tail_count = 0
    for value in returns:
        if value <= quantile:
            tail_sum += value
            tail_count += 1
    es99 = -(tail_sum / tail_count) if tail_count > 0 else np.nan

    peak = compact[0]
    max_drawdown = 0.0
    for value in compact:
        if value > peak:
            peak = value
        if peak != 0.0:
            drawdown = value / peak - 1.0
            if drawdown < max_drawdown:
                max_drawdown = drawdown

    sharpe = annual_return / annual_vol if np.isfinite(annual_vol) and annual_vol != 0.0 else np.nan
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0.0 else np.nan
    result[0] = annual_return
    result[1] = annual_vol
    result[2] = sharpe
    result[3] = var99
    result[4] = es99
    result[5] = max_drawdown
    result[6] = calmar
    return result


_BACKTEST_KERNELS = (
    uniform_weights_kernel,
    cumulative_returns_kernel,
    annual_metrics_kernel,
    portfolio_segment_path_kernel,
    portfolio_metrics_kernel,
)

for _kernel in _BACKTEST_KERNELS:
    _kernel.disable_compile()


def backtest_numba_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _BACKTEST_KERNELS
    }
    fingerprint_material = "|".join(
        [BACKTEST_ENGINE_VERSION, BACKTEST_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit({
        "engine": BACKTEST_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": BACKTEST_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(fingerprint_material.encode("utf-8")).hexdigest(),
        "nopython": all(bool(kernel.nopython_signatures) for kernel in _BACKTEST_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
        "request_time_compilation": 0,
        "fully_warmed": all(bool(kernel.signatures) for kernel in _BACKTEST_KERNELS),
    })


def warm_backtest_numba_kernels() -> dict[str, object]:
    nav = np.ascontiguousarray(
        np.array([[1.0, 1.0], [1.01, 0.99], [1.02, 1.01]], dtype=np.float64)
    )
    weights = np.ascontiguousarray(np.array([0.5, 0.5], dtype=np.float64))
    uniform = uniform_weights_kernel(2)
    cumulative = cumulative_returns_kernel(nav)
    annual_years, annual = annual_metrics_kernel(
        nav,
        np.ascontiguousarray(np.array([2024, 2024, 2025], dtype=np.int64)),
        252.0,
    )
    path, normalized = portfolio_segment_path_kernel(nav, weights, 0, 2, 1.0, 0.0)
    metrics = portfolio_metrics_kernel(np.ascontiguousarray(path), 252.0)
    if (
        uniform.size != 2
        or cumulative.size != 2
        or annual_years.size != 2
        or annual.shape != (2, 14)
        or path.size != 3
        or normalized.size != 2
        or metrics.size != 7
    ):
        raise RuntimeError("组合回测 NJIT 内核预热失败")
    audit = backtest_numba_execution_audit()
    if not audit["nopython"] or audit["python_fallback"] != 0:
        raise RuntimeError("组合回测 NJIT 内核未进入 nopython 模式")
    return audit
