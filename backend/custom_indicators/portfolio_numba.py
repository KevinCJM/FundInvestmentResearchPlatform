from __future__ import annotations

"""Numerical kernels for versioned portfolio research snapshots."""

import hashlib

import numpy as np
from numba import float64, int64, njit, types, uint8

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


PORTFOLIO_ENGINE_VERSION = "portfolio-research-njit-1.0.0"
PORTFOLIO_KERNEL_VERSION = "path-attribution-risk-2"

_F1 = float64[::1]
_F2 = float64[:, ::1]
_U1 = uint8[::1]
_NORMALIZE_RESULT = types.Tuple((_F1, int64))
_BACKTEST_RESULT = types.Tuple((_F1, _F2, _F2, int64))
_SUMMARY_RESULT = types.Tuple((_F1, _F1, _F1))
_DIAGNOSIS_RESULT = types.Tuple((_F2, _F2, _F1, _F1, _F1, _F1, _F1, _F1))
_CONTEXT_RESULT = types.Tuple((_F1, _F2, int64))


@njit(_F2(_F2), cache=False, nogil=True)
def strict_returns_kernel(nav: np.ndarray) -> np.ndarray:
    rows, assets = nav.shape
    output = np.empty((max(0, rows - 1), assets), dtype=np.float64)
    for row in range(1, rows):
        for asset in range(assets):
            previous = nav[row - 1, asset]
            current = nav[row, asset]
            if not np.isfinite(previous) or previous == 0.0 or not np.isfinite(current):
                raise ValueError("strict portfolio NAV matrix contains invalid values")
            value = current / previous - 1.0
            if not np.isfinite(value):
                raise ValueError("strict portfolio return is not finite")
            output[row - 1, asset] = value
    return output


@njit(_NORMALIZE_RESULT(_F1), cache=False, nogil=True)
def normalize_long_only_weights_kernel(weights: np.ndarray) -> tuple[np.ndarray, int]:
    output = np.empty(weights.size, dtype=np.float64)
    total = 0.0
    for idx in range(weights.size):
        value = weights[idx]
        if not np.isfinite(value) or value < 0.0:
            return np.zeros(weights.size, dtype=np.float64), 1
        output[idx] = value
        total += value
    if total <= 0.0:
        return np.zeros(weights.size, dtype=np.float64), 2
    for idx in range(output.size):
        output[idx] /= total
    return output, 0


@njit(_NORMALIZE_RESULT(_F1, float64), cache=False, nogil=True)
def validate_unit_weights_kernel(
    weights: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """Validate manual long-only weights without silently normalizing them."""

    output = np.empty(weights.size, dtype=np.float64)
    total = 0.0
    for idx in range(weights.size):
        value = weights[idx]
        if not np.isfinite(value) or value < 0.0:
            return np.zeros(weights.size, dtype=np.float64), 1
        output[idx] = value
        total += value
    if total <= 0.0:
        return np.zeros(weights.size, dtype=np.float64), 2
    if abs(total - 1.0) > tolerance:
        return output, 3
    return output, 0


@njit(_F1(int64), cache=False, nogil=True)
def equal_weights_kernel(asset_count: int) -> np.ndarray:
    if asset_count <= 0:
        raise ValueError("asset count must be positive")
    return np.full(asset_count, 1.0 / asset_count, dtype=np.float64)


@njit(_CONTEXT_RESULT(_F2, _F2, float64), cache=False, nogil=True)
def portfolio_context_kernel(
    asset_returns: np.ndarray,
    weight_path: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    rows, assets = asset_returns.shape
    if weight_path.shape != (rows, assets) or rows == 0 or assets == 0:
        return np.empty(0, dtype=np.float64), np.empty((0, 0), dtype=np.float64), 1
    derived = np.empty(rows, dtype=np.float64)
    log_returns = np.empty((rows, assets), dtype=np.float64)
    for row in range(rows):
        weight_total = 0.0
        portfolio_return = 0.0
        for asset in range(assets):
            value = asset_returns[row, asset]
            weight = weight_path[row, asset]
            if not np.isfinite(value) or not np.isfinite(weight) or value <= -1.0:
                return derived[:row], log_returns[:row], 2
            weight_total += weight
            portfolio_return += weight * value
            log_returns[row, asset] = np.log1p(value)
        if abs(weight_total - 1.0) > tolerance:
            return derived[:row], log_returns[:row], 3
        derived[row] = portfolio_return
    return derived, log_returns, 0


@njit(uint8(_F1, _F1, float64, float64), cache=False, nogil=True)
def finite_series_close_kernel(
    left: np.ndarray,
    right: np.ndarray,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> int:
    if left.size != right.size:
        return 0
    for idx in range(left.size):
        left_value = left[idx]
        right_value = right[idx]
        if not np.isfinite(left_value) or not np.isfinite(right_value):
            return 0
        tolerance = absolute_tolerance + relative_tolerance * abs(right_value)
        if abs(left_value - right_value) > tolerance:
            return 0
    return 1


@njit(_BACKTEST_RESULT(_F2, _F2, _U1), cache=False, nogil=True)
def portfolio_drift_backtest_kernel(
    asset_returns: np.ndarray,
    scheduled_weights: np.ndarray,
    schedule_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows, assets = asset_returns.shape
    if scheduled_weights.shape != (rows, assets) or schedule_mask.size != rows:
        raise ValueError("portfolio schedule shape mismatch")
    portfolio_returns = np.empty(rows, dtype=np.float64)
    daily_weights = np.empty((rows, assets), dtype=np.float64)
    contributions = np.empty((rows, assets), dtype=np.float64)
    current = np.zeros(assets, dtype=np.float64)
    active = False
    for row in range(rows):
        if schedule_mask[row] == 1:
            current, status = normalize_long_only_weights_kernel(scheduled_weights[row])
            if status != 0:
                return portfolio_returns[:row], daily_weights[:row], contributions[:row], 10 + status
            active = True
        if not active:
            return portfolio_returns[:row], daily_weights[:row], contributions[:row], 3
        total_return = 0.0
        for asset in range(assets):
            value = asset_returns[row, asset]
            if not np.isfinite(value):
                return portfolio_returns[:row], daily_weights[:row], contributions[:row], 4
            daily_weights[row, asset] = current[asset]
            contribution = current[asset] * value
            contributions[row, asset] = contribution
            total_return += contribution
        if not np.isfinite(total_return) or total_return <= -1.0:
            return portfolio_returns[:row], daily_weights[:row], contributions[:row], 5
        portfolio_returns[row] = total_return
        denominator = 1.0 + total_return
        for asset in range(assets):
            current[asset] = current[asset] * (1.0 + asset_returns[row, asset]) / denominator
    return portfolio_returns, daily_weights, contributions, 0


@njit(cache=False, nogil=True, inline="always")
def _linear_quantile(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(values.copy())
    position = probability * (ordered.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@njit(_SUMMARY_RESULT(_F1, float64, float64), cache=False, nogil=True)
def portfolio_summary_kernel(
    portfolio_returns: np.ndarray,
    periods_per_year: float,
    annual_risk_free_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = portfolio_returns.size
    if count == 0 or periods_per_year <= 0.0:
        raise ValueError("portfolio summary requires returns and positive annualization")
    nav = np.empty(count, dtype=np.float64)
    drawdown = np.empty(count, dtype=np.float64)
    cumulative = 1.0
    peak = 0.0
    max_drawdown = 0.0
    total = 0.0
    for idx in range(count):
        value = portfolio_returns[idx]
        cumulative *= 1.0 + value
        nav[idx] = cumulative
        total += value
        if idx == 0 or cumulative > peak:
            peak = cumulative
        drawdown[idx] = cumulative / peak - 1.0 if peak != 0.0 else 0.0
        if drawdown[idx] < max_drawdown:
            max_drawdown = drawdown[idx]
    annual_return = cumulative ** (periods_per_year / count) - 1.0
    annual_vol = np.nan
    if count > 1:
        mean = total / count
        squared = 0.0
        for value in portfolio_returns:
            delta = value - mean
            squared += delta * delta
        annual_vol = np.sqrt(squared / (count - 1)) * np.sqrt(periods_per_year)
    sharpe = (annual_return - annual_risk_free_rate) / annual_vol if annual_vol > 0.0 else np.nan
    q01 = _linear_quantile(portfolio_returns, 0.01)
    tail_total = 0.0
    tail_count = 0
    for value in portfolio_returns:
        if value <= q01:
            tail_total += value
            tail_count += 1
    metrics = np.array(
        [
            cumulative - 1.0,
            annual_return,
            annual_vol,
            sharpe,
            abs(max_drawdown),
            -q01,
            -(tail_total / tail_count) if tail_count else np.nan,
        ],
        dtype=np.float64,
    )
    return nav, drawdown, metrics


@njit(_DIAGNOSIS_RESULT(_F2, _F2, _F2, float64), cache=False, nogil=True)
def portfolio_diagnosis_kernel(
    asset_returns: np.ndarray,
    daily_weights: np.ndarray,
    contributions: np.ndarray,
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, assets = asset_returns.shape
    if rows < 2 or daily_weights.shape != (rows, assets) or contributions.shape != (rows, assets):
        raise ValueError("portfolio diagnosis matrices are invalid")
    means = np.empty(assets, dtype=np.float64)
    for asset in range(assets):
        total = 0.0
        for row in range(rows):
            total += asset_returns[row, asset]
        means[asset] = total / rows
    covariance = np.empty((assets, assets), dtype=np.float64)
    correlation = np.empty((assets, assets), dtype=np.float64)
    for left in range(assets):
        for right in range(assets):
            cov = 0.0
            left_var = 0.0
            right_var = 0.0
            for row in range(rows):
                left_delta = asset_returns[row, left] - means[left]
                right_delta = asset_returns[row, right] - means[right]
                cov += left_delta * right_delta
                left_var += left_delta * left_delta
                right_var += right_delta * right_delta
            covariance[left, right] = cov / (rows - 1)
            denominator = np.sqrt(left_var * right_var)
            correlation[left, right] = cov / denominator if denominator > 0.0 else 0.0

    current = daily_weights[rows - 1]
    marginal = np.empty(assets, dtype=np.float64)
    component_risk = np.empty(assets, dtype=np.float64)
    risk_shares = np.empty(assets, dtype=np.float64)
    variance = 0.0
    covariance_times_weight = covariance @ current
    for asset in range(assets):
        variance += current[asset] * covariance_times_weight[asset]
    volatility = np.sqrt(max(variance, 0.0))
    component_total = 0.0
    for asset in range(assets):
        if volatility > 0.0:
            marginal[asset] = covariance_times_weight[asset] / volatility
            component_risk[asset] = current[asset] * marginal[asset]
            component_total += component_risk[asset]
        else:
            marginal[asset] = np.nan
            component_risk[asset] = np.nan
    for asset in range(assets):
        risk_shares[asset] = component_risk[asset] / component_total if abs(component_total) > 1e-15 else np.nan

    period_returns = np.empty(assets, dtype=np.float64)
    interval_contributions = np.zeros(assets, dtype=np.float64)
    for asset in range(assets):
        compounded = 1.0
        for row in range(rows):
            compounded *= 1.0 + asset_returns[row, asset]
            interval_contributions[asset] += contributions[row, asset]
        period_returns[asset] = compounded - 1.0

    ordered = np.sort(current.copy())[::-1]
    hhi = 0.0
    top_three = 0.0
    for asset in range(assets):
        hhi += current[asset] * current[asset]
        if asset < 3:
            top_three += ordered[asset]
    concentration = np.array(
        [ordered[0], top_three, hhi, 1.0 / hhi if hhi > 0.0 else np.nan],
        dtype=np.float64,
    )
    return (
        covariance,
        correlation,
        risk_shares,
        marginal,
        component_risk,
        period_returns,
        interval_contributions,
        concentration,
    )


@njit(_F1(_F2), cache=False, nogil=True)
def turnover_path_kernel(weight_path: np.ndarray) -> np.ndarray:
    rows, assets = weight_path.shape
    output = np.empty(rows, dtype=np.float64)
    if rows == 0:
        return output
    output[0] = np.nan
    for row in range(1, rows):
        turnover = 0.0
        for asset in range(assets):
            turnover += abs(weight_path[row, asset] - weight_path[row - 1, asset])
        output[row] = turnover / 2.0
    return output


_KERNELS = (
    strict_returns_kernel,
    normalize_long_only_weights_kernel,
    validate_unit_weights_kernel,
    equal_weights_kernel,
    portfolio_context_kernel,
    finite_series_close_kernel,
    portfolio_drift_backtest_kernel,
    portfolio_summary_kernel,
    portfolio_diagnosis_kernel,
    turnover_path_kernel,
)

for _kernel in _KERNELS:
    _kernel.disable_compile()


def portfolio_numba_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _KERNELS
    }
    material = "|".join(
        [PORTFOLIO_ENGINE_VERSION, PORTFOLIO_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit({
        "engine": PORTFOLIO_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": PORTFOLIO_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(bool(kernel.nopython_signatures) for kernel in _KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
        "request_time_compilation": 0,
        "fully_warmed": all(bool(kernel.signatures) for kernel in _KERNELS),
    })


def warm_portfolio_numba_kernels() -> dict[str, object]:
    nav = np.ascontiguousarray(np.array([[1.0, 1.0], [1.01, 0.99], [1.02, 1.01]]))
    returns = strict_returns_kernel(nav)
    equal = equal_weights_kernel(2)
    normalized, status = normalize_long_only_weights_kernel(np.ascontiguousarray(equal))
    validated, validate_status = validate_unit_weights_kernel(
        np.ascontiguousarray(equal), 1e-8
    )
    portfolio_context_kernel(returns, np.ascontiguousarray(np.vstack((normalized, normalized))), 1e-8)
    finite_series_close_kernel(
        np.ascontiguousarray(np.array([0.0, 0.1], dtype=np.float64)),
        np.ascontiguousarray(np.array([0.0, 0.1], dtype=np.float64)),
        1e-10,
        1e-12,
    )
    schedule = np.ascontiguousarray(np.vstack((normalized, np.zeros(2))))
    mask = np.ascontiguousarray(np.array([1, 0], dtype=np.uint8))
    portfolio_returns, weights, contributions, backtest_status = portfolio_drift_backtest_kernel(
        returns, schedule, mask
    )
    portfolio_summary_kernel(portfolio_returns, 252.0, 0.0)
    portfolio_diagnosis_kernel(returns, weights, contributions, 252.0)
    turnover_path_kernel(np.ascontiguousarray(weights))
    if (
        status != 0
        or validate_status != 0
        or validated.size != equal.size
        or backtest_status != 0
    ):
        raise RuntimeError("组合研究 NJIT 内核预热失败")
    return portfolio_numba_execution_audit()


__all__ = [
    "equal_weights_kernel",
    "finite_series_close_kernel",
    "normalize_long_only_weights_kernel",
    "portfolio_diagnosis_kernel",
    "portfolio_context_kernel",
    "portfolio_drift_backtest_kernel",
    "portfolio_numba_execution_audit",
    "portfolio_summary_kernel",
    "strict_returns_kernel",
    "turnover_path_kernel",
    "validate_unit_weights_kernel",
    "warm_portfolio_numba_kernels",
]
