from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np
import pandas as pd
from numba import njit, types
from numba.core.registry import CPUDispatcher


OPTIMIZER_NUMBA_KERNEL_VERSION = "2.0.0"
_FLOAT64_1D = types.float64[::1]
_FLOAT64_2D = types.float64[:, ::1]
_INT64_1D = types.int64[::1]
_UINT8_2D = types.uint8[:, ::1]
_PORTFOLIO_RESULT = types.UniTuple(types.float64, 2)


@njit(_FLOAT64_1D(_FLOAT64_1D), cache=False, nogil=True)
def get_log_returns(nav_series: np.ndarray) -> np.ndarray:
    output = np.empty(max(nav_series.size - 1, 0), dtype=np.float64)
    for index in range(output.size):
        output[index] = np.log(nav_series[index + 1] / nav_series[index])
    return output


@njit(_FLOAT64_1D(_FLOAT64_1D), cache=False, nogil=True)
def get_simple_returns(nav_series: np.ndarray) -> np.ndarray:
    output = np.empty(max(nav_series.size - 1, 0), dtype=np.float64)
    for index in range(output.size):
        output[index] = nav_series[index + 1] / nav_series[index] - 1.0
    return output


@njit((_FLOAT64_2D, types.int64), cache=False, nogil=True)
def nav_matrix_returns_kernel(
    nav_values: np.ndarray,
    return_type_code: int,
) -> tuple[np.ndarray, int]:
    row_count, asset_count = nav_values.shape
    output = np.empty((max(row_count - 1, 0), asset_count), dtype=np.float64)
    if row_count < 2 or asset_count == 0:
        return output, 1
    for row_index in range(row_count - 1):
        for asset_index in range(asset_count):
            previous = nav_values[row_index, asset_index]
            current = nav_values[row_index + 1, asset_index]
            if not np.isfinite(previous) or not np.isfinite(current) or previous <= 0.0 or current <= 0.0:
                return output, 2
            ratio = current / previous
            output[row_index, asset_index] = np.log(ratio) if return_type_code == 1 else ratio - 1.0
    return output, 0


@njit(
    types.float64(_FLOAT64_1D, types.int64, types.float64, types.float64),
    cache=False,
    nogil=True,
)
def calculate_return_kernel(
    returns: np.ndarray,
    metric_code: int,
    periods_per_year: float,
    decay: float,
) -> float:
    if returns.size == 0:
        return np.nan
    total = 0.0
    for value in returns:
        if not np.isfinite(value):
            return np.nan
        total += value
    if metric_code == 1:
        return total
    mean = total / returns.size
    if metric_code == 2:
        return mean * periods_per_year
    if metric_code == 3:
        lam = min(max(decay, 0.0), 0.999999)
        weighted = returns[0]
        for index in range(1, returns.size):
            weighted = lam * weighted + (1.0 - lam) * returns[index]
        return weighted
    return mean


@njit(
    types.float64(
        _FLOAT64_1D,
        types.int64,
        types.float64,
        types.float64,
        types.int64,
        types.float64,
    ),
    cache=False,
    nogil=True,
)
def calculate_risk_kernel(
    returns: np.ndarray,
    metric_code: int,
    periods_per_year: float,
    decay: float,
    window: int,
    confidence: float,
) -> float:
    size = returns.size
    if size == 0:
        return np.nan
    for value in returns:
        if not np.isfinite(value):
            return np.nan
    if metric_code == 2:
        start = max(0, size - max(2, window)) if window > 0 else 0
        mean = returns[start]
        variance = 0.0
        lam = min(max(decay, 0.0), 0.999999)
        for index in range(start + 1, size):
            delta = returns[index] - mean
            mean = lam * mean + (1.0 - lam) * returns[index]
            variance = lam * variance + (1.0 - lam) * delta * delta
        return np.sqrt(max(variance, 0.0)) * np.sqrt(periods_per_year)
    if metric_code == 3:
        count = 0
        total = 0.0
        for value in returns:
            if value < 0.0:
                count += 1
                total += value
        if count < 2:
            return 0.0
        mean = total / count
        squared = 0.0
        for value in returns:
            if value < 0.0:
                delta = value - mean
                squared += delta * delta
        return np.sqrt(squared / (count - 1))
    if metric_code == 4 or metric_code == 5:
        ordered = np.sort(returns.copy())
        probability = 1.0 - min(max(confidence, 0.0), 1.0)
        position = probability * (size - 1)
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        quantile = ordered[lower]
        if upper != lower:
            fraction = position - lower
            quantile = ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
        if metric_code == 4:
            return -quantile
        tail_total = 0.0
        tail_count = 0
        for value in returns:
            if value <= quantile:
                tail_total += value
                tail_count += 1
        return -(tail_total / tail_count) if tail_count > 0 else 0.0
    if metric_code == 6:
        nav = 1.0
        peak = 1.0
        maximum_loss = 0.0
        for value in returns:
            if value <= -1.0:
                return np.nan
            nav *= 1.0 + value
            if nav > peak:
                peak = nav
            loss = 1.0 - nav / peak
            if loss > maximum_loss:
                maximum_loss = loss
        return maximum_loss
    if size < 2:
        return 0.0
    total = 0.0
    for value in returns:
        total += value
    mean = total / size
    squared = 0.0
    for value in returns:
        delta = value - mean
        squared += delta * delta
    volatility = np.sqrt(squared / (size - 1))
    return volatility * np.sqrt(periods_per_year) if metric_code == 1 else volatility


@njit(
    _PORTFOLIO_RESULT(_FLOAT64_1D, _FLOAT64_1D, _FLOAT64_2D),
    cache=False,
    nogil=True,
)
def compute_portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> tuple[float, float]:
    portfolio_return = 0.0
    portfolio_variance = 0.0
    for left in range(weights.size):
        portfolio_return += mean_returns[left] * weights[left]
        for right in range(weights.size):
            portfolio_variance += weights[left] * cov_matrix[left, right] * weights[right]
    return portfolio_return, np.sqrt(max(portfolio_variance, 0.0))


@njit(
    _FLOAT64_2D(types.int64, types.int64, _FLOAT64_1D, _FLOAT64_2D),
    cache=False,
    nogil=True,
)
def generate_random_portfolios(
    n_portfolios: int,
    n_assets: int,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
) -> np.ndarray:
    if n_portfolios < 0 or n_assets <= 0:
        raise ValueError("portfolio and asset counts must be positive")
    results = np.zeros((2, n_portfolios), dtype=np.float64)
    for portfolio_index in range(n_portfolios):
        weights = np.random.random(n_assets)
        total = np.sum(weights)
        if total <= 0.0:
            raise ValueError("random weight generation failed")
        weights /= total
        portfolio_return, portfolio_volatility = compute_portfolio_performance(
            weights, mean_returns, cov_matrix
        )
        results[0, portfolio_index] = portfolio_volatility
        results[1, portfolio_index] = portfolio_return
    return results


@njit((_FLOAT64_2D, _FLOAT64_1D), cache=False, nogil=True)
def portfolio_returns_kernel(
    asset_returns: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    output = np.empty(asset_returns.shape[0], dtype=np.float64)
    for row_index in range(asset_returns.shape[0]):
        total = 0.0
        for asset_index in range(asset_returns.shape[1]):
            total += asset_returns[row_index, asset_index] * weights[asset_index]
        output[row_index] = total
    return output


@njit(
    (_FLOAT64_1D, _FLOAT64_2D, _UINT8_2D, _FLOAT64_1D, _FLOAT64_1D, types.float64),
    cache=False,
    nogil=True,
)
def repair_weights_kernel(
    raw_weights: np.ndarray,
    single_bounds: np.ndarray,
    group_membership: np.ndarray,
    group_lows: np.ndarray,
    group_highs: np.ndarray,
    quantize_step: float,
) -> tuple[np.ndarray, int]:
    """Project one long-only vector into single/group bounds without fallback."""

    asset_count = raw_weights.size
    output = raw_weights.copy()
    if asset_count == 0 or single_bounds.shape[0] != asset_count:
        return output, 1
    lower_total = 0.0
    upper_total = 0.0
    for asset_index in range(asset_count):
        lower = single_bounds[asset_index, 0]
        upper = single_bounds[asset_index, 1]
        if (
            not np.isfinite(lower)
            or not np.isfinite(upper)
            or lower < 0.0
            or upper > 1.0
            or lower > upper
        ):
            return output, 1
        lower_total += lower
        upper_total += upper
    if lower_total > 1.0 + 1e-10 or upper_total < 1.0 - 1e-10:
        return output, 2

    if quantize_step > 0.0:
        for asset_index in range(asset_count):
            output[asset_index] = np.round(output[asset_index] / quantize_step) * quantize_step
    for asset_index in range(asset_count):
        value = output[asset_index]
        if not np.isfinite(value):
            return output, 1
        output[asset_index] = min(max(value, single_bounds[asset_index, 0]), single_bounds[asset_index, 1])

    for _ in range(80):
        deficit = 1.0 - np.sum(output)
        if deficit > 1e-12:
            room_total = 0.0
            for asset_index in range(asset_count):
                room_total += max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
            if room_total <= 1e-14:
                return output, 2
            for asset_index in range(asset_count):
                room = max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
                output[asset_index] += deficit * room / room_total
        elif deficit < -1e-12:
            removable_total = 0.0
            for asset_index in range(asset_count):
                removable_total += max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
            if removable_total <= 1e-14:
                return output, 2
            for asset_index in range(asset_count):
                removable = max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
                output[asset_index] += deficit * removable / removable_total

        any_group_violation = False
        for group_index in range(group_membership.shape[0]):
            group_total = 0.0
            for asset_index in range(asset_count):
                if group_membership[group_index, asset_index] != 0:
                    group_total += output[asset_index]
            lower = group_lows[group_index]
            upper = group_highs[group_index]
            if group_total < lower - 1e-11:
                any_group_violation = True
                needed = lower - group_total
                inside_room = 0.0
                outside_available = 0.0
                for asset_index in range(asset_count):
                    if group_membership[group_index, asset_index] != 0:
                        inside_room += max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
                    else:
                        outside_available += max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
                transfer = min(needed, min(inside_room, outside_available))
                if transfer <= 1e-14:
                    return output, 3
                for asset_index in range(asset_count):
                    if group_membership[group_index, asset_index] != 0:
                        room = max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
                        output[asset_index] += transfer * room / inside_room
                    else:
                        removable = max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
                        output[asset_index] -= transfer * removable / outside_available
            elif group_total > upper + 1e-11:
                any_group_violation = True
                needed = group_total - upper
                inside_available = 0.0
                outside_room = 0.0
                for asset_index in range(asset_count):
                    if group_membership[group_index, asset_index] != 0:
                        inside_available += max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
                    else:
                        outside_room += max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
                transfer = min(needed, min(inside_available, outside_room))
                if transfer <= 1e-14:
                    return output, 3
                for asset_index in range(asset_count):
                    if group_membership[group_index, asset_index] != 0:
                        removable = max(output[asset_index] - single_bounds[asset_index, 0], 0.0)
                        output[asset_index] -= transfer * removable / inside_available
                    else:
                        room = max(single_bounds[asset_index, 1] - output[asset_index], 0.0)
                        output[asset_index] += transfer * room / outside_room
        if not any_group_violation and abs(np.sum(output) - 1.0) <= 1e-10:
            break

    if abs(np.sum(output) - 1.0) > 1e-7:
        return output, 4
    for asset_index in range(asset_count):
        if output[asset_index] < single_bounds[asset_index, 0] - 1e-7 or output[asset_index] > single_bounds[asset_index, 1] + 1e-7:
            return output, 4
    for group_index in range(group_membership.shape[0]):
        group_total = 0.0
        for asset_index in range(asset_count):
            if group_membership[group_index, asset_index] != 0:
                group_total += output[asset_index]
        if group_total < group_lows[group_index] - 1e-7 or group_total > group_highs[group_index] + 1e-7:
            return output, 4
    return output, 0


@njit(
    (
        _FLOAT64_2D, _FLOAT64_2D, _UINT8_2D, _FLOAT64_1D, _FLOAT64_1D,
        _INT64_1D, _FLOAT64_1D, _INT64_1D, types.float64, types.int64,
        types.int64, types.int64, types.float64, types.float64, types.float64,
        types.float64, types.int64, types.float64, types.float64, types.int64,
        types.float64, types.float64,
    ),
    cache=False,
    nogil=True,
)
def explore_portfolios_kernel(
    asset_returns: np.ndarray,
    single_bounds: np.ndarray,
    group_membership: np.ndarray,
    group_lows: np.ndarray,
    group_highs: np.ndarray,
    round_samples: np.ndarray,
    round_steps: np.ndarray,
    round_buckets: np.ndarray,
    quantize_step: float,
    seed: int,
    return_metric_code: int,
    risk_metric_code: int,
    return_periods: float,
    risk_periods: float,
    return_decay: float,
    risk_decay: float,
    risk_window: int,
    confidence: float,
    risk_free_rate: float,
    target_code: int,
    target_return: float,
    target_risk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, int, np.ndarray, int]:
    requested_count = 0
    for value in round_samples:
        if value > 0:
            requested_count += value
    asset_count = asset_returns.shape[1]
    weights_output = np.full((requested_count, asset_count), np.nan, dtype=np.float64)
    risks = np.full(requested_count, np.nan, dtype=np.float64)
    returns = np.full(requested_count, np.nan, dtype=np.float64)
    frontier_indices = np.full(requested_count, -1, dtype=np.int64)
    special_indices = np.full(4, -1, dtype=np.int64)
    if requested_count <= 0 or asset_count <= 0 or asset_returns.shape[0] <= 0:
        return weights_output, risks, returns, 0, frontier_indices, 0, special_indices, 1

    np.random.seed(seed)
    accepted = 0
    attempted = 0
    for round_index in range(round_samples.size):
        samples = max(round_samples[round_index], 0)
        step = min(max(round_steps[round_index], 0.0), 1.0)
        bucket_span = max(round_buckets[round_index], 1) * 4
        for _ in range(samples):
            raw = np.empty(asset_count, dtype=np.float64)
            if attempted == 0:
                for asset_index in range(asset_count):
                    raw[asset_index] = 1.0
            else:
                raw_total = 0.0
                for asset_index in range(asset_count):
                    uniform = max(np.random.random(), 1e-15)
                    raw[asset_index] = -np.log(uniform)
                    raw_total += raw[asset_index]
                if raw_total <= 0.0:
                    return weights_output, risks, returns, accepted, frontier_indices, 0, special_indices, 2
                raw /= raw_total
                if round_index > 0 and accepted > 0:
                    span = min(accepted, bucket_span)
                    offset = int(np.random.random() * span)
                    base_index = accepted - 1 - min(offset, span - 1)
                    for asset_index in range(asset_count):
                        raw[asset_index] = (1.0 - step) * weights_output[base_index, asset_index] + step * raw[asset_index]
            attempted += 1
            repaired, repair_status = repair_weights_kernel(
                raw, single_bounds, group_membership, group_lows, group_highs, quantize_step
            )
            if repair_status != 0:
                continue
            portfolio_returns = portfolio_returns_kernel(asset_returns, repaired)
            return_value = calculate_return_kernel(portfolio_returns, return_metric_code, return_periods, return_decay)
            risk_value = calculate_risk_kernel(
                portfolio_returns, risk_metric_code, risk_periods, risk_decay, risk_window, confidence
            )
            if not np.isfinite(return_value) or not np.isfinite(risk_value):
                continue
            weights_output[accepted] = repaired
            risks[accepted] = risk_value
            returns[accepted] = return_value
            accepted += 1

    if accepted == 0:
        return weights_output, risks, returns, 0, frontier_indices, 0, special_indices, 3

    order = np.argsort(risks[:accepted])
    frontier_count = 0
    last_return = -np.inf
    position = 0
    while position < accepted:
        first_index = order[position]
        rounded_risk = np.round(risks[first_index] * 10000.0) / 10000.0
        best_index = first_index
        scan = position + 1
        while scan < accepted:
            candidate_index = order[scan]
            candidate_rounded = np.round(risks[candidate_index] * 10000.0) / 10000.0
            if candidate_rounded != rounded_risk:
                break
            if returns[candidate_index] > returns[best_index]:
                best_index = candidate_index
            scan += 1
        if returns[best_index] > last_return:
            frontier_indices[frontier_count] = best_index
            frontier_count += 1
            last_return = returns[best_index]
        position = scan

    minimum_risk_index = 0
    maximum_return_index = 0
    maximum_sharpe_index = -1
    maximum_sharpe = -np.inf
    target_index = -1
    target_score = np.inf
    for index in range(accepted):
        if risks[index] < risks[minimum_risk_index]:
            minimum_risk_index = index
        if returns[index] > returns[maximum_return_index]:
            maximum_return_index = index
        if risks[index] > 1e-12:
            sharpe = (returns[index] - risk_free_rate) / risks[index]
            if sharpe > maximum_sharpe:
                maximum_sharpe = sharpe
                maximum_sharpe_index = index
        if target_code == 0 and risks[index] < target_score:
            target_score = risks[index]
            target_index = index
        elif target_code == 1 and -returns[index] < target_score:
            target_score = -returns[index]
            target_index = index
        elif target_code == 2 and risks[index] > 1e-12:
            score = -returns[index] / risks[index]
            if score < target_score:
                target_score = score
                target_index = index
        elif target_code == 3 and risks[index] > 1e-12:
            score = -(returns[index] - risk_free_rate) / risks[index]
            if score < target_score:
                target_score = score
                target_index = index
        elif target_code == 4 and returns[index] >= target_return - 1e-10 and risks[index] < target_score:
            target_score = risks[index]
            target_index = index
        elif target_code == 5 and risks[index] <= target_risk + 1e-10 and -returns[index] < target_score:
            target_score = -returns[index]
            target_index = index

    special_indices[0] = maximum_sharpe_index
    special_indices[1] = minimum_risk_index
    special_indices[2] = maximum_return_index
    special_indices[3] = target_index
    status = 4 if target_code >= 0 and target_index < 0 else 0
    return weights_output, risks, returns, accepted, frontier_indices, frontier_count, special_indices, status


OPTIMIZER_NUMBA_KERNELS: tuple[CPUDispatcher, ...] = (
    get_log_returns,
    get_simple_returns,
    nav_matrix_returns_kernel,
    calculate_return_kernel,
    calculate_risk_kernel,
    compute_portfolio_performance,
    generate_random_portfolios,
    portfolio_returns_kernel,
    repair_weights_kernel,
    explore_portfolios_kernel,
)
for _dispatcher in OPTIMIZER_NUMBA_KERNELS:
    _dispatcher.disable_compile()
_OPTIMIZER_WARMED = False


def _kernel_fingerprint(dispatcher: CPUDispatcher) -> str:
    return hashlib.sha256(inspect.getsource(dispatcher.py_func).encode("utf-8")).hexdigest()


def optimizer_numba_status(*, warmed: Optional[bool] = None) -> dict[str, Any]:
    signatures = {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.nopython_signatures]
        for dispatcher in OPTIMIZER_NUMBA_KERNELS
    }
    fingerprints = {
        dispatcher.py_func.__name__: _kernel_fingerprint(dispatcher)
        for dispatcher in OPTIMIZER_NUMBA_KERNELS
    }
    aggregate_fingerprint = hashlib.sha256(
        "|".join(f"{name}:{fingerprints[name]}" for name in sorted(fingerprints)).encode("utf-8")
    ).hexdigest()
    ready = sum(bool(items) for items in signatures.values())
    is_warmed = _OPTIMIZER_WARMED if warmed is None else bool(warmed)
    return {
        "version": OPTIMIZER_NUMBA_KERNEL_VERSION,
        "kernel_version": OPTIMIZER_NUMBA_KERNEL_VERSION,
        "numba_version": numba.__version__,
        "warmed": is_warmed,
        "fully_warmed": is_warmed and ready == len(signatures),
        "kernel_coverage": f"{ready}/{len(signatures)}",
        "kernel_signatures": signatures,
        "compiled_signatures": signatures,
        "kernel_fingerprints": fingerprints,
        "fingerprint": aggregate_fingerprint,
        "backend": "numba_njit_fixed_signature",
        "execution_backend": "numba_njit_fixed_signature",
        "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in OPTIMIZER_NUMBA_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
    }


def _validate_optimizer_audit(audit: dict[str, Any]) -> dict[str, Any]:
    try:
        from backend.compute_policy import validate_execution_audit
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from compute_policy import validate_execution_audit
    return validate_execution_audit(audit)


@lru_cache(maxsize=1)
def warm_optimizer_numba_kernels() -> dict[str, Any]:
    """Exercise and validate every eager fixed-signature optimization lane."""

    global _OPTIMIZER_WARMED
    nav = np.ascontiguousarray([1.0, 1.01, 1.02], dtype=np.float64)
    weights = np.ascontiguousarray([0.5, 0.5], dtype=np.float64)
    means = np.ascontiguousarray([0.05, 0.08], dtype=np.float64)
    covariance = np.ascontiguousarray([[0.10, 0.02], [0.02, 0.20]], dtype=np.float64)
    returns = np.ascontiguousarray([[0.01, 0.00], [-0.01, 0.02], [0.005, 0.004]], dtype=np.float64)
    bounds = np.ascontiguousarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    groups = np.ascontiguousarray(np.zeros((0, 2), dtype=np.uint8))
    empty_float = np.ascontiguousarray(np.empty(0, dtype=np.float64))
    samples = np.ascontiguousarray([2], dtype=np.int64)
    steps = np.ascontiguousarray([1.0], dtype=np.float64)
    buckets = np.ascontiguousarray([2], dtype=np.int64)
    get_log_returns(nav)
    get_simple_returns(nav)
    nav_matrix_returns_kernel(returns, np.int64(0))
    calculate_return_kernel(nav, np.int64(0), np.float64(252.0), np.float64(0.94))
    calculate_risk_kernel(nav, np.int64(0), np.float64(252.0), np.float64(0.94), np.int64(60), np.float64(0.95))
    compute_portfolio_performance(weights, means, covariance)
    generate_random_portfolios(np.int64(2), np.int64(2), means, covariance)
    portfolio_returns_kernel(returns, weights)
    repair_weights_kernel(weights, bounds, groups, empty_float, empty_float, np.float64(0.0))
    explore_portfolios_kernel(
        returns, bounds, groups, empty_float, empty_float, samples, steps, buckets,
        np.float64(0.0), np.int64(42), np.int64(0), np.int64(0),
        np.float64(252.0), np.float64(252.0), np.float64(0.94), np.float64(0.94),
        np.int64(60), np.float64(0.95), np.float64(0.0), np.int64(-1),
        np.float64(0.0), np.float64(0.0),
    )
    if any(len(dispatcher.nopython_signatures) != 1 for dispatcher in OPTIMIZER_NUMBA_KERNELS):
        raise RuntimeError("optimizer NJIT kernels must each expose exactly one fixed signature")
    _OPTIMIZER_WARMED = True
    return _validate_optimizer_audit(optimizer_numba_status())


def _return_metric_code(config: Dict[str, Any]) -> int:
    metric = str(config.get("metric") or "mean").lower()
    mapping = {"mean": 0, "cumulative": 1, "annual": 2, "annual_mean": 2, "ewm": 3}
    if metric not in mapping:
        raise ValueError(f"不支持的收益指标：{metric}")
    return mapping[metric]


def _risk_metric_code(config: Dict[str, Any]) -> int:
    metric = str(config.get("metric") or "vol").lower()
    mapping = {
        "vol": 0, "std": 0, "annual_vol": 1, "ewm_vol": 2,
        "downside_vol": 3, "var": 4, "es": 5, "max_drawdown": 6,
    }
    if metric not in mapping:
        raise ValueError(f"不支持的风险指标：{metric}")
    return mapping[metric]


def calculate_return(returns: np.ndarray, config: Dict[str, Any]) -> float:
    values = np.ascontiguousarray(np.asarray(returns, dtype=np.float64).reshape(-1))
    metric_code = _return_metric_code(config)
    periods = float(252 if config.get("days") is None else config["days"])
    decay = float(0.94 if config.get("alpha") is None else config["alpha"])
    if periods <= 0.0:
        raise ValueError("年化周期数必须大于 0")
    if metric_code == 3 and not 0.0 <= decay < 1.0:
        raise ValueError("指数加权衰减因子必须位于 [0, 1) 区间")
    value = calculate_return_kernel(
        values,
        np.int64(metric_code),
        np.float64(periods),
        np.float64(decay),
    )
    if not np.isfinite(value):
        raise ValueError("收益序列包含非有限值，无法计算收益指标")
    return float(value)


def returns_from_nav_matrix(nav_values: np.ndarray, return_type: str = "simple") -> np.ndarray:
    return_type_code = {"simple": 0, "log": 1}.get(str(return_type).lower())
    if return_type_code is None:
        raise ValueError(f"不支持的收益率类型：{return_type}")
    values = np.ascontiguousarray(np.asarray(nav_values, dtype=np.float64))
    if values.ndim != 2:
        raise ValueError("净值必须是二维矩阵")
    output, status = nav_matrix_returns_kernel(values, np.int64(return_type_code))
    if status == 1:
        raise ValueError("净值矩阵至少需要两个观察值和一个资产")
    if status == 2:
        raise ValueError("净值矩阵包含缺失、非有限或非正值")
    return np.ascontiguousarray(output, dtype=np.float64)


def calculate_risk(returns: np.ndarray, config: Dict[str, Any]) -> float:
    values = np.ascontiguousarray(np.asarray(returns, dtype=np.float64).reshape(-1))
    metric_code = _risk_metric_code(config)
    periods = float(252 if config.get("days") is None else config["days"])
    decay = float(0.94 if config.get("alpha") is None else config["alpha"])
    window = int(60 if config.get("window") is None else config["window"])
    if periods <= 0.0:
        raise ValueError("年化周期数必须大于 0")
    if metric_code == 2 and not 0.0 <= decay < 1.0:
        raise ValueError("指数加权衰减因子必须位于 [0, 1) 区间")
    if metric_code == 2 and window == 1:
        raise ValueError("指数加权窗口必须为 0 或至少 2")
    confidence = float(95 if config.get("confidence") is None else config["confidence"])
    if confidence > 1.0:
        confidence /= 100.0
    if metric_code in {4, 5} and not 0.0 < confidence < 1.0:
        raise ValueError("置信水平必须位于 0 与 1 之间")
    value = calculate_risk_kernel(
        values,
        np.int64(metric_code),
        np.float64(periods),
        np.float64(decay),
        np.int64(window),
        np.float64(confidence),
    )
    if not np.isfinite(value):
        raise ValueError("收益序列无法形成有效风险指标")
    return float(value)


def _constraint_arrays(
    asset_count: int,
    single_limits: Optional[List[Tuple[float, float]]],
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    limits = [(0.0, 1.0) for _ in range(asset_count)] if single_limits is None else single_limits
    if len(limits) != asset_count:
        raise ValueError("单资产约束数量必须与资产数量一致")
    bounds = np.ascontiguousarray(limits, dtype=np.float64)
    if not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] < 0.0) or np.any(bounds[:, 1] > 1.0) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("单资产权重约束必须满足 0 ≤ lo ≤ hi ≤ 1")
    groups = group_limits or {}
    membership = np.zeros((len(groups), asset_count), dtype=np.uint8)
    lows = np.empty(len(groups), dtype=np.float64)
    highs = np.empty(len(groups), dtype=np.float64)
    for group_index, (indices, limits_pair) in enumerate(groups.items()):
        if not indices:
            raise ValueError("组合约束必须至少包含一个资产")
        for asset_index in indices:
            if asset_index < 0 or asset_index >= asset_count:
                raise ValueError("组合约束包含超出资产范围的索引")
            membership[group_index, asset_index] = 1
        lows[group_index] = float(limits_pair[0])
        highs[group_index] = float(limits_pair[1])
        if not (0.0 <= lows[group_index] <= highs[group_index] <= 1.0):
            raise ValueError("组合权重约束必须满足 0 ≤ lo ≤ hi ≤ 1")
    return np.ascontiguousarray(bounds), np.ascontiguousarray(membership), np.ascontiguousarray(lows), np.ascontiguousarray(highs)


def _round_arrays(
    rounds: Optional[List[Dict[str, Any]]],
    *,
    use_refine: bool,
    refine_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    configured = rounds or [{"samples": 100, "step": 0.99, "buckets": 50}]
    samples: list[int] = []
    steps: list[float] = []
    buckets: list[int] = []
    for item in configured:
        count = int(item.get("samples", 100))
        if count <= 0:
            raise ValueError("每轮候选数量必须大于 0")
        step = float(item.get("step", 0.5))
        if not 0.0 <= step <= 1.0:
            raise ValueError("候选搜索步长必须位于 [0, 1] 区间")
        samples.append(count)
        steps.append(step)
        buckets.append(max(1, int(item.get("buckets", 50))))
    if use_refine and refine_count > 0:
        samples.append(int(refine_count))
        steps.append(0.05)
        buckets.append(max(2, int(refine_count)))
    if sum(samples) > 200_000:
        raise ValueError("候选组合总数不能超过 200000")
    return (
        np.ascontiguousarray(samples, dtype=np.int64),
        np.ascontiguousarray(steps, dtype=np.float64),
        np.ascontiguousarray(buckets, dtype=np.int64),
    )


def _target_code(target: Optional[str]) -> int:
    if target is None:
        return -1
    mapping = {
        "min_risk": 0, "max_return": 1, "max_sharpe": 2,
        "max_sharpe_traditional": 3, "risk_min_given_return": 4,
        "return_max_given_risk": 5,
    }
    if target not in mapping:
        raise ValueError(f"不支持的优化目标：{target}")
    return mapping[target]


def _run_exploration(
    asset_returns: np.ndarray,
    return_config: Dict[str, Any],
    risk_config: Dict[str, Any],
    *,
    single_limits: Optional[List[Tuple[float, float]]],
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]],
    rounds: Optional[List[Dict[str, Any]]],
    quantize_step: Optional[float],
    use_refine: bool,
    refine_count: int,
    risk_free_rate: float,
    seed: int,
    target: Optional[str],
    target_return: Optional[float],
    target_risk: Optional[float],
):
    values = np.ascontiguousarray(np.asarray(asset_returns, dtype=np.float64))
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("收益矩阵必须至少包含一个观察值和一个资产")
    if not np.all(np.isfinite(values)):
        raise ValueError("收益矩阵包含非有限值")
    if seed < 0 or seed > 2**32 - 1:
        raise ValueError("随机种子必须位于 0 至 2^32-1 之间")
    if quantize_step is not None and quantize_step <= 0.0:
        raise ValueError("权重量化步长必须大于 0")
    bounds, membership, group_lows, group_highs = _constraint_arrays(values.shape[1], single_limits, group_limits)
    samples, steps, buckets = _round_arrays(rounds, use_refine=use_refine, refine_count=refine_count)
    confidence = float(95 if risk_config.get("confidence") is None else risk_config["confidence"])
    if confidence > 1.0:
        confidence /= 100.0
    return_code = _return_metric_code(return_config)
    risk_code = _risk_metric_code(risk_config)
    return_periods = float(252 if return_config.get("days") is None else return_config["days"])
    risk_periods = float(252 if risk_config.get("days") is None else risk_config["days"])
    return_decay = float(0.94 if return_config.get("alpha") is None else return_config["alpha"])
    risk_decay = float(0.94 if risk_config.get("alpha") is None else risk_config["alpha"])
    risk_window = int(60 if risk_config.get("window") is None else risk_config["window"])
    if return_periods <= 0.0 or risk_periods <= 0.0:
        raise ValueError("年化周期数必须大于 0")
    if return_code == 3 and not 0.0 <= return_decay < 1.0:
        raise ValueError("收益指数加权衰减因子必须位于 [0, 1) 区间")
    if risk_code == 2 and not 0.0 <= risk_decay < 1.0:
        raise ValueError("风险指数加权衰减因子必须位于 [0, 1) 区间")
    if risk_code == 2 and risk_window == 1:
        raise ValueError("指数加权窗口必须为 0 或至少 2")
    if risk_code in {4, 5} and not 0.0 < confidence < 1.0:
        raise ValueError("置信水平必须位于 0 与 1 之间")
    code = _target_code(target)
    if code == 4 and target_return is None:
        raise ValueError("需要提供目标收益率")
    if code == 5 and target_risk is None:
        raise ValueError("需要提供目标风险值")
    result = explore_portfolios_kernel(
        values, bounds, membership, group_lows, group_highs, samples, steps, buckets,
        np.float64(quantize_step or 0.0), np.int64(seed),
        np.int64(return_code), np.int64(risk_code),
        np.float64(return_periods), np.float64(risk_periods),
        np.float64(return_decay), np.float64(risk_decay),
        np.int64(risk_window), np.float64(confidence), np.float64(risk_free_rate),
        np.int64(code), np.float64(target_return or 0.0), np.float64(target_risk or 0.0),
    )
    status = int(result[-1])
    if status == 1:
        raise ValueError("收益矩阵或候选数量无效")
    if status == 2:
        raise ValueError("候选权重生成失败")
    if status == 3:
        raise ValueError("当前约束下没有可行组合")
    if status == 4:
        if code == 4:
            raise ValueError("目标收益不在当前约束的可行范围内")
        if code == 5:
            raise ValueError("目标风险不在当前约束的可行范围内")
        raise ValueError("没有满足优化目标的可行组合")
    return result


def _point(weights: np.ndarray, risks: np.ndarray, returns: np.ndarray, index: int) -> dict[str, Any]:
    return {"value": (float(risks[index]), float(returns[index])), "weights": [float(value) for value in weights[index]]}


def calculate_efficient_frontier_exploration(
    asset_returns: pd.DataFrame,
    return_config: Dict[str, Any],
    risk_config: Dict[str, Any],
    *,
    single_limits: Optional[List[Tuple[float, float]]] = None,
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]] = None,
    rounds: Optional[List[Dict[str, Any]]] = None,
    quantize_step: Optional[float] = None,
    use_slsqp_refine: bool = False,
    refine_count: int = 0,
    risk_free_rate: float = 0.0,
    seed: int = 42,
):
    asset_names = list(asset_returns.columns)
    result = _run_exploration(
        asset_returns.to_numpy(dtype=np.float64), return_config, risk_config,
        single_limits=single_limits, group_limits=group_limits, rounds=rounds,
        quantize_step=quantize_step, use_refine=use_slsqp_refine,
        refine_count=refine_count, risk_free_rate=risk_free_rate, seed=seed,
        target=None, target_return=None, target_risk=None,
    )
    weights, risks, returns, accepted, frontier_indices, frontier_count, special_indices, _ = result
    scatter = [_point(weights, risks, returns, index) for index in range(int(accepted))]
    frontier = [_point(weights, risks, returns, int(frontier_indices[index])) for index in range(int(frontier_count))]
    max_sharpe_index = int(special_indices[0])
    return {
        "asset_names": asset_names,
        "scatter": scatter,
        "frontier": frontier,
        "max_sharpe": None if max_sharpe_index < 0 else _point(weights, risks, returns, max_sharpe_index),
        "min_variance": _point(weights, risks, returns, int(special_indices[1])),
        "max_return": _point(weights, risks, returns, int(special_indices[2])),
        "execution": _validate_optimizer_audit(optimizer_numba_status()),
    }


def calculate_efficient_frontier(
    asset_returns: pd.DataFrame,
    return_config: Dict[str, Any],
    risk_config: Dict[str, Any],
    n_portfolios: int = 10000,
    risk_free_rate: float = 0.0,
):
    return calculate_efficient_frontier_exploration(
        asset_returns, return_config, risk_config,
        rounds=[{"samples": int(n_portfolios), "step": 1.0, "buckets": 100}],
        risk_free_rate=risk_free_rate, seed=42,
    )


def select_target_weights(
    asset_returns: np.ndarray,
    return_config: Dict[str, Any],
    risk_config: Dict[str, Any],
    target: str,
    *,
    single_limits: Optional[List[Tuple[float, float]]] = None,
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]] = None,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    target_risk: Optional[float] = None,
    candidate_count: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    if candidate_count <= 0:
        raise ValueError("候选组合数量必须大于 0")
    first_count = max(1, int(candidate_count * 3 // 10))
    result = _run_exploration(
        asset_returns, return_config, risk_config,
        single_limits=single_limits, group_limits=group_limits,
        rounds=[
            {"samples": first_count, "step": 1.0, "buckets": 40},
            {"samples": max(1, int(candidate_count) - first_count), "step": 0.25, "buckets": 80},
        ],
        quantize_step=None, use_refine=False, refine_count=0,
        risk_free_rate=risk_free_rate, seed=seed, target=target,
        target_return=target_return, target_risk=target_risk,
    )
    weights, _, _, _, _, _, special_indices, _ = result
    selected_index = int(special_indices[3])
    if selected_index < 0:
        raise ValueError("没有满足优化目标的可行组合")
    return np.ascontiguousarray(weights[selected_index], dtype=np.float64)


__all__ = [
    "OPTIMIZER_NUMBA_KERNELS", "OPTIMIZER_NUMBA_KERNEL_VERSION",
    "calculate_efficient_frontier", "calculate_efficient_frontier_exploration",
    "calculate_return", "calculate_return_kernel", "calculate_risk",
    "calculate_risk_kernel", "compute_portfolio_performance",
    "explore_portfolios_kernel", "generate_random_portfolios", "get_log_returns",
    "get_simple_returns", "nav_matrix_returns_kernel", "optimizer_numba_status", "portfolio_returns_kernel",
    "repair_weights_kernel", "select_target_weights", "warm_optimizer_numba_kernels",
    "returns_from_nav_matrix",
]
