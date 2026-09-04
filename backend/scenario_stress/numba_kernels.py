"""Fixed-signature NJIT kernels for scenario simulation and stress testing.

The public engine converts validated objects into stable contiguous arrays.  All
scenario mathematics is executed here; the Python layer only orchestrates and
serializes results.  Explicit signatures make unsupported dtypes/layouts fail
instead of creating a request-time specialization.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numba
import numpy as np
from numba import njit, types
from numba.core.registry import CPUDispatcher

try:
    from compute_policy import validate_execution_audit
except ImportError:  # pragma: no cover - package-style backend import
    from backend.compute_policy import validate_execution_audit


SCENARIO_STRESS_KERNEL_VERSION = "1.3.0"
SCENARIO_STRESS_ENGINE = "numba_njit_fixed_signature"

_F64_1D = types.Array(types.float64, 1, "C")
_F64_2D = types.Array(types.float64, 2, "C")
_F64_3D = types.Array(types.float64, 3, "C")
_I64_1D = types.Array(types.int64, 1, "C")
_I64_2D = types.Array(types.int64, 2, "C")
_U8_1D = types.Array(types.uint8, 1, "C")
_U8_2D = types.Array(types.uint8, 2, "C")
_WEIGHT_SUMMARY_RESULT = types.Tuple(
    (types.float64, types.float64, types.float64, types.int64)
)


@njit(
    _WEIGHT_SUMMARY_RESULT(
        _F64_1D,
        types.float64,
        types.float64,
        types.float64,
        types.float64,
    ),
    cache=True,
    nogil=True,
)
def portfolio_weight_summary_kernel(
    weights: np.ndarray,
    expected_net: float,
    tolerance: float,
    maximum_absolute_weight: float,
    maximum_gross_exposure: float,
) -> tuple[float, float, float, int]:
    """Summarize and validate long/short weights inside fixed-signature NJIT.

    Status bits are stable for both definition validation and the live UI:
    1=non-finite, 2=single weight limit, 4=net mismatch,
    8=gross limit, 16=empty or invalid validation parameters.
    """

    status = 0
    net_exposure = 0.0
    gross_exposure = 0.0
    largest_absolute_weight = 0.0
    if (
        weights.size == 0
        or not np.isfinite(expected_net)
        or not np.isfinite(tolerance)
        or tolerance < 0.0
        or not np.isfinite(maximum_absolute_weight)
        or maximum_absolute_weight < 0.0
        or not np.isfinite(maximum_gross_exposure)
        or maximum_gross_exposure < 0.0
    ):
        return np.nan, np.nan, np.nan, 16
    for weight in weights:
        if not np.isfinite(weight):
            status |= 1
            continue
        absolute_weight = abs(weight)
        net_exposure += weight
        gross_exposure += absolute_weight
        if absolute_weight > largest_absolute_weight:
            largest_absolute_weight = absolute_weight
        if absolute_weight > maximum_absolute_weight + tolerance:
            status |= 2
    if status & 1:
        return np.nan, np.nan, np.nan, status
    if abs(net_exposure - expected_net) > tolerance:
        status |= 4
    if gross_exposure > maximum_gross_exposure + tolerance:
        status |= 8
    return net_exposure, gross_exposure, largest_absolute_weight, status


@njit(
    types.Tuple((_F64_1D, types.float64, types.int64, _U8_1D))(
        _F64_1D,
        _U8_1D,
        types.int64,
        types.float64,
    ),
    cache=True,
    nogil=True,
)
def coverage_weights_kernel(
    weights: np.ndarray,
    available: np.ndarray,
    missing_policy_code: int,
    minimum_coverage: float,
) -> tuple[np.ndarray, float, int, np.ndarray]:
    """Return effective weights and a status without filling missing returns."""

    asset_count = weights.shape[0]
    effective = np.zeros(asset_count, dtype=np.float64)
    missing = np.zeros(asset_count, dtype=np.uint8)
    gross = 0.0
    covered_gross = 0.0
    covered_net = 0.0
    has_missing = False
    for asset_index in range(asset_count):
        weight = weights[asset_index]
        gross += abs(weight)
        if abs(weight) <= 1e-15:
            continue
        if available[asset_index] == 1:
            covered_gross += abs(weight)
            covered_net += weight
        else:
            missing[asset_index] = 1
            has_missing = True
    ratio = covered_gross / gross if gross > 0.0 else 0.0
    if has_missing and missing_policy_code == 0:
        return effective, ratio, 1, missing
    if ratio + 1e-12 < minimum_coverage:
        return effective, ratio, 2, missing
    needs_scale = abs(ratio - 1.0) > 1e-12
    if needs_scale and abs(covered_net) <= 1e-12:
        return effective, ratio, 3, missing
    scale = 1.0 / covered_net if needs_scale else 1.0
    for asset_index in range(asset_count):
        if available[asset_index] == 1:
            effective[asset_index] = weights[asset_index] * scale
    return effective, ratio, 0, missing


@njit(
    types.Tuple(
        (
            _F64_1D,
            _F64_2D,
            _F64_1D,
            _I64_1D,
            _U8_2D,
            _F64_1D,
            _F64_1D,
            _F64_1D,
            _F64_1D,
            _U8_1D,
            types.float64,
        )
    )(_F64_2D, _F64_1D, types.int64, types.float64, types.float64),
    cache=True,
    nogil=True,
)
def deterministic_paths_kernel(
    asset_returns: np.ndarray,
    weights: np.ndarray,
    missing_policy_code: int,
    minimum_coverage: float,
    initial_nav: float,
) -> tuple[np.ndarray, ...]:
    """Project asset returns and calculate NAV, drawdown and contributions."""

    step_count, asset_count = asset_returns.shape
    portfolio_returns = np.full(step_count, np.nan, dtype=np.float64)
    contributions = np.full((step_count, asset_count), np.nan, dtype=np.float64)
    coverage_ratios = np.zeros(step_count, dtype=np.float64)
    statuses = np.zeros(step_count, dtype=np.int64)
    missing_by_step = np.zeros((step_count, asset_count), dtype=np.uint8)
    nav_path = np.full(step_count, np.nan, dtype=np.float64)
    drawdown_path = np.full(step_count, np.nan, dtype=np.float64)
    summary = np.full(9, np.nan, dtype=np.float64)
    contribution_totals = np.zeros(asset_count, dtype=np.float64)
    contribution_missing = np.zeros(asset_count, dtype=np.uint8)
    aggregate_missing = np.zeros(asset_count, dtype=np.uint8)
    minimum_coverage_observed = np.inf
    current_nav = initial_nav
    peak_nav = initial_nav
    peak_index = -1
    max_drawdown = 0.0
    drawdown_peak_nav = initial_nav
    drawdown_start_index = -1
    trough_index = -1
    worst_step_return = np.inf

    for step_index in range(step_count):
        available = np.zeros(asset_count, dtype=np.uint8)
        for asset_index in range(asset_count):
            value = asset_returns[step_index, asset_index]
            if np.isfinite(value):
                available[asset_index] = 1
        effective, ratio, status, missing = coverage_weights_kernel(
            weights,
            available,
            missing_policy_code,
            minimum_coverage,
        )
        coverage_ratios[step_index] = ratio
        if ratio < minimum_coverage_observed:
            minimum_coverage_observed = ratio
        for asset_index in range(asset_count):
            missing_by_step[step_index, asset_index] = missing[asset_index]
            if missing[asset_index] == 1:
                aggregate_missing[asset_index] = 1
                contribution_missing[asset_index] = 1
        if status != 0:
            statuses[step_index] = status
            continue
        portfolio_return = 0.0
        for asset_index in range(asset_count):
            if available[asset_index] == 0:
                contribution_missing[asset_index] = 1
                continue
            value = asset_returns[step_index, asset_index]
            if value <= -1.0:
                status = 4
                break
            contribution = effective[asset_index] * value
            contributions[step_index, asset_index] = contribution
            contribution_totals[asset_index] += contribution
            portfolio_return += contribution
        if status == 0 and (not np.isfinite(portfolio_return) or portfolio_return <= -1.0):
            status = 5
        if status != 0:
            statuses[step_index] = status
            continue
        portfolio_returns[step_index] = portfolio_return
        if portfolio_return < worst_step_return:
            worst_step_return = portfolio_return
        current_nav *= 1.0 + portfolio_return
        if not np.isfinite(current_nav) or current_nav <= 0.0:
            statuses[step_index] = 6
            continue
        nav_path[step_index] = current_nav
        if current_nav >= peak_nav:
            peak_nav = current_nav
            peak_index = step_index
        drawdown = 1.0 - current_nav / peak_nav
        drawdown_path[step_index] = drawdown
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            drawdown_peak_nav = peak_nav
            drawdown_start_index = peak_index
            trough_index = step_index

    for asset_index in range(asset_count):
        if contribution_missing[asset_index] == 1:
            contribution_totals[asset_index] = np.nan
    has_error = False
    for step_index in range(step_count):
        if statuses[step_index] != 0:
            has_error = True
            break
    if not has_error:
        recovery_index = -1
        if trough_index >= 0:
            for step_index in range(trough_index + 1, step_count):
                if nav_path[step_index] >= drawdown_peak_nav:
                    recovery_index = step_index
                    break
        summary[0] = current_nav
        summary[1] = current_nav / initial_nav - 1.0
        summary[2] = max_drawdown
        summary[3] = worst_step_return if step_count > 0 else 0.0
        summary[4] = float(drawdown_start_index)
        summary[5] = float(trough_index)
        summary[6] = float(recovery_index)
        summary[7] = float(recovery_index - trough_index) if recovery_index >= 0 else np.nan
        summary[8] = 1.0 if trough_index < 0 or recovery_index >= 0 else 0.0
    return (
        portfolio_returns,
        contributions,
        coverage_ratios,
        statuses,
        missing_by_step,
        nav_path,
        drawdown_path,
        summary,
        contribution_totals,
        aggregate_missing,
        minimum_coverage_observed,
    )


@njit(
    _F64_2D(_F64_1D, types.int64, types.int64, types.float64),
    cache=True,
    nogil=True,
)
def expand_factor_path_kernel(
    total_shocks: np.ndarray,
    horizon: int,
    path_shape_code: int,
    severity: float,
) -> np.ndarray:
    factor_count = total_shocks.shape[0]
    result = np.zeros((horizon, factor_count), dtype=np.float64)
    for step_index in range(horizon):
        for factor_index in range(factor_count):
            if path_shape_code == 0:
                result[step_index, factor_index] = severity * total_shocks[factor_index] / horizon
            elif step_index == 0:
                result[step_index, factor_index] = severity * total_shocks[factor_index]
    return result


@njit(_F64_2D(_F64_2D, types.float64), cache=True, nogil=True)
def scale_factor_path_kernel(raw_shocks: np.ndarray, severity: float) -> np.ndarray:
    result = np.empty(raw_shocks.shape, dtype=np.float64)
    for step_index in range(raw_shocks.shape[0]):
        for factor_index in range(raw_shocks.shape[1]):
            result[step_index, factor_index] = raw_shocks[step_index, factor_index] * severity
    return result


@njit(
    types.Tuple((_F64_3D, _F64_2D))(
        types.int64,
        types.int64,
        types.int64,
        types.int64,
        types.int64,
        types.float64,
    ),
    cache=True,
    nogil=True,
)
def seeded_factor_draws_kernel(
    horizon: int,
    path_count: int,
    factor_count: int,
    seed: int,
    distribution_code: int,
    degrees_of_freedom: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible primitive draws inside the compiled execution lane."""

    np.random.seed(seed)
    factor_draws = np.empty((horizon, path_count, factor_count), dtype=np.float64)
    for step_index in range(horizon):
        for path_index in range(path_count):
            for factor_index in range(factor_count):
                factor_draws[step_index, path_index, factor_index] = np.random.standard_normal()
    if distribution_code == 1:
        chi_square_draws = np.empty((horizon, path_count), dtype=np.float64)
        for step_index in range(horizon):
            for path_index in range(path_count):
                chi_square_draws[step_index, path_index] = np.random.chisquare(degrees_of_freedom)
    else:
        chi_square_draws = np.ones((1, 1), dtype=np.float64)
    return factor_draws, chi_square_draws


@njit(
    _F64_2D(types.int64, types.int64, types.int64),
    cache=True,
    nogil=True,
)
def seeded_uniform_draws_kernel(
    path_count: int,
    draw_count: int,
    seed: int,
) -> np.ndarray:
    """Create reproducible state-transition uniforms in nopython mode."""

    np.random.seed(seed)
    draws = np.empty((path_count, draw_count), dtype=np.float64)
    for path_index in range(path_count):
        for draw_index in range(draw_count):
            draws[path_index, draw_index] = np.random.random()
    return draws


@njit(
    types.Tuple((_F64_2D, _U8_1D, types.int64))(_F64_2D),
    cache=True,
    nogil=True,
)
def normalize_transition_counts_kernel(
    transition_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Normalize historical transition counts and disclose empty self-loops."""

    state_count = transition_counts.shape[0]
    normalized = np.zeros((state_count, state_count), dtype=np.float64)
    imputed_self_loops = np.zeros(state_count, dtype=np.uint8)
    for row_index in range(state_count):
        total = 0.0
        for column_index in range(state_count):
            value = transition_counts[row_index, column_index]
            if not np.isfinite(value) or value < 0.0:
                return normalized, imputed_self_loops, 1
            total += value
        if total <= 1e-12:
            normalized[row_index, row_index] = 1.0
            imputed_self_loops[row_index] = 1
        else:
            for column_index in range(state_count):
                normalized[row_index, column_index] = (
                    transition_counts[row_index, column_index] / total
                )
    return normalized, imputed_self_loops, 0


@njit(
    types.Tuple((_F64_2D, _U8_2D))(_F64_2D, _U8_2D, _F64_2D, _U8_2D),
    cache=True,
    nogil=True,
)
def metric_deltas_kernel(
    current_values: np.ndarray,
    current_available: np.ndarray,
    reference_values: np.ndarray,
    reference_available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate comparable run deltas without inventing unavailable metrics."""

    row_count, metric_count = current_values.shape
    deltas = np.full((row_count, metric_count), np.nan, dtype=np.float64)
    available = np.zeros((row_count, metric_count), dtype=np.uint8)
    for row_index in range(row_count):
        for metric_index in range(metric_count):
            if (
                current_available[row_index, metric_index] == 1
                and reference_available[row_index, metric_index] == 1
            ):
                deltas[row_index, metric_index] = (
                    current_values[row_index, metric_index]
                    - reference_values[row_index, metric_index]
                )
                available[row_index, metric_index] = 1
    return deltas, available


@njit(
    types.Tuple((_F64_2D, types.int64, types.int64, types.int64))(
        _F64_2D,
        _F64_2D,
        _F64_1D,
        _U8_1D,
    ),
    cache=True,
    nogil=True,
)
def factor_to_asset_kernel(
    factor_shocks: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    available_assets: np.ndarray,
) -> tuple[np.ndarray, int, int, int]:
    step_count, factor_count = factor_shocks.shape
    asset_count = beta.shape[0]
    result = np.full((step_count, asset_count), np.nan, dtype=np.float64)
    for step_index in range(step_count):
        for asset_index in range(asset_count):
            if available_assets[asset_index] == 0:
                continue
            value = intercepts[asset_index]
            for factor_index in range(factor_count):
                value += beta[asset_index, factor_index] * factor_shocks[step_index, factor_index]
            if not np.isfinite(value):
                return result, 1, step_index, asset_index
            if value <= -1.0:
                return result, 2, step_index, asset_index
            result[step_index, asset_index] = value
    return result, 0, -1, -1


@njit(
    types.Tuple((_F64_2D, types.float64, types.int64))(_F64_1D, _F64_2D),
    cache=True,
    nogil=True,
)
def covariance_root_kernel(
    volatilities: np.ndarray,
    correlation: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    dimension = volatilities.shape[0]
    root = np.zeros((dimension, dimension), dtype=np.float64)
    has_positive_volatility = False
    for left_index in range(dimension):
        volatility = volatilities[left_index]
        if not np.isfinite(volatility) or volatility < 0.0:
            return root, np.nan, 1
        if volatility > 0.0:
            has_positive_volatility = True
        for right_index in range(dimension):
            value = correlation[left_index, right_index]
            if not np.isfinite(value):
                return root, np.nan, 2
            if abs(value - correlation[right_index, left_index]) > 1e-10:
                return root, np.nan, 2
            if abs(value) > 1.0 + 1e-10:
                return root, np.nan, 2
        if abs(correlation[left_index, left_index] - 1.0) > 1e-10:
            return root, np.nan, 2
    if not has_positive_volatility:
        return root, np.nan, 1
    correlation_eigenvalues = np.linalg.eigvalsh(correlation)
    minimum_correlation_eigenvalue = correlation_eigenvalues[0]
    if minimum_correlation_eigenvalue < -1e-9:
        return root, minimum_correlation_eigenvalue, 3
    covariance = np.empty((dimension, dimension), dtype=np.float64)
    for left_index in range(dimension):
        for right_index in range(dimension):
            covariance[left_index, right_index] = (
                volatilities[left_index]
                * volatilities[right_index]
                * correlation[left_index, right_index]
            )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    for column_index in range(dimension):
        scale = np.sqrt(max(eigenvalues[column_index], 0.0))
        for row_index in range(dimension):
            root[row_index, column_index] = eigenvectors[row_index, column_index] * scale
    return root, minimum_correlation_eigenvalue, 0


@njit(
    types.void(_F64_3D, _F64_2D, _F64_2D, _F64_1D, types.int64, types.float64),
    cache=True,
    nogil=True,
)
def transform_factor_draws_kernel(
    factor_draws: np.ndarray,
    chi_square_draws: np.ndarray,
    covariance_root: np.ndarray,
    means: np.ndarray,
    distribution_code: int,
    degrees_of_freedom: float,
) -> None:
    step_count, path_count, factor_count = factor_draws.shape
    transformed = np.empty(factor_count, dtype=np.float64)
    student_scale_constant = 1.0
    if distribution_code == 1:
        student_scale_constant = np.sqrt((degrees_of_freedom - 2.0) / degrees_of_freedom)
    for step_index in range(step_count):
        for path_index in range(path_count):
            scale = 1.0
            if distribution_code == 1:
                scale = student_scale_constant / np.sqrt(
                    chi_square_draws[step_index, path_index] / degrees_of_freedom
                )
            for output_factor in range(factor_count):
                value = 0.0
                for input_factor in range(factor_count):
                    value += (
                        factor_draws[step_index, path_index, input_factor]
                        * covariance_root[output_factor, input_factor]
                    )
                transformed[output_factor] = means[output_factor] + value * scale
            for factor_index in range(factor_count):
                factor_draws[step_index, path_index, factor_index] = transformed[factor_index]


@njit(
    types.Tuple((_F64_2D, _F64_1D, types.int64, types.int64, types.int64, types.float64))(
        _F64_3D,
        _F64_2D,
        _F64_1D,
        _F64_1D,
        _U8_1D,
    ),
    cache=True,
    nogil=True,
)
def monte_carlo_projection_kernel(
    factor_draws: np.ndarray,
    beta: np.ndarray,
    intercepts: np.ndarray,
    effective_weights: np.ndarray,
    available_assets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int, float]:
    step_count, path_count, factor_count = factor_draws.shape
    asset_count = beta.shape[0]
    period_returns = np.empty((path_count, step_count), dtype=np.float64)
    mean_contributions = np.zeros(asset_count, dtype=np.float64)
    minimum_return = np.inf
    for step_index in range(step_count):
        for path_index in range(path_count):
            portfolio_return = 0.0
            for asset_index in range(asset_count):
                if available_assets[asset_index] == 0:
                    continue
                log_return = intercepts[asset_index]
                for factor_index in range(factor_count):
                    log_return += (
                        beta[asset_index, factor_index]
                        * factor_draws[step_index, path_index, factor_index]
                    )
                simple_return = np.expm1(log_return)
                contribution = effective_weights[asset_index] * simple_return
                portfolio_return += contribution
                mean_contributions[asset_index] += contribution
            if portfolio_return < minimum_return:
                minimum_return = portfolio_return
            if not np.isfinite(portfolio_return):
                return period_returns, mean_contributions, 1, step_index, path_index, minimum_return
            if portfolio_return <= -1.0:
                return period_returns, mean_contributions, 2, step_index, path_index, minimum_return
            period_returns[path_index, step_index] = portfolio_return
    for asset_index in range(asset_count):
        if available_assets[asset_index] == 0:
            mean_contributions[asset_index] = np.nan
        else:
            mean_contributions[asset_index] /= path_count
    return period_returns, mean_contributions, 0, -1, -1, minimum_return


@njit(
    types.Tuple((_F64_2D, _F64_1D, _F64_1D, types.int64, types.int64, types.int64, types.float64))(
        _F64_2D,
        types.float64,
    ),
    cache=True,
    nogil=True,
)
def returns_to_nav_kernel(
    period_returns: np.ndarray,
    initial_nav: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, float]:
    path_count, step_count = period_returns.shape
    nav_paths = np.empty((path_count, step_count + 1), dtype=np.float64)
    terminal_returns = np.empty(path_count, dtype=np.float64)
    max_drawdowns = np.zeros(path_count, dtype=np.float64)
    minimum_return = np.inf
    for path_index in range(path_count):
        nav = initial_nav
        peak = initial_nav
        nav_paths[path_index, 0] = initial_nav
        for step_index in range(step_count):
            period_return = period_returns[path_index, step_index]
            if period_return < minimum_return:
                minimum_return = period_return
            if not np.isfinite(period_return):
                return nav_paths, terminal_returns, max_drawdowns, 1, step_index, path_index, minimum_return
            if period_return <= -1.0:
                return nav_paths, terminal_returns, max_drawdowns, 2, step_index, path_index, minimum_return
            nav *= 1.0 + period_return
            if not np.isfinite(nav) or nav <= 0.0:
                return nav_paths, terminal_returns, max_drawdowns, 3, step_index, path_index, minimum_return
            nav_paths[path_index, step_index + 1] = nav
            if nav > peak:
                peak = nav
            drawdown = 1.0 - nav / peak
            if drawdown > max_drawdowns[path_index]:
                max_drawdowns[path_index] = drawdown
        terminal_returns[path_index] = nav / initial_nav - 1.0
    return nav_paths, terminal_returns, max_drawdowns, 0, -1, -1, minimum_return


@njit(types.float64(_F64_1D, types.float64), cache=True, nogil=True, inline="always")
def quantile_sorted_kernel(sorted_values: np.ndarray, probability: float) -> float:
    count = sorted_values.shape[0]
    if count == 1:
        return sorted_values[0]
    position = probability * (count - 1)
    lower_index = int(np.floor(position))
    upper_index = int(np.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


@njit(
    types.Tuple((_F64_2D, _F64_1D))(_F64_2D, _F64_1D, _F64_1D, types.float64),
    cache=True,
    nogil=True,
)
def fan_statistics_kernel(
    nav_paths: np.ndarray,
    terminal_returns: np.ndarray,
    max_drawdowns: np.ndarray,
    target_return: float,
) -> tuple[np.ndarray, np.ndarray]:
    path_count, nav_step_count = nav_paths.shape
    probabilities = np.asarray((0.05, 0.25, 0.5, 0.75, 0.95), dtype=np.float64)
    fan = np.empty((5, nav_step_count), dtype=np.float64)
    for step_index in range(nav_step_count):
        sorted_nav = np.sort(nav_paths[:, step_index])
        for quantile_index in range(5):
            fan[quantile_index, step_index] = quantile_sorted_kernel(
                sorted_nav,
                probabilities[quantile_index],
            )
    sorted_returns = np.sort(terminal_returns)
    sorted_drawdowns = np.sort(max_drawdowns)
    sorted_terminal_nav = np.sort(nav_paths[:, nav_step_count - 1])
    metrics = np.empty(14, dtype=np.float64)
    for quantile_index in range(5):
        metrics[quantile_index] = quantile_sorted_kernel(
            sorted_terminal_nav,
            probabilities[quantile_index],
        )
    metrics[5] = quantile_sorted_kernel(sorted_returns, 0.05)
    metrics[6] = quantile_sorted_kernel(sorted_returns, 0.50)
    metrics[7] = quantile_sorted_kernel(sorted_returns, 0.95)
    boundary = metrics[5]
    tail_total = 0.0
    tail_count = 0
    loss_count = 0
    target_count = 0
    drawdown_total = 0.0
    for path_index in range(path_count):
        terminal_return = terminal_returns[path_index]
        if terminal_return <= boundary:
            tail_total += terminal_return
            tail_count += 1
        if terminal_return < 0.0:
            loss_count += 1
        if terminal_return >= target_return:
            target_count += 1
        drawdown_total += max_drawdowns[path_index]
    metrics[8] = max(0.0, -boundary)
    metrics[9] = max(0.0, -(tail_total / tail_count)) if tail_count > 0 else metrics[8]
    metrics[10] = loss_count / path_count
    metrics[11] = target_count / path_count
    metrics[12] = drawdown_total / path_count
    metrics[13] = quantile_sorted_kernel(sorted_drawdowns, 0.95)
    return fan, metrics


@njit(
    types.Tuple((_F64_2D, types.int64))(_F64_2D),
    cache=True,
    nogil=True,
)
def transition_validation_kernel(transition: np.ndarray) -> tuple[np.ndarray, int]:
    state_count = transition.shape[0]
    cumulative = np.empty((state_count, state_count), dtype=np.float64)
    has_random_branch = False
    for row_index in range(state_count):
        row_total = 0.0
        positive_count = 0
        for column_index in range(state_count):
            value = transition[row_index, column_index]
            if not np.isfinite(value) or value < 0.0:
                return cumulative, 1
            row_total += value
            if value > 1e-12:
                positive_count += 1
            cumulative[row_index, column_index] = row_total
        if abs(row_total - 1.0) > 1e-8:
            return cumulative, 1
        cumulative[row_index, state_count - 1] = 1.0
        if positive_count > 1:
            has_random_branch = True
    if not has_random_branch:
        return cumulative, 2
    return cumulative, 0


@njit(
    types.Tuple((_I64_2D, _F64_2D))(_F64_2D, _F64_2D, types.int64, types.int64),
    cache=True,
    nogil=True,
)
def regime_paths_kernel(
    uniform_draws: np.ndarray,
    cumulative_transition: np.ndarray,
    initial_state_index: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    path_count = uniform_draws.shape[0]
    state_count = cumulative_transition.shape[0]
    state_paths = np.empty((path_count, horizon), dtype=np.int64)
    probabilities = np.zeros((horizon, state_count), dtype=np.float64)
    for path_index in range(path_count):
        current_state = initial_state_index
        for step_index in range(horizon):
            state_paths[path_index, step_index] = current_state
            probabilities[step_index, current_state] += 1.0
            if step_index < horizon - 1:
                draw = uniform_draws[path_index, step_index]
                next_state = 0
                while (
                    next_state < state_count - 1
                    and draw > cumulative_transition[current_state, next_state]
                ):
                    next_state += 1
                current_state = next_state
    for step_index in range(horizon):
        for state_index in range(state_count):
            probabilities[step_index, state_index] /= path_count
    return state_paths, probabilities


@njit(_F64_2D(_I64_2D, _F64_1D), cache=True, nogil=True)
def state_path_returns_kernel(
    state_paths: np.ndarray,
    state_returns: np.ndarray,
) -> np.ndarray:
    path_count, step_count = state_paths.shape
    result = np.empty((path_count, step_count), dtype=np.float64)
    for path_index in range(path_count):
        for step_index in range(step_count):
            result[path_index, step_index] = state_returns[state_paths[path_index, step_index]]
    return result


@njit(_F64_1D(_I64_2D, _F64_2D), cache=True, nogil=True)
def state_weighted_contributions_kernel(
    state_paths: np.ndarray,
    state_contributions: np.ndarray,
) -> np.ndarray:
    path_count, step_count = state_paths.shape
    state_count, asset_count = state_contributions.shape
    result = np.zeros(asset_count, dtype=np.float64)
    for asset_index in range(asset_count):
        for state_index in range(state_count):
            if not np.isfinite(state_contributions[state_index, asset_index]):
                result[asset_index] = np.nan
                break
        if not np.isfinite(result[asset_index]):
            continue
        for path_index in range(path_count):
            for step_index in range(step_count):
                state_index = state_paths[path_index, step_index]
                result[asset_index] += state_contributions[state_index, asset_index]
        result[asset_index] /= path_count
    return result


@njit(
    types.Tuple(
        (
            _F64_2D,
            _I64_1D,
            _I64_1D,
            _F64_2D,
            _F64_2D,
            types.int64,
            types.int64,
            types.int64,
        )
    )(_F64_2D, _I64_1D, _I64_1D, _F64_2D, types.int64),
    cache=True,
    nogil=True,
)
def prepare_historical_state_samples_kernel(
    levels: np.ndarray,
    state_codes: np.ndarray,
    return_modes: np.ndarray,
    inline_overrides: np.ndarray,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """Build complete forward-return vectors grouped by the state at period start.

    return_modes: 0=next/current-1, 1=next value already is a simple return.
    Status: 0=ok, 1=shape/configuration error, 2=return <= -100%.
    Missing inputs remain missing and exclude the whole cross-asset observation;
    they are never filled with zero.
    """

    row_count, asset_count = levels.shape
    sample_capacity = max(row_count - 1, 0)
    grouped = np.full((sample_capacity, asset_count), np.nan, dtype=np.float64)
    offsets = np.zeros(max(state_count + 1, 1), dtype=np.int64)
    counts = np.zeros(max(state_count, 0), dtype=np.int64)
    means = np.full((max(state_count, 0), asset_count), np.nan, dtype=np.float64)
    volatilities = np.full((max(state_count, 0), asset_count), np.nan, dtype=np.float64)
    if (
        state_count < 1
        or state_codes.shape[0] != row_count
        or return_modes.shape[0] != asset_count
        or inline_overrides.shape[0] != state_count
        or inline_overrides.shape[1] != asset_count
    ):
        return grouped, offsets, counts, means, volatilities, 1, -1, -1

    raw_returns = np.full((sample_capacity, asset_count), np.nan, dtype=np.float64)
    valid_rows = np.zeros(sample_capacity, dtype=np.uint8)
    for row_index in range(sample_capacity):
        state_index = state_codes[row_index]
        if state_index < 0 or state_index >= state_count:
            continue
        complete = True
        for asset_index in range(asset_count):
            override = inline_overrides[state_index, asset_index]
            if np.isfinite(override):
                value = override
            elif return_modes[asset_index] == 0:
                current = levels[row_index, asset_index]
                following = levels[row_index + 1, asset_index]
                if not np.isfinite(current) or not np.isfinite(following) or current == 0.0:
                    complete = False
                    break
                value = following / current - 1.0
            elif return_modes[asset_index] == 1:
                value = levels[row_index + 1, asset_index]
                if not np.isfinite(value):
                    complete = False
                    break
            else:
                return grouped, offsets, counts, means, volatilities, 1, row_index, asset_index
            if not np.isfinite(value):
                complete = False
                break
            if value <= -1.0:
                return grouped, offsets, counts, means, volatilities, 2, row_index, asset_index
            raw_returns[row_index, asset_index] = value
        if complete:
            valid_rows[row_index] = 1
            counts[state_index] += 1

    for state_index in range(state_count):
        offsets[state_index + 1] = offsets[state_index] + counts[state_index]
    cursors = offsets[:-1].copy()
    for row_index in range(sample_capacity):
        if valid_rows[row_index] == 0:
            continue
        state_index = state_codes[row_index]
        target_index = cursors[state_index]
        for asset_index in range(asset_count):
            grouped[target_index, asset_index] = raw_returns[row_index, asset_index]
        cursors[state_index] += 1

    for state_index in range(state_count):
        start = offsets[state_index]
        end = offsets[state_index + 1]
        count = end - start
        if count == 0:
            continue
        for asset_index in range(asset_count):
            total = 0.0
            for sample_index in range(start, end):
                total += grouped[sample_index, asset_index]
            mean = total / count
            means[state_index, asset_index] = mean
            if count > 1:
                squared = 0.0
                for sample_index in range(start, end):
                    difference = grouped[sample_index, asset_index] - mean
                    squared += difference * difference
                volatilities[state_index, asset_index] = np.sqrt(squared / (count - 1))
            else:
                volatilities[state_index, asset_index] = 0.0
    return grouped, offsets, counts, means, volatilities, 0, -1, -1


@njit(
    types.Tuple(
        (_F64_2D, _F64_1D, types.int64, types.int64, types.int64, types.float64)
    )(_I64_2D, _F64_2D, _F64_2D, _I64_1D, _F64_1D),
    cache=True,
    nogil=True,
)
def empirical_regime_projection_kernel(
    state_paths: np.ndarray,
    uniform_draws: np.ndarray,
    grouped_samples: np.ndarray,
    state_offsets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, int, float]:
    """Bootstrap one complete historical cross-asset return vector per state step."""

    path_count, step_count = state_paths.shape
    asset_count = grouped_samples.shape[1]
    period_returns = np.empty((path_count, step_count), dtype=np.float64)
    mean_contributions = np.zeros(asset_count, dtype=np.float64)
    minimum_return = np.inf
    if (
        uniform_draws.shape[0] != path_count
        or uniform_draws.shape[1] != step_count
        or weights.shape[0] != asset_count
        or state_offsets.shape[0] < 2
    ):
        return period_returns, mean_contributions, 3, -1, -1, minimum_return
    state_count = state_offsets.shape[0] - 1
    for path_index in range(path_count):
        for step_index in range(step_count):
            state_index = state_paths[path_index, step_index]
            if state_index < 0 or state_index >= state_count:
                return period_returns, mean_contributions, 3, step_index, path_index, minimum_return
            start = state_offsets[state_index]
            end = state_offsets[state_index + 1]
            count = end - start
            if count <= 0:
                return period_returns, mean_contributions, 3, step_index, path_index, minimum_return
            draw = uniform_draws[path_index, step_index]
            selected = start + min(int(draw * count), count - 1)
            portfolio_return = 0.0
            for asset_index in range(asset_count):
                asset_return = grouped_samples[selected, asset_index]
                if not np.isfinite(asset_return):
                    return period_returns, mean_contributions, 1, step_index, path_index, minimum_return
                contribution = weights[asset_index] * asset_return
                portfolio_return += contribution
                mean_contributions[asset_index] += contribution
            if portfolio_return < minimum_return:
                minimum_return = portfolio_return
            if not np.isfinite(portfolio_return):
                return period_returns, mean_contributions, 1, step_index, path_index, minimum_return
            if portfolio_return <= -1.0:
                return period_returns, mean_contributions, 2, step_index, path_index, minimum_return
            period_returns[path_index, step_index] = portfolio_return
    for asset_index in range(asset_count):
        mean_contributions[asset_index] /= path_count
    return period_returns, mean_contributions, 0, -1, -1, minimum_return


@njit(types.uint8(types.float64, types.float64, types.int64), cache=True, nogil=True, inline="always")
def compare_limit_kernel(value: float, threshold: float, operator_code: int) -> int:
    if operator_code == 0:
        return 1 if value > threshold else 0
    if operator_code == 1:
        return 1 if value >= threshold else 0
    if operator_code == 2:
        return 1 if value < threshold else 0
    return 1 if value <= threshold else 0


@njit(
    types.Tuple((_F64_1D, _I64_1D, _I64_1D, types.int64, types.int64))(
        _F64_1D,
        _U8_1D,
        _I64_1D,
        _I64_1D,
        _F64_1D,
        types.float64,
        _F64_1D,
        _F64_1D,
        _F64_1D,
        types.int64,
    ),
    cache=True,
    nogil=True,
)
def limit_evaluation_kernel(
    metrics: np.ndarray,
    metric_available: np.ndarray,
    metric_codes: np.ndarray,
    operator_codes: np.ndarray,
    thresholds: np.ndarray,
    initial_nav: float,
    path_returns: np.ndarray,
    path_nav: np.ndarray,
    path_drawdown: np.ndarray,
    has_path: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    limit_count = metric_codes.shape[0]
    values = np.full(limit_count, np.nan, dtype=np.float64)
    breached = np.full(limit_count, -1, dtype=np.int64)
    first_indices = np.full(limit_count, -1, dtype=np.int64)
    breach_count = 0
    overall_first = -1
    for limit_index in range(limit_count):
        metric_code = metric_codes[limit_index]
        if metric_available[metric_code] == 0:
            continue
        value = metrics[metric_code]
        values[limit_index] = value
        is_breached = compare_limit_kernel(
            value,
            thresholds[limit_index],
            operator_codes[limit_index],
        )
        breached[limit_index] = is_breached
        if is_breached == 0:
            continue
        breach_count += 1
        if has_path == 0 or metric_code > 2:
            continue
        for step_index in range(path_returns.shape[0]):
            point_value = path_returns[step_index]
            if metric_code == 0:
                point_value = path_nav[step_index] / initial_nav - 1.0
            elif metric_code == 1:
                point_value = path_drawdown[step_index]
            if compare_limit_kernel(
                point_value,
                thresholds[limit_index],
                operator_codes[limit_index],
            ) == 1:
                first_indices[limit_index] = step_index
                if overall_first < 0 or step_index < overall_first:
                    overall_first = step_index
                break
    return values, breached, first_indices, breach_count, overall_first


@njit(
    types.Tuple(
        (
            types.int64,
            types.int64,
            _F64_2D,
            _F64_2D,
            _F64_2D,
            _F64_1D,
            _F64_1D,
            _F64_1D,
            _U8_1D,
            _I64_1D,
            _I64_1D,
            _I64_1D,
            _I64_1D,
            types.int64,
            types.int64,
            _F64_1D,
        )
    )(_F64_2D, _F64_1D, _F64_2D, _F64_1D, _U8_1D, types.float64, types.float64),
    cache=True,
    nogil=True,
)
def reverse_stress_kernel(
    beta: np.ndarray,
    intercepts: np.ndarray,
    bounds: np.ndarray,
    effective_weights: np.ndarray,
    available_assets: np.ndarray,
    threshold: float,
    initial_nav: float,
) -> tuple[Any, ...]:
    asset_count, factor_count = beta.shape
    max_candidates = factor_count + 1 + min(max(factor_count - 1, 0), 4)
    shocks = np.zeros((max_candidates, factor_count), dtype=np.float64)
    asset_impacts = np.full((max_candidates, asset_count), np.nan, dtype=np.float64)
    contributions = np.full((max_candidates, asset_count), np.nan, dtype=np.float64)
    portfolio_returns = np.zeros(max_candidates, dtype=np.float64)
    max_drawdowns = np.zeros(max_candidates, dtype=np.float64)
    severities = np.zeros(max_candidates, dtype=np.float64)
    meets_target = np.zeros(max_candidates, dtype=np.uint8)
    kind_codes = np.zeros(max_candidates, dtype=np.int64)
    primary_indices = np.full(max_candidates, -1, dtype=np.int64)
    secondary_indices = np.full(max_candidates, -1, dtype=np.int64)
    order = np.arange(max_candidates, dtype=np.int64)
    sensitivities = np.zeros(factor_count, dtype=np.float64)
    base_return = 0.0
    for asset_index in range(asset_count):
        if available_assets[asset_index] == 0:
            continue
        base_return += effective_weights[asset_index] * intercepts[asset_index]
        for factor_index in range(factor_count):
            sensitivities[factor_index] += (
                effective_weights[asset_index] * beta[asset_index, factor_index]
            )
    has_sensitivity = False
    for factor_index in range(factor_count):
        if abs(sensitivities[factor_index]) > 1e-12:
            has_sensitivity = True
            break
    summary = np.full(4, np.nan, dtype=np.float64)
    if not has_sensitivity:
        return (
            1,
            0,
            shocks,
            asset_impacts,
            contributions,
            portfolio_returns,
            max_drawdowns,
            severities,
            meets_target,
            kind_codes,
            primary_indices,
            secondary_indices,
            order,
            -1,
            0,
            summary,
        )

    active_masks = np.zeros((max_candidates, factor_count), dtype=np.uint8)
    candidate_count = 0
    for factor_index in range(factor_count):
        if abs(sensitivities[factor_index]) <= 1e-12:
            continue
        active_masks[candidate_count, factor_index] = 1
        kind_codes[candidate_count] = 1
        primary_indices[candidate_count] = factor_index
        candidate_count += 1
    if factor_count >= 2:
        for factor_index in range(factor_count):
            active_masks[candidate_count, factor_index] = 1
        kind_codes[candidate_count] = 2
        candidate_count += 1
        pair_count = min(factor_count - 1, 4)
        for pair_index in range(pair_count):
            active_masks[candidate_count, pair_index] = 1
            active_masks[candidate_count, pair_index + 1] = 1
            kind_codes[candidate_count] = 3
            primary_indices[candidate_count] = pair_index
            secondary_indices[candidate_count] = pair_index + 1
            candidate_count += 1

    raw_kind_codes = kind_codes.copy()
    raw_primary_indices = primary_indices.copy()
    raw_secondary_indices = secondary_indices.copy()
    target_return = -threshold
    write_count = 0
    for candidate_index in range(candidate_count):
        denominator = 0.0
        for factor_index in range(factor_count):
            if active_masks[candidate_index, factor_index] == 1:
                denominator += sensitivities[factor_index] * sensitivities[factor_index]
        multiplier = (target_return - base_return) / denominator if denominator > 1e-18 else 0.0
        candidate_shocks = np.zeros(factor_count, dtype=np.float64)
        for factor_index in range(factor_count):
            if active_masks[candidate_index, factor_index] == 0:
                continue
            value = multiplier * sensitivities[factor_index]
            value = max(bounds[factor_index, 0], min(bounds[factor_index, 1], value))
            candidate_shocks[factor_index] = value
        duplicate = False
        for prior_index in range(write_count):
            equal = True
            for factor_index in range(factor_count):
                if abs(shocks[prior_index, factor_index] - candidate_shocks[factor_index]) > 5e-13:
                    equal = False
                    break
            if equal:
                duplicate = True
                break
        if duplicate:
            continue
        for factor_index in range(factor_count):
            shocks[write_count, factor_index] = candidate_shocks[factor_index]
        kind_codes[write_count] = raw_kind_codes[candidate_index]
        primary_indices[write_count] = raw_primary_indices[candidate_index]
        secondary_indices[write_count] = raw_secondary_indices[candidate_index]
        write_count += 1
    candidate_count = write_count

    for candidate_index in range(candidate_count):
        portfolio_return = 0.0
        severity_squared = 0.0
        for factor_index in range(factor_count):
            normalizer = max(
                abs(bounds[factor_index, 0]),
                abs(bounds[factor_index, 1]),
                1e-12,
            )
            severity_squared += (shocks[candidate_index, factor_index] / normalizer) ** 2
        severities[candidate_index] = np.sqrt(severity_squared)
        for asset_index in range(asset_count):
            if available_assets[asset_index] == 0:
                continue
            asset_return = intercepts[asset_index]
            for factor_index in range(factor_count):
                asset_return += beta[asset_index, factor_index] * shocks[candidate_index, factor_index]
            asset_impacts[candidate_index, asset_index] = asset_return
            contribution = effective_weights[asset_index] * asset_return
            contributions[candidate_index, asset_index] = contribution
            portfolio_return += contribution
        if not np.isfinite(portfolio_return) or portfolio_return <= -1.0:
            return (
                2,
                candidate_count,
                shocks,
                asset_impacts,
                contributions,
                portfolio_returns,
                max_drawdowns,
                severities,
                meets_target,
                kind_codes,
                primary_indices,
                secondary_indices,
                order,
                candidate_index,
                0,
                summary,
            )
        portfolio_returns[candidate_index] = portfolio_return
        max_drawdowns[candidate_index] = max(0.0, -portfolio_return)
        if portfolio_return <= -threshold + 1e-10:
            meets_target[candidate_index] = 1

    for left_index in range(1, candidate_count):
        current = order[left_index]
        right_index = left_index - 1
        while right_index >= 0:
            prior = order[right_index]
            current_group = 0 if meets_target[current] == 1 else 1
            prior_group = 0 if meets_target[prior] == 1 else 1
            should_move = current_group < prior_group or (
                current_group == prior_group and severities[current] < severities[prior]
            )
            if not should_move:
                break
            order[right_index + 1] = prior
            right_index -= 1
        order[right_index + 1] = current
    worst_index = 0
    for candidate_index in range(1, candidate_count):
        if portfolio_returns[candidate_index] < portfolio_returns[worst_index]:
            worst_index = candidate_index
    worst_return = portfolio_returns[worst_index]
    feasible_count = 0
    for candidate_index in range(candidate_count):
        feasible_count += int(meets_target[candidate_index])
    summary[0] = initial_nav * (1.0 + worst_return)
    summary[1] = worst_return
    summary[2] = max(0.0, -worst_return)
    summary[3] = worst_return
    return (
        0,
        candidate_count,
        shocks,
        asset_impacts,
        contributions,
        portfolio_returns,
        max_drawdowns,
        severities,
        meets_target,
        kind_codes,
        primary_indices,
        secondary_indices,
        order,
        worst_index,
        feasible_count,
        summary,
    )


SCENARIO_STRESS_NUMBA_KERNELS: tuple[CPUDispatcher, ...] = (
    portfolio_weight_summary_kernel,
    coverage_weights_kernel,
    deterministic_paths_kernel,
    expand_factor_path_kernel,
    scale_factor_path_kernel,
    seeded_factor_draws_kernel,
    seeded_uniform_draws_kernel,
    normalize_transition_counts_kernel,
    metric_deltas_kernel,
    factor_to_asset_kernel,
    covariance_root_kernel,
    transform_factor_draws_kernel,
    monte_carlo_projection_kernel,
    returns_to_nav_kernel,
    quantile_sorted_kernel,
    fan_statistics_kernel,
    transition_validation_kernel,
    regime_paths_kernel,
    state_path_returns_kernel,
    state_weighted_contributions_kernel,
    prepare_historical_state_samples_kernel,
    empirical_regime_projection_kernel,
    compare_limit_kernel,
    limit_evaluation_kernel,
    reverse_stress_kernel,
)

for _dispatcher in SCENARIO_STRESS_NUMBA_KERNELS:
    _dispatcher.disable_compile()


def _signature_map() -> dict[str, list[str]]:
    return {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.signatures]
        for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS
    }


@lru_cache(maxsize=1)
def _kernel_code_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS:
        code = dispatcher.py_func.__code__
        payload = code.co_code + repr((code.co_consts, code.co_names)).encode("utf-8")
        result[dispatcher.py_func.__name__] = hashlib.sha256(payload).hexdigest()
    return result


def _kernel_fingerprint(signatures: dict[str, list[str]]) -> str:
    payload = {
        "engine": SCENARIO_STRESS_ENGINE,
        "version": SCENARIO_STRESS_KERNEL_VERSION,
        "numba": numba.__version__,
        "numpy": np.__version__,
        "signatures": signatures,
        "code_hashes": _kernel_code_hashes(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def scenario_stress_numba_status(*, warmed: bool | None = None) -> dict[str, Any]:
    signatures = _signature_map()
    ready = sum(len(items) == 1 for items in signatures.values())
    nopython = all(
        bool(dispatcher.nopython_signatures)
        and all(not compilation.objectmode for compilation in dispatcher.overloads.values())
        for dispatcher in SCENARIO_STRESS_NUMBA_KERNELS
    )
    if warmed is None:
        warmed = warm_scenario_stress_numba_kernels.cache_info().currsize > 0
    return {
        "kernel": "scenario_stress_numeric_core",
        "engine": SCENARIO_STRESS_ENGINE,
        "backend": SCENARIO_STRESS_ENGINE,
        "execution_backend": SCENARIO_STRESS_ENGINE,
        "nopython": nopython,
        "version": SCENARIO_STRESS_KERNEL_VERSION,
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
        "signatures": signatures,
        "kernel_signatures": signatures,
        "kernel_code_hashes": _kernel_code_hashes(),
        "kernel_coverage": f"{ready}/{len(signatures)}",
        "fully_warmed": bool(warmed and ready == len(signatures)),
        "fingerprint": _kernel_fingerprint(signatures),
        "python_fallback": 0,
        "object_mode": 0,
        "request_time_compilation": 0,
        "optimized_third_party_model": None,
        "random_generation": {
            "provider": "numba.random",
            "role": "seeded_sampling_in_fixed_signature_njit",
            "path_math": SCENARIO_STRESS_ENGINE,
        },
    }


def assert_scenario_stress_numba_ready() -> None:
    status = scenario_stress_numba_status()
    validate_execution_audit(status)
    expected = len(SCENARIO_STRESS_NUMBA_KERNELS)
    if status["kernel_coverage"] != f"{expected}/{expected}":
        raise RuntimeError("scenario stress NJIT fixed-signature coverage incomplete")
    if any(len(signatures) != 1 for signatures in status["signatures"].values()):
        raise RuntimeError("scenario stress NJIT signature drift detected")


@lru_cache(maxsize=1)
def warm_scenario_stress_numba_kernels() -> dict[str, Any]:
    """Exercise every production kernel before FastAPI readiness opens."""

    weights = np.ascontiguousarray([0.6, 0.4], dtype=np.float64)
    portfolio_weight_summary_kernel(
        weights,
        np.float64(1.0),
        np.float64(1e-8),
        np.float64(2.0),
        np.float64(3.0),
    )
    available = np.ascontiguousarray([1, 1], dtype=np.uint8)
    asset_returns = np.ascontiguousarray([[0.01, -0.002], [0.003, 0.004]], dtype=np.float64)
    coverage_weights_kernel(weights, available, np.int64(0), np.float64(1.0))
    deterministic_paths_kernel(asset_returns, weights, np.int64(0), np.float64(1.0), np.float64(1.0))
    total_shocks = np.ascontiguousarray([0.01, -0.02], dtype=np.float64)
    raw_path = np.ascontiguousarray([[0.01, -0.02], [0.0, 0.01]], dtype=np.float64)
    expanded = expand_factor_path_kernel(total_shocks, np.int64(2), np.int64(0), np.float64(1.0))
    scale_factor_path_kernel(raw_path, np.float64(1.0))
    seeded_factor_draws_kernel(
        np.int64(2),
        np.int64(4),
        np.int64(2),
        np.int64(7),
        np.int64(0),
        np.float64(5.0),
    )
    seeded_uniform_draws_kernel(np.int64(4), np.int64(2), np.int64(7))
    normalize_transition_counts_kernel(
        np.ascontiguousarray([[2.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    )
    metric_deltas_kernel(
        np.ascontiguousarray([[0.2, 0.0]], dtype=np.float64),
        np.ascontiguousarray([[1, 0]], dtype=np.uint8),
        np.ascontiguousarray([[0.1, 0.0]], dtype=np.float64),
        np.ascontiguousarray([[1, 0]], dtype=np.uint8),
    )
    beta = np.ascontiguousarray([[0.5, 0.1], [-0.2, 0.4]], dtype=np.float64)
    intercepts = np.ascontiguousarray([0.0, 0.0], dtype=np.float64)
    factor_to_asset_kernel(expanded, beta, intercepts, available)
    volatilities = np.ascontiguousarray([0.1, 0.2], dtype=np.float64)
    correlation = np.ascontiguousarray([[1.0, 0.2], [0.2, 1.0]], dtype=np.float64)
    covariance_root, _, _ = covariance_root_kernel(volatilities, correlation)
    draws = np.ascontiguousarray(np.ones((2, 4, 2)), dtype=np.float64)
    chi = np.ascontiguousarray(np.full((2, 4), 5.0), dtype=np.float64)
    transform_factor_draws_kernel(draws, chi, covariance_root, total_shocks, np.int64(0), np.float64(5.0))
    period_returns, _, _, _, _, _ = monte_carlo_projection_kernel(
        draws,
        beta,
        intercepts,
        weights,
        available,
    )
    nav_paths, terminal_returns, max_drawdowns, _, _, _, _ = returns_to_nav_kernel(
        period_returns,
        np.float64(1.0),
    )
    quantile_sorted_kernel(np.ascontiguousarray([0.0, 1.0], dtype=np.float64), np.float64(0.5))
    fan_statistics_kernel(nav_paths, terminal_returns, max_drawdowns, np.float64(0.0))
    transition = np.ascontiguousarray([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    cumulative, _ = transition_validation_kernel(transition)
    uniforms = np.ascontiguousarray(np.full((4, 1), 0.5), dtype=np.float64)
    state_paths, _ = regime_paths_kernel(uniforms, cumulative, np.int64(0), np.int64(2))
    state_returns = np.ascontiguousarray([0.01, -0.02], dtype=np.float64)
    state_path_returns_kernel(state_paths, state_returns)
    state_contributions = np.ascontiguousarray([[0.006, 0.004], [-0.012, -0.008]], dtype=np.float64)
    state_weighted_contributions_kernel(state_paths, state_contributions)
    historical_levels = np.ascontiguousarray(
        [[100.0, 100.0], [101.0, 99.0], [100.0, 101.0], [102.0, 100.0]],
        dtype=np.float64,
    )
    historical_states = np.ascontiguousarray([0, 0, 1, 1], dtype=np.int64)
    return_modes = np.ascontiguousarray([0, 0], dtype=np.int64)
    inline_overrides = np.ascontiguousarray(
        [[np.nan, np.nan], [np.nan, np.nan]], dtype=np.float64
    )
    grouped_samples, state_offsets, _, _, _, _, _, _ = prepare_historical_state_samples_kernel(
        historical_levels,
        historical_states,
        return_modes,
        inline_overrides,
        np.int64(2),
    )
    empirical_regime_projection_kernel(
        state_paths,
        np.ascontiguousarray(np.full(state_paths.shape, 0.5), dtype=np.float64),
        grouped_samples,
        state_offsets,
        weights,
    )
    compare_limit_kernel(np.float64(0.1), np.float64(0.0), np.int64(0))
    metrics = np.ascontiguousarray([0.01, 0.02, -0.01, 0.03, 0.04, 0.5, 0.5], dtype=np.float64)
    metric_available = np.ascontiguousarray(np.ones(7), dtype=np.uint8)
    metric_codes = np.ascontiguousarray([0, 1], dtype=np.int64)
    operator_codes = np.ascontiguousarray([2, 0], dtype=np.int64)
    thresholds = np.ascontiguousarray([-0.1, 0.01], dtype=np.float64)
    limit_evaluation_kernel(
        metrics,
        metric_available,
        metric_codes,
        operator_codes,
        thresholds,
        np.float64(1.0),
        np.ascontiguousarray([0.01, -0.01], dtype=np.float64),
        np.ascontiguousarray([1.01, 0.9999], dtype=np.float64),
        np.ascontiguousarray([0.0, 0.01], dtype=np.float64),
        np.int64(1),
    )
    bounds = np.ascontiguousarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=np.float64)
    reverse_stress_kernel(
        beta,
        intercepts,
        bounds,
        weights,
        available,
        np.float64(0.1),
        np.float64(1.0),
    )
    status = scenario_stress_numba_status(warmed=True)
    status = validate_execution_audit(status)
    expected = len(SCENARIO_STRESS_NUMBA_KERNELS)
    if status["kernel_coverage"] != f"{expected}/{expected}":
        raise RuntimeError("scenario stress NJIT kernel warmup incomplete")
    return status


# Public startup hook kept deliberately short for app lifespan integration.
warm_scenario_numba_kernels = warm_scenario_stress_numba_kernels


__all__ = [
    "SCENARIO_STRESS_ENGINE",
    "SCENARIO_STRESS_KERNEL_VERSION",
    "SCENARIO_STRESS_NUMBA_KERNELS",
    "assert_scenario_stress_numba_ready",
    "coverage_weights_kernel",
    "covariance_root_kernel",
    "deterministic_paths_kernel",
    "empirical_regime_projection_kernel",
    "expand_factor_path_kernel",
    "factor_to_asset_kernel",
    "fan_statistics_kernel",
    "limit_evaluation_kernel",
    "monte_carlo_projection_kernel",
    "metric_deltas_kernel",
    "normalize_transition_counts_kernel",
    "portfolio_weight_summary_kernel",
    "prepare_historical_state_samples_kernel",
    "regime_paths_kernel",
    "returns_to_nav_kernel",
    "reverse_stress_kernel",
    "scale_factor_path_kernel",
    "seeded_factor_draws_kernel",
    "seeded_uniform_draws_kernel",
    "scenario_stress_numba_status",
    "state_path_returns_kernel",
    "state_weighted_contributions_kernel",
    "transition_validation_kernel",
    "transform_factor_draws_kernel",
    "warm_scenario_stress_numba_kernels",
    "warm_scenario_numba_kernels",
]
