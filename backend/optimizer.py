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


from backend.qp_numba import QP_KERNELS, feasible_qp_kernel
from backend.frontier_sampling import SAMPLING_KERNELS, integer_weights_kernel, return_bucket_indices_kernel
from backend.cal_indicators.typed_numba_kernels import covariance_2d


OPTIMIZER_NUMBA_KERNEL_VERSION = "3.0.0"
_FLOAT64_1D = types.float64[::1]
_FLOAT64_2D = types.float64[:, ::1]
_INT64_1D = types.int64[::1]
_UINT8_2D = types.uint8[:, ::1]
_GRID_R1 = types.Array(types.float64, 1, "A", readonly=True)
_GRID_R2 = types.Array(types.float64, 2, "A", readonly=True)
_GRID_U2 = types.Array(types.uint8, 2, "A", readonly=True)
_PORTFOLIO_RESULT = types.UniTuple(types.float64, 2)
_FRONTIER_RESULT = types.Tuple((_INT64_1D, types.int64))
_REPRESENTATIVE_RESULT = types.UniTuple(types.int64, 3)


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
    types.float64(_FLOAT64_1D, types.int64, types.float64, types.float64, types.int64),
    cache=False,
    nogil=True,
)
def calculate_return_kernel(
    returns: np.ndarray,
    metric_code: int,
    periods_per_year: float,
    decay: float,
    window: int,
) -> float:
    if returns.size == 0 or window < 0:
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
        start = max(0, returns.size - window) if window > 0 else 0
        weighted = returns[start]
        for index in range(start + 1, returns.size):
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


@njit((_GRID_R2, _GRID_R1), cache=False, nogil=True)
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
        return integer_weights_kernel(raw_weights, single_bounds, group_membership,
                                      group_lows, group_highs, quantize_step, 50000)
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
    _FRONTIER_RESULT(_FLOAT64_1D, _FLOAT64_1D, types.int64),
    cache=False,
    nogil=True,
)
def pareto_frontier_indices_kernel(
    risks: np.ndarray,
    returns: np.ndarray,
    count: int,
) -> tuple[np.ndarray, int]:
    """Return the exact non-dominated risk/return frontier for the supplied candidates."""
    if count < 0 or count > risks.size or returns.size != risks.size:
        raise ValueError("FRONTIER_AXIS")
    indices = np.full(risks.size, -1, dtype=np.int64)
    if count == 0:
        return indices, 0
    order = np.argsort(risks[:count])
    frontier_count = 0
    best_return = -np.inf
    position = 0
    while position < count:
        first_index = order[position]
        current_risk = risks[first_index]
        best_index = first_index
        scan = position + 1
        while scan < count and risks[order[scan]] == current_risk:
            candidate_index = order[scan]
            if returns[candidate_index] > returns[best_index]:
                best_index = candidate_index
            scan += 1
        if np.isfinite(current_risk) and np.isfinite(returns[best_index]) and returns[best_index] > best_return + 1e-12:
            indices[frontier_count] = best_index
            frontier_count += 1
            best_return = returns[best_index]
        position = scan
    return indices, frontier_count


@njit(
    _REPRESENTATIVE_RESULT(_FLOAT64_1D, _FLOAT64_1D, types.int64, types.float64),
    cache=False,
    nogil=True,
)
def representative_indices_kernel(
    risks: np.ndarray,
    returns: np.ndarray,
    count: int,
    risk_free_rate: float,
) -> tuple[int, int, int]:
    """Select max-Sharpe, minimum-risk and maximum-return from one finite candidate set."""
    if count < 0 or count > risks.size or returns.size != risks.size:
        raise ValueError("REPRESENTATIVE_AXIS")
    if count == 0:
        return -1, -1, -1

    minimum_risk_index = -1
    maximum_return_index = -1
    maximum_sharpe_index = -1
    maximum_sharpe = -np.inf
    for index in range(count):
        risk_value = risks[index]
        return_value = returns[index]
        if not np.isfinite(risk_value) or not np.isfinite(return_value):
            continue
        if minimum_risk_index < 0 or risk_value < risks[minimum_risk_index]:
            minimum_risk_index = index
        if maximum_return_index < 0 or return_value > returns[maximum_return_index]:
            maximum_return_index = index
        if risk_value > 1e-12:
            sharpe = (return_value - risk_free_rate) / risk_value
            if sharpe > maximum_sharpe:
                maximum_sharpe = sharpe
                maximum_sharpe_index = index
    return maximum_sharpe_index, minimum_risk_index, maximum_return_index


@njit(
    (
        _FLOAT64_2D, _FLOAT64_2D, _UINT8_2D, _FLOAT64_1D, _FLOAT64_1D,
        _INT64_1D, _FLOAT64_1D, _INT64_1D, types.float64, types.int64,
        types.int64, types.int64, types.float64, types.float64, types.float64,
        types.float64, types.int64, types.float64, types.float64, types.int64,
        types.float64, types.float64, types.int64,
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
    return_window: int,
):
    requested_count = np.sum(round_samples)
    asset_count = asset_returns.shape[1]
    weights_output = np.full((requested_count, asset_count), np.nan, dtype=np.float64)
    risks = np.full(requested_count, np.nan, dtype=np.float64)
    returns = np.full(requested_count, np.nan, dtype=np.float64)
    frontier_indices = np.full(requested_count, -1, dtype=np.int64)
    special_indices = np.full(4, -1, dtype=np.int64)
    # Columns: start, end, accepted, selected, rejected, search-budget failures.
    round_stats = np.zeros((round_samples.size, 6), dtype=np.int64)
    selected_mask = np.zeros(requested_count, dtype=np.bool_)
    parents = np.full(requested_count, -1, dtype=np.int64)
    if requested_count <= 0 or asset_count <= 0 or asset_returns.shape[0] <= 0:
        return weights_output, risks, returns, 0, frontier_indices, 0, special_indices, round_stats, selected_mask, parents, 1

    np.random.seed(seed)
    accepted = 0
    seed_indices = np.full(requested_count, -1, dtype=np.int64)
    seed_count = 0
    for round_index in range(round_samples.size):
        samples = round_samples[round_index]
        step = round_steps[round_index]
        round_start = accepted
        for _ in range(samples):
            raw = np.empty(asset_count, dtype=np.float64)
            raw_total = 0.0
            for asset_index in range(asset_count):
                raw[asset_index] = -np.log(max(np.random.random(), 1e-15))
                raw_total += raw[asset_index]
            raw /= raw_total
            base_index = -1
            if round_index > 0 and seed_count > 0:
                base_index = seed_indices[min(int(np.random.random() * seed_count), seed_count - 1)]
                for asset_index in range(asset_count):
                    raw[asset_index] = (1.0 - step) * weights_output[base_index, asset_index] + step * raw[asset_index]
            repaired, repair_status = repair_weights_kernel(
                raw, single_bounds, group_membership, group_lows, group_highs, quantize_step
            )
            if repair_status != 0:
                if quantize_step > 0.0 and repair_status == 2:
                    round_stats[round_index, 5] += 1
                continue
            portfolio_returns = portfolio_returns_kernel(asset_returns, repaired)
            return_value = calculate_return_kernel(portfolio_returns, return_metric_code, return_periods, return_decay, return_window)
            risk_value = calculate_risk_kernel(
                portfolio_returns, risk_metric_code, risk_periods, risk_decay, risk_window, confidence
            )
            if not np.isfinite(return_value) or not np.isfinite(risk_value):
                continue
            weights_output[accepted] = repaired
            risks[accepted] = risk_value
            returns[accepted] = return_value
            parents[accepted] = base_index
            accepted += 1
        if round_index == 0:
            chosen, count = np.arange(accepted, dtype=np.int64), accepted
        else:
            chosen, count = return_bucket_indices_kernel(
                risks, returns, round_start, accepted, round_buckets[round_index])
        if count > 0:
            seed_count = count
            for slot in range(count):
                seed_indices[slot] = chosen[slot]
                selected_mask[chosen[slot]] = True
        round_stats[round_index, 0] = round_start
        round_stats[round_index, 1] = accepted
        round_stats[round_index, 2] = accepted - round_start
        round_stats[round_index, 3] = count
        round_stats[round_index, 4] = samples - (accepted - round_start)

    if accepted == 0:
        return weights_output, risks, returns, 0, frontier_indices, 0, special_indices, round_stats, selected_mask, parents, 3

    frontier_indices, frontier_count = pareto_frontier_indices_kernel(risks, returns, accepted)
    maximum_sharpe_index, minimum_risk_index, maximum_return_index = representative_indices_kernel(
        risks, returns, accepted, risk_free_rate
    )

    target_index = -1
    target_score = np.inf
    for index in range(accepted):
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
    return weights_output, risks, returns, accepted, frontier_indices, frontier_count, special_indices, round_stats, selected_mask, parents, status


@njit(
    (_FLOAT64_2D, _FLOAT64_1D, types.int64, types.int64, types.float64,
     types.float64, types.float64, types.float64, types.int64, types.float64,
     types.float64, types.int64, types.int64),
    cache=False,
    nogil=True,
)
def refine_objective_kernel(
    asset_returns: np.ndarray,
    weights: np.ndarray,
    return_metric_code: int,
    risk_metric_code: int,
    return_periods: float,
    risk_periods: float,
    return_decay: float,
    risk_decay: float,
    risk_window: int,
    confidence: float,
    risk_free_rate: float,
    objective_code: int,
    return_window: int,
) -> tuple[float, float, float]:
    portfolio_returns = portfolio_returns_kernel(asset_returns, weights)
    return_value = calculate_return_kernel(portfolio_returns, return_metric_code, return_periods, return_decay, return_window)
    risk_value = calculate_risk_kernel(
        portfolio_returns, risk_metric_code, risk_periods, risk_decay, risk_window, confidence
    )
    if not np.isfinite(return_value) or not np.isfinite(risk_value):
        return return_value, risk_value, -np.inf
    if objective_code == 0:
        score = (return_value - risk_free_rate) / risk_value if risk_value > 1e-12 else -np.inf
    elif objective_code == 1:
        score = -risk_value
    else:
        score = return_value
    return return_value, risk_value, score


@njit(
    (_FLOAT64_2D, _FLOAT64_2D, _FLOAT64_2D, _UINT8_2D, _FLOAT64_1D,
     _FLOAT64_1D, types.float64, types.int64, types.int64, types.float64,
     types.float64, types.float64, types.float64, types.int64, types.float64,
     types.float64, types.int64, types.int64),
    cache=False,
    nogil=True,
)
def refine_special_candidates_kernel(
    asset_returns: np.ndarray,
    initial_weights: np.ndarray,
    single_bounds: np.ndarray,
    group_membership: np.ndarray,
    group_lows: np.ndarray,
    group_highs: np.ndarray,
    quantize_step: float,
    return_metric_code: int,
    risk_metric_code: int,
    return_periods: float,
    risk_periods: float,
    return_decay: float,
    risk_decay: float,
    risk_window: int,
    confidence: float,
    risk_free_rate: float,
    max_iterations: int,
    return_window: int,
):
    """Deterministic pairwise-transfer local refinement for three frontier landmarks."""
    row_count, asset_count = initial_weights.shape
    refined = np.full((row_count, asset_count), np.nan, dtype=np.float64)
    returns_out = np.full(row_count, np.nan, dtype=np.float64)
    risks_out = np.full(row_count, np.nan, dtype=np.float64)
    before_scores = np.full(row_count, np.nan, dtype=np.float64)
    after_scores = np.full(row_count, np.nan, dtype=np.float64)
    iterations = np.zeros(row_count, dtype=np.int64)
    statuses = np.ones(row_count, dtype=np.int64)
    if row_count != 3 or asset_count == 0 or max_iterations < 1:
        return refined, returns_out, risks_out, before_scores, after_scores, iterations, statuses
    for objective_code in range(3):
        raw = initial_weights[objective_code]
        if np.any(~np.isfinite(raw)):
            continue
        total = 0.0
        feasible = True
        for asset_index in range(asset_count):
            value = raw[asset_index]
            if (not np.isfinite(value)
                    or value < single_bounds[asset_index, 0] - 1e-7
                    or value > single_bounds[asset_index, 1] + 1e-7):
                feasible = False
            if quantize_step > 0.0 and abs(value / quantize_step - np.round(value / quantize_step)) > 1e-7:
                feasible = False
            total += value
        if abs(total - 1.0) > 1e-7:
            feasible = False
        for group_index in range(group_membership.shape[0]):
            group_total = 0.0
            for asset_index in range(asset_count):
                if group_membership[group_index, asset_index] != 0:
                    group_total += raw[asset_index]
            if group_total < group_lows[group_index] - 1e-7 or group_total > group_highs[group_index] + 1e-7:
                feasible = False
        if not feasible:
            continue
        # Keep the original feasible candidate as the unchanged score baseline.
        current = raw.copy()
        return_value, risk_value, score = refine_objective_kernel(
            asset_returns, current, return_metric_code, risk_metric_code,
            return_periods, risk_periods, return_decay, risk_decay, risk_window,
            confidence, risk_free_rate, objective_code, return_window,
        )
        if not np.isfinite(score):
            continue
        before_scores[objective_code] = score
        step = 0.05
        loop_count = 0
        while loop_count < max_iterations and step >= 1e-4:
            best_score = score
            best = current.copy()
            best_return = return_value
            best_risk = risk_value
            for donor in range(asset_count):
                donor_room = current[donor] - single_bounds[donor, 0]
                if donor_room <= 1e-12:
                    continue
                for receiver in range(asset_count):
                    if receiver == donor:
                        continue
                    receive_room = single_bounds[receiver, 1] - current[receiver]
                    amount = min(step, min(donor_room, receive_room))
                    if amount <= 1e-12:
                        continue
                    proposal = current.copy()
                    proposal[donor] -= amount
                    proposal[receiver] += amount
                    candidate, repair_status = repair_weights_kernel(
                        proposal, single_bounds, group_membership, group_lows, group_highs, quantize_step
                    )
                    if repair_status != 0:
                        continue
                    candidate_return, candidate_risk, candidate_score = refine_objective_kernel(
                        asset_returns, candidate, return_metric_code, risk_metric_code,
                        return_periods, risk_periods, return_decay, risk_decay, risk_window,
                        confidence, risk_free_rate, objective_code, return_window,
                    )
                    if candidate_score > best_score + 1e-12:
                        best_score = candidate_score
                        best = candidate
                        best_return = candidate_return
                        best_risk = candidate_risk
            loop_count += 1
            if best_score > score + 1e-12:
                current = best
                score = best_score
                return_value = best_return
                risk_value = best_risk
            else:
                step *= 0.5
        refined[objective_code] = current
        returns_out[objective_code] = return_value
        risks_out[objective_code] = risk_value
        after_scores[objective_code] = score
        iterations[objective_code] = loop_count
        statuses[objective_code] = 0 if step < 1e-4 else 2
    return refined, returns_out, risks_out, before_scores, after_scores, iterations, statuses


@njit((_GRID_R2, _GRID_R2, _GRID_U2, _GRID_R1, _GRID_R1,
       types.int64, types.int64, _GRID_R1), cache=False, nogil=True)
def frontier_grid_model_kernel(values, bounds, groups, group_lows, group_highs,
                               return_code, risk_code, settings):
    """Build the linear return axis and constraints once; reuse the risk contract."""
    rows, assets = values.shape
    means = np.empty(assets, dtype=np.float64)
    series = np.empty(rows, dtype=np.float64)
    covariance = np.zeros((assets, assets), dtype=np.float64)
    for j in range(assets):
        for t in range(rows):
            series[t] = values[t, j]
        means[j] = calculate_return_kernel(series, return_code, settings[0], settings[2], int(settings[6]) if settings.size > 6 else 0)
    if risk_code <= 1:
        covariance = covariance_2d(values)
        if risk_code == 1:
            covariance *= settings[1]
    elif risk_code == 2:
        # Polarization uses the existing EWM risk recurrence, not a second
        # independently maintained weighting/variance definition.
        for j in range(assets):
            for t in range(rows):
                series[t] = values[t, j]
            sigma = calculate_risk_kernel(series, risk_code, settings[1], settings[3],
                                          int(settings[4]), settings[5])
            covariance[j, j] = sigma * sigma
        for j in range(assets):
            for k in range(j):
                for t in range(rows):
                    series[t] = values[t, j] + values[t, k]
                sigma = calculate_risk_kernel(series, risk_code, settings[1], settings[3],
                                              int(settings[4]), settings[5])
                covariance[j, k] = (sigma * sigma - covariance[j, j] - covariance[k, k]) * .5
                covariance[k, j] = covariance[j, k]
    count = 1 + assets * 2 + groups.shape[0] * 2
    matrix = np.zeros((count + 1, assets), dtype=np.float64)
    limits = np.empty(count + 1, dtype=np.float64)
    matrix[0] = 1.0
    limits[0] = 1.0
    for j in range(assets):
        matrix[1 + j * 2, j] = 1.0
        limits[1 + j * 2] = bounds[j, 0]
        matrix[2 + j * 2, j] = -1.0
        limits[2 + j * 2] = -bounds[j, 1]
    for k in range(groups.shape[0]):
        row = 1 + assets * 2 + k * 2
        for j in range(assets):
            matrix[row, j] = 1.0 if groups[k, j] != 0 else 0.0
            matrix[row + 1, j] = -matrix[row, j]
        limits[row] = group_lows[k]
        limits[row + 1] = -group_highs[k]
    matrix[count] = means
    limits[count] = 0.0
    return means, covariance, matrix, limits


@njit((_GRID_R2, _GRID_R1, types.int64, _GRID_R1), cache=False, nogil=True)
def grid_risk_gradient_kernel(values, weights, risk_code, settings):
    """Finite differences of the existing non-quadratic risk, with owned scratch."""
    portfolio = portfolio_returns_kernel(values, weights)
    risk = calculate_risk_kernel(portfolio, risk_code, settings[1], settings[3],
                                 int(settings[4]), settings[5])
    gradient = np.empty(weights.size, dtype=np.float64)
    plus = np.empty(values.shape[0], dtype=np.float64)
    minus = np.empty(values.shape[0], dtype=np.float64)
    epsilon = 1e-6
    for j in range(weights.size):
        for t in range(values.shape[0]):
            plus[t] = portfolio[t] + epsilon * values[t, j]
            minus[t] = portfolio[t] - epsilon * values[t, j]
        hi = calculate_risk_kernel(plus, risk_code, settings[1], settings[3],
                                   int(settings[4]), settings[5])
        lo = calculate_risk_kernel(minus, risk_code, settings[1], settings[3],
                                   int(settings[4]), settings[5])
        gradient[j] = (hi - lo) / (2.0 * epsilon)
    return risk, gradient


@njit((_GRID_R2, _GRID_R2, _GRID_R2, _GRID_R1, _GRID_R1,
       types.int64, _GRID_R1, types.int64), cache=False, nogil=True)
def grid_minimum_risk_kernel(values, covariance, matrix, limits, initial,
                             risk_code, settings, max_iterations):
    """One target's QP, or feasible BFGS-SQP for a non-quadratic risk metric.

    Every SQP subproblem has the same linear weight/group/return constraints.
    A line search between feasible points cannot relax those constraints.
    Non-smooth metrics have only a local numerical stopping test, not a global
    optimality certificate. Failed solves are never added to the plotted set.
    """
    n = initial.size
    if risk_code <= 2:
        return feasible_qp_kernel(2.0 * covariance, np.zeros(n), matrix, limits,
                                  initial, max_iterations, 1e-9)
    x = initial.copy()
    value, gradient = grid_risk_gradient_kernel(values, x, risk_code, settings)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return x, 3, 0, np.inf
    scale = max(np.max(np.abs(gradient)), abs(value), 1e-6)
    gradient /= scale
    value /= scale
    hessian = np.eye(n)
    residual = np.inf
    for iteration in range(max_iterations):
        proposal, status, _, residual = feasible_qp_kernel(
            hessian, gradient - hessian @ x, matrix, limits, x, 300, 1e-9)
        if status != 0:
            return x, status, iteration + 1, residual
        direction = proposal - x
        residual = np.max(np.abs(direction))
        # The generic lane uses 1e-6 finite differences, so its local step
        # tolerance is 1e-6 too; the exact quadratic lane keeps 1e-9 KKT tests.
        if residual <= 1e-6:
            return x, 0, iteration + 1, residual
        slope = np.dot(gradient, direction)
        alpha = 1.0
        accepted = False
        new_x = x.copy()
        new_value = value
        new_gradient = gradient.copy()
        for _ in range(30):
            new_x = x + alpha * direction
            raw_value, raw_gradient = grid_risk_gradient_kernel(values, new_x, risk_code, settings)
            new_value = raw_value / scale
            new_gradient = raw_gradient / scale
            if (np.isfinite(new_value) and np.all(np.isfinite(new_gradient))
                    and new_value <= value + 1e-4 * alpha * min(slope, 0.0)):
                accepted = True
                break
            alpha *= .5
        if not accepted:
            return x, 4, iteration + 1, residual
        step = new_x - x
        change = new_gradient - gradient
        hs = hessian @ step
        curvature = np.dot(step, change)
        model_curvature = np.dot(step, hs)
        if model_curvature > 1e-16:
            if curvature < .2 * model_curvature:
                theta = .8 * model_curvature / (model_curvature - curvature)
                change = theta * change + (1.0 - theta) * hs
                curvature = np.dot(step, change)
            if curvature > 1e-16:
                hessian += np.outer(change, change) / curvature - np.outer(hs, hs) / model_curvature
        x, value, gradient = new_x, new_value, new_gradient
    return x, 1, max_iterations, residual


@njit((_GRID_R2, _GRID_R2, _GRID_U2, _GRID_R1, _GRID_R1,
       _GRID_R1, types.int64, types.int64, _GRID_R1, types.int64,
       types.int64, types.boolean, types.float64, types.float64), cache=False, nogil=True)
def solve_frontier_grid_kernel(values, bounds, groups, group_lows, group_highs,
                               initial, return_code, risk_code, settings, point_count,
                               max_iterations, explicit_range, start_target, end_target):
    """Optimize endpoints, generate N return targets and attempt N constrained solves."""
    n = initial.size
    if (not 1 <= n <= 30 or values.shape[0] < 2 or values.shape[1] != n
            or bounds.shape != (n, 2) or groups.shape[1] != n
            or groups.shape[0] != group_lows.size or group_lows.size != group_highs.size
            or (settings.size != 6 and settings.size != 7) or not 2 <= point_count <= 200
            or not 1 <= max_iterations <= 1000 or not 0 <= return_code <= 3
            or not 0 <= risk_code <= 6):
        raise ValueError("FRONTIER_GRID_INPUT_AXIS")
    means, covariance, matrix, limits = frontier_grid_model_kernel(
        values, bounds, groups, group_lows, group_highs, return_code, risk_code, settings)
    n = initial.size
    base_matrix, base_limits = matrix[:-1], limits[:-1]
    low, low_status, low_iterations, low_residual = grid_minimum_risk_kernel(
        values, covariance, base_matrix, base_limits, initial, risk_code, settings, max_iterations)
    high, high_status, high_iterations, high_residual = feasible_qp_kernel(
        np.zeros((n, n)), -means, base_matrix, base_limits, initial, max_iterations, 1e-9)
    range_resolved = low_status == 0 and high_status == 0
    low_return, high_return = np.dot(means, low), np.dot(means, high)
    targets = np.linspace(start_target if explicit_range else low_return,
                          end_target if explicit_range else max(low_return, high_return), point_count)
    weights = np.full((point_count, n), np.nan)
    metrics = np.full((point_count, 2), np.nan)
    statuses = np.full(point_count, 5, dtype=np.int64)
    iterations = np.zeros(point_count, dtype=np.int64)
    residuals = np.full(point_count, np.nan)
    violations = np.full(point_count, np.nan)
    attempted = 0
    # Unresolved endpoints do not certify a feasible frontier range. The caller
    # still receives the requested N rows and endpoint diagnostics, not a fake curve.
    if range_resolved:
        for k in range(point_count):
            attempted += 1
            target = targets[k]
            if target > high_return + 1e-8:
                statuses[k] = 2
                continue
            limits[-1] = target
            seed = low if target <= low_return else high
            solution, status, used, residual = grid_minimum_risk_kernel(
                values, covariance, matrix, limits, seed, risk_code, settings, max_iterations)
            if status == 2:
                status = 3  # An infeasible numerical starting point is not an infeasibility proof.
            portfolio = portfolio_returns_kernel(values, solution)
            actual_return = calculate_return_kernel(portfolio, return_code, settings[0], settings[2], int(settings[6]) if settings.size > 6 else 0)
            actual_risk = calculate_risk_kernel(portfolio, risk_code, settings[1], settings[3],
                                                int(settings[4]), settings[5])
            slack = matrix @ solution - limits
            violation = max(abs(slack[0]), max(0.0, -np.min(slack[1:])))
            if status == 0 and (violation > 1e-7 or not np.isfinite(actual_risk)
                                or not np.isfinite(actual_return)):
                status = 3
            weights[k] = solution
            metrics[k, 0] = actual_risk
            metrics[k, 1] = actual_return
            statuses[k], iterations[k] = status, used
            residuals[k], violations[k] = residual, violation
    endpoint_statuses = np.array([low_status, high_status], dtype=np.int64)
    endpoint_iterations = np.array([low_iterations, high_iterations], dtype=np.int64)
    endpoint_residuals = np.array([low_residual, high_residual])
    endpoint_returns = np.array([low_return, high_return])
    return (targets, weights, metrics, statuses, iterations, residuals, violations,
            endpoint_statuses, endpoint_iterations, endpoint_residuals, endpoint_returns, attempted)


@njit((_FLOAT64_2D, _FLOAT64_1D, _FLOAT64_1D, types.int64,
       _FLOAT64_2D, _FLOAT64_2D, _INT64_1D), cache=False, nogil=True)
def append_grid_candidates_kernel(weights, risks, returns, count, grid_weights, metrics, statuses):
    """Append successful unique grid solutions to this request's owned buffer."""
    indices = np.full(statuses.size, -1, dtype=np.int64)
    duplicate_of = np.full(statuses.size, -1, dtype=np.int64)
    for i in range(statuses.size):
        if statuses[i] != 0:
            continue
        for previous in range(i):
            if indices[previous] < 0:
                continue
            if np.max(np.abs(grid_weights[i] - grid_weights[previous])) <= 1e-10:
                duplicate_of[i] = previous
                indices[i] = indices[previous]
                break
        if indices[i] < 0:
            indices[i] = count
            weights[count] = grid_weights[i]
            risks[count], returns[count] = metrics[i, 0], metrics[i, 1]
            count += 1
    return count, indices, duplicate_of


@njit((_FLOAT64_2D, _INT64_1D, _FLOAT64_1D, _FLOAT64_1D, _INT64_1D, types.int64),
      cache=False, nogil=True)
def grid_curve_membership_kernel(metrics, statuses, risks, returns, frontier_indices, frontier_count):
    """Only plot grid solutions surviving the final common Pareto filter."""
    output = np.zeros(statuses.size, dtype=np.bool_)
    for i in range(statuses.size):
        if statuses[i] != 0:
            continue
        for k in range(frontier_count):
            candidate = frontier_indices[k]
            if (abs(metrics[i, 0] - risks[candidate]) <= 1e-10
                    and abs(metrics[i, 1] - returns[candidate]) <= 1e-10):
                output[i] = True
                break
    return output


@njit((_FLOAT64_2D, _FLOAT64_2D, _FLOAT64_2D, _UINT8_2D,
       _FLOAT64_1D, _FLOAT64_1D, _INT64_1D, _FLOAT64_1D,
       types.float64, types.int64, types.int64, _FLOAT64_1D), cache=False, nogil=True)
def project_grid_weights_kernel(asset_returns, grid_weights, bounds, groups, lows, highs,
                                statuses, targets, step, return_code, risk_code, settings):
    """Produce adoptable integer portfolios without altering continuous grid evidence."""
    weights = np.full_like(grid_weights, np.nan)
    metrics = np.full((grid_weights.shape[0], 2), np.nan)
    result_status = np.full(grid_weights.shape[0], 4, dtype=np.int64)
    target_met = np.zeros(grid_weights.shape[0], dtype=np.bool_)
    for i in range(grid_weights.shape[0]):
        if statuses[i] != 0:
            continue
        repaired, status = integer_weights_kernel(grid_weights[i], bounds, groups, lows, highs, step, 50000)
        result_status[i] = status
        if status != 0:
            continue
        values = portfolio_returns_kernel(asset_returns, repaired)
        risk = calculate_risk_kernel(values, risk_code, settings[1], settings[3], int(settings[4]), settings[5])
        ret = calculate_return_kernel(values, return_code, settings[0], settings[2], int(settings[6]) if settings.size > 6 else 0)
        if not np.isfinite(risk) or not np.isfinite(ret):
            result_status[i] = 3
            continue
        weights[i] = repaired
        metrics[i, 0], metrics[i, 1] = risk, ret
        target_met[i] = ret >= targets[i] - 1e-7
    return weights, metrics, result_status, target_met


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
    pareto_frontier_indices_kernel,
    representative_indices_kernel,
    explore_portfolios_kernel,
    refine_objective_kernel,
    refine_special_candidates_kernel,
    frontier_grid_model_kernel,
    grid_risk_gradient_kernel,
    grid_minimum_risk_kernel,
    solve_frontier_grid_kernel,
    append_grid_candidates_kernel,
    grid_curve_membership_kernel,
    *QP_KERNELS,
    *SAMPLING_KERNELS,
    project_grid_weights_kernel,
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
    calculate_return_kernel(nav, np.int64(3), np.float64(252.0), np.float64(0.94), np.int64(2))
    calculate_risk_kernel(nav, np.int64(0), np.float64(252.0), np.float64(0.94), np.int64(60), np.float64(0.95))
    compute_portfolio_performance(weights, means, covariance)
    generate_random_portfolios(np.int64(2), np.int64(2), means, covariance)
    portfolio_returns_kernel(returns, weights)
    repair_weights_kernel(weights, bounds, groups, empty_float, empty_float, np.float64(0.0))
    pareto_frontier_indices_kernel(
        np.ascontiguousarray([0.1, 0.2], dtype=np.float64),
        np.ascontiguousarray([0.02, 0.03], dtype=np.float64), np.int64(2),
    )
    representative_indices_kernel(
        np.ascontiguousarray([0.1, 0.2], dtype=np.float64),
        np.ascontiguousarray([0.02, 0.03], dtype=np.float64), np.int64(2), np.float64(0.0),
    )
    explore_portfolios_kernel(
        returns, bounds, groups, empty_float, empty_float, samples, steps, buckets,
        np.float64(0.0), np.int64(42), np.int64(0), np.int64(0),
        np.float64(252.0), np.float64(252.0), np.float64(0.94), np.float64(0.94),
        np.int64(60), np.float64(0.95), np.float64(0.0), np.int64(-1),
        np.float64(0.0), np.float64(0.0), np.int64(0),
    )
    refine_objective_kernel(
        returns, weights, np.int64(0), np.int64(0), np.float64(252.0),
        np.float64(252.0), np.float64(0.94), np.float64(0.94), np.int64(60),
        np.float64(0.95), np.float64(0.0), np.int64(0), np.int64(0),
    )
    refine_special_candidates_kernel(
        returns, np.ascontiguousarray(np.vstack((weights, weights, weights))), bounds,
        groups, empty_float, empty_float, np.float64(0.0), np.int64(0), np.int64(0),
        np.float64(252.0), np.float64(252.0), np.float64(0.94), np.float64(0.94),
        np.int64(60), np.float64(0.95), np.float64(0.0), np.int64(2), np.int64(0),
    )
    grid_settings = np.array([252.0, 252.0, .94, .94, 60.0, .95, 2.0])
    for risk_code in (0, 2, 6):
        grid = solve_frontier_grid_kernel(
            returns, bounds, groups, empty_float, empty_float, weights,
            np.int64(2), np.int64(risk_code), grid_settings, np.int64(2),
            np.int64(10), False, np.float64(0.0), np.float64(0.0))
    integer_weights_kernel(weights, bounds, groups, empty_float, empty_float, np.float64(.005), np.int64(50000))
    return_bucket_indices_kernel(means, means, np.int64(0), np.int64(2), np.int64(2))
    project_grid_weights_kernel(returns, grid[1], bounds, groups, empty_float, empty_float,
                                grid[3], grid[0], np.float64(.005), np.int64(2), np.int64(6), grid_settings)
    scratch_weights = np.empty((2, 2))
    scratch_risks, scratch_returns = np.empty(2), np.empty(2)
    count, _, _ = append_grid_candidates_kernel(scratch_weights, scratch_risks, scratch_returns,
                                                np.int64(0), grid[1], grid[2], grid[3])
    frontier_indices, frontier_count = pareto_frontier_indices_kernel(scratch_risks, scratch_returns, count)
    grid_curve_membership_kernel(grid[2], grid[3], scratch_risks, scratch_returns,
                                 frontier_indices, frontier_count)
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


def _return_window(config: Dict[str, Any]) -> int:
    if _return_metric_code(config) != 3:
        return 0
    window = config.get("window", 0)
    if window is None:
        return 0
    if type(window) is not int or window < 0:
        raise ValueError("收益指数加权窗口须为非负整数；0 表示全部样本。")
    return window


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
        np.int64(_return_window(config)),
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rounds is not None and (not isinstance(rounds, list) or not rounds):
        raise ValueError("至少配置一轮随机探索")
    configured = rounds if rounds is not None else [{"samples": 100, "step": 0.99, "buckets": 50}]
    samples: list[int] = []
    steps: list[float] = []
    buckets: list[int] = []
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("探索轮次必须为参数对象")
        count = item.get("samples", 100)
        bucket_count = item.get("buckets", 50)
        if type(count) is not int or count <= 0:
            raise ValueError("每轮候选数量必须为正整数")
        if type(bucket_count) is not int or not 1 <= bucket_count <= 200000:
            raise ValueError("每轮收益分桶数必须为 1 至 200000 的整数")
        step = float(item.get("step", 0.5))
        if not 0.0 <= step <= 1.0:
            raise ValueError("候选搜索步长必须位于 [0, 1] 区间")
        samples.append(count)
        steps.append(step)
        buckets.append(bucket_count)
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
    if type(seed) is not int or seed < 0 or seed > 2**32 - 1:
        raise ValueError("随机种子必须位于 0 至 2^32-1 之间")
    if quantize_step is not None and (not np.isfinite(quantize_step) or quantize_step <= 0.0):
        raise ValueError("权重量化步长必须大于 0")
    bounds, membership, group_lows, group_highs = _constraint_arrays(values.shape[1], single_limits, group_limits)
    if quantize_step is not None:
        _, quant_status = integer_weights_kernel(
            np.ones(values.shape[1]), bounds, membership, group_lows, group_highs,
            np.float64(quantize_step), np.int64(50000))
        if quant_status != 0:
            messages = {1: "指定权重精度与单项／联合约束不存在共同可行组合。",
                        2: "离散权重可行性搜索预算耗尽，尚未确认可行；请调整约束或精度。",
                        3: "权重精度须为可整除 100% 的有限正数，且不小于 0.0000001%。"}
            raise ValueError(messages[int(quant_status)])
    samples, steps, buckets = _round_arrays(rounds)
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
        np.int64(_return_window(return_config)),
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
    use_local_refine: bool = False,
    refine_iterations: int = 0,
    frontier_grid: Optional[Dict[str, Any]] = None,
    risk_free_rate: float = 0.0,
    seed: int = 42,
):
    asset_names = list(asset_returns.columns)
    grid_count = 0
    if frontier_grid is not None:
        grid_count = frontier_grid.get("point_count", 20)
        grid_iterations = frontier_grid.get("max_iterations", 300)
        if (type(grid_count) is not int or not 2 <= grid_count <= 200
                or type(grid_iterations) is not int or not 1 <= grid_iterations <= 1000):
            raise ValueError("前沿目标点数须为 2–200 的整数；单点最大迭代次数须为 1–1000 的整数。")
        if len(asset_names) > 30 or len(group_limits or {}) > 30:
            raise ValueError("目标网格求解最多支持 30 个资产和 30 个分组约束。")
        if frontier_grid.get("weight_domain", "continuous") != "continuous":
            raise ValueError("目标网格使用连续权重，不提供未经验证的整数权重求解。")
        if quantize_step and frontier_grid.get("accept_continuous_weights") is not True:
            raise ValueError("散点启用取整时，请明确确认目标网格采用连续权重；不会静默将离散约束用于连续曲线。")
        target_start, target_end = frontier_grid.get("target_start"), frontier_grid.get("target_end")
        if (target_start is None) != (target_end is None):
            raise ValueError("自定义目标区间必须同时提供起点和终点。")
        if target_start is not None and (not np.isfinite(target_start) or not np.isfinite(target_end) or target_start > target_end):
            raise ValueError("目标收益区间须为有限数值且起点不大于终点。")
    asset_values = np.ascontiguousarray(asset_returns.to_numpy(dtype=np.float64))
    if frontier_grid is not None and asset_values.shape[0] < 2:
        raise ValueError("目标网格至少需要两个收益观察期。")
    result = _run_exploration(
        asset_values, return_config, risk_config,
        single_limits=single_limits, group_limits=group_limits, rounds=rounds,
        quantize_step=quantize_step, risk_free_rate=risk_free_rate, seed=seed,
        target=None, target_return=None, target_risk=None,
    )
    weights, risks, returns, accepted, _frontier_indices, _frontier_count, special_indices, round_stats, selected_mask, parents, _ = result
    sampled_candidates = int(accepted)
    combined_weights = np.empty((sampled_candidates + 3 + grid_count, len(asset_names)), dtype=np.float64)
    combined_risks = np.empty(sampled_candidates + 3 + grid_count, dtype=np.float64)
    combined_returns = np.empty(sampled_candidates + 3 + grid_count, dtype=np.float64)
    combined_weights[:sampled_candidates] = weights[:sampled_candidates]
    combined_risks[:sampled_candidates] = risks[:sampled_candidates]
    combined_returns[:sampled_candidates] = returns[:sampled_candidates]
    combined_count = sampled_candidates

    max_sharpe_index = int(special_indices[0])
    refinement: dict[str, Any] = {
        "requested": bool(use_local_refine),
        "algorithm": "bounded_pairwise_pattern_search_njit",
        "max_iterations": int(refine_iterations),
        "items": [],
        "accepted_points": 0,
        "global_optimum_claim": False,
    }
    if use_local_refine:
        if not 1 <= int(refine_iterations) <= 200:
            raise ValueError("局部精炼迭代次数必须位于 1 至 200 之间")
        initial = np.full((3, len(asset_names)), np.nan, dtype=np.float64)
        source_indices = (max_sharpe_index, int(special_indices[1]), int(special_indices[2]))
        for index, source_index in enumerate(source_indices):
            if source_index >= 0:
                initial[index] = weights[source_index]
        bounds, membership, group_lows, group_highs = _constraint_arrays(len(asset_names), single_limits, group_limits)
        confidence = float(95 if risk_config.get("confidence") is None else risk_config["confidence"])
        if confidence > 1.0:
            confidence /= 100.0
        refined = refine_special_candidates_kernel(
            asset_values, np.ascontiguousarray(initial), bounds, membership, group_lows, group_highs,
            np.float64(quantize_step or 0.0), np.int64(_return_metric_code(return_config)),
            np.int64(_risk_metric_code(risk_config)),
            np.float64(252 if return_config.get("days") is None else return_config["days"]),
            np.float64(252 if risk_config.get("days") is None else risk_config["days"]),
            np.float64(0.94 if return_config.get("alpha") is None else return_config["alpha"]),
            np.float64(0.94 if risk_config.get("alpha") is None else risk_config["alpha"]),
            np.int64(60 if risk_config.get("window") is None else risk_config["window"]),
            np.float64(confidence), np.float64(risk_free_rate), np.int64(refine_iterations),
            np.int64(_return_window(return_config)),
        )
        refined_weights, refined_returns, refined_risks, before_scores, after_scores, iterations, statuses = refined
        keys = ("max_sharpe", "min_variance", "max_return")
        for index, key in enumerate(keys):
            status = int(statuses[index])
            non_worsening = (
                status in (0, 2)
                and np.isfinite(before_scores[index])
                and np.isfinite(after_scores[index])
                and after_scores[index] >= before_scores[index] - 1e-12
            )
            improved = bool(non_worsening and after_scores[index] > before_scores[index] + 1e-12)
            refinement["items"].append({
                "candidate": key,
                "status": "converged" if status == 0 else "max_iterations" if status == 2 else "failed",
                "iterations": int(iterations[index]),
                "before_score": float(before_scores[index]) if np.isfinite(before_scores[index]) else None,
                "after_score": float(after_scores[index]) if np.isfinite(after_scores[index]) else None,
                "non_worsening": bool(non_worsening),
                "applied": improved,
            })
            if improved:
                combined_weights[combined_count] = refined_weights[index]
                combined_risks[combined_count] = refined_risks[index]
                combined_returns[combined_count] = refined_returns[index]
                combined_count += 1
        refinement["accepted_points"] = combined_count - sampled_candidates

    local_refined_count = combined_count - sampled_candidates
    grid_start_index = combined_count
    grid_result = None
    if frontier_grid is not None:
        bounds, membership, group_lows, group_highs = _constraint_arrays(len(asset_names), single_limits, group_limits)
        confidence = float(95 if risk_config.get("confidence") is None else risk_config["confidence"])
        if confidence > 1.0:
            confidence /= 100.0
        settings = np.array([
            252 if return_config.get("days") is None else return_config["days"],
            252 if risk_config.get("days") is None else risk_config["days"],
            .94 if return_config.get("alpha") is None else return_config["alpha"],
            .94 if risk_config.get("alpha") is None else risk_config["alpha"],
            60 if risk_config.get("window") is None else risk_config["window"], confidence,
            _return_window(return_config),
        ], dtype=np.float64)
        grid_result = solve_frontier_grid_kernel(
            asset_values, bounds, membership, group_lows, group_highs, weights[0],
            np.int64(_return_metric_code(return_config)), np.int64(_risk_metric_code(risk_config)),
            settings, np.int64(grid_count), np.int64(grid_iterations), target_start is not None,
            np.float64(target_start or 0.0), np.float64(target_end or 0.0))
        adopt_weights, adopt_metrics, adopt_statuses = grid_result[1:4]
        if quantize_step:
            adopt_weights, adopt_metrics, adopt_statuses, adopt_target_met = project_grid_weights_kernel(
                asset_values, grid_result[1], bounds, membership, group_lows, group_highs,
                grid_result[3], grid_result[0], np.float64(quantize_step),
                np.int64(_return_metric_code(return_config)), np.int64(_risk_metric_code(risk_config)), settings)
        combined_count, grid_indices, grid_duplicates = append_grid_candidates_kernel(
            combined_weights, combined_risks, combined_returns, np.int64(combined_count),
            adopt_weights, adopt_metrics, adopt_statuses)

    final_frontier_indices, final_frontier_count = pareto_frontier_indices_kernel(
        combined_risks, combined_returns, np.int64(combined_count)
    )
    final_max_sharpe, final_min_risk, final_max_return = representative_indices_kernel(
        combined_risks, combined_returns, np.int64(combined_count), np.float64(risk_free_rate)
    )
    final_indices = {
        "max_sharpe": int(final_max_sharpe),
        "min_variance": int(final_min_risk),
        "max_return": int(final_max_return),
    }
    special_points = {
        key: None if index < 0 else _point(combined_weights, combined_risks, combined_returns, index)
        for key, index in final_indices.items()
    }
    refinement["final_representatives"] = {
        key: {
            "candidate_index": index,
            "source": "unavailable" if index < 0 else "sampled" if index < sampled_candidates else "refined" if index < grid_start_index else "grid",
        }
        for key, index in final_indices.items()
    }
    scatter = [_point(combined_weights, combined_risks, combined_returns, index) for index in range(combined_count)]
    frontier = [
        _point(combined_weights, combined_risks, combined_returns, int(final_frontier_indices[index]))
        for index in range(int(final_frontier_count))
    ]
    grid_payload = None
    if grid_result is not None:
        targets, grid_weights, metrics, statuses, iterations, residuals, violations = grid_result[:7]
        if quantize_step:
            # Reference curve belongs to the continuous domain. Compact only grid outputs.
            reference_weights = np.empty_like(grid_weights)
            reference_risks, reference_returns = np.empty(grid_count), np.empty(grid_count)
            reference_count, _, continuous_duplicates = append_grid_candidates_kernel(
                reference_weights, reference_risks, reference_returns, np.int64(0), grid_weights, metrics, statuses)
            reference_indices, reference_frontier_count = pareto_frontier_indices_kernel(
                reference_risks, reference_returns, reference_count)
            plot_mask = grid_curve_membership_kernel(metrics, statuses, reference_risks, reference_returns,
                                                     reference_indices, reference_frontier_count)
        else:
            plot_mask = grid_curve_membership_kernel(metrics, statuses, combined_risks, combined_returns,
                                                     final_frontier_indices, final_frontier_count)
        status_names = ("converged", "max_iterations", "infeasible_target", "numerical_failure",
                        "line_search_failed", "range_unresolved")
        finite = lambda value: float(value) if np.isfinite(value) else None
        points = []
        for i in range(grid_count):
            candidate_index = int(grid_indices[i])
            point = {"target_index": i, "target": finite(targets[i]),
                     "status": status_names[int(statuses[i])], "iterations": int(iterations[i]),
                     "optimality_residual": finite(residuals[i]), "constraint_violation": finite(violations[i]),
                     "candidate_index": candidate_index if candidate_index >= 0 else None,
                     "duplicate_of": (int(continuous_duplicates[i]) if continuous_duplicates[i] >= 0 else None) if quantize_step else (int(grid_duplicates[i]) if grid_duplicates[i] >= 0 else None),
                     "on_frontier": bool(plot_mask[i]),
                     "value": [finite(metrics[i, 0]), finite(metrics[i, 1])],
                     "weights": [finite(value) for value in grid_weights[i]]}
            if quantize_step:
                point["adoption"] = {
                    "weight_domain": "discrete", "step": quantize_step,
                    "status": ("feasible", "infeasible", "search_budget", "numerical_failure", "grid_failed")[int(adopt_statuses[i])],
                    "weights": [finite(value) for value in adopt_weights[i]],
                    "value": [finite(value) for value in adopt_metrics[i]],
                    "target_met": bool(adopt_target_met[i]),
                    "candidate_index": point["candidate_index"],
                    "duplicate_of": int(grid_duplicates[i]) if grid_duplicates[i] >= 0 else None,
                }
            points.append(point)
        success = sum(point["status"] == "converged" for point in points)
        grid_payload = {
            "algorithm": "target_return_grid_sqp_njit", "target_axis": "return",
            "subproblem": "minimum_risk_given_return_floor", "weight_domain": "continuous",
            "adoption_weight_domain": "discrete" if quantize_step else "continuous",
            "risk_solver": "active_set_qp" if _risk_metric_code(risk_config) <= 2 else "feasible_bfgs_sqp",
            "optimality_scope": "convex_quadratic_kkt" if _risk_metric_code(risk_config) <= 2 else "local_numerical_stationarity",
            "requested_points": grid_count, "attempted_points": int(grid_result[11]),
            "solver_calls": sum(point["iterations"] > 0 for point in points),
            "infeasible_targets": sum(point["status"] == "infeasible_target" for point in points),
            "successful_points": success, "failed_points": int(grid_result[11]) - success,
            "unattempted_points": grid_count - int(grid_result[11]),
            "duplicate_targets": sum(points[i]["target"] == points[i - 1]["target"] for i in range(1, grid_count)),
            "duplicate_solutions": sum(point["duplicate_of"] is not None for point in points),
            "adoption_duplicate_solutions": sum(point.get("adoption", {}).get("duplicate_of") is not None for point in points),
            "added_candidates": int(combined_count) - grid_start_index,
            "max_iterations": grid_iterations, "constraint_tolerance": 1e-7,
            "stationarity_tolerance": 1e-9 if _risk_metric_code(risk_config) <= 2 else 1e-6,
            "inner_qp_max_iterations": 300, "line_search_max_steps": 30,
            "points": points, "curve": [point if point["on_frontier"] else None for point in points],
            "endpoints": [{"kind": kind, "status": status_names[int(grid_result[7][i])],
                           "iterations": int(grid_result[8][i]), "optimality_residual": finite(grid_result[9][i]),
                           "return": finite(grid_result[10][i])} for i, kind in enumerate(("minimum_risk", "maximum_return"))],
        }
    refined_candidates = local_refined_count
    return {
        "asset_names": asset_names,
        "scatter": scatter,
        "frontier": frontier,
        "sampled_candidates": sampled_candidates,
        "weight_domain": "discrete" if quantize_step else "continuous",
        "quantization_step": quantize_step,
        "exploration": {
            "algorithm": "return_bucket_minimum_risk_random_walk_njit", "seed": seed,
            "selected_indices": [i for i in range(sampled_candidates) if selected_mask[i]],
            "parent_indices": [int(value) for value in parents[:sampled_candidates]],
            "rounds": [{"round": i, "requested": int(row[2] + row[4]),
                        "start_index": int(row[0]), "end_index": int(row[1]),
                        "accepted": int(row[2]), "selected": int(row[3]), "rejected": int(row[4]),
                        "search_budget_failures": int(row[5]),
                        "status": "completed" if row[2] else "no_candidates_previous_seeds_retained"}
                       for i, row in enumerate(round_stats)],
        },
        "refined_candidates": refined_candidates,
        "grid_candidates": int(combined_count) - grid_start_index,
        "frontier_grid": grid_payload,
        "accepted_candidates": int(combined_count),
        **special_points,
        "refinement": refinement,
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
        quantize_step=None,
        risk_free_rate=risk_free_rate, seed=seed, target=target,
        target_return=target_return, target_risk=target_risk,
    )
    weights, special_indices = result[0], result[6]
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
    "get_simple_returns", "nav_matrix_returns_kernel", "optimizer_numba_status", "pareto_frontier_indices_kernel",
    "representative_indices_kernel",
    "portfolio_returns_kernel", "refine_objective_kernel", "refine_special_candidates_kernel", "repair_weights_kernel",
    "select_target_weights", "warm_optimizer_numba_kernels",
    "returns_from_nav_matrix",
]
