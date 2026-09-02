"""Fused fixed-signature kernels for the 35 audited scalar built-in metrics."""

from __future__ import annotations

import math

import numpy as np
from numba import float64, int16, int64, njit, prange, types, void


BUILTIN_METRIC_IDS = (
    "builtin-total-return-v2",
    "builtin-annualized-return-v2",
    "builtin-mean-return-v2",
    "builtin-median-return-v2",
    "builtin-maximum-gain-v2",
    "builtin-maximum-loss-v2",
    "builtin-positive-return-ratio-v2",
    "builtin-mean-positive-return-v2",
    "builtin-mean-negative-return-v2",
    "builtin-payoff-ratio-v2",
    "builtin-return-volatility-v2",
    "builtin-annualized-volatility-v2",
    "builtin-return-mad-v2",
    "builtin-return-range-v2",
    "builtin-return-skewness-v2",
    "builtin-return-excess-kurtosis-v2",
    "builtin-historical-var-95-v2",
    "builtin-historical-cvar-95-v2",
    "builtin-downside-deviation-v2",
    "builtin-upside-deviation-v2",
    "builtin-annualized-sharpe-v2",
    "builtin-annualized-sortino-v2",
    "builtin-omega-ratio-v2",
    "builtin-tail-ratio-95-v2",
    "builtin-maximum-drawdown-v2",
    "builtin-ulcer-index-v2",
    "builtin-calmar-ratio-v2",
    "builtin-new-high-ratio-v2",
    "builtin-adjusted-nav-slope-v2",
    "builtin-adjusted-nav-r-squared-v2",
    "builtin-average-volume-v2",
    "builtin-volume-volatility-v2",
    "builtin-highest-market-price-v2",
    "builtin-lowest-market-price-v2",
    "builtin-market-high-low-range-v2",
)
BUILTIN_METRIC_CODE = {
    indicator_id: code for code, indicator_id in enumerate(BUILTIN_METRIC_IDS)
}

STATUS_OK = 0
STATUS_INSUFFICIENT_SAMPLE = 1
STATUS_DIVIDE_BY_ZERO = 2
STATUS_DOMAIN_ERROR = 3
STATUS_NON_FINITE_RESULT = 4


@njit(cache=True, nogil=True, inline="always")
def _mean(values: np.ndarray) -> float:
    total = 0.0
    for value in values:
        total += value
    return total / values.size


@njit(cache=True, nogil=True, inline="always")
def _sample_std(values: np.ndarray) -> float:
    if values.size < 2:
        return math.nan
    mean = _mean(values)
    total = 0.0
    for value in values:
        delta = value - mean
        total += delta * delta
    return math.sqrt(total / (values.size - 1))


@njit(cache=True, nogil=True, inline="always")
def _quantile_sorted(sorted_values: np.ndarray, probability: float) -> float:
    position = (sorted_values.size - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


@njit(cache=True, nogil=True)
def _linear_fit(values: np.ndarray) -> tuple[float, float]:
    count = values.size
    mean_x = (count - 1) / 2.0
    mean_y = _mean(values)
    numerator = 0.0
    denominator = 0.0
    total = 0.0
    for index in range(count):
        centered_x = index - mean_x
        centered_y = values[index] - mean_y
        numerator += centered_x * centered_y
        denominator += centered_x * centered_x
        total += centered_y * centered_y
    if denominator <= 0.0:
        return math.nan, math.nan
    slope = numerator / denominator
    residual = 0.0
    intercept = mean_y - slope * mean_x
    for index in range(count):
        difference = values[index] - (intercept + slope * index)
        residual += difference * difference
    r_squared = math.nan if total <= 0.0 else 1.0 - residual / total
    return slope, r_squared


@njit(cache=True, nogil=True)
def _compute_metric(
    code: int,
    returns: np.ndarray,
    nav: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray,
    risk_free_per_observation: float,
) -> tuple[float, int]:
    count = returns.size
    if code <= 27 and count < 1:
        return math.nan, STATUS_INSUFFICIENT_SAMPLE
    if code in (10, 11, 20) and count < 2:
        return math.nan, STATUS_INSUFFICIENT_SAMPLE
    if code == 14 and count < 3:
        return math.nan, STATUS_INSUFFICIENT_SAMPLE
    if code == 15 and count < 4:
        return math.nan, STATUS_INSUFFICIENT_SAMPLE

    growth = 1.0
    mean = 0.0
    minimum = math.inf
    maximum = -math.inf
    positive_count = 0
    positive_sum = 0.0
    negative_count = 0
    negative_sum = 0.0
    if count:
        for value in returns:
            growth *= 1.0 + value
            mean += value
            minimum = min(minimum, value)
            maximum = max(maximum, value)
            if value > 0.0:
                positive_count += 1
                positive_sum += value
            elif value < 0.0:
                negative_count += 1
                negative_sum += value
        mean /= count

    value = math.nan
    status = STATUS_OK
    if code == 0:
        value = growth - 1.0
    elif code == 1:
        value = growth ** (252.0 / count) - 1.0 if growth >= 0.0 else math.nan
        if growth < 0.0:
            status = STATUS_DOMAIN_ERROR
    elif code == 2:
        value = mean
    elif code == 3:
        value = _quantile_sorted(np.sort(returns.copy()), 0.5)
    elif code == 4:
        value = maximum
    elif code == 5:
        value = abs(min(minimum, 0.0))
    elif code == 6:
        value = positive_count / count
    elif code == 7:
        if positive_count == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = positive_sum / positive_count
    elif code == 8:
        if negative_count == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = negative_sum / negative_count
    elif code == 9:
        if positive_count == 0 or negative_count == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            denominator = abs(negative_sum / negative_count)
            if denominator == 0.0:
                status = STATUS_DIVIDE_BY_ZERO
            else:
                value = (positive_sum / positive_count) / denominator
    elif code in (10, 11):
        value = _sample_std(returns)
        if code == 11:
            value *= math.sqrt(252.0)
    elif code == 12:
        total = 0.0
        for item in returns:
            total += abs(item - mean)
        value = total / count
    elif code == 13:
        value = maximum - minimum
    elif code in (14, 15):
        second = 0.0
        third = 0.0
        fourth = 0.0
        for item in returns:
            centered = item - mean
            square = centered * centered
            second += square
            third += square * centered
            fourth += square * square
        second /= count
        if second <= 0.0:
            status = STATUS_DOMAIN_ERROR
        elif code == 14:
            third /= count
            value = (
                math.sqrt(count * (count - 1.0))
                / (count - 2.0)
                * third
                / second**1.5
            )
        else:
            fourth /= count
            population_excess = fourth / (second * second) - 3.0
            value = ((count - 1.0) / ((count - 2.0) * (count - 3.0))) * (
                (count + 1.0) * population_excess + 6.0
            )
    elif code in (16, 17, 23):
        sorted_values = np.sort(returns.copy())
        lower_quantile = _quantile_sorted(sorted_values, 0.05)
        if code == 16:
            value = -lower_quantile
        elif code == 17:
            selected_sum = 0.0
            selected_count = 0
            for item in returns:
                if item <= lower_quantile:
                    selected_sum += item
                    selected_count += 1
            value = -selected_sum / selected_count
        else:
            upper_quantile = _quantile_sorted(sorted_values, 0.95)
            denominator = abs(lower_quantile)
            if denominator == 0.0:
                status = STATUS_DIVIDE_BY_ZERO
            else:
                value = upper_quantile / denominator
    elif code in (18, 19, 21):
        selected_sum = 0.0
        selected_count = 0
        for item in returns:
            excess = item - risk_free_per_observation
            if (code in (18, 21) and excess < 0.0) or (code == 19 and excess > 0.0):
                selected_sum += excess * excess
                selected_count += 1
        if selected_count == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            deviation = math.sqrt(selected_sum / selected_count)
            if code == 21:
                numerator = mean - risk_free_per_observation
                if deviation == 0.0:
                    status = STATUS_DIVIDE_BY_ZERO
                else:
                    value = numerator / deviation * math.sqrt(252.0)
            else:
                value = deviation
    elif code == 20:
        denominator = _sample_std(returns)
        if denominator == 0.0:
            status = STATUS_DIVIDE_BY_ZERO
        else:
            value = (mean - risk_free_per_observation) / denominator * math.sqrt(252.0)
    elif code == 22:
        gains = 0.0
        losses = 0.0
        for item in returns:
            excess = item - risk_free_per_observation
            if excess > 0.0:
                gains += excess
            elif excess < 0.0:
                losses += excess
        denominator = abs(losses)
        if denominator == 0.0:
            status = STATUS_DIVIDE_BY_ZERO
        else:
            value = gains / denominator
    elif code in (24, 25, 26, 27):
        running_max = nav[0]
        minimum_drawdown = 0.0
        squared_drawdown = 0.0
        new_highs = 1
        if not math.isfinite(running_max) or running_max <= 0.0:
            status = STATUS_DOMAIN_ERROR
        for index in range(1, nav.size):
            item = nav[index]
            if not math.isfinite(item) or item <= 0.0:
                status = STATUS_DOMAIN_ERROR
                break
            if item > running_max:
                running_max = item
                new_highs += 1
            drawdown = item / running_max - 1.0
            minimum_drawdown = min(minimum_drawdown, drawdown)
            squared_drawdown += drawdown * drawdown
        if status == STATUS_OK:
            maximum_drawdown = abs(min(minimum_drawdown, 0.0))
            if code == 24:
                value = maximum_drawdown
            elif code == 25:
                value = math.sqrt(squared_drawdown / nav.size)
            elif code == 26:
                if maximum_drawdown == 0.0:
                    status = STATUS_DIVIDE_BY_ZERO
                else:
                    value = (growth ** (252.0 / count) - 1.0) / maximum_drawdown
            else:
                value = new_highs / nav.size
    elif code in (28, 29):
        if nav.size < 2:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            slope, r_squared = _linear_fit(nav)
            value = slope if code == 28 else r_squared
            if not math.isfinite(value):
                status = STATUS_DOMAIN_ERROR
    elif code == 30:
        if primary.size == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = _mean(primary)
    elif code == 31:
        if primary.size < 2:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = _sample_std(primary)
    elif code == 32:
        if primary.size == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = np.max(primary)
    elif code == 33:
        if primary.size == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = np.min(primary)
    elif code == 34:
        if primary.size == 0 or secondary.size == 0:
            status = STATUS_INSUFFICIENT_SAMPLE
        else:
            value = np.max(primary) - np.min(secondary)
    else:
        status = STATUS_DOMAIN_ERROR

    if status == STATUS_OK and not math.isfinite(value):
        status = STATUS_NON_FINITE_RESULT
    return value, status


@njit(cache=True, nogil=True, inline="always")
def _compute_product(
    row: int,
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    metric_codes: np.ndarray,
    primary_indices: np.ndarray,
    secondary_indices: np.ndarray,
    risk_free: np.ndarray,
    output: np.ndarray,
    statuses: np.ndarray,
) -> None:
    start = starts[row]
    end = ends[row]
    if start < 0 or end - start < 2:
        for metric_index in range(metric_codes.size):
            output[row, metric_index] = math.nan
            statuses[row, metric_index] = STATUS_INSUFFICIENT_SAMPLE
        return
    nav = values[0, start:end]
    returns = np.empty(nav.size - 1, dtype=np.float64)
    for index in range(returns.size):
        returns[index] = nav[index + 1] / nav[index] - 1.0
    for metric_index in range(metric_codes.size):
        primary_index = primary_indices[metric_index]
        secondary_index = secondary_indices[metric_index]
        primary = values[primary_index, start:end] if primary_index >= 0 else nav[:0]
        secondary = values[secondary_index, start:end] if secondary_index >= 0 else nav[:0]
        value, status = _compute_metric(
            metric_codes[metric_index],
            returns,
            nav,
            primary,
            secondary,
            risk_free[metric_index],
        )
        output[row, metric_index] = value
        statuses[row, metric_index] = status


_BATCH_SIGNATURE = void(
    float64[:, ::1],
    int64[::1],
    int64[::1],
    int64[::1],
    int64[::1],
    int64[::1],
    float64[::1],
    float64[:, ::1],
    int16[:, ::1],
)

_READONLY_BATCH_SIGNATURE = void(
    types.Array(float64, 2, "C", readonly=True),
    types.Array(int64, 1, "C", readonly=True),
    types.Array(int64, 1, "C", readonly=True),
    types.Array(int64, 1, "C", readonly=True),
    types.Array(int64, 1, "C", readonly=True),
    types.Array(int64, 1, "C", readonly=True),
    types.Array(float64, 1, "C", readonly=True),
    float64[:, ::1],
    int16[:, ::1],
)


@njit([_BATCH_SIGNATURE, _READONLY_BATCH_SIGNATURE], cache=True, nogil=True)
def compute_builtin_batch_serial(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    metric_codes: np.ndarray,
    primary_indices: np.ndarray,
    secondary_indices: np.ndarray,
    risk_free: np.ndarray,
    output: np.ndarray,
    statuses: np.ndarray,
) -> None:
    for row in range(starts.size):
        _compute_product(
            row,
            values,
            starts,
            ends,
            metric_codes,
            primary_indices,
            secondary_indices,
            risk_free,
            output,
            statuses,
        )


@njit(
    [_BATCH_SIGNATURE, _READONLY_BATCH_SIGNATURE],
    cache=True,
    nogil=True,
    parallel=True,
)
def compute_builtin_batch_parallel(
    values: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    metric_codes: np.ndarray,
    primary_indices: np.ndarray,
    secondary_indices: np.ndarray,
    risk_free: np.ndarray,
    output: np.ndarray,
    statuses: np.ndarray,
) -> None:
    for row in prange(starts.size):
        _compute_product(
            row,
            values,
            starts,
            ends,
            metric_codes,
            primary_indices,
            secondary_indices,
            risk_free,
            output,
            statuses,
        )


def warm_builtin_batch_kernels() -> None:
    values = np.ascontiguousarray([[1.0, 1.01, 1.02]], dtype=np.float64)
    starts = np.ascontiguousarray([0], dtype=np.int64)
    ends = np.ascontiguousarray([3], dtype=np.int64)
    codes = np.ascontiguousarray([0], dtype=np.int64)
    indices = np.ascontiguousarray([-1], dtype=np.int64)
    risk_free = np.ascontiguousarray([0.0], dtype=np.float64)
    output = np.empty((1, 1), dtype=np.float64)
    statuses = np.empty((1, 1), dtype=np.int16)
    compute_builtin_batch_serial(
        values, starts, ends, codes, indices, indices, risk_free, output, statuses
    )
    compute_builtin_batch_parallel(
        values, starts, ends, codes, indices, indices, risk_free, output, statuses
    )


__all__ = [
    "BUILTIN_METRIC_CODE",
    "BUILTIN_METRIC_IDS",
    "STATUS_DIVIDE_BY_ZERO",
    "STATUS_DOMAIN_ERROR",
    "STATUS_INSUFFICIENT_SAMPLE",
    "STATUS_NON_FINITE_RESULT",
    "STATUS_OK",
    "compute_builtin_batch_parallel",
    "compute_builtin_batch_serial",
    "warm_builtin_batch_kernels",
]
