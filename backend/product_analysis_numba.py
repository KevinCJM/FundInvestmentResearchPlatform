from __future__ import annotations

"""Fixed-signature nopython kernels for single-product research charts.

The module contains no pandas, Pydantic, I/O or Python numerical fallback.
Every production dispatcher has an eager signature and ``cache=False`` so the
backend-package and backend-working-directory import styles cannot share an
ambiguous on-disk Numba cache.
"""

import hashlib
import math

import numpy as np
from numba import boolean, float64, int64, njit, types, uint8, uint64


PRODUCT_ANALYSIS_ENGINE_VERSION = "product-analysis-njit-1.0.0"
PRODUCT_ANALYSIS_KERNEL_VERSION = "product-chart-statistics-simulation-2"

_F1 = float64[::1]
_F2 = float64[:, ::1]
_I1 = int64[::1]
_U1 = uint8[::1]

_BOX_RESULT = types.Tuple((_F1, _F1))
_QQ_RESULT = types.Tuple((_F2, _F2))
_SIM_RESULT = types.Tuple((_F2, _F2, _F1, _F1, _F1))
_DENSITY_RESULT = types.Tuple((_F2, _F2, _F1))
_RANDOM_RESULT = types.Tuple((uint64, float64))


@njit(_F2(_F1, _I1), cache=False, nogil=True)
def moving_average_kernel(values: np.ndarray, periods: np.ndarray) -> np.ndarray:
    output = np.full((periods.size, values.size), np.nan, dtype=np.float64)
    for row in range(periods.size):
        period = periods[row]
        if period <= 0:
            raise ValueError("moving-average period must be positive")
        running_sum = 0.0
        invalid_count = 0
        for index in range(values.size):
            value = values[index]
            if np.isfinite(value):
                running_sum += value
            else:
                invalid_count += 1
            if index >= period:
                expired = values[index - period]
                if np.isfinite(expired):
                    running_sum -= expired
                else:
                    invalid_count -= 1
            if index + 1 >= period and invalid_count == 0:
                output[row, index] = running_sum / period
    return output


@njit(_I1(_F1, _F1, _F1, _F1, _F1), cache=False, nogil=True)
def technical_input_availability_kernel(
    open_values: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
) -> np.ndarray:
    if not (
        open_values.size == high.size
        and open_values.size == low.size
        and open_values.size == close.size
        and open_values.size == volume.size
    ):
        raise ValueError("technical input arrays must have equal length")
    has_ohlc = close.size > 0
    has_kdj = close.size > 0
    has_volume = False
    for index in range(close.size):
        if not (
            np.isfinite(open_values[index])
            and np.isfinite(high[index])
            and np.isfinite(low[index])
            and np.isfinite(close[index])
        ):
            has_ohlc = False
        if not (
            np.isfinite(high[index])
            and np.isfinite(low[index])
            and np.isfinite(close[index])
        ):
            has_kdj = False
        if np.isfinite(volume[index]):
            has_volume = True
    return np.array([int(has_ohlc), int(has_volume), int(has_kdj)], dtype=np.int64)


@njit(_F2(_F1, int64, float64), cache=False, nogil=True)
def bollinger_kernel(values: np.ndarray, period: int, multiplier: float) -> np.ndarray:
    if period < 2 or not np.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("invalid Bollinger parameters")
    output = np.full((3, values.size), np.nan, dtype=np.float64)
    for index in range(period - 1, values.size):
        total = 0.0
        valid = True
        for position in range(index - period + 1, index + 1):
            value = values[position]
            if not np.isfinite(value):
                valid = False
                break
            total += value
        if not valid:
            continue
        mean = total / period
        variance_sum = 0.0
        for position in range(index - period + 1, index + 1):
            difference = values[position] - mean
            variance_sum += difference * difference
        deviation = math.sqrt(variance_sum / period)
        output[0, index] = mean + multiplier * deviation
        output[1, index] = mean
        output[2, index] = mean - multiplier * deviation
    return output


@njit(_F2(_F1, _F1, _F1, int64, int64, int64), cache=False, nogil=True)
def kdj_kernel(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    k_smoothing: int,
    d_smoothing: int,
) -> np.ndarray:
    if high.size != low.size or high.size != close.size:
        raise ValueError("KDJ arrays must have equal length")
    if period < 2 or k_smoothing < 1 or d_smoothing < 1:
        raise ValueError("invalid KDJ parameters")
    output = np.full((3, close.size), np.nan, dtype=np.float64)
    previous_k = 50.0
    previous_d = 50.0
    for index in range(close.size):
        start = max(0, index - period + 1)
        highest = -np.inf
        lowest = np.inf
        valid = np.isfinite(close[index])
        for position in range(start, index + 1):
            if not np.isfinite(high[position]) or not np.isfinite(low[position]):
                valid = False
                break
            if high[position] > highest:
                highest = high[position]
            if low[position] < lowest:
                lowest = low[position]
        if not valid:
            continue
        rsv = 50.0
        if highest != lowest:
            rsv = (close[index] - lowest) / (highest - lowest) * 100.0
        current_k = ((k_smoothing - 1.0) * previous_k + rsv) / k_smoothing
        current_d = ((d_smoothing - 1.0) * previous_d + current_k) / d_smoothing
        current_j = 3.0 * current_k - 2.0 * current_d
        output[0, index] = current_k
        output[1, index] = current_d
        output[2, index] = current_j
        previous_k = current_k
        previous_d = current_d
    return output


@njit(_F1(_F1), cache=False, nogil=True)
def daily_returns_percent_kernel(close: np.ndarray) -> np.ndarray:
    if close.size < 2:
        return np.empty(0, dtype=np.float64)
    output = np.full(close.size - 1, np.nan, dtype=np.float64)
    for index in range(1, close.size):
        previous = close[index - 1]
        current = close[index]
        if np.isfinite(previous) and previous != 0.0 and np.isfinite(current):
            value = (current / previous - 1.0) * 100.0
            if np.isfinite(value):
                output[index - 1] = value
    return output


@njit(float64(_F1, float64), cache=False, nogil=True, inline="always")
def _linear_quantile(sorted_values: np.ndarray, probability: float) -> float:
    if sorted_values.size == 0:
        return np.nan
    position = (sorted_values.size - 1) * probability
    lower_index = int(math.floor(position))
    upper_index = min(sorted_values.size - 1, lower_index + 1)
    weight = position - lower_index
    return sorted_values[lower_index] + (
        sorted_values[upper_index] - sorted_values[lower_index]
    ) * weight


@njit(_F1(_F1), cache=False, nogil=True)
def return_statistics_kernel(values: np.ndarray) -> np.ndarray:
    """mean, pop std, median, positive ratio, best, worst, n, skew, kurt, JB, p."""

    valid_count = 0
    for value in values:
        if np.isfinite(value):
            valid_count += 1
    output = np.full(11, np.nan, dtype=np.float64)
    output[6] = valid_count
    if valid_count == 0:
        return output
    finite = np.empty(valid_count, dtype=np.float64)
    total = 0.0
    positive = 0
    position = 0
    best = -np.inf
    worst = np.inf
    for value in values:
        if not np.isfinite(value):
            continue
        finite[position] = value
        position += 1
        total += value
        if value > 0.0:
            positive += 1
        if value > best:
            best = value
        if value < worst:
            worst = value
    mean = total / valid_count
    variance_sum = 0.0
    for value in finite:
        difference = value - mean
        variance_sum += difference * difference
    standard_deviation = math.sqrt(variance_sum / valid_count)
    finite.sort()
    output[0] = mean
    output[1] = standard_deviation
    output[2] = _linear_quantile(finite, 0.5)
    output[3] = positive / valid_count
    output[4] = best
    output[5] = worst
    if standard_deviation <= 0.0:
        return output
    third = 0.0
    fourth = 0.0
    for value in finite:
        standardized = (value - mean) / standard_deviation
        squared = standardized * standardized
        third += squared * standardized
        fourth += squared * squared
    if valid_count > 2:
        skewness = (
            math.sqrt(valid_count * (valid_count - 1.0))
            / (valid_count - 2.0)
            * (third / valid_count)
        )
        output[7] = skewness
    if valid_count > 3:
        population_excess = fourth / valid_count - 3.0
        excess_kurtosis = (
            (valid_count - 1.0)
            / ((valid_count - 2.0) * (valid_count - 3.0))
            * ((valid_count + 1.0) * population_excess + 6.0)
        )
        output[8] = excess_kurtosis
    if np.isfinite(output[7]) and np.isfinite(output[8]):
        jb = valid_count / 6.0 * (
            output[7] * output[7] + output[8] * output[8] / 4.0
        )
        output[9] = jb
        output[10] = math.exp(-jb / 2.0)
    return output


@njit(_I1(_F1), cache=False, nogil=True)
def distribution_interpretation_codes_kernel(statistics: np.ndarray) -> np.ndarray:
    """Map skew, kurtosis and normality results to presentation codes."""

    output = np.full(3, 99, dtype=np.int64)
    skewness = statistics[7]
    if np.isfinite(skewness):
        if skewness <= -1.0:
            output[0] = -3
        elif skewness <= -0.5:
            output[0] = -2
        elif skewness <= -0.1:
            output[0] = -1
        elif skewness >= 1.0:
            output[0] = 3
        elif skewness >= 0.5:
            output[0] = 2
        elif skewness >= 0.1:
            output[0] = 1
        else:
            output[0] = 0
    kurtosis = statistics[8]
    if np.isfinite(kurtosis):
        if kurtosis >= 3.0:
            output[1] = 3
        elif kurtosis >= 1.0:
            output[1] = 2
        elif kurtosis >= 0.1:
            output[1] = 1
        elif kurtosis <= -1.0:
            output[1] = -3
        elif kurtosis <= -0.5:
            output[1] = -2
        elif kurtosis <= -0.1:
            output[1] = -1
        else:
            output[1] = 0
    p_value = statistics[10]
    if np.isfinite(p_value):
        output[2] = 1 if p_value < 0.05 else 0
    return output


@njit(_F2(_F1, float64), cache=False, nogil=True)
def histogram_normal_pdf_kernel(values: np.ndarray, bin_width: float) -> np.ndarray:
    safe_width = max(bin_width, 0.01)
    stats = return_statistics_kernel(values)
    valid_count = int(stats[6])
    if valid_count == 0:
        return np.empty((0, 6), dtype=np.float64)
    minimum = stats[5]
    maximum = stats[4]
    normalized_minimum = math.floor(minimum / safe_width) * safe_width
    normalized_maximum = math.ceil(maximum / safe_width) * safe_width
    if minimum == maximum:
        normalized_maximum = normalized_minimum + safe_width
    bin_count = max(
        1,
        int(round((normalized_maximum - normalized_minimum) / safe_width)),
    )
    output = np.zeros((bin_count, 6), dtype=np.float64)
    for index in range(bin_count):
        lower = normalized_minimum + index * safe_width
        upper = normalized_maximum if index == bin_count - 1 else lower + safe_width
        output[index, 0] = lower
        output[index, 1] = upper
        output[index, 5] = (lower + upper) / 2.0
    for value in values:
        if not np.isfinite(value):
            continue
        index = int(math.floor((value - normalized_minimum) / safe_width))
        index = max(0, min(bin_count - 1, index))
        output[index, 2] += 1.0
    standard_deviation = stats[1]
    for index in range(bin_count):
        output[index, 4] = output[index, 2] / valid_count
        if np.isfinite(standard_deviation) and standard_deviation > 0.0:
            center = output[index, 5]
            standardized = (center - stats[0]) / standard_deviation
            pdf = (
                math.exp(-0.5 * standardized * standardized)
                / (math.sqrt(2.0 * math.pi) * standard_deviation)
            )
            output[index, 3] = pdf * valid_count * safe_width
        else:
            output[index, 3] = np.nan
    return output


@njit(_BOX_RESULT(_F1), cache=False, nogil=True)
def box_plot_kernel(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_count = 0
    for value in values:
        if np.isfinite(value):
            valid_count += 1
    if valid_count < 5:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    finite = np.empty(valid_count, dtype=np.float64)
    position = 0
    for value in values:
        if np.isfinite(value):
            finite[position] = value
            position += 1
    finite.sort()
    first_quartile = _linear_quantile(finite, 0.25)
    median = _linear_quantile(finite, 0.5)
    third_quartile = _linear_quantile(finite, 0.75)
    interquartile_range = third_quartile - first_quartile
    lower_fence = first_quartile - 1.5 * interquartile_range
    upper_fence = third_quartile + 1.5 * interquartile_range
    lower_whisker = finite[0]
    upper_whisker = finite[finite.size - 1]
    for value in finite:
        if value >= lower_fence:
            lower_whisker = value
            break
    for reverse_index in range(finite.size):
        value = finite[finite.size - 1 - reverse_index]
        if value <= upper_fence:
            upper_whisker = value
            break
    outlier_count = 0
    for value in finite:
        if value < lower_whisker or value > upper_whisker:
            outlier_count += 1
    outliers = np.empty(outlier_count, dtype=np.float64)
    position = 0
    for value in finite:
        if value < lower_whisker or value > upper_whisker:
            outliers[position] = value
            position += 1
    statistics = np.array(
        [
            lower_whisker,
            first_quartile,
            median,
            third_quartile,
            upper_whisker,
            interquartile_range,
        ],
        dtype=np.float64,
    )
    return statistics, outliers


@njit(float64(float64), cache=False, nogil=True, inline="always")
def _inverse_standard_normal(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return np.nan
    a0, a1, a2 = -39.6968302866538, 220.946098424521, -275.928510446969
    a3, a4, a5 = 138.357751867269, -30.6647980661472, 2.50662827745924
    b0, b1, b2 = -54.4760987982241, 161.585836858041, -155.698979859887
    b3, b4 = 66.8013118877197, -13.2806815528857
    c0, c1, c2 = -0.00778489400243029, -0.322396458041136, -2.40075827716184
    c3, c4, c5 = -2.54973253934373, 4.37466414146497, 2.93816398269878
    d0, d1, d2, d3 = 0.00778469570904146, 0.32246712907004, 2.445134137143, 3.75440866190742
    lower_boundary = 0.02425
    if probability < lower_boundary:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
        )
    if probability > 1.0 - lower_boundary:
        q = math.sqrt(-2.0 * math.log(1.0 - probability))
        return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5) / (
            (((d0 * q + d1) * q + d2) * q + d3) * q + 1.0
        )
    q = probability - 0.5
    r = q * q
    numerator = (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
    denominator = ((((b0 * r + b1) * r + b2) * r + b3) * r + b4) * r + 1.0
    return numerator / denominator


@njit(_QQ_RESULT(_F1), cache=False, nogil=True)
def normal_qq_kernel(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_count = 0
    for value in values:
        if np.isfinite(value):
            valid_count += 1
    if valid_count < 3:
        return np.empty((0, 5), dtype=np.float64), np.empty((0, 5), dtype=np.float64)
    sorted_values = np.empty(valid_count, dtype=np.float64)
    position = 0
    for value in values:
        if np.isfinite(value):
            sorted_values[position] = value
            position += 1
    sorted_values.sort()
    theoretical_q1 = _inverse_standard_normal(0.25)
    theoretical_q3 = _inverse_standard_normal(0.75)
    observed_q1 = _linear_quantile(sorted_values, 0.25)
    observed_q3 = _linear_quantile(sorted_values, 0.75)
    slope = (observed_q3 - observed_q1) / (theoretical_q3 - theoretical_q1)
    intercept = observed_q1 - slope * theoretical_q1
    points = np.empty((valid_count, 5), dtype=np.float64)
    for index in range(valid_count):
        percentile = (index + 0.5) / valid_count
        theoretical = _inverse_standard_normal(percentile)
        tail_code = 0.0 if percentile <= 0.1 else (2.0 if percentile >= 0.9 else 1.0)
        points[index, 0] = percentile
        points[index, 1] = theoretical
        points[index, 2] = sorted_values[index]
        points[index, 3] = intercept + slope * theoretical
        points[index, 4] = tail_code
    targets = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    key_points = np.full((targets.size, 5), np.nan, dtype=np.float64)
    used = np.zeros(valid_count, dtype=np.uint8)
    for target_index in range(targets.size):
        best_index = 0
        best_distance = abs(points[0, 0] - targets[target_index])
        for index in range(1, valid_count):
            distance = abs(points[index, 0] - targets[target_index])
            if distance < best_distance:
                best_index = index
                best_distance = distance
        if used[best_index] == 0:
            used[best_index] = 1
            for column in range(5):
                key_points[target_index, column] = points[best_index, column]
    return points, key_points


@njit(_F1(_F1), cache=False, nogil=True)
def _adjusted_distribution_shape(values: np.ndarray) -> np.ndarray:
    output = np.full(5, np.nan, dtype=np.float64)
    if values.size < 2:
        return output
    total = 0.0
    for value in values:
        if not np.isfinite(value):
            return output
        total += value
    mean = total / values.size
    second = 0.0
    for value in values:
        difference = value - mean
        second += difference * difference
    second /= values.size
    if not np.isfinite(second) or second <= np.finfo(np.float64).eps:
        return output
    standard_deviation = math.sqrt(second)
    third = 0.0
    fourth = 0.0
    for value in values:
        standardized = (value - mean) / standard_deviation
        square = standardized * standardized
        third += square * standardized
        fourth += square * square
    population_skew = third / values.size
    population_kurtosis = fourth / values.size - 3.0
    output[0] = mean
    output[1] = standard_deviation
    if values.size > 2:
        output[2] = (
            math.sqrt(values.size * (values.size - 1.0))
            / (values.size - 2.0)
            * population_skew
        )
    if values.size > 3:
        output[3] = (
            (values.size - 1.0)
            / ((values.size - 2.0) * (values.size - 3.0))
            * ((values.size + 1.0) * population_kurtosis + 6.0)
        )
    output[4] = 1.0
    return output


@njit(_F1(float64, float64), cache=False, nogil=True)
def _calibrate_sinh_arcsinh(target_skewness: float, target_kurtosis: float) -> np.ndarray:
    """Return active, status, skew parameter, tail weight, mean/std/skew/kurt."""

    output = np.full(8, np.nan, dtype=np.float64)
    if not np.isfinite(target_skewness) or not np.isfinite(target_kurtosis):
        output[0] = 0.0
        output[1] = 2.0
        return output
    sample = np.empty(801, dtype=np.float64)
    skew_scale = max(0.5, abs(target_skewness))
    kurtosis_scale = max(1.0, abs(target_kurtosis))
    normal_score = (target_skewness / skew_scale) ** 2 + (
        target_kurtosis / kurtosis_scale
    ) ** 2
    best_score = np.inf
    best_skew_parameter = 0.0
    best_tail_weight = 1.0
    best_shape = np.full(5, np.nan, dtype=np.float64)

    coarse_skew = np.array([-1.5, -1.2, -0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5])
    coarse_tail = np.array([0.45, 0.55, 0.65, 0.75, 0.85, 1.0, 1.15, 1.35, 1.6, 2.0, 2.5])
    for skew_parameter in coarse_skew:
        for tail_weight in coarse_tail:
            for index in range(sample.size):
                normal_value = _inverse_standard_normal((index + 0.5) / sample.size)
                sample[index] = math.sinh(
                    (math.asinh(normal_value) + skew_parameter) / tail_weight
                )
            shape = _adjusted_distribution_shape(sample)
            if shape[4] != 1.0:
                continue
            score = ((shape[2] - target_skewness) / skew_scale) ** 2 + (
                (shape[3] - target_kurtosis) / kurtosis_scale
            ) ** 2
            if score < best_score:
                best_score = score
                best_skew_parameter = skew_parameter
                best_tail_weight = tail_weight
                best_shape = shape

    skew_step = 0.15
    tail_step = 0.12
    for _ in range(18):
        anchor_skew = best_skew_parameter
        anchor_tail = best_tail_weight
        for skew_direction in range(-1, 2):
            for tail_direction in range(-1, 2):
                if skew_direction == 0 and tail_direction == 0:
                    continue
                skew_parameter = anchor_skew + skew_direction * skew_step
                tail_weight = anchor_tail + tail_direction * tail_step
                if abs(skew_parameter) > 2.0 or tail_weight < 0.4 or tail_weight > 3.0:
                    continue
                for index in range(sample.size):
                    normal_value = _inverse_standard_normal((index + 0.5) / sample.size)
                    sample[index] = math.sinh(
                        (math.asinh(normal_value) + skew_parameter) / tail_weight
                    )
                shape = _adjusted_distribution_shape(sample)
                if shape[4] != 1.0:
                    continue
                score = ((shape[2] - target_skewness) / skew_scale) ** 2 + (
                    (shape[3] - target_kurtosis) / kurtosis_scale
                ) ** 2
                if score < best_score:
                    best_score = score
                    best_skew_parameter = skew_parameter
                    best_tail_weight = tail_weight
                    best_shape = shape
        skew_step *= 0.72
        tail_step *= 0.72

    matched = (
        abs(best_shape[2] - target_skewness)
        <= max(0.08, abs(target_skewness) * 0.15)
        and abs(best_shape[3] - target_kurtosis)
        <= max(0.25, abs(target_kurtosis) * 0.15)
    )
    status = 0.0 if matched else (1.0 if best_score < normal_score * 0.8 else 2.0)
    active = 1.0 if status < 2.0 else 0.0
    output[0] = active
    output[1] = status
    output[2] = best_skew_parameter
    output[3] = best_tail_weight
    output[4] = best_shape[0]
    output[5] = best_shape[1]
    output[6] = best_shape[2]
    output[7] = best_shape[3]
    return output


@njit(_RANDOM_RESULT(uint64), cache=False, nogil=True, inline="always")
def _next_uniform(state: int) -> tuple[int, float]:
    state = state * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
    value = ((state >> np.uint64(11)) & np.uint64(9007199254740991)) / 9007199254740992.0
    return state, value


@njit(_SIM_RESULT(_F2, _F1, float64, float64), cache=False, nogil=True)
def _summarize_simulation(
    paths: np.ndarray,
    max_drawdowns: np.ndarray,
    initial_nav: float,
    target_return: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_count, day_count = paths.shape
    sample_count = min(12, path_count)
    sample_paths = np.ascontiguousarray(paths[:sample_count].copy())
    percentiles = np.empty((5, day_count), dtype=np.float64)
    probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    scratch = np.empty(path_count, dtype=np.float64)
    for day in range(day_count):
        for path in range(path_count):
            scratch[path] = paths[path, day]
        scratch.sort()
        for index in range(probabilities.size):
            percentiles[index, day] = _linear_quantile(scratch, probabilities[index])
    terminal = np.ascontiguousarray(paths[:, day_count - 1].copy())
    terminal.sort()
    terminal_returns = np.empty(path_count, dtype=np.float64)
    loss_count = 0
    target_count = 0
    for index in range(path_count):
        value = terminal[index] / initial_nav - 1.0
        terminal_returns[index] = value
        if value < 0.0:
            loss_count += 1
        if value >= target_return:
            target_count += 1
    tail_boundary = _linear_quantile(terminal_returns, 0.05)
    tail_total = 0.0
    tail_count = 0
    for value in terminal_returns:
        if value <= tail_boundary:
            tail_total += value
            tail_count += 1
    drawdown_total = 0.0
    for value in max_drawdowns:
        drawdown_total += value
    summary = np.array(
        [
            _linear_quantile(terminal, 0.05),
            _linear_quantile(terminal, 0.25),
            _linear_quantile(terminal, 0.5),
            _linear_quantile(terminal, 0.75),
            _linear_quantile(terminal, 0.95),
            loss_count / path_count,
            max(0.0, -tail_boundary),
            max(0.0, -(tail_total / max(1, tail_count))),
            target_count / path_count,
            drawdown_total / path_count,
            _linear_quantile(terminal_returns, 0.05),
            _linear_quantile(terminal_returns, 0.5),
        ],
        dtype=np.float64,
    )
    days = np.arange(day_count, dtype=np.float64)
    return sample_paths, percentiles, terminal, summary, days


@njit(
    _SIM_RESULT(_F1, float64, int64, int64, int64, float64),
    cache=False,
    nogil=True,
)
def parametric_monte_carlo_kernel(
    returns_percent: np.ndarray,
    initial_nav: float,
    horizon_days: int,
    path_count: int,
    seed: int,
    target_return_percent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_count = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            valid_count += 1
    if valid_count < 20 or initial_nav <= 0.0 or horizon_days < 1 or path_count < 1:
        raise ValueError("parametric simulation requires valid parameters and 20 returns")
    log_returns = np.empty(valid_count, dtype=np.float64)
    position = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            log_returns[position] = math.log1p(value / 100.0)
            position += 1
    shape = _adjusted_distribution_shape(log_returns)
    mean = shape[0]
    sample_variance = 0.0
    for value in log_returns:
        difference = value - mean
        sample_variance += difference * difference
    volatility = math.sqrt(max(0.0, sample_variance / (valid_count - 1)))
    calibration = _calibrate_sinh_arcsinh(shape[2], shape[3])
    paths = np.empty((path_count, horizon_days + 1), dtype=np.float64)
    max_drawdowns = np.empty(path_count, dtype=np.float64)
    state = np.uint64(seed if seed != 0 else 1)
    for path_index in range(path_count):
        nav = initial_nav
        peak = initial_nav
        worst = 0.0
        paths[path_index, 0] = nav
        has_spare = False
        spare = 0.0
        for day in range(1, horizon_days + 1):
            if has_spare:
                normal_value = spare
                has_spare = False
            else:
                state, first = _next_uniform(state)
                state, second = _next_uniform(state)
                first = max(first, np.finfo(np.float64).eps)
                magnitude = math.sqrt(-2.0 * math.log(first))
                normal_value = magnitude * math.cos(2.0 * math.pi * second)
                spare = magnitude * math.sin(2.0 * math.pi * second)
                has_spare = True
            innovation = normal_value
            if calibration[0] == 1.0:
                transformed = math.sinh(
                    (math.asinh(normal_value) + calibration[2]) / calibration[3]
                )
                innovation = (transformed - calibration[4]) / calibration[5]
            nav *= math.exp(mean + volatility * innovation)
            if not np.isfinite(nav) or nav <= 0.0:
                raise ValueError("non-finite parametric simulation path")
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak
            if drawdown > worst:
                worst = drawdown
            paths[path_index, day] = nav
        max_drawdowns[path_index] = worst
    sample, percentiles, terminal, summary, days = _summarize_simulation(
        paths,
        max_drawdowns,
        initial_nav,
        target_return_percent / 100.0,
    )
    assumptions = np.array(
        [
            valid_count,
            target_return_percent,
            mean,
            volatility,
            shape[2],
            shape[3],
            calibration[6] if calibration[0] == 1.0 else 0.0,
            calibration[7] if calibration[0] == 1.0 else 0.0,
            calibration[1],
            calibration[2] if calibration[0] == 1.0 else np.nan,
            calibration[3] if calibration[0] == 1.0 else np.nan,
            np.nan,
        ],
        dtype=np.float64,
    )
    # The fifth return lane is reserved for days by the common result contract;
    # attach assumptions after the day vector so the service can reconstruct it.
    packed = np.empty(days.size + assumptions.size, dtype=np.float64)
    packed[: days.size] = days
    packed[days.size :] = assumptions
    return sample, percentiles, terminal, summary, packed


@njit(
    _SIM_RESULT(_F1, float64, int64, int64, int64, float64, int64),
    cache=False,
    nogil=True,
)
def stationary_block_bootstrap_kernel(
    returns_percent: np.ndarray,
    initial_nav: float,
    horizon_days: int,
    path_count: int,
    seed: int,
    target_return_percent: float,
    average_block_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_count = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            valid_count += 1
    if valid_count < 20 or initial_nav <= 0.0 or horizon_days < 1 or path_count < 1:
        raise ValueError("bootstrap simulation requires valid parameters and 20 returns")
    returns = np.empty(valid_count, dtype=np.float64)
    position = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            returns[position] = value / 100.0
            position += 1
    block_length = max(1, min(valid_count, average_block_length))
    paths = np.empty((path_count, horizon_days + 1), dtype=np.float64)
    max_drawdowns = np.empty(path_count, dtype=np.float64)
    state = np.uint64(seed if seed != 0 else 1)
    for path_index in range(path_count):
        state, first = _next_uniform(state)
        source_index = min(valid_count - 1, int(math.floor(first * valid_count)))
        nav = initial_nav
        peak = initial_nav
        worst = 0.0
        paths[path_index, 0] = nav
        for day in range(1, horizon_days + 1):
            if day > 1:
                state, decision = _next_uniform(state)
                if decision >= 1.0 / block_length:
                    source_index = (source_index + 1) % valid_count
                else:
                    state, choice = _next_uniform(state)
                    source_index = min(
                        valid_count - 1,
                        int(math.floor(choice * valid_count)),
                    )
            nav *= 1.0 + returns[source_index]
            if not np.isfinite(nav) or nav <= 0.0:
                raise ValueError("non-finite bootstrap simulation path")
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak
            if drawdown > worst:
                worst = drawdown
            paths[path_index, day] = nav
        max_drawdowns[path_index] = worst
    sample, percentiles, terminal, summary, days = _summarize_simulation(
        paths,
        max_drawdowns,
        initial_nav,
        target_return_percent / 100.0,
    )
    assumptions = np.array(
        [valid_count, target_return_percent, np.nan, np.nan, np.nan, np.nan,
         np.nan, np.nan, np.nan, np.nan, np.nan, block_length],
        dtype=np.float64,
    )
    packed = np.empty(days.size + assumptions.size, dtype=np.float64)
    packed[: days.size] = days
    packed[days.size :] = assumptions
    return sample, percentiles, terminal, summary, packed


@njit(_F1(_F1, _F1, float64), cache=False, nogil=True)
def simulation_comparison_kernel(
    parametric_summary: np.ndarray,
    bootstrap_summary: np.ndarray,
    initial_nav: float,
) -> np.ndarray:
    output = np.empty(5, dtype=np.float64)
    output[0] = abs(
        (parametric_summary[0] / initial_nav - 1.0)
        - (bootstrap_summary[0] / initial_nav - 1.0)
    )
    output[1] = abs(
        (parametric_summary[2] / initial_nav - 1.0)
        - (bootstrap_summary[2] / initial_nav - 1.0)
    )
    output[2] = abs(parametric_summary[5] - bootstrap_summary[5])
    output[3] = abs(parametric_summary[7] - bootstrap_summary[7])
    largest = max(output[0], output[1], output[2], output[3])
    output[4] = 2.0 if largest >= 0.1 else (1.0 if largest >= 0.03 else 0.0)
    return output


@njit(_DENSITY_RESULT(_F1, _F2, int64, float64), cache=False, nogil=True)
def terminal_density_kernel(
    terminal_values: np.ndarray,
    percentiles: np.ndarray,
    point_count: int,
    initial_nav: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("initial NAV must be positive")
    sorted_values = terminal_values.copy()
    sorted_values.sort()
    if sorted_values.size < 2:
        return (
            np.empty((0, 4), dtype=np.float64),
            np.empty((0, 5), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    total = 0.0
    for value in sorted_values:
        total += value
    mean = total / sorted_values.size
    variance_sum = 0.0
    for value in sorted_values:
        difference = value - mean
        variance_sum += difference * difference
    standard_deviation = math.sqrt(max(0.0, variance_sum / max(1, sorted_values.size - 1)))
    interquartile_range = _linear_quantile(sorted_values, 0.75) - _linear_quantile(sorted_values, 0.25)
    robust_scale = standard_deviation
    alternate_scale = interquartile_range / 1.34
    if robust_scale <= 0.0 or (alternate_scale > 0.0 and alternate_scale < robust_scale):
        robust_scale = alternate_scale
    minimum_bandwidth = max(abs(mean) * 0.0005, 1e-6)
    bandwidth = max(
        minimum_bandwidth,
        0.9 * robust_scale * sorted_values.size ** (-0.2),
    )
    minimum_nav = max(0.0, _linear_quantile(sorted_values, 0.01) - bandwidth * 2.0)
    maximum_nav = max(
        minimum_nav + minimum_bandwidth,
        _linear_quantile(sorted_values, 0.99) + bandwidth * 2.0,
    )
    safe_point_count = max(21, min(161, point_count))
    points = np.empty((safe_point_count, 4), dtype=np.float64)
    normalizer = sorted_values.size * bandwidth * math.sqrt(2.0 * math.pi)
    mode_index = 0
    for index in range(safe_point_count):
        nav = minimum_nav + (maximum_nav - minimum_nav) * index / (safe_point_count - 1)
        kernel_sum = 0.0
        for value in sorted_values:
            standardized = (nav - value) / bandwidth
            kernel_sum += math.exp(-0.5 * standardized * standardized)
        density = kernel_sum / normalizer
        points[index, 0] = nav
        points[index, 1] = density
        points[index, 3] = nav / initial_nav - 1.0
        if index == 0 or density > points[mode_index, 1]:
            mode_index = index
    bin_count = max(8, min(28, int(round(math.sqrt(sorted_values.size)))))
    bin_width = (maximum_nav - minimum_nav) / bin_count
    histogram = np.zeros((bin_count, 5), dtype=np.float64)
    for index in range(bin_count):
        histogram[index, 0] = minimum_nav + index * bin_width
        histogram[index, 1] = minimum_nav + (index + 1) * bin_width
    for value in sorted_values:
        raw_index = int(math.floor((value - minimum_nav) / bin_width))
        index = max(0, min(bin_count - 1, raw_index))
        histogram[index, 3] += 1.0
    max_density = points[mode_index, 1]
    density_count_factor = sorted_values.size * bin_width
    maximum_count = 1.0
    for index in range(points.shape[0]):
        points[index, 2] = points[index, 1] * density_count_factor
        if points[index, 2] > maximum_count:
            maximum_count = points[index, 2]
    for index in range(histogram.shape[0]):
        histogram[index, 2] = histogram[index, 3] / density_count_factor
        histogram[index, 4] = histogram[index, 3] / sorted_values.size
        if histogram[index, 2] > max_density:
            max_density = histogram[index, 2]
        if histogram[index, 3] > maximum_count:
            maximum_count = histogram[index, 3]
    observed_minimum = minimum_nav
    observed_maximum = maximum_nav
    for row in (0, 4):
        for value in percentiles[row]:
            if value < observed_minimum:
                observed_minimum = value
            if value > observed_maximum:
                observed_maximum = value
    padding = max(
        (observed_maximum - observed_minimum) * 0.04,
        abs(observed_maximum) * 0.001,
        0.0001,
    )
    summary = np.array(
        [
            max_density,
            points[mode_index, 0],
            minimum_nav,
            maximum_nav,
            math.ceil(maximum_count * 1.08),
            max(0.0, observed_minimum - padding),
            observed_maximum + padding,
            density_count_factor,
            bin_width,
            sorted_values.size,
        ],
        dtype=np.float64,
    )
    return points, histogram, summary


@njit(
    _F2(_I1, _F1, _I1, _I1, _I1, int64),
    cache=False,
    nogil=True,
)
def regime_performance_kernel(
    date_days: np.ndarray,
    close: np.ndarray,
    segment_start_days: np.ndarray,
    segment_end_days: np.ndarray,
    segment_state_codes: np.ndarray,
    state_count: int,
) -> np.ndarray:
    if date_days.size != close.size:
        raise ValueError("regime dates and close arrays must have equal length")
    if not (
        segment_start_days.size == segment_end_days.size
        and segment_start_days.size == segment_state_codes.size
    ):
        raise ValueError("regime segment arrays must have equal length")
    output = np.full((state_count, 6), np.nan, dtype=np.float64)
    state_by_row = np.full(date_days.size, -1, dtype=np.int64)
    for segment_index in range(segment_start_days.size):
        state_code = segment_state_codes[segment_index]
        if state_code < 0 or state_code >= state_count:
            continue
        for row in range(date_days.size):
            if segment_start_days[segment_index] <= date_days[row] <= segment_end_days[segment_index]:
                state_by_row[row] = state_code
    for state_code in range(state_count):
        observation_count = 0
        return_count = 0
        wealth = 1.0
        return_sum = 0.0
        return_square_sum = 0.0
        positive_count = 0
        worst_drawdown = np.nan
        for row in range(date_days.size):
            if state_by_row[row] == state_code:
                observation_count += 1
        for segment_index in range(segment_start_days.size):
            if segment_state_codes[segment_index] != state_code:
                continue
            previous = np.nan
            peak = np.nan
            segment_worst = 0.0
            for row in range(date_days.size):
                if not (
                    segment_start_days[segment_index]
                    <= date_days[row]
                    <= segment_end_days[segment_index]
                ):
                    continue
                current = close[row]
                if not np.isfinite(current) or current <= 0.0:
                    continue
                if not np.isfinite(peak) or current > peak:
                    peak = current
                if np.isfinite(previous):
                    value = current / previous - 1.0
                    wealth *= 1.0 + value
                    return_sum += value
                    return_square_sum += value * value
                    return_count += 1
                    if value > 0.0:
                        positive_count += 1
                drawdown = current / peak - 1.0
                if drawdown < segment_worst:
                    segment_worst = drawdown
                previous = current
            if not np.isfinite(worst_drawdown) or segment_worst < worst_drawdown:
                worst_drawdown = segment_worst
        output[state_code, 0] = observation_count
        output[state_code, 1] = return_count
        if return_count > 0:
            output[state_code, 2] = wealth - 1.0
            output[state_code, 4] = worst_drawdown
            output[state_code, 5] = positive_count / return_count
        if return_count > 1:
            mean = return_sum / return_count
            variance = (
                return_square_sum - return_count * mean * mean
            ) / (return_count - 1)
            output[state_code, 3] = math.sqrt(max(0.0, variance)) * math.sqrt(252.0)
    return output


_PRODUCTION_KERNELS = (
    moving_average_kernel,
    technical_input_availability_kernel,
    bollinger_kernel,
    kdj_kernel,
    daily_returns_percent_kernel,
    _linear_quantile,
    return_statistics_kernel,
    distribution_interpretation_codes_kernel,
    histogram_normal_pdf_kernel,
    box_plot_kernel,
    _inverse_standard_normal,
    normal_qq_kernel,
    _adjusted_distribution_shape,
    _calibrate_sinh_arcsinh,
    _next_uniform,
    _summarize_simulation,
    parametric_monte_carlo_kernel,
    stationary_block_bootstrap_kernel,
    simulation_comparison_kernel,
    terminal_density_kernel,
    regime_performance_kernel,
)


def product_analysis_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _PRODUCTION_KERNELS
    }
    material = "|".join(
        [PRODUCT_ANALYSIS_ENGINE_VERSION, PRODUCT_ANALYSIS_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return {
        "engine": PRODUCT_ANALYSIS_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": PRODUCT_ANALYSIS_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_coverage": f"{sum(bool(value) for value in signatures.values())}/{len(signatures)}",
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(bool(kernel.nopython_signatures) for kernel in _PRODUCTION_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
    }


def warm_product_analysis_numba_kernels() -> dict[str, object]:
    try:
        from backend.compute_policy import validate_execution_audit
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from compute_policy import validate_execution_audit

    values = np.ascontiguousarray(np.linspace(1.0, 1.2, 32, dtype=np.float64))
    periods = np.ascontiguousarray(np.array([5, 10], dtype=np.int64))
    dates = np.ascontiguousarray(np.arange(values.size, dtype=np.int64))
    moving_average_kernel(values, periods)
    technical_input_availability_kernel(values, values, values, values, values)
    bollinger_kernel(values, 20, 2.0)
    kdj_kernel(values + 0.01, values - 0.01, values, 9, 3, 3)
    returns = daily_returns_percent_kernel(values)
    sorted_returns = np.ascontiguousarray(np.sort(returns.copy()))
    _linear_quantile(sorted_returns, 0.5)
    return_statistics_kernel(returns)
    distribution_interpretation_codes_kernel(return_statistics_kernel(returns))
    histogram_normal_pdf_kernel(returns, 0.2)
    box_plot_kernel(returns)
    _inverse_standard_normal(0.5)
    normal_qq_kernel(returns)
    log_returns = np.ascontiguousarray(np.log1p(returns / 100.0))
    _adjusted_distribution_shape(log_returns)
    _calibrate_sinh_arcsinh(0.0, 0.0)
    _next_uniform(np.uint64(1))
    paths = np.ascontiguousarray(np.ones((2, 3), dtype=np.float64))
    drawdowns = np.ascontiguousarray(np.zeros(2, dtype=np.float64))
    _summarize_simulation(paths, drawdowns, 1.0, 0.05)
    parametric = parametric_monte_carlo_kernel(returns, 1.0, 2, 3, 1, 5.0)
    bootstrap = stationary_block_bootstrap_kernel(returns, 1.0, 2, 3, 2, 5.0, 5)
    simulation_comparison_kernel(parametric[3], bootstrap[3], 1.0)
    terminal_density_kernel(parametric[2], parametric[1], 21, 1.0)
    regime_performance_kernel(
        dates,
        values,
        np.ascontiguousarray(np.array([0], dtype=np.int64)),
        np.ascontiguousarray(np.array([31], dtype=np.int64)),
        np.ascontiguousarray(np.array([0], dtype=np.int64)),
        1,
    )
    audit = validate_execution_audit(product_analysis_execution_audit())
    if audit["nopython"] is not True or audit["python_fallback"] != 0:
        raise RuntimeError("产品详情分析 NJIT 内核预热失败")
    return audit


__all__ = [
    "bollinger_kernel",
    "box_plot_kernel",
    "daily_returns_percent_kernel",
    "distribution_interpretation_codes_kernel",
    "histogram_normal_pdf_kernel",
    "kdj_kernel",
    "moving_average_kernel",
    "normal_qq_kernel",
    "parametric_monte_carlo_kernel",
    "product_analysis_execution_audit",
    "regime_performance_kernel",
    "return_statistics_kernel",
    "simulation_comparison_kernel",
    "stationary_block_bootstrap_kernel",
    "terminal_density_kernel",
    "technical_input_availability_kernel",
    "warm_product_analysis_numba_kernels",
]
