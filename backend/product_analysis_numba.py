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
PRODUCT_ANALYSIS_KERNEL_VERSION = "product-scenario-statistics-simulation-7"

# Every simulation lane packs the same assumption block behind the day vector so
# one service-side reader can decode any of them. Slots a lane cannot fill are
# NaN rather than absent, which keeps the layout a constant.
SIMULATION_ASSUMPTION_SLOTS = 20

# The chart's right-hand panel answers "where does the NAV land", and the day it
# answers for follows the zoom window's right edge — so every day needs its own
# cross-section, not just the last one. Shipping all 252 would triple the
# response for a difference no eye can resolve, so days are strided down to
# roughly this many checkpoints. The last day is always one of them: that frame
# is the headline 期末 distribution every metric card already quotes.
DENSITY_FRAME_TARGET = 48
DENSITY_CURVE_POINTS = 41
DENSITY_BIN_COUNT = 24
# day, navLow, navHigh, binWidth, countAxisMax — then the curve, then the bins.
# Counts only: the nav each entry sits at is a uniform grid the reader rebuilds
# from navLow/navHigh, which is what keeps 240 frames a small payload.
DENSITY_FRAME_SLOTS = 5
DENSITY_FRAME_WIDTH = DENSITY_FRAME_SLOTS + DENSITY_CURVE_POINTS + DENSITY_BIN_COUNT

_F1 = float64[::1]
_F2 = float64[:, ::1]
_I1 = int64[::1]
_U1 = uint8[::1]

_BOX_RESULT = types.Tuple((_F1, _F1))
_QQ_RESULT = types.Tuple((_F2, _F2))
_SIM_RESULT = types.Tuple((_F2, _F2, _F1, _F1, _F1, _F2))
_RANDOM_RESULT = types.Tuple((uint64, float64))
_REGIME_RESULT = types.Tuple((_F2, _F2, _F1, _I1, _I1))
_REALIZED_RESULT = types.Tuple((_F1, _F1))
_FILTER_RESULT = types.Tuple((_F1, _F1))


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


@njit(_F1(_F1, float64), cache=False, nogil=True)
def _density_frame(sorted_values: np.ndarray, day: float) -> np.ndarray:
    """One day's landing distribution: a smooth curve and a histogram.

    Takes the cross-section already sorted, because the caller sorts it anyway
    to read that day's quantiles — the whole per-day panel therefore costs no
    extra sort, only the kernel sum.
    """
    frame = np.zeros(DENSITY_FRAME_WIDTH, dtype=np.float64)
    frame[0] = day
    size = sorted_values.size
    if size < 2:
        return frame
    total = 0.0
    for value in sorted_values:
        total += value
    mean = total / size
    variance_sum = 0.0
    for value in sorted_values:
        difference = value - mean
        variance_sum += difference * difference
    standard_deviation = math.sqrt(max(0.0, variance_sum / max(1, size - 1)))
    interquartile_range = _linear_quantile(sorted_values, 0.75) - _linear_quantile(sorted_values, 0.25)
    robust_scale = standard_deviation
    alternate_scale = interquartile_range / 1.34
    if robust_scale <= 0.0 or (alternate_scale > 0.0 and alternate_scale < robust_scale):
        robust_scale = alternate_scale
    # Anchored on the median, not the mean. `robust_scale` above already prefers
    # the IQR when an outlier dominates, but this floor did not: one path that
    # compounded to 1e12 — which the filtered lanes can produce, since EWMA's
    # variance has no mean reversion — dragged the mean to 1e9 and set the
    # floor, and with it the chart's whole nav axis, to six figures. The
    # quantiles were fine; the axis made them invisible.
    median_nav = _linear_quantile(sorted_values, 0.5)
    minimum_bandwidth = max(abs(median_nav) * 0.0005, 1e-6)
    bandwidth = max(minimum_bandwidth, 0.9 * robust_scale * size ** (-0.2))
    nav_low = max(0.0, _linear_quantile(sorted_values, 0.01) - bandwidth * 2.0)
    nav_high = max(
        nav_low + minimum_bandwidth,
        _linear_quantile(sorted_values, 0.99) + bandwidth * 2.0,
    )
    bin_width = (nav_high - nav_low) / DENSITY_BIN_COUNT
    frame[1] = nav_low
    frame[2] = nav_high
    frame[3] = bin_width
    # Counts, not densities: the panel's axis is "how many of my paths land
    # here", which is the only version of this number a reader can sanity-check
    # against the path budget they chose.
    count_factor = size * bin_width
    normalizer = size * bandwidth * math.sqrt(2.0 * math.pi)
    highest = 1.0
    for index in range(DENSITY_CURVE_POINTS):
        nav = nav_low + (nav_high - nav_low) * index / (DENSITY_CURVE_POINTS - 1)
        kernel_sum = 0.0
        for value in sorted_values:
            standardized = (nav - value) / bandwidth
            kernel_sum += math.exp(-0.5 * standardized * standardized)
        count = kernel_sum / normalizer * count_factor
        frame[DENSITY_FRAME_SLOTS + index] = count
        if count > highest:
            highest = count
    bin_base = DENSITY_FRAME_SLOTS + DENSITY_CURVE_POINTS
    for value in sorted_values:
        raw_index = int(math.floor((value - nav_low) / bin_width))
        index = max(0, min(DENSITY_BIN_COUNT - 1, raw_index))
        frame[bin_base + index] += 1.0
    for index in range(DENSITY_BIN_COUNT):
        if frame[bin_base + index] > highest:
            highest = frame[bin_base + index]
    frame[4] = math.ceil(highest * 1.08)
    return frame


@njit(_SIM_RESULT(_F2, _F1, float64, float64), cache=False, nogil=True)
def _summarize_simulation(
    paths: np.ndarray,
    max_drawdowns: np.ndarray,
    initial_nav: float,
    target_return: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_count, day_count = paths.shape
    sample_count = min(12, path_count)
    sample_paths = np.ascontiguousarray(paths[:sample_count].copy())
    percentiles = np.empty((5, day_count), dtype=np.float64)
    probabilities = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    # Counted back from the last day so the terminal frame is always exact; day
    # 0 is excluded because every path starts on the same value, which is a
    # spike carrying no information.
    last_day = day_count - 1
    stride = max(1, (last_day + DENSITY_FRAME_TARGET - 1) // DENSITY_FRAME_TARGET)
    frame_row_of_day = np.full(day_count, -1, dtype=np.int64)
    frame_count = 0
    day = last_day
    while day >= 1:
        frame_count += 1
        day -= stride
    row = frame_count - 1
    day = last_day
    while day >= 1:
        frame_row_of_day[day] = row
        row -= 1
        day -= stride
    frames = np.zeros((max(1, frame_count), DENSITY_FRAME_WIDTH), dtype=np.float64)
    scratch = np.empty(path_count, dtype=np.float64)
    for day in range(day_count):
        for path in range(path_count):
            scratch[path] = paths[path, day]
        scratch.sort()
        for index in range(probabilities.size):
            percentiles[index, day] = _linear_quantile(scratch, probabilities[index])
        row = frame_row_of_day[day]
        if row >= 0:
            frames[row] = _density_frame(scratch, day)
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
    # One NAV axis for both panels, fixed for the whole run: the reader is
    # comparing today's distribution against the fan it came from, and an axis
    # that rescaled as the zoom window moved would make every frame look alike.
    observed_minimum = percentiles[0, day_count - 1]
    observed_maximum = percentiles[4, day_count - 1]
    if frame_count > 0:
        observed_minimum = frames[frame_count - 1, 1]
        observed_maximum = frames[frame_count - 1, 2]
    for band in (0, 4):
        for value in percentiles[band]:
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
            max(0.0, observed_minimum - padding),
            observed_maximum + padding,
        ],
        dtype=np.float64,
    )
    days = np.arange(day_count, dtype=np.float64)
    return sample_paths, percentiles, terminal, summary, days, frames


@njit(
    _SIM_RESULT(_F1, float64, int64, int64, int64, float64, int64),
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
    shape_mode: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw iid daily log returns from a fitted marginal distribution.

    ``shape_mode`` picks the marginal: ``0`` keeps the textbook standard normal,
    which is the baseline every richer lane has to beat, and ``1`` calibrates a
    sinh-arcsinh transform to the sample's skewness and excess kurtosis. Both
    modes share one path loop, so the only difference between the two lanes the
    platform exposes is the innovation — nothing else can drift between them.
    """

    valid_count = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            valid_count += 1
    if valid_count < 20 or initial_nav <= 0.0 or horizon_days < 1 or path_count < 1:
        raise ValueError("parametric simulation requires valid parameters and 20 returns")
    if shape_mode != 0 and shape_mode != 1:
        raise ValueError("parametric shape mode must be 0 (normal) or 1 (sinh-arcsinh)")
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
    if shape_mode == 1:
        calibration = _calibrate_sinh_arcsinh(shape[2], shape[3])
    else:
        # No calibration to report: this lane's whole point is an untouched
        # standard normal, so every shape slot stays empty and the inactive flag
        # is what makes the path loop below skip the transform.
        calibration = np.full(8, np.nan, dtype=np.float64)
        calibration[0] = 0.0
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
    sample, percentiles, terminal, summary, days, frames = _summarize_simulation(
        paths,
        max_drawdowns,
        initial_nav,
        target_return_percent / 100.0,
    )
    assumptions = np.full(SIMULATION_ASSUMPTION_SLOTS, np.nan, dtype=np.float64)
    assumptions[0] = valid_count
    assumptions[1] = target_return_percent
    assumptions[2] = mean
    assumptions[3] = volatility
    assumptions[4] = shape[2]
    assumptions[5] = shape[3]
    if calibration[0] == 1.0:
        assumptions[6] = calibration[6]
        assumptions[7] = calibration[7]
        assumptions[9] = calibration[2]
        assumptions[10] = calibration[3]
    elif shape_mode == 1:
        assumptions[6] = 0.0
        assumptions[7] = 0.0
    assumptions[8] = calibration[1]
    # The fifth return lane is reserved for days by the common result contract;
    # attach assumptions after the day vector so the service can reconstruct it.
    packed = np.empty(days.size + assumptions.size, dtype=np.float64)
    packed[: days.size] = days
    packed[days.size :] = assumptions
    return sample, percentiles, terminal, summary, packed, frames


@njit(
    _SIM_RESULT(_F1, _I1, float64, int64, int64, int64, float64, int64),
    cache=False,
    nogil=True,
)
def stationary_block_bootstrap_kernel(
    returns_percent: np.ndarray,
    segment_ids: np.ndarray,
    initial_nav: float,
    horizon_days: int,
    path_count: int,
    seed: int,
    target_return_percent: float,
    average_block_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if returns_percent.size != segment_ids.size:
        raise ValueError("bootstrap return and segment axes must match")
    valid_count = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            valid_count += 1
    if valid_count < 20 or initial_nav <= 0.0 or horizon_days < 1 or path_count < 1:
        raise ValueError("bootstrap simulation requires valid parameters and 20 returns")
    valid_positions = np.empty(valid_count, dtype=np.int64)
    position = 0
    for index in range(returns_percent.size):
        if np.isfinite(returns_percent[index]) and returns_percent[index] > -100.0:
            valid_positions[position] = index
            position += 1
    block_length = max(1, min(valid_count, average_block_length))
    restart_probability = 1.0 / block_length
    restart_cdf = np.empty(valid_count, dtype=np.float64)
    restart_weight = 0.0
    for position in range(valid_count):
        index = valid_positions[position]
        has_predecessor = (
            position > 0
            and valid_positions[position - 1] == index - 1
            and segment_ids[index - 1] == segment_ids[index]
        )
        # At a boundary the next draw must restart. Uniform restart positions
        # would then over-sample segment interiors. These incoming-mass deficits
        # preserve a uniform marginal over all valid historical observations.
        restart_weight += restart_probability if has_predecessor else 1.0
        restart_cdf[position] = restart_weight
    paths = np.empty((path_count, horizon_days + 1), dtype=np.float64)
    max_drawdowns = np.empty(path_count, dtype=np.float64)
    state = np.uint64(seed if seed != 0 else 1)
    for path_index in range(path_count):
        state, first = _next_uniform(state)
        source_index = valid_positions[min(valid_count - 1, int(math.floor(first * valid_count)))]
        nav = initial_nav
        peak = initial_nav
        worst = 0.0
        paths[path_index, 0] = nav
        for day in range(1, horizon_days + 1):
            if day > 1:
                state, decision = _next_uniform(state)
                next_index = source_index + 1
                continues = (
                    next_index < returns_percent.size
                    and segment_ids[next_index] == segment_ids[source_index]
                    and np.isfinite(returns_percent[next_index])
                    and returns_percent[next_index] > -100.0
                )
                if decision >= restart_probability and continues:
                    source_index = next_index
                else:
                    state, choice = _next_uniform(state)
                    target_weight = choice * restart_weight
                    lower = 0
                    upper = valid_count - 1
                    while lower < upper:
                        middle = (lower + upper) // 2
                        if target_weight < restart_cdf[middle]:
                            upper = middle
                        else:
                            lower = middle + 1
                    source_index = valid_positions[lower]
            nav *= 1.0 + returns_percent[source_index] / 100.0
            if not np.isfinite(nav) or nav <= 0.0:
                raise ValueError("non-finite bootstrap simulation path")
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak
            if drawdown > worst:
                worst = drawdown
            paths[path_index, day] = nav
        max_drawdowns[path_index] = worst
    sample, percentiles, terminal, summary, days, frames = _summarize_simulation(
        paths,
        max_drawdowns,
        initial_nav,
        target_return_percent / 100.0,
    )
    assumptions = np.full(SIMULATION_ASSUMPTION_SLOTS, np.nan, dtype=np.float64)
    assumptions[0] = valid_count
    assumptions[1] = target_return_percent
    assumptions[11] = block_length
    packed = np.empty(days.size + assumptions.size, dtype=np.float64)
    packed[: days.size] = days
    packed[days.size :] = assumptions
    return sample, percentiles, terminal, summary, packed, frames


@njit(float64(_F1, _I1, float64, float64, float64, float64), cache=False, nogil=True)
def _garch_quasi_likelihood(
    demeaned: np.ndarray,
    segment_ids: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
    start_variance: float,
) -> float:
    """Gaussian quasi-log-likelihood of a GARCH(1,1) recursion.

    Quasi because the innovations are almost certainly not normal — that is the
    whole reason the simulation bootstraps residuals instead of drawing them.
    The estimator stays consistent under misspecified innovations, which is what
    makes this the standard first fit rather than a shortcut.
    """

    variance = start_variance
    total = 0.0
    for index in range(demeaned.size):
        # Two情景区间 years apart are two histories, not one: carrying variance
        # across the joint would fit persistence to a day that never followed.
        if index > 0 and segment_ids[index] != segment_ids[index - 1]:
            variance = start_variance
        value = demeaned[index]
        if not np.isfinite(variance) or variance <= 0.0:
            return -np.inf
        total -= 0.5 * (math.log(variance) + value * value / variance)
        variance = omega + alpha * value * value + beta * variance
    if not np.isfinite(total):
        return -np.inf
    return total


@njit(_FILTER_RESULT(_F1, _I1, int64, float64), cache=False, nogil=True)
def _volatility_filter(
    log_returns: np.ndarray,
    segment_ids: np.ndarray,
    filter_mode: int,
    ewma_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a return series into a conditional volatility path and residuals.

    EWMA *is* GARCH(1,1) with ``omega = 0``, ``alpha = 1 - lambda`` and
    ``beta = lambda``, so both modes leave the same three coefficients behind and
    the forward recursion in the simulation needs a single code path. The only
    difference is where the coefficients come from: a decay constant the user
    picks, or a variance-targeted fit to this product's own history.
    """

    count = log_returns.size
    if count < 20:
        raise ValueError("volatility filtering requires 20 valid returns")
    if segment_ids.size != count:
        raise ValueError("volatility filter return and segment axes must match")
    total = 0.0
    for value in log_returns:
        if not np.isfinite(value):
            raise ValueError("volatility filtering requires finite log returns")
        total += value
    mean = total / count
    demeaned = np.empty(count, dtype=np.float64)
    variance_sum = 0.0
    for index in range(count):
        deviation = log_returns[index] - mean
        demeaned[index] = deviation
        variance_sum += deviation * deviation
    unconditional = variance_sum / (count - 1)
    if not np.isfinite(unconditional) or unconditional <= 0.0:
        raise ValueError("volatility filtering requires a positive sample variance")

    quasi_likelihood = np.nan
    if filter_mode == 0:
        if not (0.5 <= ewma_lambda < 1.0):
            raise ValueError("EWMA decay must sit inside [0.5, 1)")
        omega = 0.0
        alpha = 1.0 - ewma_lambda
        beta = ewma_lambda
    elif filter_mode == 1:
        # Variance targeting pins omega to the sample variance, which turns a
        # three-parameter fit into a 2-D search the house grid-then-refine
        # pattern carries without scipy — and guarantees the model's long-run
        # volatility equals the sample's instead of drifting off it.
        # ponytail: grid + coordinate refinement. Swap in a real optimiser only
        # if a fit is ever shown to sit off the likelihood surface's peak.
        best_alpha = 0.05
        best_beta = 0.90
        best_score = -np.inf
        coarse_alpha = np.array([0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.25])
        coarse_beta = np.array([0.55, 0.66, 0.74, 0.81, 0.86, 0.90, 0.93, 0.96])
        for alpha_candidate in coarse_alpha:
            for beta_candidate in coarse_beta:
                if alpha_candidate + beta_candidate >= 0.9995:
                    continue
                score = _garch_quasi_likelihood(
                    demeaned,
                    segment_ids,
                    unconditional * (1.0 - alpha_candidate - beta_candidate),
                    alpha_candidate,
                    beta_candidate,
                    unconditional,
                )
                if score > best_score:
                    best_score = score
                    best_alpha = alpha_candidate
                    best_beta = beta_candidate
        alpha_step = 0.02
        beta_step = 0.02
        for _ in range(16):
            anchor_alpha = best_alpha
            anchor_beta = best_beta
            for alpha_direction in range(-1, 2):
                for beta_direction in range(-1, 2):
                    if alpha_direction == 0 and beta_direction == 0:
                        continue
                    alpha_candidate = anchor_alpha + alpha_direction * alpha_step
                    beta_candidate = anchor_beta + beta_direction * beta_step
                    if alpha_candidate < 0.0005 or alpha_candidate > 0.6:
                        continue
                    if beta_candidate < 0.0005 or beta_candidate > 0.999:
                        continue
                    if alpha_candidate + beta_candidate >= 0.9995:
                        continue
                    score = _garch_quasi_likelihood(
                        demeaned,
                        segment_ids,
                        unconditional * (1.0 - alpha_candidate - beta_candidate),
                        alpha_candidate,
                        beta_candidate,
                        unconditional,
                    )
                    if score > best_score:
                        best_score = score
                        best_alpha = alpha_candidate
                        best_beta = beta_candidate
            alpha_step *= 0.7
            beta_step *= 0.7
        alpha = best_alpha
        beta = best_beta
        omega = unconditional * (1.0 - alpha - beta)
        quasi_likelihood = best_score
    else:
        raise ValueError("volatility filter mode must be 0 (EWMA) or 1 (GARCH)")

    # A long run of identical closes drives an omega-free EWMA recursion toward
    # zero variance, which would then divide the next residual by nothing. Money
    # market funds do exactly that. ponytail: hard floor at a millionth of the
    # sample variance; a per-product floor only matters if one is ever hit for
    # long enough to distort the filtered path.
    variance_floor = unconditional * 1e-6
    residuals = np.empty(count, dtype=np.float64)
    variance = unconditional
    for index in range(count):
        # Same restart the recursion already does at index 0, applied at every
        # 情景区间 boundary: each segment is standardised by its own history only.
        if index > 0 and segment_ids[index] != segment_ids[index - 1]:
            variance = unconditional
        if variance < variance_floor:
            variance = variance_floor
        deviation = demeaned[index]
        residuals[index] = deviation / math.sqrt(variance)
        variance = omega + alpha * deviation * deviation + beta * variance
        if not np.isfinite(variance):
            raise ValueError("volatility filter produced a non-finite variance")
    if variance < variance_floor:
        variance = variance_floor
    # `variance` now holds the conditional variance of the first simulated day —
    # the one thing an unconditional model cannot know and the reason this lane
    # answers "what next" rather than "what on average".
    variance_next = variance

    # Rescale to unit variance. Variance targeting already matched the long-run
    # level, so residuals that come out slightly off unit would re-introduce the
    # bias the targeting removed.
    residual_sum = 0.0
    for value in residuals:
        residual_sum += value * value
    residual_scale = math.sqrt(residual_sum / count)
    if not np.isfinite(residual_scale) or residual_scale <= 0.0:
        raise ValueError("volatility filter produced degenerate residuals")
    for index in range(count):
        residuals[index] = residuals[index] / residual_scale

    parameters = np.array(
        [
            mean,
            omega,
            alpha,
            beta,
            variance_next,
            unconditional,
            alpha + beta,
            quasi_likelihood,
        ],
        dtype=np.float64,
    )
    return residuals, parameters


@njit(
    _SIM_RESULT(_F1, _I1, float64, int64, int64, int64, float64, int64, float64),
    cache=False,
    nogil=True,
)
def filtered_historical_simulation_kernel(
    returns_percent: np.ndarray,
    segment_ids: np.ndarray,
    initial_nav: float,
    horizon_days: int,
    path_count: int,
    seed: int,
    target_return_percent: float,
    filter_mode: int,
    ewma_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap standardised residuals, re-inflated from today's volatility.

    This is the lane that fixes the shared blind spot of the other three: they
    are all unconditional, so a研究日 sitting in a volatility spike gets the same
    fan as one sitting in a calm stretch. Here the fan starts from the
    conditional variance the filter left behind and clusters forward, and a
    large residual drawn from a calm stretch lands on top of a turbulent
    variance — which is how a path worse than anything in the history gets made
    at all.
    """

    if returns_percent.size != segment_ids.size:
        raise ValueError("filtered simulation return and segment axes must match")
    valid_count = 0
    for value in returns_percent:
        if np.isfinite(value) and value > -100.0:
            valid_count += 1
    if valid_count < 20 or initial_nav <= 0.0 or horizon_days < 1 or path_count < 1:
        raise ValueError("filtered simulation requires valid parameters and 20 returns")
    log_returns = np.empty(valid_count, dtype=np.float64)
    valid_segments = np.empty(valid_count, dtype=np.int64)
    position = 0
    for index in range(returns_percent.size):
        value = returns_percent[index]
        if np.isfinite(value) and value > -100.0:
            log_returns[position] = math.log1p(value / 100.0)
            valid_segments[position] = segment_ids[index]
            position += 1
    residuals, parameters = _volatility_filter(
        log_returns, valid_segments, filter_mode, ewma_lambda
    )
    mean = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]
    variance_start = parameters[4]
    unconditional = parameters[5]
    variance_floor = unconditional * 1e-6
    shape = _adjusted_distribution_shape(log_returns)
    residual_shape = _adjusted_distribution_shape(residuals)

    paths = np.empty((path_count, horizon_days + 1), dtype=np.float64)
    max_drawdowns = np.empty(path_count, dtype=np.float64)
    state = np.uint64(seed if seed != 0 else 1)
    for path_index in range(path_count):
        nav = initial_nav
        peak = initial_nav
        worst = 0.0
        paths[path_index, 0] = nav
        variance = variance_start
        for day in range(1, horizon_days + 1):
            if variance < variance_floor:
                variance = variance_floor
            state, draw = _next_uniform(state)
            index = int(math.floor(draw * valid_count))
            if index >= valid_count:
                index = valid_count - 1
            deviation = math.sqrt(variance) * residuals[index]
            nav *= math.exp(mean + deviation)
            if not np.isfinite(nav) or nav <= 0.0:
                raise ValueError("non-finite filtered simulation path")
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak
            if drawdown > worst:
                worst = drawdown
            paths[path_index, day] = nav
            variance = omega + alpha * deviation * deviation + beta * variance
            if not np.isfinite(variance):
                raise ValueError("non-finite filtered simulation variance")
        max_drawdowns[path_index] = worst
    sample, percentiles, terminal, summary, days, frames = _summarize_simulation(
        paths,
        max_drawdowns,
        initial_nav,
        target_return_percent / 100.0,
    )
    assumptions = np.full(SIMULATION_ASSUMPTION_SLOTS, np.nan, dtype=np.float64)
    assumptions[0] = valid_count
    assumptions[1] = target_return_percent
    assumptions[2] = mean
    assumptions[3] = math.sqrt(unconditional)
    assumptions[4] = shape[2]
    assumptions[5] = shape[3]
    assumptions[12] = math.sqrt(variance_start)
    assumptions[13] = parameters[6]
    assumptions[14] = omega
    assumptions[15] = alpha
    assumptions[16] = beta
    if filter_mode == 0:
        assumptions[17] = ewma_lambda
    assumptions[18] = residual_shape[2]
    assumptions[19] = residual_shape[3]
    packed = np.empty(days.size + assumptions.size, dtype=np.float64)
    packed[: days.size] = days
    packed[days.size :] = assumptions
    return sample, percentiles, terminal, summary, packed, frames


@njit(_REALIZED_RESULT(_F1, float64, _F2, _F1, float64), cache=False, nogil=True)
def realized_path_comparison_kernel(
    future_closes: np.ndarray,
    base_close: float,
    percentiles: np.ndarray,
    terminal_values: np.ndarray,
    initial_nav: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Score what actually happened after the research day against one model.

    The realised prices arrive here only to be *measured*; nothing computed in
    this kernel can reach the simulation inputs, which is what keeps the panel
    free of look-ahead. Coverage is treated as a spectrum: a horizon the data
    only partly covers still scores the days it has, and only the exact
    percentile rank — which compares one terminal value against a distribution
    of the same length — waits for a complete horizon.
    """

    day_count = percentiles.shape[1]
    horizon = day_count - 1
    if horizon < 1 or initial_nav <= 0.0:
        raise ValueError("realized comparison requires a positive initial nav and horizon")
    if not np.isfinite(base_close) or base_close <= 0.0:
        raise ValueError("realized comparison requires a positive base close")
    usable = future_closes.size
    if usable > horizon:
        usable = horizon
    path = np.empty(usable + 1, dtype=np.float64)
    path[0] = initial_nav
    covered = 0
    finite_count = 0
    inside_count = 0
    above_median_count = 0
    breach_count = 0
    # Signed distance outside the 5%—95% envelope, at its widest. A "first
    # breach day" was tried here and read as noise: on day one the envelope is
    # a single day of volatility wide, so almost any real move breaches it
    # while the path still sits inside the band for the other 250 days.
    worst_gap = 0.0
    worst_gap_day = -1.0
    peak = initial_nav
    worst_drawdown = 0.0
    for day in range(1, usable + 1):
        close = future_closes[day - 1]
        if not np.isfinite(close) or close <= 0.0:
            path[day] = np.nan
            continue
        nav = initial_nav * close / base_close
        path[day] = nav
        covered = day
        finite_count += 1
        if nav > peak:
            peak = nav
        drawdown = (peak - nav) / peak
        if drawdown > worst_drawdown:
            worst_drawdown = drawdown
        gap = 0.0
        if nav < percentiles[0, day]:
            gap = nav - percentiles[0, day]
        elif nav > percentiles[4, day]:
            gap = nav - percentiles[4, day]
        if gap == 0.0:
            inside_count += 1
        else:
            breach_count += 1
            if abs(gap) > abs(worst_gap):
                worst_gap = gap
                worst_gap_day = float(day)
        if nav > percentiles[2, day]:
            above_median_count += 1
    terminal_nav = np.nan
    terminal_return = np.nan
    band = -1.0
    if covered > 0:
        terminal_nav = path[covered]
        terminal_return = terminal_nav / initial_nav - 1.0
        band = 5.0
        for index in range(5):
            if terminal_nav <= percentiles[index, covered]:
                band = float(index)
                break
    percentile_rank = np.nan
    if covered == horizon and covered > 0 and terminal_values.size > 0:
        below = 0
        for value in terminal_values:
            if value <= terminal_nav:
                below += 1
        percentile_rank = below / terminal_values.size
    containment = np.nan
    above_median_ratio = np.nan
    if finite_count > 0:
        containment = inside_count / finite_count
        above_median_ratio = above_median_count / finite_count
    summary = np.array(
        [
            float(horizon),
            float(covered),
            1.0 if covered == horizon and covered > 0 else 0.0,
            terminal_nav,
            terminal_return,
            percentile_rank,
            band,
            containment,
            float(breach_count),
            worst_gap if worst_gap_day >= 0.0 else np.nan,
            worst_gap_day,
            worst_drawdown if finite_count > 0 else np.nan,
            above_median_ratio,
            float(finite_count),
        ],
        dtype=np.float64,
    )
    return path, summary


@njit(_F1(_F2, float64), cache=False, nogil=True)
def simulation_comparison_kernel(
    summaries: np.ndarray,
    initial_nav: float,
) -> np.ndarray:
    """Spread of the headline figures across every model that ran.

    A spread over N lanes rather than a difference between two: with five models
    the question stopped being "do these two agree" and became "how much of this
    number is the model's choice rather than the product's history".
    """

    lane_count = summaries.shape[0]
    if lane_count < 2 or summaries.shape[1] < 12:
        raise ValueError("model comparison needs at least two 12-slot summaries")
    if not np.isfinite(initial_nav) or initial_nav <= 0.0:
        raise ValueError("model comparison requires a positive initial nav")
    output = np.empty(5, dtype=np.float64)
    # p05 terminal nav, median terminal nav, loss probability, 95% CVaR. The
    # first two are navs and read as returns; the last two already are ratios.
    columns = np.array([0, 2, 5, 7], dtype=np.int64)
    for position in range(columns.size):
        column = columns[position]
        lowest = np.inf
        highest = -np.inf
        for lane in range(lane_count):
            value = summaries[lane, column]
            if column == 0 or column == 2:
                value = value / initial_nav - 1.0
            if not np.isfinite(value):
                raise ValueError("model comparison requires finite summaries")
            if value < lowest:
                lowest = value
            if value > highest:
                highest = value
        output[position] = highest - lowest
    largest = max(output[0], output[1], output[2], output[3])
    output[4] = 2.0 if largest >= 0.1 else (1.0 if largest >= 0.03 else 0.0)
    return output


@njit(
    _REGIME_RESULT(_I1, _F1, _I1, _I1, _I1, int64, int64, int64),
    cache=False,
    nogil=True,
)
def regime_analysis_kernel(
    date_days: np.ndarray,
    close: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    state_codes: np.ndarray,
    state_count: int,
    selected_state: int,
    selected_segment: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Keep date positions and segment boundaries; never compound disjoint paths.

    State lanes: observations, returns, segments, median length, eligible paths,
    mean return, annual volatility, up-day ratio, median/worst segment return,
    median/worst segment drawdown. All financial values are decimal returns.
    Segment lanes: observations, returns, return, drawdown, status, first/last
    row, valid prices. Status 0=complete, 1=short, 2=missing price.
    """
    if date_days.size != close.size or not (starts.size == ends.size == state_codes.size):
        raise ValueError("regime input axes must match")
    if state_count < 0 or selected_state >= state_count or selected_state < -1 or selected_segment < -1 or selected_segment >= starts.size:
        raise ValueError("invalid selected regime state or segment")
    for segment in range(starts.size):
        if state_codes[segment] < 0 or state_codes[segment] >= state_count or starts[segment] > ends[segment]:
            raise ValueError("invalid regime segment definition")
    base_returns = daily_returns_percent_kernel(close)
    by_row = np.full(close.size, -1, dtype=np.int64)
    segment_values = np.full((starts.size, 8), np.nan, dtype=np.float64)
    for segment in range(starts.size):
        observations = 0
        valid = 0
        return_count = 0
        invalid_return = False
        first = -1
        last = -1
        peak = 0.0
        worst = 0.0
        for row in range(close.size):
            if not (starts[segment] <= date_days[row] <= ends[segment]):
                continue
            if by_row[row] >= 0:
                raise ValueError("historical regime segments overlap")
            by_row[row] = segment
            observations += 1
            if first < 0:
                first = row
            last = row
            value = close[row]
            if not np.isfinite(value) or value <= 0.0:
                continue
            valid += 1
            peak = max(peak, value)
            worst = min(worst, value / peak - 1.0)
            if row > first and np.isfinite(close[row - 1]) and close[row - 1] > 0.0:
                change = base_returns[row - 1]
                if np.isfinite(change) and change > -100.0:
                    return_count += 1
                else:
                    invalid_return = True
        segment_values[segment, 0] = observations
        segment_values[segment, 1] = return_count
        segment_values[segment, 5] = first
        segment_values[segment, 6] = last
        segment_values[segment, 7] = valid
        status = 1
        if observations != valid or invalid_return:
            status = 2
        elif observations >= 2:
            change = close[last] / close[first] - 1.0
            if np.isfinite(change) and change > -1.0:
                status = 0
                segment_values[segment, 2] = change
                segment_values[segment, 3] = worst
            else:
                status = 2
        segment_values[segment, 4] = status

    selected_returns = np.full(max(0, close.size - 1), np.nan, dtype=np.float64)
    return_segments = np.full(selected_returns.size, -1, dtype=np.int64)
    selected_observations = 0
    selected_count = 0
    selected_segments = 0
    selected_first = -1
    selected_last = -1
    for row in range(close.size):
        segment = by_row[row]
        include = selected_state < 0 and selected_segment < 0
        if segment >= 0:
            include = include or (
                (selected_state < 0 or state_codes[segment] == selected_state)
                and (selected_segment < 0 or segment == selected_segment)
            )
        if not include:
            continue
        selected_observations += 1
        if selected_first < 0:
            selected_first = row
        selected_last = row
        if row < 1 or not (np.isfinite(close[row]) and close[row] > 0.0
                           and np.isfinite(close[row - 1]) and close[row - 1] > 0.0):
            continue
        if (selected_state >= 0 or selected_segment >= 0) and by_row[row - 1] != segment:
            continue
        change = base_returns[row - 1]
        if not np.isfinite(change) or change <= -100.0:
            continue
        selected_returns[row - 1] = change
        # Full-window sampling can cross states, but never a missing observation.
        return_segments[row - 1] = segment if selected_state >= 0 or selected_segment >= 0 else 0
        selected_count += 1
    for segment in range(starts.size):
        if segment_values[segment, 0] > 0 and (selected_state < 0 or state_codes[segment] == selected_state) and (selected_segment < 0 or segment == selected_segment):
            selected_segments += 1
    if state_count == 0 and selected_observations > 0:
        selected_segments = 1

    states = np.full((state_count, 12), np.nan, dtype=np.float64)
    for state in range(state_count):
        observations = 0
        segment_count = 0
        eligible_count = 0
        returns = np.full(max(0, close.size - 1), np.nan, dtype=np.float64)
        lengths = np.empty(starts.size, dtype=np.float64)
        interval_returns = np.empty(starts.size, dtype=np.float64)
        drawdowns = np.empty(starts.size, dtype=np.float64)
        for segment in range(starts.size):
            if state_codes[segment] != state or segment_values[segment, 0] == 0:
                continue
            observations += int(segment_values[segment, 0])
            lengths[segment_count] = segment_values[segment, 0]
            segment_count += 1
            if segment_values[segment, 4] == 0:
                interval_returns[eligible_count] = segment_values[segment, 2]
                drawdowns[eligible_count] = segment_values[segment, 3]
                eligible_count += 1
        for row in range(1, close.size):
            segment = by_row[row]
            if segment < 0 or by_row[row - 1] != segment or state_codes[segment] != state:
                continue
            if np.isfinite(close[row]) and close[row] > 0 and np.isfinite(close[row - 1]) and close[row - 1] > 0:
                change = base_returns[row - 1]
                if np.isfinite(change) and change > -100.0:
                    returns[row - 1] = change / 100.0
        stats = return_statistics_kernel(returns)
        states[state, 0] = observations
        states[state, 1] = stats[6]
        states[state, 2] = segment_count
        states[state, 4] = eligible_count
        states[state, 5] = stats[0]
        states[state, 7] = stats[3]
        if stats[6] >= 2:
            states[state, 6] = stats[1] * math.sqrt(stats[6] / (stats[6] - 1.0)) * math.sqrt(252.0)
        if segment_count > 0:
            ordered_lengths = np.sort(lengths[:segment_count])
            states[state, 3] = _linear_quantile(ordered_lengths, 0.5)
        if eligible_count > 0:
            ordered_returns = np.sort(interval_returns[:eligible_count])
            ordered_drawdowns = np.sort(drawdowns[:eligible_count])
            states[state, 8] = _linear_quantile(ordered_returns, 0.5)
            states[state, 9] = ordered_returns[0]
            states[state, 10] = _linear_quantile(ordered_drawdowns, 0.5)
            states[state, 11] = ordered_drawdowns[0]
    context = np.array([selected_observations, selected_count, selected_segments, selected_first, selected_last], dtype=np.int64)
    return states, segment_values, selected_returns, return_segments, context


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
    _density_frame,
    _summarize_simulation,
    parametric_monte_carlo_kernel,
    stationary_block_bootstrap_kernel,
    _garch_quasi_likelihood,
    _volatility_filter,
    filtered_historical_simulation_kernel,
    simulation_comparison_kernel,
    realized_path_comparison_kernel,
    regime_analysis_kernel,
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
    _density_frame(np.ascontiguousarray(np.array([0.98, 1.0, 1.02], dtype=np.float64)), 2.0)
    parametric = parametric_monte_carlo_kernel(returns, 1.0, 2, 3, 1, 5.0, 1)
    parametric_monte_carlo_kernel(returns, 1.0, 2, 3, 1, 5.0, 0)
    bootstrap = stationary_block_bootstrap_kernel(returns, np.zeros(returns.size, dtype=np.int64), 1.0, 2, 3, 2, 5.0, 5)
    segment_ids = np.zeros(returns.size, dtype=np.int64)
    _garch_quasi_likelihood(log_returns, segment_ids, 1e-6, 0.05, 0.9, 1e-4)
    _volatility_filter(log_returns, segment_ids, 0, 0.94)
    _volatility_filter(log_returns, segment_ids, 1, 0.94)
    filtered_historical_simulation_kernel(returns, segment_ids, 1.0, 2, 3, 3, 5.0, 0, 0.94)
    filtered_historical_simulation_kernel(returns, segment_ids, 1.0, 2, 3, 4, 5.0, 1, 0.94)
    simulation_comparison_kernel(
        np.ascontiguousarray(np.vstack((parametric[3], bootstrap[3]))), 1.0
    )
    realized_path_comparison_kernel(
        np.ascontiguousarray(np.array([1.01, np.nan], dtype=np.float64)),
        1.0,
        parametric[1],
        parametric[2],
        1.0,
    )
    regime_analysis_kernel(
        dates,
        values,
        np.ascontiguousarray(np.array([0], dtype=np.int64)),
        np.ascontiguousarray(np.array([31], dtype=np.int64)),
        np.ascontiguousarray(np.array([0], dtype=np.int64)),
        1, -1, -1,
    )
    audit = validate_execution_audit(product_analysis_execution_audit())
    if audit["nopython"] is not True or audit["python_fallback"] != 0:
        raise RuntimeError("产品详情分析 NJIT 内核预热失败")
    return audit


__all__ = [
    "SIMULATION_ASSUMPTION_SLOTS",
    "bollinger_kernel",
    "box_plot_kernel",
    "daily_returns_percent_kernel",
    "distribution_interpretation_codes_kernel",
    "filtered_historical_simulation_kernel",
    "histogram_normal_pdf_kernel",
    "kdj_kernel",
    "moving_average_kernel",
    "normal_qq_kernel",
    "parametric_monte_carlo_kernel",
    "product_analysis_execution_audit",
    "realized_path_comparison_kernel",
    "regime_analysis_kernel",
    "return_statistics_kernel",
    "simulation_comparison_kernel",
    "stationary_block_bootstrap_kernel",
    "technical_input_availability_kernel",
    "warm_product_analysis_numba_kernels",
]
