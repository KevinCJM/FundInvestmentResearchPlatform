"""Fixed-signature numerical kernels for research-series profiling."""

from __future__ import annotations

import hashlib

import numpy as np
from numba import float64, int64, njit, types, uint8

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


RESEARCH_SERIES_ENGINE_VERSION = "research-series-njit-1.0.0"
RESEARCH_SERIES_KERNEL_VERSION = "profile-statistics-1"

_F1 = float64[::1]
_I1 = int64[::1]
_U1 = uint8[::1]
_F2 = float64[:, ::1]
_I2 = int64[:, ::1]
_INDEX_PROFILE_RESULT = types.Tuple((_F1, _F1, _F1, _F1, _F1))
_MACRO_PROFILE_RESULT = types.Tuple((_F1, _F1, _F1))
_CORRELATION_RESULT = types.Tuple((_F2, _I2))
_NAT_INT64 = np.iinfo(np.int64).min


@njit(float64(_F1, float64), cache=False, nogil=True, inline="always")
def _linear_quantile_sorted(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return np.nan
    position = probability * (values.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


@njit(_F1(_F1), cache=False, nogil=True)
def distribution_summary_kernel(values: np.ndarray) -> np.ndarray:
    """Return full-sample count/missing/rate/moments and linear quantiles."""

    output = np.full(12, np.nan, dtype=np.float64)
    finite_values = np.empty(values.size, dtype=np.float64)
    finite_count = 0
    total = 0.0
    for value in values:
        if np.isfinite(value):
            finite_values[finite_count] = value
            finite_count += 1
            total += value
    missing_count = values.size - finite_count
    output[0] = float(finite_count)
    output[1] = float(missing_count)
    output[2] = float(missing_count) / values.size if values.size else np.nan
    if finite_count == 0:
        return output

    compact = np.sort(finite_values[:finite_count])
    mean = total / finite_count
    output[3] = mean
    if finite_count > 1:
        squared = 0.0
        for value in compact:
            difference = value - mean
            squared += difference * difference
        output[4] = np.sqrt(squared / (finite_count - 1))
    output[5] = compact[0]
    output[6] = _linear_quantile_sorted(compact, 0.05)
    output[7] = _linear_quantile_sorted(compact, 0.25)
    output[8] = _linear_quantile_sorted(compact, 0.50)
    output[9] = _linear_quantile_sorted(compact, 0.75)
    output[10] = _linear_quantile_sorted(compact, 0.95)
    output[11] = compact[-1]
    return output


@njit(_INDEX_PROFILE_RESULT(_F1, int64, float64), cache=False, nogil=True)
def index_profile_kernel(
    values: np.ndarray,
    rolling_window: int,
    annualization: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute index transforms without bridging missing observations."""

    if rolling_window < 2:
        raise ValueError("rolling window must be at least two")
    if annualization <= 0.0:
        raise ValueError("annualization must be positive")
    size = values.size
    normalized = np.full(size, np.nan, dtype=np.float64)
    returns = np.full(size, np.nan, dtype=np.float64)
    cumulative = np.full(size, np.nan, dtype=np.float64)
    drawdown = np.full(size, np.nan, dtype=np.float64)
    rolling_volatility = np.full(size, np.nan, dtype=np.float64)

    finite_count = 0
    total = 0.0
    for value in values:
        if np.isfinite(value):
            finite_count += 1
            total += value
    if finite_count:
        mean = total / finite_count
        variance_total = 0.0
        for value in values:
            if np.isfinite(value):
                difference = value - mean
                variance_total += difference * difference
        deviation = np.sqrt(variance_total / finite_count)
        for index in range(size):
            value = values[index]
            if np.isfinite(value):
                normalized[index] = (value - mean) / deviation if deviation > 0.0 else 0.0

    base = np.nan
    peak = np.nan
    for index in range(size):
        value = values[index]
        if not np.isfinite(value):
            continue
        if not np.isfinite(base):
            base = value
        if np.isfinite(base) and base != 0.0:
            cumulative[index] = value / base - 1.0
        if not np.isfinite(peak) or value > peak:
            peak = value
        if peak != 0.0:
            drawdown[index] = value / peak - 1.0
        if index > 0:
            previous = values[index - 1]
            if np.isfinite(previous) and previous != 0.0:
                candidate = value / previous - 1.0
                if np.isfinite(candidate):
                    returns[index] = candidate

    for end in range(rolling_window, size):
        start = end - rolling_window + 1
        count = 0
        total_return = 0.0
        for index in range(start, end + 1):
            value = returns[index]
            if np.isfinite(value):
                count += 1
                total_return += value
        if count != rolling_window:
            continue
        mean_return = total_return / count
        squared = 0.0
        for index in range(start, end + 1):
            difference = returns[index] - mean_return
            squared += difference * difference
        rolling_volatility[end] = np.sqrt(squared / (count - 1)) * np.sqrt(annualization)
    return normalized, returns, cumulative, drawdown, rolling_volatility


@njit(_MACRO_PROFILE_RESULT(_F1, int64), cache=False, nogil=True)
def macro_profile_kernel(
    values: np.ndarray,
    year_over_year_lag: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute generic percent-change and full-history percentile transforms."""

    if year_over_year_lag < 1:
        raise ValueError("year-over-year lag must be positive")
    size = values.size
    year_over_year = np.full(size, np.nan, dtype=np.float64)
    month_over_month = np.full(size, np.nan, dtype=np.float64)
    quantile = np.full(size, np.nan, dtype=np.float64)
    for index in range(size):
        current = values[index]
        if not np.isfinite(current):
            continue
        if index > 0:
            previous = values[index - 1]
            if np.isfinite(previous) and previous != 0.0:
                month_over_month[index] = current / previous - 1.0
        if index >= year_over_year_lag:
            previous_year = values[index - year_over_year_lag]
            if np.isfinite(previous_year) and previous_year != 0.0:
                year_over_year[index] = current / previous_year - 1.0

    finite_values = np.empty(size, dtype=np.float64)
    finite_indices = np.empty(size, dtype=np.int64)
    finite_count = 0
    for index in range(size):
        if np.isfinite(values[index]):
            finite_values[finite_count] = values[index]
            finite_indices[finite_count] = index
            finite_count += 1
    if finite_count == 1:
        quantile[finite_indices[0]] = 0.5
    elif finite_count > 1:
        compact = finite_values[:finite_count]
        order = np.argsort(compact)
        position = 0
        while position < finite_count:
            group_end = position
            ordered_value = compact[order[position]]
            while group_end + 1 < finite_count and compact[order[group_end + 1]] == ordered_value:
                group_end += 1
            percentile = (position + group_end) * 0.5 / (finite_count - 1)
            for member in range(position, group_end + 1):
                quantile[finite_indices[order[member]]] = percentile
            position = group_end + 1
    return year_over_year, month_over_month, quantile


@njit(_F1(_I1, _I1), cache=False, nogil=True)
def release_lag_days_kernel(
    observation_days: np.ndarray,
    available_days: np.ndarray,
) -> np.ndarray:
    if observation_days.size != available_days.size:
        raise ValueError("date arrays must have equal length")
    output = np.full(observation_days.size, np.nan, dtype=np.float64)
    for index in range(observation_days.size):
        observed = observation_days[index]
        available = available_days[index]
        if observed != _NAT_INT64 and available != _NAT_INT64:
            output[index] = float(available - observed)
    return output


@njit(_I1(int64, int64), cache=False, nogil=True)
def sample_indices_kernel(observation_count: int, maximum_points: int) -> np.ndarray:
    if observation_count < 0 or maximum_points < 1:
        raise ValueError("invalid sampling bounds")
    if observation_count == 0:
        return np.empty(0, dtype=np.int64)
    if observation_count <= maximum_points:
        return np.arange(observation_count, dtype=np.int64)
    if maximum_points == 1:
        return np.array([observation_count - 1], dtype=np.int64)
    output = np.empty(maximum_points, dtype=np.int64)
    scale = float(observation_count - 1) / float(maximum_points - 1)
    for index in range(maximum_points):
        output[index] = int(np.floor(index * scale + 0.5))
    output[0] = 0
    output[-1] = observation_count - 1
    return output


@njit(_U1(_F1), cache=False, nogil=True)
def finite_mask_kernel(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.size, dtype=np.uint8)
    for index in range(values.size):
        output[index] = 1 if np.isfinite(values[index]) else 0
    return output


@njit(int64(_F1), cache=False, nogil=True)
def infinite_count_kernel(values: np.ndarray) -> int:
    count = 0
    for index in range(values.size):
        if np.isinf(values[index]):
            count += 1
    return count


@njit(_F1(_F1), cache=False, nogil=True)
def complement_rate_kernel(coverage_rates: np.ndarray) -> np.ndarray:
    output = np.full(coverage_rates.size, np.nan, dtype=np.float64)
    for index in range(coverage_rates.size):
        value = coverage_rates[index]
        if np.isfinite(value) and 0.0 <= value <= 1.0:
            output[index] = 1.0 - value
    return output


@njit(_I1(_I1, _I1), cache=False, nogil=True)
def strict_intersection_dates_kernel(
    left_dates: np.ndarray,
    right_dates: np.ndarray,
) -> np.ndarray:
    """Intersect two sorted, unique day arrays without Python set semantics."""

    output = np.empty(min(left_dates.size, right_dates.size), dtype=np.int64)
    left_index = 0
    right_index = 0
    output_size = 0
    while left_index < left_dates.size and right_index < right_dates.size:
        left = left_dates[left_index]
        right = right_dates[right_index]
        if left == right:
            output[output_size] = left
            output_size += 1
            left_index += 1
            right_index += 1
        elif left < right:
            left_index += 1
        else:
            right_index += 1
    return output[:output_size]


@njit(_F1(_I1, _I1, _F1), cache=False, nogil=True)
def align_values_kernel(
    common_dates: np.ndarray,
    source_dates: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    if source_dates.size != source_values.size:
        raise ValueError("source dates and values must have equal length")
    output = np.full(common_dates.size, np.nan, dtype=np.float64)
    common_index = 0
    source_index = 0
    while common_index < common_dates.size and source_index < source_dates.size:
        common = common_dates[common_index]
        source = source_dates[source_index]
        if common == source:
            output[common_index] = source_values[source_index]
            common_index += 1
            source_index += 1
        elif common < source:
            common_index += 1
        else:
            source_index += 1
    return output


@njit(_F2(_F2), cache=False, nogil=True)
def standardize_matrix_kernel(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    output = np.full((rows, columns), np.nan, dtype=np.float64)
    for column in range(columns):
        count = 0
        total = 0.0
        for row in range(rows):
            value = values[row, column]
            if np.isfinite(value):
                count += 1
                total += value
        if count == 0:
            continue
        mean = total / count
        squared = 0.0
        for row in range(rows):
            value = values[row, column]
            if np.isfinite(value):
                difference = value - mean
                squared += difference * difference
        deviation = np.sqrt(squared / count)
        for row in range(rows):
            value = values[row, column]
            if np.isfinite(value):
                output[row, column] = (
                    (value - mean) / deviation if deviation > 0.0 else 0.0
                )
    return output


@njit(_CORRELATION_RESULT(_F2), cache=False, nogil=True)
def pearson_matrix_kernel(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = values.shape
    correlations = np.full((columns, columns), np.nan, dtype=np.float64)
    counts = np.zeros((columns, columns), dtype=np.int64)
    for left in range(columns):
        for right in range(left, columns):
            count = 0
            left_total = 0.0
            right_total = 0.0
            for row in range(rows):
                left_value = values[row, left]
                right_value = values[row, right]
                if np.isfinite(left_value) and np.isfinite(right_value):
                    count += 1
                    left_total += left_value
                    right_total += right_value
            counts[left, right] = count
            counts[right, left] = count
            if count < 2:
                continue
            left_mean = left_total / count
            right_mean = right_total / count
            covariance = 0.0
            left_squared = 0.0
            right_squared = 0.0
            for row in range(rows):
                left_value = values[row, left]
                right_value = values[row, right]
                if np.isfinite(left_value) and np.isfinite(right_value):
                    left_difference = left_value - left_mean
                    right_difference = right_value - right_mean
                    covariance += left_difference * right_difference
                    left_squared += left_difference * left_difference
                    right_squared += right_difference * right_difference
            denominator = np.sqrt(left_squared * right_squared)
            if denominator > 0.0:
                correlation = covariance / denominator
                correlations[left, right] = correlation
                correlations[right, left] = correlation
    return correlations, counts


@njit(_I1(_F2), cache=False, nogil=True)
def complete_case_indices_kernel(values: np.ndarray) -> np.ndarray:
    rows, columns = values.shape
    output = np.empty(rows, dtype=np.int64)
    output_size = 0
    for row in range(rows):
        complete = True
        for column in range(columns):
            if not np.isfinite(values[row, column]):
                complete = False
                break
        if complete:
            output[output_size] = row
            output_size += 1
    return output[:output_size]


@njit(_I1(_F2, int64, int64), cache=False, nogil=True)
def pair_valid_indices_kernel(
    values: np.ndarray,
    left_column: int,
    right_column: int,
) -> np.ndarray:
    rows, columns = values.shape
    if (
        left_column < 0
        or right_column < 0
        or left_column >= columns
        or right_column >= columns
    ):
        raise ValueError("pair column is out of bounds")
    output = np.empty(rows, dtype=np.int64)
    output_size = 0
    for row in range(rows):
        if np.isfinite(values[row, left_column]) and np.isfinite(
            values[row, right_column]
        ):
            output[output_size] = row
            output_size += 1
    return output[:output_size]


_DISPATCHERS = (
    _linear_quantile_sorted,
    distribution_summary_kernel,
    index_profile_kernel,
    macro_profile_kernel,
    release_lag_days_kernel,
    sample_indices_kernel,
    finite_mask_kernel,
    infinite_count_kernel,
    complement_rate_kernel,
    strict_intersection_dates_kernel,
    align_values_kernel,
    standardize_matrix_kernel,
    pearson_matrix_kernel,
    complete_case_indices_kernel,
    pair_valid_indices_kernel,
)

for _dispatcher in _DISPATCHERS:
    _dispatcher.disable_compile()


def research_series_numba_execution_audit() -> dict[str, object]:
    signatures = {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.signatures]
        for dispatcher in _DISPATCHERS
    }
    material = "|".join(
        [RESEARCH_SERIES_ENGINE_VERSION, RESEARCH_SERIES_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit(
        {
            "engine": RESEARCH_SERIES_ENGINE_VERSION,
            "backend": "numba_njit_fixed_signature",
            "kernel_version": RESEARCH_SERIES_KERNEL_VERSION,
            "kernel_signatures": signatures,
            "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
            "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in _DISPATCHERS),
            "object_mode": 0,
            "python_fallback": 0,
            "request_time_compilation": 0,
            "fully_warmed": all(bool(dispatcher.signatures) for dispatcher in _DISPATCHERS),
        }
    )


def warm_research_series_numba_kernels() -> dict[str, object]:
    values = np.ascontiguousarray(np.array([100.0, 101.0, np.nan, 103.0], dtype=np.float64))
    distribution = distribution_summary_kernel(values)
    index_result = index_profile_kernel(values, np.int64(2), np.float64(252.0))
    macro_result = macro_profile_kernel(values, np.int64(2))
    days = np.ascontiguousarray(np.array([1, 2, 3, 4], dtype=np.int64))
    lag = release_lag_days_kernel(days, days + 1)
    sample = sample_indices_kernel(np.int64(10), np.int64(4))
    mask = finite_mask_kernel(values)
    infinite_count = infinite_count_kernel(values)
    missing_rates = complement_rate_kernel(
        np.ascontiguousarray(np.array([0.8, np.nan], dtype=np.float64))
    )
    left_dates = np.ascontiguousarray(np.array([1, 2, 4, 5], dtype=np.int64))
    right_dates = np.ascontiguousarray(np.array([2, 3, 4, 5], dtype=np.int64))
    common_dates = strict_intersection_dates_kernel(left_dates, right_dates)
    aligned = align_values_kernel(
        common_dates,
        left_dates,
        np.ascontiguousarray(np.array([1.0, 2.0, np.nan, 5.0], dtype=np.float64)),
    )
    matrix = np.ascontiguousarray(
        np.column_stack((aligned, np.array([2.0, 4.0, 6.0], dtype=np.float64))),
        dtype=np.float64,
    )
    standardized = standardize_matrix_kernel(matrix)
    correlations, correlation_counts = pearson_matrix_kernel(matrix)
    complete_cases = complete_case_indices_kernel(matrix)
    pair_cases = pair_valid_indices_kernel(matrix, np.int64(0), np.int64(1))
    if (
        distribution.size != 12
        or any(item.size != values.size for item in index_result)
        or any(item.size != values.size for item in macro_result)
        or lag.size != values.size
        or sample.size != 4
        or mask.size != values.size
        or infinite_count != 0
        or missing_rates.size != 2
        or common_dates.size != 3
        or aligned.size != common_dates.size
        or standardized.shape != matrix.shape
        or correlations.shape != (2, 2)
        or correlation_counts.shape != (2, 2)
        or complete_cases.size != pair_cases.size
    ):
        raise RuntimeError("研究序列 NJIT 内核预热失败")
    return research_series_numba_execution_audit()


__all__ = [
    "RESEARCH_SERIES_ENGINE_VERSION",
    "RESEARCH_SERIES_KERNEL_VERSION",
    "complement_rate_kernel",
    "complete_case_indices_kernel",
    "distribution_summary_kernel",
    "finite_mask_kernel",
    "infinite_count_kernel",
    "align_values_kernel",
    "index_profile_kernel",
    "macro_profile_kernel",
    "pair_valid_indices_kernel",
    "pearson_matrix_kernel",
    "release_lag_days_kernel",
    "research_series_numba_execution_audit",
    "sample_indices_kernel",
    "standardize_matrix_kernel",
    "strict_intersection_dates_kernel",
    "warm_research_series_numba_kernels",
]
