"""Fixed-signature NJIT kernels used by Regime Graph v2.

All kernels are compiled eagerly at module import and compilation is disabled
before they can be reached by a request. Python only validates and orchestrates
typed contiguous arrays.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numba import float64, int64, njit, types
from numba.core.registry import CPUDispatcher

from compute_policy import NJIT_BACKEND, validate_execution_audit


KERNEL_VERSION = "regime-graph-kernels/2.3.0"
_F64 = float64[::1]
_I64 = int64[::1]
_F64_2D = float64[:, ::1]
_I64_2D = int64[:, ::1]
_POSITION_SPLIT = types.Tuple((_I64, _I64))(_I64, _I64, int64)


@njit(_F64(_F64, int64, int64), cache=True)
def unary_transform_kernel(values: np.ndarray, opcode: int, period: int) -> np.ndarray:
    size = values.shape[0]
    result = np.empty(size, dtype=np.float64)
    for index in range(size):
        result[index] = np.nan
    if opcode == 0:
        for index in range(size):
            result[index] = values[index]
        return result
    if opcode == 1:
        for index in range(size):
            value = values[index]
            if np.isfinite(value) and value > 0.0:
                result[index] = np.log(value)
        return result
    lag = period
    if lag < 1:
        lag = 1
    for index in range(lag, size):
        current = values[index]
        previous = values[index - lag]
        if opcode == 4:
            if np.isfinite(previous):
                result[index] = previous
            continue
        if not np.isfinite(current) or not np.isfinite(previous):
            continue
        if opcode == 2:
            result[index] = current - previous
        elif opcode == 3 and previous != 0.0:
            result[index] = current / previous - 1.0
    return result


@njit(_F64(_F64, _F64, int64), cache=True)
def binary_math_kernel(left: np.ndarray, right: np.ndarray, opcode: int) -> np.ndarray:
    size = left.shape[0]
    result = np.empty(size, dtype=np.float64)
    for index in range(size):
        lhs = left[index]
        rhs = right[index]
        value = np.nan
        if np.isfinite(lhs) and np.isfinite(rhs):
            if opcode == 0:
                value = lhs + rhs
            elif opcode == 1:
                value = lhs - rhs
            elif opcode == 2:
                value = lhs * rhs
            elif opcode == 3 and rhs != 0.0:
                value = lhs / rhs
        result[index] = value
    return result


@njit(_F64(_F64, int64), cache=True)
def ema_kernel(values: np.ndarray, window: int) -> np.ndarray:
    size = values.shape[0]
    result = np.empty(size, dtype=np.float64)
    span = window
    if span < 1:
        span = 1
    alpha = 2.0 / (float(span) + 1.0)
    previous = 0.0
    has_previous = False
    for index in range(size):
        value = values[index]
        if not np.isfinite(value):
            result[index] = np.nan
            continue
        if not has_previous:
            previous = value
            has_previous = True
        else:
            previous = alpha * value + (1.0 - alpha) * previous
        result[index] = previous
    return result


@njit(_F64(_F64, int64, int64), cache=True)
def rolling_kernel(values: np.ndarray, window: int, opcode: int) -> np.ndarray:
    size = values.shape[0]
    result = np.empty(size, dtype=np.float64)
    width = window
    if width < 1:
        width = 1
    for index in range(size):
        result[index] = np.nan
        if index + 1 < width:
            continue
        start = index + 1 - width
        total = 0.0
        total_sq = 0.0
        valid = True
        for cursor in range(start, index + 1):
            value = values[cursor]
            if not np.isfinite(value):
                valid = False
                break
            total += value
            total_sq += value * value
        if not valid:
            continue
        mean = total / float(width)
        variance = total_sq / float(width) - mean * mean
        if variance < 0.0 and variance > -1e-15:
            variance = 0.0
        std = np.sqrt(variance) if variance >= 0.0 else np.nan
        if opcode == 0:
            result[index] = mean
        elif opcode == 1:
            result[index] = std
        elif opcode == 2:
            if std > 0.0:
                result[index] = (values[index] - mean) / std
        elif opcode == 3:
            x_mean = 0.5 * float(width - 1)
            numerator = 0.0
            denominator = 0.0
            for offset in range(width):
                centered = float(offset) - x_mean
                numerator += centered * (values[start + offset] - mean)
                denominator += centered * centered
            if denominator > 0.0:
                result[index] = numerator / denominator
            elif width == 1:
                result[index] = 0.0
        elif opcode == 4:
            minimum = values[start]
            for cursor in range(start + 1, index + 1):
                if values[cursor] < minimum:
                    minimum = values[cursor]
            result[index] = minimum
        else:
            maximum = values[start]
            for cursor in range(start + 1, index + 1):
                if values[cursor] > maximum:
                    maximum = values[cursor]
            result[index] = maximum
    return result


@njit(_I64(_F64, float64, float64), cache=True)
def threshold_state_kernel(values: np.ndarray, upper: float, lower: float) -> np.ndarray:
    size = values.shape[0]
    states = np.empty(size, dtype=np.int64)
    for index in range(size):
        value = values[index]
        if not np.isfinite(value):
            states[index] = -1
        elif value >= upper:
            states[index] = 0
        elif value <= lower:
            states[index] = 2
        else:
            states[index] = 1
    return states


@njit(_I64(_F64, float64, float64, float64, float64), cache=True)
def hysteresis_state_kernel(
    values: np.ndarray,
    upper_enter: float,
    upper_exit: float,
    lower_enter: float,
    lower_exit: float,
) -> np.ndarray:
    size = values.shape[0]
    states = np.empty(size, dtype=np.int64)
    current = 1
    for index in range(size):
        value = values[index]
        if not np.isfinite(value):
            states[index] = -1
            continue
        if current == 0:
            if value <= lower_enter:
                current = 2
            elif value <= upper_exit:
                current = 1
        elif current == 2:
            if value >= upper_enter:
                current = 0
            elif value >= lower_exit:
                current = 1
        elif value >= upper_enter:
            current = 0
        elif value <= lower_enter:
            current = 2
        states[index] = current
    return states


@njit(_I64(_F64, _F64, float64, float64), cache=True)
def quadrant_state_kernel(
    growth: np.ndarray,
    inflation: np.ndarray,
    growth_threshold: float,
    inflation_threshold: float,
) -> np.ndarray:
    size = growth.shape[0]
    states = np.empty(size, dtype=np.int64)
    for index in range(size):
        growth_value = growth[index]
        inflation_value = inflation[index]
        if not np.isfinite(growth_value) or not np.isfinite(inflation_value):
            states[index] = -1
        elif growth_value >= growth_threshold and inflation_value < inflation_threshold:
            states[index] = 0
        elif growth_value >= growth_threshold and inflation_value >= inflation_threshold:
            states[index] = 1
        elif growth_value < growth_threshold and inflation_value >= inflation_threshold:
            states[index] = 2
        else:
            states[index] = 3
    return states


@njit(_I64(_I64, int64, int64), cache=True)
def confirmation_state_kernel(states: np.ndarray, confirmation: int, min_duration: int) -> np.ndarray:
    size = states.shape[0]
    result = np.empty(size, dtype=np.int64)
    confirm_count = confirmation
    duration_floor = min_duration
    if confirm_count < 1:
        confirm_count = 1
    if duration_floor < 1:
        duration_floor = 1
    active = -1
    active_duration = 0
    candidate = -1
    candidate_count = 0
    for index in range(size):
        observed = states[index]
        if observed < 0:
            result[index] = -1
            continue
        if active < 0:
            if observed == candidate:
                candidate_count += 1
            else:
                candidate = observed
                candidate_count = 1
            if candidate_count >= confirm_count:
                active = candidate
                active_duration = 1
            result[index] = active
            continue
        if observed == active:
            active_duration += 1
            candidate = -1
            candidate_count = 0
        elif active_duration >= duration_floor:
            if observed == candidate:
                candidate_count += 1
            else:
                candidate = observed
                candidate_count = 1
            if candidate_count >= confirm_count:
                active = candidate
                active_duration = 1
                candidate = -1
                candidate_count = 0
        result[index] = active
    return result


@njit(_F64_2D(_I64, int64), cache=True)
def state_probabilities_kernel(states: np.ndarray, state_count: int) -> np.ndarray:
    rows = states.shape[0]
    columns = state_count
    if columns < 1:
        columns = 1
    result = np.zeros((rows, columns), dtype=np.float64)
    for row in range(rows):
        state = states[row]
        if state >= 0 and state < columns:
            result[row, state] = 1.0
    return result


@njit(_F64(_I64), cache=True)
def state_confidence_kernel(states: np.ndarray) -> np.ndarray:
    result = np.empty(states.shape[0], dtype=np.float64)
    for index in range(states.shape[0]):
        result[index] = 1.0 if states[index] >= 0 else np.nan
    return result


@njit(_I64(_I64, int64), cache=True)
def state_count_kernel(states: np.ndarray, state_count: int) -> np.ndarray:
    columns = state_count
    if columns < 1:
        columns = 1
    result = np.zeros(columns + 2, dtype=np.int64)
    previous = -1
    for index in range(states.shape[0]):
        state = states[index]
        if state >= 0 and state < columns:
            result[state] += 1
            result[columns] += 1
            if previous >= 0 and state != previous:
                result[columns + 1] += 1
            previous = state
    return result


_ALIGN_SIGNATURE = types.Tuple((_I64, _I64, _I64))(_I64, _I64)


@njit(_ALIGN_SIGNATURE, cache=True)
def strict_intersection_indices_kernel(left: np.ndarray, right: np.ndarray):
    capacity = left.shape[0]
    if right.shape[0] < capacity:
        capacity = right.shape[0]
    common = np.empty(capacity, dtype=np.int64)
    left_positions = np.empty(capacity, dtype=np.int64)
    right_positions = np.empty(capacity, dtype=np.int64)
    left_index = 0
    right_index = 0
    count = 0
    while left_index < left.shape[0] and right_index < right.shape[0]:
        left_value = left[left_index]
        right_value = right[right_index]
        if left_value == right_value:
            common[count] = left_value
            left_positions[count] = left_index
            right_positions[count] = right_index
            count += 1
            left_index += 1
            right_index += 1
        elif left_value < right_value:
            left_index += 1
        else:
            right_index += 1
    return common[:count], left_positions[:count], right_positions[:count]


@njit(_F64(_F64, _I64), cache=True)
def take_float_kernel(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    result = np.empty(positions.shape[0], dtype=np.float64)
    for index in range(positions.shape[0]):
        result[index] = values[positions[index]]
    return result


@njit(_I64(_I64, _I64), cache=True)
def take_int64_kernel(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    result = np.empty(positions.shape[0], dtype=np.int64)
    for index in range(positions.shape[0]):
        result[index] = values[positions[index]]
    return result


@njit(_I64(_I64, _I64), cache=True)
def maximum_int64_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.empty(left.shape[0], dtype=np.int64)
    for index in range(left.shape[0]):
        result[index] = left[index] if left[index] >= right[index] else right[index]
    return result


@njit(_F64(_I64, _I64, _F64), cache=True)
def left_align_float_kernel(
    target_dates: np.ndarray,
    source_dates: np.ndarray,
    source_values: np.ndarray,
) -> np.ndarray:
    result = np.empty(target_dates.shape[0], dtype=np.float64)
    for index in range(target_dates.shape[0]):
        result[index] = np.nan
    target_index = 0
    source_index = 0
    while target_index < target_dates.shape[0] and source_index < source_dates.shape[0]:
        target_date = target_dates[target_index]
        source_date = source_dates[source_index]
        if target_date == source_date:
            result[target_index] = source_values[source_index]
            target_index += 1
            source_index += 1
        elif target_date < source_date:
            target_index += 1
        else:
            source_index += 1
    return result


@njit(_F64(_F64), cache=True)
def drawdown_series_kernel(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape[0], dtype=np.float64)
    peak = -np.inf
    for index in range(values.shape[0]):
        value = values[index]
        if not np.isfinite(value):
            result[index] = np.nan
            continue
        if value > peak:
            peak = value
        result[index] = value / peak - 1.0 if peak != 0.0 else np.nan
    return result


@njit(_F64(_F64, float64, float64), cache=True)
def kalman_filter_kernel(
    values: np.ndarray,
    process_variance: float,
    measurement_variance: float,
) -> np.ndarray:
    result = np.empty(values.shape[0], dtype=np.float64)
    estimate = 0.0
    estimate_variance = 1.0
    initialized = False
    process = max(process_variance, 1e-12)
    measurement = max(measurement_variance, 1e-12)
    for index in range(values.shape[0]):
        observation = values[index]
        if not np.isfinite(observation):
            result[index] = np.nan
            continue
        if not initialized:
            estimate = observation
            initialized = True
        else:
            estimate_variance += process
            gain = estimate_variance / (estimate_variance + measurement)
            estimate += gain * (observation - estimate)
            estimate_variance *= 1.0 - gain
        result[index] = estimate
    return result


@njit(_F64(_F64, float64, float64), cache=True)
def clip_kernel(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    result = np.empty(values.shape[0], dtype=np.float64)
    for index in range(values.shape[0]):
        value = values[index]
        if not np.isfinite(value):
            result[index] = np.nan
        elif value < lower:
            result[index] = lower
        elif value > upper:
            result[index] = upper
        else:
            result[index] = value
    return result


@njit(_F64(_F64, float64), cache=True)
def constant_like_kernel(anchor: np.ndarray, value: float) -> np.ndarray:
    result = np.empty(anchor.shape[0], dtype=np.float64)
    for index in range(anchor.shape[0]):
        result[index] = value if np.isfinite(anchor[index]) else np.nan
    return result


@njit(_I64(int64, int64, int64), cache=True)
def resample_positions_kernel(length: int, every: int, offset: int) -> np.ndarray:
    step = every
    if step < 1:
        step = 1
    start = offset
    if start < 0:
        start = 0
    if start >= length:
        return np.empty(0, dtype=np.int64)
    count = (length - 1 - start) // step + 1
    result = np.empty(count, dtype=np.int64)
    for index in range(count):
        result[index] = start + index * step
    return result


@njit(int64(int64, int64), cache=True)
def calendar_bucket_kernel(timestamp_ns: int, frequency_code: int) -> int:
    """Map a nanosecond timestamp to a stable civil calendar bucket."""

    day_ns = np.int64(86_400_000_000_000)
    days = timestamp_ns // day_ns
    if frequency_code == 0:
        return days
    if frequency_code == 1:
        # 1970-01-01 was Thursday; +3 anchors ISO weeks on Monday.
        return (days + 3) // 7
    shifted = days + 719468
    era = shifted // 146097 if shifted >= 0 else (shifted - 146096) // 146097
    day_of_era = shifted - era * 146097
    year_of_era = (
        day_of_era
        - day_of_era // 1460
        + day_of_era // 36524
        - day_of_era // 146096
    ) // 365
    year = year_of_era + era * 400
    day_of_year = day_of_era - (
        365 * year_of_era + year_of_era // 4 - year_of_era // 100
    )
    month_prime = (5 * day_of_year + 2) // 153
    month = month_prime + 3 if month_prime < 10 else month_prime - 9
    if month <= 2:
        year += 1
    if frequency_code == 2:
        return year * 12 + month - 1
    if frequency_code == 3:
        return year * 4 + (month - 1) // 3
    return year


_CALENDAR_RESAMPLE_SIGNATURE = types.Tuple((_F64, _I64, _I64))(
    _F64, _I64, _I64, int64, int64
)


@njit(_CALENDAR_RESAMPLE_SIGNATURE, cache=True)
def calendar_resample_kernel(
    values: np.ndarray,
    dates: np.ndarray,
    available: np.ndarray,
    frequency_code: int,
    aggregation_code: int,
):
    """Aggregate sorted observations into calendar buckets without filling gaps."""

    size = dates.shape[0]
    if size == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    bucket_count = 1
    previous_bucket = calendar_bucket_kernel(dates[0], frequency_code)
    for index in range(1, size):
        bucket = calendar_bucket_kernel(dates[index], frequency_code)
        if bucket != previous_bucket:
            bucket_count += 1
            previous_bucket = bucket
    output_values = np.empty(bucket_count, dtype=np.float64)
    output_dates = np.empty(bucket_count, dtype=np.int64)
    output_available = np.empty(bucket_count, dtype=np.int64)
    bucket_start = 0
    output_index = 0
    while bucket_start < size:
        bucket = calendar_bucket_kernel(dates[bucket_start], frequency_code)
        bucket_stop = bucket_start + 1
        while (
            bucket_stop < size
            and calendar_bucket_kernel(dates[bucket_stop], frequency_code) == bucket
        ):
            bucket_stop += 1
        if aggregation_code == 0:
            output_values[output_index] = values[bucket_start]
            output_dates[output_index] = dates[bucket_start]
            output_available[output_index] = available[bucket_start]
        elif aggregation_code == 1:
            selected = bucket_stop - 1
            output_values[output_index] = values[selected]
            output_dates[output_index] = dates[selected]
            output_available[output_index] = available[selected]
        else:
            total = 0.0
            valid = True
            latest_available = available[bucket_start]
            for index in range(bucket_start, bucket_stop):
                value = values[index]
                if not np.isfinite(value):
                    valid = False
                else:
                    total += value
                if available[index] > latest_available:
                    latest_available = available[index]
            if not valid:
                output_values[output_index] = np.nan
            elif aggregation_code == 2:
                output_values[output_index] = total / float(bucket_stop - bucket_start)
            else:
                output_values[output_index] = total
            output_dates[output_index] = dates[bucket_stop - 1]
            output_available[output_index] = latest_available
        output_index += 1
        bucket_start = bucket_stop
    return output_values, output_dates, output_available


@njit(_I64_2D(_I64, _I64), cache=True)
def disagreement_spans_kernel(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return inclusive mismatch spans for observations classified by both runs."""

    size = min(reference.shape[0], candidate.shape[0])
    spans = np.empty((size, 2), dtype=np.int64)
    count = 0
    start = -1
    for index in range(size):
        mismatch = (
            reference[index] >= 0
            and candidate[index] >= 0
            and reference[index] != candidate[index]
        )
        if mismatch and start < 0:
            start = index
        elif not mismatch and start >= 0:
            spans[count, 0] = start
            spans[count, 1] = index - 1
            count += 1
            start = -1
    if start >= 0:
        spans[count, 0] = start
        spans[count, 1] = size - 1
        count += 1
    return spans[:count]


@njit(_I64(_I64, _I64, int64), cache=True)
def pit_asof_positions_kernel(
    anchor_dates: np.ndarray,
    feature_available_dates: np.ndarray,
    max_age_days: int,
) -> np.ndarray:
    result = np.full(anchor_dates.shape[0], -1, dtype=np.int64)
    feature_index = 0
    latest = -1
    day_ns = np.int64(86_400_000_000_000)
    for anchor_index in range(anchor_dates.shape[0]):
        cutoff = anchor_dates[anchor_index]
        while (
            feature_index < feature_available_dates.shape[0]
            and feature_available_dates[feature_index] <= cutoff
        ):
            latest = feature_index
            feature_index += 1
        if latest >= 0:
            age = (cutoff - feature_available_dates[latest]) // day_ns
            if max_age_days < 0 or age <= max_age_days:
                result[anchor_index] = latest
    return result


@njit(_I64(_I64), cache=True)
def stable_time_order_kernel(values: np.ndarray) -> np.ndarray:
    """Return a stable O(T log T) order for possibly non-monotonic timestamps."""

    size = values.shape[0]
    order = np.empty(size, dtype=np.int64)
    scratch = np.empty(size, dtype=np.int64)
    for index in range(size):
        order[index] = index
    width = 1
    while width < size:
        start = 0
        while start < size:
            middle = min(start + width, size)
            stop = min(start + width + width, size)
            left = start
            right = middle
            cursor = start
            while left < middle and right < stop:
                if values[order[left]] <= values[order[right]]:
                    scratch[cursor] = order[left]
                    left += 1
                else:
                    scratch[cursor] = order[right]
                    right += 1
                cursor += 1
            while left < middle:
                scratch[cursor] = order[left]
                left += 1
                cursor += 1
            while right < stop:
                scratch[cursor] = order[right]
                right += 1
                cursor += 1
            start = stop
        for index in range(size):
            order[index] = scratch[index]
        width *= 2
    return order


@njit(_F64(_F64, _I64), cache=True)
def take_float_or_nan_kernel(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    result = np.empty(positions.shape[0], dtype=np.float64)
    for index in range(positions.shape[0]):
        position = positions[index]
        result[index] = values[position] if position >= 0 else np.nan
    return result


@njit(_I64(_I64, _I64, _I64), cache=True)
def aligned_available_kernel(
    anchor_available: np.ndarray,
    feature_available: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    result = np.empty(anchor_available.shape[0], dtype=np.int64)
    for index in range(anchor_available.shape[0]):
        position = positions[index]
        feature_value = feature_available[position] if position >= 0 else anchor_available[index]
        result[index] = max(anchor_available[index], feature_value)
    return result


@njit(_F64_2D(_F64, _F64, _F64, _F64, int64), cache=True)
def feature_matrix_kernel(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    feature_count: int,
) -> np.ndarray:
    columns = feature_count
    if columns < 1:
        columns = 1
    if columns > 4:
        columns = 4
    result = np.empty((first.shape[0], columns), dtype=np.float64)
    for index in range(first.shape[0]):
        result[index, 0] = first[index]
        if columns > 1:
            result[index, 1] = second[index]
        if columns > 2:
            result[index, 2] = third[index]
        if columns > 3:
            result[index, 3] = fourth[index]
    return result


@njit(_F64(_F64, _F64, _F64, _F64, _F64, int64, int64, float64), cache=True)
def formula_feature_kernel(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    coefficients: np.ndarray,
    feature_count: int,
    opcode: int,
    intercept: float,
) -> np.ndarray:
    result = np.empty(first.shape[0], dtype=np.float64)
    count = min(max(feature_count, 1), 4)
    for index in range(first.shape[0]):
        values = (first[index], second[index], third[index], fourth[index])
        valid = True
        for feature in range(count):
            if not np.isfinite(values[feature]):
                valid = False
                break
        if not valid:
            result[index] = np.nan
        elif opcode == 1:
            result[index] = values[0] - values[1]
        elif opcode == 2:
            result[index] = values[0] / values[1] if values[1] != 0.0 else np.nan
        else:
            total = intercept
            for feature in range(count):
                total += coefficients[feature] * values[feature]
            result[index] = total
    return result


@njit(_I64(_F64_2D), cache=True)
def finite_row_positions_kernel(matrix: np.ndarray) -> np.ndarray:
    temporary = np.empty(matrix.shape[0], dtype=np.int64)
    count = 0
    for row in range(matrix.shape[0]):
        valid = True
        for column in range(matrix.shape[1]):
            if not np.isfinite(matrix[row, column]):
                valid = False
                break
        if valid:
            temporary[count] = row
            count += 1
    return temporary[:count]


@njit(_POSITION_SPLIT, cache=True)
def split_positions_at_available_kernel(
    positions: np.ndarray,
    available: np.ndarray,
    cutoff: int,
):
    """Split valid positions into expanding training and later test rows."""

    train_count = 0
    for index in range(positions.shape[0]):
        if available[positions[index]] <= cutoff:
            train_count += 1
    train = np.empty(train_count, dtype=np.int64)
    test = np.empty(positions.shape[0] - train_count, dtype=np.int64)
    train_index = 0
    test_index = 0
    for index in range(positions.shape[0]):
        position = positions[index]
        if available[position] <= cutoff:
            train[train_index] = position
            train_index += 1
        else:
            test[test_index] = position
            test_index += 1
    return train, test


@njit(_F64_2D(_F64_2D, _I64), cache=True)
def take_matrix_rows_kernel(matrix: np.ndarray, positions: np.ndarray) -> np.ndarray:
    result = np.empty((positions.shape[0], matrix.shape[1]), dtype=np.float64)
    for row in range(positions.shape[0]):
        for column in range(matrix.shape[1]):
            result[row, column] = matrix[positions[row], column]
    return result


@njit(_F64(_F64_2D, int64), cache=True)
def matrix_column_kernel(matrix: np.ndarray, column: int) -> np.ndarray:
    result = np.empty(matrix.shape[0], dtype=np.float64)
    selected = column
    if selected < 0 or selected >= matrix.shape[1]:
        selected = 0
    for row in range(matrix.shape[0]):
        result[row] = matrix[row, selected]
    return result


_SCATTER_MODEL_SIGNATURE = types.Tuple((_I64, _F64, _F64_2D))(
    _I64, _F64_2D, _I64, _I64, int64, int64
)


@njit(_SCATTER_MODEL_SIGNATURE, cache=True)
def scatter_component_model_kernel(
    assignments: np.ndarray,
    posterior: np.ndarray,
    component_order: np.ndarray,
    positions: np.ndarray,
    total_rows: int,
    state_count: int,
):
    states = np.full(total_rows, -1, dtype=np.int64)
    confidence = np.full(total_rows, np.nan, dtype=np.float64)
    probabilities = np.zeros((total_rows, state_count), dtype=np.float64)
    component_to_state = np.empty(component_order.shape[0], dtype=np.int64)
    for rank in range(component_order.shape[0]):
        state = state_count - 1 - rank
        if state < 0:
            state = 0
        component_to_state[component_order[rank]] = state
    for row in range(assignments.shape[0]):
        target = positions[row]
        component = assignments[row]
        mapped_state = component_to_state[component]
        states[target] = mapped_state
        winner = 0.0
        for component_index in range(posterior.shape[1]):
            state = component_to_state[component_index]
            probabilities[target, state] += posterior[row, component_index]
            if posterior[row, component_index] > winner:
                winner = posterior[row, component_index]
        confidence[target] = winner
    return states, confidence, probabilities


_ENSEMBLE_SIMPLE_SIGNATURE = types.Tuple((_I64, _F64_2D, _F64, int64))(
    _I64, _I64, _I64, _I64, _F64, int64, int64, float64
)


@njit(_ENSEMBLE_SIMPLE_SIGNATURE, cache=True)
def ensemble_state_kernel(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    fourth: np.ndarray,
    weights: np.ndarray,
    member_count: int,
    state_count: int,
    threshold: float,
):
    members = member_count
    if members < 2:
        members = 2
    if members > 4:
        members = 4
    total_weight = 0.0
    for member in range(members):
        total_weight += max(weights[member], 0.0)
    if total_weight <= 0.0:
        total_weight = 1.0
    states = np.full(first.shape[0], -1, dtype=np.int64)
    probabilities = np.zeros((first.shape[0], state_count), dtype=np.float64)
    confidence = np.full(first.shape[0], np.nan, dtype=np.float64)
    conflicts = 0
    for index in range(first.shape[0]):
        values = (first[index], second[index], third[index], fourth[index])
        for member in range(members):
            state = values[member]
            if state >= 0 and state < state_count:
                probabilities[index, state] += max(weights[member], 0.0) / total_weight
        winner = 0
        for state in range(1, state_count):
            if probabilities[index, state] > probabilities[index, winner]:
                winner = state
        confidence[index] = probabilities[index, winner]
        if confidence[index] >= threshold:
            states[index] = winner
        else:
            conflicts += 1
    return states, probabilities, confidence, conflicts


@njit(_I64(_I64, _I64), cache=True)
def component_map_kernel(states: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    result = np.full(states.shape[0], -1, dtype=np.int64)
    for index in range(states.shape[0]):
        state = states[index]
        if state >= 0 and state < mapping.shape[0]:
            result[index] = mapping[state]
    return result


@njit(_I64(_I64, _I64, int64), cache=True)
def combine_states_kernel(left: np.ndarray, right: np.ndarray, opcode: int) -> np.ndarray:
    result = np.full(left.shape[0], -1, dtype=np.int64)
    for index in range(left.shape[0]):
        lhs = left[index]
        rhs = right[index]
        if opcode == 0:
            result[index] = lhs if lhs >= 0 else rhs
        elif lhs >= 0 and lhs == rhs:
            result[index] = lhs
    return result


@njit(_I64(_I64, _F64, float64), cache=True)
def confidence_gate_kernel(states: np.ndarray, confidence: np.ndarray, floor: float) -> np.ndarray:
    result = np.full(states.shape[0], -1, dtype=np.int64)
    for index in range(states.shape[0]):
        if states[index] >= 0 and np.isfinite(confidence[index]) and confidence[index] >= floor:
            result[index] = states[index]
    return result


_TEMPORAL_OUTPUT_SIGNATURE = types.Tuple((_I64, _I64, _I64))(_I64)


@njit(_TEMPORAL_OUTPUT_SIGNATURE, cache=True)
def temporal_output_kernel(states: np.ndarray):
    recognition = np.full(states.shape[0], -1, dtype=np.int64)
    effective = np.full(states.shape[0], -1, dtype=np.int64)
    reasons = np.ones(states.shape[0], dtype=np.int64)
    for index in range(states.shape[0]):
        if states[index] >= 0:
            recognition[index] = index
            reasons[index] = 0
            if index + 1 < states.shape[0]:
                effective[index] = index + 1
    return recognition, effective, reasons


@njit(_I64(_I64, _I64), cache=True)
def effective_from_recognition_kernel(recognition: np.ndarray, states: np.ndarray) -> np.ndarray:
    result = np.full(states.shape[0], -1, dtype=np.int64)
    for index in range(states.shape[0]):
        recognized = recognition[index]
        if states[index] >= 0 and recognized >= 0 and recognized + 1 < states.shape[0]:
            result[index] = recognized + 1
    return result


@njit(_F64(_I64, _I64), cache=True)
def recognition_delay_summary_kernel(
    states: np.ndarray,
    recognition: np.ndarray,
) -> np.ndarray:
    """Summarize non-negative recognition lag for classified observations."""

    result = np.full(4, np.nan, dtype=np.float64)
    usable = 0
    delayed = 0
    total_delay = 0.0
    maximum_delay = 0.0
    size = min(states.shape[0], recognition.shape[0])
    for index in range(size):
        recognized = recognition[index]
        if states[index] < 0 or recognized < index:
            continue
        delay = float(recognized - index)
        usable += 1
        total_delay += delay
        if delay > 0.0:
            delayed += 1
        if delay > maximum_delay:
            maximum_delay = delay
    result[0] = float(usable)
    if usable > 0:
        result[1] = total_delay / float(usable)
        result[2] = maximum_delay
        result[3] = float(delayed) / float(usable)
    return result


@njit(_F64(_F64_2D, _I64, int64), cache=True)
def probability_contract_kernel(
    probabilities: np.ndarray,
    states: np.ndarray,
    state_count: int,
) -> np.ndarray:
    """Validate probability shape, finiteness, bounds and classified row sums."""

    result = np.zeros(5, dtype=np.float64)
    if probabilities.shape[0] != states.shape[0] or probabilities.shape[1] != state_count:
        result[0] = 1.0
    rows = min(probabilities.shape[0], states.shape[0])
    for row in range(rows):
        state = states[row]
        if state < -1 or state >= state_count:
            result[4] += 1.0
        total = 0.0
        row_valid = True
        for column in range(probabilities.shape[1]):
            value = probabilities[row, column]
            if not np.isfinite(value):
                result[1] += 1.0
                row_valid = False
            elif value < 0.0 or value > 1.0:
                result[2] += 1.0
                row_valid = False
            else:
                total += value
        if row_valid:
            if state >= 0:
                if np.abs(total - 1.0) > 1.0e-8:
                    result[3] += 1.0
            elif np.abs(total) > 1.0e-8 and np.abs(total - 1.0) > 1.0e-8:
                result[3] += 1.0
    return result


@njit(_I64(_F64, _I64, _I64, _I64, _I64), cache=True)
def final_output_contract_kernel(
    confidence: np.ndarray,
    recognition: np.ndarray,
    effective: np.ndarray,
    reasons: np.ndarray,
    states: np.ndarray,
) -> np.ndarray:
    """Validate scalar final outputs without falling back to Python math."""

    result = np.zeros(9, dtype=np.int64)
    size = states.shape[0]
    if confidence.shape[0] != size:
        result[0] = 1
    for index in range(min(confidence.shape[0], size)):
        value = confidence[index]
        if states[index] >= 0 and (
            not np.isfinite(value) or value < 0.0 or value > 1.0
        ):
            result[1] += 1

    if recognition.shape[0] != size:
        result[2] = 1
    for index in range(min(recognition.shape[0], size)):
        value = recognition[index]
        if value < -1 or value >= size:
            result[3] += 1
        if states[index] >= 0 and value < 0:
            result[4] += 1

    if effective.shape[0] != size:
        result[5] = 1
    for index in range(min(effective.shape[0], size)):
        value = effective[index]
        if value < -1 or value >= size:
            result[6] += 1

    if reasons.shape[0] != size:
        result[7] = 1
    for index in range(min(reasons.shape[0], size)):
        if reasons[index] < 0:
            result[8] += 1
    return result


KERNELS: dict[str, CPUDispatcher] = {
    "unary_transform": unary_transform_kernel,
    "binary_math": binary_math_kernel,
    "ema": ema_kernel,
    "rolling": rolling_kernel,
    "threshold_state": threshold_state_kernel,
    "hysteresis_state": hysteresis_state_kernel,
    "quadrant_state": quadrant_state_kernel,
    "confirmation_state": confirmation_state_kernel,
    "state_probabilities": state_probabilities_kernel,
    "state_confidence": state_confidence_kernel,
    "state_count": state_count_kernel,
    "strict_intersection_indices": strict_intersection_indices_kernel,
    "take_float": take_float_kernel,
    "take_int64": take_int64_kernel,
    "maximum_int64": maximum_int64_kernel,
    "left_align_float": left_align_float_kernel,
    "drawdown_series": drawdown_series_kernel,
    "kalman_filter": kalman_filter_kernel,
    "clip": clip_kernel,
    "constant_like": constant_like_kernel,
    "resample_positions": resample_positions_kernel,
    "calendar_bucket": calendar_bucket_kernel,
    "calendar_resample": calendar_resample_kernel,
    "disagreement_spans": disagreement_spans_kernel,
    "pit_asof_positions": pit_asof_positions_kernel,
    "stable_time_order": stable_time_order_kernel,
    "take_float_or_nan": take_float_or_nan_kernel,
    "aligned_available": aligned_available_kernel,
    "feature_matrix": feature_matrix_kernel,
    "formula_feature": formula_feature_kernel,
    "finite_row_positions": finite_row_positions_kernel,
    "split_positions_at_available": split_positions_at_available_kernel,
    "take_matrix_rows": take_matrix_rows_kernel,
    "matrix_column": matrix_column_kernel,
    "scatter_component_model": scatter_component_model_kernel,
    "ensemble_state": ensemble_state_kernel,
    "component_map": component_map_kernel,
    "combine_states": combine_states_kernel,
    "confidence_gate": confidence_gate_kernel,
    "temporal_output": temporal_output_kernel,
    "effective_from_recognition": effective_from_recognition_kernel,
    "recognition_delay_summary": recognition_delay_summary_kernel,
    "probability_contract": probability_contract_kernel,
    "final_output_contract": final_output_contract_kernel,
}


for _dispatcher in KERNELS.values():
    _dispatcher.disable_compile()


def _kernel_fingerprint(dispatcher: CPUDispatcher) -> str:
    digest = hashlib.sha256()
    digest.update(KERNEL_VERSION.encode("utf-8"))
    digest.update(dispatcher.py_func.__name__.encode("utf-8"))
    digest.update(dispatcher.py_func.__code__.co_code)
    return digest.hexdigest()


def regime_graph_numba_status() -> dict[str, Any]:
    kernel_items: list[dict[str, Any]] = []
    signatures: dict[str, list[str]] = {}
    complete = True
    for kernel_id, dispatcher in KERNELS.items():
        compiled = [str(signature) for signature in dispatcher.signatures]
        nopython = [str(signature) for signature in dispatcher.nopython_signatures]
        is_complete = bool(compiled) and len(compiled) == len(nopython) and dispatcher._can_compile is False
        complete = complete and is_complete
        signatures[kernel_id] = compiled
        kernel_items.append(
            {
                "kernel_id": kernel_id,
                "compile_status": "compiled" if is_complete else "incomplete",
                "compiled_signatures": compiled,
                "kernel_fingerprint": _kernel_fingerprint(dispatcher),
                "python_fallback": 0,
            }
        )
    audit = {
        "engine": "regime_graph_v2",
        "kernel_version": KERNEL_VERSION,
        "complete": complete,
        "execution_backend": NJIT_BACKEND,
        "nopython": complete,
        "object_mode": 0,
        "python_fallback": 0,
        "python_operator_calls": 0,
        "request_time_compilation": 0,
        "kernel_signatures": signatures,
        "kernels": kernel_items,
    }
    return validate_execution_audit(audit)


def warm_regime_graph_numba_kernels() -> dict[str, Any]:
    """Verify the eager fixed signatures; never compiles a new signature."""

    return regime_graph_numba_status()


__all__ = [
    "KERNELS",
    "KERNEL_VERSION",
    "binary_math_kernel",
    "aligned_available_kernel",
    "clip_kernel",
    "combine_states_kernel",
    "component_map_kernel",
    "confirmation_state_kernel",
    "confidence_gate_kernel",
    "constant_like_kernel",
    "drawdown_series_kernel",
    "ema_kernel",
    "effective_from_recognition_kernel",
    "ensemble_state_kernel",
    "feature_matrix_kernel",
    "final_output_contract_kernel",
    "formula_feature_kernel",
    "finite_row_positions_kernel",
    "hysteresis_state_kernel",
    "left_align_float_kernel",
    "kalman_filter_kernel",
    "maximum_int64_kernel",
    "matrix_column_kernel",
    "quadrant_state_kernel",
    "regime_graph_numba_status",
    "recognition_delay_summary_kernel",
    "rolling_kernel",
    "resample_positions_kernel",
    "pit_asof_positions_kernel",
    "probability_contract_kernel",
    "scatter_component_model_kernel",
    "split_positions_at_available_kernel",
    "state_confidence_kernel",
    "state_count_kernel",
    "state_probabilities_kernel",
    "stable_time_order_kernel",
    "strict_intersection_indices_kernel",
    "take_float_kernel",
    "take_float_or_nan_kernel",
    "take_int64_kernel",
    "take_matrix_rows_kernel",
    "threshold_state_kernel",
    "temporal_output_kernel",
    "unary_transform_kernel",
    "warm_regime_graph_numba_kernels",
]
