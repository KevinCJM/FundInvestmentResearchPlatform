"""Fixed-signature NJIT kernels for historical-regime computation.

Python callers are deliberately limited to validation, data coercion and result
serialisation.  Every numerical path used by a historical-regime run is exposed
through a precompiled, auditable Numba dispatcher in this module.
"""

from __future__ import annotations

import hashlib
import inspect
import threading
from typing import Any

import numpy as np
from numba import boolean, float64, int64, njit, types, uint8


KERNEL_VERSION = "historical-regime-kernels-1.5.0"
ENGINE_VERSION = "numba-njit-fixed-signature-1.0.0"

_F1 = float64[::1]
_F2 = float64[:, ::1]
_I1 = int64[::1]
_I2 = int64[:, ::1]
_U1 = uint8[::1]
_FORMULA_INPUT_BLOCKS_RESULT = types.Tuple((_U1, _I1, _I1, int64))
_FORMULA_MISSING_MASK_RESULT = types.Tuple((_F2, _I1, int64))
_VALIDATION_WINDOWS_RESULT = types.Tuple((int64, _I2))


@njit(_F1(_F1, _F1, int64), cache=True, nogil=True)
def relative_transform_kernel(
    numerator: np.ndarray,
    denominator: np.ndarray,
    transform_code: int,
) -> np.ndarray:
    """Build ratio/log-ratio values without a Python or pandas numeric path."""

    if numerator.size != denominator.size:
        raise ValueError("relative series lengths must match")
    if transform_code not in (0, 1):
        raise ValueError("relative transform code is unsupported")
    output = np.full(numerator.size, np.nan, dtype=np.float64)
    for index in range(numerator.size):
        left = numerator[index]
        right = denominator[index]
        if not np.isfinite(left) or not np.isfinite(right) or left <= 0.0 or right <= 0.0:
            continue
        ratio = left / right
        output[index] = np.log(ratio) if transform_code == 1 else ratio
    return output


@njit(_FORMULA_INPUT_BLOCKS_RESULT(_F2), cache=False, nogil=True)
def formula_input_blocks_kernel(
    inputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build one all-input finite mask and its contiguous execution blocks."""

    observation_count = inputs.shape[1]
    complete = np.ones(observation_count, dtype=np.uint8)
    complete_count = 0
    for column in range(observation_count):
        for row in range(inputs.shape[0]):
            if not np.isfinite(inputs[row, column]):
                complete[column] = 0
                break
        if complete[column] != 0:
            complete_count += 1

    block_count = 0
    inside_block = False
    for column in range(observation_count):
        if complete[column] != 0:
            if not inside_block:
                block_count += 1
                inside_block = True
        else:
            inside_block = False

    starts = np.empty(block_count, dtype=np.int64)
    stops = np.empty(block_count, dtype=np.int64)
    block_index = 0
    inside_block = False
    for column in range(observation_count):
        if complete[column] != 0:
            if not inside_block:
                starts[block_index] = column
                inside_block = True
        elif inside_block:
            stops[block_index] = column
            block_index += 1
            inside_block = False
    if inside_block:
        stops[block_index] = observation_count
    return complete, starts, stops, complete_count


@njit(int64(_F1, int64, int64, _F1), cache=False, nogil=True)
def formula_align_block_kernel(
    result: np.ndarray,
    start: int,
    stop: int,
    output: np.ndarray,
) -> int:
    """Right-align one causal formula block to its original time coordinates."""

    if start < 0 or stop < start or stop > output.size:
        raise ValueError("formula block bounds are invalid")
    block_size = stop - start
    if result.size > block_size:
        raise ValueError("formula result exceeds its input block")
    if result.size == 0:
        return 0
    target_start = stop - result.size
    for index in range(result.size):
        output[target_start + index] = result[index]
    return result.size


@njit(_FORMULA_MISSING_MASK_RESULT(_F1, _F2), cache=False, nogil=True)
def formula_missing_numeric_mask_kernel(
    formula_values: np.ndarray,
    numeric_outputs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Mask every numeric output where the formula is missing or non-finite."""

    if numeric_outputs.shape[1] != formula_values.size:
        raise ValueError("formula and numeric output lengths must match")
    missing_count = 0
    for value in formula_values:
        if not np.isfinite(value):
            missing_count += 1
    missing_positions = np.empty(missing_count, dtype=np.int64)
    masked = numeric_outputs.copy()
    position_index = 0
    for column in range(formula_values.size):
        if np.isfinite(formula_values[column]):
            continue
        missing_positions[position_index] = column
        position_index += 1
        for row in range(masked.shape[0]):
            masked[row, column] = np.nan
    return masked, missing_positions, formula_values.size - missing_count


@njit(cache=True, nogil=True, inline="always")
def _finite_mean(values: np.ndarray, start: int, end: int, minimum: int) -> float:
    total = 0.0
    count = 0
    for index in range(start, end):
        value = values[index]
        if np.isfinite(value):
            total += value
            count += 1
    if count < minimum:
        return np.nan
    return total / count


@njit(cache=True, nogil=True, inline="always")
def _finite_std(
    values: np.ndarray,
    start: int,
    end: int,
    minimum: int,
) -> float:
    total = 0.0
    count = 0
    for index in range(start, end):
        value = values[index]
        if np.isfinite(value):
            total += value
            count += 1
    if count < minimum or count < 2:
        return np.nan
    mean = total / count
    squared = 0.0
    for index in range(start, end):
        value = values[index]
        if np.isfinite(value):
            difference = value - mean
            squared += difference * difference
    return np.sqrt(squared / (count - 1))


@njit(cache=True, nogil=True)
def _one_sided_filter_impl(
    values: np.ndarray,
    filter_code: int,
    window: int,
    process_variance: float,
    measurement_variance: float,
    zero_phase_order: int,
) -> np.ndarray:
    size = values.size
    output = np.full(size, np.nan, dtype=np.float64)
    minimum = min(window, max(3, window // 2))
    if filter_code == 0:  # EMA
        alpha = 2.0 / (window + 1.0)
        current = np.nan
        observed = 0
        for index in range(size):
            value = values[index]
            if not np.isfinite(value):
                continue
            if not np.isfinite(current):
                current = value
            else:
                current = alpha * value + (1.0 - alpha) * current
            observed += 1
            if observed >= minimum:
                output[index] = current
        return output
    if filter_code == 1:  # causal SMA
        for index in range(size):
            start = max(0, index - window + 1)
            output[index] = _finite_mean(values, start, index + 1, minimum)
        return output
    if filter_code == 2:  # scalar Kalman filter
        variance = 1.0
        current = np.nan
        for index in range(size):
            observation = values[index]
            if not np.isfinite(observation):
                continue
            if not np.isfinite(current):
                current = observation
            predicted_variance = variance + process_variance
            gain = predicted_variance / (
                predicted_variance + measurement_variance
            )
            current = current + gain * (observation - current)
            variance = (1.0 - gain) * predicted_variance
            output[index] = current
        return output

    # A forward/backward low-pass cascade.  It is deliberately non-causal and
    # therefore only admitted by the retrospective execution contract.  The
    # two directions remove phase delay without calling scipy or Python.
    cutoff = min(0.99, max(0.01, 2.0 / window))
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff)
    work = values.copy()
    for _ in range(zero_phase_order):
        current = np.nan
        for index in range(size):
            value = work[index]
            if not np.isfinite(value):
                continue
            if not np.isfinite(current):
                current = value
            else:
                current = alpha * value + (1.0 - alpha) * current
            output[index] = current
        current = np.nan
        for index in range(size - 1, -1, -1):
            value = output[index]
            if not np.isfinite(value):
                continue
            if not np.isfinite(current):
                current = value
            else:
                current = alpha * value + (1.0 - alpha) * current
            work[index] = current
        output[:] = work
    for index in range(size):
        if not np.isfinite(values[index]):
            output[index] = np.nan
    return output


_FEATURE_RETURN = types.Tuple((_F1, _F1, _F1, _F1))


@njit(
    _FEATURE_RETURN(
        _F1,
        int64,
        int64,
        int64,
        int64,
        int64,
        float64,
        float64,
        int64,
    ),
    cache=True,
    nogil=True,
)
def feature_pipeline_kernel(
    raw_values: np.ndarray,
    transform_code: int,
    filter_code: int,
    window: int,
    slope_window: int,
    volatility_window: int,
    process_variance: float,
    measurement_variance: float,
    zero_phase_order: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transform, filter, slope and volatility in one nopython path."""

    size = raw_values.size
    transformed = np.full(size, np.nan, dtype=np.float64)
    if transform_code == 0:
        transformed[:] = raw_values
    elif transform_code == 1:
        for index in range(size):
            value = raw_values[index]
            if np.isfinite(value) and value > 0.0:
                transformed[index] = np.log(value)
    elif transform_code == 2:
        for index in range(1, size):
            current = raw_values[index]
            prior = raw_values[index - 1]
            if np.isfinite(current) and np.isfinite(prior) and prior != 0.0:
                transformed[index] = current / prior - 1.0
    else:  # causal rolling z-score
        minimum = max(3, window // 2)
        for index in range(size):
            start = max(0, index - window + 1)
            mean = _finite_mean(raw_values, start, index + 1, minimum)
            deviation = _finite_std(raw_values, start, index + 1, minimum)
            value = raw_values[index]
            if (
                np.isfinite(value)
                and np.isfinite(mean)
                and np.isfinite(deviation)
                and deviation > 0.0
            ):
                transformed[index] = (value - mean) / deviation

    filtered = _one_sided_filter_impl(
        transformed,
        filter_code,
        window,
        process_variance,
        measurement_variance,
        zero_phase_order,
    )
    slope = np.full(size, np.nan, dtype=np.float64)
    for index in range(slope_window, size):
        current = filtered[index]
        prior = filtered[index - slope_window]
        if np.isfinite(current) and np.isfinite(prior):
            slope[index] = (current - prior) / slope_window

    changes = np.full(size, np.nan, dtype=np.float64)
    for index in range(1, size):
        current = transformed[index]
        prior = transformed[index - 1]
        if np.isfinite(current) and np.isfinite(prior):
            changes[index] = current - prior
    volatility = np.full(size, np.nan, dtype=np.float64)
    minimum_volatility = max(3, volatility_window // 2)
    for index in range(size):
        start = max(0, index - volatility_window + 1)
        volatility[index] = _finite_std(
            changes,
            start,
            index + 1,
            minimum_volatility,
        )
    return transformed, filtered, slope, volatility


_CONFIRM_RETURN = types.Tuple((_I1, _I1, _I1, _I1, _I1, _I1))


@njit(cache=True, nogil=True)
def _confirm_codes_impl(
    desired: np.ndarray,
    neutral_code: int,
    confirmation: int,
    minimum_duration: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = desired.size
    labels = np.full(size, -1, dtype=np.int64)
    reason_codes = np.zeros(size, dtype=np.int64)
    pending_counts = np.zeros(size, dtype=np.int64)
    transition_from = np.full(size, -1, dtype=np.int64)
    transition_to = np.full(size, -1, dtype=np.int64)
    recognition = np.arange(size, dtype=np.int64)
    current = neutral_code
    pending = -1
    pending_count = 0
    last_switch = -minimum_duration
    for index in range(size):
        candidate = desired[index]
        if candidate < 0:
            reason_codes[index] = 0
            continue
        if candidate == current:
            pending = -1
            pending_count = 0
        elif candidate != pending:
            pending = candidate
            pending_count = 1
        else:
            pending_count += 1
        if (
            pending >= 0
            and pending_count >= confirmation
            and index - last_switch >= minimum_duration
        ):
            transition_from[index] = current
            transition_to[index] = pending
            current = pending
            last_switch = index
            reason_codes[index] = 1
            pending = -1
            pending_count = 0
        elif pending >= 0:
            reason_codes[index] = 2
            pending_counts[index] = pending_count
        else:
            reason_codes[index] = 3
        labels[index] = current
    return (
        labels,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    )


_TREND_RETURN = types.Tuple((_I1, _F1, _I1, _I1, _I1, _I1, _I1, _I1))


@njit(
    _TREND_RETURN(
        _F1,
        _F1,
        float64,
        float64,
        float64,
        float64,
        int64,
        int64,
    ),
    cache=True,
    nogil=True,
)
def trend_state_kernel(
    raw_values: np.ndarray,
    slope: np.ndarray,
    upper: float,
    lower: float,
    positive_exit: float,
    negative_exit: float,
    confirmation: int,
    minimum_duration: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    size = raw_values.size
    desired = np.full(size, -1, dtype=np.int64)
    previous = 1
    for index in range(size):
        value = slope[index]
        if not np.isfinite(raw_values[index]) or not np.isfinite(value):
            continue
        if previous == 0 and value > positive_exit:
            candidate = 0
        elif previous == 2 and value < negative_exit:
            candidate = 2
        elif value >= upper:
            candidate = 0
        elif value <= lower:
            candidate = 2
        else:
            candidate = 1
        desired[index] = candidate
        previous = candidate
    (
        labels,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = _confirm_codes_impl(desired, 1, confirmation, minimum_duration)
    scale = max(abs(upper), abs(lower), 1e-12)
    confidence = np.full(size, np.nan, dtype=np.float64)
    for index in range(size):
        if np.isfinite(slope[index]) and np.isfinite(raw_values[index]):
            confidence[index] = min(1.0, abs(slope[index]) / scale)
    return (
        labels,
        confidence,
        recognition,
        desired,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    )


_TURNING_RETURN = types.Tuple((_I1, _I1, _F1, _F1, _I1))


@njit(
    _TURNING_RETURN(_F1, int64, float64),
    cache=True,
    nogil=True,
)
def turning_point_kernel(
    values: np.ndarray,
    window: int,
    minimum_move: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = values.size
    extrema_indices = np.empty(size, dtype=np.int64)
    extrema_kinds = np.empty(size, dtype=np.int64)  # 1 peak, -1 trough
    extrema_count = 0
    for index in range(window, size - window):
        current = values[index]
        if not np.isfinite(current):
            continue
        valid = True
        is_peak = True
        is_trough = True
        for other in range(index - window, index + window + 1):
            candidate = values[other]
            if not np.isfinite(candidate):
                valid = False
                break
            if current < candidate:
                is_peak = False
            if current > candidate:
                is_trough = False
        if not valid or (not is_peak and not is_trough):
            continue
        kind = 1 if is_peak else -1
        if extrema_count > 0 and extrema_kinds[extrema_count - 1] == kind:
            prior_index = extrema_indices[extrema_count - 1]
            better = (
                values[index] > values[prior_index]
                if kind == 1
                else values[index] < values[prior_index]
            )
            if better:
                extrema_indices[extrema_count - 1] = index
        else:
            extrema_indices[extrema_count] = index
            extrema_kinds[extrema_count] = kind
            extrema_count += 1

    labels = np.full(size, -1, dtype=np.int64)
    recognition = np.empty(size, dtype=np.int64)
    confidence = np.full(size, np.nan, dtype=np.float64)
    segment_move = np.full(size, np.nan, dtype=np.float64)
    extremum_kind = np.zeros(size, dtype=np.int64)
    for index in range(size):
        recognition[index] = min(index + window, size - 1)
    for extremum in range(extrema_count):
        index = extrema_indices[extremum]
        extremum_kind[index] = extrema_kinds[extremum]
    for extremum in range(extrema_count - 1):
        start = extrema_indices[extremum]
        end = extrema_indices[extremum + 1]
        kind = extrema_kinds[extremum]
        next_kind = extrema_kinds[extremum + 1]
        if end <= start or kind == next_kind:
            continue
        move = np.nan
        if values[start] != 0.0:
            move = values[end] / values[start] - 1.0
        state = 1
        if kind == -1 and np.isfinite(move) and move >= minimum_move:
            state = 0
        elif kind == 1 and np.isfinite(move) and move <= -minimum_move:
            state = 2
        for index in range(start, end + 1):
            labels[index] = state
            recognition[index] = min(end + window, size - 1)
            confidence[index] = 1.0
            segment_move[index] = move
    return labels, recognition, confidence, segment_move, extremum_kind


_MERRILL_RETURN = types.Tuple(
    (_F1, _F1, _F1, _I1, _F1, _I1, _I1, _I1, _I1, _I1)
)


@njit(
    _MERRILL_RETURN(
        _F1,
        _F1,
        int64,
        int64,
        int64,
        float64,
        float64,
        int64,
        int64,
    ),
    cache=True,
    nogil=True,
)
def merrill_clock_kernel(
    growth: np.ndarray,
    inflation: np.ndarray,
    filter_code: int,
    window: int,
    slope_window: int,
    process_variance: float,
    measurement_variance: float,
    zero_phase_order: int,
    confirmation: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    growth_filtered = _one_sided_filter_impl(
        growth,
        filter_code,
        window,
        process_variance,
        measurement_variance,
        zero_phase_order,
    )
    inflation_filtered = _one_sided_filter_impl(
        inflation,
        filter_code,
        window,
        process_variance,
        measurement_variance,
        zero_phase_order,
    )
    size = growth.size
    growth_direction = np.full(size, np.nan, dtype=np.float64)
    inflation_direction = np.full(size, np.nan, dtype=np.float64)
    desired = np.full(size, -1, dtype=np.int64)
    neutral_code = 0
    neutral_found = False
    for index in range(slope_window, size):
        if (
            np.isfinite(growth_filtered[index])
            and np.isfinite(growth_filtered[index - slope_window])
        ):
            growth_direction[index] = (
                growth_filtered[index] - growth_filtered[index - slope_window]
            )
        if (
            np.isfinite(inflation_filtered[index])
            and np.isfinite(inflation_filtered[index - slope_window])
        ):
            inflation_direction[index] = (
                inflation_filtered[index]
                - inflation_filtered[index - slope_window]
            )
        if (
            np.isfinite(growth[index])
            and np.isfinite(inflation[index])
            and np.isfinite(growth_direction[index])
            and np.isfinite(inflation_direction[index])
        ):
            growth_up = growth_direction[index] >= 0.0
            inflation_up = inflation_direction[index] >= 0.0
            if growth_up and not inflation_up:
                desired[index] = 0
            elif growth_up and inflation_up:
                desired[index] = 1
            elif not growth_up and inflation_up:
                desired[index] = 2
            else:
                desired[index] = 3
            if not neutral_found:
                neutral_code = desired[index]
                neutral_found = True
    (
        labels,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = _confirm_codes_impl(desired, neutral_code, confirmation, 1)

    magnitude = np.full(size, np.nan, dtype=np.float64)
    finite_values = np.empty(size, dtype=np.float64)
    finite_count = 0
    for index in range(size):
        if np.isfinite(growth_direction[index]) and np.isfinite(
            inflation_direction[index]
        ):
            value = np.sqrt(
                growth_direction[index] * growth_direction[index]
                + inflation_direction[index] * inflation_direction[index]
            )
            magnitude[index] = value
            finite_values[finite_count] = value
            finite_count += 1
    scale = 1.0
    if finite_count > 0:
        scale = np.median(finite_values[:finite_count])
    confidence = np.full(size, np.nan, dtype=np.float64)
    denominator = max(scale * 2.0, 1e-12)
    for index in range(size):
        if np.isfinite(magnitude[index]):
            confidence[index] = min(1.0, magnitude[index] / denominator)
    return (
        growth_filtered,
        growth_direction,
        inflation_direction,
        labels,
        confidence,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    )


@njit(_I1(_F2), cache=True, nogil=True)
def valid_rows_kernel(matrix: np.ndarray) -> np.ndarray:
    valid = np.ones(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not np.isfinite(matrix[row, column]):
                valid[row] = 0
                break
    return valid


_STANDARDIZE_RETURN = types.Tuple((_F2, _F1, _F1))


@njit(_STANDARDIZE_RETURN(_F2), cache=True, nogil=True)
def standardize_fit_kernel(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = matrix.shape
    mean = np.empty(columns, dtype=np.float64)
    scale = np.empty(columns, dtype=np.float64)
    standardized = np.empty_like(matrix)
    for column in range(columns):
        total = 0.0
        for row in range(rows):
            total += matrix[row, column]
        column_mean = total / rows
        mean[column] = column_mean
        squared = 0.0
        for row in range(rows):
            difference = matrix[row, column] - column_mean
            squared += difference * difference
        column_scale = np.sqrt(squared / rows)
        if column_scale < 1e-10:
            column_scale = 1.0
        scale[column] = column_scale
        for row in range(rows):
            standardized[row, column] = (
                matrix[row, column] - column_mean
            ) / column_scale
    return standardized, mean, scale


@njit(_F2(_F2, _F1, _F1), cache=True, nogil=True)
def apply_standardization_kernel(
    matrix: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    output = np.empty_like(matrix)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                output[row, column] = (value - mean[column]) / scale[column]
            else:
                output[row, column] = np.nan
    return output


@njit(cache=True, nogil=True, inline="always")
def _logsumexp_vector(values: np.ndarray) -> float:
    maximum = -np.inf
    for index in range(values.size):
        if values[index] > maximum:
            maximum = values[index]
    if not np.isfinite(maximum):
        return maximum
    total = 0.0
    for index in range(values.size):
        total += np.exp(values[index] - maximum)
    return maximum + np.log(max(total, 1e-300))


@njit(cache=True, nogil=True)
def _gaussian_log_density_impl(
    matrix: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    rows = matrix.shape[0]
    components = means.shape[0]
    dimensions = matrix.shape[1]
    output = np.empty((rows, components), dtype=np.float64)
    constant = dimensions * np.log(2.0 * np.pi)
    for row in range(rows):
        for component in range(components):
            log_variance = 0.0
            quadratic = 0.0
            for dimension in range(dimensions):
                variance = max(variances[component, dimension], 1e-12)
                difference = matrix[row, dimension] - means[component, dimension]
                log_variance += np.log(variance)
                quadratic += difference * difference / variance
            output[row, component] = -0.5 * (
                constant + log_variance + quadratic
            )
    return output


_GMM_FIT_RETURN = types.Tuple((_F1, _F2, _F2, int64))


@njit(_F2(_F2, int64, int64, int64, _F2), cache=True, nogil=True)
def initialize_gaussian_means_kernel(
    matrix: np.ndarray,
    components: int,
    strategy_code: int,
    random_seed: int,
    explicit_means: np.ndarray,
) -> np.ndarray:
    """Create deterministic model starting means for quantile/random/explicit modes."""

    rows, dimensions = matrix.shape
    if rows < components or components < 1:
        raise ValueError("insufficient rows for component initialization")
    means = np.empty((components, dimensions), dtype=np.float64)
    if strategy_code == 2:
        if explicit_means.shape[0] != components or explicit_means.shape[1] != dimensions:
            raise ValueError("explicit means shape mismatch")
        for component in range(components):
            for dimension in range(dimensions):
                value = explicit_means[component, dimension]
                if not np.isfinite(value):
                    raise ValueError("explicit means must be finite")
                means[component, dimension] = value
        return means
    if strategy_code == 1:
        # Park-Miller LCG gives a local, reproducible stream without touching
        # Numba's process-global RNG state. Duplicate draws are advanced to an
        # unused row so every component has a genuinely distinct starting row.
        state = random_seed % 2147483646
        if state < 0:
            state = -state
        state += 1
        selected = np.full(components, -1, dtype=np.int64)
        for component in range(components):
            state = (state * 48271) % 2147483647
            candidate = state % rows
            duplicate = True
            while duplicate:
                duplicate = False
                for prior in range(component):
                    if selected[prior] == candidate:
                        candidate = (candidate + 1) % rows
                        duplicate = True
                        break
            selected[component] = candidate
            for dimension in range(dimensions):
                means[component, dimension] = matrix[candidate, dimension]
        return means
    if strategy_code != 0:
        raise ValueError("unsupported initialization strategy")
    order = np.argsort(matrix[:, 0])
    for component in range(components):
        start = rows * component // components
        end = rows * (component + 1) // components
        count = max(end - start, 1)
        for dimension in range(dimensions):
            total = 0.0
            for offset in range(start, end):
                total += matrix[order[offset], dimension]
            means[component, dimension] = total / count
    return means


@njit(_GMM_FIT_RETURN(_F2, int64, int64, _F2), cache=True, nogil=True)
def gmm_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rows, dimensions = matrix.shape
    if initialized_means.shape[0] != components or initialized_means.shape[1] != dimensions:
        raise ValueError("initialized means shape mismatch")
    means = initialized_means.copy()
    global_variance = np.empty(dimensions, dtype=np.float64)
    for dimension in range(dimensions):
        total = 0.0
        for row in range(rows):
            total += matrix[row, dimension]
        mean = total / rows
        squared = 0.0
        for row in range(rows):
            difference = matrix[row, dimension] - mean
            squared += difference * difference
        global_variance[dimension] = max(squared / rows, 1e-4)
    variances = np.empty((components, dimensions), dtype=np.float64)
    for component in range(components):
        for dimension in range(dimensions):
            variances[component, dimension] = global_variance[dimension]
    weights = np.full(components, 1.0 / components, dtype=np.float64)
    prior_likelihood = -np.inf
    completed = 0
    for iteration in range(1, iterations + 1):
        completed = iteration
        log_joint = _gaussian_log_density_impl(matrix, means, variances)
        responsibilities = np.empty_like(log_joint)
        likelihood = 0.0
        for row in range(rows):
            for component in range(components):
                log_joint[row, component] += np.log(
                    max(weights[component], 1e-12)
                )
            normalizer = _logsumexp_vector(log_joint[row])
            likelihood += normalizer
            for component in range(components):
                responsibilities[row, component] = np.exp(
                    log_joint[row, component] - normalizer
                )
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for row in range(rows):
                effective[component] += responsibilities[row, component]
            effective[component] = max(effective[component], 1e-8)
            weights[component] = effective[component] / rows
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    total += (
                        responsibilities[row, component]
                        * matrix[row, dimension]
                    )
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for row in range(rows):
                    difference = (
                        matrix[row, dimension] - means[component, dimension]
                    )
                    total += (
                        responsibilities[row, component]
                        * difference
                        * difference
                    )
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < 1e-6:
            break
        prior_likelihood = likelihood
    return weights, means, variances, completed


@njit(_F2(_F2, _F1, _F2, _F2), cache=True, nogil=True)
def gmm_posterior_kernel(
    matrix: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    log_joint = _gaussian_log_density_impl(matrix, means, variances)
    posterior = np.empty_like(log_joint)
    for row in range(matrix.shape[0]):
        for component in range(weights.size):
            log_joint[row, component] += np.log(max(weights[component], 1e-12))
        normalizer = _logsumexp_vector(log_joint[row])
        for component in range(weights.size):
            posterior[row, component] = np.exp(
                log_joint[row, component] - normalizer
            )
    return posterior


_HMM_FIT_RETURN = types.Tuple((_F1, _F2, _F2, _F2, int64))


@njit(_HMM_FIT_RETURN(_F2, int64, int64, _F2), cache=True, nogil=True)
def markov_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Fit a hard-state Gaussian Markov-switching model.

    Unlike ``hmm_fit_kernel`` this does not run Baum-Welch.  Gaussian regimes
    are estimated first, then the transition matrix is estimated from the
    locked training assignments with Laplace smoothing.  This gives the
    Markov node distinct, auditable estimation semantics.
    """

    weights, means, variances, completed = gmm_fit_kernel(
        matrix,
        components,
        iterations,
        initialized_means,
    )
    posterior = gmm_posterior_kernel(matrix, weights, means, variances)
    assignments = np.empty(matrix.shape[0], dtype=np.int64)
    for row in range(matrix.shape[0]):
        winner = 0
        for component in range(1, components):
            if posterior[row, component] > posterior[row, winner]:
                winner = component
        assignments[row] = winner

    smoothing = 1e-3
    initial = np.full(components, smoothing, dtype=np.float64)
    initial[assignments[0]] += 1.0
    initial /= np.sum(initial)
    transition = np.full((components, components), smoothing, dtype=np.float64)
    for row in range(1, assignments.size):
        transition[assignments[row - 1], assignments[row]] += 1.0
    for source in range(components):
        total = np.sum(transition[source])
        transition[source] /= total
    return initial, transition, means, variances, completed


@njit(_HMM_FIT_RETURN(_F2, int64, int64, _F2), cache=True, nogil=True)
def hmm_fit_kernel(
    matrix: np.ndarray,
    components: int,
    iterations: int,
    initialized_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    weights, means, variances, _ = gmm_fit_kernel(
        matrix,
        components,
        min(iterations, 30),
        initialized_means,
    )
    initial = np.empty(components, dtype=np.float64)
    initial_total = 0.0
    for component in range(components):
        initial[component] = max(weights[component], 1e-8)
        initial_total += initial[component]
    initial /= initial_total
    transition = np.empty((components, components), dtype=np.float64)
    for row in range(components):
        row_total = 0.0
        for column in range(components):
            value = 0.9 if row == column and components > 1 else (
                1.0 if components == 1 else 0.1 / (components - 1)
            )
            transition[row, column] = value
            row_total += value
        for column in range(components):
            transition[row, column] /= row_total
    prior_likelihood = -np.inf
    completed = 0
    rows = matrix.shape[0]
    dimensions = matrix.shape[1]
    for iteration in range(1, iterations + 1):
        completed = iteration
        emissions = _gaussian_log_density_impl(matrix, means, variances)
        log_transition = np.empty_like(transition)
        for row in range(components):
            for column in range(components):
                log_transition[row, column] = np.log(
                    max(transition[row, column], 1e-12)
                )
        alpha = np.empty_like(emissions)
        for component in range(components):
            alpha[0, component] = (
                np.log(max(initial[component], 1e-12))
                + emissions[0, component]
            )
        workspace = np.empty(components, dtype=np.float64)
        for index in range(1, rows):
            for target in range(components):
                for source in range(components):
                    workspace[source] = (
                        alpha[index - 1, source]
                        + log_transition[source, target]
                    )
                alpha[index, target] = (
                    emissions[index, target]
                    + _logsumexp_vector(workspace)
                )
        beta = np.zeros_like(emissions)
        for index in range(rows - 2, -1, -1):
            for source in range(components):
                for target in range(components):
                    workspace[target] = (
                        log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                beta[index, source] = _logsumexp_vector(workspace)
        likelihood = _logsumexp_vector(alpha[rows - 1])
        gamma = np.empty_like(emissions)
        for index in range(rows):
            for component in range(components):
                workspace[component] = alpha[index, component] + beta[index, component]
            normalizer = _logsumexp_vector(workspace)
            for component in range(components):
                gamma[index, component] = np.exp(
                    alpha[index, component]
                    + beta[index, component]
                    - normalizer
                )
        xi_sum = np.zeros_like(transition)
        for index in range(rows - 1):
            xi_normalizer = -np.inf
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                    )
                    if xi_normalizer == -np.inf:
                        xi_normalizer = value
                    elif value > xi_normalizer:
                        xi_normalizer = value + np.log1p(
                            np.exp(xi_normalizer - value)
                        )
                    else:
                        xi_normalizer = xi_normalizer + np.log1p(
                            np.exp(value - xi_normalizer)
                        )
            for source in range(components):
                for target in range(components):
                    value = (
                        alpha[index, source]
                        + log_transition[source, target]
                        + emissions[index + 1, target]
                        + beta[index + 1, target]
                        - xi_normalizer
                    )
                    xi_sum[source, target] += np.exp(value)
        initial_total = 0.0
        for component in range(components):
            initial[component] = max(gamma[0, component], 1e-8)
            initial_total += initial[component]
        initial /= initial_total
        for source in range(components):
            row_total = 0.0
            for target in range(components):
                transition[source, target] = max(
                    xi_sum[source, target],
                    1e-8,
                )
                row_total += transition[source, target]
            for target in range(components):
                transition[source, target] /= row_total
        effective = np.zeros(components, dtype=np.float64)
        for component in range(components):
            for index in range(rows):
                effective[component] += gamma[index, component]
            effective[component] = max(effective[component], 1e-8)
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    total += gamma[index, component] * matrix[index, dimension]
                means[component, dimension] = total / effective[component]
        for component in range(components):
            for dimension in range(dimensions):
                total = 0.0
                for index in range(rows):
                    difference = matrix[index, dimension] - means[component, dimension]
                    total += gamma[index, component] * difference * difference
                variances[component, dimension] = max(
                    total / effective[component],
                    1e-5,
                )
        if abs(likelihood - prior_likelihood) < 1e-5:
            break
        prior_likelihood = likelihood
    return initial, transition, means, variances, completed


@njit(_F2(_F2, _F1, _F2, _F2, _F2), cache=True, nogil=True)
def hmm_smoothed_posterior_kernel(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    emissions = _gaussian_log_density_impl(observations, means, variances)
    rows = observations.shape[0]
    components = initial.size
    log_transition = np.empty_like(transition)
    for source in range(components):
        for target in range(components):
            log_transition[source, target] = np.log(
                max(transition[source, target], 1e-12)
            )
    alpha = np.empty_like(emissions)
    beta = np.zeros_like(emissions)
    workspace = np.empty(components, dtype=np.float64)
    for component in range(components):
        alpha[0, component] = (
            np.log(max(initial[component], 1e-12))
            + emissions[0, component]
        )
    for index in range(1, rows):
        for target in range(components):
            for source in range(components):
                workspace[source] = (
                    alpha[index - 1, source]
                    + log_transition[source, target]
                )
            alpha[index, target] = emissions[index, target] + _logsumexp_vector(
                workspace
            )
    for index in range(rows - 2, -1, -1):
        for source in range(components):
            for target in range(components):
                workspace[target] = (
                    log_transition[source, target]
                    + emissions[index + 1, target]
                    + beta[index + 1, target]
                )
            beta[index, source] = _logsumexp_vector(workspace)
    posterior = np.empty_like(emissions)
    for index in range(rows):
        for component in range(components):
            workspace[component] = alpha[index, component] + beta[index, component]
        normalizer = _logsumexp_vector(workspace)
        for component in range(components):
            posterior[index, component] = np.exp(
                alpha[index, component] + beta[index, component] - normalizer
            )
    return posterior


@njit(_F2(_F2, _F2, _F1, _F2, _F2, _F2), cache=True, nogil=True)
def hmm_filtered_posterior_kernel(
    training: np.ndarray,
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    components = initial.size
    previous = initial.copy()
    train_emissions = _gaussian_log_density_impl(training, means, variances)
    for index in range(training.shape[0]):
        current = np.empty(components, dtype=np.float64)
        maximum = np.max(train_emissions[index])
        if index == 0:
            for component in range(components):
                current[component] = previous[component] * np.exp(
                    train_emissions[index, component] - maximum
                )
        else:
            for target in range(components):
                predicted = 0.0
                for source in range(components):
                    predicted += previous[source] * transition[source, target]
                current[target] = predicted * np.exp(
                    train_emissions[index, target] - maximum
                )
        total = np.sum(current)
        previous = current / max(total, 1e-300)
    emissions = _gaussian_log_density_impl(observations, means, variances)
    posterior = np.empty((observations.shape[0], components), dtype=np.float64)
    for index in range(observations.shape[0]):
        maximum = np.max(emissions[index])
        current = np.empty(components, dtype=np.float64)
        for target in range(components):
            predicted = 0.0
            for source in range(components):
                predicted += previous[source] * transition[source, target]
            current[target] = predicted * np.exp(
                emissions[index, target] - maximum
            )
        total = np.sum(current)
        previous = current / max(total, 1e-300)
        posterior[index] = previous
    return posterior


@njit(_F2(_F2, _F1, _F2, _F2, _F2), cache=True, nogil=True)
def markov_smoothed_posterior_kernel(
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Retrospective posterior for the hard-state Markov estimator."""

    return hmm_smoothed_posterior_kernel(
        observations,
        initial,
        transition,
        means,
        variances,
    )


@njit(_F2(_F2, _F2, _F1, _F2, _F2, _F2), cache=True, nogil=True)
def markov_filtered_posterior_kernel(
    training: np.ndarray,
    observations: np.ndarray,
    initial: np.ndarray,
    transition: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    """Causal posterior for the hard-state Markov estimator."""

    return hmm_filtered_posterior_kernel(
        training,
        observations,
        initial,
        transition,
        means,
        variances,
    )


_ASSIGN_RETURN = types.Tuple((_I1, _F1))


@njit(_ASSIGN_RETURN(_F2), cache=True, nogil=True)
def posterior_assignment_kernel(
    posterior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rows, components = posterior.shape
    assignments = np.empty(rows, dtype=np.int64)
    confidence = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        winner = 0
        for component in range(1, components):
            if posterior[row, component] > posterior[row, winner]:
                winner = component
        assignments[row] = winner
        confidence[row] = posterior[row, winner]
    return assignments, confidence


@njit(_I1(_F2), cache=True, nogil=True)
def component_order_kernel(means: np.ndarray) -> np.ndarray:
    return np.argsort(means[:, 0]).astype(np.int64)


_CHANGE_RETURN = types.Tuple((_F1, _I1, _F1, _I1, _I1, _I1, _I1, _I1))


@njit(
    _CHANGE_RETURN(_F1, int64, float64, int64, boolean),
    cache=True,
    nogil=True,
)
def change_point_kernel(
    values: np.ndarray,
    window: int,
    threshold: float,
    confirmation: int,
    retrospective: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    size = values.size
    changes = np.full(size, np.nan, dtype=np.float64)
    for index in range(1, size):
        if np.isfinite(values[index]) and np.isfinite(values[index - 1]):
            changes[index] = values[index] - values[index - 1]
    score = np.full(size, np.nan, dtype=np.float64)
    for index in range(size):
        if retrospective:
            past_start = max(0, index - window + 1)
            future_end = min(size, index + window)
            past = _finite_mean(changes, past_start, index + 1, window)
            future = _finite_mean(changes, index, future_end, window)
            noise_start = max(0, index - window)
            noise_end = min(size, index + window)
            noise = _finite_std(changes, noise_start, noise_end, window)
        else:
            recent_start = max(0, index - window + 1)
            baseline_end = index - window + 1
            baseline_start = max(0, baseline_end - window)
            noise_start = max(0, index - 2 * window)
            past = _finite_mean(changes, recent_start, index + 1, window)
            future = _finite_mean(
                changes,
                baseline_start,
                max(baseline_end, baseline_start),
                window,
            )
            noise = _finite_std(changes, noise_start, index, window)
        if (
            np.isfinite(past)
            and np.isfinite(future)
            and np.isfinite(noise)
            and noise != 0.0
        ):
            score[index] = (future - past) / noise if retrospective else (
                past - future
            ) / noise
    desired = np.full(size, -1, dtype=np.int64)
    for index in range(size):
        value = score[index]
        if not np.isfinite(value):
            continue
        if value >= threshold:
            desired[index] = 0
        elif value <= -threshold:
            desired[index] = 2
        else:
            desired[index] = 1
    (
        labels,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    ) = _confirm_codes_impl(desired, 1, confirmation, 1)
    if retrospective:
        for index in range(size):
            recognition[index] = min(recognition[index] + window, size - 1)
    confidence = np.full(size, np.nan, dtype=np.float64)
    denominator = max(threshold, 1e-12)
    for index in range(size):
        if np.isfinite(score[index]):
            confidence[index] = min(1.0, abs(score[index]) / denominator)
    return (
        score,
        labels,
        confidence,
        recognition,
        reason_codes,
        pending_counts,
        transition_from,
        transition_to,
    )


_ENSEMBLE_RETURN = types.Tuple((_I1, _F2, _F1, _F1, _F1, _I1, int64))


@njit(
    _ENSEMBLE_RETURN(_I2, _F1, _F2, _F2, _I2, int64, float64),
    cache=True,
    nogil=True,
)
def ensemble_consensus_kernel(
    member_labels: np.ndarray,
    weights: np.ndarray,
    member_filtered: np.ndarray,
    member_scores: np.ndarray,
    member_recognition: np.ndarray,
    state_count: int,
    consensus_threshold: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    members, points = member_labels.shape
    total_weight = np.sum(weights)
    labels = np.full(points, -1, dtype=np.int64)
    probabilities = np.zeros((points, state_count), dtype=np.float64)
    confidence = np.full(points, np.nan, dtype=np.float64)
    filtered = np.full(points, np.nan, dtype=np.float64)
    scores = np.full(points, np.nan, dtype=np.float64)
    recognition = np.zeros(points, dtype=np.int64)
    conflicts = 0
    for point in range(points):
        for member in range(members):
            state = member_labels[member, point]
            if 0 <= state < state_count:
                probabilities[point, state] += weights[member]
        winner = 0
        for state in range(state_count):
            probabilities[point, state] /= total_weight
            if probabilities[point, state] > probabilities[point, winner]:
                winner = state
        winner_confidence = probabilities[point, winner]
        if winner_confidence >= consensus_threshold:
            labels[point] = winner
            confidence[point] = winner_confidence
        else:
            conflicts += 1
        filtered_total = 0.0
        filtered_weight = 0.0
        score_total = 0.0
        score_weight = 0.0
        maximum_recognition = 0
        for member in range(members):
            if np.isfinite(member_filtered[member, point]):
                filtered_total += weights[member] * member_filtered[member, point]
                filtered_weight += weights[member]
            if np.isfinite(member_scores[member, point]):
                score_total += weights[member] * member_scores[member, point]
                score_weight += weights[member]
            if member_recognition[member, point] > maximum_recognition:
                maximum_recognition = member_recognition[member, point]
        if filtered_weight > 0.0:
            filtered[point] = filtered_total / filtered_weight
        if score_weight > 0.0:
            scores[point] = score_total / score_weight
        recognition[point] = maximum_recognition
    return (
        labels,
        probabilities,
        confidence,
        filtered,
        scores,
        recognition,
        conflicts,
    )


@njit(_F1(_F1, int64), cache=True, nogil=True)
def path_metrics_kernel(values: np.ndarray, periods_per_year: int) -> np.ndarray:
    result = np.full(4, np.nan, dtype=np.float64)
    valid_count = 0
    for value in values:
        if np.isfinite(value):
            valid_count += 1
    if valid_count < 2:
        return result
    valid = np.empty(valid_count, dtype=np.float64)
    offset = 0
    for value in values:
        if np.isfinite(value):
            valid[offset] = value
            offset += 1
    if valid[0] != 0.0:
        total_return = valid[-1] / valid[0] - 1.0
        result[0] = total_return
        if total_return > -1.0:
            result[1] = (1.0 + total_return) ** (
                periods_per_year / max(valid_count - 1, 1)
            ) - 1.0
    return_count = valid_count - 1
    returns = np.empty(return_count, dtype=np.float64)
    finite_return_count = 0
    for index in range(return_count):
        if valid[index] != 0.0:
            value = valid[index + 1] / valid[index] - 1.0
            if np.isfinite(value):
                returns[finite_return_count] = value
                finite_return_count += 1
    if finite_return_count > 1:
        mean = 0.0
        for index in range(finite_return_count):
            mean += returns[index]
        mean /= finite_return_count
        squared = 0.0
        for index in range(finite_return_count):
            difference = returns[index] - mean
            squared += difference * difference
        result[2] = np.sqrt(squared / (finite_return_count - 1)) * np.sqrt(
            periods_per_year
        )
    all_positive = True
    peak = -np.inf
    drawdown = 0.0
    for value in valid:
        if value <= 0.0:
            all_positive = False
        if value > peak:
            peak = value
        current = value / peak - 1.0
        if current < drawdown:
            drawdown = current
    if all_positive:
        result[3] = drawdown
    return result


@njit(_F2(_F1, _I1, int64, int64), cache=True, nogil=True)
def conditional_statistics_kernel(
    values: np.ndarray,
    label_codes: np.ndarray,
    state_count: int,
    periods_per_year: int,
) -> np.ndarray:
    # observation_count, return_count, mean, annualized, volatility,
    # maximum_drawdown, sharpe, positive_rate
    output = np.full((state_count, 8), np.nan, dtype=np.float64)
    for state in range(state_count):
        observations = 0
        return_count = 0
        for index in range(values.size):
            if label_codes[index] == state:
                observations += 1
            if (
                index + 1 < values.size
                and label_codes[index] == state
                and np.isfinite(values[index])
                and np.isfinite(values[index + 1])
                and values[index] != 0.0
            ):
                return_count += 1
        output[state, 0] = observations
        output[state, 1] = return_count
        if return_count == 0:
            continue
        returns = np.empty(return_count, dtype=np.float64)
        offset = 0
        for index in range(values.size - 1):
            if (
                label_codes[index] == state
                and np.isfinite(values[index])
                and np.isfinite(values[index + 1])
                and values[index] != 0.0
            ):
                returns[offset] = values[index + 1] / values[index] - 1.0
                offset += 1
        mean = np.sum(returns) / return_count
        output[state, 2] = mean
        output[state, 3] = mean * periods_per_year
        positives = 0
        wealth = 1.0
        peak = 1.0
        maximum_drawdown = 0.0
        for value in returns:
            if value > 0.0:
                positives += 1
            wealth *= 1.0 + value
            if wealth > peak:
                peak = wealth
            drawdown = wealth / peak - 1.0
            if drawdown < maximum_drawdown:
                maximum_drawdown = drawdown
        output[state, 5] = maximum_drawdown
        output[state, 7] = positives / return_count
        if return_count > 1:
            squared = 0.0
            for value in returns:
                difference = value - mean
                squared += difference * difference
            volatility = np.sqrt(squared / (return_count - 1)) * np.sqrt(
                periods_per_year
            )
            output[state, 4] = volatility
            if volatility > 0.0:
                output[state, 6] = output[state, 3] / volatility
    return output


_TRANSITION_RETURN = types.Tuple((_I2, _F2))


@njit(_TRANSITION_RETURN(_I1, int64), cache=True, nogil=True)
def transition_matrix_kernel(
    label_codes: np.ndarray,
    state_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((state_count, state_count), dtype=np.int64)
    for index in range(1, label_codes.size):
        prior = label_codes[index - 1]
        current = label_codes[index]
        if 0 <= prior < state_count and 0 <= current < state_count:
            counts[prior, current] += 1
    probabilities = np.zeros((state_count, state_count), dtype=np.float64)
    for prior in range(state_count):
        total = 0
        for current in range(state_count):
            total += counts[prior, current]
        if total > 0:
            for current in range(state_count):
                probabilities[prior, current] = counts[prior, current] / total
    return counts, probabilities


@njit(_F1(_I1, _I1), cache=True, nogil=True)
def prefix_stability_kernel(
    base_codes: np.ndarray,
    prefix_codes: np.ndarray,
) -> np.ndarray:
    result = np.full(4, np.nan, dtype=np.float64)
    comparable = 0
    revisions = 0
    size = min(base_codes.size, prefix_codes.size)
    for index in range(size):
        if base_codes[index] >= 0 and prefix_codes[index] >= 0:
            comparable += 1
            if base_codes[index] != prefix_codes[index]:
                revisions += 1
    result[0] = comparable
    result[1] = revisions
    if comparable > 0:
        result[2] = (comparable - revisions) / comparable
        result[3] = revisions / comparable
    return result


@njit(int64(_I1), cache=True, nogil=True)
def label_flip_kernel(label_codes: np.ndarray) -> int:
    flips = 0
    prior = -1
    for code in label_codes:
        if code < 0:
            continue
        if prior >= 0 and code != prior:
            flips += 1
        prior = code
    return flips


@njit(float64(_F1), cache=True, nogil=True)
def finite_mean_kernel(values: np.ndarray) -> float:
    total = 0.0
    count = 0
    for value in values:
        if np.isfinite(value):
            total += value
            count += 1
    return total / count if count else np.nan


@njit(_I1(_I1, int64), cache=True, nogil=True)
def state_counts_kernel(label_codes: np.ndarray, state_count: int) -> np.ndarray:
    counts = np.zeros(state_count, dtype=np.int64)
    for code in label_codes:
        if 0 <= code < state_count:
            counts[code] += 1
    return counts


@njit(_F1(_I1), cache=True, nogil=True)
def label_summary_kernel(label_codes: np.ndarray) -> np.ndarray:
    output = np.full(4, np.nan, dtype=np.float64)
    classified = 0
    prior = -1
    flips = 0
    for code in label_codes:
        if code < 0:
            continue
        classified += 1
        if prior >= 0 and code != prior:
            flips += 1
        prior = code
    output[0] = classified
    output[1] = flips
    output[2] = classified / label_codes.size if label_codes.size else np.nan
    output[3] = flips / max(classified - 1, 1) if classified else np.nan
    return output


@njit(_F1(_I1, _I1), cache=True, nogil=True)
def comparison_pair_kernel(
    left_codes: np.ndarray,
    right_codes: np.ndarray,
) -> np.ndarray:
    output = np.full(4, np.nan, dtype=np.float64)
    size = min(left_codes.size, right_codes.size)
    usable_count = 0
    agreement_count = 0
    left_boundaries = np.empty(size, dtype=np.int64)
    right_boundaries = np.empty(size, dtype=np.int64)
    left_boundary_count = 0
    right_boundary_count = 0
    prior_left = -1
    prior_right = -1
    for index in range(size):
        left = left_codes[index]
        right = right_codes[index]
        if left < 0 or right < 0:
            continue
        if usable_count > 0:
            if left != prior_left:
                left_boundaries[left_boundary_count] = usable_count
                left_boundary_count += 1
            if right != prior_right:
                right_boundaries[right_boundary_count] = usable_count
                right_boundary_count += 1
        if left == right:
            agreement_count += 1
        prior_left = left
        prior_right = right
        usable_count += 1
    output[0] = usable_count
    output[1] = agreement_count
    if usable_count > 0:
        output[2] = agreement_count / usable_count
    if left_boundary_count == 0 and right_boundary_count == 0:
        output[3] = 0.0
    elif left_boundary_count > 0 and right_boundary_count > 0:
        total_distance = 0.0
        distance_count = 0
        for left_index in range(left_boundary_count):
            minimum = size + 1
            for right_index in range(right_boundary_count):
                distance = abs(
                    left_boundaries[left_index] - right_boundaries[right_index]
                )
                if distance < minimum:
                    minimum = distance
            total_distance += minimum
            distance_count += 1
        for right_index in range(right_boundary_count):
            minimum = size + 1
            for left_index in range(left_boundary_count):
                distance = abs(
                    right_boundaries[right_index] - left_boundaries[left_index]
                )
                if distance < minimum:
                    minimum = distance
            total_distance += minimum
            distance_count += 1
        output[3] = total_distance / distance_count
    return output


@njit(_VALIDATION_WINDOWS_RESULT(int64, int64), cache=True, nogil=True)
def validation_windows_kernel(
    observation_count: int,
    fold_count: int,
) -> tuple[int, np.ndarray]:
    """Build the stability prefix and expanding fold indexes in fixed NJIT."""

    prefix_length = max(5, (observation_count * 3) // 4)
    if observation_count < 0 or fold_count <= 0:
        return prefix_length, np.empty((0, 6), dtype=np.int64)
    initial = max(10, observation_count // (fold_count + 1))
    boundaries = np.empty(fold_count + 1, dtype=np.int64)
    for index in range(fold_count + 1):
        boundaries[index] = (
            initial * (fold_count - index) + observation_count * index
        ) // fold_count
    windows = np.empty((fold_count, 6), dtype=np.int64)
    for fold in range(fold_count):
        train_end = boundaries[fold]
        test_end = boundaries[fold + 1]
        windows[fold, 0] = train_end
        windows[fold, 1] = test_end
        windows[fold, 2] = train_end - 1
        windows[fold, 3] = train_end
        windows[fold, 4] = test_end - 1
        windows[fold, 5] = test_end - train_end
    return prefix_length, windows


@njit(int64(_I1), cache=True, nogil=True)
def integer_sum_kernel(values: np.ndarray) -> int:
    total = 0
    for value in values:
        total += value
    return total


@njit(int64(int64), cache=True, nogil=True)
def predecessor_count_kernel(count: int) -> int:
    return count - 1 if count > 0 else 0


@njit(float64(float64, float64, float64, uint8), cache=True, nogil=True)
def perturb_scalar_kernel(
    value: float,
    perturbation: float,
    minimum: float,
    round_to_integer: int,
) -> float:
    result = value * (1.0 + perturbation)
    if round_to_integer != 0:
        result = np.rint(result)
    return minimum if result < minimum else result


@njit(float64(int64, int64), cache=True, nogil=True)
def conflict_rate_kernel(
    conflict_count: int,
    observation_count: int,
) -> float:
    """Calculate the ensemble rejection rate without Python scalar math."""

    if observation_count <= 0:
        return np.nan
    return conflict_count / observation_count


@njit(_U1(_I2), cache=True, nogil=True)
def row_disagreement_kernel(state_codes: np.ndarray) -> np.ndarray:
    """Mark dates on which at least two classified runs disagree."""

    result = np.zeros(state_codes.shape[0], dtype=np.uint8)
    for row in range(state_codes.shape[0]):
        first = -1
        for column in range(state_codes.shape[1]):
            code = state_codes[row, column]
            if code < 0:
                continue
            if first < 0:
                first = code
            elif code != first:
                result[row] = 1
                break
    return result


_DISPATCHERS = {
    "relative_transform": relative_transform_kernel,
    "formula_input_blocks": formula_input_blocks_kernel,
    "formula_align_block": formula_align_block_kernel,
    "formula_missing_numeric_mask": formula_missing_numeric_mask_kernel,
    "feature_pipeline": feature_pipeline_kernel,
    "trend_state": trend_state_kernel,
    "turning_point": turning_point_kernel,
    "merrill_clock": merrill_clock_kernel,
    "valid_rows": valid_rows_kernel,
    "standardize_fit": standardize_fit_kernel,
    "apply_standardization": apply_standardization_kernel,
    "initialize_gaussian_means": initialize_gaussian_means_kernel,
    "gmm_fit": gmm_fit_kernel,
    "gmm_posterior": gmm_posterior_kernel,
    "hmm_fit": hmm_fit_kernel,
    "hmm_smoothed_posterior": hmm_smoothed_posterior_kernel,
    "hmm_filtered_posterior": hmm_filtered_posterior_kernel,
    "markov_fit": markov_fit_kernel,
    "markov_smoothed_posterior": markov_smoothed_posterior_kernel,
    "markov_filtered_posterior": markov_filtered_posterior_kernel,
    "posterior_assignment": posterior_assignment_kernel,
    "component_order": component_order_kernel,
    "change_point": change_point_kernel,
    "ensemble_consensus": ensemble_consensus_kernel,
    "path_metrics": path_metrics_kernel,
    "conditional_statistics": conditional_statistics_kernel,
    "transition_matrix": transition_matrix_kernel,
    "prefix_stability": prefix_stability_kernel,
    "label_flip": label_flip_kernel,
    "finite_mean": finite_mean_kernel,
    "state_counts": state_counts_kernel,
    "label_summary": label_summary_kernel,
    "comparison_pair": comparison_pair_kernel,
    "validation_windows": validation_windows_kernel,
    "integer_sum": integer_sum_kernel,
    "predecessor_count": predecessor_count_kernel,
    "perturb_scalar": perturb_scalar_kernel,
    "conflict_rate": conflict_rate_kernel,
    "row_disagreement": row_disagreement_kernel,
}

for _dispatcher in _DISPATCHERS.values():
    _dispatcher.disable_compile()
_RUNTIME_LOCK = threading.RLock()
_RUNTIME_STATUS: dict[str, Any] = {
    "complete": False,
    "engine_version": ENGINE_VERSION,
    "kernel_version": KERNEL_VERSION,
    "python_fallback": 0,
}


def _fingerprint(dispatcher: Any) -> str:
    source = inspect.getsource(dispatcher.py_func).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _metadata(kernel_id: str) -> dict[str, Any]:
    dispatcher = _DISPATCHERS[kernel_id]
    signatures = [str(signature) for signature in dispatcher.nopython_signatures]
    return {
        "kernel_id": kernel_id,
        "compiled_plan_id": f"historical-regime:{KERNEL_VERSION}:{kernel_id}",
        "compile_status": "compiled" if signatures else "not_compiled",
        "compiled_signatures": signatures,
        "kernel_fingerprint": _fingerprint(dispatcher),
        "kernel_version": KERNEL_VERSION,
        "engine_version": ENGINE_VERSION,
        "execution_backend": "numba_njit_fixed_signature",
        "njit_required": True,
        "python_fallback": 0,
        "python_operator_calls": 0,
    }


def historical_regime_numba_execution_audit() -> dict[str, Any]:
    """Build the startup policy proof for the complete numerical lane."""

    signatures = {
        kernel_id: [str(signature) for signature in dispatcher.nopython_signatures]
        for kernel_id, dispatcher in sorted(_DISPATCHERS.items())
    }
    fingerprint_material = "|".join(
        [ENGINE_VERSION, KERNEL_VERSION]
        + [
            f"{kernel_id}:{','.join(kernel_signatures)}:{_fingerprint(_DISPATCHERS[kernel_id])}"
            for kernel_id, kernel_signatures in signatures.items()
        ]
    )
    return {
        "engine": ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(
            fingerprint_material.encode("utf-8")
        ).hexdigest(),
        "nopython": all(
            bool(dispatcher.nopython_signatures)
            and all(
                not compilation.objectmode
                for compilation in dispatcher.overloads.values()
            )
            for dispatcher in _DISPATCHERS.values()
        ),
        "object_mode": 0,
        "request_time_compilation": 0,
        "python_fallback": 0,
    }


def warm_historical_regime_numba_kernels() -> dict[str, Any]:
    """Verify all fixed signatures before a service may accept requests."""

    try:
        from backend.compute_policy import validate_execution_audit
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from compute_policy import validate_execution_audit

    with _RUNTIME_LOCK:
        kernels = [_metadata(kernel_id) for kernel_id in sorted(_DISPATCHERS)]
        incomplete = [
            item["kernel_id"]
            for item in kernels
            if item["compile_status"] != "compiled"
        ]
        if incomplete:
            raise RuntimeError(
                "Historical-regime NJIT warmup incomplete: "
                + ", ".join(incomplete)
            )
        policy_audit = validate_execution_audit(
            historical_regime_numba_execution_audit()
        )
        _RUNTIME_STATUS.clear()
        _RUNTIME_STATUS.update(
            {
                **policy_audit,
                "complete": True,
                "engine_version": ENGINE_VERSION,
                "kernel_version": KERNEL_VERSION,
                "kernel_count": len(kernels),
                "kernels": kernels,
                "python_fallback": 0,
                "python_operator_calls": 0,
            }
        )
        return historical_regime_numba_status()


# Compatibility alias for callers that used the shorter provisional name.
warm_historical_regime_kernels = warm_historical_regime_numba_kernels


def historical_regime_numba_status() -> dict[str, Any]:
    with _RUNTIME_LOCK:
        status = dict(_RUNTIME_STATUS)
        if "kernels" in status:
            status["kernels"] = [dict(item) for item in status["kernels"]]
        return status


def require_historical_regime_kernels_ready() -> None:
    if not historical_regime_numba_status().get("complete"):
        raise RuntimeError(
            "Historical-regime NJIT kernels were not warmed before request handling"
        )


def execution_audit(family: str, kernel_ids: list[str]) -> dict[str, Any]:
    require_historical_regime_kernels_ready()
    kernels = [_metadata(kernel_id) for kernel_id in kernel_ids]
    fingerprint_material = "|".join(
        f"{item['kernel_id']}:{item['kernel_fingerprint']}"
        for item in kernels
    )
    return {
        "source_kind": "historical_regime_algorithm",
        "family": family,
        "compiled_plan_id": (
            f"historical-regime:{KERNEL_VERSION}:{family}"
        ),
        "compile_status": "compiled",
        "kernel_version": KERNEL_VERSION,
        "engine_version": ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "execution_backend": "numba_njit_fixed_signature",
        "kernel_signatures": {
            item["kernel_id"]: list(item["compiled_signatures"])
            for item in kernels
        },
        "kernel_fingerprint": hashlib.sha256(
            fingerprint_material.encode("utf-8")
        ).hexdigest(),
        "nopython": all(bool(item["compiled_signatures"]) for item in kernels),
        "njit_required": True,
        "object_mode": 0,
        "request_time_compilation": 0,
        "python_fallback": 0,
        "python_operator_calls": 0,
        "kernels": kernels,
    }
