"""Fixed-signature Numba kernels for the typed indicator operator registry.

The public DSL is intentionally richer than a single Python callable: an
operator can accept scalars, one-dimensional arrays and matrices.  This module
keeps that polymorphism explicit.  Every operator owns an opcode and resolves
to one of a small number of fixed-layout Numba dispatchers.  Formula plans use
the same resolver, so operator execution never calls the Python reference
implementation stored in :mod:`typed_operators`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numba
import numpy as np
from numba import njit, types
from numba.core.registry import CPUDispatcher


NUMERIC_KERNEL_VERSION = "2.2.0"
ENGINE_VERSION = "typed-numba-3"

STATUS_OK = 0
STATUS_INSUFFICIENT_SAMPLE = 1
STATUS_DIVIDE_BY_ZERO = 2
STATUS_DOMAIN_ERROR = 3
STATUS_NON_FINITE_RESULT = 4
STATUS_INVALID_PARAMETER = 5
STATUS_SINGULAR_MATRIX = 6


BASIC_OPCODES = {
    "add": 1,
    "subtract": 2,
    "multiply": 3,
    "divide": 4,
    "power": 5,
    "minimum": 6,
    "maximum": 7,
}
UNARY_OPCODES = {
    "negate": 20,
    "absolute": 21,
    "sqrt": 22,
    "log": 23,
    "exp": 24,
    "reciprocal": 25,
    "sign": 26,
    "normal_pdf": 27,
    "normal_ppf": 28,
}
COMPARISON_OPCODES = {
    "equal": 40,
    "not_equal": 41,
    "less_than": 42,
    "less_equal": 43,
    "greater_than": 44,
    "greater_equal": 45,
}
REDUCTION_OPCODES = {
    "sum": 60,
    "product": 61,
    "mean": 62,
    "min_value": 63,
    "max_value": 64,
    "variance": 65,
    "std": 66,
    "median": 67,
    "skewness": 68,
    "excess_kurtosis": 69,
    "mean_absolute_deviation": 70,
    "root_mean_square": 71,
    "argmin": 72,
    "argmax": 73,
}
SCAN_OPCODES = {
    "cumulative_sum": 80,
    "cumulative_product": 81,
    "cumulative_max": 83,
    "cumulative_min": 84,
}
MASK_REDUCTION_OPCODES = {
    "sum_where": 100,
    "mean_where": 101,
    "variance_where": 102,
    "std_where": 103,
    "min_where": 104,
    "max_where": 105,
    "median_where": 106,
}
AXIS_REDUCTION_OPCODES = {
    "sum": 120,
    "mean": 121,
    "product": 122,
    "variance": 123,
    "std": 124,
    "min": 125,
    "max": 126,
}

CANONICAL_OPERATOR_IDS = (
    "add", "subtract", "multiply", "divide", "power", "minimum", "maximum",
    "negate", "absolute", "sqrt", "clip", "log", "exp", "reciprocal", "sign",
    "equal", "not_equal", "less_than", "less_equal", "greater_than", "greater_equal",
    "logical_and", "logical_or", "logical_not", "where", "sum_where", "mean_where",
    "variance_where", "std_where", "min_where", "max_where", "median_where",
    "quantile_where", "count_true", "max_consecutive_true", "sum", "product", "mean",
    "min_value", "max_value", "variance", "std", "cumulative_sum",
    "cumulative_product", "cumulative_return", "cumulative_max", "cumulative_min",
    "drawdown_series", "new_high_mask", "rolling_mean", "rolling_std", "rolling_min",
    "rolling_max", "recursive_smooth", "divide_or_default",
    "first", "length", "lag", "difference", "median", "skewness", "excess_kurtosis",
    "mean_absolute_deviation", "root_mean_square", "argmin", "argmax", "quantile",
    "linear_slope", "linear_intercept", "linear_r_squared", "regression_standard_error",
    "normal_pdf", "normal_ppf", "last", "total_return", "annualized_return", "sum_time",
    "mean_time", "product_time", "variance_time", "std_time", "min_time", "max_time",
    "sum_asset", "mean_asset", "product_asset", "variance_asset", "std_asset", "min_asset",
    "max_asset", "transpose", "dot", "outer", "matmul", "matvec", "diag", "trace",
    "solve", "covariance", "correlation", "portfolio_returns", "quadratic_form",
    "active_returns", "last_drawdown_interval",
    "interval_start", "interval_trough", "interval_recovery", "value_at", "days_between",
    "require_positive", "require_nonnegative", "linear_fit", "fit_slope", "fit_intercept",
    "fit_residual_sum_squares", "fit_total_sum_squares", "fit_observation_count", "finite_mask",
)


def _raise(code: str) -> None:
    raise ValueError(code)


@njit(cache=True, nogil=True, inline="always")
def _binary_value(opcode: int, lhs: float, rhs: float) -> float:
    if opcode == 1:
        return lhs + rhs
    if opcode == 2:
        return lhs - rhs
    if opcode == 3:
        return lhs * rhs
    if opcode == 4:
        if rhs == 0.0:
            raise ValueError("DIVIDE_BY_ZERO")
        return lhs / rhs
    if opcode == 5:
        value = lhs**rhs
        if not math.isfinite(value):
            raise ValueError("DOMAIN_ERROR")
        return value
    if opcode == 6:
        return min(lhs, rhs)
    return max(lhs, rhs)


@njit(cache=True, nogil=True)
def binary_scalar(opcode: int, lhs: float, rhs: float) -> float:
    return _binary_value(opcode, lhs, rhs)


@njit(cache=True, nogil=True)
def binary_1d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    # Validate throwing scalar operations before allocating: an exceptional
    # nopython exit must not leak one result buffer per rolling window.
    if opcode in (4, 5):
        for index in range(lhs.size):
            _binary_value(opcode, lhs[index], rhs[index])
    result = np.empty(lhs.size, dtype=np.float64)
    for index in range(lhs.size):
        result[index] = _binary_value(opcode, lhs[index], rhs[index])
    return result


@njit(cache=True, nogil=True)
def binary_1d_right_scalar(opcode: int, lhs: np.ndarray, rhs: float) -> np.ndarray:
    if opcode in (4, 5):
        for value in lhs:
            _binary_value(opcode, value, rhs)
    result = np.empty(lhs.size, dtype=np.float64)
    for index in range(lhs.size):
        result[index] = _binary_value(opcode, lhs[index], rhs)
    return result


@njit(cache=True, nogil=True)
def binary_1d_left_scalar(opcode: int, lhs: float, rhs: np.ndarray) -> np.ndarray:
    if opcode in (4, 5):
        for value in rhs:
            _binary_value(opcode, lhs, value)
    result = np.empty(rhs.size, dtype=np.float64)
    for index in range(rhs.size):
        result[index] = _binary_value(opcode, lhs, rhs[index])
    return result


@njit(cache=True, nogil=True)
def binary_2d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.float64)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _binary_value(
                opcode,
                lhs[row, column],
                rhs[row, column],
            )
    return result


@njit(cache=True, nogil=True)
def binary_2d_right_scalar(opcode: int, lhs: np.ndarray, rhs: float) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.float64)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _binary_value(opcode, lhs[row, column], rhs)
    return result


@njit(cache=True, nogil=True)
def binary_2d_left_scalar(opcode: int, lhs: float, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(rhs.shape, dtype=np.float64)
    for row in range(rhs.shape[0]):
        for column in range(rhs.shape[1]):
            result[row, column] = _binary_value(opcode, lhs, rhs[row, column])
    return result


@njit(cache=True, nogil=True, inline="always")
def _series_safe_divide_value(lhs: float, rhs: float) -> float:
    """Return NaN for one undefined series ratio without aborting the plan."""

    if not math.isfinite(lhs) or not math.isfinite(rhs) or abs(rhs) < 1e-12:
        return np.nan
    return lhs / rhs


@njit(cache=True, nogil=True)
def series_safe_divide_1d(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.size != rhs.size:
        raise ValueError("SHAPE_MISMATCH")
    result = np.empty(lhs.size, dtype=np.float64)
    for index in range(lhs.size):
        result[index] = _series_safe_divide_value(lhs[index], rhs[index])
    return result


@njit(cache=True, nogil=True)
def series_safe_divide_1d_right_scalar(
    lhs: np.ndarray,
    rhs: float,
) -> np.ndarray:
    result = np.empty(lhs.size, dtype=np.float64)
    for index in range(lhs.size):
        result[index] = _series_safe_divide_value(lhs[index], rhs)
    return result


@njit(cache=True, nogil=True)
def series_safe_divide_1d_left_scalar(
    lhs: float,
    rhs: np.ndarray,
) -> np.ndarray:
    result = np.empty(rhs.size, dtype=np.float64)
    for index in range(rhs.size):
        result[index] = _series_safe_divide_value(lhs, rhs[index])
    return result


@njit(cache=True, nogil=True)
def series_safe_divide_2d(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    if lhs.shape != rhs.shape:
        raise ValueError("SHAPE_MISMATCH")
    result = np.empty(lhs.shape, dtype=np.float64)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _series_safe_divide_value(
                lhs[row, column],
                rhs[row, column],
            )
    return result


@njit(cache=True, nogil=True)
def series_safe_divide_2d_right_scalar(
    lhs: np.ndarray,
    rhs: float,
) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.float64)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _series_safe_divide_value(
                lhs[row, column],
                rhs,
            )
    return result


@njit(cache=True, nogil=True)
def series_safe_divide_2d_left_scalar(
    lhs: float,
    rhs: np.ndarray,
) -> np.ndarray:
    result = np.empty(rhs.shape, dtype=np.float64)
    for row in range(rhs.shape[0]):
        for column in range(rhs.shape[1]):
            result[row, column] = _series_safe_divide_value(
                lhs,
                rhs[row, column],
            )
    return result


@njit(cache=True, nogil=True, inline="always")
def _normal_ppf_value(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("DOMAIN_ERROR")
    # Acklam's rational approximation.  Coefficients are fixed constants and
    # the maximum absolute error is well below indicator display precision.
    a0, a1, a2 = -3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02
    a3, a4, a5 = 1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00
    b0, b1, b2 = -5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02
    b3, b4 = 6.680131188771972e01, -1.328068155288572e01
    c0, c1, c2 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00
    c3, c4, c5 = -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00
    d0, d1, d2, d3 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00
    lower = 0.02425
    upper = 1.0 - lower
    if probability < lower:
        q = math.sqrt(-2.0 * math.log(probability))
        return (((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    if probability > upper:
        q = math.sqrt(-2.0 * math.log(1.0-probability))
        return -(((((c0*q+c1)*q+c2)*q+c3)*q+c4)*q+c5) / ((((d0*q+d1)*q+d2)*q+d3)*q+1.0)
    q = probability - 0.5
    r = q*q
    return (((((a0*r+a1)*r+a2)*r+a3)*r+a4)*r+a5)*q / (((((b0*r+b1)*r+b2)*r+b3)*r+b4)*r+1.0)


@njit(cache=True, nogil=True, inline="always")
def _unary_value(opcode: int, value: float) -> float:
    if opcode == 20:
        return -value
    if opcode == 21:
        return abs(value)
    if opcode == 22:
        if value < 0.0:
            raise ValueError("DOMAIN_ERROR")
        return math.sqrt(value)
    if opcode == 23:
        if value <= 0.0:
            raise ValueError("DOMAIN_ERROR")
        return math.log(value)
    if opcode == 24:
        result = math.exp(value)
        if not math.isfinite(result):
            raise ValueError("DOMAIN_ERROR")
        return result
    if opcode == 25:
        if value == 0.0:
            raise ValueError("DIVIDE_BY_ZERO")
        return 1.0 / value
    if opcode == 26:
        if value > 0.0:
            return 1.0
        if value < 0.0:
            return -1.0
        return 0.0
    if opcode == 27:
        return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)
    return _normal_ppf_value(value)


@njit(cache=True, nogil=True)
def unary_scalar(opcode: int, value: float) -> float:
    return _unary_value(opcode, value)


@njit(cache=True, nogil=True)
def unary_1d(opcode: int, values: np.ndarray) -> np.ndarray:
    if opcode not in (20, 21, 26, 27):
        for value in values:
            _unary_value(opcode, value)
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        result[index] = _unary_value(opcode, values[index])
    return result


@njit(cache=True, nogil=True)
def unary_2d(opcode: int, values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape, dtype=np.float64)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[row, column] = _unary_value(opcode, values[row, column])
    return result


@njit(cache=True, nogil=True)
def clip_scalar(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("INVALID_PARAMETER")
    return min(max(value, lower), upper)


@njit(cache=True, nogil=True)
def clip_1d(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if lower > upper:
        raise ValueError("INVALID_PARAMETER")
    result = np.empty(values.size, dtype=np.float64)
    for index in range(values.size):
        result[index] = min(max(values[index], lower), upper)
    return result


@njit(cache=True, nogil=True)
def clip_2d(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    if lower > upper:
        raise ValueError("INVALID_PARAMETER")
    result = np.empty(values.shape, dtype=np.float64)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[row, column] = min(max(values[row, column], lower), upper)
    return result


@njit(cache=True, nogil=True, inline="always")
def _comparison_value(opcode: int, lhs: float, rhs: float) -> np.uint8:
    if opcode == 40:
        return np.uint8(lhs == rhs)
    if opcode == 41:
        return np.uint8(lhs != rhs)
    if opcode == 42:
        return np.uint8(lhs < rhs)
    if opcode == 43:
        return np.uint8(lhs <= rhs)
    if opcode == 44:
        return np.uint8(lhs > rhs)
    return np.uint8(lhs >= rhs)


@njit(cache=True, nogil=True)
def comparison_scalar(opcode: int, lhs: float, rhs: float) -> np.uint8:
    return _comparison_value(opcode, lhs, rhs)


@njit(cache=True, nogil=True)
def comparison_1d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(lhs.size, dtype=np.uint8)
    for index in range(lhs.size):
        result[index] = _comparison_value(opcode, lhs[index], rhs[index])
    return result


@njit(cache=True, nogil=True)
def comparison_1d_right_scalar(opcode: int, lhs: np.ndarray, rhs: float) -> np.ndarray:
    result = np.empty(lhs.size, dtype=np.uint8)
    for index in range(lhs.size):
        result[index] = _comparison_value(opcode, lhs[index], rhs)
    return result


@njit(cache=True, nogil=True)
def comparison_1d_left_scalar(opcode: int, lhs: float, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(rhs.size, dtype=np.uint8)
    for index in range(rhs.size):
        result[index] = _comparison_value(opcode, lhs, rhs[index])
    return result


@njit(cache=True, nogil=True)
def comparison_2d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.uint8)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _comparison_value(opcode, lhs[row, column], rhs[row, column])
    return result


@njit(cache=True, nogil=True)
def comparison_2d_right_scalar(opcode: int, lhs: np.ndarray, rhs: float) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.uint8)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = _comparison_value(opcode, lhs[row, column], rhs)
    return result


@njit(cache=True, nogil=True)
def comparison_2d_left_scalar(opcode: int, lhs: float, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(rhs.shape, dtype=np.uint8)
    for row in range(rhs.shape[0]):
        for column in range(rhs.shape[1]):
            result[row, column] = _comparison_value(opcode, lhs, rhs[row, column])
    return result


@njit(cache=True, nogil=True)
def logical_scalar(opcode: int, lhs: np.uint8, rhs: np.uint8) -> np.uint8:
    return np.uint8((lhs != 0 and rhs != 0) if opcode == 1 else (lhs != 0 or rhs != 0))


@njit(cache=True, nogil=True)
def logical_1d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(lhs.size, dtype=np.uint8)
    for index in range(lhs.size):
        result[index] = np.uint8((lhs[index] != 0 and rhs[index] != 0) if opcode == 1 else (lhs[index] != 0 or rhs[index] != 0))
    return result


@njit(cache=True, nogil=True)
def logical_2d(opcode: int, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    result = np.empty(lhs.shape, dtype=np.uint8)
    for row in range(lhs.shape[0]):
        for column in range(lhs.shape[1]):
            result[row, column] = np.uint8((lhs[row, column] != 0 and rhs[row, column] != 0) if opcode == 1 else (lhs[row, column] != 0 or rhs[row, column] != 0))
    return result


@njit(cache=True, nogil=True)
def logical_not_scalar(value: np.uint8) -> np.uint8:
    return np.uint8(value == 0)


@njit(cache=True, nogil=True)
def logical_not_1d(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.size, dtype=np.uint8)
    for index in range(values.size):
        result[index] = np.uint8(values[index] == 0)
    return result


@njit(cache=True, nogil=True)
def logical_not_2d(values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape, dtype=np.uint8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            result[row, column] = np.uint8(values[row, column] == 0)
    return result


@njit(cache=True, nogil=True)
def where_scalar(mask: np.uint8, when_true: float, when_false: float) -> float:
    return when_true if mask != 0 else when_false


@njit(cache=True, nogil=True)
def where_1d(mask: np.ndarray, when_true: np.ndarray, when_false: np.ndarray) -> np.ndarray:
    result = np.empty(mask.size, dtype=np.float64)
    for index in range(mask.size):
        result[index] = when_true[index] if mask[index] != 0 else when_false[index]
    return result


@njit(cache=True, nogil=True)
def where_1d_false_scalar(mask: np.ndarray, when_true: np.ndarray, when_false: float) -> np.ndarray:
    result = np.empty(mask.size, dtype=np.float64)
    for index in range(mask.size):
        result[index] = when_true[index] if mask[index] != 0 else when_false
    return result


@njit(cache=True, nogil=True)
def where_1d_true_scalar(mask: np.ndarray, when_true: float, when_false: np.ndarray) -> np.ndarray:
    result = np.empty(mask.size, dtype=np.float64)
    for index in range(mask.size):
        result[index] = when_true if mask[index] != 0 else when_false[index]
    return result


@njit(cache=True, nogil=True)
def where_2d(mask: np.ndarray, when_true: np.ndarray, when_false: np.ndarray) -> np.ndarray:
    result = np.empty(mask.shape, dtype=np.float64)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            result[row, column] = when_true[row, column] if mask[row, column] != 0 else when_false[row, column]
    return result


@njit(cache=True, nogil=True)
def where_2d_false_scalar(mask: np.ndarray, when_true: np.ndarray, when_false: float) -> np.ndarray:
    result = np.empty(mask.shape, dtype=np.float64)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            result[row, column] = when_true[row, column] if mask[row, column] != 0 else when_false
    return result


@njit(cache=True, nogil=True)
def where_2d_true_scalar(mask: np.ndarray, when_true: float, when_false: np.ndarray) -> np.ndarray:
    result = np.empty(mask.shape, dtype=np.float64)
    for row in range(mask.shape[0]):
        for column in range(mask.shape[1]):
            result[row, column] = when_true if mask[row, column] != 0 else when_false[row, column]
    return result


@njit(cache=True, nogil=True, inline="always")
def _validate_nonempty(size: int) -> None:
    if size <= 0:
        raise ValueError("INSUFFICIENT_SAMPLE")


@njit(cache=True, nogil=True)
def reduce_1d(opcode: int, values: np.ndarray, ddof: int) -> float:
    count = values.size
    _validate_nonempty(count)
    if opcode in (65, 66) and (ddof < 0 or ddof >= count):
        raise ValueError("INVALID_PARAMETER")
    if opcode in (63, 64, 72, 73):
        best_index = 0
        best = values[0]
        for index in range(1, count):
            if (opcode in (63, 72) and values[index] < best) or (opcode in (64, 73) and values[index] > best):
                best = values[index]
                best_index = index
        if opcode in (72, 73):
            return float(best_index)
        return best
    if opcode == 61:
        total = 1.0
        for value in values:
            total *= value
        return total
    mean = 0.0
    m2 = 0.0
    total = 0.0
    absolute_total = 0.0
    square_total = 0.0
    for index in range(count):
        value = values[index]
        total += value
        square_total += value * value
        delta = value - mean
        mean += delta / (index + 1)
        m2 += delta * (value - mean)
    if opcode == 60:
        return total
    if opcode == 62:
        return mean
    if opcode == 65:
        return m2 / (count - ddof)
    if opcode == 66:
        return math.sqrt(m2 / (count - ddof))
    if opcode == 70:
        for value in values:
            absolute_total += abs(value - mean)
        return absolute_total / count
    if opcode == 71:
        return math.sqrt(square_total / count)
    if opcode == 68:
        if count < 3 or m2 <= 0.0:
            raise ValueError("INSUFFICIENT_SAMPLE")
        third = 0.0
        for value in values:
            third += (value - mean) ** 3
        population_second = m2 / count
        population_third = third / count
        return math.sqrt(count * (count - 1.0)) / (count - 2.0) * population_third / population_second**1.5
    if opcode == 69:
        if count < 4 or m2 <= 0.0:
            raise ValueError("INSUFFICIENT_SAMPLE")
        fourth = 0.0
        for value in values:
            fourth += (value - mean) ** 4
        second = m2 / count
        population_excess = fourth / count / (second * second) - 3.0
        return (count - 1.0) / ((count - 2.0) * (count - 3.0)) * ((count + 1.0) * population_excess + 6.0)
    # Median uses the same interpolation convention as NumPy.
    ordered = np.sort(values.copy())
    middle = count // 2
    return ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) * 0.5


@njit(cache=True, nogil=True)
def reduce_2d(opcode: int, values: np.ndarray, ddof: int) -> float:
    flat = values.reshape(values.size)
    return reduce_1d(opcode, flat, ddof)


@njit(cache=True, nogil=True)
def reduce_1d_parameter(values: np.ndarray, ddof: float, opcode: int) -> float:
    integer = int(ddof)
    if not math.isfinite(ddof) or float(integer) != ddof:
        raise ValueError("INVALID_PARAMETER")
    return reduce_1d(opcode, values, integer)


@njit(cache=True, nogil=True)
def reduce_2d_parameter(values: np.ndarray, ddof: float, opcode: int) -> float:
    integer = int(ddof)
    if not math.isfinite(ddof) or float(integer) != ddof:
        raise ValueError("INVALID_PARAMETER")
    return reduce_2d(opcode, values, integer)


@njit(cache=True, nogil=True)
def quantile_1d(values: np.ndarray, probability: float) -> float:
    _validate_nonempty(values.size)
    if probability <= 0.0 or probability >= 1.0:
        raise ValueError("INVALID_PARAMETER")
    ordered = values.copy()
    ordered.sort()
    position = (ordered.size - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@njit(cache=True, nogil=True)
def quantile_2d(values: np.ndarray, probability: float) -> float:
    return quantile_1d(values.reshape(values.size), probability)


@njit(cache=True, nogil=True)
def scan_1d(opcode: int, values: np.ndarray) -> np.ndarray:
    _validate_nonempty(values.size)
    result = np.empty(values.size, dtype=np.float64)
    if opcode == 80:
        running = 0.0
        for index in range(values.size):
            running += values[index]
            result[index] = running
    elif opcode == 81:
        running = 1.0
        for index in range(values.size):
            running *= values[index]
            result[index] = running
    elif opcode == 83:
        running = values[0]
        for index in range(values.size):
            running = max(running, values[index])
            result[index] = running
    else:
        running = values[0]
        for index in range(values.size):
            running = min(running, values[index])
            result[index] = running
    return result


@njit(cache=True, nogil=True)
def drawdown_series_1d(values: np.ndarray) -> np.ndarray:
    """Return signed drawdowns from the running peak of a positive level path."""

    _validate_nonempty(values.size)
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("DOMAIN_ERROR")
    result = np.empty(values.size, dtype=np.float64)
    running_peak = values[0]
    result[0] = 0.0
    for index in range(1, values.size):
        value = values[index]
        if value > running_peak:
            running_peak = value
        result[index] = value / running_peak - 1.0
    return result


@njit(cache=True, nogil=True)
def new_high_mask_1d(values: np.ndarray) -> np.ndarray:
    """Mark only strict new highs; equal plateaus do not count again."""

    _validate_nonempty(values.size)
    for value in values:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("DOMAIN_ERROR")
    result = np.zeros(values.size, dtype=np.uint8)
    running_peak = values[0]
    result[0] = 1
    for index in range(1, values.size):
        value = values[index]
        if value > running_peak:
            result[index] = 1
            running_peak = value
    return result


@njit(cache=True, nogil=True)
def first_1d(values: np.ndarray) -> float:
    _validate_nonempty(values.size)
    return values[0]


@njit(cache=True, nogil=True)
def last_1d(values: np.ndarray) -> float:
    _validate_nonempty(values.size)
    return values[-1]


@njit(cache=True, nogil=True)
def length_1d(values: np.ndarray) -> float:
    return float(values.size)


@njit(cache=True, nogil=True)
def lag_1d(values: np.ndarray, periods: int) -> np.ndarray:
    if periods < 0 or periods >= values.size:
        raise ValueError("INVALID_PARAMETER")
    return values[: values.size - periods].copy() if periods else values.copy()


@njit(cache=True, nogil=True)
def lag_1d_parameter(values: np.ndarray, periods: float) -> np.ndarray:
    integer = int(periods)
    if not math.isfinite(periods) or float(integer) != periods:
        raise ValueError("INVALID_PARAMETER")
    return lag_1d(values, integer)


@njit(cache=True, nogil=True)
def difference_1d(values: np.ndarray, periods: int) -> np.ndarray:
    if periods <= 0 or periods >= values.size:
        raise ValueError("INVALID_PARAMETER")
    result = np.empty(values.size - periods, dtype=np.float64)
    for index in range(result.size):
        result[index] = values[index + periods] - values[index]
    return result


@njit(cache=True, nogil=True)
def difference_1d_parameter(values: np.ndarray, periods: float) -> np.ndarray:
    integer = int(periods)
    if not math.isfinite(periods) or float(integer) != periods:
        raise ValueError("INVALID_PARAMETER")
    return difference_1d(values, integer)


@njit(cache=True, nogil=True, inline="always")
def _integer_parameter(value: float, allow_zero: bool = False) -> int:
    integer = int(value)
    minimum = 0 if allow_zero else 1
    if not math.isfinite(value) or float(integer) != value or integer < minimum:
        raise ValueError("INVALID_PARAMETER")
    return integer


@njit(cache=True, nogil=True)
def rolling_mean_1d(
    values: np.ndarray,
    window: float,
    min_periods: float,
) -> np.ndarray:
    width = _integer_parameter(window, False)
    minimum = _integer_parameter(min_periods, False)
    if minimum > width:
        raise ValueError("INVALID_PARAMETER")
    result = np.full(values.size, np.nan, dtype=np.float64)
    running_sum = 0.0
    finite_count = 0
    for index in range(values.size):
        current = values[index]
        if math.isfinite(current):
            running_sum += current
            finite_count += 1
        if index >= width:
            expired = values[index - width]
            if math.isfinite(expired):
                running_sum -= expired
                finite_count -= 1
        if finite_count >= minimum:
            result[index] = running_sum / finite_count
    return result


@njit(cache=True, nogil=True)
def _rolling_second_moment_1d(
    values: np.ndarray,
    window: float,
    ddof: float,
    min_periods: float,
    take_sqrt: int,
) -> np.ndarray:
    width = _integer_parameter(window, False)
    degrees = _integer_parameter(ddof, True)
    minimum = _integer_parameter(min_periods, False)
    if minimum > width:
        raise ValueError("INVALID_PARAMETER")
    result = np.full(values.size, np.nan, dtype=np.float64)
    running_sum = 0.0
    running_square_sum = 0.0
    finite_count = 0
    for index in range(values.size):
        current = values[index]
        if math.isfinite(current):
            running_sum += current
            running_square_sum += current * current
            finite_count += 1
        if index >= width:
            expired = values[index - width]
            if math.isfinite(expired):
                running_sum -= expired
                running_square_sum -= expired * expired
                finite_count -= 1
        if finite_count < minimum or finite_count <= degrees:
            continue
        centered_sum = running_square_sum - running_sum * running_sum / finite_count
        if centered_sum < 0.0 and centered_sum > -1e-12:
            centered_sum = 0.0
        if centered_sum >= 0.0:
            variance = centered_sum / (finite_count - degrees)
            result[index] = math.sqrt(variance) if take_sqrt != 0 else variance
    return result


@njit(cache=True, nogil=True)
def rolling_std_1d(
    values: np.ndarray,
    window: float,
    ddof: float,
    min_periods: float,
) -> np.ndarray:
    return _rolling_second_moment_1d(values, window, ddof, min_periods, 1)


@njit(cache=True, nogil=True)
def rolling_variance_1d(
    values: np.ndarray,
    window: float,
    ddof: float,
    min_periods: float,
) -> np.ndarray:
    return _rolling_second_moment_1d(values, window, ddof, min_periods, 0)


@njit(cache=True, nogil=True)
def rolling_extreme_1d(
    values: np.ndarray,
    window: float,
    min_periods: float,
    maximum: int,
) -> np.ndarray:
    width = _integer_parameter(window, False)
    minimum = _integer_parameter(min_periods, False)
    if minimum > width:
        raise ValueError("INVALID_PARAMETER")
    result = np.full(values.size, np.nan, dtype=np.float64)
    queue = np.empty(values.size, dtype=np.int64)
    head = 0
    tail = 0
    finite_count = 0
    for index in range(values.size):
        expired_index = index - width
        if expired_index >= 0 and math.isfinite(values[expired_index]):
            finite_count -= 1
        while head < tail and queue[head] <= expired_index:
            head += 1
        current = values[index]
        if math.isfinite(current):
            finite_count += 1
            if maximum != 0:
                while head < tail and values[queue[tail - 1]] <= current:
                    tail -= 1
            else:
                while head < tail and values[queue[tail - 1]] >= current:
                    tail -= 1
            queue[tail] = index
            tail += 1
        if finite_count >= minimum and head < tail:
            result[index] = values[queue[head]]
    return result


@njit(cache=True, nogil=True)
def rolling_min_1d(
    values: np.ndarray,
    window: float,
    min_periods: float,
) -> np.ndarray:
    return rolling_extreme_1d(values, window, min_periods, 0)


@njit(cache=True, nogil=True)
def rolling_max_1d(
    values: np.ndarray,
    window: float,
    min_periods: float,
) -> np.ndarray:
    return rolling_extreme_1d(values, window, min_periods, 1)


@njit(cache=True, nogil=True)
def recursive_smooth_1d(
    values: np.ndarray,
    periods: float,
    initial: float,
) -> np.ndarray:
    width = _integer_parameter(periods, False)
    if not math.isfinite(initial):
        raise ValueError("INVALID_PARAMETER")
    result = np.full(values.size, np.nan, dtype=np.float64)
    previous = initial
    for index in range(values.size):
        current = values[index]
        if not math.isfinite(current):
            continue
        previous = ((width - 1.0) * previous + current) / width
        result[index] = previous
    return result


@njit(cache=True, nogil=True)
def divide_or_default_1d(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float,
) -> np.ndarray:
    if numerator.size != denominator.size or not math.isfinite(default):
        raise ValueError("INVALID_PARAMETER")
    result = np.full(numerator.size, np.nan, dtype=np.float64)
    for index in range(numerator.size):
        lhs = numerator[index]
        rhs = denominator[index]
        if not math.isfinite(lhs) or not math.isfinite(rhs):
            continue
        result[index] = default if abs(rhs) < 1e-12 else lhs / rhs
    return result


@njit(cache=True, nogil=True)
def axis_reduce_asset(opcode: int, values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("INSUFFICIENT_SAMPLE")
    result = np.empty(values.shape[0], dtype=np.float64)
    if opcode == 120:
        mapped = 60
    elif opcode == 121:
        mapped = 62
    elif opcode == 122:
        mapped = 61
    elif opcode == 123:
        mapped = 65
    elif opcode == 124:
        mapped = 66
    elif opcode == 125:
        mapped = 63
    else:
        mapped = 64
    for row in range(values.shape[0]):
        result[row] = reduce_1d(mapped, values[row, :], 1)
    return result


# Correct the time-axis opcode mapping in a separate tiny dispatcher.  Keeping
# axis direction explicit avoids Numba's unsupported generic ``axis`` reducers.
@njit(cache=True, nogil=True)
def axis_reduce_time_fixed(opcode: int, values: np.ndarray) -> np.ndarray:
    result = np.empty(values.shape[1], dtype=np.float64)
    if opcode == 120:
        mapped = 60
    elif opcode == 121:
        mapped = 62
    elif opcode == 122:
        mapped = 61
    elif opcode == 123:
        mapped = 65
    elif opcode == 124:
        mapped = 66
    elif opcode == 125:
        mapped = 63
    else:
        mapped = 64
    for column in range(values.shape[1]):
        result[column] = reduce_1d(mapped, values[:, column], 1)
    return result


@njit(cache=True, nogil=True)
def masked_reduce_1d(opcode: int, values: np.ndarray, mask: np.ndarray) -> float:
    selected_count = 0
    total = 0.0
    mean = 0.0
    m2 = 0.0
    minimum = math.inf
    maximum = -math.inf
    for index in range(values.size):
        if mask[index] == 0:
            continue
        value = values[index]
        selected_count += 1
        total += value
        delta = value - mean
        mean += delta / selected_count
        m2 += delta * (value - mean)
        minimum = min(minimum, value)
        maximum = max(maximum, value)
    if selected_count == 0:
        raise ValueError("INSUFFICIENT_SAMPLE")
    if opcode == 100:
        return total
    if opcode == 101:
        return mean
    if opcode in (102, 103):
        if selected_count < 2:
            raise ValueError("INSUFFICIENT_SAMPLE")
        variance = m2 / (selected_count - 1)
        return variance if opcode == 102 else math.sqrt(variance)
    if opcode == 104:
        return minimum
    if opcode == 105:
        return maximum
    selected = np.empty(selected_count, dtype=np.float64)
    cursor = 0
    for index in range(values.size):
        if mask[index] != 0:
            selected[cursor] = values[index]
            cursor += 1
    return reduce_1d(67, selected, 1)


@njit(cache=True, nogil=True)
def masked_reduce_2d(opcode: int, values: np.ndarray, mask: np.ndarray) -> float:
    return masked_reduce_1d(opcode, values.reshape(values.size), mask.reshape(mask.size))


@njit(cache=True, nogil=True)
def masked_quantile_1d(values: np.ndarray, mask: np.ndarray, probability: float) -> float:
    count = 0
    for value in mask:
        count += int(value != 0)
    if count == 0:
        raise ValueError("INSUFFICIENT_SAMPLE")
    selected = np.empty(count, dtype=np.float64)
    cursor = 0
    for index in range(values.size):
        if mask[index] != 0:
            selected[cursor] = values[index]
            cursor += 1
    return quantile_1d(selected, probability)


@njit(cache=True, nogil=True)
def masked_quantile_2d(values: np.ndarray, mask: np.ndarray, probability: float) -> float:
    return masked_quantile_1d(values.reshape(values.size), mask.reshape(mask.size), probability)


@njit(cache=True, nogil=True)
def count_true_1d(mask: np.ndarray) -> float:
    count = 0
    for value in mask:
        count += int(value != 0)
    return float(count)


@njit(cache=True, nogil=True)
def count_true_2d(mask: np.ndarray) -> float:
    return count_true_1d(mask.reshape(mask.size))


@njit(cache=True, nogil=True)
def max_consecutive_true_1d(mask: np.ndarray) -> float:
    longest = 0
    current = 0
    for value in mask:
        if value != 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return float(longest)


@njit(cache=True, nogil=True)
def transpose_2d(values: np.ndarray) -> np.ndarray:
    return values.T.copy()


@njit(cache=True, nogil=True)
def dot_1d(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.dot(lhs, rhs))


@njit(cache=True, nogil=True)
def outer_1d(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return np.outer(lhs, rhs)


@njit(cache=True, nogil=True)
def matmul_2d(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs @ rhs


@njit(cache=True, nogil=True)
def matvec_2d(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return lhs @ rhs


@njit(cache=True, nogil=True)
def diag_1d(values: np.ndarray) -> np.ndarray:
    return np.diag(values)


@njit(cache=True, nogil=True)
def diag_2d(values: np.ndarray) -> np.ndarray:
    return np.diag(values)


@njit(cache=True, nogil=True)
def trace_2d(values: np.ndarray) -> float:
    return float(np.trace(values))


@njit(cache=True, nogil=True)
def solve_2d(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    size = vector.size
    coefficients = matrix.copy()
    rhs = vector.copy()
    for pivot_column in range(size):
        pivot_row = pivot_column
        pivot_magnitude = abs(coefficients[pivot_row, pivot_column])
        for candidate in range(pivot_column + 1, size):
            candidate_magnitude = abs(coefficients[candidate, pivot_column])
            if candidate_magnitude > pivot_magnitude:
                pivot_magnitude = candidate_magnitude
                pivot_row = candidate
        if pivot_magnitude <= 1e-14:
            raise ValueError("SINGULAR_MATRIX")
        if pivot_row != pivot_column:
            for column in range(size):
                temporary = coefficients[pivot_column, column]
                coefficients[pivot_column, column] = coefficients[pivot_row, column]
                coefficients[pivot_row, column] = temporary
            temporary_rhs = rhs[pivot_column]
            rhs[pivot_column] = rhs[pivot_row]
            rhs[pivot_row] = temporary_rhs
        pivot = coefficients[pivot_column, pivot_column]
        for row in range(pivot_column + 1, size):
            factor = coefficients[row, pivot_column] / pivot
            coefficients[row, pivot_column] = 0.0
            for column in range(pivot_column + 1, size):
                coefficients[row, column] -= factor * coefficients[pivot_column, column]
            rhs[row] -= factor * rhs[pivot_column]
    result = np.empty(size, dtype=np.float64)
    for reverse_index in range(size):
        row = size - 1 - reverse_index
        total = rhs[row]
        for column in range(row + 1, size):
            total -= coefficients[row, column] * result[column]
        pivot = coefficients[row, row]
        if abs(pivot) <= 1e-14:
            raise ValueError("SINGULAR_MATRIX")
        result[row] = total / pivot
    return result


@njit(cache=True, nogil=True)
def covariance_1d(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.size < 2:
        raise ValueError("INSUFFICIENT_SAMPLE")
    mean_lhs = reduce_1d(62, lhs, 1)
    mean_rhs = reduce_1d(62, rhs, 1)
    total = 0.0
    for index in range(lhs.size):
        total += (lhs[index] - mean_lhs) * (rhs[index] - mean_rhs)
    return total / (lhs.size - 1)


@njit(cache=True, nogil=True)
def covariance_2d(values: np.ndarray) -> np.ndarray:
    if values.shape[0] < 2:
        raise ValueError("INSUFFICIENT_SAMPLE")
    columns = values.shape[1]
    result = np.empty((columns, columns), dtype=np.float64)
    for left in range(columns):
        for right in range(left, columns):
            value = covariance_1d(values[:, left], values[:, right])
            result[left, right] = value
            result[right, left] = value
    return result


@njit(cache=True, nogil=True)
def correlation_1d(lhs: np.ndarray, rhs: np.ndarray) -> float:
    covariance = covariance_1d(lhs, rhs)
    lhs_std = reduce_1d(66, lhs, 1)
    rhs_std = reduce_1d(66, rhs, 1)
    if lhs_std == 0.0 or rhs_std == 0.0:
        raise ValueError("NON_FINITE_RESULT")
    return covariance / (lhs_std * rhs_std)


@njit(cache=True, nogil=True)
def correlation_2d(values: np.ndarray) -> np.ndarray:
    covariance = covariance_2d(values)
    columns = values.shape[1]
    result = np.empty((columns, columns), dtype=np.float64)
    deviations = np.empty(columns, dtype=np.float64)
    for column in range(columns):
        deviations[column] = math.sqrt(covariance[column, column])
        if deviations[column] == 0.0:
            raise ValueError("NON_FINITE_RESULT")
    for left in range(columns):
        for right in range(columns):
            result[left, right] = covariance[left, right] / (deviations[left] * deviations[right])
    return result


@dataclass(frozen=True)
class KernelSpec:
    operator_id: str
    operator_version: str
    opcode: int
    kernel_version: str
    execution_lane: str
    input_signatures: tuple[str, ...]
    output_signature: str
    serial_kernels: tuple[CPUDispatcher, ...]
    parallel_policy: str
    status_contract: str
    warmup_cases: tuple[str, ...]

    @property
    def serial_kernel(self) -> CPUDispatcher:
        return self.serial_kernels[0]

    @property
    def compiled_signatures(self) -> tuple[str, ...]:
        values: list[str] = []
        for dispatcher in self.serial_kernels:
            values.extend(f"{dispatcher.py_func.__name__}{signature}" for signature in dispatcher.signatures)
        return tuple(values)

    def catalog_entry(self) -> dict[str, Any]:
        return {
            "njit_supported": True,
            "kernel_version": self.kernel_version,
            "execution_lane": self.execution_lane,
            "opcode": self.opcode,
            "input_signatures": list(self.input_signatures),
            "output_signature": self.output_signature,
            "parallel_policy": self.parallel_policy,
            "status_contract": self.status_contract,
            "warmup_cases": list(self.warmup_cases),
            "compiled_signatures": list(self.compiled_signatures),
            "warmup_status": "ready" if self.compiled_signatures else "pending",
        }


def _unique_dispatchers(values: Iterable[CPUDispatcher]) -> tuple[CPUDispatcher, ...]:
    result: list[CPUDispatcher] = []
    seen: set[int] = set()
    for value in values:
        if id(value) not in seen:
            seen.add(id(value))
            result.append(value)
    return tuple(result)


def _operator_dispatchers(operator_id: str) -> tuple[CPUDispatcher, ...]:
    from .primitive_access import ACCESS_KERNELS
    from .regression_state import FIT_PROJECTION_KERNELS, linear_fit_pair_kernel, linear_fit_time_kernel
    from .operator_lowering import COMPOSITE_DEPENDENCIES
    if operator_id in ACCESS_KERNELS:
        return (ACCESS_KERNELS[operator_id],)
    if operator_id in FIT_PROJECTION_KERNELS:
        return (FIT_PROJECTION_KERNELS[operator_id],)
    if operator_id == "linear_fit":
        return (linear_fit_time_kernel, linear_fit_pair_kernel)
    if operator_id in COMPOSITE_DEPENDENCIES:
        return _unique_dispatchers(kernel for name in COMPOSITE_DEPENDENCIES[operator_id] for kernel in _operator_dispatchers(name))
    if operator_id in BASIC_OPCODES:
        return (binary_scalar, binary_1d, binary_1d_right_scalar, binary_1d_left_scalar, binary_2d, binary_2d_right_scalar, binary_2d_left_scalar)
    if operator_id in UNARY_OPCODES:
        return (unary_scalar, unary_1d, unary_2d)
    if operator_id == "clip":
        return (clip_scalar, clip_1d, clip_2d)
    if operator_id in COMPARISON_OPCODES:
        return (comparison_scalar, comparison_1d, comparison_1d_right_scalar, comparison_1d_left_scalar, comparison_2d, comparison_2d_right_scalar, comparison_2d_left_scalar)
    if operator_id in {"logical_and", "logical_or"}:
        return (logical_scalar, logical_1d, logical_2d)
    if operator_id == "logical_not":
        return (logical_not_scalar, logical_not_1d, logical_not_2d)
    if operator_id == "where":
        return (where_scalar, where_1d, where_1d_false_scalar, where_1d_true_scalar, where_2d, where_2d_false_scalar, where_2d_true_scalar)
    if operator_id in REDUCTION_OPCODES:
        if operator_id in {"variance", "std"}:
            return (reduce_1d, reduce_2d, reduce_1d_parameter, reduce_2d_parameter)
        return (reduce_1d, reduce_2d)
    if operator_id == "quantile":
        return (quantile_1d, quantile_2d)
    if operator_id in SCAN_OPCODES:
        return (scan_1d,)
    if operator_id == "drawdown_series":
        return (drawdown_series_1d,)
    if operator_id == "new_high_mask":
        return (new_high_mask_1d,)
    if operator_id == "first":
        return (first_1d,)
    if operator_id == "last":
        return (last_1d,)
    if operator_id == "length":
        return (length_1d,)
    if operator_id == "lag":
        return (lag_1d_parameter,)
    if operator_id == "difference":
        return (difference_1d_parameter,)
    if operator_id == "rolling_mean":
        return (rolling_mean_1d,)
    if operator_id == "rolling_std":
        return (rolling_std_1d,)
    if operator_id == "rolling_min":
        return (rolling_min_1d,)
    if operator_id == "rolling_max":
        return (rolling_max_1d,)
    if operator_id == "recursive_smooth":
        return (recursive_smooth_1d,)
    if operator_id == "divide_or_default":
        return (divide_or_default_1d,)
    if operator_id.endswith("_time"):
        return (axis_reduce_time_fixed,)
    if operator_id.endswith("_asset"):
        return (axis_reduce_asset,)
    if operator_id in MASK_REDUCTION_OPCODES:
        return (masked_reduce_1d, masked_reduce_2d)
    if operator_id == "quantile_where":
        return (masked_quantile_1d, masked_quantile_2d)
    if operator_id == "count_true":
        return (count_true_1d, count_true_2d)
    if operator_id == "max_consecutive_true":
        return (max_consecutive_true_1d,)
    mapping = {
        "transpose": (transpose_2d,), "dot": (dot_1d,), "outer": (outer_1d,),
        "matmul": (matmul_2d,), "matvec": (matvec_2d,), "diag": (diag_1d, diag_2d),
        "trace": (trace_2d,), "solve": (solve_2d,), "covariance": (covariance_1d, covariance_2d),
        "correlation": (correlation_1d, correlation_2d),
    }
    return mapping[operator_id]


@lru_cache(maxsize=1)
def get_numba_kernel_registry() -> dict[str, KernelSpec]:
    from .drawdown_interval import INTERVAL_KERNELS
    from .primitive_access import ACCESS_KERNELS
    from .regression_state import FIT_PROJECTION_KERNELS, linear_fit_pair_kernel, linear_fit_time_kernel
    fixed_kernels = {name: (kernel,) for name, kernel in {**INTERVAL_KERNELS, **ACCESS_KERNELS, **FIT_PROJECTION_KERNELS}.items()}
    fixed_kernels["linear_fit"] = (linear_fit_time_kernel, linear_fit_pair_kernel)
    registry: dict[str, KernelSpec] = {}
    for opcode, operator_id in enumerate(CANONICAL_OPERATOR_IDS, start=1):
        if operator_id in fixed_kernels:
            continue  # Structured signatures are registered below.
        lane = "numba_blas" if operator_id in {"dot", "outer", "matmul", "matvec", "solve", "portfolio_returns", "quadratic_form"} else "numba"
        registry[operator_id] = KernelSpec(
            operator_id=operator_id,
            operator_version=(
                "2.3.0"
                if operator_id
                in {
                    "rolling_mean",
                    "rolling_std",
                    "rolling_min",
                    "rolling_max",
                    "recursive_smooth",
                    "divide_or_default",
                }
                else "2.2.0"
            ),
            opcode=opcode,
            kernel_version=NUMERIC_KERNEL_VERSION,
            execution_lane=lane,
            input_signatures=(
                "float64",
                "float64[::1]",
                "float64[:,::1]",
                "uint8[::1]",
                "uint8[:,::1]",
            ),
            output_signature=(
                "uint8 | uint8[::1] | uint8[:,::1]"
                if operator_id in COMPARISON_OPCODES
                or operator_id
                in {"logical_and", "logical_or", "logical_not", "new_high_mask", "finite_mask"}
                else "float64 | float64[::1] | float64[:,::1]"
            ),
            serial_kernels=_unique_dispatchers(_operator_dispatchers(operator_id)),
            parallel_policy="outer_product_dimension_only",
            status_contract="stable_integer_status_at_batch_boundary",
            warmup_cases=("scalar", "float64[::1]", "float64[:,::1]"),
        )
    for operator_id, dispatchers in fixed_kernels.items():
        registry[operator_id] = KernelSpec(
            operator_id=operator_id, operator_version="2.3.0", opcode=len(registry) + 1,
            kernel_version=NUMERIC_KERNEL_VERSION, execution_lane="numba",
            input_signatures=tuple(str(signature) for dispatcher in dispatchers for signature in dispatcher.signatures),
            output_signature=" | ".join(sorted({str(signature.return_type) for dispatcher in dispatchers for signature in dispatcher.nopython_signatures})),
            serial_kernels=dispatchers, parallel_policy="outer_product_dimension_only",
            status_contract="typed_state_or_scalar_with_independent_missing_status",
            warmup_cases=("explicit_fixed_signature",),
        )
    return registry


def _compile(dispatcher: CPUDispatcher, signatures: Iterable[tuple[Any, ...]]) -> None:
    for signature in signatures:
        dispatcher.compile(signature)


@lru_cache(maxsize=1)
def warm_numba_kernel_registry() -> dict[str, Any]:
    """Compile every fixed layout used by the current canonical operators."""

    f1 = types.float64[::1]
    f2 = types.float64[:, ::1]
    u1 = types.uint8[::1]
    u2 = types.uint8[:, ::1]
    i8 = types.int64
    f8 = types.float64
    for dispatcher in (binary_scalar,):
        _compile(dispatcher, ((i8, f8, f8),))
    _compile(binary_1d, ((i8, f1, f1),))
    _compile(binary_1d_right_scalar, ((i8, f1, f8),))
    _compile(binary_1d_left_scalar, ((i8, f8, f1),))
    _compile(binary_2d, ((i8, f2, f2),))
    _compile(binary_2d_right_scalar, ((i8, f2, f8),))
    _compile(binary_2d_left_scalar, ((i8, f8, f2),))
    _compile(series_safe_divide_1d, ((f1, f1),))
    _compile(series_safe_divide_1d_right_scalar, ((f1, f8),))
    _compile(series_safe_divide_1d_left_scalar, ((f8, f1),))
    _compile(series_safe_divide_2d, ((f2, f2),))
    _compile(series_safe_divide_2d_right_scalar, ((f2, f8),))
    _compile(series_safe_divide_2d_left_scalar, ((f8, f2),))
    _compile(unary_scalar, ((i8, f8),))
    _compile(unary_1d, ((i8, f1),))
    _compile(unary_2d, ((i8, f2),))
    _compile(clip_scalar, ((f8, f8, f8),))
    _compile(clip_1d, ((f1, f8, f8),))
    _compile(clip_2d, ((f2, f8, f8),))
    _compile(comparison_scalar, ((i8, f8, f8),))
    _compile(comparison_1d, ((i8, f1, f1),))
    _compile(comparison_1d_right_scalar, ((i8, f1, f8),))
    _compile(comparison_1d_left_scalar, ((i8, f8, f1),))
    _compile(comparison_2d, ((i8, f2, f2),))
    _compile(comparison_2d_right_scalar, ((i8, f2, f8),))
    _compile(comparison_2d_left_scalar, ((i8, f8, f2),))
    _compile(logical_scalar, ((i8, types.uint8, types.uint8),))
    _compile(logical_1d, ((i8, u1, u1),))
    _compile(logical_2d, ((i8, u2, u2),))
    _compile(logical_not_scalar, ((types.uint8,),))
    _compile(logical_not_1d, ((u1,),))
    _compile(logical_not_2d, ((u2,),))
    _compile(where_scalar, ((types.uint8, f8, f8),))
    _compile(where_1d, ((u1, f1, f1),))
    _compile(where_1d_false_scalar, ((u1, f1, f8),))
    _compile(where_1d_true_scalar, ((u1, f8, f1),))
    _compile(where_2d, ((u2, f2, f2),))
    _compile(where_2d_false_scalar, ((u2, f2, f8),))
    _compile(where_2d_true_scalar, ((u2, f8, f2),))
    _compile(reduce_1d, ((i8, f1, i8),))
    _compile(reduce_2d, ((i8, f2, i8),))
    _compile(reduce_1d_parameter, ((f1, f8, i8),))
    _compile(reduce_2d_parameter, ((f2, f8, i8),))
    _compile(quantile_1d, ((f1, f8),))
    _compile(quantile_2d, ((f2, f8),))
    _compile(scan_1d, ((i8, f1),))
    _compile(drawdown_series_1d, ((f1,),))
    _compile(new_high_mask_1d, ((f1,),))
    for dispatcher in (first_1d, last_1d, length_1d):
        _compile(dispatcher, ((f1,),))
    _compile(lag_1d_parameter, ((f1, f8),))
    _compile(difference_1d_parameter, ((f1, f8),))
    _compile(rolling_mean_1d, ((f1, f8, f8),))
    _compile(rolling_std_1d, ((f1, f8, f8, f8),))
    _compile(rolling_variance_1d, ((f1, f8, f8, f8),))
    _compile(rolling_min_1d, ((f1, f8, f8),))
    _compile(rolling_max_1d, ((f1, f8, f8),))
    _compile(recursive_smooth_1d, ((f1, f8, f8),))
    _compile(divide_or_default_1d, ((f1, f1, f8),))
    _compile(axis_reduce_time_fixed, ((i8, f2),))
    _compile(axis_reduce_asset, ((i8, f2),))
    _compile(masked_reduce_1d, ((i8, f1, u1),))
    _compile(masked_reduce_2d, ((i8, f2, u2),))
    _compile(masked_quantile_1d, ((f1, u1, f8),))
    _compile(masked_quantile_2d, ((f2, u2, f8),))
    _compile(count_true_1d, ((u1,),))
    _compile(count_true_2d, ((u2,),))
    _compile(max_consecutive_true_1d, ((u1,),))
    _compile(transpose_2d, ((f2,),))
    _compile(dot_1d, ((f1, f1),))
    _compile(outer_1d, ((f1, f1),))
    _compile(matmul_2d, ((f2, f2),))
    _compile(matvec_2d, ((f2, f1),))
    _compile(diag_1d, ((f1,),))
    _compile(diag_2d, ((f2,),))
    _compile(trace_2d, ((f2,),))
    _compile(solve_2d, ((f2, f1),))
    _compile(covariance_1d, ((f1, f1),))
    _compile(covariance_2d, ((f2,),))
    _compile(correlation_1d, ((f1, f1),))
    _compile(correlation_2d, ((f2,),))
    registry = get_numba_kernel_registry()
    missing = [operator_id for operator_id, spec in registry.items() if not spec.compiled_signatures]
    if missing:
        raise RuntimeError(f"NJIT kernel warmup incomplete: {', '.join(missing)}")
    return kernel_registry_status(warmed=True)


def kernel_registry_status(*, warmed: bool | None = None) -> dict[str, Any]:
    registry = get_numba_kernel_registry()
    if warmed is None:
        warmed = warm_numba_kernel_registry.cache_info().currsize > 0
    ready = sum(bool(spec.compiled_signatures) for spec in registry.values())
    return {
        "engine_version": ENGINE_VERSION,
        "kernel_version": NUMERIC_KERNEL_VERSION,
        "numba_version": numba.__version__,
        "operator_coverage": f"{ready}/{len(registry)}",
        "canonical_operators": len(registry),
        "warmed": warmed,
        "python_fallback": 0,
        "python_operator_calls": 0,
        "execution_lanes": {"numba_fused": True, "numba_blas": True, "python_fallback": 0},
    }


def kernel_catalog_entry(operator_id: str) -> dict[str, Any]:
    if operator_id == "rolling_apply":
        return {"njit_supported": True, "kernel_version": NUMERIC_KERNEL_VERSION,
                "execution_lane": "compiler_scoped_interval", "compiled_signatures": [],
                "warmup_status": "formula_preparation_required", "status_contract": "interval_scalar_to_aligned_series",
                "execution_scope": "deferred_interval_body", "materializes_windows": False}
    from .operator_lowering import COMPOSITE_DEPENDENCIES, COMPILER_FUSED_OPERATOR_IDS
    if operator_id in COMPILER_FUSED_OPERATOR_IDS:
        dispatchers = (
            rolling_mean_1d,
            rolling_std_1d,
            rolling_variance_1d,
            rolling_min_1d,
            rolling_max_1d,
        )
        signatures = sorted(
            {
                str(signature)
                for dispatcher in dispatchers
                for signature in dispatcher.signatures
            }
        )
        return {
            "njit_supported": True,
            "kernel_version": NUMERIC_KERNEL_VERSION,
            "execution_lane": "compiler_fused_no_materialization",
            "compiled_signatures": signatures,
            "warmup_status": "ready" if signatures else "pending",
            "expanded_operators": ["mean", "std", "variance", "min_value", "max_value"],
            "status_contract": "logical_window_fused_into_fixed_signature_reducer",
        }
    if operator_id in COMPOSITE_DEPENDENCIES:
        # Historical catalogs describe a compiler expansion, not an installed
        # second kernel or proof that a composite formula is already prepared.
        return {
            "njit_supported": True,
            "kernel_version": NUMERIC_KERNEL_VERSION,
            "execution_lane": "compiler_expansion",
            "compiled_signatures": [],
            "warmup_status": "formula_preparation_required",
            "expanded_operators": list(COMPOSITE_DEPENDENCIES[operator_id]),
            "status_contract": "current_primitive_dag",
        }
    return get_numba_kernel_registry()[operator_id].catalog_entry()


__all__ = [
    "CANONICAL_OPERATOR_IDS", "ENGINE_VERSION", "KernelSpec", "NUMERIC_KERNEL_VERSION",
    "drawdown_series_1d", "new_high_mask_1d", "rolling_mean_1d", "rolling_std_1d",
    "rolling_variance_1d", "rolling_min_1d", "rolling_max_1d", "recursive_smooth_1d", "divide_or_default_1d",
    "series_safe_divide_1d", "series_safe_divide_1d_right_scalar",
    "series_safe_divide_1d_left_scalar", "series_safe_divide_2d",
    "series_safe_divide_2d_right_scalar", "series_safe_divide_2d_left_scalar",
    "get_numba_kernel_registry", "kernel_catalog_entry", "kernel_registry_status",
    "warm_numba_kernel_registry",
]
