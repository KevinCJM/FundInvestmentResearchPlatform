from __future__ import annotations

"""Fixed-signature nopython kernels for fit, correlation and class analytics."""

import hashlib

import numpy as np
from numba import float64, int64, njit, types, uint8

try:
    from backend.compute_policy import validate_execution_audit
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit


FIT_ENGINE_VERSION = "fit-analytics-njit-1.0.0"
FIT_KERNEL_VERSION = "fit-correlation-risk-2"

_F1 = float64[::1]
_F2 = float64[:, ::1]
_U1 = uint8[::1]
_U2 = uint8[:, ::1]
_WEIGHT_RESULT = types.Tuple((_F2, _U1))
_CLASS_RESULT = types.Tuple((_F2, _F2, _F2))
_ROLLING_RESULT = types.Tuple((_F2, _F2))


@njit(_U2(_F2), cache=False, nogil=True)
def finite_matrix_mask_kernel(values: np.ndarray) -> np.ndarray:
    """Return the only numerical gate used by JSON result serialization."""

    output = np.zeros(values.shape, dtype=np.uint8)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if np.isfinite(values[row, column]):
                output[row, column] = 1
    return output


@njit(_F2(_F2), cache=False, nogil=True)
def returns_from_nav_kernel(nav: np.ndarray) -> np.ndarray:
    rows, columns = nav.shape
    output_rows = rows - 1 if rows > 0 else 0
    output = np.full((output_rows, columns), np.nan, dtype=np.float64)
    for column in range(columns):
        for row in range(1, rows):
            previous = nav[row - 1, column]
            current = nav[row, column]
            if np.isfinite(previous) and previous != 0.0 and np.isfinite(current):
                value = current / previous - 1.0
                if np.isfinite(value):
                    output[row - 1, column] = value
    return output


@njit(_WEIGHT_RESULT(_F2), cache=False, nogil=True)
def normalize_weight_matrix_kernel(raw_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assets, classes = raw_weights.shape
    output = np.zeros((assets, classes), dtype=np.float64)
    valid = np.zeros(classes, dtype=np.uint8)
    for class_idx in range(classes):
        total = 0.0
        for asset_idx in range(assets):
            weight = raw_weights[asset_idx, class_idx]
            if np.isfinite(weight) and weight > 0.0:
                output[asset_idx, class_idx] = weight
                total += weight
        if total > 0.0:
            valid[class_idx] = 1
            for asset_idx in range(assets):
                output[asset_idx, class_idx] /= total
    return output, valid


@njit(_F2(int64[::1], int64[::1], _F1, int64, int64), cache=False, nogil=True)
def assemble_weight_matrix_kernel(
    asset_indexes: np.ndarray,
    class_indexes: np.ndarray,
    values: np.ndarray,
    asset_count: int,
    class_count: int,
) -> np.ndarray:
    if asset_indexes.size != class_indexes.size or asset_indexes.size != values.size:
        raise ValueError("weight coordinate arrays must have the same length")
    output = np.zeros((asset_count, class_count), dtype=np.float64)
    for idx in range(values.size):
        asset_idx = asset_indexes[idx]
        class_idx = class_indexes[idx]
        if asset_idx < 0 or asset_idx >= asset_count or class_idx < 0 or class_idx >= class_count:
            raise ValueError("weight coordinate is outside the matrix")
        value = values[idx]
        if np.isfinite(value):
            output[asset_idx, class_idx] += value
    return output


@njit(_F2(_F2, _F2), cache=False, nogil=True)
def weighted_returns_kernel(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    rows, assets = returns.shape
    if weights.shape[0] != assets:
        raise ValueError("weight matrix does not match return matrix")
    classes = weights.shape[1]
    output = np.zeros((rows, classes), dtype=np.float64)
    for row in range(rows):
        for class_idx in range(classes):
            value = 0.0
            for asset_idx in range(assets):
                value += returns[row, asset_idx] * weights[asset_idx, class_idx]
            output[row, class_idx] = value
    return output


@njit(cache=False, nogil=True, inline="always")
def _mean(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    total = 0.0
    for value in values:
        total += value
    return total / values.size


@njit(cache=False, nogil=True, inline="always")
def _sample_std(values: np.ndarray, mean: float) -> float:
    if values.size < 2:
        return np.nan
    total = 0.0
    for value in values:
        delta = value - mean
        total += delta * delta
    return np.sqrt(total / (values.size - 1))


@njit(cache=False, nogil=True, inline="always")
def _linear_quantile(values: np.ndarray, probability: float) -> float:
    if values.size == 0:
        return np.nan
    ordered = np.sort(values.copy())
    position = probability * (ordered.size - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


@njit(cache=False, nogil=True, inline="always")
def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    size = left.size
    if size < 2 or right.size != size:
        return 0.0
    left_mean = _mean(left)
    right_mean = _mean(right)
    covariance = 0.0
    left_variance = 0.0
    right_variance = 0.0
    for idx in range(size):
        left_delta = left[idx] - left_mean
        right_delta = right[idx] - right_mean
        covariance += left_delta * right_delta
        left_variance += left_delta * left_delta
        right_variance += right_delta * right_delta
    denominator = np.sqrt(left_variance * right_variance)
    if denominator <= 0.0:
        return 0.0
    value = covariance / denominator
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


@njit(_CLASS_RESULT(_F2, float64), cache=False, nogil=True)
def class_nav_corr_metrics_kernel(
    class_returns: np.ndarray,
    periods_per_year: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, classes = class_returns.shape
    if rows == 0 or classes == 0:
        raise ValueError("class return matrix must not be empty")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")

    nav = np.empty((rows, classes), dtype=np.float64)
    corr = np.zeros((classes, classes), dtype=np.float64)
    metrics = np.empty((classes, 7), dtype=np.float64)
    for class_idx in range(classes):
        cumulative = 1.0
        for row in range(rows):
            cumulative *= 1.0 + class_returns[row, class_idx]
            nav[row, class_idx] = 1.0 if row == 0 else cumulative

    for left_idx in range(classes):
        left = class_returns[:, left_idx]
        for right_idx in range(classes):
            corr[left_idx, right_idx] = _correlation(left, class_returns[:, right_idx])

    for class_idx in range(classes):
        values = class_returns[:, class_idx]
        mean = _mean(values)
        annual_return = mean * periods_per_year
        std = _sample_std(values, mean)
        annual_vol = std * np.sqrt(periods_per_year) if np.isfinite(std) else np.nan
        sharpe = annual_return / annual_vol if np.isfinite(annual_vol) and annual_vol != 0.0 else np.nan
        q01 = _linear_quantile(values, 0.01)
        tail_total = 0.0
        tail_count = 0
        for value in values:
            if value <= q01:
                tail_total += value
                tail_count += 1
        var99 = -q01
        es99 = -(tail_total / tail_count) if tail_count > 0 else np.nan
        peak = nav[0, class_idx]
        max_drawdown = 0.0
        for row in range(rows):
            value = nav[row, class_idx]
            if value > peak:
                peak = value
            if peak != 0.0:
                drawdown = value / peak - 1.0
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0.0 else np.nan
        metrics[class_idx, 0] = annual_return
        metrics[class_idx, 1] = annual_vol
        metrics[class_idx, 2] = sharpe
        metrics[class_idx, 3] = var99
        metrics[class_idx, 4] = es99
        metrics[class_idx, 5] = max_drawdown
        metrics[class_idx, 6] = calmar
    return nav, corr, metrics


@njit(_ROLLING_RESULT(_F2, int64, int64), cache=False, nogil=True)
def rolling_correlation_kernel(
    returns: np.ndarray,
    target_index: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = returns.shape
    if target_index < 0 or target_index >= columns:
        raise ValueError("target index is outside the return matrix")
    if window <= 1:
        raise ValueError("window must be greater than one")
    rolling = np.zeros((rows, columns), dtype=np.float64)
    metrics = np.zeros((columns, 7), dtype=np.float64)
    target = returns[:, target_index]
    for column in range(columns):
        values = returns[:, column]
        overall = _correlation(values, target)
        for row in range(rows):
            if row + 1 >= window:
                start = row + 1 - window
                rolling[row, column] = _correlation(values[start : row + 1], target[start : row + 1])
        finite = rolling[:, column]
        mean = _mean(finite)
        median = _linear_quantile(finite, 0.5)
        std = _sample_std(finite, mean)
        if not np.isfinite(std):
            std = 0.0
        second = 0.0
        third = 0.0
        fourth = 0.0
        total = 0.0
        for value in finite:
            total += value
            delta = value - mean
            squared = delta * delta
            second += squared
            third += squared * delta
            fourth += squared * squared
        second /= finite.size if finite.size else 1
        third /= finite.size if finite.size else 1
        fourth /= finite.size if finite.size else 1
        skew = third / (second ** 1.5 + 1e-12) if finite.size > 2 else 0.0
        kurtosis = fourth / (second * second + 1e-12) - 3.0 if finite.size > 2 else 0.0
        metrics[column, 0] = overall
        metrics[column, 1] = total
        metrics[column, 2] = mean
        metrics[column, 3] = median
        metrics[column, 4] = std
        metrics[column, 5] = skew
        metrics[column, 6] = kurtosis
    return rolling, metrics


@njit(_F1(_F2, float64), cache=False, nogil=True)
def class_consistency_kernel(returns: np.ndarray, periods_per_year: float) -> np.ndarray:
    rows, columns = returns.shape
    output = np.empty(3, dtype=np.float64)
    output[:] = np.nan
    if rows < 2 or columns < 2:
        return output

    corr_total = 0.0
    corr_count = 0
    for left_idx in range(columns):
        for right_idx in range(left_idx + 1, columns):
            corr_total += _correlation(returns[:, left_idx], returns[:, right_idx])
            corr_count += 1
    output[0] = corr_total / corr_count if corr_count else np.nan

    means = np.empty(columns, dtype=np.float64)
    for column in range(columns):
        means[column] = _mean(returns[:, column])
    covariance = np.zeros((columns, columns), dtype=np.float64)
    for left_idx in range(columns):
        for right_idx in range(columns):
            total = 0.0
            for row in range(rows):
                total += (
                    (returns[row, left_idx] - means[left_idx])
                    * (returns[row, right_idx] - means[right_idx])
                )
            covariance[left_idx, right_idx] = total / (rows - 1)

    trace = 0.0
    for column in range(columns):
        trace += covariance[column, column]
    if trace > 1e-12:
        vector = np.full(columns, 1.0 / np.sqrt(columns), dtype=np.float64)
        for _ in range(100):
            next_vector = covariance @ vector
            norm = np.sqrt(np.sum(next_vector * next_vector))
            if norm <= 1e-15:
                break
            next_vector /= norm
            distance = np.sum(np.abs(next_vector - vector))
            vector = next_vector
            if distance < 1e-12:
                break
        largest = float(vector @ covariance @ vector)
        output[1] = largest / trace

    maximum_tracking_error = 0.0
    for column in range(columns):
        differences = np.empty(rows, dtype=np.float64)
        for row in range(rows):
            row_total = 0.0
            for other_column in range(columns):
                row_total += returns[row, other_column]
            differences[row] = returns[row, column] - row_total / columns
        difference_mean = _mean(differences)
        tracking_error = _sample_std(differences, difference_mean) * np.sqrt(periods_per_year)
        if np.isfinite(tracking_error) and tracking_error > maximum_tracking_error:
            maximum_tracking_error = tracking_error
    output[2] = maximum_tracking_error
    return output


_PUBLIC_KERNELS = (
    finite_matrix_mask_kernel,
    returns_from_nav_kernel,
    assemble_weight_matrix_kernel,
    normalize_weight_matrix_kernel,
    weighted_returns_kernel,
    class_nav_corr_metrics_kernel,
    rolling_correlation_kernel,
    class_consistency_kernel,
)

for _kernel in _PUBLIC_KERNELS:
    _kernel.disable_compile()


def fit_numba_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _PUBLIC_KERNELS
    }
    material = "|".join(
        [FIT_ENGINE_VERSION, FIT_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return validate_execution_audit({
        "engine": FIT_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": FIT_KERNEL_VERSION,
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(
            bool(kernel.nopython_signatures)
            and all(
                not compilation.objectmode
                for compilation in kernel.overloads.values()
            )
            for kernel in _PUBLIC_KERNELS
        ),
        "object_mode": 0,
        "python_fallback": 0,
        "request_time_compilation": 0,
    })


def warm_fit_numba_kernels() -> dict[str, object]:
    nav = np.ascontiguousarray(
        np.array([[1.0, 1.0], [1.01, 0.99], [1.02, 1.01]], dtype=np.float64)
    )
    returns = returns_from_nav_kernel(nav)
    finite_matrix_mask_kernel(nav)
    raw_weights = assemble_weight_matrix_kernel(
        np.ascontiguousarray(np.array([0, 1], dtype=np.int64)),
        np.ascontiguousarray(np.array([0, 1], dtype=np.int64)),
        np.ascontiguousarray(np.array([1.0, 1.0], dtype=np.float64)),
        2,
        2,
    )
    weights, valid = normalize_weight_matrix_kernel(raw_weights)
    class_returns = weighted_returns_kernel(returns, weights)
    class_nav_corr_metrics_kernel(class_returns, 252.0)
    rolling_correlation_kernel(class_returns, 0, 2)
    class_consistency_kernel(class_returns, 252.0)
    if not np.all(valid == 1):
        raise RuntimeError("拟合分析 NJIT 权重内核预热失败")
    audit = validate_execution_audit(fit_numba_execution_audit())
    if not audit["nopython"] or audit["python_fallback"] != 0:
        raise RuntimeError("拟合分析 NJIT 内核未进入 nopython 模式")
    return audit


__all__ = [
    "class_consistency_kernel",
    "class_nav_corr_metrics_kernel",
    "assemble_weight_matrix_kernel",
    "fit_numba_execution_audit",
    "finite_matrix_mask_kernel",
    "normalize_weight_matrix_kernel",
    "returns_from_nav_kernel",
    "rolling_correlation_kernel",
    "warm_fit_numba_kernels",
    "weighted_returns_kernel",
]
