"""Independent numerical operators for joint product risk and class-budget QP."""

import os
import numpy as np
from numba import njit, types
from backend.qp_numba import matrix_qp_kernel

V = types.Array(types.float64, 1, "A", readonly=True)
M = types.Array(types.float64, 2, "A", readonly=True)
INT_VECTOR = types.Array(types.int64, 1, "A", readonly=True)
N, F = types.int64, types.float64
_WARMED_PID = None


@njit((M, N, N, F), cache=True, nogil=True)
def covariance_kernel(values, start, end, annualization):
    if not 0 <= start < end <= values.shape[0] or end - start < 2 or annualization <= 0:
        raise ValueError("IMPLEMENTATION_SAMPLE")
    n, k = end - start, values.shape[1]
    means = np.zeros(k)
    covariance = np.zeros((k, k))
    for t in range(start, end):
        for j in range(k):
            if not np.isfinite(values[t, j]):
                raise ValueError("IMPLEMENTATION_NONFINITE")
            means[j] += values[t, j] / n
    for t in range(start, end):
        for i in range(k):
            for j in range(k):
                covariance[i, j] += (
                    (values[t, i] - means[i])
                    * (values[t, j] - means[j])
                    * annualization
                    / (n - 1)
                )
    return covariance


@njit((M, M, M), cache=True, nogil=True)
def residuals_kernel(factors, products, coefficients):
    if factors.shape[0] != products.shape[0] or coefficients.shape != (
        products.shape[1],
        factors.shape[1] + 1,
    ):
        raise ValueError("IMPLEMENTATION_RESIDUAL_AXIS")
    result = np.empty_like(products)
    for t in range(products.shape[0]):
        for j in range(products.shape[1]):
            fitted = coefficients[j, factors.shape[1]]
            for k in range(factors.shape[1]):
                fitted += factors[t, k] * coefficients[j, k]
            result[t, j] = products[t, j] - fitted
    return result


@njit((M,), cache=True, nogil=True)
def validate_covariance_kernel(covariance):
    if (
        covariance.shape[0] < 1
        or covariance.shape[0] != covariance.shape[1]
        or not np.all(np.isfinite(covariance))
    ):
        raise ValueError("IMPLEMENTATION_COVARIANCE")
    scale = max(1.0, np.max(np.abs(covariance)))
    if np.max(np.abs(covariance - covariance.T)) > 1e-10 * scale:
        raise ValueError("IMPLEMENTATION_COVARIANCE_SYMMETRY")
    if np.linalg.eigvalsh(covariance).min() < -1e-10 * scale:
        raise ValueError("IMPLEMENTATION_COVARIANCE_PSD")


@njit((M,), cache=True, nogil=True)
def design_diagnostics_kernel(covariance):
    """Rank and spectral condition of centered, standardized training factors."""
    validate_covariance_kernel(covariance)
    n = covariance.shape[0]
    correlation = np.empty((n, n))
    for i in range(n):
        if covariance[i, i] <= 0:
            raise ValueError("IMPLEMENTATION_DESIGN_CONSTANT")
        for j in range(n):
            correlation[i, j] = covariance[i, j] / np.sqrt(
                covariance[i, i] * covariance[j, j]
            )
    eigenvalues = np.linalg.eigvalsh(correlation)
    rank = np.count_nonzero(eigenvalues > eigenvalues[-1] * 1e-12)
    if rank < n:
        raise ValueError("IMPLEMENTATION_DESIGN_ILL_CONDITIONED")
    condition = np.sqrt(eigenvalues[-1] / eigenvalues[0])
    return rank, condition


@njit((M, M, M, V, V, V, V), cache=True, nogil=True)
def product_risk_kernel(
    beta, residual_covariance, class_covariance, class_means, weights, saa, target
):
    n, k = beta.shape
    if (
        residual_covariance.shape != (n, n)
        or class_covariance.shape != (k, k)
        or weights.size != n
        or class_means.size != k
        or saa.size != k
        or target.size != k
    ):
        raise ValueError("IMPLEMENTATION_RISK_AXIS")
    validate_covariance_kernel(residual_covariance)
    validate_covariance_kernel(class_covariance)
    if (
        not np.all(np.isfinite(beta))
        or not np.all(np.isfinite(class_means))
        or not np.all(np.isfinite(weights))
        or not np.all(np.isfinite(saa))
        or not np.all(np.isfinite(target))
    ):
        raise ValueError("IMPLEMENTATION_RISK_NONFINITE")
    covariance = np.empty((n, n))
    means = np.zeros(n)
    exposure = np.zeros(k)
    for i in range(n):
        for a in range(k):
            exposure[a] += weights[i] * beta[i, a]
            means[i] += beta[i, a] * class_means[a]
        for j in range(n):
            value = residual_covariance[i, j]
            for a in range(k):
                for b in range(k):
                    value += beta[i, a] * class_covariance[a, b] * beta[j, b]
            covariance[i, j] = value
    variance, residual, expected, active, implementation = 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(n):
        expected += weights[i] * means[i]
        for j in range(n):
            variance += weights[i] * covariance[i, j] * weights[j]
            residual += weights[i] * residual_covariance[i, j] * weights[j]
    for a in range(k):
        for b in range(k):
            active += (
                (exposure[a] - saa[a]) * class_covariance[a, b] * (exposure[b] - saa[b])
            )
            implementation += (
                (exposure[a] - target[a])
                * class_covariance[a, b]
                * (exposure[b] - target[b])
            )
    return (
        np.array(
            [
                expected,
                np.sqrt(max(0.0, variance)),
                np.sqrt(max(0.0, active + residual)),
                np.sqrt(max(0.0, implementation + residual)),
                np.sqrt(max(0.0, residual)),
            ]
        ),
        covariance,
        means,
        exposure,
    )


@njit((V, INT_VECTOR, V, V), cache=True, nogil=True)
def budget_kernel(weights, classes, target, upper):
    if (
        weights.size != classes.size
        or weights.size != upper.size
        or target.size < 1
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(upper))
    ):
        raise ValueError("IMPLEMENTATION_BUDGET_AXIS")
    actual = np.zeros(target.size)
    valid = weights.size == classes.size and upper.size == weights.size
    for i in range(weights.size):
        if not 0 <= classes[i] < target.size or not np.isfinite(weights[i]):
            raise ValueError("IMPLEMENTATION_BUDGET_AXIS")
        actual[classes[i]] += weights[i]
        valid = valid and 0 <= weights[i] <= upper[i] + 1e-8
    return (
        actual,
        valid
        and np.max(np.abs(actual - target)) <= 1e-8
        and abs(np.sum(weights) - 1.0) <= 1e-8,
    )


@njit((M, M, V, INT_VECTOR, V, V), cache=True, nogil=True)
def implementation_qp_kernel(
    product_returns, class_returns, target, classes, upper, unit_cost
):
    """Annual TE variance plus a declared unit-cost penalty, not a cash objective."""
    n, k = product_returns.shape[1], target.size
    if (
        product_returns.shape[0] != class_returns.shape[0]
        or class_returns.shape[1] != k
        or classes.size != n
        or upper.size != n
        or unit_cost.size != n
        or n < 1
        or k < 1
        or not np.all(np.isfinite(target))
        or np.any(target < 0)
        or abs(np.sum(target) - 1.0) > 1e-8
        or not np.all(np.isfinite(upper))
        or np.any(upper < 0)
        or np.any(upper > 1)
        or not np.all(np.isfinite(unit_cost))
        or np.any(unit_cost < 0)
        or np.any(classes < 0)
        or np.any(classes >= k)
    ):
        raise ValueError("IMPLEMENTATION_QP_INPUT")
    panel = np.empty((product_returns.shape[0], n + 1))
    panel[:, :n] = product_returns
    panel[:, n] = 0.0
    for t in range(panel.shape[0]):
        for j in range(k):
            panel[t, n] += class_returns[t, j] * target[j]
    covariance = covariance_kernel(panel, 0, panel.shape[0], 252.0)
    equality = np.zeros((k, n))
    initial = np.zeros(n)
    for j in range(k):
        remaining = target[j]
        for i in range(n):
            if classes[i] == j:
                equality[j, i] = 1.0
                initial[i] = min(upper[i], remaining)
                remaining -= initial[i]
        if remaining > 1e-8:
            return initial, 4, 0, np.full(4, np.nan)
    inequalities = np.zeros((2 * n, n))
    limits = np.zeros(2 * n)
    hessian = np.empty((n, n))
    linear = np.empty(n)
    for i in range(n):
        inequalities[i, i], inequalities[n + i, i] = 1.0, -1.0
        limits[n + i] = -upper[i]
        linear[i] = -2.0 * covariance[i, n] + unit_cost[i]
        for j in range(n):
            hessian[i, j] = 2.0 * covariance[i, j]
    return matrix_qp_kernel(
        hessian, linear, equality, target, inequalities, limits, initial, 500, 1e-8
    )


KERNELS = (
    covariance_kernel,
    design_diagnostics_kernel,
    residuals_kernel,
    validate_covariance_kernel,
    product_risk_kernel,
    budget_kernel,
    implementation_qp_kernel,
)
for _kernel in KERNELS:
    _kernel.disable_compile()


def audit():
    ready = _WARMED_PID == os.getpid() and all(
        len(k.nopython_signatures) == 1 and not k._can_compile for k in KERNELS
    )
    return {
        "version": "product-implementation-risk/1.0",
        "complete": ready,
        "python_fallback": 0,
        "request_time_compilation": 0,
    }


def require_ready():
    if not audit()["complete"]:
        raise RuntimeError("产品实施计算尚未完成本进程预热。")


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    p = np.array([[0.01, 0.02], [-0.02, 0.01], [0.015, -0.01]])
    c = covariance_kernel(p, 0, 3, 252.0)
    design_diagnostics_kernel(c)
    residuals_kernel(p, p, np.zeros((2, 3)))
    product_risk_kernel(
        np.eye(2),
        np.zeros((2, 2)),
        c,
        np.array([0.05, 0.02]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
        np.array([0.5, 0.5]),
    )
    budget_kernel(
        np.array([0.5, 0.5]),
        np.array([0, 1], dtype=np.int64),
        np.array([0.5, 0.5]),
        np.ones(2),
    )
    implementation_qp_kernel(
        p,
        p,
        np.array([0.5, 0.5]),
        np.array([0, 1], dtype=np.int64),
        np.ones(2),
        np.zeros(2),
    )
    _WARMED_PID = os.getpid()
    require_ready()
    return audit()
