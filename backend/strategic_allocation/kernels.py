"""Fixed-signature forward moments and bounded policy candidate evaluation.

History is read-only, arbitrary-stride input. Small owned constraint arrays use
exactly the existing optimizer's ABI; only outputs/work buffers are allocated.
"""
from __future__ import annotations

import os

import numpy as np
from numba import float64, int64, njit, types

from backend.cal_indicators.typed_numba_kernels import covariance_2d
from backend.optimizer import repair_weights_kernel

V = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
CV = float64[::1]
CM = float64[:, ::1]
GM = types.uint8[:, ::1]
_WARMED_PID: int | None = None
VERSION = "forward-policy-moments/1.2.0"


@njit((M, float64, int64), cache=True, nogil=True)
def historical_risk_kernel(returns, shrinkage, periods):
    """Sample covariance with an explicit, fixed diagonal shrinkage coefficient."""
    rows, columns = returns.shape
    if rows < 20 or columns < 1 or not 0 <= shrinkage <= 1 or periods < 1:
        raise ValueError("RISK_SAMPLE_INVALID")
    means = np.zeros(columns, dtype=np.float64)
    for t in range(rows):
        for i in range(columns):
            if not np.isfinite(returns[t, i]) or returns[t, i] <= -1:
                raise ValueError("RISK_SAMPLE_INVALID")
            means[i] += returns[t, i]
    covariance = covariance_2d(returns)
    volatility = np.empty(columns, dtype=np.float64)
    correlation = np.empty((columns, columns), dtype=np.float64)
    for i in range(columns):
        means[i] *= periods / rows
        for j in range(columns):
            covariance[i, j] *= periods * (1.0 if i == j else 1.0 - shrinkage)
        volatility[i] = np.sqrt(covariance[i, i])
        if not np.isfinite(volatility[i]) or volatility[i] <= 1e-12:
            raise ValueError("RISK_ZERO_VARIANCE")
    for i in range(columns):
        for j in range(columns):
            correlation[i, j] = covariance[i, j] / (volatility[i] * volatility[j])
    return covariance, volatility, correlation, means


@njit((V, M), cache=True, nogil=True)
def cma_covariance_kernel(volatility, correlation):
    """Reject invalid matrices rather than manufacturing a nearest PSD matrix."""
    count = volatility.size
    if count < 1 or correlation.shape != (count, count):
        raise ValueError("CMA_MATRIX_SHAPE")
    covariance = np.empty((count, count), dtype=np.float64)
    checked_correlation = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        if not np.isfinite(volatility[i]) or volatility[i] <= 0:
            raise ValueError("CMA_VOLATILITY")
        if abs(correlation[i, i] - 1) > 1e-10:
            raise ValueError("CMA_CORRELATION_DIAGONAL")
        for j in range(count):
            value = correlation[i, j]
            if not np.isfinite(value) or value < -1 - 1e-12 or value > 1 + 1e-12:
                raise ValueError("CMA_CORRELATION_RANGE")
            if abs(value - correlation[j, i]) > 1e-10:
                raise ValueError("CMA_CORRELATION_SYMMETRY")
            covariance[i, j] = volatility[i] * value * volatility[j]
            checked_correlation[i, j] = value
    eigenvalues = np.linalg.eigvalsh(checked_correlation)
    if eigenvalues[0] < -1e-10:
        raise ValueError("CMA_CORRELATION_PSD")
    return covariance, eigenvalues[0]


@njit((V, V, M, V, float64, float64), cache=True, nogil=True)
def portfolio_moments_kernel(weights, means, covariance, uncertainty, aversion, penalty):
    """One shared moment evaluation, including its Euler risk decomposition."""
    count = weights.size
    if means.size != count or uncertainty.size != count or covariance.shape != (count, count):
        raise ValueError("POLICY_AXIS_MISMATCH")
    if not np.isfinite(aversion) or not np.isfinite(penalty) or aversion < 0 or penalty < 0:
        raise ValueError("POLICY_PARAMETERS")
    variance, expected, haircut = 0.0, 0.0, 0.0
    contributions = np.empty(count, dtype=np.float64)
    total = 0.0
    for i in range(count):
        if (not np.isfinite(weights[i]) or weights[i] < -1e-10
                or not np.isfinite(means[i]) or not np.isfinite(uncertainty[i]) or uncertainty[i] < 0):
            raise ValueError("POLICY_INPUT_INVALID")
        total += weights[i]
        expected += weights[i] * means[i]
        haircut += abs(weights[i]) * uncertainty[i] * penalty
        marginal = 0.0
        for j in range(count):
            if not np.isfinite(covariance[i, j]):
                raise ValueError("POLICY_INPUT_INVALID")
            marginal += covariance[i, j] * weights[j]
        contributions[i] = weights[i] * marginal
        variance += contributions[i]
    if abs(total - 1.0) > 1e-8 or variance < -1e-10:
        raise ValueError("POLICY_INPUT_INVALID")
    variance = max(0.0, variance)
    for i in range(count):
        contributions[i] = contributions[i] / variance if variance > 1e-20 else np.nan
    metrics = np.array([expected, np.sqrt(variance), expected - haircut,
                        expected - 0.5 * aversion * variance,
                        expected - haircut - 0.5 * aversion * variance])
    return metrics, contributions


@njit((V, V, M), cache=True, nogil=True)
def expected_active_risk_kernel(weights, baseline_weights, covariance):
    """Forward tracking error of target weights versus the frozen policy weights."""
    count = weights.size
    if baseline_weights.size != count or covariance.shape != (count, count) or count < 1:
        raise ValueError("POLICY_AXIS_MISMATCH")
    target_total = 0.0
    baseline_total = 0.0
    for i in range(count):
        if (not np.isfinite(weights[i]) or not np.isfinite(baseline_weights[i])
                or weights[i] < -1e-10 or baseline_weights[i] < -1e-10):
            raise ValueError("POLICY_INPUT_INVALID")
        target_total += weights[i]
        baseline_total += baseline_weights[i]
    if abs(target_total - 1.0) > 1e-8 or abs(baseline_total - 1.0) > 1e-8:
        raise ValueError("POLICY_INPUT_INVALID")
    variance = 0.0
    for i in range(count):
        delta_i = weights[i] - baseline_weights[i]
        for j in range(count):
            value = covariance[i, j]
            if not np.isfinite(value):
                raise ValueError("POLICY_INPUT_INVALID")
            variance += delta_i * value * (weights[j] - baseline_weights[j])
    if variance < -1e-10:
        raise ValueError("POLICY_INPUT_INVALID")
    return np.sqrt(max(variance, 0.0))


@njit((V, V, V), cache=True, nogil=True)
def expected_excess_return_kernel(weights, benchmark_weights, means):
    if weights.size == 0 or weights.size != means.size or benchmark_weights.size != means.size:
        raise ValueError("POLICY_AXIS_MISMATCH")
    result = 0.0
    for i in range(means.size):
        if not np.isfinite(weights[i]) or not np.isfinite(benchmark_weights[i]) or not np.isfinite(means[i]):
            raise ValueError("POLICY_INPUT_INVALID")
        result += (weights[i] - benchmark_weights[i]) * means[i]
    return result


@njit((V, V), cache=True, nogil=True)
def risk_budget_error_kernel(contributions, budget):
    """Squared distance between signed Euler risk shares and declared targets."""
    if contributions.size == 0 or contributions.size != budget.size:
        raise ValueError("POLICY_AXIS_MISMATCH")
    error, total = 0.0, 0.0
    for i in range(budget.size):
        if not np.isfinite(budget[i]) or budget[i] < 0 or not np.isfinite(contributions[i]):
            raise ValueError("POLICY_RISK_BUDGET_INVALID")
        total += budget[i]
        difference = contributions[i] - budget[i]
        error += difference * difference
    if abs(total - 1.0) > 1e-8:
        raise ValueError("POLICY_RISK_BUDGET_INVALID")
    return error


@njit((V, M, V, CM, GM, CV, CV, float64, float64, float64, float64, V, float64, float64, int64, int64, V),
      cache=True, nogil=True)
def policy_candidates_with_budget_kernel(means, covariance, uncertainty, bounds, groups, group_low, group_high,
                                         aversion, penalty, min_return, max_volatility, benchmark_weights,
                                         benchmark_te_limit, target_excess, samples, seed, risk_budget):
    """One finite search for all objectives; optional risk-budget fit is not an exact solution."""
    count = means.size
    if (count < 1 or count > 30 or bounds.shape != (count, 2) or groups.shape[1] != count
            or groups.shape[0] != group_low.size or group_low.size != group_high.size
            or samples < 1 or samples > 5000 or benchmark_weights.size not in (0, count)
            or risk_budget.size not in (0, count)):
        raise ValueError("POLICY_SEARCH_SHAPE")
    has_budget = int(risk_budget.size == count)
    if has_budget:
        budget_total = 0.0
        for i in range(count):
            if not np.isfinite(risk_budget[i]) or risk_budget[i] < 0:
                raise ValueError("POLICY_RISK_BUDGET_INVALID")
            budget_total += risk_budget[i]
        if abs(budget_total - 1.0) > 1e-8:
            raise ValueError("POLICY_RISK_BUDGET_INVALID")
    method_count = 4 + has_budget
    selected_weights = np.zeros((method_count, count), dtype=np.float64)
    selected_metrics = np.full((method_count, 5), np.nan)
    selected_contributions = np.full((method_count, count), np.nan)
    best = np.full(method_count, -np.inf)
    raw = np.empty(count, dtype=np.float64)
    state = int(seed)
    accepted = 0
    # The local PRNG never reseeds NumPy's process/thread random generator.
    has_benchmark = int(benchmark_weights.size == count)
    if has_benchmark:
        if not np.isfinite(benchmark_te_limit) or benchmark_te_limit < 0 or not np.isfinite(target_excess):
            raise ValueError("POLICY_PARAMETERS")
        expected_active_risk_kernel(benchmark_weights, benchmark_weights, covariance)
    for candidate in range(samples + count + 1 + has_benchmark):
        offset = candidate - has_benchmark
        if has_benchmark and candidate == 0:
            for i in range(count):
                raw[i] = benchmark_weights[i]
        elif offset == 0:
            for i in range(count):
                raw[i] = 1.0 / count
        elif offset <= count:
            for i in range(count):
                raw[i] = 1.0 if i == offset - 1 else 0.0
        else:
            total = 0.0
            for i in range(count):
                state = (state * 1664525 + 1013904223) % 4294967296
                raw[i] = -np.log((state + 0.5) / 4294967296.0)
                total += raw[i]
            for i in range(count):
                raw[i] /= total
        weights, status = repair_weights_kernel(raw, bounds, groups, group_low, group_high, 0.0)
        if status != 0:
            continue
        valid = abs(np.sum(weights) - 1) <= 1e-8
        for i in range(count):
            valid = valid and bounds[i, 0] - 1e-8 <= weights[i] <= bounds[i, 1] + 1e-8
        for g in range(groups.shape[0]):
            value = 0.0
            for i in range(count):
                if groups[g, i]:
                    value += weights[i]
            valid = valid and group_low[g] - 1e-8 <= value <= group_high[g] + 1e-8
        if not valid:
            continue
        metrics, contributions = portfolio_moments_kernel(weights, means, covariance, uncertainty, aversion, penalty)
        if metrics[0] < min_return - 1e-10 or metrics[1] > max_volatility + 1e-10:
            continue
        if has_benchmark:
            if expected_active_risk_kernel(weights, benchmark_weights, covariance) > benchmark_te_limit + 1e-10:
                continue
            if expected_excess_return_kernel(weights, benchmark_weights, means) < target_excess - 1e-10:
                continue
        accepted += 1
        scores = (-metrics[1], metrics[3], metrics[4], metrics[0])
        for method in range(4):
            if scores[method] > best[method] + 1e-14:
                best[method] = scores[method]
                selected_weights[method, :] = weights
                selected_metrics[method, :] = metrics
                selected_contributions[method, :] = contributions
        if has_budget:
            # A zero-variance portfolio has undefined Euler risk shares. Do not
            # manufacture zero contributions or a successful budget fit.
            defined = True
            for i in range(count):
                if not np.isfinite(contributions[i]):
                    defined = False
                    break
            squared_error = risk_budget_error_kernel(contributions, risk_budget) if defined else np.inf
            if -squared_error > best[4] + 1e-14:
                best[4] = -squared_error
                selected_weights[4, :] = weights
                selected_metrics[4, :] = metrics
                selected_contributions[4, :] = contributions
    return selected_weights, selected_metrics, selected_contributions, accepted


@njit((V, M, V, CM, GM, CV, CV, float64, float64, float64, float64, V, float64, float64, int64, int64),
      cache=True, nogil=True)
def policy_candidates_kernel(means, covariance, uncertainty, bounds, groups, group_low, group_high,
                             aversion, penalty, min_return, max_volatility, benchmark_weights,
                             benchmark_te_limit, target_excess, samples, seed):
    """Preserve the existing four-candidate ABI via the single current search."""
    return policy_candidates_with_budget_kernel(
        means, covariance, uncertainty, bounds, groups, group_low, group_high,
        aversion, penalty, min_return, max_volatility, benchmark_weights,
        benchmark_te_limit, target_excess, samples, seed, np.empty(0, dtype=np.float64))


KERNELS = (historical_risk_kernel, cma_covariance_kernel, portfolio_moments_kernel,
           expected_active_risk_kernel, expected_excess_return_kernel,
           policy_candidates_with_budget_kernel, policy_candidates_kernel, risk_budget_error_kernel)
for dispatcher in KERNELS:
    dispatcher.disable_compile()


def execution_audit():
    complete = _WARMED_PID == os.getpid() and all(
        len(k.signatures) == 1 and k.nopython_signatures
        and not any(v.objectmode for v in k.overloads.values()) for k in KERNELS
    )
    return {"engine": VERSION, "backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "nopython": bool(complete), "fully_warmed": bool(complete), "complete": bool(complete),
            "object_mode": 0, "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in KERNELS}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("战略配置计算内核未完成启动预热。")


def warm_strategic_kernels():
    global _WARMED_PID
    _WARMED_PID = None
    sample = np.arange(60, dtype=np.float64).reshape(30, 2) * 0.0001
    historical_risk_kernel(sample, 0.1, 252)
    cov, _ = cma_covariance_kernel(np.array([0.15, 0.05]), np.eye(2))
    expected_active_risk_kernel(np.array([0.55, 0.45]), np.array([0.5, 0.5]), cov)
    policy_candidates_kernel(np.array([0.06, 0.03]), cov, np.array([0.02, 0.005]),
                             np.array([[0., 1.], [0., 1.]]), np.zeros((0, 2), dtype=np.uint8),
                             np.empty(0), np.empty(0), 5., 1., 0., 1., np.empty(0), 1., 0., 200, 42)
    policy_candidates_with_budget_kernel(np.array([0.06, 0.03]), cov, np.array([0.02, 0.005]),
                             np.array([[0., 1.], [0., 1.]]), np.zeros((0, 2), dtype=np.uint8),
                             np.empty(0), np.empty(0), 5., 1., 0., 1., np.empty(0), 1., 0., 200, 42,
                             np.array([0.5, 0.5]))
    risk_budget_error_kernel(np.array([0.6, 0.4]), np.array([0.5, 0.5]))
    expected_excess_return_kernel(np.array([0.6, 0.4]), np.array([0.5, 0.5]), np.array([0.06, 0.03]))
    from .goal_kernels import warm_goal_kernels
    warm_goal_kernels()
    _WARMED_PID = os.getpid()
    audit = execution_audit()
    if not audit["complete"]:
        _WARMED_PID = None
        raise RuntimeError("战略配置固定签名预热失败。")
    return audit
