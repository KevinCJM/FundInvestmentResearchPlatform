"""Fixed-signature LTCMA estimation kernels, independent of I/O and publication.

Sample covariance reuses the common reference/statistics implementation. NIW is
one conjugate update; state statistics preserve a common, typed state/date axis.
"""
from __future__ import annotations

import math
import os
import hashlib
import inspect

import numpy as np
from numba import njit, float64, int64, types

from .reference_evidence_kernels import annual_moments
from .kernels import cma_covariance_kernel
from backend.historical_regimes.numba_kernels import transition_matrix_kernel

V = types.Array(float64, 1, 'A', readonly=True)
M = types.Array(float64, 2, 'A', readonly=True)
I = types.Array(int64, 1, 'A', readonly=True)
VERSION = "ltcma-statistics/1.1.0"
_WARMED_PID = None
_WARMED_FINGERPRINT = None


@njit((float64, float64, float64), cache=True, nogil=True)
def beta_fraction(a, b, x):
    """Bounded modified-Lentz continued fraction for regularized incomplete beta."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    tiny = 1e-300
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 513):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            return h
    raise ValueError("LTCMA_BETA_NONCONVERGENCE")


@njit((float64, float64, float64), cache=True, nogil=True)
def regularized_beta(x, a, b):
    if not 0 <= x <= 1 or a <= 0 or b <= 0:
        raise ValueError("LTCMA_BETA_INPUT")
    if x == 0.0 or x == 1.0:
        return x
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return factor * beta_fraction(a, b, x) / a
    return 1.0 - factor * beta_fraction(b, a, 1.0 - x) / b


@njit((float64, float64), cache=True, nogil=True)
def student_t_quantile(probability, degrees):
    """Numerical quantile, not a normal 1.96 approximation; supports upper tail."""
    if (not np.isfinite(probability) or not np.isfinite(degrees)
            or not .5 <= probability <= .9999 or not 2.0 < degrees <= 200000):
        raise ValueError("LTCMA_T_QUANTILE_RANGE")
    if probability == .5:
        return 0.0
    lo, hi = 0.0, 1.0
    target_tail = 1.0 - probability
    for _ in range(30):
        tail = .5 * regularized_beta(degrees / (degrees + hi * hi), degrees / 2.0, .5)
        if tail <= target_tail:
            break
        hi *= 2.0
    else:
        raise ValueError("LTCMA_T_QUANTILE_BRACKET")
    for _ in range(80):
        mid = (lo + hi) / 2.0
        tail = .5 * regularized_beta(degrees / (degrees + mid * mid), degrees / 2.0, .5)
        if tail > target_tail:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


@njit((M,), cache=True, nogil=True)
def statistical_covariance_diagnostics(covariance):
    """PSD diagnostics with deterministic cash allowed and correlation undefined."""
    n = covariance.shape[0]
    if n < 1 or n > 30 or covariance.shape[1] != n:
        raise ValueError("LTCMA_COVARIANCE_SHAPE")
    for i in range(n):
        for j in range(n):
            if not np.isfinite(covariance[i, j]) or abs(covariance[i, j] - covariance[j, i]) > 1e-10:
                raise ValueError("LTCMA_COVARIANCE_INVALID")
        if covariance[i, i] < 0 or covariance[i, i] > 9:
            raise ValueError("LTCMA_COVARIANCE_DIAGONAL")
    minimum = np.linalg.eigvalsh(covariance)[0]
    if minimum < -1e-10:
        raise ValueError("LTCMA_COVARIANCE_NOT_PSD")
    vol = np.empty(n, dtype=np.float64)
    corr = np.full((n, n), np.nan, dtype=np.float64)
    active = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(n):
        vol[i] = math.sqrt(covariance[i, i])
        if vol[i] > 0:
            active[count] = i
            count += 1
    for i in range(n):
        for j in range(n):
            if vol[i] > 0 and vol[j] > 0:
                corr[i, j] = covariance[i, j] / (vol[i] * vol[j])
            elif abs(covariance[i, j]) > 1e-12:
                raise ValueError("LTCMA_ZERO_VARIANCE_CROSS_TERM")
    block = np.empty((count, count))
    for i in range(count):
        for j in range(count):
            block[i, j] = corr[active[i], active[j]]
    correlation_minimum = np.linalg.eigvalsh(block)[0] if count else 0.0
    if correlation_minimum < -1e-8:
        raise ValueError("LTCMA_CORRELATION_NOT_PSD")
    return vol, corr, correlation_minimum


@njit((V, M), cache=True, nogil=True)
def manual_covariance_with_cash(volatility, correlation):
    """Reuse the original risk block; declared deterministic cash contributes zero."""
    n = volatility.size
    if n < 1 or n > 30 or correlation.shape != (n, n):
        raise ValueError("CMA_MATRIX_SHAPE")
    indices = np.empty(n, dtype=np.int64)
    count = 0
    for i in range(n):
        if not np.isfinite(volatility[i]) or volatility[i] < 0 or volatility[i] > 3:
            raise ValueError("CMA_VOLATILITY")
        if volatility[i] > 0:
            indices[count] = i
            count += 1
        for j in range(n):
            if not np.isfinite(correlation[i, j]):
                raise ValueError("CMA_CORRELATION_RANGE")
            if (volatility[i] == 0 or volatility[j] == 0) and abs(correlation[i, j] - (1.0 if i == j else 0.0)) > 1e-10:
                raise ValueError("LTCMA_CASH_CORRELATION")
    if count == n:
        return cma_covariance_kernel(volatility, correlation)
    output = np.zeros((n, n))
    if count == 0:
        return output, 0.0
    block_vol = np.empty(count)
    block_corr = np.empty((count, count))
    for i in range(count):
        block_vol[i] = volatility[indices[i]]
        for j in range(count):
            block_corr[i, j] = correlation[indices[i], indices[j]]
    covariance, minimum = cma_covariance_kernel(block_vol, block_corr)
    for i in range(count):
        for j in range(count):
            output[indices[i], indices[j]] = covariance[i, j]
    return output, minimum


@njit((M, float64), cache=True, nogil=True)
def historical_estimate(returns, shrinkage):
    means, covariance, _, _ = annual_moments(returns, shrinkage, np.int64(252))
    rows, n = returns.shape
    # Explicit iid marginal intervals for the sample mean, not joint coverage.
    quantile = student_t_quantile(.975, float(rows - 1))
    mean_covariance = np.empty((n, n))
    half_width = np.empty(n)
    for i in range(n):
        for j in range(n):
            mean_covariance[i, j] = covariance[i, j] * 252.0 / rows
        half_width[i] = quantile * math.sqrt(mean_covariance[i, i])
    return means, covariance, mean_covariance, half_width


@njit((M, V, M, float64, float64), cache=True, nogil=True)
def niw_update(returns, prior_mean, prior_psi, kappa, nu):
    rows, n = returns.shape
    if (prior_mean.size != n or prior_psi.shape != (n, n) or rows < 20
            or not np.isfinite(kappa) or kappa <= 0 or not np.isfinite(nu) or nu <= n + 1):
        raise ValueError("LTCMA_NIW_PRIOR")
    for i in range(n):
        if not np.isfinite(prior_mean[i]):
            raise ValueError("LTCMA_NIW_PRIOR")
        for j in range(n):
            if not np.isfinite(prior_psi[i, j]) or abs(prior_psi[i, j] - prior_psi[j, i]) > 1e-12:
                raise ValueError("LTCMA_NIW_PRIOR")
    np.linalg.cholesky(prior_psi)  # A genuine PD prior is required, no invented jitter.
    sample_mean, sample_covariance, _, _ = annual_moments(returns, 0., np.int64(252))
    posterior_kappa, posterior_nu = kappa + rows, nu + rows
    degrees = posterior_nu - n + 1.0
    quantile = student_t_quantile(.975, degrees)
    posterior_mean = np.empty(n)
    psi = np.empty((n, n))
    annual_mean = np.empty(n)
    annual_risk = np.empty((n, n))
    mean_uncertainty = np.empty((n, n))
    half_width = np.empty(n)
    for i in range(n):
        posterior_mean[i] = (kappa * prior_mean[i] + rows * sample_mean[i] / 252.0) / posterior_kappa
        annual_mean[i] = 252.0 * posterior_mean[i]
        for j in range(n):
            psi[i, j] = (prior_psi[i, j] + sample_covariance[i, j] * ((rows - 1.0) / 252.0)
                + kappa * rows / posterior_kappa * (sample_mean[i] / 252.0 - prior_mean[i])
                * (sample_mean[j] / 252.0 - prior_mean[j]))
            base_risk = psi[i, j] / (posterior_nu - n - 1.0)
            annual_risk[i, j] = 252.0 * base_risk
            mean_uncertainty[i, j] = 252.0 ** 2 * base_risk / posterior_kappa
        half_width[i] = 252.0 * quantile * math.sqrt(psi[i, i] / (posterior_kappa * degrees))
    return annual_mean, annual_risk, mean_uncertainty, half_width, posterior_mean, psi, posterior_kappa, posterior_nu


@njit((V, M, float64, float64), cache=True, nogil=True)
def recenter_niw_prior(annual_mean, annual_covariance, mean_strength, risk_strength):
    n = annual_mean.size
    if annual_covariance.shape != (n, n) or mean_strength <= 0 or risk_strength <= 0:
        raise ValueError("LTCMA_NIW_PRIOR")
    return annual_mean / 252., annual_covariance * (risk_strength / 252.), mean_strength, n + 1. + risk_strength


@njit((M, I, int64, float64), cache=True, nogil=True)
def conditional_state_moments(returns, states, count, shrinkage):
    """One common state axis and indexed Welford scatter, no copied state panels."""
    rows, n = returns.shape
    if states.size != rows or count < 1 or count > 60 or not 0 <= shrinkage <= 1:
        raise ValueError("LTCMA_STATE_AXIS")
    counts = np.zeros(count, dtype=np.int64)
    means = np.zeros((count, n))
    scatter = np.zeros((count, n, n))
    delta = np.empty(n)
    for t in range(rows):
        state = states[t]
        if state == -1:
            continue
        if state < 0 or state >= count:
            raise ValueError("LTCMA_STATE_CODE")
        counts[state] += 1
        for i in range(n):
            value = returns[t, i]
            if not np.isfinite(value) or value <= -1:
                raise ValueError("LTCMA_STATE_RETURN")
            delta[i] = value - means[state, i]
            means[state, i] += delta[i] / counts[state]
        for i in range(n):
            for j in range(n):
                scatter[state, i, j] += delta[i] * (returns[t, j] - means[state, j])
    for s in range(count):
        if counts[s]:
            for i in range(n):
                for j in range(i, n):
                    value = (scatter[s, i, j] + scatter[s, j, i]) / (2.0 * counts[s])
                    if i != j:
                        value *= 1.0 - shrinkage
                    scatter[s, i, j], scatter[s, j, i] = value, value
    return counts, means, scatter


@njit((I,), cache=True, nogil=True)
def occupancy_probabilities(counts):
    total = 0
    for i in range(counts.size):
        if counts[i] < 0:
            raise ValueError("LTCMA_STATE_COUNTS")
        total += counts[i]
    if total < 20:
        raise ValueError("LTCMA_STATE_SAMPLE")
    result = np.empty(counts.size)
    for i in range(counts.size):
        result[i] = counts[i] / total
    return result


@njit((V, M, int64), cache=True, nogil=True)
def annualize_moments(mean, covariance, periods):
    if periods != 252:
        raise ValueError("LTCMA_FREQUENCY")
    return mean * periods, covariance * periods


@njit((I, int64), cache=True, nogil=True)
def regime_transition_diagnostics_kernel(states, state_count):
    """Reuse transition counts; unknown labels break adjacency and missing rows stay unknown.

    The existing shared counter requires mutable C-contiguous int64. Its one
    code-vector ABI copy is explicit; no return panel or per-state panel is copied.
    Status: 0 unique irreducible stationary distribution; 1 missing row;
    2 reducibility not certified; 3 numerical stationary solve unavailable.
    """
    if state_count < 1 or state_count > 60:
        raise ValueError("LTCMA_STATE_AXIS")
    for value in states:
        if value < -1 or value >= state_count:
            raise ValueError("LTCMA_STATE_CODE")
    counts, probabilities = transition_matrix_kernel(states.copy(), state_count)
    duration = np.full(state_count, np.nan)
    stationary = np.full(state_count, np.nan)
    status = 0
    for i in range(state_count):
        total = 0
        for j in range(state_count):
            total += counts[i, j]
        if total == 0:
            probabilities[i, :] = np.nan
            status = 1
        elif probabilities[i, i] < 1.0:
            duration[i] = 1.0 / (1.0 - probabilities[i, i])
    if status:
        return counts, probabilities, duration, stationary, status
    reach = np.zeros((state_count, state_count), dtype=np.bool_)
    for i in range(state_count):
        for j in range(state_count):
            reach[i, j] = probabilities[i, j] > 0 or i == j
    for k in range(state_count):
        for i in range(state_count):
            for j in range(state_count):
                reach[i, j] = reach[i, j] or (reach[i, k] and reach[k, j])
    if not np.all(reach):
        return counts, probabilities, duration, stationary, 2
    system = probabilities.T.copy()
    target = np.zeros(state_count)
    for i in range(state_count):
        system[i, i] -= 1.0
        system[state_count - 1, i] = 1.0
    target[state_count - 1] = 1.0
    try:
        candidate = np.linalg.solve(system, target)
    except Exception:
        return counts, probabilities, duration, stationary, 3
    if not np.isfinite(candidate).all() or np.any(candidate < 0) or np.any(candidate > 1) or abs(np.sum(candidate) - 1.0) > 1e-9:
        return counts, probabilities, duration, stationary, 3
    for j in range(state_count):
        residual = -candidate[j]
        for i in range(state_count):
            residual += candidate[i] * probabilities[i, j]
        if abs(residual) > 1e-9:
            return counts, probabilities, duration, stationary, 3
    return counts, probabilities, duration, candidate, 0


@njit((int64, int64), cache=True, nogil=True)
def sample_horizon_diagnostics_kernel(observations, horizon_years):
    if observations < 0 or horizon_years < 1 or horizon_years > 30:
        raise ValueError("LTCMA_SAMPLE_HORIZON")
    years = observations / 252.0
    return years, years < 3.0, years / horizon_years


KERNELS = (beta_fraction, regularized_beta, student_t_quantile, statistical_covariance_diagnostics,
           manual_covariance_with_cash, historical_estimate, niw_update, recenter_niw_prior,
           conditional_state_moments, occupancy_probabilities, annualize_moments,
           regime_transition_diagnostics_kernel, sample_horizon_diagnostics_kernel)
for kernel in KERNELS:
    kernel.disable_compile()


def execution_audit():
    dispatchers = (*KERNELS, transition_matrix_kernel)
    fingerprint = hashlib.sha256(repr([(inspect.getsource(k.py_func), [str(s) for s in k.signatures]) for k in dispatchers]).encode()).hexdigest()
    compiled = all(k.nopython_signatures and not k._can_compile for k in dispatchers)
    ready = bool(compiled and _WARMED_PID == os.getpid() and fingerprint == _WARMED_FINGERPRINT)
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "fingerprint": fingerprint, "complete": ready, "fully_warmed": ready,
            "nopython": bool(compiled), "object_mode": 0, "python_fallback": 0,
            "request_time_compilation": 0,
            "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in dispatchers}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("LTCMA_STATISTICS_NOT_READY: 统计模型尚未完成本进程预热。")


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    _WARMED_PID = None
    raw = np.column_stack((np.sin(np.arange(40)) * .01, np.cos(np.arange(40)) * .002))
    returns = raw[::-1]
    returns.flags.writeable = False
    mean, cov, _, _ = historical_estimate(returns, .1)
    prior = recenter_niw_prior(mean, cov, 20., 20.)
    niw_update(returns, *prior)
    counts, _, _ = conditional_state_moments(returns, np.arange(40, dtype=np.int64) % 2, 2, 0.)
    occupancy_probabilities(counts)
    statistical_covariance_diagnostics(np.diag(np.array([0., .01])))
    manual_covariance_with_cash(np.array([0., .1]), np.eye(2))
    annualize_moments(mean / 252., cov / 252., 252)
    regime_transition_diagnostics_kernel(np.arange(40, dtype=np.int64) % 2, 2)
    sample_horizon_diagnostics_kernel(40, 10)
    _WARMED_PID = os.getpid()
    _WARMED_FINGERPRINT = execution_audit()["fingerprint"]
    return execution_audit()
