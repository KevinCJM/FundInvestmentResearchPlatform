"""Fixed readonly arbitrary-stride ABI for explicit BL and scenario CMA.

Only outputs and small linear algebra workspaces are allocated. There is no
inverse, covariance repair, persistence, policy search, or Python fallback.
"""
from __future__ import annotations

import os

import numpy as np
from numba import float64, njit, types

from .kernels import cma_covariance_kernel

V = types.Array(float64, 1, "A", readonly=True)
M = types.Array(float64, 2, "A", readonly=True)
D = types.Array(float64, 3, "A", readonly=True)
VERSION = "explicit-cma-models/1.1.0"
_WARMED_PID: int | None = None


@njit((M,), cache=True, nogil=True)
def covariance_diagnostics_kernel(covariance):
    """Convert an explicit covariance at the boundary; reuse the CMA PSD gate."""
    count = covariance.shape[0]
    if count < 1 or count > 30 or covariance.shape[1] != count:
        raise ValueError("CMA_MODEL_MATRIX_AXIS")
    volatility = np.empty(count, dtype=np.float64)
    correlation = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        if not np.isfinite(covariance[i, i]) or covariance[i, i] <= 0 or covariance[i, i] > 9:
            raise ValueError("CMA_MODEL_VARIANCE_RANGE")
        volatility[i] = np.sqrt(covariance[i, i])
    for i in range(count):
        for j in range(count):
            if not np.isfinite(covariance[i, j]):
                raise ValueError("CMA_MODEL_COVARIANCE_FINITE")
            correlation[i, j] = covariance[i, j] / (volatility[i] * volatility[j])
    _, minimum_eigenvalue = cma_covariance_kernel(volatility, correlation)
    return volatility, correlation, minimum_eigenvalue


@njit((V,), cache=True, nogil=True)
def validate_effective_returns_kernel(means):
    for value in means:
        if not np.isfinite(value) or value < -0.5 or value > 2:
            raise ValueError("CMA_MODEL_RETURN_RANGE")


@njit((M, V, M, V, V, float64, float64, float64), cache=True, nogil=True)
def black_litterman_update_kernel(covariance, market_weights, picks, view_returns, view_std,
                                 delta, tau, risk_free_rate):
    """Joint Gaussian mean update; risk remains the caller's explicit Sigma.

    q_excess = q_total - rf * P1 also handles relative (+1/-1) views.
    The posterior mean covariance uses Joseph's equivalent PSD-stable form.
    """
    count, views = market_weights.size, view_returns.size
    if (count < 1 or count > 30 or covariance.shape != (count, count)
            or views > 60 or picks.shape != (views, count) or view_std.size != views):
        raise ValueError("CMA_BL_AXIS")
    if (not np.isfinite(delta) or delta <= 0 or not np.isfinite(tau)
            or tau <= 0 or not np.isfinite(risk_free_rate) or not -0.5 <= risk_free_rate <= 2):
        raise ValueError("CMA_BL_PARAMETERS")
    covariance_diagnostics_kernel(covariance)
    total = 0.0
    for weight in market_weights:
        if not np.isfinite(weight) or weight < 0 or weight > 1:
            raise ValueError("CMA_BL_MARKET_WEIGHTS")
        total += weight
    if abs(total - 1) > 1e-8:
        raise ValueError("CMA_BL_MARKET_WEIGHTS")
    prior = np.zeros(count, dtype=np.float64)
    prior_covariance = np.empty((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(count):
            prior[i] += delta * covariance[i, j] * market_weights[j]
            prior_covariance[i, j] = tau * covariance[i, j]
            if not np.isfinite(prior_covariance[i, j]):
                raise ValueError("CMA_BL_PRIOR_FINITE")
        if not np.isfinite(prior[i]):
            raise ValueError("CMA_BL_PRIOR_FINITE")
    residual = np.empty(views, dtype=np.float64)
    omega = np.empty(views, dtype=np.float64)
    for v in range(views):
        gross, exposure, projected = 0.0, 0.0, 0.0
        for i in range(count):
            p = picks[v, i]
            if not np.isfinite(p):
                raise ValueError("CMA_BL_VIEW_PICK")
            gross += abs(p)
            exposure += p
            projected += p * prior[i]
        absolute = abs(exposure - 1.0) <= 1e-10
        if not 1e-12 <= gross <= 4.0 or (not absolute and abs(exposure) > 1e-10):
            raise ValueError("CMA_BL_VIEW_PICK")
        q = view_returns[v]
        if not np.isfinite(q) or (absolute and not -0.5 <= q <= 2) or (not absolute and not -2.5 <= q <= 2.5):
            raise ValueError("CMA_BL_VIEW_RETURN")
        omega[v] = view_std[v] * view_std[v]
        if not np.isfinite(view_std[v]) or view_std[v] <= 0 or not np.isfinite(omega[v]) or omega[v] <= 0:
            raise ValueError("CMA_BL_VIEW_STD")
        residual[v] = q - risk_free_rate * exposure - projected
    means = prior + risk_free_rate
    posterior = prior_covariance.copy()
    gain_transpose = np.empty((views, count), dtype=np.float64)
    view_prior_variance = np.empty(views, dtype=np.float64)
    if views:
        # Small workspaces are owned/contiguous for LAPACK; inputs stay shared.
        cross = np.zeros((count, views), dtype=np.float64)
        system = np.zeros((views, views), dtype=np.float64)
        for i in range(count):
            for v in range(views):
                for j in range(count):
                    cross[i, v] += prior_covariance[i, j] * picks[v, j]
        for v in range(views):
            for u in range(views):
                for i in range(count):
                    system[v, u] += picks[v, i] * cross[i, u]
            view_prior_variance[v] = system[v, v]
            system[v, v] += omega[v]
        # Cholesky is a numerical SPD gate, never a regularizer or inverse.
        np.linalg.cholesky(system)
        solved = np.linalg.solve(system, cross.T.copy())
        gain_transpose = solved
        transform = np.eye(count)
        for i in range(count):
            for v in range(views):
                means[i] += solved[v, i] * residual[v]
                for j in range(count):
                    transform[i, j] -= solved[v, i] * picks[v, j]
        propagated = transform @ prior_covariance @ transform.T
        for i in range(count):
            for j in range(count):
                value = propagated[i, j]
                for v in range(views):
                    value += solved[v, i] * omega[v] * solved[v, j]
                if not np.isfinite(value):
                    raise ValueError("CMA_BL_POSTERIOR_FINITE")
                posterior[i, j] = value
        for i in range(count):
            for j in range(i):
                value = (posterior[i, j] + posterior[j, i]) * 0.5
                posterior[i, j], posterior[j, i] = value, value
    validate_effective_returns_kernel(means)
    return means, posterior, prior, gain_transpose, residual, view_prior_variance, omega


@njit((M, V, M, V, V, float64, float64, float64), cache=True, nogil=True)
def black_litterman_kernel(covariance, market_weights, picks, view_returns, view_std,
                          delta, tau, risk_free_rate):
    """Keep the three-result ABI; there is only one Gaussian update implementation."""
    result = black_litterman_update_kernel(covariance, market_weights, picks, view_returns,
                                          view_std, delta, tau, risk_free_rate)
    return result[0], result[1], result[2]


@njit((M, V, V, M, V, V, V), cache=True, nogil=True)
def black_litterman_diagnostics_kernel(covariance, market_weights, prior, gain_transpose,
                                      residual, view_prior_variance, omega):
    """Explain an already solved update; influence is not a probability or blend weight."""
    count, views = prior.size, residual.size
    if (covariance.shape != (count, count) or market_weights.size != count
            or gain_transpose.shape != (views, count) or view_prior_variance.size != views or omega.size != views):
        raise ValueError("CMA_BL_AXIS")
    market_variance, market_excess = 0.0, 0.0
    contributions = np.empty((views, count), dtype=np.float64)
    shift = np.zeros(count, dtype=np.float64)
    ratio = np.empty(views, dtype=np.float64)
    for i in range(count):
        market_excess += market_weights[i] * prior[i]
        for j in range(count):
            market_variance += market_weights[i] * covariance[i, j] * market_weights[j]
        for v in range(views):
            contributions[v, i] = gain_transpose[v, i] * residual[v]
            shift[i] += contributions[v, i]
    for v in range(views):
        ratio[v] = view_prior_variance[v] / omega[v]
    volatility = np.sqrt(max(0.0, market_variance))
    sharpe = market_excess / volatility if volatility > 1e-15 else np.nan
    return market_excess, volatility, sharpe, ratio, contributions, shift


@njit((V, M, D, types.boolean), cache=True, nogil=True)
def mixture_moments_kernel(probabilities, scenario_means, risk_covariances, shared_risk):
    """Frequency-neutral total covariance, shared by scenarios and state evidence."""
    scenarios, count = scenario_means.shape
    if (probabilities.size != scenarios or scenarios < 1 or scenarios > 60 or count < 1 or count > 30
            or risk_covariances.shape != (1 if shared_risk else scenarios, count, count)):
        raise ValueError("CMA_SCENARIO_AXIS")
    means = np.zeros(count, dtype=np.float64)
    total = 0.0
    for s in range(scenarios):
        p = probabilities[s]
        if not np.isfinite(p) or p < 0 or p > 1:
            raise ValueError("CMA_SCENARIO_PROBABILITY")
        total += p
        for i in range(count):
            if not np.isfinite(scenario_means[s, i]):
                raise ValueError("CMA_SCENARIO_MEAN")
            means[i] += p * scenario_means[s, i]
    if abs(total - 1.0) > 1e-8:
        raise ValueError("CMA_SCENARIO_PROBABILITY")
    within = np.zeros((count, count), dtype=np.float64)
    between = np.zeros((count, count), dtype=np.float64)
    for s in range(scenarios):
        risk = 0 if shared_risk else s
        for i in range(count):
            for j in range(count):
                if not np.isfinite(risk_covariances[risk, i, j]):
                    raise ValueError("CMA_SCENARIO_RISK")
                within[i, j] += probabilities[s] * risk_covariances[risk, i, j]
                between[i, j] += probabilities[s] * (scenario_means[s, i] - means[i]) * (scenario_means[s, j] - means[j])
    return means, within + between, within, between


@njit((V, M, D, types.boolean), cache=True, nogil=True)
def scenario_mixture_kernel(probabilities, scenario_means, risk_covariances, shared_risk):
    """Explicit distribution moments, including between-scenario mean risk."""
    scenarios, count = scenario_means.shape
    expected_risks = 1 if shared_risk else scenarios
    if (scenarios < 1 or scenarios > 60 or count < 1 or count > 30
            or probabilities.size != scenarios or risk_covariances.shape != (expected_risks, count, count)):
        raise ValueError("CMA_SCENARIO_AXIS")
    for s in range(expected_risks):
        covariance_diagnostics_kernel(risk_covariances[s])
    total = 0.0
    means = np.zeros(count, dtype=np.float64)
    for s in range(scenarios):
        p = probabilities[s]
        if not np.isfinite(p) or p < 0 or p > 1:
            raise ValueError("CMA_SCENARIO_PROBABILITY")
        total += p
        validate_effective_returns_kernel(scenario_means[s])
        for i in range(count):
            means[i] += p * scenario_means[s, i]
    if abs(total - 1) > 1e-8:
        raise ValueError("CMA_SCENARIO_PROBABILITY")
    means, covariance, within, between = mixture_moments_kernel(
        probabilities, scenario_means, risk_covariances, shared_risk)
    validate_effective_returns_kernel(means)
    covariance_diagnostics_kernel(covariance)
    return means, covariance, within, between


KERNELS = (covariance_diagnostics_kernel, validate_effective_returns_kernel,
           black_litterman_update_kernel, black_litterman_kernel, black_litterman_diagnostics_kernel,
           mixture_moments_kernel, scenario_mixture_kernel)
for dispatcher in KERNELS:
    dispatcher.disable_compile()


def execution_audit():
    dispatchers = (*KERNELS, cma_covariance_kernel)
    compiled = all(len(k.signatures) == len(k.nopython_signatures) == 1 and not k._can_compile
                   and not any(v.objectmode for v in k.overloads.values()) for k in dispatchers)
    ready = compiled and _WARMED_PID == os.getpid()
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "complete": bool(ready), "fully_warmed": bool(ready), "nopython": bool(compiled),
            "object_mode": 0, "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": {k.__name__: [str(s) for s in k.signatures] for k in dispatchers}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("CMA_MODEL_NOT_READY: CMA模型尚未完成本进程启动预热。")


def warm():
    """Call during each worker's startup; never from a calculation request."""
    global _WARMED_PID
    _WARMED_PID = None
    covariance = np.diag(np.array([0.04, 0.01]))[::-1, ::-1]
    weights = np.array([0.4, 0.6])[::-1]
    picks = np.array([[1., -1.], [0., 1.]])[:, ::-1]
    q, std = np.array([0.02, 0.06]), np.array([0.1, 0.2])
    for array in (covariance, weights, picks, q, std):
        array.flags.writeable = False
    black_litterman_kernel(covariance, weights, picks, q, std, 2., 0.05, 0.02)
    black_litterman_kernel(covariance, weights, picks[:0], q[:0], std[:0], 2., 0.05, 0.02)
    solved = black_litterman_update_kernel(covariance, weights, picks, q, std, 2., .05, .02)
    black_litterman_diagnostics_kernel(covariance, weights, solved[2], solved[3], solved[4], solved[5], solved[6])
    means = np.array([[0.04, 0.03], [0.08, 0.02]])[:, ::-1]
    risks = covariance[None, :, :]
    scenario_mixture_kernel(np.array([0.4, 0.6]), means, risks, True)
    scenario_mixture_kernel(np.array([0.4, 0.6]), means, np.stack([covariance, covariance]), False)
    _WARMED_PID = os.getpid()
    if not execution_audit()["complete"]:
        _WARMED_PID = None
        raise RuntimeError("CMA_MODEL_WARMUP_FAILED")
    return execution_audit()
