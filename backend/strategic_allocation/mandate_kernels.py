"""Bounded reference candidate checks and funding search using shared NJIT primitives."""
from __future__ import annotations

import hashlib
import inspect
import os

import numpy as np
from numba import njit, types

from . import goal_kernels as goals
from .kernels import portfolio_moments_kernel, expected_active_risk_kernel, expected_excess_return_kernel
from .risk_scale_kernels import classify_risk_kernel, BOUNDARY_TOL

F, N = types.float64, types.int64
V = types.Array(F, 1, "A", readonly=True)
M = types.Array(F, 2, "A", readonly=True)
D = types.Array(F, 3, "A", readonly=True)
I = types.Array(N, 1, "A", readonly=True)
VERSION = "mandate-reference-search/1.0.0"
_WARMED_PID = None


@njit((M, I, V, M, V, M, M, V, V, V, F, F, F, F, F, F, V, F), cache=True, nogil=True)
def reference_candidate_checks_kernel(weights, statuses, means, covariance, uncertainty,
                                      bounds, groups, group_lows, group_highs, benchmark,
                                      cap, return_floor, te_cap, excess_floor, aversion,
                                      penalty, risk_caps, reference_minimum):
    """Evaluate each real candidate, never convert a failed solve to a valid weight."""
    count, assets = weights.shape
    if (statuses.size != count or means.size != assets or covariance.shape != (assets, assets)
            or uncertainty.size != assets or bounds.shape != (assets, 2)
            or groups.shape != (group_lows.size, assets) or group_highs.size != group_lows.size
            or benchmark.size not in (0, assets) or risk_caps.size != 5
            or not np.isfinite(cap) or cap < 0):
        raise ValueError("MANDATE_CANDIDATE_AXIS")
    metrics = np.full((count, 7), np.nan)
    levels = np.zeros(count, dtype=np.int64)
    eligible = np.zeros(count, dtype=np.int64)
    for row in range(count):
        if statuses[row] != 0:
            continue
        w = weights[row]
        valid = True
        total_weight = 0.
        for i in range(assets):
            total_weight += w[i]
            valid = valid and np.isfinite(w[i]) and bounds[i, 0] - 1e-8 <= w[i] <= bounds[i, 1] + 1e-8
        valid = valid and abs(total_weight - 1.) <= 1e-8
        for group in range(groups.shape[0]):
            total = 0.
            for i in range(assets):
                total += groups[group, i] * w[i]
            valid = valid and group_lows[group] - 1e-8 <= total <= group_highs[group] + 1e-8
        if not valid:
            continue
        evaluated, _ = portfolio_moments_kernel(w, means, covariance, uncertainty, aversion, penalty)
        metrics[row, :5] = evaluated
        level, _ = classify_risk_kernel(evaluated[1], risk_caps, reference_minimum, BOUNDARY_TOL)
        levels[row] = level
        valid = evaluated[1] <= cap + BOUNDARY_TOL and evaluated[0] >= return_floor - 1e-10
        if benchmark.size:
            metrics[row, 5] = expected_active_risk_kernel(w, benchmark, covariance)
            metrics[row, 6] = expected_excess_return_kernel(w, benchmark, means)
            valid = valid and metrics[row, 5] <= te_cap + 1e-10 and metrics[row, 6] >= excess_floor - 1e-10
        eligible[row] = int(valid)
    return metrics, levels, eligible


@njit((M, I, D, F, V, V, F, F, F, N, N), cache=True, nogil=True)
def reference_funding_search_kernel(metrics, eligible, draws, initial, inflows, outflows,
                                    target, fee, probability, method, periods):
    """Search evidence only: keep draws shared and choose before validation is seen."""
    count = eligible.size
    if (metrics.shape[0] != count or metrics.shape[1] < 2 or not .5 <= probability <= .99
            or draws.shape[0] != inflows.size or inflows.size != outflows.size or inflows.size == 0
            or draws.shape[1] == 0 or draws.shape[2] != 1 or not np.isfinite(initial) or initial <= 0
            or not np.isfinite(target) or target < 0):
        raise ValueError("MANDATE_SEARCH_INPUT")
    # One streaming validation pass; do not allocate a path-sized Boolean mask.
    for month in range(inflows.size):
        if not np.isfinite(inflows[month]) or not np.isfinite(outflows[month]) or inflows[month] < 0 or outflows[month] < 0:
            raise ValueError("MANDATE_SEARCH_INPUT")
        for path in range(draws.shape[1]):
            if not np.isfinite(draws[month, path, 0]):
                raise ValueError("MANDATE_SEARCH_INPUT")
    outcomes = np.full((count, 3), np.nan)
    selected = -1
    for row in range(count):
        if eligible[row] == 0:
            continue
        drift, scale = goals.funding_monthly_parameters_kernel(metrics[row, 0], metrics[row, 1], fee, method, periods)
        successes = goals.funding_capital_successes_kernel(draws, drift, scale, initial, inflows, outflows, target, 1.)
        low, high = goals.wilson_interval_kernel(successes, draws.shape[1])
        outcomes[row, 0], outcomes[row, 1], outcomes[row, 2] = successes / draws.shape[1], low, high
        if low < probability:
            continue
        if (selected < 0 or metrics[row, 1] < metrics[selected, 1] - 1e-12
                or abs(metrics[row, 1] - metrics[selected, 1]) <= 1e-12 and outcomes[row, 0] > outcomes[selected, 0] + 1e-12):
            selected = row
    return outcomes, selected


@njit((M, I), cache=True, nogil=True)
def minimum_reference_candidate_kernel(metrics, eligible):
    """Stable minimum-risk selection when there is no cash success event."""
    if metrics.shape[0] != eligible.size or metrics.shape[1] < 2:
        raise ValueError("MANDATE_CANDIDATE_AXIS")
    selected = -1
    for row in range(eligible.size):
        if eligible[row] and (selected < 0 or metrics[row, 1] < metrics[selected, 1] - 1e-12):
            selected = row
    return selected


@njit((M, I, M), cache=True, nogil=True)
def diagnostic_reference_candidate_kernel(metrics, eligible, outcomes):
    """Choose one fixed capital-diagnostic candidate, never an adoption recommendation."""
    if metrics.shape[0] != eligible.size or outcomes.shape != (eligible.size, 3):
        raise ValueError("MANDATE_CANDIDATE_AXIS")
    selected = -1
    for row in range(eligible.size):
        if not eligible[row] or not np.isfinite(outcomes[row, 0]):
            continue
        if (selected < 0 or outcomes[row, 0] > outcomes[selected, 0] + 1e-12
                or abs(outcomes[row, 0] - outcomes[selected, 0]) <= 1e-12 and metrics[row, 1] < metrics[selected, 1] - 1e-12):
            selected = row
    return selected


@njit((M, I, F, F), cache=True, nogil=True)
def reference_reachability_kernel(metrics, levels, cap, return_floor):
    """Report which stated bound actually binds. Reads evaluated candidates only."""
    if metrics.shape[0] != levels.size or metrics.shape[1] < 2 or not np.isfinite(cap) or cap < 0:
        raise ValueError("MANDATE_CANDIDATE_AXIS")
    best_return, best_row = -np.inf, -1
    least_volatility, least_row = np.inf, -1
    for row in range(levels.size):
        # levels[row] > 0 marks a solved candidate that already passed bounds and groups.
        if levels[row] == 0 or not np.isfinite(metrics[row, 0]) or not np.isfinite(metrics[row, 1]):
            continue
        if metrics[row, 1] <= cap + BOUNDARY_TOL and metrics[row, 0] > best_return:
            best_return, best_row = metrics[row, 0], row
        if (np.isfinite(return_floor) and metrics[row, 0] >= return_floor - 1e-10
                and metrics[row, 1] < least_volatility):
            least_volatility, least_row = metrics[row, 1], row
    return best_return, best_row, least_volatility, least_row


KERNELS = (reference_candidate_checks_kernel, reference_funding_search_kernel, minimum_reference_candidate_kernel,
           diagnostic_reference_candidate_kernel, reference_reachability_kernel)
for _kernel in KERNELS:
    _kernel.disable_compile()


def execution_audit():
    signatures = {k.__name__: [str(s) for s in k.signatures] for k in KERNELS}
    fingerprint = hashlib.sha256((VERSION + repr(signatures) + "".join(inspect.getsource(k.py_func) for k in KERNELS)).encode()).hexdigest()
    complete = _WARMED_PID == os.getpid() and all(len(k.nopython_signatures) == 1 and not k._can_compile for k in KERNELS)
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION, "complete": complete,
            "fully_warmed": complete, "nopython": complete, "python_fallback": 0,
            "object_mode": 0, "request_time_compilation": 0, "kernel_signatures": signatures, "fingerprint": fingerprint}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("MANDATE_REFERENCE_NOT_READY")


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    weights = np.array([[1., 0.], [.5, .5]])
    metrics, _, eligible = reference_candidate_checks_kernel(weights, np.zeros(2, dtype=np.int64),
        np.array([.02, .08]), np.diag(np.array([.0001, .04])), np.zeros(2),
        np.array([[0., 1.], [0., 1.]]), np.empty((0, 2)), np.empty(0), np.empty(0),
        np.empty(0), .25, -np.inf, .1, 0., 5., 1., np.array([.02, .05, .1, .15, .25]), .01)
    draws, _ = goals.seeded_factor_draws_kernel(12, 8, 1, 42, 0, 5.)
    flows = np.zeros(12)
    for method in (0, 1):
        outcomes, _ = reference_funding_search_kernel(metrics, eligible, draws, 100., flows, flows, 90., 0., .5, method, 252)
        diagnostic_reference_candidate_kernel(metrics, eligible, outcomes)
    minimum_reference_candidate_kernel(metrics, eligible)
    reference_reachability_kernel(metrics, np.ones(2, dtype=np.int64), .25, .05)
    _WARMED_PID = os.getpid()
    require_ready()
    return execution_audit()
