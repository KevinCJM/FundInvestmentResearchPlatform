"""Matrix-only, long-only efficient frontier; no history or production solver fallback.

All array inputs have one readonly arbitrary-stride float64 ABI. Bounds are N×2;
groups are B×N binary membership coefficients (overlap is allowed). Outputs own
memory. Call warm() once in EACH worker, then solve_frontier() for requests.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import threading

import numpy as np
from numba import njit, types

from backend.qp_numba import (QP_KERNELS, constraint_violation_kernel,
                             matrix_qp_kernel, prepare_constraints_kernel, prepared_matrix_qp_kernel)
from backend.strategic_allocation.kernels import portfolio_moments_kernel

V = types.Array(types.float64, 1, "A", readonly=True)
M = types.Array(types.float64, 2, "A", readonly=True)
I = types.Array(types.int64, 1, "A", readonly=True)
VERSION = "matrix-frontier/1.0.0"
PRIMAL_TOL = 1e-7
KKT_TOL = 1e-9
RETURN_TOL = 1e-8
PSD_TOL = 1e-12
STATUS_NAMES = {0: "optimal_to_tolerance", 1: "iteration_limit", 2: "infeasible_start",
                3: "numerical_failure", 4: "infeasible_certified",
                5: "feasibility_not_found", 6: "range_unresolved"}
_WARMED_PID = None
_WARMED_FINGERPRINT = None
_LOCK = threading.RLock()


@njit((V, M, M, M, V, V), cache=False, nogil=True)
def frontier_constraints_kernel(means, covariance, bounds, groups, group_lows, group_highs):
    """Validate moments and build shared E/e/G/h; return a structural status."""
    n = means.size
    b = groups.shape[0]
    if (not 1 <= n <= 30 or covariance.shape != (n, n) or bounds.shape != (n, 2)
            or groups.shape[1] != n or group_lows.size != b or group_highs.size != b or b > 200):
        raise ValueError("MOMENT_FRONTIER_AXIS")
    if (not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariance))
            or not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(groups))
            or not np.all(np.isfinite(group_lows)) or not np.all(np.isfinite(group_highs))):
        raise ValueError("MOMENT_FRONTIER_NONFINITE")
    scale = max(1.0, np.max(np.abs(covariance)))
    if np.max(np.abs(covariance - covariance.T)) > PSD_TOL * scale:
        raise ValueError("MOMENT_FRONTIER_ASYMMETRIC")
    if np.linalg.eigvalsh(covariance.copy())[0] < -PSD_TOL * scale:
        raise ValueError("MOMENT_FRONTIER_NOT_PSD")
    e = np.ones((1, n))
    ev = np.ones(1)
    g = np.zeros((2 * n + 2 * b, n))
    h = np.zeros(2 * n + 2 * b)
    status = 0
    for i in range(n):
        if bounds[i, 0] < 0 or bounds[i, 1] > 1:
            raise ValueError("MOMENT_FRONTIER_LONG_ONLY")
        if bounds[i, 0] > bounds[i, 1]:
            status = 4
        g[2*i, i], g[2*i+1, i] = 1.0, -1.0
        h[2*i], h[2*i+1] = bounds[i, 0], -bounds[i, 1]
    if np.sum(bounds[:, 0]) > 1.0 + PRIMAL_TOL or np.sum(bounds[:, 1]) < 1.0 - PRIMAL_TOL:
        status = 4
    for k in range(b):
        low, high = 0.0, 0.0
        outside_low, outside_high = 0.0, 0.0
        for i in range(n):
            if groups[k, i] != 0.0 and groups[k, i] != 1.0:
                raise ValueError("MOMENT_FRONTIER_GROUP_MEMBERSHIP")
            if groups[k, i] == 1.0:
                low += bounds[i, 0]
                high += bounds[i, 1]
            else:
                outside_low += bounds[i, 0]
                outside_high += bounds[i, 1]
        low, high = max(low, 1.0 - outside_high), min(high, 1.0 - outside_low)
        if (group_lows[k] > group_highs[k] or group_lows[k] > high + PRIMAL_TOL
                or group_highs[k] < low - PRIMAL_TOL):
            status = 4
        for j in range(k):
            if np.all(groups[k] == groups[j]):
                if max(group_lows[j], group_lows[k]) > min(group_highs[j], group_highs[k]) + PRIMAL_TOL:
                    status = 4
        g[2*n+2*k], g[2*n+2*k+1] = groups[k], -groups[k]
        h[2*n+2*k], h[2*n+2*k+1] = group_lows[k], -group_highs[k]
    return e, ev, g, h, status


@njit((M, V, M, V, types.int64), cache=False, nogil=True)
def phase_one_kernel(equalities, values, inequalities, limits, max_iterations):
    """Return (seed, status, iterations, [primal,KKT,dual,complementarity,t]).

    Positive optimized t alone returns 5, NOT certified infeasibility. Structural
    inconsistent equalities/zero rows return 4. Iteration exhaustion stays 1.
    """
    n = equalities.shape[1]
    matrix, bound, rank, status = prepare_constraints_kernel(equalities, values, inequalities, limits)
    diagnostics = np.full(5, np.nan)
    seed = np.zeros(n)
    if status:
        return seed, status, 0, diagnostics
    for k in range(rank):
        seed += bound[k] * matrix[k]
    residual = constraint_violation_kernel(equalities, values, inequalities, limits, seed)
    if residual <= PRIMAL_TOL:
        diagnostics[:] = 0.0
        diagnostics[0] = residual
        return seed, 0, 0, diagnostics
    e = np.zeros((rank, n + 1))
    e[:, :n] = matrix[:rank]
    g = np.zeros((limits.size + 1, n + 1))
    h = np.zeros(limits.size + 1)
    g[:-1, :n] = matrix[rank:]
    g[:, n] = 1.0
    h[:-1] = bound[rank:]
    x = np.zeros(n + 1)
    x[:n] = seed
    for j in range(limits.size):
        x[n] = max(x[n], h[j] - np.dot(g[j, :n], seed))
    x[n] += 1.0
    linear = np.zeros(n + 1)
    linear[n] = 1.0
    result, status, used, d = matrix_qp_kernel(
        np.zeros((n + 1, n + 1)), linear, e, bound[:rank], g, h, x, max_iterations, KKT_TOL)
    seed = result[:n].copy()
    diagnostics[:4], diagnostics[4] = d, result[n]
    diagnostics[0] = constraint_violation_kernel(equalities, values, inequalities, limits, seed)
    if status == 0:
        status = 0 if diagnostics[0] <= PRIMAL_TOL and result[n] <= PRIMAL_TOL else 5
    return seed, status, used, diagnostics


@njit((V, M, M, M, V, V, types.int64, types.int64), cache=False, nogil=True)
def frontier_moments_kernel(means, covariance, bounds, groups, group_lows, group_highs,
                            point_count, max_iterations):
    """Return the 14-array/scalar tuple documented in m1-report.md.

    Endpoint rows: raw GMV, return-maximal GMV, maximum return LP, minimum
    variance on that maximum-return face. Metrics columns: volatility, return.
    Each point diagnostic: original primal, normalized KKT/dual/complementarity.
    """
    if not 2 <= point_count <= 200 or not 1 <= max_iterations <= 1000:
        raise ValueError("MOMENT_FRONTIER_BUDGET")
    e, ev, g, h, structural = frontier_constraints_kernel(
        means, covariance, bounds, groups, group_lows, group_highs)
    n = means.size
    targets = np.full(point_count, np.nan)
    weights = np.full((point_count, n), np.nan)
    metrics = np.full((point_count, 2), np.nan)
    statuses = np.full(point_count, 6, dtype=np.int64)
    iterations = np.zeros(point_count, dtype=np.int64)
    diagnostics = np.full((point_count, 4), np.nan)
    endpoint_weights = np.full((4, n), np.nan)
    endpoint_metrics = np.full((4, 2), np.nan)
    endpoint_status = np.full(4, 6, dtype=np.int64)
    endpoint_iterations = np.zeros(4, dtype=np.int64)
    endpoint_diagnostics = np.full((4, 4), np.nan)
    phase_status, phase_iterations = structural, 0
    phase_diagnostics = np.full(5, np.nan)
    if structural == 0:
        seed, phase_status, phase_iterations, phase_diagnostics = phase_one_kernel(e, ev, g, h, max_iterations)
        if phase_status == 0:
            zero = np.zeros(n)
            hessian = 2.0 * covariance
            low, st, it, d = matrix_qp_kernel(hessian, zero, e, ev, g, h, seed, max_iterations, KKT_TOL)
            endpoint_weights[0], endpoint_status[0] = low, st
            endpoint_iterations[0], endpoint_diagnostics[0] = it, d
            if st == 0:
                tie_e = np.empty((n+1, n))
                tie_ev = np.empty(n+1)
                tie_e[0], tie_ev[0] = e[0], 1.0
                tie_e[1:] = covariance
                tie_ev[1:] = covariance @ low
                low, st, it, d = matrix_qp_kernel(np.zeros((n, n)), -means, tie_e, tie_ev,
                                                g, h, low, max_iterations, KKT_TOL)
                endpoint_weights[1], endpoint_status[1] = low, st
                endpoint_iterations[1], endpoint_diagnostics[1] = it, d
                # Verify the original quadratic face, not a perturbed objective.
                v0 = np.dot(endpoint_weights[0], covariance @ endpoint_weights[0])
                v1 = np.dot(low, covariance @ low)
                if st == 0 and abs(v1-v0) > PSD_TOL * max(1.0, abs(v0)):
                    endpoint_status[1] = 3
            high, st, it, d = matrix_qp_kernel(np.zeros((n, n)), -means, e, ev, g, h,
                                             seed, max_iterations, KKT_TOL)
            endpoint_weights[2], endpoint_status[2] = high, st
            endpoint_iterations[2], endpoint_diagnostics[2] = it, d
            if st == 0:
                right_e = np.empty((2, n))
                right_ev = np.empty(2)
                right_e[0], right_e[1] = e[0], means
                right_ev[0], right_ev[1] = 1.0, np.dot(means, high)
                high, st, it, d = matrix_qp_kernel(hessian, zero, right_e, right_ev, g, h,
                                                 high, max_iterations, KKT_TOL)
                endpoint_weights[3], endpoint_status[3] = high, st
                endpoint_iterations[3], endpoint_diagnostics[3] = it, d
            for j in range(4):
                if endpoint_status[j] == 0:
                    try:
                        values, _ = portfolio_moments_kernel(endpoint_weights[j], means, covariance, zero, 0., 0.)
                        endpoint_metrics[j, 0], endpoint_metrics[j, 1] = values[1], values[0]
                    except Exception:
                        endpoint_status[j] = 3
            if endpoint_status[1] == 0 and endpoint_status[3] == 0:
                low_return, high_return = endpoint_metrics[1, 1], endpoint_metrics[3, 1]
                if high_return >= low_return - RETURN_TOL:
                    targets = np.linspace(low_return, max(low_return, high_return), point_count)
                    grid_g = np.empty((g.shape[0]+1, n))
                    grid_h = np.empty(h.size+1)
                    grid_g[:-1], grid_g[-1] = g, means
                    grid_h[:-1], grid_h[-1] = h, targets[0]
                    grid_matrix, grid_bound, grid_rank, prepared_status = prepare_constraints_kernel(e, ev, grid_g, grid_h)
                    target_norm = np.sqrt(np.dot(means, means))
                    if target_norm == 0.0:
                        target_norm = 1.0
                    previous = low.copy()
                    for j in range(point_count):
                        grid_h[-1] = targets[j]
                        grid_bound[-1] = targets[j]/target_norm
                        # Endpoint solves also use the ORIGINAL grid QP so every row's
                        # KKT diagnostics refer to variance, not the tie-breaking LP.
                        previous_return = np.dot(means, previous)
                        if j == point_count - 1:
                            previous = high.copy()
                        elif previous_return < targets[j]:
                            denominator = high_return - previous_return
                            if denominator <= 0.0:
                                previous = high.copy()
                            else:
                                fraction = min(1.0, max(0.0, (targets[j] - previous_return) / denominator))
                                previous = (1.0 - fraction) * previous + fraction * high
                        if prepared_status:
                            statuses[j] = prepared_status
                            continue
                        solution, st, it, d = prepared_matrix_qp_kernel(
                            hessian, zero, e, ev, grid_g, grid_h, grid_matrix, grid_bound,
                            grid_rank, previous, max_iterations, KKT_TOL)
                        d[0] = constraint_violation_kernel(e, ev, grid_g, grid_h, solution)
                        if st == 0 and d[0] > PRIMAL_TOL:
                            st = 3
                        if st == 0:
                            try:
                                values, _ = portfolio_moments_kernel(solution, means, covariance, zero, 0., 0.)
                                metrics[j, 0], metrics[j, 1] = values[1], values[0]
                                previous = solution.copy()
                            except Exception:
                                st = 3
                        # Retain failed iterates as diagnostics; status gates their use.
                        weights[j], statuses[j], iterations[j], diagnostics[j] = solution, st, it, d
    return (targets, weights, metrics, statuses, iterations, diagnostics,
            endpoint_weights, endpoint_metrics, endpoint_status, endpoint_iterations, endpoint_diagnostics,
            phase_status, phase_iterations, phase_diagnostics)


@njit((M, I, types.float64), cache=False, nogil=True)
def frontier_curve_indices_kernel(metrics, statuses, tolerance):
    """Return (retained original indices, status): 0 valid, 1 gap, 2 degenerate, 3 invalid.

    A failed point forbids segmentation across its gap. Duplicate/dominated nodes
    are removed by indices only; raw solver evidence is never overwritten.
    """
    n = statuses.size
    if metrics.shape != (n, 2) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("FRONTIER_CURVE_AXIS")
    indices = np.empty(n, dtype=np.int64)
    if n == 0:
        return indices[:0], 2
    if np.any(statuses != 0):
        return indices[:0], 1
    if not np.all(np.isfinite(metrics)) or np.any(metrics[:, 0] < 0):
        return indices[:0], 3
    count = 0
    for i in range(n):
        if i and (metrics[i, 0] < metrics[i-1, 0]-tolerance or metrics[i, 1] < metrics[i-1, 1]-tolerance):
            return indices[:0], 3
        dominated = False
        for j in range(n):
            if j == i:
                continue
            if (metrics[j, 0] <= metrics[i, 0]+tolerance and metrics[j, 1] >= metrics[i, 1]-tolerance
                    and (metrics[j, 0] < metrics[i, 0]-tolerance or metrics[j, 1] > metrics[i, 1]+tolerance)):
                dominated = True
                break
        if dominated:
            continue
        if count and (metrics[i, 0] <= metrics[indices[count-1], 0]+tolerance
                      or metrics[i, 1] <= metrics[indices[count-1], 1]+tolerance):
            continue
        indices[count] = i
        count += 1
    return indices[:count], 0 if count >= 2 else 2


KERNELS = (frontier_constraints_kernel, phase_one_kernel, frontier_moments_kernel,
           frontier_curve_indices_kernel)
DISPATCHERS = (*QP_KERNELS, portfolio_moments_kernel, *KERNELS)
for _dispatcher in KERNELS:
    _dispatcher.disable_compile()


def execution_audit():
    signatures = {k.__name__: [str(s) for s in k.signatures] for k in DISPATCHERS}
    fingerprints = {k.__name__: hashlib.sha256(inspect.getsource(k.py_func).encode()).hexdigest()
                    for k in DISPATCHERS}
    fingerprint = hashlib.sha256(repr((VERSION, signatures, fingerprints)).encode()).hexdigest()
    compiled = all(len(k.signatures) == len(k.nopython_signatures) == 1 and not k._can_compile
                   and not any(v.objectmode for v in k.overloads.values()) for k in DISPATCHERS)
    ready = compiled and _WARMED_PID == os.getpid() and _WARMED_FINGERPRINT == fingerprint
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "complete": ready, "fully_warmed": ready, "warmed_pid": _WARMED_PID,
            "pid": os.getpid(), "nopython": compiled, "object_mode": 0,
            "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": signatures, "kernel_fingerprints": fingerprints, "fingerprint": fingerprint}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("MOMENT_FRONTIER_NOT_READY")


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    with _LOCK:
        _WARMED_PID = None
        means = np.array([.02, .08])[::-1]
        covariance = np.diag(np.array([0., .04]))[::-1, ::-1]
        bounds = np.array([[0., 1.], [0., 1.]])[::-1]
        groups = np.zeros((0, 2))
        empty = np.empty(0)
        for a in (means, covariance, bounds, groups, empty):
            a.flags.writeable = False
        result = frontier_moments_kernel(means, covariance, bounds, groups, empty, empty, 31, 1000)
        indices, status = frontier_curve_indices_kernel(result[2], result[3], 1e-10)
        if result[11] != 0 or status != 0 or indices.size != 31 or np.any(result[3]):
            raise RuntimeError("MOMENT_FRONTIER_WARMUP_FAILED")
        constrained = frontier_moments_kernel(
            means, covariance, bounds, np.array([[1., 0.]]), np.array([.7]), np.array([.9]), 31, 1000)
        tied = frontier_moments_kernel(
            np.array([.02, .04, .1]), np.array([[.01, .01, 0.], [.01, .01, 0.], [0., 0., .04]]),
            np.tile(np.array([[0., 1.]]), (3, 1)), np.zeros((0, 3)), empty, empty, 31, 1000)
        if constrained[11] != 0 or np.any(constrained[3]) or np.any(tied[3]):
            raise RuntimeError("MOMENT_FRONTIER_PHASE_TIE_WARMUP_FAILED")
        # Exercise the compatibility adapter as well as the general call chain.
        from backend.qp_numba import feasible_qp_kernel
        old = feasible_qp_kernel(np.eye(2), np.zeros(2), np.ones((1, 2)), np.ones(1),
                                 np.array([.5, .5]), 1000, KKT_TOL)
        if old[1] != 0:
            raise RuntimeError("MOMENT_FRONTIER_QP_WARMUP_FAILED")
        _WARMED_FINGERPRINT = execution_audit()["fingerprint"]
        _WARMED_PID = os.getpid()
        require_ready()
        return execution_audit()


def solve_frontier(means, covariance, bounds, groups, group_lows, group_highs,
                   point_count=101, max_iterations=1000):
    """Readiness-checked service entry; arrays must already be float64 (no hidden copy)."""
    require_ready()
    if isinstance(point_count, bool) or not isinstance(point_count, (int, np.integer)):
        raise ValueError("MOMENT_FRONTIER_POINT_COUNT")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, (int, np.integer)):
        raise ValueError("MOMENT_FRONTIER_MAX_ITERATIONS")
    return frontier_moments_kernel(means, covariance, bounds, groups, group_lows, group_highs,
                                   point_count, max_iterations)
