"""Bounded outer-approximation building blocks for common-model return problems.

The nonlinear state is the accumulated set of supporting half-spaces. All LPs
delegate to the shared active-set core; Phase I never interprets solver failure
as infeasibility. Only a positive dual lower bound can certify a conflict.
"""
from __future__ import annotations

import os
import numpy as np
from numba import njit, types
from backend.qp_numba import bounded_lp_kernel

V = types.Array(types.float64, 1, 'A', readonly=True)
M = types.Array(types.float64, 2, 'A', readonly=True)
T = types.Array(types.float64, 3, 'A', readonly=True)
I = types.int64
F = types.float64
_WARMED_PID = None


@njit((M, T, M, M, V, V, V, F, F, F, F, V, I), cache=True, nogil=True)
def prepare_problem_kernel(means, risks, bounds, groups, lows, highs, benchmark,
                           floor, vol_cap, te_cap, excess, references, objective):
    models, n = means.shape
    if (models < 1 or models > 20 or n < 1 or n > 30 or risks.shape != (models, n, n)
            or bounds.shape != (n, 2) or groups.shape != (lows.size, n) or highs.size != lows.size
            or benchmark.size not in (0, n) or objective not in (0, 1)
            or references.size != models or vol_cap < 0 or te_cap < 0
            or not np.isfinite(vol_cap) or not np.isfinite(te_cap) or not np.isfinite(excess)
            or not (np.isfinite(floor) or floor == -np.inf)):
        raise ValueError('COMPATIBILITY_AXIS')
    for value in (means, bounds, groups):
        if not np.all(np.isfinite(value)):
            raise ValueError('COMPATIBILITY_NONFINITE')
    if (not np.all(np.isfinite(risks)) or not np.all(np.isfinite(references))
            or not np.all(np.isfinite(lows)) or not np.all(np.isfinite(highs))
            or np.any(bounds[:, 0] < 0) or np.any(bounds[:, 1] > 1)
            or np.any(bounds[:, 0] > bounds[:, 1]) or np.any(lows > highs)):
        raise ValueError('COMPATIBILITY_INPUT')
    for m in range(models):
        if np.max(np.abs(risks[m] - risks[m].T)) > 1e-10 or np.min(np.linalg.eigvalsh(risks[m])) < -1e-10:
            raise ValueError('COMPATIBILITY_RISK_PSD')
    if benchmark.size:
        if (not np.all(np.isfinite(benchmark)) or np.any(benchmark < 0)
                or abs(np.sum(benchmark) - 1) > 1e-8):
            raise ValueError('COMPATIBILITY_BENCHMARK')
    # x=(long-only weights, epigraph). Bound q from the observed finite means;
    # fitted model output need not share the manual input editor's range.
    d = n + 1
    matrix = np.zeros((1 + 2*d + 2*lows.size + 3*models, d))
    limits = np.zeros(matrix.shape[0])
    matrix[0, :n] = 1.
    limits[0] = 1.
    lower = np.zeros(d); upper = np.ones(d)
    epigraph_bound = 2.*np.max(np.abs(means)) + np.max(np.abs(references)) + 1.
    if not np.isfinite(epigraph_bound):
        raise ValueError('COMPATIBILITY_RETURN_SCALE')
    lower[n] = 0. if objective == 1 else -epigraph_bound
    upper[n] = epigraph_bound
    row = 1
    for j in range(d):
        matrix[row, j] = 1.
        matrix[row+1, j] = -1.
        limits[row] = bounds[j, 0] if j < n else lower[j]
        limits[row+1] = -bounds[j, 1] if j < n else -upper[j]
        row += 2
    for g in range(lows.size):
        matrix[row, :n] = groups[g]; limits[row] = lows[g]
        matrix[row+1, :n] = -groups[g]; limits[row+1] = -highs[g]
        row += 2
    for m in range(models):
        offset = np.dot(means[m], benchmark) if benchmark.size else 0.
        if np.isfinite(floor):
            matrix[row, :n] = means[m]; limits[row] = floor; row += 1
        if benchmark.size:
            matrix[row, :n] = means[m]; limits[row] = offset + excess; row += 1
        matrix[row, :n] = means[m]; matrix[row, n] = 1.
        limits[row] = offset + (references[m] if objective == 1 else 0.)
        row += 1
    # Normalize once; subsequent support rows use the same convention.
    for k in range(1, row):
        norm = np.sqrt(np.dot(matrix[k], matrix[k]))
        if norm > 0:
            matrix[k] /= norm; limits[k] /= norm
    initial = np.zeros(d); initial[:n] = 1. / n
    linear = np.zeros(d); linear[n] = 1.
    return matrix[:row].copy(), limits[:row].copy(), linear, lower, upper, initial


@njit((M, V, V, V, V, V, I), cache=True, nogil=True)
def linear_master_kernel(matrix, limits, linear, lower, upper, initial, iterations):
    """Find a feasible master seed with a bounded max-slack Phase I LP.

    Status 4: positive Phase I lower bound. Status 5: no feasible seed was
    verified. Neither an iteration limit nor an invalid start proves no solution.
    """
    n = initial.size
    violation = 0.
    for k in range(1, limits.size):
        violation = max(violation, limits[k] - np.dot(matrix[k], initial))
    seed = initial.copy()
    if violation > 1e-10:
        # Hard simplex/box remain unslacked; all other rows share one slack.
        count = limits.size + 2*(n+1)
        a = np.zeros((count, n+1)); b = np.zeros(count)
        a[0, :n] = matrix[0]; b[0] = limits[0]
        start = np.zeros(n+1)
        start[:n-1] = 1. / (n-1)
        slack = 0.
        for k in range(1, limits.size):
            a[k, :n] = matrix[k]; a[k, n] = 1.; b[k] = limits[k]
            slack = max(slack, limits[k] - np.dot(matrix[k], start[:n]))
        start[n] = slack + 1.
        lo = np.zeros(n+1); hi = np.ones(n+1)
        lo[:n] = lower; hi[:n] = upper; hi[n] = slack + 2.
        for j in range(n+1):
            k = limits.size + 2*j
            a[k, j] = 1.; b[k] = lo[j]
            a[k+1, j] = -1.; b[k+1] = -hi[j]
        cost = np.zeros(n+1); cost[n] = 1.
        result, status, used, checks = bounded_lp_kernel(cost, a, b, start, lo, hi, iterations, 1e-11)
        if np.isfinite(checks[4]) and checks[4] > 1e-8:
            return result[:n].copy(), 4, used, checks
        violation = 0.
        for k in range(1, limits.size):
            violation = max(violation, limits[k] - np.dot(matrix[k], result[:n]))
        if violation > 1e-10 or not np.all(np.isfinite(result)):
            return result[:n].copy(), 5, used, checks
        seed = result[:n].copy()
    result, status, used, checks = bounded_lp_kernel(linear, matrix, limits, seed, lower, upper, iterations, 1e-11)
    # The historical QP ABI permits a looser primal residual. A new common
    # policy must independently satisfy its own constraint tolerance.
    if status == 0:
        violation = abs(np.dot(matrix[0], result) - limits[0])
        for k in range(1, limits.size):
            violation = max(violation, limits[k] - np.dot(matrix[k], result))
        for j in range(n):
            violation = max(violation, lower[j] - result[j], result[j] - upper[j])
        if violation > 1e-10 or not np.all(np.isfinite(result)):
            status = 3
    return result, status, used, checks


@njit((V, T, V, F, F), cache=True, nogil=True)
def quadratic_support_kernel(point, risks, benchmark, vol_cap, te_cap):
    """Evaluate original risks and emit normalized valid supporting cuts."""
    models, n, _ = risks.shape
    kinds = 2 if benchmark.size else 1
    rows = np.zeros((models*kinds, n+1)); rhs = np.zeros(models*kinds)
    violations = np.zeros(models*kinds)
    used = 0
    for m in range(models):
        for kind in range(kinds):
            delta = point[:n].copy()
            cap = vol_cap
            if kind:
                delta -= benchmark
                cap = te_cap
            marginal = risks[m] @ delta
            variance = np.dot(delta, marginal)
            risk = np.sqrt(max(0., variance))
            violations[m*kinds+kind] = risk-cap
            if risk > cap + 1e-10:
                rows[used, :n] = -2.*marginal
                rhs[used] = variance-cap*cap-2.*np.dot(marginal, point[:n])
                norm = np.sqrt(np.dot(rows[used], rows[used]))
                if norm > 0:
                    rows[used] /= norm; rhs[used] /= norm
                used += 1
    return rows[:used].copy(), rhs[:used].copy(), violations


@njit((M, V), cache=True, nogil=True)
def worst_model_summary_kernel(metrics, probability_lowers):
    if metrics.shape[0] < 1 or metrics.shape[1] != 5 or not np.all(np.isfinite(metrics)):
        raise ValueError('COMPATIBILITY_METRICS')
    result = np.empty(5)
    for j in range(5):
        result[j] = np.max(metrics[:, j]) if j == 1 else np.min(metrics[:, j])
    worst = -1
    if probability_lowers.size:
        if probability_lowers.size != metrics.shape[0] or not np.all(np.isfinite(probability_lowers)):
            raise ValueError('COMPATIBILITY_FUNDING')
        worst = int(np.argmin(probability_lowers))
    return result, worst


KERNELS = (prepare_problem_kernel, linear_master_kernel, quadratic_support_kernel, worst_model_summary_kernel, bounded_lp_kernel)
for kernel in KERNELS:
    kernel.disable_compile()


def execution_audit():
    complete = _WARMED_PID == os.getpid() and all(len(k.signatures) == 1 and k.nopython_signatures
        and not k._can_compile and not any(o.objectmode for o in k.overloads.values()) for k in KERNELS)
    return {'backend': 'numba_njit_fixed_signature', 'kernel_version': 'multi-cma-outer-approximation/1.0.0',
            'complete': bool(complete), 'fully_warmed': bool(complete), 'nopython': bool(complete),
            'object_mode': 0, 'python_fallback': 0, 'request_time_compilation': 0,
            'kernel_signatures': {k.__name__: [str(s) for s in k.signatures] for k in KERNELS}}


def require_ready():
    if not execution_audit()['complete']:
        raise RuntimeError('多 CMA 共同约束内核尚未完成本进程预热。')


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    means = np.array([[.1, .02], [.02, .1]])[:, ::-1]
    risks = np.array([np.eye(2)*.04, np.eye(2)*.04])[:, ::-1, ::-1]
    bounds = np.array([[0., 1.], [0., 1.]])
    for x in (means, risks, bounds): x.flags.writeable = False
    args = prepare_problem_kernel(means, risks, bounds, np.zeros((0,2)), np.empty(0), np.empty(0),
                                  np.empty(0), .058, .16, 1., 0., np.zeros(2), 0)
    a, b, f, lo, hi, x = args
    linear_master_kernel(a, b, f, lo, hi, x, 300)
    quadratic_support_kernel(np.array([1.,0.,0.]), risks, np.empty(0), .16, 1.)
    worst_model_summary_kernel(np.zeros((2,5)), np.array([.9,.8]))
    _WARMED_PID = os.getpid()
    require_ready()
    return execution_audit()
