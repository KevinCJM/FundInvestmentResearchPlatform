"""Bounded scheduling of the shared LP core and true quadratic risk checks."""
from __future__ import annotations

import time
import numpy as np
from . import compatibility_kernels as numeric

MAX_CUTS = 2048
MASTER_ITERATIONS = 500
OBJECTIVE_TOLERANCE = 1e-7
MAX_SECONDS = 30.


def solve(means, risks, bounds, groups, lows, highs, benchmark, floor, vol_cap,
          te_cap, excess, references, objective, *, max_iterations=64, deadline=None):
    """Return deterministic evidence; wall-clock time is a budget, not a hash input.

    No fallback candidate is manufactured after exhaustion. A feasible point
    with an open objective gap is explicitly approximate, never a global claim.
    """
    numeric.require_ready()
    if not 1 <= max_iterations <= 128:
        raise ValueError('COMPATIBILITY_ITERATION_BUDGET')
    deadline = time.monotonic() + MAX_SECONDS if deadline is None else deadline
    base, limits, cost, lower, upper, initial = numeric.prepare_problem_kernel(
        means, risks, bounds, groups, lows, highs, benchmark, float(floor), float(vol_cap),
        float(te_cap), float(excess), references, int(objective))
    count = len(limits)
    matrix = np.empty((count + MAX_CUTS, initial.size)); rhs = np.empty(count + MAX_CUTS)
    matrix[:count] = base; rhs[:count] = limits
    current = initial
    bound = -np.inf
    feasible = None
    best_value = np.inf
    status = 'iteration_limit'
    masters = 0
    residuals = None
    phase_one_bound = None
    for iteration in range(max_iterations):
        if time.monotonic() >= deadline:
            status = 'time_budget'; break
        # Views retain one bounded support-cut workspace for the whole solve.
        current, code, used, checks = numeric.linear_master_kernel(
            matrix[:count], rhs[:count], cost, lower, upper, current, MASTER_ITERATIONS)
        masters += int(used)
        if code == 4:
            phase_one_bound = float(checks[4])
            status = 'infeasible'; break
        if code == 5 or not np.all(np.isfinite(current)):
            status = 'phase_one_unresolved'; break
        if np.isfinite(checks[4]):
            bound = max(bound, float(checks[4]))
        cuts, cut_limits, residuals = numeric.quadratic_support_kernel(
            current, risks, benchmark, float(vol_cap), float(te_cap))
        # The shared master already checks primal residuals. A failed LP cannot
        # authorize weights merely because its nonlinear risks happen to pass.
        if code == 0 and not len(cuts):
            feasible = current[:-1].copy()
            best_value = float(current[-1])
            status = 'converged' if best_value-bound <= OBJECTIVE_TOLERANCE else 'verified_feasible'
            break
        if code != 0:
            status = 'master_iteration_limit' if code == 1 else 'numerical_failure'; break
        if count + len(cuts) > len(rhs):
            status = 'cut_budget'; break
        matrix[count:count+len(cuts)] = cuts
        rhs[count:count+len(cuts)] = cut_limits
        count += len(cuts)
    return {'weights': feasible, 'status': status, 'objective_value': best_value if feasible is not None else None,
            'lower_bound': bound if np.isfinite(bound) else None,
            'objective_gap': max(0., best_value-bound) if feasible is not None and np.isfinite(bound) else None,
            'phase_one_lower_bound': phase_one_bound, 'iterations': iteration+1,
            'master_iterations': masters, 'support_cuts': count-len(limits),
            'risk_residuals': residuals.tolist() if residuals is not None else [],
            'search_domain': 'continuous_convex_outer_approximation',
            'objective_tolerance': OBJECTIVE_TOLERANCE,
            'phase_one': 'bounded_maximum_normalized_slack_with_dual_lower_bound',
            'limits': {'outer_iterations': max_iterations, 'master_iterations': MASTER_ITERATIONS,
                       'support_cuts': MAX_CUTS, 'shared_wall_seconds': MAX_SECONDS}}
