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
    workspace = numeric.prepare_outer_workspace_kernel(
        base, limits, initial, MAX_CUTS, risks.shape[0] * (2 if benchmark.size else 1))
    state, values, matrix, rhs, current, weights, residuals = workspace
    timed_out = False
    # Only the wall-clock scheduler stays in Python. Every numerical transition,
    # stopping decision and support cut belongs to the compiled state machine.
    # Yield after each master to retain the existing non-preemptive time budget.
    while state[0] == 0:
        if time.monotonic() >= deadline:
            timed_out = True
            break
        numeric.advance_outer_approximation_kernel(
            state, values, matrix, rhs, current, weights, residuals, cost, lower, upper,
            risks, benchmark, float(vol_cap), float(te_cap), max_iterations,
            MASTER_ITERATIONS, OBJECTIVE_TOLERANCE, 1)
    statuses = ('running', 'converged', 'verified_feasible', 'infeasible',
                'phase_one_unresolved', 'iteration_limit', 'master_iteration_limit',
                'numerical_failure', 'cut_budget')
    objective_value, bound, gap, phase_one_bound = (
        float(value) if np.isfinite(value) else None for value in values)
    return {'weights': weights if state[0] in (1, 2) else None,
            'status': 'time_budget' if timed_out else statuses[state[0]],
            'objective_value': objective_value, 'lower_bound': bound, 'objective_gap': gap,
            'phase_one_lower_bound': phase_one_bound, 'iterations': int(state[1]) + int(timed_out),
            'master_iterations': int(state[2]), 'support_cuts': int(state[3] - state[4]),
            'risk_residuals': residuals[:state[5]].tolist(),
            'search_domain': 'continuous_convex_outer_approximation',
            'objective_tolerance': OBJECTIVE_TOLERANCE,
            'phase_one': 'bounded_maximum_normalized_slack_with_dual_lower_bound',
            'limits': {'outer_iterations': max_iterations, 'master_iterations': MASTER_ITERATIONS,
                       'support_cuts': MAX_CUTS, 'shared_wall_seconds': MAX_SECONDS}}
