"""Feasible active-set QP/LP steps for target-grid portfolio optimization.

The first constraint is an equality; subsequent rows mean A @ x >= b.
Inputs are read-only views. Only the iterate and the small KKT workspace are
owned here. No external Python solver, callback or request-time compilation.
"""
from __future__ import annotations

import numpy as np
from numba import njit, types

_R1 = types.Array(types.float64, 1, "A", readonly=True)
_R2 = types.Array(types.float64, 2, "A", readonly=True)
_I1 = types.Array(types.int64, 1, "A", readonly=True)
_RESULT = types.Tuple((types.float64[::1], types.int64, types.int64, types.float64))


@njit(types.boolean(_R2, _I1, types.int64, types.int64), cache=False, nogil=True)
def independent_constraint_kernel(matrix, active, size, candidate):
    """Reject dependent active rows rather than solving a singular KKT system."""
    columns = matrix.shape[1]
    basis = np.zeros((size, columns), dtype=np.float64)
    for k in range(size):
        vector = matrix[active[k]].copy()
        for j in range(k):
            coefficient = np.dot(vector, basis[j])
            vector -= coefficient * basis[j]
        norm = np.sqrt(np.dot(vector, vector))
        if norm <= 1e-12:
            return False
        basis[k] = vector / norm
    vector = matrix[candidate].copy()
    original_norm = np.sqrt(np.dot(vector, vector))
    for _ in range(2):
        for j in range(size):
            vector -= np.dot(vector, basis[j]) * basis[j]
    return np.sqrt(np.dot(vector, vector)) > 1e-10 * max(1.0, original_norm)


@njit(_RESULT(_R2, _R1, _R2, _R1, _R1, types.int64, types.float64), cache=False, nogil=True)
def feasible_qp_kernel(hessian, linear, matrix, limits, initial, max_iterations, tolerance):
    """Minimize .5*x'H*x + f'x from a feasible point, with explicit KKT status.

    Status: 0 optimum to tolerance, 1 iteration limit, 2 infeasible start,
    3 numerical failure. H=0 is an LP: move to a blocking face, not a
    regularized approximation of the requested linear objective.
    """
    n = initial.size
    m = limits.size
    if (n == 0 or m == 0 or hessian.shape != (n, n) or linear.size != n
            or matrix.shape != (m, n) or max_iterations < 1 or tolerance <= 0):
        raise ValueError("QP_INPUT_AXIS")
    x = initial.copy()
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(hessian))
            or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(limits))):
        return x, 3, 0, np.inf
    slack = np.empty(m, dtype=np.float64)
    for row in range(m):
        slack[row] = -limits[row]
        for j in range(n):
            slack[row] += matrix[row, j] * x[j]
    if abs(slack[0]) > 1e-7 or (m > 1 and np.min(slack[1:]) < -1e-7):
        return x, 2, 0, np.inf
    scale = max(np.max(np.abs(hessian)), np.max(np.abs(linear)), 1e-12)
    h = hessian / scale
    f = linear / scale
    is_linear = np.max(np.abs(hessian)) == 0.0
    direction_hessian = h.copy()
    # A tiny positive diagonal regularizes only the direction equation, not
    # the objective or its gradient. Success is checked against the original H.
    for j in range(n):
        direction_hessian[j, j] += 1.0 if is_linear else 1e-10
    active = np.full(n, -1, dtype=np.int64)
    active[0] = 0
    active_count = 1
    selected = np.zeros(m, dtype=np.bool_)
    selected[0] = True
    residual = np.inf
    for iteration in range(max_iterations):
        gradient = h @ x + f
        kkt = np.zeros((n + active_count, n + active_count), dtype=np.float64)
        rhs = np.zeros(n + active_count, dtype=np.float64)
        kkt[:n, :n] = direction_hessian
        rhs[:n] = -gradient
        for k in range(active_count):
            for j in range(n):
                coefficient = matrix[active[k], j]
                kkt[j, n + k] = coefficient
                kkt[n + k, j] = coefficient
        try:
            solution = np.linalg.solve(kkt, rhs)
        except Exception:
            return x, 3, iteration + 1, np.inf
        direction = solution[:n]
        multipliers = solution[n:]
        stationarity = gradient.copy()
        for k in range(active_count):
            stationarity += multipliers[k] * matrix[active[k]]
        residual = np.max(np.abs(stationarity))
        if np.max(np.abs(direction)) <= tolerance:
            remove = -1
            largest = tolerance
            for k in range(1, active_count):
                # With A*x >= b and this KKT sign, inequality lambda <= 0.
                if multipliers[k] > largest:
                    largest = multipliers[k]
                    remove = k
            if remove < 0:
                for row in range(m):
                    slack[row] = -limits[row]
                    for j in range(n):
                        slack[row] += matrix[row, j] * x[j]
                if (residual <= tolerance * 10 and abs(slack[0]) <= 1e-7
                        and (m == 1 or np.min(slack[1:]) >= -1e-7)):
                    return x, 0, iteration + 1, residual
                return x, 3, iteration + 1, residual
            selected[active[remove]] = False
            for k in range(remove, active_count - 1):
                active[k] = active[k + 1]
            active_count -= 1
            continue
        alpha = np.inf if is_linear else 1.0
        blocker = -1
        for row in range(1, m):
            if selected[row]:
                continue
            slope = 0.0
            room = -limits[row]
            for j in range(n):
                slope += matrix[row, j] * direction[j]
                room += matrix[row, j] * x[j]
            if slope >= -1e-12:
                continue
            candidate_alpha = max(0.0, room) / -slope
            if candidate_alpha <= alpha and independent_constraint_kernel(matrix, active, active_count, row):
                alpha = candidate_alpha
                blocker = row
        if not np.isfinite(alpha):
            return x, 3, iteration + 1, residual
        x += alpha * direction
        if blocker >= 0 and active_count < n:
            active[active_count] = blocker
            active_count += 1
            selected[blocker] = True
    return x, 1, max_iterations, residual


QP_KERNELS = (independent_constraint_kernel, feasible_qp_kernel)
for _kernel in QP_KERNELS:
    _kernel.disable_compile()
