"""Feasible active-set QP/LP steps for target-grid portfolio optimization.

The historical adapter retains one equality. The common core accepts an explicit
equality count; the matrix entry point rank-reduces and scales E/e and G/h.
Inputs are read-only views. Only the iterate and the small KKT workspace are
owned here. No external Python solver, callback or request-time compilation.
"""
from __future__ import annotations

import numpy as np
from numba import njit, types

_R1 = types.Array(types.float64, 1, "A", readonly=True)
_R2 = types.Array(types.float64, 2, "A", readonly=True)
_I1 = types.Array(types.int64, 1, "A", readonly=True)
_DETAIL = types.Tuple((types.float64[::1], types.int64, types.int64, types.float64[::1]))
_CERT_DETAIL = types.Tuple((types.float64[::1], types.int64, types.int64, types.float64[::1], types.float64[::1]))
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


@njit(_CERT_DETAIL(_R2, _R1, _R2, _R1, _R1, types.int64, types.float64, types.int64, types.boolean), cache=False, nogil=True)
def _active_set_qp_solve(hessian, linear, matrix, limits, initial, max_iterations, tolerance, equality_count, kkt_stopping):
    """One numerical core: minimize .5*x'H*x + f'x from a feasible point.

    First equality_count rows are permanent equalities, the rest are >=.
    kkt_stopping enables PSD stationarity stopping for the new matrix ABI.

    Status: 0 optimum to tolerance, 1 iteration limit, 2 infeasible start,
    3 numerical failure. H=0 is an LP: move to a blocking face, not a
    regularized approximation of the requested linear objective.
    """
    n = initial.size
    m = limits.size
    if (n == 0 or equality_count < 0 or equality_count > min(n, m) or hessian.shape != (n, n) or linear.size != n
            or matrix.shape != (m, n) or max_iterations < 1 or tolerance <= 0):
        raise ValueError("QP_INPUT_AXIS")
    diagnostics = np.full(4, np.inf)
    dual = np.zeros(m, dtype=np.float64)
    x = initial.copy()
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(hessian))
            or not np.all(np.isfinite(linear)) or not np.all(np.isfinite(matrix))
            or not np.all(np.isfinite(limits))):
        return x, 3, 0, diagnostics, dual
    slack = np.empty(m, dtype=np.float64)
    for row in range(m):
        slack[row] = -limits[row]
        for j in range(n):
            slack[row] += matrix[row, j] * x[j]
    if (equality_count > 0 and np.max(np.abs(slack[:equality_count])) > 1e-7) or (m > equality_count and np.min(slack[equality_count:]) < -1e-7):
        return x, 2, 0, diagnostics, dual
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
    active_count = equality_count
    selected = np.zeros(m, dtype=np.bool_)
    for k in range(equality_count):
        active[k] = k
        selected[k] = True
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
            diagnostics[1] = np.inf
            return x, 3, iteration + 1, diagnostics, dual
        direction = solution[:n]
        multipliers = solution[n:]
        dual[:] = 0.0
        for k in range(active_count):
            dual[active[k]] = -multipliers[k] * scale
        stationarity = gradient.copy()
        for k in range(active_count):
            stationarity += multipliers[k] * matrix[active[k]]
        residual = np.max(np.abs(stationarity))
        diagnostics[1] = residual
        diagnostics[2], diagnostics[3] = 0.0, 0.0
        for k in range(equality_count, active_count):
            diagnostics[2] = max(diagnostics[2], multipliers[k])
            room = np.dot(matrix[active[k]], x) - limits[active[k]]
            diagnostics[3] = max(diagnostics[3], abs(multipliers[k] * room))
        # PSD null directions can remain noisy after stationarity is satisfied.
        # The historical ABI retains its original direction-only stopping rule.
        if np.max(np.abs(direction)) <= tolerance or (kkt_stopping and residual <= tolerance):
            remove = -1
            largest = tolerance
            for k in range(equality_count, active_count):
                # With A*x >= b and this KKT sign, inequality lambda <= 0.
                if multipliers[k] > largest:
                    largest = multipliers[k]
                    remove = k
            if remove < 0:
                for row in range(m):
                    slack[row] = -limits[row]
                    for j in range(n):
                        slack[row] += matrix[row, j] * x[j]
                diagnostics[0] = 0.0
                for row in range(m):
                    violation = abs(slack[row]) if row < equality_count else max(0.0, -slack[row])
                    diagnostics[0] = max(diagnostics[0], violation)
                if (residual <= tolerance * 10 and diagnostics[0] <= 1e-7):
                    return x, 0, iteration + 1, diagnostics, dual
                return x, 3, iteration + 1, diagnostics, dual
            selected[active[remove]] = False
            for k in range(remove, active_count - 1):
                active[k] = active[k + 1]
            active_count -= 1
            continue
        alpha = np.inf if is_linear else 1.0
        blocker = -1
        for row in range(equality_count, m):
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
            return x, 3, iteration + 1, diagnostics, dual
        x += alpha * direction
        if blocker >= 0 and active_count < n:
            active[active_count] = blocker
            active_count += 1
            selected[blocker] = True
    return x, 1, max_iterations, diagnostics, dual


@njit(_DETAIL(_R2, _R1, _R2, _R1, _R1, types.int64, types.float64, types.int64, types.boolean), cache=False, nogil=True)
def active_set_qp_core(hessian, linear, matrix, limits, initial, max_iterations, tolerance, equality_count, kkt_stopping):
    """Preserve the existing QP ABI while sharing the primal/dual solve."""
    x, status, used, diagnostics, _ = _active_set_qp_solve(
        hessian, linear, matrix, limits, initial, max_iterations, tolerance, equality_count, kkt_stopping)
    return x, status, used, diagnostics


@njit((_R1, _R2, _R1, _R1, _R1, _R1, types.int64, types.float64), cache=False, nogil=True)
def bounded_lp_kernel(linear, matrix, limits, initial, lower, upper, max_iterations, tolerance):
    """LP with one equality, >= rows and declared finite bounding box.

    Dual inequalities are clipped to nonnegative values. The remaining
    stationarity error is minimized over the box, giving a conservative lower
    bound even when the active-set iteration did not converge. The caller must
    encode the same box in its constraints; a residual is never an infeasibility
    certificate by itself. diagnostics[4:] = lower bound, primal-dual gap.
    """
    n = linear.size
    if (lower.size != n or upper.size != n or not np.all(np.isfinite(lower))
            or not np.all(np.isfinite(upper)) or np.any(lower > upper) or limits.size < 1):
        raise ValueError("LP_BOUNDING_BOX")
    x, status, used, checks, dual = _active_set_qp_solve(
        np.zeros((n, n)), linear, matrix, limits, initial, max_iterations, tolerance, 1, True)
    residual = linear.copy()
    bound = 0.0
    magnitude = 1.0
    for row in range(limits.size):
        multiplier = dual[row] if row == 0 else max(0.0, dual[row])
        bound += multiplier * limits[row]
        magnitude += abs(multiplier * limits[row])
        for j in range(n):
            residual[j] -= multiplier * matrix[row, j]
            magnitude += abs(multiplier * matrix[row, j]) * max(abs(lower[j]), abs(upper[j]))
    for j in range(n):
        bound += residual[j] * (lower[j] if residual[j] >= 0 else upper[j])
    # Floating arithmetic reserve, separate from the caller's optimality tolerance.
    bound -= 1e-12 * magnitude
    result = np.empty(6)
    result[:4] = checks
    result[4] = bound
    result[5] = np.dot(linear, x) - bound
    return x, status, used, result


@njit(_RESULT(_R2, _R1, _R2, _R1, _R1, types.int64, types.float64), cache=False, nogil=True)
def feasible_qp_kernel(hessian, linear, matrix, limits, initial, max_iterations, tolerance):
    """Historical ABI: first row equality, remaining rows >=; statuses 0/1/2/3.

    The algorithm, tolerances, iteration budget and scalar residual are retained.
    """
    if limits.size == 0:
        raise ValueError("QP_INPUT_AXIS")
    x, status, used, diagnostics = active_set_qp_core(
        hessian, linear, matrix, limits, initial, max_iterations, tolerance, 1, False)
    return x, status, used, diagnostics[1]


@njit((_R2, _R1, _R2, _R1), cache=False, nogil=True)
def prepare_constraints_kernel(equalities, values, inequalities, limits):
    """Orthonormal equality basis plus unit inequality rows; retain original gate.

    Return combined matrix, limits, equality rank, status (4 = inconsistent).
    Two-pass modified Gram-Schmidt applies identical operations to the RHS.
    """
    n = equalities.shape[1]
    if values.size != equalities.shape[0] or inequalities.shape != (limits.size, n):
        raise ValueError("QP_CONSTRAINT_AXIS")
    basis = np.zeros((n, n))
    rhs = np.zeros(n)
    rank = 0
    if (not np.all(np.isfinite(equalities)) or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(inequalities)) or not np.all(np.isfinite(limits))):
        return basis, rhs, rank, 3
    for i in range(values.size):
        v = equalities[i].copy()
        norm = np.sqrt(np.dot(v, v))
        if not np.isfinite(norm) or (norm == 0.0 and np.any(v != 0.0)):
            return basis, rhs, rank, 3
        if norm == 0.0:
            if abs(values[i]) > 1e-7:
                return basis, rhs, rank, 4
            continue
        # Only exact proportional input rows certify a nonzero-RHS conflict.
        # A merely small singular direction can still have a very large solution.
        for previous in range(i):
            pivot = -1
            for j in range(n):
                if equalities[previous, j] != 0.0:
                    pivot = j
                    break
            if pivot >= 0:
                ratio = equalities[i, pivot] / equalities[previous, pivot]
                if np.isfinite(ratio) and np.all(equalities[i] == ratio*equalities[previous]):
                    difference = values[i] - ratio*values[previous]
                    if np.isfinite(difference) and abs(difference) > 1e-7:
                        return basis, rhs, rank, 4
        v /= norm
        value = values[i] / norm
        for _ in range(2):
            for j in range(rank):
                c = np.dot(v, basis[j])
                v -= c * basis[j]
                value -= c * rhs[j]
        remaining = np.sqrt(np.dot(v, v))
        if remaining <= 1e-10:
            if abs(value) > 1e-7:
                return basis, rhs, rank, 3
        elif rank < n:
            basis[rank] = v / remaining
            rhs[rank] = value / remaining
            rank += 1
        else:
            return basis, rhs, rank, 3
    matrix = np.zeros((rank + limits.size, n))
    bound = np.zeros(rank + limits.size)
    matrix[:rank] = basis[:rank]
    bound[:rank] = rhs[:rank]
    for i in range(limits.size):
        norm = np.sqrt(np.dot(inequalities[i], inequalities[i]))
        if not np.isfinite(norm) or (norm == 0.0 and np.any(inequalities[i] != 0.0)):
            return matrix, bound, rank, 3
        if norm == 0.0:
            if limits[i] > 1e-7:
                return matrix, bound, rank, 4
            norm = 1.0
        matrix[rank + i] = inequalities[i] / norm
        bound[rank + i] = limits[i] / norm
    return matrix, bound, rank, 0


@njit(types.float64(_R2, _R1, _R2, _R1, _R1), cache=False, nogil=True)
def constraint_violation_kernel(equalities, values, inequalities, limits, x):
    violation = 0.0
    for i in range(values.size):
        violation = max(violation, abs(np.dot(equalities[i], x) - values[i]))
    for i in range(limits.size):
        violation = max(violation, limits[i] - np.dot(inequalities[i], x))
    return violation


@njit(_DETAIL(_R2, _R1, _R2, _R1, _R2, _R1, _R2, _R1, types.int64, _R1, types.int64, types.float64), cache=False, nogil=True)
def prepared_matrix_qp_kernel(hessian, linear, equalities, values, inequalities, limits,
                              matrix, bound, rank, initial, max_iterations, tolerance):
    """Internal prepared-matrix entry, avoiding row copies at each grid target.

    matrix/bound/rank MUST come from prepare_constraints_kernel(E,e,G,h).
    Only a target RHS may be updated, in both original and normalized units.
    Returns x,status,iterations,[original primal,normalized KKT,dual,complementarity].
    """
    n = initial.size
    if (n < 1 or hessian.shape != (n, n) or linear.size != n
            or equalities.shape != (values.size, n) or inequalities.shape != (limits.size, n)
            or not 1 <= max_iterations <= 1000 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("QP_INPUT_AXIS")
    if (not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(linear))
            or not np.all(np.isfinite(equalities)) or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(inequalities)) or not np.all(np.isfinite(limits))
            or not np.all(np.isfinite(initial))):
        return np.full(n, np.nan), 3, 0, np.full(4, np.nan)
    if matrix.shape != (bound.size, n) or not 0 <= rank <= min(n, bound.size):
        raise ValueError("QP_PREPARED_AXIS")
    if constraint_violation_kernel(equalities, values, inequalities, limits, initial) > 1e-7:
        return initial.copy(), 2, 0, np.full(4, np.nan)
    x, status, used, diagnostics = active_set_qp_core(
        hessian, linear, matrix, bound, initial, max_iterations, tolerance, rank, True)
    diagnostics[0] = constraint_violation_kernel(equalities, values, inequalities, limits, x)
    if status == 0 and (not np.all(np.isfinite(x)) or not np.all(np.isfinite(diagnostics))
                       or diagnostics[0] > 1e-7 or diagnostics[2] > tolerance
                       or diagnostics[3] > tolerance * 10):
        status = 3
    return x, status, used, diagnostics


@njit(_DETAIL(_R2, _R1, _R2, _R1, _R2, _R1, _R1, types.int64, types.float64), cache=False, nogil=True)
def matrix_qp_kernel(hessian, linear, equalities, values, inequalities, limits,
                     initial, max_iterations, tolerance):
    """Explicit E*x=e, G*x>=h entry to the same active-set core.

    Caller supplies a convex H (the frontier validates PSD once per request).
    Returns x,status,iterations,[original primal,normalized KKT,dual,complementarity].
    Status 4 is structural inconsistency; 2 only a bad start; 3 includes ambiguous
    numerical rank. Nonfinite diagnostics always have a nonzero status.
    """
    n = initial.size
    if (n < 1 or hessian.shape != (n, n) or linear.size != n
            or equalities.shape != (values.size, n) or inequalities.shape != (limits.size, n)
            or not 1 <= max_iterations <= 1000 or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("QP_INPUT_AXIS")
    if (not np.all(np.isfinite(equalities)) or not np.all(np.isfinite(values))
            or not np.all(np.isfinite(inequalities)) or not np.all(np.isfinite(limits))):
        return np.full(n, np.nan), 3, 0, np.full(4, np.nan)
    matrix, bound, rank, status = prepare_constraints_kernel(equalities, values, inequalities, limits)
    if status:
        return initial.copy(), status, 0, np.full(4, np.nan)
    return prepared_matrix_qp_kernel(hessian, linear, equalities, values, inequalities, limits,
                                     matrix, bound, rank, initial, max_iterations, tolerance)


QP_KERNELS = (independent_constraint_kernel, _active_set_qp_solve, active_set_qp_core, bounded_lp_kernel, feasible_qp_kernel,
              prepare_constraints_kernel, constraint_violation_kernel,
              prepared_matrix_qp_kernel, matrix_qp_kernel)
for _kernel in QP_KERNELS:
    _kernel.disable_compile()
