"""Independent frontier geometry, constrained segmentation, bands and portraits.

Direction-angle variation is NOT a second derivative curvature estimator.
There is one suffix-DP engine for the three automatic segmentation objectives.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import threading

import numpy as np
from numba import njit, types

from backend import frontier_moments

V = types.Array(types.float64, 1, "A", readonly=True)
M = types.Array(types.float64, 2, "A", readonly=True)
I = types.Array(types.int64, 1, "A", readonly=True)
VERSION = "risk-scale-numeric/1.1.0"
BOUNDARY_TOL = 1e-10
SPAN_TOL = 1e-10
ALGORITHMS = {
    "frontier_shape_dp_v2": 0,
    "equal_volatility_v1": 1,
    "equal_arclength_v1": 2,
    "equal_return_v1": 3,
    "manual_volatility_bands_v1": 4,
}
SEGMENTATION_STATUS = {0: "ok", 1: "invalid_frontier", 2: "degenerate_frontier",
                       3: "segmentation_insufficient_resolution", 4: "frontier_gap"}
_WARMED_PID = None
_WARMED_FINGERPRINT = None
_LOCK = threading.RLock()


@njit((M, I, types.float64), cache=False, nogil=True)
def frontier_geometry_kernel(metrics, indices, span_tolerance):
    """Return x,y,angles,lengths,prefix,status. prefix rows are H,Htheta,Htheta².

    Node arrays have length J+1, edge arrays J. status 2 means zero span,
    1 invalid/nonmonotonic nodes. Failed solver points must be gated beforehand.
    """
    n = indices.size
    if metrics.shape[1] != 2 or n > 200 or not np.isfinite(span_tolerance) or span_tolerance <= 0:
        raise ValueError("RISK_GEOMETRY_AXIS")
    x, y = np.full(n, np.nan), np.full(n, np.nan)
    angles, lengths = np.full(max(0, n-1), np.nan), np.full(max(0, n-1), np.nan)
    prefix = np.zeros((3, n))
    for i in range(n):
        if indices[i] < 0 or indices[i] >= metrics.shape[0] or (i and indices[i] <= indices[i-1]):
            raise ValueError("RISK_GEOMETRY_INDICES")
        if not np.all(np.isfinite(metrics[indices[i]])) or metrics[indices[i], 0] < 0:
            return x, y, angles, lengths, prefix, 1
    if n < 2:
        return x, y, angles, lengths, prefix, 2
    dx = metrics[indices[-1], 0] - metrics[indices[0], 0]
    dy = metrics[indices[-1], 1] - metrics[indices[0], 1]
    if dx < 0 or dy < 0:
        return x, y, angles, lengths, prefix, 1
    if dx <= span_tolerance or dy <= span_tolerance:
        return x, y, angles, lengths, prefix, 2
    for i in range(n):
        x[i] = (metrics[indices[i], 0]-metrics[indices[0], 0])/dx
        y[i] = (metrics[indices[i], 1]-metrics[indices[0], 1])/dy
        if i:
            rx, ry = x[i]-x[i-1], y[i]-y[i-1]
            if rx <= 0 or ry <= 0:
                return x, y, angles, lengths, prefix, 1
            angle = np.arctan2(ry, rx)
            length = np.hypot(rx, ry)
            angles[i-1], lengths[i-1] = angle, length
            prefix[0, i] = prefix[0, i-1]+length
            prefix[1, i] = prefix[1, i-1]+length*angle
            prefix[2, i] = prefix[2, i-1]+length*angle*angle
    if not np.all(np.isfinite(prefix)):
        return x, y, angles, lengths, prefix, 1
    return x, y, angles, lengths, prefix, 0


@njit(types.float64(M, types.int64, types.int64), cache=False, nogil=True)
def interval_shape_cost_kernel(prefix, left, right):
    """Weighted within-interval direction dispersion, O(1) from prefix sums."""
    if prefix.shape[0] != 3 or left < 0 or right <= left or right >= prefix.shape[1]:
        raise ValueError("RISK_INTERVAL_AXIS")
    mass = prefix[0, right]-prefix[0, left]
    first = prefix[1, right]-prefix[1, left]
    second = prefix[2, right]-prefix[2, left]
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(first) or not np.isfinite(second):
        raise ValueError("RISK_INTERVAL_NONFINITE")
    square = first*first/mass
    value = second-square
    rounding = 64*np.finfo(np.float64).eps*max(1.0, abs(second), abs(square))
    if not np.isfinite(value) or value < -rounding:
        raise ValueError("RISK_INTERVAL_NEGATIVE_COST")
    return max(0.0, value)


@njit(types.float64(V, V, M, types.int64, types.int64, types.int64, types.int64), cache=False, nogil=True)
def segmentation_cost_kernel(x, y, prefix, left, right, completed_segments, objective):
    if objective == 0:
        return interval_shape_cost_kernel(prefix, left, right)
    if completed_segments == 5:
        return 0.0
    if objective == 1:
        coordinate = x[right]
    elif objective == 2:
        coordinate = prefix[0, right]/prefix[0, -1]
    else:
        coordinate = y[right]
    distance = coordinate-completed_segments/5.0
    return distance*distance


@njit((V, V, V, M, types.int64, types.int64, types.float64, types.float64, types.float64), cache=False, nogil=True)
def segment_frontier_kernel(x, y, angles, prefix, algorithm, min_edges, min_span, tie_tolerance, near_linear_threshold):
    """Return cuts[4], status, chosen_cost, optimum, fallback_code (1=near linear).

    All objectives O(5 J²) time / O(5 J) memory. Equal-grid modes minimize the
    sum of squared normalized cut distances under the same interval constraints.
    Earliest complete lexicographic path within ONE global objective tolerance.
    """
    n = x.size
    cuts = np.full(4, -1, dtype=np.int64)
    if (not 0 <= algorithm <= 3 or min_edges < 1 or not np.isfinite(min_span)
            or not 0 < min_span <= .2 or not np.isfinite(tie_tolerance) or tie_tolerance < 0
            or not np.isfinite(near_linear_threshold) or near_linear_threshold < 0
            or prefix.shape != (3, n) or angles.size != max(0, n-1) or n > 200):
        raise ValueError("RISK_SEGMENTATION_PARAMETERS")
    if n < 2:
        return cuts, 3, np.nan, np.nan, 0
    if (y.size != n or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.all(np.isfinite(angles)) or not np.all(np.isfinite(prefix))
            or abs(x[0]) > 1e-12 or abs(x[-1]-1.) > 1e-12 or abs(y[0]) > 1e-12 or abs(y[-1]-1.) > 1e-12
            or np.any((x[1:]-x[:-1]) <= 0) or np.any((y[1:]-y[:-1]) <= 0) or np.any((prefix[0, 1:]-prefix[0, :-1]) <= 0)
            or np.any(angles < 0) or np.any(angles > np.pi/2)):
        return cuts, 1, np.nan, np.nan, 0
    objective = algorithm
    fallback = 0
    if algorithm == 0 and np.max(angles)-np.min(angles) < near_linear_threshold:
        objective, fallback = 2, 1
    if n-1 < 5*min_edges:
        return cuts, 3, np.nan, np.nan, fallback
    suffix = np.full((6, n), np.inf)
    suffix[0, n-1] = 0.0
    for remaining in range(1, 6):
        completed = 6-remaining
        for left in range(n-1, -1, -1):
            for right in range(left+min_edges, n):
                if x[right]-x[left]+1e-14 < min_span or not np.isfinite(suffix[remaining-1, right]):
                    continue
                cost = segmentation_cost_kernel(x, y, prefix, left, right, completed, objective)
                candidate = cost+suffix[remaining-1, right]
                if candidate < suffix[remaining, left]:
                    suffix[remaining, left] = candidate
    optimum = suffix[5, 0]
    if not np.isfinite(optimum):
        return cuts, 3, np.nan, np.nan, fallback
    used = 0.0
    left = 0
    for stage in range(1, 5):
        selected = -1
        for right in range(left+min_edges, n):
            if x[right]-x[left]+1e-14 < min_span or not np.isfinite(suffix[5-stage, right]):
                continue
            cost = segmentation_cost_kernel(x, y, prefix, left, right, stage, objective)
            if used+cost+suffix[5-stage, right] <= optimum+tie_tolerance:
                selected = right
                used += cost
                break
        if selected < 0:
            return cuts, 1, np.nan, optimum, fallback
        cuts[stage-1], left = selected, selected
    used += segmentation_cost_kernel(x, y, prefix, left, n-1, 5, objective)
    if used > optimum+tie_tolerance:
        return cuts, 1, used, optimum, fallback
    return cuts, 0, used, optimum, fallback


@njit((V,), cache=False, nogil=True)
def validate_caps_kernel(caps):
    if caps.size != 5:
        raise ValueError("RISK_CAPS_AXIS")
    for i in range(5):
        if not np.isfinite(caps[i]) or caps[i] < 0 or (i and caps[i] <= caps[i-1]):
            raise ValueError("RISK_CAPS_STRICT_ORDER")


@njit(types.int64(types.float64, types.float64, types.float64), cache=False, nogil=True)
def risk_within_cap_kernel(risk, cap, tolerance):
    """1 within, 0 exceeds, -1 unavailable; exactly the classifier comparison."""
    if not np.isfinite(tolerance) or tolerance < 0 or not np.isfinite(cap) or cap < 0:
        raise ValueError("RISK_CAP_COMPARISON")
    if not np.isfinite(risk) or risk < 0:
        return -1
    return 1 if risk-cap <= tolerance else 0


@njit((types.float64, V, types.float64, types.float64), cache=False, nogil=True)
def classify_risk_kernel(risk, caps, reference_minimum, tolerance):
    """(level, flags): 0 unavailable, 1..5 bands, 6 above_scale.

    flags bit 1: below reference minimum; bit 2: within boundary tolerance.
    Unknown/negative risk never becomes C1. Equality belongs to the lower band.
    """
    validate_caps_kernel(caps)
    if not np.isfinite(reference_minimum) or reference_minimum < 0:
        raise ValueError("RISK_REFERENCE_MINIMUM")
    within = risk_within_cap_kernel(risk, caps[0], tolerance)
    if within == -1:
        return 0, 0
    flags = 1 if risk < reference_minimum-tolerance else 0
    for i in range(5):
        if abs(risk-caps[i]) <= tolerance:
            flags |= 2
        if risk_within_cap_kernel(risk, caps[i], tolerance) == 1:
            return i+1, flags
    return 6, flags


@njit((V, V, types.float64), cache=False, nogil=True)
def manual_volatility_bands_kernel(risks, caps, tolerance):
    """Return cuts[4], caps copy, not_calibrated[5]. -1 means no node below cap.

    Manual caps require no five-segment density. Empty reference is permitted;
    all bands then have uncalibrated coverage and no invented representative.
    """
    validate_caps_kernel(caps)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("RISK_BOUNDARY_TOLERANCE")
    if not np.all(np.isfinite(risks)) or np.any(risks < 0) or np.any((risks[1:]-risks[:-1]) < 0):
        raise ValueError("RISK_MANUAL_REFERENCE")
    cuts = np.full(4, -1, dtype=np.int64)
    not_calibrated = np.ones(5, dtype=np.int64)
    for band in range(5):
        if risks.size:
            not_calibrated[band] = 1 if risk_within_cap_kernel(caps[band], risks[-1], tolerance) == 0 else 0
        if band < 4:
            for j in range(risks.size):
                if risk_within_cap_kernel(risks[j], caps[band], tolerance) == 1:
                    cuts[band] = j
    return cuts, caps.copy(), not_calibrated


@njit((V, V, V, types.float64), cache=False, nogil=True)
def representative_nodes_kernel(risks, cumulative_length, caps, tolerance):
    """Band-local arclength midpoint, earliest node on ties; -1 for empty bands.

    Midpoint is halfway between the FIRST and LAST admissible nodes' cumulative
    arclength. A boundary node belongs solely to the lower band.
    """
    validate_caps_kernel(caps)
    if (risks.size != cumulative_length.size or not np.all(np.isfinite(risks)) or np.any(risks < 0)
            or not np.all(np.isfinite(cumulative_length)) or np.any((risks[1:]-risks[:-1]) < 0)
            or np.any((cumulative_length[1:]-cumulative_length[:-1]) < 0) or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("RISK_REPRESENTATIVE_AXIS")
    representatives = np.full(5, -1, dtype=np.int64)
    first, last = np.full(5, -1, dtype=np.int64), np.full(5, -1, dtype=np.int64)
    for i in range(risks.size):
        band, _ = classify_risk_kernel(risks[i], caps, 0., tolerance)
        if 1 <= band <= 5:
            if first[band-1] < 0:
                first[band-1] = i
            last[band-1] = i
    for band in range(5):
        if first[band] < 0:
            continue
        target = .5*(cumulative_length[first[band]]+cumulative_length[last[band]])
        distance = np.inf
        for i in range(first[band], last[band]+1):
            value = abs(cumulative_length[i]-target)
            tie = 16*np.finfo(np.float64).eps*max(1.0, abs(target))
            if value < distance-tie:
                representatives[band], distance = i, value
    return representatives


@njit((V, types.float64, types.float64), cache=False, nogil=True)
def historical_tail_kernel(returns, alpha, minimum_tail_mass):
    """(VaR, ES, tail_mass, status): 0 ok, 1 thin tail, 2 unavailable.

    Empirical inverse CDF VaR; fractional upper loss-tail ES. No annualization,
    no loss clipping; ties counted by probability mass, not value threshold.
    """
    if not 0 < alpha < 1 or not np.isfinite(minimum_tail_mass) or minimum_tail_mass <= 0:
        raise ValueError("RISK_TAIL_PARAMETERS")
    n = returns.size
    if n == 0 or not np.all(np.isfinite(returns)) or np.any(returns < -1):
        return np.nan, np.nan, 0., 2
    losses = -returns.copy()
    losses.sort()
    q = (1-alpha)*n
    # Stabilize mathematically integral probability masses near binary roundoff.
    nearest = np.rint(q)
    if abs(q-nearest) <= 8*np.finfo(np.float64).eps*max(1., q) and nearest > 0:
        q = nearest
    count = int(np.floor(q))
    fraction = q-count
    total = 0.0
    for i in range(count):
        total += losses[n-1-i]
    if fraction > 0:
        total += fraction*losses[n-1-count]
    quantile_mass = alpha*n
    nearest = np.rint(quantile_mass)
    if abs(quantile_mass-nearest) <= 8*np.finfo(np.float64).eps*max(1., quantile_mass):
        quantile_mass = nearest
    position = max(0, min(n-1, int(np.ceil(quantile_mass))-1))
    return losses[position], total/q, q, 1 if q < minimum_tail_mass else 0


@njit((V,), cache=False, nogil=True)
def historical_drawdown_kernel(returns):
    """(maximum drawdown, status 0/2); initial NAV=1 participates in the peak."""
    if returns.size == 0:
        return np.nan, 2
    value, peak, drawdown = 1.0, 1.0, 0.0
    for r in returns:
        if not np.isfinite(r) or r < -1:
            return np.nan, 2
        value *= 1+r
        if not np.isfinite(value):
            return np.nan, 2
        peak = max(peak, value)
        drawdown = max(drawdown, 1-value/peak)
    return drawdown, 0


@njit((V, V, V, V, types.float64, types.float64), cache=False, nogil=True)
def boundary_stability_kernel(caps_base, caps_dense, endpoints_base, endpoints_dense,
                              endpoint_tolerance, boundary_tolerance):
    """(max normalized boundary shift, status): 0 stable,1 unstable,2 endpoints,3 degenerate.

    Endpoints are [GMV volatility,GMV return,right volatility,right return].
    Dense boundaries always use the BASE volatility span after endpoint checking.
    """
    validate_caps_kernel(caps_base)
    validate_caps_kernel(caps_dense)
    if (endpoints_base.size != 4 or endpoints_dense.size != 4 or not np.isfinite(endpoint_tolerance)
            or endpoint_tolerance < 0 or not np.isfinite(boundary_tolerance) or boundary_tolerance < 0):
        raise ValueError("RISK_STABILITY_AXIS")
    if (not np.all(np.isfinite(endpoints_base)) or not np.all(np.isfinite(endpoints_dense))
            or np.max(np.abs(endpoints_base-endpoints_dense)) > endpoint_tolerance):
        return np.nan, 2
    span = endpoints_base[2]-endpoints_base[0]
    if span <= SPAN_TOL:
        return np.nan, 3
    shift = np.max(np.abs(caps_base[:4]-caps_dense[:4]))/span
    return shift, 1 if shift > boundary_tolerance else 0


KERNELS = (frontier_geometry_kernel, interval_shape_cost_kernel, segmentation_cost_kernel,
           segment_frontier_kernel, validate_caps_kernel, risk_within_cap_kernel,
           classify_risk_kernel, manual_volatility_bands_kernel, representative_nodes_kernel,
           historical_tail_kernel, historical_drawdown_kernel, boundary_stability_kernel)
for _dispatcher in KERNELS:
    _dispatcher.disable_compile()


def execution_audit():
    dependency = frontier_moments.execution_audit()
    signatures = {k.__name__: [str(s) for s in k.signatures] for k in KERNELS}
    fingerprints = {k.__name__: hashlib.sha256(inspect.getsource(k.py_func).encode()).hexdigest() for k in KERNELS}
    signatures = {**dependency["kernel_signatures"], **signatures}
    fingerprints = {**dependency["kernel_fingerprints"], **fingerprints}
    fingerprint = hashlib.sha256(repr((VERSION, signatures, fingerprints)).encode()).hexdigest()
    compiled = all(len(k.signatures) == len(k.nopython_signatures) == 1 and not k._can_compile
                   and not any(v.objectmode for v in k.overloads.values()) for k in KERNELS)
    compiled = compiled and dependency["nopython"]
    ready = dependency["complete"] and compiled and _WARMED_PID == os.getpid() and _WARMED_FINGERPRINT == fingerprint
    return {"backend": "numba_njit_fixed_signature", "kernel_version": VERSION,
            "complete": ready, "fully_warmed": ready, "warmed_pid": _WARMED_PID, "pid": os.getpid(),
            "nopython": compiled, "object_mode": 0, "python_fallback": 0, "request_time_compilation": 0,
            "kernel_signatures": signatures, "kernel_fingerprints": fingerprints, "fingerprint": fingerprint}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("RISK_SCALE_NOT_READY")


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    with _LOCK:
        _WARMED_PID = None
        if not frontier_moments.execution_audit()["complete"]:
            frontier_moments.warm()
        values = np.linspace(.01, .3, 61)
        metrics = np.column_stack((values, np.sqrt(values)))[::2]
        indices = np.arange(metrics.shape[0], dtype=np.int64)
        metrics.flags.writeable = indices.flags.writeable = False
        geometry = frontier_geometry_kernel(metrics, indices, SPAN_TOL)
        if geometry[5]:
            raise RuntimeError("RISK_SCALE_GEOMETRY_WARMUP_FAILED")
        x, y, angles, _, prefix, _ = geometry
        for algorithm in range(4):
            cuts, status, _, _, _ = segment_frontier_kernel(x, y, angles, prefix, algorithm, 5, .05, 1e-12, .02)
            if status:
                raise RuntimeError("RISK_SCALE_SEGMENT_WARMUP_FAILED")
        caps = np.array([.05, .1, .15, .2, .3])
        manual_volatility_bands_kernel(metrics[:, 0], caps, BOUNDARY_TOL)
        representative_nodes_kernel(metrics[:, 0], prefix[0], caps, BOUNDARY_TOL)
        historical_tail_kernel(np.array([-.1, .02, .01]), .95, 5.)
        historical_drawdown_kernel(np.array([-.1, .02, .01]))
        endpoints = np.array([.01, .02, .3, .1])
        boundary_stability_kernel(caps, caps, endpoints, endpoints, 1e-8, .02)
        _WARMED_FINGERPRINT = execution_audit()["fingerprint"]
        _WARMED_PID = os.getpid()
        require_ready()
        return execution_audit()


def segment_frontier(metrics, statuses, algorithm_id="frontier_shape_dp_v2", *,
                     manual_caps=None, min_edges=5, min_span=.05,
                     tie_tolerance=1e-12, near_linear_threshold=.02):
    """Service orchestration; all numerical work stays in the independent kernels.

    Returned indices/cuts/representatives reference ORIGINAL frontier rows, never
    a copied weight matrix. A gap blocks automatic segmentation. Manual bands
    remain valid with no usable reference and explicitly missing representatives.
    """
    require_ready()
    if algorithm_id not in ALGORITHMS:
        raise ValueError("RISK_SCALE_ALGORITHM")
    if isinstance(min_edges, bool) or not isinstance(min_edges, (int, np.integer)):
        raise ValueError("RISK_SCALE_MIN_EDGES")
    indices, curve_status = frontier_moments.frontier_curve_indices_kernel(metrics, statuses, BOUNDARY_TOL)
    geometry = frontier_geometry_kernel(metrics, indices, SPAN_TOL)
    x, y, angles, lengths, prefix, geometry_status = geometry
    result = {"algorithm_id": algorithm_id, "algorithm_version": VERSION,
              "parameters": {"min_edges": min_edges, "min_span": min_span, "tie_tolerance": tie_tolerance,
                             "near_linear_threshold": near_linear_threshold, "boundary_tolerance": BOUNDARY_TOL},
              "curve_indices": indices, "cut_node_indices": np.full(4, -1, dtype=np.int64),
              "risk_caps": np.full(5, np.nan), "representative_node_indices": np.full(5, -1, dtype=np.int64),
              "status": 0, "fallback_reason": None, "objective": None, "optimal_objective": None,
              "not_calibrated": np.zeros(5, dtype=np.int64), "curve_status": curve_status,
              "geometry_status": geometry_status}
    if algorithm_id == "manual_volatility_bands_v1":
        if manual_caps is None:
            raise ValueError("RISK_SCALE_MANUAL_CAPS_REQUIRED")
        if curve_status not in (0, 2):
            indices = np.empty(0, dtype=np.int64)
            result["curve_indices"] = indices
        risks = metrics[indices, 0]  # Small node metadata only, never weights or a historical panel.
        cuts, caps, uncalibrated = manual_volatility_bands_kernel(risks, manual_caps, BOUNDARY_TOL)
        cumulative = prefix[0] if geometry_status == 0 else np.zeros(indices.size)
        result["not_calibrated"] = uncalibrated
    else:
        if curve_status == 1:
            result["status"] = 4
            return result
        if curve_status != 0 or geometry_status != 0:
            result["status"] = 2 if curve_status == 2 or geometry_status == 2 else 1
            return result
        cuts, status, chosen, optimum, fallback = segment_frontier_kernel(
            x, y, angles, prefix, ALGORITHMS[algorithm_id], min_edges, min_span, tie_tolerance, near_linear_threshold)
        result["status"] = status
        result["objective"], result["optimal_objective"] = float(chosen), float(optimum)
        result["fallback_reason"] = "near_linear_frontier" if fallback else None
        if status:
            return result
        caps = np.empty(5)
        caps[:4], caps[4] = metrics[indices[cuts], 0], metrics[indices[-1], 0]
        validate_caps_kernel(caps)
        risks, cumulative = metrics[indices, 0], prefix[0]
    representatives = representative_nodes_kernel(risks, cumulative, caps, BOUNDARY_TOL)
    result["risk_caps"] = caps
    for j, cut in enumerate(cuts):
        if cut >= 0:
            result["cut_node_indices"][j] = indices[cut]
    for j, representative in enumerate(representatives):
        if representative >= 0:
            result["representative_node_indices"][j] = indices[representative]
    return result
