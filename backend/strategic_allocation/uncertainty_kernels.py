"""Joint uncertainty radii; Gaussian, NIW t and Hotelling calibrations stay distinct."""
from __future__ import annotations

import math
import os
import hashlib
import inspect
import numpy as np
from numba import float64, int64, njit
from .cma_statistical_kernels import regularized_beta, beta_fraction

VERSION = "mean-uncertainty-radius/1.0.0"
_WARMED_PID = None
_WARMED_FINGERPRINT = None


@njit((float64, int64), cache=True, nogil=True)
def chi_square_cdf(value, degrees):
    if not np.isfinite(value) or value < 0 or degrees < 1 or degrees > 30:
        raise ValueError("SAA_UNCERTAINTY_QUANTILE_INPUT")
    if value == 0:
        return 0.0
    x = value / 2.0
    # Integer/half-integer gamma recurrence, bounded by the 30-asset contract.
    shape = .5 if degrees % 2 else 1.0
    result = math.erf(math.sqrt(x)) if degrees % 2 else -math.expm1(-x)
    while shape + .5 < degrees / 2.0:
        result -= math.exp(shape * math.log(x) - x - math.lgamma(shape + 1.0))
        shape += 1.0
    return min(1.0, max(0.0, result))


@njit((float64, int64), cache=True, nogil=True)
def chi_square_quantile(probability, degrees):
    if not np.isfinite(probability) or not .5 <= probability <= .999 or degrees < 1 or degrees > 30:
        raise ValueError("SAA_UNCERTAINTY_QUANTILE_INPUT")
    low, high = 0.0, float(degrees)
    for _ in range(64):
        if chi_square_cdf(high, degrees) >= probability:
            break
        high *= 2.0
    else:
        raise ValueError("SAA_UNCERTAINTY_QUANTILE_BRACKET")
    for _ in range(80):
        mid = (low + high) / 2.0
        if chi_square_cdf(mid, degrees) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


@njit((float64, int64, float64), cache=True, nogil=True)
def f_quantile(probability, first_degrees, second_degrees):
    if (not np.isfinite(probability) or not .5 <= probability <= .999 or first_degrees < 1
            or first_degrees > 30 or not np.isfinite(second_degrees) or not 0 < second_degrees <= 200000):
        raise ValueError("SAA_UNCERTAINTY_QUANTILE_INPUT")
    low, high = 0.0, 1.0
    for _ in range(90):
        mid = (low + high) / 2.0
        cdf = regularized_beta(mid, first_degrees / 2.0, second_degrees / 2.0)
        if cdf < probability:
            low = mid
        else:
            high = mid
    z = (low + high) / 2.0
    if z >= 1.0:
        raise ValueError("SAA_UNCERTAINTY_QUANTILE_BRACKET")
    return second_degrees * z / (first_degrees * (1.0 - z))


@njit((float64, int64, int64, float64), cache=True, nogil=True)
def uncertainty_radius_kernel(probability, dimension, method, information):
    """method: 0 Gaussian; 1 NIW covariance (not t scale); 2 Hotelling sample mean."""
    if dimension < 0 or dimension > 30 or method < 0 or method > 2 or not np.isfinite(probability) or not .5 <= probability <= .999:
        raise ValueError("SAA_UNCERTAINTY_RADIUS_INPUT")
    if dimension == 0:
        return 0.0
    if method == 0:
        return math.sqrt(chi_square_quantile(probability, dimension))
    if not np.isfinite(information):
        raise ValueError("SAA_UNCERTAINTY_RADIUS_INPUT")
    if method == 1:
        if information <= 2:
            raise ValueError("SAA_UNCERTAINTY_RADIUS_INPUT")
        squared = dimension * (information - 2.0) / information * f_quantile(probability, dimension, information)
    else:
        if information <= dimension:
            raise ValueError("SAA_UNCERTAINTY_SAMPLE_TOO_SHORT")
        squared = dimension * (information - 1.0) / (information - dimension) * f_quantile(probability, dimension, information - dimension)
    return math.sqrt(squared)


KERNELS = (chi_square_cdf, chi_square_quantile, f_quantile, uncertainty_radius_kernel)
for kernel in KERNELS:
    kernel.disable_compile()


def execution_audit():
    dispatchers = (*KERNELS, regularized_beta, beta_fraction)
    fingerprint = hashlib.sha256(repr([(inspect.getsource(k.py_func), [str(s) for s in k.signatures]) for k in dispatchers]).encode()).hexdigest()
    compiled = all(len(k.signatures) == len(k.nopython_signatures) == 1 and not k._can_compile
                   and not any(v.objectmode for v in k.overloads.values()) for k in dispatchers)
    ready = compiled and _WARMED_PID == os.getpid() and _WARMED_FINGERPRINT == fingerprint
    return {"backend":"numba_njit_fixed_signature", "kernel_version":VERSION, "complete":bool(ready),
            "fully_warmed":bool(ready), "fingerprint":fingerprint, "nopython":bool(compiled), "object_mode":0,
            "python_fallback":0, "request_time_compilation":0,
            "kernel_signatures":{k.__name__:[str(s) for s in k.signatures] for k in dispatchers}}


def require_ready():
    if not execution_audit()["complete"]:
        raise RuntimeError("SAA_UNCERTAINTY_NOT_READY: 均值不确定性内核未完成本进程预热。")


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    _WARMED_PID = None
    _WARMED_FINGERPRINT = None
    for method, information in ((0,0.), (1,30.), (2,40.)):
        uncertainty_radius_kernel(.95, 2, method, information)
    chi_square_quantile(.68, 3)
    _WARMED_PID = os.getpid()
    _WARMED_FINGERPRINT = execution_audit()["fingerprint"]
    return execution_audit()
