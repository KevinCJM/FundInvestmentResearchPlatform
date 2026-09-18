"""Reusable reference evidence steps: adjacency, held-weight recursion and moments.

The recursion is coupled through previous holdings; source IO/alignment remains
outside. Missing held observations terminate the path, never renormalize survivors.
"""
import os
import hashlib
import inspect
import numpy as np
from numba import njit, float64, int64, types
from backend.cal_indicators.typed_numba_kernels import covariance_2d

V = types.Array(float64, 1, 'A', readonly=True)
M = types.Array(float64, 2, 'A', readonly=True)
I = types.Array(int64, 1, 'A', readonly=True)
VERSION = 'reference-evidence/1.0.0'
_WARMED_PID = None
_WARMED_FINGERPRINT = None


@njit((M,), cache=True, nogil=True)
def adjacent_returns(levels):
    rows, columns = levels.shape
    result = np.empty((max(0, rows - 1), columns), dtype=np.float64)
    for t in range(1, rows):
        for j in range(columns):
            previous, current = levels[t - 1, j], levels[t, j]
            result[t - 1, j] = (current / previous - 1.0
                if np.isfinite(previous) and np.isfinite(current) and previous > 0 and current > 0 else np.nan)
    return result


@njit((M, V, I), cache=True, nogil=True)
def proxy_returns(returns, target, reset_before):
    rows, columns = returns.shape
    if target.size != columns or reset_before.size != rows:
        raise ValueError('PROXY_SHAPE')
    result = np.full(rows, np.nan)
    held = target.copy()
    for t in range(rows):
        if reset_before[t]:
            held[:] = target
        value = 0.0
        for j in range(columns):
            if held[j] == 0:
                continue
            r = returns[t, j]
            if not np.isfinite(r) or r <= -1:
                return result
            value += held[j] * r
        if not np.isfinite(value) or value <= -1:
            return result
        result[t] = value
        for j in range(columns):
            if held[j] != 0:
                held[j] *= (1.0 + returns[t, j]) / (1.0 + value)
    return result


@njit((M, float64, int64), cache=True, nogil=True)
def annual_moments(returns, shrinkage, periods):
    """Shared sample covariance primitive; annual arithmetic, explicit shrinkage.

    Unlike the older risk-reference facade, zero variance is valid. Its undefined
    correlations are reported as null by the boundary, never used to build risk.
    """
    rows, columns = returns.shape
    if rows < 20 or columns < 1 or columns > 30 or periods != 252 or not 0 <= shrinkage <= 1:
        raise ValueError('REFERENCE_SAMPLE_INVALID')
    mean = np.zeros(columns)
    for t in range(rows):
        for j in range(columns):
            if not np.isfinite(returns[t, j]) or returns[t, j] <= -1:
                raise ValueError('REFERENCE_SAMPLE_INVALID')
            mean[j] += returns[t, j]
    cov = covariance_2d(returns)
    volatility = np.empty(columns)
    correlation = np.full((columns, columns), np.nan)
    for i in range(columns):
        mean[i] *= periods / rows
        for j in range(columns):
            cov[i, j] *= periods * (1.0 if i == j else 1.0 - shrinkage)
        volatility[i] = np.sqrt(cov[i, i])
    for i in range(columns):
        for j in range(columns):
            if volatility[i] > 0 and volatility[j] > 0:
                correlation[i, j] = cov[i, j] / (volatility[i] * volatility[j])
    return mean, cov, volatility, correlation


@njit((V, V), cache=True, nogil=True)
def boundary_difference(left, right):
    if left.size != right.size:
        raise ValueError('BOUNDARY_SHAPE')
    return right - left


KERNELS = (adjacent_returns, proxy_returns, annual_moments, boundary_difference)
for _kernel in KERNELS:
    _kernel.disable_compile()


def audit():
    fingerprint = hashlib.sha256(repr([(inspect.getsource(k.py_func), [str(x) for x in k.signatures]) for k in KERNELS]).encode()).hexdigest()
    return {'backend': 'numba_njit_fixed_signature', 'kernel_version': VERSION,
            'complete': _WARMED_PID == os.getpid() and fingerprint == _WARMED_FINGERPRINT and all(k.nopython_signatures and not k._can_compile for k in KERNELS),
            'warmed_pid': _WARMED_PID, 'fingerprint': fingerprint, 'python_fallback': 0, 'request_time_compilation': 0}


def require_ready():
    if not audit()['complete']:
        raise RuntimeError('REFERENCE_COMPUTE_NOT_READY')


def warm():
    global _WARMED_PID, _WARMED_FINGERPRINT
    x = np.array([[1.0, 1.0], [1.01, 1.02], [1.02, 1.01]], dtype=np.float64)
    r = adjacent_returns(x)
    proxy_returns(r, np.array([0.5, 0.5]), np.zeros(2, dtype=np.int64))
    annual_moments(np.tile(r, (10, 1)), 0.1, np.int64(252))
    boundary_difference(np.zeros(5), np.ones(5))
    _WARMED_FINGERPRINT = audit()['fingerprint']
    _WARMED_PID = os.getpid()
    return audit()
