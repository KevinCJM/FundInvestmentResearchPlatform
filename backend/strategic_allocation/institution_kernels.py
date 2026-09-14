"""Small independent snapshot diagnostics, fixed ABI, readonly strided inputs."""
import os
import numpy as np
from numba import float64, njit, types

V = types.Array(float64, 1, 'A', readonly=True)
B = types.Array(types.boolean, 1, 'A', readonly=True)
_WARMED_PID = None
VERSION = 'institution-snapshot/1.0.0'


@njit((V,), cache=True, nogil=True)
def balance_sheet_kernel(values):
    """Assets and confirmed net assets; commitments are disclosed separately."""
    if values.size != 4:
        raise ValueError('INSTITUTION_BALANCE_AXIS')
    for value in values:
        if np.isinf(value) or (not np.isnan(value) and value < 0):
            raise ValueError('INSTITUTION_BALANCE_VALUE')
    result = np.full(2, np.nan)
    if not np.isnan(values[0]) and not np.isnan(values[1]):
        result[0] = values[0] + values[1]
        if not np.isnan(values[2]):
            result[1] = result[0] - values[2]
    return result


@njit((V, B), cache=True, nogil=True)
def cash_weight_kernel(weights, eligible):
    if weights.size != eligible.size:
        raise ValueError('INSTITUTION_CASH_AXIS')
    total = 0.0
    for i in range(weights.size):
        if not np.isfinite(weights[i]) or weights[i] < -1e-10:
            raise ValueError('INSTITUTION_CASH_WEIGHT')
        if eligible[i]:
            total += weights[i]
    return total


KERNELS = (balance_sheet_kernel, cash_weight_kernel)
for kernel in KERNELS:
    kernel.disable_compile()


def execution_audit():
    complete = _WARMED_PID == os.getpid() and all(len(k.signatures) == 1 and k.nopython_signatures
        and not any(o.objectmode for o in k.overloads.values()) for k in KERNELS)
    return {'engine': VERSION, 'backend': 'numba_njit_fixed_signature', 'kernel_version': VERSION,
            'complete': bool(complete), 'fully_warmed': bool(complete), 'nopython': bool(complete),
            'object_mode': 0, 'python_fallback': 0, 'request_time_compilation': 0,
            'kernel_signatures': {k.__name__: [str(s) for s in k.signatures] for k in KERNELS}}


def require_ready():
    if not execution_audit()['complete']:
        raise RuntimeError('机构诊断内核未完成启动预热。')


def warm():
    global _WARMED_PID
    _WARMED_PID = None
    raw = np.zeros(8)
    raw.flags.writeable = False
    balance_sheet_kernel(raw[::2])
    mask = np.ones(8, dtype=np.bool_)
    mask.flags.writeable = False
    cash_weight_kernel(raw[::2], mask[::2])
    _WARMED_PID = os.getpid()
    require_ready()
    return execution_audit()
