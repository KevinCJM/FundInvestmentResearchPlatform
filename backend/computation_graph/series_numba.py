"""Fixed-signature availability propagation for causal series expressions."""
import numpy as np
from numba import float64, int64, njit, types


@njit(int64[::1](int64[::1]), cache=True)
def causal_available_kernel(available):
    result = np.empty(available.size, dtype=np.int64)
    latest = np.iinfo(np.int64).min
    for index in range(available.size):
        latest = max(latest, available[index])
        result[index] = latest
    return result


causal_available_kernel.disable_compile()


@njit(int64(float64[::1]), cache=True)
def valid_series_output_kernel(values):
    for index in range(values.size):
        if np.isinf(values[index]):
            return 0
    return 1


valid_series_output_kernel.disable_compile()


@njit(types.UniTuple(int64, 2)(types.Array(float64, 1, 'A', readonly=True)), cache=True, nogil=True)
def series_validity_counts_kernel(values):
    """Output-boundary finite/Inf counts; missing observations remain NaN."""
    finite, infinite = 0, 0
    for value in values:
        finite += np.isfinite(value)
        infinite += np.isinf(value)
    return finite, infinite


series_validity_counts_kernel.disable_compile()
