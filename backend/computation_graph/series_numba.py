"""Fixed-signature availability propagation for causal series expressions."""
import numpy as np
from numba import float64, int64, njit


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
