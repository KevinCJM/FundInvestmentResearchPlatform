"""Compatibility facade over the typed-numba-3 numeric backend.

The helpers in this module remain for offline parity tests. Production typed
plans execute only fixed-signature kernels from ``typed_numba_kernels``;
matrix kernels may call Numba-supported BLAS/LAPACK inside NJIT dispatchers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

import numba
import numpy as np

from cal_indicators.numba_finance_math import (
    cumulative_simple_returns,
    sequence_cumprod,
    sequence_cumsum,
    sequence_max,
    sequence_mean,
    sequence_min,
    sequence_prod,
    sequence_std,
    sequence_sum,
    sequence_variance,
)


NUMERIC_BACKEND_VERSION = "2.2.0"


def _flat_float64(values: Any) -> np.ndarray:
    """Return a flat float64 view whenever the input layout permits it.

    ``np.asarray`` is zero-copy for an existing float64 ndarray. ``ravel`` also
    preserves storage for contiguous arrays; an unavoidable copy is made only
    for incompatible dtype/layout inputs.
    """

    return np.asarray(values, dtype=np.float64).ravel(order="K")


def reduce_sum(values: Any) -> float:
    return float(sequence_sum(_flat_float64(values)))


def reduce_product(values: Any) -> float:
    return float(sequence_prod(_flat_float64(values)))


def reduce_mean(values: Any) -> float:
    return float(sequence_mean(_flat_float64(values)))


def reduce_min(values: Any) -> float:
    return float(sequence_min(_flat_float64(values)))


def reduce_max(values: Any) -> float:
    return float(sequence_max(_flat_float64(values)))


def reduce_variance(values: Any, ddof: int) -> float:
    return float(sequence_variance(_flat_float64(values), int(ddof)))


def reduce_std(values: Any, ddof: int) -> float:
    return float(sequence_std(_flat_float64(values), int(ddof)))


def scan_sum(values: Any) -> np.ndarray:
    return sequence_cumsum(_flat_float64(values))


def scan_product(values: Any) -> np.ndarray:
    return sequence_cumprod(_flat_float64(values))


def scan_return(values: Any) -> np.ndarray:
    return cumulative_simple_returns(_flat_float64(values))


_WARM_KERNELS: tuple[Callable[..., Any], ...] = (
    sequence_sum,
    sequence_prod,
    sequence_mean,
    sequence_min,
    sequence_max,
    sequence_variance,
    sequence_std,
    sequence_cumsum,
    sequence_cumprod,
    cumulative_simple_returns,
)


@lru_cache(maxsize=1)
def warm_typed_numeric_backend() -> dict[str, Any]:
    """Exercise all typed Numba kernels once and expose auditable status."""

    sample = np.asarray([0.01, -0.02, 0.03, 0.005], dtype=np.float64)
    sequence_sum(sample)
    sequence_prod(sample)
    sequence_mean(sample)
    sequence_min(sample)
    sequence_max(sample)
    sequence_variance(sample, 1)
    sequence_std(sample, 1)
    sequence_cumsum(sample)
    sequence_cumprod(sample)
    cumulative_simple_returns(sample)
    from cal_indicators.typed_numba_kernels import warm_numba_kernel_registry

    warm_numba_kernel_registry()
    return numeric_backend_status(warmed=True)


def numeric_backend_status(*, warmed: bool | None = None) -> dict[str, Any]:
    from cal_indicators.typed_numba_kernels import kernel_registry_status

    if warmed is None:
        warmed = warm_typed_numeric_backend.cache_info().currsize > 0
    signatures = {
        function.py_func.__name__: [str(signature) for signature in function.signatures]
        for function in _WARM_KERNELS
    }
    return {
        "version": NUMERIC_BACKEND_VERSION,
        "warmed": warmed,
        "numba_version": numba.__version__,
        "numpy_version": np.__version__,
        "policy": {
            "all_canonical_operators": "numba_njit_fixed_signature",
            "one_dimensional_reductions_and_scans": "numba_njit_fixed_signature",
            "elementwise_broadcasting": "numba_explicit_loop_scalar_only",
            "matrix_and_linear_algebra": "numba_wrapped_blas_lapack",
            "input_conversion": "zero_copy_when_float64_layout_compatible",
            "python_fallback": "forbidden",
        },
        "compiled_signatures": signatures,
        "typed_numba_3": kernel_registry_status(),
    }


__all__ = [
    "NUMERIC_BACKEND_VERSION",
    "numeric_backend_status",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_product",
    "reduce_std",
    "reduce_sum",
    "reduce_variance",
    "scan_product",
    "scan_return",
    "scan_sum",
    "warm_typed_numeric_backend",
]
