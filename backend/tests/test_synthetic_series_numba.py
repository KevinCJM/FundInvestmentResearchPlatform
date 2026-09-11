from __future__ import annotations

import numpy as np

from compute_policy import validate_execution_audit
from synthetic_series_numba import (
    synthetic_ohlcv_kernel,
    synthetic_series_execution_audit,
    warm_synthetic_series_numba_kernel,
)


def test_synthetic_series_is_fixed_signature_nopython() -> None:
    audit = validate_execution_audit(warm_synthetic_series_numba_kernel())
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["python_fallback"] == 0
    assert all(audit["kernel_signatures"].values())


def test_synthetic_series_is_deterministic_and_structurally_valid() -> None:
    first = synthetic_ohlcv_kernel(23, 100.0, 32)
    second = synthetic_ohlcv_kernel(23, 100.0, 32)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (32, 5)
    assert np.isfinite(first).all()
    assert np.all(first[:, 2] >= np.maximum(first[:, 0], first[:, 1]))
    assert np.all(first[:, 3] <= np.minimum(first[:, 0], first[:, 1]))
    assert np.all(first[:, 3] >= 0.1)


def test_synthetic_series_large_run_does_not_compile_new_signature() -> None:
    warm_synthetic_series_numba_kernel()
    signatures_before = synthetic_series_execution_audit()["kernel_signatures"]
    output = synthetic_ohlcv_kernel(41, np.nan, 5000)
    assert output.shape == (5000, 5)
    assert signatures_before == synthetic_series_execution_audit()["kernel_signatures"]
