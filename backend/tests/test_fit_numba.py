import sys
from pathlib import Path

import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fit_numba import (
    class_consistency_kernel,
    class_nav_corr_metrics_kernel,
    finite_matrix_mask_kernel,
    fit_numba_execution_audit,
    returns_from_nav_kernel,
    rolling_correlation_kernel,
    warm_fit_numba_kernels,
)
from fit import compute_nav_performance_payload


def test_fit_kernels_use_only_fixed_signature_nopython_lanes():
    audit = warm_fit_numba_kernels()

    assert audit["backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["python_fallback"] == 0
    assert all(audit["kernel_signatures"].values())


def test_nav_returns_do_not_fill_or_bridge_missing_value():
    nav = np.ascontiguousarray(
        np.array([[100.0], [np.nan], [110.0]], dtype=np.float64)
    )

    result = returns_from_nav_kernel(nav)

    assert np.isnan(result).all()


def test_fit_statistics_match_expected_shapes_and_bounds():
    returns = np.ascontiguousarray(
        np.array(
            [
                [0.01, -0.01],
                [0.02, 0.00],
                [-0.01, 0.02],
                [0.00, 0.01],
            ],
            dtype=np.float64,
        )
    )

    nav, corr, metrics = class_nav_corr_metrics_kernel(returns, 252.0)
    rolling, rolling_metrics = rolling_correlation_kernel(returns, 0, 3)
    consistency = class_consistency_kernel(returns, 252.0)

    assert nav.shape == returns.shape
    assert nav[0].tolist() == [1.0, 1.0]
    assert corr.shape == (2, 2)
    assert np.all(corr <= 1.0) and np.all(corr >= -1.0)
    assert metrics.shape == (2, 7)
    assert rolling.shape == returns.shape
    assert rolling_metrics.shape == (2, 7)
    assert consistency.shape == (3,)


def test_large_fit_request_does_not_compile_new_signatures():
    warm_fit_numba_kernels()
    signatures_before = fit_numba_execution_audit()["kernel_signatures"]
    rows = 5_001
    base = np.linspace(100.0, 180.0, rows)
    nav = np.ascontiguousarray(
        np.column_stack((base, base * 1.02, base[::-1] + 100.0)),
        dtype=np.float64,
    )

    returns = returns_from_nav_kernel(nav)
    class_nav_corr_metrics_kernel(returns, 252.0)
    rolling_correlation_kernel(returns, 0, 60)
    class_consistency_kernel(returns, 252.0)
    finite_matrix_mask_kernel(nav)

    assert signatures_before == fit_numba_execution_audit()["kernel_signatures"]


def test_fit_performance_payload_is_njit_computed_and_audited():
    nav = pd.DataFrame(
        {
            "股票": [1.0, 1.1, 1.2],
            "债券": [1.0, 0.99, 1.01],
        },
        index=pd.to_datetime(["2024-01-02", "2024-12-31", "2025-01-02"]),
    )

    payload = compute_nav_performance_payload(nav)

    assert np.isclose(payload["cumulative_returns"]["股票"], 0.2)
    assert payload["annual_metrics"]["years"] == [2024, 2025]
    assert payload["execution"]["fit_analytics"]["python_fallback"] == 0
    assert payload["execution"]["performance_metrics"]["backend"] == "numba_njit_fixed_signature"
