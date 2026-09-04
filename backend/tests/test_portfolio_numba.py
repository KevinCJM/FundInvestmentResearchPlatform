import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from custom_indicators.portfolio_numba import (
    equal_weights_kernel,
    portfolio_diagnosis_kernel,
    portfolio_drift_backtest_kernel,
    portfolio_numba_execution_audit,
    portfolio_summary_kernel,
    strict_returns_kernel,
    turnover_path_kernel,
    validate_unit_weights_kernel,
    warm_portfolio_numba_kernels,
)


def test_portfolio_research_kernels_are_fixed_signature_nopython():
    audit = warm_portfolio_numba_kernels()

    assert audit["backend"] == "numba_njit_fixed_signature"
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["object_mode"] == 0
    assert audit["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0
    assert audit["fully_warmed"] is True
    assert all(audit["kernel_signatures"].values())


def test_portfolio_drift_and_attribution_accounting_identity():
    returns = np.ascontiguousarray(
        np.array([[0.01, -0.01], [0.02, 0.00], [-0.01, 0.02]], dtype=np.float64)
    )
    schedule = np.zeros_like(returns)
    schedule[0] = equal_weights_kernel(2)
    mask = np.ascontiguousarray(np.array([1, 0, 0], dtype=np.uint8))

    portfolio_returns, weights, contributions, status = portfolio_drift_backtest_kernel(
        returns, np.ascontiguousarray(schedule), mask
    )
    nav, drawdown, metrics = portfolio_summary_kernel(portfolio_returns, 252.0, 0.0)
    diagnosis = portfolio_diagnosis_kernel(returns, weights, contributions, 252.0)

    assert status == 0
    np.testing.assert_allclose(contributions.sum(axis=1), portfolio_returns, atol=1e-15)
    assert nav.shape == drawdown.shape == portfolio_returns.shape
    assert metrics.shape == (7,)
    assert diagnosis[0].shape == (2, 2)


def test_manual_weight_validation_is_njit_and_never_normalizes() -> None:
    accepted, accepted_status = validate_unit_weights_kernel(
        np.ascontiguousarray(np.array([0.7, 0.3], dtype=np.float64)), 1e-8
    )
    rejected, rejected_status = validate_unit_weights_kernel(
        np.ascontiguousarray(np.array([60.0, 40.0], dtype=np.float64)), 1e-8
    )

    np.testing.assert_array_equal(accepted, np.array([0.7, 0.3]))
    np.testing.assert_array_equal(rejected, np.array([60.0, 40.0]))
    assert accepted_status == 0
    assert rejected_status == 3


def test_large_portfolio_path_adds_no_request_time_signature():
    warm_portfolio_numba_kernels()
    signatures_before = portfolio_numba_execution_audit()["kernel_signatures"]
    rows = 5_001
    levels = np.linspace(100.0, 150.0, rows)
    nav = np.ascontiguousarray(
        np.column_stack((levels, levels * 1.01, levels[::-1] + 100.0)),
        dtype=np.float64,
    )
    returns = strict_returns_kernel(nav)
    schedule = np.zeros_like(returns)
    schedule[0] = equal_weights_kernel(3)
    mask = np.zeros(returns.shape[0], dtype=np.uint8)
    mask[0] = 1

    portfolio_returns, weights, contributions, status = portfolio_drift_backtest_kernel(
        returns, np.ascontiguousarray(schedule), np.ascontiguousarray(mask)
    )
    portfolio_summary_kernel(portfolio_returns, 252.0, 0.0)
    portfolio_diagnosis_kernel(returns, weights, contributions, 252.0)
    turnover_path_kernel(np.ascontiguousarray(weights[::250]))

    assert status == 0
    assert portfolio_returns.size == 5_000
    assert signatures_before == portfolio_numba_execution_audit()["kernel_signatures"]
