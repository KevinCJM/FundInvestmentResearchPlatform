from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from research_series import numba_kernels as kernels  # noqa: E402


def test_research_series_kernels_are_fixed_signature_and_fail_closed() -> None:
    audit = kernels.warm_research_series_numba_kernels()

    assert audit["backend"] == "numba_njit_fixed_signature"
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True
    assert audit["object_mode"] == 0
    assert audit["python_fallback"] == 0
    assert audit["request_time_compilation"] == 0
    assert audit["fully_warmed"] is True
    assert all(audit["kernel_signatures"].values())
    assert all(dispatcher._can_compile is False for dispatcher in kernels._DISPATCHERS)


def test_index_profile_preserves_missing_and_does_not_bridge_returns() -> None:
    values = np.ascontiguousarray(
        np.array([100.0, 110.0, np.nan, 121.0, 133.1], dtype=np.float64)
    )

    normalized, returns, cumulative, drawdown, rolling_volatility = (
        kernels.index_profile_kernel(values, np.int64(2), np.float64(252.0))
    )

    assert np.isnan(normalized[2])
    assert np.isclose(returns[1], np.float64(0.1))
    assert np.isnan(returns[2])
    assert np.isnan(returns[3])
    assert np.isclose(returns[4], np.float64(0.1))
    assert cumulative[0] == 0.0
    assert np.isclose(cumulative[-1], np.float64(0.331))
    assert drawdown[2] != drawdown[2]
    assert np.isnan(rolling_volatility).all()


def test_macro_profile_distribution_release_lag_and_sampling_use_full_input() -> None:
    values = np.ascontiguousarray(
        np.array([100.0, 102.0, 104.0, np.nan, 108.0], dtype=np.float64)
    )
    yoy, mom, quantile = kernels.macro_profile_kernel(values, np.int64(2))
    distribution = kernels.distribution_summary_kernel(values)
    observation_days = np.ascontiguousarray(np.array([1, 2, 3, 4, 5], dtype=np.int64))
    available_days = np.ascontiguousarray(
        np.array([3, 4, np.iinfo(np.int64).min, 8, 10], dtype=np.int64)
    )
    release_lag = kernels.release_lag_days_kernel(observation_days, available_days)
    sampled = kernels.sample_indices_kernel(np.int64(values.size), np.int64(3))

    assert np.isclose(yoy[2], np.float64(0.04))
    assert np.isclose(mom[1], np.float64(0.02))
    assert np.isnan(yoy[3]) and np.isnan(mom[3]) and np.isnan(quantile[3])
    assert distribution[0] == 4
    assert distribution[1] == 1
    assert np.isclose(distribution[2], 0.2)
    assert release_lag.tolist()[:2] == [2.0, 2.0]
    assert np.isnan(release_lag[2])
    assert sampled.tolist() == [0, 2, 4]


def test_large_profile_request_adds_no_numba_signature() -> None:
    kernels.warm_research_series_numba_kernels()
    before = kernels.research_series_numba_execution_audit()["kernel_signatures"]
    values = np.ascontiguousarray(np.linspace(80.0, 160.0, 50_000), dtype=np.float64)

    transformed = kernels.index_profile_kernel(
        values,
        np.int64(60),
        np.float64(252.0),
    )
    kernels.macro_profile_kernel(values, np.int64(12))
    kernels.distribution_summary_kernel(transformed[1])
    kernels.sample_indices_kernel(np.int64(values.size), np.int64(500))
    kernels.finite_mask_kernel(transformed[0])
    kernels.complement_rate_kernel(
        np.ascontiguousarray(np.array([0.0, 0.5, 1.0], dtype=np.float64))
    )
    dates = np.ascontiguousarray(np.arange(values.size, dtype=np.int64))
    common = kernels.strict_intersection_dates_kernel(dates, dates)
    aligned = kernels.align_values_kernel(common, dates, values)
    matrix = np.ascontiguousarray(np.column_stack((aligned, aligned)))
    kernels.standardize_matrix_kernel(matrix)
    kernels.pearson_matrix_kernel(matrix)
    kernels.complete_case_indices_kernel(matrix)
    kernels.pair_valid_indices_kernel(matrix, np.int64(0), np.int64(1))
    kernels.infinite_count_kernel(values)

    assert before == kernels.research_series_numba_execution_audit()["kernel_signatures"]


def test_compare_kernels_use_strict_dates_pairwise_finite_and_complete_cases() -> None:
    left_dates = np.ascontiguousarray(np.array([1, 2, 3, 5], dtype=np.int64))
    right_dates = np.ascontiguousarray(np.array([2, 3, 4, 5], dtype=np.int64))
    common = kernels.strict_intersection_dates_kernel(left_dates, right_dates)
    left = kernels.align_values_kernel(
        common,
        left_dates,
        np.ascontiguousarray(np.array([10.0, 20.0, np.nan, 50.0])),
    )
    right = kernels.align_values_kernel(
        common,
        right_dates,
        np.ascontiguousarray(np.array([40.0, 60.0, 80.0, 100.0])),
    )
    matrix = np.ascontiguousarray(np.column_stack((left, right)), dtype=np.float64)

    standardized = kernels.standardize_matrix_kernel(matrix)
    correlations, counts = kernels.pearson_matrix_kernel(matrix)
    complete = kernels.complete_case_indices_kernel(matrix)
    pair = kernels.pair_valid_indices_kernel(matrix, np.int64(0), np.int64(1))

    assert common.tolist() == [2, 3, 5]
    assert left[0] == 20.0
    assert np.isnan(left[1])
    assert right.tolist() == [40.0, 60.0, 100.0]
    assert standardized.shape == (3, 2)
    assert counts.tolist() == [[2, 2], [2, 3]]
    assert np.isclose(correlations[0, 1], 1.0)
    assert complete.tolist() == [0, 2]
    assert pair.tolist() == [0, 2]
