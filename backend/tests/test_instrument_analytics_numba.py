from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.compute_policy import validate_execution_audit  # noqa: E402
from backend import benchmark_indicator_engine  # noqa: E402
from backend.instrument_analytics_numba import (  # noqa: E402
    adjusted_nav_anomaly_mask_kernel,
    candle_metrics_kernel,
    contiguous_blocks_kernel,
    data_quality_masks_kernel,
    encoded_category_counts_kernel,
    encoded_unique_count_kernel,
    int_less_than_count_kernel,
    int_range_count_kernel,
    instrument_analytics_numba_execution_audit,
    nav_metrics_kernel,
    period_window_quality_kernel,
    positive_finite_sum_kernel,
    ranking_order_kernel,
    simple_log_returns_kernel,
    warm_instrument_analytics_numba_kernels,
)
from backend.services import instrument_analytics  # noqa: E402


def _signature_snapshot() -> dict[str, tuple[str, ...]]:
    audit = instrument_analytics_numba_execution_audit()
    return {
        name: tuple(signatures)
        for name, signatures in audit["kernel_signatures"].items()
    }


def test_instrument_analytics_kernels_are_fixed_signature_nopython() -> None:
    audit = warm_instrument_analytics_numba_kernels()
    validated = validate_execution_audit(audit)

    assert validated["execution_backend"] == "numba_njit_fixed_signature"
    assert validated["nopython"] is True
    assert validated["object_mode"] == 0
    assert validated["python_fallback"] == 0
    assert all(
        len(signatures) == 1
        for signatures in validated["kernel_signatures"].values()
    )


def test_series_returns_use_locked_fixed_signature_kernel() -> None:
    warm_instrument_analytics_numba_kernels()
    signatures_before = tuple(simple_log_returns_kernel.signatures)
    values = np.ascontiguousarray(np.linspace(1.0, 2.0, 5_001), dtype=np.float64)

    simple, logarithmic, status = simple_log_returns_kernel(values)

    assert status == 0
    assert simple == pytest.approx(values[1:] / values[:-1] - 1.0)
    assert logarithmic == pytest.approx(np.log(values[1:] / values[:-1]))
    assert tuple(simple_log_returns_kernel.signatures) == signatures_before
    with pytest.raises(TypeError, match="No matching definition"):
        simple_log_returns_kernel(values.astype(np.float32))
    assert tuple(simple_log_returns_kernel.signatures) == signatures_before


def test_data_quality_classification_uses_one_locked_njit_signature() -> None:
    warm_instrument_analytics_numba_kernels()
    signatures_before = tuple(data_quality_masks_kernel.signatures)
    nat = np.iinfo(np.int64).min
    missing, invalid, future, anomaly, stale = data_quality_masks_kernel(
        np.ascontiguousarray([np.nan, np.inf, 0.0, 1.0], dtype=np.float64),
        np.ascontiguousarray([nat, 11, 9, 12], dtype=np.int64),
        np.ascontiguousarray([0.0, 1.0, np.nan, 2.0], dtype=np.float64),
        np.ascontiguousarray([1, 0, 1, 1], dtype=np.uint8),
        np.ascontiguousarray([10.0, 20.0, 8.0, 9.0], dtype=np.float64),
        np.int64(10),
        np.float64(7.0),
    )
    assert missing.tolist() == [1, 0, 0, 0]
    assert invalid.tolist() == [0, 1, 1, 0]
    assert future.tolist() == [0, 1, 0, 1]
    assert anomaly.tolist() == [0, 1, 0, 1]
    assert stale.tolist() == [1, 0, 1, 1]
    with pytest.raises(TypeError, match="No matching definition"):
        data_quality_masks_kernel(
            np.ascontiguousarray([1.0], dtype=np.float32),
            np.ascontiguousarray([1], dtype=np.int64),
            np.ascontiguousarray([0.0], dtype=np.float64),
            np.ascontiguousarray([1], dtype=np.uint8),
            np.ascontiguousarray([0.0], dtype=np.float64),
            np.int64(10),
            np.float64(7.0),
        )
    assert tuple(data_quality_masks_kernel.signatures) == signatures_before


def test_large_instrument_path_uses_actual_call_chain_without_new_signatures() -> None:
    warm_instrument_analytics_numba_kernels()
    before = _signature_snapshot()
    dates = pd.bdate_range("2000-01-03", periods=5_000)
    trend = np.linspace(1.0, 2.0, dates.size)
    drawdown = np.ones(dates.size)
    drawdown[3_500:3_700] = np.linspace(1.0, 0.75, 200)
    drawdown[3_700:] = np.linspace(0.75, 1.05, dates.size - 3_700)
    adjusted = trend * drawdown
    frame = pd.DataFrame(
        {
            "date": dates,
            "adj_nav": adjusted,
            "unit_nav": adjusted,
            "accum_nav": adjusted,
        }
    )

    record, unit_tail = instrument_analytics._compute_nav_record(
        "etf",
        "510300.SH",
        frame,
        "fixture",
        dates,
    )

    assert record["observation_count"] == 5_000
    assert record["annual_volatility_1y"] is not None
    assert record["max_drawdown_3y"] is not None
    assert len(unit_tail) == 31
    assert before == _signature_snapshot()


def test_nav_candle_ranking_and_quality_numeric_parity() -> None:
    one_year = np.ascontiguousarray(
        np.array([1.0, 1.01, 0.99, 1.03, 1.02] * 20, dtype=np.float64)
    )
    three_year = np.ascontiguousarray(
        np.linspace(1.0, 1.4, 220, dtype=np.float64)
    )
    three_year[100:150] *= np.linspace(1.0, 0.7, 50)
    metrics = nav_metrics_kernel(one_year, one_year, one_year, three_year, 365)
    reference_returns = one_year[1:] / one_year[:-1] - 1.0

    assert metrics[0] == pytest.approx(one_year[-1] / one_year[0] - 1.0)
    assert metrics[4] == pytest.approx(reference_returns.std(ddof=1) * np.sqrt(252.0))
    assert metrics[6] == pytest.approx(
        reference_returns.mean() / reference_returns.std(ddof=1) * np.sqrt(252.0)
    )

    close = np.ascontiguousarray(np.linspace(1.0, 1.2, 25))
    amount = np.ascontiguousarray(np.arange(1.0, 26.0))
    unit_nav = np.full(25, np.nan, dtype=np.float64)
    unit_nav[-1] = close[-1] / 1.01
    candle, common_index = candle_metrics_kernel(close, amount, amount * 10.0, unit_nav)
    assert common_index == 24
    assert candle[2] == pytest.approx(0.01)
    assert candle[3] == pytest.approx(np.arange(6.0, 26.0).mean())

    values = np.ascontiguousarray(np.array([0.2, np.nan, 0.5, 0.5, -0.1]))
    code_rank = np.ascontiguousarray(np.array([4, 0, 2, 1, 3], dtype=np.int64))
    order = ranking_order_kernel(values, code_rank, np.uint8(0))
    assert order.tolist() == [3, 2, 0, 4]

    days = np.ascontiguousarray(np.arange(100, 110, dtype=np.int64))
    complete = period_window_quality_kernel(
        days,
        100,
        109,
        days,
        np.empty(0, dtype=np.int64),
        10,
        0.9,
        5,
    )
    assert complete[0] == 1
    assert complete[5] == pytest.approx(1.0)


def test_anomaly_kernel_handles_missing_rows_without_python_fallback() -> None:
    adjusted = np.ascontiguousarray(np.array([1.0, np.nan, 100.0], dtype=np.float64))
    reference = np.ascontiguousarray(np.array([1.0, np.nan, 1.01], dtype=np.float64))

    result = adjusted_nav_anomaly_mask_kernel(
        adjusted,
        reference,
        reference,
        0.20,
        0.10,
        1.0,
    )

    assert result.tolist() == [0, 0, 1]


def test_encoded_category_counts_and_unique_count_are_njit() -> None:
    codes = np.ascontiguousarray(
        np.array([2, 0, 2, -1, 1, 0, 2], dtype=np.int64)
    )

    counts = encoded_category_counts_kernel(codes, 3)

    assert counts.tolist() == [2, 1, 3]
    assert encoded_unique_count_kernel(codes) == 3


def test_generic_count_sum_and_contiguous_block_kernels_keep_one_signature_at_5000_points() -> None:
    warm_instrument_analytics_numba_kernels()
    before = _signature_snapshot()
    values = np.ascontiguousarray(np.linspace(-2.0, 3.0, 5_000, dtype=np.float64))
    values[10] = np.nan
    values[20] = np.inf
    integers = np.ascontiguousarray(np.arange(5_000, dtype=np.int64))
    mask = np.zeros(5_000, dtype=np.uint8)
    mask[2:5] = 1
    mask[8:10] = 1
    mask[4_998:] = 1

    expected_sum = float(values[np.isfinite(values) & (values > 0.0)].sum())
    assert positive_finite_sum_kernel(values) == pytest.approx(expected_sum)
    assert int_range_count_kernel(integers, 25, 125) == 101
    assert int_range_count_kernel(integers, 125, 25) == 0
    assert int_less_than_count_kernel(integers, 125) == 125
    positions, split_boundaries = contiguous_blocks_kernel(mask)
    assert positions.tolist() == [2, 3, 4, 8, 9, 4_998, 4_999]
    assert split_boundaries.tolist() == [3, 5]

    assert before == _signature_snapshot()
    assert len(positive_finite_sum_kernel.nopython_signatures) == 1
    assert len(int_range_count_kernel.nopython_signatures) == 1
    assert len(int_less_than_count_kernel.nopython_signatures) == 1
    assert len(contiguous_blocks_kernel.nopython_signatures) == 1


def test_public_response_exposes_policy_validated_execution(tmp_path: Path) -> None:
    info = pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "name": "沪深300ETF",
                "status_code": "L",
                "status": "上市交易",
                "issue_amount": 100.0,
                "list_date": pd.Timestamp("2020-01-02"),
            }
        ]
    )
    info_path = tmp_path / "etf_info_df.parquet"
    info.to_parquet(info_path, index=False)

    response = instrument_analytics.build_analytics_response(
        kind="etf",
        data_dir=tmp_path,
        info_files={"etf": info_path},
    )

    execution = validate_execution_audit(response["execution"])
    assert execution["nopython"] is True
    assert execution["python_fallback"] == 0


def test_indicator_benchmark_fails_closed_on_python_fallback(monkeypatch) -> None:
    class _Spec:
        compiled_signatures = ("kernel(Array(float64, 1, C))",)

    monkeypatch.setattr(
        benchmark_indicator_engine,
        "get_numba_kernel_registry",
        lambda: {"total_return": _Spec()},
    )
    monkeypatch.setattr(
        benchmark_indicator_engine,
        "kernel_registry_status",
        lambda: {"warmed": True, "kernel_version": "test"},
    )
    compliant = {
        "execution": {
            "engine_version": "test",
            "execution_lanes": {"python_fallback": 0},
            "typed_batch_fallback": 0,
            "python_operator_calls": 0,
        }
    }

    audit = benchmark_indicator_engine.validate_benchmark_run_execution(compliant)
    assert audit["execution_backend"] == "numba_njit_fixed_signature"

    compliant["execution"]["execution_lanes"]["python_fallback"] = 1
    with pytest.raises(ValueError, match="Python 数值回退"):
        benchmark_indicator_engine.validate_benchmark_run_execution(compliant)
