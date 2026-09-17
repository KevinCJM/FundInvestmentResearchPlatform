"""后复权因子与复权 OHLC 的口径、失败关闭与 ETL 落盘。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_sources.price_adjustment import (
    AdjustmentPolicyError,
    DEFAULT_FACTOR_POLICY,
    attach_adjusted_prices,
    pre_close_factor,
)
from data_sources.price_adjustment_numba import (
    price_adjustment_execution_audit,
    warm_price_adjustment_kernels,
)


def _candle(closes, *, previous=None, code="510300.SH", start="2025-01-02"):
    closes = np.asarray(closes, dtype=float)
    if previous is None:
        previous = np.r_[closes[0], closes[:-1]]
    previous = np.asarray(previous, dtype=float)
    return pd.DataFrame(
        {
            "ts_code": code,
            "date": pd.bdate_range(start, periods=len(closes)),
            "open": closes - 0.05,
            "high": closes + 0.10,
            "low": closes - 0.10,
            "close": closes,
            "pre_close": previous,
        }
    )


def test_factor_is_one_until_an_event_and_reproduces_the_real_return() -> None:
    closes = [10.0, 10.2, 9.9, 10.1]
    # 第三天除息 0.3：前收盘价被下调，价格跌幅里有 0.3 不是真实亏损。
    previous = [10.0, 10.0, 10.2 - 0.3, 9.9]
    frame, stats = attach_adjusted_prices(_candle(closes, previous=previous), None, policy="pre_close")

    factor = frame["adj_factor"].to_numpy()
    np.testing.assert_allclose(factor[:2], [1.0, 1.0])
    np.testing.assert_allclose(factor[2:], 10.2 / 9.9)
    np.testing.assert_allclose(
        frame["adj_close"].pct_change().to_numpy()[1:],
        np.asarray(closes[1:]) / np.asarray(previous[1:]) - 1.0,
        atol=1e-12,
    )
    assert stats["codes_with_adjustment_events"] == 1
    assert stats["codes_by_factor_source"] == {"pre_close": 1}


def test_a_broken_ratio_voids_the_rest_of_the_series_instead_of_skipping_it() -> None:
    frame = _candle([10.0, 10.2, 9.9, 10.1], previous=[10.0, 10.0, np.nan, 9.9])
    result, stats = attach_adjusted_prices(frame, None, policy="pre_close")

    assert result["adj_factor"].notna().tolist() == [True, True, False, False]
    assert result["adj_close"].notna().tolist() == [True, True, False, False]
    assert result["adj_factor_source"].isna().tolist() == [False, False, True, True]
    assert stats["rows_without_factor"] == 2


def test_source_policy_leaves_uncovered_codes_without_adjusted_prices() -> None:
    frame = pd.concat(
        [_candle([10.0, 10.2, 9.9]), _candle([3.0, 3.1, 3.2], code="159915.SZ")],
        ignore_index=True,
    )
    official = pd.DataFrame(
        {
            "ts_code": "510300.SH",
            "date": pd.bdate_range("2025-01-02", periods=3),
            "adj_factor": [2.0, 2.0, 2.06],
        }
    )
    result, stats = attach_adjusted_prices(frame, official, policy="source")

    covered = result[result["ts_code"] == "510300.SH"]
    missing = result[result["ts_code"] == "159915.SZ"]
    # 数据源因子归一到各标的首个交易日，两条来源才可比。
    np.testing.assert_allclose(covered["adj_factor"].to_numpy(), [1.0, 1.0, 1.03])
    assert covered["adj_factor_source"].eq("source").all()
    assert missing["adj_factor"].isna().all() and missing["adj_close"].isna().all()
    assert stats["codes_by_factor_source"] == {"source": 1, "none": 1}


def test_partial_official_coverage_is_not_interpolated() -> None:
    frame = _candle([10.0, 10.2, 9.9])
    official = pd.DataFrame(
        {"ts_code": "510300.SH", "date": pd.bdate_range("2025-01-02", periods=2), "adj_factor": [2.0, 2.0]}
    )
    result, _ = attach_adjusted_prices(frame, official, policy="source")
    assert result["adj_factor"].isna().all()

    filled, _ = attach_adjusted_prices(frame, official, policy="source_then_pre_close")
    assert filled["adj_factor_source"].eq("pre_close").all()


def test_unknown_policy_is_rejected() -> None:
    with pytest.raises(AdjustmentPolicyError):
        attach_adjusted_prices(_candle([10.0, 10.1]), None, policy="whatever")


def test_factors_never_cross_product_boundaries() -> None:
    frame = pd.concat(
        [
            _candle([10.0, 10.2], previous=[10.0, 9.0]),
            _candle([3.0, 3.1], code="159915.SZ"),
        ],
        ignore_index=True,
    )
    factor = pre_close_factor(frame.sort_values(["ts_code", "date"]).reset_index(drop=True))
    assert factor.tolist()[:2] == [1.0, 1.0]  # 159915.SZ 无事件
    np.testing.assert_allclose(factor.tolist()[2:], [1.0, 10.0 / 9.0])


def test_the_factor_chain_runs_on_precompiled_njit_kernels() -> None:
    audit = warm_price_adjustment_kernels()
    assert audit == price_adjustment_execution_audit()
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["nopython"] is True and audit["fully_warmed"] is True
    assert audit["python_fallback"] == 0 and audit["request_time_compilation"] == 0
    # A declared signature per kernel is what keeps an ETL run out of the compiler.
    assert audit["kernel_signatures"] and all(audit["kernel_signatures"].values())


def test_an_unusable_previous_close_voids_the_rest_instead_of_dividing_by_it() -> None:
    for broken in (0.0, -1.0, np.nan):
        frame = _candle([10.0, 10.2, 9.9, 10.1], previous=[10.0, 10.0, broken, 9.9])
        result, _ = attach_adjusted_prices(frame, None, policy="pre_close")
        assert result["adj_factor"].notna().tolist() == [True, True, False, False]
        assert np.isfinite(result["adj_close"].to_numpy()[:2]).all()


def test_etl_step_writes_the_columns_back_into_the_candle_table(tmp_path: Path) -> None:
    import T01_get_data as script

    _candle([10.0, 10.2, 9.9], previous=[10.0, 10.0, 10.2]).assign(
        change=0.0, pct_chg=0.0, vol=1.0, amount=1.0, name="沪深300ETF"
    ).to_parquet(tmp_path / "etf_daily_candle_df.parquet", index=False)

    with pytest.raises(FileNotFoundError):
        script.save_price_adjustment(tmp_path, "source")

    script.save_price_adjustment(tmp_path, "pre_close")
    written = pd.read_parquet(tmp_path / "etf_daily_candle_df.parquet")
    assert {"adj_factor", "adj_factor_source", "adj_open", "adj_high", "adj_low", "adj_close"} <= set(written.columns)
    assert written["amount"].notna().all()  # 原有列不丢
    np.testing.assert_allclose(written["adj_close"].to_numpy(), [10.0, 10.2, 9.9])


def test_the_registered_default_leaves_no_product_without_an_adjusted_price(tmp_path: Path) -> None:
    """数据源因子只覆盖少数标的，默认口径仍须让全量同步后每只标的都有复权价格。"""
    import T01_get_data as script
    from data_sources.task_catalog import task_specs

    frame = pd.concat(
        [_candle([10.0, 10.2, 9.9], previous=[10.0, 10.0, 10.2]),
         _candle([3.0, 3.1, 3.2], code="159915.SZ")],
        ignore_index=True,
    ).assign(change=0.0, pct_chg=0.0, vol=1.0, amount=1.0, name="x")
    frame.to_parquet(tmp_path / "etf_daily_candle_df.parquet", index=False)
    pd.DataFrame(
        {"ts_code": "510300.SH", "trade_date": ["20250102", "20250103", "20250106"],
         "adj_factor": [2.0, 2.0, 2.06]}
    ).to_parquet(tmp_path / "fund_adj_factor_df.parquet", index=False)

    default = next(f for f in task_specs()["tushare.price_adjustment"]["parameters"]
                   if f["name"] == "factor_policy")["default"]
    assert default == DEFAULT_FACTOR_POLICY
    script.save_price_adjustment(tmp_path, default)

    written = pd.read_parquet(tmp_path / "etf_daily_candle_df.parquet")
    assert written["adj_close"].notna().all()
    # 有数据源因子的仍走数据源，其余才推导——默认口径不会把已有因子挤掉。
    assert dict(written.groupby("ts_code")["adj_factor_source"].first()) == {
        "159915.SZ": "pre_close", "510300.SH": "source",
    }
