from dataclasses import fields
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from backend.timing_research import data


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    pd.DataFrame({"ts_code": ["510300.SH"], "name": ["沪深300ETF"], "instrument_type": ["etf"],
                  "list_date": [dates[0]], "delist_date": [None], "qdii_type": ["非QDII"]}).to_parquet(root / data.INFO_FILE)
    pd.DataFrame({"exchange": ["SSE"] * 4, "cal_date": [d.strftime("%Y%m%d") for d in dates],
                  "is_open": [1] * 4}).to_parquet(root / data.CALENDAR_FILE)
    candle = pd.DataFrame({"ts_code": ["510300.SH"] * 3, "date": dates[[0, 2, 3]],
                           "open": [10., 5., 5.2], "high": [11., 5.5, 5.7], "low": [9., 4.5, 4.9],
                           "close": [10.5, 5.3, 5.5], "vol": [100., 200., 180.],
                           "unused_large_field": ["not read"] * 3})
    candle.to_parquet(root / data.CANDLE_FILE)
    pd.DataFrame({"ts_code": ["510300.SH"] * 3, "date": dates[[0, 2, 3]], "adj_factor": [1., 2., 2.]}).to_parquet(root / data.FACTOR_FILE)
    manifest = {"schema_version": 1, "snapshot_dir": "snapshot", "validation": {"status": "passed"},
                "files": {p.name: p.stat().st_size for p in root.iterdir()}}
    (tmp_path / "tushare_active.json").write_text(json.dumps(manifest))
    return tmp_path, root


def test_hfq_real_ohlcv_retains_missing_calendar_day(snapshot):
    base, _ = snapshot
    bars = data.load_etf_bars(base, "510300.SH", "2020-01-01", "2024-01-05")
    np.testing.assert_allclose(bars.close, [10.5, np.nan, 10.6, 11.], equal_nan=True)
    np.testing.assert_allclose(bars.raw_close, [10.5, np.nan, 5.3, 5.5], equal_nan=True)
    np.testing.assert_allclose(bars.volume, [100., np.nan, 200., 180.], equal_nan=True)
    assert bars.dates[0] == np.datetime64("2024-01-02", "D").astype(np.int64)
    assert bars.available_days[1] == data.MISSING_DAY
    assert bars.available_days[0] == bars.dates[0]
    from backend.timing_research.numeric import availability_status_kernel
    assert availability_status_kernel(bars.dates, bars.available_days, 0, len(bars.dates)) == 0
    assert bars.lineage["calendar_count"] == 4
    assert bars.lineage["observation_count"] == 3
    assert bars.lineage["revision_history_guaranteed"] is False
    assert bars.lineage["source_hash"].startswith("sha256:")
    assert any("缺失行情" in warning for warning in bars.warnings)
    for field in fields(bars):
        array = getattr(bars, field.name)
        if isinstance(array, np.ndarray):
            assert array.ndim == 1 and array.flags.c_contiguous
            assert array.dtype == (np.int64 if field.name in {"dates", "available_days"} else np.float64)
            assert not array.flags.writeable
    with pytest.raises(ValueError):
        bars.close[0] = 42
    view = bars.close[1:3]
    assert np.shares_memory(view, bars.close)
    assert not view.flags.writeable


def test_raw_arrays_alias_and_dont_require_factor(snapshot):
    base, root = snapshot
    (root / data.FACTOR_FILE).unlink()
    bars = data.load_etf_bars(base, "510300.SH", None, "2024-01-05", "raw")
    for field in data.OHLC:
        assert getattr(bars, field) is getattr(bars, "raw_" + field)
    assert any("不复权" in warning for warning in bars.warnings)


def test_qfq_anchor_is_cutoff_bound_and_future_perturbation_has_no_effect(snapshot):
    base, root = snapshot
    early = data.load_etf_bars(base, "510300.SH", None, "2024-01-02", "qfq")
    np.testing.assert_array_equal(early.close, [10.5])
    later = data.load_etf_bars(base, "510300.SH", None, "2024-01-05", "qfq")
    np.testing.assert_allclose(later.close, [5.25, np.nan, 5.3, 5.5], equal_nan=True)
    factors = pd.read_parquet(root / data.FACTOR_FILE)
    factors.loc[2, "adj_factor"] = 100
    factors.to_parquet(root / data.FACTOR_FILE)
    again = data.load_etf_bars(base, "510300.SH", None, "2024-01-02", "qfq")
    np.testing.assert_array_equal(again.close, early.close)
    assert again.lineage["source_hash"] == early.lineage["source_hash"]
    assert early.lineage["qfq_anchor"] == {"date": "2024-01-02", "factor": 1.}


@pytest.mark.parametrize("missing", ["file", "row", "nan", "zero"])
def test_adjustment_never_fabricates_missing_factors(snapshot, missing):
    base, root = snapshot
    path = root / data.FACTOR_FILE
    if missing == "file":
        path.unlink()
    else:
        table = pd.read_parquet(path)
        if missing == "row":
            table = table.iloc[:2]
        else:
            table.loc[2, "adj_factor"] = np.nan if missing == "nan" else 0.
        table.to_parquet(path)
    with pytest.raises(data.ValidationError, match="复权因子"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_missing_ohlc_is_not_substituted_with_nav(snapshot):
    base, root = snapshot
    frame = pd.read_parquet(root / data.CANDLE_FILE).drop(columns="open")
    frame["adj_nav"] = 10.
    frame.to_parquet(root / data.CANDLE_FILE)
    with pytest.raises(data.ValidationError, match="open"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_nulls_are_preserved_and_bad_prices_rejected(snapshot):
    base, root = snapshot
    frame = pd.read_parquet(root / data.CANDLE_FILE)
    frame.loc[0, "close"] = np.nan
    frame.to_parquet(root / data.CANDLE_FILE)
    assert np.isnan(data.load_etf_bars(base, "510300.SH", None, "2024-01-05").close[0])
    frame.loc[0, "close"] = 0.
    frame.to_parquet(root / data.CANDLE_FILE)
    with pytest.raises(data.ValidationError, match="非正价格"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_duplicate_days_fail_closed(snapshot):
    base, root = snapshot
    frame = pd.read_parquet(root / data.CANDLE_FILE)
    pd.concat([frame, frame.iloc[:1]], ignore_index=True).to_parquet(root / data.CANDLE_FILE)
    with pytest.raises(data.ValidationError, match="同日"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_requires_activated_files_and_snapshot(snapshot):
    base, _ = snapshot
    manifest_path = base / "tushare_active.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"][data.CANDLE_FILE]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(data.ValidationError, match="活跃快照缺少"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    manifest_path.unlink()
    with pytest.raises(data.ValidationError, match="激活"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_projection_pushdown_reads_each_product_file_once(snapshot, monkeypatch):
    base, _ = snapshot
    original = data._read_table
    calls = []

    def tracked(path, columns, predicate, **kwargs):
        calls.append((path.name, columns, str(predicate)))
        return original(path, columns, predicate, **kwargs)

    monkeypatch.setattr(data, "_read_table", tracked)
    data.load_etf_bars(base, "510300.SH", "2024-01-04", "2024-01-05")
    candles = [call for call in calls if call[0] == data.CANDLE_FILE]
    assert len(candles) == 1
    assert set(candles[0][1]) == {"date", "open", "high", "low", "close", "vol"}
    assert "510300.SH" in candles[0][2] and "2024-01-04" in candles[0][2]
    assert sum(call[0] == data.FACTOR_FILE for call in calls) == 1


def test_compact_date_fields_and_actual_shared_kernel_calls(snapshot, monkeypatch):
    base, root = snapshot
    for name in (data.CANDLE_FILE, data.FACTOR_FILE):
        table = pd.read_parquet(root / name)
        table["trade_date"] = table.pop("date").dt.strftime("%Y%m%d")
        table.to_parquet(root / name)
    calls = []
    original = data.adjusted_price_kernel
    before = tuple(original.signatures)

    def traced(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(data, "adjusted_price_kernel", traced)
    bars = data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    assert len(calls) == 4
    assert tuple(original.signatures) == before
    assert original.nopython_signatures
    assert bars.close[2] == 10.6


def test_maximum_bars_rejected_before_materializing_unbounded_data(snapshot, monkeypatch):
    _, root = snapshot
    with pytest.raises(data.ValidationError, match="最多支持"):
        data._read_table(root / data.CANDLE_FILE, ["date"], None, limit=2)


def test_snapshot_change_during_read_is_rejected(snapshot, monkeypatch):
    base, root = snapshot
    original = data._dated_table

    def changing(*args, **kwargs):
        result = original(*args, **kwargs)
        if args[0].name == data.FACTOR_FILE:
            path = root / data.INFO_FILE
            content = path.read_bytes()
            path.write_bytes(content)
        return result

    monkeypatch.setattr(data, "_dated_table", changing)
    with pytest.raises(data.ValidationError, match="发生变化"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_cross_border_and_late_available_data_fail_explicitly(snapshot):
    base, root = snapshot
    path = root / data.CANDLE_FILE
    frame = pd.read_parquet(path)
    frame["available_at"] = ["2024-01-03", "2024-01-04", "2024-01-06"]
    frame.to_parquet(path)
    with pytest.raises(data.ValidationError, match="截止日后"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    info = pd.read_parquet(root / data.INFO_FILE)
    info["qdii_type"] = "QDII"
    info.to_parquet(root / data.INFO_FILE)
    with pytest.raises(data.ValidationError, match="跨境 ETF"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_factor_availability_cannot_cross_cutoff(snapshot):
    base, root = snapshot
    frame = pd.read_parquet(root / data.FACTOR_FILE)
    frame["available_at"] = ["2024-01-02", "2024-01-04", "2024-01-06"]
    frame.to_parquet(root / data.FACTOR_FILE)
    with pytest.raises(data.ValidationError, match="复权因子可得日期"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_known_late_factor_cannot_be_used_on_earlier_market_date(snapshot):
    base, root = snapshot
    frame = pd.read_parquet(root / data.FACTOR_FILE)
    frame["available_at"] = ["2024-01-04", "2024-01-04", "2024-01-05"]
    frame.to_parquet(root / data.FACTOR_FILE)
    with pytest.raises(data.ValidationError, match="晚于对应行情日"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    frame.loc[0, "available_at"] = None
    frame.to_parquet(root / data.FACTOR_FILE)
    with pytest.raises(data.ValidationError, match="无效日期"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_calendar_tail_and_missing_interior_days_cannot_shorten_request(snapshot):
    base, root = snapshot
    calendar = pd.read_parquet(root / data.CALENDAR_FILE)
    calendar.iloc[:-1].to_parquet(root / data.CALENDAR_FILE)
    with pytest.raises(data.ValidationError, match="日历未完整覆盖"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    calendar.iloc[[0, 2, 3]].to_parquet(root / data.CALENDAR_FILE)
    with pytest.raises(data.ValidationError, match="日历未完整覆盖"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_actual_last_bar_must_cover_last_open_day(snapshot):
    base, root = snapshot
    path = root / data.CANDLE_FILE
    frame = pd.read_parquet(path)
    frame.iloc[:-1].to_parquet(path)
    with pytest.raises(data.ValidationError, match="未覆盖截止日前最后"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    frame.loc[2, "close"] = np.nan
    frame.to_parquet(path)
    with pytest.raises(data.ValidationError, match="未覆盖截止日前最后"):
        data.load_etf_bars(base, "510300.SH", None, "2024-01-05")


def test_calendar_verified_weekend_end_accepts_last_friday(snapshot):
    base, root = snapshot
    calendar = pd.read_parquet(root / data.CALENDAR_FILE)
    weekend = pd.DataFrame({"exchange": ["SSE", "SSE"], "cal_date": ["20240106", "20240107"], "is_open": [0, 0]})
    pd.concat([calendar, weekend], ignore_index=True).to_parquet(root / data.CALENDAR_FILE)
    bars = data.load_etf_bars(base, "510300.SH", None, "2024-01-07")
    assert bars.lineage["cutoff"] == "2024-01-07"
    assert bars.lineage["actual_end"] == "2024-01-05"
    assert len(bars.dates) == 4


@pytest.mark.parametrize("listed", [20240104, "20240104", "2024-01-04", pd.Timestamp("2024-01-04")])
def test_lifecycle_date_encoding_and_non_january_first_boundary(snapshot, listed):
    base, root = snapshot
    info = pd.read_parquet(root / data.INFO_FILE)
    info["list_date"] = [listed]
    info["delist_date"] = [20240105]
    info.to_parquet(root / data.INFO_FILE)
    bars = data.load_etf_bars(base, "510300.SH", None, "2024-01-05")
    assert bars.lineage["actual_start"] == "2024-01-04"
    assert len(bars.dates) == 2


def test_compact_and_iso_date_predicates_do_not_cross_match(tmp_path):
    from datetime import date
    import pyarrow.parquet as pq
    values = ["20120101", "20120527", "20120528", "20120601", "20120910", "20120911",
              "2012-01-01", "2012-05-27", "2012-05-28", "2012-06-01", "2012-09-10", "2012-09-11"]
    table = pa.table({"cal_date": values})
    path = tmp_path / "calendar.parquet"
    pq.write_table(table, path)
    predicate = data._date_predicate(table.schema, "cal_date", date(2012, 5, 28), date(2012, 9, 10))
    selected = data._read_table(path, ["cal_date"], predicate)["cal_date"].to_pylist()
    assert selected == ["20120528", "20120601", "20120910", "2012-05-28", "2012-06-01", "2012-09-10"]
