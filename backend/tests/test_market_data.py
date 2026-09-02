from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from backend import market_data
from backend.market_data_validation import validate_tushare_snapshot
from backend.services.instrument_analytics import rebuild_analytics_snapshot


def _write_snapshot(snapshot: Path, filenames: tuple[str, ...]) -> None:
    snapshot.mkdir(parents=True)
    if set(filenames) != set(market_data.CORE_SNAPSHOT_FILES):
        for filename in filenames:
            (snapshot / filename).write_bytes(b"fixture")
        return
    pd.DataFrame(
        [{"ts_code": "510050.SH", "name": "ETF", "status_code": "L"}]
    ).to_parquet(snapshot / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "name": "基金", "status_code": "L"}]
    ).to_parquet(snapshot / "fund_info_df.parquet", index=False)
    dates = pd.date_range("2026-08-28", periods=3)
    pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": date, "adj_nav": 1 + index / 100, "unit_nav": 1 + index / 100}
            for index, date in enumerate(dates)
        ]
    ).to_parquet(snapshot / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": date, "adj_nav": 2 + index / 100}
            for index, date in enumerate(dates)
        ]
    ).to_parquet(snapshot / "fund_nav_df.parquet", index=False)
    pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": date, "close": 1 + index / 100, "vol": 10, "amount": 20}
            for index, date in enumerate(dates)
        ]
    ).to_parquet(snapshot / "etf_daily_candle_df.parquet", index=False)
    pd.DataFrame(
        [
            {"exchange": "SSE", "cal_date": date.strftime("%Y%m%d"), "is_open": 1}
            for date in dates
        ]
    ).to_parquet(snapshot / "trade_day_df.parquet", index=False)
    rebuild_analytics_snapshot(snapshot)


def test_resolver_falls_back_to_legacy_data_without_manifest(tmp_path: Path) -> None:
    assert market_data.resolve_tushare_data_dir(tmp_path) == tmp_path.resolve()


def test_activate_switches_manifest_only_after_validation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "verified"
    _write_snapshot(snapshot, market_data.CORE_SNAPSHOT_FILES)

    payload = market_data.activate_tushare_snapshot(snapshot, base_dir=tmp_path)

    assert payload["snapshot_dir"] == "snapshots/verified"
    assert market_data.resolve_tushare_data_dir(tmp_path) == snapshot.resolve()
    persisted = json.loads((tmp_path / market_data.ACTIVE_MANIFEST_NAME).read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 1


def test_activation_failure_keeps_previous_manifest(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    _write_snapshot(valid, market_data.CORE_SNAPSHOT_FILES)
    market_data.activate_tushare_snapshot(valid, base_dir=tmp_path)
    before = (tmp_path / market_data.ACTIVE_MANIFEST_NAME).read_bytes()

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    with pytest.raises(market_data.MarketDataManifestError, match="缺少必要文件"):
        market_data.activate_tushare_snapshot(invalid, base_dir=tmp_path)

    assert (tmp_path / market_data.ACTIVE_MANIFEST_NAME).read_bytes() == before


def test_manifest_cannot_escape_data_directory(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-market-data"
    outside.mkdir(exist_ok=True)
    (tmp_path / market_data.ACTIVE_MANIFEST_NAME).write_text(
        json.dumps({"schema_version": 1, "snapshot_dir": str(outside)}),
        encoding="utf-8",
    )

    with pytest.raises(market_data.MarketDataManifestError, match="必须位于"):
        market_data.resolve_tushare_data_dir(tmp_path, strict=True)
    assert market_data.resolve_tushare_data_dir(tmp_path) == tmp_path.resolve()


def test_activation_rejects_files_changed_after_validation(tmp_path: Path) -> None:
    snapshot = tmp_path / "candidate"
    _write_snapshot(snapshot, market_data.CORE_SNAPSHOT_FILES)
    report = validate_tushare_snapshot(snapshot)
    info = pd.read_parquet(snapshot / "fund_info_df.parquet")
    info.loc[0, "name"] = "验收后被修改"
    info.to_parquet(snapshot / "fund_info_df.parquet", index=False)

    with pytest.raises(market_data.MarketDataManifestError, match="验收后发生变化"):
        market_data.activate_tushare_snapshot(
            snapshot,
            base_dir=tmp_path,
            validation_report=report,
        )

    assert not (tmp_path / market_data.ACTIVE_MANIFEST_NAME).exists()
