from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend import app as backend_app  # noqa: E402
from backend import trading_calendar  # noqa: E402


def _activate(root: Path, snapshot: Path) -> None:
    (root / "tushare_active.json").write_text(
        json.dumps({"schema_version": 1, "snapshot_dir": snapshot.name}),
        encoding="utf-8",
    )


def test_trading_calendar_cache_tracks_manifest_snapshot_identity(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "snapshot-v1"
    second = tmp_path / "snapshot-v2"
    first.mkdir()
    second.mkdir()
    pd.DataFrame(
        [{"exchange": "SSE", "cal_date": "20260828", "is_open": 1}]
    ).to_parquet(first / "trade_day_df.parquet", index=False)
    pd.DataFrame(
        [{"exchange": "SSE", "cal_date": "20260831", "is_open": 1}]
    ).to_parquet(second / "trade_day_df.parquet", index=False)
    monkeypatch.setattr(trading_calendar, "DATA_DIR", tmp_path)
    trading_calendar._load_calendar.cache_clear()

    _activate(tmp_path, first)
    first_days = trading_calendar.get_trading_days()
    _activate(tmp_path, second)
    second_days = trading_calendar.get_trading_days()

    assert first_days.tolist() == [pd.Timestamp("2026-08-28")]
    assert second_days.tolist() == [pd.Timestamp("2026-08-31")]
    trading_calendar._load_calendar.cache_clear()


def test_legacy_etf_search_prefers_active_manifest_parquet(monkeypatch, tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot-v1"
    snapshot.mkdir()
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "name": "沪深300ETF",
                "management": "当前管理人",
                "found_date": pd.Timestamp("2020-01-01"),
            }
        ]
    ).to_parquet(snapshot / "etf_info_df.parquet", index=False)
    (tmp_path / "etf_universe.json").write_text(
        json.dumps([{"code": "STALE.SH", "name": "陈旧ETF"}], ensure_ascii=False),
        encoding="utf-8",
    )
    _activate(tmp_path, snapshot)
    monkeypatch.setattr(backend_app, "DATA_DIR", tmp_path)
    backend_app._cached_universe_with_mtime.cache_clear()

    universe = backend_app._get_universe()

    assert universe == [
        {
            "code": "510300.SH",
            "name": "沪深300ETF",
            "management": "当前管理人",
            "found_date": "2020-01-01",
        }
    ]
    backend_app._cached_universe_with_mtime.cache_clear()
