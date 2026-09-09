from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from backend.market_data_validation import (
    SnapshotValidationError,
    _independent_nav_metrics,
    _validate_history,
    _source_fingerprint,
    validate_tushare_snapshot,
)


def _write_valid_snapshot(path: Path) -> None:
    path.mkdir()
    pd.DataFrame(
        [{"ts_code": "510050.SH", "name": "ETF", "status_code": "L"}]
    ).to_parquet(path / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "name": "基金", "status_code": "L"}]
    ).to_parquet(path / "fund_info_df.parquet", index=False)
    dates = pd.date_range("2026-08-28", periods=3)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "date": date, "adj_nav": 1.0 + index / 100} for index, date in enumerate(dates)]
    ).to_parquet(path / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "date": date, "adj_nav": 2.0 + index / 100} for index, date in enumerate(dates)]
    ).to_parquet(path / "fund_nav_df.parquet", index=False)
    pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": date, "close": 1.0, "vol": 10.0, "amount": 20.0}
            for date in dates
        ]
    ).to_parquet(path / "etf_daily_candle_df.parquet", index=False)
    etf_fingerprint = _source_fingerprint(path / "etf_daily_df.parquet")
    fund_fingerprint = _source_fingerprint(path / "fund_nav_df.parquet")
    pd.DataFrame(
        [
            {
                "instrument_type": "etf",
                "ts_code": "510050.SH",
                "latest_date": dates[-1],
                "as_of": dates[-1],
                "observation_count": 3,
                "return_1y": np.nan,
                "nav_source_fingerprint": etf_fingerprint,
            },
            {
                "instrument_type": "fund",
                "ts_code": "000001.OF",
                "latest_date": dates[-1],
                "as_of": dates[-1],
                "observation_count": 3,
                "return_1y": np.nan,
                "nav_source_fingerprint": fund_fingerprint,
            },
        ]
    ).to_parquet(path / "instrument_metrics_snapshot.parquet", index=False)


def test_complete_snapshot_passes_read_only_gate(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)

    report = validate_tushare_snapshot(snapshot)

    assert report["status"] == "passed"
    assert report["datasets"]["fund_info"]["rows"] == 1
    assert report["datasets"]["analytics_snapshot"]["by_kind"] == {"etf": 1, "fund": 1}


def test_duplicate_history_key_fails_gate(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    frame = pd.read_parquet(snapshot / "fund_nav_df.parquet")
    pd.concat([frame, frame.iloc[[-1]]], ignore_index=True).to_parquet(
        snapshot / "fund_nav_df.parquet", index=False
    )

    with pytest.raises(SnapshotValidationError, match="duplicate_keys"):
        validate_tushare_snapshot(snapshot)


def test_nullable_source_nav_is_reported_but_nonpositive_nav_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "nav.parquet"
    pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": pd.Timestamp("2026-08-28"), "adj_nav": 1.0},
            {"ts_code": "510050.SH", "date": pd.Timestamp("2026-08-29"), "adj_nav": np.nan},
        ]
    ).to_parquet(path, index=False)

    report, _, _ = _validate_history(path, "nav", 30)
    assert report["missing_values"] == 1

    invalid = pd.read_parquet(path)
    invalid.loc[1, "adj_nav"] = 0.0
    invalid.to_parquet(path, index=False)
    with pytest.raises(SnapshotValidationError, match="invalid_values"):
        _validate_history(path, "nav", 30)


def test_independent_metric_check_uses_full_period_coverage_rules() -> None:
    open_dates = pd.bdate_range("2026-05-29", "2026-09-01")
    missing = set(open_dates[12:20])
    observed = [date for date in open_dates if date not in missing]
    frame = pd.DataFrame(
        {
            "date": observed,
            "adj_nav": np.linspace(1.0, 1.1, len(observed)),
            "unit_nav": np.linspace(1.0, 1.1, len(observed)),
        }
    )

    metrics = _independent_nav_metrics(frame, open_dates)

    assert metrics["return_1m"] is not None
    assert metrics["return_3m"] is None


def test_empty_metrics_snapshot_cannot_pass_for_nonempty_information_tables(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    pd.DataFrame(
        columns=[
            "instrument_type",
            "ts_code",
            "latest_date",
            "observation_count",
            "nav_source_fingerprint",
        ]
    ).to_parquet(snapshot / "instrument_metrics_snapshot.parquet", index=False)

    with pytest.raises(SnapshotValidationError, match="指标行不足抽样门槛"):
        validate_tushare_snapshot(snapshot)


def test_snapshot_source_fingerprint_must_match_validated_history(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    nav = pd.read_parquet(snapshot / "fund_nav_df.parquet")
    nav.loc[len(nav)] = {
        "ts_code": "000001.OF",
        "date": pd.Timestamp("2026-09-01"),
        "adj_nav": 2.1,
    }
    nav.to_parquet(snapshot / "fund_nav_df.parquet", index=False)

    with pytest.raises(SnapshotValidationError, match="源文件指纹不一致"):
        validate_tushare_snapshot(snapshot)


@pytest.mark.parametrize("failure", [None, "value", "missing_result", "error", "status", "metadata"])
def test_configured_metric_gate_uses_recorded_version_and_uncached_values(tmp_path, monkeypatch, failure):
    from backend.custom_indicators import service as indicator_service

    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    path = snapshot / "instrument_metrics_snapshot.parquet"
    frame = pd.read_parquet(path)
    frame["sharpe_1y"] = 0.25
    frame["sharpe_1y__status"] = "ok"
    frame.to_parquet(path, index=False)
    item = {"field": "sharpe_1y", "indicator_id": "configured-sharpe", "indicator_revision": 7, "period": "1Y"}
    metadata = {"items": [item], "configured_count": 1}
    if failure == "metadata":
        metadata["items"] = []
    (snapshot / "instrument_metrics_snapshot.meta.json").write_text(json.dumps(metadata))
    calls = []

    class Evaluator:
        def __init__(self, **kwargs):
            assert kwargs == {"workspace_data_dir": tmp_path, "market_data_dir": snapshot}

        def warm_snapshot_numba_plans(self, items):
            assert items == [item]
            calls.append("warm")

        def evaluate(self, **kwargs):
            assert calls == ["warm"]
            assert kwargs["indicator_versions"] == {"configured-sharpe": 7}
            assert kwargs["prefer_snapshot"] is False
            assert kwargs["include_series"] is False
            assert kwargs["period"] == "1Y"
            calls.append("evaluate")
            return {"results": [] if failure == "missing_result" else [
                {"target": target, "indicator_id": "configured-sharpe", "status": "error" if failure == "error" else "warning" if failure == "status" else "ok", "value": 0.5 if failure == "value" else 0.25}
                for target in kwargs["targets"]
            ]}

    monkeypatch.setattr(indicator_service, "CustomIndicatorService", Evaluator)
    if failure:
        with pytest.raises(SnapshotValidationError):
            validate_tushare_snapshot(snapshot)
    else:
        report = validate_tushare_snapshot(snapshot)
        assert report["datasets"]["analytics_snapshot"]["configured_metric_samples"] == {"sharpe_1y": 2}
        assert pd.read_parquet(path)["sharpe_1y"].tolist() == [0.25, 0.25]


def test_configured_snapshot_missing_metadata_is_not_treated_as_fixed_formula(tmp_path):
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    path = snapshot / "instrument_metrics_snapshot.parquet"
    frame = pd.read_parquet(path)
    frame["return_1y__status"] = "unavailable"
    frame.to_parquet(path, index=False)
    with pytest.raises(SnapshotValidationError, match="版本记录"):
        validate_tushare_snapshot(snapshot)


def test_history_code_coverage_blocks_internally_consistent_truncation(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    fund_info = pd.DataFrame(
        [
            {"ts_code": f"{index:06d}.OF", "name": f"基金{index}", "status_code": "L"}
            for index in range(1, 41)
        ]
    )
    fund_info.to_parquet(snapshot / "fund_info_df.parquet", index=False)

    with pytest.raises(SnapshotValidationError, match="代码覆盖率过低"):
        validate_tushare_snapshot(snapshot, sample_size=1)


def test_strict_real_gate_requires_minimum_sample_population(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)

    with pytest.raises(SnapshotValidationError, match="真实验收门槛"):
        validate_tushare_snapshot(snapshot, strict=True)


def test_future_history_date_is_rejected(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    _write_valid_snapshot(snapshot)
    candle = pd.read_parquet(snapshot / "etf_daily_candle_df.parquet")
    start = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() + pd.Timedelta(days=5)
    candle["date"] = pd.date_range(start, periods=len(candle))
    candle.to_parquet(snapshot / "etf_daily_candle_df.parquet", index=False)

    with pytest.raises(SnapshotValidationError, match="未来最新日期"):
        validate_tushare_snapshot(snapshot)
