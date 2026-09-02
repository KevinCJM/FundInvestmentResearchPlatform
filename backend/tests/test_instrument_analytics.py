from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = ROOT / "backend"
for path in (ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.services import instrument_analytics, instrument_routes  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_small_table_cache():
    instrument_analytics._read_small_parquet_cached.cache_clear()
    yield
    instrument_analytics._read_small_parquet_cached.cache_clear()


def _write_info_files(data_dir: Path, *, include_fund: bool = True) -> dict[str, Path]:
    common = {
        "type": "契约型开放式",
        "custodian": "托管行",
        "invest_type": "被动指数型",
        "market": "上交所",
        "m_fee": 0.5,
        "c_fee": 0.1,
    }
    etf = pd.DataFrame(
        [
            {
                **common,
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "management": "管理人甲",
                "fund_type": "股票型",
                "status": "上市交易",
                "status_code": "L",
                "issue_amount": 100.0,
                "index_code": "000016.SH",
                "index_name": "上证50",
                "list_date": pd.Timestamp("2020-01-02"),
                "found_date": pd.Timestamp("2019-12-01"),
            },
            {
                **common,
                "ts_code": "510051.SH",
                "name": "已摘牌ETF",
                "management": "管理人乙",
                "fund_type": "股票型",
                "status": "摘牌",
                "status_code": "D",
                "issue_amount": 200.0,
                "list_date": pd.Timestamp("2021-03-04"),
                "found_date": pd.Timestamp("2020-12-01"),
            },
        ]
    )
    etf_path = data_dir / "etf_info_df.parquet"
    etf.to_parquet(etf_path, index=False)
    fund_path = data_dir / "fund_info_df.parquet"
    if include_fund:
        fund = pd.DataFrame(
            [
                {
                    **common,
                    "ts_code": "000001.OF",
                    "name": "场外基金A",
                    "management": "管理人甲",
                    "fund_type": "混合型",
                    "market": "场外",
                    "status": "存续",
                    "status_code": "L",
                "issue_amount": 300.0,
                "purc_startdate": pd.Timestamp("2018-05-10"),
                "redm_startdate": pd.Timestamp("2018-05-11"),
                "found_date": pd.Timestamp("2018-05-06"),
                    "list_date": pd.Timestamp("2099-01-01"),
                },
                {
                    **common,
                    "ts_code": "000002.OF",
                    "name": "场外基金B",
                    "management": "管理人丙",
                    "fund_type": "债券型",
                    "market": "场外",
                    "status": "到期/终止",
                    "status_code": "D",
                    "issue_amount": 400.0,
                    "found_date": pd.Timestamp("2019-07-08"),
                    "list_date": pd.NaT,
                },
            ]
        )
        fund.to_parquet(fund_path, index=False)
    return {"etf": etf_path, "fund": fund_path}


def _write_snapshot(data_dir: Path) -> None:
    pd.DataFrame(
        [
            {"ts_code": "510050.SH", "date": pd.Timestamp("2026-08-31"), "adj_nav": 1.0},
            {"ts_code": "510051.SH", "date": pd.Timestamp("2026-08-31"), "adj_nav": 1.0},
        ]
    ).to_parquet(data_dir / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-30"), "adj_nav": 1.0},
            {"ts_code": "000002.OF", "date": pd.Timestamp("2026-08-29"), "adj_nav": 1.0},
        ]
    ).to_parquet(data_dir / "fund_nav_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "date": pd.Timestamp("2026-08-31"), "close": 1.0}]
    ).to_parquet(data_dir / "etf_daily_candle_df.parquet", index=False)
    etf_fingerprint = instrument_analytics._source_fingerprint(data_dir / "etf_daily_df.parquet")
    fund_fingerprint = instrument_analytics._source_fingerprint(data_dir / "fund_nav_df.parquet")
    candle_fingerprint = instrument_analytics._source_fingerprint(
        data_dir / "etf_daily_candle_df.parquet"
    )
    pd.DataFrame(
        [
            {
                "instrument_type": "etf",
                "ts_code": "510050.SH",
                "as_of": pd.Timestamp("2026-08-31"),
                "latest_date": pd.Timestamp("2026-08-31"),
                "observation_count": 500,
                "return_1y": 0.08,
                "amount_avg_20d": 1000.0,
                "latest_candle_date": pd.Timestamp("2026-08-31"),
                "nav_source_fingerprint": etf_fingerprint,
                "candle_source_fingerprint": candle_fingerprint,
            },
            {
                "instrument_type": "fund",
                "ts_code": "000001.OF",
                "as_of": pd.Timestamp("2026-08-30"),
                "latest_date": pd.Timestamp("2026-08-30"),
                "observation_count": 450,
                "return_1y": 0.12,
                "nav_source_fingerprint": fund_fingerprint,
            },
            {
                "instrument_type": "fund",
                "ts_code": "000002.OF",
                "as_of": pd.Timestamp("2026-08-29"),
                "latest_date": pd.Timestamp("2026-08-29"),
                "observation_count": 300,
                "return_1y": 0.50,
                "nav_source_fingerprint": fund_fingerprint,
            },
        ]
    ).to_parquet(data_dir / instrument_analytics.SNAPSHOT_FILENAME, index=False)


def test_all_market_summary_uses_status_code_and_never_combines_issue_amount(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)
    _write_snapshot(tmp_path)

    response = instrument_analytics.build_analytics_response(
        kind="all", data_dir=tmp_path, info_files=info_files
    )

    assert response["status"] == "complete"
    assert response["summary"]["all"]["share_code_count"] == 4
    assert response["summary"]["all"]["active_count"] == 2
    assert response["summary"]["all"]["inactive_count"] == 2
    assert "issue_amount_total" not in response["summary"]["all"]
    assert response["summary"]["etf"]["issue_amount_total"] == 300.0
    assert response["summary"]["fund"]["issue_amount_total"] == 700.0
    assert response["summary"]["etf"]["nav_covered_count"] == 1
    assert response["summary"]["fund"]["nav_covered_count"] == 2
    assert response["summary"]["etf"]["index_covered_count"] == 1
    assert response["summary"]["etf"]["liquidity_covered_count"] == 1
    assert response["summary"]["fund"]["purchase_redemption_covered_count"] == 1
    assert any(
        warning["code"] == "ETF_NAV_PARTIAL"
        and warning["message"] == "部分产品代码缺少分析快照，相关业绩指标将显示为空。"
        for warning in response["data_quality"]["warnings"]
    )
    assert response["availability"] == {
        "etf_info": "ready",
        "fund_info": "ready",
        "analysis_snapshot": "ready",
    }
    assert response["units"]["return_1y"] == "ratio"
    assert response["segments"]["etf"]["distributions"]["m_fee"]
    assert response["segments"]["etf"]["latest_products"][0]["ts_code"] == "510051.SH"
    assert response["segments"]["etf"]["latest_products"][0]["list_date"] == "2021-03-04"
    assert response["segments"]["fund"]["latest_products"][0]["ts_code"] == "000002.OF"
    assert response["segments"]["fund"]["latest_products"][0]["found_date"] == "2019-07-08"


def test_missing_snapshot_keeps_structure_but_marks_response_partial(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)

    response = instrument_analytics.build_analytics_response(
        kind="all", data_dir=tmp_path, info_files=info_files
    )

    assert response["status"] == "partial"
    assert response["availability"]["analysis_snapshot"] == "missing"
    assert response["summary"]["all"]["share_code_count"] == 4
    assert response["summary"]["etf"]["nav_coverage_rate"] == 0
    assert any(
        warning["code"] == "ANALYTICS_SNAPSHOT_MISSING"
        for warning in response["data_quality"]["warnings"]
    )


def test_event_dates_are_etf_list_date_and_fund_found_date(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)

    response = instrument_analytics.build_trend_response(
        kind="all", data_dir=tmp_path, info_files=info_files
    )

    assert response["series"]["etf"]["date_field"] == "list_date"
    assert [point["year"] for point in response["series"]["etf"]["points"]] == [2020, 2021]
    assert response["series"]["fund"]["date_field"] == "found_date"
    assert [point["year"] for point in response["series"]["fund"]["points"]] == [2018, 2019]
    assert 2099 not in [point["year"] for point in response["series"]["fund"]["points"]]


def test_missing_fund_data_is_partial_http_200(tmp_path: Path, monkeypatch) -> None:
    info_files = _write_info_files(tmp_path, include_fund=False)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", info_files)
    app = FastAPI()
    app.include_router(instrument_routes.router)

    response = TestClient(app).get("/api/instruments/analytics?kind=all")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "partial"
    assert payload["segments"]["etf"]["availability"] == "ready"
    assert payload["segments"]["fund"]["availability"] == "missing"
    assert payload["summary"]["fund"]["share_code_count"] == 0


def test_rankings_use_small_snapshot_and_default_to_active_products(tmp_path: Path, monkeypatch) -> None:
    info_files = _write_info_files(tmp_path)
    _write_snapshot(tmp_path)
    original_read = instrument_analytics.pd.read_parquet

    def guarded_read(path, *args, **kwargs):
        if Path(path).name in {"fund_nav_df.parquet", "etf_daily_df.parquet", "etf_daily_candle_df.parquet"}:
            raise AssertionError("request path must not read full history")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(instrument_analytics.pd, "read_parquet", guarded_read)

    response = instrument_analytics.build_rankings_response(
        kind="fund",
        metric="return_1y",
        data_dir=tmp_path,
        info_files=info_files,
    )

    assert response["status"] == "complete"
    assert response["total"] == 1
    assert response["items"][0]["ts_code"] == "000001.OF"
    assert response["items"][0]["value"] == pytest.approx(0.12)


def test_changed_nav_source_marks_snapshot_stale_without_reading_history(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)
    _write_snapshot(tmp_path)
    fund_path = tmp_path / "fund_nav_df.parquet"
    changed = pd.read_parquet(fund_path)
    changed.loc[len(changed)] = {
        "ts_code": "000001.OF",
        "date": pd.Timestamp("2026-09-01"),
        "adj_nav": 1.01,
    }
    changed.to_parquet(fund_path, index=False)

    response = instrument_analytics.build_analytics_response(
        kind="fund", data_dir=tmp_path, info_files=info_files
    )
    ranking = instrument_analytics.build_rankings_response(
        kind="fund", metric="m_fee", data_dir=tmp_path, info_files=info_files
    )

    assert response["status"] == "partial"
    assert response["availability"]["analysis_snapshot"] == "stale"
    assert response["summary"]["fund"]["nav_covered_count"] == 0
    assert ranking["status"] == "unavailable"
    assert ranking["items"] == []


def test_kind_specific_as_of_does_not_leak_etf_date_into_fund(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)
    _write_snapshot(tmp_path)

    response = instrument_analytics.build_analytics_response(
        kind="fund", data_dir=tmp_path, info_files=info_files
    )

    assert response["as_of"] == "2026-08-30"


def test_rankings_exclude_active_but_stale_snapshot_rows(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)
    fund = pd.read_parquet(info_files["fund"])
    fund = pd.concat(
        [
            fund,
            pd.DataFrame(
                [
                    {
                        **fund.iloc[0].to_dict(),
                        "ts_code": "000003.OF",
                        "name": "陈旧净值基金",
                        "status": "存续",
                        "status_code": "L",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fund.to_parquet(info_files["fund"], index=False)
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "date": pd.Timestamp("2026-08-31"), "adj_nav": 1.0},
            {"ts_code": "000003.OF", "date": pd.Timestamp("2026-07-01"), "adj_nav": 1.0},
        ]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    source_fingerprint = instrument_analytics._source_fingerprint(tmp_path / "fund_nav_df.parquet")
    pd.DataFrame(
        [
            {
                "instrument_type": "fund",
                "ts_code": "000001.OF",
                "latest_date": pd.Timestamp("2026-08-31"),
                "observation_count": 500,
                "return_1y": 0.12,
                "nav_source_fingerprint": source_fingerprint,
            },
            {
                "instrument_type": "fund",
                "ts_code": "000003.OF",
                "latest_date": pd.Timestamp("2026-07-01"),
                "observation_count": 500,
                "return_1y": 0.99,
                "nav_source_fingerprint": source_fingerprint,
            },
        ]
    ).to_parquet(tmp_path / instrument_analytics.SNAPSHOT_FILENAME, index=False)

    response = instrument_analytics.build_rankings_response(
        kind="fund", metric="return_1y", data_dir=tmp_path, info_files=info_files
    )

    assert response["total"] == 1
    assert response["items"][0]["ts_code"] == "000001.OF"


def _series_rows(code: str, dates: pd.DatetimeIndex, start: float, *, include_unit_nav: bool) -> list[dict]:
    rows = []
    for index, date in enumerate(dates):
        value = start * (1.0 + 0.001 * index)
        row = {"ts_code": code, "date": date, "adj_nav": value}
        if include_unit_nav:
            row["unit_nav"] = value
        rows.append(row)
    return rows


def test_rebuild_snapshot_streams_row_groups_and_computes_etf_specific_metrics(tmp_path: Path, monkeypatch) -> None:
    dates = pd.date_range("2025-01-01", periods=420, freq="D")
    etf_rows = _series_rows("510050.SH", dates, 1.0, include_unit_nav=True)
    etf_rows += _series_rows("510051.SH", dates[:30], 2.0, include_unit_nav=True)
    pd.DataFrame(etf_rows).sort_values(["ts_code", "date"]).to_parquet(
        tmp_path / "etf_daily_df.parquet", index=False, row_group_size=37
    )
    fund_rows = _series_rows("000001.OF", dates, 1.5, include_unit_nav=False)
    pd.DataFrame(fund_rows).to_parquet(
        tmp_path / "fund_nav_df.parquet", index=False, row_group_size=41
    )
    unit_by_date = {row["date"]: row["unit_nav"] for row in etf_rows if row["ts_code"] == "510050.SH"}
    candle_dates = dates[-25:]
    candle = pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "date": date,
                "close": unit_by_date[date] * 1.01,
                "amount": float(index + 1),
                "vol": float((index + 1) * 10),
            }
            for index, date in enumerate(candle_dates)
        ]
    )
    candle.to_parquet(tmp_path / "etf_daily_candle_df.parquet", index=False, row_group_size=7)

    def forbid_full_pandas_read(*_args, **_kwargs):
        raise AssertionError("snapshot builder must stream with pyarrow row groups")

    monkeypatch.setattr(instrument_analytics.pd, "read_parquet", forbid_full_pandas_read)

    summary = instrument_analytics.rebuild_analytics_snapshot(tmp_path)
    snapshot = parquet_read(tmp_path / instrument_analytics.SNAPSHOT_FILENAME)

    assert summary["rows"] == 3
    assert summary["by_kind"] == {"etf": 2, "fund": 1}
    first = snapshot[(snapshot["instrument_type"] == "etf") & (snapshot["ts_code"] == "510050.SH")].iloc[0]
    assert first["observation_count"] == 420
    assert first["return_1y"] > 0
    assert first["premium_discount_latest"] == pytest.approx(0.01)
    assert first["premium_discount_date"] == dates[-1]
    assert first["amount_avg_20d"] == pytest.approx(np.mean(np.arange(6, 26)))
    assert first["volume_avg_20d"] == pytest.approx(np.mean(np.arange(6, 26) * 10))


def test_snapshot_short_history_leaves_risk_metrics_null(tmp_path: Path) -> None:
    dates = pd.date_range("2026-01-01", periods=50, freq="D")
    pd.DataFrame(
        _series_rows("000001.OF", dates, 1.0, include_unit_nav=False)
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)

    instrument_analytics.rebuild_analytics_snapshot(tmp_path)
    row = parquet_read(tmp_path / instrument_analytics.SNAPSHOT_FILENAME).iloc[0]

    assert pd.isna(row["annual_volatility_1y"])
    assert pd.isna(row["sharpe_1y"])
    assert pd.isna(row["max_drawdown_3y"])
    assert pd.isna(row["calmar_3y"])


def test_snapshot_rejects_sparse_series_even_when_old_minimum_count_is_met(tmp_path: Path) -> None:
    dates = pd.date_range("2025-08-31", "2026-08-31", freq="5D")
    pd.DataFrame(
        _series_rows("000001.OF", dates, 1.0, include_unit_nav=False)
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)

    instrument_analytics.rebuild_analytics_snapshot(tmp_path)
    row = parquet_read(tmp_path / instrument_analytics.SNAPSHOT_FILENAME).iloc[0]

    assert len(dates) >= 60
    assert pd.isna(row["return_1y"])
    assert pd.isna(row["annual_volatility_1y"])
    assert pd.isna(row["sharpe_1y"])
    assert row["quality_reason_1y"] == "insufficient_density"
    assert row["coverage_ratio_1y"] < 0.8


def test_snapshot_quarantines_adjusted_nav_dislocation_only_in_affected_windows(tmp_path: Path) -> None:
    dates = pd.bdate_range("2025-08-28", "2026-08-28")
    smooth = 1.0 + np.arange(len(dates)) * 0.0001
    adjusted = smooth.copy()
    adjusted[dates >= pd.Timestamp("2026-06-02")] *= 100.0
    pd.DataFrame(
        {
            "ts_code": "003816.OF",
            "date": dates,
            "unit_nav": smooth,
            "accum_nav": smooth,
            "adj_nav": adjusted,
        }
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    info_path = tmp_path / "fund_info_df.parquet"
    pd.DataFrame(
        [
            {
                "ts_code": "003816.OF",
                "name": "银华货币-B",
                "status": "存续",
                "status_code": "L",
                "fund_type": "货币型",
            }
        ]
    ).to_parquet(info_path, index=False)

    instrument_analytics.rebuild_analytics_snapshot(tmp_path)
    row = parquet_read(tmp_path / instrument_analytics.SNAPSHOT_FILENAME).iloc[0]

    one_month_target = dates[-1] - pd.DateOffset(months=1)
    one_month_anchor = np.flatnonzero(dates <= one_month_target)[-1]
    assert row["adj_nav_anomaly_count"] == 1
    assert row["return_1m"] == pytest.approx(smooth[-1] / smooth[one_month_anchor] - 1)
    assert pd.isna(row["return_3m"])
    assert pd.isna(row["return_1y"])
    assert pd.isna(row["annual_volatility_1y"])
    assert pd.isna(row["max_drawdown_3y"])
    assert row["quality_reason_1m"] is None
    assert row["quality_reason_3m"] == "adjusted_nav_anomaly"
    assert row["quality_reason_1y"] == "adjusted_nav_anomaly"

    one_year_ranking = instrument_analytics.build_rankings_response(
        kind="fund",
        metric="return_1y",
        data_dir=tmp_path,
        info_files={"fund": info_path},
    )
    one_month_ranking = instrument_analytics.build_rankings_response(
        kind="fund",
        metric="return_1m",
        data_dir=tmp_path,
        info_files={"fund": info_path},
    )
    assert one_year_ranking["items"] == []
    assert one_month_ranking["items"][0]["ts_code"] == "003816.OF"
    assert one_month_ranking["items"][0]["observation_count"] == row["observation_count_1m"]


def parquet_read(path: Path) -> pd.DataFrame:
    return instrument_analytics.parquet.read_table(path).to_pandas()


def test_snapshot_write_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    previous_path = tmp_path / instrument_analytics.SNAPSHOT_FILENAME
    pd.DataFrame(
        [{"instrument_type": "etf", "ts_code": "OLD.SH", "return_1y": 0.1}]
    ).to_parquet(previous_path, index=False)
    pd.DataFrame(
        _series_rows("510050.SH", pd.date_range("2026-01-01", periods=10), 1.0, include_unit_nav=True)
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)

    def fail_write(*_args, **_kwargs):
        raise RuntimeError("simulated snapshot write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(RuntimeError, match="simulated snapshot"):
        instrument_analytics.rebuild_analytics_snapshot(tmp_path)

    preserved = parquet_read(previous_path)
    assert preserved["ts_code"].tolist() == ["OLD.SH"]
    assert list(tmp_path.glob(f".{instrument_analytics.SNAPSHOT_FILENAME}.*.tmp")) == []


def test_rankings_reject_inactive_share_classes_even_when_requested(tmp_path: Path) -> None:
    info_files = _write_info_files(tmp_path)
    _write_snapshot(tmp_path)

    with pytest.raises(ValueError, match="status_code=L"):
        instrument_analytics.build_rankings_response(
            kind="fund",
            metric="return_1y",
            active_only=False,
            data_dir=tmp_path,
            info_files=info_files,
        )


def test_etf_twenty_day_averages_require_twenty_finite_observations() -> None:
    dates = pd.date_range("2026-08-01", periods=20, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(1.0, 1.2, 20),
            "amount": [*np.arange(1.0, 20.0), np.inf],
            "vol": np.arange(1.0, 21.0),
        }
    )

    record = instrument_analytics._compute_candle_record(frame, {}, "fingerprint")

    assert record["amount_avg_20d"] is None
    assert record["volume_avg_20d"] == pytest.approx(10.5)


def test_event_trend_ignores_non_finite_issue_amounts() -> None:
    frame = pd.DataFrame(
        [
            {"found_date": pd.Timestamp("2026-01-01"), "issue_amount": 100.0},
            {"found_date": pd.Timestamp("2026-02-01"), "issue_amount": np.inf},
        ]
    )

    trend = instrument_analytics._event_trend(frame, "fund")

    assert trend["points"] == [{"year": 2026, "count": 2, "total_issue_amount": 100.0}]
