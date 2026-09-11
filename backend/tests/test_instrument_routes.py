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

from backend.services import instrument_routes  # noqa: E402
from backend.services import instrument_analytics  # noqa: E402
from backend import fit  # noqa: E402


def _write_info_files(data_dir: Path) -> None:
    common = {
        "custodian": "托管行",
        "type": "契约型开放式",
        "invest_type": "指数型",
        "status": "存续",
        "issue_amount": 10000.0,
        "m_fee": 0.5,
        "c_fee": 0.1,
        "found_date": pd.Timestamp("2020-01-01"),
    }
    pd.DataFrame(
        [{
            "ts_code": "510050.SH",
            "code": "510050",
            "name": "上证50ETF",
            "management": "华夏基金",
            "fund_type": "股票型",
            "market": "上交所",
            "list_date": pd.Timestamp("2020-01-02"),
            "delist_date": pd.Timestamp("2026-12-31"),
            **common,
        }]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [{
            "ts_code": "000001.OF",
            "code": "000001",
            "name": "华夏成长",
            "management": "华夏基金",
            "fund_type": "混合型",
            "market": "场外",
            "due_date": pd.Timestamp("2030-12-31"),
            **common,
        }]
    ).to_parquet(data_dir / "fund_info_df.parquet", index=False)


def test_instrument_search_combines_etf_and_public_fund(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_search(
        q="华夏", kind="all", sort_by="name", sort_dir="asc", page=1, page_size=10
    )

    assert response["total"] == 2
    assert {item["instrument_type"] for item in response["items"]} == {"etf", "fund"}
    assert {item["code"] for item in response["items"]} == {"510050.SH", "000001.OF"}


def test_product_query_keeps_fund_universe_separate(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "nav_date": "20260630",
                "net_asset": 300_000_000.0,
                "total_netasset": np.nan,
            },
            {
                "ts_code": "000001.OF",
                "nav_date": "20260828",
                "net_asset": 320_000_000.0,
                "total_netasset": 350_000_000.0,
            },
            {
                "ts_code": "999999.OF",
                "nav_date": "20260828",
                "net_asset": 990_000_000.0,
                "total_netasset": np.nan,
            },
        ]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_products(
        kind="fund",
        q="",
        fund_type=None,
        fund_category=None,
        invest_type=None,
        market=None,
        status=None,
        management=None,
        custodian=None,
        page=1,
        page_size=20,
        sort_by="issue_amount",
        sort_dir="desc",
    )

    assert response["kind"] == "fund"
    assert response["total"] == 1
    assert response["items"][0]["ts_code"] == "000001.OF"
    assert "current_size" not in response["items"][0]
    assert response["items"][0]["snapshot_values"] == {}
    assert response["summary"]["universe_total"] == 1
    assert response["summary"]["active_count"] == 1
    assert response["summary"]["active_rate"] == pytest.approx(1.0)
    assert response["execution"]["backend"] == "numba_njit_fixed_signature"
    assert response["execution"]["request_time_compilation"] == 0


def _write_product_filter_fixture(data_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "ts_code": "510001.SH",
                "code": "510001",
                "name": "十点收益ETF",
                "management": "示例基金",
                "fund_type": "股票型",
                "status": "上市",
                "list_date": pd.Timestamp("2020-01-01"),
            },
            {
                "ts_code": "510002.SH",
                "code": "510002",
                "name": "二十点收益ETF",
                "management": "示例基金",
                "fund_type": "股票型",
                "status": "上市",
                "list_date": pd.Timestamp("2022-01-01"),
            },
            {
                "ts_code": "510003.SH",
                "code": "510003",
                "name": "指标缺失ETF",
                "management": "示例基金",
                "fund_type": "股票型",
                "status": "上市",
                "list_date": pd.Timestamp("2023-01-01"),
            },
        ]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "code": "000001",
                "name": "示例基金",
                "management": "示例基金",
                "fund_type": "混合型",
                "status": "存续",
                "found_date": pd.Timestamp("2021-05-01"),
            }
        ]
    ).to_parquet(data_dir / "fund_info_df.parquet", index=False)
    nav_path = data_dir / "etf_daily_df.parquet"
    pd.DataFrame(
        [
            {"ts_code": "510001.SH", "date": pd.Timestamp("2026-08-31"), "adj_nav": 1.1},
            {"ts_code": "510002.SH", "date": pd.Timestamp("2026-08-31"), "adj_nav": 1.2},
        ]
    ).to_parquet(nav_path, index=False)
    fingerprint = instrument_analytics._source_fingerprint(nav_path)
    pd.DataFrame(
        [
            {
                "instrument_type": "etf",
                "ts_code": "510001.SH",
                "as_of": pd.Timestamp("2026-08-31"),
                "latest_date": pd.Timestamp("2026-08-31"),
                "nav_source_fingerprint": fingerprint,
                "return_1y": 0.10,
            },
            {
                "instrument_type": "etf",
                "ts_code": "510002.SH",
                "as_of": pd.Timestamp("2026-08-31"),
                "latest_date": pd.Timestamp("2026-08-31"),
                "nav_source_fingerprint": fingerprint,
                "return_1y": 0.20,
            },
            {
                "instrument_type": "etf",
                "ts_code": "510003.SH",
                "as_of": pd.Timestamp("2026-08-31"),
                "latest_date": pd.Timestamp("2026-08-31"),
                "nav_source_fingerprint": fingerprint,
                "return_1y": np.nan,
            },
        ]
    ).to_parquet(data_dir / "instrument_metrics_snapshot.parquet", index=False)


def _product_filter_client(monkeypatch, data_dir: Path) -> TestClient:
    monkeypatch.setattr(instrument_routes, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        instrument_routes,
        "INSTRUMENT_FILES",
        {
            "etf": data_dir / "etf_info_df.parquet",
            "fund": data_dir / "fund_info_df.parquet",
        },
    )
    instrument_analytics._read_small_parquet_cached.cache_clear()
    app = FastAPI()
    app.include_router(instrument_routes.router)
    return TestClient(app)


@pytest.mark.parametrize(
    ("operator", "value", "expected_codes"),
    [
        ("gte", "10", {"510001.SH", "510002.SH"}),
        ("lte", "20", {"510001.SH", "510002.SH"}),
        ("gt", "10", {"510002.SH"}),
        ("lt", "20", {"510001.SH"}),
        ("eq", "10", {"510001.SH"}),
    ],
)
def test_product_query_filters_validated_snapshot_metrics(
    monkeypatch,
    tmp_path: Path,
    operator: str,
    value: str,
    expected_codes: set[str],
) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params={"kind": "etf", "condition": f"return_1y|{operator}|{value}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert {item["ts_code"] for item in payload["items"]} == expected_codes
    assert all(item["ts_code"] != "510003.SH" for item in payload["items"])
    assert payload["snapshot"]["status"] == "ready"
    assert payload["applied_conditions"][0]["operator"] == operator


def test_product_query_filters_etf_listing_date(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params={"kind": "etf", "condition": "list_date|gte|2022-01-01"},
    )

    assert response.status_code == 200
    assert {item["ts_code"] for item in response.json()["items"]} == {"510002.SH", "510003.SH"}
    assert response.json()["condition_fields"][0]["label"] == "上市日期"


def test_product_query_displays_only_requested_snapshot_metrics(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    default_response = client.get("/api/instruments/products", params={"kind": "etf"})
    selected_response = client.get(
        "/api/instruments/products",
        params=[("kind", "etf"), ("snapshot_metric", "return_1y")],
    )

    assert default_response.status_code == 200
    assert default_response.json()["items"][0]["snapshot_values"] == {}
    assert selected_response.status_code == 200
    payload = selected_response.json()
    assert payload["selected_snapshot_metrics"] == ["return_1y"]
    assert payload["items"][0]["snapshot_values"]["return_1y"] in {0.1, 0.2}
    assert payload["items"][0]["snapshot_value_dates"]["return_1y"] == "2026-08-31"
    assert any(
        field["field"] == "return_1y" and field["label"] == "近1年收益率"
        for field in payload["snapshot_metric_fields"]
    )
    fields = {field["field"]: field for field in payload["snapshot_metric_fields"]}
    assert fields["return_1y"]["metric_source"] == "built_in"
    assert fields["return_1y"]["metric_source_label"] == "内置指标"
    assert fields["return_1y"]["metric_type"] == "return"
    assert fields["return_1y"]["metric_type_label"] == "收益型指标"
    assert fields["current_size"]["metric_source"] == "system_derived"
    assert fields["current_size"]["metric_type_label"] == "规模指标"


def test_product_selection_returns_all_matching_identities(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products/selection",
        params=[
            ("kind", "etf"),
            ("fund_type", "股票型"),
            ("condition", "list_date|gte|2022-01-01"),
        ],
    )

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "code": "510002.SH",
                "ts_code": "510002.SH",
                "name": "二十点收益ETF",
                "instrument_type": "etf",
            },
            {
                "code": "510003.SH",
                "ts_code": "510003.SH",
                "name": "指标缺失ETF",
                "instrument_type": "etf",
            },
        ],
        "total": 2,
        "kind": "etf",
    }


def test_product_query_defaults_to_ten_items(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get("/api/instruments/products", params={"kind": "etf"})

    assert response.status_code == 200
    assert response.json()["page_size"] == 10


@pytest.mark.parametrize("kind", ["etf", "fund"])
def test_product_query_exposes_and_filters_legacy_qdii_classification(
    monkeypatch, tmp_path: Path, kind: str
) -> None:
    _write_info_files(tmp_path)
    path = tmp_path / ("etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet")
    frame = pd.read_parquet(path)
    frame.loc[:, "name"] = frame["name"].astype(str) + "(QDII)"
    frame.to_parquet(path, index=False)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params=[("kind", kind), ("qdii_type", "QDII")],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["qdii_type"] == "QDII"
    assert payload["items"][0]["qdii_source"] == "legacy_info.name_marker"
    assert payload["available_filters"]["qdii_type"] == [
        {"value": "QDII", "label": "QDII", "count": 1}
    ]


def test_product_query_uses_founding_date_for_public_funds(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params={"kind": "fund", "condition": "found_date|eq|2021-05-01"},
    )

    assert response.status_code == 200
    assert [item["ts_code"] for item in response.json()["items"]] == ["000001.OF"]
    assert response.json()["condition_fields"][0]["label"] == "成立日期"


def test_product_query_combines_multiple_conditions_with_and(monkeypatch, tmp_path: Path) -> None:
    _write_product_filter_fixture(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params=[
            ("kind", "etf"),
            ("condition", "list_date|gte|2021-01-01"),
            ("condition", "return_1y|lte|20"),
        ],
    )

    assert response.status_code == 200
    assert [item["ts_code"] for item in response.json()["items"]] == ["510002.SH"]


def test_product_query_rejects_metric_filter_when_snapshot_is_missing(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    client = _product_filter_client(monkeypatch, tmp_path)

    response = client.get(
        "/api/instruments/products",
        params={"kind": "etf", "condition": "return_1y|gte|10"},
    )

    assert response.status_code == 409
    assert "分析快照未就绪" in response.json()["detail"]


def test_asset_allocation_loader_combines_etf_and_public_fund_nav(tmp_path: Path) -> None:
    pd.DataFrame(
        [{"ts_code": "510050.SH", "name": "上证50ETF", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.0}]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "name": "华夏成长", "date": pd.Timestamp("2026-08-28"), "adj_nav": 3.0}]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)

    result = fit._load_adj_nav(tmp_path, ["510050.SH", "000001.OF"], ["上证50ETF", "华夏成长"])

    assert set(result["ts_code"]) == {"510050.SH", "000001.OF"}


def test_public_fund_detail_uses_real_nav_timeseries(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    pd.DataFrame(
        [
            {"ts_code": "000001.OF", "name": "华夏成长", "date": pd.Timestamp("2026-08-27"), "adj_nav": 2.0},
            {"ts_code": "000001.OF", "name": "华夏成长", "date": pd.Timestamp("2026-08-28"), "adj_nav": 2.1},
        ]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_product_detail("000001.OF", kind="fund")

    assert response["kind"] == "fund"
    assert response["name"] == "华夏成长"
    assert response["base_info"]["found_date"] == "2020-01-01"
    assert response["base_info"]["due_date"] == "2030-12-31"
    assert [point["close"] for point in response["timeseries"]] == [2.0, 2.1]
    assert {
        field: response["timeseries"][0][field]
        for field in ("open", "high", "low", "volume")
    } == {"open": None, "high": None, "low": None, "volume": None}
    assert response["execution"]["python_fallback"] == 0


def test_etf_detail_uses_real_nav_timeseries_without_synthetic_fallback(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "date": pd.Timestamp("2026-08-27"),
                "nav_date": "20260827",
                "adj_nav": 3.0,
                "net_asset": 800_000_000.0,
                "total_netasset": np.nan,
            },
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "date": pd.Timestamp("2026-08-28"),
                "nav_date": "20260828",
                "adj_nav": 3.2,
                "net_asset": 900_000_000.0,
                "total_netasset": 1_000_000_000.0,
            },
        ]
    ).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "trade_date": "20260827",
                "date": pd.Timestamp("2026-08-27"),
                "total_share": 200_000.0,
                "nav": 3.0,
            },
            {
                "ts_code": "510050.SH",
                "trade_date": "20260828",
                "date": pd.Timestamp("2026-08-28"),
                "total_share": 250_000.0,
                "nav": 3.2,
            },
        ]
    ).to_parquet(tmp_path / "etf_share_size_df.parquet", index=False)
    instrument_analytics.rebuild_analytics_snapshot(tmp_path)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_product_detail("510050.SH", kind="etf")

    assert response["kind"] == "etf"
    assert response["name"] == "上证50ETF"
    assert response["base_info"]["list_date"] == "2020-01-02"
    assert response["base_info"]["delist_date"] == "2026-12-31"
    assert [point["close"] for point in response["timeseries"]] == [3.0, 3.2]
    assert {
        field: response["timeseries"][0][field]
        for field in ("open", "high", "low", "volume")
    } == {"open": None, "high": None, "low": None, "volume": None}
    assert response["execution"]["python_fallback"] == 0
    assert response["metrics"]["current_size"] == 800_000.0
    assert response["metrics"]["current_size_as_of"] == "2026-08-28"
    assert response["metrics"]["current_size_source"] == "instrument_metrics_snapshot"
    assert response["metrics"]["current_share"] == 250_000.0
    assert response["metrics"]["current_unit_nav"] == 3.2


def test_fund_detail_does_not_substitute_net_asset_for_share_times_unit_nav(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "date": pd.Timestamp("2026-06-30"),
                "nav_date": "20260630",
                "adj_nav": 2.0,
                "net_asset": 320_000_000.0,
                "total_netasset": np.nan,
            }
        ]
    ).to_parquet(tmp_path / "fund_nav_df.parquet", index=False)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_product_detail("000001.OF", kind="fund")

    assert response["metrics"]["current_size"] is None
    assert response["metrics"]["current_size_as_of"] is None
    assert response["metrics"]["current_size_source"] is None


def test_product_detail_returns_empty_timeseries_when_real_nav_is_missing(monkeypatch, tmp_path: Path) -> None:
    _write_info_files(tmp_path)
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(instrument_routes, "INSTRUMENT_FILES", {
        "etf": tmp_path / "etf_info_df.parquet",
        "fund": tmp_path / "fund_info_df.parquet",
    })

    response = instrument_routes.instrument_product_detail("510050.SH", kind="etf")

    assert response["timeseries"] == []


def test_rankings_api_rejects_active_only_false_without_reading_data() -> None:
    app = FastAPI()
    app.include_router(instrument_routes.router)

    response = TestClient(app).get("/api/instruments/analytics/rankings?active_only=false")

    assert response.status_code == 400
    assert "status_code=L" in response.json()["detail"]


def test_instrument_json_helpers_drop_non_finite_values() -> None:
    assert instrument_routes._serialize(np.inf) is None
    assert instrument_routes._serialize(-np.inf) is None
    assert instrument_routes._safe_stat(pd.Series([1.0, np.inf]), "sum") == 1.0


def _seed_research_series(data_dir: Path) -> None:
    dates = pd.bdate_range("2014-12-24", periods=10)
    pd.DataFrame(
        [
            {"ts_code": "510300.SH", "date": date, "adj_nav": 1.0 + index * 0.01}
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "etf_daily_df.parquet", index=False)


def test_product_research_series_stops_at_the_platform_research_day(monkeypatch, tmp_path: Path) -> None:
    """The whole point: a 2014 research day must not read 2015 rows.

    Regression guard for 产品研究 showing every row on disk while the header
    badge claimed a research day — the page never asked for the口径, so the
    cut has to live in the loader every panel shares.
    """

    from backend.pit.settings import PitSettingsRepository

    _seed_research_series(tmp_path)
    monkeypatch.setenv("TUSHARE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)

    points, context, future = instrument_routes._load_product_research_points("etf", "510300.SH", "adjusted_nav")
    assert points[-1]["date"] == "2015-01-06"
    assert context["asOf"] is None
    # No research day means no "after", so 未来模拟 has nothing to look back at.
    assert future == []

    PitSettingsRepository(tmp_path).update(None, "RESEARCH", as_of="2014-12-31")
    cut, cut_context, cut_future = instrument_routes._load_product_research_points("etf", "510300.SH", "adjusted_nav")
    assert cut[-1]["date"] == "2014-12-31"
    assert cut_context["asOf"] == "2014-12-31"
    assert any("2014-12-31" in warning for warning in cut_context["warnings"])
    # The rows after the研究日 come back in their own lane so 未来模拟 can be
    # scored against them — and only there, never as research input.
    assert [point["date"] for point in cut_future] == ["2015-01-01", "2015-01-02", "2015-01-05", "2015-01-06"]
    assert not any(point["date"] in {item["date"] for item in cut} for point in cut_future)

    PitSettingsRepository(tmp_path).update(None, "RESEARCH", as_of="2010-01-01")
    with pytest.raises(ValueError, match="2010-01-01"):
        instrument_routes._load_product_research_points("etf", "510300.SH", "adjusted_nav")


def test_current_size_moves_with_the_research_day(monkeypatch, tmp_path: Path) -> None:
    """规模 is computed (份额 × 单位净值), so it is not a "latest row" attribute.

    The page header used to read the last row of the validated snapshot, which
    has no `as_of` and therefore reported today's size under any research day.
    """

    from backend.pit.settings import PitSettingsRepository

    dates = pd.bdate_range("2014-12-29", periods=4)
    pd.DataFrame(
        [
            {
                "ts_code": "510300.SH",
                "date": date,
                "total_share": 1000.0 + index * 10,
                "nav": 2.0,
                "total_size": (1000.0 + index * 10) * 2.0,
            }
            for index, date in enumerate(dates)
        ]
    ).to_parquet(tmp_path / "etf_share_size_df.parquet", index=False)
    monkeypatch.setenv("TUSHARE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(instrument_routes, "DATA_DIR", tmp_path)

    PitSettingsRepository(tmp_path).update(None, "RESEARCH", as_of="2014-12-31")
    on_31 = instrument_routes._load_current_size("etf", "510300.SH")
    assert on_31["current_size_as_of"] == "2014-12-31"
    assert on_31["current_size"] == pytest.approx(2040.0)
    assert on_31["current_size_source"] == "etf_share_size_pit"

    # A day earlier is a different size, not the same number with a new label.
    PitSettingsRepository(tmp_path).update(None, "RESEARCH", as_of="2014-12-30")
    on_30 = instrument_routes._load_current_size("etf", "510300.SH")
    assert on_30["current_size_as_of"] == "2014-12-30"
    assert on_30["current_size"] == pytest.approx(2020.0)

    # Before the series begins there is nothing to report, and nothing invented.
    PitSettingsRepository(tmp_path).update(None, "RESEARCH", as_of="2010-01-01")
    assert instrument_routes._load_current_size("etf", "510300.SH")["current_size"] is None
