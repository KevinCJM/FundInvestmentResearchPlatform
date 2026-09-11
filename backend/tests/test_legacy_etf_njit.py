from __future__ import annotations

import json
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

from backend import app as app_module  # noqa: E402
from backend import instrument_analytics_numba  # noqa: E402


def _payload(response) -> dict:
    return json.loads(response.body)


def _write_legacy_fixture(data_dir: Path) -> None:
    pd.DataFrame(
        [
            {
                "ts_code": "510001.SH",
                "code": "510001",
                "name": "产品甲",
                "management": "管理人甲",
                "custodian": "托管行甲",
                "fund_type": "股票型",
                "type": "公募基金",
                "invest_type": "被动指数型",
                "market": "上交所",
                "status": "上市交易",
                "status_code": "L",
                "issue_amount": 100.0,
                "m_fee": 0.5,
                "c_fee": 0.1,
                "exp_return": 8.0,
                "duration_year": 5.0,
                "list_date": pd.Timestamp("2020-01-02"),
                "found_date": pd.Timestamp("2019-12-01"),
            },
            {
                "ts_code": "510002.SH",
                "code": "510002",
                "name": "产品乙",
                "management": "管理人甲",
                "custodian": "托管行甲",
                "fund_type": "债券型",
                "type": "公募基金",
                "invest_type": "主动管理型",
                "market": "深交所",
                "status": "摘牌",
                "status_code": "D",
                "issue_amount": 200.0,
                "m_fee": 0.7,
                "c_fee": 0.2,
                "exp_return": 4.0,
                "duration_year": 3.0,
                "list_date": pd.Timestamp("2021-03-04"),
                "found_date": pd.Timestamp("2021-02-01"),
            },
            {
                "ts_code": "510003.SH",
                "code": "510003",
                "name": "产品丙",
                "management": "管理人乙",
                "custodian": "托管行乙",
                "fund_type": "股票型",
                "type": "公募基金",
                "invest_type": "被动指数型",
                "market": "上交所",
                "status": "上市交易",
                "status_code": "L",
                "issue_amount": 300.0,
                "m_fee": 1.0,
                "c_fee": 0.15,
                "exp_return": 10.0,
                "duration_year": 2.0,
                "list_date": pd.Timestamp("2022-05-06"),
                "found_date": pd.Timestamp("2022-04-01"),
            },
        ]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "510001.SH",
                "date": pd.Timestamp("2024-01-02"),
                "open": np.nan,
                "high": 1.02,
                "low": 0.98,
                "close": 1.0,
                "vol": np.nan,
            },
            {
                "ts_code": "510001.SH",
                "date": pd.Timestamp("2024-01-03"),
                "open": 1.05,
                "high": 1.12,
                "low": 1.0,
                "close": 1.1,
                "vol": 100.0,
            },
        ]
    ).to_parquet(data_dir / "etf_daily_candle_df.parquet", index=False)


def _configure_modules(monkeypatch, data_dir: Path):
    analytics_module = sys.modules[app_module.build_legacy_etf_analytics_response.__module__]
    routes_module = sys.modules[app_module.unified_instrument_products.__module__]
    monkeypatch.setattr(app_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(routes_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(
        routes_module,
        "INSTRUMENT_FILES",
        {
            "etf": data_dir / "etf_info_df.parquet",
            "fund": data_dir / "fund_info_df.parquet",
        },
    )
    analytics_module._read_small_parquet_cached.cache_clear()
    return analytics_module


def test_legacy_analytics_and_trend_preserve_contract_on_fixed_njit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_legacy_fixture(tmp_path)
    analytics_module = _configure_modules(monkeypatch, tmp_path)
    signatures_before = instrument_analytics_numba.instrument_analytics_numba_execution_audit()[
        "kernel_signatures"
    ]

    analytics = _payload(app_module.etf_analytics())
    trend = _payload(app_module.etf_list_trend("management", ["管理人甲"]))

    assert analytics["summary"]["total_count"] == 3
    assert analytics["summary"]["active_count"] == 2
    assert analytics["summary"]["unique_managements"] == 2
    assert analytics["summary"]["total_issue_amount"] == 600.0
    assert analytics["summary"]["avg_m_fee"] == pytest.approx(2.2 / 3.0)
    assert analytics["summary"]["avg_c_fee"] == pytest.approx(0.15)
    assert analytics["summary"]["avg_exp_return"] == pytest.approx(22.0 / 3.0)
    assert analytics["summary"]["avg_duration_year"] == pytest.approx(10.0 / 3.0)
    assert analytics["top_management"][0] == {
        "name": "管理人甲",
        "count": 2,
        "total_issue_amount": 300.0,
    }
    assert analytics["market_issue_summary"][0]["market"] == "上交所"
    assert analytics["market_issue_summary"][0]["total_issue_amount"] == 400.0
    assert analytics["fee_by_fund_type"][0]["fund_type"] == "股票型"
    assert analytics["fee_by_fund_type"][0]["avg_m_fee"] == 0.75
    assert analytics["top_issue_amount"][0]["ts_code"] == "510003.SH"
    assert trend["dimension"] == "management"
    assert [item["year"] for item in trend["list_trend"]] == [2020, 2021]
    assert [item["total_issue_amount"] for item in trend["list_trend"]] == [100.0, 200.0]
    for payload in (analytics, trend):
        assert payload["execution"]["execution_backend"] == "numba_njit_fixed_signature"
        assert payload["execution"]["nopython"] is True
        assert payload["execution"]["python_fallback"] == 0
        assert payload["execution"]["request_time_compilation"] == 0
    signatures_after = instrument_analytics_numba.instrument_analytics_numba_execution_audit()[
        "kernel_signatures"
    ]
    assert signatures_after == signatures_before
    analytics_module._read_small_parquet_cached.cache_clear()


def test_legacy_products_delegate_to_unified_njit_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_legacy_fixture(tmp_path)
    _configure_modules(monkeypatch, tmp_path)

    response = _payload(
        app_module.etf_products(
            q="",
            fund_type=None,
            organization_type=None,
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
    )

    assert response["total"] == 3
    assert response["items"][0]["ts_code"] == "510003.SH"
    assert response["summary"]["avg_exp_return"] == 22.0 / 3.0
    assert response["summary"]["avg_duration_year"] == 10.0 / 3.0
    assert response["summary"]["median_issue_amount"] == 200.0
    assert response["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert response["execution"]["python_fallback"] == 0


def test_legacy_detail_uses_real_points_and_never_synthesizes_missing_prices(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_legacy_fixture(tmp_path)
    _configure_modules(monkeypatch, tmp_path)

    real = _payload(app_module.etf_product_detail("510001.SH"))
    missing = _payload(app_module.etf_product_detail("510002.SH"))

    assert len(real["timeseries"]) == 2
    assert real["timeseries"][0]["close"] == 1.0
    assert real["timeseries"][0]["open"] is None
    assert real["timeseries"][0]["volume"] is None
    assert missing["timeseries"] == []
    assert real["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert real["execution"]["python_fallback"] == 0
