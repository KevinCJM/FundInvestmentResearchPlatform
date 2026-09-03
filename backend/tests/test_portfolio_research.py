from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ConflictError, ValidationError
from custom_indicators.portfolio_service import PortfolioResearchService
from custom_indicators.service import CustomIndicatorService
from services import portfolio_routes


def _write_market_data(data_dir: Path, periods: int = 90) -> None:
    dates = pd.bdate_range("2025-01-02", periods=periods)
    pd.DataFrame(
        [
            {"ts_code": "510001.SH", "code": "510001", "name": "测试 ETF A"},
            {"ts_code": "510002.SH", "code": "510002", "name": "测试 ETF B"},
        ]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    rows = []
    for index, date in enumerate(dates):
        rows.append(
            {
                "ts_code": "510001.SH",
                "date": date,
                "adj_nav": 1.0 * (1.002 ** index),
                "name": "测试 ETF A",
            }
        )
        # B deliberately misses every ninth real date. Strict intersection must
        # remove the date instead of forward-filling it.
        if index % 9:
            rows.append(
                {
                    "ts_code": "510002.SH",
                    "date": date,
                    "adj_nav": 2.0 * (1.001 ** index),
                    "name": "测试 ETF B",
                }
            )
    pd.DataFrame(rows).to_parquet(data_dir / "etf_daily_df.parquet", index=False)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "code": "000001", "name": "测试公募基金"}]
    ).to_parquet(data_dir / "fund_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "date": date,
                "adj_nav": 1.5 * (1.0015 ** index),
                "name": "测试公募基金",
            }
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "fund_nav_df.parquet", index=False)
    # Typed-v2 must not use this deliberately contradictory exchange close.
    pd.DataFrame(
        [
            {"ts_code": code, "date": date, "close": 100.0 - index, "name": code}
            for code in ("510001.SH", "510002.SH")
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "etf_daily_candle_df.parquet", index=False)


def _definition(strategy: dict | None = None, rebalance: dict | None = None) -> dict:
    return {
        "components": [
            {"kind": "etf", "product_id": "510001", "name": "测试 ETF A"},
            {"kind": "etf", "product_id": "510002", "name": "测试 ETF B"},
        ],
        "strategy": strategy or {"type": "equal_weight"},
        "constraints": {"min_weight": 0.0, "max_weight": 1.0},
        "rebalance": rebalance or {"enabled": False},
        "alignment": "strict_intersection",
    }


def _service(tmp_path: Path) -> PortfolioResearchService:
    _write_market_data(tmp_path)
    return PortfolioResearchService(tmp_path, tmp_path)


def test_target_repository_versions_restart_and_reference_protection(tmp_path: Path) -> None:
    service = _service(tmp_path)
    created = service.create_target({"name": "组合 A", "definition": _definition()})
    updated = service.update_target(
        created["id"],
        1,
        {"name": "组合 A v2", "definition": _definition()},
    )
    assert updated["revision"] == 2
    assert service.targets.get(created["id"], 1)["name"] == "组合 A"
    with pytest.raises(ConflictError, match="刷新后重试"):
        service.update_target(created["id"], 1, {"name": "冲突", "definition": _definition()})

    restarted = PortfolioResearchService(tmp_path, tmp_path)
    assert restarted.get_target(created["id"])["revision"] == 2
    run = restarted.run_target(created["id"])
    assert restarted.get_run(run["id"])["immutable"] is True
    with pytest.raises(ConflictError, match="运行快照引用"):
        restarted.delete_target(created["id"], 2)


def test_strict_intersection_adjusted_nav_and_next_day_weight_contract(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = service.create_target(
        {
            "name": "严格交集",
            "definition": _definition(
                {"type": "manual", "weights": [0.7, 0.3]},
                {"enabled": True, "mode": "monthly", "which": "first"},
            ),
        }
    )
    run = service.run_target(target["id"])

    source_dates = pd.bdate_range("2025-01-02", periods=90)
    expected_common_points = sum(1 for index in range(len(source_dates)) if index % 9)
    assert run["observation_count"] == expected_common_points - 1
    assert run["summary"]["cumulative_return"] > 0  # adjusted NAV rises; close-price fixture falls
    assert run["weight_path"][0]["decision_date"] < run["weight_path"][0]["effective_date"]
    assert list(run["weight_path"][0]["weights"]) == run["asset_order"]
    assert run["common_date_hash"]
    assert len(run["daily_weights"]) == run["observation_count"]
    daily_weights = np.asarray(run["daily_weights"], dtype=np.float64)
    asset_returns = np.asarray(run["asset_returns"], dtype=np.float64)
    portfolio_returns = np.asarray(run["portfolio_returns"], dtype=np.float64)
    assert np.allclose(
        portfolio_returns,
        np.sum(daily_weights * asset_returns, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.any(np.abs(daily_weights[1:] - daily_weights[:-1]) > 1e-12)

    diagnosis = service.diagnose(run["id"])
    assert diagnosis["correlation"]["labels"] == run["asset_order"]
    assert len(diagnosis["risk_contributions"]) == 2
    assert diagnosis["concentration_summary"]["effective_holdings"] > 1
    assert diagnosis["contribution_series"][0]["date"] == run["dates"][0]

    cached = service.run_target(target["id"])
    assert cached["cache"]["hit"] is True
    _write_market_data(tmp_path, periods=91)
    refreshed = service.run_target(target["id"])
    assert refreshed["cache"]["hit"] is False
    assert refreshed["data_fingerprints"] != run["data_fingerprints"]


def test_manual_weights_are_never_silently_normalized(tmp_path: Path) -> None:
    service = _service(tmp_path)
    with pytest.raises(ValidationError) as error:
        service.create_target(
            {
                "name": "错误权重",
                "definition": _definition({"type": "manual", "weights": [60, 40]}),
            }
        )
    assert error.value.code == "WEIGHTS_NOT_NORMALIZED"


def test_etf_and_fund_can_share_one_strict_portfolio_snapshot(tmp_path: Path) -> None:
    service = _service(tmp_path)
    definition = _definition()
    definition["components"] = [
        {"kind": "etf", "product_id": "510001", "name": "测试 ETF A"},
        {"kind": "fund", "product_id": "000001.OF", "name": "测试公募基金"},
    ]
    target = service.create_target({"name": "ETF 基金混合", "definition": definition})

    run = service.run_target(target["id"])

    assert [item["kind"] for item in run["assets"]] == ["etf", "fund"]
    assert run["observation_count"] == 89
    assert run["summary"]["cumulative_return"] > 0


def test_diagnosis_reuses_saved_dynamic_portfolio_indicator(tmp_path: Path) -> None:
    service = _service(tmp_path)
    target = service.create_target({"name": "动态组合", "definition": _definition()})
    run = service.run_target(target["id"])
    indicator = CustomIndicatorService(tmp_path, tmp_path).create_indicator(
        {
            "name": "动态组合累计收益",
            "description": "按快照每日权重计算。",
            "expression": (
                r"\prod\left(\operatorname{sum_asset}\left(\mathbf{R}\cdot\mathbf{W}\right)+1\right)-1"
            ),
            "periods": ["1Y"],
            "unit": "%",
            "display_format": "percent",
            "precision": 2,
            "direction": "higher_better",
            "annual_risk_free_rate_percent": 0,
            "dsl_version": "2.0.0",
            "operator_registry_version": "2.0.0",
            "context_kind": "portfolio",
            "output_contract": "scalar",
        }
    )

    diagnosis = service.diagnose(run["id"], [indicator["id"]])

    assert diagnosis["custom_indicators"][0]["name"] == "动态组合累计收益"
    assert diagnosis["custom_indicators"][0]["value"] == pytest.approx(
        run["summary"]["cumulative_return"]
    )


@pytest.mark.parametrize(
    "strategy",
    [
        {"type": "risk_budget", "budgets": [0.5, 0.5], "lookback_observations": 20},
        {"type": "target_optimization", "target": "min_volatility", "lookback_observations": 20},
    ],
)
def test_dynamic_strategies_use_history_and_produce_finite_weights(tmp_path: Path, strategy: dict) -> None:
    service = _service(tmp_path)
    target = service.create_target(
        {
            "name": strategy["type"],
            "definition": _definition(strategy, {"enabled": True, "mode": "monthly"}),
        }
    )
    run = service.run_target(target["id"])
    assert run["observation_count"] > 10
    assert run["weight_path"]
    for item in run["weight_path"]:
        weights = list(item["weights"].values())
        assert all(value >= 0 for value in weights)
        assert sum(weights) == pytest.approx(1.0)


def test_routes_crud_run_diagnose_scenario_and_export(monkeypatch, tmp_path: Path) -> None:
    service = _service(tmp_path)
    monkeypatch.setattr(portfolio_routes, "portfolio_service", service)
    app = FastAPI()
    app.include_router(portfolio_routes.router)
    client = TestClient(app)

    created_response = client.post(
        "/api/research-targets",
        json={"name": "API 组合", "kind": "portfolio", "definition": _definition()},
    )
    assert created_response.status_code == 201
    target = created_response.json()
    run_response = client.post(f"/api/research-targets/{target['id']}/run", json={})
    assert run_response.status_code == 201
    run = run_response.json()
    assert client.get("/api/research-targets").json()["items"][0]["id"] == target["id"]
    listing = client.get("/api/portfolio-runs").json()["items"]
    assert listing[0]["id"] == run["id"]
    assert "asset_returns" not in listing[0]

    diagnosis = client.post(f"/api/portfolio-runs/{run['id']}/diagnose", json={})
    assert diagnosis.status_code == 200
    assert diagnosis.json()["components"]
    scenario = client.post(
        f"/api/portfolio-runs/{run['id']}/scenario",
        json={"name": "测试情景", "start_date": "2025-02-01", "end_date": "2025-04-01"},
    )
    assert scenario.status_code == 200
    assert scenario.json()["name"] == "测试情景"
    assert scenario.json()["metrics"]
    exported_csv = client.get(f"/api/portfolio-runs/{run['id']}/export?format=csv&table=correlation")
    exported_zip = client.get(
        f"/api/portfolio-runs/{run['id']}/export"
        "?format=zip&scenario_start=2025-02-01&scenario_end=2025-04-01"
    )
    assert exported_csv.status_code == 200
    assert exported_csv.content.startswith(b"\xef\xbb\xbfasset_key")
    assert exported_zip.status_code == 200
    assert exported_zip.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(exported_zip.content)) as bundle:
        assert {"summary.csv", "scenario-summary.csv", "scenario-series.csv"}.issubset(bundle.namelist())
