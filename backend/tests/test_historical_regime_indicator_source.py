from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from custom_indicators.service import CustomIndicatorService
from historical_regimes.service import HistoricalRegimeService
from services import historical_regime_routes


def _write_etf_data(data_dir: Path, periods: int = 180) -> None:
    dates = pd.bdate_range("2020-01-02", periods=periods)
    returns = np.r_[
        np.full(periods // 3, 0.006),
        np.full(periods // 3, -0.005),
        np.full(periods - 2 * (periods // 3), 0.001),
    ]
    nav = np.cumprod(1.0 + returns)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        {
            "ts_code": "510050.SH",
            "name": "上证50ETF",
            "date": dates,
            "ann_date": dates,
            "adj_nav": nav,
        }
    ).to_parquet(data_dir / "etf_daily_df.parquet", index=False)


def _indicator_draft(name: str, expression: str) -> dict[str, object]:
    return {
        "name": name,
        "description": "历史情景逐期指标测试",
        "expression": expression,
        "periods": ["1W"],
        "unit": "",
        "display_format": "number",
        "precision": 6,
        "direction": "higher_better",
        "indicator_type": "other",
        "annual_risk_free_rate_percent": 0.0,
    }


def _regime_definition(indicator: dict[str, object]) -> dict[str, object]:
    return {
        "name": "版本化指标历史识别",
        "description": "锁定指标中心的不可变修订版。",
        "template_id": "custom-indicator-regime",
        "target": {
            "kind": "indicator",
            "indicator_id": indicator["id"],
            "indicator_revision": indicator["revision"],
            "product_kind": "etf",
            "product_id": "510050.SH",
            "period": "1W",
            "availability_mode": "point_in_time",
        },
        "features": {
            "transform": "identity",
            "filter": "ema",
            "window": 3,
            "slope_window": 1,
            "volatility_window": 5,
        },
        "algorithm": {
            "family": "causal_filter",
            "parameters": {
                "bull_enter": 0.0001,
                "bull_exit": 0.0,
                "bear_enter": -0.0001,
                "bear_exit": 0.0,
                "confirmation": 1,
                "min_duration": 1,
            },
        },
        "states": [],
        "validation": {"walk_forward": False, "folds": 2},
        "usage_intent": "taa",
    }


def _services(tmp_path: Path, periods: int = 180):
    _write_etf_data(tmp_path, periods)
    indicators = CustomIndicatorService(tmp_path, tmp_path)
    regimes = HistoricalRegimeService(
        tmp_path,
        tmp_path,
        indicator_service=indicators,
    )
    return indicators, regimes


def test_exact_typed_indicator_revision_executes_as_audited_series(
    tmp_path: Path,
) -> None:
    indicators, regimes = _services(tmp_path)
    first = indicators.create_indicator(
        _indicator_draft("滚动平均收益", "mean(returns)")
    )
    second = indicators.update_indicator(
        str(first["id"]),
        int(first["revision"]),
        _indicator_draft("滚动平均收益反向", "-mean(returns)"),
    )
    indicators.warm_indicator_revision(str(first["id"]), 1)
    indicators.warm_indicator_revision(str(first["id"]), 2)

    first_definition = regimes.create_definition(_regime_definition(first))
    second_draft = _regime_definition(second)
    second_draft["name"] = "版本化指标历史识别 R2"
    second_definition = regimes.create_definition(second_draft)
    first_run = regimes.run(
        {"id": first_definition["id"], "revision": 1},
        "realtime",
    )
    second_run = regimes.run(
        {"id": second_definition["id"], "revision": 1},
        "realtime",
    )

    snapshot = first_run["data_snapshot"]
    assert snapshot["kind"] == "indicator"
    assert snapshot["indicator_id"] == first["id"]
    assert snapshot["indicator_revision"] == 1
    assert snapshot["indicator_definition_hash"]
    assert snapshot["series_hash"]
    assert snapshot["product_data"]["fingerprint"]
    assert snapshot["plan"]["compile_status"] == "compiled"
    assert snapshot["plan"]["compiled_signatures"]
    assert snapshot["plan"]["njit_required"] is True
    assert snapshot["plan"]["python_fallback"] == 0
    assert snapshot["plan"]["python_operator_calls"] == 0
    assert snapshot["typed_ast"]["nodes"]
    assert snapshot["typed_ast"]["edges"]
    assert snapshot["dag"]["roots"]["result"] == snapshot["typed_ast"]["root"]
    assert first_run["calculation_audits"][0]["source_kind"] == "indicator"
    assert first_run["diagnostics"][-1]["code"] == "NJIT_INDICATOR_VERSION_EXECUTED"
    assert any(point["value"] is None for point in first_run["series"])
    assert all(
        point["value"] is None or point["value"] != 0.0
        for point in first_run["series"][:5]
    )
    first_values = [point["value"] for point in first_run["series"]]
    second_values = [point["value"] for point in second_run["series"]]
    assert first_values != second_values
    assert snapshot["indicator_definition_hash"] != second_run["data_snapshot"][
        "indicator_definition_hash"
    ]


def test_all_indicator_revisions_are_in_startup_warmup(tmp_path: Path) -> None:
    indicators, _ = _services(tmp_path)
    first = indicators.create_indicator(
        _indicator_draft("第一版", "mean(returns)")
    )
    indicators.update_indicator(
        str(first["id"]),
        1,
        _indicator_draft("第二版", "-mean(returns)"),
    )

    revisions = [
        item
        for item in indicators.indicators.list_all_versions()
        if item.get("id") == first["id"]
    ]
    status = indicators.warm_numba_plans()

    assert {item["revision"] for item in revisions} == {1, 2}
    assert (
        status["indicator_plans"] + status["time_series_plans"]
        >= len(indicators.indicators.list()) + 1
    )


def test_unwarmed_and_legacy_indicator_versions_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    indicators, regimes = _services(tmp_path)
    typed = indicators.create_indicator(
        _indicator_draft("未预热模拟", "mean(returns)")
    )
    definition = regimes.create_definition(_regime_definition(typed))
    monkeypatch.setattr(
        "custom_indicators.service.get_cached_numba_plan",
        lambda _plan: None,
    )
    with pytest.raises(ValidationError) as unwarmed:
        regimes.run({"id": definition["id"], "revision": 1}, "realtime")
    assert unwarmed.value.code == "INDICATOR_REVISION_NOT_WARMED"

    legacy = indicators.get_indicator("builtin-cumulative-return")
    with pytest.raises(ValidationError) as unsupported:
        regimes.create_definition(_regime_definition(legacy))
    assert unsupported.value.code == "INDICATOR_NJIT_REQUIRED"


def test_historical_indicator_series_exceeds_old_500_point_limit(
    tmp_path: Path,
) -> None:
    indicators, _ = _services(tmp_path, periods=5000)
    definition = indicators.create_indicator(
        _indicator_draft("长序列", "mean(returns)")
    )
    indicators.warm_indicator_revision(str(definition["id"]), 1)

    result = indicators.evaluate_historical_series(
        indicator_id=str(definition["id"]),
        indicator_revision=1,
        product_kind="etf",
        product_id="510050.SH",
        period="1W",
    )
    repeated = indicators.evaluate_historical_series(
        indicator_id=str(definition["id"]),
        indicator_revision=1,
        product_kind="etf",
        product_id="510050.SH",
        period="1W",
    )

    assert len(result["series"]) == 5000
    assert result["snapshot"]["series_limit"] == 5000
    assert result["snapshot"]["plan"]["python_fallback"] == 0
    assert result["snapshot"]["plan"]["python_operator_calls"] == 0
    assert repeated["snapshot"]["plan"]["compiled_plan_id"] == result[
        "snapshot"
    ]["plan"]["compiled_plan_id"]
    assert repeated["snapshot"]["series_hash"] == result["snapshot"][
        "series_hash"
    ]
    assert result["series"][0]["value"] is None
    assert result["series"][0]["status"] == "unavailable"


def test_indicator_target_route_meta_and_run_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    indicators, regimes = _services(tmp_path)
    indicator = indicators.create_indicator(
        _indicator_draft("API 指标", "mean(returns)")
    )
    indicators.warm_indicator_revision(str(indicator["id"]), 1)
    monkeypatch.setattr(
        historical_regime_routes,
        "historical_regime_service",
        regimes,
    )
    app = FastAPI()
    app.include_router(historical_regime_routes.router)
    client = TestClient(app)

    meta = client.get("/api/historical-regimes/meta")
    assert meta.status_code == 200
    body = meta.json()
    assert "indicator" in {item["id"] for item in body["data_sources"]}
    catalog_item = next(
        item for item in body["indicator_catalog"] if item["id"] == indicator["id"]
    )
    assert catalog_item["revision"] == 1
    assert catalog_item["njit_required"] is True
    assert any(item["value"] == "1W" for item in body["indicator_periods"])
    assert all(
        item["njit_supported"] is True
        and item["execution_backend"] == "numba_njit_fixed_signature"
        for item in body["formula_language"]["functions"]
    )

    draft = _regime_definition(indicator)
    blocked_definition = client.post(
        "/api/historical-regimes/definitions", json=draft
    )
    assert blocked_definition.status_code == 409
    assert (
        blocked_definition.json()["detail"]["code"]
        == "V1_DEFINITION_READ_ONLY"
    )

    saved = regimes.create_definition(draft)
    seeded_run = regimes.run(
        {"id": saved["id"], "revision": saved["revision"]},
        "realtime",
    )
    executed = client.get(f"/api/historical-regimes/runs/{seeded_run['id']}")
    assert executed.status_code == 200, executed.text
    run = executed.json()
    assert run == seeded_run
    assert run["target"]["kind"] == "indicator"
    assert run["data_snapshot"]["indicator_revision"] == 1
    assert run["calculation_audit"]["plan"]["python_fallback"] == 0


def test_indicator_reference_is_immutable_when_current_version_changes(
    tmp_path: Path,
) -> None:
    indicators, regimes = _services(tmp_path)
    first = indicators.create_indicator(
        _indicator_draft("精确旧版", "mean(returns)")
    )
    saved_definition = regimes.create_definition(_regime_definition(first))
    before = copy.deepcopy(regimes.get_definition(str(saved_definition["id"]), 1))
    indicators.update_indicator(
        str(first["id"]),
        1,
        _indicator_draft("当前新版", "-mean(returns)"),
    )
    indicators.warm_numba_plans()

    run = regimes.run(
        {"id": saved_definition["id"], "revision": 1},
        "realtime",
    )

    assert run["target"]["indicator_revision"] == 1
    assert regimes.get_definition(str(saved_definition["id"]), 1) == before
    assert run["data_snapshot"]["indicator_revision"] == 1
