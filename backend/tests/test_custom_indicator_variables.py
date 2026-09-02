from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.series_provider import (
    InstrumentIdentity,
    ProductVariableSeries,
    load_product_variable_series,
    select_variable_window_fast,
    select_variable_window,
)
from custom_indicators.service import CustomIndicatorService, SUPPORTED_PERIODS
from custom_indicators.errors import ValidationError
from custom_indicators.variable_registry import variable_catalog
from services import custom_indicator_routes


def _write_variable_data(data_dir: Path) -> None:
    dates = pd.bdate_range("2026-01-02", periods=10)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    nav_rows = []
    candle_rows = []
    for index, date in enumerate(dates):
        nav_rows.append(
            {
                "ts_code": "510050.SH",
                "date": date,
                "ann_date": date + pd.Timedelta(days=1),
                "adj_nav": 1.0 + index * 0.01,
                "unit_nav": 0.9 + index * 0.01,
                "accum_nav": 1.0 + index * 0.01,
                "accum_div": 0.1 if index == 3 else None,
            }
        )
        candle_rows.append(
            {
                "ts_code": "510050.SH",
                "date": date,
                "open": 2.0 + index * 0.01,
                "high": 2.1 + index * 0.01,
                "low": 1.9 + index * 0.01,
                "close": 2.05 + index * 0.01,
                "pre_close": 2.0 + index * 0.01,
                "change": 0.05,
                "pct_chg": 2.5,
                "vol": None if index == 2 else 1000.0 + index,
                "amount": 2000.0 + index,
            }
        )
    # The last NAV observation is not yet public at this cutoff.
    nav_rows[-1]["ann_date"] = pd.Timestamp("2026-02-01")
    pd.DataFrame(nav_rows).to_parquet(data_dir / "etf_daily_df.parquet", index=False)
    pd.DataFrame(candle_rows).to_parquet(
        data_dir / "etf_daily_candle_df.parquet", index=False
    )


def _draft(**overrides):
    payload = {
        "name": "均值指标",
        "description": "",
        "expression": "mean(returns)",
        "unit": "%",
        "display_format": "percent",
        "precision": 2,
        "direction": "higher_better",
        "annual_risk_free_rate_percent": 1.5,
    }
    payload.update(overrides)
    return payload


def test_dependency_provider_enforces_announcement_time_and_inner_join(
    tmp_path: Path,
) -> None:
    _write_variable_data(tmp_path)

    source = load_product_variable_series(
        "etf",
        "510050",
        ["adjusted_nav", "market_close", "price_return", "volume"],
        tmp_path,
        "2026-01-31",
    )

    assert source is not None
    assert len(source.frame) == 8  # one unannounced NAV row and one missing volume row
    assert source.frame["price_return"].eq(0.025).all()
    assert source.frame["volume"].isna().sum() == 0
    assert {item["availability_field"] for item in source.lineage} == {
        "ann_date",
        "date",
    }
    assert set(source.fingerprints) == {"nav", "candle"}

    window = select_variable_window(source, "1W", "2026-01-31")
    assert window.observation_count == 5
    assert window.context["market_close"].shape == (6,)
    assert window.context["returns"].shape == (5,)
    assert window.context["window_elapsed_days"] > 0
    assert window.common_date_hash


def test_period_policy_and_exact_scalar_builtin_catalog(tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    omitted = service.create_indicator(_draft())
    empty = service.create_indicator(_draft(name="空周期字段", periods=[]))

    for definition in (omitted, empty):
        assert definition["period_policy"] == "all_supported"
        assert definition["periods"] == list(SUPPORTED_PERIODS)
        assert definition["dsl_version"] == "2.2.0"
        assert definition["numeric_kernel_version"] == "2.2.0"
        assert definition["required_variables"] == ["returns"]

    typed_builtins = [
        item
        for item in service.list_indicators()["items"]
        if item.get("source") == "built_in" and item.get("dsl_version") == "2.2.0"
    ]
    assert len(typed_builtins) == 35
    assert sum(item["applicable_product_kinds"] == ["etf"] for item in typed_builtins) == 5
    for definition in typed_builtins:
        result = service.validate(definition)
        assert result["valid"], (definition["id"], result["diagnostics"])
        assert result["dag"]["roots"].keys() == {"result"}
        assert definition["required_variables"]
        assert definition["methodology"]
        assert definition["formula_version"] == "2.2.0"
        assert definition["minimum_observations"] >= 1
        assert definition["semantic_differences"]

    measures = {item["id"]: item["output_measure"] for item in typed_builtins}
    assert measures["builtin-positive-return-ratio-v2"] == "dimensionless"
    assert measures["builtin-new-high-ratio-v2"] == "dimensionless"


def test_period_catalog_exposes_rolling_calendar_and_lifetime_semantics(
    tmp_path: Path,
) -> None:
    metadata = CustomIndicatorService(tmp_path, tmp_path).meta()
    periods = {item["value"]: item for item in metadata["periods"]}

    assert periods["1Y"]["label"] == "近 1 年"
    assert periods["1Y"]["kind"] == "rolling"
    assert periods["Y1"]["label"] == "去年"
    assert periods["Y2"]["label"] == "前年"
    assert periods["W1"]["label"] == "上周"
    assert periods["W2"]["label"] == "上上周"
    assert periods["ALL"]["label"] == "成立以来"
    assert periods["ALL"]["kind"] == "lifetime"


def test_fast_window_uses_same_calendar_and_lifetime_boundaries() -> None:
    dates = pd.bdate_range("2024-12-31", "2026-08-31")
    adjusted_nav = 1.0 + pd.Series(range(len(dates)), dtype=float).to_numpy() * 0.0001
    source = ProductVariableSeries(
        identity=InstrumentIdentity("etf", "TEST.SH", "TEST.SH", "测试 ETF"),
        frame=pd.DataFrame({"date": dates, "adjusted_nav": adjusted_nav}),
        fingerprints={"nav": "fixture"},
        fingerprint="fixture",
        data_latest_date="2026-08-31",
        requested_variables=("returns",),
    )

    last_year = select_variable_window_fast(source, "Y1", "2026-08-31")
    lifetime = select_variable_window_fast(source, "ALL", "2026-08-31")

    assert last_year.start_date == "2024-12-31"
    assert last_year.end_date == "2025-12-31"
    assert lifetime.start_date == "2024-12-31"
    assert lifetime.end_date == "2026-08-31"


def test_variable_catalog_exposes_product_contract_fields() -> None:
    catalog = {item["id"]: item for item in variable_catalog("single_product")}

    assert catalog["price_change"]["type"]["semantic_dimension"] == "raw_market_price"
    assert catalog["market_close"]["type"]["semantic_dimension"] == "raw_market_price"
    assert catalog["previous_close"]["type"]["semantic_dimension"] == "raw_market_price"
    assert catalog["risk_free_rate_per_observation"]["structural_type"] == "scalar"
    assert "risk_free_rate_per_period" in catalog["risk_free_rate_per_observation"]["aliases"]
    for item in catalog.values():
        assert {
            "structural_type",
            "context_kinds",
            "canonical_field",
            "source_bindings",
            "availability_policy",
            "measure",
            "scale",
            "timing",
            "availability_tier",
            "missing_policy",
        } <= item.keys()
    assert catalog["adjusted_nav"]["source_bindings"]["etf"]["configured"] is True
    assert catalog["adjusted_nav"]["source_bindings"]["fund"]["configured"] is True
    assert catalog["market_high"]["source_bindings"]["etf"]["source_field"] == "high"
    assert catalog["market_high"]["source_bindings"]["fund"]["configured"] is False
    assert catalog["market_high"]["alternative_variables"] == [
        "adjusted_nav",
        "returns",
    ]


def test_meta_exposes_versioned_operator_contract(tmp_path: Path) -> None:
    meta = CustomIndicatorService(tmp_path, tmp_path).meta()
    assert meta["templates"] == []
    assert meta["predefined_calculations"] == []
    operators = {item["id"]: item for item in meta["operators"]}
    clip = operators["clip"]
    assert clip["label"] == "逐元素数值限幅"
    assert {
        "family",
        "parameters",
        "type_rules",
        "semantic_rules",
        "output_rule",
        "latex_alias",
        "examples",
        "cost",
    } <= clip.keys()
    assert operators["count_true"]["parameters"][0]["allowed_shapes"] == ["mask"]
    assert operators["where"]["parameters"][0]["allowed_shapes"] == ["mask"]
    assert [item["name"] for item in operators["linear_slope"]["parameters"]] == ["values"]
    assert {item["arity"] for item in operators["linear_slope"]["parameter_sets"]} == {1, 2}
    assert [item["name"] for item in operators["covariance"]["parameters"]] == ["asset_returns"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("variable_registry_version", "future-variables"),
        ("data_contract_version", "future-data"),
        ("context_schema_version", "future-context"),
    ],
)
def test_typed_21_definition_rejects_uninstalled_protocol_versions(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)

    with pytest.raises(ValidationError) as caught:
        service.create_indicator(_draft(**{field: value}))

    assert caught.value.code == f"UNSUPPORTED_{field.upper()}"


def test_variable_availability_route_reports_partial_fields(
    monkeypatch, tmp_path: Path
) -> None:
    _write_variable_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)

    response = client.post(
        "/api/custom-indicators/variables/availability",
        json={
            "kind": "etf",
            "product_id": "510050",
            "variable_ids": ["adjusted_nav", "accumulated_dividend", "volume"],
            "period": "1W",
            "as_of": "2026-01-31",
        },
    )

    assert response.status_code == 200
    body = response.json()
    statuses = {item["variable_id"]: item["status"] for item in body["items"]}
    assert statuses["adjusted_nav"] == "available"
    assert statuses["accumulated_dividend"] == "partial"
    assert statuses["volume"] == "partial"
    shapes = {item["variable_id"]: item["actual_shape"] for item in body["items"]}
    assert shapes["adjusted_nav"] == [6]
    assert shapes["volume"] == [6]
    assert body["period"] == "1W"
    windows = {item["variable_id"]: item["window"] for item in body["items"]}
    assert windows["adjusted_nav"]["observation_count"] == 5
    assert windows["volume"]["observation_count"] == 5
    assert body["variable_registry_version"] == "2.1.0"
    assert body["source_fingerprints"]


def test_typed_evaluation_uses_requested_market_variable(tmp_path: Path) -> None:
    _write_variable_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)

    response = service.evaluate(
        indicator_ids=["builtin-average-volume-v2"],
        inline_definition=None,
        targets=[{"kind": "etf", "product_id": "510050"}],
        period="1W",
        as_of="2026-01-31",
    )

    result = response["results"][0]
    assert result["value"] is not None
    assert result["status"] == "warning"
    assert result["warnings"][0]["code"] == "INPUT_COVERAGE_PARTIAL"
    assert result["input_requirements"]["status"] == "partial"
    assert result["input_requirements"]["partial_inputs"][0]["label"] == "成交量"
    assert result["window"]["common_date_hash"]
    assert {item["dataset"] for item in result["window"]["data_lineage"]} == {
        "etf_daily_df.parquet",
        "etf_daily_candle_df.parquet",
    }


def test_historical_as_of_requires_nav_announcement_date(tmp_path: Path) -> None:
    _write_variable_data(tmp_path)
    nav_path = tmp_path / "etf_daily_df.parquet"
    nav = pd.read_parquet(nav_path).drop(columns=["ann_date"])
    nav.to_parquet(nav_path, index=False)
    service = CustomIndicatorService(tmp_path, tmp_path)

    response = service.evaluate(
        indicator_ids=["builtin-total-return-v2"],
        inline_definition=None,
        targets=[{"kind": "etf", "product_id": "510050"}],
        period="1W",
        as_of="2026-01-31",
    )

    result = response["results"][0]
    assert result["value"] is None
    assert result["status"] == "warning"
    assert result["warnings"][0]["code"] == "ANN_DATE_UNAVAILABLE"
