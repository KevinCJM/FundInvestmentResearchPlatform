from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.service import CustomIndicatorService
from services import custom_indicator_routes


def _write_etf_data(data_dir: Path) -> None:
    dates = pd.bdate_range("2026-01-02", periods=70)
    pd.DataFrame(
        [{"ts_code": "510050.SH", "code": "510050", "name": "上证50ETF"}]
    ).to_parquet(data_dir / "etf_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "510050.SH",
                "name": "上证50ETF",
                "date": date,
                "adj_nav": 1.0 + index * 0.01,
            }
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "etf_daily_df.parquet", index=False)


def _write_fund_data(data_dir: Path) -> None:
    dates = pd.bdate_range("2026-01-02", periods=70)
    pd.DataFrame(
        [{"ts_code": "000001.OF", "code": "000001", "name": "华夏成长"}]
    ).to_parquet(data_dir / "fund_info_df.parquet", index=False)
    pd.DataFrame(
        [
            {
                "ts_code": "000001.OF",
                "name": "华夏成长",
                "date": date,
                "adj_nav": 2.0 + index * 0.005,
            }
            for index, date in enumerate(dates)
        ]
    ).to_parquet(data_dir / "fund_nav_df.parquet", index=False)


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    service = CustomIndicatorService(tmp_path, tmp_path)
    # Standalone router tests do not run backend.app's lifespan.  Honor the
    # production contract explicitly: plans are compiled before any request.
    service.warm_numba_plans()
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    return TestClient(app)


def _draft(name: str = "API 累计收益") -> dict:
    return {
        "name": name,
        "description": "API 测试",
        "expression": r"\left(\prod\left(\mathbf{r}+1\right)\right)-1",
        "periods": ["1W"],
        "unit": "%",
        "display_format": "percent",
        "precision": 2,
        "direction": "higher_better",
        "annual_risk_free_rate_percent": 1.5,
    }


def test_meta_list_and_interactive_validation_contract(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)

    meta = client.get("/api/custom-indicators/meta")
    listing = client.get("/api/custom-indicators")
    invalid = client.post(
        "/api/custom-indicators/validate",
        json={"name": "非法", "expression": "unknown(returns)", "periods": ["1W"]},
    )

    assert meta.status_code == 200
    assert meta.json()["workspace_scope"] == "shared"
    assert listing.status_code == 200
    visible_items = listing.json()["items"]
    assert listing.json()["total"] == len(visible_items)
    assert all(item.get("result_kind", "scalar") in {"scalar", "time_series"} for item in visible_items)
    assert {item["id"] for item in visible_items} >= {
        "builtin-last-maximum-drawdown-rate", "builtin-maximum-drawdown-start-date",
        "builtin-maximum-drawdown-trough-date", "builtin-maximum-drawdown-duration-days",
        "builtin-maximum-drawdown-recovery-date", "builtin-maximum-drawdown-recovery-days",
        "builtin-maximum-drawdown-total-days",
    }
    assert sum(item["name"] == "累计收益率" for item in listing.json()["items"]) == 1
    assert all(item["catalog_status"] == "current" for item in listing.json()["items"])
    risk_listing = client.get("/api/custom-indicators?indicator_type=risk")
    assert risk_listing.status_code == 200
    assert risk_listing.json()["items"]
    assert all(
        item["indicator_type"] == "risk"
        for item in risk_listing.json()["items"]
    )
    compatibility = client.get("/api/custom-indicators?include_compatibility=true")
    assert compatibility.status_code == 200
    assert compatibility.json()["total"] == len(compatibility.json()["items"])
    assert compatibility.json()["total"] > listing.json()["total"]
    hidden_legacy = next(
        item
        for item in compatibility.json()["items"]
        if item["id"] == "builtin-cumulative-return"
    )
    assert hidden_legacy["catalog_status"] == "compatibility"
    assert hidden_legacy["ui_exposed"] is False
    assert {
        item["name"]
        for item in listing.json()["items"]
        if item.get("context_kind") == "portfolio"
    } == {"组合累计收益率", "组合波动率"}
    assert invalid.status_code == 200
    assert invalid.json()["valid"] is False
    assert invalid.json()["diagnostics"][0]["code"] == "UNKNOWN_FUNCTION"


def test_time_series_builder_requires_fixed_constants_and_runtime_cannot_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = _client(monkeypatch, tmp_path)

    invalid_compose = client.post(
        "/api/custom-indicators/compose",
        json={
            "operator_id": "rolling_mean",
            "context": "single_product",
            "arguments": [
                {"parameter": "values", "source": "variable", "value": "market_close"},
                {"parameter": "window", "source": "variable", "value": "observation_count"},
            ],
        },
    )
    valid_compose = client.post(
        "/api/custom-indicators/compose",
        json={
            "operator_id": "rolling_mean",
            "context": "single_product",
            "arguments": [
                {"parameter": "values", "source": "variable", "value": "market_close"},
                {"parameter": "window", "source": "constant", "value": 20},
            ],
        },
    )
    runtime_override = client.post(
        "/api/custom-indicators/evaluate-series",
        json={
            "indicator_instances": [
                {
                    "indicator_id": "builtin-close-moving-average-series",
                    "parameters": {"window": 5},
                }
            ],
            "target": {"kind": "etf", "product_id": "510050.SH"},
            "period": "ALL",
        },
    )

    assert invalid_compose.status_code == 422
    assert invalid_compose.json()["detail"]["code"] == (
        "SERIES_CONFIGURATION_MUST_BE_CONSTANT"
    )
    assert valid_compose.status_code == 200
    assert "rolling_mean" in valid_compose.json()["normalized_expression"]
    assert "20.0" in valid_compose.json()["normalized_expression"]
    assert "rolling_mean" not in valid_compose.json()["display_latex"]
    assert runtime_override.status_code == 422
    assert runtime_override.json()["detail"]["code"] == (
        "SERIES_PARAMETERS_FIXED_IN_DEFINITION"
    )


def test_legacy_typed_requests_infer_the_matching_registry_version(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = _client(monkeypatch, tmp_path)

    validated = client.post(
        "/api/custom-indicators/validate",
        json={
            "name": "旧版累计收益",
            "expression": "product(returns + 1) - 1",
            "dsl_version": "2.0.0",
            "context_kind": "single_product",
        },
    )
    composed = client.post(
        "/api/custom-indicators/compose",
        json={
            "operator_id": "product",
            "context": "single_product",
            "dsl_version": "2.0.0",
            "arguments": [
                {"parameter": "values", "source": "variable", "value": "returns"}
            ],
        },
    )

    assert validated.status_code == 200
    assert validated.json()["valid"] is True
    assert validated.json()["operator_registry_version"] == "2.0.0"
    assert composed.status_code == 200
    assert composed.json()["operator_registry_version"] == "2.0.0"


def test_market_metric_remains_visible_and_reports_missing_fund_field(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_fund_data(tmp_path)
    client = _client(monkeypatch, tmp_path)

    fund_catalog = client.get("/api/custom-indicators?product_kind=fund")
    evaluated = client.post(
        "/api/custom-indicators/evaluate",
        json={
            "indicator_ids": ["builtin-average-volume-v2"],
            "targets": [{"kind": "fund", "product_id": "000001.OF"}],
            "period": "1W",
        },
    )

    assert fund_catalog.status_code == 200
    assert "builtin-average-volume-v2" in {
        item["id"] for item in fund_catalog.json()["items"]
    }
    assert evaluated.status_code == 200
    assert evaluated.json()["summary"]["unavailable"] == 1
    result = evaluated.json()["results"][0]
    assert result["status"] == "unavailable"
    assert result["value"] is None
    assert result["warnings"][0]["code"] == "INDICATOR_NOT_APPLICABLE"
    assert result["presentation"]["applicable_product_kinds"] == ["etf"]
    assert result["input_requirements"]["status"] == "blocked"
    blocking = result["input_requirements"]["blocking_inputs"]
    assert len(blocking) == 1
    assert blocking[0]["variable_id"] == "volume"
    assert blocking[0]["label"] == "成交量"
    assert blocking[0]["status"] == "source_unavailable"
    assert blocking[0]["reason_code"] == "SOURCE_UNAVAILABLE_FOR_PRODUCT"
    assert "成交量" in result["warnings"][0]["message"]
    assert result["target_data"]["data_latest_date"] is not None


def test_batch_availability_reports_each_selected_product_and_field(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_fund_data(tmp_path)
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/custom-indicators/variables/availability",
        json={
            "targets": [{"kind": "fund", "product_id": "000001.OF"}],
            "variable_ids": ["adjusted_nav", "market_high", "market_low"],
            "period": "1W",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {"target_count": 1, "variable_count": 3}
    availability = {item["variable_id"]: item for item in body["items"]}
    assert availability["adjusted_nav"]["status"] == "available"
    assert availability["market_high"]["status"] == "source_unavailable"
    assert availability["market_low"]["status"] == "source_unavailable"
    assert availability["market_high"]["target_statuses"][0]["target"]["name"] == "华夏成长"
    assert "1/1" in availability["adjusted_nav"]["reason"]["message"]
    assert "0/1" in availability["market_high"]["reason"]["message"]


def test_batch_availability_rejects_conflicting_or_more_than_ten_targets(
    monkeypatch,
    tmp_path: Path,
) -> None:
    client = _client(monkeypatch, tmp_path)
    target = {"kind": "fund", "product_id": "000001.OF"}

    conflicting = client.post(
        "/api/custom-indicators/variables/availability",
        json={"kind": "fund", "product_id": "000001.OF", "targets": [target]},
    )
    too_many = client.post(
        "/api/custom-indicators/variables/availability",
        json={"targets": [target] * 11},
    )

    assert conflicting.status_code == 422
    assert conflicting.json()["detail"]["code"] == "AVAILABILITY_TARGET_CONFLICT"
    assert too_many.status_code == 422
    assert too_many.json()["detail"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_fund_market_range_reports_both_missing_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _write_fund_data(tmp_path)
    client = _client(monkeypatch, tmp_path)

    response = client.post(
        "/api/custom-indicators/evaluate",
        json={
            "indicator_ids": ["builtin-market-high-low-range-v2"],
            "targets": [{"kind": "fund", "product_id": "000001.OF"}],
            "period": "1W",
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["value"] is None
    assert result["status"] == "unavailable"
    blocking = result["input_requirements"]["blocking_inputs"]
    assert [item["label"] for item in blocking] == ["最高价", "最低价"]
    assert all(item["status"] == "source_unavailable" for item in blocking)
    assert "最高价、最低价" in result["warnings"][0]["message"]


def test_typed_compose_infer_and_portfolio_snapshot_evaluation(monkeypatch, tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)

    composed = client.post(
        "/api/custom-indicators/compose",
        json={
            "template_id": "cumulative-return",
            "context": "single_product",
            "arguments": [{"parameter": "values", "source": "variable", "value": "returns"}],
        },
    )
    inferred = client.post(
        "/api/custom-indicators/infer",
        json={"expression": r"\operatorname{matvec}\left(\mathbf{R},\mathbf{w}\right)", "context": "portfolio"},
    )
    reusable = service.get_indicator("builtin-total-return-v2")
    composed_indicator = client.post(
        "/api/custom-indicators/compose",
        json={
            "indicator_id": reusable["id"],
            "indicator_revision": reusable["revision"],
            "context": reusable["context_kind"],
            "dsl_version": reusable["dsl_version"],
            "operator_registry_version": reusable["operator_registry_version"],
            "variable_registry_version": reusable["variable_registry_version"],
            "data_contract_version": reusable["data_contract_version"],
            "context_schema_version": reusable["context_schema_version"],
            "arguments": [],
        },
    )
    assert composed.status_code == 200
    assert composed.json()["shape"] == "scalar"
    assert inferred.status_code == 200
    assert inferred.json()["shape"] == "series"
    assert composed_indicator.status_code == 200
    assert composed_indicator.json()["expression"] == reusable["expression"]
    assert composed_indicator.json()["indicator_origin"]["indicator_revision"] == reusable["revision"]

    run = service.portfolio_runs.create(
        {
            "target_id": "target-test",
            "target_revision": 1,
            "target_name": "API 组合",
            "context_schema": "portfolio-v2",
            "asset_order": ["etf:A", "fund:B"],
            "data_fingerprints": {"etf:A": "one", "fund:B": "two"},
            "common_date_hash": "dates-hash",
            "requested_as_of": None,
            "effective_as_of": "2026-01-06",
            "actual_start_date": "2026-01-02",
            "actual_end_date": "2026-01-06",
            "observation_count": 3,
            "asset_returns": [[0.01, 0.02], [0.015, -0.005], [0.0, 0.01]],
            "daily_weights": [[0.6, 0.4], [0.6, 0.4], [0.6, 0.4]],
            "benchmark_returns": None,
            "warnings": [],
        }
    )
    payload = {
        "run_id": run["id"],
        "inline_definition": {
            **_draft("组合波动率"),
            "expression": (
                r"\sqrt{\operatorname{dot}\left(\mathbf{w},"
                r"\operatorname{matvec}\left(\operatorname{covariance}\left(\mathbf{R}\right),\mathbf{w}\right)\right)}"
            ),
            "context_kind": "portfolio",
            "dsl_version": "2.0.0",
            "operator_registry_version": "2.0.0",
        },
    }
    validation = client.post(
        "/api/custom-indicators/validate",
        json=payload["inline_definition"],
    )
    assert validation.status_code == 200
    assert validation.json()["valid"] is True
    payload["compile_token"] = validation.json()["compile_token"]
    first = client.post("/api/custom-indicators/evaluate-portfolio", json=payload)
    second = client.post("/api/custom-indicators/evaluate-portfolio", json=payload)
    assert first.status_code == 200
    assert first.json()["results"][0]["value"] is not None
    assert first.json()["results"][0]["runtime_trace"]["nodes"]
    assert second.json()["cache"]["hits"] == 1


def test_crud_revision_conflict_and_delete_contract(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)
    created_response = client.post("/api/custom-indicators", json=_draft())
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["dsl_version"] == "2.3.0"
    assert created["numeric_kernel_version"] == "2.2.0"
    assert created["period_policy"] == "all_supported"
    assert created["periods"] == list(custom_indicator_routes.SUPPORTED_PERIODS)
    assert created["context_kind"] == "single_product"

    update_payload = {**_draft("更新后指标"), "revision": created["revision"]}
    updated_response = client.put(f"/api/custom-indicators/{created['id']}", json=update_payload)
    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["revision"] == 2

    stale_update = client.put(f"/api/custom-indicators/{created['id']}", json=update_payload)
    assert stale_update.status_code == 409
    assert stale_update.json()["detail"]["code"] == "REVISION_CONFLICT"
    stale_delete = client.delete(f"/api/custom-indicators/{created['id']}?revision=1")
    assert stale_delete.status_code == 409
    assert stale_delete.json()["detail"]["code"] == "REVISION_CONFLICT"

    deleted = client.delete(f"/api/custom-indicators/{created['id']}?revision=2")
    assert deleted.status_code == 200
    assert deleted.json() == {"deleted_id": created["id"]}

    missing_revision = client.delete(f"/api/custom-indicators/{created['id']}")
    assert missing_revision.status_code == 422
    assert missing_revision.json()["detail"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_real_data_evaluate_and_stable_422_error(monkeypatch, tmp_path: Path) -> None:
    _write_etf_data(tmp_path)
    client = _client(monkeypatch, tmp_path)

    evaluated = client.post(
        "/api/custom-indicators/evaluate",
        json={
            "indicator_ids": ["builtin-cumulative-return"],
            "targets": [{"kind": "etf", "product_id": "510050"}],
            "period": "1W",
            "include_series": True,
        },
    )
    unsupported = client.post(
        "/api/custom-indicators/evaluate",
        json={
            "indicator_ids": ["builtin-cumulative-return"],
            "targets": [{"kind": "etf", "product_id": "510050"}],
            "period": "2D",
        },
    )

    assert evaluated.status_code == 200
    body = evaluated.json()
    assert body["summary"] == {"total": 1, "ok": 1, "warning": 0, "error": 0}
    assert body["results"][0]["window"]["observation_count"] == 5
    assert body["results"][0]["series"]
    assert body["results"][0]["presentation"]["indicator_id"] == "builtin-cumulative-return"
    assert body["results"][0]["presentation"]["value_scale"] == 100.0
    assert unsupported.status_code == 422
    assert unsupported.json()["detail"]["code"] == "INVALID_PERIOD"


def test_evaluation_plan_create_list_run_and_delete(monkeypatch, tmp_path: Path) -> None:
    _write_etf_data(tmp_path)
    client = _client(monkeypatch, tmp_path)
    create_response = client.post(
        "/api/evaluation-plans",
        json={
            "name": "API 评价方案",
            "description": "",
            "product_kind": "etf",
            "indicators": [
                {
                        "indicator_id": "builtin-total-return-v2",
                    "period": "1W",
                    "weight": 100,
                }
            ],
            "targets": [{"kind": "etf", "product_id": "510050.SH"}],
            "product_selection": {
                "query": "上证50",
                "filters": {
                    "fund_type": ["股票型"],
                    "invest_type": ["被动指数型"],
                    "qdii_type": ["非QDII"],
                    "market": ["上交所"],
                    "status": ["上市交易"],
                    "management": [],
                    "custodian": [],
                },
                "conditions": [
                    {"field": "return_1y", "operator": "gte", "value": "5"}
                ],
                "selection_mode": "all_matching",
            },
            "missing_policy": "strict",
        },
    )

    assert create_response.status_code == 201
    plan = create_response.json()
    assert plan["product_kind"] == "etf"
    assert plan["indicators"][0]["indicator_revision"] == 1
    assert plan["product_selection"] == {
        "query": "上证50",
        "filters": {
            "fund_type": ["股票型"],
            "invest_type": ["被动指数型"],
            "qdii_type": ["非QDII"],
            "market": ["上交所"],
            "status": ["上市交易"],
            "management": [],
            "custodian": [],
        },
        "conditions": [{"field": "return_1y", "operator": "gte", "value": "5"}],
        "selection_mode": "all_matching",
    }
    listing = client.get("/api/evaluation-plans?kind=etf")
    fund_listing = client.get("/api/evaluation-plans?kind=fund")
    run = client.post(f"/api/evaluation-plans/{plan['id']}/run", json={})
    assert listing.json()["total"] == 1
    assert fund_listing.json()["total"] == 0
    assert run.status_code == 200
    assert run.json()["ranked_count"] == 1
    assert run.json()["rows"][0]["rank"] == 1
    assert run.json()["normalization"]["effective_weight_total"] == 1.0
    value = run.json()["rows"][0]["values"][0]
    assert value["effective_weight"] == 1.0
    assert value["weighted_contribution"] == 50.0
    assert value["presentation"]["revision"] == 1
    assert value["window"]["observation_count"] == 5

    deleted = client.delete(f"/api/evaluation-plans/{plan['id']}?revision=1")
    assert deleted.status_code == 200


def test_evaluation_plan_result_paging_route(monkeypatch, tmp_path: Path) -> None:
    service = CustomIndicatorService(tmp_path, tmp_path)
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)
    result_id = service.run_results.store(
        {
            "plan_id": "plan-paged",
            "plan_revision": 1,
            "run_at": "2026-09-01T00:00:00+00:00",
            "as_of": None,
            "ranked_count": 2,
            "excluded_count": 0,
            "normalization": {},
            "execution": {},
            "rows": [
                {
                    "rank": index + 1,
                    "score": float(2 - index),
                    "status": "ranked",
                    "target": {"kind": "etf", "product_id": f"51000{index}.SH"},
                    "values": [],
                }
                for index in range(2)
            ],
        }
    )

    response = client.get(
        f"/api/evaluation-plan-runs/{result_id}?page=2&page_size=1"
    )

    assert response.status_code == 200
    assert response.json()["rows"][0]["rank"] == 2
    assert response.json()["pagination"]["total"] == 2


def test_integrated_app_registers_each_custom_route_once() -> None:
    from app import app

    schema = app.openapi()
    expected = {
        ("/api/custom-indicators/meta", "GET"),
        ("/api/custom-indicators", "GET"),
        ("/api/custom-indicators", "POST"),
        ("/api/custom-indicators/validate", "POST"),
        ("/api/custom-indicators/compose", "POST"),
        ("/api/custom-indicators/infer", "POST"),
        ("/api/custom-indicators/evaluate", "POST"),
        ("/api/custom-indicators/evaluate-portfolio", "POST"),
        ("/api/custom-indicators/snapshot-config", "GET"),
        ("/api/custom-indicators/snapshot-config", "PUT"),
        ("/api/evaluation-plans", "GET"),
        ("/api/evaluation-plans", "POST"),
        ("/api/evaluation-plan-runs/{result_id}", "GET"),
    }
    operation_ids = []
    for path, method in expected:
        operation = schema["paths"][path][method.lower()]
        operation_ids.append(operation["operationId"])
    assert len(operation_ids) == len(set(operation_ids))


def test_snapshot_indicator_config_routes(monkeypatch, tmp_path: Path) -> None:
    client = _client(monkeypatch, tmp_path)

    current = client.get("/api/custom-indicators/snapshot-config")
    assert current.status_code == 200
    assert current.json()["items"]

    updated = client.put(
        "/api/custom-indicators/snapshot-config",
        json={
            "revision": current.json()["revision"],
            "items": [
                {
                    "indicator_id": "builtin-total-return-v2",
                    "indicator_revision": 1,
                    "period": "ALL",
                }
            ],
        },
    )
    assert updated.status_code == 200
    assert updated.json()["items"][0]["period"] == "ALL"

    stale = client.put(
        "/api/custom-indicators/snapshot-config",
        json={"revision": current.json()["revision"], "items": []},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "REVISION_CONFLICT"
