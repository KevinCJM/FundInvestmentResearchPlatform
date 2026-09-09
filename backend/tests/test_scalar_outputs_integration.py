"""Offline end-to-end contracts for named scalar outputs and consumers."""
from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from custom_indicators.service import CustomIndicatorService
from test_scalar_indicator_outputs import bundle_draft, service


def target():
    return {"kind": "etf", "product_id": "510050.SH"}


def plan_draft(indicator, output_id="mean", **changes):
    item = {"indicator_id": indicator["id"], "indicator_revision": indicator["revision"],
            "output_id": output_id, "period": "ALL", "weight": 100}
    item.update(changes)
    return {"name": "多结果评分", "product_kind": "etf", "targets": [target()],
            "indicators": [item], "missing_policy": "strict"}


def test_http_validate_create_evaluate_and_compose(monkeypatch, service):
    from services import custom_indicator_routes as routes

    monkeypatch.setattr(routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        draft = bundle_draft()
        validation = client.post("/api/custom-indicators/validate", json=draft)
        assert validation.status_code == 200, validation.text
        assert validation.json()["valid"]
        evaluated = client.post("/api/custom-indicators/evaluate", json={
            "inline_definition": draft, "compile_token": validation.json()["compile_token"],
            "targets": [target()], "period": "ALL",
        })
        assert evaluated.status_code == 200, evaluated.text
        assert len(evaluated.json()["results"][0]["outputs"]) == 3
        created = client.post("/api/custom-indicators", json=draft)
        assert created.status_code in {200, 201}, created.text
        indicator = created.json()
        selected = client.post("/api/custom-indicators/evaluate", json={
            "indicator_refs": [{"indicator_id": indicator["id"], "indicator_revision": 1, "output_id": "mean"}],
            "targets": [target()], "period": "ALL",
        })
        assert selected.status_code == 200, selected.text
        assert selected.json()["results"][0]["output_id"] == "mean"
        composed = client.post("/api/custom-indicators/compose", json={
            "indicator_id": indicator["id"], "indicator_revision": 1, "output_id": "mean", "context": "single_product",
        })
        assert composed.status_code == 200, composed.text
        assert composed.json()["shape"] == "scalar"
        assert composed.json()["indicator_origin"]["output_id"] == "mean"
        unspecified = client.post("/api/custom-indicators/compose", json={"indicator_id": indicator["id"], "indicator_revision": 1})
        assert unspecified.status_code == 422
        assert unspecified.json()["detail"]["code"] == "OUTPUT_REQUIRED"


def test_selected_result_scores_despite_unselected_failure_and_keeps_version(service):
    indicator = service.create_indicator(bundle_draft())
    plan = service.create_plan(plan_draft(indicator))
    first = service.run_plan(plan["id"])
    assert first["ranked_count"] == 1
    value = first["rows"][0]["values"][0]
    assert value["output_id"] == "mean"
    assert value["indicator_revision"] == 1
    assert value["presentation"]["value_scale"] == 100
    changed = bundle_draft()
    changed["scalar_outputs"][0]["expression"] = "mean(returns) * 4"
    changed["scalar_outputs"][0]["label"] = "新版平均值"
    service.update_indicator(indicator["id"], 1, changed)
    second = service.run_plan(plan["id"])
    assert second["rows"][0]["values"][0]["value"] == value["value"]
    assert second["rows"][0]["values"][0]["presentation"]["name"] == value["presentation"]["name"]


def test_plan_requires_output_and_explicit_neutral_direction(service):
    indicator = service.create_indicator(bundle_draft())
    invalid = plan_draft(indicator)
    invalid["indicators"][0].pop("output_id")
    with pytest.raises(ValidationError) as unspecified:
        service.create_plan(invalid)
    assert unspecified.value.code == "OUTPUT_REQUIRED"
    with pytest.raises(ValidationError):
        service.create_plan(plan_draft(indicator, "twice"))
    valid = service.create_plan(plan_draft(indicator, "twice", direction="lower_better"))
    run = service.run_plan(valid["id"])
    assert run["rows"][0]["values"][0]["direction_overridden"]
    invalid = plan_draft(indicator)
    invalid["indicators"].append(deepcopy(invalid["indicators"][0]))
    with pytest.raises(ValidationError) as repeated:
        service.create_plan(invalid)
    assert repeated.value.code == "DUPLICATE_PLAN_INDICATOR"


def test_two_outputs_are_separate_scoring_items_and_pagination_survives(service, monkeypatch):
    indicator = service.create_indicator(bundle_draft())
    fields = plan_draft(indicator)
    fields["indicators"].append({**fields["indicators"][0], "output_id": "twice", "direction": "higher_better"})
    plan = service.create_plan(fields)
    result = service.run_plan(plan["id"])
    assert [value["output_id"] for value in result["rows"][0]["values"]] == ["mean", "twice"]
    assert sum(value["effective_weight"] for value in result["rows"][0]["values"]) == pytest.approx(1)
    # Retention uses the same repository/schema as ordinary scalar plan runs.
    result_id = service.run_results.store(result)
    page = service.get_plan_run_result(result_id, page=1, page_size=1)
    assert page["rows"][0]["values"][1]["output_id"] == "twice"


def test_cache_statistics_and_per_output_windows_are_honest(service):
    draft = bundle_draft()
    draft["scalar_outputs"][1]["expression"] = "mean(volume)"
    indicator = service.create_indicator(draft)
    kwargs = {"indicator_ids": [indicator["id"]], "inline_definition": None, "targets": [target()], "period": "ALL"}
    first = service.evaluate(**kwargs)
    second = service.evaluate(**kwargs)
    assert first["cache"] == {"hits": 0, "misses": 1}
    assert second["cache"] == {"hits": 1, "misses": 0}
    bundle = first["results"][0]
    assert bundle["window_scope"] == "per_output"
    assert bundle["window"]["start_date"] is None
    assert bundle["outputs"][0]["window"]["observation_count"] > 0


def portfolio_snapshot():
    return {"id": "test-immutable-run", "target_name": "研究组合", "actual_start_date": "2026-01-02",
            "actual_end_date": "2026-01-06", "effective_as_of": "2026-01-06", "observation_count": 3,
            "asset_returns": [[0.01, 0.02], [-0.01, 0.01], [0.02, 0.04]],
            "daily_weights": [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], "warnings": []}


def test_portfolio_bundle_is_warmed_and_missing_benchmark_is_isolated(service, monkeypatch):
    draft = bundle_draft()
    draft["context_kind"] = "portfolio"
    draft["scalar_outputs"] = [
        {"id": "portfolio", "label": "组合平均收益", "expression": "mean(portfolio_returns)", "display_format": "percent"},
        {"id": "benchmark", "label": "基准平均收益", "expression": "mean(benchmark_returns)"},
    ]
    indicator = service.create_indicator(draft)
    # Same workspace, a fresh facade as used by PortfolioResearchService.
    other = CustomIndicatorService(service.workspace_data_dir, service.market_data_dir)
    import custom_indicators.scalar_bundle_service as module
    monkeypatch.setattr(module, "compile_scalar_bundle", lambda *_args, **_kwargs: pytest.fail("request-time compilation"))
    result = other.evaluate_portfolio_snapshot([indicator["id"]], portfolio_snapshot())
    bundle = result["results"][0]
    assert bundle["target"]["product_id"] == "test-immutable-run"
    assert bundle["period"] == "snapshot"
    assert bundle["outputs"][0]["value"] == pytest.approx(0.015)
    assert bundle["outputs"][1]["value"] is None
    assert bundle["outputs"][1]["status"] == "unavailable"
    assert result["execution"]["python_fallback"] == 0


def test_long_display_names_do_not_overflow_internal_scalar_validation(service):
    draft = bundle_draft()
    draft["name"] = "研" * 80
    draft["scalar_outputs"][0]["label"] = "均" * 80
    indicator = service.create_indicator(draft)
    result = service.evaluate(indicator_ids=[indicator["id"]], inline_definition=None, targets=[target()], period="ALL")
    assert result["results"][0]["outputs"][0]["value"] is not None
    assert result["results"][0]["outputs"][0]["presentation"]["output_label"] == "均" * 80


def test_data_generation_change_fails_closed(service, monkeypatch):
    indicator = service.create_indicator(bundle_draft())
    import custom_indicators.scalar_bundle_service as module
    generations = iter(["before", "after", "after"])
    monkeypatch.setattr(module, "market_data_generation", lambda *_: next(generations, "after"))
    with pytest.raises(ValidationError) as changed:
        service.evaluate(indicator_ids=[indicator["id"]], inline_definition=None, targets=[target()], period="ALL")
    assert changed.value.code == "DATA_GENERATION_CHANGED"
