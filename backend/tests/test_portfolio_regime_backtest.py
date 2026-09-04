from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import services.portfolio_routes as portfolio_routes
import services.strategy_routes as strategy_routes
from custom_indicators.errors import ConflictError, ValidationError
from custom_indicators.portfolio_service import PortfolioResearchService
from historical_regimes.v2_contracts import definition_content_hash, parse_definition_v2
from portfolio_regime import (
    PublishedRegimeBacktestReference,
    PublishedRegimeBacktestResolver,
    _conditioning_audit,
    _stored_run_snapshot_hash,
    condition_nav_backtest,
)


def _regime_definition() -> dict[str, Any]:
    rows = []
    start = date(2024, 1, 1)
    for index in range(12):
        observation = start + timedelta(days=index)
        rows.append(
            {
                "observation_date": observation.isoformat(),
                "available_at": observation.isoformat(),
                "value": 100.0 + index,
            }
        )
    return {
        "schema_version": "2.0",
        "name": "正式回测状态",
        "description": "测试锁定发布版本。",
        "graph": {
            "nodes": [
                {
                    "id": "source",
                    "type": "source.inline",
                    "parameters": {"rows": rows, "frequency": "daily"},
                },
                {
                    "id": "classifier",
                    "type": "model.threshold",
                    "parameters": {"upper": 0.5, "lower": -0.5},
                    "inputs": {"value": {"node_id": "source", "port": "value"}},
                },
            ],
            "outputs": {"state": {"node_id": "classifier", "port": "state"}},
            "exposed_node_ids": ["classifier"],
        },
        "states": [
            {"id": "bull", "label": "牛市", "role": "positive", "color": "#16a34a", "order": 1},
            {"id": "sideways", "label": "震荡", "role": "neutral", "color": "#64748b", "order": 2},
            {"id": "bear", "label": "熊市", "role": "negative", "color": "#dc2626", "order": 3},
        ],
        "evaluation_targets": [],
        "validation": {"walk_forward": True, "folds": 2},
        "usage_intent": "formal_backtest",
    }


def _seed_regime_run(
    resolver: PublishedRegimeBacktestResolver,
    *,
    effective_start: str,
    effective_change: str,
    usage: str = "formal_backtest",
    publication_id: str = "publication-formal",
    publication_hash: str | None = None,
) -> tuple[dict[str, Any], PublishedRegimeBacktestReference]:
    definition = resolver.definitions.create(_regime_definition())
    definition_hash = definition_content_hash(parse_definition_v2(definition))
    states = definition["states"]
    run_payload = {
        "schema_version": "2.0",
        "definition_id": definition["id"],
        "definition_revision": definition["revision"],
        "definition_snapshot_hash": definition_hash,
        "name": "锁定状态运行",
        "mode": "realtime",
        "definition": definition,
        "states": states,
        "series": [
            {
                "observation_date": effective_start,
                "recognized_at": effective_start,
                "effective_date": effective_start,
                "state_id": "bull",
                "state_label": "牛市",
                "confidence": 0.9,
            },
            {
                "observation_date": effective_change,
                "recognized_at": effective_change,
                "effective_date": effective_change,
                "state_id": "bear",
                "state_label": "熊市",
                "confidence": 0.8,
            },
        ],
        "governance": {
            "formal_gate_passed": True,
            "publish_eligible_usages": ["research_display", "formal_backtest"],
        },
        "application_bindings": [],
    }
    run_payload["content_hash"] = _stored_run_snapshot_hash(run_payload)
    run = resolver.runs.create(run_payload)
    publication = {
        "id": publication_id,
        "usage": usage,
        "published_at": "2025-01-01T00:00:00+00:00",
        "run_id": run["id"],
        "definition_revision": definition["revision"],
        "run_content_hash": publication_hash or run["content_hash"],
        "gate": (
            "comprehensive_formal_gate_passed"
            if usage == "formal_backtest"
            else "research_with_recorded_restrictions"
        ),
    }
    resolver.runs.add_publications(run["id"], [publication])
    return resolver.runs.get(run["id"]), PublishedRegimeBacktestReference(
        run_id=run["id"],
        publication_id=publication_id,
    )


def _write_asset_nv(path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    rows = []
    for index, current in enumerate(dates):
        rows.extend(
            [
                {
                    "date": current,
                    "asset_name": "ClassA",
                    "asset_alloc_name": "demo",
                    "nv": 100.0 * (1.01**index),
                },
                {
                    "date": current,
                    "asset_name": "ClassB",
                    "asset_alloc_name": "demo",
                    "nv": 100.0 * (1.002**index),
                },
            ]
        )
    pd.DataFrame(rows).to_parquet(path / "asset_nv.parquet", index=False)


def _write_product_nav(path: Path) -> None:
    dates = pd.bdate_range("2025-01-02", periods=40)
    pd.DataFrame(
        [
            {"ts_code": "510001.SH", "code": "510001", "name": "ETF A"},
            {"ts_code": "510002.SH", "code": "510002", "name": "ETF B"},
        ]
    ).to_parquet(path / "etf_info_df.parquet", index=False)
    rows = []
    for index, current in enumerate(dates):
        rows.extend(
            [
                {
                    "ts_code": "510001.SH",
                    "date": current,
                    "adj_nav": 1.0 * (1.002**index),
                    "name": "ETF A",
                },
                {
                    "ts_code": "510002.SH",
                    "date": current,
                    "adj_nav": 2.0 * (1.001**index),
                    "name": "ETF B",
                },
            ]
        )
    pd.DataFrame(rows).to_parquet(path / "etf_daily_df.parquet", index=False)


def _portfolio_definition() -> dict[str, Any]:
    return {
        "components": [
            {"kind": "etf", "product_id": "510001", "name": "ETF A"},
            {"kind": "etf", "product_id": "510002", "name": "ETF B"},
        ],
        "strategy": {"type": "equal_weight"},
        "constraints": {"min_weight": 0.0, "max_weight": 1.0},
        "rebalance": {"enabled": False},
        "alignment": "strict_intersection",
    }


def test_conditioning_uses_period_start_and_fixed_njit_signatures(tmp_path: Path) -> None:
    resolver = PublishedRegimeBacktestResolver(tmp_path)
    _, reference = _seed_regime_run(
        resolver,
        effective_start="2024-01-01",
        effective_change="2024-01-03",
    )
    resolved = resolver.resolve(reference)
    signatures_before = _conditioning_audit()["kernel_signatures"]

    result = condition_nav_backtest(
        resolved,
        pd.date_range("2024-01-01", periods=6, freq="D"),
        {"策略 A": [1.0, 1.01, 1.02, 1.00, 0.99, 1.03]},
    )

    by_date = {item["date"]: item for item in result["period_states"]}
    assert by_date["2024-01-03"]["state_id"] == "bull"
    assert by_date["2024-01-04"]["state_id"] == "bear"
    assert result["alignment"]["same_period_end_signal_allowed"] is False
    assert result["binding"]["definition_snapshot_hash"]
    assert result["binding"]["run_content_hash"]
    assert len(result["conditional_performance"]["策略 A"]) == 3
    assert result["execution"]["execution_backend"] == "numba_njit_fixed_signature"
    assert result["execution"]["python_callback"] is False
    assert result["execution"]["request_time_compilation"] == 0
    assert signatures_before == _conditioning_audit()["kernel_signatures"]


def test_resolver_rejects_non_formal_and_broken_publication_lineage(tmp_path: Path) -> None:
    product_resolver = PublishedRegimeBacktestResolver(tmp_path / "product")
    _, product_reference = _seed_regime_run(
        product_resolver,
        effective_start="2024-01-01",
        effective_change="2024-01-03",
        usage="product_research",
    )
    with pytest.raises(ValidationError) as product_error:
        product_resolver.resolve(product_reference)
    assert product_error.value.code == "FORMAL_BACKTEST_PUBLICATION_REQUIRED"
    with pytest.raises(ValidationError) as missing_publication:
        product_resolver.resolve(
            {
                "run_id": product_reference.run_id,
                "publication_id": "publication-never-selected",
            }
        )
    assert missing_publication.value.code == "REGIME_PUBLICATION_NOT_FOUND"

    broken_resolver = PublishedRegimeBacktestResolver(tmp_path / "broken")
    _, broken_reference = _seed_regime_run(
        broken_resolver,
        effective_start="2024-01-01",
        effective_change="2024-01-03",
        publication_hash="sha256-broken",
    )
    with pytest.raises(ConflictError) as broken_error:
        broken_resolver.resolve(broken_reference)
    assert broken_error.value.code == "REGIME_PUBLICATION_LINEAGE_MISMATCH"

    legacy_resolver = PublishedRegimeBacktestResolver(tmp_path / "legacy")
    legacy_payload = {
        "schema_version": "1.0",
        "name": "旧运行",
        "application_bindings": [],
    }
    legacy_payload["content_hash"] = _stored_run_snapshot_hash(legacy_payload)
    legacy_run = legacy_resolver.runs.create(legacy_payload)
    legacy_resolver.runs.add_publications(
        legacy_run["id"],
        [
            {
                "id": "publication-legacy",
                "usage": "formal_backtest",
                "run_id": legacy_run["id"],
                "run_content_hash": legacy_run["content_hash"],
                "definition_revision": 1,
                "gate": "comprehensive_formal_gate_passed",
            }
        ],
    )
    with pytest.raises(ValidationError) as legacy_error:
        legacy_resolver.resolve(
            {
                "run_id": legacy_run["id"],
                "publication_id": "publication-legacy",
            }
        )
    assert legacy_error.value.code == "FORMAL_BACKTEST_REQUIRES_V2_REGIME"


def test_strategy_backtest_reference_is_optional_and_exact(tmp_path: Path, monkeypatch) -> None:
    _write_asset_nv(tmp_path)
    resolver = PublishedRegimeBacktestResolver(tmp_path)
    _, reference = _seed_regime_run(
        resolver,
        effective_start="2024-01-01",
        effective_change="2024-01-04",
    )
    monkeypatch.setattr(strategy_routes, "DATA_DIR", tmp_path)
    monkeypatch.setattr(strategy_routes, "regime_backtest_resolver", resolver)
    strategy = strategy_routes.StrategySpec(
        type="fixed",
        name="固定组合",
        classes=[
            strategy_routes.StrategyClassItem(name="ClassA", weight=0.5),
            strategy_routes.StrategyClassItem(name="ClassB", weight=0.5),
        ],
    )

    plain = strategy_routes.api_backtest(
        strategy_routes.BacktestRequest(alloc_name="demo", strategies=[strategy])
    )
    assert "regime_conditioning" not in plain

    conditioned = strategy_routes.api_backtest(
        strategy_routes.BacktestRequest.model_validate(
            {
                "alloc_name": "demo",
                "strategies": [strategy.model_dump()],
                "regime": reference.model_dump(),
            }
        )
    )
    assert conditioned["regime_conditioning"]["binding"]["run_id"] == reference.run_id
    assert (
        conditioned["regime_conditioning"]["binding"]["publication_id"]
        == reference.publication_id
    )
    assert "固定组合" in conditioned["regime_conditioning"]["conditional_performance"]


def test_portfolio_run_persists_locked_regime_conditioning_and_cache_key(tmp_path: Path) -> None:
    _write_product_nav(tmp_path)
    resolver = PublishedRegimeBacktestResolver(tmp_path)
    raw_run, reference = _seed_regime_run(
        resolver,
        effective_start="2025-01-02",
        effective_change="2025-01-20",
    )
    second_publication = {
        "id": "publication-formal-2",
        "usage": "formal_backtest",
        "published_at": "2025-01-02T00:00:00+00:00",
        "run_id": raw_run["id"],
        "definition_revision": raw_run["definition_revision"],
        "run_content_hash": raw_run["content_hash"],
        "gate": "comprehensive_formal_gate_passed",
    }
    resolver.runs.add_publications(raw_run["id"], [second_publication])
    service = PortfolioResearchService(
        tmp_path,
        tmp_path,
        regime_backtest_resolver=resolver,
    )
    target = service.create_target(
        {"name": "组合 A", "definition": _portfolio_definition()}
    )

    plain = service.run_target(target["id"])
    assert "regime_conditioning" not in plain

    first = service.run_target(target["id"], historical_regime=reference)
    assert first["regime_conditioning"]["binding"]["publication_id"] == reference.publication_id
    assert first["regime_conditioning"]["period_states"]
    assert "组合 A" in first["regime_conditioning"]["conditional_performance"]
    assert (
        service.get_run(first["id"])["regime_conditioning"]["binding"]["run_content_hash"]
        == raw_run["content_hash"]
    )
    listed = next(item for item in service.list_runs() if item["id"] == first["id"])
    assert "period_states" not in listed["regime_conditioning"]
    assert listed["regime_conditioning"]["period_states_count"] == first["observation_count"]

    second = service.run_target(
        target["id"],
        historical_regime={
            "run_id": raw_run["id"],
            "publication_id": second_publication["id"],
        },
    )
    assert second["cache"]["hit"] is False
    assert second["regime_conditioning"]["binding"]["publication_id"] == "publication-formal-2"


def test_portfolio_route_requires_complete_explicit_reference(tmp_path: Path, monkeypatch) -> None:
    _write_product_nav(tmp_path)
    service = PortfolioResearchService(tmp_path, tmp_path)
    target = service.create_target(
        {"name": "组合 API", "definition": _portfolio_definition()}
    )
    monkeypatch.setattr(portfolio_routes, "portfolio_service", service)
    app = FastAPI()
    app.include_router(portfolio_routes.router)
    client = TestClient(app)

    response = client.post(
        f"/api/research-targets/{target['id']}/run",
        json={"regime": {"run_id": "regime-run-only"}},
    )

    assert response.status_code == 422
