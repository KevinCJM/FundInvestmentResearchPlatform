from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from historical_regimes.service import HistoricalRegimeService
from services import historical_regime_routes


def _rows(count: int = 64, *, missing_at: int | None = None) -> list[dict[str, Any]]:
    start = date(2021, 1, 1)
    rows: list[dict[str, Any]] = []
    for index in range(count):
        value = 100.0 + index if index < 32 else 132.0 - 0.7 * (index - 32)
        if index == missing_at:
            value = None
        observation_date = start + timedelta(days=index)
        rows.append(
            {
                "observation_date": observation_date.isoformat(),
                "available_at": (observation_date + timedelta(days=1)).isoformat(),
                "value": value,
                "revision": 1,
            }
        )
    return rows


def _definition(
    formula: str,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "name": "自定义因果公式路由测试",
        "target": {
            "kind": "inline",
            "series_id": "formula-fixture",
            "name": "公式固定样本",
            "frequency": "daily",
            "rows": rows if rows is not None else _rows(),
        },
        "features": {
            "formula": formula,
            "transform": "identity",
            "filter": "ema",
            "window": 4,
            "slope_window": 2,
            "volatility_window": 4,
        },
        "algorithm": {
            "family": "causal_filter",
            "parameters": {
                "bull_enter": 0.1,
                "bull_exit": 0.02,
                "bear_enter": -0.1,
                "bear_exit": -0.02,
                "confirmation": 1,
                "min_duration": 1,
            },
        },
        "states": [
            {"id": "bull", "label": "牛市", "role": "positive", "order": 1},
            {"id": "sideways", "label": "震荡市", "role": "neutral", "order": 2},
            {"id": "bear", "label": "熊市", "role": "negative", "order": 3},
        ],
        "validation": {"walk_forward": False},
        "usage_intent": "research_display",
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    service = HistoricalRegimeService(tmp_path, tmp_path)
    monkeypatch.setattr(historical_regime_routes, "historical_regime_service", service)
    app = FastAPI()
    app.state.historical_regime_service = service
    app.include_router(historical_regime_routes.router)
    return TestClient(app)


def _service(client: TestClient) -> HistoricalRegimeService:
    return client.app.state.historical_regime_service


def _run(client: TestClient, definition: dict[str, Any]) -> dict[str, Any]:
    prepared = client.post(
        "/api/historical-regimes/formulas/prepare",
        json={"definition": definition, "mode": "realtime"},
    )
    assert prepared.status_code == 200, prepared.text
    preparation = prepared.json()
    assert preparation["required"] is True
    assert preparation["request_time_compilation"] == 0
    # v1 execution is no longer an HTTP write surface.  Seed an immutable
    # historical run through the legacy service, then verify the compatibility
    # read endpoint still exposes it unchanged.
    seeded = _service(client).run(
        definition,
        "realtime",
        compile_token=preparation["compile_token"],
    )
    response = client.get(f"/api/historical-regimes/runs/{seeded['id']}")
    assert response.status_code == 200, response.text
    assert response.json() == seeded
    return response.json()


def test_formula_contract_and_compilation_audit_are_exposed_by_api(client: TestClient) -> None:
    meta = client.get("/api/historical-regimes/meta")
    assert meta.status_code == 200
    language = meta.json()["formula_language"]
    assert language["allowlist_version"] == "typed-njit-causal-scope-3"
    assert language["njit_required"] is True
    assert language["python_fallback"] == 0
    assert language["provenance"]["field"] == "features.formula_provenance"
    assert {item["id"] for item in language["functions"]} >= {
        "lag",
        "difference",
        "difference",
        "cumulative_sum",
        "cumulative_max",
    }

    run = _run(client, _definition("difference(log(value), 3)"))
    audit = run["formula_diagnostics"]
    assert audit["execution_backend"] == "numba_njit_fixed_signature"
    assert audit["compile_status"] == "compiled"
    assert audit["python_fallback"] == 0
    assert audit["python_operator_calls"] == 0
    assert audit["dag"]["roots"]["result"] is not None
    assert audit["allowlist_version"] == language["allowlist_version"]
    assert audit["evaluator_version"]
    assert audit["referenced_columns"] == ["value"]
    assert run["data_snapshot"]["feature_formula"]["fingerprint"] == audit["fingerprint"]
    assert run["algorithm_diagnostics"]["feature_formula"]["normalized_expression"]
    assert run["series"][12]["value"] == pytest.approx(_rows()[12]["value"])
    assert run["series"][12]["features"]["formula_input"] != pytest.approx(
        run["series"][12]["value"]
    )


def test_v1_formula_run_endpoint_is_read_only_even_after_preparation(client: TestClient) -> None:
    definition = _definition("log(value)")
    missing = client.post(
        "/api/historical-regimes/run",
        json={"definition": definition, "mode": "realtime"},
    )
    assert missing.status_code == 409
    assert missing.json()["detail"]["code"] == "V1_RUN_READ_ONLY"

    prepared = client.post(
        "/api/historical-regimes/formulas/prepare",
        json={"definition": definition, "mode": "realtime"},
    ).json()
    mismatched = client.post(
        "/api/historical-regimes/run",
        json={
            "definition": _definition("value * 2"),
            "mode": "realtime",
            "compile_token": prepared["compile_token"],
        },
    )
    assert mismatched.status_code == 409
    assert mismatched.json()["detail"]["code"] == "V1_RUN_READ_ONLY"


def test_formula_api_is_prefix_invariant(client: TestClient) -> None:
    rows = _rows(64)
    formula = "difference(log(value), 8)"
    full = _run(client, _definition(formula, rows=rows))
    prefix = _run(client, _definition(formula, rows=rows[:48]))

    for full_point, prefix_point in zip(full["series"][:48], prefix["series"]):
        assert full_point["state_id"] == prefix_point["state_id"]
        if full_point["features"]["formula_input"] is None:
            assert prefix_point["features"]["formula_input"] is None
        else:
            assert full_point["features"]["formula_input"] == pytest.approx(
                prefix_point["features"]["formula_input"]
            )


@pytest.mark.parametrize(
    ("formula", "expected_code"),
    [
        ("value.__class__", "FORMULA_NODE_FORBIDDEN"),
        ("value[-1]", "FORMULA_NODE_FORBIDDEN"),
        ("lead(value, 1)", "FORMULA_FUNCTION_FORBIDDEN"),
        ("__import__('os')", "FORMULA_FUNCTION_FORBIDDEN"),
    ],
)
def test_formula_api_rejects_unsafe_expressions(
    client: TestClient,
    formula: str,
    expected_code: str,
) -> None:
    response = client.post(
        "/api/historical-regimes/formulas/prepare",
        json={"definition": _definition(formula), "mode": "realtime"},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == expected_code
    assert detail["field"] == "features.formula"


def test_formula_api_keeps_missing_input_null_and_unclassified(client: TestClient) -> None:
    run = _run(client, _definition("value * 2", rows=_rows(missing_at=27)))
    point = run["series"][27]

    assert point["value"] is None
    assert point["features"]["formula_input"] is None
    assert point["state_id"] == "unclassified"
    assert point["confidence"] is None
    assert point["probabilities"] == {}
    assert any("未用 0 替代" in reason for reason in point["reasons"])


def test_unverified_formula_column_cannot_reach_formal_or_taa_publication(
    client: TestClient,
) -> None:
    rows = _rows()
    for index, row in enumerate(rows):
        row["future_return"] = (
            float(rows[index + 1]["value"]) / float(row["value"]) - 1.0
            if index + 1 < len(rows)
            else None
        )
    definition = _definition("future_return", rows=rows)
    created = _service(client).create_definition(definition)
    reference = {"id": created["id"], "revision": created["revision"]}
    prepared = client.post(
        "/api/historical-regimes/formulas/prepare",
        json={"definition": reference, "mode": "realtime"},
    )
    assert prepared.status_code == 200, prepared.text
    run = _service(client).run(
        reference,
        "realtime",
        compile_token=prepared.json()["compile_token"],
    )

    assert run["formula_diagnostics"]["input_provenance"]["verified"] is False
    assert "formal_backtest" not in run["causality"]["publish_eligible_usages"]
    assert "taa" not in run["causality"]["publish_eligible_usages"]
    publish = client.post(
        f"/api/historical-regimes/runs/{run['id']}/publish",
        json={"usage": "taa"},
    )
    assert publish.status_code == 409
    assert publish.json()["detail"]["code"] == "V1_RUN_PUBLICATION_READ_ONLY"
