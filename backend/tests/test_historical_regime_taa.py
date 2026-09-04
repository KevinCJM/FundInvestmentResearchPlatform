from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from historical_regimes.service import HistoricalRegimeService
from historical_regimes.taa import run_taa_backtest, warm_taa_numba_kernels
from services import historical_regime_routes


def _synthetic_run() -> dict[str, Any]:
    return {
        "id": "regime-run-synthetic",
        "content_hash": "source-run-hash",
        "definition_id": "regime-synthetic",
        "definition_revision": 1,
        "states": [{"id": "bull", "label": "牛市"}, {"id": "bear", "label": "熊市"}],
        "series": [
            {
                "observation_date": "2024-01-01",
                "recognized_at": "2024-01-01",
                "effective_date": "2024-01-02",
                "state_id": "bull",
                "probabilities": {"bear": 1.0},
                "probability_source": "model",
                "confidence": 1.0,
            },
            {
                "observation_date": "2024-01-02",
                "recognized_at": "2024-01-02",
                "effective_date": "2024-01-03",
                "state_id": "bear",
                "probabilities": {"bull": 1.0},
                "probability_source": "model",
                "confidence": 1.0,
            },
        ],
    }


def _taa_request(*, transaction_cost_bps: float = 0.0) -> dict[str, Any]:
    return {
        "asset_returns": [
            {"date": "2024-01-02", "equity": 0.0, "bond": 0.0},
            {"date": "2024-01-03", "equity": 0.0, "bond": 0.0},
            {"date": "2024-01-04", "equity": 0.0, "bond": 0.0},
        ],
        "base_weights": {"equity": 0.5, "bond": 0.5},
        "state_tilts": {
            "bull": {"equity": 0.2, "bond": -0.2},
            "bear": {"equity": -0.2, "bond": 0.2},
        },
        "limits": {"min_weight": 0.0, "max_weight": 1.0, "max_abs_tilt": 0.3},
        "transaction_cost_bps": transaction_cost_bps,
        "confidence_floor": 0.5,
    }


def test_taa_uses_probabilities_and_only_strictly_prior_effective_dates() -> None:
    result = run_taa_backtest(_synthetic_run(), _taa_request(), {"passed": True})
    path = result["weights"]

    assert path[0]["date"] == "2024-01-02"
    assert path[0]["regime_effective_date"] is None
    assert path[0]["fallback_to_base"] is True
    assert path[0]["weights"] == {"bond": 0.5, "equity": 0.5}

    # state_id 故意写成 bull，但概率为 bear；权重必须服从概率而不是标签。
    assert path[1]["regime_effective_date"] == "2024-01-02"
    assert path[1]["probabilities"] == {"bull": 0.0, "bear": 1.0}
    assert path[1]["probability_source"] == "model"
    assert path[1]["weights"] == pytest.approx({"bond": 0.7, "equity": 0.3})
    assert path[2]["regime_effective_date"] == "2024-01-03"
    assert path[2]["weights"] == pytest.approx({"bond": 0.3, "equity": 0.7})
    assert all(
        item["regime_effective_date"] is None or item["regime_effective_date"] < item["date"]
        for item in path
    )
    assert result["timing_policy"]["same_day_signal_allowed"] is False


def test_taa_cost_and_snapshot_are_deterministic() -> None:
    first = run_taa_backtest(_synthetic_run(), _taa_request(transaction_cost_bps=100), {"passed": True})
    second = run_taa_backtest(_synthetic_run(), _taa_request(transaction_cost_bps=100), {"passed": True})

    assert first == second
    assert first["turnover_and_cost"]["total_turnover"] == pytest.approx(0.6)
    assert first["turnover_and_cost"]["total_transaction_cost"] == pytest.approx(0.005992)
    assert first["taa"]["metrics"]["total_return"] == pytest.approx(-0.005992)
    assert first["baseline"]["metrics"]["total_return"] == pytest.approx(0.0)
    assert first["snapshot_hash"]


def test_state_contributions_reconcile_to_gross_active_return() -> None:
    request = _taa_request()
    request["asset_returns"][1]["equity"] = 0.1
    result = run_taa_backtest(_synthetic_run(), request, {"passed": True})

    for period in result["weights"]:
        assert sum(period["state_contributions"].values()) == pytest.approx(
            period["gross_excess_return"]
        )
    assert sum(item["gross_excess_return"] for item in result["weights"]) == pytest.approx(
        result["excess"]["gross_active_return_sum"]
    )


def test_confidence_floor_falls_back_but_keeps_source_probabilities_for_audit() -> None:
    run = _synthetic_run()
    run["series"][0]["confidence"] = 0.2
    result = run_taa_backtest(run, _taa_request(), {"passed": True})
    period = result["weights"][1]

    assert period["fallback_to_base"] is True
    assert period["fallback_reason"] == "confidence_below_floor"
    assert period["probabilities"] == {"bull": 0.0, "bear": 1.0}
    assert period["allocation_probabilities"] == {}
    assert period["weights"] == {"bond": 0.5, "equity": 0.5}


def test_limits_scale_tilt_without_changing_weight_sum() -> None:
    request = _taa_request()
    request["limits"]["max_abs_tilt"] = 0.1
    result = run_taa_backtest(_synthetic_run(), request, {"passed": True})
    period = result["weights"][1]

    assert period["tilt_scale"] == pytest.approx(0.5)
    assert period["weights"] == pytest.approx({"bond": 0.6, "equity": 0.4})
    assert sum(period["weights"].values()) == pytest.approx(1.0)


def test_non_daily_returns_only_use_signals_available_at_period_start() -> None:
    run = _synthetic_run()
    run["series"] = [
        {
            "observation_date": "2024-01-14",
            "recognized_at": "2024-01-15",
            "effective_date": "2024-01-15",
            "probabilities": {"bull": 1.0},
            "probability_source": "model",
            "confidence": 1.0,
        }
    ]
    request = _taa_request()
    request["asset_returns"] = [
        {"period_start": "2024-01-01", "date": "2024-01-31", "equity": 0.02, "bond": 0.01},
        {"period_start": "2024-02-01", "date": "2024-02-29", "equity": 0.01, "bond": 0.00},
    ]
    request["periods_per_year"] = 12
    result = run_taa_backtest(run, request, {"passed": True})

    assert result["weights"][0]["fallback_reason"] == "no_effective_regime"
    assert result["weights"][0]["weights"] == {"bond": 0.5, "equity": 0.5}
    assert result["weights"][1]["regime_effective_date"] == "2024-01-15"
    assert result["weights"][1]["weights"] == pytest.approx({"bond": 0.3, "equity": 0.7})
    assert result["metrics_policy"]["annualization_periods"] == 12
    assert result["timing_policy"]["signal_rule"] == "regime.effective_date <= asset_return.period_start"


def test_stale_signal_falls_back_and_policy_is_hashed() -> None:
    request = _taa_request()
    request["asset_returns"] = [
        {"period_start": "2024-02-01", "date": "2024-02-02", "equity": 0.0, "bond": 0.0},
        {"period_start": "2030-01-01", "date": "2030-01-02", "equity": 0.0, "bond": 0.0},
    ]
    request["max_signal_age_days"] = 31
    result = run_taa_backtest(_synthetic_run(), request, {"passed": True})

    assert result["weights"][0]["fallback_to_base"] is False
    assert result["weights"][1]["fallback_reason"] == "stale_regime_signal"
    assert result["input_snapshot"]["parameters_hash"]
    assert result["parameters"]["max_signal_age_days"] == 31


def test_saa_and_taa_use_the_same_rebalancing_cost_when_no_tilt_is_applied() -> None:
    request = _taa_request(transaction_cost_bps=100)
    request["state_tilts"] = {
        "bull": {"equity": 0.0, "bond": 0.0},
        "bear": {"equity": 0.0, "bond": 0.0},
    }
    request["asset_returns"] = [
        {"date": "2024-01-02", "equity": 0.1, "bond": 0.0},
        {"date": "2024-01-03", "equity": -0.05, "bond": 0.0},
    ]
    result = run_taa_backtest(_synthetic_run(), request, {"passed": True})

    assert result["taa"]["nav"] == result["baseline"]["nav"]
    assert result["turnover_and_cost"]["total_transaction_cost"] == pytest.approx(
        result["turnover_and_cost"]["baseline_total_transaction_cost"]
    )
    assert result["baseline"]["policy"] == "periodic_target_weight_with_same_cost_model"


def test_taa_numerical_path_is_fixed_signature_njit_without_python_fallback() -> None:
    request = _taa_request()
    start = date(2010, 1, 1)
    request["asset_returns"] = [
        {
            "period_start": (start + timedelta(days=index)).isoformat(),
            "date": (start + timedelta(days=index + 1)).isoformat(),
            "equity": 0.0002 if index % 2 == 0 else -0.0001,
            "bond": 0.00005,
        }
        for index in range(5000)
    ]
    request["max_signal_age_days"] = 3650
    result = run_taa_backtest(_synthetic_run(), request, {"passed": True})
    execution = result["execution"]

    assert len(result["weights"]) == 5000
    assert execution["backend"] == "numba_njit_fixed_signature"
    assert execution["njit_required"] is True
    assert execution["nopython"] is True
    assert execution["object_mode"] == 0
    assert execution["python_fallback"] == 0
    assert execution["request_time_compilation"] == 0
    assert len(execution["compiled_signatures"]) == 5
    assert execution["kernel_fingerprint"]


def test_taa_njit_kernels_are_warmed_before_request_execution() -> None:
    status = warm_taa_numba_kernels()

    assert status["complete"] is True
    assert status["backend"] == "numba_njit_fixed_signature"
    assert status["nopython"] is True
    assert status["object_mode"] == 0
    assert status["python_fallback"] == 0
    assert status["request_time_compilation"] == 0
    assert all(status["kernel_signatures"].values())
    assert len(status["compiled_signatures"]) == 5


def _regime_rows(count: int = 48) -> list[dict[str, Any]]:
    start = date(2022, 1, 3)
    rows = []
    for index in range(count):
        observation = start + timedelta(days=index)
        value = 100.0 + index if index < count // 2 else 124.0 - 0.8 * (index - count // 2)
        rows.append(
            {
                "observation_date": observation.isoformat(),
                "available_at": observation.isoformat(),
                "value": value,
            }
        )
    return rows


def _definition() -> dict[str, Any]:
    return {
        "name": "TAA 下游复用测试",
        "target": {
            "kind": "inline",
            "series_id": "taa-fixture",
            "name": "TAA 固定样本",
            "frequency": "daily",
            "rows": _regime_rows(),
        },
        "features": {
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
                "bull_exit": 0.01,
                "bear_enter": -0.1,
                "bear_exit": -0.01,
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
        "usage_intent": "taa",
    }


def _asset_returns(*, missing: bool = False) -> list[dict[str, Any]]:
    rows = []
    for index, source in enumerate(_regime_rows()):
        row = {
            "date": source["observation_date"],
            "equity": 0.002 if index % 2 == 0 else -0.001,
            "bond": 0.0003,
        }
        if missing and index == 20:
            row.pop("bond")
        rows.append(row)
    return rows


def _route_request(*, missing: bool = False, cost: float = 20.0) -> dict[str, Any]:
    return {
        "asset_returns": _asset_returns(missing=missing),
        "base_weights": {"equity": 0.6, "bond": 0.4},
        "state_tilts": {
            "bull": {"equity": 0.15, "bond": -0.15},
            "sideways": {"equity": 0.0, "bond": 0.0},
            "bear": {"equity": -0.15, "bond": 0.15},
        },
        "transaction_cost_bps": cost,
        "confidence_floor": 0.0,
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    service = HistoricalRegimeService(tmp_path, tmp_path)
    monkeypatch.setattr(historical_regime_routes, "historical_regime_service", service)
    app = FastAPI()
    app.state.historical_regime_service = service
    app.include_router(historical_regime_routes.router)
    return TestClient(app)


def test_taa_route_is_registered_in_production_app() -> None:
    from app import app as production_app

    assert "/api/historical-regimes/runs/{run_id}/taa-backtest" in production_app.openapi()["paths"]


def test_meta_exposes_taa_backtest_timing_and_gate(client: TestClient) -> None:
    response = client.get("/api/historical-regimes/meta")

    assert response.status_code == 200
    contract = response.json()["taa_backtest"]
    assert contract["required_run"]["published_usage_any_of"] == ["taa", "formal_backtest"]
    assert "effective_date" in contract["timing_rule"]
    assert contract["missing_return_policy"] == "reject"


def _saved_run(client: TestClient, *, publish_usage: str | None) -> dict[str, Any]:
    service: HistoricalRegimeService = client.app.state.historical_regime_service
    definition = service.create_definition(_definition())
    run = service.run(
        {"id": definition["id"], "revision": definition["revision"]},
        "realtime",
    )
    if publish_usage is not None:
        service.publish(run["id"], publish_usage, "既有 v1 发布记录")
    stored = client.get(f"/api/historical-regimes/runs/{run['id']}")
    assert stored.status_code == 200, stored.text
    return stored.json()


@pytest.mark.parametrize("usage", ["taa", "formal_backtest"])
def test_taa_route_accepts_published_run_and_is_reproducible(client: TestClient, usage: str) -> None:
    run = _saved_run(client, publish_usage=usage)
    endpoint = f"/api/historical-regimes/runs/{run['id']}/taa-backtest"
    first = client.post(endpoint, json=_route_request())
    second = client.post(endpoint, json=_route_request())

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_payload = first.json()
    assert first_payload == second.json()
    assert first_payload["gate"]["passed"] is True
    assert first_payload["gate"]["publication_usages"] == [usage]
    assert first_payload["input_snapshot"]["regime_run_content_hash"] == run["content_hash"]
    assert first_payload["baseline"]["nav"]
    assert first_payload["taa"]["nav"]
    assert first_payload["weights"]
    assert first_payload["state_contributions"]
    assert first_payload["turnover_and_cost"]["total_transaction_cost"] >= 0
    assert all(
        item["regime_effective_date"] is None or item["regime_effective_date"] < item["date"]
        for item in first_payload["weights"]
    )


def test_taa_route_rejects_unpublished_run(client: TestClient) -> None:
    run = _saved_run(client, publish_usage=None)
    response = client.post(
        f"/api/historical-regimes/runs/{run['id']}/taa-backtest",
        json=_route_request(),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "TAA_RUN_NOT_PUBLISHED"


def test_taa_route_rejects_missing_asset_return_instead_of_using_zero(client: TestClient) -> None:
    run = _saved_run(client, publish_usage="taa")
    response = client.post(
        f"/api/historical-regimes/runs/{run['id']}/taa-backtest",
        json=_route_request(missing=True),
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "MISSING_ASSET_RETURN"
    assert "不能按 0 填充" in response.json()["detail"]["message"]


def test_transaction_cost_reduces_taa_nav(client: TestClient) -> None:
    run = _saved_run(client, publish_usage="taa")
    endpoint = f"/api/historical-regimes/runs/{run['id']}/taa-backtest"
    no_cost = client.post(endpoint, json=_route_request(cost=0.0))
    with_cost = client.post(endpoint, json=_route_request(cost=100.0))

    assert no_cost.status_code == 200, no_cost.text
    assert with_cost.status_code == 200, with_cost.text
    assert with_cost.json()["turnover_and_cost"]["total_transaction_cost"] > 0
    assert with_cost.json()["taa"]["nav"][-1]["value"] < no_cost.json()["taa"]["nav"][-1]["value"]


@pytest.mark.parametrize(
    ("changes", "expected_code"),
    [
        ({"immutable": False}, "REGIME_RUN_NOT_IMMUTABLE"),
        ({"mode": "retrospective"}, "TAA_REQUIRES_REALTIME_RUN"),
        ({"content_hash": "tampered"}, "REGIME_RUN_SNAPSHOT_MISMATCH"),
    ],
)
def test_taa_service_rejects_invalid_source_run_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, Any],
    expected_code: str,
) -> None:
    service = HistoricalRegimeService(tmp_path, tmp_path)
    saved = service.create_definition(_definition())
    run = service.run({"id": saved["id"], "revision": saved["revision"]}, "realtime")
    invalid_run = {**run, **changes}
    monkeypatch.setattr(service.runs, "get", lambda _: invalid_run)

    with pytest.raises(ValidationError) as error:
        service.taa_backtest(run["id"], _route_request())
    assert error.value.code == expected_code


@pytest.mark.parametrize(
    "causality_changes",
    [
        {"uses_future_data": True},
        {"realtime_eligible": False},
        {"repaints": True},
        {"is_causal": False},
    ],
)
def test_taa_service_rejects_every_noncausal_gate_dimension(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    causality_changes: dict[str, Any],
) -> None:
    service = HistoricalRegimeService(tmp_path, tmp_path)
    saved = service.create_definition(_definition())
    run = service.run({"id": saved["id"], "revision": saved["revision"]}, "realtime")
    invalid_run = {
        **run,
        "causality": {**run["causality"], **causality_changes},
    }
    monkeypatch.setattr(service.runs, "get", lambda _: invalid_run)

    with pytest.raises(ValidationError) as error:
        service.taa_backtest(run["id"], _route_request())
    assert error.value.code == "TAA_REQUIRES_CAUSAL_RUN"
