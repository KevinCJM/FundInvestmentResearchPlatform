"""Forward qualification integration, explicit legacy hashes and TAA times."""
import copy
from datetime import datetime, timezone

import pytest
from historical_regimes.reliability.contracts import Study
from historical_regimes.reliability.consumer import calibrated_output
from historical_regimes.reliability.execution import model_binding_hash
from historical_regimes.reliability import kernels
from historical_regimes.v2_contracts import parse_definition_v2
from test_historical_regime_v2 import _definition
from test_regime_reliability_consumer import _eligible_fixture


def test_absent_qualification_preserves_study_payload_and_binding():
    payload = _definition()
    original = {"purpose": "realtime_recognition", "family": "market_trend",
                "reference": {"run_id": "r", "publication_id": "p", "content_hash": "a" * 64},
                "state_mapping": None, "calibration_id": None}
    payload["study"] = original
    definition = parse_definition_v2(payload)
    assert definition.model_dump(mode="json")["study"] == original
    before = model_binding_hash(definition)
    payload["study"] = {**original, "calibration_id": "calibration", "qualification_id": "forward-" + "1" * 32}
    assert model_binding_hash(parse_definition_v2(payload)) == before
    payload["graph"]["nodes"][2]["parameters"]["window"] = 4
    assert model_binding_hash(parse_definition_v2(payload)) != before
    with pytest.raises(ValueError):
        Study(purpose="historical_reference", family="market_trend", qualification_id="q")
    with pytest.raises(ValueError):
        Study(purpose="realtime_recognition", family="market_trend", qualification_id="q")


def test_new_verified_forward_window_does_not_rewrite_expired_candidate():
    kernels.warm()
    run = _eligible_fixture()
    context = run["_reliability"]
    context["artifact"]["report"]["calibration"]["deployment_eligible"] = False
    context["qualification"] = {"id": "forward-q", "status": "qualified",
        "available_from": "2025-01-01T12:00:00+00:00", "expires_at": "2025-02-01T00:00:00+00:00"}
    context["verified_at"] = "2025-01-10T12:00:00+00:00"
    run["definition"]["study"]["qualification_id"] = "forward-q"
    point = {**run["series"][0], "observation_date": "2025-01-02",
             "recognized_at": "2025-01-02", "effective_date": "2025-01-02"}
    original = copy.deepcopy(run)
    q, confidence, reason = calibrated_output(run, point, "2025-01-02", ["bull", "bear"])
    assert reason is None and confidence == .2 and q == {"bull": .2, "bear": .8}
    assert run == original
    for day, reason in [("2025-01-01", "calibration_not_yet_available"),
                        ("2025-02-01", "calibration_expired"),
                        ("2025-01-11", "calibration_future_decision")]:
        assert calibrated_output(run, point, day, ["bull", "bear"])[2] == reason
    context["qualification"]["status"] = "pending"
    assert calibrated_output(run, point, "2025-01-02", ["bull", "bear"])[2] == "calibration_qualification_mismatch"


def test_service_and_routes_expose_forward_pending_without_fabricating_data(tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from historical_regimes.v2_service import RegimeGraphV2Service
    from services import historical_regime_routes as routes
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    assert graph.prospective._audit()["python_fallback"] == 0
    monkeypatch.setattr(routes, "regime_graph_v2_service", graph)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.get('/api/historical-regimes/prospective/catalog').json() == {"items": []}
        bad = client.post('/api/historical-regimes/prospective/register', json={
            "calibration_id": "reliability-" + "a" * 64,
            "eligible": True, "registered_at": "2010-01-01"})
        assert bad.status_code in (400, 422)
        bad = client.post('/api/historical-regimes/prospective/forward-bad/capture', json={"as_of": "2010-01-01"})
        assert bad.status_code in (400, 422)
        assert client.get('/api/historical-regimes/prospective/protocols/forward-bad/progress').status_code == 404
