"""Synthetic immutable scenario artifacts for offline CMA integration tests.

These calibration numbers are test data, not evidence of market forecasting skill.
The real publication resolver, hash readers and NJIT calibrator are exercised.
"""
import copy
from datetime import date, timedelta
import json
from types import SimpleNamespace

from backend.historical_regimes.v2_contracts import parse_definition_v2, definition_content_hash
from backend.historical_regimes.v2_service import RegimeGraphV2Service, _stored_run_snapshot_hash, _content_hash
from backend.historical_regimes.repository import RegimeRunRepository
from backend.historical_regimes.reliability.service import ReliabilityService
from backend.historical_regimes.reliability.execution import model_binding_hash


def create_scenario_fixture(service, days):
    """Return (historical run, realtime run, exact published reference)."""
    root = service.cma.evidence.regime_root
    root.mkdir(parents=True, exist_ok=True)
    cutoff = str(date.today())
    previous = str(date.today() - timedelta(days=2))
    labels = [{"id": "growth", "label": "增长", "role": "positive", "color": "#16a34a", "order": 1},
              {"id": "stress", "label": "压力", "role": "negative", "color": "#dc2626", "order": 2}]
    definition = parse_definition_v2({"schema_version": "2.0", "id": "regime-scenario-reference", "revision": 1,
        "name": "离线情景参考", "states": labels,
        "graph": {"nodes": [{"id": "source", "type": "source.inline", "parameters": {
            "rows": [{"observation_date": cutoff, "available_at": cutoff, "value": 1.}], "frequency": "daily"}},
            {"id": "classifier", "type": "model.threshold", "parameters": {"upper": .1, "lower": -.1},
             "inputs": {"value": {"node_id": "source", "port": "value"}}}],
            "outputs": {"state": {"node_id": "classifier", "port": "state"}}},
        "study": {"purpose": "historical_reference", "family": "market_trend"},
        "usage_intent": "research_display"})
    historical = {"id": "regime-run-scenario-reference", "name": "增长与压力历史参考", "schema_version": "2.0",
        "immutable": True, "mode": "retrospective", "frequency": "daily", "as_of": cutoff,
        "definition_id": definition.id, "definition_revision": 1,
        "definition": definition.model_dump(mode="json"), "definition_snapshot_hash": definition_content_hash(definition),
        "states": [s.model_dump(mode="json") for s in definition.states], "application_bindings": [],
        "publications": [], "series": [{"observation_date": str(day.date()), "available_at": str(day.date()),
             "recognized_at": str(day.date()), "state_id": labels[(i // 8) % 2]["id"]} for i, day in enumerate(days)]}
    historical["content_hash"] = _stored_run_snapshot_hash(historical)
    reference = {"run_id": historical["id"], "publication_id": "pub-scenario-reference", "content_hash": historical["content_hash"]}
    historical["publications"] = [{"id": reference["publication_id"], "run_id": historical["id"],
        "run_content_hash": historical["content_hash"], "definition_revision": 1, "usage": "research_display",
        "published_at": previous + "T00:00:00Z", "fit_mode": "retrospective"}]
    calibration_id = "reliability-" + "a" * 64
    realtime_definition = parse_definition_v2({**definition.model_dump(mode="json"), "id": "regime-scenario-current", "default_mode": "realtime",
        "study": {"purpose": "realtime_recognition", "family": "market_trend", "reference": reference,
                  "calibration_id": calibration_id}})
    realtime = {**copy.deepcopy(historical), "id": "regime-run-scenario-current", "name": "已校准当前判断",
        "mode": "realtime", "definition_id": realtime_definition.id,
        "definition": realtime_definition.model_dump(mode="json"),
        "definition_snapshot_hash": definition_content_hash(realtime_definition),
        "causality": {"is_causal": True, "uses_future_data": False, "repaints": False, "realtime_eligible": True},
        "publications": [], "series": [{"observation_date": str(days[-1].date()), "available_at": str(days[-1].date()),
            "recognized_at": str(days[-1].date()), "effective_date": str(days[-1].date()),
            "state_id": "growth", "probabilities": {"growth": .7, "stress": .3}, "confidence": .7}]}
    realtime["content_hash"] = _stored_run_snapshot_hash(realtime)
    realtime["publications"] = [{"id": "pub-scenario-current", "run_id": realtime["id"],
        "run_content_hash": realtime["content_hash"], "definition_revision": 1, "usage": "taa",
        "published_at": previous + "T00:00:00Z", "gate": "comprehensive_formal_gate_passed"}]
    run_path = root / "historical_regime_runs.json"
    existing = json.loads(run_path.read_text(encoding="utf-8"))["items"] if run_path.exists() else []
    existing = [row for row in existing if row.get("id") not in {historical["id"], realtime["id"]}]
    run_path.write_text(json.dumps({"items": [*existing, historical, realtime]}), encoding="utf-8")
    definitions = {definition.id: definition.model_dump(mode="json"), realtime_definition.id: realtime_definition.model_dump(mode="json")}
    graph = SimpleNamespace(workspace_data_dir=root, runs=RegimeRunRepository(root / "historical_regime_runs.json"),
        get_definition=lambda identifier, revision: definitions[identifier], _hydrate_run=copy.deepcopy,
        reference_quality=SimpleNamespace(catalog=lambda: {"items": []}))
    graph.reliability = ReliabilityService(graph)
    graph.hydrate_run_snapshot = lambda raw: RegimeGraphV2Service.hydrate_run_snapshot(graph, raw)
    graph.resolve_taa_run = lambda identifier: RegimeGraphV2Service.resolve_taa_run(graph, identifier)
    artifact = {"id": calibration_id, "created_at": previous + "T00:00:00Z", "immutable": True,
        "request": {"definition_id": realtime_definition.id, "reference": reference},
        "report": {"states": historical["states"],
            "verification": {"recognition_ready": True, "verified_states": ["growth", "stress"]},
            "lineage": {"model_binding_hash": model_binding_hash(realtime_definition),
                        "state_mapping": {row["id"]: row["id"] for row in labels}},
            "calibration": {"deployment_eligible": True, "available_from": previous,
                "expires_on": str(date.today() + timedelta(days=10)), "confidence_floor": .1,
                "method": "class_frequency", "parameters": {"counts": [[8., 2.], [3., 7.]], "temperature": None}}}}
    artifact["content_hash"] = _content_hash(artifact)
    store = graph.reliability._store(calibration_id)
    with store.locked():
        store.write_unlocked({"items": [artifact]})
    service.cma.evidence.prepare_regime_reader()
    service.cma.evidence.bind_scenario_graph(graph)
    return historical, realtime, reference
