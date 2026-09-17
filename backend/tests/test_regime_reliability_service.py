import copy
from datetime import date, datetime, timedelta, timezone
import numpy as np
import pytest
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.reliability.references import resolve_reference
from historical_regimes.reliability.execution import probability_provenance
from historical_regimes.v2_contracts import parse_definition_v2
from test_historical_regime_v2 import _definition
from test_historical_regime_v2_p1 import _upload_source


@pytest.fixture
def case(tmp_path):
    graph = RegimeGraphV2Service(tmp_path,tmp_path)
    rows = [{"observation_date":(date(2020,1,1)+timedelta(days=i)).isoformat(),
             "available_at":(date(2020,1,1)+timedelta(days=i)).isoformat(),
             "value":100+10*np.sin(i/4)} for i in range(240)]
    payload = _definition(rows)
    payload["graph"]["nodes"][0] = _upload_source(graph,rows)
    payload["study"] = {"purpose":"historical_reference","family":"market_trend"}
    saved = graph.create_definition(payload)
    plan = graph.prepare(saved)
    run = graph.run_saved({"schema_version":"2.0","id":saved["id"],"revision":1},
                          "retrospective","2020-08-27",plan["compile_token"])
    published = graph.publish(run["id"],"research_display")
    ref = {"run_id":run["id"],"publication_id":published["publication"]["id"],"content_hash":run["content_hash"]}
    payload["study"] = {"purpose":"realtime_recognition","family":"market_trend","reference":ref}
    model = graph.create_definition(payload)
    graph.prepare(model)
    request = {"definition_id":model["id"],"revision":1,"reference":ref,
               "policy":{"calibration_end":"2020-04-30","validation_end":"2020-06-30",
                         "test_end":"2020-08-27","minimum_samples":2,"minimum_class_samples":1,"minimum_segments":1}}
    return graph,model,ref,request


def test_reference_projection_integrity_and_mode(case):
    graph,model,ref,request = case
    assert graph.reliability.references()["items"][0]["run_id"] == ref["run_id"]
    assert resolve_reference(graph,ref)[0]["series"]
    for field in ("publication_id","content_hash"):
        invalid = {**ref,field:"b"*64}
        with pytest.raises(Exception):
            resolve_reference(graph,invalid)
    raw = graph.runs.get(ref["run_id"])
    with graph.runs.store.locked():
        content=graph.runs.store.read_unlocked()
        content["items"][0]["name"]="tampered"
        graph.runs.store.write_unlocked(content)
    assert graph.reliability.references() == {"items":[]}


def test_preview_actual_executor_blocks_and_reference_semantics(case):
    graph,model,ref,request=case
    preview = graph.reliability.preview(request)
    report = preview["report"]
    assert report["lineage"]["prediction_method"] == "fixed_rule_causal_replay"
    assert report["status"] == "retrospective_only"
    assert report["calibration"]["deployment_eligible"] is False
    assert report["calibration"]["method"] == "class_frequency"
    assert report["calibration"]["label_known_at"] >= datetime.now(timezone.utc).date().isoformat()
    assert set(report["sample"]["blocks"]) == {"calibration","validation","test"}
    assert report["execution"]["python_fallback"] == 0
    assert report["confidence_interval"]["status"] == "unavailable"
    assert all(p["calibrated_confidence"] is None or p["calibrated_confidence"] ==
               p["calibrated_probabilities"][p["predicted_state"]] for p in report["points"])


def test_preview_rejects_wrong_axis_and_temperature(case):
    graph,model,ref,request=case
    invalid=copy.deepcopy(request);invalid["policy"]["calibration_method"]="temperature"
    with pytest.raises(Exception,match="temperature"):
        graph.reliability.preview(invalid)
    model["graph"]["nodes"][0]["parameters"]["frequency"]="monthly"
    graph.update_definition(model["id"],1,model)
    invalid={**request,"revision":2}
    with pytest.raises(Exception,match="同频率"):
        graph.reliability.preview(invalid)


def test_confirmation_immutable_idempotent_and_no_client_reports(case):
    graph,model,ref,request=case
    preview=graph.reliability.preview(request)
    body={"request":preview["request"],"preview_hash":preview["preview_hash"]}
    confirmed=graph.reliability.confirm(body)
    assert graph.reliability.confirm(body)==confirmed
    assert graph.reliability.get(confirmed["id"])==confirmed
    assert len(graph.reliability.catalog()["items"])==1
    assert "series" not in graph.runs.get(ref["run_id"])
    with pytest.raises(ValueError):
        graph.reliability.confirm({**body,"report":{"status":"eligible"}})
    changed=copy.deepcopy(body);changed["request"]["policy"]["bins"]=7
    with pytest.raises(Exception,match="不一致"):
        graph.reliability.confirm(changed)
    with pytest.raises(Exception):
        graph.reliability.get("../../outside")
    store=graph.reliability._store(confirmed["id"])
    with store.locked():
        data=store.read_unlocked();data["items"][0]["report"]["status"]="eligible";store.write_unlocked(data)
    with pytest.raises(Exception,match="完整性"):
        graph.reliability.get(confirmed["id"])


def test_probability_provenance_tracks_confirmation_not_unrelated_model():
    payload=_definition()
    definition=parse_definition_v2(payload)
    provenance=probability_provenance(definition)
    assert provenance["final_state_path"]==["confirmed","classifier"]
    assert not provenance["temperature_supported"]


def test_preview_expiry_request_prepare_and_tamper(case):
    graph,model,ref,request=case
    preview=graph.reliability.preview(request)
    graph.reliability._previews[preview["preview_hash"]]["expires"]=0
    with pytest.raises(Exception,match="过期"):
        graph.reliability.confirm({"request":preview["request"],"preview_hash":preview["preview_hash"]})
    graph._plans.clear()
    with pytest.raises(Exception):
        graph.reliability.preview(request)


def test_routes_real_preview_and_error_envelope(case,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services import historical_regime_routes as routes
    graph,_,_,request=case
    monkeypatch.setattr(routes,"regime_graph_v2_service",graph)
    app=FastAPI();app.include_router(routes.router)
    with TestClient(app) as client:
        assert client.get("/api/historical-regimes/references").status_code==200
        response=client.post("/api/historical-regimes/reliability/preview",json=request)
        assert response.status_code==200,response.text
        body=response.json()
        confirmed=client.post("/api/historical-regimes/reliability/confirm",json={"request":body["request"],"preview_hash":body["preview_hash"]})
        assert confirmed.status_code==200,confirmed.text
        assert client.get("/api/historical-regimes/reliability/reports/"+confirmed.json()["id"]).json()==confirmed.json()
        evidence = client.get("/api/historical-regimes/reliability/reports/"+confirmed.json()["id"]+"/recognition-evidence")
        assert evidence.status_code == 200, evidence.text
        contract = evidence.json()
        assert contract["kind"] == "regime_recognition_evidence"
        assert contract["production_eligible"] is False
        assert contract["unverified_state_policy"] == "do_not_authorize_unverified_states"
        legacy = client.get("/api/historical-regimes/reliability/reports/"+confirmed.json()["id"]+"/cma-evidence")
        assert legacy.status_code == 409
        assert legacy.json()["detail"]["code"] == "REALTIME_CMA_EVIDENCE_REMOVED"
        assert client.get("/api/historical-regimes/reliability/catalog").status_code==200
        assert client.post("/api/historical-regimes/reliability/confirm",json={"report":{"status":"eligible"}}).status_code in (400,422)


def test_latent_oos_reuses_frozen_fold_executor(case):
    graph,model,ref,request=case
    model["graph"]={"nodes":[model["graph"]["nodes"][0],
        {"id":"features","type":"feature.matrix","parameters":{},"inputs":{"feature_1":{"node_id":"source","port":"value"}}},
        {"id":"model","type":"model.gmm","parameters":{"components":3,"initial_train_size":30,"iterations":5},
         "inputs":{"features":{"node_id":"features","port":"features"}}},
        {"id":"confirmed","type":"post.confirmation","parameters":{"confirmation":2,"min_duration":2},
         "inputs":{"state":{"node_id":"model","port":"state"}}}],
        "outputs":{"state":{"node_id":"confirmed","port":"state"},"probabilities":{"node_id":"model","port":"probabilities"}}}
    updated=graph.update_definition(model["id"],1,model)
    graph.prepare(updated)
    preview=graph.reliability.preview({**request,"revision":2})
    report=preview["report"]
    assert report["lineage"]["prediction_method"]=="expanding_walk_forward"
    assert report["calibration"]["method"]=="temperature"
    assert report["lineage"]["folds"]
    for fold in report["lineage"]["folds"]:
        assert fold["status"]=="completed"
        audit=fold["model_audits"]["model"]
        assert audit["walk_forward_refit"] and audit["label_mapping_locked_before_classification"]
        assert audit["training_cutoff_available_at"]==fold["training_as_of"]
        assert audit["classification_start_index"]>audit["training_end_index"]


def test_exact_alignment_unknown_reference_and_prediction_abstention(case):
    from historical_regimes.reliability.report import build_report
    from historical_regimes.reliability.contracts import Policy
    graph,model,ref,request=case
    reference,publication=resolve_reference(graph,ref)
    predictions=copy.deepcopy(reference["series"])
    missing_date=predictions.pop(10)["observation_date"]
    predictions[12]["state_id"]="not_a_state"
    reference["series"][20]["state_id"]="unclassified"
    definition=parse_definition_v2(model)
    lineage={"probability_provenance":{"temperature_supported":False,"type":"deterministic_state"},"temporal_audit":{}}
    report=build_report(definition,reference,publication,predictions,lineage,Policy(**request["policy"]),"2020-08-27")
    assert report["sample"]["missing_prediction"]==1
    assert report["sample"]["invalid_prediction_labels"]==1
    assert sum(map(sum,report["classification"]["confusion"]))==len(reference["series"])-report["sample"]["unknown_reference"]
    missing=next(p for p in report["points"] if p["observation_date"]==missing_date)
    assert missing["predicted_state"] is None and missing["calibrated_confidence"] is None
    assert missing["decision_status"]=="abstained"
    predictions.reverse()
    with pytest.raises(Exception,match="严格递增"):
        build_report(definition,reference,publication,predictions,lineage,Policy(**request["policy"]),"2020-08-27")


@pytest.mark.parametrize("prediction", ["unclassified", "bull"])
def test_reference_sample_size_does_not_replace_usable_calibration_predictions(case, prediction):
    from historical_regimes.reliability.report import build_report
    from historical_regimes.reliability.contracts import Policy

    graph, model, ref, request = case
    reference, publication = resolve_reference(graph, ref)
    states = [state["id"] for state in reference["states"]]
    for index, point in enumerate(reference["series"]):
        point["state_id"] = states[(index // 10) % len(states)]
    predictions = copy.deepcopy(reference["series"])
    for point in predictions:
        point["state_id"] = prediction
        point["recognized_at"] = point["observation_date"]
    lineage = {"probability_provenance": {"temperature_supported": False,
               "type": "deterministic_state"}, "temporal_audit": {}}
    report = build_report(parse_definition_v2(model), reference, publication, predictions,
                          lineage, Policy(**request["policy"]), "2020-08-27")
    assert report["sample"]["blocks"]["calibration"]["samples"] >= 2
    assert min(report["sample"]["blocks"]["calibration"]["per_class"].values()) > 0
    assert report["calibration"]["fitted"] is False
    assert "insufficient_usable_calibration_predictions" in report["calibration"]["reasons"]
    assert all(point["calibrated_confidence"] is None for point in report["points"])


def test_reference_series_bytes_tampering_and_source_change(case):
    graph,model,ref,request=case
    preview=graph.reliability.preview(request)
    raw=graph.runs.get(ref["run_id"])
    manifest=raw["series_artifact"]
    # Stored reference hash binds the external series checksum; hydrate must check it.
    assert manifest.get("checksum")
    with graph.runs.store.locked():
        data=graph.runs.store.read_unlocked()
        data["items"][0]["series_artifact"]["checksum"]="0"*64
        graph.runs.store.write_unlocked(data)
    with pytest.raises(Exception):
        graph.reliability.confirm({"request":preview["request"],"preview_hash":preview["preview_hash"]})
