"""Offline forward clocks and real temporary Parquet -> graph -> journal path."""
import copy
import json
from datetime import datetime, date, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from historical_regimes.v2_service import RegimeGraphV2Service, _content_hash
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.reliability.prospective import ProspectiveService, ForwardPolicy, install
from historical_regimes.reliability import prospective_kernels as pk
from test_historical_regime_v2 import _definition


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


@pytest.fixture
def setup(tmp_path, request):
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    # Existing legacy unbound index definitions can grow; current create_definition
    # intentionally binds immutable snapshots. Seed that supported stored contract
    # through the real version repository, with no patched source/execution gates.
    def save_live_definition(payload):
        if getattr(request, "param", None) == "bound":
            return graph.create_definition(payload)
        fields = parse_definition_v2(payload).model_dump(mode="json", exclude={"id", "revision", "created_at", "updated_at"})
        return graph.definitions.create(fields)
    registration = datetime.now(timezone.utc).replace(hour=12, minute=0, second=0, microsecond=0) + timedelta(days=2)
    start = registration.date() - timedelta(days=180)
    rows = []

    def extend(end):
        next_day = start + timedelta(days=len(rows))
        while next_day <= end:
            i = (next_day - start).days
            rows.append({"ts_code": "000300.SH", "trade_date": next_day.isoformat(),
                         "available_at": next_day.isoformat(), "close": (300., 155., 10.)[(i // 5) % 3]})
            next_day += timedelta(days=1)
        pd.DataFrame(rows).to_parquet(tmp_path / "index_daily_df.parquet", index=False)

    extend(registration.date() - timedelta(days=1))
    definition = _definition()
    definition["graph"] = {"nodes": [
        {"id": "source", "type": "source.index", "parameters": {
            "ts_code": "000300.SH", "source_api": "index_daily", "field": "close", "frequency": "daily"}},
        {"id": "classifier", "type": "model.threshold", "parameters": {"upper": 200., "lower": 50.},
         "inputs": {"value": {"node_id": "source", "port": "value"}}}],
        "outputs": {"state": {"node_id": "classifier", "port": "state"}}}
    definition["study"] = {"purpose": "historical_reference", "family": "market_trend"}
    historical = save_live_definition(definition)
    plan = graph.prepare(historical)

    def reference(end, published_at=None):
        run = graph.run_saved({"schema_version": "2.0", "id": historical["id"], "revision": 1},
                              "retrospective", end.isoformat(), plan["compile_token"])
        pub = graph.publish(run["id"], "research_display")["publication"]
        if published_at:
            # Trusted publishing clock seam only. Run snapshots remain genuine.
            with graph.runs.store.locked():
                data = graph.runs.store.read_unlocked()
                for item in data["items"]:
                    if item["id"] == run["id"]:
                        for p in item["publications"]:
                            if p["id"] == pub["id"]:
                                p["published_at"] = published_at.isoformat()
                graph.runs.store.write_unlocked(data)
        return {"run_id": run["id"], "publication_id": pub["id"], "content_hash": run["content_hash"]}

    ref = reference(registration.date() - timedelta(days=1))
    definition["study"] = {"purpose": "realtime_recognition", "family": "market_trend", "reference": ref}

    def candidate(mapping=None, latent=False):
        payload = copy.deepcopy(definition)
        if latent:
            payload["graph"]["nodes"].append({
                "id": "features", "type": "feature.matrix", "parameters": {},
                "inputs": {"feature_1": {"node_id": "source", "port": "value"}}})
            payload["graph"]["nodes"][1] = {
                "id": "classifier", "type": "model.gmm",
                "parameters": {"components": 3, "initial_train_size": 30, "iterations": 60},
                "inputs": {"features": {"node_id": "features", "port": "features"}}}
            payload["graph"]["outputs"]["probabilities"] = {"node_id": "classifier", "port": "probabilities"}
        if mapping:
            payload["study"]["state_mapping"] = mapping
        model = save_live_definition(payload)
        graph.prepare(model)
        preview = graph.reliability.preview({"definition_id": model["id"], "revision": 1, "reference": ref,
            "policy": {"calibration_end": (start + timedelta(days=139 if latent else 59)).isoformat(),
                       "validation_end": (start + timedelta(days=159 if latent else 119)).isoformat(),
                       "test_end": (start + timedelta(days=179)).isoformat(),
                       "minimum_samples": 30, "minimum_class_samples": 5, "minimum_segments": 3}})
        assert preview["report"]["calibration"]["fitted"], json.dumps(preview["report"]["sample"]["blocks"]["calibration"])
        artifact = graph.reliability.confirm({"request": preview["request"], "preview_hash": preview["preview_hash"]})
        return artifact, model

    clock = Clock(registration)
    service = ProspectiveService(graph, graph.reliability, clock=clock)
    service.warm()
    policy = {"observation_window": 60, "minimum_observations": 60, "minimum_class_observations": 5,
              "minimum_class_complete_regimes": 1, "block_size": 5, "minimum_complete_blocks": 3,
              "minimum_coverage": 1., "minimum_agreement": .8, "minimum_brier_improvement": .01}
    return dict(graph=graph, service=service, clock=clock, registration=registration, candidate=candidate,
                extend=extend, reference=reference, policy=policy, rows=rows, root=tmp_path, original_ref=ref)


def register(case, mapping=None):
    artifact, model = case["candidate"](mapping)
    protocol = case["service"].register({"calibration_id": artifact["id"], "policy": case["policy"]})
    return protocol, artifact, model


def capture_days(case, protocol, n=60, skip=()):
    observations = []
    for day in range(1, n + 1):
        d = case["registration"].date() + timedelta(days=day)
        case["extend"](d)
        case["clock"].value = datetime.combine(d + timedelta(days=1), datetime.min.time(), timezone.utc) + timedelta(hours=12)
        if day not in skip:
            result = case["service"].capture(protocol["id"])
            assert result["status"] == "captured", result
            observations.append(result["observation"])
    return observations


def assess(case, protocol):
    end = case["registration"].date() + timedelta(days=60)
    case["clock"].value += timedelta(seconds=1)
    ref = case["reference"](end, case["clock"].value)
    case["clock"].value += timedelta(seconds=1)
    return case["service"].assess(protocol["id"], {"reference": ref})


def test_registration_frozen_no_backdated_inputs_and_idempotence(setup):
    c = setup
    protocol, artifact, _ = register(c)
    assert protocol["status"] == "pending"
    assert protocol["calibrator"]["deployment_eligible"] is False
    assert c["service"].get_protocol(protocol["id"]) == protocol
    assert c["service"].catalog() == {"items": [protocol]}
    assert c["service"].register({"calibration_id": artifact["id"], "policy": c["policy"]}) == protocol
    with pytest.raises(Exception, match="更换门槛"):
        c["service"].register({"calibration_id": artifact["id"], "policy": {**c["policy"], "minimum_agreement": .9}})
    with pytest.raises(ValueError):
        c["service"].register({"calibration_id": artifact["id"], "start_date": "2000-01-01"})
    assert c["service"].capture(protocol["id"])["reason"] == "no_post_registration_day"
    assert c["graph"].reliability.get(artifact["id"])["content_hash"] == artifact["content_hash"]


def test_artifact_tamper_and_unfitted(setup):
    c = setup
    artifact, _ = c["candidate"]()
    store = c["graph"].reliability._store(artifact["id"])
    with store.locked():
        data = store.read_unlocked()
        data["items"][0]["report"]["calibration"]["fitted"] = False
        store.write_unlocked(data)
    with pytest.raises(Exception):
        c["service"].register({"calibration_id": artifact["id"]})
    with store.locked():
        data = store.read_unlocked()
        data["items"][0]["content_hash"] = _content_hash({k: v for k, v in data["items"][0].items() if k != "content_hash"})
        store.write_unlocked(data)
    with pytest.raises(Exception, match="原始预览"):
        c["service"].register({"calibration_id": artifact["id"]})


def test_real_capture_latest_duplicate_no_backfill_and_source_revision(setup):
    c = setup
    protocol, _, _ = register(c)
    obs = capture_days(c, protocol, 3, skip=(1, 2))[0]
    assert obs["observation_date"] == (c["registration"].date() + timedelta(days=3)).isoformat()
    assert obs["as_of"] < obs["recorded_at"][:10]
    assert obs["confidence"] == obs["probabilities"][obs["chosen"]]
    assert obs["data_snapshots"]["source"]["fingerprint"]
    assert obs["temporal_audit"]["verified"]
    assert c["service"].capture(protocol["id"])["status"] == "duplicate"
    c["rows"][0]["close"] = 100.
    pd.DataFrame(c["rows"]).to_parquet(c["root"] / "index_daily_df.parquet", index=False)
    with pytest.raises(Exception, match="修订"):
        c["service"].capture(protocol["id"])


def test_perfect_future_qualification_consumer_expiry_and_bindings(setup):
    c = setup
    protocol, artifact, _ = register(c)
    captures = capture_days(c, protocol)
    result = assess(c, protocol)
    assert result["status"] == "qualified", result["reasons"]
    assert result["metrics"]["paired_samples"] == 60
    assert result["metrics"]["complete_blocks"] == 12
    assert result["metrics"]["classification"]["accuracy"] == 1.
    assert result["statistical_significance_claim"] is False
    assert c["service"].get_qualification(result["id"]) == result
    args = (result["id"], artifact["id"], protocol["model_binding_hash"])
    c["clock"].value += timedelta(seconds=1)
    assert c["service"].verify_qualification(*args)["status"] == "qualified"
    with pytest.raises(Exception, match="过去"):
        c["service"].verify_qualification(*args, decision_at=captures[-1]["observation_date"])
    with pytest.raises(Exception, match="过去"):
        c["service"].verify_qualification(*args, decision_at=protocol["registration_day"])
    with pytest.raises(Exception, match="不属于"):
        c["service"].verify_qualification(result["id"], artifact["id"], "0"*64)
    with pytest.raises(Exception, match="不属于"):
        c["service"].verify_qualification(*args, reference_definition={})
    c["rows"][0]["close"] = 999.
    pd.DataFrame(c["rows"]).to_parquet(c["root"] / "index_daily_df.parquet", index=False)
    with pytest.raises(Exception, match="修订"):
        c["service"].verify_qualification(*args)
    c["clock"].value = datetime.fromisoformat(result["expires_at"])
    with pytest.raises(Exception, match="过期"):
        c["service"].verify_qualification(*args)
    assert c["graph"].reliability.get(artifact["id"])["report"]["calibration"]["deployment_eligible"] is False


def test_poor_mapping_only_qualifies_states_that_independently_pass(setup):
    c = setup
    protocol, _, _ = register(c, {"bull": "bear", "bear": "bull", "sideways": "sideways"})
    captures = capture_days(c, protocol)
    assert any(o["confidence"] < max(o["probabilities"]) for o in captures)
    result = assess(c, protocol)
    assert result["metrics"]["global_agreement_meets_policy"] is False
    assert result["status"] in {"qualified", "rejected"}
    by_state = {row["state_id"]: row for row in result["state_evidence"]}
    assert by_state["bull"]["status"] != "qualified"
    assert by_state["bear"]["status"] != "qualified"
    assert {"bull", "bear"} <= set(result["fallback_states"])
    if result["status"] == "qualified":
        assert result["outcome"] == "partially_qualified"
        assert result["qualified_states"] == ["sideways"]
    assert c["service"].capture(protocol["id"])["status"] == "closed"


def test_missing_captures_not_compressed_and_qualification_tamper(setup):
    c = setup
    protocol, artifact, _ = register(c)
    capture_days(c, protocol, skip=(10, 20, 30))
    result = assess(c, protocol)
    assert result["status"] == "rejected"
    assert result["metrics"]["axis_rows"] == 60
    assert result["metrics"]["paired_samples"] == 57
    assert result["metrics"]["complete_blocks"] == 9
    assert "insufficient_capture_coverage" in result["reasons"]
    with c["service"].journal.locked():
        data = c["service"].journal.read_unlocked()
        data["items"][-1]["status"] = "qualified"
        data["items"][-1]["content_hash"] = _content_hash({k: v for k, v in data["items"][-1].items() if k not in {"content_hash", "signature"}})
        c["service"].journal.write_unlocked(data)
    with pytest.raises(Exception, match="日志"):
        c["service"].verify_qualification(result["id"], artifact["id"], protocol["model_binding_hash"])


def test_pending_short_window_stale_and_reference_identity(setup):
    c = setup
    protocol, _, _ = register(c)
    capture_days(c, protocol, 3)
    result = assess(c, protocol)
    assert result["status"] == "pending"
    assert result["metrics"]["paired_samples"] == 3
    assert "future_observation_window_incomplete" in result["reasons"]
    c["clock"].value += timedelta(days=10)
    assert c["service"].capture(protocol["id"])["status"] == "stale"
    bad = {**result["reference"], "content_hash": "0"*64}
    with pytest.raises(Exception):
        c["service"].assess(protocol["id"], {"reference": bad})


def test_route_inputs_forbid_client_time_and_warm_is_required(setup):
    c = setup
    fresh = ProspectiveService(c["graph"], c["graph"].reliability, clock=c["clock"])
    artifact, _ = c["candidate"]()
    with pytest.raises(RuntimeError, match="NOT_READY"):
        fresh.register({"calibration_id": artifact["id"]})
    app = FastAPI()
    install(app, lambda: c["service"], lambda fn, *args: fn(*args))
    with TestClient(app) as client:
        assert client.post("/api/historical-regimes/prospective/register", json={"calibration_id": artifact["id"], "registered_at": "2000-01-01"}).status_code == 422
        response = client.post("/api/historical-regimes/prospective/register", json={"calibration_id": artifact["id"], "policy": c["policy"]})
        assert response.status_code == 200, response.text
        protocol_id = response.json()["id"]
        assert client.post(f"/api/historical-regimes/prospective/{protocol_id}/capture", json={"as_of": "2000-01-01"}).status_code == 422
        assert client.get("/api/historical-regimes/prospective/catalog").json()["items"][0]["id"] == response.json()["id"]


def test_kernel_independent_oracle_readonly_stride_and_missing():
    pk.warm()
    n = 60
    day_owner = np.arange(2*n, dtype=np.int64)
    days = day_owner[::2]
    y_owner = np.repeat(np.arange(12) % 3, 10).astype(np.int64)
    y = y_owner[::2]
    q_owner = np.zeros((n, 6))
    q = q_owner[:, ::2]
    q[np.arange(n), y] = .8
    q[q == 0.] = .1
    base = np.array([1/3, 1/3, 1/3])
    for value in (days, y, q, base):
        value.flags.writeable = False
    signatures = tuple(pk.forward_blocks_kernel.signatures)
    support, cycles, blocks, pairs, coverage, count, worst = pk.forward_blocks_kernel(days, y, y, q, base, 5, 7)
    oracle = ((base - np.eye(3)[y])**2).sum(axis=1) - ((q - np.eye(3)[y])**2).sum(axis=1)
    np.testing.assert_allclose(blocks, oracle.reshape(-1, 5).mean(axis=1))
    assert np.shares_memory(q, q_owner) and np.shares_memory(days, day_owner)
    assert pairs == n and count == 12 and coverage == 1.
    assert tuple(pk.forward_blocks_kernel.signatures) == signatures
    assert pk.audit()["python_fallback"] == 0
    bad = q.copy(); bad[9] = np.nan
    result = pk.forward_blocks_kernel(days, y, y, bad, base, 5, 7)
    assert result[3] == 59 and result[5] == 11 and np.isnan(result[2][1])
    assert (result[1] <= cycles).all()


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 2.])
def test_kernel_invalid_probabilities_empty_and_pid(bad, monkeypatch):
    pk.warm()
    y = np.array([0, 1, 0, 1, 0], np.int64)
    days = np.arange(5, dtype=np.int64)
    q = np.array([[.8, .2]] * 5); q[2, 0] = bad
    base = np.array([.5, .5])
    assert pk.forward_blocks_kernel(days, y, y, q, base, 5, 7)[5] == 0
    empty = pk.forward_blocks_kernel(days[:0], y[:0], y[:0], q[:0], base, 5, 7)
    assert empty[3] == 0 and np.isnan(empty[4])
    monkeypatch.setattr(pk, "_PID", -1)
    with pytest.raises(RuntimeError, match="NOT_READY"):
        pk.audit()


def test_no_future_captures_pending_and_genuine_unfitted_rejected(setup):
    c = setup
    protocol, artifact, model = register(c)
    c["clock"].value += timedelta(seconds=1)
    result = c["service"].assess(protocol["id"], {"reference": c["original_ref"]})
    assert result["status"] == "pending"
    assert result["metrics"]["coverage"] is None
    assert "no_forward_captures" in result["reasons"]
    request = copy.deepcopy(artifact["request"])
    request["policy"]["minimum_samples"] = 120
    preview = c["graph"].reliability.preview(request)
    assert not preview["report"]["calibration"]["fitted"]
    saved = c["graph"].reliability.confirm({"request": preview["request"], "preview_hash": preview["preview_hash"]})
    with pytest.raises(Exception, match="尚未拟合"):
        c["service"].register({"calibration_id": saved["id"]})


def test_labels_published_before_capture_cannot_be_laundered(setup):
    c = setup
    protocol, _, _ = register(c)
    d = c["registration"].date() + timedelta(days=1)
    c["extend"](d)
    c["clock"].value += timedelta(days=2)
    c["reference"](d, c["clock"].value - timedelta(seconds=1))
    result = c["service"].capture(protocol["id"])
    assert result["reason"] == "reference_label_already_published"
    with c["service"].journal.locked():
        assert not any(i["kind"] == "observation" for i in c["service"].journal.read_unlocked()["items"])


def test_other_reference_definition_and_future_publication_rejected(setup):
    c = setup
    protocol, _, _ = register(c)
    capture_days(c, protocol, 3)
    future_ref = c["reference"](c["clock"].value.date() - timedelta(days=1), c["clock"].value + timedelta(days=1))
    with pytest.raises(Exception, match="当前已发布"):
        c["service"].assess(protocol["id"], {"reference": future_ref})
    original = c["graph"].runs.get(c["original_ref"]["run_id"])
    other = c["graph"].create_definition(original["definition"])
    plan = c["graph"].prepare(other)
    run = c["graph"].run_saved({"schema_version": "2.0", "id": other["id"], "revision": 1},
                              "retrospective", c["clock"].value.date().isoformat(), plan["compile_token"])
    publication = c["graph"].publish(run["id"], "research_display")["publication"]
    ref = {"run_id": run["id"], "publication_id": publication["id"], "content_hash": run["content_hash"]}
    with pytest.raises(Exception, match="历史定义"):
        c["service"].assess(protocol["id"], {"reference": ref})


def test_kernel_bad_brier_gap_and_invalid_dimensions():
    pk.warm()
    y = np.repeat(np.arange(12) % 3, 5).astype(np.int64)
    days = np.arange(60, dtype=np.int64)
    pred = (y + 1) % 3
    q = np.eye(3)[pred]
    base = np.array([1/3, 1/3, 1/3])
    result = pk.forward_blocks_kernel(days, y, pred, q, base, 5, 7)
    np.testing.assert_allclose(result[2], -4/3)
    assert result[-1] < 0
    days[2:] += 20
    gap = pk.forward_blocks_kernel(days, y, pred, q, base, 5, 7)
    assert np.isnan(gap[2][0]) and gap[5] == 11
    with pytest.raises(ValueError):
        pk.forward_blocks_kernel(days[:-1], y, pred, q, base, 5, 7)
    with pytest.raises(ValueError):
        pk.forward_blocks_kernel(days[::-1], y, pred, q, base, 5, 7)
    with pytest.raises(TypeError):
        pk.forward_blocks_kernel(days.astype(np.int32), y, pred, q, base, 5, 7)


def test_latent_capture_uses_frozen_declared_initial_training_rule(setup):
    c = setup
    artifact, _ = c["candidate"](latent=True)
    assert artifact["report"]["lineage"]["prediction_method"] == "expanding_walk_forward"
    assert artifact["report"]["calibration"]["method"] == "temperature"
    protocol = c["service"].register({"calibration_id": artifact["id"], "policy": c["policy"]})
    assert protocol["model_fit_origin"] == "declared_initial_training_interval_then_causal_inference"
    observations = capture_days(c, protocol, 3)
    initial = protocol["initial_model_audits"]["classifier"]
    for obs in observations:
        audit = obs["model_audits"]["classifier"]
        assert audit["training_count"] == 30
        assert audit["training_end_index"] == initial["training_end_index"]
        assert audit["initialization_fingerprint"] == initial["initialization_fingerprint"]
        assert audit["label_mapping_locked_before_classification"]
        assert not audit["walk_forward_refit"]
        assert obs["model_binding_hash"] == protocol["model_binding_hash"]


def test_missing_reference_dates_stay_on_capture_time_axis(setup):
    c = setup
    protocol, _, _ = register(c)
    observations = capture_days(c, protocol, 10, skip=(4,))
    # A subsequent source vintage omits a day that was seen at capture time.
    # Publishing is real; the immutable reference definition is untouched.
    missing = (c["registration"].date() + timedelta(days=4)).isoformat()
    reduced = [row for row in c["rows"] if row["trade_date"] != missing]
    pd.DataFrame(reduced).to_parquet(c["root"] / "index_daily_df.parquet", index=False)
    result = assess(c, protocol)
    assert result["metrics"]["axis_rows"] == 10
    assert result["metrics"]["paired_samples"] == 9
    assert "reference_labels_not_mature" in result["reasons"]
    assert any(missing in observation["observed_dates"] for observation in observations)


def test_loaded_executor_change_invalidates_protocol(setup, monkeypatch):
    c = setup
    protocol, _, _ = register(c)
    original = RegimeGraphV2Service._execute_numeric_node
    monkeypatch.setattr(RegimeGraphV2Service, "_execute_numeric_node",
                        lambda *args, **kwargs: original(*args, **kwargs))
    with pytest.raises(Exception, match="血缘"):
        c["service"].get_protocol(protocol["id"])


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_current_snapshot_bound_definitions_remain_pending_without_rebinding(setup):
    c = setup
    protocol, _, _ = register(c)
    assert protocol["forward_source_mode"] == "immutable_source_pending_new_observations"
    c["clock"].value += timedelta(days=2)
    assert c["service"].capture(protocol["id"])["reason"] == "no_post_registration_observation"
    c["extend"](c["registration"].date() + timedelta(days=1))
    with pytest.raises(Exception, match="校验值"):
        c["service"].capture(protocol["id"])


def test_state_level_forward_qualification_allows_common_states_and_falls_back_rare_state():
    from historical_regimes.reliability.prospective import _state_qualification

    states = ["bull", "sideways", "bear"]
    # Paired observations are abundant for bull/bear but scarce for sideways.
    support = np.array([[8, 2, 8], [8, 2, 8]], np.int64)
    cycles = np.array([3, 1, 3], np.int64)
    y = np.array([0,0,0,0,1,1,2,2,2,2,0,0,0,0,2,2,2,2], np.int64)
    accepted = np.array([0,0,0,0,-1,-1,2,2,2,2,0,0,0,0,2,2,2,0], np.int64)
    policy = {"minimum_class_observations": 2, "minimum_class_complete_regimes": 2,
              "minimum_state_precision": .65}
    rows, qualified, fallback = _state_qualification(states, support, cycles, y, accepted, policy)
    by_state = {row["state_id"]: row for row in rows}
    assert qualified == ["bull", "bear"]
    assert fallback == ["sideways"]
    assert by_state["sideways"]["status"] == "insufficient_evidence"
    assert "insufficient_complete_state_regimes" in by_state["sideways"]["reasons"]
    assert by_state["bull"]["status"] == by_state["bear"]["status"] == "qualified"
