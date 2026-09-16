"""Frozen index successors through real temporary Parquet and graph execution."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from historical_regimes.reliability import source_versions as sv
from historical_regimes.reliability.prospective import ProspectiveService
from historical_regimes.reliability.execution import model_binding_hash
from historical_regimes.v2_contracts import parse_definition_v2
from test_regime_prospective import setup, register


def successor(case, name="next", days=60, edit=None):
    rows = copy.deepcopy(case["rows"])
    start = datetime.fromisoformat(rows[0]["trade_date"]).date()
    end = case["registration"].date() + timedelta(days=days)
    day = start + timedelta(days=len(rows))
    while day <= end:
        i = (day - start).days
        rows.append({"ts_code": "000300.SH", "trade_date": day.isoformat(),
                     "available_at": day.isoformat(), "close": (300., 155., 10.)[(i // 5) % 3]})
        day += timedelta(days=1)
    if edit:
        edit(rows)
    root = case["root"] / name
    root.mkdir()
    pd.DataFrame(rows).to_parquet(root / "index_daily_df.parquet", index=False)
    # Isolated fixture manifest; source resolution/checksum/PIT are not patched.
    (case["root"] / "tushare_active.json").write_text(json.dumps({
        "schema_version": 1, "snapshot_dir": name, "snapshot_id": name, "generation": name}))
    case["clock"].value = case["registration"] + timedelta(days=2)
    return root


def publish_updated(case, identity, end):
    graph = case["graph"]
    payload = graph.get_definition(identity["definition_id"], identity["revision"])
    plan = graph.prepare(payload)
    run = graph.run_saved({"schema_version": "2.0", "id": payload["id"], "revision": payload["revision"]},
                          "retrospective", end.isoformat(), plan["compile_token"])
    pub = graph.publish(run["id"], "research_display")["publication"]
    # Same trusted publisher clock seam used by the existing forward tests.
    with graph.runs.store.locked():
        data = graph.runs.store.read_unlocked()
        for item in data["items"]:
            if item["id"] == run["id"]:
                for p in item["publications"]:
                    if p["id"] == pub["id"]:
                        p["published_at"] = case["clock"].value.isoformat()
        graph.runs.store.write_unlocked(data)
    return {"run_id": run["id"], "publication_id": pub["id"], "content_hash": run["content_hash"]}


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_explicit_snapshot_successor_capture_reference_and_qualification(setup):
    c = setup
    protocol, artifact, model = register(c)
    original_bytes = (c["root"] / "index_daily_df.parquet").read_bytes()
    original_definition = c["graph"].get_definition(model["id"], 1)
    successor(c)
    preview = sv.preview(c["service"], protocol["id"])
    assert preview["prefix_unchanged"] and preview["added_observations"] == 2
    assert c["service"].get_progress(protocol["id"])["current_source_version"] is None
    accepted = sv.confirm(c["service"], protocol["id"], {"preview_hash": preview["preview_hash"]})
    assert sv.confirm(c["service"], protocol["id"], {"preview_hash": preview["preview_hash"]}) == accepted
    assert accepted["reference_definition"]["revision"] == 2
    assert c["graph"].get_definition(model["id"], 1) == original_definition
    assert (c["root"] / "index_daily_df.parquet").read_bytes() == original_bytes
    assert c["graph"].reliability.get(artifact["id"])["content_hash"] == artifact["content_hash"]
    for day in range(1, 61):
        c["clock"].value = c["registration"] + timedelta(days=day + 1)
        point = c["service"].capture(protocol["id"])
        assert point["status"] == "captured", point
        assert point["observation"]["observation_date"] == (c["registration"].date() + timedelta(days=day)).isoformat()
        assert point["observation"]["execution_model_binding_hash"] == accepted["model_binding_hash"]
    c["clock"].value += timedelta(seconds=1)
    ref = publish_updated(c, accepted["reference_definition"], c["registration"].date() + timedelta(days=60))
    c["clock"].value += timedelta(seconds=1)
    result = c["service"].assess(protocol["id"], {"reference": ref})
    assert result["status"] == "qualified", result["reasons"]
    assert result["metrics"]["paired_samples"] == 60
    assert c["service"].get_qualification(result["id"]) == result
    c["clock"].value += timedelta(seconds=1)
    assert c["service"].verify_qualification(result["id"], artifact["id"], accepted["model_binding_hash"])["status"] == "qualified"
    with pytest.raises(Exception, match="不属于"):
        c["service"].verify_qualification(result["id"], artifact["id"], protocol["model_binding_hash"])
    # The real shared TAA consumer must accept only the approved source binding.
    from historical_regimes.reliability.consumer import attach_calibration, calibrated_output
    c["graph"].prospective = c["service"]
    c["clock"].value += timedelta(days=1)
    adopted = sv.rebind(parse_definition_v2(model), accepted["model_bindings"]).model_dump(mode="json")
    adopted["study"].update(calibration_id=artifact["id"], qualification_id=result["id"])
    source_run = {"definition": adopted, "definition_id": model["id"], "states": adopted["states"], "series": [point["observation"]["raw_point"]]}
    attached = attach_calibration(source_run, c["graph"].reliability)
    assert attached["_reliability"]["error"] is None
    state_ids = [s["id"] for s in adopted["states"]]
    assert calibrated_output(attached, source_run["series"][0], c["clock"].value.date().isoformat(), state_ids)[2] == 'calibration_signal_time_missing'
    # A current run supplies the actual next-observation execution clock; a
    # capture's then-unknown tail effective_date must never be invented.
    plan = c["graph"].prepare(adopted)
    execution = c["graph"]._execute_graph(None, parse_definition_v2(adopted), 'realtime', c["clock"].value.date().isoformat(), plan=c["graph"]._validate_plan(parse_definition_v2(adopted), plan['compile_token']))
    executable = next(p for p in reversed(execution['series']) if p.get('effective_date'))
    _, confidence, reason = calibrated_output(attached, executable, c["clock"].value.date().isoformat(), state_ids)
    assert reason is None and confidence == 1.
    assert "_reliability" not in source_run
    changed = copy.deepcopy(source_run)
    changed["definition"]["graph"]["nodes"][1]["parameters"]["upper"] = 999.
    assert attach_calibration(changed, c["graph"].reliability)["_reliability"]["error"] is not None
    # Restart: accepted-source plans warm explicitly, never on capture fallback.
    c["graph"]._plans.clear()
    c["graph"]._plans_by_graph_hash.clear()
    restored = ProspectiveService(c["graph"], c["graph"].reliability, clock=c["clock"])
    restored.warm()
    assert restored.get_progress(protocol["id"])["current_source_version"] == accepted
    assert restored.verify_qualification(result["id"], artifact["id"], accepted["model_binding_hash"])["status"] == "qualified"


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
@pytest.mark.parametrize("edit", [lambda rows: rows[0].update(close=1.),
    lambda rows: rows[2].update(available_at=rows[3]["available_at"]), lambda rows: rows.pop(10)])
def test_changed_history_cannot_be_laundered_as_new_snapshot(setup, edit):
    c = setup
    protocol, _, _ = register(c)
    successor(c, edit=edit)
    with pytest.raises(Exception, match="修订|缺失"):
        sv.preview(c["service"], protocol["id"])
    assert c["service"].get_progress(protocol["id"])["current_source_version"] is None


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_preview_expiry_source_change_no_new_rows_and_untrusted_payload(setup):
    c = setup
    protocol, _, _ = register(c)
    with pytest.raises(Exception, match="没有新增"):
        sv.preview(c["service"], protocol["id"])
    successor(c)
    p = sv.preview(c["service"], protocol["id"])
    c["service"]._source_previews[p["preview_hash"]]["expires"] = 0
    with pytest.raises(Exception, match="过期"):
        sv.confirm(c["service"], protocol["id"], {"preview_hash": p["preview_hash"]})
    p = sv.preview(c["service"], protocol["id"])
    successor(c, name="different")
    with pytest.raises(Exception, match="变化"):
        sv.confirm(c["service"], protocol["id"], {"preview_hash": p["preview_hash"]})
    with pytest.raises(ValueError):
        sv.confirm(c["service"], protocol["id"], {"preview_hash": p["preview_hash"], "approved": True})


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_changed_reference_math_is_never_overwritten(setup):
    c = setup
    protocol, _, _ = register(c)
    successor(c)
    ref_id = protocol["reference_definition"]["definition_id"]
    definition = c["graph"].get_definition(ref_id, 1)
    definition["graph"]["nodes"][1]["parameters"]["upper"] = 999.
    changed = c["graph"].update_definition(ref_id, 1, definition)
    with pytest.raises(Exception, match="参考算法已修改"):
        sv.preview(c["service"], protocol["id"])
    assert c["graph"].get_definition(ref_id, 2) == changed


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_published_successor_label_still_blocks_backfill(setup):
    c = setup
    protocol, _, _ = register(c)
    successor(c)
    p = sv.preview(c["service"], protocol["id"])
    accepted = sv.confirm(c["service"], protocol["id"], {"preview_hash": p["preview_hash"]})
    publish_updated(c, accepted["reference_definition"], c["clock"].value.date() - timedelta(days=1))
    c["clock"].value += timedelta(seconds=1)
    assert c["service"].capture(protocol["id"])["reason"] == "reference_label_already_published"


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_two_successors_preserve_prior_definitions_and_revision_is_explicit(setup):
    c = setup
    protocol, _, _ = register(c)
    first_path = successor(c, days=2)
    first_bytes = (first_path / "index_daily_df.parquet").read_bytes()
    preview = sv.preview(c["service"], protocol["id"])
    first = sv.confirm(c["service"], protocol["id"], {"preview_hash": preview["preview_hash"]})
    c["service"].capture(protocol["id"])
    successor(c, name="second", days=3)
    c["clock"].value += timedelta(days=1)
    preview = sv.preview(c["service"], protocol["id"])
    second = sv.confirm(c["service"], protocol["id"], {"preview_hash": preview["preview_hash"]})
    assert second["reference_definition"]["revision"] == 3
    assert (first_path / "index_daily_df.parquet").read_bytes() == first_bytes
    progress = c["service"].get_progress(protocol["id"])
    assert [r["revision"] for r in progress["reference_definitions"]] == [1, 2, 3]
    assert progress["current_source_version"] == second
    assert c["service"].capture(protocol["id"])["status"] == "captured"


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_source_routes_forbid_client_paths_dates_and_status(setup):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from historical_regimes.reliability.prospective import install
    c = setup
    protocol, _, _ = register(c)
    successor(c)
    app = FastAPI()
    install(app, lambda: c["service"], lambda fn, *args: fn(*args))
    url = '/api/historical-regimes/prospective/' + protocol['id'] + '/sources'
    with TestClient(app) as client:
        assert client.post(url + '/preview', json={'as_of': '2000-01-01'}).status_code == 422
        preview = client.post(url + '/preview', json={})
        assert preview.status_code == 200, preview.text
        body = {'preview_hash': preview.json()['preview_hash']}
        assert client.post(url + '/confirm', json={**body, 'status': 'accepted'}).status_code == 422
        assert client.post(url + '/confirm', json=body).status_code == 200


@pytest.mark.parametrize("setup", ["bound"], indirect=True)
def test_engine_invalidated_protocol_does_not_block_other_research_startup(setup, monkeypatch):
    from historical_regimes.reliability import prospective
    c = setup
    protocol, _, _ = register(c)
    successor(c)
    p = sv.preview(c["service"], protocol["id"])
    sv.confirm(c["service"], protocol["id"], {"preview_hash": p["preview_hash"]})
    monkeypatch.setattr(prospective, '_engine_hash', lambda graph: 'changed-engine')
    restored = ProspectiveService(c["graph"], c["graph"].reliability, clock=c["clock"])
    audit = restored.warm()
    assert audit['source_versions'] == {'prepared': 0, 'blocked': [
        {'protocol_id': protocol['id'], 'code': 'PROSPECTIVE_LINEAGE'}]}
    assert restored._audit()['python_fallback'] == 0
    with pytest.raises(Exception, match='血缘已变化'):
        restored.capture(protocol['id'])
