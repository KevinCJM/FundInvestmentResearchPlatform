"""Reusable full/incremental plans, explicit scope and frozen run parameters."""
from __future__ import annotations

import copy
import json
import sys
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))
from backend.data_sources import acquisition, etl_service
from backend.data_sources.batches import capture_batch
from backend.data_sources.etl_models import EtlDefinition, EtlRunOptions
from backend.data_sources.etl_parameters import bind_parameters
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.etl_templates import tushare_fund_workflow
from backend.data_sources.models import CenterError, InterfaceConfig
from backend.data_sources.store import SourceStore
from backend.services import etl_routes


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(etl_service, '_launch', etl_service._launch_inline)
    monkeypatch.setenv("DATA_SOURCE_CENTER_ENABLED", "true")
    result = SourceStore(tmp_path)
    result.seed()
    return result


def definition(store):
    record = store.get("interface", "akshare.fund_nav")
    return {"name": "Reusable NAV flow", "steps": [
        {"id": "download", "name": "NAV", "kind": "download", "mode": "inherit", "source_id": "akshare", "interface_id": record["config"]["id"], "interface_revision": record["revision"], "params": {"start_date": "20240101", "end_date": "20240110"}},
        {"id": "map", "name": "Map", "kind": "map", "inputs": ["download"]},
        {"id": "resolve", "name": "Resolve", "kind": "resolve", "table_id": "market.nav_daily", "inputs": ["map"], "include_history": False},
    ]}


def wait(store, run):
    for _ in range(500):
        result = EtlStore(store).get_run(run["run_id"])
        if result["status"] != "RUNNING":
            return result
        time.sleep(0.01)
    pytest.fail("offline run did not finish")


def request(plan, mode="incremental", parameters=None):
    return {"definition": plan, "options": {"mode": mode, "parameters": parameters or {}}, "request_id": str(uuid.uuid4()), "confirm": True}


def response(*args, **kwargs):
    return {}, [{"symbol": "000001", "净值日期": "2024-01-05", "单位净值": 1.25}]


def test_one_saved_revision_runs_full_incremental_and_full_again(store, monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition, "fetch_with_retry", lambda *args: (calls.append(args[3]) or response()))
    saved = etl_service.save_workflow(store, "reusable", definition(store), 0)
    for mode in ("full", "incremental", "full"):
        payload = {"workflow_id": "reusable", "expected_revision": 1, "request_id": str(uuid.uuid4()), "confirm": True, "options": {"mode": mode}}
        run = wait(store, etl_service.start(store, payload))
        assert run["status"] == "SUCCEEDED", run
        assert run["options"]["mode"] == run["steps"][0]["mode"] == mode
    assert [value["start_date"] for value in calls] == ["20240101", "20240102", "20240101"]
    assert EtlStore(store).workflow("reusable") == saved


def test_explicit_legacy_step_mode_is_preserved(store, monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition, "fetch_with_retry", lambda *args: (calls.append(args[3]) or response()))
    plan = definition(store)
    plan["steps"][0]["mode"] = "full"
    for _ in range(2):
        assert wait(store, etl_service.start(store, request(plan)))["status"] == "SUCCEEDED"
    assert [value["start_date"] for value in calls] == ["20240101", "20240101"]


def test_mode_is_part_of_idempotency_key(store, monkeypatch):
    monkeypatch.setattr(acquisition, "fetch_with_retry", response)
    payload = request(definition(store), "full")
    wait(store, etl_service.start(store, payload))
    payload["options"]["mode"] = "incremental"
    with pytest.raises(CenterError) as error:
        etl_service.start(store, payload)
    assert error.value.code == "ETL_REQUEST_CONFLICT"


@pytest.mark.parametrize("options", [{"mode": "bad"}, {"mode": None}, {"parameters": {"x": 1}}, {"unknown": True}])
def test_invalid_run_options_fail_before_download(store, monkeypatch, options):
    monkeypatch.setattr(acquisition, "fetch_with_retry", lambda *args: pytest.fail("network must not run"))
    payload = request(definition(store)); payload["options"] = options
    with pytest.raises(CenterError):
        etl_service.start(store, payload)


def test_named_parameters_are_required_not_written_to_workflow(store):
    plan = tushare_fund_workflow(store)
    saved = etl_service.save_workflow(store, "tushare_funds", plan.model_dump(mode="json"), 0)
    assert saved["revision"] == 1
    assert len(plan.steps) == 17
    assert {s.mode for s in plan.steps if s.kind == "download"} == {"full", "inherit"}
    with pytest.raises(CenterError) as error:
        bind_parameters(plan, EtlRunOptions())
    assert error.value.code == "ETL_PARAMETER_REQUIRED"
    options = EtlRunOptions(mode="full", parameters={"etf_code": "510300.SH", "fund_code": "000001.OF", "end_date": "2024-01-05"})
    bound = bind_parameters(plan, options)
    navs = [s for s in bound.steps if s.kind == "download" and s.id in {"etf_nav", "fund_nav"}]
    assert {s.params["ts_code"] for s in navs} == {"510300.SH", "000001.OF"}
    assert all(s.params["start_date"] == "20100101" and s.params["end_date"] == "20240105" for s in navs)
    assert all("ts_code" not in s.params for s in plan.steps)
    assert etl_service.inspect_plan(store, bound)["plan"][-1]["kind"] == "snapshot"
    assert EtlStore(store).workflow("tushare_funds") == saved


@pytest.mark.parametrize("parameters,code", [({"etf_code": "510300.SH"}, "ETL_PARAMETER_REQUIRED"), ({"extra": "x"}, "ETL_PARAMETER_UNKNOWN"), ({"etf_code": "510300.SH", "fund_code": "000001.OF", "end_date": "bad"}, "ETL_PARAMETER_INVALID")])
def test_template_bad_scope_is_rejected(store, parameters, code):
    with pytest.raises(CenterError) as error:
        bind_parameters(tushare_fund_workflow(store), EtlRunOptions(parameters=parameters))
    assert error.value.code == code


def test_unresolved_parameter_binding_cannot_be_saved(store):
    plan = definition(store); plan["steps"][0]["parameter_bindings"] = {"symbol": "not_declared"}
    assert not etl_service.validate(store, plan)["valid"]


def test_resume_uses_frozen_request_range_not_new_watermark(store, monkeypatch):
    calls = []
    def fail(*args):
        calls.append(args[3]); raise CenterError("OFFLINE_FAILURE", "test")
    monkeypatch.setattr(acquisition, "fetch_with_retry", fail)
    run = wait(store, etl_service.start(store, request(definition(store))))
    assert run["status"] == "FAILED"
    frozen = run["frozen"]["downloads"]["download"]
    with store.connection() as db:
        db.execute("INSERT OR REPLACE INTO source_sync_checkpoint VALUES (?,?)", (frozen["checkpoint"], "2024-01-09"))
    monkeypatch.setattr(acquisition, "fetch_with_retry", lambda *args: (calls.append(args[3]) or response()))
    resumed = wait(store, etl_service.resume(store, run["run_id"], True))
    assert resumed["status"] == "SUCCEEDED"
    assert [v["start_date"] for v in calls] == ["20240101", "20240101"]


def test_matching_history_does_not_import_other_product(store, monkeypatch):
    record = store.get("interface", "akshare.fund_nav")
    for symbol in ("000001", "000002"):
        config = InterfaceConfig.model_validate(record["config"])
        config.params.update(symbol=symbol, start_date="20240101", end_date="20240104")
        capture_batch(store, config, [{"symbol": symbol, "净值日期": "2024-01-04", "单位净值": 1.2}], config.params, "test-" + symbol)
    monkeypatch.setattr(acquisition, "fetch_with_retry", response)
    plan = definition(store)
    plan["steps"][-1].update(include_history=True, history_scope="matching_inputs")
    run = wait(store, etl_service.start(store, request(plan)))
    assert run["status"] == "SUCCEEDED", run
    assert len(run["frozen"]["history"]["resolve"]) == 1
    assert run["steps"][-1]["rows"] == 2


def test_template_endpoint_only_returns_draft_then_normal_save(store, monkeypatch):
    monkeypatch.setattr(etl_routes, "get_store", lambda: store)
    app = FastAPI(); app.include_router(etl_routes.router)
    client = TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1'); base = "/api/data-sources/etl"
    draft = client.get(base + "/templates/tushare-funds")
    assert draft.status_code == 200, draft.text
    assert client.get(base + "/workflows").json() == []
    result = client.put(base + "/workflows/tushare_funds", json=draft.json())
    assert result.status_code == 200, result.text
    assert client.put(base + "/workflows/tushare_funds", json=draft.json()).status_code == 409
    assert client.get(base + "/runs").json() == []
    store.seed()
    assert len(EtlStore(store).workflows()) == 1
    missing = client.post(base + "/validate", json={"definition": draft.json()["definition"], "options": {"mode": "full"}})
    assert missing.json()["errors"][0]["code"] == "ETL_PARAMETER_REQUIRED"
