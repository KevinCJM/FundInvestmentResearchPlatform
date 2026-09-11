"""HTTP contract tests use an isolated research repository and real kernels."""
import time
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.timing_research.service import TimingResearchService
from test_timing_service import fixture_bars


@pytest.fixture
def client(tmp_path, monkeypatch):
    from services import timing_research_routes as routes
    service = TimingResearchService(tmp_path, loader=lambda *args: fixture_bars())
    service.warm()
    monkeypatch.setattr(routes, "timing_service", service)
    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app) as value:
        yield value
    service.close()


def test_http_author_run_detail_release_bind(client):
    prefix = "/api/timing-research"
    catalog = client.get(prefix + "/catalog").json()
    definition = catalog["templates"][0]["definition"]
    saved = client.post(prefix + "/definitions", json=definition)
    assert saved.status_code == 201, saved.text
    assert client.get(prefix + "/definitions").json()["items"]
    update = client.put(prefix + "/definitions/" + saved.json()["id"], json={**definition, "revision": 1})
    assert update.json()["revision"] == 2
    assert client.put(prefix + "/definitions/" + saved.json()["id"], json={**definition, "revision": 1}).status_code == 409
    prepared = client.post(prefix + "/prepare", json={"definition": definition}).json()
    request = {"definition": definition, "compile_token": prepared["compile_token"], "targets": [{"product_id": "510300.SH"}],
               "start_date": "2020-01-01", "end_date": "2021-12-31", "holdout_start": "2021-01-01"}
    response = client.post(prefix + "/runs", json=request)
    assert response.status_code == 202, response.text
    job = response.json()
    deadline = time.monotonic() + 30
    while job["status"] not in {"completed", "failed"} and time.monotonic() < deadline:
        time.sleep(.01)
        job = client.get(prefix + "/jobs/" + job["id"]).json()
    assert job["status"] == "completed", job
    run = client.get(prefix + "/runs/" + job["run_id"])
    assert run.status_code == 200, run.text
    assert run.json()["execution"]["request_time_compilation"] == 0
    detail = client.get(prefix + "/runs/" + job["run_id"] + "/products/510300.SH?limit=3")
    assert len(detail.json()["curve"]) == 3
    assert detail.json()["execution"]["nopython"] is True
    release = client.post(prefix + "/releases", json={"run_id": job["run_id"], "note": "研究引用"})
    assert release.status_code == 201, release.text
    assert release.json()["execution_authorized"] is False
    bound = client.post(prefix + "/bindings", json={"release_id": release.json()["id"], "context": "pre_investment"})
    assert bound.status_code == 201, bound.text
    assert client.get(prefix + "/bindings").json()["items"]
    assert client.get(prefix + "/releases").json()["items"]


def test_http_safe_validation_and_not_found(client):
    prefix = "/api/timing-research"
    invalid = client.post(prefix + "/prepare", json={"definition": {"name": "empty"}})
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "REQUEST_VALIDATION_ERROR"
    missing = client.get(prefix + "/jobs/unknown")
    assert missing.status_code == 404
    assert client.post(prefix + "/bindings", json={"release_id": "x", "context": "live_execution"}).status_code == 422
