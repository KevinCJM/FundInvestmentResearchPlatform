"""Independent center lifecycle, using the same offline product fixtures as SAA."""
import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.custom_indicators.errors import ConflictError, ValidationError
from backend.strategic_allocation.cma_center_contracts import CmaCenterPublish, CmaDraftWrite, CmaRetire
from backend.strategic_allocation.routes import build_router
from backend.tests.test_strategic_allocation import workspace, warm, definition, saved_inputs


def published(service, key="ltcma-test-operation"):
    request = definition()
    preview = service.preview_cma(request)
    body = CmaCenterPublish(request=request, preview_hash=preview["preview_hash"],
                            confirm=True, idempotency_key=key)
    return service.publish_cma(body), body


def test_center_keeps_exact_old_calculation_and_readonly_preview(workspace):
    service, _ = workspace
    old = service.preview_cma(definition())
    assert service.cma.preview(definition()) == old
    assert service.cma.list()["items"] == []
    assert service.cma.drafts.read() == []
    assert not service.artifacts.root.exists()
    assert not service.cma.drafts.document.path.exists()


def test_idempotent_publication_and_conflict(workspace):
    service, _ = workspace
    first, body = published(service)
    assert service.publish_cma(body) == first
    assert len(service.cma.list()["items"]) == 1
    changed = body.model_copy(update={"copied_from_id": first["id"]})
    with pytest.raises(ConflictError, match="同一幂等键"):
        service.publish_cma(changed)
    assert service.cma.view(first["id"])["version"] == first


def test_retirement_blocks_new_policy_not_frozen_history(workspace):
    service, _ = workspace
    _, cma, request = saved_inputs(service)
    prior = copy.deepcopy(service.get_cma(cma["id"]))
    service.cma.retire(cma["id"], CmaRetire(confirm=True, content_hash=cma["content_hash"], reason="更换长期研究假设"))
    assert service.get_cma(cma["id"]) == prior
    assert service.cma.list()["total"] == 0
    assert service.cma.list(include_retired=True)["items"][0]["retired"]
    with pytest.raises(ValidationError, match="停止新引用"):
        service.preview_policy(request)
    assert service.cma.view(cma["id"])["retired"]


def test_retirement_hash_and_idempotence(workspace):
    service, _ = workspace
    cma, _ = published(service)
    with pytest.raises(ConflictError):
        service.cma.retire(cma["id"], CmaRetire(confirm=True, content_hash="0" * 64, reason="更换长期研究假设"))
    body = CmaRetire(confirm=True, content_hash=cma["content_hash"], reason="更换长期研究假设")
    assert service.cma.retire(cma["id"], body) == service.cma.retire(cma["id"], body)
    assert len([x for x in service.artifacts.list("retirement") if x.get("artifact_type") == "cma_retirement"]) == 1


def test_drafts_allow_incomplete_inputs_and_reject_stale_revision(workspace):
    service, _ = workspace
    body = CmaDraftWrite(name="待研究假设", editable_definition={"assets": [], "source": None})
    first = service.cma.drafts.save(body)
    update = body.model_copy(update={"expected_revision": 1, "name": "新草稿名称"})
    second = service.cma.drafts.save(update, first["id"])
    assert second["revision"] == 2
    with pytest.raises(ConflictError):
        service.cma.drafts.save(update, first["id"])
    with pytest.raises(ConflictError):
        service.cma.drafts.delete(first["id"], 1)
    service.cma.drafts.delete(first["id"], 2)
    assert service.cma.drafts.read() == []
    assert service.cma.list()["total"] == 0


def test_api_static_paths_and_no_fake_capabilities(workspace):
    service, _ = workspace
    app = FastAPI(); app.include_router(build_router(service))
    client = TestClient(app)
    base = "/api/strategic-allocation/cma"
    assert client.get(base).json()["items"] == []
    response = client.get(base + "/capabilities")
    assert response.status_code == 200
    assert {x["id"] for x in response.json()["methods"]} >= {"manual", "black_litterman", "scenario_mixture"}
    assert client.get(base + "/drafts").status_code == 200
    assert client.get(base + "?limit=201").status_code == 422
    draft = client.post(base + "/drafts", json={"name": "初始研究", "editable_definition": {}})
    assert draft.status_code == 201
    assert client.get(base + "/drafts/" + draft.json()["id"]).status_code == 200
    assert client.request("DELETE", base + "/drafts/" + draft.json()["id"], json={"expected_revision": 1}).status_code == 200


def test_nonfinite_draft_is_not_cleaned_to_zero():
    with pytest.raises(ValueError):
        CmaDraftWrite(name="坏数值", editable_definition={"annual_return": float("nan")})


def test_prior_retirement_after_calculation_blocks_new_publication(workspace, monkeypatch):
    from backend.tests.test_ltcma_evidence import modern_manual
    from backend.tests.test_ltcma_statistics import request, publish

    service, _ = workspace
    service.warm()
    prior = publish(service, modern_manual(), "retirement-race-prior")
    definition = request("bayesian_niw",
        prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
        mean_prior_observations=20., covariance_prior_observations=20.)
    preview = service.cma.preview(definition)
    original = service.cma.calculation

    def retire_after_calculation(value):
        result = original(value)
        service.cma.retire(prior["id"], CmaRetire(confirm=True,
            content_hash=prior["content_hash"], reason="计算完成后停止先验新引用"))
        return result

    monkeypatch.setattr(service.cma, "calculation", retire_after_calculation)
    with pytest.raises(ValidationError, match="停止新引用"):
        service.cma.publish(CmaCenterPublish(request=definition,
            preview_hash=preview["preview_hash"], confirm=True,
            idempotency_key="retirement-race-posterior"))
    assert service.cma.list(include_retired=True)["total"] == 1
    assert service.cma.get(prior["id"]) == prior


def test_niw_publication_holds_lifecycle_lock_and_replays_after_retirement(workspace, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    from backend.tests.test_ltcma_evidence import modern_manual
    from backend.tests.test_ltcma_statistics import request, publish

    service, _ = workspace
    service.warm()
    prior = publish(service, modern_manual(), "locked-prior-publication")
    definition = request("bayesian_niw",
        prior_ref={"id": prior["id"], "content_hash": prior["content_hash"]},
        mean_prior_observations=20., covariance_prior_observations=20.)
    preview = service.cma.preview(definition)
    body = CmaCenterPublish(request=definition, preview_hash=preview["preview_hash"],
        confirm=True, idempotency_key="locked-posterior-publication")
    original = service.artifacts.save
    checked = []

    def save_with_lock_probe(*args, **kwargs):
        with service.artifacts.governance_lock.lock_path.open("a+") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(service.artifacts, "save", save_with_lock_probe)
    posterior = service.cma.publish(body)
    assert checked == [True]
    service.cma.retire(prior["id"], CmaRetire(confirm=True,
        content_hash=prior["content_hash"], reason="新版本发布后停止先验新引用"))
    assert service.cma.publish(body) == posterior
    assert service.cma.get(posterior["id"]) == posterior
