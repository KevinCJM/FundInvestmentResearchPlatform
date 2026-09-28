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


def test_options_sections_do_not_load_unrequested_studies(workspace, monkeypatch):
    service, _ = workspace
    def unexpected(*args, **kwargs):
        raise AssertionError('unrequested research was loaded')
    monkeypatch.setattr(service.cma.evidence, '_regime_items', unexpected)
    monkeypatch.setattr(service.cma, 'list', unexpected)
    app = FastAPI(); app.include_router(build_router(service))
    client = TestClient(app)
    url = '/api/strategic-allocation/cma/study-options'
    base = client.get(url, params={'section': 'base'})
    assert base.status_code == 200
    assert base.json()['allocations'] and base.json()['assumptions'] == []
    assert 'scenario_options' not in base.json()
    monkeypatch.setattr(service.cma, 'list', lambda **kwargs: {'items': [{'id': 'prior'}]})
    monkeypatch.setattr(service, 'catalog', unexpected)
    assert client.get(url, params={'section': 'priors'}).json() == {'assumptions': [{'id': 'prior'}]}
    assert client.get(url, params={'section': 'invalid'}).status_code == 422
    assert client.get(url, params={'section': 'base', 'as_of': '9999-12-31'}).status_code == 422


def test_duplicate_name_is_rejected_until_the_earlier_version_retires(workspace):
    """名称是清单里区分同范围、同方法版本的唯一线索，重名会让研究员无法辨认。"""
    service, _ = workspace
    first, body = published(service)
    again = body.model_copy(update={"idempotency_key": "ltcma-test-operation-2"})
    with pytest.raises(ConflictError, match="名称已存在"):
        service.publish_cma(again)
    assert service.cma.study_options(section="base")["existing_names"] == [first["name"]]
    service.cma.retire(first["id"], CmaRetire(confirm=True, content_hash=first["content_hash"],
                                              reason="更换长期研究假设"))
    assert service.cma.study_options(section="base")["existing_names"] == []
    assert service.publish_cma(again)["name"] == first["name"]


def test_edit_updates_one_study_while_copy_creates_another_and_history_stays_frozen(workspace):
    from backend.strategic_allocation.cma_center_contracts import CmaCenterUpdate
    service, _ = workspace
    _, first, policy_request = saved_inputs(service)
    from backend.strategic_allocation.contracts import PublishPolicyRequest
    policy_preview = service.preview_policy(policy_request)
    policy = service.publish_policy(PublishPolicyRequest(request=policy_request,
        preview_hash=policy_preview['preview_hash'], candidate_id='robust-utility',
        name='引用修改前结果的 SAA', reason='保留修改前的配置研究依据'))
    old_arrays = service.artifacts.arrays(first['id'])['covariance'].copy()
    request = definition().model_copy(update={'name': first['name'], 'source': '修改后的研究依据'})
    preview = service.cma.preview(request)
    body = CmaCenterUpdate(request=request, preview_hash=preview['preview_hash'], confirm=True,
                           idempotency_key='edit-original-study', expected_content_hash=first['content_hash'])
    app = FastAPI(); app.include_router(build_router(service)); client = TestClient(app)
    url = '/api/strategic-allocation/cma/' + first['id']
    result = client.patch(url, json=body.model_dump(mode='json'))
    assert result.status_code == 200, result.text
    updated = result.json()
    assert updated['study_id'] == first['id']
    assert updated['supersedes_cma_id'] == first['id']
    assert 'copied_from_id' not in updated
    assert updated['definition']['source'] == '修改后的研究依据'
    assert [x['id'] for x in service.cma.list(include_retired=True)['items']] == [updated['id']]
    assert [x['id'] for x in service.catalog()['assumptions']] == [updated['id']]
    assert service.cma.active_names() == [first['name']]
    assert [x['id'] for x in service.cma.study_options(section='priors')['assumptions']] == [updated['id']]
    restored = client.get('/api/strategic-allocation/cma/study-options', params={'section': 'priors', 'selected_prior_id': first['id']})
    assert restored.status_code == 200
    assert {x['id'] for x in restored.json()['assumptions']} == {first['id'], updated['id']}
    assert client.patch(url, json=body.model_dump(mode='json')).json() == updated
    assert service.get_cma(first['id']) == first
    assert (service.artifacts.arrays(first['id'])['covariance'] == old_arrays).all()
    assert service.baselines.get_baseline(policy['id']) == policy
    assert service.cma.get(first['id']) == first  # 冻结版本仍可精确读取；
    with pytest.raises(ValidationError, match='已有新版本'):  # 但已被替代的版本不能接入新的下游工作。
        service.cma.require_selectable(first['id'])
    stale = body.model_copy(update={'idempotency_key': 'second-editor-stale'})
    assert client.patch(url, json=stale.model_dump(mode='json')).status_code == 409
    with pytest.raises(ConflictError, match='修改或删除'):
        service.cma.retire(first['id'], CmaRetire(confirm=True, content_hash=first['content_hash'], reason='过期页面尝试删除'))
    second_body = body.model_copy(update={'expected_content_hash': updated['content_hash'], 'idempotency_key': 'edit-same-study-again'})
    second = service.cma.publish(second_body, updated['id'])
    assert second['study_id'] == first['id']
    assert service.cma.list()['total'] == 1
    copied_request = request.model_copy(update={'name': '独立复制研究'})
    copied = service.cma.publish(CmaCenterPublish(request=copied_request,
        preview_hash=service.cma.preview(copied_request)['preview_hash'], confirm=True,
        idempotency_key='copy-separate-study', copied_from_id=second['id']))
    assert copied['copied_from_id'] == second['id'] and 'supersedes_cma_id' not in copied
    assert service.cma.list()['total'] == 2
    conflicting = second_body.model_copy(update={'request': copied_request,
        'preview_hash': service.cma.preview(copied_request)['preview_hash'],
        'expected_content_hash': second['content_hash'], 'idempotency_key': 'edit-name-conflict'})
    with pytest.raises(ConflictError, match='名称已存在'):
        service.cma.publish(conflicting, second['id'])
    service.cma.retire(second['id'], CmaRetire(confirm=True, content_hash=second['content_hash'], reason='删除当前研究方案'))
    assert [x['id'] for x in service.cma.list()['items']] == [copied['id']]
    assert service.cma.list(include_retired=True)['total'] == 2  # No resurrected predecessors.
    with pytest.raises(ConflictError, match='修改或删除'):
        service.cma.publish(second_body.model_copy(update={'expected_content_hash': second['content_hash'],
                            'idempotency_key': 'edit-deleted-study'}), second['id'])


def test_edit_rejects_wrong_hash_and_recovers_interrupted_save(workspace, monkeypatch):
    from backend.strategic_allocation.cma_center_contracts import CmaCenterUpdate
    from backend.custom_indicators.repository import AtomicJsonStore
    service, _ = workspace
    first, published_body = published(service)
    body = CmaCenterUpdate(**published_body.model_dump(mode='json'), expected_content_hash='0' * 64)
    body = body.model_copy(update={'idempotency_key': 'edit-interrupted-operation'})
    with pytest.raises(ConflictError, match='内容不一致'):
        service.cma.publish(body, first['id'])
    body = body.model_copy(update={'expected_content_hash': first['content_hash']})
    original_write = AtomicJsonStore.write_unlocked
    def fail_completion(store, value):
        if store.path.name == 'operations.json' and value['items'][-1]['complete']:
            raise OSError('simulated failure after index registration')
        return original_write(store, value)
    with monkeypatch.context() as patch:
        patch.setattr(AtomicJsonStore, 'write_unlocked', fail_completion)
        with pytest.raises(OSError):
            service.cma.publish(body, first['id'])
    assert service.cma.list()['total'] == 1
    recovered = service.cma.publish(body, first['id'])
    assert service.cma.publish(body, first['id']) == recovered
    assert service.cma.list()['total'] == 1 and service.get_cma(first['id']) == first
    draft = service.cma.drafts.save(CmaDraftWrite(name=first['name'], editable_definition={},
        editing_ref={'id': recovered['id'], 'content_hash': recovered['content_hash']}))
    assert service.cma.drafts.get(draft['id'])['editing_ref'] == draft['editing_ref']
