"""LTCMA 列表带出目标、路径与范围的版本；上游删除停止引用，上游改版需更新。"""
from datetime import date, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.strategic_allocation.routes import build_router
from backend.strategic_allocation.service import StrategicAllocationService

P = '/api/strategic-allocation'


@pytest.fixture(scope='module', autouse=True)
def warmed():
    from backend.strategic_allocation import institution_kernels, kernels
    kernels.warm_strategic_kernels()
    institution_kernels.warm()


def setup(tmp_path):
    client = TestClient(FastAPI())
    client.app.include_router(build_router(StrategicAllocationService(tmp_path / 'research', tmp_path / 'market')))
    body = {'name': '上游范围', 'as_of': str(date.today()), 'currency': 'CNY', 'source': '', 'assets': [
        {'id': 'growth', 'name': '增长资产', 'role': 'growth', 'liquidity': 'liquid', 'currency': 'CNY'},
        {'id': 'cash', 'name': '现金', 'role': 'liquidity', 'liquidity': 'liquid', 'currency': 'CNY'}]}
    preview = client.post(P + '/universes/preview', json=body)
    scope = client.post(P + '/universes/confirm', json={'request': body, 'preview_hash': preview.json()['preview_hash']})
    assert scope.status_code == 201, scope.text
    study = {'definition': {
        'name': '上游目标', 'as_of': str(date.today()), 'review_date': str(date.today() + timedelta(days=180)),
        'currency': 'CNY', 'horizon_years': 5, 'target_return': 0., 'max_volatility': .3, 'max_tracking_error': .1,
        'boundary_reason': '测试上游删除。', 'strategic_universe_id': scope.json()['id']}}
    diagnosed = client.post(P + '/mandates/preview', json=study)
    assert diagnosed.status_code == 200, diagnosed.text
    mandate = client.post(P + '/mandates/confirm', json={
        'request': study, 'preview_hash': diagnosed.json()['preview_hash'], 'acknowledge_limits': True})
    assert mandate.status_code == 201, mandate.text
    bound = client.post(P + '/universes/' + scope.json()['id'] + '/mandate', json={'mandate_id': mandate.json()['id']})
    assert bound.status_code == 200, bound.text
    assumptions = {
        'name': '上游假设', 'strategic_universe_id': scope.json()['id'], 'as_of': str(date.today()),
        'currency': 'CNY', 'horizon_years': 5, 'source': '离线前瞻输入。', 'basis_confirmed': True,
        'assets': [
            {'id': 'growth', 'role': 'growth', 'liquidity': 'liquid', 'rationale': '增长风险预算',
             'annual_return': .08, 'annual_volatility': .18, 'mean_uncertainty': .02},
            {'id': 'cash', 'role': 'liquidity', 'liquidity': 'liquid', 'rationale': '经营资金储备',
             'annual_return': .02, 'annual_volatility': .01, 'mean_uncertainty': .001},
        ], 'correlation': [[1., 0.], [0., 1.]]}
    cma_preview = client.post(P + '/cma/preview', json=assumptions)
    cma = client.post(P + '/cma', json={'request': assumptions, 'preview_hash': cma_preview.json()['preview_hash']})
    assert cma.status_code == 201, cma.text
    return client, mandate.json(), scope.json(), cma.json()


def refs(item):
    return {ref['kind']: {k: ref[k] for k in ('id', 'name', 'number', 'status')} for ref in item['upstream']}


@pytest.mark.parametrize('deleted', ['scope', 'mandate'])
def test_list_carries_group_and_upstream_deletion_retires_cma(tmp_path, deleted):
    client, mandate, scope, cma = setup(tmp_path)
    item = client.get(P + '/cma').json()['items'][0]
    assert item['research_path'] == 'strategy_first' and item['retired'] is False
    assert refs(item) == {
        'mandate': {'id': mandate['id'], 'name': '上游目标', 'number': 1, 'status': 'current'},
        'strategic_scope': {'id': scope['id'], 'name': '上游范围', 'number': 1, 'status': 'current'}}
    assert item['version']['number'] == 1 and item['usable'] == {'status': 'ready', 'reasons': []}
    target = P + ('/universes/' + scope['id'] if deleted == 'scope' else '/mandates/' + mandate['id'])
    assert client.delete(target).status_code == 200
    assert client.get(P + '/cma').json()['items'] == []
    item = client.get(P + '/cma', params={'include_retired': True}).json()['items'][0]
    kind = 'strategic_scope' if deleted == 'scope' else 'mandate'
    assert item['retired'] is True and item['usable']['status'] == 'blocked'
    assert item['usable']['reasons'][0]['code'] == 'upstream_deleted' and item['usable']['reasons'][0]['kind'] == kind
    assert client.get(P + '/cma/' + cma['id'] + '/view').json()['retired'] is True
    policy = client.post(P + '/policy/preview', json={'mandate_id': mandate['id'], 'cma_id': cma['id']})
    assert policy.status_code >= 400
    if deleted == 'scope':  # 目标仍有效时，拦截来自 LTCMA 自身的上游删除检查。
        assert 'CMA_UPSTREAM_DELETED' in policy.text


def edit_mandate(client, mandate):
    study = {'definition': {**mandate['definition'], 'boundary_reason': '目标改版。'}, 'replaces_mandate_id': mandate['id']}
    study['definition'] = {k: v for k, v in study['definition'].items() if k in {
        'name', 'as_of', 'review_date', 'currency', 'horizon_years', 'target_return', 'max_volatility',
        'max_tracking_error', 'boundary_reason', 'strategic_universe_id'}}
    diagnosed = client.post(P + '/mandates/preview', json={'definition': study['definition']})
    assert diagnosed.status_code == 200, diagnosed.text
    edited = client.post(P + '/mandates/confirm', json={
        'request': {'definition': study['definition']}, 'replaces_mandate_id': mandate['id'],
        'preview_hash': diagnosed.json()['preview_hash'], 'acknowledge_limits': True})
    assert edited.status_code == 201, edited.text
    return edited.json()


def test_mandate_edit_makes_cma_stale_and_blocks_new_saa(tmp_path):
    client, mandate, scope, cma = setup(tmp_path)
    edited = edit_mandate(client, mandate)
    item = client.get(P + '/cma').json()['items'][0]
    assert refs(item)['mandate'] == {'id': mandate['id'], 'name': '上游目标', 'number': 1, 'status': 'superseded'}
    assert next(ref for ref in item['upstream'] if ref['kind'] == 'mandate')['latest_number'] == 2
    assert item['retired'] is False and item['usable']['status'] == 'stale'
    assert item['usable']['reasons'] == [{'code': 'upstream_superseded', 'kind': 'mandate', 'name': '上游目标',
                                          'number': 1, 'latest_number': 2}]
    policy = client.post(P + '/policy/preview', json={'mandate_id': edited['id'], 'cma_id': cma['id']})
    assert policy.status_code == 422 and 'CMA_NOT_CURRENT' in policy.text


def test_saved_saa_turns_stale_when_its_mandate_is_revised(tmp_path):
    client, mandate, scope, cma = setup(tmp_path)
    request = {'mandate_id': mandate['id'], 'cma_id': cma['id'], 'candidate_count': 200}
    preview = client.post(P + '/policy/preview', json=request)
    assert preview.status_code == 200, preview.text
    saved = client.post(P + '/policies', json={'request': request, 'preview_hash': preview.json()['preview_hash'],
                                               'candidate_id': 'minimum-risk', 'name': '版本测试方案', 'reason': '验证版本传递。'})
    assert saved.status_code == 201, saved.text
    [item] = client.get(P + '/policies').json()['items']
    assert item['usable']['status'] == 'ready'
    assert [(ref['kind'], ref['number'], ref['status']) for ref in item['upstream']] == [
        ('mandate', 1, 'current'), ('cma', 1, 'current')]
    edit_mandate(client, mandate)
    [item] = client.get(P + '/policies').json()['items']
    # 方案钉住旧目标，不跟随新版本；目标与 LTCMA 都提示需更新。
    assert item['usable']['status'] == 'stale'
    assert [(ref['kind'], ref['status'], ref['usable']) for ref in item['upstream']] == [
        ('mandate', 'superseded', 'ready'), ('cma', 'current', 'stale')]


def test_scope_edit_can_upgrade_to_current_mandate_of_the_same_lineage_only(tmp_path):
    client, mandate, scope, cma = setup(tmp_path)
    edited = edit_mandate(client, mandate)
    body = {**scope['definition'], 'name': '上游范围', 'source': ''}
    body = {k: body[k] for k in ('name', 'as_of', 'currency', 'source', 'assets')}
    preview = client.post(P + '/universes/preview', json=body).json()['preview_hash']
    upgraded = client.post(P + '/universes/confirm', json={'request': body, 'preview_hash': preview,
                                                           'replaces_universe_id': scope['id'], 'mandate_id': edited['id']})
    assert upgraded.status_code == 201, upgraded.text
    assert upgraded.json()['mandate_id'] == edited['id'] and upgraded.json()['supersedes_universe_id'] == scope['id']
    [current] = [item for item in client.get(P + '/catalog').json()['strategic_universes'] if item['id'] == upgraded.json()['id']]
    assert current['version']['number'] == 2 and current['usable']['status'] == 'ready'
    # 旧 LTCMA 钉住旧范围 v1：需更新，可在修改中改选范围 v2。
    item = client.get(P + '/cma').json()['items'][0]
    assert refs(item)['strategic_scope']['status'] == 'superseded' and item['usable']['status'] == 'stale'
    other = client.post(P + '/universes/confirm', json={'request': body, 'preview_hash': preview,
                                                        'replaces_universe_id': upgraded.json()['id'], 'mandate_id': mandate['id']})
    assert other.status_code == 409 and 'SCOPE_MANDATE_CONFLICT' in other.text


@pytest.mark.parametrize('limits,expected', [({'min_weight': 0, 'max_weight': .2}, 'stale'),
                                            ({'min_weight': .1, 'max_weight': 1}, 'stale'),
                                            ({'min_weight': 0, 'max_weight': 1}, 'ready')])
def test_scope_equivalence_checks_weight_limits_without_rewriting_history(tmp_path, limits, expected):
    from copy import deepcopy
    client, mandate, scope, cma = setup(tmp_path)
    body = deepcopy(scope['definition'])
    body['assets'][0]['weight_limits'] = limits
    preview = client.post(P + '/universes/preview', json=body)
    updated = client.post(P + '/universes/confirm', json={
        'request': body, 'preview_hash': preview.json()['preview_hash'], 'replaces_universe_id': scope['id']})
    assert updated.status_code == 201, updated.text
    # This old economic fingerprint intentionally excludes allocation limits.
    assert updated.json()['scope_fingerprint'] == scope['scope_fingerprint']
    item = client.get(P + '/cma').json()['items'][0]
    assert item['usable']['status'] == expected
    assert item['upstream'][-1]['equivalent'] is (expected == 'ready')
    result = client.post(P + '/policy/preview', json={
        'mandate_id': mandate['id'], 'cma_id': cma['id'], 'candidate_count': 200})
    assert result.status_code == (422 if expected == 'stale' else 200), result.text
    if expected == 'stale':
        assert 'CMA_NOT_CURRENT' in result.text
    assert client.get(P + '/cma/' + cma['id']).json() == cma


@pytest.mark.parametrize('deleted', ['scope', 'mandate'])
def test_deleting_series_head_blocks_older_dependencies(tmp_path, deleted):
    client, mandate, scope, cma = setup(tmp_path)
    if deleted == 'mandate':
        head = edit_mandate(client, mandate)
        path = '/mandates/'
    else:
        body = {**scope['definition'], 'source': '仅修改说明'}
        preview = client.post(P + '/universes/preview', json=body)
        head = client.post(P + '/universes/confirm', json={
            'request': body, 'preview_hash': preview.json()['preview_hash'],
            'replaces_universe_id': scope['id']}).json()
        path = '/universes/'
    assert client.delete(P + path + head['id']).status_code == 200
    assert client.get(P + '/cma').json()['items'] == []
    item = client.get(P + '/cma', params={'include_retired': True}).json()['items'][0]
    assert item['usable']['status'] == 'blocked' and item['retired']
    assert any(r['code'] == 'upstream_deleted' for r in item['usable']['reasons'])
    assert client.get(P + '/cma/' + cma['id']).json() == cma
    assert client.get(P + '/cma/study-options', params={'section': 'base'}).json()['existing_names'] == []


def test_deleted_lineage_preserves_version_identity_for_every_artifact_kind():
    from backend.strategic_allocation.versioning import STRATEGIC_TYPES, version_index, reference, usability, single_version
    for parent_key in [value[1] for value in STRATEGIC_TYPES.values()] + ['supersedes_snapshot_id']:
        index = version_index([{'id': 'old'}, {'id': 'head', parent_key: 'old'}], parent_key, ['head'])
        assert index['old']['status'] == 'superseded'
        assert index['old']['number'] == 1 and index['old']['latest_id'] is None
        assert usability(index['old'], [])['status'] == 'blocked'
        upstream = reference('cma', 'old', '历史', index['old'])
        assert usability(single_version(lineage_id='downstream'), [upstream])['status'] == 'blocked'
