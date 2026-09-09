import copy

import pytest
from custom_indicators.errors import ValidationError

from historical_regimes.node_preview import node_preview_definition
from historical_regimes.v2_contracts import validate_definition_v2
from historical_regimes.v2_numba import regime_graph_numba_status
from test_historical_regime_v2 import client, classic_service, v2_service, _definition, _wait_for_preview


def unfinished():
    definition = _definition()
    definition['graph']['outputs'] = {}
    definition['graph']['nodes'].append({'id': 'unfinished', 'type': 'model.threshold', 'inputs': {}, 'parameters': {'upper': None}})
    definition['evaluation_targets'] = [{'id': 'unfinished'}]
    return definition


def start(client, definition, node='source', port='value', mode='realtime', as_of=None):
    target = {'node_id': node, 'port': port}
    prepared = client.post('/api/historical-regimes/prepare', json={'definition': definition, 'preview_target': target})
    assert prepared.status_code == 200, prepared.text
    response = client.post('/api/historical-regimes/preview-runs', json={
        'definition': definition, 'preview_target': target,
        'compile_token': prepared.json()['compile_token'], 'mode': mode, 'as_of': as_of,
    })
    return response, prepared.json()


def test_single_node_preview_ignores_unfinished_graph_without_mutating_or_saving(client, v2_service):
    definition = unfinished()
    original = copy.deepcopy(definition)
    response, _ = start(client, definition)
    assert response.status_code == 202, response.text
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    assert job['result']['result_kind'] == 'node_preview'
    assert job['result']['diagnostics']['required_node_ids'] == ['source']
    rows = client.get(f"/api/historical-regimes/preview-runs/{job['id']}/series").json()
    assert rows['total'] == 80
    assert rows['items'][0]['value'] == 100
    assert rows['node_id'] == 'source'
    assert client.get(f"/api/historical-regimes/preview-runs/{job['id']}/overview").status_code == 409
    assert v2_service.list_definitions() == []
    assert definition == original
    assert client.post('/api/historical-regimes/prepare', json={'definition': definition}).status_code == 422


def test_partial_transform_uses_same_warmed_kernels_and_rejects_incomplete_ancestors(client):
    definition = unfinished()
    before = regime_graph_numba_status()['kernel_signatures']
    response, plan = start(client, definition, 'returns')
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    rows = client.get(f"/api/historical-regimes/preview-runs/{job['id']}/series").json()['items']
    assert rows[0]['value'] is None
    assert rows[1]['value'] == pytest.approx(0.01)
    assert job['result']['diagnostics']['python_fallback'] == 0
    assert regime_graph_numba_status()['kernel_signatures'] == before
    definition['graph']['nodes'][1]['inputs'] = {}
    response = client.post('/api/historical-regimes/prepare', json={'definition': definition, 'preview_target': {'node_id': 'returns', 'port': 'value'}})
    assert response.status_code == 422


def test_target_and_token_are_bound_and_preview_projection_cannot_be_saved(client):
    definition = unfinished()
    _, plan = start(client, definition)
    response = client.post('/api/historical-regimes/preview-runs', json={'definition': definition, 'preview_target': {'node_id': 'returns', 'port': 'value'}, 'compile_token': plan['compile_token']})
    assert response.status_code == 422
    for target in ({'node_id': 'missing', 'port': 'value'}, {'node_id': 'source', 'port': 'missing'}):
        assert client.post('/api/historical-regimes/prepare', json={'definition': definition, 'preview_target': target}).status_code == 422
    projected = node_preview_definition(definition, {'node_id': 'source', 'port': 'value'}).model_dump(mode='json')
    assert client.post('/api/historical-regimes/v2/definitions', json={'definition': projected}).status_code == 422


def test_realtime_guard_applies_to_preview_ancestors_and_asof_is_preserved(client):
    definition = unfinished()
    definition['graph']['nodes'].append({'id': 'pivots', 'type': 'pivot.local_extrema', 'inputs': {'value': {'node_id': 'source', 'port': 'value'}}, 'parameters': {}})
    response, _ = start(client, definition, as_of='2020-01-11')
    job = _wait_for_preview(client, response.json()['id'])
    rows = client.get(f"/api/historical-regimes/preview-runs/{job['id']}/series").json()
    assert rows['total'] == 10
    response, _ = start(client, definition, 'pivots', 'pivot', mode='realtime')
    assert response.status_code == 422
    assert 'NON_CAUSAL_REALTIME_GRAPH' in response.text


def test_cycles_and_conflicting_edges_are_rejected_in_selected_dependencies():
    definition = unfinished()
    definition['graph']['nodes'][0]['inputs'] = {'value': {'node_id': 'returns', 'port': 'value'}}
    with pytest.raises(ValidationError):
        validate_definition_v2(node_preview_definition(definition, {'node_id': 'returns', 'port': 'value'}))
    definition = unfinished()
    definition['graph']['edges'] = [{'source': {'node_id': 'returns', 'port': 'value'}, 'target': {'node_id': 'returns', 'port': 'value'}}]
    with pytest.raises(ValidationError):
        node_preview_definition(definition, {'node_id': 'returns', 'port': 'value'})


def test_upstream_catalog_uses_frozen_labels_and_only_transitive_ancestors(client, v2_service):
    definition = unfinished()
    definition['graph']['nodes'][0]['label'] = '沪深300'
    definition['graph']['nodes'][1]['label'] = '日收益'
    definition['graph']['nodes'][2]['label'] = '平滑结果'
    response, _ = start(client, definition, 'smooth', as_of='2020-01-11')
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    definition['graph']['nodes'][0]['label'] = '已改名但未试算'
    signatures = regime_graph_numba_status()['kernel_signatures']
    path = f"/api/historical-regimes/preview-runs/{job['id']}/series"
    result = client.get(path, params={'node_id': 'smooth'}).json()
    assert result['node_label'] == '平滑结果'
    assert [(x['node_id'], x['node_label'], x['distance']) for x in result['upstream_outputs']] == [('returns', '日收益', 1), ('source', '沪深300', 2)]
    for item in result['upstream_outputs']:
        assert item['plottable'] and item['port_label'] == '数值序列'
        series = client.get(path, params={'node_id': item['node_id'], 'port': item['port']}).json()
        assert series['total'] == result['total'] == 10
        assert series['items'][-1]['observation_date'] == '2020-01-10'
    assert client.get(path, params={'node_id': 'source'}).json()['upstream_outputs'] == []
    assert client.get(path, params={'node_id': 'unfinished'}).status_code == 404
    assert regime_graph_numba_status()['kernel_signatures'] == signatures


def test_upstream_catalog_deduplicates_shared_ancestors_and_works_with_edges(client):
    definition = unfinished()
    definition['graph']['nodes'].append({'id': 'sum', 'type': 'math.add', 'inputs': {'left': {'node_id': 'returns', 'port': 'value'}, 'right': {'node_id': 'smooth', 'port': 'value'}}})
    definition['graph']['edges'] = [{'source': ref, 'target': {'node_id': node['id'], 'port': name}} for node in definition['graph']['nodes'] for name, ref in node.get('inputs', {}).items()]
    response, _ = start(client, definition, 'sum')
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    result = client.get(f"/api/historical-regimes/preview-runs/{job['id']}/series").json()
    ids = [item['node_id'] for item in result['upstream_outputs']]
    assert ids == ['returns', 'smooth', 'source']


def test_upstream_catalog_marks_enum_outputs_unavailable_for_numeric_overlay(client):
    response, _ = start(client, unfinished(), 'confirmed', 'state')
    job = _wait_for_preview(client, response.json()['id'])
    assert job['status'] == 'completed', job
    result = client.get(f"/api/historical-regimes/preview-runs/{job['id']}/series").json()
    state = next(item for item in result['upstream_outputs'] if item['node_id'] == 'classifier' and item['port'] == 'state')
    assert state['plottable'] is False
    assert state['unavailable_reason']
    assert next(item for item in result['upstream_outputs'] if item['node_id'] == 'source')['plottable'] is True
