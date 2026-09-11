"""Roundtrip, safe syntax and mode checks for the isolated regime editor."""
import copy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from historical_regimes.v2_contracts import definition_content_hash, parse_definition_v2
from historical_regimes.v2_templates import list_templates_v2, instantiate_template_v2


def resolve(definition, source_kind='graph', source='', mode='retrospective'):
    return resolve_authoring(AuthoringRequest(definition=definition, source_kind=source_kind, source=source, mode=mode))


@pytest.mark.parametrize('template', list_templates_v2(), ids=lambda t: t['id'])
def test_all_templates_graph_formula_roundtrip(template):
    raw = template['definition']
    first = resolve(raw)
    assert first['valid'], first['diagnostics']
    again = resolve(raw, 'formula', first['source'])
    assert again['valid'], again['diagnostics']
    assert first['source'] == again['source']
    assert definition_content_hash(parse_definition_v2(raw)) == definition_content_hash(parse_definition_v2(again['definition']))
    assert again['compile_status'] == 'not_requested'


@pytest.mark.parametrize('source', [
    'import os', 'x = __import__("os")', 'x = open("/tmp/secret")',
    'x = source_inline(**{})', 'x = source_inline(rows=print(1))',
    'x = source_inline(rows=[i for i in range(10)])', 'while True: pass',
    'x = source_inline()\nx = source_inline()', 'output(state=x.__class__.__base__)',
    'output(state=x.state)\noutput(state=x.state)', 'x = source_inline(_id=1)',
    'x = source_inline(rows=[])\noutput(state=x.value, state=x.value)',
])
def test_unsafe_or_ambiguous_source_rejected(source):
    result = resolve(instantiate_template_v2('peak-trough-daily-v2'), 'formula', source)
    assert not result['valid'] and result['definition'] is None
    assert result['diagnostics']


def test_constants_edit_without_touching_pivots_and_mode_prohibition():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    first = resolve(raw)
    source = first['source'].replace('value=0.03', 'value=0.05')
    edited = resolve(raw, 'formula', source)
    assert edited['valid'], edited['diagnostics']
    assert edited['definition']['graph']['nodes'][4]['parameters']['value'] == .05
    assert edited['definition']['graph']['nodes'][1] == first['definition']['graph']['nodes'][1]
    blocked = resolve(raw, 'formula', source, 'realtime')
    assert not blocked['valid']
    assert any(d['code'] == 'NON_CAUSAL_REALTIME_GRAPH' for d in blocked['diagnostics'])


def test_identifier_collision_label_version_and_shared_input_preserved():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    graph = raw['graph']
    mapping = {'market': 'market-x', 'pivots': 'market_x'}
    for node in graph['nodes']:
        node['id'] = mapping.get(node['id'], node['id'])
        for ref in node.get('inputs', {}).values():
            ref['node_id'] = mapping.get(ref['node_id'], ref['node_id'])
    graph['exposed_node_ids'] = [mapping.get(x, x) for x in graph['exposed_node_ids']]
    first = resolve(raw)
    assert first['valid'], first['diagnostics']
    again = resolve(raw, 'formula', first['source'])
    assert again['valid'], again['diagnostics']
    assert first['definition'] == again['definition']
    bad = resolve(raw, 'formula', first['source'].replace('_version=1', '_version=999', 1))
    assert not bad['valid']


def test_real_authoring_route_is_readonly(monkeypatch):
    from services import historical_regime_routes
    monkeypatch.setattr(historical_regime_routes.regime_graph_v2_service, 'prepare', lambda *_: pytest.fail('must not prepare'))
    app = FastAPI(); app.include_router(historical_regime_routes.router)
    response = TestClient(app).post('/api/historical-regimes/authoring/resolve', json={
        'definition': instantiate_template_v2('peak-trough-daily-v2'), 'source_kind': 'graph', 'mode': 'retrospective'})
    assert response.status_code == 200
    assert response.json()['valid'], response.json()
    assert response.json()['compile_status'] == 'not_requested'
