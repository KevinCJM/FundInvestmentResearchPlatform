"""ETL canvas persistence and actual graph-to-executor compilation, offline."""
import copy
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'backend'):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from backend.data_sources import acquisition, etl_service as etl
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.etl_graph import graph_schemas
from backend.data_sources.store import SourceStore
from backend.data_sources.models import CenterError
from backend.tests.test_etl_workflows import plan, payload, finished, nav_fetch


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(etl, '_launch', etl._launch_inline)
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    result = SourceStore(tmp_path); result.seed(); return result


def graph(store):
    result = plan(store)
    result['graph_version'] = 1
    result['canvas'] = {'version': 1, 'positions': {'download': {'x': 321, 'y': 55}}, 'viewport': {'x': 12, 'y': 34, 'zoom': .7}}
    return result


def test_unordered_dag_compiles_and_actual_executor_preserves_data_inputs(store, monkeypatch):
    definition = graph(store); definition['steps'].reverse()
    monkeypatch.setattr(acquisition, 'fetch_with_retry', nav_fetch)
    saved = etl.save_workflow(store, 'graph_flow', definition, 0)
    assert saved['definition']['steps'][0]['id'] == 'resolve'
    assert saved['definition']['canvas']['positions']['download']['x'] == 321
    validated = etl.validate(store, definition)
    assert validated['valid']
    assert [step['id'] for step in validated['steps']] == ['download', 'mapping', 'resolve']
    result = finished(store, etl.start(store, payload(definition)))
    assert result['status'] == 'SUCCEEDED', result
    assert [step['id'] for step in result['steps']] == ['download', 'mapping', 'resolve']
    assert result['definition']['canvas'] is None
    assert result['template_definition']['canvas'] == saved['definition']['canvas']
    assert result['steps'][-1]['output']['rows'] == 1
    assert not (store.root / 'tushare_active.json').exists()


def test_control_dependency_changes_order_but_is_not_data(store, monkeypatch):
    definition = graph(store)
    first = definition['steps'][0]
    other = {**first, 'id': 'other', 'name': 'other', 'params': {**first['params'], 'symbol': '000002'}}
    first['after'] = ['other']
    definition['steps'].append(other)
    calls = []
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *args: (calls.append(args[3].get('symbol')) or nav_fetch()))
    result = finished(store, etl.start(store, payload(definition)))
    assert result['status'] == 'SUCCEEDED'
    assert [s['id'] for s in result['steps']] == ['other', 'download', 'mapping', 'resolve']
    assert result['definition']['steps'][1]['inputs'] == []
    assert result['definition']['steps'][1]['after'] == ['other']
    assert len(calls) == 2


@pytest.mark.parametrize('change', ['self', 'cycle', 'missing', 'duplicate', 'wrong_data', 'layout_ghost', 'layout_nan'])
def test_bad_graphs_fail_before_network(store, monkeypatch, change):
    definition = graph(store)
    if change == 'self': definition['steps'][0]['after'] = ['download']
    elif change == 'cycle': definition['steps'][0]['after'] = ['resolve']
    elif change == 'missing': definition['steps'][0]['after'] = ['ghost']
    elif change == 'duplicate': definition['steps'][2]['after'] = ['download', 'download']
    elif change == 'wrong_data': definition['steps'][1].update(inputs=[], after=['download'])
    elif change == 'layout_ghost': definition['canvas']['positions']['ghost'] = {'x': 1, 'y': 2}
    else: definition['canvas']['positions']['download']['x'] = float('nan')
    assert not etl.validate(store, definition)['valid']
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *a: pytest.fail('unexpected network'))
    with pytest.raises(CenterError): etl.start(store, payload(definition))


def test_old_list_definitions_retain_forward_reference_rejection(store):
    definition = plan(store); definition['steps'].reverse()
    assert not etl.validate(store, definition)['valid']


def test_layout_does_not_change_compiled_business_definition(store):
    first = etl.parse_definition(graph(store))
    changed = copy.deepcopy(first.model_dump())
    changed['canvas']['positions']['download'] = {'x': -500, 'y': 999}
    second = etl.parse_definition(changed)
    assert first.compiled().model_dump() == second.compiled().model_dump()


def test_port_catalog_is_source_neutral_and_control_is_optional():
    schemas = graph_schemas()
    assert {s['id'] for s in schemas} == {'download', 'map', 'resolve', 'snapshot', 'task'}
    for schema in schemas:
        control = next(p for p in schema['inputs'] if p['id'] == 'after')
        assert control['value_type'] == 'control' and not control['required'] and control['multiple']
    assert next(s for s in schemas if s['id'] == 'map')['inputs'][0]['multiple'] is False
