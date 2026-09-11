"""Real graph execution of current and locked built-ins in both centers."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cal_indicators.typed_dsl import compose_typed_expression, TypedDslError
from computation_graph.causal_series import causal_violations, series_variable_types, SYSTEM_CONTEXT_NAMES
from custom_indicators.errors import ValidationError
from custom_indicators.runtime_context import single_product_scalar_context
from custom_indicators.service import _built_in_indicators
from historical_regimes.formula import CAUSAL_OPERATOR_IDS
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.v2_templates import instantiate_template_v2
from historical_regimes.v2_contracts import parse_definition_v2, inspect_definition_v2
from timing_research.catalog import build_catalog
from timing_research.contracts import Definition
from timing_research.graph import GraphRuntime, _formula_plan
from test_builtin_series_rolling_migration import IDS, inputs, compile_definition, compute


@pytest.fixture(scope='module')
def source_service():
    definitions = [item for item in _built_in_indicators() if item.get('result_kind') == 'time_series' and item['revision'] in {1, 2, 3}]
    return SimpleNamespace(indicators=SimpleNamespace(list_all_versions=lambda: definitions),
        get_indicator=lambda identifier, revision: copy.deepcopy(next(x for x in definitions if x['id'] == identifier and x['revision'] == revision)))


@pytest.fixture(scope='module')
def regime_service(tmp_path_factory, source_service):
    from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
    warm_historical_regime_numba_kernels()
    root = tmp_path_factory.mktemp('cross-center-regime')
    return RegimeGraphV2Service(root, root, indicator_service=source_service)


def _context(definition, missing=False):
    data = inputs(80, missing=False)
    if missing:
        data['market_low'][6:10] = np.nan
        data['market_high'][20] = np.nan
    dates = data['observation_dates'].astype('int64').astype('datetime64[D]')
    data.update(single_product_scalar_context(definition, dates, returns=data['returns']))
    for value in data.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return data, dates


def _regime_graph(metadata, data, dates):
    raw = instantiate_template_v2('peak-trough-daily-v2')
    nodes, bindings = [], {}
    ports = metadata['inputs']
    # A KDJ fixture shares one proven source axis. High/low derive transparently
    # from that source; unrelated axes must NOT be silently declared aligned.
    is_kdj = {p['name'] for p in ports} == {'market_close', 'market_high', 'market_low'}
    if is_kdj:
        ports = [p for p in ports if p['name'] == 'market_close']
    for index, port in enumerate(ports):
        name, identifier = port['name'], f'source_{index}'
        assert name not in SYSTEM_CONTEXT_NAMES
        nodes.append({'id': identifier, 'type': 'source.inline', 'parameters': {'frequency': 'daily',
            'rows': [{'observation_date': str(day), 'available_at': str(day), 'value': float(v) if np.isfinite(v) else None}
                     for day, v in zip(dates, data[name])]}})
        bindings[name] = {'node_id': identifier, 'port': 'value'}
    if is_kdj:
        for field, offset in [('market_high', .8), ('market_low', -.9)]:
            nodes.append({'id': field, 'type': 'feature.formula', 'inputs': {'feature_1': bindings['market_close']},
                          'parameters': {'expression': f'feature_1 + ({offset})'}})
            bindings[field] = {'node_id': field, 'port': 'value'}
    nodes.append({'id': 'metric', 'type': metadata['id'], 'inputs': bindings, 'parameters': {}})
    first = metadata['outputs'][0]['name']
    nodes.append({'id': 'classifier', 'type': 'model.range_threshold', 'inputs': {'value': {'node_id': 'metric', 'port': first}},
                  'parameters': {'lower': 0., 'upper': 1.}})
    outputs = {'state': {'node_id': 'classifier', 'port': 'state'}}
    outputs.update({port['name']: {'node_id': 'metric', 'port': port['name']} for port in metadata['outputs']})
    raw['graph'] = {'nodes': nodes, 'outputs': outputs, 'exposed_node_ids': [],
                    'channel_metadata': {port['name']: {'label': port.get('label') or port['name']}
                                         for port in metadata['outputs']}}
    return raw


@pytest.mark.parametrize('revision', [1, 2, 3])
@pytest.mark.parametrize('indicator_id', IDS)
def test_regime_versioned_execution_matches_indicator_engine(source_service, regime_service, indicator_id, revision, monkeypatch):
    definition = source_service.get_indicator(indicator_id, revision)
    metadata = next(x for x in NODE_REGISTRY.values() if x.get('indicator_reference', {}).get('id') == indicator_id
                    and x['indicator_reference']['revision'] == revision)
    assert metadata.get('available'), metadata.get('unavailable_reason')
    data, dates = _context(definition)
    if 'kdj' in indicator_id:
        for name in ('market_close', 'market_high', 'market_low'):
            data[name] = data[name].copy()
            data[name][6] = np.nan
            data[name].setflags(write=False)
    raw = _regime_graph(metadata, data, dates)
    inspection = inspect_definition_v2(parse_definition_v2(raw))
    assert inspection['valid'], inspection
    expected_plan, expected_compiled = compile_definition(definition)
    expected = dict(zip(expected_compiled.channel_names, compute(expected_compiled, data)))
    prepared = regime_service.prepare(raw)
    import historical_regimes.formula as formula
    monkeypatch.setattr(formula, 'compile_numba_series_plan', lambda *_a, **_k: pytest.fail('formal execution must not compile'))
    result = regime_service._execute_graph(None, parse_definition_v2(raw), 'realtime', None, plan=prepared)
    for port, values in expected.items():
        actual = result['node_outputs']['metric'][port]
        np.testing.assert_allclose(actual.values, values, equal_nan=True, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(actual.dates, dates.astype('datetime64[ns]').astype(np.int64))
        audit = result['result']['diagnostics']['formula_audits'][f'metric:{port}' if port != 'value' else 'metric']
        assert audit['python_fallback'] == 0
        assert audit['request_time_compilation'] == 0


def _timing_graph(metadata):
    nodes, connections = [], {}
    for index, port in enumerate(metadata['inputs']):
        name = port['name']
        assert name not in SYSTEM_CONTEXT_NAMES
        if name == 'returns':
            nodes.extend([
                {'id': 'nav', 'label': '净值', 'op': 'source', 'parameters': {'field': 'close'}},
                {'id': 'prior', 'label': '前值', 'op': 'indicator.lag', 'inputs': {'values': 'nav.value'}, 'parameters': {'periods': 1}},
                {'id': 'returns', 'label': '收益', 'op': 'formula', 'inputs': {'a': 'nav.value', 'b': 'prior.value'}, 'parameters': {'expression': 'a/b-1'}},
            ])
            connections[name] = 'returns.value'
        else:
            field = {'market_close': 'close', 'market_high': 'high', 'market_low': 'low', 'volume': 'volume', 'adjusted_nav': 'close'}[name]
            identifier = f'source_{index}'
            nodes.append({'id': identifier, 'label': name, 'op': 'source', 'parameters': {'field': field}})
            connections[name] = f'{identifier}.value'
    nodes.append({'id': 'metric', 'label': '引用指标', 'op': metadata['id'], 'inputs': connections})
    previous = None
    for index, port in enumerate(metadata['outputs']):
        identifier = f'condition_{index}'
        nodes.append({'id': identifier, 'label': '条件', 'op': 'compare', 'inputs': {'left': f'metric.{port["name"]}'}, 'parameters': {'threshold': 0.}})
        if previous:
            nodes.append({'id': f'joint_{index}', 'label': '合取', 'op': 'all', 'inputs': {'left': previous, 'right': f'{identifier}.value'}})
            previous = f'joint_{index}.value'
        else:
            previous = f'{identifier}.value'
    return Definition(name='跨中心滚动验证', nodes=nodes, entry=previous)


@pytest.mark.parametrize('revision', [1, 2, 3])
@pytest.mark.parametrize('indicator_id', IDS)
def test_timing_versioned_execution_matches_indicator_engine(source_service, indicator_id, revision, monkeypatch):
    definition = source_service.get_indicator(indicator_id, revision)
    catalog = build_catalog(source_service)
    metadata = next(x for x in catalog.values() if x.get('indicator_reference', {}).get('id') == indicator_id
                    and x['indicator_reference']['revision'] == revision)
    assert metadata['indicator_reference']['revision'] == revision
    data, dates = _context(definition, missing='kdj' in indicator_id)
    _, compiled = compile_definition(definition)
    expected = dict(zip(compiled.channel_names, compute(compiled, data)))
    runtime = GraphRuntime(catalog)
    prepared = runtime.prepare(_timing_graph(metadata))
    assert len({id(prepared.formulas[f'metric.{port}']) for port in expected}) == 1
    import timing_research.graph as graph
    monkeypatch.setattr(graph, 'compile_numba_series_plan', lambda *_a, **_k: pytest.fail('formal execution must not compile'))
    bars = SimpleNamespace(dates=dates.astype(np.int64), close=data['adjusted_nav'] if 'sharpe' in indicator_id else data['market_close'],
                           high=data['market_high'], low=data['market_low'], volume=data['volume'])
    actual = runtime.evaluate(prepared, bars)
    for port, values in expected.items():
        np.testing.assert_allclose(actual[f'metric.{port}'], values, equal_nan=True, rtol=1e-10, atol=1e-12)
    assert runtime.audit(prepared)['python_fallback'] == 0


@pytest.mark.parametrize('expression', [
    'a + mean(a)', 'rolling_apply(mean(a),3) + mean(a)', 'a * observation_count',
])
def test_scope_does_not_launder_full_sample_value_uses(expression):
    plan = compose_typed_expression(expression, variable_types=series_variable_types(['a']), output_contract='series')
    assert causal_violations(plan, CAUSAL_OPERATOR_IDS)
    with pytest.raises(ValidationError):
        _formula_plan(expression, {'a': 'source.value'})


def test_system_context_cannot_be_connected_as_user_data():
    with pytest.raises(TypedDslError, match='系统上下文'):
        series_variable_types(['a', 'annual_risk_free_rate_decimal'])
