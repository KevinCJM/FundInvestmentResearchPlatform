"""Presentation contracts: actual graph parameters, roots and safe mathematical text."""
import copy

import pytest

from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from historical_regimes.v2_templates import instantiate_template_v2, list_templates_v2


def resolve(definition, **kwargs):
    return resolve_authoring(AuthoringRequest(definition=definition, source_kind='graph', **kwargs))


@pytest.mark.parametrize('template', list_templates_v2(), ids=lambda item: item['id'])
def test_all_templates_have_math_for_every_output(template):
    raw = copy.deepcopy(template['definition'])
    before = copy.deepcopy(raw)
    result = resolve(raw)
    assert result['valid'], result['diagnostics']
    assert not result.get('math_error')
    assert set(result['display_latex']) == set(raw['graph']['outputs'])
    for output, steps in result['formula_steps'].items():
        assert steps and steps[-1]['node_id'] == raw['graph']['outputs'][output]['node_id']
        assert len({(step['node_id'], step['port']) for step in steps}) == len(steps)
    assert raw == before
    assert result['compile_status'] == 'not_requested'


def test_peak_thresholds_and_interval_fraction_use_actual_values_and_bounds():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    raw['graph']['nodes'][4]['parameters']['value'] = .12
    raw['graph']['nodes'][6]['parameters'] = {'upper': 99, 'lower': -99}
    result = resolve(raw)
    latex = result['display_latex']['state']
    assert r'\begin{cases}' in latex and '>0.12' in latex and '<-0.03' in latex
    assert '99' not in latex  # connected constant wins over unused parameter
    assert r'\le' in latex and r'\neg\mathcal{V}_t' in latex and '未识别' in latex
    steps = result['formula_steps']['state']
    change = next(step for step in steps if step['node_id'] == 'change')
    assert r'\frac{' in change['latex']
    pivots = next(step for step in steps if step['node_id'] == 'pivots')
    assert {'左窗口', '右窗口', '首窗口', '尾窗口'} <= {p['label'] for p in pivots['parameters']}


def test_numeric_formulas_use_shared_renderer_with_aliases_and_per_root_dependencies():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    raw['graph']['nodes'].append({'id': 'ratio', 'type': 'feature.formula', 'parameters': {
        'expression': 'close / rolling_mean(close, 20, 1) - 1', 'variables': {'close': 'feature_1'}},
        'inputs': {'feature_1': {'node_id': 'market', 'port': 'value'}}})
    raw['graph']['outputs']['ratio'] = {'node_id': 'ratio', 'port': 'value'}
    raw['graph']['channel_metadata'] = {'ratio': {'label': '变化_2% & {净值}'}}
    result = resolve(raw)
    assert result['valid'], result['diagnostics']
    assert not result.get('math_error')
    latex = result['display_latex']['ratio']
    assert r'\frac{' in latex and r'\_' in latex and r'\%' in latex and r'\&' in latex
    assert 'rolling_mean(' not in latex and 'close' not in latex
    assert [step['node_id'] for step in result['formula_steps']['ratio']] == ['market', 'ratio']
    assert 'ratio' not in [step['node_id'] for step in result['formula_steps']['state']]


def test_typed_registry_window_default_and_parameter_changes_are_rendered():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    raw['graph']['nodes'].append({'id': 'smooth', 'type': 'indicator.rolling_mean', 'parameters': {},
        'inputs': {'values': {'node_id': 'market', 'port': 'value'}}})
    raw['graph']['outputs']['smooth'] = {'node_id': 'smooth', 'port': 'value'}
    raw['graph']['channel_metadata'] = {'smooth': {'label': '平滑'}}
    first = resolve(raw)
    assert r'\mu_{t,20}' in first['display_latex']['smooth']
    raw['graph']['nodes'][-1]['parameters']['window'] = 60
    assert r'\mu_{t,60}' in resolve(raw)['display_latex']['smooth']


def test_math_presentation_does_not_bypass_realtime_rejection():
    result = resolve(instantiate_template_v2('peak-trough-daily-v2'), mode='realtime')
    assert not result['valid']
    assert any(item['code'] == 'NON_CAUSAL_REALTIME_GRAPH' for item in result['diagnostics'])
    assert result['compile_status'] == 'not_requested'


def test_invalid_graph_has_no_misleading_mathematical_preview():
    raw = instantiate_template_v2('peak-trough-daily-v2')
    raw['graph']['nodes'][-1]['inputs']['value']['node_id'] = 'missing'
    result = resolve(raw)
    assert not result['valid'] and not result.get('display_latex')
