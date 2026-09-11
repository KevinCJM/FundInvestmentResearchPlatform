import numpy as np
import pytest
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.temporal_capability import analyze_temporal
from historical_regimes.temporal_audit import audit_execution
from causality.temporal_numba import TEMPORAL_KERNELS, compare_available_prefix, perturb_available_tail
from custom_indicators.errors import ValidationError


def definition(count=96):
    dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-01-01') + count)
    return {'schema_version': '2.0', 'name': '时点审计', 'graph': {'nodes': [
        {'id': 'market', 'type': 'source.inline', 'parameters': {'rows': [
            {'observation_date': str(d), 'available_at': str(d), 'value': float(100 + i + np.sin(i))} for i, d in enumerate(dates)]}},
        {'id': 'returns', 'type': 'transform.return', 'inputs': {'value': {'node_id': 'market', 'port': 'value'}}},
        {'id': 'classifier', 'type': 'model.threshold', 'parameters': {'upper': .01, 'lower': -.01}, 'inputs': {'value': {'node_id': 'returns', 'port': 'value'}}},
    ], 'outputs': {'state': {'node_id': 'classifier', 'port': 'state'}}},
        'states': [{'id': 'bull', 'label': '牛', 'role': 'positive', 'order': 0}, {'id': 'flat', 'label': '平', 'role': 'neutral', 'order': 1}, {'id': 'bear', 'label': '熊', 'role': 'negative', 'order': 2}],
        'validation': {'walk_forward': False}, 'usage_intent': 'research_display'}


@pytest.fixture
def service(tmp_path):
    return RegimeGraphV2Service(workspace_data_dir=tmp_path, market_data_dir=tmp_path)


def test_unused_hindsight_does_not_contaminate_selected_output():
    data = definition()
    data['graph']['nodes'].append({'id': 'hindsight', 'type': 'pivot.local_extrema', 'inputs': {'value': {'node_id': 'market', 'port': 'value'}}})
    parsed = parse_definition_v2(data)
    result = analyze_temporal(parsed, NODE_REGISTRY)
    assert result['status'] == 'conditional'
    assert 'hindsight' not in result['outputs']['state']['node_ids']
    RegimeGraphV2Service._validate_realtime_graph(parsed, 'realtime')


def test_manual_hindsight_is_not_numerical_repainting():
    from historical_regimes.v2_templates import instantiate_template_v2
    parsed = parse_definition_v2(instantiate_template_v2('manual-historical-events-v1'))
    report = analyze_temporal(parsed, NODE_REGISTRY)
    assert report['semantic_hindsight'] and not report['may_repaint']
    assert report['status'] == 'retrospective_required'
    with pytest.raises(ValidationError):
        RegimeGraphV2Service._validate_realtime_graph(parsed, 'realtime')


def test_model_capability_depends_on_mode_and_output():
    data = definition()
    data['graph']['nodes'][1] = {'id': 'returns', 'type': 'feature.matrix', 'inputs': {'feature_1': {'node_id': 'market', 'port': 'value'}}}
    data['graph']['nodes'][2] = {'id': 'classifier', 'type': 'model.hmm', 'parameters': {'initial_train_size': 20}, 'inputs': {'features': {'node_id': 'returns', 'port': 'features'}}}
    parsed = parse_definition_v2(data)
    assert analyze_temporal(parsed, NODE_REGISTRY, 'realtime')['status'] == 'conditional'
    assert analyze_temporal(parsed, NODE_REGISTRY, 'retrospective')['status'] == 'retrospective_required'
    parsed.graph.outputs['state'].port = 'score'
    assert analyze_temporal(parsed, NODE_REGISTRY, 'retrospective')['status'] == 'conditional'


def test_real_njit_graph_tail_and_prefix_audit(service):
    payload = definition(); parsed = parse_definition_v2(payload); plan = service.prepare(payload)
    execution = service._execute_graph(None, parsed, 'realtime', None, plan=plan)
    raw = execution['node_outputs']['market']['value'].values.copy()
    before = {k: tuple(v.signatures) for k, v in TEMPORAL_KERNELS.items()}
    report = audit_execution(service, parsed, 'realtime', None, plan, execution)
    assert report['numerical_verdict'] == 'causal', report
    assert report['verified'], report
    assert report['coverage']['executions'] >= 12
    np.testing.assert_array_equal(execution['node_outputs']['market']['value'].values, raw)
    assert {k: tuple(v.signatures) for k, v in TEMPORAL_KERNELS.items()} == before
    assert all(not k._can_compile for k in TEMPORAL_KERNELS.values())


def test_bad_kernel_declaration_cannot_overrule_observed_leak(service, monkeypatch):
    parsed = parse_definition_v2(definition()); plan = service.prepare(definition())
    original = service._execute_numeric_node
    def contaminated(node, outputs, *args, **kwargs):
        result = original(node, outputs, *args, **kwargs)
        if node.id == 'returns':
            result['value'].values[:] = outputs['market']['value'].values[-1]
        return result
    monkeypatch.setattr(service, '_execute_numeric_node', contaminated)
    execution = service._execute_graph(None, parsed, 'realtime', None, plan=plan)
    report = audit_execution(service, parsed, 'realtime', None, plan, execution)
    assert report['numerical_verdict'] == 'leak'
    assert not report['verified']
    assert any(f['node_id'] == 'returns' for f in report['findings'])


def test_timestamp_comparison_does_not_right_align_weekly_or_delayed_data():
    dates = np.array([1, 8, 15], dtype=np.int64); available = np.array([2, 9, 16], dtype=np.int64)
    base = np.array([[1.], [2.], [3.]]); altered = base.copy(); altered[1:] *= 10
    assert compare_available_prefix(base, dates, available, altered, dates, available, 8, 0)[0] == 0
    assert compare_available_prefix(base, dates, available, altered, dates, available, 9, 0)[0] == 1
    future = perturb_available_tail(base, available, 8, 1)
    assert future[0, 0] == 1 and future[1, 0] == 12
    assert not np.shares_memory(future, base)


def test_insufficient_samples_are_unknown_not_passed(service):
    parsed = parse_definition_v2(definition(5)); plan = service.prepare(definition(5))
    execution = service._execute_graph(None, parsed, 'realtime', None, plan=plan)
    report = audit_execution(service, parsed, 'realtime', None, plan, execution)
    assert report['status'] == 'audit_unknown' and not report['verified']


def test_source_vintage_obligation_is_not_lost():
    parsed = parse_definition_v2(definition())
    report = analyze_temporal(parsed, NODE_REGISTRY, snapshots={'market': {'revision_policy': 'latest_vintage'}})
    assert report['status'] == 'retrospective_required'
    assert report['reasons'][0]['path'][-1] == 'classifier.state'
