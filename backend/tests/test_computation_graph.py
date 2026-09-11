"""Shared graph topology and strict historical-regime compatibility."""
import random
import sys
from collections import deque
from pathlib import Path
import pytest
from pydantic import ValidationError
ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / 'backend'):
    if str(path) not in sys.path: sys.path.insert(0, str(path))
from backend.computation_graph import CanvasLayout, GraphPortRef, dependency_order
from historical_regimes.v2_contracts import GraphPortRefV2, RegimeDefinitionV2, _topological_order


def reference(ids, edges):
    degree = dict.fromkeys(ids, 0); consumers = {key: [] for key in ids}
    for source, target in edges:
        degree[target] += 1; consumers[source].append(target)
    queue = deque(key for key in ids if not degree[key]); result = []
    while queue:
        key = queue.popleft(); result.append(key)
        for target in consumers[key]:
            degree[target] -= 1
            if not degree[target]: queue.append(target)
    return result


def test_fifo_preserves_original_regime_order():
    rng = random.Random(42)
    for count in range(1, 40):
        ids = [f'n{i}' for i in range(count)]
        edges = [(ids[i], ids[j]) for i in range(count) for j in range(i + 1, count) if rng.random() < .15]
        if edges: edges += edges[:2]
        assert dependency_order(ids, edges).order == reference(ids, edges)


def test_etl_uses_original_position_for_ready_node_ties():
    ids = ['source', 'map', 'other']; edges = [('source', 'map')]
    assert dependency_order(ids, edges).order == ['source', 'other', 'map']
    assert dependency_order(ids, edges, tie_break='node_order').order == ids


def test_missing_duplicate_cycle_and_empty():
    result = dependency_order(['a', 'b', 'b'], [('missing', 'b'), ('a', 'b'), ('b', 'a')])
    assert result.missing == [('missing', 'b')]
    assert result.duplicates == ['b'] and result.cyclic and not result.order
    assert dependency_order([], []).order == []


@pytest.mark.parametrize('x', [float('nan'), float('inf'), 1e20])
def test_layout_rejects_nonfinite_and_unbounded_positions(x):
    with pytest.raises(ValidationError): CanvasLayout(positions={'node': {'x': x, 'y': 0}})


def test_regime_ref_retains_narrow_id_contract():
    assert GraphPortRef(node_id='etl.fund').node_id == 'etl.fund'
    with pytest.raises(ValidationError): GraphPortRefV2(node_id='etl.fund')
    assert GraphPortRefV2(node_id='node_1').model_dump() == {'node_id': 'node_1', 'port': 'value'}


def test_regime_diagnostics_are_unchanged():
    definition = RegimeDefinitionV2.model_validate({'name': 'regression', 'graph': {'nodes': [
        {'id': 'a', 'type': 'source.constant', 'inputs': {'missing': {'node_id': 'ghost'}, 'value': {'node_id': 'b'}}},
        {'id': 'b', 'type': 'source.constant', 'inputs': {'value': {'node_id': 'a'}}},
    ], 'outputs': {'state': {'node_id': 'b'}}}, 'states': [{'id': 's1', 'label': 'one', 'order': 0}, {'id': 's2', 'label': 'two', 'order': 1}]})
    order, diagnostics = _topological_order(definition)
    assert order == []
    assert diagnostics == [
        {'code': 'UNKNOWN_INPUT_NODE', 'path': 'graph.nodes.a.inputs.missing', 'message': '输入引用了不存在的节点 ghost。', 'severity': 'error'},
        {'code': 'GRAPH_CYCLE', 'path': 'graph.nodes', 'message': '图谱存在循环依赖。', 'severity': 'error'},
    ]
