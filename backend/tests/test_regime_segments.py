"""Independent numerical references and production graph contracts."""
import copy
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from historical_regimes.segment_numba import SEGMENT_KERNELS, local_extrema_kernel, between_pivots_kernel, interval_statistic_kernel, range_threshold_kernel
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_service import RegimeGraphV2Service, _kernel_ids_for_definition
from historical_regimes.v2_templates import instantiate_template_v2


def definition(prices=None):
    prices = np.array([99., 100, 102, 101, 100, 101, 105, 104, 98, 99, 100, 99]) if prices is None else prices
    raw = instantiate_template_v2('peak-trough-daily-v2')
    raw['graph']['nodes'][0] = {'id': 'market', 'type': 'source.inline', 'parameters': {'frequency': 'daily', 'rows': [
        {'observation_date': day.date().isoformat(), 'available_at': day.date().isoformat(), 'value': float(price)}
        for day, price in zip(pd.date_range('2020-01-01', periods=len(prices)), prices)]}}
    raw['graph']['nodes'][1]['parameters'] = {'left_window': 1, 'right_window': 1, 'head_window': 0, 'tail_window': 0}
    return raw


def reference_pivots(prices, left, right, head, tail):
    result = np.zeros(len(prices))
    valid = np.isfinite(prices) & (prices > 0)
    result[~valid] = np.nan
    ids = np.flatnonzero(valid)
    for block in np.split(ids, np.flatnonzero(np.diff(ids) != 1) + 1):
        if not len(block):
            continue
        chosen = []
        for t in block:
            if t - block[0] < max(left, head) or block[-1] - t < max(right, tail):
                continue
            kind = 1 if np.all(prices[t] > prices[t-left:t]) and np.all(prices[t] >= prices[t+1:t+right+1]) else -1 if np.all(prices[t] < prices[t-left:t]) and np.all(prices[t] <= prices[t+1:t+right+1]) else 0
            if not kind:
                continue
            if chosen and chosen[-1][1] == kind:
                if kind * (prices[t] - prices[chosen[-1][0]]) > 0:
                    chosen[-1] = (t, kind)
            elif not chosen or kind * (prices[t] - prices[chosen[-1][0]]) > 0:
                chosen.append((t, kind))
        for t, kind in chosen:
            result[t] = kind
    return result


@pytest.mark.parametrize('windows', [(1, 1, 0, 0), (3, 1, 5, 0), (1, 4, 0, 6), (8, 8, 6, 6)])
@pytest.mark.parametrize('seed', range(3))
def test_pivots_and_independent_statistics(windows, seed):
    prices = np.exp(np.random.default_rng(seed).normal(0, .02, 250).cumsum()) * 100
    prices[50:53] = np.nan
    prices[110] = np.inf
    prices[150] = 0
    prices[170:173] = prices[170]
    pivots, pivot_prices = local_extrema_kernel(prices, *windows)
    np.testing.assert_array_equal(pivots, reference_pivots(prices, *windows))
    np.testing.assert_array_equal(pivot_prices, np.where(np.abs(pivots) == 1, prices, np.nan))
    starts, ends = between_pivots_kernel(pivots)
    for opcode in range(5):
        actual = interval_statistic_kernel(prices, starts, ends, opcode, 1)
        expected = np.full(len(prices), np.nan)
        for left in np.flatnonzero(starts == np.arange(len(prices))):
            right = ends[left]
            path = prices[left:right+1]
            assert np.all(np.isfinite(path)) and np.all(path > 0)
            returns = np.diff(path) / path[:-1]
            movement = np.abs(np.diff(path)).sum()
            values = [path[-1]/path[0]-1, path.max()/path.min()-1,
                      np.std(returns, ddof=1) if len(returns) > 1 else np.nan,
                      right-left, abs(path[-1]-path[0])/movement if movement else 0.]
            expected[left:right] = values[opcode]
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14)
    np.testing.assert_array_equal(local_extrema_kernel(prices, *windows)[0], pivots)


@pytest.mark.parametrize('prices', [[], [100], [100]*12, [np.nan, np.inf, 0, -1]])
def test_empty_invalid_flat_and_large_window(prices):
    data = np.asarray(prices, dtype=np.float64)
    pivots, _ = local_extrema_kernel(data, 50, 20, 0, 0)
    starts, ends = between_pivots_kernel(pivots)
    assert np.all(starts == -1) and np.all(ends == -1)
    assert np.isnan(interval_statistic_kernel(data, starts, ends, 0, 1)).all()


def test_single_small_wave_threshold_equality_and_unknown():
    values = np.array([.03, -.03, .0301, -.0301, 0, np.nan])
    states, errors = range_threshold_kernel(values, np.full(6, .03), np.full(6, -.03))
    assert errors == 0
    np.testing.assert_array_equal(states, [1, 1, 0, 2, 1, -1])
    assert range_threshold_kernel(values, np.full(6, -.04), np.full(6, -.03))[1] == 6
    assert range_threshold_kernel(values, np.array([.03]), np.array([-.03]))[1] == 1


def test_interval_bad_boundary_path_and_ddof():
    prices = np.array([100., 101., 100.])
    starts, ends = np.array([0, 0, -1]), np.array([2, 2, -1])
    result = interval_statistic_kernel(prices, starts, ends, 2, 0)
    assert result[0] == pytest.approx(np.std([.01, -1/101]))
    ends[1] = 1
    assert np.isnan(interval_statistic_kernel(prices, starts, ends, 0, 1)).all()
    ends[:] = [2, 2, -1]
    prices[1] = np.nan
    assert np.isnan(interval_statistic_kernel(prices, starts, ends, 0, 1)).all()
    assert np.isnan(interval_statistic_kernel(prices, starts[:1], ends, 0, 1)).all()


def test_production_chain_independent_thresholds_persistence_and_compiled_paths(tmp_path):
    raw = definition()
    parsed = parse_definition_v2(raw)
    assert inspect_definition_v2(parsed)['valid']
    required_kernels = set(_kernel_ids_for_definition(parsed))
    assert {'local_extrema', 'between_pivots', 'interval_statistic', 'range_threshold'} <= required_kernels
    assert not {'phase_direction', 'boundary_line'} & required_kernels
    before = {key: list(kernel.signatures) for key, kernel in SEGMENT_KERNELS.items()}
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    result = service._execute_graph(None, parsed, 'retrospective', None)
    assert result['series'][2]['state_id'] == 'sideways'
    assert result['series'][4]['state_id'] == 'bull'
    assert result['series'][6]['state_id'] == 'bear'
    assert result['series'][-1]['state_id'] == 'unclassified'
    assert all(not row['executable'] and row['effective_date'] is None for row in result['series'])
    assert all(row['recognized_at'] == '2020-01-12' for row in result['series'] if row['state_code'] >= 0)
    modified = copy.deepcopy(raw)
    modified['graph']['nodes'][4]['parameters']['value'] = .10
    modified['graph']['nodes'][5]['parameters']['value'] = -.10
    changed = service._execute_graph(None, parse_definition_v2(modified), 'retrospective', None)
    assert [r['features']['pivot'] for r in changed['series']] == [r['features']['pivot'] for r in result['series']]
    assert changed['series'][4]['state_id'] == 'sideways'
    manifest = service._persist_series(result['series'])
    restored = pq.read_table(service.artifact_dir / (manifest['checksum'].split(':')[1] + '.parquet')).to_pylist()
    assert json.loads(restored[2]['features_json']) == result['series'][2]['features']
    assert before == {key: list(kernel.signatures) for key, kernel in SEGMENT_KERNELS.items()}
    assert all(kernel.nopython_signatures and not kernel._can_compile for kernel in SEGMENT_KERNELS.values())
    with pytest.raises(TypeError):
        local_extrema_kernel(np.ones(20, dtype=np.float32), 1, 1, 0, 0)


def test_realtime_rejected_before_io_and_boundaries_cannot_mix(tmp_path, monkeypatch):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    monkeypatch.setattr(service, '_resolve_sources', lambda *args, **kwargs: pytest.fail('unexpected I/O'))
    with pytest.raises(Exception, match='实时'):
        service._execute_graph(None, parse_definition_v2(definition()), 'realtime', None)
    raw = definition()
    duplicate = copy.deepcopy(raw['graph']['nodes'][2]); duplicate['id'] = 'other_segments'
    raw['graph']['nodes'].append(duplicate)
    raw['graph']['nodes'][3]['inputs']['end']['node_id'] = duplicate['id']
    assert any(e['code'] == 'SEGMENT_BOUNDARIES_MISMATCH' for e in inspect_definition_v2(parse_definition_v2(raw))['errors'])


@pytest.mark.parametrize('name,value', [('left_window', 0), ('right_window', 1.5), ('head_window', -1), ('tail_window', True)])
def test_window_contract(name, value):
    raw = definition(); raw['graph']['nodes'][1]['parameters'][name] = value
    assert not inspect_definition_v2(parse_definition_v2(raw))['valid']


def test_optional_statistics_do_not_classify_or_shift_pivots(tmp_path):
    raw = definition()
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    first = service._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    for name in ('amplitude', 'volatility', 'duration', 'efficiency'):
        raw['graph']['nodes'].append({'id': name, 'type': f'segment.{name}',
            'inputs': copy.deepcopy(raw['graph']['nodes'][3]['inputs'])})
        raw['graph']['exposed_node_ids'].append(name)
    after = service._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    assert [r['state_code'] for r in first['series']] == [r['state_code'] for r in after['series']]
    assert after['series'][2]['features']['segment_duration'] == 2
    assert after['series'][2]['features']['segment_amplitude'] == pytest.approx(.02)
    assert after['series'][2]['features']['segment_efficiency'] == pytest.approx(1)


def test_full_input_availability_and_output_wrapper_do_not_erase_retrospective_status(tmp_path):
    raw = definition()
    raw['graph']['nodes'][0]['parameters']['rows'][1]['available_at'] = '2020-02-01'
    raw['graph']['nodes'].append({'id': 'final', 'type': 'output.state', 'inputs': {'state': {'node_id': 'classifier', 'port': 'state'}}})
    raw['graph']['outputs']['state'] = {'node_id': 'final', 'port': 'state'}
    result = RegimeGraphV2Service(tmp_path, tmp_path)._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    assert all(r['recognized_at'] == '2020-02-01' and r['effective_date'] is None for r in result['series'] if r['state_code'] >= 0)
    assert result['series'][2]['features']['phase_return'] == pytest.approx(100/102-1)


def test_unrelated_exposed_price_statistics_cannot_replace_classification_evidence(tmp_path):
    raw = definition()
    raw['graph']['nodes'].append({'id': 'other_price', 'type': 'source.constant',
        'parameters': {'value': 100.}, 'inputs': {'anchor': {'node_id': 'market', 'port': 'value'}}})
    inputs = copy.deepcopy(raw['graph']['nodes'][3]['inputs'])
    inputs['value'] = {'node_id': 'other_price', 'port': 'value'}
    raw['graph']['nodes'].append({'id': 'other_change', 'type': 'segment.change', 'inputs': inputs})
    raw['graph']['exposed_node_ids'].append('other_change')
    result = RegimeGraphV2Service(tmp_path, tmp_path)._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    assert result['series'][2]['features']['phase_return'] == pytest.approx(100/102-1)


def test_delayed_threshold_input_cannot_receive_an_earlier_recognition_date(tmp_path):
    raw = definition()
    rows = copy.deepcopy(raw['graph']['nodes'][0]['parameters']['rows'])
    for row in rows:
        row['value'] = .03
    rows[2]['available_at'] = '2020-02-10'
    raw['graph']['nodes'][4] = {'id': 'upper', 'type': 'source.inline',
        'parameters': {'frequency': 'daily', 'rows': rows}}
    raw['graph']['nodes'].append({'id': 'bound_alignment', 'type': 'align.strict_intersection',
        'inputs': {'left': {'node_id': 'change', 'port': 'value'}, 'right': {'node_id': 'upper', 'port': 'value'}}})
    raw['graph']['nodes'][5]['inputs']['anchor'] = {'node_id': 'bound_alignment', 'port': 'left'}
    raw['graph']['nodes'][6]['inputs']['value'] = {'node_id': 'bound_alignment', 'port': 'left'}
    raw['graph']['nodes'][6]['inputs']['upper_bound'] = {'node_id': 'bound_alignment', 'port': 'right'}
    result = RegimeGraphV2Service(tmp_path, tmp_path)._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    assert result['series'][2]['recognized_at'] == '2020-02-10'
    assert not result['series'][2]['executable']
