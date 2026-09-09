"""Independent list-based reference, invariants and public graph execution."""
import copy
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from historical_regimes.peak_trough_numba import PEAK_TROUGH_KERNELS, peak_trough_kernel, peak_trough_sideways_kernel, peak_trough_asymmetric_kernel
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_numba import regime_graph_numba_status
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.v2_service import RegimeGraphV2Service, _kernel_ids_for_definition
from historical_regimes.v2_templates import get_template_v2, instantiate_template_v2


def price_path():
    return np.interp(np.arange(97), [0, 16, 32, 48, 64, 80, 96], [100, 160, 100, 180, 80, 160, 100])


def reference(prices, window, min_phase, min_cycle, endpoint_window, exception,
              right_window=None, tail_window=None):
    """Slow list/slice specification, with no compiled helper calls."""
    states = np.full(len(prices), -1, dtype=np.int64)
    rw = window if right_window is None else right_window
    tw = endpoint_window if tail_window is None else tail_window
    pivots = np.zeros(len(prices))
    finite = np.isfinite(prices) & (prices > 0)
    pivots[~finite] = np.nan
    blocks = np.split(np.flatnonzero(finite), np.flatnonzero(np.diff(np.flatnonzero(finite)) > 1) + 1)
    def alternate(turns):
        kept = []
        for index, sign in turns:
            if kept and sign == kept[-1][1]:
                if sign * (prices[index] - prices[kept[-1][0]]) > 0:
                    kept[-1] = (index, sign)
            elif not kept or sign * (prices[index] - prices[kept[-1][0]]) > 0:
                kept.append((index, sign))
        return kept
    for block in blocks:
        if not len(block):
            continue
        lo, hi = block[0], block[-1] + 1
        turns = []
        for t in block:
            if t - lo < max(window, endpoint_window) or hi - 1 - t < max(rw, tw):
                continue
            left, right = prices[t - window:t], prices[t + 1:t + rw + 1]
            for sign in (1, -1):
                if np.all(sign * prices[t] > sign * left) and np.all(sign * prices[t] >= sign * right):
                    turns.append((t, sign))
        turns = alternate(turns)
        while turns:
            t, sign = turns[0]
            if np.any(sign * prices[lo:t] > sign * prices[t]):
                turns.pop(0)
                continue
            t, sign = turns[-1]
            if np.any(sign * prices[t + 1:hi] > sign * prices[t]):
                turns.pop()
                continue
            short_cycles = [i for i in range(len(turns) - 2) if turns[i + 2][0] - turns[i][0] < min_cycle]
            if short_cycles:
                i = short_cycles[0]
                left, sign = turns[i]
                right = turns[i + 2][0]
                at = i if sign * (prices[right] - prices[left]) > 0 else i + 1
                turns = alternate(turns[:at] + turns[at + 2:])
                continue
            short_phases = [i for i in range(len(turns) - 1)
                            if turns[i + 1][0] - turns[i][0] < min_phase
                            and abs(prices[turns[i + 1][0]] / prices[turns[i][0]] - 1) <= exception]
            if not short_phases:
                break
            i = short_phases[0]
            at = i
            if i + 1 == len(turns) - 1:
                at = i + 1
            elif i > 0:
                before = abs(np.log(prices[turns[i][0]] / prices[turns[i - 1][0]]))
                after = abs(np.log(prices[turns[i + 2][0]] / prices[turns[i + 1][0]]))
                at = i + 1 if before >= after else i
            turns = alternate(turns[:at] + turns[at + 1:])
        for index, sign in turns:
            pivots[index] = sign
        for (start, sign), (end, _) in zip(turns, turns[1:]):
            states[start:end] = 0 if sign == -1 else 1
    return states, pivots


@pytest.mark.parametrize("window,phase,cycle,edge,exception", [(1, 4, 8, 0, .2), (3, 10, 16, 2, .1), (8, 4, 16, 6, .2)])
def test_njit_matches_list_reference_and_retained_duration_rules(window, phase, cycle, edge, exception):
    for seed in range(5):
        prices = np.exp(np.random.default_rng(seed).normal(0, .04, 300).cumsum()) * 100
        prices[120:123] = np.nan
        prices[250] = 0
        output = peak_trough_kernel(prices, window, phase, cycle, edge, exception)
        expected = reference(prices, window, phase, cycle, edge, exception)
        for actual, wanted in zip(output, expected):
            np.testing.assert_array_equal(actual, wanted)
        classified = np.flatnonzero(output[0] >= 0)
        for t in classified:
            left, right = output[2][t], output[3][t]
            assert right - left >= phase or abs(output[4][t]) > exception
            assert np.isfinite(prices[left:right + 1]).all()
            assert output[4][t] > 0 if output[0][t] == 0 else output[4][t] < 0
            assert output[5][t] == pytest.approx(prices[left] + (prices[right] - prices[left]) * (t - left) / (right - left))
        indices = np.flatnonzero(np.isfinite(output[1]) & (output[1] != 0))
        for a, b, c in zip(indices, indices[1:], indices[2:]):
            if np.isfinite(prices[a:c + 1]).all() and (prices[a:c + 1] > 0).all():
                assert output[1][a] == output[1][c] == -output[1][b]
                assert c - a >= cycle


def test_monthly_ps_defaults_and_half_open_endpoints():
    states, pivot, starts, ends, moves, line = peak_trough_kernel(price_path(), 8, 4, 16, 6, .2)
    assert np.flatnonzero(pivot).tolist() == [16, 32, 48, 64, 80]
    assert pivot[[16, 32, 48, 64, 80]].tolist() == [1, -1, 1, -1, 1]
    assert (states[:16] == -1).all() and (states[80:] == -1).all()
    assert (states[16:32] == 1).all() and (states[32:48] == 0).all()
    assert (starts[32:48] == 32).all() and (ends[32:48] == 48).all()
    assert moves[32] == pytest.approx(.8)
    assert line[80] == price_path()[80] and np.isnan(line[81])


def test_granularity_changes_retain_fewer_turns_and_short_crashes_have_phase_exception():
    prices = price_path()
    normal = peak_trough_kernel(prices, 8, 4, 16, 6, .2)
    coarse = peak_trough_kernel(prices, 8, 4, 50, 6, .2)
    assert np.count_nonzero(coarse[1]) < np.count_nonzero(normal[1])
    crash = np.array([100., 120, 60, 110, 50, 90, 80])
    retained = peak_trough_kernel(crash, 1, 4, 2, 0, .2)
    suppressed = peak_trough_kernel(crash, 1, 4, 2, 0, 10.)
    assert retained[0][1] == 1 and retained[4][1] == pytest.approx(-.5)
    assert np.count_nonzero(retained[1]) > np.count_nonzero(suppressed[1])


@pytest.mark.parametrize("prices", [np.array([]), np.array([100.]), np.ones(40) * 100,
                                     np.arange(1., 41.), np.array([np.nan, np.inf, -np.inf, 0., -1.])])
def test_insufficient_flat_monotonic_and_invalid_series_stay_unknown(prices):
    result = peak_trough_kernel(prices, 3, 4, 8, 0, .2)
    assert (result[0] == -1).all()
    assert not np.isinf(result[4]).any()


def test_plateau_ties_are_deterministic_and_missing_never_bridges():
    prices = np.array([100., 120, 120, 100, 80, 80, 100, 120, 120, 100])
    first = peak_trough_kernel(prices, 1, 1, 2, 0, .2)
    assert np.flatnonzero(first[1]).tolist() == [1, 4, 7]
    np.testing.assert_array_equal(first[0], peak_trough_kernel(prices, 1, 1, 2, 0, .2)[0])
    prices[5] = np.nan
    split = peak_trough_kernel(prices, 1, 1, 2, 0, .2)
    assert split[0][5] == -1 and np.isnan(split[5][5])
    assert all(not (left < 5 < right) for left, right in zip(split[2], split[3]) if left >= 0)


def definition():
    raw = instantiate_template_v2('peak-trough-ps-v2')
    dates = pd.date_range('2010-01-31', periods=97, freq='ME')
    raw['graph']['nodes'][0] = {'id': 'market', 'type': 'source.inline', 'parameters': {'frequency': 'monthly', 'rows': [
        {'observation_date': day.date().isoformat(), 'available_at': (day + pd.Timedelta(days=1)).date().isoformat(), 'value': float(price)}
        for day, price in zip(dates, price_path())]}}
    return raw


def test_service_runs_njit_and_freezes_nonexecutable_retrospective_evidence(tmp_path):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    parsed = parse_definition_v2(definition())
    assert inspect_definition_v2(parsed)['valid']
    assert set(PEAK_TROUGH_KERNELS) <= set(_kernel_ids_for_definition(parsed))
    before = {key: list(kernel.signatures) for key, kernel in PEAK_TROUGH_KERNELS.items()}
    result = service._execute_graph(None, parsed, 'retrospective', None)
    assert {row['state_id'] for row in result['series']} == {'bull', 'bear', 'unclassified'}
    for row in result['series']:
        assert not row['executable'] and row['effective_index'] == -1 and row['effective_date'] is None
        assert row['is_final'] is False
        if row['state_code'] >= 0:
            assert row['recognition_index'] == 96
            assert row['recognized_at'] == result['series'][-1]['available_at']
    assert result['series'][16]['features']['pivot'] == 1
    manifest = service._persist_series(result['series'])
    restored = pq.read_table(service.artifact_dir / (manifest['checksum'].split(':')[1] + '.parquet')).to_pylist()
    assert json.loads(restored[32]['features_json']) == result['series'][32]['features']
    assert regime_graph_numba_status()['complete']
    assert before == {key: list(kernel.signatures) for key, kernel in PEAK_TROUGH_KERNELS.items()}
    assert all(kernel.nopython_signatures and not kernel._can_compile for kernel in PEAK_TROUGH_KERNELS.values())
    with pytest.raises(TypeError):
        peak_trough_kernel(price_path().astype(np.float32), 8, 4, 16, 6, .2)


def test_realtime_rejected_before_source_reads_and_catalog_marks_correct_mode(tmp_path, monkeypatch):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    monkeypatch.setattr(service, '_resolve_sources', lambda *args, **kwargs: pytest.fail('must reject before I/O'))
    with pytest.raises(Exception, match='实时'):
        service._execute_graph(None, parse_definition_v2(definition()), 'realtime', None)
    node = NODE_REGISTRY['model.peak_trough']
    assert node['supports_realtime'] is False and node['causal'] is False and node['repaints'] is True
    template = get_template_v2('peak-trough-ps-v2')
    assert template['supported_modes'] == ['retrospective']
    assert template['default_mode'] == 'retrospective'
    assert get_template_v2('bull-bear-causal-v2')['default_mode'] == 'realtime'


def test_dating_knowledge_uses_latest_availability_even_on_an_earlier_observation(tmp_path):
    raw = definition()
    raw['graph']['nodes'][0]['parameters']['rows'][32]['available_at'] = '2020-01-01'
    result = RegimeGraphV2Service(tmp_path, tmp_path)._execute_graph(
        None, parse_definition_v2(raw), 'retrospective', None)
    classified = [row for row in result['series'] if row['state_code'] >= 0]
    assert classified
    assert all(row['recognized_at'] == '2020-01-01' and not row['executable'] for row in classified)


@pytest.mark.parametrize('parameter,value', [('window', 0), ('window', 1.5), ('min_phase', 0),
                                           ('min_cycle', 1), ('endpoint_window', -1), ('amplitude_exception', -0.1)])
def test_invalid_parameters_fail_before_computation(parameter, value):
    raw = definition()
    raw['graph']['nodes'][-1]['parameters'][parameter] = value
    assert not inspect_definition_v2(parse_definition_v2(raw))['valid']


def test_future_data_can_revise_dating_and_incorrect_state_semantics_rejected():
    full = peak_trough_kernel(price_path(), 8, 4, 16, 6, .2)[0]
    short = peak_trough_kernel(price_path()[:50].copy(), 8, 4, 16, 6, .2)[0]
    assert np.any(short != full[:50])  # Expected: this is deliberately retrospective.
    raw = definition()
    raw['states'].reverse()
    assert not inspect_definition_v2(parse_definition_v2(raw))['valid']


def sideways(prices, small=.03, width=.06, efficiency=.25, duration=20, enabled=1):
    base = peak_trough_kernel(prices, 1, 1, 2, 0, .2)
    return base, peak_trough_sideways_kernel(prices, base[0], base[2], base[3],
                                           enabled, 2, 1, small, width, efficiency, duration)


def sideways_reference(prices, base, small, width, efficiency, duration):
    """Brute-force complete candidate slices, independent of streaming kernel."""
    states, _, starts, ends, *_ = base
    result = np.where(states == 1, 2, states)
    widths, ers = np.full(len(prices), np.nan), np.full(len(prices), np.nan)
    lefts, rights = np.full(len(prices), -1), np.full(len(prices), -1)
    counts = np.zeros(len(prices), dtype=np.int64)
    phases = [(i, int(ends[i])) for i in range(len(prices)) if states[i] >= 0 and starts[i] == i]
    index = 0
    while index < len(phases):
        first = phases[index][0]
        if first and result[first - 1] == 1:
            index += 1
            continue
        candidates = []
        for last in range(index, len(phases)):
            selected = phases[index:last + 1]
            if any(a[1] != b[0] for a, b in zip(selected, selected[1:])):
                break
            if any(abs(prices[b] / prices[a] - 1) > small for a, b in selected):
                break
            end = selected[-1][1]
            sample = prices[first:end + 1]
            span = sample.max() / sample.min() - 1
            if span > width:
                break
            total = np.abs(np.diff(sample)).sum()
            er = abs(sample[-1] - sample[0]) / total if total else 0.
            turns = [selected[0][0]] + [b for _, b in selected]
            peaks = [prices[t] for t in turns if base[1][t] == 1]
            troughs = [prices[t] for t in turns if base[1][t] == -1]
            directional = len(peaks) >= 2 and len(troughs) >= 2 and (
                (np.all(np.diff(peaks) > 0) and np.all(np.diff(troughs) > 0)) or
                (np.all(np.diff(peaks) < 0) and np.all(np.diff(troughs) < 0)))
            if len(selected) >= 2 and end - first >= duration and er <= efficiency and not directional:
                candidates.append((last, end, span, er, len(selected)))
        if candidates:
            last, end, span, er, count = candidates[-1]
            result[first:end], widths[first:end], ers[first:end] = 1, span, er
            lefts[first:end], rights[first:end], counts[first:end] = first, end, count
            index = last + 1
        else:
            index += 1
    return result, widths, ers, lefts, rights, counts


@pytest.mark.parametrize('seed', range(5))
def test_sideways_matches_independent_slices_and_full_interval_constraints(seed):
    prices = np.ascontiguousarray(100 * np.exp(np.random.default_rng(seed).normal(0, .004, 400).cumsum()))
    prices[140] = np.nan
    for params in ((.03, .06, .25, 20), (.02, .03, .4, 6), (.01, .02, .1, 10)):
        base, actual = sideways(prices, *params)
        expected = sideways_reference(prices, base, *params)
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)
        assert (actual[0][base[0] < 0] == -1).all()
        for start in np.flatnonzero((actual[0] == 1) & (np.r_[True, actual[0][:-1] != 1])):
            stop = start
            while stop < len(prices) and actual[0][stop] == 1:
                stop += 1
            assert stop == actual[4][start] and start == actual[3][start]
            assert prices[start:stop+1].max() / prices[start:stop+1].min() - 1 <= params[1] + 1e-12


def test_sideways_merges_small_reverse_waves_but_preserves_unknown_tail_and_raw_pivots():
    prices = np.interp(np.arange(36), np.arange(0, 36, 5), [101, 102, 100, 102, 100, 102, 100, 101])
    base, result = sideways(prices)
    assert np.flatnonzero(result[0] == 1).tolist() == list(range(5, 30))
    assert result[5][5] == 5 and result[1][5] == pytest.approx(.02)
    assert result[2][5] == pytest.approx(.2)
    assert (result[0][30:] == -1).all()
    assert base[2][15] == 15 and result[3][15] == 5
    for params in ((.01, .06, .25, 20), (.03, .01, .25, 20), (.03, .06, .25, 30)):
        assert not (sideways(prices, *params)[1][0] == 1).any()
    _, strict_er = sideways(prices, efficiency=.1)
    assert strict_er[4][5] == 25  # Retain longest qualifying earlier endpoint.
    _, disabled = sideways(prices, enabled=0)
    np.testing.assert_array_equal(disabled[0], np.where(base[0] == 1, 2, base[0]))
    assert np.isnan(disabled[1]).all()
    scaled = sideways(prices * 1000)[1]
    for a, b in zip(result, scaled):
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_slow_directional_drift_and_large_crash_rebound_are_not_sideways():
    anchors = [100, 102, 101, 103, 102, 104, 103, 105, 104, 106, 105, 107, 106]
    drift = np.interp(np.arange(61), np.arange(0, 61, 5), anchors)
    assert not (sideways(drift, duration=6)[1][0] == 1).any()
    crash = np.interp(np.arange(36), np.arange(0, 36, 5), [100, 120, 80, 120, 80, 120, 80, 100])
    assert not (sideways(crash, small=.6, duration=6)[1][0] == 1).any()


@pytest.mark.parametrize('prices', [np.array([]), np.array([100.]), np.full(50, 100.),
                                   np.array([np.nan, np.inf, -np.inf, 0., -1.])])
def test_sideways_invalid_empty_or_unbounded_data_remain_unknown(prices):
    _, result = sideways(prices)
    assert (result[0] == -1).all() and np.isnan(result[1]).all()


def test_daily_template_three_states_persist_evidence_and_legacy_two_states_still_work(tmp_path):
    raw = instantiate_template_v2('peak-trough-daily-legacy-v1')
    assert raw['graph']['nodes'][-1]['parameters']['sideways_min_duration'] == 20
    assert get_template_v2('peak-trough-ps-v2')['version'] == 2
    prices = np.interp(np.arange(36), np.arange(0, 36, 5), [101, 102, 100, 102, 100, 102, 100, 101])
    raw['graph']['nodes'][0] = {'id': 'market', 'type': 'source.inline', 'parameters': {'frequency': 'daily', 'rows': [
        {'observation_date': date.date().isoformat(), 'value': float(price)}
        for date, price in zip(pd.bdate_range('2020-01-01', periods=len(prices)), prices)]}}
    params = raw['graph']['nodes'][-1]['parameters']
    params.update(left_window=1, right_window=1, head_window=0, tail_window=0, min_phase=1, min_cycle=2)
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    parsed = parse_definition_v2(raw)
    assert inspect_definition_v2(parsed)['valid']
    signatures = {key: list(kernel.signatures) for key, kernel in PEAK_TROUGH_KERNELS.items()}
    result = service._execute_graph(None, parsed, 'retrospective', None)
    row = result['series'][5]
    assert row['state_id'] == 'sideways' and row['state_code'] == 1
    assert row['probabilities'] == {'bull': 0., 'sideways': 1., 'bear': 0.}
    assert row['features']['sideways_swing_count'] == 5 and '合并为震荡' in row['reasons'][0]
    assert all(not r['executable'] for r in result['series'])
    manifest = service._persist_series(result['series'])
    restored = pq.read_table(service.artifact_dir / (manifest['checksum'].split(':')[1] + '.parquet')).to_pylist()
    assert json.loads(restored[5]['features_json']) == row['features']
    legacy = copy.deepcopy(raw)
    legacy['states'] = [legacy['states'][0], legacy['states'][2]]
    assert not inspect_definition_v2(parse_definition_v2(legacy))['valid']
    for key in ('sideways_enabled', 'small_swing_threshold', 'sideways_max_range', 'sideways_max_efficiency', 'sideways_min_duration'):
        legacy['graph']['nodes'][-1]['parameters'].pop(key)
    legacy_params = legacy['graph']['nodes'][-1]['parameters']
    for key in ('left_window', 'right_window', 'head_window', 'tail_window'):
        legacy_params.pop(key)
    legacy_params.update(window=1, endpoint_window=0)
    legacy_result = service._execute_graph(None, parse_definition_v2(legacy), 'retrospective', None)
    np.testing.assert_array_equal([r['state_code'] for r in legacy_result['series']], peak_trough_kernel(prices, 1, 1, 2, 0, .2)[0])
    legacy_params['right_window'] = 6
    override = service._execute_graph(None, parse_definition_v2(legacy), 'retrospective', None)
    np.testing.assert_array_equal([r['state_code'] for r in override['series']], peak_trough_asymmetric_kernel(prices, 1, 6, 1, 2, 0, 0, .2)[0])
    assert signatures == {key: list(kernel.signatures) for key, kernel in PEAK_TROUGH_KERNELS.items()}
    assert all(kernel.nopython_signatures and not kernel._can_compile for kernel in PEAK_TROUGH_KERNELS.values())
    base = peak_trough_kernel(prices, 1, 1, 2, 0, .2)
    with pytest.raises(TypeError):
        peak_trough_sideways_kernel(prices.astype(np.float32), base[0], base[2], base[3], 1, 2, 1, .03, .06, .25, 20)


@pytest.mark.parametrize('parameter,value', [('sideways_enabled', 1), ('small_swing_threshold', -1),
    ('sideways_max_range', float('nan')), ('sideways_max_efficiency', 1), ('sideways_min_duration', 1.5)])
def test_invalid_sideways_parameters_fail_closed(parameter, value):
    raw = instantiate_template_v2('peak-trough-daily-legacy-v1')
    raw['graph']['nodes'][-1]['parameters'][parameter] = value
    assert not inspect_definition_v2(parse_definition_v2(raw))['valid']


@pytest.mark.parametrize('windows', [(8, 2, 6, 1), (2, 8, 1, 6), (4, 1, 0, 10), (1, 3, 10, 0)])
def test_four_windows_match_reference_and_symmetric_compatibility(windows):
    left, right, head, tail = windows
    prices = np.ascontiguousarray(100 * np.exp(np.random.default_rng(24).normal(0, .01, 400).cumsum()))
    prices[200] = np.nan
    result = peak_trough_asymmetric_kernel(prices, left, right, 3, 6, head, tail, .2)
    expected = reference(prices, left, 3, 6, head, .2, right, tail)
    np.testing.assert_array_equal(result[0], expected[0])
    np.testing.assert_allclose(result[1], expected[1], equal_nan=True)
    symmetric = peak_trough_asymmetric_kernel(prices, left, left, 3, 6, head, head, .2)
    for a, b in zip(symmetric, peak_trough_kernel(prices, left, 3, 6, head, .2)):
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_shorter_right_and_tail_can_retain_a_more_recent_turn_without_extrapolation():
    prices = np.interp(np.arange(31), [0, 6, 12, 18, 27, 30], [100, 130, 90, 140, 80, 100])
    original = peak_trough_asymmetric_kernel(prices, 4, 4, 1, 2, 0, 6, .2)
    recent = peak_trough_asymmetric_kernel(prices, 4, 2, 1, 2, 0, 1, .2)
    assert np.flatnonzero(original[1])[-1] == 18
    assert np.flatnonzero(recent[1])[-1] == 27
    assert recent[0][26] == 1 and (recent[0][27:] == -1).all()


@pytest.mark.parametrize('parameter,value', [('left_window', 0), ('right_window', 0),
    ('head_window', -1), ('tail_window', 1.5)])
def test_invalid_split_windows_rejected(parameter, value):
    raw = instantiate_template_v2('peak-trough-daily-legacy-v1')
    raw['graph']['nodes'][-1]['parameters'][parameter] = value
    assert not inspect_definition_v2(parse_definition_v2(raw))['valid']
