"""Business-semantic regressions, with independent bucket and exhaustive integer oracles."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

import optimizer
from backend.frontier_sampling import integer_weights_kernel, return_bucket_indices_kernel


def _integer(raw, step=.005, bounds=None, groups=None, lows=None, highs=None, budget=50000):
    n = len(raw)
    return integer_weights_kernel(np.array(raw, dtype=float),
        np.array(bounds if bounds is not None else [[0., 1.]] * n, dtype=float).reshape(n, 2),
        np.array(groups, dtype=np.uint8) if groups is not None else np.empty((0, n), dtype=np.uint8),
        np.array(lows or [], dtype=float), np.array(highs or [], dtype=float), float(step), budget)


@pytest.mark.parametrize('step', [.001, .002, .005])
def test_precision_survives_sum_and_overlapping_group_constraints(step):
    for raw in ([.3334, .3334, .3332], [.02, .92, .06], [.4, .2, .4]):
        weights, status = _integer(raw, step, [[.05, .8]] * 3, [[1, 1, 0], [0, 1, 1]], [.3, .4], [.65, .85])
        assert status == 0
        np.testing.assert_allclose(weights / step, np.round(weights / step), atol=1e-9)
        assert weights.sum() == pytest.approx(1, abs=1e-12)
        assert np.all((weights >= .05) & (weights <= .8))
        assert .3 - 1e-12 <= weights[0] + weights[1] <= .65 + 1e-12
        assert .4 - 1e-12 <= weights[1] + weights[2] <= .85 + 1e-12
        again, status = _integer(weights, step, [[.05, .8]] * 3, [[1, 1, 0], [0, 1, 1]], [.3, .4], [.65, .85])
        np.testing.assert_array_equal(weights, again)


def test_integer_feasibility_matches_exhaustive_oracle():
    rng = np.random.default_rng(481)
    grid = np.array([point for point in product(range(6), repeat=4) if sum(point) == 5]) / 5
    groups = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 1]], dtype=np.uint8)
    group_sums = grid @ groups.T
    for _ in range(80):
        bounds = np.sort(rng.integers(0, 6, size=(4, 2)), axis=1) / 5
        limits = np.sort(rng.integers(0, 6, size=(3, 2)), axis=1) / 5
        feasible = np.all((grid >= bounds[:, 0] - 1e-12) & (grid <= bounds[:, 1] + 1e-12), axis=1)
        feasible &= np.all((group_sums >= limits[:, 0] - 1e-12) & (group_sums <= limits[:, 1] + 1e-12), axis=1)
        weights, status = integer_weights_kernel(rng.dirichlet(np.ones(4)), bounds, groups,
            np.ascontiguousarray(limits[:, 0]), np.ascontiguousarray(limits[:, 1]), .2, 50000)
        assert (status == 0) == bool(feasible.any())
        if feasible.any():
            assert np.any(np.all(np.isclose(grid[feasible], weights, atol=1e-12), axis=1))
        else:
            assert status == 1


def test_budget_is_not_misreported_as_infeasibility_and_invalid_inputs_fail():
    _, status = _integer([.9, .05, .05], .1, groups=[[1, 0, 0]], lows=[.1], highs=[.2], budget=0)
    assert status == 2
    assert _integer([.5, .5], .1, [[.45, .49], [0, 1]])[1] == 1
    for step in [float('nan'), float('inf'), 0., -.1, .03]:
        assert _integer([.5, .5], step)[1] == 3
    assert _integer([np.nan, .5])[1] == 3
    assert _integer([])[1] == 3


def test_integer_input_is_not_mutated_and_fine_grid_cost_is_bounded():
    raw = np.array([1., 0., 0.]); original = raw.copy()
    bounds = np.array([[0., .01], [0., 1.], [0., 1.]])
    result, status = integer_weights_kernel(raw, bounds, np.empty((0, 3), dtype=np.uint8), np.empty(0), np.empty(0), 1e-9, 50000)
    assert status == 0
    np.testing.assert_array_equal(raw, original)
    assert result[0] <= .01
    assert not np.shares_memory(raw, result)
    assert integer_weights_kernel._can_compile is False


def test_bucket_oracle_includes_max_endpoint_and_first_ties():
    returns = np.array([0., .1, .11, .8, 1., 1.])
    risks = np.array([.9, .3, .3, .2, .1, .1])
    selected, count = return_bucket_indices_kernel(risks, returns, 0, 6, 2)
    assert list(selected[:count]) == [1, 4]
    selected, count = return_bucket_indices_kernel(risks, np.ones(6), 0, 6, 3)
    assert list(selected[:count]) == [4, 5, 3]
    assert return_bucket_indices_kernel(risks, returns, 2, 2, 3)[1] == 0
    with pytest.raises(ValueError, match='NONFINITE'):
        return_bucket_indices_kernel(np.array([np.nan]), np.array([1.]), 0, 1, 2)
    with pytest.raises(ValueError, match='AXIS'):
        return_bucket_indices_kernel(risks, returns, -1, 2, 3)


def _explore(risk='annual_vol', **kwargs):
    rng = np.random.default_rng(137)
    frame = pd.DataFrame(rng.normal([.0002, .0005, .0009], [.006, .012, .02], (252, 3)), columns=['A', 'B', 'C'])
    return optimizer.calculate_efficient_frontier_exploration(frame,
        {'metric': 'annual_mean'}, {'metric': risk}, single_limits=[(.05, .8)] * 3,
        group_limits={(0, 1): (.3, .85)},
        rounds=[{'samples': 100, 'step': 1., 'buckets': 10},
                {'samples': 180, 'step': .25, 'buckets': 12},
                {'samples': 220, 'step': .1, 'buckets': 20}], **kwargs)


def test_frozen_parent_sets_and_risk_driven_selection_are_restored():
    result = _explore()
    audit = result['exploration']
    selected = set(audit['selected_indices'])
    previous_seeds = set(range(100))
    assert set(audit['parent_indices'][:100]) == {-1}
    for row in audit['rounds'][1:]:
        start, end = row['start_index'], row['end_index']
        assert set(audit['parent_indices'][start:end]) <= previous_seeds
        # Independent equal-width grouping, without invoking production selector.
        points = np.array([p['value'] for p in result['scatter'][start:end]])
        count = 12 if start == 100 else 20
        edges = np.linspace(points[:, 1].min(), points[:, 1].max(), count + 1)
        bins = np.minimum(np.searchsorted(edges, points[:, 1], side='right') - 1, count - 1)
        oracle = set()
        for bucket in range(count):
            members = np.flatnonzero(bins == bucket)
            if len(members):
                oracle.add(start + members[np.argmin(points[members, 0])])
        assert selected & set(range(start, end)) == oracle
        previous_seeds = oracle
    alternative = _explore('max_drawdown')
    first = np.array([p['weights'] for p in result['scatter']])
    other = np.array([p['weights'] for p in alternative['scatter']])
    np.testing.assert_array_equal(first[:280], other[:280])
    assert not np.allclose(first[280:], other[280:])
    assert result == _explore()
    assert not np.array_equal(first, np.array([p['weights'] for p in _explore(seed=8)['scatter']]))


@pytest.mark.parametrize('step', [.001, .005])
def test_quantized_grid_and_refinement_share_adoptable_domain(step):
    result = _explore(quantize_step=step, use_local_refine=True, refine_iterations=30,
        frontier_grid={'point_count': 20, 'max_iterations': 300, 'accept_continuous_weights': True})
    assert result['weight_domain'] == 'discrete'
    for point in result['scatter']:
        weights = np.array(point['weights'])
        np.testing.assert_allclose(weights / step, np.round(weights / step), atol=1e-9)
        assert weights.sum() == pytest.approx(1, abs=1e-12)
        assert .3 - 1e-12 <= weights[0] + weights[1] <= .85 + 1e-12
    values = np.array([p['value'] for p in result['scatter']])
    assert result['max_return']['value'][1] == pytest.approx(values[:, 1].max())
    assert result['min_variance']['value'][0] == pytest.approx(values[:, 0].min())
    assert result['max_sharpe']['value'][1] / result['max_sharpe']['value'][0] == pytest.approx(np.max(values[:, 1] / values[:, 0]))
    for point in result['frontier']:
        r, ret = point['value']
        assert not np.any((values[:, 0] <= r) & (values[:, 1] >= ret) & ((values[:, 0] < r - 1e-10) | (values[:, 1] > ret + 1e-10)))
    success = [p for p in result['frontier_grid']['points'] if p['status'] == 'converged']
    assert len(success) == 20
    assert any(not np.allclose(np.array(p['weights']) / step, np.round(np.array(p['weights']) / step), atol=1e-9) for p in success)
    for point in success:
        adopted = point['adoption']
        assert adopted['status'] == 'feasible'
        assert adopted['value'] == list(result['scatter'][point['candidate_index']]['value'])
        assert adopted['weights'] == result['scatter'][point['candidate_index']]['weights']
        assert adopted['target_met'] == (adopted['value'][1] >= point['target'] - 1e-7)


@pytest.mark.parametrize('rounds', [[], [{'samples': 2.5}], [{'samples': True}], [{'buckets': 0}], [{'step': np.nan}]])
def test_exploration_rejects_silent_parameter_coercion(rounds):
    with pytest.raises(ValueError):
        optimizer._round_arrays(rounds)


def test_integer_reads_strided_readonly_views_without_new_signatures():
    owner = np.array([.3334, -99., .3334, -99., .3332, -99.])
    raw = owner[::2]; raw.setflags(write=False)
    bounds_owner = np.zeros((6, 4)); bounds_owner[::2, 2] = 1
    bounds = bounds_owner[::2, ::2]; bounds.setflags(write=False)
    groups = np.array([[1, 1, 0]], dtype=np.uint8); groups.setflags(write=False)
    lows, highs = np.array([.3]), np.array([.8])
    lows.setflags(write=False); highs.setflags(write=False)
    before = list(integer_weights_kernel.nopython_signatures)
    weights, status = integer_weights_kernel(raw, bounds, groups, lows, highs, .005, 50000)
    assert status == 0
    assert np.shares_memory(raw, owner) and np.shares_memory(bounds, bounds_owner)
    np.testing.assert_array_equal(raw, [.3334, .3334, .3332])
    assert before == list(integer_weights_kernel.nopython_signatures)
    assert weights.sum() == pytest.approx(1.)


@pytest.fixture
def frontier_client(monkeypatch, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.services import analytics_routes as routes
    from pit.context import build_context
    rng = np.random.default_rng(819)
    nav = np.cumprod(1 + rng.normal(.0004, .01, (70, 3)), axis=0)
    pd.DataFrame([{'asset_alloc_name': 'sampling-api', 'asset_name': name, 'date': day, 'nv': nav[i, j]}
        for i, day in enumerate(pd.bdate_range('2024-01-01', periods=70))
        for j, name in enumerate(['A', 'B', 'C'])]).to_parquet(tmp_path / 'asset_nv.parquet', index=False)
    monkeypatch.setattr(routes, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(routes, 'resolve_request_context', lambda *_: build_context('2024-12-31'))
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        yield client


def _request(**changes):
    return dict(alloc_name='sampling-api', start_date='2024-01-01', end_date='2024-12-31',
                return_metric={'metric': 'annual'}, risk_metric={'metric': 'annual_vol'}, **changes)


@pytest.mark.parametrize('changes', [
    {'constraints': {'single_limits': {'unknown': {'lo': .1}}}},
    {'constraints': {'single_limits': {'A': {'lo': -.1}}}},
    {'constraints': {'group_limits': [{'assets': ['unknown'], 'lo': .1}]}},
    {'constraints': {'group_limits': [{'assets': ['A', 'A']}]}},
    {'constraints': {'group_limits': [{'assets': ['A', 'B']}, {'assets': ['B', 'A']}]}},
    {'constraints': {'group_limits': [{'assets': []}]}},
    {'exploration': {'seed': True}}, {'exploration': {'seed': 1.2}},
    {'exploration': {'rounds': [{'samples': 1.2}]}},
    {'quantization': {'step': .005}, 'constraints': {'single_limits': {'A': {'lo': .331, 'hi': .332}}}},
])
def test_api_refuses_ignored_or_coerced_constraints(frontier_client, changes):
    response = frontier_client.post('/api/efficient-frontier', json=_request(**changes))
    assert response.status_code == 400, response.text
    assert response.json()['detail']


def test_api_seed_round_audit_and_integer_adoption(frontier_client):
    request = _request(exploration={'seed': 9, 'rounds': [{'samples': 50}, {'samples': 80, 'step': .2, 'buckets': 7}]},
                       quantization={'step': .005}, frontier_grid={'point_count': 20, 'accept_continuous_weights': True})
    response = frontier_client.post('/api/efficient-frontier', json=request)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['exploration']['seed'] == 9
    assert result['exploration']['rounds'][1]['selected'] <= 7
    assert result['weight_domain'] == 'discrete'
    assert result['quantization_step'] == .005
    for point in result['scatter']:
        np.testing.assert_allclose(np.array(point['weights']) / .005, np.round(np.array(point['weights']) / .005), atol=1e-9)


@pytest.mark.parametrize('window', [0, 20, 60, 1000])
def test_return_window_changes_the_actual_axis_consistently(window):
    rng = np.random.default_rng(251)
    values = rng.normal([.0002, .0008], [.007, .012], (90, 2))
    # Earlier returns differ materially: an ignored window cannot pass this check.
    values[:40] += .004
    config = {'metric': 'ewm', 'alpha': .8, 'window': window}
    result = optimizer.calculate_efficient_frontier_exploration(pd.DataFrame(values), config,
        {'metric': 'annual_vol'}, rounds=[{'samples': 30}], use_local_refine=True, refine_iterations=10,
        quantize_step=.005, frontier_grid={'point_count': 20, 'accept_continuous_weights': True})
    points = result['scatter'] + [p for p in result['frontier_grid']['points'] if p['status'] == 'converged']
    for point in points:
        portfolio = values @ point['weights']
        sample = portfolio[-window:] if window else portfolio
        expected = sample[0]
        for value in sample[1:]:
            expected = .8 * expected + .2 * value
        assert point['value'][1] == pytest.approx(expected, abs=1e-12)
        assert optimizer.calculate_return(portfolio, config) == pytest.approx(expected, abs=1e-12)
    if window == 20:
        full = optimizer.calculate_return(values[:, 0].copy(), {**config, 'window': 0})
        recent = optimizer.calculate_return(values[:, 0].copy(), config)
        assert abs(full - recent) > 1e-8


def test_zero_step_uses_only_frozen_seeds_and_bucket_count_changes_next_round():
    values = pd.DataFrame(np.random.default_rng(85).normal(.0003, .01, (80, 3)))
    def run(buckets):
        return optimizer.calculate_efficient_frontier_exploration(values, {'metric': 'annual'}, {'metric': 'annual_vol'},
            rounds=[{'samples': 60}, {'samples': 60, 'step': .2, 'buckets': buckets}, {'samples': 40, 'step': 0.}])
    first, other = run(4), run(12)
    for index in range(120, 160):
        parent = first['exploration']['parent_indices'][index]
        assert first['scatter'][index]['weights'] == first['scatter'][parent]['weights']
    assert first['scatter'][120:] != other['scatter'][120:]
