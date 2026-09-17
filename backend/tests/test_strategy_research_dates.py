"""The same explicit ending date bounds fitting, schedules, cache and NAV."""
from datetime import date, timedelta
from concurrent.futures import Future

import pandas as pd
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

import backend.services.strategy_routes as routes


class ImmediateExecutor:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, function, value):
        result = Future()
        try:
            result.set_result(function(value))
        except Exception as exc:
            result.set_exception(exc)
        return result


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    days = pd.date_range('2024-01-01', periods=12)
    rows = [{"date": day, "asset_name": name, "asset_alloc_name": "dates",
             "nv": 1. + rate * t, "available_at": day}
            for name, rate in [("A", .01), ("B", .002)] for t, day in enumerate(days)]
    pd.DataFrame(rows).to_parquet(tmp_path / 'asset_nv.parquet', index=False)
    monkeypatch.setattr(routes, 'DATA_DIR', tmp_path)
    monkeypatch.setattr(routes, 'ProcessPoolExecutor', ImmediateExecutor)
    with routes._schedule_cache_lock:
        routes._schedule_cache.clear()
    return tmp_path


def fixed():
    return routes.StrategySpec(type='fixed', name='constant', classes=[
        routes.StrategyClassItem(name='A', weight=.6), routes.StrategyClassItem(name='B', weight=.4)])


def test_backtest_cutoff_ignores_prices_after_end_and_reports_actual_dates(workspace):
    request = routes.BacktestRequest(alloc_name='dates', start_date='2024-01-02', end_date='2024-01-08', strategies=[fixed()])
    first = routes.api_backtest(request)
    assert first['dates'][-1] == '2024-01-08'
    assert first['research_interval']['requested_end'] == '2024-01-08'
    assert first['research_interval']['actual_end'] == '2024-01-08'
    path = workspace / 'asset_nv.parquet'
    data = pd.read_parquet(path)
    data.loc[data.date > '2024-01-08', 'nv'] *= 1.5
    data.to_parquet(path, index=False)
    second = routes.api_backtest(request)
    assert first['series'] == second['series']
    assert routes.api_backtest(request.model_copy(update={'end_date': None}))['dates'][-1] == '2024-01-12'


def test_point_weight_estimation_respects_explicit_end(workspace, monkeypatch):
    captured = []

    def fit(nav, *args, **kwargs):
        captured.append(nav.index[-1])
        return [.6, .4]

    monkeypatch.setattr(routes, 'compute_target_weights', fit)
    request = routes.ComputeWeightsRequest(alloc_name='dates', end_date='2024-01-05', strategy=routes.StrategySpec(
        type='target', classes=[routes.StrategyClassItem(name='A'), routes.StrategyClassItem(name='B')], target='min_risk'))
    result = routes.api_compute_weights(request)
    assert result['weights'] == [.6, .4]
    assert captured == [pd.Timestamp('2024-01-05')]


def test_schedule_cache_is_partitioned_by_end_and_reused_only_in_same_scope(workspace, monkeypatch):
    calls = []

    def worker(args):
        calls.append(args)
        return {'date': args['date'], 'weights': [.6, .4]}

    monkeypatch.setattr(routes, '_compute_weight_for_date', worker)
    strategy = routes.StrategySpec(type='target', classes=[routes.StrategyClassItem(name='A'), routes.StrategyClassItem(name='B')],
        rebalance={'enabled': True, 'recalc': True, 'mode': 'fixed', 'fixedInterval': 2},
        model={'window_mode': 'rollingN', 'data_len': 2})
    request = routes.ComputeScheduleRequest(alloc_name='dates', end_date='2024-01-08', strategy=strategy)
    first = routes.api_compute_schedule_weights(request)
    second = routes.api_compute_schedule_weights(request.model_copy(update={'end_date': date(2024, 1, 10)}))
    assert first['cache_key'] != second['cache_key']
    assert max(first['dates']) <= '2024-01-08'
    assert all(max(arg['nav_split']['index'])[:10] <= '2024-01-10' for arg in calls)
    seen = []

    def backtest(nav, strategies, **kwargs):
        seen.append(strategies[0].get('precomputed_weights'))
        return {'dates': [str(nav.index[0].date()), str(nav.index[-1].date())], 'series': {}, 'markers': {}}

    monkeypatch.setattr(routes, 'backtest_portfolio', backtest)
    matched = strategy.model_copy(update={'precomputed': first['cache_key']})
    routes.api_backtest(routes.BacktestRequest(alloc_name='dates', end_date='2024-01-08', strategies=[matched]))
    routes.api_backtest(routes.BacktestRequest(alloc_name='dates', end_date='2024-01-10', strategies=[matched]))
    assert seen[0] is not None
    assert seen[1] is None


@pytest.mark.parametrize('end,start', [
    (date.today() + timedelta(days=1), None), (date(2024, 1, 5), '2024-01-05'),
    (date(2024, 1, 1), None), (date(2024, 1, 5), 'not-a-date'),
])
def test_invalid_intervals_fail_before_calculation(workspace, end, start):
    with pytest.raises(HTTPException) as error:
        routes.api_backtest(routes.BacktestRequest(alloc_name='dates', start_date=start, end_date=end, strategies=[fixed()]))
    assert error.value.status_code == 400


def test_invalid_end_date_rejected_at_request_boundary():
    with pytest.raises(ValidationError):
        routes.BacktestRequest(alloc_name='dates', end_date='2024-02-31', strategies=[fixed()])
