"""Regression oracles for the 2026-09-16 audit; all inputs are isolated and offline."""
import copy
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.testclient import TestClient

from custom_indicators.errors import IndicatorDomainError, ValidationError
from historical_regimes.data import _parse_date
from historical_regimes.event_library import EventLibraryService
from historical_regimes.event_routes import install_event_routes
from historical_regimes.reliability import kernels as reliability_kernels
from historical_regimes.reliability.diagnostic_kernels import horizon_profile_kernel
from historical_regimes.segment_numba import drawdown_cycle_realtime_kernel
from historical_regimes.taa import run_taa_backtest
from scenario_stress import numba_kernels as nk
from scenario_stress.service import ScenarioStressService
from services.custom_indicator_routes import StableValidationRoute
from test_historical_regime_taa import _taa_request
from test_regime_reliability_consumer import _eligible_fixture
from test_tactical_allocation_service import workspace


@pytest.fixture(scope='module', autouse=True)
def warmed():
    nk.warm_scenario_numba_kernels()
    reliability_kernels.warm()


def partial_run(start, end):
    run = _eligible_fixture()
    context = run['_reliability']
    context['artifact']['created_at'] = str(start - timedelta(days=1)) + 'T00:00:00Z'
    context['artifact']['report']['calibration'].update(
        parameters={'counts': [[8., 2.], [2., 8.]], 'temperature': None},
        available_from=str(start), expires_on=str(end), deployment_eligible=False)
    context['qualification'] = {'id': 'forward-q', 'status': 'qualified',
        'qualified_states': ['bull'], 'fallback_states': ['bear'],
        'available_from': str(start - timedelta(days=1)) + 'T00:00:00Z', 'expires_at': str(end) + 'T23:59:59Z'}
    context['verified_at'] = str(end) + 'T12:00:00Z'
    run['definition']['study']['qualification_id'] = 'forward-q'
    run['series'] = [{**run['series'][0], 'observation_date': str(start - timedelta(days=2)),
                      'recognized_at': str(start - timedelta(days=2)), 'effective_date': str(start - timedelta(days=2))}]
    return run


def test_partial_qualification_keeps_unverified_probability_at_baseline():
    from datetime import date
    run = partial_run(date(2024, 1, 1), date(2024, 3, 31))
    request = _taa_request()
    request['state_tilts'] = {'bull': {'equity': .1, 'bond': -.1}, 'bear': {'equity': -.5, 'bond': .5}}
    before = copy.deepcopy(run)
    result = run_taa_backtest(run, request, {'passed': True})
    assert result['weights'][0]['fallback_reason'] == 'no_safe_period_start'
    for row in result['weights'][1:]:
        assert row['probabilities'] == {'bull': .8, 'bear': .2}
        assert row['allocation_probabilities'] == {'bull': .8, 'bear': 0.}
        assert row['weights'] == pytest.approx({'equity': .58, 'bond': .42})
        assert row['state_contributions']['bear'] == 0.
    assert run == before
    run['series'][0]['state_id'] = 'bear'
    assert all(row['fallback_to_base'] for row in run_taa_backtest(run, request, {'passed': True})['weights'])


def test_tactical_partial_qualification_uses_same_mass_without_renormalizing(workspace):
    service, _, request = workspace
    service.warm()
    run = partial_run(request.start_date, request.as_of + timedelta(days=1))
    service.regime_resolver = lambda _: (run, {'passed': True})
    baseline = service.create_baseline({'alloc_name': '配置', 'name': 'Unconstrained oracle', 'as_of': str(request.as_of), 'weights': {'股票': .6, '债券': .4}})
    body = request.model_copy(update={'baseline_id': baseline['id'], 'signal_mode': 'regime', 'regime_run_id': 'qualified-fixture',
        'state_tilts': {'bull': {'股票': .1, '债券': -.1}, 'bear': {'股票': -.5, '债券': .5}},
        'max_signal_age_days': 3650, 'confidence_floor': .1, 'selected_candidate_id': 'scale-4'})
    result = service.preview(body)
    for row in result['audit']['signal']['signal_timing']:
        assert row['probabilities'] == {'bull': .8, 'bear': .2}
        assert row['allocation_probabilities'] == {'bull': .8, 'bear': 0.}
    for row in result['weight_path']:
        assert row['weights'] == pytest.approx({'股票': .68, '债券': .32})


@pytest.mark.parametrize('dates,days', [
    (['2020-01-31', '2020-02-29', '2020-03-31'], 31),
    (['2019-12-31', '2020-01-31', '2020-02-29'], 29),
    (['2020-01-03', '2020-01-10', '2020-01-17'], 7),
    (['2020-01-02', '2020-01-03', '2020-01-06'], 3),
])
def test_calendar_duration_is_half_open_until_next_observation(dates, days):
    axis = np.array(dates, dtype='datetime64[D]').astype(np.int64)
    per, summary = horizon_profile_kernel(np.array([0, 1, 0], np.int64), axis, 2)
    assert per[1, 0] == 1 and per[1, 2] == 1
    assert per[1, 5] == days
    assert per[0, 0] == 0  # both boundary episodes are censored
    assert summary[2] == axis[-1] - axis[0]
    censored, _ = horizon_profile_kernel(np.array([-1, 1, 0], np.int64), axis, 2)
    assert censored[1, 0] == 0


def test_zero_exit_drawdown_is_valid_and_recovers_at_previous_high():
    prices = np.array([100., 80., 85., 100., 101.])
    assert drawdown_cycle_realtime_kernel(prices, 9, .12, .05, 0.).tolist() == [0, 2, 1, 0, 0]
    broken = np.array([100., np.nan, 85., 100.])
    assert drawdown_cycle_realtime_kernel(broken, 9, .12, .05, 0.).tolist() == [0, -1, -1, -1]


def test_event_list_accepts_unspecified_dates_with_production_validation(tmp_path):
    library = EventLibraryService(tmp_path)
    router = APIRouter(route_class=StableValidationRoute)
    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except IndicatorDomainError as exc:
            raise HTTPException(exc.status_code, detail=exc.detail()) from exc
    install_event_routes(router, lambda: SimpleNamespace(event_library=library), call)
    app = FastAPI(); app.include_router(router)
    with TestClient(app) as client:
        response = client.get('/api/historical-regimes/event-library/events')
        assert response.status_code == 200, response.text
        assert client.get('/api/historical-regimes/event-library/events?start=2020-01-01').status_code == 200
        assert client.get('/api/historical-regimes/event-library/events?start=wrong').status_code == 422


@pytest.mark.parametrize('revision', ['abc', '1', 1.9, True, 0, -1, [], {}])
def test_invalid_revision_is_a_route_domain_error_not_500(tmp_path, monkeypatch, revision):
    from services import scenario_stress_routes as routes
    monkeypatch.setattr(routes, 'scenario_stress_service', ScenarioStressService(tmp_path, tmp_path))
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        response = client.post('/api/scenario-stress/run', json={'definition': {'id': 'missing', 'revision': revision}})
    assert response.status_code == 422, response.text
    assert response.json()['detail']['code'] == 'INVALID_SCENARIO_REVISION'


@pytest.mark.parametrize('weights,coverage,status', [
    ([.5, -.8, 1.3], .5, 3), ([1.4, -1., .6], .8, 4), ([.6, .2, .2], .8, 0),
])
def test_degrade_rechecks_net_direction_and_leverage(weights, coverage, status):
    effective, ratio, actual, _ = nk.coverage_weights_kernel(
        np.array(weights, np.float64), np.array([1, 1, 0], np.uint8), np.int64(1), np.float64(coverage))
    assert actual == status and ratio + 1e-12 >= coverage
    if not status:
        np.testing.assert_allclose(effective, [.75, .25, 0.])
    from scenario_stress.engine import _raise_coverage_status
    if status:
        with pytest.raises(ValidationError) as error:
            _raise_coverage_status({'mapping': {'minimum_coverage': coverage}}, {'id': 'p', 'label': 'p'}, status, ratio, ['missing'], 'mapping')
        assert error.value.code == ('DEGRADED_WEIGHT_NORMALIZATION_FAILED' if status == 3 else 'DEGRADED_WEIGHT_LIMIT_EXCEEDED')


@pytest.mark.parametrize('value', ['0001-01-01', '1600-01-01', '9999-12-31'])
def test_dates_outside_numeric_axis_fail_as_domain_validation(value):
    with pytest.raises(ValidationError) as error:
        _parse_date(value, 'as_of')
    assert error.value.code == 'DATE_OUT_OF_RANGE'


def test_horizon_levels_and_pending_response_have_separate_oracles():
    path = np.array([[-.1, -2.], [0., 0.], [.1, 3.]])
    original = path.copy()
    levels, pending = nk.horizon_level_evidence_kernel(path, np.array([1, 0], np.int64), np.int64(2))
    np.testing.assert_allclose(levels, [-.1, -2.])
    assert pending == 3.
    recovery, no_pending = nk.horizon_level_evidence_kernel(np.array([[-.1], [1 / .9 - 1]]), np.array([1], np.int64), np.int64(2))
    assert abs(recovery[0]) < 1e-12 and no_pending == 0.
    np.testing.assert_array_equal(path, original)
    for periods in [-1, 4]:
        with pytest.raises(ValueError):
            nk.horizon_level_evidence_kernel(path, np.array([1, 0], np.int64), np.int64(periods))
    with pytest.raises(ValueError):
        nk.horizon_level_evidence_kernel(path, np.array([1], np.int64), np.int64(1))


def test_readiness_requires_current_worker_warmup_and_compilation_closed(monkeypatch):
    nk.warm_scenario_numba_kernels.cache_clear()
    assert not nk.scenario_stress_numba_status()['fully_warmed']
    with pytest.raises(RuntimeError, match='warmup incomplete'):
        nk.assert_scenario_stress_numba_ready()
    nk.warm_scenario_numba_kernels()
    nk.assert_scenario_stress_numba_ready()
    with monkeypatch.context() as m:
        m.setattr(nk, '_WARMED_PID', -1)
        with pytest.raises(RuntimeError, match='warmup incomplete'):
            nk.assert_scenario_stress_numba_ready()
    with monkeypatch.context() as m:
        m.setattr(nk.SCENARIO_STRESS_NUMBA_KERNELS[0], '_can_compile', True)
        assert nk.scenario_stress_numba_status()['request_time_compilation'] == 1
        with pytest.raises(Exception):
            nk.assert_scenario_stress_numba_ready()
    nk._reset_worker_warmup()
    assert not nk.scenario_stress_numba_status()['fully_warmed']
    assert nk.warm_scenario_numba_kernels()['fully_warmed']


@pytest.mark.parametrize('evaluation_target', [False, True])
def test_indicator_definition_replays_frozen_values_after_append_and_restart(tmp_path, evaluation_target):
    from custom_indicators.service import CustomIndicatorService
    from historical_regimes.v2_service import RegimeGraphV2Service
    from historical_regimes.v2_contracts import parse_definition_v2
    from test_historical_regime_indicator_source import _write_etf_data, _indicator_draft
    from test_historical_regime_v2_p1 import _definition
    market = tmp_path / 'market'; market.mkdir()
    _write_etf_data(market)
    indicators = CustomIndicatorService(market, market)
    indicator = indicators.create_indicator(_indicator_draft('Frozen mean return', 'mean(returns)'))
    indicators.warm_indicator_revision(indicator['id'], indicator['revision'])
    graph = RegimeGraphV2Service(tmp_path / 'graph', market, indicator_service=indicators)
    raw = _definition()
    raw['graph']['nodes'][0] = {'id': 'source', 'type': 'source.indicator', 'parameters': {
        'indicator_id': indicator['id'], 'indicator_revision': indicator['revision'],
        'product_kind': 'etf', 'product_id': '510050.SH', 'period': '1W', 'frequency': 'daily',
        'availability_mode': 'point_in_time'}}
    if evaluation_target:
        raw['evaluation_targets'] = [{'id': 'measure', 'name': '冻结评价指标', 'primary': True,
            'source': {'kind': 'indicator', **raw['graph']['nodes'][0]['parameters']}}]
    saved = graph.create_definition(raw)
    original = copy.deepcopy(saved)
    parsed = parse_definition_v2(saved)
    def read(definition, mode='realtime', as_of=None):
        return graph._resolve_sources(definition, {'source'}, mode, as_of)[0]['source']
    def assert_evaluation(definition, mode, as_of=None):
        if not evaluation_target:
            return
        from historical_regimes.v2_service import PortValue, _as_dates, _as_values
        bundle = read(definition, mode, as_of)
        axis = PortValue(_as_values(bundle.frame), _as_dates(bundle.frame, 'observation_date'), _as_dates(bundle.frame, 'available_at'))
        values, snapshot, _, _ = graph._evaluation_values(definition, axis, {'source': axis}, mode, as_of)
        np.testing.assert_allclose(values, axis.values, equal_nan=True)
        assert snapshot['fingerprint'] == bundle.snapshot['fingerprint']
    before = read(parsed)
    snapshot = saved['graph']['nodes'][0]['parameters']['indicator_data_snapshot']
    assert snapshot['frozen_series']['content_hash']
    short = read(parsed, 'retrospective', '2020-04-01')
    assert 0 < len(short.frame) < len(before.frame)
    assert short.snapshot['fingerprint'] == before.snapshot['fingerprint']
    # A legacy definition lacks a frozen array, but retains its original boundary.
    legacy = copy.deepcopy(saved)
    legacy['graph']['nodes'][0]['parameters']['indicator_data_snapshot'].pop('frozen_series')
    if evaluation_target:
        legacy['evaluation_targets'][0]['source']['indicator_data_snapshot'].pop('frozen_series')
    old_parsed = parse_definition_v2(legacy)
    frame = pd.read_parquet(market / 'etf_daily_df.parquet')
    last = frame.iloc[[-1]].copy()
    last['date'] += pd.offsets.BDay(1); last['ann_date'] = last['date']; last['adj_nav'] *= 1.001
    pd.concat([frame, last], ignore_index=True).to_parquet(market / 'etf_daily_df.parquet', index=False)
    refreshed = CustomIndicatorService(market, market)
    refreshed.warm_indicator_revision(indicator['id'], indicator['revision'])
    graph = RegimeGraphV2Service(tmp_path / 'graph', market, indicator_service=refreshed)
    for definition in (parsed, old_parsed):
        for mode in ('realtime', 'retrospective'):
            actual = read(definition, mode)
            pd.testing.assert_series_equal(actual.frame['value'], before.frame['value'])
            assert actual.snapshot['fingerprint'] == before.snapshot['fingerprint']
            assert len(read(definition, mode, '2020-04-01').frame) == len(short.frame)
            assert_evaluation(definition, mode)
            assert_evaluation(definition, mode, '2020-04-01')
    upgraded = graph._freeze_source_versions(legacy)
    assert upgraded['graph']['nodes'][0]['parameters']['indicator_data_snapshot']['frozen_series']
    if evaluation_target:
        assert upgraded['evaluation_targets'][0]['source']['indicator_data_snapshot']['frozen_series']
    assert graph.get_definition(saved['id'], saved['revision']) == original
    # Full graph execution without a cutoff must also accept decoded buffers;
    # slicing at as_of can otherwise hide read-only signature incompatibilities.
    prepared = graph.prepare(saved)
    run = graph.run_saved({'schema_version': '2.0', 'id': saved['id'], 'revision': saved['revision']},
                          'realtime', None, prepared['compile_token'])
    assert run['series_summary']['row_count'] == len(before.frame)
    changed = copy.deepcopy(saved)
    changed['graph']['nodes'][0]['parameters']['product_id'] = '510300.SH'
    with pytest.raises(ValidationError) as error:
        read(parse_definition_v2(changed))
    assert error.value.code == 'INDICATOR_SOURCE_BINDING_MISMATCH'


def test_bound_index_and_evaluation_read_validated_directory_despite_env_override(tmp_path, monkeypatch):
    from historical_regimes.v2_service import RegimeGraphV2Service
    from historical_regimes.v2_contracts import parse_definition_v2
    from test_historical_regime_v2 import _definition
    a = tmp_path / 'a'; a.mkdir()
    b = tmp_path / 'b'; b.mkdir()
    rows = pd.DataFrame({'ts_code': ['000300.SH'] * 5, 'trade_date': pd.date_range('2020-01-01', periods=5),
                         'close': [100., 101., 102., 103., 104.]})
    rows.to_parquet(a / 'index_daily_df.parquet', index=False)
    other = rows.copy(); other['close'] *= 10
    other.to_parquet(b / 'index_daily_df.parquet', index=False)
    graph = RegimeGraphV2Service(tmp_path / 'workspace', a)
    raw = _definition()
    spec = {'ts_code': '000300.SH', 'source_api': 'index_daily', 'field': 'close'}
    raw['graph']['nodes'][0] = {'id': 'source', 'type': 'source.index', 'parameters': spec}
    raw['evaluation_targets'] = [{'id': 'csi300', 'name': 'CSI300', 'source': {'kind': 'index', **spec}, 'primary': True}]
    saved = graph.create_definition(raw)
    parsed = parse_definition_v2(saved)
    monkeypatch.setenv('TUSHARE_DATA_DIR', str(b))
    bundle = graph._resolve_sources(parsed, {'source'}, 'realtime', None)[0]['source']
    assert bundle.frame['value'].tolist() == rows['close'].tolist()
    root = graph._bound_source_root('source.index', parsed.evaluation_targets[0].source)
    from historical_regimes.data import resolve_target
    target = resolve_target(parsed.evaluation_targets[0].source, 'retrospective', None, root, resolved_data_dir=True)
    assert target.frame['value'].tolist() == rows['close'].tolist()


def test_relative_resolver_forwards_a_prevalidated_root_to_both_children(tmp_path, monkeypatch):
    from historical_regimes import data
    calls = []
    def index(spec, mode, as_of, root, *, resolved_data_dir=False):
        calls.append((root, resolved_data_dir))
        dates = pd.date_range('2020-01-01', periods=2)
        return data.DataBundle(pd.DataFrame({'observation_date': dates, 'available_at': dates,
                                             'value': [1., 2.], 'is_final': [True, True]}), {})
    monkeypatch.setattr(data, '_index_bundle', index)
    result = data.resolve_target({'kind': 'relative', 'numerator': {'kind': 'index'}, 'denominator': {'kind': 'index'}},
                                 'retrospective', None, tmp_path, resolved_data_dir=True)
    assert calls == [(tmp_path, True), (tmp_path, True)]
    assert result.frame['value'].tolist() == [0., 0.]  # Default log_ratio of identical series.
