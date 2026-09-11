"""End-to-end regressions from the project audit; all data is disposable."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.service import CustomIndicatorService
from custom_indicators.series_provider import load_product_chart_series
from services import analytics_routes
import fit


@pytest.fixture
def chart_data(tmp_path):
    dates = pd.bdate_range('2026-01-02', periods=8)
    prices = np.array([10., np.nan, 30., 40., 50., 60., 70., 80.])
    nav = np.array([100., 110., 99., 118.8, 106.92, 128.304, 141.1344, 127.02096])
    pd.DataFrame([{'ts_code': '510300.SH', 'code': '510300', 'name': 'Audit ETF'}]).to_parquet(tmp_path / 'etf_info_df.parquet', index=False)
    pd.DataFrame({'ts_code': '510300.SH', 'date': dates, 'ann_date': dates, 'adj_nav': nav}).to_parquet(tmp_path / 'etf_daily_df.parquet', index=False)
    pd.DataFrame({'ts_code': '510300.SH', 'date': dates, 'close': prices, 'open': prices, 'high': prices + 1, 'low': prices - 1, 'vol': 1000.}).to_parquet(tmp_path / 'etf_daily_candle_df.parquet', index=False)
    service = CustomIndicatorService(tmp_path, tmp_path)
    try:
        yield service, dates, prices, nav
    finally:
        service.close_compute_engine()


def _chart(service, expression, anchor, period='ALL', max_points=5000):
    saved = service.create_indicator({'name': 'Audit regression', 'result_kind': 'time_series', 'axis_anchor': anchor,
        'dsl_version': '2.4.0', 'operator_registry_version': '2.4.0', 'context_kind': 'single_product',
        'series_outputs': [{'id': 'value', 'label': 'Result', 'expression': expression, 'output_measure': 'auto'}]})
    response = service.evaluate_series(indicator_instances=[{'indicator_id': saved['id']}], target={'kind': 'etf', 'product_id': '510300.SH'}, period=period, max_points=max_points)
    assert response['execution']['request_time_compilation'] == 0
    assert response['execution']['python_fallback'] == 0
    return response['results'][0]


def test_missing_anchor_preserves_date_axis_and_invalid_window(chart_data):
    service, dates, _, _ = chart_data
    result = _chart(service, 'rolling_apply(mean(market_close),2)', 'market_close')
    assert result['dates'] == dates.strftime('%Y-%m-%d').tolist()
    assert result['channels'][0]['values'] == [None, None, None, 35., 45., 55., 65., 75.]


def test_returns_axis_keeps_first_valid_rolling_result(chart_data):
    service, dates, _, _ = chart_data
    level_axis = _chart(service, 'rolling_apply(mean(returns),3)', 'adjusted_nav')
    return_axis = _chart(service, 'rolling_apply(mean(returns),3)', 'returns')
    assert return_axis['dates'] == level_axis['dates'] == dates.strftime('%Y-%m-%d').tolist()
    np.testing.assert_allclose(np.asarray(return_axis['channels'][0]['values'], dtype=float),
                               np.asarray(level_axis['channels'][0]['values'], dtype=float), equal_nan=True)
    assert return_axis['channels'][0]['values'][3] == pytest.approx(1 / 15)


def test_missing_nav_does_not_bridge_gap_to_create_return(chart_data, tmp_path):
    service, dates, _, nav = chart_data
    nav[2] = np.nan
    pd.DataFrame({'ts_code': '510300.SH', 'date': dates, 'ann_date': dates, 'adj_nav': nav}).to_parquet(tmp_path / 'etf_daily_df.parquet', index=False)
    result = _chart(service, 'rolling_apply(mean(returns),1)', 'returns')
    assert result['dates'] == dates.strftime('%Y-%m-%d').tolist()
    assert result['channels'][0]['values'][2:4] == [None, None]
    assert result['channels'][0]['values'][4] == pytest.approx(-.1)


@pytest.mark.parametrize('expression', [
    'rolling_apply(mean(returns),3)',
    'rolling_apply(mean(returns) * window_elapsed_days,3)',
])
def test_chart_display_slice_matches_full_history_first_visible_result(chart_data, expression):
    service, _, _, _ = chart_data
    full = _chart(service, expression, 'returns')
    tail = _chart(service, expression, 'returns', max_points=3)
    assert tail['dates'] == full['dates'][-3:]
    assert tail['channels'][0]['values'] == pytest.approx(full['channels'][0]['values'][-3:])


def test_concurrent_fit_results_own_their_lineage_and_availability(tmp_path):
    dates = pd.bdate_range('2026-01-02', periods=12)
    frames = []
    for code, label, lag in [('510300.SH', 'Delayed ETF', 3), ('510050.SH', 'Same-day ETF', 0)]:
        frames.append(pd.DataFrame({'ts_code': code, 'name': label, 'date': dates, 'ann_date': dates + pd.Timedelta(days=lag),
                                    'adj_nav': 1 + np.arange(len(dates)) * .01}))
    pd.concat(frames).to_parquet(tmp_path / 'etf_daily_df.parquet', index=False)
    done_a, done_b = threading.Event(), threading.Event()

    def compute(code, label):
        return fit.compute_classes_nav(tmp_path, [fit.ClassSpec('one', 'Asset', [fit.ETFSpec(code, label, 1.)])],
                                       dates[0], as_of='2026-02-01')

    def a():
        result = compute('510300.SH', 'Delayed ETF')
        done_a.set()
        assert done_b.wait(30)
        return result

    def b():
        assert done_a.wait(30)
        try:
            return compute('510050.SH', 'Same-day ETF')
        finally:
            done_b.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_a, future_b = executor.submit(a), executor.submit(b)
        first, second = future_a.result(60), future_b.result(60)
    assert hasattr(first, 'available_at'), 'fit must return request-owned PIT data, not a process-global last result'
    day = pd.Timestamp('2026-01-06')
    assert first.available_at.loc[day] == day + pd.Timedelta(days=3)
    assert second.available_at.loc[day] == day
    assert first.lineage is not second.lineage
    assert first.nav.index.equals(second.nav.index)


def test_allocation_save_persists_its_own_pit_after_interleaved_fit(tmp_path, monkeypatch):
    import app as api

    dates = pd.bdate_range('2026-01-02', periods=12)
    pd.concat([
        pd.DataFrame({'ts_code': code, 'name': label, 'date': dates,
                      'ann_date': dates + pd.Timedelta(days=lag), 'adj_nav': 1 + np.arange(len(dates)) * .01})
        for code, label, lag in [('510300.SH', 'Delayed ETF', 3), ('510050.SH', 'Same-day ETF', 0)]
    ]).to_parquet(tmp_path / 'etf_daily_df.parquet', index=False)
    monkeypatch.setattr(api, 'DATA_DIR', tmp_path)

    def interleaved(*args, **kwargs):
        own = fit.compute_classes_nav(*args, **kwargs)
        fit.compute_classes_nav(tmp_path, [fit.ClassSpec('other', 'Other', [fit.ETFSpec('510050.SH', 'Same-day ETF', 1.)])],
                                dates[0], as_of='2026-01-16')
        return own

    monkeypatch.setattr(api, 'compute_classes_nav', interleaved)
    response = api.save_allocation(api.SaveRequest(asset_alloc_name='PIT-owned', as_of='2026-02-01', classes=[
        {'id': 'one', 'name': 'Asset', 'etfs': [{'code': '510300.SH', 'name': 'Delayed ETF', 'weight': 1.}]}]))
    assert isinstance(response, dict) and response['ok'], response
    saved = pd.read_parquet(tmp_path / 'asset_nv.parquet').set_index('date')
    assert saved.loc[pd.Timestamp('2026-01-06'), 'available_at'] == pd.Timestamp('2026-01-09')
    assert response['lineage']['pit']['as_of'] == '2026-02-01'
    assert set(saved['as_of']) == {'2026-02-01'}


@pytest.mark.parametrize('body,field', [
    ({'startDate': 'not-a-date', 'classes': []}, 'startDate'),
    ({'startDate': '2026-01-02', 'classes': []}, 'classes'),
    ({'startDate': 'NaT', 'classes': []}, 'startDate'),
])
def test_fit_invalid_parameters_are_client_errors(body, field):
    app = FastAPI()
    app.include_router(analytics_routes.router)
    response = TestClient(app, raise_server_exceptions=False).post('/api/fit-classes', json=body)
    assert response.status_code in {400, 422}
    assert field in response.text
