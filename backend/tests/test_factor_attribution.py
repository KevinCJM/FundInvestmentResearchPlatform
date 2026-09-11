"""Independent arithmetic checks and offline API tests for contribution attribution."""
import copy
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.factor_research import attribution_kernels as k
from backend.factor_research import numba_kernels as nk
from backend.factor_research.attribution import validate_attribution_size
from backend.factor_research.contracts import AttributionFields
from backend.custom_indicators.errors import IndicatorDomainError
from backend.services.factor_research_routes import build_router
from backend.tests import test_factor_research_service as factor_fixtures

context = factor_fixtures.context


def sample(n=160):
    rng = np.random.default_rng(914)
    factors = np.ascontiguousarray(rng.normal(0, .009, (n, 2)))
    rf = np.full(n, .0001)
    returns = np.ascontiguousarray((factors @ np.array([.6, .4]) + rf + .0002).reshape(n, 1))
    days = np.arange(n, dtype=np.int64)
    available = days.reshape(n, 1).copy()
    beta, stats = nk.attribution_kernel(returns, factors, rf, 90, 1)
    return returns, factors, rf, days, available, beta, stats


def exposure(args, rolling=True, minimum=30, model=1):
    returns, factors, rf, days, available, beta, stats = args
    return k.exposure_path_kernel(returns, factors, rf, available, days, beta, stats,
                                  90, 40, minimum, 10, int(rolling), model)


def test_fixed_daily_components_close_and_reuse_exact_exposures():
    args = sample()
    r, x, rf, _, _, beta, _ = args
    betas, meta, _, _ = exposure(args, rolling=False)
    values, checks, status = k.daily_contributions_kernel(r, x, rf, betas, meta)
    np.testing.assert_allclose(betas[:, 0], np.broadcast_to(beta[0], (len(r), 3)))
    np.testing.assert_allclose(values[:, 0, :2], x * [.6, .4], atol=1e-12)
    np.testing.assert_allclose(values[:, 0, 2], rf)
    np.testing.assert_allclose(values[:, 0, 3], .0002, atol=1e-12)
    np.testing.assert_allclose(values[:, 0, 4], 0, atol=1e-12)
    np.testing.assert_allclose(values.sum(axis=2), r, atol=1e-12)
    np.testing.assert_allclose(checks[:, :, 1], 0, atol=1e-12)
    assert np.all(status == 0)


def test_two_day_wealth_link_is_not_simple_addition_or_independent_compounding():
    r = np.array([[.10], [-.05]])
    # One factor + RF + intercept + residual. Day 2 starts with wealth1.1.
    values = np.array([[[.08, 0, .01, .01]], [[-.04, 0, .01, -.02]]])
    curves, summary = k.link_contributions_kernel(r, values, np.zeros((2, 1), np.int64), 0, 2)
    expected = np.array([.08 - 1.1 * .04, 0, .01 + 1.1 * .01, .01 - 1.1 * .02])
    np.testing.assert_allclose(summary[0, :4], expected)
    assert summary[0, 7] == pytest.approx(.045)  # Actual compound return.
    assert summary[0, 8] == pytest.approx(.045)  # Sum of linked contributions.
    assert summary[0, 9] == pytest.approx(0, abs=1e-12)
    assert not np.isclose(values.sum(), summary[0, 7])
    _, second = k.link_contributions_kernel(r, values, np.zeros((2, 1), np.int64), 1, 2)
    np.testing.assert_allclose(second[0, :4], values[1, 0])
    assert curves[-1, 0, 4] == pytest.approx(.045)


def test_zero_returns_negative_contributions_and_empty_intervals():
    r = np.zeros((3, 1))
    values = np.tile(np.array([[[.01, 0, 0, -.01]]]), (3, 1, 1))
    status = np.zeros((3, 1), np.int64)
    _, summary = k.link_contributions_kernel(r, values, status, 0, 3)
    np.testing.assert_allclose(summary[0, :4], [.03, 0, 0, -.03])
    assert summary[0, 7] == 0 and np.isnan(summary[0, 10])
    curves, summary = k.link_contributions_kernel(r, values, status, 2, 2)
    assert len(curves) == 0 and summary[0, -1] == 2
    assert np.isnan(summary[0, :4]).all()


def test_missing_contribution_breaks_link_but_not_known_actual_return():
    r = np.full((5, 1), .01)
    values = np.zeros((5, 1, 4))
    values[:, 0, 0] = .01
    values[2] = np.nan
    status = np.zeros((5, 1), np.int64)
    status[2] = 6
    curves, summary = k.link_contributions_kernel(r, values, status, 0, 5)
    assert summary[0, -1] == 1
    assert summary[0, 7] == pytest.approx(1.01 ** 5 - 1)
    assert np.isnan(summary[0, :4]).all()
    assert np.isnan(curves[2:, 0, :4]).all()
    assert summary[0, 5] == 4
    _, later = k.link_contributions_kernel(r, values, status, 3, 5)
    assert later[0, -1] == 0
    r[2] = np.nan
    _, broken = k.link_contributions_kernel(r, values, status, 0, 5)
    assert np.isnan(broken[0, 7])


@pytest.mark.parametrize('missing', [np.nan, np.inf])
def test_missing_rf_never_becomes_zero_or_residual(missing):
    args = sample()
    path, meta, _, _ = exposure(args, rolling=False)
    args[2][100] = missing
    values, checks, status = k.daily_contributions_kernel(args[0], args[1], args[2], path, meta)
    assert status[100, 0] == 6
    assert np.isnan(values[100]).all() and np.isnan(checks[100]).all()


def test_rolling_reference_window_lag_and_future_invariance():
    args = sample()
    path, meta, _, _ = exposure(args)
    assert np.all(meta[:40, :, 0] == 4)
    assert np.all(meta[40:, :, 3] < np.arange(40, 160)[:, None])
    for t in range(40, 160, 10):
        expected, stats = nk.attribution_kernel(args[0][t-40:t].copy(), args[1][t-40:t].copy(),
                                               args[2][t-40:t].copy(), 40, 1)
        np.testing.assert_allclose(path[t], expected, atol=1e-12)
        assert meta[t, 0, 1] == stats[0, 0]
    changed = [value.copy() for value in args]
    changed[0][100:] += .3
    changed[1][100:] *= -5
    changed_path, _, _, _ = exposure(changed)
    np.testing.assert_allclose(path[:101], changed_path[:101], equal_nan=True)


def test_rolling_delayed_announcements_excluded_and_failed_refit_not_carried_forward():
    args = sample()
    args[4][40:80] = 9999  # Both previously recorded dates and current date unavailable.
    path, meta, _, _ = exposure(args)
    assert meta[40, 0, 0] == 0
    assert meta[80, 0, 0] == 1
    assert np.isnan(path[80:90]).all()
    assert meta[80, 0, 1] == 0
    # The failed window keeps 40 time positions rather than reaching back for replacements.
    assert meta[80, 0, 2] == 40 and meta[80, 0, 3] == 79


def test_rolling_rbsa_constraints_and_singular_ols_status():
    args = sample()
    path, meta, _, _ = exposure(args, model=0)
    np.testing.assert_allclose(path[40:, 0, :2].sum(axis=1), 1, atol=1e-10)
    assert np.all(path[40:, 0, :2] >= 0)
    args[1][:, 1] = args[1][:, 0]
    path, meta, _, _ = exposure(args)
    assert np.all(meta[40:, :, 0] == 3)
    assert np.isnan(path[40:]).all()


def test_fixed_signatures_and_dtype_guard():
    assert k.warm_attribution_kernels()['complete']
    for kernel in k.KERNELS:
        assert len(kernel.signatures) == len(kernel.nopython_signatures) == 1
    with pytest.raises(TypeError):
        k.return_availability_kernel(np.ones((3, 2), dtype=np.int32))


def request(fields, **extra):
    keys = ('name', 'product_kind', 'targets', 'start_date', 'end_date', 'oos_date')
    return {**{key: fields[key] for key in keys}, 'model': 'rbsa', 'indices': ['INDEX0', 'INDEX1'], **extra}


def test_service_fixed_compatibility_and_immutable_contribution_snapshot(context, monkeypatch):
    service, fields, _ = context
    spies = {}
    for name in ('exposure_path_kernel', 'daily_contributions_kernel', 'link_contributions_kernel'):
        spies[name] = Mock(wraps=getattr(k, name))
        monkeypatch.setattr(k, name, spies[name])
    run = service.run_attribution(request(fields))
    analysis = run['attribution']
    assert analysis['mode'] == 'fixed'
    assert all(item['kind'] != 'risk_free' for item in analysis['components'])
    assert run['execution']['python_fallback'] == 0
    for spy in spies.values():
        assert spy.called and spy.call_args.args[0].dtype == np.float64
    for product in analysis['products']:
        for summary in product['summaries'].values():
            assert summary['status'] == 'complete'
            assert sum(summary['contributions']) == pytest.approx(summary['total_return'], abs=1e-10)
        result = next(item for item in run['results'] if item['code'] == product['code'])
        assert product['daily'][0]['exposures'] == [item['value'] for item in result['exposures']]
        assert product['daily'][0]['exposure_basis'] == 'retrospective_fit'
    saved = service.artifacts.get(run['id'])
    arrays = np.load(service.artifacts.root / (run['id'] + '.npz'), allow_pickle=False)
    assert arrays['exposure_path'].dtype == np.float64
    assert arrays['daily_status'].dtype == np.int64
    service.run_attribution(request(fields))
    assert service.artifacts.get(run['id']) == saved


def test_service_rolling_and_maturity_dates(context):
    service, fields, _ = context
    run = service.run_attribution(request(fields, exposure_mode='rolling', rolling_window=63,
                                           min_observations=40, refit_step=21))
    analysis = run['attribution']
    assert analysis['warmup_days'] == 63
    for product in analysis['products']:
        for row in product['daily'][63:]:
            assert row['fit_end'] < row['date']
        assert product['summaries']['all']['days'] == len(product['daily']) - 63
        assert product['summaries']['all']['start_date'] == product['daily'][63]['date']
        assert product['summaries']['all']['reconciliation_error'] == pytest.approx(0, abs=1e-10)
    assert analysis['summary_basis'] == 'last_scheduled_fit_and_walk_forward_oos'


def test_ff3_includes_rf_separately_and_raw_daily_identity(context):
    service, fields, _ = context
    dates = pd.bdate_range(fields['start_date'], fields['end_date'])
    rng = np.random.default_rng(31)
    factors = rng.normal(0, .01, (len(dates), 3))
    dataset = service.add_dataset({'name': '离线中国因子', 'source_url': 'https://example.org/offline',
        'market': 'CN', 'currency': 'CNY', 'construction': '离线独立因子测试，不是真实市场。',
        'rows': [{'date': str(day.date()), 'MKT_RF': values[0], 'SMB': values[1], 'HML': values[2], 'RF': .0001}
                 for day, values in zip(dates, factors)]})
    fields = request(fields, model='ff3', dataset_id=dataset['id'], indices=[])
    run = service.run_attribution(fields)
    analysis = run['attribution']
    assert analysis['dependent_return'] == 'excess'
    assert [c['id'] for c in analysis['components']][-3:] == ['rf', 'intercept', 'residual']
    for row in analysis['products'][0]['daily']:
        assert row['contributions'][3] == .0001
        assert sum(row['contributions']) == pytest.approx(row['actual_return'], abs=1e-12)


def test_api_cold_errors_and_legacy_default(context):
    service, fields, _ = context
    app = FastAPI()
    app.include_router(build_router(service))
    with TestClient(app) as client:
        body = request(fields)
        service._ready = False
        assert client.post('/api/factor-research/attributions', json=body).status_code == 503
        service.warm()
        invalid = {**body, 'exposure_mode': 'rolling', 'rolling_window': 30, 'min_observations': 60}
        assert client.post('/api/factor-research/attributions', json=invalid).status_code == 422
        response = client.post('/api/factor-research/attributions', json=body)
        assert response.status_code == 201, response.text
        assert response.json()['request']['exposure_mode'] == 'fixed'
        invalid = {**body, 'exposure_mode': 'rolling', 'rolling_window': 1260}
        response = client.post('/api/factor-research/attributions', json=invalid)
        assert response.status_code == 422 and '窗口' in response.text


def test_size_limits_fail_closed():
    config = {'exposure_mode': 'rolling', 'rolling_window': 60, 'min_observations': 40, 'refit_step': 1}
    with pytest.raises(IndicatorDomainError, match='200万'):
        validate_attribution_size(config, 6000, 40, 8)
    with pytest.raises(IndicatorDomainError, match='2500'):
        validate_attribution_size(config, 1200, 3, 2)
