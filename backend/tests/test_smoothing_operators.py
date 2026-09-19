"""Independent numerical references plus real graph/preview/temporal contracts."""
import copy
import time
from datetime import date, timedelta
import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt, savgol_filter

from computation_graph.smoothing_numba import (
    SMOOTHING_KERNELS, butterworth_zero_phase_kernel as butterworth,
    savitzky_golay_centered_kernel as savgol, ehlers_error_correcting_kernel as ehlers,
    smoothing_available_kernel,
)
from computation_graph.smoothing_operators import SMOOTHING_OPERATORS
from historical_regimes.authoring import AuthoringRequest, resolve_authoring
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_numba import KERNELS, regime_graph_numba_status
from historical_regimes.v2_service import PortValue, RegimeGraphV2Service, _kernel_ids_for_definition
from historical_regimes.v2_templates import get_template_v2
from custom_indicators.errors import ValidationError
from test_historical_regime_v2 import _definition
import test_historical_regime_v2 as graph_fixtures

client = graph_fixtures.client
classic_service = graph_fixtures.classic_service
v2_service = graph_fixtures.v2_service


@pytest.mark.parametrize("period", [3, 20, 63, 5000])
def test_butterworth_matches_scipy_sos_and_segment_boundaries(period):
    x = np.random.default_rng(42).normal(size=260).cumsum() + 100
    x[80:82] = [np.nan, np.inf]
    actual = butterworth(x, period)
    for start, end in ((0, 80), (82, 260)):
        expected = sosfiltfilt(butter(2, 2 / period, output="sos"), x[start:end], padlen=9)
        np.testing.assert_allclose(actual[start:end], expected, rtol=1e-8, atol=1e-8)
    assert np.isnan(actual[80:82]).all()
    assert np.isnan(butterworth(x[:9], period)).all()


@pytest.mark.parametrize("window,order", [(3, 0), (5, 4), (21, 3), (63, 5), (501, 5)])
def test_savgol_matches_scipy_and_polynomial_projection(window, order):
    x = np.random.default_rng(7).normal(size=600)
    half = window // 2
    actual = savgol(x, window, order)
    if window <= 63:
        expected = savgol_filter(x, window, order)[half:-half]
    else:
        # Large unscaled powers in SciPy's coefficient solve lose ~3e-8 here.
        # An independent scaled SVD oracle preserves the polynomial moments.
        design = np.vander(np.linspace(-1, 1, window), order + 1, increasing=True)
        target = np.eye(order + 1)[0]
        weights = np.linalg.lstsq(design.T, target, rcond=None)[0]
        expected = np.convolve(x, weights[::-1], mode='valid')
    np.testing.assert_allclose(actual[half:-half], expected, atol=2e-9)
    assert np.isnan(actual[:half]).all() and np.isnan(actual[-half:]).all()
    polynomial = np.linspace(-1, 1, 600) ** order
    np.testing.assert_allclose(savgol(polynomial, window, order)[half:-half], polynomial[half:-half], atol=1e-12)
    x[300] = np.inf
    assert np.isnan(savgol(x, window, order)[300-half:301+half]).all()


def test_ehlers_matches_independent_grid_reference_and_is_prefix_invariant():
    x = np.random.default_rng(3).normal(size=160).cumsum() + 100
    period, limit, alpha = 20, 50, 2 / 21
    from scipy.signal import lfilter
    ema = lfilter([alpha], [1, -(1-alpha)], x, zi=[(1-alpha)*x[0]])[0]
    expected = np.full(len(x), np.nan)
    previous = x[0]
    gains = np.arange(-limit, limit+1) / 10
    for t in range(1, len(x)):
        candidates = alpha * (ema[t] + gains * (x[t]-previous)) + (1-alpha)*previous
        previous = candidates[np.argmin(np.abs(x[t]-candidates))]
        if t >= period-1:
            expected[t] = previous
    np.testing.assert_allclose(ehlers(x, period, limit), expected, atol=1e-12)
    for end in [0, 1, 19, 20, 61, 160]:
        np.testing.assert_array_equal(ehlers(x[:end], period, limit), ehlers(x, period, limit)[:end])
    changed = x.copy(); changed[90:] += 500
    np.testing.assert_array_equal(ehlers(changed, period, limit)[:90], ehlers(x, period, limit)[:90])
    x[60] = np.nan
    np.testing.assert_array_equal(ehlers(x, period, limit)[61:], ehlers(x[61:], period, limit))


@pytest.mark.parametrize("kernel,args", [(butterworth, (20,)), (savgol, (21, 3)), (ehlers, (20, 50))])
def test_fixed_readonly_strided_views_empty_missing_and_determinism(kernel, args):
    base = np.arange(200, dtype=np.float64)
    for view in (base, base[::2], base[::-1]):
        view.flags.writeable = False
        before = base.copy()
        assert np.shares_memory(base, view)
        signatures = tuple(kernel.signatures)
        actual = kernel(view, *args)
        np.testing.assert_array_equal(actual, kernel(view, *args))
        np.testing.assert_array_equal(base, before)
        assert not np.shares_memory(actual, base)
        assert actual.dtype == np.float64 and tuple(kernel.signatures) == signatures
    for size in [0, 1, 2, 100]:
        assert np.isnan(kernel(np.full(size, np.nan), *args)).all()
    constant = kernel(np.ones(120), *args)
    expected_count = 120 if kernel is butterworth else 100 if kernel is savgol else 101
    assert np.isfinite(constant).sum() == expected_count
    np.testing.assert_allclose(constant[np.isfinite(constant)], 1, atol=1e-12)
    assert not kernel._can_compile and kernel.nopython_signatures
    with pytest.raises(TypeError):
        kernel(np.arange(20, dtype=np.float32), *args)


@pytest.mark.parametrize("kernel,args", [(butterworth, (2,)), (savgol, (20, 3)), (savgol, (3, 3)), (ehlers, (20, 101))])
def test_kernel_rejects_invalid_parameters(kernel, args):
    with pytest.raises(ValueError, match="INVALID_SMOOTHING_PARAMETERS"):
        kernel(np.ones(30), *args)


def definition_for(identifier):
    raw = _definition()
    node = next(n for n in raw['graph']['nodes'] if n['id'] == 'smooth')
    node.update(type=identifier, parameters={})
    return raw


@pytest.mark.parametrize("identifier", list(SMOOTHING_OPERATORS))
def test_real_graph_roundtrip_runtime_preview_and_temporal_gates(identifier, tmp_path):
    raw = definition_for(identifier)
    original = copy.deepcopy(raw)
    source = resolve_authoring(AuthoringRequest(definition=raw, source_kind="graph"))
    assert source['valid'], source
    again = resolve_authoring(AuthoringRequest(definition=raw, source_kind="formula", source=source['source']))
    assert again['valid'] and again['source'] == source['source']
    assert raw == original
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    definition = parse_definition_v2(raw)
    spec = SMOOTHING_OPERATORS[identifier]
    assert spec['kernel_id'] in _kernel_ids_for_definition(definition)
    before = regime_graph_numba_status()['kernel_signatures']
    plan = service.prepare(raw)
    mode = 'realtime' if spec['causal'] else 'retrospective'
    job = service.create_preview(raw, mode=mode, compile_token=plan['compile_token'])
    for _ in range(200):
        job = service.get_preview(job['id'])
        if job['status'] in {'completed', 'failed'}:
            break
        time.sleep(.02)
    assert job['status'] == 'completed', job
    result = job['result']
    assert result['diagnostics']['python_fallback'] == 0
    assert result['diagnostics']['request_time_compilation'] == 0
    if not spec['causal']:
        with pytest.raises(ValidationError):
            service.create_preview(raw, mode='realtime', compile_token=plan['compile_token'])
        points = service.preview_series(job['id'])['items']
        classified = [p for p in points if p['state_code'] >= 0]
        assert classified
        assert all(p['effective_date'] is None and not p['executable'] for p in points)
        assert all(p['recognized_at'] == raw['graph']['nodes'][0]['parameters']['rows'][-1]['available_at'] for p in classified)
        assert all('事后滤波' in p['reasons'][0] and '峰谷' not in p['reasons'][0] for p in classified)
    assert regime_graph_numba_status()['kernel_signatures'] == before
    assert inspect_definition_v2(parse_definition_v2(raw))['valid']
    target = {'node_id': 'smooth', 'port': 'value'}
    preview_plan = service.prepare(raw, preview_target=target)
    single = service.create_preview(raw, mode=mode, preview_target=target, compile_token=preview_plan['compile_token'])
    for _ in range(200):
        single = service.get_preview(single['id'])
        if single['status'] in {'completed', 'failed'}:
            break
        time.sleep(.02)
    assert single['status'] == 'completed', single
    rows = service.preview_series(single['id'])['items']
    actual = np.array([np.nan if row['value'] is None else row['value'] for row in rows])
    raw_values = np.array([row['value'] for row in raw['graph']['nodes'][0]['parameters']['rows']])
    returns = np.r_[np.nan, raw_values[1:]/raw_values[:-1]-1]
    expected = SMOOTHING_KERNELS[spec['kernel_id']](returns, *(field['default'] for field in spec['parameters'].values()))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert regime_graph_numba_status()['kernel_signatures'] == before


def test_availability_and_future_dependency_are_not_hidden():
    axis = np.array([1, 4, 3, 5], dtype=np.int64)
    np.testing.assert_array_equal(smoothing_available_kernel(axis, 0), [1, 4, 4, 5])
    np.testing.assert_array_equal(smoothing_available_kernel(axis, 1), [5, 5, 5, 5])
    x = np.sin(np.arange(120) / 12)
    changed = x.copy(); changed[90:] += 3
    assert not np.allclose(butterworth(x, 20)[:90], butterworth(changed, 20)[:90])
    assert not np.allclose(savgol(x, 21, 3)[80:90], savgol(changed, 21, 3)[80:90])


def test_smoothed_templates_keep_returns_on_raw_prices_and_distinct_versions():
    for suffix in ('butterworth', 'savgol'):
        template = get_template_v2(f'market-trend-smoothed-{suffix}-reference-v2')
        assert template['version'] == 2
        definition = template['definition']
        nodes = {n['id']: n for n in definition['graph']['nodes']}
        assert nodes['pivots']['inputs']['value']['node_id'] == 'smooth'
        assert nodes['filtered']['inputs']['value']['node_id'] == 'smooth'
        assert nodes['change']['inputs']['value']['node_id'] == 'market'
        assert inspect_definition_v2(parse_definition_v2(definition))['valid']
    old = get_template_v2('market-trend-reference-csi300-v1')['definition']
    assert not any(n['type'] in SMOOTHING_OPERATORS for n in old['graph']['nodes'])


def test_all_butterworth_periods_match_independent_sos():
    values = np.random.default_rng(96).normal(size=37)
    for period in range(3, 5001):
        expected = sosfiltfilt(butter(2, 2 / period, output='sos'), values, padlen=9)
        np.testing.assert_allclose(butterworth(values, period), expected, rtol=2e-8, atol=2e-9,
                                   err_msg=f'period={period}')


@pytest.mark.parametrize('period', [3, 20, 63, 5000])
@pytest.mark.parametrize('scale', [1e-300, -1e-300, 1e308, -1e308])
def test_butterworth_finite_extremes_preserve_dc_and_segment_independence(period, scale):
    values = np.full(40, scale)
    values[15] = np.nan
    actual = butterworth(values, period)
    assert np.isfinite(actual).sum() == 39
    np.testing.assert_allclose(actual[np.isfinite(actual)] / scale, 1, atol=2e-9, rtol=0)
    np.testing.assert_array_equal(actual[16:], butterworth(values[16:], period))


def test_zero_phase_impulse_symmetry_and_squared_cutoff_gain():
    impulse = np.zeros(1001); impulse[500] = 1
    for result in (butterworth(impulse, 20), savgol(impulse, 21, 3)):
        np.testing.assert_allclose(result, result[::-1], atol=1e-14, equal_nan=True)
        assert np.nanargmax(result) == 500
        assert np.nansum(result) == pytest.approx(1, abs=1e-12)
    t = np.arange(4000)
    wave = np.cos(2*np.pi*t/40)
    filtered = butterworth(wave, 40)
    # Interior comprises whole periods; a cutoff sinusoid must retain phase and half its amplitude.
    np.testing.assert_allclose(filtered[800:-800], .5*wave[800:-800], atol=1e-12)


def test_all_1496_valid_savgol_parameter_pairs_match_scaled_svd():
    rng = np.random.default_rng(213)
    pairs = 0
    for window in range(3, 502, 2):
        values = rng.normal(size=window + 12)
        for order in range(min(5, window - 1) + 1):
            design = np.vander(np.linspace(-1, 1, window), order + 1, increasing=True)
            weights = np.linalg.lstsq(design.T, np.eye(order + 1)[0], rcond=None)[0]
            expected = np.convolve(values, weights[::-1], mode='valid')
            actual = savgol(values, window, order)
            half = window // 2
            assert np.isfinite(actual).sum() == 13
            np.testing.assert_allclose(actual[half:-half], expected, atol=1e-12,
                                       err_msg=f'window={window}, order={order}')
            pairs += 1
    assert pairs == 1496


@pytest.mark.parametrize('period', [2, 20, 5000])
@pytest.mark.parametrize('limit', [0, 1, 50, 100])
def test_ehlers_parameter_edges_match_independent_reference(period, limit):
    from scipy.signal import lfilter
    values = np.random.default_rng(53).normal(size=period + 25).cumsum()
    alpha = 2 / (period + 1)
    ema = lfilter([alpha], [1, -(1-alpha)], values, zi=[(1-alpha)*values[0]])[0]
    previous = values[0]
    gains = np.arange(-limit, limit+1) / 10
    expected = np.full(values.size, np.nan)
    for t in range(1, values.size):
        candidates = alpha*(ema[t]+gains*(values[t]-previous))+(1-alpha)*previous
        previous = candidates[np.argmin(abs(values[t]-candidates))]
        if t >= period-1:
            expected[t] = previous
    actual = ehlers(values, period, limit)
    assert np.isfinite(actual).sum() == 26
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)
    if limit == 0:
        double_ema = lfilter([alpha], [1, -(1-alpha)], ema, zi=[(1-alpha)*values[0]])[0]
        np.testing.assert_allclose(actual[period-1:], double_ema[period-1:], atol=1e-11)


@pytest.mark.parametrize('identifier', list(SMOOTHING_OPERATORS))
def test_actual_node_execution_borrows_input_and_keeps_axis(identifier, tmp_path, monkeypatch):
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    node = parse_definition_v2(definition_for(identifier)).graph.nodes[2]
    base = np.linspace(80, 110, 320)
    values = base[::2]; values.flags.writeable = False
    dates = np.arange(160, dtype=np.int64)
    available = dates.copy(); available[30] = 80; available.flags.writeable = False
    source = PortValue(values, dates, available)
    spec = SMOOTHING_OPERATORS[identifier]
    original = SMOOTHING_KERNELS[spec['kernel_id']]
    assert KERNELS[spec['kernel_id']] is original
    calls = []
    def capture(received, *args):
        assert received is values and np.shares_memory(received, base)
        calls.append(True)
        return original(received, *args)
    monkeypatch.setitem(SMOOTHING_KERNELS, spec['kernel_id'], capture)
    result = service._execute_numeric_node(node, {'returns': {'value': source}}, 3,
        'realtime' if spec['causal'] else 'retrospective', None, None, {}, {}, {})['value']
    assert calls == [True] and original.nopython_signatures and not original._can_compile
    assert result.dates is dates and not np.shares_memory(result.values, values)
    np.testing.assert_array_equal(base, np.linspace(80, 110, 320))
    np.testing.assert_array_equal(result.available, np.maximum.accumulate(available) if spec['causal'] else np.full(160, available.max()))


@pytest.mark.parametrize('identifier', list(SMOOTHING_OPERATORS)[:2])
def test_lag_does_not_turn_hindsight_into_realtime(identifier, tmp_path, monkeypatch):
    raw = definition_for(identifier)
    raw['graph']['nodes'].append({'id': 'lagged', 'type': 'transform.lag', 'parameters': {'window': 40},
        'inputs': {'value': {'node_id': 'smooth', 'port': 'value'}}})
    raw['graph']['nodes'][3]['inputs']['value']['node_id'] = 'lagged'
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    monkeypatch.setattr(service, '_resolve_sources', lambda *a, **k: pytest.fail('must reject before reading sources'))
    with pytest.raises(ValidationError) as caught:
        service._execute_graph(None, parse_definition_v2(raw), 'realtime', None)
    assert caught.value.code == 'NON_CAUSAL_REALTIME_GRAPH'


@pytest.mark.parametrize('suffix', ['butterworth', 'savgol'])
def test_smoothed_template_actual_returns_and_tail_are_raw_and_nonexecutable(suffix, tmp_path):
    raw = get_template_v2(f'market-trend-smoothed-{suffix}-reference-v2')['definition']
    t = np.arange(900)
    prices = 100*(1+.3*np.sin(t/45))+.9*np.sin(t*.8)
    rows = [{'observation_date': (date(2020, 1, 1)+timedelta(days=int(i))).isoformat(),
             'available_at': (date(2020, 1, 2)+timedelta(days=int(i))).isoformat(), 'value': float(x)} for i,x in enumerate(prices)]
    raw['graph']['nodes'][0] = {'id': 'market', 'type': 'source.inline', 'parameters': {'rows': rows, 'frequency': 'daily'}}
    execution = RegimeGraphV2Service(tmp_path, tmp_path)._execute_graph(None, parse_definition_v2(raw), 'retrospective', None)
    outputs = execution['node_outputs']
    changes = outputs['change']['value'].values
    starts, ends = outputs['segments']['start'].values, outputs['segments']['end'].values
    valid = np.flatnonzero(np.isfinite(changes))
    assert len(valid) > 500
    np.testing.assert_allclose(changes[valid], prices[ends[valid]]/prices[starts[valid]]-1, atol=1e-14)
    assert any(row['state_id']=='bull' for row in execution['series'])
    assert any(row['state_id']=='bear' for row in execution['series'])
    assert execution['series'][-1]['state_id'] == 'unclassified'
    np.testing.assert_allclose([row['features']['index_value'] for row in execution['series']], prices)
    np.testing.assert_allclose(np.array([row['features']['filtered_index'] for row in execution['series']], dtype=float),
                               outputs['smooth']['value'].values, equal_nan=True)
    np.testing.assert_allclose(np.array([row['features']['phase_return'] for row in execution['series']], dtype=float),
                               changes, equal_nan=True)
    assert all(row['effective_date'] is None and not row['executable'] for row in execution['series'])


@pytest.mark.parametrize('identifier', list(SMOOTHING_OPERATORS))
@pytest.mark.parametrize('invalid', [None, True, 20.5, '20', -1, 5001, 10**400], ids=['null', 'bool', 'fraction', 'text', 'negative', 'too_large', 'huge_integer'])
def test_api_rejects_invalid_filter_parameters_without_server_error(identifier, invalid, client):
    raw = definition_for(identifier)
    parameter = 'window' if identifier.endswith('centered') else 'period'
    raw['graph']['nodes'][2]['parameters'] = {parameter: invalid}
    response = client.post('/api/historical-regimes/prepare', json={'definition': raw})
    assert response.status_code == 422, response.text
    assert response.json()['detail']['code'] == 'INVALID_REGIME_GRAPH_V2'


@pytest.mark.parametrize('parameters', [{'window': 20}, {'window': 3, 'polyorder': 3}, {'gain_limit': 101}])
def test_api_rejects_joint_sg_constraints_and_gain_limit(parameters, client):
    raw = definition_for('filter.ehlers_error_correcting' if 'gain_limit' in parameters else 'filter.savitzky_golay_centered')
    raw['graph']['nodes'][2]['parameters'] = parameters
    response = client.post('/api/historical-regimes/prepare', json={'definition': raw})
    assert response.status_code == 422, response.text
