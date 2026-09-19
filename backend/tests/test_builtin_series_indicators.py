"""Built-in series indicators: one definition each, rolling width adjustable."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from cal_indicators.typed_dsl import compose_typed_series_bundle, TypedDslError
from cal_indicators.typed_numba_plan import compile_numba_series_plan
from cal_indicators.typed_types import ValueType
from custom_indicators.service import _built_in_indicators
from custom_indicators.variable_registry import variable_types

DEFAULT_WINDOWS = {
    'builtin-close-moving-average-series': 20,
    'builtin-bollinger-bands-series': 20,
    'builtin-volume-moving-average-series': 10,
    'builtin-kdj-series': 9,
    'builtin-rolling-5d-annualized-sharpe-series': 5,
}
IDS = sorted(DEFAULT_WINDOWS)


@pytest.fixture(scope="module")
def definitions():
    return {item['id']: item for item in _built_in_indicators()
            if item.get('result_kind') == 'time_series'}


def compile_definition(definition):
    parameters = {str(item['id']): ValueType.scalar(semantic_dimension='count')
                  for item in definition['parameter_schema']}
    plan = compose_typed_series_bundle({item['id']: item['expression'] for item in definition['series_outputs']},
        variable_types={**variable_types('single_product', definition['dsl_version']), **parameters},
        parameter_names=frozenset(parameters),
        dsl_version=definition['dsl_version'], operator_registry_version=definition['operator_registry_version'])
    return plan, compile_numba_series_plan(plan)


def inputs(size=80, missing=False):
    x = np.arange(size, dtype=np.float64)
    close = 100 + x * .07 + np.sin(x / 3)
    nav = 1 + x * .001 + np.sin(x / 4) * .008
    factor = np.where(x < size // 2, 1., 1.05)
    data = {'market_close': close, 'market_high': close + .8, 'market_low': close - .9,
            'adjusted_close': close * factor, 'adjusted_high': (close + .8) * factor,
            'adjusted_low': (close - .9) * factor, 'adjusted_open': (close - .5) * factor,
            'volume': 1000 + x * 3 + np.cos(x) * 100, 'adjusted_nav': nav,
            'returns': np.r_[np.nan, nav[1:] / nav[:-1] - 1],
            'observation_dates': 20000 + x, 'annual_risk_free_rate_decimal': .015,
            'risk_free_rate_per_observation': 1.015 ** (1 / 252) - 1, 'periods_per_year': 252.,
            'observation_count': float(size - 1), 'window_elapsed_days': float(size - 1),
            'risk_free_return_window': 1.015 ** ((size - 1) / 365) - 1}
    if missing:
        for key in ('market_close', 'market_high', 'market_low', 'adjusted_close',
                    'adjusted_high', 'adjusted_low', 'volume', 'returns'):
            data[key][6] = np.nan
            data[key][12] = np.inf
        data['market_low'][25:36] = np.nan
        data['market_high'][40:51] = np.nan
        data['adjusted_low'][25:36] = np.nan
        data['adjusted_high'][40:51] = np.nan
    return data


def compute(compiled, context, window=None):
    """Run one plan; ``window`` overrides whatever width the context carries."""

    if window is not None:
        context = {**context, 'window': float(window)}
    return compiled.compute(tuple(context[name] for name in compiled.context_names))


def reference_window(values, window, reduce):
    """A window yields a value only when all of its observations are finite."""

    output = np.full(len(values), np.nan)
    for index in range(window - 1, len(values)):
        chunk = values[index - window + 1:index + 1]
        if np.all(np.isfinite(chunk)):
            output[index] = reduce(chunk)
    return output


def reference_kdj(context, window):
    low, high, close = (context[key] for key in ('adjusted_low', 'adjusted_high', 'adjusted_close'))
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    k_state = d_state = 50.
    for index in range(len(close)):
        start = max(0, index - window + 1)
        low_window, high_window = low[start:index + 1], high[start:index + 1]
        lows, highs = low_window[np.isfinite(low_window)], high_window[np.isfinite(high_window)]
        if not lows.size or not highs.size or not np.isfinite(close[index]):
            continue
        denominator = highs.max() - lows.min()
        rsv = 50. if abs(denominator) < 1e-12 else (close[index] - lows.min()) * 100 / denominator
        k_state = (2 * k_state + rsv) / 3
        d_state = (2 * d_state + k_state) / 3
        k[index], d[index] = k_state, d_state
    return k, d, 3 * k - 2 * d


def reference_channels(indicator_id, context, window):
    """Independent NumPy answers for every channel of one built-in."""

    if indicator_id == 'builtin-close-moving-average-series':
        return [reference_window(context['adjusted_close'], window, np.mean)]
    if indicator_id == 'builtin-volume-moving-average-series':
        return [reference_window(context['volume'], window, np.mean)]
    if indicator_id == 'builtin-bollinger-bands-series':
        middle = reference_window(context['adjusted_close'], window, np.mean)
        deviation = reference_window(context['adjusted_close'], window, lambda chunk: np.std(chunk, ddof=0))
        return [middle + 2 * deviation, middle, middle - 2 * deviation]
    if indicator_id == 'builtin-kdj-series':
        return list(reference_kdj(context, window))
    annualization = np.sqrt(context['periods_per_year'])
    excess = context['risk_free_rate_per_observation']
    return [reference_window(context['returns'], window,
                             lambda chunk: (chunk.mean() - excess) / np.std(chunk, ddof=1) * annualization)]


@pytest.mark.parametrize('indicator_id', IDS)
def test_catalog_carries_one_revision_that_opens_only_its_rolling_width(definitions, indicator_id):
    definition = definitions[indicator_id]
    assert definition['revision'] == 1
    assert definition['source'] == 'built_in' and definition['read_only']
    assert definition['name'].startswith('N 日')
    assert definition['parameter_contract_version'] == '1.0'
    assert [(item['id'], item['type'], item['default'], item['minimum'], item['maximum'], item['step'])
            for item in definition['parameter_schema']] == [
        ('window', 'integer', DEFAULT_WINDOWS[indicator_id], 2, 1000, 1)]
    plan, compiled = compile_definition(definition)
    operations = {node.operator_id for node in plan.nodes}
    assert 'rolling_apply' in operations
    assert not operations.intersection({'rolling_window', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'})
    for channel in definition['series_outputs']:
        assert 'window' in channel['expression']
    delegates = [value for value in compiled.dispatcher.py_func.__globals__.values() if hasattr(value, 'interval_source')]
    assert delegates
    assert all('interval_body(' in kernel.rolling_source and not kernel._can_compile for kernel in delegates)


@pytest.mark.parametrize('indicator_id', IDS)
@pytest.mark.parametrize('missing', [False, True])
def test_every_channel_matches_an_independent_reference_at_several_widths(definitions, indicator_id, missing):
    context = inputs(missing=missing)
    _, compiled = compile_definition(definitions[indicator_id])
    before = {key: value.copy() for key, value in context.items() if isinstance(value, np.ndarray)}
    for window in (2, DEFAULT_WINDOWS[indicator_id], 25):
        actual = compute(compiled, context, window)
        expected = reference_channels(indicator_id, context, window)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True)
    for key, value in before.items():
        np.testing.assert_array_equal(context[key], value)


@pytest.mark.parametrize('case', ['normal', 'missing', 'flat', 'zero', 'unbounded_j', 'short'])
def test_kdj_short_windows_missing_and_recursive_state_match_independent_reference(definitions, case):
    context = inputs(size=7 if case == 'short' else 80, missing=case == 'missing')
    if case in {'flat', 'zero'}:
        for key in ('adjusted_close', 'adjusted_high', 'adjusted_low'):
            context[key][:] = 100. if case == 'flat' else 0.
    if case == 'unbounded_j':
        context['adjusted_close'][30:40] *= 10
    _, compiled = compile_definition(definitions['builtin-kdj-series'])
    actual = compute(compiled, context, 9)
    for left, right in zip(actual, reference_kdj(context, 9)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(actual[0][0])
    if case == 'unbounded_j':
        assert np.nanmax(actual[2]) > 100


@pytest.mark.parametrize('indicator_id,scopes,smoothing', [
    ('builtin-bollinger-bands-series', 2, 0), ('builtin-kdj-series', 2, 2),
])
def test_multi_channel_shared_subgraphs_are_not_repeated(definitions, indicator_id, scopes, smoothing):
    plan, compiled = compile_definition(definitions[indicator_id])
    counts = Counter(node.operator_id for node in plan.nodes)
    assert counts['rolling_apply'] == scopes
    assert counts['recursive_smooth'] == smoothing
    for node in plan.nodes:
        if node.operator_id in {'rolling_apply', 'recursive_smooth'}:
            assert compiled.source.count(f'    n{node.node_id} = ') == 1


def test_partial_window_minimum_and_mask_are_general_not_kdj_specific():
    plan = compose_typed_series_bundle({'value': 'rolling_apply(mean_where(market_close,finite_mask(market_close)),width,minimum)'},
        variable_types={**variable_types('single_product', '2.4.0'), 'width': ValueType.scalar(semantic_dimension='count'),
                        'minimum': ValueType.scalar(semantic_dimension='count')},
        parameter_names=frozenset({'width', 'minimum'}))
    compiled = compile_numba_series_plan(plan)
    context = inputs(size=8)
    context.update(width=3., minimum=1., market_close=np.array([1., np.nan, 3., np.inf, 5., 0., 7., 8.]))
    signatures = tuple(compiled.dispatcher.signatures)
    expected = [1., 1., 2., 3., 4., 2.5, 4., 5.]
    np.testing.assert_allclose(compiled.compute(tuple(context[name] for name in compiled.context_names))[0], expected)
    context['minimum'] = 2.
    np.testing.assert_allclose(compiled.compute(tuple(context[name] for name in compiled.context_names))[0],
                               [np.nan, np.nan, 2., np.nan, 4., 2.5, 4., 5.], equal_nan=True)
    assert signatures == tuple(compiled.dispatcher.signatures)
    context['minimum'] = 4.
    with pytest.raises(ValueError, match='MIN_PERIODS'):
        compiled.compute(tuple(context[name] for name in compiled.context_names))


def test_partial_minimum_cannot_exceed_window_at_parse_time():
    with pytest.raises(TypedDslError, match='最少有效观察数'):
        compose_typed_series_bundle({'value': 'rolling_apply(mean(market_close),3,4)'})


@pytest.fixture(scope='module')
def service(tmp_path_factory):
    from custom_indicators.service import CustomIndicatorService
    from test_custom_indicator_time_series import _write_market_data
    from cal_indicators.typed_numba_kernels import warm_numba_kernel_registry
    root = tmp_path_factory.mktemp('builtin-series')
    _write_market_data(root)
    service = CustomIndicatorService(root, root)
    warm_numba_kernel_registry()
    versions = [item for item in service.indicators.list_all_versions() if item.get('result_kind') == 'time_series']
    assert len(versions) == len(IDS)
    for definition in versions:
        service.series_service.warm(definition)
    yield service
    service.close_compute_engine()


@pytest.mark.parametrize('indicator_id', IDS)
def test_one_warmed_plan_serves_every_width(service, indicator_id, monkeypatch):
    from custom_indicators import series_service
    default = DEFAULT_WINDOWS[indicator_id]
    monkeypatch.setattr(series_service, '_compile_definition', lambda *a, **kw: pytest.fail('request-time compilation'))
    response = service.evaluate_series(
        indicator_instances=[{'indicator_id': indicator_id},
                             {'indicator_id': indicator_id, 'parameters': {'window': default}},
                             {'indicator_id': indicator_id, 'parameters': {'window': default * 2}}],
        target={'kind': 'etf', 'product_id': '510300.SH'}, period='ALL')
    assert response['summary']['ok'] == 3, response
    assert len(response['execution']['compiled_plan_ids']) == 1
    assert response['execution']['request_time_compilation'] == 0
    assert response['execution']['python_fallback'] == 0
    implicit, explicit, widened = response['results']
    # An omitted parameter and the same value spelled out are one calculation.
    for left, right in zip(implicit['channels'], explicit['channels']):
        assert left['values'] == right['values']
    # KDJ carries recursive state, so it loads full history and reports no lookback.
    if implicit['history_policy'] == 'lookback':
        assert implicit['lookback_observations'] == default
        assert widened['lookback_observations'] == default * 2
    assert any(left['values'] != right['values']
               for left, right in zip(implicit['channels'], widened['channels']))


def test_window_outside_the_declared_range_is_rejected(service):
    from custom_indicators.errors import ValidationError
    for window in (1, 1001, 20.5):
        with pytest.raises(ValidationError):
            service.evaluate_series(
                indicator_instances=[{'indicator_id': 'builtin-close-moving-average-series',
                                      'parameters': {'window': window}}],
                target={'kind': 'etf', 'product_id': '510300.SH'}, period='ALL')


def test_opening_a_further_input_still_produces_a_separate_user_indicator(service):
    from custom_indicators.series_parameters import inspect_parameter_inputs, bind_parameter_input
    builtin = service.get_indicator('builtin-kdj-series')
    candidate = next(item for item in inspect_parameter_inputs(builtin)['candidates']
                     if item['operator_id'] == 'recursive_smooth' and item['argument'] == 'periods')
    opened = bind_parameter_input(builtin, candidate_id=candidate['id'])
    opened['name'] = '自定义可调平滑 KDJ'
    saved = service.create_indicator(opened)
    assert len(saved['parameter_schema']) == 2
    assert [item['id'] for item in service.get_indicator(builtin['id'])['parameter_schema']] == ['window']


@pytest.mark.parametrize('indicator_id', IDS)
def test_builtin_graph_and_native_excel_are_resolvable(service, monkeypatch, indicator_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services import custom_indicator_routes
    import xml.etree.ElementTree as ET
    import zipfile
    monkeypatch.setattr(custom_indicator_routes, 'indicator_service', service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)
    definition = service.get_indicator(indicator_id)
    context = {'result_kind': 'time_series', 'context_kind': 'single_product',
               'dsl_version': '2.4.0', 'operator_registry_version': '2.4.0',
               'parameter_contract_version': definition['parameter_contract_version'],
               'parameter_schema': definition['parameter_schema']}
    graph = client.post('/api/custom-indicators/graph/resolve', json={**context, 'source_kind': 'formula',
        'expressions': {item['id']: item['expression'] for item in definition['series_outputs']}}).json()
    assert graph['valid'], graph
    recovered = client.post('/api/custom-indicators/graph/resolve', json={**context, 'source_kind': 'graph', 'graph': graph['graph']}).json()
    assert recovered['valid'] and recovered['expressions'] == graph['expressions'], recovered
    artifact = service.export_excel(indicator_ids=[indicator_id], inline_definition=None,
        targets=[{'kind': 'etf', 'product_id': '510300.SH'}], period='1W')
    try:
        with zipfile.ZipFile(artifact.path) as archive:
            assert archive.testzip() is None
            namespace = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            formulas = [node.text or '' for name in archive.namelist() if name.startswith('xl/worksheets/sheet')
                        for node in ET.fromstring(archive.read(name)).findall('.//s:f', namespace)]
            assert formulas
            assert not any('rolling_apply(' in formula or '_xlfn.' in formula for formula in formulas)
            if indicator_id == 'builtin-kdj-series':
                assert any('SUMPRODUCT(' in formula and '>=1' in formula for formula in formulas)
                assert any('ISNUMBER(' in formula for formula in formulas)
    finally:
        artifact.cleanup()


def test_partial_window_is_causal_and_preserves_unknown_future():
    from causality import audit_expression, Verdict
    report = audit_expression('rolling_apply(min_where(market_low,finite_mask(market_low)),9,1)')
    assert report.verdict is Verdict.CAUSAL, report.as_dict()
