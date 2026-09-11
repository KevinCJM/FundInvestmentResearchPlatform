"""v3 built-ins use whole-interval scopes without rewriting historical contracts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pytest

from cal_indicators.typed_dsl import compose_typed_series_bundle, TypedDslError
from cal_indicators.typed_numba_plan import compile_numba_series_plan
from cal_indicators.typed_types import ValueType
from custom_indicators.service import _built_in_indicators
from custom_indicators.variable_registry import variable_types

HISTORY_HASHES = {
    "builtin-close-moving-average-series@1": "50cb603d578462a7571a516e0fc8367da9201773076cd9e26185a61cc26ca879",
    "builtin-bollinger-bands-series@1": "4e1201ca225d24b49274b1cb9e4f98d7531cf462eb73e22038c786db75ddc3c3",
    "builtin-volume-moving-average-series@1": "7e075080db9656a66a141d0cb05ec44d76434b80a635c9ce5b8f17bc52904a33",
    "builtin-kdj-series@1": "112d9aaa1a34c6da2677f9a74b141bfc358a5e0f64c273e871ff1b2bdfe44511",
    "builtin-rolling-5d-annualized-sharpe-series@1": "6da9b45799994b4393c3ec7c65a8c7235fdbd35c707206b2815294c667848a0d",
    "builtin-close-moving-average-series@2": "05ae14e1a3e5175c68da70f344f8763e6499d9c7c48fb1d5a26e5100caa29cd1",
    "builtin-bollinger-bands-series@2": "5035e71938abdc622127c0865a98855d502677a7b2868b8bd9197b9e8738fc69",
    "builtin-volume-moving-average-series@2": "c0edf7a389d79df41eddbcc46015d9e9f8a89f86baa0cbc040e61b7e722273b2",
    "builtin-kdj-series@2": "695482834b5b00d1c8f9d5f0c5f60a89634d84df8a69acdb5006870f6c223d7c",
    "builtin-rolling-5d-annualized-sharpe-series@2": "02d52300bbb9c2b8529c096a089f96e227c0a8cd9bc68fd65caafb8f5a1d70c4",
}
IDS = sorted({key.split('@')[0] for key in HISTORY_HASHES})


@pytest.fixture(scope="module")
def definitions():
    return {(item['id'], item['revision']): item for item in _built_in_indicators()
            if item.get('result_kind') == 'time_series'}


def compile_definition(definition):
    plan = compose_typed_series_bundle({item['id']: item['expression'] for item in definition['series_outputs']},
        variable_types=variable_types('single_product', definition['dsl_version']),
        dsl_version=definition['dsl_version'], operator_registry_version=definition['operator_registry_version'])
    return plan, compile_numba_series_plan(plan)


def inputs(size=80, missing=False):
    x = np.arange(size, dtype=np.float64)
    close = 100 + x * .07 + np.sin(x / 3)
    nav = 1 + x * .001 + np.sin(x / 4) * .008
    data = {'market_close': close, 'market_high': close + .8, 'market_low': close - .9,
            'volume': 1000 + x * 3 + np.cos(x) * 100, 'adjusted_nav': nav,
            'returns': np.r_[np.nan, nav[1:] / nav[:-1] - 1],
            'observation_dates': 20000 + x, 'annual_risk_free_rate_decimal': .015,
            'risk_free_rate_per_observation': 1.015 ** (1 / 252) - 1, 'periods_per_year': 252.,
            'observation_count': float(size - 1), 'window_elapsed_days': float(size - 1),
            'risk_free_return_window': 1.015 ** ((size - 1) / 365) - 1}
    if missing:
        for key in ('market_close', 'market_high', 'market_low', 'volume', 'returns'):
            data[key][6] = np.nan
            data[key][12] = np.inf
        data['market_low'][25:36] = np.nan
        data['market_high'][40:51] = np.nan
    return data


def compute(compiled, context):
    return compiled.compute(tuple(context[name] for name in compiled.context_names))


def test_all_ten_historical_definition_hashes_are_unchanged(definitions):
    for key, expected in HISTORY_HASHES.items():
        name, revision = key.split('@')
        value = {k: v for k, v in definitions[name, int(revision)].items() if k not in {'created_at', 'updated_at'}}
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                          separators=(',', ':'), default=str).encode()).hexdigest()
        assert digest == expected, key


@pytest.mark.parametrize('indicator_id', IDS)
def test_current_definition_and_executable_plan_use_generic_scope(definitions, indicator_id):
    definition = definitions[indicator_id, 3]
    plan, compiled = compile_definition(definition)
    operations = {node.operator_id for node in plan.nodes}
    assert 'rolling_apply' in operations
    assert not operations.intersection({'rolling_window', 'rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max'})
    assert definition['parameter_schema'] == []
    previous = definitions[indicator_id, 2]
    assert [(x['id'], x['label'], x['precision'], x['display_format']) for x in definition['series_outputs']] == [
        (x['id'], x['label'], x['precision'], x['display_format']) for x in previous['series_outputs']]
    delegates = [value for value in compiled.dispatcher.py_func.__globals__.values() if hasattr(value, 'interval_source')]
    assert delegates
    assert all('interval_body(' in kernel.rolling_source and not kernel._can_compile for kernel in delegates)
    if 'sharpe' in indicator_id:
        assert definition['rolling_source']['transform_version'] == '3.0.0'
        assert definition['rolling_source']['definition_hash'] == previous['rolling_source']['definition_hash']


@pytest.mark.parametrize('indicator_id', IDS)
@pytest.mark.parametrize('missing', [False, True])
def test_v2_v3_every_channel_matches_including_null_positions(definitions, indicator_id, missing):
    context = inputs(missing=missing)
    _, previous = compile_definition(definitions[indicator_id, 2])
    _, current = compile_definition(definitions[indicator_id, 3])
    before = {key: value.copy() for key, value in context.items() if isinstance(value, np.ndarray)}
    for old, new in zip(compute(previous, context), compute(current, context)):
        np.testing.assert_array_equal(np.isnan(old), np.isnan(new))
        np.testing.assert_allclose(new, old, rtol=1e-8, atol=1e-9, equal_nan=True)
    for key, value in before.items():
        np.testing.assert_array_equal(context[key], value)


def reference_kdj(context):
    low, high, close = (context[key] for key in ('market_low', 'market_high', 'market_close'))
    k = np.full(len(close), np.nan)
    d = np.full(len(close), np.nan)
    k_state = d_state = 50.
    for index in range(len(close)):
        low_window, high_window = low[max(0, index - 8):index + 1], high[max(0, index - 8):index + 1]
        lows, highs = low_window[np.isfinite(low_window)], high_window[np.isfinite(high_window)]
        if not lows.size or not highs.size or not np.isfinite(close[index]):
            continue
        denominator = highs.max() - lows.min()
        rsv = 50. if abs(denominator) < 1e-12 else (close[index] - lows.min()) * 100 / denominator
        k_state = (2 * k_state + rsv) / 3
        d_state = (2 * d_state + k_state) / 3
        k[index], d[index] = k_state, d_state
    return k, d, 3 * k - 2 * d


@pytest.mark.parametrize('case', ['normal', 'missing', 'flat', 'zero', 'unbounded_j', 'short'])
def test_kdj_short_windows_missing_and_recursive_state_match_independent_reference(definitions, case):
    context = inputs(size=7 if case == 'short' else 80, missing=case == 'missing')
    if case in {'flat', 'zero'}:
        for key in ('market_close', 'market_high', 'market_low'):
            context[key][:] = 100. if case == 'flat' else 0.
    if case == 'unbounded_j':
        context['market_close'][30:40] *= 10
    _, compiled = compile_definition(definitions['builtin-kdj-series', 3])
    actual = compute(compiled, context)
    for left, right in zip(actual, reference_kdj(context)):
        np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(actual[0][0])
    if case == 'unbounded_j':
        assert np.nanmax(actual[2]) > 100


@pytest.mark.parametrize('indicator_id,scopes,smoothing', [
    ('builtin-bollinger-bands-series', 2, 0), ('builtin-kdj-series', 2, 2),
])
def test_multi_channel_shared_subgraphs_are_not_repeated(definitions, indicator_id, scopes, smoothing):
    plan, compiled = compile_definition(definitions[indicator_id, 3])
    counts = Counter(node.operator_id for node in plan.nodes)
    assert counts['rolling_apply'] == scopes
    assert counts['recursive_smooth'] == smoothing
    for node in plan.nodes:
        if node.operator_id in {'rolling_apply', 'recursive_smooth'}:
            assert compiled.source.count(f'    n{node.node_id} = ') == 1


def test_partial_window_minimum_and_mask_are_general_not_kdj_specific():
    plan = compose_typed_series_bundle({'value': 'rolling_apply(mean_where(market_close,finite_mask(market_close)),width,minimum)'},
        variable_types={**variable_types('single_product', '2.4.0'), 'width': ValueType.scalar(semantic_dimension='count'),
                        'minimum': ValueType.scalar(semantic_dimension='count')})
    compiled = compile_numba_series_plan(plan)
    context = inputs(size=8)
    context.update(width=3., minimum=1., market_close=np.array([1., np.nan, 3., np.inf, 5., 0., 7., 8.]))
    signatures = tuple(compiled.dispatcher.signatures)
    expected = [1., 1., 2., 3., 4., 2.5, 4., 5.]
    np.testing.assert_allclose(compute(compiled, context)[0], expected)
    context['minimum'] = 2.
    np.testing.assert_allclose(compute(compiled, context)[0], [np.nan, np.nan, 2., np.nan, 4., 2.5, 4., 5.], equal_nan=True)
    assert signatures == tuple(compiled.dispatcher.signatures)
    context['minimum'] = 4.
    with pytest.raises(ValueError, match='MIN_PERIODS'):
        compute(compiled, context)


def test_partial_minimum_cannot_exceed_window_at_parse_time():
    with pytest.raises(TypedDslError, match='最少有效观察数'):
        compose_typed_series_bundle({'value': 'rolling_apply(mean(market_close),3,4)'})


@pytest.fixture(scope='module')
def migration_service(tmp_path_factory):
    from custom_indicators.service import CustomIndicatorService
    from test_custom_indicator_time_series import _write_market_data
    from cal_indicators.typed_numba_kernels import warm_numba_kernel_registry
    root = tmp_path_factory.mktemp('builtin-series-migration')
    _write_market_data(root)
    service = CustomIndicatorService(root, root)
    warm_numba_kernel_registry()
    versions = [item for item in service.indicators.list_all_versions() if item.get('result_kind') == 'time_series']
    assert len(versions) == 15
    for definition in versions:
        service.series_service.warm(definition)
    yield service
    service.close_compute_engine()


def test_repository_and_actual_api_execution_use_v3_but_allow_v2(migration_service, monkeypatch):
    from custom_indicators import series_service
    service = migration_service
    current = [item for item in service.list_indicators()['items'] if item.get('result_kind') == 'time_series']
    assert len(current) == 5 and {item['revision'] for item in current} == {3}
    monkeypatch.setattr(series_service, '_compile_definition', lambda *a, **kw: pytest.fail('request-time compilation'))
    instances = [{'indicator_id': item['id'], 'indicator_revision': revision} for item in current for revision in (2, 3)]
    response = service.evaluate_series(indicator_instances=instances, target={'kind': 'etf', 'product_id': '510300.SH'}, period='ALL')
    assert response['summary']['ok'] == 10, response
    assert response['execution']['python_fallback'] == 0
    assert response['execution']['request_time_compilation'] == 0
    for old, new in zip(response['results'][::2], response['results'][1::2]):
        assert old['indicator_revision'] == 2 and new['indicator_revision'] == 3
        for left, right in zip(old['channels'], new['channels']):
            np.testing.assert_allclose(np.asarray(left['values'], dtype=float), np.asarray(right['values'], dtype=float),
                                       equal_nan=True, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('indicator_id', IDS)
def test_current_builtin_graph_and_native_excel_are_resolvable(migration_service, monkeypatch, indicator_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services import custom_indicator_routes
    import xml.etree.ElementTree as ET
    import zipfile
    service = migration_service
    monkeypatch.setattr(custom_indicator_routes, 'indicator_service', service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    client = TestClient(app)
    definition = service.get_indicator(indicator_id)
    context = {'result_kind': 'time_series', 'context_kind': 'single_product',
               'dsl_version': '2.4.0', 'operator_registry_version': '2.4.0'}
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


def test_new_scope_parameters_remain_author_controlled(migration_service):
    from custom_indicators.series_parameters import inspect_parameter_inputs, bind_parameter_input
    service = migration_service
    builtin = service.get_indicator('builtin-close-moving-average-series')
    assert not builtin['parameter_schema']
    candidate = next(item for item in inspect_parameter_inputs(builtin)['candidates'] if item['operator_id'] == 'rolling_apply')
    opened = bind_parameter_input(builtin, candidate_id=candidate['id'])
    opened['name'] = '迁移后自定义可调均线'
    saved = service.create_indicator(opened)
    parameter = saved['parameter_schema'][0]['id']
    results = service.evaluate_series(indicator_instances=[{'indicator_id': saved['id']},
        {'indicator_id': saved['id'], 'parameters': {parameter: 10}}],
        target={'kind': 'etf', 'product_id': '510300.SH'}, period='ALL')
    assert results['summary']['ok'] == 2
    assert [item['lookback_observations'] for item in results['results']] == [20, 10]
    assert len(results['execution']['compiled_plan_ids']) == 1
    assert service.get_indicator(builtin['id'])['parameter_schema'] == []


@pytest.mark.parametrize('indicator_id', IDS)
def test_explicit_v1_reference_remains_executable_after_v3_migration(migration_service, indicator_id):
    response = migration_service.evaluate_series(
        indicator_instances=[{'indicator_id': indicator_id, 'indicator_revision': revision} for revision in (1, 3)],
        target={'kind': 'etf', 'product_id': '510300.SH'}, period='ALL',
    )
    assert response['summary']['ok'] == 2, response
    historical, current = response['results']
    assert historical['indicator_revision'] == 1
    assert current['indicator_revision'] == 3
    for old, new in zip(historical['channels'], current['channels']):
        np.testing.assert_allclose(np.asarray(old['values'], dtype=float), np.asarray(new['values'], dtype=float),
                                   equal_nan=True, rtol=1e-8, atol=1e-9)
    assert response['execution']['request_time_compilation'] == 0


def test_partial_window_is_causal_and_preserves_unknown_future():
    from causality import audit_expression, Verdict
    report = audit_expression('rolling_apply(min_where(market_low,finite_mask(market_low)),9,1)')
    assert report.verdict is Verdict.CAUSAL, report.as_dict()
