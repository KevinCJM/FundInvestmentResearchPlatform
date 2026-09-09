"""ETF/fund source identity, PIT dates and the actual warmed preview path."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from custom_indicators.errors import ValidationError
from historical_regimes.data import resolve_target
from historical_regimes.node_preview import node_preview_definition
from historical_regimes.numba_kernels import warm_historical_regime_numba_kernels
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_registry import node_catalog
from research_series.numba_kernels import warm_research_series_numba_kernels
from research_series.product_sources import PRODUCT_SOURCES
from research_series.service import ResearchSeriesService
from test_regime_series_builder import graph


@pytest.fixture(scope="module", autouse=True)
def warm():
    warm_historical_regime_numba_kernels()
    warm_research_series_numba_kernels()


@pytest.fixture
def market(tmp_path):
    root = tmp_path / "data"
    snap = root / "snapshot-products"
    snap.mkdir(parents=True)
    pd.DataFrame({"ts_code": ["510300.SH", "999999.SH"], "name": ["沪深300ETF", "未下载ETF"]}).to_parquet(snap / "etf_info_df.parquet")
    pd.DataFrame({"ts_code": ["000001.OF", "000003.OF"], "name": ["华夏成长", "其他基金"]}).to_parquet(snap / "fund_info_df.parquet")
    pd.DataFrame({"ts_code": ["510300.SH"] * 4, "trade_date": ["20240102", "20240103", "20240104", "20240105"],
                  "close": [3., 3.3, np.nan, 3.6], "open": [2.9, 3.2, 3.1, 3.5]}).to_parquet(snap / "etf_daily_candle_df.parquet")
    pd.DataFrame({"ts_code": ["000001.OF"] * 5, "nav_date": ["20240102", "20240103", "20240104", "20240105", "20240102"],
                  "ann_date": ["20240103", "20240104", "20240105", "20240108", "20240109"],
                  "unit_nav": [1., 1.1, np.nan, 1.2, 9.], "accum_nav": [2., 2.1, np.nan, 2.2, 9.], "adj_nav": [3., 3.1, np.nan, 3.2, 9.]}).to_parquet(snap / "fund_nav_df.parquet")
    manifest = {"schema_version": 1, "snapshot_dir": snap.name, "files": {p.name: p.stat().st_size for p in snap.iterdir()}}
    (root / "tushare_active.json").write_text(json.dumps(manifest))
    return root, snap, ResearchSeriesService(root)


@pytest.mark.parametrize("kind,query,code,default", [("etf", "沪深300", "510300.SH", "close"), ("fund", "华夏成长", "000001.OF", "unit_nav")])
def test_catalog_search_binding_fields_and_registry(market, kind, query, code, default):
    root, snap, catalog = market
    result = catalog.catalog(kind=kind, query=query)
    assert result["total"] == 1
    item = result["items"][0]
    assert item["regime_node_type"] == f"source.{kind}"
    assert item["binding_parameters"]["ts_code"] == code
    assert item["default_field"] == default
    assert item["binding_parameters"]["file_checksum"].startswith("sha256:")
    assert catalog.catalog(kind=kind, query=code.lower())["items"] == result["items"]
    assert catalog.catalog(kind=kind, status="not_downloaded")["total"] == 1
    assert catalog.catalog(kind=kind, limit=1, offset=1)["total"] == 2
    assert len(catalog.catalog(kind=kind, limit=1, offset=1)["items"]) == 1
    metadata = next(n for n in node_catalog()["items"] if n["id"] == f"source.{kind}")
    assert not metadata.get("authoring_hidden") and metadata["category"] == "source"
    assert metadata["parameter_schema"]["properties"]["field"]["option_source"] == "research_series.fields"


def test_unpublished_files_are_not_offered(market):
    root, snap, service = market
    manifest = json.loads((root / "tushare_active.json").read_text())
    del manifest["files"]["fund_nav_df.parquet"]
    (root / "tushare_active.json").write_text(json.dumps(manifest))
    assert service.catalog(kind="fund", status="available")["items"] == []
    assert all(i["binding_parameters"] == {} for i in service.catalog(kind="fund")["items"])


def test_fund_nav_uses_announcement_cutoff_and_first_release(market):
    root, _, catalog = market
    spec = {"kind": "fund", "ts_code": "000001.OF"}
    early = resolve_target(spec, "realtime", "2024-01-04", root)
    assert early.frame["value"].tolist() == [1., 1.1]
    assert early.frame["available_at"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04"]
    latest = resolve_target(spec, "retrospective", "2024-01-09", root)
    assert latest.frame["value"].iloc[0] == 9.
    realtime = resolve_target(spec, "realtime", "2024-01-09", root)
    assert realtime.frame["value"].iloc[0] == 1.
    with pytest.raises(ValidationError, match="复权净值"):
        resolve_target({**spec, "field": "adj_nav"}, "realtime", None, root)
    assert resolve_target({**spec, "field": "adj_nav"}, "retrospective", None, root).frame["value"].iloc[1] == 3.1
    profile = catalog.profile(series_id="fund:fund_nav:000001.OF", as_of="2024-01-04")
    assert profile["values"]["raw"] == [1., 1.1]
    assert profile["pit"]["available_at"] == ["2024-01-03", "2024-01-04"]
    assert profile["binding"]["node_type"] == "source.fund"
    assert profile["execution"]["python_fallback"] == 0


def test_missing_release_dates_fail_realtime_and_fields_fail_closed(market):
    root, snap, _ = market
    path = snap / "fund_nav_df.parquet"
    frame = pd.read_parquet(path); frame.loc[0, "ann_date"] = None; frame.to_parquet(path)
    spec = {"kind": "fund", "ts_code": "000001.OF"}
    with pytest.raises(ValidationError, match="公告日期"):
        resolve_target(spec, "realtime", None, root)
    assert not resolve_target(spec, "retrospective", "2024-01-04", root).snapshot["pit"]["supported"]
    with pytest.raises(ValidationError, match="数值字段"):
        resolve_target({"kind": "etf", "ts_code": "510300.SH", "field": "ann_date"}, "realtime", None, root)
    with pytest.raises(ValidationError, match="行情来源"):
        resolve_target({"kind": "etf", "ts_code": "510300.SH", "source_api": "fund_nav"}, "realtime", None, root)


@pytest.mark.parametrize("kind", ["etf", "fund"])
def test_single_node_and_downstream_preview_snapshot_checks(market, tmp_path, kind, monkeypatch):
    root, snap, catalog = market
    service = RegimeGraphV2Service(tmp_path / "workspace", root)
    item = catalog.catalog(kind=kind, status="available")["items"][0]
    raw = graph(parameters={"window": 2, "min_periods": 2})
    raw["graph"]["nodes"][0] = {"id": "price", "type": f"source.{kind}", "label": item["name"], "parameters": item["binding_parameters"]}
    raw["graph"]["outputs"] = {}
    single = node_preview_definition(raw, {"node_id": "price", "port": "value"})
    plan = service.prepare(raw, preview_target={"node_id": "price", "port": "value"})
    result = service._execute_graph(None, single, "realtime", None, plan=plan)
    expected = np.array([3., 3.3, np.nan, 3.6] if kind == "etf" else [1., 1.1, np.nan, 1.2])
    np.testing.assert_allclose(result["node_outputs"]["price"]["value"].values, expected, equal_nan=True)
    target = {"node_id": "smooth", "port": "value"}
    downstream = node_preview_definition(raw, target)
    plan = service.prepare(raw, preview_target=target)
    monkeypatch.setattr(service, "_prepare_formula_nodes", lambda *_: pytest.fail("cannot compile in calculation"))
    result = service._execute_graph(None, downstream, "realtime", None, plan=plan)
    np.testing.assert_allclose(result["node_outputs"]["smooth"]["value"].values, pd.Series(expected).rolling(2).mean(), equal_nan=True)
    assert result["result"]["diagnostics"]["python_fallback"] == 0
    assert result["result"]["diagnostics"]["request_time_compilation"] == 0
    assert raw["graph"]["outputs"] == {}
    parameters = copy.deepcopy(item["binding_parameters"])
    parameters["source_file"] = "etf_daily_df.parquet"
    with pytest.raises(ValidationError, match="文件"):
        service._bound_source_root(f"source.{kind}", parameters)
    source = snap / item["dataset"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValidationError, match="校验"):
        service._bound_source_root(f"source.{kind}", item["binding_parameters"])


def test_research_catalog_route_accepts_both_kinds(market, monkeypatch):
    from services import research_series_routes as routes
    monkeypatch.setattr(routes, "research_series_service", market[2])
    app = FastAPI(); app.include_router(routes.router)
    with TestClient(app) as client:
        for kind in ("etf", "fund"):
            response = client.get("/api/research-series/catalog", params={"kind": kind, "status": "available"})
            assert response.status_code == 200, response.text
            assert response.json()["total"] == 1


@pytest.mark.parametrize('mode', ['realtime', 'retrospective'])
def test_two_etfs_subtract_on_common_dates_with_explicit_alignment(market, tmp_path, mode):
    from historical_regimes.v2_contracts import validate_definition_v2
    from historical_regimes.v2_numba import regime_graph_numba_status

    root, snap, catalog = market
    left_dates = pd.date_range('2024-01-01', periods=9)
    right_dates = pd.date_range('2024-01-03', periods=9)
    left_values = np.arange(100., 109.)
    left_values[4] = np.nan
    frame = pd.concat([
        pd.DataFrame({'ts_code': '513500.SH', 'trade_date': left_dates.strftime('%Y%m%d'), 'close': left_values, 'available_at': left_dates}),
        pd.DataFrame({'ts_code': '510300.SH', 'trade_date': right_dates.strftime('%Y%m%d'), 'close': np.arange(10., 19.), 'available_at': right_dates + pd.Timedelta(days=1)}),
    ], ignore_index=True)
    frame.to_parquet(snap / 'etf_daily_candle_df.parquet')
    pd.DataFrame({'ts_code': ['513500.SH', '510300.SH'], 'name': ['标普500ETF', '沪深300ETF']}).to_parquet(snap / 'etf_info_df.parquet')
    nodes = [{'id': key, 'type': 'source.etf', 'parameters': catalog.catalog(kind='etf', query=code)['items'][0]['binding_parameters']}
             for key, code in [('sp500', '513500.SH'), ('hs300', '510300.SH')]]
    inputs = {'left': {'node_id': 'sp500', 'port': 'value'}, 'right': {'node_id': 'hs300', 'port': 'value'}}
    nodes.append({'id': 'difference', 'type': 'math.subtract', 'label': 'ETF价差', 'inputs': copy.deepcopy(inputs)})
    raw = {'schema_version': '2.0', 'name': 'ETF价差', 'graph': {'nodes': nodes, 'outputs': {}}}
    target = {'node_id': 'difference', 'port': 'value'}
    service = RegimeGraphV2Service(tmp_path / 'workspace', root)
    with pytest.raises(ValidationError) as error:
        service.prepare(raw, preview_target=target)
    diagnostic = error.value.diagnostics[0]
    assert diagnostic['code'] == 'EXPLICIT_ALIGNMENT_REQUIRED'
    assert diagnostic['node_id'] == 'difference'
    assert 'ETF价差' in diagnostic['message'] and '共同日期' in diagnostic['message']

    nodes.insert(2, {'id': 'common_dates', 'type': 'align.strict_intersection', 'inputs': inputs})
    nodes[-1]['inputs'] = {side: {'node_id': 'common_dates', 'port': side} for side in ['left', 'right']}
    projected = node_preview_definition(raw, target)
    assert validate_definition_v2(projected)['valid']
    plan = service.prepare(raw, preview_target=target)
    signatures = regime_graph_numba_status()['kernel_signatures']
    result = service._execute_graph(None, projected, mode, '2024-01-11', plan=plan)
    output = result['node_outputs']['difference']['value']
    expected = left_values[2:] - np.arange(10., 17.)
    np.testing.assert_allclose(output.values, expected, equal_nan=True)
    assert pd.to_datetime(output.dates).tolist() == list(pd.date_range('2024-01-03', periods=7))
    assert pd.to_datetime(output.available).tolist() == list(pd.date_range('2024-01-04', periods=7))
    assert result['result']['diagnostics']['python_fallback'] == 0
    assert result['result']['diagnostics']['request_time_compilation'] == 0
    assert regime_graph_numba_status()['kernel_signatures'] == signatures
    assert raw['graph']['outputs'] == {}


def test_adjusted_etf_fields_require_published_factors_and_rebase_after_cutoff(market):
    from research_series.product_sources import ADJUSTMENT_FILE
    root, snap, catalog = market
    base = {'kind': 'etf', 'ts_code': '510300.SH', 'field': 'close_qfq'}
    missing = catalog.catalog(kind='etf', query='510300')['items'][0]
    assert next(f for f in missing['fields'] if f['name'] == 'close_qfq')['available'] is False
    factors = pd.DataFrame({'ts_code': ['510300.SH'] * 4, 'trade_date': ['20240102', '20240103', '20240104', '20240105'], 'adj_factor': [1., 2., 2., 4.]})
    path = snap / ADJUSTMENT_FILE
    factors.to_parquet(path)
    with pytest.raises(ValidationError, match='复权因子'):
        resolve_target(base, 'retrospective', None, root)
    manifest = json.loads((root / 'tushare_active.json').read_text())
    manifest['files'][ADJUSTMENT_FILE] = path.stat().st_size
    (root / 'tushare_active.json').write_text(json.dumps(manifest))
    item = catalog.catalog(kind='etf', query='510300')['items'][0]
    assert next(f for f in item['fields'] if f['name'] == 'close_qfq')['available'] is True
    spec = {**item['binding_parameters'], **base}
    early = resolve_target(spec, 'retrospective', '2024-01-03', root)
    np.testing.assert_allclose(early.frame['value'], [1.5, 3.3])
    assert not early.snapshot['pit']['supported']
    assert early.frame['available_at'].dt.strftime('%Y-%m-%d').tolist() == ['2024-01-03'] * 2
    assert early.snapshot['adjustment']['checksum'] == spec['adjustment_checksum']
    later = resolve_target(spec, 'retrospective', None, root)
    np.testing.assert_allclose(later.frame['value'], [0.75, 1.65, np.nan, 3.6], equal_nan=True)
    back = resolve_target({**spec, 'field': 'close_hfq'}, 'retrospective', None, root)
    np.testing.assert_allclose(back.frame['value'], [3., 6.6, np.nan, 14.4], equal_nan=True)
    for field in ['close_qfq', 'close_hfq']:
        with pytest.raises(ValidationError, match='事后分析'):
            resolve_target({**spec, 'field': field}, 'realtime', None, root)
    profile = catalog.profile(series_id='etf:fund_daily:510300.SH', field='close_qfq', end_date='2024-01-03', availability_mode='latest')
    assert profile['values']['raw'] == [1.5, 3.3]
    assert profile['binding_parameters']['adjustment_checksum'] == spec['adjustment_checksum']
    factors.loc[1, 'adj_factor'] = np.nan
    factors.to_parquet(path)
    with pytest.raises(ValidationError, match='版本'):
        resolve_target(spec, 'retrospective', None, root)
    with pytest.raises(ValidationError, match='有效复权因子'):
        resolve_target(base, 'retrospective', None, root)


@pytest.mark.parametrize('prices,factors,forward,expected', [
    ([], [], 0, []), ([10., 11., 12.], [1., 2., 3.], 0, [10., 22., 36.]),
    ([10., 11., 12.], [1., 2., 3.], 1, [10./3., 22./3., 12.]),
    ([np.nan, np.inf, 12.], [1., 2., 3.], 1, [np.nan, np.nan, 12.]),
])
def test_fixed_adjusted_price_kernel(prices, factors, forward, expected):
    from research_series.numba_kernels import adjusted_price_kernel
    before = list(adjusted_price_kernel.signatures)
    actual = adjusted_price_kernel(np.array(prices, dtype=np.float64), np.array(factors, dtype=np.float64), np.int64(forward))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert actual.dtype == np.float64
    assert list(adjusted_price_kernel.signatures) == before
    np.testing.assert_allclose(actual, adjusted_price_kernel(np.array(prices, dtype=np.float64), np.array(factors, dtype=np.float64), np.int64(forward)), equal_nan=True)


@pytest.mark.parametrize('factors', [[0.], [-1.], [np.nan], [np.inf], []])
def test_adjusted_price_kernel_rejects_missing_invalid_factors(factors):
    from research_series.numba_kernels import adjusted_price_kernel
    with pytest.raises(ValueError):
        adjusted_price_kernel(np.array([10.]), np.array(factors, dtype=np.float64), np.int64(1))
    with pytest.raises(TypeError):
        adjusted_price_kernel(np.array([10.], dtype=np.float32), np.array([1.]), np.int64(0))


def _add_etf_nav(root, snap, code="510300.SH", published=True):
    path = snap / "etf_daily_df.parquet"
    pd.DataFrame({"ts_code": [code] * 5,
                  "nav_date": ["20240102", "20240103", "20240104", "20240105", "20240102"],
                  "ann_date": ["20240103", "20240104", "20240105", "20240108", "20240109"],
                  "adj_nav": [10., 10.1, np.nan, 10.3, 99.]}).to_parquet(path)
    if published:
        manifest = json.loads((root / "tushare_active.json").read_text())
        manifest["files"][path.name] = path.stat().st_size
        (root / "tushare_active.json").write_text(json.dumps(manifest))
    return path


def test_etf_adjusted_nav_binds_existing_nav_file_without_price_factors(market, tmp_path):
    root, snap, catalog = market
    path = _add_etf_nav(root, snap)
    item = catalog.catalog(kind="etf", query="510300")["items"][0]
    nav = next(field for field in item["fields"] if field["name"] == "adj_nav")
    assert nav["available"] and item["default_field"] == "close"
    binding = nav["binding_parameters"]
    assert binding["source_api"] == "fund_nav" and binding["source_file"] == path.name
    assert "adjustment_checksum" not in binding
    assert binding["file_checksum"] != item["binding_parameters"]["file_checksum"]
    bundle = resolve_target({**binding, "kind": "etf"}, "retrospective", "2024-01-08", root)
    np.testing.assert_allclose(bundle.frame["value"], [10., 10.1, np.nan, 10.3], equal_nan=True)
    assert bundle.snapshot["file"] == path.name and not bundle.snapshot["pit"]["supported"]
    assert bundle.snapshot["pit"]["available_at_field"] == "ann_date"
    assert bundle.frame["available_at"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]
    profile = catalog.profile(series_id=item["id"], field="adj_nav", as_of="2024-01-04", availability_mode="latest")
    assert profile["values"]["raw"] == [10., 10.1]
    assert profile["binding_parameters"] == binding
    assert profile["execution"]["python_fallback"] == 0
    with pytest.raises(ValidationError, match="事后分析"):
        resolve_target({**binding, "kind": "etf"}, "realtime", None, root)

    raw = graph(parameters={"window": 2, "min_periods": 2})
    raw["graph"]["nodes"][0] = {"id": "price", "type": "source.etf", "parameters": binding}
    raw["graph"]["outputs"] = {}
    service = RegimeGraphV2Service(tmp_path / "workspace", root)
    target = {"node_id": "smooth", "port": "value"}
    plan = service.prepare(raw, preview_target=target)
    result = service._execute_graph(None, node_preview_definition(raw, target), "retrospective", "2024-01-08", plan=plan)
    np.testing.assert_allclose(result["node_outputs"]["smooth"]["value"].values, [np.nan, 10.05, np.nan, np.nan], equal_nan=True)
    assert result["result"]["diagnostics"]["python_fallback"] == 0
    assert result["result"]["diagnostics"]["request_time_compilation"] == 0
    assert raw["graph"]["outputs"] == {}
    with pytest.raises(ValidationError, match="数据文件"):
        service._bound_source_root("source.etf", {**item["binding_parameters"], "field": "adj_nav"})
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValidationError, match="校验"):
        service._bound_source_root("source.etf", binding)


def test_unpublished_etf_nav_is_disabled_and_does_not_fall_back_to_market_price(market):
    from research_series.service import ResearchSeriesError
    root, snap, catalog = market
    _add_etf_nav(root, snap, published=False)
    item = catalog.catalog(kind="etf", query="510300")["items"][0]
    nav = next(field for field in item["fields"] if field["name"] == "adj_nav")
    assert nav["available"] is False and "binding_parameters" not in nav
    with pytest.raises(ResearchSeriesError, match="清单"):
        catalog.profile(series_id=item["id"], field="adj_nav", availability_mode="latest")


def test_etf_with_nav_only_is_selectable_without_exchange_prices(market):
    root, snap, catalog = market
    _add_etf_nav(root, snap, code="999999.SH")
    item = catalog.catalog(kind="etf", query="999999")["items"][0]
    assert item["status"] == "available" and item["default_field"] == "adj_nav"
    assert item["binding_parameters"]["source_api"] == "fund_nav"
    profile = catalog.profile(series_id=item["id"], field="adj_nav", availability_mode="latest", as_of="2024-01-04")
    assert profile["values"]["raw"] == [10., 10.1]
