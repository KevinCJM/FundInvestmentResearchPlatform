"""Offline wire contracts for membership pagination and query receipts."""
from types import SimpleNamespace

import pandas as pd
import pytest
import T01_get_data as script
from backend.data_sources.presets import default_interfaces
from backend.data_sources.mapping import validate_mapping
from backend.data_sources.models import CenterError


def args():
    return SimpleNamespace(start_date='20260901', end_date='20260906', resume=True)


@pytest.mark.parametrize('api', ['index_member_all', 'ci_index_member', 'ths_member', 'dc_member', 'tdx_member', 'index_weight'])
def test_presets_and_pagination_contract(api):
    config = next(c for c in default_interfaces() if c.api_name == api)
    assert config.pagination.mode == 'offset'
    assert validate_mapping(config)['valid']
    assert all(f.data_type == 'string' for f in config.source_fields if f.name == 'con_name')
    if api == 'ths_member':
        assert config.policy.requests_per_minute == 180
        assert config.policy.min_interval_seconds >= 60 / 180
    if api == 'index_weight':
        assert config.pagination.max_pages == 100


@pytest.mark.parametrize('case', ['valid', 'duplicate', 'oversize', 'wrong_code', 'wrong_date', 'missing_date', 'missing_member', 'empty', 'empty_then_data', 'limit'])
def test_pages_are_complete_or_fail_closed(monkeypatch, case):
    monkeypatch.setitem(script.API_ROW_LIMITS, 'dc_member', 2)
    calls = []
    def fetch(_pro, api, limiter, options, **kw):
        calls.append(kw)
        offset = kw['offset']
        if case == 'empty' or (case == 'empty_then_data' and len(calls) == 1):
            return pd.DataFrame()
        if offset and case != 'duplicate':
            return pd.DataFrame()
        frame = pd.DataFrame([{'ts_code': 'BK1.DC', 'con_code': str(n), 'trade_date': '20260904'}
                              for n in range(3 if case == 'oversize' else 2)])
        if case == 'wrong_code': frame['ts_code'] = 'BK2.DC'
        if case == 'wrong_date': frame['trade_date'] = '20260907'
        if case == 'missing_date': frame = frame.drop(columns='trade_date')
        if case == 'missing_member': frame['con_code'] = None
        return frame
    monkeypatch.setattr(script, '_call_index_api', fetch)
    pro = object()
    if case == 'limit':
        config = next(c.model_copy(deep=True) for c in default_interfaces() if c.api_name == 'dc_member')
        config.pagination.max_pages = 1
        pro = SimpleNamespace(interfaces={'dc_member':config}, source=SimpleNamespace(policy=config.policy))
    call = lambda: script.fetch_constituent_pages(pro, 'dc_member', None, args(), ts_code='BK1.DC', trade_date='20260904')
    if case in {'valid', 'empty', 'empty_then_data'}:
        result = call()
        assert len(result) == (0 if case == 'empty' else 2)
        assert len(calls) == (2 if case == 'empty' else 4 if case == 'empty_then_data' else 3)
    else:
        with pytest.raises(CenterError, match='CONSTITUENT_'):
            call()


@pytest.mark.parametrize('api', ['index_member_all', 'ci_index_member'])
def test_industry_identity_keeps_three_levels(api):
    frame = pd.DataFrame([{'l1_code':'L1', 'l2_code':'L2', 'l3_code':'L3',
                           'ts_code':'600000.SH', 'name':'stock', 'in_date':'20200101'}])
    result = script._normalise_member_frame(frame, api, None)
    assert set(result.index_code) == {'L1','L2','L3'}
    assert set(result.con_code) == {'600000.SH'}


@pytest.mark.parametrize('empty', [False, True])
def test_receipts_reuse_verified_queries_and_reject_tampering(tmp_path, monkeypatch, empty):
    calls = []
    def fetch(*_args, **_kw):
        calls.append(1)
        return pd.DataFrame() if empty else pd.DataFrame([{'con_code':'600000.SH'}])
    monkeypatch.setattr(script, 'fetch_constituent_pages', fetch)
    call = lambda: script.cached_constituent_request(object(), 'ths_member', None, args(), tmp_path, ts_code='A.TI')
    first, second = call(), call()
    if empty:
        assert first.empty and second.empty
    else:
        pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1
    next(tmp_path.glob('*.parquet')).write_bytes(b'broken')
    with pytest.raises(CenterError, match='CHECKPOINT_INVALID'): call()


def test_weight_page_budget_splits_dates_and_preserves_completed_children(tmp_path, monkeypatch):
    calls = []
    fail = [True]
    def fetch(_pro, api, _limiter, _args, **params):
        calls.append((params['start_date'], params['end_date']))
        if params['start_date'] != params['end_date']:
            raise CenterError('CONSTITUENT_PAGE_LIMIT', '分页预算')
        if params['start_date'] == '20260902' and fail[0]:
            raise CenterError('SOURCE_TIMEOUT', '有界网络失败')
        return pd.DataFrame([{'index_code':'A.SH', 'con_code':'600000.SH', 'trade_date':params['start_date'], 'weight':1.0}])
    monkeypatch.setattr(script, 'fetch_constituent_pages', fetch)
    def download():
        return script.cached_constituent_request(object(), 'index_weight', None, args(), tmp_path,
                                                index_code='A.SH', start_date='20260901', end_date='20260902')
    with pytest.raises(CenterError, match='有界网络失败'): download()
    fail[0] = False
    result = download()
    assert len(result) == 2
    assert calls.count(('20260901', '20260901')) == 1
    before = len(calls)
    pd.testing.assert_frame_equal(download(), result)
    assert len(calls) == before


@pytest.mark.parametrize('case', ['single_day', 'network', 'invalid', 'empty_children'])
def test_weight_split_is_bounded_and_never_hides_other_failures(tmp_path, monkeypatch, case):
    calls = []
    def fetch(_pro, api, _limiter, _args, **params):
        calls.append(params)
        if case == 'empty_children' and params['start_date'] == params['end_date']:
            return pd.DataFrame()
        code = 'SOURCE_TIMEOUT' if case == 'network' else 'CONSTITUENT_RESPONSE_INVALID' if case == 'invalid' else 'CONSTITUENT_PAGE_LIMIT'
        raise CenterError(code, code)
    monkeypatch.setattr(script, 'fetch_constituent_pages', fetch)
    with pytest.raises(CenterError) as error:
        script.cached_constituent_request(object(), 'index_weight', None, args(), tmp_path,
            index_code='A.SH', start_date='20260901', end_date='20260901' if case == 'single_day' else '20260902')
    expected = {'single_day':'INDEX_WEIGHT_DAY_PAGE_LIMIT', 'network':'SOURCE_TIMEOUT', 'invalid':'CONSTITUENT_RESPONSE_INVALID', 'empty_children':'INDEX_WEIGHT_SPLIT_INCONSISTENT'}
    assert error.value.code == expected[case]
    assert len(calls) == (3 if case == 'empty_children' else 1)


def test_latest_weight_looks_back_by_month_and_stops_after_newest_nonempty(monkeypatch):
    options = args(); options.start_date = '19991231'
    calls = []
    def fetch(*_args, **params):
        calls.append(params)
        return pd.DataFrame() if params['start_date'] == '20260901' else pd.DataFrame([{'trade_date':'20260831'}])
    monkeypatch.setattr(script, 'cached_constituent_request', fetch)
    result = script.latest_index_weight(object(), None, options, None, 'A.SH')
    assert result.trade_date.tolist() == ['20260831']
    assert [(p['start_date'],p['end_date']) for p in calls] == [('20260901','20260906'),('20260801','20260831')]


def test_member_failure_never_writes_success_stage(tmp_path, monkeypatch):
    pd.DataFrame([{'source_api':'ths_index', 'quote_source_api':'ths_daily', 'ts_code':'A.TI'}]).to_parquet(tmp_path/'index_catalog_df.parquet')
    def fetch(_pro, api, *_args, **kwargs):
        if api == 'ths_member': raise RuntimeError('offline failure')
        return pd.DataFrame([{'index_code':'L1', 'con_code':'600000.SH'}])
    monkeypatch.setattr(script, 'cached_constituent_request', fetch)
    options = args(); options.limit = None; options.max_workers = 2
    with pytest.raises(CenterError) as error:
        script.save_index_constituents(object(), tmp_path, None, options)
    assert error.value.code == 'CONSTITUENT_INCOMPLETE'
    assert not (tmp_path/'index_members_df.parquet').exists()
    assert not (tmp_path/'.tushare_stage_index_constituents_members.json').exists()


def test_missing_interfaces_fail_before_reading_data_or_submitting_requests(tmp_path):
    with pytest.raises(CenterError) as error:
        script.save_index_constituents(SimpleNamespace(interfaces={}), tmp_path, None, args())
    assert error.value.code == 'API_NOT_CONFIGURED'
    assert 'ths_member' in error.value.message


def test_weight_single_day_obeys_configured_budget_and_never_accepts_a_full_last_page(monkeypatch):
    monkeypatch.setitem(script.API_ROW_LIMITS, 'index_weight', 2)
    config = next(c.model_copy(deep=True) for c in default_interfaces() if c.api_name == 'index_weight')
    config.pagination.max_pages = 20
    pro = SimpleNamespace(interfaces={'index_weight': config}, source=SimpleNamespace(policy=config.policy))
    calls = []
    def fetch(_pro, _api, _limiter, _args, **kw):
        calls.append(kw['offset'])
        return pd.DataFrame([{'index_code': 'A.SH', 'con_code': str(n), 'trade_date': '20260904', 'weight': 1.0}
                             for n in range(kw['offset'], min(kw['offset'] + kw['limit'], 43))])
    monkeypatch.setattr(script, '_call_index_api', fetch)
    def download():
        return script.fetch_constituent_pages(pro, 'index_weight', None, args(),
                                             index_code='A.SH', start_date='20260904', end_date='20260904')
    with pytest.raises(CenterError) as exc:
        download()
    assert exc.value.code == 'CONSTITUENT_PAGE_LIMIT' and len(calls) == 20
    calls.clear(); config.pagination.max_pages = 100
    assert len(download()) == 43 and len(calls) == 22
