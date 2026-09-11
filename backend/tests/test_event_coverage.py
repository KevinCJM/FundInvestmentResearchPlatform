"""Coverage scheduling and the real collector, exclusively offline fixtures."""
import copy
import json
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.data_sources import event_coverage as coverage
from backend.data_sources.models import CenterError, Pagination
from backend.tests.test_fund_event_download import run, row


def test_empty_first_page_is_confirmed_exactly_twice(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame()
    fetch.pagination_config = Pagination(mode='offset', page_size=1000, max_pages=20)
    assert run(tmp_path, fetch, latest=True).empty
    assert len(calls) == 2
    record = coverage.load(tmp_path / 'fund_portfolio_df.parquet', verify=True)
    assert list(record['days']) == ['2010-10-26']


def test_market_pages_never_expand_to_all_funds(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        assert 'ts_code' not in params
        offset = params['offset']
        return pd.DataFrame([row(symbol=str(i)) for i in range(offset, min(offset + 2, 5))])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    assert len(run(tmp_path, fetch, latest=True)) == 5
    assert [p['offset'] for p in calls] == [0, 2, 4]


def test_same_cutoff_no_requests_new_day_only_one_partition():
    evidence = {'days': {'2026-09-08': '2026-09-09T01:00:00+00:00', '2026-09-09': '2026-09-10T01:00:00+00:00'}}
    args = (date(2020, 1, 1), date(2026, 9, 8))
    assert coverage.plan_dates(*args, date(2026, 9, 9), evidence)['query_dates'] == []
    plan = coverage.plan_dates(*args, date(2026, 9, 10), evidence)
    assert plan['query_dates'] == ['20260910']
    assert plan['reused_query_days'] == 2


def test_holes_and_revision_schedule_are_separate():
    evidence = {'days': {'2026-09-08': '2026-09-09T01:00:00+00:00', '2026-09-10': '2026-09-11T01:00:00+00:00'}}
    plan = coverage.plan_dates(date(2020, 1, 1), date(2026, 9, 8), date(2026, 9, 10), evidence)
    assert plan['query_dates'] == ['20260909']
    plan = coverage.plan_dates(date(2020, 1, 1), date(2026, 9, 8), date(2026, 9, 10), evidence, revision_interval=1)
    assert plan['revision_query_days'] == 1 and plan['new_query_days'] == 1
    assert plan['query_dates'] == ['20260908', '20260909']


def test_no_coverage_does_not_apply_trading_day_overlap():
    plan = coverage.plan_dates(date(2000, 1, 1), date(2026, 8, 31), date(2026, 9, 9), None)
    assert plan['query_dates'][0] == '20260831'
    assert plan['coverage_known'] is False


@pytest.mark.parametrize('fault', ['data', 'schema', 'day', 'symlink'])
def test_corrupt_coverage_is_never_used_to_skip(tmp_path, fault):
    path = tmp_path / 'fund_portfolio_df.parquet'
    pd.DataFrame([row()]).to_parquet(path)
    session = SimpleNamespace(inceptions={'000001.OF': None}, dates=['20101026'], api_name='fund_portfolio', fields=['ts_code'])
    original = coverage.commit(path, session, 'source', None)
    assert coverage.load(path, 'other-source', verify=True) is None
    side = coverage.sidecar(path)
    if fault == 'data': path.write_bytes(b'changed')
    if fault == 'schema': side.write_text('{}')
    if fault == 'day':
        original['days'] = {'invalid': 'now'}; side.write_text(json.dumps(original))
    if fault == 'symlink':
        content = side.read_bytes(); side.unlink(); target = tmp_path / 'outside.json'; target.write_bytes(content); side.symlink_to(target)
    with pytest.raises(CenterError): coverage.load(path, verify=True)


def test_automatic_repeat_reuses_bytes_and_does_not_fetch(tmp_path):
    def fetch(**params): return pd.DataFrame([row()])
    run(tmp_path, fetch, latest=True)
    path = tmp_path / 'fund_portfolio_df.parquet'
    original = path.read_bytes()
    def forbidden(**params): pytest.fail('covered range must not fetch')
    result = run(tmp_path, forbidden, latest=True, automatic_event_plan={
        'query_dates': [], 'history_start': '20000101', 'reused_query_days': 1})
    assert len(result) == 1 and path.read_bytes() == original


def test_new_scope_does_not_borrow_old_scope_coverage(tmp_path):
    path = tmp_path / 'fund_portfolio_df.parquet'; pd.DataFrame([row()]).to_parquet(path)
    one = SimpleNamespace(inceptions={'A': None}, dates=['20240101'], api_name='fund_portfolio', fields=['ts_code'])
    previous = coverage.commit(path, one, 'source', None)
    two = copy.copy(one); two.inceptions = {'A': None, 'B': None}; two.dates = ['20240102']
    result = coverage.commit(path, two, 'source', previous)
    assert list(result['days']) == ['2024-01-02']


def test_new_fund_backfill_does_not_redownload_existing_fund(tmp_path):
    run(tmp_path, lambda **params: pd.DataFrame([row()]), latest=True)
    calls = []
    def fetch(**params):
        calls.append(params)
        assert params['ts_code'] == '000002.OF'
        return pd.DataFrame([row(code='000002.OF')])
    result = run(tmp_path, fetch, latest=True,
        universe=pd.DataFrame([{'ts_code': code, 'found_date': '20040101'} for code in ['000001.OF', '000002.OF']]),
        automatic_event_plan={'query_dates': [], 'history_start': '20101026', 'reused_query_days': 1})
    assert set(result.ts_code) == {'000001.OF', '000002.OF'} and len(calls) == 1
    assert coverage.load(tmp_path/'fund_portfolio_df.parquet', verify=True)['universe'] == ['000001.OF', '000002.OF']
