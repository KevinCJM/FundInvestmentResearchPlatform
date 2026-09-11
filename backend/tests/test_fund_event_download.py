"""Offline collector regression: no credentials or real provider calls."""
from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import T01_get_data as script
from backend.data_sources.fund_events import FundEventDownload, fund_inceptions
from backend.data_sources.models import CenterError, DownloadPolicy, Pagination


@pytest.mark.parametrize('null_rows', [0, 1])
@pytest.mark.parametrize('reverse', [False, True])
def test_ordered_merge_promotes_null_without_losing_lineage(tmp_path, null_rows, reverse):
    first, second = tmp_path / 'first.parquet', tmp_path / 'second.parquet'
    pq.write_table(pa.table({'id': pa.array(range(null_rows), type=pa.int64()),
                            'source_api': pa.nulls(null_rows)}), first)
    pq.write_table(pa.table({'id': [2], 'source_api': ['fund_div']}), second)
    paths = [second, first] if reverse else [first, second]
    output = tmp_path / 'merged.parquet'
    assert script._consolidate_ordered_parts(paths, output) == null_rows + 1
    table = pq.read_table(output)
    assert table.schema.field('source_api').type == pa.string()
    expected = ['fund_div'] + [None] * null_rows if reverse else [None] * null_rows + ['fund_div']
    assert table['source_api'].to_pylist() == expected


def test_ordered_merge_rejects_concrete_type_conflict_atomically(tmp_path):
    first, second, output = [tmp_path / name for name in ('a.parquet', 'b.parquet', 'out.parquet')]
    pq.write_table(pa.table({'value': [1]}), first)
    pq.write_table(pa.table({'value': ['2']}), second)
    pq.write_table(pa.table({'value': [99]}), output)
    before = output.read_bytes()
    with pytest.raises(pa.ArrowTypeError):
        script._consolidate_ordered_parts([first, second], output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob('*.tmp'))


def row(code='000001.OF', date='20101026', **changes):
    return dict(ts_code=code, ann_date=date, end_date='20100930',
                symbol='600000.SH', mkv=2.0) | changes


@pytest.mark.parametrize('latest', [False, True])
def test_single_day_holdings_pages_are_complete_checkpointed_and_reusable(tmp_path, latest):
    calls = []
    def fetch(**params):
        calls.append(params)
        if not params.get('ts_code') and 'offset' not in params:
            raise script.ResponseTruncatedError('market cap')
        assert params['ann_date'] == '20101026' and 'start_date' not in params
        offset = params['offset']
        return pd.DataFrame([row(symbol=str(i)) for i in range(offset, min(offset + 2, 5))])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    fetch.download_policy = DownloadPolicy(max_rows_per_request=2)
    result = run(tmp_path, fetch, latest=latest)
    assert len(result) == 5 and result.symbol.tolist() == list('01234')
    assert [p['offset'] for p in calls if 'offset' in p] == [0, 2, 4]
    calls.clear()
    assert len(run(tmp_path, fetch, latest=latest)) == 5
    assert not calls


@pytest.mark.parametrize('case', ['repeat', 'internal_duplicate', 'oversized', 'wrong_code', 'wrong_day', 'missing_key', 'exhausted', 'permission', 'budget'])
def test_invalid_pages_never_publish(tmp_path, case):
    calls = []
    def fetch(**params):
        calls.append(params)
        offset = params['offset']
        if case == 'permission': raise CenterError('SOURCE_PERMISSION_OR_PARAMS', 'Denied')
        rows = [row(symbol=str(offset + i)) for i in range(2)]
        if case == 'repeat': rows = [row(symbol=str(i)) for i in range(2)]
        if case == 'internal_duplicate': rows[1] = {**rows[0], 'mkv': 99.0}
        if case == 'oversized': rows.append(row(symbol='extra'))
        if case == 'wrong_code': rows[0]['ts_code'] = 'other'
        if case == 'wrong_day': rows[0]['ann_date'] = '20101027'
        if case == 'missing_key': del rows[0]['symbol']
        return pd.DataFrame(rows)
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=2)
    with pytest.raises(CenterError):
        run(tmp_path, fetch, latest=False, fund_event_max_requests=1 if case == 'budget' else 20)
    assert not (tmp_path / 'fund_portfolio_df.parquet').exists()
    assert len(calls) <= 2


def test_page_empty_is_rechecked_and_transient_empty_does_not_truncate(tmp_path):
    calls = []
    def fetch(**params):
        offset = params['offset']
        calls.append(offset)
        if offset == 2 and calls.count(2) == 1:
            return pd.DataFrame()
        return pd.DataFrame([row(symbol=str(i)) for i in range(offset, min(offset + 2, 4))])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    assert len(run(tmp_path, fetch, latest=False)) == 4
    assert calls == [0, 2, 2, 4, 4]


@pytest.mark.parametrize('missing', [None, float('nan')])
def test_identical_rows_within_page_keep_raw_cursor_and_reuse_receipt(tmp_path, missing):
    calls = []
    rows = [row(symbol='A', stk_float_ratio=missing), row(symbol='A', stk_float_ratio=missing), row(symbol='B')]
    def fetch(**params):
        calls.append(params['offset'])
        return pd.DataFrame(rows[params['offset']:params['offset'] + params['limit']])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=3)
    fetch.download_policy = DownloadPolicy(max_rows_per_request=2)
    assert run(tmp_path, fetch, latest=True).symbol.tolist() == ['A', 'B']
    assert calls == [0, 2]  # Dedup leaves one row but the raw first page is full.
    calls.clear()
    assert run(tmp_path, fetch, latest=True).symbol.tolist() == ['A', 'B']
    assert not calls


@pytest.mark.parametrize('missing', [None, float('nan')])
def test_identical_cross_page_rows_require_stable_full_pass(tmp_path, missing):
    calls = []
    rows = [row(symbol=s, stk_float_ratio=missing) for s in ['A', 'B', 'B', 'C', 'D']]
    def fetch(**params):
        calls.append(params['offset'])
        return pd.DataFrame(rows[params['offset']:params['offset'] + params['limit']])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    assert run(tmp_path, fetch, latest=True).symbol.tolist() == ['A', 'B', 'C', 'D']
    assert calls == [0, 2, 4, 0, 2, 4]
    calls.clear()
    assert len(run(tmp_path, fetch, latest=True)) == 4
    assert not calls


@pytest.mark.parametrize('change', ['value', 'order', 'terminal', 'budget'])
def test_cross_page_validation_failure_never_creates_complete_receipt(tmp_path, change):
    calls = []
    rows = [row(symbol=s) for s in ['A', 'B', 'B', 'C', 'D']]
    def fetch(**params):
        offset = params['offset']
        calls.append(offset)
        selected = [dict(r) for r in rows[offset:offset + params['limit']]]
        if calls.count(offset) > 1:
            if change == 'value' and offset == 0: selected[0]['mkv'] += 1e-14
            if change == 'order' and offset == 0: selected.reverse()
            if change == 'terminal' and offset == 4: selected = []
        return pd.DataFrame(selected)
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    with pytest.raises(CenterError):
        run(tmp_path, fetch, latest=True, fund_event_max_requests=3 if change == 'budget' else 20)
    assert not (tmp_path / 'fund_portfolio_df.parquet').exists()
    assert not list(tmp_path.rglob('*_market.json'))
    assert len(calls) <= 7


@pytest.mark.parametrize('latest', [True, False])
@pytest.mark.parametrize('same_page', [True, False])
def test_stable_conflicting_values_are_losslessly_quarantined(tmp_path, latest, same_page):
    import json
    from backend.data_sources.fund_event_conflicts import VALUES
    calls = []
    original = row(symbol='B', amount=38357., mkv=699019.63)
    revised = row(symbol='B', amount=38000., mkv=692740.)
    rows = [original, revised, row(symbol='C')] if same_page else [row(symbol='A'), original, revised, row(symbol='C'), row(symbol='D')]
    def fetch(**params):
        calls.append(params['offset'])
        return pd.DataFrame(rows[params['offset']:params['offset'] + params['limit']])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    result = run(tmp_path, fetch, latest=latest)
    conflict = result.loc[result.symbol.eq('B')]
    assert len(conflict) == 1 and conflict[VALUES].isna().all().all()
    assert conflict.availability_status.eq('source_conflict').all()
    assert result.loc[~result.symbol.eq('B')].availability_status.eq('announced_date').all()
    assert calls == ([0, 2, 0, 2] if same_page else [0, 2, 4, 0, 2, 4])
    evidence = pd.read_parquet(tmp_path / 'fund_portfolio_conflicts.parquet')
    assert set(evidence.amount) == {38000., 38357.} and len(evidence) == 2
    quality = json.loads((tmp_path / 'fund_portfolio_df.parquet.quality.meta.json').read_text())
    assert quality['acquisition_complete'] and not quality['publishable']
    assert quality['conflicting_keys'] == 1
    calls.clear()
    again = run(tmp_path, fetch, latest=latest)
    assert not calls and again.loc[again.symbol.eq('B'), VALUES].isna().all().all()
    assert len(pd.read_parquet(tmp_path / 'fund_portfolio_conflicts.parquet')) == 2


@pytest.mark.parametrize('ambiguous_batch', [False, True])
def test_incremental_revision_updates_old_value_without_misclassifying_batches(tmp_path, ambiguous_batch):
    # The immutable old snapshot and a new request are different acquisitions.
    old = script._prepare_fund_event_rows(
        pd.DataFrame([row(amount=38357., mkv=699019.63), row(symbol='other', mkv=7.)]),
        fields=script.FUND_PORTFOLIO_FIELDS, source_api='fund_portfolio', observation_column='end_date')
    old['ingested_at'] = '2000-01-01T00:00:00+00:00'
    snapshot = tmp_path / 'old_snapshot.parquet'
    script.save_dataframe(old, snapshot, quiet=True)
    original = snapshot.read_bytes()
    work = tmp_path / 'new_acquisition'
    work.mkdir()
    (work / 'fund_portfolio_df.parquet').write_bytes(original)
    incoming = [row(amount=38000., mkv=692740.)]
    if ambiguous_batch:
        incoming.append(row(amount=38357., mkv=699019.63))
    def fetch(**params):
        return pd.DataFrame(incoming)
    result = run(work, fetch, latest=True)
    current = result.loc[result.symbol.eq('600000.SH')].iloc[0]
    assert len(result) == 2 and result.loc[result.symbol.eq('other'), 'mkv'].iloc[0] == 7.
    assert snapshot.read_bytes() == original
    assert pd.Timestamp(current.ingested_at) > pd.Timestamp('2000-01-01', tz='UTC')
    if ambiguous_batch:
        assert pd.isna(current.mkv) and pd.isna(current.amount)
        assert current.availability_status == 'source_conflict'
    else:
        assert current.mkv == 692740. and current.amount == 38000.
        assert current.availability_status == 'announced_date'
        assert not (work / 'fund_portfolio_df.parquet.quality.meta.json').exists()


def test_conflict_evidence_tampering_blocks_checkpoint_reuse(tmp_path):
    rows = [row(mkv=2.), row(mkv=2. + 1e-14)]
    def fetch(**params):
        return pd.DataFrame(rows[params['offset']:params['offset'] + params['limit']])
    fetch.pagination_config = Pagination(mode='offset', page_size=3, max_pages=2)
    run(tmp_path, fetch, latest=True)
    evidence = next(tmp_path.rglob('conflicts/*.parquet'))
    evidence.write_bytes(b'corrupted')
    with pytest.raises(CenterError, match='冲突'):
        run(tmp_path, fetch, latest=True)


def test_overlap_verification_confirms_empty_terminal_page(tmp_path):
    calls = []
    rows = [row(symbol=s) for s in ['A', 'B', 'B', 'C']]
    def fetch(**params):
        calls.append(params['offset'])
        return pd.DataFrame(rows[params['offset']:params['offset'] + params['limit']])
    fetch.pagination_config = Pagination(mode='offset', page_size=2, max_pages=4)
    assert len(run(tmp_path, fetch, latest=True)) == 3
    assert calls == [0, 2, 4, 4, 0, 2, 4, 4]


def test_old_single_day_split_is_requeried_using_enabled_pages(tmp_path):
    args = arguments(tmp_path, latest=False)
    session = FundEventDownload(directory=script.history_checkpoint_dir(tmp_path/'fund_portfolio_df.parquet', args),
        dates=['20101026'], universe=pd.DataFrame([{'ts_code':'000001.OF','found_date':'20040101'}]),
        api_name='fund_portfolio', fields=script.FUND_PORTFOLIO_FIELDS, smoke=False, strategy='fund_announcement_history')
    session._record('20101026-20101026', '000001.OF', 'SPLIT')
    def fetch(**params):
        assert params['offset'] == 0
        return pd.DataFrame([row()])
    fetch.pagination_config = Pagination(mode='offset', page_size=1000)
    assert len(run(tmp_path, fetch, latest=False)) == 1


@pytest.mark.parametrize('code', ['SOURCE_CONNECTION', 'SOURCE_DNS', 'SOURCE_TIMEOUT', 'SOURCE_TRANSIENT', 'SOURCE_RATE_LIMIT'])
def test_exhausted_retry_preserves_safe_error_category(code, monkeypatch, capsys):
    from backend.data_sources.transport import TransientSourceError
    calls, waits = [], []
    def fetch():
        calls.append(1)
        raise TransientSourceError(code, '安全的来源错误说明。', 502) from OSError('private-credential')
    with pytest.raises(CenterError) as error:
        script.call_tushare_api(fetch, script.RateLimiter(10000), max_retries=3,
            backoff_sec=0, wait_on_rate_limit_sec=60, retry_jitter_sec=0,
            context='fund_portfolio A', interrupt_wait=waits.append)
    assert error.value.code == code and '已尝试 3 次' in error.value.message
    assert len(calls) == 3 and len(waits) == 2
    assert 'private-credential' not in error.value.message + capsys.readouterr().out
    if code == 'SOURCE_RATE_LIMIT': assert all(wait >= 60 for wait in waits)


@pytest.mark.parametrize('error', [ValueError('bad data'), OSError(28, 'disk full'), RuntimeError('bug')])
def test_non_network_failures_are_not_retried(error):
    calls = []
    def fetch():
        calls.append(1)
        raise error
    with pytest.raises(type(error)):
        script.call_tushare_api(fetch, script.RateLimiter(10000), max_retries=3,
            backoff_sec=0, wait_on_rate_limit_sec=0, context='fund_portfolio A')
    assert calls == [1]


@pytest.mark.parametrize('code', ['SOURCE_CONNECTION', 'SOURCE_DNS', 'SOURCE_TIMEOUT'])
def test_network_outage_waits_for_recovery_without_extra_attempts(code):
    from backend.data_sources.transport import TransientSourceError
    calls, waits = [], []
    def fetch():
        calls.append(1)
        raise TransientSourceError(code, '网络暂时中断', 502)
    fetch.download_policy = DownloadPolicy(max_attempts=3, read_timeout_seconds=30)
    with pytest.raises(CenterError):
        script.call_tushare_api(fetch, script.RateLimiter(10000), max_retries=10,
            backoff_sec=2, wait_on_rate_limit_sec=60, retry_jitter_sec=0,
            context='ths_daily A', interrupt_wait=waits.append)
    assert len(calls) == 3
    assert waits == [30, 60]


def arguments(path, **changes):
    args = script.parse_args(['--output-dir', str(path), '--start-date', '20101026',
                              '--end-date', '20101026', '--max-workers', '1',
                              '--max-retries', '1', '--backoff-sec', '0',
                              '--retry-jitter-sec', '0'])
    vars(args).update(changes)
    # Tests of the announcement path use explicit incremental bounds; full
    # history/range partition tests opt out below.
    if 'latest' not in changes:
        args.latest = True
    args.automatic_start_date = args.start_date
    return args


def run(path, fetch, *, universe=None, **changes):
    class Pro:
        fund_portfolio = staticmethod(fetch)
    if universe is None:
        universe = pd.DataFrame([{'ts_code': '000001.OF', 'found_date': '20040101'}])
    universe = universe.assign(name='离线基金')
    script.save_fund_portfolio(Pro(), path, script.RateLimiter(10_000),
                               arguments(path, **changes), fund_info=universe)
    return pd.read_parquet(path / 'fund_portfolio_df.parquet')


def test_cap_does_not_discard_predecessor_history_by_current_inception(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        if 'ts_code' not in params:
            raise script.ResponseTruncatedError('cap')
        return pd.DataFrame([row(params['ts_code'])])
    universe = pd.DataFrame([
        {'ts_code':'000001.OF', 'found_date':'20040101', 'due_date':'20091231'},
        {'ts_code':'000770.OF', 'found_date':'20141028'},
        {'ts_code':'001345.OF', 'found_date':'20150526'},
        {'ts_code':'000002.OF', 'found_date':None},
    ])
    result = run(tmp_path, fetch, universe=universe)
    assert [p.get('ts_code') for p in calls] == [None, '000001.OF', '000002.OF', '000770.OF', '001345.OF']
    assert result.ts_code.tolist() == ['000001.OF', '000002.OF', '000770.OF', '001345.OF']
    assert result.available_at.eq(pd.Timestamp('2010-10-26')).all()
    assert result.end_date.eq(pd.Timestamp('2010-09-30')).all()


def test_unknown_and_conflicting_inceptions_remain_candidates():
    universe = pd.DataFrame([
        {'ts_code':'A', 'found_date':'20140101'}, {'ts_code':'A', 'found_date':None},
        {'ts_code':'B', 'found_date':'invalid'}, {'ts_code':'C', 'found_date':'2010-01-02'},
    ])
    assert fund_inceptions(universe) == {'A':None, 'B':None, 'C':'20100102'}


def test_historical_disclosure_before_current_contract_is_retained(tmp_path):
    def fetch(**params):
        return pd.DataFrame([row('A')])
    result = run(tmp_path, fetch, universe=pd.DataFrame([{'ts_code':'A','found_date':'20200101'}]))
    assert len(result) == 1


def test_confirmed_empty_is_reused_but_old_unverified_marker_is_not(tmp_path):
    args = arguments(tmp_path)
    old = script.history_checkpoint_dir(tmp_path / 'fund_portfolio_df.parquet', args)
    old.mkdir(parents=True)
    (old / '20101026.empty').write_text('no data\n')
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame()
    assert run(tmp_path, fetch).empty
    assert len(calls) == 2
    assert run(tmp_path, fetch).empty
    assert len(calls) == 2
    assert (old / '20101026.empty').exists()


def test_transient_empty_recovery_is_not_cached_as_empty(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame() if len(calls) == 1 else pd.DataFrame([row()])
    assert len(run(tmp_path, fetch)) == 1
    assert len(calls) == 2


def test_partial_fund_checkpoint_survives_failure_and_resume(tmp_path):
    universe = pd.DataFrame([{'ts_code':'000001.OF'}, {'ts_code':'000002.OF'}])
    calls = []
    failing = True
    def fetch(**params):
        calls.append(params.get('ts_code'))
        if 'ts_code' not in params:
            raise script.ResponseTruncatedError('cap')
        if params['ts_code'] == '000002.OF' and failing:
            raise CenterError('SOURCE_PERMISSION', 'Denied')
        return pd.DataFrame([row(params['ts_code'])])
    with pytest.raises(CenterError, match='Denied'):
        run(tmp_path, fetch, universe=universe)
    assert not (tmp_path / 'fund_portfolio_df.parquet').exists()
    assert list(tmp_path.rglob('*_000001.OF.parquet'))
    failing = False
    calls.clear()
    result = run(tmp_path, fetch, universe=universe)
    assert calls == ['000002.OF']
    assert len(result) == 2


def test_permanent_failure_stops_later_dates_and_preserves_output(tmp_path):
    target = tmp_path / 'fund_portfolio_df.parquet'
    pd.DataFrame({'existing':[123]}).to_parquet(target)
    before = target.read_bytes()
    calls = []
    def fetch(**params):
        calls.append(params)
        raise CenterError('SOURCE_PERMISSION', 'Denied')
    with pytest.raises(CenterError):
        run(tmp_path, fetch, end_date='20101126', max_retries=3)
    assert len(calls) == 1
    assert target.read_bytes() == before


def test_inflight_success_is_checkpointed_without_dispatching_more(tmp_path):
    entered = threading.Barrier(2)
    calls = []
    def fetch(**params):
        calls.append(params['ann_date'])
        entered.wait(timeout=5)
        if params['ann_date'] == '20101026':
            raise CenterError('SOURCE_PERMISSION', 'Denied')
        return pd.DataFrame([row(date=params['ann_date'])])
    with pytest.raises(CenterError):
        run(tmp_path, fetch, end_date='20101126', max_workers=2)
    assert sorted(calls) == ['20101026', '20101027']
    assert list(tmp_path.rglob('20101027_market.parquet'))


@pytest.mark.parametrize('kind', ['wrong_date', 'missing_key', 'bad_report', 'wrong_code'])
def test_invalid_responses_fail_closed(tmp_path, kind):
    def fetch(**params):
        value = row()
        if kind == 'wrong_date': value['ann_date'] = '20101025'
        if kind == 'missing_key': value.pop('symbol')
        if kind == 'bad_report': value['end_date'] = 'bad'
        if kind == 'wrong_code':
            if 'ts_code' not in params: raise script.ResponseTruncatedError('cap')
            value['ts_code'] = '999999.OF'
        return pd.DataFrame([value])
    with pytest.raises(CenterError) as error:
        run(tmp_path, fetch)
    assert error.value.code == 'FUND_EVENT_RESPONSE'
    assert not (tmp_path / 'fund_portfolio_df.parquet').exists()


def test_single_fund_cap_is_explicit_failure_not_infinite_fallback(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        raise script.ResponseTruncatedError('cap')
    with pytest.raises(CenterError) as error:
        run(tmp_path, fetch)
    assert error.value.code == 'FUND_EVENT_UNSPLITTABLE'
    assert len(calls) == 2


def test_corrupt_checkpoint_cannot_be_silently_reused(tmp_path):
    run(tmp_path, lambda **params: pd.DataFrame([row()]))
    part = next(tmp_path.rglob('20101026_market.parquet'))
    part.write_bytes(b'corrupt')
    with pytest.raises(CenterError) as error:
        run(tmp_path, lambda **params: pytest.fail('Must validate checkpoint first'))
    assert error.value.code == 'FUND_EVENT_CHECKPOINT'


@pytest.mark.parametrize('change', ['universe', 'source'])
def test_checkpoint_isolation_by_universe_and_source(tmp_path, change):
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame([row()])
    run(tmp_path, fetch)
    if change == 'source':
        run(tmp_path, fetch, source_configuration_hash='different')
    else:
        run(tmp_path, fetch, universe=pd.DataFrame([{'ts_code':'000001.OF','found_date':None}]))
    assert len(calls) == 2


@pytest.mark.parametrize('empty', [True, False])
def test_smoke_never_uses_more_than_one_request(tmp_path, empty):
    calls = []
    def fetch(**params):
        calls.append(params)
        if empty: return pd.DataFrame()
        raise script.ResponseTruncatedError('cap')
    if empty:
        run(tmp_path, fetch, smoke=True)
    else:
        with pytest.raises(script.ResponseTruncatedError):
            run(tmp_path, fetch, smoke=True)
    assert len(calls) == 1


def test_budget_counts_retries_and_stops_without_final_output(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        raise ConnectionError('offline transient')
    with pytest.raises(CenterError) as error:
        run(tmp_path, fetch, max_retries=3, fund_event_max_requests=2)
    assert error.value.code == 'FUND_EVENT_BUDGET'
    assert len(calls) == 2


@pytest.mark.parametrize('limit, code', [('idle_timeout','FUND_EVENT_IDLE'), ('max_runtime','FUND_EVENT_RUNTIME')])
def test_wait_bounds_without_sleep(tmp_path, limit, code):
    session = FundEventDownload(directory=tmp_path, dates=['20101026'],
                                universe=pd.DataFrame([{'ts_code':'A'}]),
                                api_name='fund_portfolio', fields=[], smoke=False, **{limit:1})
    session.started -= 10
    if limit == 'idle_timeout': session.activity -= 10
    with pytest.raises(CenterError) as error: session.check()
    assert error.value.code == code


def test_sorted_full_output_and_incremental_overlap(tmp_path):
    def fetch(**params):
        return pd.DataFrame([row(date=params['ann_date'])])
    result = run(tmp_path, fetch, end_date='20101028', max_workers=2)
    assert result.available_at.tolist() == list(pd.date_range('20101026','20101028'))
    def revised(**params):
        value = row(date=params['ann_date']); value['mkv'] = 9.0
        return pd.DataFrame([value])
    result = run(tmp_path, revised, start_date='20101028', end_date='20101029', latest=True)
    assert result.available_at.tolist() == list(pd.date_range('20101026','20101029'))
    assert result.mkv.tolist() == [2.0, 2.0, 9.0, 9.0]


def test_policy_cap_and_concurrency_are_not_relaxed(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame([row(), row(symbol='600001.SH')])
    fetch.download_policy = DownloadPolicy(max_rows_per_request=2, max_concurrency=1)
    with pytest.raises(CenterError) as error: run(tmp_path, fetch, max_workers=16)
    assert error.value.code == 'FUND_EVENT_UNSPLITTABLE'
    assert len(calls) == 2


@pytest.mark.parametrize('failure', [None, 'SOURCE_PERMISSION_OR_PARAMS', 'SOURCE_ROW_CAP'])
@pytest.mark.parametrize('strategy', ['announcement', 'history'])
def test_worker_only_acknowledges_fully_resolved_market_caps(tmp_path, monkeypatch, failure, strategy):
    from backend.data_sources.acquisition import fingerprint
    from backend.data_sources.credentials import save_credential
    from backend.data_sources.runtime import ConfiguredTushareClient
    from backend.data_sources.store import SourceStore
    from backend.data_sources.task_catalog import task_specs
    from backend.data_sources.task_worker import acquire
    store = SourceStore(tmp_path); store.seed()
    # Exercise the still-supported unpaged cap-splitting contract explicitly.
    from backend.data_sources.models import InterfaceConfig
    record = store.get('interface', 'tushare.fund_portfolio')
    config = InterfaceConfig.model_validate(record['config']); config.pagination.mode = 'none'
    store.save(config, record['revision'])
    save_credential(store, 'tushare', 'offline-placeholder')
    records = [r for kind in ('source','interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    def call(self, interface, **params):
        if 'ts_code' not in params: raise CenterError('SOURCE_ROW_CAP', 'cap')
        if strategy == 'history' and params['start_date'] == '20101025' and params['end_date'] == '20101026':
            raise CenterError('SOURCE_ROW_CAP', 'cap')
        if failure: raise CenterError(failure, 'offline failure')
        if strategy == 'history' and not params['start_date'] <= '20101026' <= params['end_date']:
            return pd.DataFrame()
        return pd.DataFrame([row()])
    def operation(args, actions, *, client):
        args.max_workers = 1
        args.latest, args.automatic_start_date = strategy == 'announcement', args.start_date
        script.save_fund_portfolio(client, args.output_dir, script.RateLimiter(10_000), args,
                                  fund_info=pd.DataFrame([{'ts_code':'000001.OF','name':'离线基金'}]))
    monkeypatch.setattr(ConfiguredTushareClient, '_call', call)
    monkeypatch.setattr(script, '_run_actions', operation)
    payload = dict(root=str(tmp_path), source_id='tushare', source_hash=fingerprint(records),
                   params={'start_date':'20101025' if strategy == 'history' else '20101026',
                           'end_date':'20101026'}, mode='full', has_baseline=False)
    if failure:
        with pytest.raises(CenterError):
            acquire(payload, task_specs()['tushare.fund_portfolio'], tmp_path / 'out')
        assert not (tmp_path / 'out' / 'fund_portfolio_df.parquet').exists()
    else:
        acquire(payload, task_specs()['tushare.fund_portfolio'], tmp_path / 'out')
        assert len(pd.read_parquet(tmp_path / 'out' / 'fund_portfolio_df.parquet')) == 1


def test_shared_quota_wait_is_cooperatively_cancellable(tmp_path, monkeypatch):
    from backend.data_sources import quota as module
    from backend.data_sources.store import SourceStore
    quota = module.SharedQuota(SourceStore(tmp_path))
    attempts = []
    def check():
        attempts.append(1)
        if len(attempts) == 2: raise CenterError('STOP', 'offline cancel')
    monkeypatch.setattr(quota, 'reserve', lambda *args: 60)
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: None)
    with pytest.raises(CenterError) as error:
        with quota.acquire('tushare', 'fund_portfolio', DownloadPolicy(), DownloadPolicy(), 1, check=check):
            pytest.fail('No quota acquired')
    assert error.value.code == 'STOP'
    assert len(attempts) == 2


def test_full_history_fetches_by_fund_once_and_retains_predecessor(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        assert 'ann_date' not in params and params['start_date'] == '19991231'
        return pd.DataFrame([row(params['ts_code'], date='20000117', end_date='19991231')])
    universe = pd.DataFrame([{'ts_code':'000264.OF','found_date':'20130715'},
                             {'ts_code':'000595.OF','found_date':'20140404'}])
    result = run(tmp_path, fetch, universe=universe, latest=False, start_date='19991231', end_date='20101026')
    assert len(calls) == 2  # Not thousands of announcement dates x both funds.
    assert result.ann_date.eq(pd.Timestamp('20000117')).all()
    assert len(result) == 2
    calls.clear()
    assert len(run(tmp_path, fetch, universe=universe, latest=False,
                   start_date='19991231', end_date='20101026')) == 2
    assert not calls


def test_announcement_range_cap_covers_both_halves_without_gaps(tmp_path):
    source = [row(date=date, end_date='20101231') for date in ['20110101','20110110','20110120','20110131']]
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame([r for r in source if params['start_date'] <= r['ann_date'] <= params['end_date']])
    fetch.download_policy = DownloadPolicy(max_rows_per_request=2, max_attempts=1, backoff_seconds=0)
    result = run(tmp_path, fetch, latest=False, start_date='20110101', end_date='20110131')
    assert result.ann_date.tolist() == [pd.Timestamp(r['ann_date']) for r in source]
    assert len(calls) < 100
    assert not any('ann_date' in p or 'offset' in p for p in calls)


def test_report_dates_before_announcement_range_are_retained(tmp_path):
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame([row(date='20101026', end_date='20000331')])
    result = run(tmp_path, fetch, latest=False)
    assert len(calls) == 1 and len(result) == 1
    assert result.end_date.iloc[0] == pd.Timestamp('20000331')


@pytest.mark.parametrize('date', ['20101025', '20101027'])
def test_full_history_rejects_ignored_announcement_bounds(tmp_path, date):
    def fetch(**params):
        return pd.DataFrame([row(date=date)])
    with pytest.raises(CenterError) as error:
        run(tmp_path, fetch, latest=False)
    assert error.value.code == 'FUND_EVENT_RESPONSE'


def test_observed_000011_range_keeps_lagged_report_and_boundaries(tmp_path):
    """Regression of the 2026-09-08 live failure: report date < request start."""
    def fetch(**params):
        assert params['start_date'] == '20150826' and params['end_date'] == '20210301'
        return pd.DataFrame([row('000011.OF', date=date, end_date=end)
                             for date, end in [('20150826','20150630'),
                                               ('20150829','20150630'),
                                               ('20210301','20201231')]])
    result = run(tmp_path, fetch, latest=False, start_date='20150826', end_date='20210301',
                 universe=pd.DataFrame([{'ts_code':'000011.OF','found_date':'20040812'}]))
    assert len(result) == 3
    assert result.available_at.tolist() == list(pd.to_datetime(['20150826','20150829','20210301']))
    assert result.end_date.iloc[0] == pd.Timestamp('20150630')


def test_single_fund_single_announcement_cap_still_fails_closed(tmp_path):
    def fetch(**params):
        return pd.DataFrame([row(), row(symbol='600001.SH')])
    fetch.download_policy = DownloadPolicy(max_rows_per_request=2, max_attempts=1)
    with pytest.raises(CenterError) as error:
        run(tmp_path, fetch, latest=False)
    assert error.value.code == 'FUND_EVENT_UNSPLITTABLE'
    assert not (tmp_path / 'fund_portfolio_df.parquet').exists()


def test_new_date_axis_does_not_reuse_v3_receipts(tmp_path):
    import hashlib
    import json
    from backend.services.refresh_runtime import atomic_write_json
    universe = pd.DataFrame([{'ts_code':'000001.OF','found_date':'20040101'}])
    parent = script.history_checkpoint_dir(tmp_path / 'fund_portfolio_df.parquet', arguments(tmp_path, latest=False))
    contract = dict(version=3, strategy='fund_report_history', api='fund_portfolio',
                    fields=script.FUND_PORTFOLIO_FIELDS, dates=['20101026'],
                    inceptions=fund_inceptions(universe), smoke=False)
    old = parent / ('events_v3_' + hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:20])
    old.mkdir(parents=True)
    receipt = old / '19000101-20101026_000001.OF.json'
    atomic_write_json(receipt, dict(date='19000101-20101026', code='000001.OF', status='EMPTY', confirmations=2))
    calls = []
    def fetch(**params):
        calls.append(params)
        return pd.DataFrame([row()])
    assert len(run(tmp_path, fetch, latest=False, universe=universe)) == 1
    assert len(calls) == 1 and receipt.exists()
    assert len(list(parent.glob('events_v4_*/contract.json'))) == 1


def test_bounded_external_merge_is_sorted_deduplicated_and_atomic(tmp_path):
    from backend.data_sources.fund_event_merge import merge_event_parts
    files = []
    for index in range(40):
        path = tmp_path / f'{index}.parquet'
        pd.DataFrame({'key':['a', str(100-index)], 'value':[index, index]}).sort_values('key').to_parquet(path)
        files.append(path)
    output = tmp_path / 'merged.parquet'
    merge_event_parts(files, output, ['key'], lambda: None)
    result = pd.read_parquet(output)
    assert result.key.tolist() == sorted(set(result.key))
    assert result[result.key == 'a'].value.iloc[0] == 39
    original = output.read_bytes()
    def fail(): raise CenterError('STOP', 'offline cancel')
    with pytest.raises(CenterError): merge_event_parts(files, output, ['key'], fail)
    assert output.read_bytes() == original
    assert not list(tmp_path.glob('.event-merge-*'))
