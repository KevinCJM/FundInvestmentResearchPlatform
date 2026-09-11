"""Offline cap splitting, receipt integrity, retries and lineage."""
from types import SimpleNamespace

import pandas as pd
import pytest

import T01_get_data as script
from backend.data_sources.models import CenterError, DownloadPolicy
from backend.tests import test_etl_dataset_tasks as dataset_fixtures

store = dataset_fixtures.store


def run(tmp_path, operation, **changes):
    args = script.parse_args(['--start-date', '20260901', '--end-date', '20260904', '--resume',
                              '--max-retries', '1', '--backoff-sec', '0', '--retry-jitter-sec', '0'])
    vars(args).update(changes)
    return script._fetch_macro_range(SimpleNamespace(repo_daily=operation), script.RateLimiter(10000), args,
                                    api_name='repo_daily', start_date=args.start_date, end_date=args.end_date,
                                    chunk_days=90, output_dir=tmp_path)


def row(day):
    return {'ts_code': '204001.SH', 'trade_date': day, 'close': 1.5}


def test_cap_splits_both_halves_and_resume_preserves_original_collection_time(tmp_path):
    calls = []
    def operation(start_date, end_date):
        calls.append((start_date, end_date))
        rows = [row(d.strftime('%Y%m%d')) for d in pd.date_range(start_date, end_date)]
        return pd.DataFrame(rows[:2])
    operation.download_policy = DownloadPolicy(max_rows_per_request=2)
    result = run(tmp_path, operation)
    assert result.trade_date.tolist() == ['20260901', '20260902', '20260903', '20260904']
    assert len(calls) == 7
    assert result.availability_status.eq('date_only').all()
    calls.clear()
    repeated = run(tmp_path, operation)
    pd.testing.assert_frame_equal(result, repeated)
    assert not calls
    with pytest.raises(CenterError) as error: run(tmp_path, operation, resume=False)
    assert error.value.code == 'MACRO_RESUME_REQUIRED'


def test_completed_leaves_are_reused_after_later_connection_failure(tmp_path):
    calls, failing = [], [True]
    def operation(start_date, end_date):
        calls.append((start_date, end_date))
        if start_date == '20260903' and failing[0]: raise ConnectionError('offline connection')
        return pd.DataFrame([row(d.strftime('%Y%m%d')) for d in pd.date_range(start_date, end_date)][:2])
    operation.download_policy = DownloadPolicy(max_rows_per_request=2)
    with pytest.raises(CenterError): run(tmp_path, operation)
    failing[0] = False
    calls.clear()
    assert len(run(tmp_path, operation)) == 4
    assert all(start >= '20260903' for start, _ in calls)


def test_empty_is_independently_confirmed_then_cached(tmp_path):
    calls = []
    def operation(**params):
        calls.append(params)
        return pd.DataFrame()
    assert run(tmp_path, operation).empty
    assert len(calls) == 2
    assert run(tmp_path, operation).empty and len(calls) == 2


def test_save_all_rate_tables_uses_receipts_and_keeps_raw_units(tmp_path):
    calls = []
    class Pro:
        def shibor(self, **params):
            calls.append('shibor')
            return pd.DataFrame([{'date': '20260901', 'on': 1.5}])
        def shibor_lpr(self, **params):
            calls.append('lpr')
            return pd.DataFrame([{'date': '20260901', '1y': 3.0}])
        def repo_daily(self, **params):
            calls.append('repo')
            return pd.DataFrame([row('20260901')])
    args = script.parse_args(['--start-date', '20260901', '--end-date', '20260904', '--resume'])
    script.save_macro_rates(Pro(), tmp_path, script.RateLimiter(10000), args)
    assert len(calls) == 3
    assert pd.read_parquet(tmp_path / 'macro_repo_daily_df.parquet').close.tolist() == [1.5]
    before = {p.name: p.read_bytes() for p in tmp_path.glob('*.parquet')}
    script.save_macro_rates(Pro(), tmp_path, script.RateLimiter(10000), args)
    assert len(calls) == 3
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('*.parquet')}


def test_worker_clears_only_the_capped_parent_after_both_children_succeed(store, monkeypatch):
    from backend.data_sources import task_worker
    from backend.data_sources.acquisition import fingerprint
    from backend.data_sources.runtime import ConfiguredTushareClient
    from backend.data_sources.task_catalog import get_task
    calls = []
    def operation(self, interface, **params):
        calls.append(params)
        if params['start_date'] != params['end_date']:
            raise CenterError('SOURCE_ROW_CAP', 'offline configured cap')
        return pd.DataFrame([row(params['start_date'])])
    monkeypatch.setattr(ConfiguredTushareClient, '_call', operation)
    def actions(args, actions, client):
        result = script._fetch_macro_range(client, script.RateLimiter(10000), args, api_name='repo_daily',
                    start_date=args.start_date, end_date=args.end_date, chunk_days=90, output_dir=store.root)
        assert len(result) == 2
    monkeypatch.setattr(script, '_run_actions', actions)
    current = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    result = task_worker.acquire({'root':str(store.root), 'source_id':'tushare',
                                'source_hash':fingerprint(current), 'mode':'full',
                                'params':{'start_date':'20260901', 'end_date':'20260902'}},
                               get_task('tushare.macro_rates'), store.root)
    assert len(calls) == 3 and result['warnings'] == 0


@pytest.mark.parametrize('case', ['day_cap', 'outside', 'duplicate', 'missing', 'permission', 'budget', 'split_empty'])
def test_invalid_partition_fails_closed(tmp_path, case):
    calls = []
    def operation(start_date, end_date):
        calls.append((start_date, end_date))
        if case == 'permission': raise CenterError('SOURCE_PERMISSION_OR_PARAMS', 'offline denied')
        if case in {'day_cap', 'budget', 'split_empty'}:
            if case == 'split_empty' and start_date == end_date: return pd.DataFrame()
            return pd.DataFrame([row('20260901'), row('20260902')])
        if case == 'outside': return pd.DataFrame([row('20260831')])
        if case == 'duplicate': return pd.DataFrame([row('20260901'), row('20260901')])
        return pd.DataFrame([{'close': 1}])
    operation.download_policy = DownloadPolicy(max_rows_per_request=2 if case in {'day_cap','budget','split_empty'} else 1000)
    with pytest.raises(CenterError) as error:
        run(tmp_path, operation, **({'smoke': True} if case == 'budget' else {}))
    assert error.value.code == {'day_cap': 'MACRO_DAY_ROW_CAP', 'budget': 'MACRO_REQUEST_BUDGET',
                                'permission': 'SOURCE_PERMISSION_OR_PARAMS'}.get(case, 'MACRO_RANGE_INVALID')
    if case == 'permission': assert len(calls) == 1


@pytest.mark.parametrize('case', ['tamper', 'orphan', 'symlink'])
def test_checkpoint_corruption_never_refetches_or_overwrites(tmp_path, case):
    calls = []
    def operation(**params):
        calls.append(params)
        return pd.DataFrame([row('20260901')])
    run(tmp_path, operation)
    path = next(tmp_path.rglob('*.parquet'))
    if case == 'tamper': path.write_bytes(b'broken')
    if case == 'orphan': path.with_suffix('.json').unlink()
    if case == 'symlink':
        original = path.with_suffix('.saved')
        path.rename(original); path.symlink_to(original)
    calls.clear()
    with pytest.raises(CenterError): run(tmp_path, operation)
    assert not calls
