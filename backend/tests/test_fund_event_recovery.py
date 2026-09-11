"""Offline v4 migration proof: reuse validated leaves, fetch only missing work."""
import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import T01_get_data as script
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.fund_event_recovery import import_history_receipts, import_dividend_receipts
from backend.data_sources.fund_events import FundEventDownload
from backend.data_sources.models import CenterError
from backend.data_sources.store import SourceStore


def dividend_fixture(tmp_path):
    journal = EtlStore(SourceStore(tmp_path))
    parts, dest = tmp_path / 'old', tmp_path / 'new' / 'parts'
    dest.mkdir(parents=True)
    universe = pd.DataFrame([{'ts_code': 'A', 'found_date': '20240101'}])
    universe.to_parquet(dest.parent / 'fund_info_df.parquet')
    dates = ['20240101', '20240102', '20240103']
    session = FundEventDownload(directory=parts, dates=dates, universe=universe,
                                api_name='fund_div', fields=script.FUND_DIVIDEND_FIELDS, smoke=False)
    frame = script._prepare_fund_event_rows(pd.DataFrame([dict(ts_code='A', ann_date='20240102',
        ex_date='20240103', pay_date='20240105', div_cash=0.1)]), fields=script.FUND_DIVIDEND_FIELDS,
        source_api='fund_div', observation_column='ex_date')
    for day, value in [('20240101', frame.iloc[:0]), ('20240102', frame)]:
        path, _ = session._paths(day, None)
        script.save_dataframe(value, path, quiet=True)
        session._record(day, None, 'COMPLETE', rows=len(value), sha256=journal.artifact(path)['checksum'])
    session._record('20240103', None, 'EMPTY', confirmations=2)
    step = SimpleNamespace(params={'start_date': dates[0], 'end_date': dates[-1]})
    return journal, parts, dest, universe, session, frame, step


def test_dividend_recovery_reuses_all_dates_without_requests_and_keeps_evidence(tmp_path):
    journal, parts, dest, universe, old, frame, step = dividend_fixture(tmp_path)
    evidence = import_dividend_receipts(journal, parts, dest, step)
    assert evidence['complete'] == 2 and evidence['empty'] == 1
    session = FundEventDownload(directory=dest, dates=old.dates, universe=universe,
                                api_name='fund_div', fields=old.fields, smoke=False)
    paths = session.run(fetch=lambda *args: pytest.fail('Must reuse downloaded dividends'),
                        prepare=lambda value: value, save=script.save_dataframe,
                        cap_error=script.ResponseTruncatedError, max_workers=2)
    output = tmp_path / 'result.parquet'
    assert script._consolidate_ordered_parts(paths, output) == 1
    assert pq.read_table(output).equals(pa.Table.from_pandas(frame, preserve_index=False), check_metadata=False)
    for item in evidence['files']:
        assert journal.checked_path(item['source']).read_bytes() == journal.checked_path(item['imported']).read_bytes()


@pytest.mark.parametrize('case', ['checksum', 'rows', 'available', 'code', 'source', 'observation',
                                  'empty', 'contract', 'symlink', 'split', 'request_day', 'missing_column'])
def test_dividend_import_rejects_invalid_evidence(tmp_path, case):
    journal, parts, dest, universe, old, frame, step = dividend_fixture(tmp_path)
    path, receipt = old._paths('20240102', None)
    if case in {'available', 'code', 'source', 'observation', 'missing_column'}:
        if case == 'available': frame['available_at'] = pd.Timestamp('20250101')
        if case == 'code': frame['ts_code'] = 'B'
        if case == 'source': frame['source_api'] = 'fund_nav'
        if case == 'observation': frame['observation_date'] = pd.Timestamp('20250101')
        if case == 'missing_column': frame = frame.drop(columns=['ingested_at'])
        script.save_dataframe(frame, path, quiet=True)
        old._record('20240102', None, 'COMPLETE', rows=1, sha256=journal.artifact(path)['checksum'])
    elif case == 'checksum': path.write_bytes(b'changed')
    elif case in {'rows', 'request_day'}:
        record = json.loads(receipt.read_text())
        record['rows' if case == 'rows' else 'date'] = 2 if case == 'rows' else '20250101'
        receipt.write_text(json.dumps(record))
    elif case == 'empty': old._record('20240103', None, 'EMPTY', confirmations=1)
    elif case == 'contract': (old.directory / 'contract.json').write_text('{}')
    elif case == 'split': old._record('20240102', None, 'SPLIT')
    elif case == 'symlink':
        other = tmp_path / 'other.parquet'
        path.rename(other)
        path.symlink_to(other)
    with pytest.raises(CenterError):
        import_dividend_receipts(journal, parts, dest, step)


def fixture(tmp_path):
    journal = EtlStore(SourceStore(tmp_path))
    parts, dest = tmp_path / 'old-parts', tmp_path / 'new' / 'parts'
    dest.mkdir(parents=True)
    universe = pd.DataFrame([{'ts_code': code, 'found_date': '20240101'} for code in ['A', 'B', 'C']])
    universe.to_parquet(dest.parent / 'fund_info_df.parquet')
    fields = script.FUND_PORTFOLIO_FIELDS
    session = FundEventDownload(directory=parts, dates=['20240101', '20240102'], universe=universe,
                                api_name='fund_portfolio', fields=fields, smoke=False, strategy='fund_announcement_history')
    frame = script._prepare_fund_event_rows(pd.DataFrame([dict(ts_code='A', ann_date='20240102',
        end_date='20231231', symbol='600000.SH', mkv=1., amount=2.)]), fields=fields,
        source_api='fund_portfolio', observation_column='end_date')
    path = session._paths('20240101-20240102', 'A')[0]
    script.save_dataframe(frame, path, quiet=True)
    session._record('20240101-20240102', 'A', 'COMPLETE', rows=1, sha256=journal.artifact(path)['checksum'])
    session._record('20240101-20240102', 'B', 'EMPTY', confirmations=2)
    (session.directory / 'failure.json').write_text('{}')
    (session.directory / 'unfinished.parquet.tmp').write_text('incomplete')
    step = SimpleNamespace(params={'start_date':'20240101', 'end_date':'20240102'})
    return journal, parts, dest, universe, session, frame, step


def test_v4_import_reuses_complete_and_confirmed_empty_then_fetches_missing(tmp_path):
    journal, parts, dest, universe, old, frame, step = fixture(tmp_path)
    evidence = import_history_receipts(journal, parts, dest, step)
    assert evidence['complete'] == evidence['empty'] == 1
    assert not list(dest.rglob('failure.json')) and not list(dest.rglob('*.tmp'))
    for item in evidence['files']:
        assert journal.checked_path(item['source']).read_bytes() == journal.checked_path(item['imported']).read_bytes()
    new = FundEventDownload(directory=dest, dates=old.dates, universe=universe, api_name='fund_portfolio',
                            fields=old.fields, smoke=False, strategy='fund_announcement_history')
    calls = []
    def fetch(code, start, end):
        calls.append(code)
        assert code == 'C'
        return pd.DataFrame([dict(ts_code=code, ann_date='20240102', end_date='20231231', symbol='600000.SH')])
    def prepare(value):
        return script._prepare_fund_event_rows(value, fields=old.fields, source_api='fund_portfolio', observation_column='end_date')
    paths = new.run_history(fetch=fetch, prepare=prepare, save=script.save_dataframe,
                           cap_error=script.ResponseTruncatedError, max_workers=1,
                           sort_columns=['available_at', 'ts_code', 'end_date', 'symbol'])
    assert calls == ['C']
    assert pd.read_parquet(paths[0]).ts_code.tolist() == ['A', 'C']


@pytest.mark.parametrize('case', ['checksum','rows','outside','available','source','code','report','empty','contract','symlink'])
def test_v4_import_rejects_invalid_evidence(tmp_path, case):
    journal, parts, dest, universe, old, frame, step = fixture(tmp_path)
    path, receipt = old._paths('20240101-20240102', 'A')
    if case in {'outside','available','source','code','report'}:
        if case == 'outside': frame['ann_date'] = pd.Timestamp('20250101')
        if case == 'available': frame['available_at'] = pd.Timestamp('20240101')
        if case == 'source': frame['source_api'] = 'fund_nav'
        if case == 'code': frame['ts_code'] = 'B'
        if case == 'report': frame['observation_date'] = pd.Timestamp('20230101')
        script.save_dataframe(frame, path, quiet=True)
        old._record('20240101-20240102', 'A', 'COMPLETE', rows=1, sha256=journal.artifact(path)['checksum'])
    elif case == 'checksum': path.write_bytes(b'changed')
    elif case == 'rows':
        record=json.loads(receipt.read_text());record['rows']=2;receipt.write_text(json.dumps(record))
    elif case == 'empty': old._record('20240101-20240102', 'B', 'EMPTY', confirmations=1)
    elif case == 'contract': (old.directory / 'contract.json').write_text('{}')
    elif case == 'symlink':
        content=path.read_bytes();path.unlink();other=tmp_path/'other.parquet';other.write_bytes(content);path.symlink_to(other)
    with pytest.raises(CenterError):
        import_history_receipts(journal, parts, dest, step)


def test_v3_is_not_imported_as_v4(tmp_path):
    journal, parts, dest, universe, old, frame, step = fixture(tmp_path)
    old.directory.rename(parts / 'events_v3_wrong_axis')
    assert import_history_receipts(journal, parts, dest, step)['files'] == []


def test_single_day_split_is_immutable_audit_not_success_or_working_cache(tmp_path):
    journal, parts, dest, universe, old, frame, step = fixture(tmp_path)
    old._record('20240102-20240102', 'C', 'SPLIT')
    source = old._paths('20240102-20240102', 'C')[1]
    before = source.read_bytes()
    evidence = import_history_receipts(journal, parts, dest, step)
    assert evidence['requery'] == 1 and evidence['split'] == 0
    copied = dest / old.directory.name
    assert not (copied / source.name).exists()
    audit = copied / 'requery_evidence' / source.name
    assert audit.read_bytes() == before == source.read_bytes()
    assert any(journal.checked_path(item['imported']) == audit for item in evidence['files'])
    new = FundEventDownload(directory=dest, dates=old.dates, universe=universe, api_name='fund_portfolio',
                            fields=old.fields, smoke=False, strategy='fund_announcement_history')
    new._record('20240102-20240102', 'C', 'COMPLETE', rows=1)
    for item in evidence['files']:
        journal.checked_path(item['imported'])  # A later successful retry cannot invalidate imported evidence.
