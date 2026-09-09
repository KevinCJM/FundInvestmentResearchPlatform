"""Offline THS checkpoint recovery through the normal guarded resume path."""
import copy
import json
import uuid
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

import T01_get_data as script
from backend.data_sources import etl_service as etl, task_catalog, task_runtime
from backend.data_sources.etl_migration import stage_recovery
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.runtime import ConfiguredTushareClient
from backend.data_sources.task_workspace import read_inventory
from backend.tests.test_etl_dataset_tasks import store, small_plan, request, finish


@pytest.fixture
def stopped_index(store, monkeypatch):
    specs = copy.deepcopy(task_catalog.task_specs())
    specs['tushare.calendar']['provides'].append('index_catalog')
    monkeypatch.setattr(task_catalog, 'task_specs', lambda: specs)
    plan = small_plan()
    plan['steps'][1].update(task_id='tushare.index_concept')
    for step in plan['steps']:
        step['params'] = {'start_date': '20000101', 'end_date': '20100101'}
    calls, fail = [], [True]

    def fetch(**params):
        calls.append(params)
        code = params['ts_code']
        if code == 'C.TI' and fail[0]:
            raise CenterError('SOURCE_DB_BUSY', 'offline lock timeout')
        if code == 'B.TI':
            return pd.DataFrame()
        return pd.DataFrame([{'ts_code': code, 'trade_date': params['start_date'], 'close': 1.0}])

    def worker(payload, check, lock):
        check()
        path = Path(payload['directory'])
        if payload['task_id'] == 'tushare.calendar':
            pd.DataFrame({'ts_code': ['A.TI', 'B.TI', 'C.TI'], 'name': ['A', 'B', 'C'],
                          'quote_source_api': ['ths_daily'] * 3}).to_parquet(path / 'index_catalog_df.parquet')
        else:
            args = script.parse_args(['--start-date', '20000101', '--end-date', '20100101', '--max-workers', '1'])
            args.source_configuration_hash = ConfiguredTushareClient('offline-test-credential', root=store.root).configuration_hash
            script.save_index_history_api(type('Pro', (), {'ths_daily': staticmethod(fetch)})(),
                                          path, script.RateLimiter(100000), args, 'ths_daily')
        return {'received_rows': 1, 'warnings': 0, 'mapped_rejected_batches': 0}

    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    old = finish(store, etl.start(store, request(plan)))
    assert old['status'] == 'FAILED'
    fail[0] = False
    calls.clear()
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'new-execution')
    work = store.root / 'etl_runs' / old['run_id'] / 'company' / 'work'
    parts = next(work.glob('.*_parts_*'))
    return old, work, parts, calls


@pytest.mark.parametrize('tampered', [None, 'parquet', 'empty'])
def test_verified_index_resume_preserves_nonempty_and_rechecks_unproven_empty(store, stopped_index, tampered):
    old, work, parts, calls = stopped_index
    original = copy.deepcopy(old)
    original_bytes = {p.name: p.read_bytes() for p in (parts / 'segments').iterdir()}
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    journal = EtlStore(store)
    receipt = json.loads(journal.checked_path(journal.get_run(new['run_id'])['recovery_receipt']).read_text())
    shards = receipt['shards']
    assert (shards['copied'], shards['empty_recheck'], shards['missing_segments']) == (2, 2, 2)
    assert calls == []
    if tampered:
        imported = next(x['imported'] for x in shards['files'] if x['imported']['path'].endswith('.' + tampered))
        journal.checked_path(imported).write_bytes(b'changed')
        with pytest.raises(CenterError, match='校验和'):
            etl.resume(store, new['run_id'], True)
        assert calls == []
    else:
        done = finish(store, etl.resume(store, new['run_id'], True))
        assert done['status'] == 'SUCCEEDED', done
        assert len(calls) == 6  # C: two missing ranges; B: two independently confirmed empties.
        assert {c['ts_code'] for c in calls} == {'B.TI', 'C.TI'}
        value = read_inventory(journal, done['steps'][1]['output']['workspace'])
        result = pd.read_parquet(journal.checked_path(value['files']['index_ths_daily_df.parquet']))
        assert len(result) == 4 and set(result.ts_code) == {'A.TI', 'C.TI'}
    assert journal.get_run(old['run_id']) == original
    assert {p.name: p.read_bytes() for p in (parts / 'segments').iterdir()} == original_bytes
    assert not (store.root / 'tushare_active.json').exists()


@pytest.mark.parametrize('case', ['code', 'source', 'date', 'duplicate', 'columns', 'empty_parquet',
                                 'truncated', 'unknown_file', 'unknown_chunk', 'symlink', 'conflict',
                                 'bad_empty', 'catalog', 'marker', 'later_api', 'per_code'])
def test_index_import_rejects_invalid_or_unsupported_checkpoints(store, stopped_index, case):
    old, work, parts, calls = stopped_index
    path = next((parts / 'segments').glob('*.parquet'))
    frame = pd.read_parquet(path)
    if case == 'code': frame['ts_code'] = 'OTHER.TI'
    if case == 'source': frame['source_api'] = 'index_daily'
    if case == 'date': frame['trade_date'] = pd.Timestamp('1999-01-01')
    if case == 'duplicate': frame = pd.concat([frame, frame])
    if case == 'columns': frame = frame.drop(columns='close')
    if case == 'empty_parquet': frame = frame.iloc[:0]
    if case in {'code', 'source', 'date', 'duplicate', 'columns', 'empty_parquet'}:
        frame.to_parquet(path)
    elif case == 'truncated': path.write_bytes(b'broken')
    elif case == 'unknown_file': (parts / 'segments' / 'unknown.tmp').write_text('partial')
    elif case == 'unknown_chunk': path.rename(path.with_name('A.TI__19991231_20000101.parquet'))
    elif case == 'symlink': (parts / 'segments' / 'D.TI__20000101_20091228.parquet').symlink_to(path)
    elif case == 'conflict': script.mark_empty_checkpoint(path.with_suffix('.empty'))
    elif case == 'bad_empty': next((parts / 'segments').glob('*.empty')).write_text('changed')
    elif case == 'catalog': (work / 'index_catalog_df.parquet').write_bytes(b'changed')
    elif case == 'marker': (work.parent / 'work_input.json').write_text('{}')
    elif case == 'later_api': (work / '.index_dc_daily_df_parts_unknown').mkdir()
    elif case == 'per_code': frame.to_parquet(parts / 'A.TI.parquet')
    with pytest.raises((CenterError, pa.ArrowException)):
        stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    assert calls == []
