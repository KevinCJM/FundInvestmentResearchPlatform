"""Automatic planning and execution: temporary snapshots, no vendor traffic."""
import json
import time
import uuid
from datetime import date, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.data_sources import auto_incremental as auto, etl_service as etl, task_runtime
from backend.data_sources.acquisition import fingerprint
from backend.data_sources.credentials import save_credential
from backend.data_sources.etl_models import EtlDefinition, EtlParameter, EtlStep
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.store import SourceStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(etl, '_launch', etl._launch_inline)
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    monkeypatch.delenv('TUSHARE_DATA_DIR', raising=False)
    monkeypatch.setattr(auto, 'cutoff_date', lambda: date(2024, 1, 14))
    store = SourceStore(tmp_path)
    store.seed()
    save_credential(store, 'tushare', 'offline-fixture')
    snapshot = tmp_path / 'snapshot'; snapshot.mkdir()
    (tmp_path / 'tushare_active.json').write_text(json.dumps({'schema_version': 1, 'snapshot_dir': 'snapshot'}))
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(31)]
    pq.write_table(pa.table({'exchange': ['SSE'] * 31, 'cal_date': [d.strftime('%Y%m%d') for d in days],
                            'is_open': [int(d.weekday() < 5) for d in days]}), snapshot / 'trade_day_df.parquet')
    pq.write_table(pa.table({'ts_code': ['000001.OF', '000001.OF'], 'date': [date(2024, 1, 1), date(2024, 1, 10)],
                            'adj_nav': [1., 1.1]}), snapshot / 'fund_nav_df.parquet')
    return store


def definition():
    steps = []
    for i, action in enumerate(('calendar', 'fund_info', 'fund_nav')):
        steps.append(EtlStep(id=action, name=action, kind='task', task_id='tushare.' + action,
                             source_id='tushare', inputs=[steps[-1].id] if steps else [],
                             parameter_bindings={'start_date': 'start', 'end_date': 'end'}))
    return EtlDefinition(name='auto-test', steps=steps, parameters=[
        EtlParameter(id='start', label='开始', data_type='date', default='1999-01-01', date_format='compact'),
        EtlParameter(id='end', label='结束', data_type='date', date_format='compact')]).model_dump(mode='json')


OPTIONS = {'mode': 'auto_incremental', 'parameters': {}}


def test_auto_ignores_dates_and_uses_actual_metadata(store):
    checked = etl.validate(store, definition(), OPTIONS)
    assert checked['valid'], checked
    step = checked['auto_plan']['steps'][-1]
    assert step['latest_date'] == '2024-01-10'
    assert (step['start_date'], step['end_date']) == ('20240104', '20240114')
    assert not (store.root / 'etl_runs').exists()


@pytest.mark.parametrize('invalid', ['missing', 'empty', 'corrupt_manifest', 'future', 'symlink', 'stale'])
def test_invalid_baseline_never_falls_back_to_full(store, invalid):
    file = store.root / 'snapshot' / 'fund_nav_df.parquet'
    if invalid == 'missing': file.unlink()
    if invalid == 'empty': pq.write_table(pa.table({'date': pa.array([], type=pa.date32())}), file)
    if invalid == 'corrupt_manifest': (store.root / 'tushare_active.json').write_text('{}')
    if invalid == 'future': pq.write_table(pa.table({'date': [date(2024, 2, 1)]}), file)
    if invalid == 'stale': pq.write_table(pa.table({'date': [date(2020, 1, 1)]}), file)
    if invalid == 'symlink':
        file.rename(store.root / 'outside.parquet'); file.symlink_to(store.root / 'outside.parquet')
    checked = etl.validate(store, definition(), OPTIONS)
    assert not checked['valid'], checked
    assert not (store.root / 'etl_runs').exists()


def test_each_dataset_has_own_date_and_no_unrelated_baseline_copy(store):
    pq.write_table(pa.table({'date': [date(2024, 1, 12)]}), store.root / 'snapshot' / 'etf_daily_df.parquet')
    pq.write_table(pa.table({'date': [date(2020, 1, 1)]}), store.root / 'snapshot' / 'unrelated.parquet')
    result = auto.plan(store, etl.parse_definition(definition()))
    assert set(result['baseline']['files']) == {'fund_nav_df.parquet', 'trade_day_df.parquet'}
    steps = etl.parse_definition(definition()).steps
    steps.extend([EtlStep(id='etf_info', name='ETF', kind='task', task_id='tushare.etf_info', source_id='tushare', inputs=['fund_nav']),
                  EtlStep(id='nav', name='净值', kind='task', task_id='tushare.nav', source_id='tushare', inputs=['etf_info'])])
    result = auto.plan(store, EtlDefinition(name='mixed', steps=steps, parameters=etl.parse_definition(definition()).parameters))['public']
    assert result['steps'][-1]['latest_date'] == '2024-01-12'
    assert result['steps'][2]['latest_date'] == '2024-01-10'


def test_plan_must_be_previewed_and_rejects_file_change(store):
    payload = {'request_id': uuid.uuid4().hex, 'confirm': True, 'definition': definition(), 'options': OPTIONS}
    with pytest.raises(CenterError, match='预览'):
        etl.start(store, payload)
    checked = etl.validate(store, definition(), OPTIONS)
    payload['auto_plan_id'] = checked['auto_plan']['plan_id']
    pq.write_table(pa.table({'date': [date(2024, 1, 11)]}), store.root / 'snapshot' / 'fund_nav_df.parquet')
    with pytest.raises(CenterError, match='预览'):
        etl.start(store, payload)


def test_real_etl_path_passes_frozen_snapshot_and_auto_window(store, monkeypatch):
    calls = []
    original = (store.root / 'snapshot' / 'fund_nav_df.parquet').read_bytes()
    def worker(payload, check, lock):
        check(); calls.append(payload)
        assert payload['has_baseline'] and payload['mode'] == 'auto_incremental'
        work = Path(payload['directory'])
        if payload['task_id'] == 'tushare.fund_nav':
            assert payload['auto_step']['start_date'] == '20240104'
            assert (work / 'fund_nav_df.parquet').read_bytes() == original
            pq.write_table(pa.table({'date': [date(2024, 1, 12)]}), work / 'fund_nav_df.parquet')
        return {'received_rows': 1, 'warnings': 0, 'mode': 'auto_incremental'}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    checked = etl.validate(store, definition(), OPTIONS)
    result = etl.start(store, {'request_id': uuid.uuid4().hex, 'confirm': True, 'definition': definition(),
                              'options': OPTIONS, 'auto_plan_id': checked['auto_plan']['plan_id']})
    journal = EtlStore(store)
    for _ in range(300):
        result = journal.get_run(result['run_id'])
        if result['status'] != 'RUNNING': break
        time.sleep(.02)
    assert result['status'] == 'SUCCEEDED', result.get('error')
    assert len(calls) == 3
    assert result['auto_plan']['snapshot'] == 'snapshot'
    assert (store.root / 'snapshot' / 'fund_nav_df.parquet').read_bytes() == original


def test_worker_missing_code_backfill_then_frozen_date_update(store, monkeypatch):
    import T01_get_data as script
    from backend.data_sources.task_worker import acquire
    from backend.data_sources.task_catalog import task_specs
    calls = []
    monkeypatch.setattr(script, '_run_actions', lambda args, actions, **kwargs: calls.append(args))
    records = [r for kind in ('source', 'interface') for r in store.list(kind) if r['config'].get('source_id', r['config']['id']) == 'tushare']
    payload = {'root': str(store.root), 'source_id': 'tushare', 'source_hash': fingerprint(records),
               'params': {'start_date': '20240104', 'end_date': '20240114'}, 'mode': 'auto_incremental',
               'has_baseline': True, 'auto_step': {'start_date': '20240104', 'history_start': '20100101'}}
    result = acquire(payload, task_specs()['tushare.fund_nav'], store.root / 'work')
    assert calls[0].missing_only and not calls[0].latest and calls[0].start_date == '20100101'
    assert calls[1].latest and calls[1].automatic_start_date == '20240104'
    assert result['mode'] == 'auto_incremental'
    assert calls[1].max_workers <= 16


def test_parquet_without_statistics_is_supported(tmp_path):
    file = tmp_path / 'no_stats.parquet'
    pq.write_table(pa.table({'date': ['20240102', None, '20240105']}), file, write_statistics=False)
    assert auto.date_bounds(file, 'date') == (date(2024, 1, 2), date(2024, 1, 5))


def test_calendar_refresh_is_explicit_for_outdated_calendar(store, monkeypatch):
    monkeypatch.setattr(auto, 'cutoff_date', lambda: date(2024, 2, 5))
    checked = etl.validate(store, definition(), OPTIONS)
    assert checked['valid'], checked
    assert checked['auto_plan']['steps'][0]['start_date'] == '20240201'
    assert checked['auto_plan']['steps'][0]['end_date'] == '20240205'


def test_auto_preview_api_is_read_only_and_redacted(store, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.services import etl_routes
    app = FastAPI(); app.include_router(etl_routes.router)
    monkeypatch.setattr(etl_routes, 'get_store', lambda: store)
    response = TestClient(app).post('/api/data-sources/etl/validate', json={'definition': definition(), 'options': OPTIONS})
    assert response.status_code == 200 and response.json()['valid']
    assert 'offline-fixture' not in response.text
    assert not (store.root / 'etl_runs').exists()


def test_collector_uses_frozen_window_even_after_backfill_advanced_latest(store, monkeypatch):
    import T01_get_data as script
    args = script.parse_args(['--output-dir', str(store.root / 'snapshot'), '--start-date', '20240104', '--end-date', '20240114'])
    args.automatic_start_date = '20240104'
    captured = []
    monkeypatch.setattr(script, 'load_open_trade_dates', lambda *a, **kw: captured.append(kw) or [])
    monkeypatch.setattr(script, 'latest_parquet_date', lambda *a: pytest.fail('must use frozen window'))
    script.save_latest_public_fund_nav(None, args.output_dir, None, args)
    assert captured[0]['start_date'] == '20240104'
