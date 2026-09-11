"""Offline bounded-download and configuration-ownership regression tests."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from backend.data_sources import service, sync_jobs
from backend.data_sources.models import CenterError, InterfaceConfig, SourceConfig
from backend.data_sources.store import SourceStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    store = SourceStore(tmp_path)
    store.seed()
    return store


def finished(store, job):
    for _ in range(300):
        result = next(j for j in sync_jobs.list_jobs(store) if j['job_id'] == job['job_id'])
        if result['status'] != 'RUNNING': return result
        time.sleep(.01)
    pytest.fail('offline task did not complete')


def test_saved_configuration_survives_initialization_and_deletion(store):
    entry = store.get('interface', 'tushare.fund_daily')
    config = InterfaceConfig.model_validate(entry['config'])
    config.api_name = 'another_api'; config.entitlement_confirmed = True
    config.response.records_path = 'data.rows'
    config.pagination.mode = 'page'; config.pagination.cursor_param = 'page'
    saved = service.save(store, 'interface', config.model_dump(mode='json'), entry['revision'])
    store.seed()
    assert store.get('interface', config.id) == saved
    store.delete('interface', config.id, saved['revision'])
    store.seed()
    with pytest.raises(CenterError): store.get('interface', config.id)


def test_multiple_configs_can_use_one_api(store):
    config = InterfaceConfig.model_validate(store.get('interface', 'tushare.fund_daily')['config'])
    config.id = 'tushare.another_quote_scope'
    config.params = {'ts_code': '159915.SZ'}
    assert service.save(store, 'interface', config.model_dump(mode='json'), 0)['revision'] == 1


def test_sync_honors_checkpoint_and_resolves_candidates(store, monkeypatch):
    record = store.get('interface', 'akshare.fund_nav')
    calls = []
    def fetch(_store, _source, _interface, params):
        calls.append(params)
        return {}, [dict(symbol='000001', 净值日期='2024-01-05', 单位净值=1.25)]
    monkeypatch.setattr(sync_jobs, 'fetch_with_retry', fetch)
    for mode in ('full', 'incremental'):
        job = sync_jobs.start_sync(store, record['config']['id'], record['revision'], {'start_date':'20240101', 'end_date':'20240110'}, mode)
        result = finished(store, job)
        assert result['status'] == 'SUCCEEDED', result
        assert result['batch']['status'] == 'VALIDATED_CANDIDATE'
        assert result['resolutions'][0]['status'] == 'CANDIDATE_READY'
    assert calls[1]['start_date'] == '20240102'
    assert not (store.root / 'tushare_active.json').exists()


def test_empty_response_is_not_download_success(store, monkeypatch):
    monkeypatch.setattr(sync_jobs, 'fetch_with_retry', lambda *args: ({}, []))
    record = store.get('interface', 'akshare.fund_nav')
    result = finished(store, sync_jobs.start_sync(store, record['config']['id'], record['revision'], {}, 'incremental'))
    assert result['status'] == 'EMPTY'
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_sync_checkpoint').fetchone()[0] == 0


def test_preflight_revision_and_disabled_interface(store):
    record = store.get('interface', 'akshare.fund_nav')
    with pytest.raises(CenterError, match='修改'):
        sync_jobs.start_sync(store, record['config']['id'], 0, {}, 'full')
    config = InterfaceConfig.model_validate(record['config']); config.enabled = False
    saved = store.save(config, record['revision'])
    with pytest.raises(CenterError, match='启用'):
        sync_jobs.start_sync(store, config.id, saved['revision'], {}, 'full')


def test_exhausted_pagination_cannot_save_partial_data(store, monkeypatch):
    source = SourceConfig(id='sample', name='Offline fixture', base_url='https://example.com')
    store.save(source, 0)
    config = InterfaceConfig.model_validate(store.get('interface', 'tushare.fund_daily')['config'])
    config.id = 'sample.quotes'; config.source_id = source.id
    config.pagination.mode = 'offset'; config.pagination.page_size = 1; config.pagination.max_pages = 1
    saved = store.save(config, 0)
    monkeypatch.setattr(sync_jobs, 'fetch_with_retry', lambda *args: ({}, [dict(ts_code='510300.SH', trade_date='20240105', close=11)]))
    result = finished(store, sync_jobs.start_sync(store, config.id, saved['revision'], {}, 'full'))
    assert result['code'] == 'SOURCE_PAGINATION_INCOMPLETE'
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_run').fetchone()[0] == 0
