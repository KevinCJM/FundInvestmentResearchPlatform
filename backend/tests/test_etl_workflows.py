"""Offline ETL end-to-end orchestration, isolation and recovery contracts."""
from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
if str(ROOT / 'backend') not in sys.path: sys.path.insert(0, str(ROOT / 'backend'))
from backend.data_sources import acquisition, etl_service as etl
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError, InterfaceConfig
from backend.data_sources.store import SourceStore
from backend.services import etl_routes


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(etl, '_launch', etl._launch_inline)
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    store = SourceStore(tmp_path); store.seed()
    return store


def plan(store, *, kind='nav', name='Offline ETL'):
    record = store.get('interface', 'akshare.fund_nav' if kind == 'nav' else 'akshare.etf_daily')
    table = 'market.nav_daily' if kind == 'nav' else 'market.quote_daily'
    return {'name': name, 'steps': [
        {'id': 'download', 'name': '下载', 'kind': 'download', 'source_id': 'akshare', 'interface_id': record['config']['id'], 'interface_revision': record['revision'], 'params': {'start_date': '20240101', 'end_date': '20240110'}},
        {'id': 'mapping', 'name': '映射', 'kind': 'map', 'inputs': ['download']},
        {'id': 'resolve', 'name': '取值', 'kind': 'resolve', 'inputs': ['mapping'], 'table_id': table, 'include_history': False},
    ]}


def payload(definition):
    return {'definition': definition, 'confirm': True, 'request_id': str(uuid.uuid4())}


def finished(store, run):
    for _ in range(1000):
        result = EtlStore(store).get_run(run['run_id'])
        if result['status'] != 'RUNNING': return result
        time.sleep(.01)
    pytest.fail('offline task timed out')


def nav_fetch(*args, **kwargs):
    return {}, [{'symbol': '000001', '净值日期': '2024-01-05', '单位净值': 1.25}]


def test_ordered_download_map_resolve_and_repeated_submission(store, monkeypatch):
    calls = []
    def fetch(*args, **kwargs):
        calls.append(args[3]); return nav_fetch()
    monkeypatch.setattr(acquisition, 'fetch_with_retry', fetch)
    request = payload(plan(store))
    run = finished(store, etl.start(store, request))
    assert run['status'] == 'SUCCEEDED', run
    assert [s['status'] for s in run['steps']] == ['SUCCEEDED'] * 3
    assert calls[0]['start_date'] == '20240101'
    assert len(calls) == 1
    assert etl.start(store, request)['run_id'] == run['run_id']
    assert len(calls) == 1
    assert run['steps'][0]['output'].get('batch') is None
    assert run['steps'][1]['output']['batch']['published'] is False
    assert run['steps'][2]['output']['resolution']['summary']['selected_rows'] == 1
    assert not (store.root / 'tushare_active.json').exists()
    assert 'frozen' not in etl.start(store, request)


@pytest.mark.parametrize('alter', ['forward', 'duplicate', 'wrong_source', 'snapshot_missing', 'wrong_table', 'secret'])
def test_invalid_plans_fail_before_network(store, monkeypatch, alter):
    definition = plan(store)
    if alter == 'forward': definition['steps'] = list(reversed(definition['steps']))
    elif alter == 'duplicate': definition['steps'][1]['id'] = 'download'
    elif alter == 'wrong_source': definition['steps'][0]['source_id'] = 'tushare'
    elif alter == 'snapshot_missing': definition['steps'].append({'id':'snapshot','name':'快照','kind':'snapshot','inputs':['resolve']})
    elif alter == 'wrong_table': definition['steps'][2]['table_id'] = 'master.instrument'
    elif alter == 'secret': definition['steps'][0]['params']['api_key'] = 'do-not-leak'
    assert not etl.validate(store, definition)['valid']
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *a: pytest.fail('unexpected network'))
    with pytest.raises(CenterError): etl.start(store, payload(definition))


def test_workflow_cas_and_deletion_preserve_history(store):
    saved = etl.save_workflow(store, 'my_flow', plan(store), 0)
    assert saved['revision'] == 1
    with pytest.raises(CenterError): etl.save_workflow(store, 'my_flow', plan(store), 0)
    EtlStore(store).delete_workflow('my_flow', 1)
    assert EtlStore(store).workflows() == []
    with pytest.raises(CenterError): etl.save_workflow(store, 'my_flow', plan(store), 0)


def test_resume_keeps_successful_download_and_mapping(store, monkeypatch):
    count = []
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *args: (count.append(1) or nav_fetch()))
    original = etl.resolve_saved
    monkeypatch.setattr(etl, 'resolve_saved', lambda *a, **kw: (_ for _ in ()).throw(CenterError('TEST_FAIL', 'test failure')))
    run = finished(store, etl.start(store, payload(plan(store))))
    assert run['status'] == 'FAILED'
    assert run['steps'][0]['status'] == run['steps'][1]['status'] == 'SUCCEEDED'
    monkeypatch.setattr(etl, 'resolve_saved', original)
    resumed = finished(store, etl.resume(store, run['run_id'], True))
    assert resumed['status'] == 'SUCCEEDED', resumed
    assert count == [1]
    assert resumed['steps'][0]['attempt'] == 1
    assert resumed['steps'][2]['attempt'] == 2


def test_resume_rejects_changed_artifact(store, monkeypatch):
    monkeypatch.setattr(acquisition, 'fetch_with_retry', nav_fetch)
    monkeypatch.setattr(etl, 'resolve_saved', lambda *a, **k: (_ for _ in ()).throw(CenterError('TEST_FAIL', 'test')))
    run = finished(store, etl.start(store, payload(plan(store))))
    raw = store.root / run['steps'][0]['artifacts'][0]['path']; raw.write_text('[]')
    with pytest.raises(CenterError, match='制品'): etl.resume(store, run['run_id'], True)


def test_mapping_failure_keeps_raw_and_does_not_advance_checkpoint(store, monkeypatch):
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *a: ({}, [{'symbol':'000001','净值日期':'2024-01-05','单位净值':'broken'}]))
    result = finished(store, etl.start(store, payload(plan(store))))
    assert [s['status'] for s in result['steps']] == ['SUCCEEDED', 'FAILED', 'SKIPPED']
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_sync_checkpoint').fetchone()[0] == 0


def test_incremental_reuses_committed_watermark(store, monkeypatch):
    calls = []
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *a: (calls.append(a[3]) or nav_fetch()))
    for mode in ['full', 'incremental']:
        definition = plan(store); definition['steps'][0]['mode'] = mode
        assert finished(store, etl.start(store, payload(definition)))['status'] == 'SUCCEEDED'
    assert calls[1]['start_date'] == '20240102'


def test_cancel_global_lock_and_no_mapping_after_cancel(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def fetch(*a):
        entered.set(); release.wait(5); return nav_fetch()
    monkeypatch.setattr(acquisition, 'fetch_with_retry', fetch)
    run = etl.start(store, payload(plan(store)))
    assert entered.wait(3)
    with pytest.raises(CenterError, match='已有'): etl.start(store, payload(plan(store)))
    etl.cancel(store, run['run_id']); release.set()
    result = finished(store, run)
    assert result['status'] == 'CANCELLED'
    assert result['steps'][1]['status'] == 'SKIPPED'


def test_empty_data_not_success_by_default(store, monkeypatch):
    monkeypatch.setattr(acquisition, 'fetch_with_retry', lambda *a: ({}, []))
    result = finished(store, etl.start(store, payload(plan(store))))
    assert result['code'] == 'ETL_EMPTY_DATA'
    assert result['steps'][1]['status'] == 'SKIPPED'


def test_snapshot_dependencies_execute_in_position_and_use_only_resolved_files(store, monkeypatch):
    record = store.get('interface', 'tushare.etf_basic')
    # Local fixture uses the same master mapping with a no-auth transport.
    from backend.data_sources.models import SourceConfig
    store.save(SourceConfig(id='fixture', name='fixture', base_url='https://example.com'), 0)
    config = InterfaceConfig.model_validate(record['config']); config.id='fixture.master'; config.source_id='fixture'
    store.save(config, 0)
    from backend.data_sources.resolution_store import get_policy, save_policy
    rule = get_policy(store); rule['config']['default_source_priority'].insert(0, 'fixture')
    save_policy(store, rule['config'], rule['revision'])
    definition = plan(store)
    master_download = {'id':'master_download', 'name':'产品', 'kind':'download', 'source_id':'fixture', 'interface_id':config.id, 'interface_revision':1}
    definition['steps'][:0] = [master_download, {'id':'master_map','name':'产品映射','kind':'map','inputs':['master_download']}, {'id':'master_resolve','name':'产品取值','kind':'resolve','inputs':['master_map'],'table_id':'master.instrument','include_history':False}]
    definition['steps'].append({'id':'snapshot','name':'快照','kind':'snapshot','inputs':['master_resolve','resolve']})
    def fetch(store, source, interface, params):
        if source.id == 'fixture': return {}, [{'ts_code':'000001.OF','csname':'test fund','list_status':'L'}]
        return nav_fetch()
    monkeypatch.setattr(acquisition, 'fetch_with_retry', fetch)
    seen = []
    def build(journal, run, directory, inputs, check):
        assert all(path.is_file() for path in inputs.values())
        assert set(inputs) == {'market.nav_daily', 'master.instrument'}
        seen.append(inputs)
        return {'rows':1, 'published':False}
    monkeypatch.setattr(etl, '_snapshot_process', build)
    result = finished(store, etl.start(store, payload(definition)))
    assert result['status'] == 'SUCCEEDED', result
    assert len(seen) == 1
    assert not (store.root / 'instrument_metrics_snapshot.parquet').exists()


def test_explicit_inputs_do_not_read_unrelated_mapping_steps(store, monkeypatch):
    definition = plan(store)
    first, mapped, resolved = definition['steps']
    second = {**first, 'id':'other_download', 'name':'其他下载', 'params':{**first['params'],'symbol':'000002'}}
    other_map = {'id':'other_map','name':'其他映射','kind':'map','inputs':['other_download']}
    definition['steps'] = [first, mapped, second, other_map, resolved]
    def fetch(store, source, interface, params):
        return {}, [{'symbol':params['symbol'], '净值日期':'2024-01-05','单位净值':1.25}]
    monkeypatch.setattr(acquisition, 'fetch_with_retry', fetch)
    result = finished(store, etl.start(store, payload(definition)))
    assert result['status'] == 'SUCCEEDED', result
    assert result['steps'][-1]['output']['resolution']['summary']['selected_rows'] == 1
    assert result['steps'][-1]['output']['resolution']['input_batches'] == 1


def test_resolution_policy_is_frozen_before_download(store, monkeypatch):
    from backend.data_sources.resolution_store import get_policy, save_policy
    before = get_policy(store)
    def fetch(*args):
        changed = get_policy(store)
        changed['config']['default_source_priority'] = ['tushare']
        save_policy(store, changed['config'], changed['revision'])
        return nav_fetch()
    monkeypatch.setattr(acquisition, 'fetch_with_retry', fetch)
    result = finished(store, etl.start(store, payload(plan(store))))
    assert result['status'] == 'SUCCEEDED'
    resolution = result['steps'][-1]['output']['resolution']
    assert resolution['policy_revision'] == before['revision']
    assert resolution['summary']['selected_rows'] == 1


def test_changed_source_revision_prevents_resume(store, monkeypatch):
    monkeypatch.setattr(acquisition, 'fetch_with_retry', nav_fetch)
    monkeypatch.setattr(etl, 'resolve_saved', lambda *a, **k: (_ for _ in ()).throw(CenterError('FAIL','test')))
    run = finished(store, etl.start(store, payload(plan(store))))
    entry = store.get('interface', 'akshare.fund_nav')
    updated = InterfaceConfig.model_validate(entry['config']); updated.name = 'changed'
    store.save(updated, entry['revision'])
    with pytest.raises(CenterError, match='配置已改变'):
        etl.resume(store, run['run_id'], True)


def test_missing_owner_is_marked_interrupted_and_resumes(store, monkeypatch):
    monkeypatch.setattr(acquisition, 'fetch_with_retry', nav_fetch)
    original = etl.resolve_saved
    monkeypatch.setattr(etl, 'resolve_saved', lambda *a, **k: (_ for _ in ()).throw(CenterError('FAIL','test')))
    run = finished(store, etl.start(store, payload(plan(store))))
    run['owner_pid'] = 2**30; run['status'] = 'RUNNING'; run['steps'][-1]['status'] = 'RUNNING'
    journal = EtlStore(store); journal.save_run(run)
    assert journal.interrupted(journal.get_run(run['run_id']))['status'] == 'INTERRUPTED'
    monkeypatch.setattr(etl, 'resolve_saved', original)
    result = finished(store, etl.resume(store, run['run_id'], True))
    assert result['status'] == 'SUCCEEDED'
    assert result['steps'][0]['attempt'] == 1


def test_routes_same_origin_readonly_and_real_plan_run(store, monkeypatch):
    monkeypatch.setattr(etl_routes, 'get_store', lambda: store)
    monkeypatch.setattr(acquisition, 'fetch_with_retry', nav_fetch)
    app=FastAPI(); app.include_router(etl_routes.router)
    client=TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1')
    path='/api/data-sources/etl'
    assert client.get(path+'/workflows').status_code == 200
    assert client.post(path+'/validate', json={'definition':plan(store)}).json()['valid']
    assert client.post(path+'/runs', json=payload(plan(store)), headers={'Origin':'https://bad.invalid'}).status_code == 403
    response=client.post(path+'/runs', json=payload(plan(store)))
    assert response.status_code == 200, response.text
    assert finished(store,response.json())['status'] == 'SUCCEEDED'
    assert client.get(path+'/runs').json()[0]['steps'][-1]['status'] == 'SUCCEEDED'
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED','false')
    assert client.post(path+'/runs',json=payload(plan(store))).status_code == 403
