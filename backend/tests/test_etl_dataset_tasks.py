"""Offline full-universe task contracts, isolation and frontend metadata."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.acquisition import fingerprint
from backend.data_sources.credentials import save_credential
from backend.data_sources.etl_models import EtlDefinition, EtlStep
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.etl_templates import tushare_all_data_workflow
from backend.data_sources.models import CenterError, InterfaceConfig
from backend.data_sources.presets import API_SPECS
from backend.data_sources.service import catalog
from backend.data_sources.store import SourceStore
from backend.data_sources.task_catalog import ACTION_APIS, task_specs
from backend.data_sources.task_workspace import read_inventory
from backend.services import etl_routes


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(etl, '_launch', etl._launch_inline)
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    store = SourceStore(tmp_path)
    store.seed()
    save_credential(store, 'tushare', 'offline-test-credential')
    return store


def small_plan():
    return EtlDefinition(name='Generic dataset flow', steps=[
        EtlStep(id='calendar', name='交易日历', kind='task', task_id='tushare.calendar', source_id='tushare', params={'start_date':'20240101','end_date':'20240105'}),
        EtlStep(id='company', name='机构', kind='task', task_id='tushare.fund_company', source_id='tushare', inputs=['calendar'], params={'start_date':'20240101','end_date':'20240105'}),
    ]).model_dump(mode='json')


def request(plan, mode='full'):
    return {'definition':plan,'options':{'mode':mode,'parameters':{}},'confirm':True,'request_id':str(uuid.uuid4())}


def finish(store, run):
    journal = EtlStore(store)
    for _ in range(500):
        result = journal.get_run(run['run_id'])
        if result['status'] != 'RUNNING':
            # terminal persistence happens immediately before lock release
            from backend.services.refresh_runtime import is_file_lock_held
            if not is_file_lock_held(store.root / '.tushare_refresh.lock'):
                return result
        time.sleep(.02)
    pytest.fail('offline job timed out')


def fake_worker(payload, check, lock):
    check()
    assert lock.acquired
    path = Path(payload['directory'])
    name = payload['task_id'].split('.')[-1] + '.parquet'
    pq.write_table(pa.table({'value':[payload['params']['end_date']]}), path / name)
    return {'received_rows':1, 'warnings':0, 'mapped_rejected_batches':0}


def test_full_template_covers_every_action_and_api_without_single_codes(store):
    definition = tushare_all_data_workflow(store)
    assert len(definition.steps) == 31
    assert {s.task_id for s in definition.steps} == set(task_specs())
    assert set().union(*(set(v) for v in ACTION_APIS.values())) == set(API_SPECS)
    assert [p.id for p in definition.parameters] == ['start_date','end_date']
    assert definition.steps[-1].task_id == 'local.analytics_snapshot'
    assert all(s.mode == 'inherit' for s in definition.steps)
    assert all(set(s.params).isdisjoint({'limit','ts_code','symbol'}) for s in definition.steps)
    assert etl.validate(store, definition.model_dump(mode='json'))['valid']
    saved = etl.save_workflow(store, 'all_data', definition.model_dump(mode='json'), 0)
    assert saved['revision'] == 1
    assert EtlStore(store).runs() == []


@pytest.mark.parametrize('change', ['unknown','dependency','path','limit','source','forward'])
def test_invalid_tasks_fail_before_network(store, change):
    definition = small_plan()
    if change == 'unknown': definition['steps'][0]['task_id'] = 'shell.run'
    if change == 'dependency': definition['steps'][0]['task_id'] = 'tushare.nav'
    if change == 'path': definition['steps'][0]['params']['output_dir'] = '/tmp/outside'
    if change == 'limit': definition['steps'][0]['params']['limit'] = 1
    if change == 'source': definition['steps'][0]['source_id'] = 'akshare'
    if change == 'forward': definition['steps'][0]['inputs'] = ['company']
    assert not etl.validate(store, definition)['valid']
    with pytest.raises(CenterError):
        etl.start(store, request(definition))


def test_disabled_or_narrowed_endpoint_blocks_full_data(store):
    entry = store.get('interface','tushare.trade_cal')
    config = InterfaceConfig.model_validate(entry['config'])
    config.enabled = False
    store.save(config, entry['revision'])
    assert not etl.validate(store, small_plan())['valid']
    config.enabled = True
    config.params = {'ts_code':'510300.SH'}
    store.save(config, entry['revision'] + 1)
    assert etl.validate(store, small_plan())['errors'][0]['code'] == 'ETL_TASK_NARROWED'


def test_workspace_full_incremental_and_inputs_are_immutable(store, monkeypatch):
    calls = []
    def worker(payload, check, lock):
        calls.append(payload)
        return fake_worker(payload,check,lock)
    monkeypatch.setattr(task_runtime,'run_worker',worker)
    first = finish(store, etl.start(store, request(small_plan())))
    assert first['status'] == 'SUCCEEDED', first
    journal = EtlStore(store)
    first_inventory = read_inventory(journal, first['steps'][0]['output']['workspace'])
    assert set(first_inventory['files']) == {'calendar.parquet'}
    plan = small_plan()
    for s in plan['steps']: s['params']['end_date'] = '20240110'
    second = finish(store, etl.start(store, request(plan, 'incremental')))
    assert second['status'] == 'SUCCEEDED', second
    assert second['frozen']['task_baseline']['run_id'] == first['run_id']
    assert calls[2]['has_baseline'] is True
    # Rewriting a clone did not change a predecessor's checksum or row values.
    assert read_inventory(journal, first['steps'][0]['output']['workspace']) == first_inventory
    assert pq.read_table(journal.checked_path(first_inventory['files']['calendar.parquet'])).to_pylist() == [{'value':'20240105'}]
    third = finish(store, etl.start(store, request(plan, 'full')))
    assert third['status'] == 'SUCCEEDED'
    assert third['frozen']['task_baseline'] is None
    assert calls[4]['has_baseline'] is False
    assert not (store.root/'tushare_active.json').exists()


def test_resume_does_not_repeat_successful_dataset(store, monkeypatch):
    calls = []
    def worker(payload, check, lock):
        calls.append((payload['task_id'],payload['resume']))
        if len(calls) == 2:
            raise CenterError('OFFLINE_FAILURE','test interrupted task')
        return fake_worker(payload,check,lock)
    monkeypatch.setattr(task_runtime,'run_worker',worker)
    run = finish(store, etl.start(store, request(small_plan())))
    assert run['status'] == 'FAILED'
    resumed = finish(store, etl.resume(store,run['run_id'],True))
    assert resumed['status'] == 'SUCCEEDED', resumed
    assert calls == [('tushare.calendar',False),('tushare.fund_company',False),('tushare.fund_company',True)]
    assert resumed['steps'][0]['attempt'] == 1


def test_modified_successful_files_refuse_resume(store, monkeypatch):
    monkeypatch.setattr(task_runtime,'run_worker',fake_worker)
    run = finish(store, etl.start(store, request(small_plan())))
    journal = EtlStore(store)
    inventory = read_inventory(journal,run['steps'][0]['output']['workspace'])
    journal.checked_path(inventory['files']['calendar.parquet']).write_bytes(b'changed')
    run['status'] = 'FAILED'; journal.save_run(run)
    with pytest.raises(CenterError,match='校验和'):
        etl.resume(store,run['run_id'],True)


def test_task_catalog_and_request_forms_are_configuration_driven(store, monkeypatch):
    payload = catalog(store)
    assert len(payload['etl_tasks']) == 31
    config = next(c for c in payload['interfaces'] if c['config']['id'] == 'tushare.trade_cal')
    assert {f['name'] for f in config['request_fields']} == {'exchange','start_date','end_date'}
    assert all('handler' not in task and 'action' not in task for task in payload['etl_tasks'])
    interface = InterfaceConfig.model_validate(config['config'])
    interface.request_fields = []
    store.save(interface, config['revision'])
    assert next(c for c in catalog(store)['interfaces'] if c['config']['id'] == interface.id)['request_fields'] == []
    monkeypatch.setattr(etl_routes,'get_store',lambda:store)
    app = FastAPI(); app.include_router(etl_routes.router)
    client = TestClient(app)
    assert len(client.get('/api/data-sources/etl/tasks').json()['tasks']) == 31
    templates = client.get('/api/data-sources/etl/templates').json()
    assert len(templates[0]['definition']['steps']) == 31
    assert 'offline-test-credential' not in json.dumps(templates)


def test_real_adapter_invokes_existing_collector_without_limit(store, monkeypatch):
    import T01_get_data as script
    from backend.data_sources.task_worker import acquire
    called = []
    def operation(args, actions, *, client):
        called.append((args,actions,client))
    monkeypatch.setattr(script,'_run_actions',operation)
    records = [r for kind in ('source','interface') for r in store.list(kind) if r['config'].get('source_id',r['config']['id']) == 'tushare']
    acquire({'root':str(store.root),'source_id':'tushare','source_hash':fingerprint(records),'params':{'start_date':'20240101','end_date':'20240105'},'mode':'full','has_baseline':False},task_specs()['tushare.calendar'],store.root/'out')
    assert called[0][0].limit is None
    assert not called[0][0].latest
    assert called[0][1] == ['calendar']
    assert called[0][2].capture is True


@pytest.mark.parametrize('failure', [None, 'permission', 'cap', 'wrong_date', 'wrong_code', 'missing_date'])
def test_index_transport_split_through_real_worker_acceptance(store, monkeypatch, failure):
    import T01_get_data as script
    from backend.data_sources import runtime
    from backend.data_sources.task_worker import acquire
    monkeypatch.setattr(script, 'FUTURES_INDEX_UNIVERSE', [('A.NH', 'A')])
    calls = []
    def request(url, method, body, headers, policy):
        params = body['params']; calls.append(dict(params))
        if params['start_date'] != params['end_date'] or failure == 'cap':
            items = [['A.NH', '20260901', 1.0]] * 2001
        elif failure == 'permission':
            return json.dumps({'code': -1, 'msg': 'no permission'})
        else:
            items = [['OTHER.NH' if failure == 'wrong_code' else 'A.NH',
                      None if failure == 'missing_date' else '19990101' if failure == 'wrong_date'
                      else params['start_date'], 1.0]]
        return json.dumps({'code': 0, 'data': {'fields': ['ts_code', 'trade_date', 'close'], 'items': items}})
    monkeypatch.setattr(runtime, 'request', request)
    records = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    payload = dict(root=str(store.root), source_id='tushare', source_hash=fingerprint(records),
                   params={'start_date': '20260901', 'end_date': '20260902'}, mode='full', has_baseline=False)
    path = store.root / 'out' / 'index_futures_daily_df.parquet'
    if failure:
        with pytest.raises(CenterError):
            acquire(payload, task_specs()['tushare.index_futures'], path.parent)
        assert not path.exists()
    else:
        result = acquire(payload, task_specs()['tushare.index_futures'], path.parent)
        assert len(calls) == 3 and result['batches'] == 2
        assert pq.ParquetFile(path).metadata.num_rows == 2
        calls.clear()
        acquire(payload, task_specs()['tushare.index_futures'], path.parent)
        assert not calls  # Complete disk checkpoints need no network replay.


def test_partition_acknowledgement_cannot_hide_later_permission_failure(store, monkeypatch):
    import T01_get_data as script
    from backend.data_sources.runtime import ConfiguredTushareClient
    from backend.data_sources.task_worker import acquire
    calls = []
    def call(self, interface, **params):
        calls.append(params)
        raise CenterError('SOURCE_ROW_CAP' if len(calls) == 1 else 'SOURCE_PERMISSION_OR_PARAMS', 'offline')
    def operation(args, actions, *, client):
        for _ in range(2):
            try: client.fut_index_daily(ts_code='A.NH', start_date='20260901', end_date='20260902')
            except CenterError: pass
        client.acknowledge_partition('fut_index_daily', ts_code='A.NH', start_date='20260901', end_date='20260902')
    monkeypatch.setattr(ConfiguredTushareClient, '_call', call)
    monkeypatch.setattr(script, '_run_actions', operation)
    records = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    with pytest.raises(CenterError, match='尚未成功'):
        acquire(dict(root=str(store.root), source_id='tushare', source_hash=fingerprint(records),
                     params={}, mode='full', has_baseline=False), task_specs()['tushare.index_futures'], store.root / 'out')


def test_warning_mapping_is_not_hidden(store, monkeypatch):
    def worker(*args):
        result = fake_worker(*args)
        result['mapped_rejected_batches'] = 2
        return result
    monkeypatch.setattr(task_runtime,'run_worker',worker)
    run = finish(store, etl.start(store,request(small_plan())))
    assert run['warning_count'] == 4
    assert '告警' in run['message']
    assert run['published'] is False
