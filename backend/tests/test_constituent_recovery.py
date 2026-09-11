"""Execution upgrades must preserve verified query receipts, not blindly reuse files."""
import copy
import hashlib
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
import T01_get_data as script
from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.etl_migration import stage_recovery
from backend.data_sources.etl_models import EtlDefinition, EtlStep
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.runtime import ConfiguredTushareClient
from backend.tests.test_etl_dataset_tasks import request, finish, fake_worker
from backend.tests import test_etl_dataset_tasks as dataset_fixtures

store = dataset_fixtures.store  # Explicit pytest fixture export, not a hidden import.


@pytest.mark.parametrize('case', ['valid', 'checksum', 'rows', 'identity', 'date', 'orphan', 'unknown', 'symlink', 'marker', 'final_stage'])
def test_verified_member_query_import(store, monkeypatch, case):
    plan = EtlDefinition(name='成分下载', steps=[
        EtlStep(id=action, name=action, kind='task', task_id='tushare.'+action, source_id='tushare',
                inputs=[previous] if previous else [], params={'start_date':'20260901', 'end_date':'20260906'})
        for action, previous in [('calendar', None), ('index_catalog', 'calendar'), ('index_constituents', 'index_catalog')]
    ])
    calls = []
    def fetch(*args, **params):
        calls.append(params)
        return pd.DataFrame() if params['ts_code'] == 'A.TI' else pd.DataFrame([
            {'ts_code':'BK1.DC', 'con_code':'600000.SH', 'trade_date':'20260904'}])
    monkeypatch.setattr(script, 'fetch_constituent_pages', fetch)
    def queries(directory, resume):
        args = script.parse_args(['--start-date','20260901','--end-date','20260906', *(['--resume'] if resume else [])])
        args.source_configuration_hash = ConfiguredTushareClient('offline-test-credential', root=store.root).configuration_hash
        parts = script.history_checkpoint_dir(directory / 'index_members_queries.parquet', args)
        for api, params in [('ths_member', {'ts_code':'A.TI'}), ('dc_member', {'ts_code':'BK1.DC', 'trade_date':'20260904'})]:
            script.cached_constituent_request(object(), api, None, args, parts, **params)
        return parts
    def worker(payload, check, lock):
        directory = Path(payload['directory'])
        action = payload['task_id'].split('.')[-1]
        if action == 'calendar':
            pd.DataFrame([{'exchange':'SSE', 'is_open':1, 'cal_date':'20260904'}]).to_parquet(directory / 'trade_day_df.parquet')
        elif action == 'index_catalog':
            pd.DataFrame([{'source_api':'ths_index','ts_code':'A.TI'}, {'source_api':'dc_index','ts_code':'BK1.DC'}]).to_parquet(directory / 'index_catalog_df.parquet')
        else:
            queries(directory, payload['resume'])
            raise CenterError('SOURCE_DB_BUSY', 'offline local contention')
        return {'received_rows':1, 'warnings':0, 'mapped_rejected_batches':0}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    old = finish(store, etl.start(store, request(plan.model_dump(mode='json'))))
    assert old['status'] == 'FAILED' and len(calls) == 2
    journal = EtlStore(store)
    original = copy.deepcopy(old)
    work = store.root / 'etl_runs' / old['run_id'] / 'index_constituents' / 'work'
    parts = next(work.glob('.index_members_queries_parts_*'))
    receipt = next(p for p in parts.glob('*.json') if json.loads(p.read_text())['rows'] == 1)
    data = receipt.with_suffix('.parquet')
    meta = json.loads(receipt.read_text())
    if case == 'checksum': data.write_bytes(b'corrupt')
    if case == 'rows': meta['rows'] = 8
    if case == 'identity': meta['request']['params']['ts_code'] = 'BK2.DC'
    if case == 'date':
        frame = pd.read_parquet(data); frame['trade_date'] = '20260907'; frame.to_parquet(data)
        meta['sha256'] = hashlib.sha256(data.read_bytes()).hexdigest()
    if case in {'rows', 'identity', 'date'}: receipt.write_text(json.dumps(meta))
    if case == 'orphan': receipt.unlink()
    if case == 'unknown': (parts / 'unknown.json').write_text('{}')
    if case == 'symlink': (parts / ('a'*64 + '.json')).symlink_to(receipt)
    if case == 'marker': (work.parent / 'work_input.json').write_text('{}')
    if case == 'final_stage': pd.DataFrame({'value':[1]}).to_parquet(work / 'index_members_df.parquet')
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda:'new-execution')
    if case != 'valid':
        with pytest.raises((CenterError, RuntimeError)):
            stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    else:
        new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
        assert new['name'] == old['name']  # No machine suffix accumulated.
        evidence = json.loads(journal.checked_path(journal.get_run(new['run_id'])['recovery_receipt']).read_text())
        assert evidence['shards']['copied'] == 2 and evidence['shards']['confirmed_empty'] == 1
        assert len(evidence['shards']['files']) == 4
        def resumed(payload, check, lock):
            assert payload['resume'] is True  # New run's first attempt still owns imported checkpoints.
            queries(Path(payload['directory']), payload['resume'])
            return fake_worker(payload, check, lock)
        monkeypatch.setattr(task_runtime, 'run_worker', resumed)
        assert finish(store, etl.resume(store, new['run_id'], True))['status'] == 'SUCCEEDED'
    assert len(calls) == 2  # Migration and resume never re-fetch complete queries.
    assert journal.get_run(old['run_id']) == original
    assert not (store.root / 'tushare_active.json').exists()


@pytest.mark.parametrize('larger_page_budget', [False, True])
def test_completed_member_stage_and_weights_survive_verified_execution_upgrade(store, monkeypatch, larger_page_budget):
    from backend.data_sources.models import InterfaceConfig
    record = store.get('interface', 'tushare.index_weight')
    config = InterfaceConfig.model_validate(record['config'])
    config.pagination.max_pages = 20
    store.save(config, record['revision'])
    plan = EtlDefinition(name='权重下载恢复', steps=[
        EtlStep(id=action, name=action, kind='task', task_id='tushare.'+action, source_id='tushare',
                inputs=[previous] if previous else [], params={'start_date':'20260901', 'end_date':'20260906'})
        for action, previous in [('calendar', None), ('index_catalog', 'calendar'), ('index_constituents', 'index_catalog')]
    ])
    calls = []
    def fetch(_pro, api, _limiter, _args, **params):
        calls.append((api, params))
        if api == 'ci_index_member':
            return pd.DataFrame()
        return pd.DataFrame([{'index_code':'A.SH', 'con_code':'600000.SH',
                              'trade_date':'20260904', 'weight':1.0}])
    monkeypatch.setattr(script, 'fetch_constituent_pages', fetch)
    def options():
        args = script.parse_args(['--start-date','20260901','--end-date','20260906','--resume'])
        args.source_configuration_hash = ConfiguredTushareClient('offline-test-credential', root=store.root).configuration_hash
        return args
    def worker(payload, check, lock):
        directory = Path(payload['directory'])
        action = payload['task_id'].split('.')[-1]
        if action == 'calendar':
            pd.DataFrame([{'exchange':'SSE', 'is_open':1, 'cal_date':'20260904'}]).to_parquet(directory / 'trade_day_df.parquet')
        elif action == 'index_catalog':
            pd.DataFrame([{'source_api':'index_basic','quote_source_api':'index_daily', 'ts_code':code, 'status':'active'}
                          for code in ['A.SH', 'B.SH']]).to_parquet(directory / 'index_catalog_df.parquet')
        else:
            args = options()
            members = script.history_checkpoint_dir(directory / 'index_members_queries.parquet', args)
            frames = [script.cached_constituent_request(object(), api, None, args, members)
                      for api in ['index_member_all', 'ci_index_member']]
            script._normalise_member_frame(frames[0], 'index_member_all', None).to_parquet(directory / 'index_members_df.parquet')
            (directory / '.tushare_stage_index_constituents_members.json').write_text(json.dumps({'schema_version':1, 'stage':'index_constituents_members'}))
            weights = script.history_checkpoint_dir(directory / 'index_weights_df.parquet', args)
            raw = script.cached_constituent_request(object(), 'index_weight', None, args, weights / 'queries',
                    index_code='A.SH', start_date='20260901', end_date='20260906')
            script._normalise_member_frame(raw, 'index_weight', 'A.SH').to_parquet(weights / 'A.SH.parquet')
            raise CenterError('INDEX_WEIGHT_INCOMPLETE', 'B.SH 分页未完成')
        return {'received_rows':1, 'warnings':0, 'mapped_rejected_batches':0}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    old = finish(store, etl.start(store, request(plan.model_dump(mode='json'))))
    original = copy.deepcopy(old)
    assert old['status'] == 'FAILED', old
    journal = EtlStore(store)
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda:'new-execution')
    if larger_page_budget:
        record = store.get('interface', 'tushare.index_weight')
        config = InterfaceConfig.model_validate(record['config'])
        config.pagination.max_pages = 100
        store.save(config, record['revision'])
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    if larger_page_budget:
        change, = journal.get_run(new['run_id'])['frozen']['pagination_policy_migration']
        assert change['kind'] == 'pagination_budget'
        assert change['before']['max_pages'] == 20 and change['after']['max_pages'] == 100
    evidence = json.loads(journal.checked_path(journal.get_run(new['run_id'])['recovery_receipt']).read_text())['shards']
    assert evidence['copied'] == 2 and evidence['weights']['copied'] == 1
    assert evidence['rebuilt_members']['rows'] == 1
    assert len(evidence['files']) == 7
    work = store.root / 'etl_runs' / new['run_id'] / 'index_constituents' / 'work'
    assert not (work / 'index_members_df.parquet').exists()  # Rebuild, not blindly trust the stage flag.
    assert len(list(work.glob('.index_weights_df_parts_*/A.SH.parquet'))) == 1
    assert len(calls) == 3  # Complete queries imported with no supplier re-fetch.
    assert journal.get_run(old['run_id']) == original
    assert not (store.root / 'tushare_active.json').exists()
