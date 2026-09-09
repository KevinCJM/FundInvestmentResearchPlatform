"""Offline legacy result import, never an in-place contract bypass."""
import copy
import json
import uuid
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.etl_migration import stage_recovery, _portfolio_parts
from backend.data_sources.etl_models import EtlStep
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError, InterfaceConfig
from backend.data_sources.task_workspace import read_inventory
from backend.services.refresh_runtime import InterProcessFileLock
from backend.tests.test_etl_dataset_tasks import store, small_plan, request, finish, fake_worker


@pytest.mark.parametrize('case', ['valid','unconfirmed','nonempty','wrong_suffix','symlink','tampered'])
def test_adjustment_scope_recovery_is_explicit_and_preserves_empty_audit(store, monkeypatch, case):
    from pathlib import Path
    import pandas as pd
    import T01_get_data as script
    from backend.data_sources import task_catalog
    from backend.data_sources.runtime import ConfiguredTushareClient
    from backend.data_sources.etl_models import EtlDefinition

    current = copy.deepcopy(task_catalog.task_specs())
    prior = copy.deepcopy(current)
    prior['tushare.fund_adjustment'].update(name='公募基金复权因子', category='公募基金', requires=['fund_info'])
    monkeypatch.setattr(task_catalog, 'task_specs', lambda: prior)
    definition = EtlDefinition(name='Scope recovery', steps=[
        EtlStep(id=action, name=action, kind='task', task_id='tushare.'+action, source_id='tushare',
                inputs=[previous] if previous else [], params={'start_date':'20260901','end_date':'20260904'})
        for action, previous in [('etf_info',None),('fund_info','etf_info'),('calendar','fund_info'),('fund_adjustment','calendar')]
    ])
    def worker(payload, check, lock):
        path = Path(payload['directory'])
        action = payload['task_id'].split('.')[-1]
        if action == 'fund_adjustment':
            args = script.parse_args(['--start-date','20260901','--end-date','20260904'])
            args.source_configuration_hash = ConfiguredTushareClient('offline-test-credential',root=store.root).configuration_hash
            parts = script.history_checkpoint_dir(path / 'fund_adj_factor_df.parquet',args)
            script.mark_empty_checkpoint(parts / ('510300.SH.empty' if case == 'wrong_suffix' else '000001.OF.empty'))
            if case == 'nonempty': pq.write_table(pa.table({'value':[1]}),parts / '000001.OF.parquet')
            if case == 'symlink': (parts / '000002.OF.empty').symlink_to(parts / '000001.OF.empty')
            raise CenterError('OFFLINE_EMPTY_SCOPE','Wrong scope')
        if action in {'etf_info','fund_info'}:
            pd.DataFrame([{'ts_code':'510300.SH' if action=='etf_info' else '000001.OF', 'name':'test'}]).to_parquet(path / (action+'_df.parquet'))
        else:
            pq.write_table(pa.table({'date':['20260904']}),path/'calendar.parquet')
        return {'received_rows':1, 'warnings':0, 'mapped_rejected_batches':0}
    monkeypatch.setattr(task_runtime,'run_worker',worker)
    old = finish(store,etl.start(store,request(definition.model_dump(mode='json'))))
    assert old['status']=='FAILED' and old['steps'][-1]['status']=='FAILED'
    original = copy.deepcopy(old)
    monkeypatch.setattr(task_catalog,'task_specs',lambda: current)
    monkeypatch.setattr(etl,'execution_fingerprint',lambda:'new-execution')
    if case not in {'valid','tampered'}:
        with pytest.raises(CenterError):
            stage_recovery(store,old['run_id'],uuid.uuid4().hex,confirm=True,adjustment_scope=case!='unconfirmed')
        return
    new = stage_recovery(store,old['run_id'],uuid.uuid4().hex,confirm=True,adjustment_scope=True)
    journal = EtlStore(store)
    receipt = json.loads(journal.checked_path(journal.get_run(new['run_id'])['recovery_receipt']).read_text())
    assert receipt['shards']['excluded_scope_empty']==1
    assert not list((journal.root/'etl_runs'/new['run_id']/'fund_adjustment'/'work').rglob('*.empty'))
    assert new['definition']['steps'][-1]['name']=='ETF 复权因子'
    if case == 'tampered':
        journal.checked_path(receipt['shards']['files'][0]['imported']).write_bytes(b'changed')
        with pytest.raises(CenterError,match='校验和'):
            etl.resume(store,new['run_id'],True)
    else:
        monkeypatch.setattr(task_runtime,'run_worker',fake_worker)
        assert finish(store,etl.resume(store,new['run_id'],True))['status']=='SUCCEEDED'
    assert journal.get_run(old['run_id'])==original


def old_run(store, monkeypatch):
    def worker(payload, check, lock):
        if payload['task_id'] == 'tushare.fund_company':
            raise CenterError('OFFLINE_FAILURE', 'offline')
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    run = finish(store, etl.start(store, request(small_plan())))
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'new-execution')
    return run


def test_new_run_imports_prefix_and_keeps_original_unchanged(store, monkeypatch):
    old = old_run(store, monkeypatch)
    original = copy.deepcopy(old)
    identifier = uuid.uuid4().hex
    new = stage_recovery(store, old['run_id'], identifier, confirm=True)
    assert new['recovered_from'] == old['run_id']
    assert new['steps'][0]['status'] == 'SUCCEEDED'
    assert new['steps'][0]['imported_from']['execution_fingerprint'] == old['frozen']['execution_fingerprint']
    assert new['steps'][1]['status'] == 'PENDING'
    assert {window['run_id'] for window in new['collection_timing']['windows']} == {old['run_id']}
    assert len(new['collection_timing']['windows']) == 2  # Includes the old partial step.
    assert EtlStore(store).get_run(old['run_id']) == original
    assert stage_recovery(store, old['run_id'], identifier, confirm=True)['run_id'] == identifier
    calls = []
    def worker(payload, check, lock):
        calls.append(payload['task_id'])
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    completed = finish(store, etl.resume(store, identifier, True))
    assert completed['status'] == 'SUCCEEDED'
    assert calls == ['tushare.fund_company']
    assert EtlStore(store).get_run(old['run_id']) == original
    assert not (store.root / 'tushare_active.json').exists()


@pytest.mark.parametrize('tampered', [False, True])
def test_dividend_merge_recovery_uses_checkpoints_via_normal_resume(store, monkeypatch, tampered):
    from pathlib import Path
    import pandas as pd
    import T01_get_data as script
    from backend.data_sources.runtime import ConfiguredTushareClient

    plan = small_plan()
    plan['steps'][0].update(task_id='tushare.fund_info')
    plan['steps'][1].update(task_id='tushare.fund_dividend')
    for step in plan['steps']:
        step['params']['end_date'] = '20240103'
    calls = []
    def fetch(**params):
        calls.append(params)
        date = params['ann_date']
        if date == '20240103':
            return pd.DataFrame()
        return pd.DataFrame([dict(ts_code='B' if date == '20240101' else 'A', ann_date=date,
                                  ex_date='20240105', pay_date='20240106', div_cash=.1)])
    def worker(payload, check, lock):
        path = Path(payload['directory'])
        if payload['task_id'] == 'tushare.fund_info':
            pd.DataFrame([{'ts_code':'A', 'name':'A', 'found_date':'20240101'}]).to_parquet(path / 'fund_info_df.parquet')
        else:
            args = script.parse_args(['--start-date', '20240101', '--end-date', '20240103', '--max-workers', '1'])
            args.source_configuration_hash = ConfiguredTushareClient('offline-test-credential', root=store.root).configuration_hash
            script.save_fund_dividend(type('Pro', (), {'fund_div': staticmethod(fetch)})(),
                                     path, script.RateLimiter(100000), args)
        return {'received_rows': 1, 'warnings': 0, 'mapped_rejected_batches': 0}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    consolidate = script._consolidate_ordered_parts
    def fail_merge(*args):
        raise CenterError('OFFLINE_ARROW_NULL_FAILURE', '合并失败')
    monkeypatch.setattr(script, '_consolidate_ordered_parts', fail_merge)
    old = finish(store, etl.start(store, request(plan)))
    assert old['status'] == 'FAILED' and len(calls) == 4
    original = copy.deepcopy(old)
    monkeypatch.setattr(script, '_consolidate_ordered_parts', consolidate)
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'new-execution')
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    journal = EtlStore(store)
    receipt = json.loads(journal.checked_path(journal.get_run(new['run_id'])['recovery_receipt']).read_text())
    assert receipt['shards']['copied'] == 2 and receipt['shards']['confirmed_empty'] == 1
    if tampered:
        copied = next(item['imported'] for item in receipt['shards']['files'] if item['imported']['path'].endswith('.parquet'))
        journal.checked_path(copied).write_bytes(b'changed')
        with pytest.raises(CenterError, match='校验和'):
            etl.resume(store, new['run_id'], True)
    else:
        completed = finish(store, etl.resume(store, new['run_id'], True))
        assert completed['status'] == 'SUCCEEDED', completed
        result = read_inventory(journal, completed['steps'][1]['output']['workspace'])
        path = journal.checked_path(result['files']['fund_dividend_df.parquet'])
        assert pq.ParquetFile(path).metadata.num_rows == 1
    assert len(calls) == 4  # No additional supplier calls during migration/resume.
    assert journal.get_run(old['run_id']) == original
    assert not (store.root / 'tushare_active.json').exists()


@pytest.mark.parametrize('case', ['unconfirmed', 'locked', 'tampered', 'source_changed', 'contract_changed', 'auto'])
def test_import_fails_closed(store, monkeypatch, case):
    old = old_run(store, monkeypatch)
    journal = EtlStore(store)
    lock = None
    if case == 'locked':
        lock = InterProcessFileLock(store.root / '.tushare_refresh.lock')
        assert lock.acquire(owner='orphan')
    elif case == 'tampered':
        value = read_inventory(journal, old['steps'][0]['output']['workspace'])
        journal.checked_path(next(iter(value['files'].values()))).write_bytes(b'broken')
    elif case == 'source_changed':
        old['frozen']['task_sources']['tushare']['hash'] = 'changed'
        journal.save_run(old)
    elif case == 'contract_changed':
        old['frozen']['tasks']['company']['spec']['provides'] = ['different']
        journal.save_run(old)
    elif case == 'auto':
        old['options']['mode'] = 'auto_incremental'
        journal.save_run(old)
    try:
        with pytest.raises(CenterError):
            stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=case != 'unconfirmed')
        assert len(journal.runs()) == 1
    finally:
        if lock:
            lock.release()


def test_changed_migration_receipt_blocks_resume(store, monkeypatch):
    old = old_run(store, monkeypatch)
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    journal = EtlStore(store)
    journal.checked_path(new['recovery_receipt']).write_bytes(b'changed')
    with pytest.raises(CenterError, match='校验和'):
        etl.resume(store, new['run_id'], True)


def test_changed_imported_history_receipt_blocks_resume(store, monkeypatch):
    old = old_run(store, monkeypatch)
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    journal = EtlStore(store)
    run = journal.get_run(new['run_id'])
    path = journal.checked_path(run['recovery_receipt'])
    receipt = json.loads(path.read_text())
    shard = path.parent / 'range.json'
    artifact = journal.write_json(shard, {'status':'EMPTY', 'confirmations':2})
    receipt['shards']['history'] = {'files':[{'imported':artifact}]}
    run['recovery_receipt'] = journal.write_json(path, receipt)
    journal.save_run(run)
    shard.write_text('{}')
    with pytest.raises(CenterError, match='校验和'):
        etl.resume(store, new['run_id'], True)


@pytest.mark.parametrize('case', ['valid', 'not_accepted', 'row_cap', 'rate', 'page_size', 'cursor', 'wrong_api', 'params'])
def test_only_explicit_holdings_pagination_enable_can_migrate(store, monkeypatch, case):
    old = old_run(store, monkeypatch)
    original = copy.deepcopy(old)
    identifier = 'tushare.fund_company' if case == 'wrong_api' else 'tushare.fund_portfolio'
    record = store.get('interface', identifier)
    config = InterfaceConfig.model_validate(record['config'])
    config.pagination.mode = 'offset'
    if case == 'row_cap': config.policy.max_rows_per_request += 1
    if case == 'rate': config.policy.requests_per_minute += 1
    if case == 'page_size': config.pagination.page_size -= 1
    if case == 'cursor': config.pagination.cursor_param = 'page'
    if case == 'params': config.params['period'] = '20251231'
    store.save(config, record['revision'])
    kwargs = {'pagination_changes': (identifier,)} if case != 'not_accepted' else {}
    if case == 'valid':
        new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True, **kwargs)
        journal = EtlStore(store)
        evidence = journal.get_run(new['run_id'])['frozen']['pagination_policy_migration']
        assert evidence[0]['before']['mode'] == 'none' and evidence[0]['after']['mode'] == 'offset'
        assert journal.get_run(old['run_id']) == original
        monkeypatch.setattr(task_runtime, 'run_worker', fake_worker)
        assert finish(store, etl.resume(store, new['run_id'], True))['status'] == 'SUCCEEDED'
    else:
        with pytest.raises(CenterError):
            stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True, **kwargs)


@pytest.mark.parametrize('case', ['rate_only', 'old_optional_defaults', 'not_accepted', 'row_cap', 'params', 'fields', 'timeout'])
def test_rate_migration_requires_explicit_acceptance_and_unchanged_data_contract(store, monkeypatch, case):
    old = old_run(store, monkeypatch)
    if case == 'old_optional_defaults':
        from backend.data_sources.acquisition import fingerprint
        snapshot = old['frozen']['task_sources']['tushare']
        for item in snapshot['records']:
            if item['config']['id'] == 'tushare.fund_company':
                item['config'].pop('request_fields', None)
        snapshot['hash'] = fingerprint(snapshot['records'])
        EtlStore(store).save_run(old)
    original = copy.deepcopy(old)
    record = store.get('interface', 'tushare.fund_company')
    config = InterfaceConfig.model_validate(record['config'])
    config.policy.requests_per_minute = 400
    config.policy.min_interval_seconds = .15
    if case == 'row_cap': config.policy.max_rows_per_request += 1
    if case == 'params': config.params['ts_code'] = '001.OF'
    if case == 'fields': config.params['fields'] = 'ts_code'
    if case == 'timeout': config.policy.read_timeout_seconds += 1
    store.save(config, record['revision'])
    kwargs = {'rate_changes': ('tushare.fund_company',)} if case != 'not_accepted' else {}
    if case in {'rate_only', 'old_optional_defaults'}:
        new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True, **kwargs)
        journal = EtlStore(store)
        frozen = journal.get_run(new['run_id'])['frozen']
        assert frozen['rate_policy_migration'][0]['after']['requests_per_minute'] == 400
        assert journal.get_run(old['run_id']) == original
        monkeypatch.setattr(task_runtime, 'run_worker', fake_worker)
        assert finish(store, etl.resume(store, new['run_id'], True))['status'] == 'SUCCEEDED'
    else:
        with pytest.raises(CenterError):
            stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True, **kwargs)


@pytest.mark.parametrize('case', ['valid', 'valid_new_rate', 'wrong_date', 'symlink'])
def test_holdings_partial_import_validates_lineage_and_does_not_trust_empty_markers(store, monkeypatch, case):
    import T01_get_data as script
    from backend.data_sources.runtime import ConfiguredTushareClient
    from types import SimpleNamespace
    old = old_run(store, monkeypatch)
    journal = EtlStore(store)
    step = EtlStep(id='portfolio', name='持仓', kind='task', task_id='tushare.fund_portfolio', source_id='tushare',
                   inputs=['calendar'], params={'start_date': '20240101', 'end_date': '20240105'})
    work = journal.root / 'etl_runs' / old['run_id'] / step.id / 'work'
    work.mkdir(parents=True)
    client = ConfiguredTushareClient('offline-test-credential', root=store.root)
    args = SimpleNamespace(start_date='20240101', end_date='20240105', source_configuration_hash=client.configuration_hash)
    parts = script.history_checkpoint_dir(work / 'fund_portfolio_df.parquet', args)
    parts.mkdir()
    day = datetime(2024, 1, 3 if case == 'wrong_date' else 2)
    shard = parts / '20240102.parquet'
    pq.write_table(pa.table({'ts_code':['001.OF'], 'symbol':['001.SZ'], 'ann_date':[day], 'end_date':[day],
                            'available_at':[day], 'source_api':['fund_portfolio'], 'mkv':[1.], 'amount':[1.]}), shard)
    script.mark_empty_checkpoint(parts / '20240101.empty')
    if case == 'symlink':
        (parts / '20240103.parquet').symlink_to(shard)
    predecessor = old['steps'][0]['output']['workspace']
    producer = old['frozen']['execution_fingerprint']
    journal.write_json(work.parent / 'work_input.json', {'input':predecessor, 'task':step.model_dump(mode='json'), 'execution':producer})
    target = store.root / 'new-work'
    target.mkdir()
    target_frozen = None
    if case == 'valid_new_rate':
        record = store.get('interface', 'tushare.fund_portfolio')
        config = InterfaceConfig.model_validate(record['config'])
        config.policy.requests_per_minute = 400
        config.policy.min_interval_seconds = .15
        store.save(config, record['revision'])
        target_frozen = copy.deepcopy(old['frozen'])
        target_frozen['task_sources']['tushare']['records'] = [r for kind in ('source', 'interface') for r in store.list(kind)
                                                            if r['config'].get('source_id', r['config']['id']) == 'tushare']
    if case not in {'valid', 'valid_new_rate'}:
        with pytest.raises(CenterError):
            _portfolio_parts(journal, old, step, predecessor, target, producer)
    else:
        evidence = _portfolio_parts(journal, old, step, predecessor, target, producer, target_frozen=target_frozen)
        assert evidence['copied'] == 1 and evidence['empty_recheck'] == 1
        assert not list(target.rglob('*.empty'))
        copied = journal.checked_path(evidence['files'][0]['imported'])
        if case == 'valid_new_rate':
            updated = ConfiguredTushareClient('offline-test-credential', root=store.root)
            args.source_configuration_hash = updated.configuration_hash
            assert copied.parent == script.history_checkpoint_dir(target / 'fund_portfolio_df.parquet', args)
            assert copied.parent.name != parts.name
        assert copied.stat().st_ino != shard.stat().st_ino
        assert (parts / '20240101.empty').exists()
        journal.checked_path(evidence['day_import_receipt'])
        # The current collector must actually consume the explicit migration
        # receipt, not just copy an obsolete checkpoint directory and refetch.
        import pandas as pd
        from backend.data_sources.fund_events import FundEventDownload
        session = FundEventDownload(directory=copied.parent, dates=['20240102'],
                                    universe=pd.DataFrame([{'ts_code':'001.OF'}]), api_name='fund_portfolio',
                                    fields=script.FUND_PORTFOLIO_FIELDS, smoke=False)
        def prepare(frame):
            return script._prepare_fund_event_rows(frame, fields=script.FUND_PORTFOLIO_FIELDS,
                                                  source_api='fund_portfolio', observation_column='end_date')
        result = session.run(fetch=lambda *args: pytest.fail('Verified import must not hit network'),
                             prepare=prepare, save=script.save_dataframe,
                             cap_error=script.ResponseTruncatedError, max_workers=1)
        assert pd.read_parquet(result[0]).available_at.iloc[0] == pd.Timestamp(day)
