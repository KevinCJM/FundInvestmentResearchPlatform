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
    response = TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1').post(
        '/api/data-sources/etl/validate', json={'definition': definition(), 'options': OPTIONS})
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


def completed_candidate(store):
    from backend.data_sources.task_workspace import inventory
    journal = EtlStore(store)
    identifier = uuid.uuid4().hex
    directory = store.root / 'etl_runs' / identifier
    work = directory / 'work'; work.mkdir(parents=True)
    pq.write_table(pa.table({'date': [date(2024, 1, 9)], 'adj_nav': [1.2]}), work / 'fund_nav_df.parquet')
    artifact = inventory(journal, work, directory / 'workspace.json', ['fund_nav'], 'tushare')
    records = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    run = {'run_id': identifier, 'request_hash': 'test', 'name': '已完成下载', 'status': 'SUCCEEDED',
           'finished_at': '2024-01-10T00:00:00Z', 'definition': definition(),
           'frozen': {'task_sources': {'tushare': {'records': records}}},
           'steps': [{'status': 'SUCCEEDED', 'output': {'workspace': artifact}}]}
    journal.save_run(run, create=True)
    return run, work / 'fund_nav_df.parquet'


def test_missing_baseline_offers_completed_download_but_never_selects_it(store):
    run, file = completed_candidate(store)
    (store.root / 'snapshot' / file.name).unlink()
    checked = etl.validate(store, definition(), OPTIONS)
    assert not checked['valid']
    plan = checked['auto_plan']
    assert plan['errors'][0]['code'] == 'AUTO_BASELINE_NOT_ACTIVE'
    assert plan['errors'][0]['step_id'] == 'fund_nav'
    assert plan['steps'][-1]['strategy'] == 'blocked'
    assert plan['baseline_choices'][0]['run_id'] == run['run_id']
    assert plan['supplemental_baseline']['run_id'] is None


def test_selected_supplement_is_frozen_without_activating_or_overwriting(store):
    run, file = completed_candidate(store)
    manifest = (store.root / 'tushare_active.json').read_bytes()
    active = store.root / 'snapshot' / file.name
    # A newer active table always wins over the user's selected candidate.
    automatic = auto.plan(store, etl.parse_definition(definition()), baseline_run_id=run['run_id'])
    assert automatic['public']['steps'][-1]['latest_date'] == '2024-01-10'
    assert automatic['baseline']['supplements'] == {}
    active.unlink()
    automatic = auto.plan(store, etl.parse_definition(definition()), baseline_run_id=run['run_id'])
    assert automatic['public']['ready']
    assert automatic['public']['steps'][-1]['latest_date'] == '2024-01-09'
    assert automatic['public']['warnings']
    target = store.root / 'frozen' / 'data'
    auto.freeze_baseline(EtlStore(store), automatic, target)
    assert (target / file.name).read_bytes() == file.read_bytes()
    assert (store.root / 'tushare_active.json').read_bytes() == manifest
    assert not active.exists()


def test_supplement_checksum_is_enforced_even_with_unchanged_stat(store):
    import os
    run, file = completed_candidate(store)
    (store.root / 'snapshot' / file.name).unlink()
    automatic = auto.plan(store, etl.parse_definition(definition()), baseline_run_id=run['run_id'])
    stat = file.stat()
    raw = bytearray(file.read_bytes()); raw[10] ^= 1; file.write_bytes(raw)
    os.utime(file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(CenterError, match='校验和'):
        auto.freeze_baseline(EtlStore(store), automatic, store.root / 'frozen' / 'data')
    assert not (store.root / 'frozen' / 'data').exists()


@pytest.mark.parametrize('invalid', ['failed', 'foreign', 'mapping', 'inventory', 'symlink'])
def test_invalid_candidates_never_enable_download(store, invalid):
    run, file = completed_candidate(store)
    (store.root / 'snapshot' / file.name).unlink()
    if invalid == 'failed': run['status'] = 'FAILED'
    if invalid == 'foreign': run['frozen']['task_sources'] = {}
    if invalid == 'mapping':
        next(r for r in run['frozen']['task_sources']['tushare']['records'] if r['config']['id'] == 'tushare.fund_nav')['config']['mappings'] = []
    if invalid == 'inventory': (store.root / run['steps'][0]['output']['workspace']['path']).write_text('{}')
    if invalid == 'symlink':
        file.rename(file.with_suffix('.other')); file.symlink_to(file.with_suffix('.other'))
    EtlStore(store).save_run(run)
    checked = etl.validate(store, definition(), {**OPTIONS, 'auto_baseline_run_id': run['run_id']})
    assert not checked['valid']


def test_empty_active_table_does_not_silently_use_candidate(store):
    run, _ = completed_candidate(store)
    pq.write_table(pa.table({'date': pa.array([], type=pa.date32())}), store.root / 'snapshot' / 'fund_nav_df.parquet')
    checked = etl.validate(store, definition(), {**OPTIONS, 'auto_baseline_run_id': run['run_id']})
    assert not checked['valid']
    assert checked['auto_plan']['errors'][0]['code'] == 'AUTO_BASELINE_EMPTY'


def test_exclusion_proposal_preserves_true_dependencies_but_not_order_only(store):
    from backend.data_sources.auto_baseline import exclusion_proposal
    from backend.data_sources.etl_dependencies import plan_dependencies
    graph = plan_dependencies(etl.parse_definition(definition()))
    # Explicit custom dependency must not be relaxed just because the registry
    # does not require it; a pure ordering edge does not remove the next node.
    graph.steps[1].inputs = [graph.steps[0].id]; graph.steps[1].after = []
    graph.steps.append(EtlStep(id='independent', name='独立', kind='task', task_id='tushare.fund_company',
                              source_id='tushare', after=['fund_nav']))
    original = graph.model_dump(mode='json')
    proposal = exclusion_proposal(graph, {'calendar'})
    assert proposal['rebuild_dependencies'] is False
    assert [s['id'] for s in proposal['definition']['steps']] == ['independent']
    assert proposal['definition']['steps'][0]['after'] == []
    assert graph.model_dump(mode='json') == original


def test_plan_id_changes_with_selected_candidate_and_rejects_wrong_mode(store):
    run, file = completed_candidate(store)
    (store.root / 'snapshot' / file.name).unlink()
    plain = etl.validate(store, definition(), OPTIONS)
    selected = etl.validate(store, definition(), {**OPTIONS, 'auto_baseline_run_id': run['run_id']})
    assert selected['valid']
    assert selected['auto_plan']['plan_id'] != plain['auto_plan']['plan_id']
    assert not etl.validate(store, definition(), {'mode': 'full', 'auto_baseline_run_id': run['run_id']})['valid']


def test_explicit_acquisition_baseline_reuses_newer_file_without_publishing(store):
    from backend.data_sources.etl_models import EtlRunOptions
    from backend.data_sources.task_workspace import inventory
    run, file = completed_candidate(store)
    pq.write_table(pa.table({'date': [date(2024, 1, 12)], 'adj_nav': [1.3]}), file)
    journal = EtlStore(store)
    run['steps'][0]['output']['workspace'] = inventory(journal, file.parent, file.parent.parent/'workspace.json', ['fund_nav'], 'tushare')
    journal.save_run(run)
    original = (store.root / 'snapshot' / file.name).read_bytes()
    options = EtlRunOptions(mode='auto_incremental', auto_baseline_scope='acquisition')
    automatic = auto.plan(store, etl.parse_definition(definition()), baseline_run_id=run['run_id'], options=options)
    assert automatic['public']['steps'][-1]['latest_date'] == '2024-01-12'
    assert file.name not in automatic['baseline']['files']
    target = store.root / 'frozen' / 'data'; auto.freeze_baseline(journal, automatic, target)
    assert (target / file.name).read_bytes() == file.read_bytes()
    assert (store.root / 'snapshot' / file.name).read_bytes() == original


@pytest.mark.parametrize('case', ['valid', 'second_unknown', 'second_marker', 'second_corrupt', 'failed_dependency'])
def test_sparse_automatic_recovery_preserves_success_and_checks_every_failed_workspace(store, monkeypatch, case):
    import copy
    from backend.data_sources.etl_dependencies import plan_dependencies
    from backend.data_sources.etl_migration import stage_recovery
    from backend.services.etl_recovery import _shape_reason
    from backend.tests.test_etl_dataset_tasks import finish
    graph = plan_dependencies(etl.parse_definition(definition()))
    for identifier, task, previous in [('company', 'fund_company', 'fund_nav'),
                                        ('benchmark', 'fund_benchmark', 'company'),
                                        ('company2', 'fund_company', 'benchmark')]:
        graph.steps.append(EtlStep(id=identifier, name=identifier, kind='task', task_id='tushare.' + task,
                                  source_id='tushare', after=[previous]))
    graph.steps.append(EtlStep(id='blocked', name='blocked', kind='task', task_id='tushare.fund_company',
                              source_id='tushare', inputs=['fund_nav'], after=['company2']))
    def worker(payload, check, lock):
        if payload['task_id'] in {'tushare.fund_nav', 'tushare.fund_benchmark'}:
            raise CenterError('OFFLINE', 'offline failure')
        pq.write_table(pa.table({'code': ['A']}), Path(payload['directory']) / (payload['task_id'].split('.')[-1] + '.parquet'))
        return {'received_rows': 1, 'warnings': 0}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    checked = etl.validate(store, graph.model_dump(mode='json'), OPTIONS)
    assert checked['valid'], checked
    old = finish(store, etl.start(store, {'request_id': uuid.uuid4().hex, 'confirm': True,
        'definition': graph.model_dump(mode='json'), 'options': OPTIONS, 'auto_plan_id': checked['auto_plan']['plan_id']}))
    assert [s['status'] for s in old['steps']] == ['SUCCEEDED', 'SUCCEEDED', 'FAILED', 'SUCCEEDED', 'FAILED', 'SUCCEEDED', 'SKIPPED']
    journal = EtlStore(store)
    second = store.root / 'etl_runs' / old['run_id'] / 'benchmark'
    if case == 'second_unknown': (second / 'work' / 'unknown.partial').write_bytes(b'preserve')
    if case == 'second_marker': (second / 'work_input.json').write_text('{}')
    if case == 'second_corrupt':
        next((second / 'work').glob('*.parquet')).write_bytes(b'corrupt')
    if case == 'failed_dependency':
        old['definition']['steps'][3]['inputs'] = ['fund_nav']
        journal.save_run(old)
        assert _shape_reason(old)
    else:
        assert _shape_reason(old) is None
    original = copy.deepcopy(journal.get_run(old['run_id']))
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'new-sparse-code')
    monkeypatch.setattr(auto, 'cutoff_date', lambda: date(2024, 1, 20))
    if case != 'valid':
        with pytest.raises(CenterError): stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    else:
        new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
        assert [s['status'] for s in new['steps']] == ['SUCCEEDED', 'SUCCEEDED', 'PENDING', 'SUCCEEDED', 'PENDING', 'SUCCEEDED', 'PENDING']
        frozen = journal.get_run(new['run_id'])['frozen']
        assert frozen['auto_plan'] == old['frozen']['auto_plan']
        assert frozen['task_baseline'] == old['frozen']['task_baseline']
        receipt = json.loads(journal.checked_path(new['recovery_receipt']).read_text())
        assert [p['step_id'] for p in receipt['partials']] == ['fund_nav', 'benchmark']
        calls = []
        def repaired(payload, check, lock):
            calls.append(payload['task_id'])
            return {'received_rows': 0, 'warnings': 0}
        monkeypatch.setattr(task_runtime, 'run_worker', repaired)
        done = finish(store, etl.resume(store, new['run_id'], True))
        assert done['status'] == 'SUCCEEDED', done.get('error')
        assert calls == ['tushare.fund_nav', 'tushare.fund_benchmark', 'tushare.fund_company']
    assert journal.get_run(old['run_id']) == original


def test_automatic_graph_recovery_preserves_frozen_baseline_and_dates(store, monkeypatch):
    import copy
    from backend.data_sources.etl_dependencies import plan_dependencies
    from backend.data_sources.etl_migration import stage_recovery
    from backend.services.etl_recovery import _shape_reason
    from backend.data_sources.etl_partial_recovery import _check_unhandled_partial
    graph = plan_dependencies(etl.parse_definition(definition()))
    def worker(payload, check, lock):
        work = Path(payload['directory'])
        if payload['task_id'] == 'tushare.fund_nav':
            raise CenterError('OFFLINE', 'offline failure')
        pq.write_table(pa.table({'code': ['A']}), work / (payload['task_id'].split('.')[-1] + '.parquet'))
        return {'received_rows': 1, 'warnings': 0}
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    checked = etl.validate(store, graph.model_dump(mode='json'), OPTIONS)
    old = etl.start(store, {'request_id': uuid.uuid4().hex, 'confirm': True, 'definition': graph.model_dump(mode='json'),
                          'options': OPTIONS, 'auto_plan_id': checked['auto_plan']['plan_id']})
    journal = EtlStore(store)
    for _ in range(300):
        old = journal.get_run(old['run_id'])
        if old['status'] != 'RUNNING': break
        time.sleep(.02)
    assert old['status'] == 'FAILED'
    original = copy.deepcopy(old)
    assert _shape_reason(old) is None
    _check_unhandled_partial(journal, old)
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'new-code')
    monkeypatch.setattr(auto, 'cutoff_date', lambda: date(2024, 1, 20))
    recovered = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    new = journal.get_run(recovered['run_id'])
    assert new['frozen']['auto_plan'] == old['frozen']['auto_plan']
    assert new['frozen']['task_baseline'] == old['frozen']['task_baseline']
    assert new['auto_plan']['cutoff_date'] == '2024-01-14'
    assert journal.get_run(old['run_id']) == original
    monkeypatch.setattr(task_runtime, 'run_worker', lambda payload, check, lock: {'received_rows': 0, 'warnings': 0})
    etl.resume(store, new['run_id'], True)
    for _ in range(300):
        new = journal.get_run(new['run_id'])
        if new['status'] != 'RUNNING': break
        time.sleep(.02)
    assert new['status'] == 'SUCCEEDED', new.get('error')
