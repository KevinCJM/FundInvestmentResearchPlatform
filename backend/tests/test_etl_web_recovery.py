"""Web recovery stays asynchronous, observable and uses the guarded real importer."""
import copy
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.services import etl_recovery as recovery, etl_routes
from backend.data_sources import etl_partial_recovery as partial_recovery
from backend.services.refresh_runtime import InterProcessFileLock
from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.etl_store import EtlStore
from backend.tests.test_etl_dataset_tasks import fake_worker, finish
from backend.tests.test_etl_migration import old_run
from backend.tests.test_etl_executor import eventually
from backend.tests import test_etl_dataset_tasks as dataset_fixtures, test_etl_executor as executor_fixtures

store = dataset_fixtures.store
api = executor_fixtures.api


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(etl_routes, 'get_store', lambda: store)
    monkeypatch.setattr(recovery, 'process_birth', lambda pid: 'offline-birth')
    app = FastAPI(); app.include_router(etl_routes.router)
    return TestClient(app, client=('127.0.0.1', 1234), base_url='http://127.0.0.1')


def queued(store, monkeypatch, client, old):
    jobs = []
    monkeypatch.setattr(recovery, '_launch', lambda s, j, lock: jobs.append(j))
    payload = {'request_id': uuid.uuid4().hex, 'confirm': True}
    response = client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json=payload)
    assert response.status_code == 202, response.text
    assert response.json()['status'] == 'QUEUED'
    assert len(jobs) == 1
    return jobs[0], payload


@pytest.mark.parametrize('same_version', [True, False])
def test_click_queues_before_gb_reads_then_reuses_success_and_starts(store, monkeypatch, client, same_version):
    old = old_run(store, monkeypatch)
    original = copy.deepcopy(EtlStore(store).get_run(old['run_id']))
    if same_version:
        monkeypatch.setattr(etl, 'execution_fingerprint', lambda: old['frozen']['execution_fingerprint'])
    with monkeypatch.context() as m:
        m.setattr(EtlStore, 'checked_path', lambda *a: pytest.fail('HTTP submission must not read GB files'))
        job, payload = queued(store, monkeypatch, client, old)
        response = client.get(f'/api/data-sources/etl/runs/{old["run_id"]}')
        assert response.json()['recovery']['job']['phase'] == '等待校验'
        assert 'owner' not in response.json()['recovery']['job']
        duplicate = client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json=payload)
        assert duplicate.json()['id'] == job['id']
    calls = []
    def worker(payload, check, lock):
        calls.append(payload['task_id'])
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    recovery.execute(store, job)
    status = recovery.job_status(store, old['run_id'])
    assert status['status'] == 'SUCCEEDED', status
    assert status['target_run_id'] == (old['run_id'] if same_version else job['id'])
    result = finish(store, {'run_id': status['target_run_id']})
    assert result['status'] == 'SUCCEEDED' and not result['published']
    assert calls == ['tushare.fund_company']
    assert any('校验' in item['message'] for item in status['logs'])
    if not same_version:
        assert EtlStore(store).get_run(old['run_id']) == original
        assert client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/resume', json={'confirm': True}).status_code == 409
        value = client.get(f'/api/data-sources/etl/runs/{old["run_id"]}').json()
        assert value['recovery']['successor']['run_id'] == job['id']


@pytest.mark.parametrize('case', ['changed_file', 'held_lock', 'config', 'partial', 'version_race', 'unexpected_error'])
def test_recovery_failures_surface_after_click_preserve_original(store, monkeypatch, client, case):
    old = old_run(store, monkeypatch)
    original = copy.deepcopy(EtlStore(store).get_run(old['run_id']))
    job, _ = queued(store, monkeypatch, client, old)
    lock = InterProcessFileLock(store.root / '.tushare_refresh.lock')
    if case == 'held_lock':
        assert lock.acquire(owner='existing-download')
    elif case == 'changed_file':
        artifact = old['steps'][0]['output']['workspace']
        from backend.data_sources.task_workspace import read_inventory
        file = next(iter(read_inventory(EtlStore(store), artifact)['files'].values()))
        EtlStore(store).checked_path(file).write_bytes(b'corrupt')
    elif case == 'config':
        old_frozen = EtlStore(store).get_run(old['run_id'])
        old_frozen['frozen']['tasks']['company']['spec']['name'] = 'different contract'
        EtlStore(store).save_run(old_frozen); original = copy.deepcopy(old_frozen)
    elif case == 'partial':
        work = store.root / 'etl_runs' / old['run_id'] / 'company' / 'work'
        (work / 'unknown.partial').write_bytes(b'keep-me')
    elif case == 'version_race':
        monkeypatch.setattr(etl, 'execution_fingerprint', lambda: 'third-execution')
    else:
        monkeypatch.setattr(partial_recovery, '_check_unhandled_partial', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('secret-token-value')))
    try:
        recovery.execute(store, job)
        status = recovery.job_status(store, old['run_id'])
        assert status['status'] == 'FAILED'
        assert status['code']
        assert 'secret-token-value' not in json.dumps(status)
        assert EtlStore(store).get_run(old['run_id']) == original
        assert not recovery.successor_map(store).get(old['run_id'])
        assert client.get(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery').json() == status
        if case == 'partial': assert (work / 'unknown.partial').read_bytes() == b'keep-me'
    finally:
        lock.release()


def test_same_version_unsupported_migration_shape_does_not_block_normal_resume(store, monkeypatch, client):
    old = old_run(store, monkeypatch)
    monkeypatch.setattr(etl, 'execution_fingerprint', lambda: old['frozen']['execution_fingerprint'])
    monkeypatch.setattr(recovery, '_shape_reason', lambda *a: pytest.fail('same-version needs no migration'))
    monkeypatch.setattr(partial_recovery, '_check_unhandled_partial', lambda *a, **kw: pytest.fail('same-version checkpoint contract unchanged'))
    monkeypatch.setattr(task_runtime, 'run_worker', fake_worker)
    job, _ = queued(store, monkeypatch, client, old)
    recovery.execute(store, job)
    assert recovery.job_status(store, old['run_id'])['status'] == 'SUCCEEDED'
    finish(store, {'run_id': old['run_id']})


@pytest.mark.parametrize('case', ['unconfirmed', 'invalid_id', 'override', 'cross_origin', 'read_only'])
def test_request_boundary_fails_before_spawn(store, monkeypatch, client, case):
    old = old_run(store, monkeypatch)
    monkeypatch.setattr(recovery, '_launch', lambda *a: pytest.fail('must not spawn'))
    payload = {'request_id': uuid.uuid4().hex, 'confirm': True}
    if case == 'unconfirmed': payload['confirm'] = False
    if case == 'invalid_id': payload['request_id'] = '../escape'
    if case == 'override': payload['accept_rate_change'] = ['tushare.ths_daily']
    if case == 'read_only': monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'false')
    response = client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json=payload,
                           headers={'Origin': 'https://attacker.invalid'} if case == 'cross_origin' else {})
    assert response.status_code in {403, 422}


def test_api_restart_read_only_reconnect_dead_owner_and_retry(store, monkeypatch, client):
    old = old_run(store, monkeypatch)
    job, payload = queued(store, monkeypatch, client, old)
    # A fresh API instance has no in-memory job registry to recover.
    app = FastAPI(); app.include_router(etl_routes.router)
    fresh = TestClient(app)
    assert fresh.get(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery').json()['id'] == job['id']
    monkeypatch.setattr(recovery, 'process_birth', lambda pid: 'different-owner')
    assert fresh.get(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery').json()['status'] == 'INTERRUPTED'
    # Reading did not rewrite the persisted source state.
    assert json.loads(recovery._path(store, old['run_id']).read_text())['status'] == 'QUEUED'


def test_successor_redirect_is_idempotent_and_transitive(store, monkeypatch, client):
    old = old_run(store, monkeypatch)
    from backend.data_sources.etl_migration import stage_recovery
    new = stage_recovery(store, old['run_id'], uuid.uuid4().hex, confirm=True)
    second = stage_recovery(store, new['run_id'], uuid.uuid4().hex, confirm=True)
    monkeypatch.setattr(recovery, '_launch', lambda *a: pytest.fail('must not launch duplicate'))
    value = client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json={'request_id':uuid.uuid4().hex,'confirm':True}).json()
    assert value['status'] == 'SUCCEEDED' and value['target_run_id'] == second['run_id']


def test_jobs_global_gate_rejects_another_request(store, monkeypatch, client):
    old = old_run(store, monkeypatch)
    gate = InterProcessFileLock(store.root / '.etl_recovery.lock')
    assert gate.acquire(owner='other-recovery')
    try:
        response = client.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json={'request_id':uuid.uuid4().hex,'confirm':True})
        assert response.status_code == 409
        assert response.json()['detail']['code'] == 'ETL_RECOVERY_RUNNING'
    finally:
        gate.release()


@pytest.mark.parametrize('case', ['valid', 'valid_import', 'valid_prior_audit', 'orphan_parquet', 'unknown_directory', 'old_protocol', 'temporary', 'symlink', 'bad_import', 'bad_audit'])
def test_event_recovery_never_silently_discards_unknown_partial_files(tmp_path, case):
    from backend.data_sources.models import CenterError
    parts = tmp_path / 'parts'; parts.mkdir()
    events = parts / ('events_v4_' + 'a' * 20); events.mkdir()
    (events / 'contract.json').write_text('{}')
    if case in {'valid', 'valid_import', 'valid_prior_audit'}:
        if case == 'valid_import': (parts / 'verified_day_imports.json').write_text('{"version":1,"producer":"old-code","days":{}}')
        if case == 'valid_prior_audit':
            audit = events / 'requery_evidence'; audit.mkdir()
            (audit / '20260901_market.json').write_text('{"status":"SPLIT","date":"20260901","code":null}')
        (events / '20260901_market.json').write_text('{"status":"COMPLETE"}')
        (events / '20260901_market.parquet').write_bytes(b'validated by dedicated importer later')
        partial_recovery._check_event_layout(parts)
        return
    if case == 'orphan_parquet': (events / 'orphan.parquet').write_bytes(b'preserve')
    if case == 'unknown_directory': (events / 'pending_pages').mkdir()
    if case == 'old_protocol': (parts / 'events_v3_old').mkdir()
    if case == 'temporary': (parts / 'download.tmp').write_bytes(b'preserve')
    if case == 'symlink': (parts / '20260901.parquet').symlink_to(events / 'contract.json')
    if case == 'bad_import': (parts / 'verified_day_imports.json').write_text('{"version":1,"producer":"old","days":{"20260901":{"sha256":"bad","rows":1}}}')
    if case == 'bad_audit':
        audit = events / 'requery_evidence'; audit.mkdir()
        (audit / '20260901_market.json').write_text('{"status":"COMPLETE","date":"20260901","code":null}')
    with pytest.raises(CenterError, match='原数据保留'):
        partial_recovery._check_event_layout(parts)


def test_real_detached_recovery_survives_api_exit_and_resumes_with_no_redownload(api, monkeypatch):
    from backend.services.refresh_runtime import is_file_lock_held
    store, journal, http, start_api = api
    original_fingerprint = etl.execution_fingerprint
    monkeypatch.setattr(etl, '_launch', etl._launch_inline)
    old = old_run(store, monkeypatch)
    monkeypatch.setattr(etl, 'execution_fingerprint', original_fingerprint)
    parent = start_api()
    response = http.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json={'request_id':uuid.uuid4().hex,'confirm':True})
    assert response.status_code == 202, response.text
    eventually(lambda: (store.root / 'recovery.called').exists())
    try:
        parent.kill(); parent.wait(timeout=10)
        assert is_file_lock_held(store.root / '.etl_recovery.lock')
        start_api()
        status = http.get(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery').json()
        assert status['status'] in recovery.ACTIVE
        # Repeated clicks while the API has restarted reconnect the same job.
        again = http.post(f'/api/data-sources/etl/runs/{old["run_id"]}/recovery', json={'request_id':uuid.uuid4().hex,'confirm':True})
        assert again.json()['id'] == response.json()['id']
    finally:
        (store.root / 'recovery.release').touch()
        (store.root / 'fund_company.release').touch()
    eventually(lambda: recovery.job_status(store, old['run_id'])['status'] not in recovery.ACTIVE)
    status = recovery.job_status(store, old['run_id'])
    assert status['status'] == 'SUCCEEDED', status
    eventually(lambda: journal.get_run(old['run_id'])['status'] == 'SUCCEEDED')
    assert not (store.root / 'calendar.called').exists()  # Completed prefix never requested again.
    assert (store.root / 'fund_company.called').exists()
    eventually(lambda: not is_file_lock_held(store.root / '.etl_recovery.lock'))
