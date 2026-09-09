"""Real API restarts, inherited flock and offline worker completion receipts."""
import copy
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from backend.data_sources import etl_executor as executor
from backend.data_sources import etl_service as etl
from backend.data_sources import task_runtime
from backend.data_sources.credentials import save_credential
from backend.data_sources.etl_store import EtlStore, public_run
from backend.data_sources.models import CenterError
from backend.data_sources.store import SourceStore, utc_now
from backend.data_sources.task_receipt import read_receipt, write_receipt
from backend.services.refresh_runtime import InterProcessFileLock, is_file_lock_held
from backend.tests.test_etl_dataset_tasks import small_plan, request

REPO = Path(__file__).resolve().parents[2]


def eventually(probe, timeout=25):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = probe()
        if result:
            return result
        time.sleep(.1)
    pytest.fail('condition not reached before timeout')


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_SOURCE_CENTER_ENABLED', 'true')
    store = SourceStore(tmp_path)
    store.seed()
    save_credential(store, 'tushare', 'offline-test-credential')
    journal = EtlStore(store)
    children = []
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    client = httpx.Client(base_url=f'http://127.0.0.1:{port}', trust_env=False, timeout=3)

    def start_api():
        child = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'backend.tests.etl_lifecycle_fixture:app',
                                  '--host', '127.0.0.1', '--port', str(port), '--log-level', 'error'], cwd=REPO,
                                 env={**os.environ, 'ETL_TEST_ROOT': str(tmp_path)},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        children.append(child)
        def ready():
            assert child.poll() is None, 'fixture API exited'
            try:
                return client.get('/health').status_code == 200
            except httpx.TransportError:
                return False
        eventually(ready)
        return child

    yield store, journal, client, start_api
    for run in journal.runs():
        journal.request_cancel(run['run_id'])
    for action in ('calendar', 'fund_company'):
        (tmp_path / (action + '.release')).touch()
    eventually(lambda: not is_file_lock_held(tmp_path / '.tushare_refresh.lock'), timeout=12)
    for child in children:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
    client.close()


@pytest.mark.parametrize('api_signal', [signal.SIGTERM, signal.SIGKILL])
def test_api_restart_keeps_worker_and_advances_without_duplicates(api, api_signal):
    store, journal, client, start_api = api
    parent = start_api()
    payload = request(small_plan())
    response = client.post('/api/data-sources/etl/runs', json=payload)
    assert response.status_code == 200, response.text
    identifier = response.json()['run_id']
    eventually(lambda: (store.root / 'calendar.called').exists())
    original = journal.get_run(identifier)
    assert executor.executor_alive(original)
    parent.send_signal(api_signal)
    parent.wait(timeout=10)
    assert is_file_lock_held(store.root / '.tushare_refresh.lock')
    # No API is serving when the executor completes step one and starts step two.
    (store.root / 'calendar.release').touch()
    eventually(lambda: (store.root / 'fund_company.called').exists())
    start_api()
    assert client.get('/health').json()['connected'] == 1
    current = client.get('/api/data-sources/etl/runs/' + identifier).json()
    assert current['status'] == 'RUNNING', current
    assert current['execution']['state'] == 'CONNECTED'
    assert current['steps'][0]['status'] == 'SUCCEEDED'
    assert journal.get_run(identifier)['executor'] == original['executor']
    assert client.post('/api/data-sources/etl/runs', json=payload).json()['run_id'] == identifier
    assert client.post('/api/data-sources/etl/runs', json=request(small_plan())).status_code == 409
    assert client.post(f'/api/data-sources/etl/runs/{identifier}/resume', json={'confirm': True}).status_code == 409
    (store.root / 'fund_company.release').touch()
    eventually(lambda: journal.get_run(identifier)['status'] == 'SUCCEEDED')
    final = client.get('/api/data-sources/etl/runs/' + identifier).json()
    assert [s['attempt'] for s in final['steps']] == [1, 1]
    assert [s['rows'] for s in final['steps']] == [1, 1]
    for step in final['steps']:
        receipt = json.loads((store.root / 'etl_runs' / identifier / step['id'] / '1' / 'worker_result.json').read_text())
        assert receipt['result']['ok'] is True
    assert 'executor' not in final and 'owner_pid' not in final


def test_cancellation_after_api_restart_stops_worker_and_releases_lock(api):
    store, journal, client, start_api = api
    parent = start_api()
    identifier = client.post('/api/data-sources/etl/runs', json=request(small_plan())).json()['run_id']
    eventually(lambda: (store.root / 'calendar.called').exists())
    worker_pid = int((store.root / 'calendar.called').read_text())
    parent.terminate(); parent.wait(timeout=10)
    start_api()
    assert client.post(f'/api/data-sources/etl/runs/{identifier}/cancel', json={}).status_code == 200
    eventually(lambda: journal.get_run(identifier)['status'] == 'CANCELLED')
    eventually(lambda: not is_file_lock_held(store.root / '.tushare_refresh.lock'))
    eventually(lambda: executor.process_birth(worker_pid) is None)
    assert not (store.root / 'fund_company.called').exists()


def saved_run(tmp_path):
    journal = EtlStore(SourceStore(tmp_path))
    run = {'run_id': 'identity', 'request_hash': 'hash', 'status': 'RUNNING', 'attempt': 1,
           'owner_pid': os.getpid(), 'owner_instance': 'owner', 'steps': [],
           'executor': {'protocol': 1, 'pid': os.getpid(), 'birth': executor.process_birth(os.getpid()), 'token': 'a' * 32}}
    journal.save_run(run, create=True)
    return journal, run


def test_pid_reuse_and_stale_poll_cannot_overwrite_new_result(tmp_path, monkeypatch):
    journal, run = saved_run(tmp_path)
    stale = copy.deepcopy(run)
    monkeypatch.setattr(executor, 'process_birth', lambda pid: 'reused')
    run['status'] = 'SUCCEEDED'; journal.save_run(run)
    assert journal.interrupted(stale)['status'] == 'SUCCEEDED'
    run['status'] = 'RUNNING'; journal.save_run(run)
    assert journal.interrupted(run)['status'] == 'INTERRUPTED'


def test_heartbeat_identity_and_age_fail_closed(tmp_path):
    journal, run = saved_run(tmp_path)
    assert executor.observe(journal, run)['execution']['state'] == 'HEARTBEAT_PENDING'
    beat = {**run['executor'], 'run_id': run['run_id'], 'attempt': 1, 'heartbeat_at': utc_now()}
    path = executor.control_path(journal, run)
    journal.write_json(path, beat)
    assert executor.observe(journal, run)['execution']['state'] == 'CONNECTED'
    for changes in ({'token': 'wrong'}, {'heartbeat_at': '2000-01-01T00:00:00+00:00'}, {'attempt': 2}):
        journal.write_json(path, {**beat, **changes})
        assert executor.observe(journal, run)['execution']['state'] == 'HEARTBEAT_PENDING'


def test_legacy_telemetry_is_read_only_and_not_a_completion(tmp_path):
    journal, run = saved_run(tmp_path)
    run.update(status='INTERRUPTED', steps=[{'id': 'calendar', 'kind': 'task', 'status': 'INTERRUPTED', 'attempt': 1}])
    run.pop('executor')
    journal.save_run(run)
    path = tmp_path / 'etl_runs' / run['run_id'] / 'calendar' / '1' / 'progress.json'
    journal.write_json(path, {'activity_at': utc_now(), 'sequence': 10})
    observed = executor.observe(journal, run)
    assert observed['execution']['state'] == 'WORKER_OBSERVED'
    assert observed['steps'][0]['worker_only'] is True
    assert journal.get_run(run['run_id'])['steps'][0] == run['steps'][0]
    assert observed['status'] == 'INTERRUPTED'
    journal.write_json(path, {'activity_at': '2000-01-01T00:00:00+00:00'})
    assert 'execution' not in executor.observe(journal, run)


def test_launch_failure_is_terminal_and_releases_lock(tmp_path, monkeypatch):
    journal, run = saved_run(tmp_path)
    lock = InterProcessFileLock(tmp_path / '.tushare_refresh.lock')
    assert lock.acquire()
    monkeypatch.setattr(executor, 'runner_command', lambda: ['/nonexistent-offline-executor'])
    with pytest.raises(CenterError, match='启动失败'):
        executor.launch(journal, run, lock)
    assert journal.get_run(run['run_id'])['status'] == 'FAILED'
    assert not is_file_lock_held(lock.path)


def test_abnormal_exit_does_not_accept_success_receipt(tmp_path, monkeypatch):
    payload = {'result_path': str(tmp_path / 'step' / '1' / 'worker_result.json'),
               'directory': str(tmp_path / 'step' / 'work'), 'worker_token': 'a' * 32, 'task_id': 'tushare.calendar'}
    program = ('import json,sys; from backend.data_sources.task_receipt import write_receipt; '
               'write_receipt(json.load(sys.stdin), {"ok": True, "result": {"received_rows": 1}}); sys.exit(1)')
    monkeypatch.setattr(task_runtime, 'worker_command', lambda: [sys.executable, '-c', program])
    with InterProcessFileLock(tmp_path / '.tushare_refresh.lock') as lock:
        with pytest.raises(CenterError, match='退出异常'):
            task_runtime.run_worker(payload, lambda: None, lock)


@pytest.mark.parametrize('mutation', ['missing', 'token', 'task', 'oversize', 'malformed', 'no_result', 'path'])
def test_completion_receipts_reject_unverified_success(tmp_path, mutation):
    path = tmp_path / 'step' / '1' / 'worker_result.json'
    payload = {'result_path': str(path), 'directory': str(tmp_path / 'step' / 'work'), 'worker_token': 'a' * 32, 'task_id': 'tushare.calendar'}
    write_receipt(payload, {'ok': True, 'result': {'received_rows': 1}})
    assert read_receipt(payload)['ok'] is True
    if mutation == 'missing': path.unlink()
    elif mutation == 'token': payload['worker_token'] = 'b' * 32
    elif mutation == 'task': payload['task_id'] = 'tushare.fund_company'
    elif mutation == 'oversize': path.write_text(' ' * 131073)
    elif mutation == 'malformed': path.write_text('{')
    elif mutation == 'no_result': write_receipt(payload, {'ok': True})
    elif mutation == 'path': payload['result_path'] = str(tmp_path / 'worker_result.json')
    with pytest.raises(CenterError):
        read_receipt(payload)
