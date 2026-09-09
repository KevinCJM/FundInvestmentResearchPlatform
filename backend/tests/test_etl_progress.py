"""Offline telemetry contracts: no vendor calls, live before child exit."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from backend.data_sources import etl_service as etl, task_runtime
from backend.data_sources.models import CenterError
from backend.data_sources.task_progress import TaskProgressLog, progress_monitor, progress_path
from backend.tests.test_etl_dataset_tasks import store, small_plan, request, finish, fake_worker


def test_explicit_stage_progress_and_redaction_before_persistence(tmp_path):
    path = tmp_path / 'progress.json'
    log = TaskProgressLog(path, 'example-private-credential')
    log.write('[INFO] 净值 进度 50/100，example-private-')
    log.write('credential token=another-private-value\n')
    log.finish()
    value = json.loads(path.read_text())
    assert value['completed'] == 50 and value['total'] == 100
    assert 'example-private' not in path.read_text()
    assert 'another-private' not in path.read_text()
    log.write('[INFO] 第 2/3 次尝试，日期 2026/09/07\n')
    assert log.state['completed'] == 50
    log.write('[STAGE] 已完成拉取，正在合并\n')
    assert log.state['total'] is None
    log.write('[INFO] 流式归并进度 100/200（50.0%）\n')
    assert log.state['phase'] == '合并历史数据' and log.state['unit'] == '行'
    log.write('[DONE] 完成\n')
    assert log.state['total'] is None


def test_thread_safe_bounded_logs_and_throttled_writes(tmp_path, monkeypatch):
    from backend.data_sources import task_progress
    writes = []
    monkeypatch.setattr(task_progress, 'atomic_bytes', lambda p, b: writes.append(b))
    monkeypatch.setattr(task_progress.time, 'monotonic', lambda: 1.0)
    log = TaskProgressLog(tmp_path / 'progress.json', 'private-value')
    def produce(i):
        log.write('[WARN] private-'); log.write('value\n'); log.batch(3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(produce, range(200)))
    assert len(writes) == 1
    log.finish()
    assert len(writes) == 2
    value = json.loads(writes[-1])
    assert value['batches'] == 200 and value['received_rows'] == 600
    assert len(value['logs']) == 20 and log.warnings == 200
    assert 'private' not in writes[-1].decode() and len(log.tail) <= 8000
    log.write('x' * 20000 + '\n'); log.finish()
    assert '超长日志已省略' in log.tail


@pytest.mark.parametrize('text', ['进度 3/2', '进度 0/0', '日期 2026/09/07', '第 1/3 次重试'])
def test_no_fake_percent(text):
    log = TaskProgressLog()
    log.write(text + '\n')
    assert log.state['total'] is None


def test_progress_path_cannot_escape_attempt(tmp_path):
    output = tmp_path / 'step' / 'work'
    assert progress_path({'progress_path': str(tmp_path / 'step/1/progress.json')}, output)
    with pytest.raises(CenterError):
        progress_path({'progress_path': str(tmp_path / 'elsewhere/progress.json')}, output)
    (tmp_path / 'step/1').mkdir(parents=True)
    (tmp_path / 'step/1/progress.json').symlink_to(tmp_path / 'outside')
    with pytest.raises(CenterError):
        progress_path({'progress_path': str(tmp_path / 'step/1/progress.json')}, output)


def test_running_api_receives_progress_and_failure_keeps_last_logs(store, monkeypatch):
    from backend.data_sources.etl_store import EtlStore, public_run
    seen = []
    def worker(payload, check, lock):
        log = TaskProgressLog(Path(payload['progress_path']))
        log.write('[INFO] 持仓 日期进度 100/800\n'); log.batch(42); log.finish()
        check()
        value = public_run(EtlStore(store).runs()[0])
        seen.append(value['steps'][0])
        raise CenterError('OFFLINE_FAILURE', '离线测试失败')
    monkeypatch.setattr(task_runtime, 'run_worker', worker)
    result = finish(store, etl.start(store, request(small_plan())))
    assert seen[0]['status'] == 'RUNNING'
    assert seen[0]['progress']['completed'] == 100
    assert seen[0]['progress']['received_rows'] == 42
    assert seen[0]['heartbeat_at']
    assert result['steps'][0]['progress']['logs'][-1]['message'].endswith('100/800')
    assert result['status'] == 'FAILED'
    def resume_worker(payload, check, lock):
        check()
        if payload['task_id'] == 'tushare.calendar':
            resumed = EtlStore(store).runs()[0]['steps'][0]
            assert resumed['progress']['completed'] is None
            assert resumed['progress']['logs'] == []
            assert 'finished_at' not in resumed and 'error' not in resumed
        return fake_worker(payload, check, lock)
    monkeypatch.setattr(task_runtime, 'run_worker', resume_worker)
    assert finish(store, etl.resume(store, result['run_id'], True))['status'] == 'SUCCEEDED'


@pytest.mark.parametrize('cancel', [False, True])
def test_actual_subprocess_progress_before_exit_and_cancellation(tmp_path, monkeypatch, cancel):
    from backend.services.refresh_runtime import InterProcessFileLock
    real_popen = subprocess.Popen
    children, observed = [], []
    child_code = '''import json, sys, time
from pathlib import Path
from backend.data_sources.task_progress import TaskProgressLog
p = json.load(sys.stdin)
log = TaskProgressLog(Path(p['progress_path']))
log.write('[INFO] 日期进度 2/10\\n'); log.finish()
time.sleep(2)
print(json.dumps({'ok': True, 'result': {'rows': 3}}), flush=True)
'''
    def launch(command, **kwargs):
        child = real_popen([sys.executable, '-c', child_code], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(task_runtime.subprocess, 'Popen', launch)
    class Journal:
        def save_run(self, run):
            if state.get('progress', {}).get('completed') == 2:
                observed.append(children[0].poll())
                if cancel:
                    raise CenterError('ETL_CANCELLED', '取消')
    state = {}
    poll = progress_monitor(Journal(), {}, state, tmp_path / 'progress.json', lambda: None)
    lock = InterProcessFileLock(tmp_path / 'test.lock'); assert lock.acquire(owner='test')
    try:
        if cancel:
            with pytest.raises(CenterError, match='取消'):
                task_runtime.run_worker({'progress_path': str(tmp_path / 'progress.json')}, poll, lock)
            assert children[0].poll() is not None
        else:
            assert task_runtime.run_worker({'progress_path': str(tmp_path / 'progress.json')}, poll, lock) == {'rows': 3}
        assert None in observed  # The live child, not just its final receipt.
    finally:
        lock.release()


def test_broken_or_oversized_telemetry_does_not_break_download(tmp_path):
    class Journal:
        def save_run(self, run): pass
    path = tmp_path / 'progress.json'
    path.write_text('x' * 70000)
    state = {}
    progress_monitor(Journal(), {}, state, path, lambda: None)()
    assert 'progress' not in state and state['heartbeat_at']


def test_real_worker_adapter_flushes_sanitized_log_on_error(store, tmp_path, monkeypatch):
    import T01_get_data as script
    from backend.data_sources.task_worker import acquire
    from backend.data_sources.acquisition import fingerprint
    from backend.data_sources.task_catalog import task_specs
    records = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == 'tushare']
    work = tmp_path / 'step/work'; work.mkdir(parents=True)
    path = tmp_path / 'step/1/progress.json'
    def operation(args, actions, *, client):
        print('[INFO] 净值 进度 25/50，offline-test-credential')
        client.on_batch({'source_rows': 10, 'status': 'EMPTY'})
        raise RuntimeError('offline failure')
    monkeypatch.setattr(script, '_run_actions', operation)
    with pytest.raises(RuntimeError, match='offline failure'):
        acquire({'root': str(store.root), 'source_id': 'tushare', 'source_hash': fingerprint(records),
                 'params': {}, 'mode': 'full', 'progress_path': str(path)}, task_specs()['tushare.calendar'], work)
    value = json.loads(path.read_text())
    assert value['completed'] == 25 and value['received_rows'] == 10
    assert 'offline-test-credential' not in path.read_text()
