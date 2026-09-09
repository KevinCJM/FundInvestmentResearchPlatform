"""API-independent execution identity, heartbeat and read-only reconnection."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .batches import atomic_bytes
from .models import CenterError
from .store import utc_now


def process_birth(pid):
    """A PID alone is not ownership: compare its OS start identity as well."""
    if not isinstance(pid, int) or pid <= 0:
        return None
    try:
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            return None if fields[0] == 'Z' else fields[19]
        value = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'lstart=', '-o', 'stat='],
                               capture_output=True, text=True, timeout=2, check=False).stdout.strip()
        return value.rsplit(None, 1)[0] if value and not value.rsplit(None, 1)[-1].startswith('Z') else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def control_path(journal, run):
    return journal.root / 'etl_runs' / run['run_id'] / 'executor' / f"{run['attempt']}.json"


def read_small_json(path, limit=65536):
    try:
        if path.is_symlink():
            return None
        with path.open('rb') as handle:
            raw = handle.read(limit + 1)
        value = json.loads(raw) if len(raw) <= limit else None
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def executor_alive(run):
    executor = run.get('executor') or {}
    return bool(executor.get('birth') and executor['birth'] == process_birth(executor.get('pid')))


def runner_command():
    return [sys.executable, '-m', 'backend.data_sources.etl_runner']


def launch(journal, run, lock):
    from .etl_store import public_run
    token = uuid.uuid4().hex
    child = None
    try:
        child = subprocess.Popen([*runner_command(), str(journal.root.resolve()), run['run_id'], str(lock.fileno()), token],
                                 cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True, pass_fds=(lock.fileno(),))
        birth = process_birth(child.pid)
        if birth is None:
            raise CenterError('ETL_EXECUTOR_START', '独立执行器启动失败，未启动下载。')
        run.update(owner_pid=child.pid, owner_instance=token,
                   executor={'protocol': 1, 'token': token, 'pid': child.pid, 'birth': birth})
        journal.save_run(run)
    except BaseException:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        current = journal.get_run(run['run_id'])
        if current['status'] == 'RUNNING' and current['attempt'] == run['attempt']:
            current.update(status='FAILED', code='ETL_EXECUTOR_START',
                           error='独立执行器启动失败；已完成结果和检查点保留。', finished_at=utc_now())
            journal.save_run(current)
        raise CenterError('ETL_EXECUTOR_START', '独立执行器启动失败；请检查运行记录后重试。') from None
    finally:
        # LOCK_UN here would also release the child's flock. A plain close is
        # essential, including when the API exits in the handover window.
        lock.close_inherited_copy()
    threading.Thread(target=child.wait, daemon=True, name='etl-reap-' + run['run_id'][:8]).start()
    return public_run(journal.get_run(run['run_id']))


def heartbeat(journal, run):
    stop = threading.Event()
    def pulse():
        while not stop.is_set():
            value = {**run['executor'], 'run_id': run['run_id'], 'attempt': run['attempt'],
                     'heartbeat_at': utc_now()}
            try:
                atomic_bytes(control_path(journal, run), json.dumps(value).encode())
            except OSError:
                pass  # A missing heartbeat is surfaced as unknown, never success.
            stop.wait(2)
    worker = threading.Thread(target=pulse, daemon=True, name='etl-heartbeat')
    worker.start()
    return stop


def observe(journal, run):
    """Project durable progress without adopting or mutating another executor."""
    value = copy.deepcopy(run)
    executor = run.get('executor') or {}
    if executor.get('protocol') == 1 and run['status'] == 'RUNNING':
        beat = read_small_json(control_path(journal, run)) or {}
        matched = all(beat.get(k) == executor.get(k) for k in ('pid', 'birth', 'token')) and beat.get('attempt') == run['attempt'] and beat.get('run_id') == run['run_id']
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat['heartbeat_at'])).total_seconds()
        except (KeyError, TypeError, ValueError):
            age = None
        fresh = matched and age is not None and 0 <= age <= 15
        value['execution'] = {'mode': 'independent', 'state': 'CONNECTED' if fresh else 'HEARTBEAT_PENDING',
                              'heartbeat_at': beat.get('heartbeat_at') if matched else None,
                              'message': ('已连接独立任务执行器；API 服务重启不会停止下载或后续步骤。' if fresh else
                                          '独立执行器进程仍在，正在等待新的心跳；暂不能确认其处理进度，请勿重复启动。')}
        return value
    if run['status'] != 'INTERRUPTED':
        return value
    # Old workers cannot grow a completion protocol after launch. Show their
    # recent telemetry honestly; never manufacture success or start a rival.
    for step in value['steps']:
        if step['status'] != 'INTERRUPTED' or step.get('kind') != 'task' or not step.get('attempt'):
            continue
        path = journal.root / 'etl_runs' / run['run_id'] / step['id'] / str(step['attempt']) / 'progress.json'
        progress = read_small_json(path)
        if not progress:
            continue
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(progress['activity_at'])).total_seconds()
        except (KeyError, TypeError, ValueError):
            continue
        step['progress'] = progress
        step.pop('heartbeat_at', None)
        if 0 <= age <= 30:
            step['worker_only'] = True
            value['execution'] = {'mode': 'legacy', 'state': 'WORKER_OBSERVED', 'heartbeat_at': progress['activity_at'],
                                  'message': '后台下载进度仍在更新，但旧任务的调度已中断；仅恢复进度展示，不能自动推进后续步骤。请勿重复启动。'}
            value['error'] = None
    return value


def reconnect(journal):
    with journal.sources.connection() as db:
        rows = db.execute("SELECT id FROM etl_run WHERE json_extract(body,'$.status')='RUNNING'").fetchall()
    connected = interrupted = 0
    for row in rows:
        result = journal.interrupted(journal.get_run(row[0]))
        connected += int(result['status'] == 'RUNNING' and bool(result.get('executor')))
        interrupted += int(result['status'] == 'INTERRUPTED')
    return {'connected': connected, 'interrupted': interrupted}
