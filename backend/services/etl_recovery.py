"""Observable Web recovery using the same verified importer as the operator CLI.

This is control-plane orchestration, not another collector. No policy override,
fingerprint rewrite, checkpoint deletion or publication is allowed here.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from backend.data_sources import etl_service as etl
from backend.data_sources.batches import atomic_bytes
from backend.data_sources.etl_executor import process_birth, read_small_json
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.models import CenterError
from backend.data_sources.store import SourceStore, utc_now
from backend.services.refresh_runtime import InterProcessFileLock

ACTIVE = {'QUEUED', 'RUNNING'}


def _uuid(value):
    try:
        return uuid.UUID(value).hex
    except (ValueError, TypeError, AttributeError):
        raise CenterError('ETL_REQUEST_ID_REQUIRED', '恢复请求需要有效的任务和请求 ID。', 422) from None


def _path(store, identifier):
    return store.root / 'etl_recoveries' / (_uuid(identifier) + '.json')


def _write(store, job):
    job['updated_at'] = utc_now()
    atomic_bytes(_path(store, job['source_run_id']), json.dumps(job, ensure_ascii=False).encode())


def _public(job):
    if not job:
        return None
    result = {k: v for k, v in job.items() if k not in {'owner', 'execution_fingerprint'}}
    owner = job.get('owner', {})
    if job['status'] in ACTIVE and (not owner.get('birth') or process_birth(owner.get('pid')) != owner['birth']):
        result.update(status='INTERRUPTED', message='恢复执行器已退出，原数据保留。请重新检查后恢复；不要删除检查点。')
    return result


def job_status(store, identifier):
    return _public(read_small_json(_path(store, identifier)))


def successor_map(store):
    """Use durable JSON metadata, including runs beyond the UI's 30 rows.

    The supported macOS SQLite build does not necessarily include JSON1.
    """
    with store.connection() as db:
        rows = db.execute('SELECT body FROM etl_run').fetchall()
    records = [json.loads(row[0]) for row in rows]
    direct = {row['recovered_from']: {'run_id': row['run_id'], 'status': row['status'], 'name': row['name']}
              for row in sorted(records, key=lambda r: (r.get('created_at', ''), r['run_id'])) if row.get('recovered_from')}
    result = {}
    for parent, child in direct.items():
        seen = {parent}
        while child['run_id'] in direct and child['run_id'] not in seen:
            seen.add(child['run_id'])
            child = direct[child['run_id']]
        result[parent] = child
    return result


def _shape_reason(run):
    if run.get('options', {}).get('mode') == 'auto_incremental' or run.get('frozen', {}).get('task_baseline'):
        return '自动增量或已有基线的跨版本迁移暂不支持；旧数据保留，不能跳过基线核验。'
    prior, pending, count = None, False, 0
    definitions = run.get('definition', {}).get('steps', [])
    if len(definitions) != len(run['steps']):
        return '任务记录与定义不一致，需核验原任务。'
    for definition, step in zip(definitions, run['steps']):
        if (definition['id'] != step['id'] or definition['kind'] != 'task'
                or definition.get('inputs', []) != ([prior] if prior else []) or definition.get('after')):
            return '此计算图尚无可验证的跨版本迁移规则，不能直接复用数据。'
        if step['status'] == 'SUCCEEDED':
            if pending:
                return '已完成节点不是连续前缀，暂不能跨版本恢复。'
            count += 1
        else:
            pending = True
        prior = step['id']
    return None if count and pending else '跨版本恢复需要同时存在已完成数据和未完成步骤。'


def describe(store, run, recovery, successors=None):
    value = dict(recovery)
    successor = (successors if successors is not None else successor_map(store)).get(run['run_id'])
    job = job_status(store, run['run_id'])
    value.update(successor=successor, job=job)
    return value


def guard_successor(store, identifier):
    successor = successor_map(store).get(identifier)
    if successor:
        raise CenterError('ETL_RUN_SUPERSEDED', '此任务已有恢复后的后续任务，请查看后续任务，不能重复启动原任务。', 409)


def _check_unhandled_partial(journal, old):
    """Never let the CLI's generic zero-shard result discard unknown partial work."""
    state = next(s for s in old['steps'] if s['status'] != 'SUCCEEDED')
    definition = next(s for s in old['definition']['steps'] if s['id'] == state['id'])
    if definition['task_id'] == 'tushare.index_concept':
        return  # The concept importer validates the whole work directory.
    directory = journal.root / 'etl_runs' / old['run_id'] / state['id']
    marker, work = directory / 'work_input.json', directory / 'work'
    if not marker.exists() and not work.exists():
        return
    from backend.data_sources.task_workspace import read_inventory
    predecessor = next(s for s in old['steps'] if s['id'] == definition['inputs'][0])['output']['workspace']
    expected = {'input': predecessor, 'task': etl.parse_definition(old['definition']).steps[len([s for s in old['steps'] if s['status'] == 'SUCCEEDED'])].model_dump(mode='json'),
                'execution': old['frozen']['execution_fingerprint']}
    if marker.is_symlink() or read_small_json(marker, 1048576) != expected or work.is_symlink() or not work.is_dir():
        raise CenterError('ETL_MIGRATION_BLOCKED', '原工作区身份不一致，不能自动恢复。', 409)
    files = read_inventory(journal, predecessor)['files']
    allowed = set(files)
    if definition['task_id'] in {'tushare.fund_portfolio', 'tushare.fund_dividend'}:
        from backend.data_sources.etl_migration import _parts_name
        from backend.data_sources.etl_models import EtlStep
        name = _parts_name(old['frozen']['task_sources'][definition['source_id']]['records'],
                           EtlStep.model_validate(definition), definition['task_id'].split('.')[-1] + '_df')
        if (work / name).exists():
            allowed.add(name)
            _check_event_layout(work / name)
    if {p.name for p in work.iterdir()} != allowed:
        raise CenterError('ETL_PARTIAL_MIGRATION_UNSUPPORTED', '当前节点已有尚不支持迁移的部分文件；已下载数据保留，需增加对应校验规则，不能直接丢弃重下。', 409)
    for name, artifact in files.items():
        path = work / name
        if path.is_symlink() or not path.is_file() or journal.artifact(path)['checksum'] != artifact['checksum']:
            raise CenterError('ETL_ARTIFACT_CHANGED', '原工作区数据与前置清单不一致，停止恢复。', 409)


def _check_event_layout(parts):
    """Unreferenced files are not implicitly disposable migration inputs."""
    def require(ok):
        if not ok:
            raise CenterError('ETL_PARTIAL_MIGRATION_UNSUPPORTED', '基金检查点含未支持的部分文件，需核验后才能迁移；原数据保留。', 409)
    require(parts.is_dir() and not parts.is_symlink())
    for item in parts.iterdir():
        require(not item.is_symlink())
        if item.is_file():
            require(bool(re.fullmatch(r'\d{8}\.(parquet|empty)', item.name)))
            continue
        require(item.is_dir() and bool(re.fullmatch(r'events_v4_[a-f0-9]{20}', item.name)))
        for evidence in item.iterdir():
            require(evidence.is_file() and not evidence.is_symlink() and evidence.suffix in {'.json', '.parquet'})
            if evidence.suffix == '.parquet':
                receipt = read_small_json(evidence.with_suffix('.json'), 1048576)
                require(bool(receipt and receipt.get('status') == 'COMPLETE'))


def start(store, identifier, payload):
    etl._writable()
    identifier, request_id = _uuid(identifier), _uuid(payload.get('request_id'))
    if payload.get('confirm') is not True or set(payload) - {'request_id', 'confirm'}:
        raise CenterError('CONFIRM_ETL_RECOVERY', '请确认校验迁移与跨日期风险；此入口不允许覆盖版本或修改采集策略。', 422)
    previous = job_status(store, identifier)
    if previous and (previous['id'] == request_id or previous['status'] in ACTIVE):
        return previous
    journal = EtlStore(store)
    run = journal.interrupted(journal.get_run(identifier))
    successor = successor_map(store).get(identifier)
    if successor or run['status'] == 'RUNNING':
        return {'id': request_id, 'source_run_id': identifier, 'target_run_id': successor['run_id'] if successor else identifier,
                'status': 'SUCCEEDED', 'phase': '已有下载任务', 'message': '已有后续任务或原任务正在运行，请查看其进度；未重复启动。',
                'created_at': utc_now(), 'updated_at': utc_now(), 'logs': []}
    if run['status'] not in {'FAILED', 'INTERRUPTED', 'CANCELLED'}:
        raise CenterError('ETL_NOT_RESUMABLE', '该任务已完成，无需恢复。', 409)
    gate = InterProcessFileLock(store.root / '.etl_recovery.lock')
    if not gate.acquire(owner='etl-recovery'):
        raise CenterError('ETL_RECOVERY_RUNNING', '已有恢复校验正在运行，请等待完成。', 409)
    try:
        latest = job_status(store, identifier)
        if latest and latest['status'] in ACTIVE:
            return latest
        job = {'id': request_id, 'source_run_id': identifier, 'target_run_id': identifier,
               'status': 'QUEUED', 'phase': '等待校验', 'message': '已接收恢复请求，正在启动独立校验任务。',
               'created_at': utc_now(), 'logs': [], 'execution_fingerprint': etl.execution_fingerprint(),
               'owner': {'pid': os.getpid(), 'birth': process_birth(os.getpid()), 'token': uuid.uuid4().hex}}
        _write(store, job)
        _launch(store, job, gate)
        return job_status(store, identifier)
    finally:
        gate.close_inherited_copy()  # Never unlock the child's shared flock.


def runner_command():
    return [sys.executable, '-m', 'backend.services.etl_recovery']


def _launch(store, job, gate):
    child = None
    try:
        child = subprocess.Popen([*runner_command(),
                                  str(store.root.resolve()), job['source_run_id'], str(gate.fileno()), job['owner']['token']],
                                 cwd=Path(__file__).resolve().parents[2], stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True, pass_fds=(gate.fileno(),))
        birth = process_birth(child.pid)
        if not birth:
            raise RuntimeError('missing process identity')
        job['owner'].update(pid=child.pid, birth=birth)
        _write(store, job)
        threading.Thread(target=child.wait, daemon=True, name='etl-recovery-reaper').start()
    except Exception:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        job.update(status='FAILED', code='ETL_RECOVERY_START', message='恢复执行器未能启动，原数据保留。')
        _write(store, job)
        raise CenterError('ETL_RECOVERY_START', job['message']) from None


def execute(store, job):
    """Only the detached owner writes progress. API restarts merely reconnect."""
    from backend.data_sources.etl_migration import stage_recovery
    mutex, stop = threading.Lock(), threading.Event()

    def update(message, **values):
        with mutex:
            job.update(values, message=message)
            job['logs'] = [*job['logs'], {'at': utc_now(), 'message': message}][-12:]
            _write(store, job)

    def pulse():
        while not stop.wait(2):
            with mutex:
                _write(store, job)

    beat = threading.Thread(target=pulse, daemon=True, name='etl-recovery-heartbeat')
    beat.start()
    try:
        guard_successor(store, job['source_run_id'])
        if job['execution_fingerprint'] != etl.execution_fingerprint():
            raise CenterError('ETL_IMPLEMENTATION_CHANGED', '恢复执行器启动前代码已变化，请重新检查后恢复。', 409)
        update('正在核验任务合同和部分文件；不会删除旧数据。', status='RUNNING', phase='核验兼容性')
        journal = EtlStore(store)
        old = journal.interrupted(journal.get_run(job['source_run_id']))
        status = etl.recovery_status(store, old)
        blockers = [item for item in status['blockers'] if item['code'] != 'ETL_IMPLEMENTATION_CHANGED']
        if blockers:
            raise CenterError(blockers[0]['code'], '\n'.join(item['message'] for item in blockers), 409)
        if not status['can_resume']:
            reason = _shape_reason(old)
            if reason:
                raise CenterError('ETL_MIGRATION_BLOCKED', reason, 409)
            _check_unhandled_partial(journal, old)
            update('正在校验并复用已完成文件，大文件可能耗时；可刷新页面查看状态。', phase='校验与迁移')
            staged = stage_recovery(store, job['source_run_id'], job['id'], confirm=True, progress=update)
            update(staged['message'], phase='启动下载', target_run_id=staged['run_id'])
        else:
            update('正在核验原任务已完成文件；版本一致，校验通过后从断点续跑。', phase='校验与续跑')
        result = etl.resume(store, job['target_run_id'], True)
        update('恢复已完成，下载任务已启动，请查看后续任务进度。', status='SUCCEEDED', phase='恢复完成',
               target_run_id=result['run_id'])
    except Exception as exc:
        update(exc.message if isinstance(exc, CenterError) else '恢复校验未完成，原数据和检查点保留；请查看原因后再试。',
               status='FAILED', phase='恢复未完成', code=exc.code if isinstance(exc, CenterError) else 'ETL_RECOVERY_FAILED')
    finally:
        stop.set(); beat.join(timeout=3)


def main():
    root, identifier, descriptor, token = sys.argv[1:]
    store = SourceStore(Path(root))
    gate = InterProcessFileLock.from_inherited_fd(store.root / '.etl_recovery.lock', int(descriptor))
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = read_small_json(_path(store, identifier))
            owner = (job or {}).get('owner', {})
            if owner.get('token') == token and owner.get('pid') == os.getpid() and owner.get('birth') == process_birth(os.getpid()):
                break
            time.sleep(.05)
        else:
            return  # Parent did not persist ownership: never start migration.

        def timeout(*_):
            raise CenterError('ETL_RECOVERY_TIMEOUT', '恢复校验超过两小时，检查点保留；请检查磁盘或执行器后重试。')
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(7200)
        execute(store, job)
    finally:
        signal.alarm(0)
        gate.release()


if __name__ == '__main__':
    main()
