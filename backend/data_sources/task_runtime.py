"""Generic registered-task execution and immutable workspace chaining."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from .acquisition import fingerprint
from .models import CenterError
from .task_catalog import get_task
from .task_workspace import inventory, materialize, read_inventory


def freeze_tasks(store, definition, frozen, options):
    tasks = frozen.get('tasks', {})
    if not tasks:
        return
    source_ids = {step.source_id for step in definition.steps if step.kind == 'task' and step.source_id}
    frozen['task_sources'] = {}
    for source_id in sorted(source_ids):
        records = [r for kind in ('source', 'interface') for r in store.list(kind)
                   if r['config'].get('source_id', r['config']['id']) == source_id]
        frozen['task_sources'][source_id] = {'records': records, 'hash': fingerprint(records)}
    # Same topology, source revisions and start boundary. A later end date is
    # permitted; changing the starting scope must not silently reuse an old base.
    topology = [{'task': s.task_id, 'source': s.source_id, 'inputs': s.inputs,
                 'params': {k: v for k, v in s.params.items() if k != 'end_date'}, 'strategy': s.mode} for s in definition.steps if s.kind == 'task']
    frozen['task_baseline_key'] = fingerprint({'topology': topology, 'sources': frozen['task_sources'], 'execution': frozen.get('execution_fingerprint')})
    frozen['task_baseline'] = None
    if options.mode in {'full', 'auto_incremental'}:
        return
    cutoff = max((s.params.get('end_date', '') for s in definition.steps if s.kind == 'task'), default='')
    with store.connection() as db:
        rows = db.execute("SELECT body FROM etl_run WHERE json_extract(body,'$.frozen.task_baseline_key')=? AND json_extract(body,'$.status')='SUCCEEDED' ORDER BY updated_at DESC LIMIT 30", (frozen['task_baseline_key'],)).fetchall()
    for row in rows:
        previous = json.loads(row[0])
        dates = [s['params'].get('end_date', '') for s in previous['definition']['steps'] if s['kind'] == 'task']
        if max(dates, default='') > cutoff:
            continue
        output = next((s.get('output') for s in reversed(previous['steps']) if s['kind'] == 'task' and s['status'] == 'SUCCEEDED'), None)
        if output and output.get('workspace'):
            from .etl_store import EtlStore
            read_inventory(EtlStore(store), output['workspace'])
            frozen['task_baseline'] = {'run_id': previous['run_id'], 'workspace': output['workspace']}
            break


def verify_sources(store, frozen):
    for source_id, snapshot in frozen.get('task_sources', {}).items():
        current = [r for kind in ('source', 'interface') for r in store.list(kind)
                   if r['config'].get('source_id', r['config']['id']) == source_id]
        if fingerprint(current) != snapshot['hash']:
            raise CenterError('ETL_CONFIG_CHANGED', '数据集任务的来源或接口已改变，请创建新运行。', 409)


def worker_command():
    return [sys.executable, '-m', 'backend.data_sources.task_worker']


def run_worker(payload, check, lock):
    root = Path(__file__).resolve().parents[2]
    with subprocess.Popen(worker_command(),
                          cwd=root, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, pass_fds=(lock.fileno(),)) as child:
        try:
            child.stdin.write(json.dumps(payload)); child.stdin.close(); child.stdin = None
            while True:
                check()
                try:
                    stdout, _ = child.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            raise
    try:
        if payload.get('result_path'):
            from .task_receipt import read_receipt
            result = read_receipt(payload)
        else:
            result = json.loads(stdout)
    except (ValueError, TypeError):
        raise CenterError('ETL_TASK_PROCESS', '数据集工作进程异常退出；可从检查点继续。') from None
    if child.returncode != 0 and result.get('ok'):
        raise CenterError('ETL_TASK_PROCESS', '工作进程退出异常，不能仅凭完成回执确认成功。')
    if not result.get('ok'):
        raise CenterError(result.get('code', 'ETL_TASK_FAILED'), result.get('message', '数据集任务未完成。'))
    return result['result']


def execute_task(journal, run, step, states, directory, check, lock, timeout):
    from .etl_parameters import download_mode, parse_run_options
    spec = get_task(step.task_id)
    frozen = run['frozen']
    verify_sources(journal.sources, frozen)
    if step.inputs:
        predecessor = states[step.inputs[0]]['output']['workspace']
    else:
        predecessor = (frozen.get('task_baseline') or {}).get('workspace')
    # Incomplete task checkpoints are private and may be reused within this run.
    work = directory.parent / 'work'
    marker = directory.parent / 'work_input.json'
    identity = {'input': predecessor, 'task': step.model_dump(mode='json'), 'execution': frozen['execution_fingerprint']}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise CenterError('ETL_WORKSPACE_CHANGED', '工作区的输入合同已改变，不能沿用检查点。')
        before = read_inventory(journal, predecessor) if predecessor else {'files': {}, 'capabilities': []}
    else:
        if work.exists():
            shutil.rmtree(work)
        before = materialize(journal, predecessor, work)
        journal.write_json(marker, identity)
    mode = download_mode(step.mode, parse_run_options(run.get('options')))
    for artifact in frozen['config_artifacts']:
        journal.checked_path(artifact)
    payload = {'root': str(journal.root.resolve()), 'directory': str(work.resolve()),
               'task_id': step.task_id, 'source_id': step.source_id, 'params': step.params,
               'mode': mode, 'has_baseline': bool(frozen.get('task_baseline')),
               'source_hash': frozen.get('task_sources', {}).get(step.source_id, {}).get('hash'),
               'resume': states[step.id]['attempt'] > 1, 'timeout': timeout,
               'config_dir': str((journal.root / 'etl_runs' / run['run_id'] / 'config').resolve())}
    if frozen.get('auto_plan'):
        payload['auto_step'] = next(item for item in frozen['auto_plan']['steps'] if item['id'] == step.id)
    from .task_progress import progress_monitor
    payload['progress_path'] = str((directory / 'progress.json').resolve())
    payload['result_path'] = str((directory / 'worker_result.json').resolve())
    payload['worker_token'] = uuid.uuid4().hex
    journal.write_json(directory / 'worker_contract.json', {
        'protocol': 1, 'task_id': step.task_id, 'token': payload['worker_token'],
        'run_id': run['run_id'], 'step_id': step.id, 'attempt': states[step.id]['attempt'],
        'execution_fingerprint': frozen['execution_fingerprint'],
    })
    poll = progress_monitor(journal, run, states[step.id], directory / 'progress.json', check)
    try:
        result = run_worker(payload, poll, lock)
    finally:
        # Preserve the last logs on success, failure, timeout and cancellation.
        progress_monitor(journal, run, states[step.id], directory / 'progress.json', lambda: None)()
    check()
    states[step.id]['progress'].update(phase='校验输出文件', message='下载工作进程已结束，正在校验文件并生成候选清单。', completed=None, total=None)
    journal.save_run(run)
    capability = list(dict.fromkeys(before['capabilities'] + spec['provides']))
    artifact = inventory(journal, work, directory / 'workspace.json', capability, step.source_id or before.get('source_id'))
    return {**result, 'workspace': artifact, 'capabilities': capability, 'task_id': step.task_id,
            'rows': result.get('received_rows', result.get('rows', 0)), 'published': False,
            'baseline_run_id': (frozen.get('task_baseline') or {}).get('run_id'),
            'boundary': '私有兼容数据集及来源映射候选；不是已合并发布的标准研究数据。'}
