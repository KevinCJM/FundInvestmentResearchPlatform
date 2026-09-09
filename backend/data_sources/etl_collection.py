"""Small collection-time metadata, not historical availability/PIT certification.

Execution windows are conservative evidence for older runs, not batch timestamps.
Never scan data artifacts when projecting warnings for a status request.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from .models import CenterError
from .store import utc_now

ZONE = ZoneInfo('Asia/Shanghai')
BOUNDARY = ('采集时间不等于净值日期、公告日期或历史可得时点；跨日可能混入供应商修订，'
            '但不等于数据必然错误，同日采集也不代表已通过 PIT 校验。')


def _time(value):
    try:
        result = datetime.fromisoformat(value)
        return result if result.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def _collects(run, step):
    if step['kind'] == 'download':
        return True
    if step['kind'] != 'task':
        return False
    spec = run.get('frozen', {}).get('tasks', {}).get(step['id'], {}).get('spec', {})
    if 'network' in spec:
        return spec['network'] is True
    definition = next((s for s in (run.get('definition') or {}).get('steps', [])
                       if s['id'] == step['id']), {})
    if definition.get('task_id'):
        from .task_catalog import get_task
        try:
            return get_task(definition['task_id']).get('network', False)
        except CenterError:
            pass  # Unknown historical tasks cannot certify absence of collection.
    return True


def collection_windows(run):
    """Retain origins and previous attempts before their live counters reset."""
    windows = {(w['run_id'], w['step_id'], w['attempt']): dict(w)
               for w in run.get('collection_history', [])}
    for step in run.get('steps', []):
        if not _collects(run, step) or not (step.get('started_at') or step.get('attempt', 0)):
            continue
        origin = step.get('imported_from', {}).get('run_id', run['run_id'])
        key = (origin, step['id'], step.get('attempt', 1))
        if key in windows:
            continue
        receipt = step.get('progress', {}).get('collection_window', {})
        first, last = _time(receipt.get('first_at')), _time(receipt.get('last_at'))
        basis = 'batch_receipts'
        if not first or not last or last < first:
            first = _time(step.get('started_at'))
            last = _time(step.get('finished_at')) or _time(step.get('progress', {}).get('activity_at')) or first
            basis = 'execution_window'
        if not first or not last or last < first:
            first = last = None
            basis = 'unknown'
        windows[key] = {'run_id': origin, 'step_id': step['id'], 'name': step['name'],
                        'attempt': key[2], 'first_at': first.isoformat() if first else None,
                        'last_at': last.isoformat() if last else None, 'basis': basis}
    return list(windows.values())


def collection_timing(run):
    windows = collection_windows(run)
    times = [stamp.astimezone(ZONE).date().isoformat() for window in windows
             for key in ('first_at', 'last_at') if (stamp := _time(window.get(key)))]
    first, last = (min(times), max(times)) if times else (None, None)
    incomplete = any(w['basis'] != 'batch_receipts' for w in windows)
    warnings = []
    if first and first != last:
        warnings.append({'code': 'ETL_CROSS_DATE_COLLECTION', 'message':
                         f'采集记录已跨日期（{first} 至 {last}，北京时间），下载内容的时点可能不一致。'
                         '允许继续下载；用于研究或回测前，请核对供应商修订及历史可得时点。'})
    if incomplete:
        warnings.append({'code': 'ETL_COLLECTION_TIME_ESTIMATED', 'message':
                         '部分记录没有批次采集时间，仅展示节点执行时间范围或未知时点，不能据此确认 PIT 一致。'})
    return {'timezone': ZONE.key, 'first_date': first, 'last_date': last,
            'cross_date': bool(first and first != last), 'windows': windows,
            'warnings': warnings, 'boundary': BOUNDARY,
            'scope': '本流程及续跑、迁移沿用的采集记录；不覆盖增量基线的全部历史批次。'}


def resume_warnings(run, at=None):
    pending = any(s['status'] != 'SUCCEEDED' and _collects(run, s) for s in run.get('steps', []))
    if not pending:
        return []  # Local mapping/analytics tomorrow is not another collection.
    timing = collection_timing(run)
    today = _time(at or utc_now()).astimezone(ZONE).date().isoformat()
    if timing['first_date'] and any(day != today for day in (timing['first_date'], timing['last_date'])):
        return [{'code': 'ETL_CROSS_DATE_RESUME', 'message':
                 f'跨日期续跑提醒：已有采集记录为 {timing["first_date"]} 至 {timing["last_date"]}，'
                 f'本次将于 {today}（北京时间）继续下载。新旧内容可能来自不同修订时点，PIT 可能不一致；'
                 '这是风险警告，不因跨日期禁止续跑。'}]
    if not timing['first_date'] and (run.get('attempt', 0) or run.get('recovered_from')):
        return [{'code': 'ETL_COLLECTION_TIME_UNKNOWN', 'message':
                 '旧采集时点不完整，无法确认是否跨日期。仍可继续下载，但不能据此确认 PIT 一致。'}]
    return []


def record_resume(run, at):
    run['collection_history'] = collection_windows(run)
    run.setdefault('resume_events', []).append({'attempt': run['attempt'] + 1, 'at': at,
                                               'warnings': resume_warnings(run, at)})
