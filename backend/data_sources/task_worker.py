"""Isolated registered dataset worker. No user code, paths or credentials in argv."""
from __future__ import annotations

import contextlib
import json
import signal
import sys
from pathlib import Path

from .acquisition import fingerprint
from .credentials import read_credential
from .models import CenterError
from .runtime import ConfiguredTushareClient
from .store import SourceStore
from .task_catalog import get_task
from .task_progress import TaskProgressLog, progress_path


TailLog = TaskProgressLog


def acquire(payload: dict, spec: dict, output: Path) -> dict:
    import T01_get_data as script
    store = SourceStore(Path(payload['root']))
    source_id = payload['source_id']
    current = [r for kind in ('source', 'interface') for r in store.list(kind)
               if r['config'].get('source_id', r['config']['id']) == source_id]
    if fingerprint(current) != payload['source_hash']:
        raise CenterError('ETL_CONFIG_CHANGED', '来源配置发生变化，拒绝继续旧任务。', 409)
    token = read_credential(store, source_id)
    log = TailLog(progress_path(payload, output), token)
    import threading
    mutex = threading.Lock()
    failures = set()
    capped = set()
    stats = {'batches': 0, 'mapped_rejected_batches': 0, 'received_rows': 0, 'mapping_issues': []}
    def captured(batch):
        with mutex:
            stats['batches'] += 1
            stats['received_rows'] += batch['source_rows']
            if batch['status'] == 'REJECTED':
                stats['mapped_rejected_batches'] += 1
                if len(stats['mapping_issues']) < 10:
                    stats['mapping_issues'].append({'interface_id': batch['interface_id'], 'batch_id': batch['batch_id']})
        log.batch(batch['source_rows'])
    class TrackedClient(ConfiguredTushareClient):
        def _call(self, interface, **params):
            key = fingerprint([interface.id, params])
            try:
                result = super()._call(interface, **params)
            except Exception as exc:
                with mutex:
                    failures.add(key)
                    if isinstance(exc, CenterError) and exc.code == 'SOURCE_ROW_CAP':
                        capped.add(key)
                    else:
                        capped.discard(key)
                raise
            with mutex:
                failures.discard(key)
                capped.discard(key)
            return result
        def acknowledge_partition(self, api_name, **params):
            # Used only after all replacement event/index shards validate. Never
            # suppress permission/network errors or unresolved/truncated leaves.
            if api_name not in {'fund_portfolio', 'fund_div', 'shibor', 'shibor_lpr', 'repo_daily', *script.INDEX_HISTORY_FILES}:
                return
            key = fingerprint([self.interfaces[api_name].id, params])
            with mutex:
                if key in capped:
                    capped.discard(key)
                    failures.discard(key)
    client = TrackedClient(token, root=store.root, source_id=source_id, capture=True, on_batch=captured)
    args = script.parse_args(['--output-dir', str(output), '--start-date', payload['params'].get('start_date', '20100101'),
                              '--end-date', payload['params'].get('end_date', '20100101')])
    # Runtime policy may only tighten saved source/interface limits.
    args.max_workers = min(args.max_workers, client.source.policy.max_concurrency)
    args.max_calls_per_minute = min(args.max_calls_per_minute, client.source.policy.requests_per_minute)
    args.min_call_interval_sec = max(args.min_call_interval_sec, client.source.policy.min_interval_seconds)
    args.max_retries = min(args.max_retries, client.source.policy.max_attempts)
    for api, attribute in [('fund_basic', 'max_fund_basic_pages'), ('fund_nav', 'max_fund_nav_pages'), ('fund_manager', 'max_fund_manager_pages')]:
        if api in client.interfaces:
            setattr(args, attribute, min(getattr(args, attribute), client.interfaces[api].pagination.max_pages))
    if 'fund_nav' in client.interfaces:
        args.fund_nav_page_size = min(args.fund_nav_page_size, client.interfaces['fund_nav'].pagination.page_size)
    args.limit = None
    automatic = payload.get('auto_step')
    if payload['mode'] == 'auto_incremental' and (not automatic or not payload.get('has_baseline')):
        raise CenterError('AUTO_BASELINE_REQUIRED', '自动增量缺少已冻结基线，拒绝全量回退。')
    args.latest = payload['mode'] in {'incremental', 'auto_incremental'} and payload.get('has_baseline', False)
    if automatic:
        args.automatic_start_date = automatic['start_date']
        if 'query_dates' in automatic:
            args.automatic_event_plan = automatic
        args.max_latest_days = 370  # Planner fails closed for gaps beyond one year.
    args.resume = payload.get('resume', False)
    args.source_configuration_hash = client.configuration_hash
    with contextlib.ExitStack() as stack:
        stack.callback(log.finish)
        stack.enter_context(contextlib.redirect_stdout(log))
        stack.enter_context(contextlib.redirect_stderr(log))
        if automatic and spec['action'] in {'nav', 'fund_nav', 'candle'}:
            # Determine missing codes before today's broad date requests add them.
            # Existing collector filters IDs, clips by lifecycle, checkpoints each
            # missing code and merges into a private copy; no full-market replay.
            from copy import copy
            missing = copy(args)
            missing.latest, missing.missing_only = False, True
            missing.start_date = automatic['history_start']
            script._run_actions(missing, [spec['action']], client=client)
        script._run_actions(args, [spec['action']], client=client)
    if failures:
        # Some legacy collectors deliberately retain partial directories. A full
        # ETL node must not mistake their warnings for complete acquisition.
        marker = script._action_marker_path(output, spec['action'])
        marker.unlink(missing_ok=True)
        raise CenterError('ETL_DATASET_INCOMPLETE', f'有 {len(failures)} 个请求尚未成功，当前节点未完成；请检查权限和接口状态后从检查点继续。')
    stats.update(log_tail=log.tail.replace(token, '[REDACTED]'), warnings=log.warnings,
                 mode='auto_incremental' if automatic else 'incremental' if args.latest else 'full', unrestricted_universe=True)
    quality = output / 'fund_portfolio_df.parquet.quality.meta.json'
    if quality.exists():
        stats['data_quality'] = json.loads(quality.read_text())
        stats['warnings'] = max(1, stats['warnings'])
    return stats


def execute(payload: dict) -> dict:
    spec = get_task(payload['task_id'])
    root = Path(payload['root']).resolve()
    output = Path(payload['directory']).resolve()
    if (root / 'etl_runs') not in output.parents:
        raise CenterError('ETL_WORKSPACE_PATH', '批量任务只允许写入私有 ETL 目录。')
    if spec['handler'] == 'tushare_dataset':
        return acquire(payload, spec, output)
    if spec['handler'] == 'analytics_snapshot':
        import os
        import shutil
        os.environ.pop('TUSHARE_DATA_DIR', None)
        backend = str(Path(__file__).resolve().parents[1])
        if backend not in sys.path:
            sys.path.insert(0, backend)
        runtime_config = output / '.runtime_config'
        if runtime_config.exists():
            shutil.rmtree(runtime_config)
        shutil.copytree(payload['config_dir'], runtime_config)
        from backend.services.instrument_analytics import rebuild_analytics_snapshot
        log = TailLog(progress_path(payload, output))
        with contextlib.ExitStack() as stack:
            stack.callback(log.finish)
            stack.enter_context(contextlib.redirect_stdout(log))
            stack.enter_context(contextlib.redirect_stderr(log))
            result = rebuild_analytics_snapshot(output, workspace_data_dir=runtime_config)
        return {**result, 'warnings': log.warnings, 'snapshot_mode': 'explicit_compatibility_workspace'}
    raise CenterError('ETL_TASK_UNKNOWN', '任务没有已登记的执行器。')


def main():
    def timeout(*_):
        raise CenterError('ETL_TIMEOUT', '数据集任务达到最长运行时间，检查点保留。')
    payload = None
    try:
        payload = json.load(sys.stdin)
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, max(1, payload['timeout']))
        result = {'ok': True, 'result': execute(payload)}
    except CenterError as exc:
        result = {'ok': False, 'code': exc.code, 'message': exc.message}
    except Exception as exc:
        result = {'ok': False, 'code': 'DATASET_TASK_FAILED', 'message': f'数据集任务失败（{type(exc).__name__}），原始检查点保留；未发布数据。'}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    if payload is not None:
        from .task_receipt import write_receipt
        write_receipt(payload, result)
    print(json.dumps(result, ensure_ascii=False, default=str, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
