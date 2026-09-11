"""Read-only, snapshot-bound acquisition planning (metadata/I/O, not analytics)."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

from .acquisition import fingerprint
from .models import CenterError

# No inference from a filename or a user's arbitrary endpoint. These contracts
# describe the existing registered collectors, not additional provider access.
DAILY = {
    'nav': ('etf_daily_df.parquet', 'date'),
    'fund_nav': ('fund_nav_df.parquet', 'date'),
    'candle': ('etf_daily_candle_df.parquet', 'date'),
    'etf_share': ('etf_share_size_df.parquet', 'date'),
}
REFERENCE = {'etf_info', 'fund_info', 'fund_company', 'calendar', 'stock_basic',
             'index_info', 'etf_index', 'index_catalog', 'fund_manager',
             'fund_benchmark', 'macro_cycle', 'macro_money_credit', 'macro_release_calendar'}
DERIVED = {'fund_scale', 'index_coverage', ''}
HISTORY = {
    **{key: [value] for key, value in DAILY.items()},
    'index_domestic': [('index_daily_df.parquet', 'trade_date')],
    'index_industry': [('index_sw_daily_df.parquet', 'trade_date'), ('index_ci_daily_df.parquet', 'trade_date')],
    'index_concept': [('index_ths_daily_df.parquet', 'trade_date'), ('index_dc_daily_df.parquet', 'trade_date'), ('index_tdx_daily_df.parquet', 'trade_date')],
    'index_global': [('index_global_daily_df.parquet', 'trade_date')],
    'index_futures': [('index_futures_daily_df.parquet', 'trade_date')],
    'index_valuation': [('index_daily_basic_df.parquet', 'trade_date')],
    'fund_portfolio': [('fund_portfolio_df.parquet', 'available_at')],
    'fund_dividend': [('fund_dividend_df.parquet', 'available_at')],
    'fund_adjustment': [('fund_adj_factor_df.parquet', 'date')],
    'macro_rates': [('macro_shibor_df.parquet', 'observation_date'), ('macro_lpr_df.parquet', 'observation_date'), ('macro_repo_daily_df.parquet', 'observation_date')],
}


def supported(action: str) -> bool:
    return action in HISTORY or action in REFERENCE or action in DERIVED


def cutoff_date() -> date:
    # Yesterday is a request boundary, never a claim that the vendor has already
    # disclosed everything. Persisted latest NAV advances only on returned data.
    return datetime.now(ZoneInfo('Asia/Shanghai')).date() - timedelta(days=1)


def _date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def date_bounds(path: Path, column: str) -> tuple[date, date]:
    """Read Parquet statistics; only project date columns when stats are absent."""
    source = pq.ParquetFile(path)
    index = source.schema_arrow.get_field_index(column)
    if index < 0 or not source.metadata.num_rows:
        raise CenterError('AUTO_BASELINE_EMPTY', f'{path.name} 无有效日期基线，请先初始化该数据集。', 422)
    dates = []
    for group in range(source.metadata.num_row_groups):
        stats = source.metadata.row_group(group).column(index).statistics
        if stats is not None and stats.has_min_max:
            dates.extend((_date(stats.min), _date(stats.max)))
        else:
            for batch in source.iter_batches(columns=[column], row_groups=[group], batch_size=65536, use_threads=False):
                valid = [_date(value) for value in batch.column(0).to_pylist() if value is not None]
                if valid:
                    dates.extend((min(valid), max(valid)))
    if not dates:
        raise CenterError('AUTO_BASELINE_EMPTY', f'{path.name} 没有可用日期，不能自动推断。', 422)
    return min(dates), max(dates)


def active_snapshot(root: Path) -> Path:
    from backend.market_data import resolve_tushare_data_dir, read_active_manifest
    try:
        # Never fall back to an older directory when an active manifest is bad.
        if read_active_manifest(root) is None:
            raise ValueError('missing manifest')
        path = resolve_tushare_data_dir(root, strict=True).resolve()
        if root.resolve() not in path.parents:
            raise ValueError('outside data root')
        return path
    except (RuntimeError, OSError, ValueError) as exc:
        raise CenterError('AUTO_SNAPSHOT_REQUIRED', '没有有效活跃快照，无法自动增量；请先初始化并激活数据快照。', 422) from exc


def plan(store, definition, *, today_cutoff: date | None = None, baseline_run_id: str | None = None, options=None) -> dict:
    from .task_catalog import get_task
    from .auto_baseline import candidate_choices, exclusion_proposal, supplemental_files
    from .etl_models import EtlRunOptions
    from . import event_coverage
    options = options or EtlRunOptions(mode='auto_incremental')
    end = today_cutoff or cutoff_date()
    root = store.root.resolve()
    snapshot = active_snapshot(root)
    files = {p.name: file_identity(p) for p in snapshot.glob('*.parquet') if p.is_file() and not p.is_symlink()}
    wanted = {name: (column, get_task(s.task_id)['api_slots']) for s in definition.steps if s.kind == 'task'
              for name, column in HISTORY.get(get_task(s.task_id)['action'], [])
              if options.auto_baseline_scope == 'acquisition' or not (snapshot / name).exists() and not (snapshot / name).is_symlink()}
    supplements = {}
    if baseline_run_id:
        from .etl_store import EtlStore
        try:
            supplements = supplemental_files(store, EtlStore(store).get_run(baseline_run_id), wanted, end)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CenterError('AUTO_CANDIDATE_INVALID', '所选候选文件或日期无效，请重新选择基线；不会启动下载。', 422) from exc
    choices = candidate_choices(store, wanted, end)
    # Permit explicit use of a completed acquisition without publishing it.
    # Never replace an existing baseline with an older candidate.
    for name, entry in list(supplements.items()):
        if name in files:
            column = wanted[name][0]
            if date_bounds(snapshot / name, column)[1] > _date(entry['latest_date']):
                del supplements[name]
            else:
                del files[name]
    calendar = snapshot / 'trade_day_df.parquet'
    if calendar.name not in files:
        raise CenterError('AUTO_CALENDAR_REQUIRED', '活跃快照缺少交易日历，不能推断自动增量范围。', 422)
    entries = pq.read_table(calendar, columns=['exchange', 'cal_date', 'is_open']).to_pylist()
    sse = [r for r in entries if str(r['exchange']).upper() == 'SSE']
    days = sorted({_date(r['cal_date']) for r in sse if str(r['is_open']) == '1'})
    if not sse:
        raise CenterError('AUTO_CALENDAR_REQUIRED', '交易日历缺少上交所记录。', 422)
    if max(_date(r['cal_date']) for r in sse) < end and not any(s.task_id == 'tushare.calendar' for s in definition.steps):
        raise CenterError('AUTO_CALENDAR_STALE', '交易日历尚未覆盖昨日，请先刷新交易日历，再预览自动增量。', 422)
    bound, steps, issues = definition.model_copy(deep=True), [], []
    needed = {'trade_day_df.parquet'}
    sources = {}
    for step in bound.steps:
        if step.kind not in {'task'}:
            raise CenterError('AUTO_DATASET_TASK_REQUIRED', '自动增量使用已登记的数据集节点；单接口流程请使用手动增量，或切换自动增量下载工作台。', 422)
        spec = get_task(step.task_id)
        action = spec['action']
        needed.update(name for name, _ in HISTORY.get(action, []))
        if action.startswith('macro_'):
            needed.update(name for name in files if name.startswith('macro_'))  # Preserve small vintage tables.
        if action == 'index_coverage':
            needed.update(name for name in files if name.startswith('index_'))
        if spec['network'] and step.source_id != 'tushare':
            raise CenterError('AUTO_SOURCE_MISMATCH', '当前活跃快照仅可作为原 Tushare 来源的基线，不可混用其他来源。', 422)
        if step.source_id and step.source_id not in sources:
            sources[step.source_id] = [r for kind in ('source', 'interface') for r in store.list(kind)
                                       if r['config'].get('source_id', r['config']['id']) == step.source_id]
        if spec['network']:
            configs = {r['config']['id']: r['config'] for r in sources[step.source_id]}
            if configs[step.source_id].get('transport') != 'tushare' or any(
                configs.get(step.source_id + '.' + api, {}).get('api_name') != api for api in spec['api_slots']
            ):
                raise CenterError('AUTO_SOURCE_MISMATCH', '来源或接口已改为其他协议/数据，不能沿用活跃 Tushare 快照。', 422)
        item = {'id': step.id, 'name': step.name, 'strategy': 'refresh', 'latest_date': None,
                'start_date': end.strftime('%Y%m%d'), 'end_date': end.strftime('%Y%m%d'),
                'files': [], 'message': '基础信息整表刷新；不重取基金净值历史。'}
        issue_start = len(issues)
        if not supported(action):
            issues.append({'code': 'AUTO_UNSUPPORTED', 'message': f'{step.name} 暂不支持自动推断日期；请从本次流程排除，或单独选择手动更新。'})
        elif action in HISTORY:
            bounds = []
            for name, column in HISTORY[action]:
                if name not in files and name not in supplements:
                    available = any(any(f['name'] == name for f in c['files']) for c in choices)
                    issues.append({'code': 'AUTO_BASELINE_NOT_ACTIVE' if available else 'AUTO_BASELINE_REQUIRED', 'file': name,
                                   'message': (f'{step.name}（{name}）已有完成的下载，但未在活跃快照中启用；可选择已完成下载补足基线。'
                                               if available else f'{step.name} 的活跃快照缺少 {name}；请先初始化，或从本次流程排除。不会自动转为全量。')})
                    continue
                try:
                    source = snapshot / name if name in files else root / supplements[name]['path']
                    first, latest = date_bounds(source, column)
                    if latest > end:
                        raise CenterError('AUTO_FUTURE_BASELINE', f'{step.name} 最新日期晚于昨日，请检查快照日期。', 422)
                    previous = [d for d in days if d <= latest]
                    start = previous[-5] if len(previous) >= 5 else (previous[0] if previous else latest)
                    bounds.append((first, latest, start))
                    item['files'].append({'name': name, 'latest_date': latest.isoformat(),
                                          'baseline_run_id': supplements.get(name, {}).get('run_id')})
                    if action in event_coverage.EVENT_FILES:
                        import hashlib, json
                        records = sorted(sources[step.source_id], key=lambda r: ('source' if 'transport' in r['config'] else 'interface', r['config']['id']))
                        source_hash = hashlib.sha256(json.dumps([{'config': r['config'], 'revision': r['revision']} for r in records], sort_keys=True).encode()).hexdigest()
                        coverage = event_coverage.load(source, source_hash)
                        details = event_coverage.plan_dates(first, latest, end, coverage,
                            revision_interval=options.event_revision_interval_days,
                            revision_window=options.event_revision_window_days, purpose=options.event_update_purpose)
                        item.update(details)
                        ledger = event_coverage.sidecar(source)
                        if ledger.exists():
                            needed.add(ledger.name)
                            if name in files:
                                files[ledger.name] = file_identity(ledger)
                            else:
                                # The chosen candidate inventory must cover both bytes
                                # and ledger; a stray adjacent JSON is not evidence.
                                from .auto_baseline import candidate_inventory
                                entry = candidate_inventory(store, EtlStore(store).get_run(baseline_run_id))['files'].get(ledger.name)
                                if not entry:
                                    raise CenterError('EVENT_COVERAGE_INVALID', '候选清单缺少查询覆盖凭据。')
                                supplements[ledger.name] = {**entry, **file_identity(ledger), 'run_id': baseline_run_id}
                except (ValueError, OSError, CenterError) as exc:
                    issues.append({'code': getattr(exc, 'code', 'AUTO_DATE_INVALID'), 'file': name,
                                   'message': getattr(exc, 'message', f'{name} 日期或文件无效。')})
            if bounds:
                first, latest, start = min(b[0] for b in bounds), min(b[1] for b in bounds), min(b[2] for b in bounds)
                item.update(strategy='incremental', latest_date=latest.isoformat(), start_date=start.strftime('%Y%m%d'),
                            history_start=first.strftime('%Y%m%d'), message='回查最近 5 个交易日，补充新增日期；迟报数据以下次实际返回为准。')
                if action in {'nav', 'fund_nav', 'candle'}:
                    item['message'] += ' 无历史代码单独补齐，已有代码不重拉全历史。'
                if action in event_coverage.EVENT_FILES and 'query_dates' in item:
                    item['start_date'] = item['query_dates'][0] if item['query_dates'] else end.strftime('%Y%m%d')
                    item['message'] = (f'新增/缺少覆盖 {item["new_query_days"]} 天，修订复核 {item["revision_query_days"]} 天，'
                                       f'复用已查区间 {item["reused_query_days"]} 天。按公告日查询，不按季度或净值交易日猜测。')
                    if not item['coverage_known']:
                        item['message'] += ' 旧数据没有查询覆盖凭据，仅从最新公告日开始；不代表更早历史已完整。'
                    config = configs.get('tushare.fund_portfolio' if action == 'fund_portfolio' else 'tushare.fund_div', {})
                    item['request_estimate'] = {'minimum': len(item['query_dates']),
                        'page_ceiling': len(item['query_dates']) * config.get('pagination', {}).get('max_pages', 1),
                        'note': '单轮分页安全上限，不含有界重试和空页复核；发现跨页重复时会完整复读该公告日。新增基金身份补历史单独显示。'}
                if (end - start).days > 366:
                    issues.append({'code': 'AUTO_GAP_TOO_LARGE', 'message': f'{step.name} 缺口超过一年，请先分段手动补齐，避免自动任务意外长时间运行。'})
        elif action in DERIVED:
            item.update(strategy='derive', message='读取前置完整候选数据重新生成，不访问外部接口。')
        if action == 'calendar':
            calendar_end = max(_date(r['cal_date']) for r in sse)
            item.update(latest_date=calendar_end.isoformat(), start_date=(min(calendar_end, end) + timedelta(days=1)).strftime('%Y%m%d'),
                        message='只补充日历未覆盖的日期；已覆盖截止日时复用现有日历。')
            # The collector itself skips an already-covered calendar. Bound
            # dates still need to satisfy the registered request schema.
            item['start_date'] = min(item['start_date'], item['end_date'])
        if spec['parameters']:
            step.params = {'start_date': item['start_date'], 'end_date': item['end_date']}
        step.mode = 'inherit'
        for issue in issues[issue_start:]:
            issue['step_id'] = step.id
        if len(issues) > issue_start:
            item.update(strategy='blocked', message='未安排下载：请先处理该节点的拦截项。')
        steps.append(item)
    snapshot_id = snapshot.relative_to(root).as_posix()
    files = {name: identity for name, identity in files.items() if name in needed}
    public = {'snapshot': snapshot_id, 'cutoff_date': end.isoformat(), 'lookback_trade_days': 5,
              'steps': steps, 'errors': issues, 'ready': not issues, 'published': False,
              'supplemental_baseline': {'run_id': baseline_run_id, 'files': list(supplements), 'scope': options.auto_baseline_scope},
              'warnings': ([('使用已完成下载作为本次私有采集基线' if options.auto_baseline_scope == 'acquisition' else '仅用已完成下载补足缺失表') + '；不激活或覆盖研究快照。采集跨日期，使用前仍需核验修订与 PIT。'] if supplements else [])}
    public['plan_id'] = fingerprint({'plan': public, 'definition': definition.compiled().model_dump(mode='json'),
                                    'files': files, 'supplements': supplements, 'sources': sources})
    # Discovery/proposals are advisory and cannot invalidate a confirmed plan
    # merely because an unrelated task completed in the meantime.
    public['baseline_choices'] = choices
    public['exclusion_proposal'] = exclusion_proposal(definition, {e['step_id'] for e in issues})
    return {'public': public, 'definition': bound, 'baseline': {'snapshot': snapshot_id, 'files': files, 'supplements': supplements}}


def freeze_baseline(journal, automatic: dict, directory: Path) -> dict:
    """Freeze a read-verified copy, never expose a writable active-data path."""
    from .task_workspace import clone_file, inventory
    baseline = automatic['baseline']
    snapshot = (journal.root / baseline['snapshot']).resolve()
    if active_snapshot(journal.root) != snapshot:
        raise CenterError('AUTO_PLAN_CHANGED', '活跃快照已经切换，请重新预览。', 409)
    directory.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        for name, identity in baseline['files'].items():
            source = snapshot / name
            if source.is_symlink() or file_identity(source) != identity:
                raise CenterError('AUTO_PLAN_CHANGED', '快照文件已经变化，请重新预览。', 409)
            target = directory / name
            created.append(target)
            clone_file(source, target)
            if file_identity(source) != identity:
                raise CenterError('AUTO_PLAN_CHANGED', '复制期间快照文件变化，拒绝启动。', 409)
        for name, entry in baseline.get('supplements', {}).items():
            from .auto_baseline import candidate_path
            source = candidate_path(journal.root, entry)
            if file_identity(source) != {k: entry[k] for k in ('size', 'mtime_ns')}:
                raise CenterError('AUTO_PLAN_CHANGED', '补充基线或活跃快照已变化，请重新预览。', 409)
            journal.checked_path(entry)
            target = directory / name
            created.append(target)
            clone_file(source, target)
            # Verify copied bytes, not just metadata, before any network task.
            journal.checked_path({'path': target.relative_to(journal.root).as_posix(), 'checksum': entry['checksum']})
        from .event_coverage import load as load_coverage
        for name in ('fund_portfolio_df.parquet', 'fund_dividend_df.parquet'):
            if (directory / name).exists():
                load_coverage(directory / name, verify=True)
        if active_snapshot(journal.root) != snapshot:
            raise CenterError('AUTO_PLAN_CHANGED', '复制期间活跃快照切换，拒绝启动。', 409)
        return inventory(journal, directory, directory.parent / 'workspace.json', [], 'tushare')
    except Exception:
        # Only discard copies created by this failed initialization, never the
        # active snapshot or an existing ETL checkpoint.
        for target in created:
            target.unlink(missing_ok=True)
        directory.rmdir()
        raise
