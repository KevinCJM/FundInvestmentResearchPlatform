"""Explicit, offline import of legacy dataset results into a new execution run.

This is an operator recovery tool, not a fingerprint override. It accepts only
unchanged linear dataset contracts. Historical producer hashes remain attached
to imported results; incomplete/empty outputs never become successful steps.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from . import etl_service as etl
from .acquisition import fingerprint
from .etl_parameters import parse_run_options
from .etl_store import EtlStore, public_run
from .models import CenterError, InterfaceConfig
from .store import SourceStore, utc_now
from .task_runtime import verify_sources
from .task_workspace import clone_file, materialize, read_inventory


def _require(condition, message):
    if not condition:
        raise CenterError('ETL_MIGRATION_BLOCKED', message, 409)


def _policy_comparison(value, allowed, pagination_changes=()):
    """Compare data semantics; exclude only explicitly approved policy fields."""
    if isinstance(value, list):
        return [_policy_comparison(item, allowed, pagination_changes) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: _policy_comparison(item, allowed, pagination_changes) for key, item in value.items()}
    if isinstance(value.get('config'), dict) and value['config'].get('id') in set(allowed) | set(pagination_changes):
        # Old JSON may omit optional defaults which saving through the current
        # model materializes (e.g. request_fields=None). Compare parsed semantics.
        if 'api_name' in value['config']:
            result['config'] = InterfaceConfig.model_validate(value['config']).model_dump(mode='json')
        result.pop('revision', None)
        result.pop('updated_at', None)
        if value['config']['id'] in allowed:
            for key in ('requests_per_minute', 'min_interval_seconds'):
                result['config']['policy'].pop(key, None)
        if value['config']['id'] in pagination_changes:
            result['config']['pagination'].pop('mode', None)
    return result


def _verify_policy_changes(journal, old, allowed, pagination_changes):
    if not allowed and not pagination_changes:
        verify_sources(journal.sources, old['frozen'])
        return []
    changes, found = [], set()
    accepted = set(allowed) | set(pagination_changes)
    for source_id, snapshot in old['frozen']['task_sources'].items():
        current = [r for kind in ('source', 'interface') for r in journal.sources.list(kind)
                   if r['config'].get('source_id', r['config']['id']) == source_id]
        prior = snapshot['records']
        _require(fingerprint(prior) == snapshot['hash'], '旧来源冻结记录不完整。')
        sort = lambda records: sorted(records, key=lambda record: record['config']['id'])
        _require(_policy_comparison(sort(prior), allowed, pagination_changes)
                 == _policy_comparison(sort(current), allowed, pagination_changes),
                 '除已明确接受的限频或分页启用外，来源或接口合同发生变化，拒绝导入。')
        indexed = {r['config']['id']:r for r in current}
        for record in prior:
            identifier = record['config']['id']
            if identifier not in accepted:
                continue
            found.add(identifier)
            _require('api_name' in record['config'], '只能明确接受接口限频变化，不能改变来源认证或地址。')
            if identifier in pagination_changes:
                before = InterfaceConfig.model_validate(record['config'])
                after = InterfaceConfig.model_validate(indexed[identifier]['config'])
                _require(before.api_name == after.api_name == 'fund_portfolio'
                         and before.pagination.mode == 'none' and after.pagination.mode == 'offset'
                         and after.pagination.cursor_param == 'offset' and after.pagination.limit_param == 'limit',
                         '只允许已实测持仓接口从无分页启用 offset/limit，其他分页合同不允许自动迁移。')
                changes.append({'kind':'pagination', 'interface_id':identifier,
                                'source_revision':record['revision'], 'target_revision':indexed[identifier]['revision'],
                                'before':before.pagination.model_dump(), 'after':after.pagination.model_dump()})
            if identifier not in allowed:
                continue
            fields = ('requests_per_minute', 'min_interval_seconds')
            changes.append({'interface_id':identifier, 'source_revision':record['revision'],
                            'target_revision':indexed[identifier]['revision'],
                            'before':{k:record['config']['policy'][k] for k in fields},
                            'after':{k:indexed[identifier]['config']['policy'][k] for k in fields}})
    _require(found == accepted, '接受策略变化的接口不属于旧任务。')
    return changes


def _adjustment_scope_contract(before, after):
    """One explicit correction; not a general task-contract override."""
    before, after = copy.deepcopy(before), copy.deepcopy(after)
    expected = {'name': '公募基金复权因子', 'category': '公募基金', 'requires': ['fund_info']}
    corrected = {'name': 'ETF 复权因子', 'category': 'ETF', 'requires': ['etf_info', 'calendar']}
    _require(before.get('spec', {}).get('id') == after.get('spec', {}).get('id') == 'tushare.fund_adjustment',
             '只能迁移复权因子标的范围。')
    for key in expected:
        _require(before['spec'].get(key) == expected[key] and after['spec'].get(key) == corrected[key],
                 '复权因子范围不是已核定的场外目录到 ETF 目录修正。')
        before['spec'][key] = corrected[key]
    _require(before == after, '除复权因子名称、分类和前置目录外，任务合同不得变化。')


def _contracts(journal, old, rate_changes=(), pagination_changes=(), adjustment_scope=False):
    _require(old['status'] in {'FAILED', 'CANCELLED', 'INTERRUPTED'}, '旧任务必须已经停止。')
    definition = etl.parse_definition(old['definition'])
    options = parse_run_options(old.get('options'))
    _require(options.mode != 'auto_incremental' and not old['frozen'].get('task_baseline'),
             '该恢复工具只支持无基线的旧数据集任务，不能迁移自动增量或已有增量基线。')
    _require([s['id'] for s in old['steps']] == [s.id for s in definition.steps], '步骤记录与定义不一致。')
    policy_evidence = _verify_policy_changes(journal, old, rate_changes, pagination_changes)
    if adjustment_scope:
        affected = [s for s in definition.steps if s.task_id == 'tushare.fund_adjustment']
        first_pending = next((s['id'] for s in old['steps'] if s['status'] != 'SUCCEEDED'), None)
        _require(len(affected) == 1 and affected[0].id == first_pending,
                 '范围修正只允许发生在当前未完成的复权因子节点。')
        affected[0].name = 'ETF 复权因子'
    frozen = etl._freeze(journal.sources, definition, options)
    frozen['rate_policy_migration'] = [item for item in policy_evidence if item.get('kind') != 'pagination']
    frozen['pagination_policy_migration'] = [item for item in policy_evidence if item.get('kind') == 'pagination']
    # A newly available baseline must never change the legacy download mode.
    frozen['task_baseline'] = None
    prior, pending, completed = None, False, []
    for step, state in zip(definition.steps, old['steps']):
        _require(step.kind == 'task' and step.inputs == ([prior] if prior else []) and not step.after,
                 '仅支持已核验的单链数据集任务；其他计算图需单独迁移设计。')
        before = _policy_comparison(old['frozen']['tasks'].get(step.id), rate_changes, pagination_changes)
        after = _policy_comparison(frozen['tasks'].get(step.id), rate_changes, pagination_changes)
        if adjustment_scope and step.task_id == 'tushare.fund_adjustment':
            _adjustment_scope_contract(before, after)
        else:
            _require(before == after, f'{step.name} 的任务、来源或接口合同发生变化，不能导入。')
        if state['status'] == 'SUCCEEDED':
            _require(not pending, '已完成节点必须是连续前缀。')
            output = state.get('output', {})
            _require(output.get('task_id') == step.task_id and output.get('workspace'), '缺少成功数据集清单。')
            value = read_inventory(journal, output['workspace'])
            _require(set(frozen['tasks'][step.id]['spec']['provides']) <= set(value['capabilities']), '清单能力与任务不一致。')
            for entry in value['files'].values():
                path = journal.checked_path(entry)
                if path.suffix == '.parquet':
                    _require(pq.ParquetFile(path).metadata.num_rows == entry['rows'], '数据行数与冻结清单不一致。')
            for artifact in state.get('artifacts', []):
                journal.checked_path(artifact)
            completed.append(state['id'])
        else:
            pending = True
        prior = step.id
    _require(completed and pending, '必须同时存在已完成数据和未完成步骤。')
    return definition, options, frozen, completed


def _parts_name(records, step, filename='fund_portfolio_df'):
    records = sorted(records, key=lambda r: ('source' if 'transport' in r['config'] else 'interface', r['config']['id']))
    config_hash = hashlib.sha256(json.dumps([{'config': r['config'], 'revision': r['revision']} for r in records], sort_keys=True).encode()).hexdigest()
    suffix = hashlib.sha256(config_hash.encode()).hexdigest()[:16]
    return f".{filename}_parts_{step.params['start_date']}_{step.params['end_date']}_{suffix}"


def _adjustment_scope_parts(journal, old, step, predecessor, target, producer_hash, progress, target_frozen):
    """Archive invalid-scope empty markers; never promote them as ETF coverage."""
    from .etl_models import EtlStep
    prior_step = EtlStep.model_validate(next(s for s in old['definition']['steps'] if s['id'] == step.id))
    old_work = journal.root / 'etl_runs' / old['run_id'] / step.id / 'work'
    marker = old_work.parent / 'work_input.json'
    _require(marker.is_file() and not marker.is_symlink() and not old_work.is_symlink(), '缺少原复权因子工作区合同。')
    expected = {'input': predecessor, 'task': prior_step.model_dump(mode='json'), 'execution': producer_hash}
    _require(json.loads(marker.read_text()) == expected, '原复权因子输入合同不一致。')
    _require(not (old_work / 'fund_adj_factor_df.parquet').exists(), '原任务已有复权因子输出，不能按全空范围错误迁移。')
    before = read_inventory(journal, predecessor)
    _require({'etf_info', 'calendar'} <= set(before['capabilities'])
             and 'etf_info_df.parquet' in before['files'], '前置快照缺少 ETF 目录或交易日历。')
    records = old['frozen']['task_sources'][step.source_id]['records']
    parts = old_work / _parts_name(records, prior_step, 'fund_adj_factor_df')
    _require(parts.is_dir() and not parts.is_symlink(), '原复权因子检查点目录无效。')
    # Outside the working cache: old empty responses are audit, not new-scope evidence.
    audit = target.parent / 'scope_recovery_evidence'
    audit.mkdir()
    evidence = []
    for path in sorted(parts.iterdir()):
        _require(path.is_file() and not path.is_symlink()
                 and re.fullmatch(r'[A-Za-z0-9_-]+\.OF\.empty', path.name),
                 '原复权因子存在非空、非场外或未完成分片，须先核验，不能自动丢弃。')
        artifact = journal.artifact(path)
        clone_file(path, audit / path.name)
        imported = journal.artifact(audit / path.name)
        _require(imported['checksum'] == artifact['checksum'], '原空标记审计复制失败。')
        evidence.append({'source': artifact, 'imported': imported})
    progress(f'保留 {len(evidence)} 个错误场外范围的空标记作为审计，不作为 ETF 完成缓存。')
    return {'copied': 0, 'empty_recheck': 0, 'files': evidence,
            'excluded_scope_empty': len(evidence), 'scope_correction': 'off_exchange_to_etf'}


def _dividend_parts(journal, old, step, predecessor, target, producer_hash, progress, target_frozen):
    """Import v4 completed announcement days, including verified zero-row days."""
    from .fund_event_recovery import import_dividend_receipts

    work_root = journal.root / 'etl_runs' / old['run_id'] / step.id
    marker = work_root / 'work_input.json'
    if not marker.exists():
        return {'copied': 0, 'empty_recheck': 0}
    _require(not marker.is_symlink() and not work_root.is_symlink(), '检查点路径不得为符号链接。')
    expected = {'input': predecessor, 'task': step.model_dump(mode='json'), 'execution': producer_hash}
    _require(json.loads(marker.read_text()) == expected, '旧检查点输入、范围或执行合同不一致。')
    records = old['frozen']['task_sources'][step.source_id]['records']
    parts = work_root / 'work' / _parts_name(records, step, 'fund_dividend_df')
    _require(parts.is_dir() and not parts.is_symlink() and not parts.parent.is_symlink(),
             '已执行分红任务缺少预期检查点目录，不能静默全量重取。')
    target_records = target_frozen['task_sources'][step.source_id]['records']
    dest = target / _parts_name(target_records, step, 'fund_dividend_df')
    dest.mkdir()
    result = import_dividend_receipts(journal, parts, dest, step, progress)
    return {'copied': result['complete'], 'empty_recheck': 0, 'files': result['files'],
            'confirmed_empty': result['empty']}


def _portfolio_parts(journal, old, step, predecessor, target, producer_hash, progress=lambda _: None, target_frozen=None):
    """Only import readable nonempty daily holdings shards with matching lineage.

    Legacy empty markers have no independently verifiable result contract, so
    leave them in the old directory for audit and re-query them in the new run.
    Never copy temporary files or a legacy action-success marker.
    """
    if step.task_id != 'tushare.fund_portfolio':
        return {'copied': 0, 'empty_recheck': 0}
    work_root = journal.root / 'etl_runs' / old['run_id'] / step.id
    marker = work_root / 'work_input.json'
    if not marker.exists():
        return {'copied': 0, 'empty_recheck': 0}
    _require(not marker.is_symlink() and not work_root.is_symlink(), '检查点路径不得为符号链接。')
    expected = {'input': predecessor, 'task': step.model_dump(mode='json'), 'execution': producer_hash}
    _require(json.loads(marker.read_text()) == expected, '旧检查点输入、范围或执行合同不一致。')
    records = old['frozen']['task_sources'][step.source_id]['records']
    name = _parts_name(records, step)
    parts = work_root / 'work' / name
    if not parts.exists():
        return {'copied': 0, 'empty_recheck': 0}
    _require(not parts.is_symlink() and not parts.parent.is_symlink(), '检查点路径不得为符号链接。')
    target_records = (target_frozen or old['frozen'])['task_sources'][step.source_id]['records']
    dest = target / _parts_name(target_records, step)
    dest.mkdir()
    evidence, empty_recheck = [], 0
    required = {'ts_code', 'symbol', 'ann_date', 'end_date', 'available_at', 'source_api', 'mkv', 'amount'}
    for part in sorted(parts.iterdir()):
        _require(not part.is_symlink(), '检查点文件不得为符号链接。')
        if part.suffix == '.empty':
            empty_recheck += 1
            continue
        if part.suffix != '.parquet':
            continue
        _require(bool(re.fullmatch(r'\d{8}', part.stem)), '日期分片名称不合法。')
        day = datetime.strptime(part.stem, '%Y%m%d').date()
        _require(step.params['start_date'] <= part.stem <= step.params['end_date'], '分片超出冻结范围。')
        parquet = pq.ParquetFile(part)
        _require(parquet.metadata.num_rows > 0 and required <= set(parquet.schema_arrow.names), '持仓分片缺少必要字段或为空。')
        # Decode every column to reject truncated pages, without recomputing math.
        for batch in parquet.iter_batches(batch_size=16384):
            for row in batch.to_pylist():
                ann, available = row['ann_date'], row['available_at']
                ann = ann.date() if isinstance(ann, datetime) else ann
                available = available.date() if isinstance(available, datetime) else available
                _require(ann == day and available == day and row['source_api'] == 'fund_portfolio'
                         and row['ts_code'] and row['symbol'], '分片日期或来源口径不一致。')
        source = journal.artifact(part)
        clone_file(part, dest / part.name)
        copied = journal.artifact(dest / part.name)
        _require(source['checksum'] == copied['checksum'], '复制后的分片校验和不一致。')
        evidence.append({'source': source, 'imported': copied, 'rows': parquet.metadata.num_rows})
    # Explicit bridge into the current receipt protocol, not a second collector.
    imports = {Path(item['imported']['path']).stem: {'sha256': item['imported']['checksum'],
                                                   'rows': item['rows']} for item in evidence}
    receipt = dest / 'verified_day_imports.json'
    journal.write_json(receipt, {'version': 1, 'producer': producer_hash, 'days': imports})
    from .fund_event_recovery import import_history_receipts
    history = import_history_receipts(journal, parts, dest, step, progress)
    return {'copied': len(evidence), 'empty_recheck': empty_recheck, 'files': evidence,
            'history': history,
            'day_import_receipt': journal.artifact(receipt)}


def stage_recovery(store, identifier: str, request_id: str, *, confirm=False, progress=lambda _: None,
                   rate_changes=(), pagination_changes=(), adjustment_scope=False):
    """Stage a new stopped run. Start it only via the normal, guarded resume API."""
    etl._writable()
    _require(confirm is True, '请明确确认跨版本导入；旧任务不会被修改。')
    try:
        identifier, new_id = uuid.UUID(identifier).hex, uuid.UUID(request_id).hex
    except (ValueError, TypeError, AttributeError):
        raise CenterError('ETL_REQUEST_ID_REQUIRED', '请提供有效的旧运行和新请求 UUID。', 422) from None
    _require(identifier != new_id, '必须创建新的运行 ID。')
    journal = EtlStore(store)
    lock = etl._lock(store)
    try:
        with store.connection() as db:
            found = db.execute('SELECT id FROM etl_run WHERE id=?', (new_id,)).fetchone()
        if found:
            previous = journal.get_run(new_id)
            _require(previous.get('recovered_from') == identifier, '请求 ID 已被其他任务使用。')
            return public_run(previous)
        old = journal.get_run(identifier)
        progress('正在核验旧来源、任务合同和已完成文件的校验和。')
        definition, options, frozen, completed = _contracts(journal, old, rate_changes, pagination_changes, adjustment_scope)
        directory = journal.root / 'etl_runs' / new_id
        _require(not directory.exists(), '新目录已存在，请检查未完成迁移或使用新的请求 ID。')
        directory.mkdir(parents=True)
        # Immutable pre-migration snapshot doubles as a recoverable audit backup.
        backup = journal.write_json(directory / 'recovery' / 'original_run.json', old)
        config = directory / 'config'
        config.mkdir()
        frozen['config_artifacts'] = []
        for artifact in old['frozen']['config_artifacts']:
            source = journal.checked_path(artifact)
            clone_file(source, config / source.name)
            frozen['config_artifacts'].append(journal.artifact(config / source.name))
        steps = []
        producer = old['frozen']['execution_fingerprint']
        for step, state in zip(definition.steps, old['steps']):
            if step.id in completed:
                imported = copy.deepcopy(state)
                imported['imported_from'] = {'run_id': identifier, 'step_id': step.id, 'execution_fingerprint': producer}
                steps.append(imported)
            else:
                steps.append({'id': step.id, 'name': step.name, 'kind': step.kind, 'status': 'PENDING', 'attempt': 0})
        first = definition.steps[len(completed)]
        predecessor = steps[len(completed) - 1]['output']['workspace']
        progress('已完成节点核验通过，正在复制可验证的数据分片和回执。')
        work = directory / first.id / 'work'
        materialize(journal, predecessor, work)
        importer = _dividend_parts if first.task_id == 'tushare.fund_dividend' else _portfolio_parts
        if first.task_id == 'tushare.index_concept':
            from .index_history_recovery import import_ths_segments
            importer = import_ths_segments
        if adjustment_scope:
            importer = _adjustment_scope_parts
        shards = importer(journal, old, first, predecessor, work, producer, progress, frozen)
        journal.write_json(work.parent / 'work_input.json', {
            'input': predecessor, 'task': first.model_dump(mode='json'), 'execution': frozen['execution_fingerprint'],
        })
        receipt = journal.write_json(directory / 'recovery' / 'manifest.json', {
            'format': 'verified_dataset_import_v1', 'original_run': backup,
            'source_execution': producer, 'target_execution': frozen['execution_fingerprint'],
            'rate_policy_changes': frozen['rate_policy_migration'],
            'pagination_policy_changes': frozen['pagination_policy_migration'],
            'adjustment_scope_correction': adjustment_scope,
            'imported_steps': completed, 'partial_step': first.id, 'shards': shards,
            'boundary': '复用旧版本已校验文件，不宣称新旧算法数值等价；无独立复核证据的旧空结果重新查询。',
        })
        _require(etl.execution_fingerprint() == frozen['execution_fingerprint'], '迁移期间代码发生变化，未创建可执行任务。')
        verify_sources(store, frozen)
        from .etl_collection import collection_windows
        run = {'run_id': new_id, 'request_hash': fingerprint(['verified_dataset_import_v1', identifier, new_id]),
               'collection_history': collection_windows(old),
               'recovered_from': identifier, 'recovery_receipt': receipt, 'workflow_id': old.get('workflow_id'),
               'workflow_revision': old.get('workflow_revision'), 'definition': definition.model_dump(mode='json'),
               'template_definition': old.get('template_definition'), 'options': options.model_dump(mode='json'),
               'name': old['name'] + '（恢复）', 'status': 'INTERRUPTED', 'created_at': utc_now(),
               'attempt': 0, 'published': False, 'frozen': frozen, 'steps': steps,
               'error': None, 'message': f'已核验并复用 {len(completed)} 个完成节点、{shards["copied"]} 个完整日期分片、'
               f'{shards.get("history", {}).get("complete", 0)} 个非空区间及 {shards.get("history", {}).get("empty", 0)} 个已复核空区间；'
               f'{shards.get("confirmed_empty", 0)} 个已复核空日期；'
               f'{shards["empty_recheck"]} 个旧空结果将重新查询。等待继续执行。'}
        journal.save_run(run, create=True)
        return public_run(run)
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--confirm', action='store_true')
    parser.add_argument('--accept-rate-change', action='append', default=[],
                        help='Explicit interface ID whose request rate/spacing change may be migrated; all data semantics remain exact.')
    parser.add_argument('--accept-pagination-enable', action='append', default=[],
                        help='Explicit holdings interface enabling verified offset/limit; no other pagination fields may change.')
    parser.add_argument('--accept-fund-adjustment-scope', action='store_true',
                        help='Explicitly correct an all-empty off-exchange factor task to the existing ETF universe.')
    args = parser.parse_args()
    result = stage_recovery(SourceStore(), args.run_id, args.request_id, confirm=args.confirm,
                            rate_changes=args.accept_rate_change,
                            pagination_changes=args.accept_pagination_enable,
                            adjustment_scope=args.accept_fund_adjustment_scope,
                            progress=lambda message: print(message, flush=True))
    print(json.dumps({key: result.get(key) for key in ('run_id', 'recovered_from', 'status', 'message')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
