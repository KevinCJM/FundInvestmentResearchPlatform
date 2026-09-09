"""Verified import of the current THS date-segment checkpoint format."""
from __future__ import annotations

import json
import re
from datetime import date, datetime, time

import pyarrow.parquet as pq

from .models import CenterError
from .task_workspace import clone_file, read_inventory


def _require(condition, message):
    if not condition:
        raise CenterError('ETL_MIGRATION_BLOCKED', message, 409)


def _copy_evidence(journal, path, destination, artifact):
    journal.checked_path(artifact)
    clone_file(path, destination)
    imported = journal.artifact(destination)
    _require(imported['checksum'] == artifact['checksum'], '指数分片复制后校验和不一致。')
    return {'source': artifact, 'imported': imported}


def _validate_segment(path, code, start, end):
    parquet = pq.ParquetFile(path)
    _require(parquet.metadata.num_rows > 0
             and {'ts_code', 'trade_date', 'source_api', 'close'} <= set(parquet.schema_arrow.names),
             '指数日期分片为空或缺少必要字段。')
    seen = set()
    # Decode every column; a readable footer alone does not prove intact pages.
    for batch in parquet.iter_batches(batch_size=16384):
        values = batch.to_pydict()
        for source, actual_code, day in zip(values['source_api'], values['ts_code'], values['trade_date']):
            if isinstance(day, datetime):
                _require(day.tzinfo is None and day.time() == time(), '指数分片交易日类型不一致。')
                day = day.date()
            _require(isinstance(day, date) and start <= day <= end
                     and source == 'ths_daily' and actual_code == code,
                     '指数分片来源、代码或日期超出冻结合同。')
            _require(day not in seen, '指数日期分片存在重复业务键。')
            seen.add(day)
    _require(len(seen) == parquet.metadata.num_rows, '指数分片完整解码行数不一致。')
    return len(seen)


def import_ths_segments(journal, old, step, predecessor, target, producer, progress, frozen):
    """Do not import unproven empty markers or discard unsupported partial APIs."""
    import T01_get_data as script
    from .etl_migration import _parts_name

    _require(step.task_id == 'tushare.index_concept' and not old['frozen'].get('task_baseline'),
             '指数分片导入仅支持无基线的概念行情历史任务。')
    work_root = journal.root / 'etl_runs' / old['run_id'] / step.id
    work = work_root / 'work'
    marker = work_root / 'work_input.json'
    if not work_root.exists():
        return {'copied': 0, 'empty_recheck': 0}
    _require(not work_root.is_symlink() and not work.is_symlink()
             and marker.is_file() and not marker.is_symlink(), '指数检查点工作区合同缺失或路径无效。')
    _require(json.loads(marker.read_text()) == {
        'input': predecessor, 'task': step.model_dump(mode='json'), 'execution': producer,
    }, '原指数分片输入、日期或执行合同不一致。')
    before = read_inventory(journal, predecessor)
    catalog_entry = before['files'].get('index_catalog_df.parquet')
    _require(catalog_entry is not None, '前置快照缺少指数目录。')
    catalog_path = work / 'index_catalog_df.parquet'
    _require(catalog_path.is_file() and not catalog_path.is_symlink()
             and journal.artifact(catalog_path)['checksum'] == catalog_entry['checksum'],
             '原工作区指数目录与冻结前置目录不一致。')
    catalog = pq.read_table(journal.checked_path(catalog_entry), columns=['ts_code', 'quote_source_api']).to_pydict()
    universe = {str(code) for code, api in zip(catalog['ts_code'], catalog['quote_source_api']) if api == 'ths_daily'}
    _require(bool(universe), '冻结目录不包含同花顺指数。')
    name = _parts_name(old['frozen']['task_sources'][step.source_id]['records'], step, 'index_ths_daily_df')
    parts, segments = work / name, work / name / 'segments'
    # Only this producer's first API is supported. Later API/finalized/per-code
    # outputs require their own receipt validation, never a silent full refetch.
    allowed = set(before['files']) | {name}
    _require(all(p.name in allowed and not p.is_symlink() for p in work.iterdir()),
             '概念行情已有其他输出或未知分片，不能静默丢弃后迁移。')
    _require(parts.is_dir() and not parts.is_symlink() and segments.is_dir() and not segments.is_symlink()
             and {p.name for p in parts.iterdir()} == {'segments'},
             '仅支持尚未合并的同花顺日期分片，不能丢弃其他检查点。')
    args = script.parse_args(['--start-date', step.params['start_date'], '--end-date', step.params['end_date']])
    chunks = set(script.iter_date_chunks(args.start_date, args.end_date, args.history_chunk_days))
    dest = target / _parts_name(frozen['task_sources'][step.source_id]['records'], step, 'index_ths_daily_df') / 'segments'
    dest.mkdir(parents=True)
    audit = target.parent / 'index_empty_evidence'
    audit.mkdir()
    evidence, counts, seen = [], {'copied': 0, 'empty_recheck': 0, 'rows': 0}, set()
    for path in sorted(segments.iterdir()):
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)__(\d{8})_(\d{8})\.(parquet|empty)', path.name)
        _require(path.is_file() and not path.is_symlink() and match is not None, '指数分片名称或路径不合法。')
        code, start, end, kind = match.groups()
        _require(code in universe and (start, end) in chunks and path.stem not in seen,
                 '指数分片不属于冻结目录/日期网格或存在冲突标记。')
        seen.add(path.stem)
        artifact = journal.artifact(path)
        if kind == 'empty':
            _require(path.read_bytes() == b'no data\n', '旧指数空标记内容无效。')
            item = _copy_evidence(journal, path, audit / path.name, artifact)
            item['recheck_required'] = True
            counts['empty_recheck'] += 1
        else:
            rows = _validate_segment(path, code, datetime.strptime(start, '%Y%m%d').date(), datetime.strptime(end, '%Y%m%d').date())
            item = _copy_evidence(journal, path, dest / path.name, artifact)
            item['rows'] = rows
            counts['copied'] += 1
            counts['rows'] += rows
        evidence.append(item)
        if len(evidence) % 250 == 0:
            progress(f'已核验 {len(evidence)} 个指数分片：复用 {counts["copied"]}，空结果待复核 {counts["empty_recheck"]}。')
    return {**counts, 'files': evidence, 'format': 'verified_ths_segments_v1',
            'missing_segments': len(universe) * len(chunks) - len(seen),
            'pit_boundary': '保持原文件及采集审计；历史行情不包含历史供应商修订时点证明。'}
