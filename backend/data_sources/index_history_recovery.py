"""Verified concept/futures recovery, including derived per-code/merged files."""
from __future__ import annotations

import json
import re
import tempfile
from datetime import date, datetime, time
from itertools import zip_longest
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .models import CenterError
from .index_checkpoints import read_empty_evidence
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


def _validate_segment(path, code, start, end, api='ths_daily'):
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
                     and source == api and actual_code == code,
                     '指数分片来源、代码或日期超出冻结合同。')
            _require(day not in seen, '指数日期分片存在重复业务键。')
            seen.add(day)
    _require(len(seen) == parquet.metadata.num_rows, '指数分片完整解码行数不一致。')
    return len(seen)


def _same_values(actual, expected):
    """Exact data validation, allowing Arrow null/dtype promotion during merge."""
    try:
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError:
        raise CenterError('ETL_MIGRATION_BLOCKED', '指数汇总文件与日期分片内容不一致。', 409) from None


def _derived_evidence(journal, parts, final, universe, chunks, segment_map, audit, progress):
    """Validate derived outputs, retain as audit, rebuild after empty rechecks.

    A merged file cannot establish that an old `no data` marker was independently
    confirmed. Leaf segments and separately verified empty receipts own coverage.
    """
    import T01_get_data as script

    evidence, complete, nonempty = [], set(), []
    for path in sorted(parts.iterdir()):
        if path.name == 'segments':
            continue
        _require(path.is_file() and not path.is_symlink() and path.suffix in {'.parquet', '.empty'}
                 and path.stem in universe and path.stem not in complete,
                 '指数代码汇总检查点名称、代码或类型无效。')
        code = path.stem
        artifact = journal.artifact(path)
        leaves = segment_map.get(code, {})
        _require(set(leaves) == chunks, '代码汇总缺少完整日期分片证据。')
        sources = [leaves[key] for key in sorted(chunks) if leaves[key].suffix == '.parquet']
        if path.suffix == '.empty':
            _require(not sources and path.read_bytes() == b'no data\n', '代码空标记与日期分片冲突。')
        else:
            _require(bool(sources), '代码汇总存在未被日期分片覆盖的数据。')
            expected = pd.concat([pd.read_parquet(p) for p in sources], ignore_index=True).sort_values(['ts_code', 'trade_date'])
            _same_values(pd.read_parquet(path), expected)
            nonempty.append(path)
        complete.add(code)
        item = _copy_evidence(journal, path, audit / path.name, artifact)
        item['derived_audit_only'] = True
        evidence.append(item)
        if len(complete) % 250 == 0:
            progress(f'已核验 {len(complete)} 个代码汇总；保留审计，待空区间复核后本地重建。')
    if final.exists():
        _require(complete == universe and bool(nonempty) and final.is_file() and not final.is_symlink(),
                 '合并输出缺少完整代码和日期分片证据。')
        progress(f'正在逐批核验 {final.name} 与全部来源分片，未重新下载。')
        artifact = journal.artifact(final)
        # Bounded-memory reconstruction checks every column, not only row counts.
        with tempfile.TemporaryDirectory(prefix='index-verify-', dir=audit) as directory:
            expected_path = Path(directory) / final.name
            script.consolidate_parquet_parts(nonempty, expected_path)
            left = pq.ParquetFile(final).iter_batches(batch_size=16384, use_threads=False)
            right = pq.ParquetFile(expected_path).iter_batches(batch_size=16384, use_threads=False)
            for actual, expected in zip_longest(left, right):
                _require(actual is not None and expected is not None, '合并输出行数不一致。')
                _same_values(actual.to_pandas(), expected.to_pandas())
        item = _copy_evidence(journal, final, audit / final.name, artifact)
        item['derived_audit_only'] = True
        evidence.append(item)
    return evidence


def import_index_segments(journal, old, step, predecessor, target, producer, progress, frozen):
    """Import proven leaf coverage; archive old empties and reject unknown APIs."""
    import T01_get_data as script
    from .etl_migration import _parts_name

    _require(step.task_id in {'tushare.index_concept', 'tushare.index_futures'}
             and not old['frozen'].get('task_baseline'),
             '指数分片导入仅支持无基线的概念/期货行情历史任务。')
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
    futures = step.task_id == 'tushare.index_futures'
    apis = ('fut_index_daily',) if futures else ('ths_daily', 'dc_daily', 'tdx_daily')
    if futures:
        frozen_codes = {str(code) for code, source in zip(catalog['ts_code'], catalog['quote_source_api'])
                        if source == 'fut_index_daily'}
        _require(frozen_codes == {code for code, _ in script.FUTURES_INDEX_UNIVERSE},
                 '南华指数目录与当前执行代码范围不同，不能自动迁移。')
    names = {api: _parts_name(old['frozen']['task_sources'][step.source_id]['records'], step,
                              Path(script.INDEX_HISTORY_FILES[api]).stem) for api in apis}
    allowed = set(before['files']) | set(names.values()) | {script.INDEX_HISTORY_FILES[api] for api in apis}
    _require(all(p.name in allowed and not p.is_symlink() for p in work.iterdir()),
             '指数行情已有其他输出或未知分片，不能静默丢弃后迁移。')
    for filename, entry in before['files'].items():
        path = work / filename
        _require(path.is_file() and journal.artifact(path)['checksum'] == entry['checksum'],
                 '原工作区前置文件与冻结清单不一致。')
    args = script.parse_args(['--start-date', step.params['start_date'], '--end-date', step.params['end_date']])
    chunks = set(script.iter_date_chunks(args.start_date, args.end_date, args.history_chunk_days))
    result = {'copied': 0, 'empty_recheck': 0, 'confirmed_empty': 0, 'rows': 0, 'missing_segments': 0, 'files': []}
    for api in apis:
        universe = {str(code) for code, source in zip(catalog['ts_code'], catalog['quote_source_api']) if source == api}
        parts, final = work / names[api], work / script.INDEX_HISTORY_FILES[api]
        if not parts.exists():
            _require(not final.exists(), '已存在合并文件却缺少来源分片，不能自动迁移。')
            continue
        partial = _import_api(journal, step, target, frozen, progress, api, universe, chunks, parts, final)
        for key in result:
            result[key] += partial[key]
    return {**result, 'format': 'verified_futures_segments_v1' if futures else 'verified_concept_segments_v2',
            'pit_boundary': '保持原文件及采集审计；历史行情不包含历史供应商修订时点证明。'}


def _import_api(journal, step, target, frozen, progress, api, universe, chunks, parts, final):
    import T01_get_data as script
    from .etl_migration import _parts_name

    segments = parts / 'segments'
    _require(bool(universe) and parts.is_dir() and not parts.is_symlink()
             and segments.is_dir() and not segments.is_symlink(), '指数日期分片目录或标的目录无效。')
    dest = target / _parts_name(frozen['task_sources'][step.source_id]['records'], step,
                                Path(script.INDEX_HISTORY_FILES[api]).stem) / 'segments'
    dest.mkdir(parents=True)
    audit = target.parent / 'index_recovery_evidence' / api
    audit.mkdir(parents=True)
    evidence, counts, seen = [], {'copied': 0, 'empty_recheck': 0, 'confirmed_empty': 0, 'rows': 0}, set()
    segment_map = {}
    for path in sorted(segments.iterdir()):
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)__(\d{8})_(\d{8})\.(parquet|empty)', path.name)
        _require(path.is_file() and not path.is_symlink() and match is not None, '指数分片名称或路径不合法。')
        code, start, end, kind = match.groups()
        _require(code in universe and (start, end) in chunks and path.stem not in seen,
                 '指数分片不属于冻结目录/日期网格或存在冲突标记。')
        seen.add(path.stem)
        segment_map.setdefault(code, {})[(start, end)] = path
        artifact = journal.artifact(path)
        if kind == 'empty':
            proof = read_empty_evidence(path, api, code, start, end, parts.name)
            # A policy-name change requires requerying; never relabel evidence.
            reuse = proof is not None and dest.parent.name == parts.name
            item = _copy_evidence(journal, path, (dest if reuse else audit) / path.name, artifact)
            item['recheck_required'] = not reuse
            counts['confirmed_empty' if reuse else 'empty_recheck'] += 1
        else:
            rows = _validate_segment(path, code, datetime.strptime(start, '%Y%m%d').date(), datetime.strptime(end, '%Y%m%d').date(), api)
            item = _copy_evidence(journal, path, dest / path.name, artifact)
            item['rows'] = rows
            counts['copied'] += 1
            counts['rows'] += rows
        evidence.append(item)
        if len(evidence) % 250 == 0:
            progress(f'已核验 {len(evidence)} 个指数分片：复用 {counts["copied"]}，'
                     f'已复核空区间 {counts["confirmed_empty"]}，空结果待复核 {counts["empty_recheck"]}。')
    derived = audit / 'derived'
    derived.mkdir()
    evidence.extend(_derived_evidence(journal, parts, final, universe, chunks, segment_map, derived, progress))
    return {**counts, 'files': evidence, 'missing_segments': len(universe) * len(chunks) - len(seen)}
