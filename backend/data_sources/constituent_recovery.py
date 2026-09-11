"""Preserve completed member queries across a verified execution-only upgrade."""
import hashlib
import json
import re
from datetime import datetime

import pandas as pd

from .models import CenterError
from .task_workspace import clone_file, read_inventory


def require(value, message):
    if not value:
        raise CenterError('ETL_MIGRATION_BLOCKED', message, 409)


def import_constituent_queries(journal, old, step, predecessor, target, producer, progress, frozen):
    import T01_get_data as script
    from .etl_migration import _parts_name

    work = journal.root / 'etl_runs' / old['run_id'] / step.id / 'work'
    marker = work.parent / 'work_input.json'
    if not work.exists() and not marker.exists():
        return {'copied': 0, 'empty_recheck': 0}
    require(work.is_dir() and not work.is_symlink() and marker.is_file() and not marker.is_symlink(), '成分检查点工作区路径无效。')
    require(json.loads(marker.read_text()) == {'input': predecessor, 'task': step.model_dump(mode='json'), 'execution': producer}, '成分检查点输入合同不一致。')
    before = read_inventory(journal, predecessor)
    source_records = old['frozen']['task_sources'][step.source_id]['records']
    target_records = frozen['task_sources'][step.source_id]['records']
    name = _parts_name(source_records, step, 'index_members_queries')
    weights_name = _parts_name(source_records, step, 'index_weights_df')
    stage_name = '.tushare_stage_index_constituents_members.json'
    require({p.name for p in work.iterdir()} <= set(before['files']) | {name, weights_name, 'index_members_df.parquet', stage_name},
            '成分节点含未知文件，不能静默丢弃。')
    for filename, entry in before['files'].items():
        path = work / filename
        require(path.is_file() and not path.is_symlink() and journal.artifact(path)['checksum'] == entry['checksum'], '成分前置文件与冻结清单不一致。')
    parts = work / name
    if not parts.exists():
        require(not any((work / item).exists() for item in (weights_name, 'index_members_df.parquet', stage_name)),
                '已完成成分或权重缺少原始成分查询回执，不能跳过校验。')
        return {'copied': 0, 'empty_recheck': 0}
    require(parts.is_dir() and not parts.is_symlink(), '成分查询目录无效。')
    catalog = pd.read_parquet(work / 'index_catalog_df.parquet')
    calendar = pd.read_parquet(work / 'trade_day_df.parquet')
    days = pd.to_datetime(calendar.loc[calendar.exchange.eq('SSE') & calendar.is_open.eq(1), 'cal_date'])
    days = days[days.le(pd.Timestamp(step.params['end_date']))]
    require(not days.empty, '成分检查点缺少冻结交易日。')
    day = days.max().strftime('%Y%m%d')
    expected = {}
    requests = [(api, {}) for api in ('index_member_all', 'ci_index_member')]
    for api, source in [('ths_member', 'ths_index'), ('dc_member', 'dc_index'), ('tdx_member', 'tdx_index')]:
        requests.extend((api, {'ts_code': str(code), **({'trade_date': day} if api != 'ths_member' else {})})
                        for code in catalog.loc[catalog.source_api.eq(source), 'ts_code'].dropna().unique())
    for api, params in requests:
        identity = {'version': 1, 'api': api, 'params': params}
        expected[hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()] = identity
    files = list(parts.iterdir())
    require(all(p.is_file() and not p.is_symlink() and re.fullmatch(r'[a-f0-9]{64}\.(json|parquet)', p.name) for p in files), '成分检查点含未知文件。')
    receipts = {p.stem for p in files if p.suffix == '.json'}
    require(receipts == {p.stem for p in files if p.suffix == '.parquet'} and receipts <= set(expected), '成分查询回执不成对或超出冻结范围。')
    dest = target / _parts_name(target_records, step, 'index_members_queries')
    dest.mkdir()
    result = {'copied': 0, 'empty_recheck': 0, 'confirmed_empty': 0, 'files': []}
    member_frames = []
    for key in sorted(receipts):
        receipt, path = parts / (key + '.json'), parts / (key + '.parquet')
        meta = json.loads(receipt.read_text())
        require(meta.get('request') == expected[key], '成分查询参数与文件名不一致。')
        require(isinstance(meta.get('collected_at'), str) and datetime.fromisoformat(meta['collected_at']).tzinfo is not None, '成分回执缺少采集时点。')
        artifact = journal.artifact(path)
        require(artifact['checksum'] == meta.get('sha256'), '成分文件校验和不一致。')
        frame = pd.read_parquet(path)  # Decode all columns; a footer is not enough.
        require(len(frame) == meta.get('rows'), '成分文件行数不一致。')
        script.validate_constituent_frame(frame, expected[key]['api'], expected[key]['params'], set())
        if not frame.empty:
            member_frames.append(script._normalise_member_frame(frame, expected[key]['api'], expected[key]['params'].get('ts_code')))
        for original in (receipt, path):
            source = journal.artifact(original)
            clone_file(original, dest / original.name)
            imported = journal.artifact(dest / original.name)
            require(imported['checksum'] == source['checksum'], '复制后成分检查点校验和不一致。')
            result['files'].append({'source': source, 'imported': imported})
        result['copied'] += 1
        result['confirmed_empty'] += int(frame.empty)
        if result['copied'] % 250 == 0:
            progress(f"已核验 {result['copied']} 个完整成分查询；保留原采集时点，没有重新请求供应商。")
    members, stage = work / 'index_members_df.parquet', work / stage_name
    if members.exists() or stage.exists() or (work / weights_name).exists():
        require(receipts == set(expected) and members.is_file() and not members.is_symlink()
                and stage.is_file() and not stage.is_symlink(), '成分完成阶段缺少完整查询回执、文件或标记。')
        require(bool(member_frames), '成分阶段标记完成但没有非空原始查询。')
        rebuilt = pd.concat(member_frames, ignore_index=True).drop_duplicates(
            ['source_api', 'index_code', 'con_code', 'in_date', 'out_date', 'trade_date'], keep='last')
        _same_frame(pd.read_parquet(members), rebuilt)
        marker_value = json.loads(stage.read_text())
        require(marker_value.get('stage') == 'index_constituents_members'
                and marker_value.get('schema_version') == 1, '成分阶段标记合同无效。')
        # Derived final file is re-built locally from the imported verified raw
        # queries. Do not trust or copy a bare success flag to the new version.
        result['rebuilt_members'] = {'source': journal.artifact(members), 'rows': len(rebuilt)}
        if (work / weights_name).exists():
            from .weight_recovery import import_weights
            result['weights'] = import_weights(journal, work / weights_name,
                target / _parts_name(target_records, step, 'index_weights_df'), step, catalog, progress)
            result['files'].extend(result['weights'].pop('files'))
    return {**result, 'format': 'verified_constituent_queries_v1'}


def _same_frame(actual, expected):
    require(set(actual.columns) == set(expected.columns), '派生成分/权重文件字段不一致。')
    columns = sorted(actual.columns)
    try:
        pd.testing.assert_frame_equal(actual[columns].sort_values(columns).reset_index(drop=True),
                                      expected[columns].sort_values(columns).reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError:
        require(False, '派生成分/权重文件与已校验原始查询不一致。')
