"""Shared fail-closed validation of every unfinished recovery workspace."""
import re

from . import etl_service as etl
from .etl_executor import read_small_json
from .models import CenterError


def _check_unhandled_partial(journal, old, *, adjustment_scope=False):
    for state in old['steps']:
        task = old['frozen']['tasks'].get(state['id'], {})
        if adjustment_scope and task.get('spec', {}).get('id') == 'tushare.fund_adjustment':
            continue  # The explicit scope importer validates all excluded evidence.
        if state['status'] != 'SUCCEEDED':
            _check_partial_step(journal, old, state)


def _check_partial_step(journal, old, state):
    """Never let the CLI's generic zero-shard result discard unknown partial work."""
    definition = next(s for s in old['definition']['steps'] if s['id'] == state['id'])
    if definition['task_id'] in {'tushare.index_concept', 'tushare.index_futures', 'tushare.index_constituents', 'tushare.macro_cycle', 'tushare.macro_rates'}:
        return  # The selected importer validates the whole work directory.
    directory = journal.root / 'etl_runs' / old['run_id'] / state['id']
    marker, work = directory / 'work_input.json', directory / 'work'
    if not marker.exists() and not work.exists() and not marker.is_symlink() and not work.is_symlink():
        return
    from backend.data_sources.task_workspace import read_inventory
    if old.get('options', {}).get('mode') == 'auto_incremental':
        from backend.data_sources.task_runtime import task_input
        task = next(s for s in etl.parse_definition(old['definition']).steps if s.id == state['id'])
        if not (directory / 'inputs.json').is_file():
            raise CenterError('ETL_MIGRATION_BLOCKED', '原自动增量缺少冻结输入清单。', 409)
        predecessor = task_input(journal, old, task, {s['id']: s for s in old['steps']}, directory / '1')
    else:
        predecessor = next(s for s in old['steps'] if s['id'] == definition['inputs'][0])['output']['workspace']
    expected = {'input': predecessor, 'task': next(s for s in etl.parse_definition(old['definition']).steps if s.id == state['id']).model_dump(mode='json'),
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
            if item.name == 'verified_day_imports.json':
                from .resolution_store import _checksum
                import pyarrow.parquet as pq
                record = read_small_json(item, 1048576)
                require(isinstance(record, dict) and record.get('version') == 1
                        and isinstance(record.get('producer'), str) and isinstance(record.get('days'), dict))
                for day, entry in record['days'].items():
                    require(bool(re.fullmatch(r'\d{8}', day)) and isinstance(entry, dict))
                    part = parts / (day + '.parquet')
                    require(part.is_file() and not part.is_symlink() and _checksum(part) == entry.get('sha256')
                            and pq.ParquetFile(part).metadata.num_rows == entry.get('rows'))
                continue
            require(bool(re.fullmatch(r'\d{8}\.(parquet|empty)', item.name)))
            continue
        require(item.is_dir() and bool(re.fullmatch(r'events_v4_[a-f0-9]{20}', item.name)))
        for evidence in item.iterdir():
            if evidence.name == 'conflicts':
                from .fund_event_conflicts import checked_part
                require(evidence.is_dir() and not evidence.is_symlink())
                for raw in evidence.iterdir():
                    require(raw.is_file() and not raw.is_symlink() and raw.suffix == '.parquet')
                    receipt = read_small_json(item / (raw.stem + '.json'), 1048576)
                    require(bool(receipt and receipt.get('status') == 'COMPLETE'
                                 and receipt.get('quality_status') == 'CONFLICTED'
                                 and receipt.get('conflict_evidence', {}).get('path') == 'conflicts/' + raw.name))
                    checked_part(item, receipt['conflict_evidence'])
                continue
            if evidence.name == 'requery_evidence':
                require(evidence.is_dir() and not evidence.is_symlink())
                for audit in evidence.iterdir():
                    require(audit.is_file() and not audit.is_symlink()
                            and bool(re.fullmatch(r'\d{8}(?:-\d{8})?_[A-Za-z0-9._-]+\.json', audit.name)))
                    record = read_small_json(audit, 1048576)
                    require(isinstance(record, dict) and record.get('status') == 'SPLIT'
                            and audit.stem == f"{record.get('date')}_{record.get('code') or 'market'}")
                continue  # Immutable prior audit stays in the original run, never completion evidence.
            require(evidence.is_file() and not evidence.is_symlink() and evidence.suffix in {'.json', '.parquet'})
            if evidence.suffix == '.parquet':
                receipt = read_small_json(evidence.with_suffix('.json'), 1048576)
                require(bool(receipt and receipt.get('status') == 'COMPLETE'))
