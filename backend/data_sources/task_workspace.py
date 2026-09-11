"""Private immutable workspace inventories and copy-on-write materialization."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
from pathlib import Path

from .models import CenterError


def clone_file(source: Path, target: Path) -> None:
    """Never hard-link writable workspaces to an immutable predecessor."""
    if source.is_symlink() or target.exists():
        raise CenterError('ETL_WORKSPACE_PATH', '工作区路径无效，不能覆盖输入文件。')
    if sys.platform == 'darwin':
        library = ctypes.CDLL('/usr/lib/libSystem.B.dylib', use_errno=True)
        if library.clonefile(os.fsencode(source), os.fsencode(target), 0) == 0:
            return
    shutil.copy2(source, target)


def read_inventory(journal, artifact: dict) -> dict:
    value = json.loads(journal.checked_path(artifact).read_text())
    if value.get('format') != 'market_files_v1':
        raise CenterError('ETL_WORKSPACE_FORMAT', '前置结果不是已校验的数据集工作区。')
    for name, entry in value['files'].items():
        if Path(name).name != name or not name.endswith(('.parquet', '.meta.json')):
            raise CenterError('ETL_WORKSPACE_PATH', '工作区文件名无效。')
        journal.checked_path(entry)
    return value


def materialize(journal, artifact: dict | None, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    inventory = read_inventory(journal, artifact) if artifact else {'files': {}, 'capabilities': []}
    if inventory.get('quality_issues'):
        raise CenterError('ETL_SOURCE_QUALITY_BLOCKED', '输入工作区含未解决的供应商数值冲突，不能作为计算或下载基线；原始证据保留。')
    for name, entry in inventory['files'].items():
        clone_file(journal.checked_path(entry), directory / name)
    return inventory


def merge_inventories(journal, artifacts: list[dict]) -> dict:
    """Join explicit inputs; never pick an arbitrary winner by input order."""
    result = {'format': 'market_files_v1', 'files': {}, 'capabilities': [], 'source_id': None}
    for artifact in artifacts:
        value = read_inventory(journal, artifact)
        if value.get('quality_issues'):
            result.setdefault('quality_issues', []).extend(value['quality_issues'])
        source = value.get('source_id')
        if source and result['source_id'] and source != result['source_id']:
            raise CenterError('ETL_TASK_SOURCE_MIX', '不能合并不同来源的数据工作区。')
        result['source_id'] = source or result['source_id']
        result['capabilities'] = sorted(set(result['capabilities']) | set(value['capabilities']))
        for name, entry in value['files'].items():
            prior = result['files'].get(name)
            if prior and prior['checksum'] == entry['checksum']:
                entry = {**entry, 'supersedes': sorted(set(prior.get('supersedes', [])) | set(entry.get('supersedes', [])))}
            if prior and prior['checksum'] != entry['checksum']:
                if prior['checksum'] in entry.get('supersedes', []):
                    pass  # Verified descendant of this exact file version.
                elif entry['checksum'] in prior.get('supersedes', []):
                    continue
                else:
                    raise CenterError('ETL_WORKSPACE_CONFLICT', f'{name} 存在不同版本且无替代关系，请明确唯一数据来源。')
            result['files'][name] = entry
    return result


def inventory(journal, directory: Path, path: Path, capabilities: list[str], source_id: str | None, *, before: dict | None = None) -> dict:
    import pyarrow.parquet as pq
    files = {}
    for file in sorted([*directory.glob('*.parquet'), *directory.glob('*.meta.json')]):
        entry = {**journal.artifact(file), 'rows': pq.ParquetFile(file).metadata.num_rows if file.suffix == '.parquet' else None}
        prior = (before or {}).get('files', {}).get(file.name)
        if prior:
            entry['supersedes'] = sorted(set(prior.get('supersedes', [])) | ({prior['checksum']} if prior['checksum'] != entry['checksum'] else set()))
        files[file.name] = entry
    if not any(name.endswith('.parquet') for name in files):
        raise CenterError('ETL_TASK_EMPTY', '任务未产生数据文件，不能标记为成功。')
    value = {'format': 'market_files_v1', 'files': files, 'capabilities': capabilities, 'source_id': source_id}
    issues = []
    for name in files:
        if not name.endswith('.quality.meta.json'):
            continue
        issue = json.loads((directory / name).read_text())
        if (issue.get('version') != 1 or issue.get('status') != 'CONFLICTED'
                or issue.get('publishable') is not False
                or files.get(issue.get('file'), {}).get('checksum') != issue.get('data_checksum')
                or files.get(issue.get('evidence_file'), {}).get('checksum') != issue.get('evidence_checksum')):
            raise CenterError('ETL_QUALITY_EVIDENCE', '数值冲突清单与数据或原始证据不匹配。')
        issues.append(issue)
    if issues:
        value['quality_issues'] = issues
    return journal.write_json(path, value)
