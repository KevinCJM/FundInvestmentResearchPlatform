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
    for name, entry in inventory['files'].items():
        clone_file(journal.checked_path(entry), directory / name)
    return inventory


def inventory(journal, directory: Path, path: Path, capabilities: list[str], source_id: str | None) -> dict:
    import pyarrow.parquet as pq
    files = {}
    for file in sorted([*directory.glob('*.parquet'), *directory.glob('*.meta.json')]):
        files[file.name] = {**journal.artifact(file), 'rows': pq.ParquetFile(file).metadata.num_rows if file.suffix == '.parquet' else None}
    if not any(name.endswith('.parquet') for name in files):
        raise CenterError('ETL_TASK_EMPTY', '任务未产生数据文件，不能标记为成功。')
    value = {'format': 'market_files_v1', 'files': files, 'capabilities': capabilities, 'source_id': source_id}
    return journal.write_json(path, value)
