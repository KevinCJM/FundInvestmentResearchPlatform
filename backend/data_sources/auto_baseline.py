"""Explicit reuse of completed candidate data, without activating a snapshot."""
from __future__ import annotations

import json
from pathlib import Path

from .etl_store import EtlStore
from .models import CenterError


def candidate_path(root: Path, entry: dict) -> Path:
    relative = Path(entry['path'])
    path = root / relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or root.resolve() not in path.resolve().parents or not path.is_file()):
        raise CenterError('AUTO_CANDIDATE_INVALID', '候选文件缺失或路径不安全，不能复用。', 422)
    if not isinstance(entry.get('checksum'), str) or len(entry['checksum']) != 64:
        raise CenterError('AUTO_CANDIDATE_INVALID', '候选文件没有有效的校验和。', 422)
    return path


def candidate_inventory(store, run: dict) -> dict:
    """Only completed, immutable, single-source cumulative workspaces qualify.

    Preview verifies the small inventory, not GB data. Selected file checksums
    are verified during freeze, before any supplier request.
    """
    if run.get('status') != 'SUCCEEDED' or run.get('cancel_requested'):
        raise CenterError('AUTO_CANDIDATE_INVALID', '只能选择已成功完成的下载作为补充基线。', 422)
    if any(s.get('output', {}).get('data_quality', {}).get('status') == 'CONFLICTED' for s in run.get('steps', [])):
        raise CenterError('AUTO_CANDIDATE_INVALID', '该次采集存在未解决的持仓数值冲突，不能作为补充基线。', 422)
    records = run.get('frozen', {}).get('task_sources', {}).get('tushare', {}).get('records', [])
    configs = {r['config']['id']: r['config'] for r in records}
    if configs.get('tushare', {}).get('transport') != 'tushare':
        raise CenterError('AUTO_CANDIDATE_INVALID', '候选来源不是已记录的 Tushare 数据。', 422)
    step = next((s for s in reversed(run.get('steps', [])) if s.get('output', {}).get('workspace')), None)
    if not step or step.get('status') != 'SUCCEEDED':
        raise CenterError('AUTO_CANDIDATE_INVALID', '下载记录没有完整的候选清单。', 422)
    artifact = step['output']['workspace']
    candidate_path(store.root, artifact)
    value = json.loads(EtlStore(store).checked_path(artifact).read_text())
    if value.get('format') != 'market_files_v1' or value.get('source_id') != 'tushare':
        raise CenterError('AUTO_CANDIDATE_INVALID', '候选数据格式或来源不兼容。', 422)
    return {'files': value['files'], 'inventory': artifact, 'configs': configs}


def supplemental_files(store, run: dict, wanted: dict[str, tuple[str, list[str]]], end) -> dict:
    from .auto_incremental import date_bounds, file_identity
    value = candidate_inventory(store, run)
    selected = {}
    for name, (column, apis) in wanted.items():
        entry = value['files'].get(name)
        if not entry or not entry.get('rows'):
            continue
        if any(value['configs'].get('tushare.' + api, {}).get('api_name') != api for api in apis):
            continue
        # Transport identity alone cannot justify mixing a changed mapping.
        for api in apis:
            current = store.get('interface', 'tushare.' + api)['config']
            original = value['configs']['tushare.' + api]
            if any(current.get(k) != original.get(k) for k in ('api_name', 'source_fields', 'mappings')):
                raise CenterError('AUTO_CANDIDATE_SCHEMA', '候选数据的接口字段或映射已变更，不能直接补入基线。', 422)
        path = candidate_path(store.root, entry)
        first, latest = date_bounds(path, column)
        if latest > end:
            continue
        selected[name] = {**entry, **file_identity(path), 'latest_date': latest.isoformat(),
                          'first_date': first.isoformat(), 'run_id': run['run_id']}
    return selected


def candidate_choices(store, wanted, end) -> list[dict]:
    if not wanted:
        return []
    # Query only successful records, and keep planning bounded. No data-dir scan.
    with store.connection() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='etl_run'").fetchone():
            return []
        rows = db.execute("SELECT body FROM etl_run WHERE json_extract(body,'$.status')='SUCCEEDED' ORDER BY updated_at DESC LIMIT 20").fetchall()
    choices = []
    for row in rows:
        try:
            run = json.loads(row[0])
            files = supplemental_files(store, run, wanted, end)
            if files:
                choices.append({'run_id': run['run_id'], 'name': run['name'],
                                'finished_at': run.get('finished_at'),
                                'files': [{'name': name, 'rows': e['rows'], 'latest_date': e['latest_date']} for name, e in files.items()]})
        except (CenterError, OSError, ValueError, KeyError, TypeError):
            continue  # Invalid candidates never become an implicit fallback.
    return choices


def exclusion_proposal(definition, blocked: set[str]) -> dict | None:
    """Return an editable proposal; never silently relax an existing DAG."""
    from .etl_dependencies import plan_dependencies
    from .etl_models import EtlDefinition
    if not blocked:
        return None
    # Old serial templates represented ordering as workspace dependencies.
    # Rebuilding those links requires explicit acceptance of this proposal.
    rebuilt = definition.graph_version != 1
    try:
        graph = plan_dependencies(definition) if rebuilt else definition
    except CenterError:
        return None  # A proposal must not conceal the original validation errors.
    removed = set(blocked)
    for step in graph.execution_steps():
        if set(step.inputs) & removed:
            removed.add(step.id)
    kept = [s.model_copy(update={'after': [i for i in s.after if i not in removed]})
            for s in graph.steps if s.id not in removed]
    if not kept:
        return None
    value = graph.model_dump(mode='json')
    value.update(steps=[s.model_dump(mode='json') for s in kept], canvas=None)
    result = EtlDefinition.model_validate(value)
    return {'definition': result.model_dump(mode='json'), 'rebuild_dependencies': rebuilt,
            'excluded': [{'id': s.id, 'name': s.name, 'reason': 'blocked' if s.id in blocked else 'dependency'}
                         for s in graph.steps if s.id in removed]}
