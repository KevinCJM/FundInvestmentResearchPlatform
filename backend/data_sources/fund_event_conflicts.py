"""Lossless quarantine at the holdings I/O boundary, never a value resolver."""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from backend.services.refresh_runtime import atomic_write_json, read_json_object
from .models import CenterError

KEYS = ['ts_code', 'ann_date', 'end_date', 'symbol']
VALUES = ['mkv', 'amount', 'stk_mkv_ratio', 'stk_float_ratio']
STATUS = 'source_conflict'


def segregate(frame):
    """Keep identity, replace ambiguous values with missing, retain all variants."""
    unique = frame.drop_duplicates().copy()
    ambiguous = unique.duplicated(KEYS, keep=False)
    evidence = unique.loc[ambiguous].copy()
    clean = unique.drop_duplicates(KEYS).copy()
    keys = pd.MultiIndex.from_frame(evidence[KEYS])
    mask = pd.MultiIndex.from_frame(clean[KEYS]).isin(keys)
    clean['_source_conflict'] = mask
    for column in VALUES:
        if column in clean:
            clean.loc[mask, column] = float('nan')
    return clean, evidence


def write_part(directory, date, code, evidence):
    from .fund_events import _checksum
    folder = directory / 'conflicts'
    folder.mkdir(exist_ok=True)
    path = folder / (date + '_' + (code or 'market') + '.parquet')
    temporary = path.with_suffix('.tmp')
    evidence.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return {'path': str(path.relative_to(directory)), 'sha256': _checksum(path),
            'rows': len(evidence), 'keys': len(evidence[KEYS].drop_duplicates()),
            'recorded_at': datetime.now(timezone.utc).isoformat(), 'source_api': 'fund_portfolio'}


def checked_part(directory, evidence):
    from .fund_events import _checksum
    relative = Path(evidence.get('path', ''))
    path = directory / relative
    if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2
            or relative.parts[0] != 'conflicts' or path.is_symlink()
            or path.parent.is_symlink() or not path.is_file()
            or _checksum(path) != evidence.get('sha256')):
        raise CenterError('FUND_EVENT_CHECKPOINT', '持仓冲突原始证据缺失或校验失败。')
    frame = pd.read_parquet(path)
    if (not set(KEYS) <= set(frame) or len(frame) != evidence.get('rows')
            or frame.empty or frame.duplicated().any()
            or len(frame[KEYS].drop_duplicates()) != evidence.get('keys')
            or not frame.duplicated(KEYS, keep=False).all()):
        raise CenterError('FUND_EVENT_CHECKPOINT', '持仓冲突证据的业务键或行数不符。')
    return frame


def finish(path, session):
    """Bind unresolved evidence to the private candidate; never certify it."""
    from .fund_events import _checksum
    report = path.with_name('fund_portfolio_conflicts.parquet')
    metadata = path.with_name(path.name + '.quality.meta.json')
    frames = [checked_part(session.directory, item) for item in session.conflicts.values()]
    # Never silently erase unresolved evidence inherited from a prior candidate.
    if metadata.exists():
        prior = read_json_object(metadata)
        if report.is_symlink() or not report.is_file() or _checksum(report) != prior.get('evidence_checksum'):
            raise CenterError('FUND_EVENT_CHECKPOINT', '已有持仓冲突清单校验失败。')
        frames.insert(0, pd.read_parquet(report))
    if not frames:
        return None
    evidence = pd.concat(frames, ignore_index=True).drop_duplicates()
    temporary = report.with_suffix('.tmp')
    evidence.to_parquet(temporary, index=False)
    os.replace(temporary, report)
    issue = {'version': 1, 'status': 'CONFLICTED', 'file': path.name,
             'data_checksum': _checksum(path), 'evidence_file': report.name,
             'evidence_checksum': _checksum(report), 'variant_rows': len(evidence),
             'conflicting_keys': len(evidence[KEYS].drop_duplicates()),
             'acquisition_complete': True, 'publishable': False,
             'message': '采集完成；供应商持仓数值冲突已隔离，数值留空，禁止作为标准研究数据或发布。'}
    atomic_write_json(metadata, issue)
    print(f'[WARN] 持仓采集完成，{issue["conflicting_keys"]} 个业务键存在数值冲突；'
          '全部原始值已隔离保存，未选择任意值、未发布。')
    return issue


def validate_placeholders(path, evidence):
    import pyarrow.parquet as pq
    keys = set(evidence[KEYS].itertuples(index=False, name=None))
    observed = set()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16384, columns=KEYS + VALUES + ['availability_status']):
        for row in batch.to_pylist():
            if row['availability_status'] != STATUS:
                continue
            def day(value):
                return value.strftime('%Y%m%d') if hasattr(value, 'strftime') else str(value)
            key = row['ts_code'], day(row['ann_date']), day(row['end_date']), row['symbol']
            if key not in keys or any(row[name] is not None for name in VALUES) or key in observed:
                raise CenterError('FUND_EVENT_CHECKPOINT', '冲突占位行与证据不一致，不能恢复。')
            observed.add(key)
    if observed != keys:
        raise CenterError('FUND_EVENT_CHECKPOINT', '冲突原始证据缺少对应空值占位行。')
