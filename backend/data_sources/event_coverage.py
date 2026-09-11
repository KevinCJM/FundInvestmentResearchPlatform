"""Acquisition coverage, never a publication/PIT certificate (metadata only).

One bounded sidecar travels with its exact Parquet artifact. Missing coverage
means unknown, not empty. Query dates and observation dates are independent.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .acquisition import fingerprint
from .models import CenterError
from backend.services.refresh_runtime import atomic_write_json, read_json_object

EVENT_FILES = {'fund_portfolio': 'fund_portfolio_df.parquet', 'fund_dividend': 'fund_dividend_df.parquet'}


def sidecar(path: Path) -> Path:
    return path.with_name(path.name + '.coverage.meta.json')


def load(path: Path, source_hash: str | None = None, *, verify=False) -> dict | None:
    file = sidecar(path)
    if not file.exists() and not file.is_symlink():
        return None
    if file.is_symlink() or file.stat().st_size > 8 * 1024 * 1024:
        raise CenterError('EVENT_COVERAGE_INVALID', '查询覆盖清单路径或大小异常，不能作为跳过请求的依据。')
    value = read_json_object(file)
    if (value.get('version') != 1 or value.get('file') != path.name
            or value.get('date_axis') != 'ann_date' or value.get('complete') is not True
            or not isinstance(value.get('days'), dict)
            or not isinstance(value.get('universe'), list)
            or len(value.get('data_checksum', '')) != 64):
        raise CenterError('EVENT_COVERAGE_INVALID', '查询覆盖清单不完整，不能推进下载水位。')
    for day, checked in value['days'].items():
        try:
            if date.fromisoformat(day).isoformat() != day:
                raise ValueError()
            datetime.fromisoformat(checked)
        except (ValueError, TypeError):
            raise CenterError('EVENT_COVERAGE_INVALID', '查询覆盖日期无效。') from None
    if verify:
        from .fund_events import _checksum
        if path.is_symlink() or _checksum(path) != value['data_checksum']:
            raise CenterError('EVENT_COVERAGE_INVALID', '覆盖清单与实际数据校验和不一致，禁止跳过查询。')
    if source_hash is not None and value.get('source_hash') != source_hash:
        return None  # A changed source/field contract needs fresh observations.
    return value


def plan_dates(first: date, latest: date, end: date, coverage: dict | None,
               *, revision_interval=7, revision_window=90, purpose='update') -> dict:
    """Bounded calendar scheduling; no numerical financial computation."""
    known = (coverage or {}).get('days', {})
    # No proof: requery the last observation, but do not repeat a 5-SSE-day
    # overlap on a quarterly event source. Historical gaps remain explicit.
    begin = min(date.fromisoformat(min(known)), latest) if known else latest
    if purpose == 'recheck':
        begin = max(first, end - timedelta(days=revision_window - 1))
    else:
        begin = max(begin, end - timedelta(days=366))
    fresh, missing, revision = [], [], []
    for offset in range(max(0, (end - begin).days + 1)):
        day = begin + timedelta(days=offset)
        checked = known.get(day.isoformat())
        if checked is None:
            missing.append(day.strftime('%Y%m%d'))
        elif (purpose == 'recheck' or
              (day >= end - timedelta(days=revision_window - 1)
               and (end + timedelta(days=1) - datetime.fromisoformat(checked).date()).days >= revision_interval)):
            revision.append(day.strftime('%Y%m%d'))
        else:
            fresh.append(day.strftime('%Y%m%d'))
    query = sorted(set(missing + revision))
    return {'query_dates': query, 'new_query_days': len(missing), 'revision_query_days': len(revision),
            'reused_query_days': len(fresh), 'coverage_through': max(known) if known else None,
            'last_checked_at': max(known.values()) if known else None,
            'coverage_known': bool(coverage), 'revision_interval_days': revision_interval,
            'revision_window_days': revision_window, 'purpose': purpose}


def commit(path: Path, session, source_hash: str, previous: dict | None) -> dict:
    """Only after every requested unit AND final merge succeed, bind coverage.

    All-zero results still advance query coverage; they never create fake rows.
    A changed universe invalidates prior coverage, so new IDs cannot be skipped.
    """
    from .fund_events import _checksum
    codes = sorted(session.inceptions)
    days = dict(previous['days']) if previous and previous['universe'] == codes else {}
    now = datetime.now(timezone.utc).isoformat()
    for day in session.dates:
        days[datetime.strptime(day, '%Y%m%d').date().isoformat()] = now
    # Keep recent coverage bounded; older gaps require explicit historical repair.
    kept = sorted(days)[-1461:]
    value = {'version': 1, 'file': path.name, 'date_axis': 'ann_date', 'source_hash': source_hash,
             'data_checksum': _checksum(path), 'complete': True, 'universe': codes,
             'days': {key: days[key] for key in kept}, 'published': False,
             'query_contract': fingerprint([session.api_name, session.fields, 'ann_date', codes])}
    atomic_write_json(sidecar(path), value)
    return value
