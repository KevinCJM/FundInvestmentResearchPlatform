"""Small, versioned evidence inside index empty markers (no extra shard files)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from .models import CenterError
from backend.services.refresh_runtime import atomic_write_json


def _validate(value, api, code, start, end, checkpoint):
    try:
        expected = {'format': 'index_empty_v1', 'api': api, 'code': code,
                    'start': start, 'end': end, 'checkpoint': checkpoint}
        if not isinstance(value, dict) or set(value) != {*expected, 'confirmations'}:
            raise ValueError
        if any(value[key] != item for key, item in expected.items()):
            raise ValueError
        confirmations = value['confirmations']
        if not isinstance(confirmations, list) or len(confirmations) != 2:
            raise ValueError
        ids, previous = set(), None
        for request in confirmations:
            if set(request) != {'id', 'started_at', 'finished_at', 'rows'}:
                raise ValueError
            if type(request['rows']) is not int or request['rows'] != 0:
                raise ValueError
            identifier = request['id']
            if not re.fullmatch(r'[a-f0-9]{32}', identifier) or identifier in ids:
                raise ValueError
            ids.add(identifier)
            first, last = (datetime.fromisoformat(request[key]) for key in ('started_at', 'finished_at'))
            if (first.tzinfo is None or last.tzinfo is None or first > last
                    or last > datetime.now(timezone.utc) or previous is not None and first < previous):
                raise ValueError
            previous = last
    except (ValueError, TypeError, KeyError, OverflowError):
        raise CenterError('INDEX_EMPTY_EVIDENCE_INVALID', '指数空区间的两次独立请求凭据或来源合同无效。') from None
    return value


def write_empty_evidence(path, api, code, start, end, checkpoint, confirmations):
    value = {'format': 'index_empty_v1', 'api': api, 'code': code,
             'start': start, 'end': end, 'checkpoint': checkpoint, 'confirmations': confirmations}
    atomic_write_json(path, _validate(value, api, code, start, end, checkpoint))


def read_empty_evidence(path, api, code, start, end, checkpoint):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 8192:
        raise CenterError('INDEX_EMPTY_EVIDENCE_INVALID', '指数空区间凭据路径或大小无效。')
    raw = path.read_bytes()
    if raw == b'no data\n':
        return None  # Compatibility only; cross-version import must recheck.
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        raise CenterError('INDEX_EMPTY_EVIDENCE_INVALID', '指数空区间凭据无法解析。') from None
    return _validate(value, api, code, start, end, checkpoint)
