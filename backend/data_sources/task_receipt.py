"""Atomic completion receipts tied to one private worker attempt."""
from __future__ import annotations

import json
from pathlib import Path

from .batches import atomic_bytes
from .etl_executor import read_small_json
from .models import CenterError
from .store import utc_now


def receipt_path(payload):
    if not payload.get('result_path'):
        return None
    path = Path(payload['result_path'])
    work = Path(payload['directory'])
    if path.is_symlink() or path.name != 'worker_result.json' or path.resolve().parent.parent != work.resolve().parent:
        raise CenterError('ETL_RECEIPT_PATH', '完成回执必须位于该工作进程的私有尝试目录。')
    if not isinstance(payload.get('worker_token'), str) or len(payload['worker_token']) != 32:
        raise CenterError('ETL_RECEIPT_IDENTITY', '缺少工作进程的尝试身份。')
    return path


def write_receipt(payload, result):
    path = receipt_path(payload)
    if path:
        value = {'protocol': 1, 'token': payload['worker_token'], 'task_id': payload['task_id'],
                 'finished_at': utc_now(), 'result': result}
        encoded = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False).encode()
        if len(encoded) > 131072:
            raise CenterError('ETL_RECEIPT_SIZE', '工作进程回执超过允许大小，未确认成功。')
        atomic_bytes(path, encoded)


def read_receipt(payload):
    path = receipt_path(payload)
    value = read_small_json(path, 131072) if path else None
    if not value or value.get('protocol') != 1 or value.get('token') != payload['worker_token'] or value.get('task_id') != payload['task_id']:
        raise CenterError('ETL_RECEIPT_INVALID', '完成回执缺失或不属于当前尝试，不能将节点标记为成功。')
    result = value.get('result')
    if not isinstance(result, dict) or type(result.get('ok')) is not bool:
        raise CenterError('ETL_RECEIPT_INVALID', '完成回执格式错误，检查点保留。')
    if result['ok'] and not isinstance(result.get('result'), dict):
        raise CenterError('ETL_RECEIPT_INVALID', '完成回执缺少结果，检查点保留。')
    return result
