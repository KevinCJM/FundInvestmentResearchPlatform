"""Bounded date partitions with immutable receipts; no numerical transforms."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from .batches import atomic_bytes
from .models import CenterError


def require(condition, message):
    if not condition:
        raise CenterError('MACRO_RANGE_INVALID', message)


class MacroRangeDownload:
    def __init__(self, directory, *, api, configuration, row_limit, max_requests=100_000):
        require(api in {'shibor', 'shibor_lpr', 'repo_daily'}, '未登记的宏观日期接口。')
        self.directory = Path(directory)
        require(not self.directory.is_symlink(), '宏观检查点目录不能是符号链接。')
        self.api, self.configuration, self.row_limit = api, configuration, row_limit
        self.max_requests, self.requests = max_requests, 0

    def before_request(self):
        self.requests += 1
        if self.requests > self.max_requests:
            raise CenterError('MACRO_REQUEST_BUDGET', '宏观分片达到请求预算，已完成检查点保留。')

    def validate(self, frame, start, end):
        if frame.empty:
            return
        date_column = 'trade_date' if self.api == 'repo_daily' else 'date'
        keys = ['ts_code', date_column] if self.api == 'repo_daily' else [date_column]
        require(set(keys) <= set(frame.columns) and not frame[keys].isna().any().any(), '宏观响应缺少日期或业务键。')
        dates = pd.to_datetime(frame[date_column].astype(str), format='%Y%m%d', errors='coerce')
        require(dates.notna().all() and dates.between(pd.Timestamp(start), pd.Timestamp(end)).all(),
                '宏观响应日期无效或超出请求区间，不能静默裁剪。')
        require(not frame.duplicated(keys).any(), '宏观响应业务键重复。')
        if self.api == 'repo_daily':
            require(frame.ts_code.astype(str).str.strip().ne('').all(), '回购代码不能为空。')

    def collect(self, start, end, *, fetch, cap_error, save, prepare, acknowledge=lambda *_: None):
        require(start <= end, '宏观分片日期范围无效。')
        identity = {'version': 1, 'api': self.api, 'configuration': self.configuration,
                    'row_limit': self.row_limit, 'start': start, 'end': end}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        receipt, path = (self.directory / (key + suffix) for suffix in ('.json', '.parquet'))
        require(not receipt.is_symlink() and not path.is_symlink(), '宏观检查点不能是符号链接。')
        if receipt.exists():
            try:
                meta = json.loads(receipt.read_text())
                stamp = datetime.fromisoformat(meta['collected_at'])
                require(meta['request'] == identity and stamp.tzinfo is not None
                        and stamp <= datetime.now(timezone.utc), '宏观检查点合同或采集时点无效。')
                require(meta['status'] in {'COMPLETE', 'SPLIT'}, '宏观检查点状态无效。')
                if meta['status'] == 'COMPLETE':
                    require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == meta['sha256'],
                            '宏观检查点缺失或校验和不一致。')
                    frame = pd.read_parquet(path)
                    self.validate(frame, start, end)
                    require(len(frame) == meta['rows'] and len(frame) < self.row_limit, '宏观检查点行数或截断合同不一致。')
                    require(meta['confirmations'] >= (2 if frame.empty else 1), '宏观空响应缺少独立复核。')
                    print(f'[INFO] {self.api} {start}—{end} 已校验复用分片（{len(frame)} 行）。', flush=True)
                    return [] if frame.empty else [prepare(frame, meta['collected_at'])]
                require(not path.exists(), '宏观拆分标记与数据文件冲突。')
            except (KeyError, ValueError, TypeError, OSError) as exc:
                raise CenterError('MACRO_CHECKPOINT_INVALID', '宏观检查点无法完整读取；保留文件并停止。') from exc
        else:
            # A crash between Parquet and receipt writes leaves evidence. Never
            # overwrite or silently treat this unacknowledged file as complete.
            require(not path.exists(), '宏观分片存在未登记文件，需要核验后恢复。')
            confirmations = 0
            try:
                frame = fetch(start, end)
                confirmations += 1
                if frame.empty:
                    print(f'[INFO] {self.api} {start}—{end} 空响应，独立复核。', flush=True)
                    frame = fetch(start, end)
                    confirmations += 1
                self.validate(frame, start, end)
                if len(frame) >= self.row_limit:
                    raise cap_error('configured row cap')
            except cap_error:
                if start == end:
                    raise CenterError('MACRO_DAY_ROW_CAP', f'{self.api} {start} 单日仍达到行数上限；拒绝保存不完整结果。') from None
                meta = {'request': identity, 'status': 'SPLIT', 'collected_at': datetime.now(timezone.utc).isoformat()}
                atomic_bytes(receipt, json.dumps(meta).encode())
            else:
                save(frame, path, quiet=True)
                meta = {'request': identity, 'status': 'COMPLETE', 'rows': len(frame),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                        'confirmations': confirmations, 'collected_at': datetime.now(timezone.utc).isoformat()}
                atomic_bytes(receipt, json.dumps(meta).encode())
                print(f'[INFO] {self.api} {start}—{end} 分片完成（{len(frame)} 行）。', flush=True)
                return [] if frame.empty else [prepare(frame, meta['collected_at'])]
        # A SPLIT receipt is not successful coverage. Both disjoint children
        # must complete before the capped parent can be acknowledged.
        left, right = datetime.strptime(start, '%Y%m%d'), datetime.strptime(end, '%Y%m%d')
        require(left < right, '宏观单日拆分标记无效。')
        middle = left + timedelta(days=(right - left).days // 2)
        print(f'[INFO] {self.api} {start}—{end} 触顶，拆分于 {middle:%Y%m%d}。', flush=True)
        options = dict(fetch=fetch, cap_error=cap_error, save=save, prepare=prepare, acknowledge=acknowledge)
        frames = self.collect(start, middle.strftime('%Y%m%d'), **options)
        frames += self.collect((middle + timedelta(days=1)).strftime('%Y%m%d'), end, **options)
        require(bool(frames), '宏观触顶区间拆分后全部为空，响应矛盾；拒绝标记完成。')
        acknowledge(start, end)
        return frames
