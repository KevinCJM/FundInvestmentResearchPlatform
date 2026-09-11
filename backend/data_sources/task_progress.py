"""Bounded, redacted task telemetry; never part of the download data contract."""
from __future__ import annotations

import io
import json
import re
import threading
import time
from pathlib import Path

from .batches import atomic_bytes
from .store import utc_now


def redact(text: str, secret: str = '') -> str:
    if secret:
        text = text.replace(secret, '[REDACTED]')
    text = re.sub(r'(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+', r'\1[REDACTED]', text)
    return re.sub(r'''(?i)(["']?(?:token|api_key|password|secret)["']?\s*[:=]\s*["']?)[^\s,"'&;]+''', r'\1[REDACTED]', text)


class TaskProgressLog(io.TextIOBase):
    def __init__(self, path: Path | None = None, secret: str = ''):
        self.path, self.secret = path, secret
        self.tail, self.warnings = '', 0
        self._lock = threading.RLock()
        self._pending: dict[int, str | None] = {}
        self._last_write = float('-inf')
        self.state = {'sequence': 0, 'phase': '准备数据', 'message': '正在初始化下载工作进程。',
                      'completed': None, 'total': None, 'unit': '项', 'batches': 0, 'received_rows': 0,
                      'activity_at': utc_now(), 'logs': []}

    def _line(self, text):
        text = redact(text, self.secret).strip()
        if not text:
            return
        text = text[:600]  # 20 UTF-8 log entries remain below the 64 KiB reader cap.
        self.tail = (self.tail + text + '\n')[-8000:]
        self.warnings += int('[WARN]' in text)
        now = utc_now()
        self.state.update(activity_at=now, message=text, sequence=self.state['sequence'] + 1)
        self.state['logs'] = (self.state['logs'] + [{'at': now, 'message': text}])[-20:]
        coverage = re.fullmatch(r'\[COVERAGE\] 日期 (\d+)；复用 (\d+)；复核 (\d+)；检查点 (\d+)；请求 (\d+)。', text)
        if coverage:
            self.state['event_coverage'] = dict(zip(('queried_days', 'reused_days', 'revision_days', 'checkpoints', 'requests'), map(int, coverage.groups())))
            self.state['message'] = '公告区间已校验并记录查询覆盖；返回行数不等于新增行数。'
        page = re.search(r'(\S+) 公告日 (\d{8}) 分页 (\d+)/(\d+)', text)
        if page:
            self.state['page_progress'] = {'scope': page[1], 'date': page[2], 'page': int(page[3]), 'limit': int(page[4])}
        fund = re.search(r'(\d{8}) 基金补抓 (\d+)/(\d+)', text)
        if fund:
            self.state['fund_progress'] = {'date': fund[1], 'completed': int(fund[2]), 'total': int(fund[3])}
        if text.startswith(('[STAGE]', '[DONE]', '[OK]')):
            phase = ('分页一致性复核' if '跨页重复稳定性复核' in text else
                     '合并与校验' if '合并' in text else '本地处理' if text.startswith('[OK]') else '执行阶段')
            self.state.update(phase=phase, completed=None, total=None)
        verification = re.search(r'公告日 (\d{8}) 页面复核 (\d+)/(\d+)', text)
        if verification and 0 < int(verification[2]) <= int(verification[3]):
            self.state.update(phase='分页一致性复核', completed=int(verification[2]),
                              total=int(verification[3]), unit='页')
            self.state.pop('page_progress', None)
        # Only explicit progress markers; never interpret dates, pages or retry
        # attempt fractions as a node's completion percentage.
        match = re.search(r'(?:进度|已处理历史行)\s*(\d+)/(\d+)', text)
        if match and 0 <= int(match[1]) <= int(match[2]) and int(match[2]) > 0:
            merging = '历史行' in text or '归并' in text or '合并' in text
            self.state.update(phase='合并历史数据' if merging else '下载分片',
                              completed=int(match[1]), total=int(match[2]), unit='行' if merging else '项')

    def write(self, value):
        with self._lock:
            thread = threading.get_ident()
            pending = self._pending.get(thread, '')
            for part in value.splitlines(keepends=True):
                if pending is not None:
                    pending += part
                    if len(pending) > 16384:
                        pending = None  # Do not publish a truncated secret.
                if part.endswith(('\n', '\r')):
                    self._line(pending if pending is not None else '[INFO] 超长日志已省略。')
                    pending = ''
            self._pending[thread] = pending
            self._publish()
        return len(value)

    def batch(self, rows):
        with self._lock:
            captured_at = utc_now()
            window = self.state.setdefault('collection_window', {'first_at': captured_at})
            window['last_at'] = captured_at
            self.state.update(batches=self.state['batches'] + 1,
                              received_rows=self.state['received_rows'] + rows,
                              activity_at=utc_now(), sequence=self.state['sequence'] + 1)
            self._publish()

    def _publish(self, force=False):
        if self.path is None or not force and time.monotonic() - self._last_write < 1:
            return
        self._last_write = time.monotonic()
        try:
            atomic_bytes(self.path, json.dumps(self.state, ensure_ascii=False).encode())
        except OSError:
            # Telemetry failure must not lose successfully fetched data.
            return

    def finish(self):
        with self._lock:
            for pending in self._pending.values():
                if pending:
                    self._line(pending)
            self._pending.clear()
            self._publish(force=True)


def progress_path(payload, output):
    value = payload.get('progress_path')
    if value is None:
        return None
    path = Path(value)
    if path.is_symlink() or path.name != 'progress.json' or path.resolve().parent.parent != output.resolve().parent:
        from .models import CenterError
        raise CenterError('ETL_PROGRESS_PATH', '进度文件必须位于本次任务的私有尝试目录。')
    return path


def progress_monitor(journal, run, state, path, check):
    """Poll a bounded sidecar without changing the worker's final JSON protocol."""
    last = 0.0

    def poll():
        nonlocal last
        check()
        telemetry = None
        try:
            if not path.is_symlink():
                with path.open('rb') as handle:
                    data = handle.read(65537)
                if len(data) <= 65536:
                    telemetry = json.loads(data)
        except (OSError, ValueError):
            pass
        previous = state.get('progress', {})
        changed = isinstance(telemetry, dict) and telemetry.get('sequence') != previous.get('sequence')
        if changed or time.monotonic() - last >= 5:
            if changed:
                state['progress'] = telemetry
            state['heartbeat_at'] = utc_now()
            journal.save_run(run)
            last = time.monotonic()
    return poll
