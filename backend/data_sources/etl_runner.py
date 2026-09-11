"""Detached per-run coordinator. API processes are clients, not its parents."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path

from . import etl_service as etl
from .etl_executor import heartbeat, process_birth
from .etl_store import EtlStore
from .models import CenterError
from .store import SourceStore, utc_now


def main():
    root, identifier, descriptor, token = sys.argv[1:]
    lock = etl.InterProcessFileLock.from_inherited_fd(Path(root) / '.tushare_refresh.lock', int(descriptor))
    journal = EtlStore(SourceStore(Path(root)))
    run, stop = None, None
    try:
        # Parent persists ownership after Popen; before that we cannot do work.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = journal.get_run(identifier)
            executor = run.get('executor', {})
            if executor.get('token') == token and executor.get('pid') == os.getpid() and executor.get('birth') == process_birth(os.getpid()):
                break
            time.sleep(.05)
        else:
            raise CenterError('ETL_EXECUTOR_OWNERSHIP', '执行器未获得匹配的任务身份，拒绝执行。')
        stop = heartbeat(journal, run)
        signal.signal(signal.SIGTERM, lambda *_: journal.request_cancel(identifier))
        signal.signal(signal.SIGINT, lambda *_: journal.request_cancel(identifier))
        if etl.execution_fingerprint() != run['frozen']['execution_fingerprint']:
            raise CenterError('ETL_IMPLEMENTATION_CHANGED', '执行器启动前代码版本已改变，拒绝混用版本。', 409)
        if any(s['kind'] in {'map', 'resolve'} for s in run['steps']):
            from .resolution_kernels import warm_resolution_kernels
            if not warm_resolution_kernels()['complete']:
                raise CenterError('ETL_WARMUP_FAILED', '独立执行器数值内核预热未完成。')
        etl.execute(journal, run, lock)
    except Exception as exc:
        # Never publish exception strings from credentials, providers or files.
        current = journal.get_run(identifier)
        if current.get('executor', {}).get('token') == token:
            current.update(status='FAILED', code=exc.code if isinstance(exc, CenterError) else 'ETL_EXECUTOR_FAILED',
                           error=exc.message if isinstance(exc, CenterError) else '独立执行器未能完成初始化；检查点保留。', finished_at=utc_now())
            journal.save_run(current)
    finally:
        if stop:
            stop.set()
        lock.release()


if __name__ == '__main__':
    main()
