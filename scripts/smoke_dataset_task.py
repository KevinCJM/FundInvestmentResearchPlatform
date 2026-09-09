"""Explicit single-request smoke for the registered dataset worker; temporary only."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.data_sources.acquisition import fingerprint
from backend.data_sources.credentials import read_credential, save_credential
from backend.data_sources.models import CenterError, SourceConfig
from backend.data_sources.store import SourceStore
from backend.data_sources.task_runtime import run_worker
from backend.services.refresh_runtime import InterProcessFileLock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-network', action='store_true')
    args = parser.parse_args()
    if not args.confirm_network:
        parser.error('Explicit --confirm-network required; no request performed.')
    actual = SourceStore(ROOT / 'data')
    lock = InterProcessFileLock(actual.root / '.tushare_refresh.lock')
    if not lock.acquire(owner='single-dataset-smoke'):
        raise CenterError('DATA_TASK_RUNNING', '已有任务运行，未发起请求。')
    try:
        with tempfile.TemporaryDirectory(prefix='dataset-smoke-') as temporary:
            store = SourceStore(Path(temporary)); store.seed()
            source = SourceConfig.model_validate(store.get('source','tushare')['config'])
            source.policy.max_attempts = 1
            source.policy.max_concurrency = 1
            source.policy.max_runtime_seconds = 90
            store.save(source,1)
            save_credential(store,'tushare',read_credential(actual,'tushare'))
            current = [r for kind in ('source','interface') for r in store.list(kind) if r['config'].get('source_id',r['config']['id']) == 'tushare']
            output = store.root/'etl_runs'/'smoke'/'calendar'; output.mkdir(parents=True)
            deadline = time.monotonic() + 90
            def check():
                if time.monotonic() >= deadline:
                    raise CenterError('SMOKE_TIMEOUT','Smoke timeout')
            payload = {'root':str(store.root),'directory':str(output),'task_id':'tushare.calendar','source_id':'tushare',
                       'source_hash':fingerprint(current),'params':{'start_date':'20240102','end_date':'20240105'},
                       'mode':'full','has_baseline':False,'resume':False,'timeout':90}
            result = run_worker(payload,check,lock)
            import pyarrow.parquet as pq
            rows = pq.ParquetFile(output/'trade_day_df.parquet').metadata.num_rows
            assert result['batches'] == 1 and rows == 4 and not result['mapped_rejected_batches'], result
            from backend.data_sources.etl_service import execution_fingerprint
            print(json.dumps({'status':'passed','task':'tushare.calendar','requests':1,'rows':rows,
                              'temporary_only':True,'published':False,'execution_hash':execution_fingerprint()}))
    finally:
        lock.release()


if __name__ == '__main__':
    main()
