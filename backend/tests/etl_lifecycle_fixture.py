"""Offline subprocess fixture: production launcher/runner/worker, fake acquisition only."""
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from backend.data_sources import etl_executor, task_runtime
from backend.data_sources.etl_store import EtlStore
from backend.data_sources.store import SourceStore
from backend.services import etl_routes, etl_recovery


def fixture_command(mode):
    return [sys.executable, '-m', 'backend.tests.etl_lifecycle_fixture', mode]


@asynccontextmanager
async def lifespan(app):
    store = SourceStore(Path(os.environ['ETL_TEST_ROOT']))
    etl_routes.get_store = lambda: store
    etl_executor.runner_command = lambda: fixture_command('runner')
    etl_recovery.runner_command = lambda: fixture_command('recovery')
    app.state.reconnected = etl_executor.reconnect(EtlStore(store))
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(etl_routes.router)


@app.get('/health')
def health():
    return app.state.reconnected


def offline_acquire(payload, spec, output):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from backend.data_sources.task_progress import TaskProgressLog, progress_path
    root = Path(payload['root'])
    action = payload['task_id'].split('.')[-1]
    # Exclusive creation proves retries/restarts did not silently duplicate work.
    with (root / (action + '.called')).open('x') as handle:
        handle.write(str(os.getpid()))
    log = TaskProgressLog(progress_path(payload, output))
    deadline = time.monotonic() + 50
    while not (root / (action + '.release')).exists():
        if time.monotonic() > deadline:
            raise RuntimeError('offline fixture timed out')
        log.write('[INFO] 离线测试分片仍在推进\n')
        time.sleep(.1)
    pq.write_table(pa.table({'value': ['offline']}), output / (action + '.parquet'))
    log.finish()
    return {'received_rows': 1, 'warnings': 0, 'mapped_rejected_batches': 0}


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'runner':
        from backend.data_sources import etl_runner
        task_runtime.worker_command = lambda: fixture_command('worker')
        etl_runner.main()
    elif mode == 'worker':
        from backend.data_sources import task_worker
        task_worker.acquire = offline_acquire
        task_worker.main()
    elif mode == 'recovery':
        etl_executor.runner_command = lambda: fixture_command('runner')
        original = etl_recovery.execute
        def held_recovery(store, job):
            (store.root / 'recovery.called').write_text(str(os.getpid()))
            deadline = time.monotonic() + 40
            while not (store.root / 'recovery.release').exists():
                if time.monotonic() > deadline:
                    raise RuntimeError('offline recovery fixture timed out')
                time.sleep(.1)
            original(store, job)
        etl_recovery.execute = held_recovery
        etl_recovery.main()
    else:
        raise ValueError('unknown fixture mode')
