"""Contention retries are local transactions, not duplicate supplier requests."""
import sqlite3
from contextlib import contextmanager
from threading import Thread
import time

import pytest

from backend.data_sources.store import SourceStore
from backend.data_sources.models import CenterError
from backend.data_sources.presets import default_interfaces
from backend.data_sources.batches import capture_batch


def test_current_seed_never_claims_write_lock(tmp_path, monkeypatch):
    store = SourceStore(tmp_path); store.seed()
    original = store.connection; statements = []
    @contextmanager
    def trace():
        with original() as db:
            db.set_trace_callback(statements.append)
            yield db
    monkeypatch.setattr(store, 'connection', trace)
    writer = sqlite3.connect(store.path)
    writer.execute('BEGIN IMMEDIATE')
    try:
        store.seed()  # Must finish while another connection owns the writer slot.
    finally:
        writer.rollback(); writer.close()
    assert statements and all(s.startswith('SELECT') for s in statements)


def test_lock_retry_rolls_back_and_persists_only_once(tmp_path, monkeypatch):
    store = SourceStore(tmp_path)
    monkeypatch.setattr('backend.data_sources.store.time.sleep', lambda _: None)
    calls = []
    def transaction(db):
        calls.append(1)
        db.execute("INSERT INTO source_quota VALUES ('test',1,1)")
        if len(calls) == 1:
            exc = sqlite3.OperationalError('private')
            exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise exc
        return 'saved'
    assert store.database_operation(transaction) == 'saved'
    assert len(calls) == 2
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_quota').fetchone()[0] == 1


@pytest.mark.parametrize('code,attempts', [('SOURCE_DB_BUSY', 3), ('SOURCE_DB_FULL', 1), ('SOURCE_DB_IO', 1), ('SOURCE_PERMISSION_OR_PARAMS', 1)])
def test_only_busy_is_retried_and_retries_are_bounded(tmp_path, monkeypatch, code, attempts):
    store = SourceStore(tmp_path); calls = []
    monkeypatch.setattr('backend.data_sources.store.time.sleep', lambda _: None)
    def fail(db):
        calls.append(1)
        raise CenterError(code, 'safe error')
    with pytest.raises(CenterError): store.database_operation(fail)
    assert len(calls) == attempts


def test_real_sqlite_lock_release_can_finish_without_network(tmp_path, monkeypatch):
    store = SourceStore(tmp_path)
    connect = sqlite3.connect
    locker = connect(store.path, check_same_thread=False)
    locker.execute('BEGIN IMMEDIATE')
    monkeypatch.setattr(sqlite3, 'connect', lambda path, **kwargs: connect(path, timeout=0.02))
    def release():
        time.sleep(0.12)
        locker.rollback()
    thread = Thread(target=release); thread.start()
    try:
        store.database_operation(lambda db: db.execute("INSERT INTO source_quota VALUES ('retry',1,1)"))
    finally:
        thread.join(); locker.close()
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_quota').fetchone()[0] == 1


def test_capture_db_retry_keeps_one_batch(tmp_path, monkeypatch):
    store = SourceStore(tmp_path)
    original = store.connection; attempts = []
    @contextmanager
    def busy_once():
        attempts.append(1)
        if len(attempts) == 1: raise CenterError('SOURCE_DB_BUSY', 'busy')
        with original() as db: yield db
    monkeypatch.setattr(store, 'connection', busy_once)
    monkeypatch.setattr('backend.data_sources.store.time.sleep', lambda _: None)
    interface = next(i for i in default_interfaces() if i.api_name == 'ths_member')
    # Supplier rows already obtained; no supplier function belongs to this retry.
    result = capture_batch(store, interface, [], {'ts_code': 'A.TI'}, 'frozen')
    assert capture_batch(store, interface, [], {'ts_code': 'A.TI'}, 'frozen')['batch_id'] == result['batch_id']
    with original() as db: assert db.execute('SELECT COUNT(*) FROM source_run').fetchone()[0] == 1


def test_batch_insert_retry_does_not_rewrite_files(tmp_path, monkeypatch):
    from backend.data_sources import batches
    store = SourceStore(tmp_path)
    original = store.connection; transactions = []
    @contextmanager
    def busy_after_insert():
        transactions.append(1)
        with original() as db:
            yield db
            if len(transactions) == 2:  # Lookup succeeds; insert must roll back.
                exc = sqlite3.OperationalError('private'); exc.sqlite_errorcode = sqlite3.SQLITE_BUSY
                raise exc
    monkeypatch.setattr(store, 'connection', busy_after_insert)
    monkeypatch.setattr('backend.data_sources.store.time.sleep', lambda _: None)
    writes = []; atomic = batches.atomic_bytes
    def write(path, data):
        writes.append(path.name); return atomic(path, data)
    monkeypatch.setattr(batches, 'atomic_bytes', write)
    interface = next(i for i in default_interfaces() if i.api_name == 'ths_member')
    capture_batch(store, interface, [], {}, 'frozen')
    assert writes == ['raw.json', 'manifest.json'] and len(transactions) == 3
    with original() as db: assert db.execute('SELECT COUNT(*) FROM source_run').fetchone()[0] == 1


def test_quota_records_grant_time_after_waiting_for_database(tmp_path, monkeypatch):
    from backend.data_sources.quota import SharedQuota
    from backend.data_sources.models import DownloadPolicy
    store = SourceStore(tmp_path); quota = SharedQuota(store)
    times = iter([10., 25.])
    monkeypatch.setattr('backend.data_sources.quota.time.monotonic', lambda:next(times))
    assert quota.reserve([('source:test', DownloadPolicy())], 1, 100., 'lease') == 0
    with store.connection() as db:
        assert db.execute('SELECT called_at FROM source_quota').fetchone()[0] == 115.
