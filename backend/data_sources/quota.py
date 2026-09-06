"""Shared source/API request and row budgets across threads and processes."""
from __future__ import annotations
import time
import uuid
from contextlib import contextmanager
from typing import Iterator

from .models import CenterError, DownloadPolicy
from .store import SourceStore


class SharedQuota:
    def __init__(self, store: SourceStore) -> None:
        self.store = store
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS source_lease (lease_id TEXT, quota_key TEXT, expires_at REAL)")
            db.execute("CREATE INDEX IF NOT EXISTS lease_key ON source_lease(quota_key,expires_at)")

    def reserve(self, policies: list[tuple[str, DownloadPolicy]], rows: int, now: float, lease_id: str) -> float:
        """Reserve both levels atomically, or return a bounded wait duration."""
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM source_quota WHERE called_at<=?", (now - 60,))
            db.execute("DELETE FROM source_lease WHERE expires_at<=?", (now,))
            wait = 0.0
            for key, policy in policies:
                existing = db.execute("SELECT called_at,reserved_rows FROM source_quota WHERE quota_key=? ORDER BY called_at", (key,)).fetchall()
                active = db.execute("SELECT COUNT(*) FROM source_lease WHERE quota_key=?", (key,)).fetchone()[0]
                if active >= policy.max_concurrency:
                    wait = max(wait, 0.1)
                if existing:
                    wait = max(wait, policy.min_interval_seconds - (now - existing[-1][0]))
                if len(existing) >= policy.requests_per_minute:
                    wait = max(wait, 60 - (now - existing[0][0]))
                if policy.rows_per_minute is not None:
                    if rows > policy.rows_per_minute:
                        raise CenterError("ROW_QUOTA_TOO_SMALL", "单次预留行数超过来源或接口的每分钟行数额度。")
                    used = sum(item[1] for item in existing)
                    if used + rows > policy.rows_per_minute and existing:
                        wait = max(wait, 60 - (now - existing[0][0]))
            if wait > 0:
                return wait
            for key, policy in policies:
                db.execute("INSERT INTO source_quota VALUES (?,?,?)", (key, now, rows))
                expiry = now + policy.connect_timeout_seconds + policy.read_timeout_seconds + 10
                db.execute("INSERT INTO source_lease VALUES (?,?,?)", (lease_id, key, expiry))
        return 0.0

    @contextmanager
    def acquire(self, source_id: str, api_key: str, source: DownloadPolicy, interface: DownloadPolicy, rows: int) -> Iterator[None]:
        policies = [("source:" + source_id, source), ("api:" + source_id + ":" + api_key, interface)]
        lease_id = uuid.uuid4().hex
        deadline = time.monotonic() + min(source.max_runtime_seconds, interface.max_runtime_seconds)
        while True:
            wait = self.reserve(policies, rows, time.time(), lease_id)
            if wait <= 0:
                break
            if time.monotonic() + wait >= deadline:
                raise CenterError("QUOTA_WAIT_TIMEOUT", "等待共享配额超过任务上限。", 429)
            time.sleep(min(wait, 1.0))
        try:
            yield
        finally:
            with self.store.connection() as db:
                db.execute("DELETE FROM source_lease WHERE lease_id=?", (lease_id,))
