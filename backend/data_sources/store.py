"""SQLite configuration revisions, isolated from market snapshots."""
from __future__ import annotations

import json
import os
import sqlite3
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .models import CenterError, InterfaceConfig, SourceConfig
from backend.data_storage import guard_path

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "data"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SourceStore:
    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        guard_path(self.root, write=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "data_sources.sqlite3"
        with self.connection() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS source_config (
                kind TEXT NOT NULL, id TEXT NOT NULL, source_id TEXT NOT NULL,
                revision INTEGER NOT NULL, builtin INTEGER NOT NULL DEFAULT 0,
                body TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(kind,id));
            CREATE TABLE IF NOT EXISTS source_config_revision (
                kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                body TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(kind,id,revision));
            CREATE TABLE IF NOT EXISTS source_quota (
                quota_key TEXT NOT NULL, called_at REAL NOT NULL, reserved_rows INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS quota_time ON source_quota(quota_key,called_at);
            CREATE TABLE IF NOT EXISTS source_run (
                id TEXT PRIMARY KEY, interface_id TEXT NOT NULL, status TEXT NOT NULL,
                snapshot TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS source_run_created ON source_run(created_at DESC);
            CREATE TABLE IF NOT EXISTS source_preset (
                kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                PRIMARY KEY(kind,id));
            """)
        os.chmod(self.path, 0o600)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        guard_path(self.root)
        db = None
        try:
            db = sqlite3.connect(self.path, timeout=15)
            db.row_factory = sqlite3.Row
            with db:
                yield db
        except sqlite3.OperationalError as exc:
            # Never leak SQL, paths or bound configuration values. This is a
            # local storage failure, not permission to retry a supplier request.
            code = getattr(exc, 'sqlite_errorcode', 0) & 0xff
            errors = {
                sqlite3.SQLITE_BUSY: ('SOURCE_DB_BUSY', '控制数据库锁等待超时'),
                sqlite3.SQLITE_LOCKED: ('SOURCE_DB_BUSY', '控制数据库被并发事务占用'),
                sqlite3.SQLITE_FULL: ('SOURCE_DB_FULL', '控制数据库所在磁盘空间不足'),
                sqlite3.SQLITE_IOERR: ('SOURCE_DB_IO', '控制数据库磁盘读写失败，请检查数据盘连接'),
                sqlite3.SQLITE_CANTOPEN: ('SOURCE_DB_OPEN', '控制数据库无法打开，请检查数据盘与权限'),
                sqlite3.SQLITE_READONLY: ('SOURCE_DB_READONLY', '控制数据库不可写，请检查数据盘权限'),
            }
            identifier, message = errors.get(code, ('SOURCE_DB_OPERATIONAL', '控制数据库操作失败'))
            raise CenterError(identifier, message + '；检查点保留，未发布数据。', 503) from None
        finally:
            if db is not None:
                db.close()

    def seed(self) -> None:
        from .presets import default_interfaces, default_source
        from .akshare_presets import akshare_interfaces, akshare_source
        configs = [("source", default_source()), ("source", akshare_source())]
        configs.extend(("interface", item) for item in (*default_interfaces(), *akshare_interfaces()))
        # Catalog GETs call seed too. A current catalog must remain read-only:
        # do not claim the downloader's writer lock merely to inspect defaults.
        with self.connection() as db:
            baseline = {(r[0], r[1]): r[2] for r in db.execute('SELECT kind,id,body FROM source_preset')}
            present = {(r[0], r[1]) for r in db.execute('SELECT kind,id FROM source_config')}
            retired = {(r[0], r[1]) for r in db.execute('SELECT DISTINCT kind,id FROM source_config_revision')}
        required = False
        for kind, config in configs:
            identity = (kind, config.id)
            encoded = json.dumps(config.model_dump(mode='json'), ensure_ascii=False, sort_keys=True)
            parent_exists = kind == 'source' or ('source', config.source_id) in present
            if baseline.get(identity) != encoded or (identity not in present and identity not in retired and parent_exists):
                required = True
                break
        if not required:
            return
        # Defaults changed or first startup: recheck atomically under the lock.
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for kind, config in configs:
                body = config.model_dump(mode="json")
                encoded = json.dumps(body, ensure_ascii=False, sort_keys=True)
                previous = db.execute("SELECT * FROM source_config WHERE kind=? AND id=?", (kind, config.id)).fetchone()
                baseline = db.execute("SELECT body FROM source_preset WHERE kind=? AND id=?", (kind, config.id)).fetchone()
                # Bootstrap is a one-time insert, never an upgrade of saved user
                # settings. Revision history is also the tombstone after deletion.
                retired = db.execute("SELECT 1 FROM source_config_revision WHERE kind=? AND id=?", (kind, config.id)).fetchone()
                parent_missing = kind == "interface" and not db.execute("SELECT 1 FROM source_config WHERE kind='source' AND id=?", (body["source_id"],)).fetchone()
                if previous is None and not retired and not parent_missing:
                    now, revision = utc_now(), 1 if previous is None else previous["revision"] + 1
                    if previous is not None:
                        db.execute("INSERT OR IGNORE INTO source_config_revision VALUES (?,?,?,?,?)", (kind, config.id, previous["revision"], previous["body"], previous["updated_at"]))
                    db.execute("INSERT OR REPLACE INTO source_config VALUES (?,?,?,?,?,?,?)", (kind, config.id, body.get("source_id", config.id), revision, 1, encoded, now))
                    db.execute("INSERT OR IGNORE INTO source_config_revision VALUES (?,?,?,?,?)", (kind, config.id, revision, encoded, now))
                if baseline is None or baseline[0] != encoded:
                    db.execute("INSERT OR REPLACE INTO source_preset VALUES (?,?,?)", (kind, config.id, encoded))

    def database_operation(self, operation, *, check=None, max_attempts=3):
        """Retry ONLY a rolled-back local DB transaction, never a supplier call.

        Callbacks must contain SQLite work only. Each connection already bounds
        lock waiting to 15s; three attempts tolerate short external-disk stalls.
        Disk full, I/O, permission and contract errors remain fail-closed.
        """
        if max_attempts < 1:
            raise ValueError('max_attempts must be positive')
        for attempt in range(max_attempts):
            if check is not None:
                check()
            try:
                with self.connection() as db:
                    return operation(db)
            except CenterError as exc:
                if exc.code != 'SOURCE_DB_BUSY' or attempt + 1 == max_attempts:
                    raise
                time.sleep(0.25 * 2 ** attempt + random.uniform(0, 0.25))

    @staticmethod
    def decode(row: sqlite3.Row) -> dict[str, Any]:
        return {"config": json.loads(row["body"]), "revision": row["revision"], "builtin": bool(row["builtin"]), "updated_at": row["updated_at"]}

    def get(self, kind: str, identifier: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM source_config WHERE kind=? AND id=?", (kind, identifier)).fetchone()
        if row is None:
            raise CenterError("CONFIG_NOT_FOUND", "未找到指定配置。", 404)
        return self.decode(row)

    def list(self, kind: str) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM source_config WHERE kind=? ORDER BY builtin DESC,id", (kind,)).fetchall()
        return [self.decode(row) for row in rows]

    def save(self, config: SourceConfig | InterfaceConfig, expected_revision: int) -> dict[str, Any]:
        kind = "source" if isinstance(config, SourceConfig) else "interface"
        body = config.model_dump(mode="json")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT * FROM source_config WHERE kind=? AND id=?", (kind, config.id)).fetchone()
            actual = previous["revision"] if previous else 0
            if actual != expected_revision:
                raise CenterError("REVISION_CONFLICT", "配置已被修改，请重新加载后再保存。", 409)
            if previous is None and db.execute("SELECT 1 FROM source_config_revision WHERE kind=? AND id=?", (kind, config.id)).fetchone():
                raise CenterError("CONFIG_ID_RETIRED", "该 ID 已有历史记录，请为新配置使用新的 ID。", 409)
            if kind == "interface":
                parent = db.execute("SELECT body FROM source_config WHERE kind='source' AND id=?", (config.source_id,)).fetchone()
                if parent is None:
                    raise CenterError("SOURCE_NOT_FOUND", "请先保存数据源。")
                if previous and previous["source_id"] != config.source_id:
                    raise CenterError("SOURCE_IMMUTABLE", "已有接口不能迁移来源，请新建接口。")
            revision, now = actual + 1, utc_now()
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False)
            db.execute("INSERT OR REPLACE INTO source_config VALUES (?,?,?,?,?,?,?)", (kind, config.id, body.get("source_id", config.id), revision, previous["builtin"] if previous else 0, encoded, now))
            db.execute("INSERT INTO source_config_revision VALUES (?,?,?,?,?)", (kind, config.id, revision, encoded, now))
        return self.get(kind, config.id)

    def delete(self, kind: str, identifier: str, expected_revision: int) -> None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM source_config WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            if row is None:
                raise CenterError("CONFIG_NOT_FOUND", "配置不存在。", 404)
            if row["revision"] != expected_revision:
                raise CenterError("REVISION_CONFLICT", "配置已更新，请刷新。", 409)
            if kind == "source" and db.execute("SELECT 1 FROM source_config WHERE kind='interface' AND source_id=?", (identifier,)).fetchone():
                raise CenterError("SOURCE_IN_USE", "请先删除该来源下的接口。", 409)
            db.execute("DELETE FROM source_config WHERE kind=? AND id=?", (kind, identifier))
        if kind == "source":
            credential = self.root / ".tushare_token" if identifier == "tushare" else self.root / ".source_credentials" / identifier
            credential.unlink(missing_ok=True)
            (self.root / ".source_credentials" / (identifier + ".binding.json")).unlink(missing_ok=True)
