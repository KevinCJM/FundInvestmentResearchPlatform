"""ETL revisions and durable execution journals in the existing control database."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .acquisition import fingerprint
from .batches import atomic_bytes
from .etl_models import EtlDefinition
from .models import CenterError
from .resolution_store import _checksum
from .store import SourceStore, utc_now


class EtlStore:
    def __init__(self, source_store: SourceStore):
        self.sources = source_store
        self.root = source_store.root
        with source_store.connection() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS etl_workflow (id TEXT PRIMARY KEY, revision INTEGER NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS etl_workflow_revision (id TEXT NOT NULL, revision INTEGER NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY(id,revision));
            CREATE TABLE IF NOT EXISTS etl_run (id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, body TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL);
            """)

    def workflows(self) -> list[dict]:
        with self.sources.connection() as db:
            rows = db.execute("SELECT * FROM etl_workflow ORDER BY updated_at DESC,id").fetchall()
        return [{"id": r["id"], "revision": r["revision"], "definition": json.loads(r["body"]), "updated_at": r["updated_at"]} for r in rows]

    def workflow(self, identifier: str) -> dict:
        for item in self.workflows():
            if item["id"] == identifier:
                return item
        raise CenterError("ETL_NOT_FOUND", "流程不存在或已删除。", 404)

    def save_workflow(self, identifier: str, definition: EtlDefinition, expected: int) -> dict:
        with self.sources.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision FROM etl_workflow WHERE id=?", (identifier,)).fetchone()
            if (row[0] if row else 0) != expected:
                raise CenterError("REVISION_CONFLICT", "流程已被修改，请重新加载或另存为新流程。", 409)
            if not row and db.execute("SELECT 1 FROM etl_workflow_revision WHERE id=?", (identifier,)).fetchone():
                raise CenterError("ETL_ID_RETIRED", "该流程 ID 已停用，请使用新的 ID。", 409)
            values = (identifier, expected + 1, definition.model_dump_json(), utc_now())
            db.execute("INSERT OR REPLACE INTO etl_workflow VALUES (?,?,?,?)", values)
            db.execute("INSERT INTO etl_workflow_revision VALUES (?,?,?,?)", values)
        return self.workflow(identifier)

    def delete_workflow(self, identifier: str, expected: int) -> None:
        with self.sources.connection() as db:
            cursor = db.execute("DELETE FROM etl_workflow WHERE id=? AND revision=?", (identifier, expected))
            if cursor.rowcount != 1:
                raise CenterError("REVISION_CONFLICT", "流程不存在或已修改，请重新加载。", 409)

    def get_run(self, identifier: str) -> dict:
        with self.sources.connection() as db:
            row = db.execute("SELECT body,cancel_requested FROM etl_run WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise CenterError("ETL_RUN_NOT_FOUND", "运行记录不存在。", 404)
        run = json.loads(row[0])
        run["cancel_requested"] = bool(row[1])
        return run

    def save_run(self, run: dict, *, create: bool = False) -> None:
        run["updated_at"] = utc_now()
        with self.sources.connection() as db:
            if create:
                db.execute("INSERT INTO etl_run(id,request_hash,body,updated_at) VALUES (?,?,?,?)", (run["run_id"], run["request_hash"], json.dumps(run, ensure_ascii=False, default=str), run["updated_at"]))
            else:
                db.execute("UPDATE etl_run SET body=?,updated_at=? WHERE id=?", (json.dumps(run, ensure_ascii=False, default=str), run["updated_at"], run["run_id"]))

    def runs(self) -> list[dict]:
        with self.sources.connection() as db:
            rows = db.execute("SELECT id FROM etl_run ORDER BY updated_at DESC LIMIT 30").fetchall()
        return [self.get_run(row[0]) for row in rows]

    def request_cancel(self, identifier: str, value: bool = True) -> None:
        self.get_run(identifier)
        with self.sources.connection() as db:
            db.execute("UPDATE etl_run SET cancel_requested=? WHERE id=?", (int(value), identifier))

    def artifact(self, path: Path) -> dict:
        return {"path": path.relative_to(self.root).as_posix(), "checksum": _checksum(path)}

    def checked_path(self, artifact: dict) -> Path:
        path = (self.root / artifact["path"]).resolve()
        if self.root.resolve() not in path.parents or not path.is_file() or _checksum(path) != artifact.get("checksum"):
            raise CenterError("ETL_ARTIFACT_CHANGED", "步骤制品缺失或校验和变化，拒绝继续；请新建运行。", 409)
        return path

    def write_json(self, path: Path, value) -> dict:
        atomic_bytes(path, json.dumps(value, ensure_ascii=False, default=str, allow_nan=False).encode())
        return self.artifact(path)

    def interrupted(self, run: dict) -> dict:
        if run["status"] != "RUNNING":
            return run
        try:
            if run.get('executor'):
                from .etl_executor import executor_alive
                if not executor_alive(run):
                    raise ProcessLookupError
            os.kill(run["owner_pid"], 0)
        except ProcessLookupError:
            # A late status poll must not overwrite a completion or a new owner.
            with self.sources.connection() as db:
                db.execute('BEGIN IMMEDIATE')
                row = db.execute('SELECT body,cancel_requested FROM etl_run WHERE id=?', (run['run_id'],)).fetchone()
                latest = json.loads(row[0])
                latest['cancel_requested'] = bool(row[1])
                same_owner = all(latest.get(k) == run.get(k) for k in ('owner_pid', 'owner_instance', 'executor', 'attempt'))
                if latest['status'] == 'RUNNING' and same_owner:
                    latest['status'] = 'INTERRUPTED'
                    latest['error'] = ('独立任务执行器已退出，检查点保留；需核验下载锁和执行版本后续跑。' if latest.get('executor') else
                                       '调度服务已退出，后台工作进程可能仍在执行。已完成步骤保留，续跑前需核验下载锁和执行版本。')
                    for step in latest['steps']:
                        if step['status'] == 'RUNNING':
                            step['status'] = 'INTERRUPTED'
                    latest['updated_at'] = utc_now()
                    db.execute('UPDATE etl_run SET body=?,updated_at=? WHERE id=?',
                               (json.dumps(latest, ensure_ascii=False, default=str), latest['updated_at'], run['run_id']))
                return latest
        return run


def public_run(run: dict, *, detail: bool = True) -> dict:
    from .etl_collection import collection_timing
    hidden = {"frozen", "owner_pid", "owner_instance", "executor", "request_hash"}
    result = {k: v for k, v in run.items() if k not in hidden}
    if not detail:
        result.pop("definition", None)
        result.pop("template_definition", None)
    result['collection_timing'] = collection_timing(run)
    return result
