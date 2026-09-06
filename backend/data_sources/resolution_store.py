"""CAS policies and immutable, replayable multi-source candidate releases."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .batches import atomic_bytes
from .mapping import arrow_type, convert
from .models import CenterError
from .resolution import TABLES_BY_ID, resolve_records, _timestamp
from .resolution_models import ResolutionConfig
from .store import SourceStore, utc_now

# Bump whenever selection/validation semantics change; old runs stay immutable.
RESOLUTION_ENGINE_VERSION = "1.0.0"


def _init(store: SourceStore) -> None:
    with store.connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS resolution_config (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS resolution_revision (revision INTEGER PRIMARY KEY, body TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS resolution_run (id TEXT PRIMARY KEY, table_id TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
        """)
        value, now = ResolutionConfig().model_dump_json(), utc_now()
        db.execute("INSERT OR IGNORE INTO resolution_config VALUES (1,1,?,?)", (value, now))
        db.execute("INSERT OR IGNORE INTO resolution_revision VALUES (1,?,?)", (value, now))


def get_policy(store: SourceStore) -> dict:
    _init(store)
    with store.connection() as db:
        row = db.execute("SELECT * FROM resolution_config WHERE id=1").fetchone()
        runs = db.execute("SELECT result FROM resolution_run ORDER BY created_at DESC LIMIT 10").fetchall()
    return {"config": json.loads(row["body"]), "revision": row["revision"], "updated_at": row["updated_at"],
            "runs": [json.loads(run[0]) for run in runs]}


def save_policy(store: SourceStore, payload: dict, expected_revision: int) -> dict:
    _init(store)
    config = ResolutionConfig.model_validate(payload)
    known = {item["config"]["id"] for item in store.list("source")}
    if set(config.default_source_priority + [s for t in config.tables for s in t.source_priority]) - known:
        raise CenterError("RULE_SOURCE_UNKNOWN", "优先级包含已删除或不存在的数据源。")
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT revision FROM resolution_config WHERE id=1").fetchone()
        if row[0] != expected_revision:
            raise CenterError("REVISION_CONFLICT", "取值规则已被修改，请重新加载。", 409)
        revision, now = expected_revision + 1, utc_now()
        db.execute("UPDATE resolution_config SET revision=?,body=?,updated_at=? WHERE id=1", (revision, config.model_dump_json(), now))
        db.execute("INSERT INTO resolution_revision VALUES (?,?,?)", (revision, config.model_dump_json(), now))
    return get_policy(store)


def _checksum(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load_candidates(store: SourceStore, table_id: str, start_date: str | None = None, end_date: str | None = None, as_of: str | None = None, *, batch_ids: list[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    definition = TABLES_BY_ID.get(table_id)
    if definition is None or not definition.source_mappable:
        raise CenterError("RESOLUTION_TARGET_INVALID", "请选择外部业务数据表。")
    first = date.fromisoformat(start_date) if start_date else None
    last = date.fromisoformat(end_date) if end_date else None
    if first and last and first > last:
        raise CenterError("DATE_RANGE_INVALID", "开始日期不能晚于结束日期。")
    date_column = next((f.name for f in definition.fields if f.role == "observation_time"), None)
    if date_column is None:
        date_column = next((n for n in ("valid_from", "effective_from", "calendar_date") if n in {f.name for f in definition.fields}), None)
    filters = []
    if date_column and first:
        filters.append((date_column, ">=", first))
    if date_column and last:
        filters.append((date_column, "<=", last))
    with store.connection() as db:
        if batch_ids is None:
            batches = db.execute("SELECT result FROM source_run WHERE EXISTS (SELECT 1 FROM json_each(source_run.result,'$.tables') WHERE json_extract(value,'$.table_id')=?) ORDER BY created_at DESC LIMIT 1001", (table_id,)).fetchall()
        else:
            batches = db.execute("SELECT result FROM source_run WHERE id IN (SELECT value FROM json_each(?)) ORDER BY created_at DESC,id", (json.dumps(batch_ids),)).fetchall()
            if len(batches) != len(set(batch_ids)):
                raise CenterError("ETL_INPUT_MISSING", "锁定的候选批次已缺失，不能使用其他批次替代。")
    if len(batches) > 1000:
        raise CenterError("RESOLUTION_BATCH_LIMIT", "候选批次数超过当前安全上限，需要先按数据分区归档后合并；不会截取部分批次冒充完整结果。")
    enabled_sources = {s["config"]["id"] for s in store.list("source") if s["config"]["enabled"]}
    rows, inputs, issues, seen = [], [], {}, set()
    cutoff = _timestamp(as_of)
    for batch_row in batches:
        batch = json.loads(batch_row[0])
        if cutoff is not None and _timestamp(batch['created_at']) > cutoff:
            continue
        source = batch["source_id"]
        if source not in enabled_sources:
            issues[source] = "SOURCE_DISABLED"
            continue
        for item in batch["tables"]:
            if item["table_id"] != table_id:
                continue
            if source not in seen:
                if item["status"] == "REJECTED":
                    issues[source] = "LATEST_SOURCE_BATCH_REJECTED"
                seen.add(source)
            if not item.get("artifact"):
                continue
            path = (store.root / item["artifact"]).resolve()
            root = (store.root / "mapped_candidates").resolve()
            if root not in path.parents or not path.is_file():
                raise CenterError("CANDIDATE_UNAVAILABLE", "候选文件缺失或路径无效，无法证明输入完整。")
            checksum = _checksum(path)
            if item.get("checksum") and item["checksum"] != checksum:
                raise CenterError("CANDIDATE_CHECKSUM_INVALID", "候选校验和不一致，不能使用被修改的候选。")
            verified = bool(item.get("checksum"))
            table = pq.read_table(path, filters=filters or None)
            expected = pa.schema([pa.field(f.name, arrow_type(f.data_type), nullable=f.nullable) for f in definition.fields])
            if not table.schema.equals(expected, check_metadata=False):
                raise CenterError("CANDIDATE_SCHEMA_INVALID", "候选字段合同校验失败，请重新下载。")
            if len(rows) + table.num_rows > 100000:
                raise CenterError("RESOLUTION_LIMIT", "日期范围内超过 100000 条候选，请缩小日期区间。")
            values = table.to_pylist()
            if any(row.get("source_id") != source or row.get("source_batch_id") != batch["batch_id"] for row in values):
                raise CenterError("CANDIDATE_LINEAGE_INVALID", "候选文件与批次来源不一致，已拒绝合并。")
            rows.extend(values)
            inputs.append({"batch_id": batch["batch_id"], "artifact": item["artifact"], "checksum": checksum, "rows": table.num_rows, "verified": verified})
    return rows, sorted(inputs, key=lambda v: v["artifact"]), issues


def resolve_saved(store: SourceStore, table_id: str, expected_revision: int, start_date: str | None = None,
                  end_date: str | None = None, as_of: str | None = None, *,
                  batch_ids: list[str] | None = None, frozen_policy: dict | None = None) -> dict:
    saved = frozen_policy if frozen_policy is not None else get_policy(store)
    if saved["revision"] != expected_revision:
        raise CenterError("REVISION_CONFLICT", "规则已修改，请重新加载后合并。", 409)
    config = ResolutionConfig.model_validate(saved["config"])
    rows, inputs, issues = load_candidates(store, table_id, start_date, end_date, as_of, batch_ids=batch_ids)
    frozen = {"engine_version": RESOLUTION_ENGINE_VERSION, "table_id": table_id, "policy": saved["config"], "policy_revision": expected_revision,
              "inputs": inputs, "source_issues": issues, "start_date": start_date, "end_date": end_date, "as_of": as_of}
    run_id = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    with store.connection() as db:
        prior = db.execute("SELECT result FROM resolution_run WHERE id=?", (run_id,)).fetchone()
    if prior:
        previous = json.loads(prior[0])
        path = (store.root / previous["artifact"]).resolve()
        if (store.root / "resolved_candidates").resolve() not in path.parents or not path.is_file() or _checksum(path) != previous.get("checksum"):
            raise CenterError("RESOLUTION_CHECKSUM_INVALID", "已保存取值结果的校验和不一致，拒绝复用。")
        return previous
    unverified = frozenset(item["batch_id"] for item in inputs if not item["verified"])
    result = resolve_records(table_id, rows, config, as_of=as_of, source_issues=issues, unverified_batches=unverified)
    result["summary"]["unverified_input_batches"] = len(unverified)
    definition = TABLES_BY_ID[table_id]
    schema = pa.schema([pa.field(f.name, arrow_type(f.data_type), nullable=f.nullable) for f in definition.fields],
                       metadata={b"table_id": table_id.encode(), b"status": b"resolved_candidate", b"policy_revision": str(expected_revision).encode(), b"engine_version": RESOLUTION_ENGINE_VERSION.encode()})
    normalized = [{f.name: convert(row.get(f.name), f) for f in definition.fields} for row in result["rows"]]
    table = pa.Table.from_pylist(normalized, schema=schema)
    parent = store.root / "resolved_candidates"
    parent.mkdir(exist_ok=True)
    target = parent / run_id
    review_count = result["summary"].get("CONFLICT", 0) + result["summary"].get("BLOCKED", 0) + result["summary"].get("rejected_input_rows", 0)
    summary = {"run_id": run_id, "engine_version": RESOLUTION_ENGINE_VERSION, "table_id": table_id, "policy_revision": expected_revision,
               "summary": result["summary"], "status": "NEEDS_REVIEW" if review_count else "CANDIDATE_READY" if table.num_rows else "NO_ELIGIBLE_DATA",
               "published": False, "created_at": utc_now(), "as_of": as_of,
               "artifact": f"resolved_candidates/{run_id}/data.parquet", "decisions": result["decisions"][:100],
               "excluded": result["excluded"][:100], "input_batches": len(inputs), "execution": result["execution"]}
    with tempfile.TemporaryDirectory(prefix=".resolve-", dir=parent) as tmp:
        directory = Path(tmp)
        pq.write_table(table, directory / "data.parquet", compression="zstd")
        atomic_bytes(directory / "decisions.json", json.dumps({**result, "rows": []}, ensure_ascii=False, default=str).encode())
        summary["checksum"] = _checksum(directory / "data.parquet")
        atomic_bytes(directory / "manifest.json", json.dumps({**summary, "snapshot": frozen}, ensure_ascii=False, default=str).encode())
        if target.exists():
            # Recover a crash after atomic directory rename, before catalog commit.
            manifest = json.loads((target / "manifest.json").read_text())
            if manifest.get("snapshot") != frozen or _checksum(target / "data.parquet") != manifest.get("checksum"):
                raise CenterError("RESOLUTION_CHECKSUM_INVALID", "取值制品校验失败，不能覆盖已有版本。")
            summary = {k: v for k, v in manifest.items() if k != "snapshot"}
        else:
            os.replace(directory, target)
    with store.connection() as db:
        db.execute("INSERT OR IGNORE INTO resolution_run VALUES (?,?,?,?)", (run_id, table_id, json.dumps(summary, ensure_ascii=False, default=str), summary["created_at"]))
    return summary
