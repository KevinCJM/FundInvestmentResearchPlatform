"""Replayable raw batches and schema-checked candidates, never active datasets."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .mapping import map_table, validate_mapping
from .models import InterfaceConfig
from .store import SourceStore, utc_now
from backend.data_storage import guard_path


def atomic_bytes(path: Path, data: bytes) -> None:
    guard_path(path, write=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_batch(store: SourceStore, interface: InterfaceConfig, rows: list[dict[str, Any]], context: dict[str, Any], configuration_hash: str) -> dict[str, Any]:
    encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str, allow_nan=False).encode()
    raw_hash = hashlib.sha256(encoded).hexdigest()
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
    batch_id = hashlib.sha256(f"batch-v2:{interface.id}:{configuration_hash}:{raw_hash}:{context_hash}".encode()).hexdigest()
    with store.connection() as db:
        previous = db.execute("SELECT result FROM source_run WHERE id=?", (batch_id,)).fetchone()
    if previous is not None:
        return json.loads(previous[0])
    directory = store.root / "mapped_candidates" / interface.source_id / batch_id
    atomic_bytes(directory / "raw.json", encoded)
    validation = validate_mapping(interface)
    reports = []
    for index, mapping in enumerate(interface.mappings):
        if not mapping.enabled:
            continue
        if not validation["valid"]:
            reports.append({"table_id": mapping.target_table, "status": "REJECTED", "errors": validation["errors"]})
            continue
        table, errors = map_table([{**context, **row} for row in rows], mapping, interface.source_id, batch_id)
        entry = {"table_id": mapping.target_table, "status": "REJECTED" if errors else "VALIDATED_CANDIDATE", "rows": table.num_rows, "rejected_rows": len(errors), "errors": errors[:20]}
        if not errors and table.num_rows:
            path = directory / f"{index}_{mapping.target_table}.parquet"
            descriptor, temporary = tempfile.mkstemp(dir=directory)
            os.close(descriptor)
            try:
                pq.write_table(table, temporary, compression="zstd")
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
            entry["artifact"] = path.relative_to(store.root).as_posix()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(block)
            entry["checksum"] = digest.hexdigest()
        reports.append(entry)
    return _record_batch(store, interface, rows, reports, directory, batch_id, configuration_hash, raw_hash)


def _record_batch(store, interface, rows, reports, directory, batch_id, configuration_hash, raw_hash):
    status = "EMPTY" if not rows else "REJECTED" if not reports or any(report["status"] == "REJECTED" for report in reports) else "VALIDATED_CANDIDATE"
    result = {"batch_id": batch_id, "interface_id": interface.id, "source_id": interface.source_id,
              "status": status, "source_rows": len(rows), "tables": reports, "published": False,
              "configuration_hash": configuration_hash, "raw_checksum": raw_hash, "batch_format_version": 2,
              "created_at": utc_now(), "message": "标准化候选批次；尚未完成跨批次去重、外键核验和数据版本发布。"}
    atomic_bytes(directory / "manifest.json", json.dumps(result, ensure_ascii=False).encode())
    with store.connection() as db:
        db.execute("INSERT OR IGNORE INTO source_run (id,interface_id,status,snapshot,result,created_at) VALUES (?,?,?,?,?,?)", (batch_id, interface.id, status, json.dumps(interface.model_dump(mode="json"), ensure_ascii=False), json.dumps(result, ensure_ascii=False), result["created_at"]))
    return result


def recent_batches(store: SourceStore) -> list[dict[str, Any]]:
    with store.connection() as db:
        rows = db.execute("SELECT result FROM source_run ORDER BY created_at DESC LIMIT 30").fetchall()
    return [json.loads(row[0]) for row in rows]
