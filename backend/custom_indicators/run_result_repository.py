"""Bounded, file-backed paging for large synchronous evaluation results."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from custom_indicators.errors import IndicatorDomainError, NotFoundError


class EvaluationRunResultRepository:
    def __init__(
        self,
        directory: Path,
        *,
        ttl_seconds: int = 1_800,
        max_runs: int = 20,
        max_bytes: int = 2 * 1024 * 1024 * 1024,
        row_group_size: int = 500,
    ) -> None:
        self.directory = directory
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_runs = max(1, int(max_runs))
        self.max_bytes = max(1, int(max_bytes))
        self.row_group_size = max(1, int(row_group_size))
        self._lock = threading.RLock()

    def _paths(self, result_id: str) -> tuple[Path, Path]:
        if not result_id or any(character not in "0123456789abcdef" for character in result_id):
            raise NotFoundError("RESULT_NOT_FOUND", "评价结果不存在。")
        return (
            self.directory / f"{result_id}.json",
            self.directory / f"{result_id}.parquet",
        )

    def _read_manifest(self, result_id: str) -> dict[str, Any]:
        manifest_path, data_path = self._paths(result_id)
        if not manifest_path.exists() or not data_path.exists():
            raise NotFoundError("RESULT_NOT_FOUND", "评价结果不存在。")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NotFoundError("RESULT_NOT_FOUND", "评价结果不存在。") from exc
        if float(manifest.get("expires_at_epoch", 0.0)) <= time.time():
            self._delete_files(manifest_path, data_path)
            raise IndicatorDomainError(
                "RESULT_EXPIRED",
                "评价结果已过期，请重新运行方案。",
                status_code=410,
            )
        return manifest

    @staticmethod
    def _delete_files(*paths: Path) -> None:
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def cleanup(self) -> None:
        with self._lock:
            if not self.directory.exists():
                return
            now = time.time()
            records: list[tuple[float, int, Path, Path]] = []
            for manifest_path in self.directory.glob("*.json"):
                data_path = manifest_path.with_suffix(".parquet")
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    expires_at = float(manifest.get("expires_at_epoch", 0.0))
                    created_at = float(manifest.get("created_at_epoch", 0.0))
                except (OSError, ValueError, json.JSONDecodeError):
                    self._delete_files(manifest_path, data_path)
                    continue
                if expires_at <= now or not data_path.exists():
                    self._delete_files(manifest_path, data_path)
                    continue
                size = int(data_path.stat().st_size) + int(manifest_path.stat().st_size)
                records.append((created_at, size, manifest_path, data_path))
            records.sort(key=lambda item: item[0], reverse=True)
            retained_bytes = 0
            for index, (_, size, manifest_path, data_path) in enumerate(records):
                if index >= self.max_runs or retained_bytes + size > self.max_bytes:
                    self._delete_files(manifest_path, data_path)
                else:
                    retained_bytes += size

    def clear(self) -> int:
        """Remove only transient evaluation result files."""

        removed = 0
        with self._lock:
            if not self.directory.exists():
                return 0
            for path in self.directory.iterdir():
                if path.is_file() and path.suffix in {".json", ".parquet", ".tmp"}:
                    try:
                        path.unlink()
                        removed += 1
                    except FileNotFoundError:
                        pass
        return removed

    def store(self, result: dict[str, Any]) -> str:
        rows = list(result.get("rows") or [])
        result_id = uuid.uuid4().hex
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            manifest_path, data_path = self._paths(result_id)
            temp_manifest = manifest_path.with_suffix(".json.tmp")
            temp_data = data_path.with_suffix(".parquet.tmp")
            table = pa.table(
                {
                    "rank": pa.array([row.get("rank") for row in rows], type=pa.int64()),
                    "score": pa.array([row.get("score") for row in rows], type=pa.float64()),
                    "status": [str(row.get("status") or "") for row in rows],
                    "product_id": [
                        str((row.get("target") or {}).get("product_id") or "")
                        for row in rows
                    ],
                    "row_json": [
                        json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                        for row in rows
                    ],
                }
            )
            pq.write_table(
                table,
                temp_data,
                compression="zstd",
                row_group_size=self.row_group_size,
            )
            with temp_data.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_data, data_path)
            created_at = time.time()
            manifest = {
                "schema_version": 1,
                "result_id": result_id,
                "created_at_epoch": created_at,
                "created_at": datetime.fromtimestamp(
                    created_at, tz=timezone.utc
                ).isoformat(),
                "expires_at_epoch": created_at + self.ttl_seconds,
                "row_count": len(rows),
                "row_group_size": self.row_group_size,
                "summary": {
                    key: result.get(key)
                    for key in (
                        "plan_id",
                        "plan_revision",
                        "run_at",
                        "as_of",
                        "ranked_count",
                        "excluded_count",
                        "normalization",
                        "execution",
                    )
                },
            }
            with temp_manifest.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_manifest, manifest_path)
            self.cleanup()
        return result_id

    def page(
        self,
        result_id: str,
        *,
        page: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, min(int(page_size), 500))
        with self._lock:
            manifest = self._read_manifest(result_id)
            _, data_path = self._paths(result_id)
            total = int(manifest["row_count"])
            start = (page - 1) * page_size
            end = min(start + page_size, total)
            if start >= total and total:
                raise IndicatorDomainError(
                    "PAGE_OUT_OF_RANGE",
                    "评价结果页码超出范围。",
                    status_code=422,
                    field="page",
                )
            parquet = pq.ParquetFile(data_path)
            first_group = start // self.row_group_size
            last_group = (max(start, end - 1)) // self.row_group_size
            groups = list(range(first_group, last_group + 1)) if end > start else []
            if groups:
                table = parquet.read_row_groups(groups, columns=["row_json"])
                group_start = first_group * self.row_group_size
                local_start = start - group_start
                payloads = table.column("row_json").slice(
                    local_start, end - start
                ).to_pylist()
                rows = [json.loads(payload) for payload in payloads]
            else:
                rows = []
            page_count = (total + page_size - 1) // page_size if total else 0
            return {
                **manifest["summary"],
                "result_id": result_id,
                "rows": rows,
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "page_count": page_count,
                    "has_next": end < total,
                    "expires_at": datetime.fromtimestamp(
                        float(manifest["expires_at_epoch"]), tz=timezone.utc
                    ).isoformat(),
                },
            }


__all__ = ["EvaluationRunResultRepository"]
