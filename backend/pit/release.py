"""Immutable data releases: which vintage of the data answered the question.

An `as_of` alone does not make research reproducible. Re-run the same `as_of`
next month and the provider may have restated the rows underneath you. A release
pins the files themselves, so "same as_of, same release" is a real guarantee and
"same as_of, different release" is a visible difference rather than a mystery.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # pragma: no cover - platform specific
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

from .audit import audit_all

try:
    from backend.market_data import resolve_market_data_file
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_market_data_file


RELEASE_STORE = "data_releases.json"
SCHEMA_VERSION = 1


class DataReleaseError(ValueError):
    """Raised when a release cannot be created or found."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class DataReleaseStore:
    """Same atomic-write discipline as the product-pool store: temp file,
    fsync, rename, plus an advisory lock so two封版 cannot interleave."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @contextmanager
    def locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            with self.lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "releases": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReleaseError("数据版本存储无法读取，请检查 data/data_releases.json。") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("releases"), list):
            raise DataReleaseError("数据版本存储格式无效。")
        return payload

    def write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


def _table_fingerprint(item: dict[str, Any], size: Optional[int]) -> str:
    """A metadata digest, deliberately not a row-level content hash.

    Hashing 1.5M rows on every封版 buys almost nothing: any parquet rewrite moves
    the byte count, the row count or the date range. The field is named
    `fingerprint` rather than `content_hash` so nobody reads more into it than
    it can carry.
    """

    material = json.dumps(
        {
            "dataset_id": item["dataset_id"],
            "file": item["file"],
            "rows": item["rows"],
            "bytes": size,
            "event_range": item["event_range"],
            "availability_range": item["availability_range"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


class DataReleaseRepository:
    def __init__(self, path: Path) -> None:
        self.store = DataReleaseStore(path)

    @staticmethod
    def _order_key(item: dict[str, Any]) -> tuple[int, str]:
        """Newest-first ordering that does not depend on the clock.

        Two releases sealed inside the same wall-clock tick used to sort
        arbitrarily, which then chained `parent_release_id` to the wrong parent.
        The sequence number is monotonic by construction, so it survives both
        coarse timestamps and an NTP step backwards.
        """

        return (int(item.get("sequence") or 0), str(item.get("created_at") or ""))

    def list_releases(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        releases = list(payload["releases"])
        releases.sort(key=self._order_key, reverse=True)
        return releases

    def get(self, release_id: str) -> dict[str, Any]:
        wanted = str(release_id or "").strip()
        for release in self.list_releases():
            if release.get("id") == wanted:
                return release
        raise DataReleaseError(f"未找到数据版本 {wanted}。")

    def latest(self) -> Optional[dict[str, Any]]:
        releases = self.list_releases()
        return releases[0] if releases else None

    def create(self, data_dir: Path, name: str, note: str = "") -> dict[str, Any]:
        label = str(name or "").strip()
        if not label:
            raise DataReleaseError("数据版本名称不能为空。")
        if len(label) > 120:
            raise DataReleaseError("数据版本名称不能超过 120 个字符。")

        audit = audit_all(data_dir)
        tables: list[dict[str, Any]] = []
        for item in audit["datasets"]:
            if not item["present"]:
                continue
            # The active Tushare manifest can point the real files at a snapshot
            # subdirectory; sizing `data_dir / file` there silently yields None
            # and drops the byte count out of the fingerprint.
            try:
                path = resolve_market_data_file(item["file"], data_dir)
            except Exception:  # noqa: BLE001 - a broken manifest must not block封版
                path = data_dir / item["file"]
            size = path.stat().st_size if path.exists() else None
            tables.append(
                {
                    "dataset_id": item["dataset_id"],
                    "label": item["label"],
                    "file": item["file"],
                    "rows": item["rows"],
                    "bytes": size,
                    "grade": item["grade"],
                    "event_range": item["event_range"],
                    "availability_range": item["availability_range"],
                    "available_through": item.get("available_through"),
                    "fingerprint": _table_fingerprint(item, size),
                }
            )
        if not tables:
            raise DataReleaseError("没有任何可封版的数据集，请先完成数据下载。")

        with self.store.locked():
            payload = self.store.read_unlocked()
            existing = sorted(payload["releases"], key=self._order_key, reverse=True)
            parent = existing[0]["id"] if existing else None
            next_sequence = (int(existing[0].get("sequence") or 0) + 1) if existing else 1
            release = {
                "id": f"release-{uuid.uuid4().hex[:12]}",
                "sequence": next_sequence,
                "name": label,
                "note": str(note or "").strip()[:500],
                "created_at": _utc_now(),
                "parent_release_id": parent,
                "immutable": True,
                "tables": tables,
                "summary": audit["summary"],
                "release_fingerprint": hashlib.sha256(
                    "|".join(sorted(table["fingerprint"] for table in tables)).encode("utf-8")
                ).hexdigest(),
            }
            payload["releases"].append(release)
            payload["schema_version"] = SCHEMA_VERSION
            self.store.write_unlocked(payload)
        return release


__all__ = ["DataReleaseError", "DataReleaseRepository", "DataReleaseStore", "RELEASE_STORE"]
