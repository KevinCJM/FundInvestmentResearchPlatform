"""Atomic persistence for mutable product pools and immutable snapshots."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:  # pragma: no cover - Windows fallback for local development
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

from .domain import content_hash
from .errors import (
    ProductPoolConflictError,
    ProductPoolNotFoundError,
    ProductPoolValidationError,
)

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AtomicProductPoolStore:
    """One-file transaction boundary for pool drafts, versions, and universes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(path.suffix + ".lock")
        self._thread_lock = threading.RLock()

    @staticmethod
    def default_payload() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "pools": [],
            "versions": [],
            "universe_snapshots": [],
        }

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
            return self.default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProductPoolValidationError(
                "PRODUCT_POOL_STORAGE_CORRUPT",
                "产品池存储文件无法读取。",
            ) from exc
        if not isinstance(payload, dict):
            raise ProductPoolValidationError(
                "PRODUCT_POOL_STORAGE_CORRUPT",
                "产品池存储格式无效。",
            )
        for key in ("pools", "versions", "universe_snapshots"):
            if not isinstance(payload.get(key), list):
                raise ProductPoolValidationError(
                    "PRODUCT_POOL_STORAGE_CORRUPT",
                    f"产品池存储字段 {key} 格式无效。",
                )
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    def write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
            try:
                directory_fd = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            except OSError:  # pragma: no cover - filesystem-specific durability
                pass
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)


class ProductPoolRepository:
    """Versioned draft repository with atomic publication."""

    def __init__(self, path: Path) -> None:
        self.store = AtomicProductPoolStore(path)

    @staticmethod
    def _pool_entry(payload: dict[str, Any], pool_id: str) -> dict[str, Any]:
        for entry in payload["pools"]:
            current = entry.get("current") if isinstance(entry, dict) else None
            if isinstance(current, dict) and current.get("id") == pool_id:
                return entry
        raise ProductPoolNotFoundError("PRODUCT_POOL_NOT_FOUND", "未找到指定产品池。")

    @staticmethod
    def _version(payload: dict[str, Any], version_id: str) -> dict[str, Any]:
        for item in payload["versions"]:
            if isinstance(item, dict) and item.get("id") == version_id:
                return item
        raise ProductPoolNotFoundError("PRODUCT_POOL_VERSION_NOT_FOUND", "未找到指定产品池版本。")

    def list_pools(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        items = [copy.deepcopy(entry["current"]) for entry in payload["pools"]]
        return sorted(items, key=lambda item: str(item.get("updated_at") or ""), reverse=True)

    def get_pool(self, pool_id: str, revision: int | None = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        entry = self._pool_entry(payload, pool_id)
        current = entry["current"]
        if revision is None or int(current.get("revision", 0)) == revision:
            return copy.deepcopy(current)
        for historical in entry.get("history", []):
            if int(historical.get("revision", 0)) == revision:
                return copy.deepcopy(historical)
        raise ProductPoolNotFoundError(
            "PRODUCT_POOL_REVISION_NOT_FOUND",
            "未找到指定产品池修订版本。",
        )

    def create_pool(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **copy.deepcopy(fields),
            "id": f"pool-{uuid.uuid4().hex}",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
            "current_version_id": None,
            "published_at": None,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["pools"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return copy.deepcopy(current)

    def update_pool(
        self,
        pool_id: str,
        expected_revision: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            entry = self._pool_entry(payload, pool_id)
            current = entry["current"]
            if int(current.get("revision", 0)) != expected_revision:
                raise ProductPoolConflictError(
                    "PRODUCT_POOL_REVISION_CONFLICT",
                    "产品池已被其他操作更新，请刷新后重试。",
                    field="revision",
                )
            entry.setdefault("history", []).append(copy.deepcopy(current))
            updated = {
                **copy.deepcopy(current),
                **copy.deepcopy(fields),
                "id": pool_id,
                "revision": expected_revision + 1,
                "created_at": current["created_at"],
                "updated_at": utc_now(),
            }
            entry["current"] = updated
            self.store.write_unlocked(payload)
        return copy.deepcopy(updated)

    def archive_pool(self, pool_id: str, expected_revision: int) -> dict[str, Any]:
        return self.update_pool(pool_id, expected_revision, {"state": "archived"})

    def publish_pool(
        self,
        pool_id: str,
        expected_revision: int,
        version_fields: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Publish a version and advance the mutable pool in one file transaction."""

        with self.store.locked():
            payload = self.store.read_unlocked()
            entry = self._pool_entry(payload, pool_id)
            current = entry["current"]
            if int(current.get("revision", 0)) != expected_revision:
                raise ProductPoolConflictError(
                    "PRODUCT_POOL_REVISION_CONFLICT",
                    "产品池已被其他操作更新，请刷新后重试。",
                    field="revision",
                )
            sequence = 1 + max(
                (
                    int(item.get("version", 0))
                    for item in payload["versions"]
                    if item.get("pool_id") == pool_id
                ),
                default=0,
            )
            now = utc_now()
            version = {
                **copy.deepcopy(version_fields),
                "id": f"pool-version-{uuid.uuid4().hex}",
                "pool_id": pool_id,
                "version": sequence,
                "pool_revision": expected_revision,
                "created_at": now,
                "immutable": True,
            }
            payload["versions"].append(version)
            entry.setdefault("history", []).append(copy.deepcopy(current))
            updated = {
                **copy.deepcopy(current),
                "state": "active",
                "revision": expected_revision + 1,
                "updated_at": now,
                "published_at": now,
                "current_version_id": version["id"],
            }
            entry["current"] = updated
            self.store.write_unlocked(payload)
        return copy.deepcopy(updated), copy.deepcopy(version)

    def list_versions(
        self,
        *,
        pool_id: str | None = None,
        active_on: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        items = []
        for item in payload["versions"]:
            if pool_id and item.get("pool_id") != pool_id:
                continue
            if active_on:
                starts = str(item.get("effective_from") or "")
                ends = str(item.get("effective_to") or "")
                if not starts or starts > active_on or (ends and ends < active_on):
                    continue
            items.append(copy.deepcopy(item))
        return sorted(
            items,
            key=lambda item: (str(item.get("pool_id")), int(item.get("version", 0))),
            reverse=True,
        )

    def get_version(self, version_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return copy.deepcopy(self._version(payload, version_id))

    def create_universe_snapshot(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **copy.deepcopy(fields),
            "id": f"universe-{uuid.uuid4().hex}",
            "created_at": utc_now(),
            "immutable": True,
        }
        # The snapshot claims to be immutable; without a digest that claim is
        # unverifiable, and `membership.py` has always read this field back.
        # Hashed after id/created_at so the digest covers the identity too.
        snapshot["content_hash"] = content_hash(snapshot)
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["universe_snapshots"].append(snapshot)
            self.store.write_unlocked(payload)
        return copy.deepcopy(snapshot)

    def get_universe_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["universe_snapshots"]:
            if item.get("id") == snapshot_id:
                return copy.deepcopy(item)
        raise ProductPoolNotFoundError(
            "INVESTABLE_UNIVERSE_NOT_FOUND",
            "未找到指定可投资域快照。",
        )


class InvestableUniverseRepository:
    """Dedicated immutable-universe repository used by downstream research.

    The storage envelope intentionally matches ``AtomicProductPoolStore`` so
    existing local files and the legacy ProductPoolRepository snapshot methods
    remain readable without a migration.
    """

    def __init__(self, path: Path) -> None:
        self.store = AtomicProductPoolStore(path)

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **copy.deepcopy(fields),
            "id": str(fields.get("id") or f"universe-{uuid.uuid4().hex}"),
            "created_at": str(fields.get("created_at") or utc_now()),
            "immutable": True,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            if any(item.get("id") == snapshot["id"] for item in payload["universe_snapshots"]):
                raise ProductPoolConflictError(
                    "INVESTABLE_UNIVERSE_ALREADY_EXISTS",
                    "可投资域快照已存在。",
                )
            payload["universe_snapshots"].append(snapshot)
            self.store.write_unlocked(payload)
        return copy.deepcopy(snapshot)

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return [copy.deepcopy(item) for item in payload["universe_snapshots"]]

    def get(self, snapshot_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["universe_snapshots"]:
            if item.get("id") == snapshot_id:
                return copy.deepcopy(item)
        raise ProductPoolNotFoundError(
            "INVESTABLE_UNIVERSE_NOT_FOUND",
            "未找到指定可投资域快照。",
        )


__all__ = [
    "AtomicProductPoolStore",
    "InvestableUniverseRepository",
    "ProductPoolRepository",
    "utc_now",
]
