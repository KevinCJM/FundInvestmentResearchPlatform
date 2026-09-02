"""Atomic JSON repositories for workspace-shared indicators and plans."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

try:  # Linux/macOS production and development environments.
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback remains process-safe
    fcntl = None

from .errors import ConflictError, IndicatorDomainError, NotFoundError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AtomicJsonStore:
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
            return {"schema_version": 1, "items": []}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndicatorDomainError(
                "STORAGE_CORRUPT",
                "工作区指标数据无法读取，请检查数据文件。",
                status_code=500,
            ) from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise IndicatorDomainError(
                "STORAGE_CORRUPT",
                "工作区指标数据格式无效。",
                status_code=500,
            )
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
                json.dump(payload, handle, ensure_ascii=False, indent=2)
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


class IndicatorRepository:
    def __init__(self, path: Path, built_ins: list[dict[str, Any]]) -> None:
        self.store = AtomicJsonStore(path)
        self.built_ins = {item["id"]: dict(item) for item in built_ins}

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        custom = [dict(entry["current"]) for entry in payload["items"]]
        return [dict(item) for item in self.built_ins.values()] + custom

    def get(self, indicator_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        built_in = self.built_ins.get(indicator_id)
        if built_in is not None:
            if revision not in (None, int(built_in["revision"])):
                raise NotFoundError("INDICATOR_VERSION_NOT_FOUND", "未找到指定的内置指标版本。")
            return dict(built_in)
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current["id"] != indicator_id:
                continue
            if revision is None or int(current["revision"]) == revision:
                return dict(current)
            for historical in entry.get("history", []):
                if int(historical["revision"]) == revision:
                    return dict(historical)
            raise NotFoundError("INDICATOR_VERSION_NOT_FOUND", "未找到指定的指标版本。")
        raise NotFoundError("INDICATOR_NOT_FOUND", "未找到指定指标。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **fields,
            "id": f"indicator-{uuid.uuid4().hex}",
            "revision": 1,
            "source": "custom",
            "read_only": False,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["items"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return dict(current)

    def update(self, indicator_id: str, expected_revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        if indicator_id in self.built_ins:
            raise ConflictError("BUILT_IN_READ_ONLY", "内置指标为只读，请先复制为自定义指标。")
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current["id"] != indicator_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "指标已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                entry.setdefault("history", []).append(dict(current))
                updated = {
                    **fields,
                    "id": indicator_id,
                    "revision": expected_revision + 1,
                    "source": "custom",
                    "read_only": False,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                self.store.write_unlocked(payload)
                return dict(updated)
        raise NotFoundError("INDICATOR_NOT_FOUND", "未找到指定指标。")

    def delete(self, indicator_id: str, expected_revision: int) -> None:
        if indicator_id in self.built_ins:
            raise ConflictError("BUILT_IN_READ_ONLY", "内置指标不能删除。")
        with self.store.locked():
            payload = self.store.read_unlocked()
            for index, entry in enumerate(payload["items"]):
                current = entry["current"]
                if current["id"] != indicator_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "指标已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                payload["items"].pop(index)
                self.store.write_unlocked(payload)
                return
            raise NotFoundError("INDICATOR_NOT_FOUND", "未找到指定指标。")


class PlanRepository:
    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return [dict(entry["current"]) for entry in payload["items"]]

    def get(self, plan_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current["id"] != plan_id:
                continue
            if revision is None or int(current["revision"]) == revision:
                return dict(current)
            for historical in entry.get("history", []):
                if int(historical["revision"]) == revision:
                    return dict(historical)
            raise NotFoundError("PLAN_VERSION_NOT_FOUND", "未找到指定评价方案版本。")
        raise NotFoundError("PLAN_NOT_FOUND", "未找到指定评价方案。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **fields,
            "id": f"plan-{uuid.uuid4().hex}",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["items"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return dict(current)

    def update(self, plan_id: str, expected_revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current["id"] != plan_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "评价方案已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                entry.setdefault("history", []).append(dict(current))
                updated = {
                    **fields,
                    "id": plan_id,
                    "revision": expected_revision + 1,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                self.store.write_unlocked(payload)
                return dict(updated)
        raise NotFoundError("PLAN_NOT_FOUND", "未找到指定评价方案。")

    def delete(self, plan_id: str, expected_revision: int) -> None:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for index, entry in enumerate(payload["items"]):
                current = entry["current"]
                if current["id"] != plan_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "评价方案已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                payload["items"].pop(index)
                self.store.write_unlocked(payload)
                return
            raise NotFoundError("PLAN_NOT_FOUND", "未找到指定评价方案。")

    def references_indicator(self, indicator_id: str) -> bool:
        return any(
            item.get("indicator_id") == indicator_id
            for plan in self.list()
            for item in plan.get("indicators", [])
        )

    def archive_and_reset(self, archive_directory: Path, marker: str) -> dict[str, Any]:
        """Atomically archive active plans once, then create an empty store."""

        with self.store.locked():
            payload = self.store.read_unlocked()
            migration = payload.get("migration") or {}
            if migration.get("marker") == marker:
                return {**dict(migration), "applied": False}
            archive_path: Path | None = None
            if payload.get("items"):
                archive_directory.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                archive_path = archive_directory / f"evaluation_plans.pre-{marker}.{timestamp}.json"
                archive_store = AtomicJsonStore(archive_path)
                archive_store.write_unlocked(
                    {
                        **payload,
                        "archived_at": utc_now(),
                        "archive_reason": marker,
                    }
                )
            migration = {
                "marker": marker,
                "migrated_at": utc_now(),
                "archived_file": str(archive_path) if archive_path else None,
                "applied": True,
            }
            self.store.write_unlocked(
                {
                    "schema_version": 2,
                    "items": [],
                    "migration": migration,
                }
            )
            return dict(migration)
