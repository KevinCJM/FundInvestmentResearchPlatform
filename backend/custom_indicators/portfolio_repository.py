"""Versioned research-target and immutable portfolio-run repositories."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Optional

from .errors import ConflictError, NotFoundError
from .repository import AtomicJsonStore, utc_now


class ResearchTargetRepository:
    """Store current and historical target revisions with optimistic locking."""

    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return [dict(entry["current"]) for entry in payload["items"]]

    def get(self, target_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current["id"] != target_id:
                continue
            if revision is None or int(current["revision"]) == revision:
                return dict(current)
            for historical in entry.get("history", []):
                if int(historical["revision"]) == revision:
                    return dict(historical)
            raise NotFoundError("RESEARCH_TARGET_VERSION_NOT_FOUND", "未找到指定的研究组合版本。")
        raise NotFoundError("RESEARCH_TARGET_NOT_FOUND", "未找到指定研究组合。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **fields,
            "id": f"target-{uuid.uuid4().hex}",
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 2
            payload["items"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return dict(current)

    def update(self, target_id: str, expected_revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current["id"] != target_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "研究组合已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                entry.setdefault("history", []).append(dict(current))
                updated = {
                    **fields,
                    "id": target_id,
                    "revision": expected_revision + 1,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                payload["schema_version"] = 2
                self.store.write_unlocked(payload)
                return dict(updated)
        raise NotFoundError("RESEARCH_TARGET_NOT_FOUND", "未找到指定研究组合。")

    def delete(self, target_id: str, expected_revision: int) -> None:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for index, entry in enumerate(payload["items"]):
                current = entry["current"]
                if current["id"] != target_id:
                    continue
                if int(current["revision"]) != expected_revision:
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "研究组合已被其他操作更新，请刷新后重试。",
                        field="revision",
                    )
                payload["items"].pop(index)
                self.store.write_unlocked(payload)
                return
        raise NotFoundError("RESEARCH_TARGET_NOT_FOUND", "未找到指定研究组合。")


class PortfolioRunRepository:
    """Append-only store for immutable, reproducible portfolio snapshots."""

    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return [dict(item) for item in reversed(payload["items"])]

    def get(self, run_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["items"]:
            if item.get("id") == run_id:
                return dict(item)
        raise NotFoundError("PORTFOLIO_RUN_NOT_FOUND", "未找到指定组合运行快照。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **fields,
            "id": f"run-{uuid.uuid4().hex}",
            "created_at": utc_now(),
            "immutable": True,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 2
            payload["items"].append(snapshot)
            self.store.write_unlocked(payload)
        return dict(snapshot)

    def references_target(self, target_id: str) -> bool:
        return any(item.get("target_id") == target_id for item in self.list())
