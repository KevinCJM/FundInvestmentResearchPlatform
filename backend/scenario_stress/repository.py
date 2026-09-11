"""Versioned scenario definitions and append-only analytical run snapshots."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path
from typing import Any, Optional

from custom_indicators.errors import ConflictError, NotFoundError
from custom_indicators.repository import AtomicJsonStore, utc_now


class ScenarioDefinitionRepository:
    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self, include_archived: bool = False) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        current = [copy.deepcopy(entry["current"]) for entry in reversed(payload["items"])]
        if not include_archived:
            current = [item for item in current if not item.get("archived")]
        return current

    def get(self, definition_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current.get("id") != definition_id:
                continue
            for candidate in [current, *entry.get("history", [])]:
                if revision is None or int(candidate.get("revision", 0)) == int(revision):
                    return copy.deepcopy(candidate)
            raise NotFoundError("SCENARIO_DEFINITION_VERSION_NOT_FOUND", "未找到指定的情景定义版本。")
        raise NotFoundError("SCENARIO_DEFINITION_NOT_FOUND", "未找到指定的情景定义。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **copy.deepcopy(fields),
            "id": f"scenario-{uuid.uuid4().hex}",
            "revision": 1,
            "archived": False,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 2
            payload["items"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return copy.deepcopy(current)

    def update(self, definition_id: str, expected_revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current.get("id") != definition_id:
                    continue
                if current.get("archived"):
                    raise ConflictError("SCENARIO_DEFINITION_ARCHIVED", "情景定义已归档，不能继续修改。")
                if int(current.get("revision", 0)) != int(expected_revision):
                    raise ConflictError("REVISION_CONFLICT", "情景定义已被更新，请刷新后重试。", field="revision")
                entry.setdefault("history", []).append(copy.deepcopy(current))
                updated = {
                    **copy.deepcopy(fields),
                    "id": definition_id,
                    "revision": expected_revision + 1,
                    "archived": False,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                payload["schema_version"] = 2
                self.store.write_unlocked(payload)
                return copy.deepcopy(updated)
        raise NotFoundError("SCENARIO_DEFINITION_NOT_FOUND", "未找到指定的情景定义。")

    def archive(self, definition_id: str, expected_revision: int) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current.get("id") != definition_id:
                    continue
                if int(current.get("revision", 0)) != int(expected_revision):
                    raise ConflictError("REVISION_CONFLICT", "情景定义已被更新，请刷新后重试。", field="revision")
                if current.get("archived"):
                    return copy.deepcopy(current)
                entry.setdefault("history", []).append(copy.deepcopy(current))
                archived = {
                    **copy.deepcopy(current),
                    "revision": expected_revision + 1,
                    "archived": True,
                    "archived_at": utc_now(),
                    "updated_at": utc_now(),
                }
                entry["current"] = archived
                self.store.write_unlocked(payload)
                return copy.deepcopy(archived)
        raise NotFoundError("SCENARIO_DEFINITION_NOT_FOUND", "未找到指定的情景定义。")


class ScenarioRunRepository:
    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self, definition_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        items = payload["items"]
        if definition_id:
            items = [item for item in items if item.get("definition_id") == definition_id]
        return [copy.deepcopy(item) for item in reversed(items)]

    def get(self, run_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["items"]:
            if item.get("id") == run_id:
                return copy.deepcopy(item)
        raise NotFoundError("SCENARIO_RUN_NOT_FOUND", "未找到指定的情景模拟运行快照。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **copy.deepcopy(fields),
            "id": f"scenario-run-{uuid.uuid4().hex}",
            "created_at": utc_now(),
            "immutable": True,
            "publications": [],
            "application_bindings": [],
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 2
            payload["items"].append(snapshot)
            self.store.write_unlocked(payload)
        return copy.deepcopy(snapshot)

    def add_publications(self, run_id: str, publications: list[dict[str, Any]]) -> dict[str, Any]:
        """Append governance only; the analytical content hash remains unchanged."""

        with self.store.locked():
            payload = self.store.read_unlocked()
            for item in payload["items"]:
                if item.get("id") != run_id:
                    continue
                item.setdefault("publications", []).extend(copy.deepcopy(publications))
                item.setdefault("application_bindings", []).extend(
                    {
                        "usage": publication["usage"],
                        "name": f"{item.get('name', '情景模拟')} · {publication['usage']}",
                        "publication_id": publication["id"],
                        "run_id": run_id,
                        "definition_revision": item.get("definition_revision"),
                        "run_content_hash": item.get("content_hash"),
                        "status": "active",
                    }
                    for publication in publications
                )
                self.store.write_unlocked(payload)
                return copy.deepcopy(item)
        raise NotFoundError("SCENARIO_RUN_NOT_FOUND", "未找到指定的情景模拟运行快照。")
