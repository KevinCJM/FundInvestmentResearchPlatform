"""Versioned definitions and append-only historical regime runs."""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Optional

from custom_indicators.errors import ConflictError, NotFoundError
from custom_indicators.repository import AtomicJsonStore, utc_now


class RegimeDefinitionRepository:
    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        return [copy.deepcopy(entry["current"]) for entry in reversed(payload["items"])]

    def get(self, definition_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current.get("id") != definition_id:
                continue
            candidates = [current, *entry.get("history", [])]
            if revision is None:
                return copy.deepcopy(current)
            for candidate in candidates:
                if int(candidate.get("revision", 0)) == int(revision):
                    return copy.deepcopy(candidate)
            raise NotFoundError("REGIME_DEFINITION_VERSION_NOT_FOUND", "未找到指定的情景定义版本。")
        raise NotFoundError("REGIME_DEFINITION_NOT_FOUND", "未找到指定的情景定义。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        current = {
            **copy.deepcopy(fields),
            "id": f"regime-{uuid.uuid4().hex}",
            "revision": 1,
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
                if int(current.get("revision", 0)) != int(expected_revision):
                    raise ConflictError("REVISION_CONFLICT", "情景定义已被更新，请刷新后重试。", field="revision")
                entry.setdefault("history", []).append(copy.deepcopy(current))
                updated = {
                    **copy.deepcopy(fields),
                    "id": definition_id,
                    "revision": expected_revision + 1,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                payload["schema_version"] = 2
                self.store.write_unlocked(payload)
                return copy.deepcopy(updated)
        raise NotFoundError("REGIME_DEFINITION_NOT_FOUND", "未找到指定的情景定义。")


class RegimeRunRepository:
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
        raise NotFoundError("REGIME_RUN_NOT_FOUND", "未找到指定的历史情景运行快照。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **copy.deepcopy(fields),
            "id": f"regime-run-{uuid.uuid4().hex}",
            "created_at": utc_now(),
            "immutable": True,
            "publications": [],
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 2
            payload["items"].append(snapshot)
            self.store.write_unlocked(payload)
        return copy.deepcopy(snapshot)

    def add_publications(self, run_id: str, publications: list[dict[str, Any]]) -> dict[str, Any]:
        """Append governance records without changing the immutable analytical snapshot."""

        with self.store.locked():
            payload = self.store.read_unlocked()
            for item in payload["items"]:
                if item.get("id") != run_id:
                    continue
                existing = item.setdefault("publications", [])
                existing.extend(copy.deepcopy(publications))
                bindings = item.setdefault("application_bindings", [])
                bindings.extend(
                    {
                        "usage": publication["usage"],
                        "name": f"{item.get('name', '历史情景')} · {publication['usage']}",
                        "publication_id": publication["id"],
                        "run_id": run_id,
                        "revision": item.get("definition_revision"),
                        "status": "active",
                    }
                    for publication in publications
                )
                self.store.write_unlocked(payload)
                return copy.deepcopy(item)
        raise NotFoundError("REGIME_RUN_NOT_FOUND", "未找到指定的历史情景运行快照。")


class RegimeGraphAssetRepository:
    """Versioned user templates and reusable graph fragments."""

    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        items = [entry["current"] for entry in payload["items"]]
        if kind is not None:
            items = [item for item in items if item.get("kind") == kind]
        return [copy.deepcopy(item) for item in reversed(items)]

    def get(self, asset_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for entry in payload["items"]:
            current = entry["current"]
            if current.get("id") != asset_id:
                continue
            candidates = [current, *entry.get("history", [])]
            if revision is None:
                return copy.deepcopy(current)
            for candidate in candidates:
                if int(candidate.get("revision", 0)) == int(revision):
                    return copy.deepcopy(candidate)
            raise NotFoundError(
                "REGIME_GRAPH_ASSET_VERSION_NOT_FOUND",
                "未找到指定的用户模板或子图版本。",
            )
        raise NotFoundError(
            "REGIME_GRAPH_ASSET_NOT_FOUND",
            "未找到指定的用户模板或子图。",
        )

    def create(self, kind: str, fields: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        prefix = "regime-user-template" if kind == "template" else "regime-subgraph"
        current = {
            **copy.deepcopy(fields),
            "id": f"{prefix}-{uuid.uuid4().hex}",
            "kind": kind,
            "revision": 1,
            "created_at": now,
            "updated_at": now,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 1
            payload["items"].append({"current": current, "history": []})
            self.store.write_unlocked(payload)
        return copy.deepcopy(current)

    def update(
        self,
        asset_id: str,
        expected_revision: int,
        fields: dict[str, Any],
    ) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
            for entry in payload["items"]:
                current = entry["current"]
                if current.get("id") != asset_id:
                    continue
                if int(current.get("revision", 0)) != int(expected_revision):
                    raise ConflictError(
                        "REVISION_CONFLICT",
                        "用户模板或子图已被更新，请刷新后重试。",
                        field="revision",
                    )
                entry.setdefault("history", []).append(copy.deepcopy(current))
                updated = {
                    **copy.deepcopy(fields),
                    "id": asset_id,
                    "kind": current["kind"],
                    "revision": expected_revision + 1,
                    "created_at": current["created_at"],
                    "updated_at": utc_now(),
                }
                entry["current"] = updated
                self.store.write_unlocked(payload)
                return copy.deepcopy(updated)
        raise NotFoundError(
            "REGIME_GRAPH_ASSET_NOT_FOUND",
            "未找到指定的用户模板或子图。",
        )


class RegimeExperimentRepository:
    """Append-only batch experiment summaries."""

    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    def list(self, definition_id: Optional[str] = None) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        items = payload["items"]
        if definition_id:
            items = [item for item in items if item.get("definition_id") == definition_id]
        return [copy.deepcopy(item) for item in reversed(items)]

    def get(self, experiment_id: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["items"]:
            if item.get("id") == experiment_id:
                return copy.deepcopy(item)
        raise NotFoundError("REGIME_EXPERIMENT_NOT_FOUND", "未找到指定的批量实验。")

    def create(self, fields: dict[str, Any]) -> dict[str, Any]:
        snapshot = {
            **copy.deepcopy(fields),
            "id": f"regime-experiment-{uuid.uuid4().hex}",
            "created_at": utc_now(),
            "immutable": True,
        }
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 1
            payload["items"].append(snapshot)
            self.store.write_unlocked(payload)
        return copy.deepcopy(snapshot)


class RegimePlanManifestRepository:
    """Persistent, non-authorizing descriptors for prepared v2 graph plans.

    Execution tokens deliberately remain process-local.  The manifest is only
    an auditable recipe for startup/explicit prewarm and can never authorize a
    run by itself.
    """

    def __init__(self, path: Path) -> None:
        self.store = AtomicJsonStore(path)

    @staticmethod
    def _assert_secret_free(value: Any, path: str = "manifest") -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).lower().replace("-", "_")
                if key == "token" or key.endswith("_token"):
                    raise ConflictError(
                        "REGIME_PLAN_MANIFEST_SECRET_REJECTED",
                        "执行计划持久化清单禁止保存运行令牌。",
                        field=f"{path}.{raw_key}",
                    )
                RegimePlanManifestRepository._assert_secret_free(
                    child,
                    f"{path}.{raw_key}",
                )
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                RegimePlanManifestRepository._assert_secret_free(
                    child,
                    f"{path}.{index}",
                )
            return
        if isinstance(value, str) and value.startswith("rg2-"):
            raise ConflictError(
                "REGIME_PLAN_MANIFEST_SECRET_REJECTED",
                "执行计划持久化清单禁止保存运行令牌。",
                field=path,
            )

    @staticmethod
    def _recipe_hash(manifest: dict[str, Any]) -> str:
        recipe = copy.deepcopy(manifest)
        for key in (
            "manifest_content_hash",
            "definition_bindings",
            "first_prepared_at",
            "last_prepared_at",
            "prepare_count",
        ):
            recipe.pop(key, None)
        encoded = json.dumps(
            recipe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def _validate_manifest(cls, manifest: dict[str, Any]) -> None:
        cls._assert_secret_free(manifest)
        if (
            manifest.get("schema_version") != "1.0"
            or manifest.get("manifest_kind") != "regime_graph_v2_execution_plan"
        ):
            raise ConflictError(
                "REGIME_PLAN_MANIFEST_SCHEMA_INVALID",
                "执行计划持久化清单的类型或版本无效。",
            )
        preparation_hash = str(manifest.get("preparation_hash") or "")
        graph_hash = str(manifest.get("graph_hash") or "")
        reported_hash = str(manifest.get("manifest_content_hash") or "")
        if not all(
            len(value) == 64 and all(character in "0123456789abcdef" for character in value)
            for value in (preparation_hash, graph_hash, reported_hash)
        ):
            raise ConflictError(
                "REGIME_PLAN_MANIFEST_HASH_INVALID",
                "执行计划持久化清单缺少有效的内容指纹。",
            )
        if reported_hash != cls._recipe_hash(manifest):
            raise ConflictError(
                "REGIME_PLAN_MANIFEST_INTEGRITY_MISMATCH",
                "执行计划持久化清单完整性校验失败，拒绝预热或覆盖。",
            )

    def list(self) -> list[dict[str, Any]]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["items"]:
            self._validate_manifest(item)
        return [copy.deepcopy(item) for item in reversed(payload["items"])]

    def get(self, preparation_hash: str) -> dict[str, Any]:
        with self.store.locked():
            payload = self.store.read_unlocked()
        for item in payload["items"]:
            if item.get("preparation_hash") == preparation_hash:
                self._validate_manifest(item)
                return copy.deepcopy(item)
        raise NotFoundError(
            "REGIME_PLAN_MANIFEST_NOT_FOUND",
            "未找到指定的历史情景执行计划清单。",
        )

    def upsert(self, fields: dict[str, Any]) -> dict[str, Any]:
        manifest = copy.deepcopy(fields)
        preparation_hash = str(manifest.get("preparation_hash") or "")
        if not preparation_hash:
            raise ConflictError(
                "REGIME_PLAN_MANIFEST_HASH_REQUIRED",
                "执行计划持久化清单缺少 preparation_hash。",
                field="preparation_hash",
            )
        self._assert_secret_free(manifest)
        now = utc_now()
        with self.store.locked():
            payload = self.store.read_unlocked()
            payload["schema_version"] = 1
            for current in payload["items"]:
                self._validate_manifest(current)
            for index, current in enumerate(payload["items"]):
                if current.get("preparation_hash") != preparation_hash:
                    continue
                bindings = {
                    (
                        str(item.get("definition_id") or ""),
                        int(item.get("revision") or 0),
                    ): copy.deepcopy(item)
                    for item in current.get("definition_bindings", [])
                    if isinstance(item, dict)
                }
                for item in manifest.get("definition_bindings", []):
                    if not isinstance(item, dict):
                        continue
                    key = (
                        str(item.get("definition_id") or ""),
                        int(item.get("revision") or 0),
                    )
                    if key[0]:
                        bindings[key] = copy.deepcopy(item)
                updated = {
                    **manifest,
                    "definition_bindings": list(bindings.values()),
                    "first_prepared_at": current.get("first_prepared_at") or now,
                    "last_prepared_at": now,
                    "prepare_count": int(current.get("prepare_count") or 0) + 1,
                }
                self._validate_manifest(updated)
                payload["items"][index] = updated
                self.store.write_unlocked(payload)
                return copy.deepcopy(updated)
            created = {
                **manifest,
                "first_prepared_at": now,
                "last_prepared_at": now,
                "prepare_count": 1,
            }
            self._validate_manifest(created)
            payload["items"].append(created)
            self.store.write_unlocked(payload)
        return copy.deepcopy(created)
