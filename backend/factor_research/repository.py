"""Atomic versioned definitions and immutable runs, using workspace file locks."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError


def clean(value: Any) -> Any:
    """Serialization boundary only; missing numerical results remain null."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def checked_id(value: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,119}", value):
        raise ValidationError("INVALID_FACTOR_ID", "因子研究对象 ID 无效。")
    return value


class VersionStore:
    def __init__(self, path: Path, prefix: str, builtins=()):
        self.store = AtomicJsonStore(path)
        self.prefix = prefix
        self.builtins = {x["id"]: copy.deepcopy(x) for x in builtins}

    def list(self):
        with self.store.locked():
            items = self.store.read_unlocked()["items"]
        return copy.deepcopy(list(self.builtins.values()) + [x["current"] for x in items])

    def get(self, object_id: str, revision: int | None = None):
        if object_id in self.builtins:
            item = self.builtins[object_id]
            if revision in (None, item["revision"]):
                return copy.deepcopy(item)
        with self.store.locked():
            items = self.store.read_unlocked()["items"]
        for entry in items:
            if entry["current"]["id"] == object_id:
                for item in [entry["current"], *entry["history"]]:
                    if revision in (None, item["revision"]):
                        return copy.deepcopy(item)
        raise NotFoundError("FACTOR_OBJECT_NOT_FOUND", "未找到研究对象或指定版本。")

    def save(self, fields: dict, object_id: str | None = None, revision: int | None = None):
        if object_id in self.builtins:
            raise ConflictError("BUILT_IN_READ_ONLY", "内置模板只读，请复制后编辑。")
        now = utc_now()
        with self.store.locked():
            payload = self.store.read_unlocked()
            if object_id is None:
                current = {**clean(fields), "id": self.prefix + uuid.uuid4().hex,
                           "revision": 1, "created_at": now, "updated_at": now, "read_only": False}
                payload["items"].append({"current": current, "history": []})
            else:
                entry = next((x for x in payload["items"] if x["current"]["id"] == object_id), None)
                if entry is None:
                    raise NotFoundError("FACTOR_OBJECT_NOT_FOUND", "研究对象不存在。")
                if entry["current"]["revision"] != revision:
                    raise ConflictError("REVISION_CONFLICT", "对象已更新，请刷新后重试。")
                entry["history"].append(copy.deepcopy(entry["current"]))
                current = {**clean(fields), "id": object_id, "revision": revision + 1,
                           "created_at": entry["current"]["created_at"], "updated_at": now, "read_only": False}
                entry["current"] = current
            self.store.write_unlocked(payload)
        return copy.deepcopy(current)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root
        self.index = AtomicJsonStore(root / "index.json")

    def list(self, kind: str | None = None):
        with self.index.locked():
            items = self.index.read_unlocked()["items"]
        return [x for x in reversed(items) if kind is None or x["kind"] == kind]

    def get(self, object_id: str):
        store = AtomicJsonStore(self.root / (checked_id(object_id) + ".json"))
        with store.locked():
            if not store.path.exists():
                raise NotFoundError("FACTOR_ARTIFACT_NOT_FOUND", "研究运行或发布不存在。")
            return store.read_unlocked()["items"][0]

    def load_arrays(self, object_id: str) -> dict[str, np.ndarray]:
        """Read the immutable numeric inputs, never today's replacement snapshot."""
        item = self.get(object_id)
        path = self.root / (checked_id(object_id) + ".npz")
        if not path.is_file() or not item.get("input_checksum"):
            raise ValidationError("FACTOR_INPUT_SNAPSHOT_MISSING", "该运行缺少冻结输入，请重新运行特征研究后构建收益。")
        try:
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: np.ascontiguousarray(archive[key]) for key in archive.files}
            digest = hashlib.sha256()
            for key in sorted(arrays):
                array = arrays[key]
                if array.dtype not in (np.dtype("float64"), np.dtype("int64")):
                    raise ValueError("unsupported dtype")
                digest.update(key.encode())
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(memoryview(array).cast("B"))
            if digest.hexdigest() != item["input_checksum"]:
                raise ValueError("checksum mismatch")
        except (ValueError, OSError, EOFError) as exc:
            raise ValidationError("FACTOR_INPUT_SNAPSHOT_INVALID", "冻结输入损坏或校验和不一致，已停止构建。") from exc
        return arrays

    def save(self, kind: str, fields: dict, arrays: dict[str, np.ndarray] | None = None):
        object_id = "factor-" + kind + "-" + uuid.uuid4().hex
        item = {**clean(fields), "id": object_id, "kind": kind, "created_at": utc_now()}
        self.root.mkdir(parents=True, exist_ok=True)
        if arrays is not None:
            digest = hashlib.sha256()
            for key in sorted(arrays):
                array = np.ascontiguousarray(arrays[key])
                digest.update(key.encode())
                digest.update(str((array.dtype.str, array.shape)).encode())
                digest.update(array.tobytes())
            item["input_checksum"] = digest.hexdigest()
            fd, temporary = tempfile.mkstemp(dir=self.root, suffix=".npz")
            try:
                with os.fdopen(fd, "wb") as handle:
                    np.savez_compressed(handle, **arrays)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self.root / (object_id + ".npz"))
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        record = AtomicJsonStore(self.root / (object_id + ".json"))
        with record.locked():
            record.write_unlocked({"schema_version": 1, "items": [item]})
        with self.index.locked():
            payload = self.index.read_unlocked()
            payload["items"].append({key: item.get(key) for key in
                                     ("id", "kind", "name", "created_at", "study_id", "study_revision", "run_id",
                                      "return_plan_id", "return_plan_revision", "source_method")})
            self.index.write_unlocked(payload)
        return copy.deepcopy(item)
