"""Guarded, immutable research artifacts on the application's managed data disk."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from backend.data_storage import fsync_dir, guard_path
from backend.factor_research.repository import clean

ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,119}$")
ARRAY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MANIFEST_MAX_BYTES = 8_000_000
SUMMARY_KEYS = (
    "id", "kind", "name", "created_at", "stage", "model_id", "model_revision",
    "run_id", "release_id", "target_keys", "as_of", "effective_at", "expires_at",
    "cache_key", "content_hash", "method", "frequency", "publishable", "entry",
    "artifact_type", "scheme_id", "version_number", "base_currency", "risk_basis_id",
    "research_as_of", "review_due_at",
)


def digest_json(value: Any) -> str:
    encoded = json.dumps(clean(value), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_id(value: str) -> str:
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValidationError("INVALID_RESEARCH_ID", "研究成果标识无效。")
    return value


class ArtifactRepository:
    """JSON metadata + checked read-only NPY inputs; no pickle or mutable results."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.index = AtomicJsonStore(self.root / "index.json")
        self.governance_lock = AtomicJsonStore(self.root / "governance.json")

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        guard_path(self.root)
        if not self.index.path.exists():
            return []
        if self.index.path.is_symlink() or self.index.path.stat().st_size > 16_000_000:
            raise ValidationError("RESEARCH_INDEX_INVALID", "研究成果目录损坏或过大。")
        # The writer uses atomic replacement. Reads do not create folders or locks.
        items = self.index.read_unlocked()["items"]
        return [item for item in reversed(items) if kind is None or item.get("kind") == kind]

    def find(self, kind: str, cache_key: str) -> dict[str, Any] | None:
        match = next((item for item in self.list(kind) if item.get("cache_key") == cache_key), None)
        return self.get(match["id"], kind) if match else None

    def _folder(self, object_id: str) -> Path:
        folder = self.root / checked_id(object_id)
        if folder.is_symlink() or folder.resolve().parent != self.root.resolve():
            raise ValidationError("RESEARCH_PATH_INVALID", "研究成果路径无效。")
        return folder

    def get(self, object_id: str, kind: str | None = None) -> dict[str, Any]:
        guard_path(self.root)
        path = self._folder(object_id) / "manifest.json"
        if not path.is_file():
            raise NotFoundError("RESEARCH_ARTIFACT_NOT_FOUND", "未找到研究成果；请检查所选版本及数据磁盘。")
        try:
            if path.is_symlink() or path.stat().st_size > MANIFEST_MAX_BYTES:
                raise ValueError("invalid manifest")
            item = json.loads(path.read_text(encoding="utf-8"))
            checksum = item.get("content_hash")
            if item.get("id") != object_id or not checksum or digest_json(
                {key: value for key, value in item.items() if key != "content_hash"}
            ) != checksum:
                raise ValueError("invalid checksum")
        except (ValueError, TypeError, AttributeError, OSError) as exc:
            raise ValidationError("RESEARCH_ARTIFACT_CORRUPT", "研究成果校验失败，已停止使用；不会重新计算替代原结果。") from exc
        if kind is not None and item.get("kind") != kind:
            raise ValidationError("RESEARCH_ARTIFACT_KIND", "所选研究成果类型不匹配。")
        return item

    def arrays(self, object_id: str, names: tuple[str, ...] | None = None) -> dict[str, np.ndarray]:
        item = self.get(object_id)
        descriptors = item.get("arrays", {})
        if names is not None:
            if any(key not in descriptors for key in names):
                raise ValidationError("RESEARCH_ARRAY_MISSING", "研究成果缺少所需冻结数组。")
            descriptors = {key: descriptors[key] for key in names}
        result = {}
        for key, descriptor in descriptors.items():
            if not ARRAY_PATTERN.fullmatch(key) or descriptor.get("file") != f"{key}.npy":
                raise ValidationError("RESEARCH_ARRAY_INVALID", "冻结数组清单无效。")
            path = self._folder(object_id) / descriptor["file"]
            try:
                if (path.is_symlink() or not path.is_file() or path.stat().st_size > 256_000_000
                        or file_hash(path) != descriptor["sha256"]):
                    raise ValueError("array checksum mismatch")
                value = np.load(path, mmap_mode="r", allow_pickle=False)
                if (value.dtype.str != descriptor["dtype"] or list(value.shape) != descriptor["shape"]
                        or value.dtype not in (np.dtype("float64"), np.dtype("int64"))
                        or not value.flags.c_contiguous):
                    raise ValueError("array contract mismatch")
                value.flags.writeable = False
                result[key] = value
            except (OSError, ValueError, KeyError, EOFError) as exc:
                raise ValidationError("RESEARCH_ARRAY_CORRUPT", "冻结输入数组损坏或不符合数值契约，已停止使用。") from exc
        return result

    @staticmethod
    def _read_operations(store: AtomicJsonStore) -> dict:
        if store.path.exists() and (store.path.is_symlink() or store.path.stat().st_size > 16_000_000):
            raise ValidationError("IDEMPOTENCY_CORRUPT", "发布操作记录损坏或超过容量。")
        payload = store.read_unlocked()
        for item in payload["items"]:
            if (not isinstance(item, dict) or not isinstance(item.get("key"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["key"])
                    or not isinstance(item.get("request_hash"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", item["request_hash"])
                    or type(item.get("complete")) is not bool or not isinstance(item.get("kind"), str)):
                raise ValidationError("IDEMPOTENCY_CORRUPT", "发布操作记录结构无效。")
            checked_id(item.get("id"))
        return payload

    def idempotent_result(self, key: str, request_hash: str) -> dict[str, Any] | None:
        """Read-only replay lookup; interrupted promotion is finished by save()."""
        guard_path(self.root)
        operations = AtomicJsonStore(self.root / "operations.json")
        if not operations.path.exists():
            return None
        token = hashlib.sha256(key.encode("utf-8")).hexdigest()
        operation = next((x for x in self._read_operations(operations)["items"] if x["key"] == token), None)
        if operation is None:
            return None
        if operation["request_hash"] != request_hash:
            raise ConflictError("IDEMPOTENCY_CONFLICT", "同一幂等键已用于不同输入。")
        if operation.get("complete"):
            item = self.get(operation["id"], operation["kind"])
            if item.get("idempotency") != {"key": token, "request_hash": request_hash}:
                raise ValidationError("IDEMPOTENCY_CORRUPT", "发布恢复记录与冻结成果不一致。")
            self.arrays(item["id"])
            return item
        return None

    def save(self, kind: str, fields: dict[str, Any], arrays: dict[str, np.ndarray] | None = None,
             *, idempotency_key: str | None = None, request_hash: str | None = None) -> dict[str, Any]:
        """Optional exact-operation replay; existing callers keep random immutable IDs.

        Reservation precedes promotion. Recovery checks only its reserved directory,
        including all NPY checksums, then repairs the index without a directory scan.
        """
        if idempotency_key is None:
            return self._save(kind, fields, arrays)
        guard_path(self.root, write=True)
        if (not isinstance(idempotency_key, str) or not 8 <= len(idempotency_key) <= 160
                or not isinstance(request_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", request_hash)):
            raise ValidationError("IDEMPOTENCY_INVALID", "幂等发布需要有效操作键和请求指纹。")
        operations = AtomicJsonStore(self.root / "operations.json")
        token = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        with operations.locked():
            payload = self._read_operations(operations)
            operation = next((x for x in payload["items"] if x["key"] == token), None)
            if operation is not None and (operation["request_hash"] != request_hash or operation["kind"] != kind):
                raise ConflictError("IDEMPOTENCY_CONFLICT", "同一幂等键已用于不同输入。")
            if operation is None:
                operation = {"key": token, "kind": kind, "request_hash": request_hash,
                             "id": f"research-{kind}-{uuid.uuid4().hex}", "complete": False}
                payload["items"].append(operation)
                if len(json.dumps(payload).encode()) > 16_000_000:
                    raise ValidationError("IDEMPOTENCY_CAPACITY", "发布操作记录达到容量上限。")
                operations.write_unlocked(payload)
            destination = self._folder(operation["id"])
            if destination.exists():
                item = self.get(operation["id"], kind)
                if item.get("idempotency") != {"key": token, "request_hash": request_hash}:
                    raise ValidationError("IDEMPOTENCY_CORRUPT", "发布恢复记录与冻结成果不一致。")
                self.arrays(item["id"])
                self._register(item)
            else:
                item = self._save(kind, {**fields, "idempotency": {"key": token, "request_hash": request_hash}},
                                  arrays, object_id=operation["id"])
            operation["complete"] = True
            operations.write_unlocked(payload)
            return item

    def _register(self, item: dict[str, Any]) -> None:
        with self.index.locked():
            payload = self.index.read_unlocked()
            previous = next((x for x in payload["items"] if x["id"] == item["id"]), None)
            if previous is not None:
                if previous.get("content_hash") != item["content_hash"]:
                    raise ValidationError("RESEARCH_INDEX_INVALID", "成果索引与冻结内容不一致。")
                return
            payload["items"].append({key: item[key] for key in SUMMARY_KEYS if key in item})
            self.index.write_unlocked(payload)

    def _save(self, kind: str, fields: dict[str, Any], arrays: dict[str, np.ndarray] | None = None,
              *, object_id: str | None = None) -> dict[str, Any]:
        guard_path(self.root, write=True)
        if kind not in {"run", "release", "retirement", "series", "preview", "impact"}:
            raise ValueError("unsupported artifact kind")
        object_id = object_id or f"research-{kind}-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".writing-", dir=self.root))
        destination = self._folder(object_id)
        promoted = False
        try:
            descriptors = {}
            for key, raw in (arrays or {}).items():
                if not ARRAY_PATTERN.fullmatch(key):
                    raise ValueError("invalid numeric array name")
                value = np.asarray(raw)
                if value.dtype not in (np.dtype("float64"), np.dtype("int64")) or value.nbytes > 256_000_000:
                    raise ValueError("unsupported numerical array")
                # One explicit persistence boundary; no per-node materialization.
                path = temporary / f"{key}.npy"
                with path.open("wb") as target:
                    np.save(target, value, allow_pickle=False)
                    target.flush()
                    os.fsync(target.fileno())
                descriptors[key] = {"file": path.name, "sha256": file_hash(path),
                                    "dtype": value.dtype.str, "shape": list(value.shape)}
            item = {**clean(fields), "id": object_id, "kind": kind, "schema_version": 1,
                    "created_at": utc_now(), "immutable": True, "arrays": descriptors}
            item["content_hash"] = digest_json(item)
            encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
            if len(encoded) > MANIFEST_MAX_BYTES:
                raise ValidationError("RESEARCH_ARTIFACT_CAPACITY", "研究成果超过容量上限，请精简说明或对账记录后重试。")
            with (temporary / "manifest.json").open("wb") as target:
                target.write(encoded)
                target.flush()
                os.fsync(target.fileno())
            fsync_dir(temporary)
            guard_path(self.root, write=True)
            os.rename(temporary, destination)
            promoted = True
            fsync_dir(self.root)
            self._register(item)
            return item
        finally:
            if not promoted and temporary.exists():
                # Only this exact request-owned temporary directory is removed.
                shutil.rmtree(temporary)
