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

from backend.custom_indicators.errors import NotFoundError, ValidationError
from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from backend.data_storage import fsync_dir, guard_path
from backend.factor_research.repository import clean

ID_PATTERN = re.compile(r"^[a-z][a-z0-9-]{0,119}$")
ARRAY_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SUMMARY_KEYS = (
    "id", "kind", "name", "created_at", "stage", "model_id", "model_revision",
    "run_id", "release_id", "target_keys", "as_of", "effective_at", "expires_at",
    "cache_key", "content_hash", "method", "frequency", "publishable", "entry",
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
            if path.is_symlink() or path.stat().st_size > 8_000_000:
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

    def save(self, kind: str, fields: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        guard_path(self.root, write=True)
        if kind not in {"run", "release", "retirement", "series", "preview", "impact"}:
            raise ValueError("unsupported artifact kind")
        object_id = f"research-{kind}-{uuid.uuid4().hex}"
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
            with (temporary / "manifest.json").open("w", encoding="utf-8") as target:
                json.dump(item, target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                target.flush()
                os.fsync(target.fileno())
            fsync_dir(temporary)
            guard_path(self.root, write=True)
            os.rename(temporary, destination)
            promoted = True
            fsync_dir(self.root)
            with self.index.locked():
                payload = self.index.read_unlocked()
                payload["items"].append({key: item[key] for key in SUMMARY_KEYS if key in item})
                self.index.write_unlocked(payload)
            return item
        finally:
            if not promoted and temporary.exists():
                # Only this exact request-owned temporary directory is removed.
                shutil.rmtree(temporary)
