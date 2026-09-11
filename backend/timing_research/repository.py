"""Atomic research artifacts, with per-product data and immutable run manifests.

Only serialization and file orchestration live here. Completed product files are
their own recovery records: an interrupted run can finish without recomputing or
overwriting them. Directory scans read small manifests, not every price series.
"""
from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import re
import tempfile
import uuid
import zipfile

import numpy as np

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.custom_indicators.repository import AtomicJsonStore, utc_now

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 128 * 1024 * 1024
MAX_BARS = 12_000
_ID = re.compile(r"[a-z][a-z0-9-]{0,119}")
_PRODUCT = re.compile(r"\d{6}\.(SH|SZ)")
_ARRAY_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,99}")


def clean(value):
    """JSON output boundary; numerical missingness remains null."""
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def _encoded(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(value) -> str:
    return "sha256:" + hashlib.sha256(_encoded(clean(value))).hexdigest()


def _id(value: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValidationError("TIMING_INVALID_ID", "择时研究对象 ID 无效。")
    return value


def _code(value: str) -> str:
    if not isinstance(value, str) or not _PRODUCT.fullmatch(value):
        raise ValidationError("TIMING_INVALID_PRODUCT", "ETF 产品代码无效。")
    return value


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    total = 0
    for key in sorted(arrays):
        array = arrays[key]
        if (not _ARRAY_KEY.fullmatch(key) or not isinstance(array, np.ndarray)
                or array.dtype not in (np.dtype("float64"), np.dtype("int64"))
                or array.ndim not in (1, 2) or array.shape[0] > MAX_BARS):
            raise ValidationError("TIMING_ARRAY_CONTRACT", "冻结输入必须是最多 12000 行的 float64/int64 一维或二维数组。")
        total += array.nbytes
        if total > MAX_ARRAY_BYTES:
            raise ValidationError("TIMING_ARRAY_BUDGET", "单产品冻结输入超过内存预算。")
        digest.update(_encoded([key, array.dtype.str, array.shape]))
        if not array.size:
            continue
        if array.flags.c_contiguous:
            digest.update(memoryview(array).cast("B"))
        else:
            # Bounded serialization buffer for arbitrary-stride shared views.
            iterator = np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                                 op_flags=[["readonly", "contig"]], order="C", buffersize=8192)
            for block in iterator:
                digest.update(memoryview(block).cast("B"))
    return "sha256:" + digest.hexdigest()


class TimingRepository:
    def __init__(self, base: Path):
        self.root = Path(base).expanduser().resolve() / "timing_research"
        self._lock = AtomicJsonStore(self.root / ".repository.json")

    @contextmanager
    def _locked(self):
        self._path(".repository.json.lock")
        with self._lock.locked():
            yield

    def _path(self, *parts: str) -> Path:
        current = self.root
        for part in ("", *parts):
            current = current / part
            if current.is_symlink():
                raise ValidationError("TIMING_STORAGE_PATH", "研究存储目录或文件不能是符号链接。")
        if self.root.resolve() not in current.resolve().parents and current.resolve() != self.root.resolve():
            raise ValidationError("TIMING_STORAGE_PATH", "研究存储路径超出工作区。")
        return current

    @staticmethod
    def _read(path: Path):
        if not path.is_file():
            raise NotFoundError("TIMING_ARTIFACT_NOT_FOUND", "未找到择时研究对象或冻结结果。")
        try:
            if path.stat().st_size > MAX_JSON_BYTES:
                raise ValueError("oversized JSON")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            payload = envelope["payload"]
            if envelope["sha256"] != content_hash(payload):
                raise ValueError("checksum mismatch")
            return payload
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValidationError("TIMING_ARTIFACT_CORRUPT", "研究文件损坏或内容校验失败，已停止读取。") from exc

    @staticmethod
    def _sync_directory(path: Path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _write(self, path: Path, payload, *, replace=False):
        if path.exists() and not replace:
            raise ConflictError("TIMING_ARTIFACT_IMMUTABLE", "已冻结的研究文件不能覆盖。")
        normalized = clean(payload)
        encoded = _encoded({"schema_version": 1, "sha256": content_hash(normalized), "payload": normalized})
        if len(encoded) > MAX_JSON_BYTES:
            raise ValidationError("TIMING_RESULT_BUDGET", "单产品结果超过存储预算，请减少计算步骤。")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".timing-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def list_definitions(self):
        directory = self._path("definitions")
        return sorted([self._read(self._path("definitions", p.name))["current"]
                       for p in directory.glob("*.json")], key=lambda item: item["updated_at"], reverse=True)

    def get_definition(self, object_id: str, revision: int | None = None):
        item = self._read(self._path("definitions", _id(object_id) + ".json"))
        for value in [item["current"], *item["history"]]:
            if revision is None or value["revision"] == revision:
                return value
        raise NotFoundError("TIMING_REVISION_NOT_FOUND", "未找到该择时算法版本。")

    def save_definition(self, fields: dict, object_id: str | None = None, revision: int | None = None):
        object_id = _id(object_id) if object_id else "timing-def-" + uuid.uuid4().hex
        path = self._path("definitions", object_id + ".json")
        with self._locked():
            previous = self._read(path) if path.exists() else None
            if previous is None and revision is not None:
                raise NotFoundError("TIMING_DEFINITION_NOT_FOUND", "未找到要更新的择时算法。")
            if previous and (type(revision) is not int or revision != previous["current"]["revision"]):
                raise ConflictError("REVISION_CONFLICT", "算法已更新，请刷新后重试。")
            now = utc_now()
            current = {**clean(fields), "id": object_id, "revision": revision + 1 if previous else 1,
                       "created_at": previous["current"]["created_at"] if previous else now, "updated_at": now}
            history = [*previous["history"], previous["current"]] if previous else []
            self._write(path, {"current": current, "history": history}, replace=previous is not None)
        return current

    def create_run(self, definition: dict, request: dict) -> str:
        run_id = "timing-run-" + uuid.uuid4().hex
        record = {"id": run_id, "name": definition.get("name", "择时研究"), "created_at": utc_now(),
                  "definition_snapshot": clean(definition), "request_snapshot": clean(request),
                  "definition_hash": content_hash(definition), "request_hash": content_hash(request),
                  "status": "running", "immutable": False}
        with self._locked():
            self._write(self._path("runs", run_id, "pending.json"), record)
        return run_id

    def _array_path(self, run_id, code):
        return self._path("runs", _id(run_id), _code(code) + ".npz")

    def _save_arrays(self, run_id, code, arrays):
        checksum = _array_hash(arrays)
        path = self._array_path(run_id, code)
        if path.exists():
            if _array_hash(self._load_arrays_path(path)) != checksum:
                raise ConflictError("TIMING_INPUT_IMMUTABLE", "该产品已有不同的冻结输入，不能覆盖。")
            return checksum
        descriptor, temporary = tempfile.mkstemp(prefix=".timing-input-", suffix=".npz", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                np.savez(handle, **arrays)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return checksum

    @staticmethod
    def _load_arrays_path(path):
        try:
            if path.stat().st_size > MAX_ARRAY_BYTES + 1024 * 1024:
                raise ValueError("oversized input archive")
            with zipfile.ZipFile(path) as archive:
                if sum(item.file_size for item in archive.infolist()) > MAX_ARRAY_BYTES + 1024 * 1024:
                    raise ValueError("oversized decoded input")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            _array_hash(arrays)
            for array in arrays.values():
                array.setflags(write=False)
            return arrays
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
            raise ValidationError("TIMING_INPUT_CORRUPT", "冻结输入损坏或格式无效，已停止重放。") from exc

    @staticmethod
    def _product_summary(result, checksum):
        fields = ("product_id", "status", "error", "summary", "diagnostics", "warnings", "lineage")
        return {**{key: result[key] for key in fields if key in result}, "detail_loaded": False,
                "result_hash": checksum, "input_checksum": result.get("input_checksum"),
                "data_hash": result.get("lineage", {}).get("source_hash")}

    def save_product(self, run_id: str, code: str, result: dict, arrays: dict | None = None):
        run_id, code = _id(run_id), _code(code)
        with self._locked():
            self._read(self._path("runs", run_id, "pending.json"))
            if self._path("runs", run_id, "manifest.json").exists():
                raise ConflictError("TIMING_RUN_IMMUTABLE", "研究运行已完成，不能追加或修改产品结果。")
            record = {**clean(result), "product_id": code, "detail_loaded": True}
            if arrays is not None:
                record["input_checksum"] = self._save_arrays(run_id, code, arrays)
            elif record.get("status") == "ok":
                raise ValidationError("TIMING_INPUT_REQUIRED", "成功的研究结果必须同时冻结计算输入。")
            path = self._path("runs", run_id, code + ".json")
            if path.exists():
                if content_hash(self._read(path)) != content_hash(record):
                    raise ConflictError("TIMING_PRODUCT_IMMUTABLE", "该产品已有不同的冻结结果，不能覆盖。")
            else:
                self._write(path, record)
            return self._product_summary(record, content_hash(record))

    def finish_run(self, run_id: str, summary: dict | None = None):
        run_id = _id(run_id)
        with self._locked():
            path = self._path("runs", run_id, "manifest.json")
            if path.exists():
                return self.get_run(run_id)
            pending = self._read(self._path("runs", run_id, "pending.json"))
            products = []
            directory = self._path("runs", run_id)
            requested = [item["product_id"] for item in pending["request_snapshot"].get("targets", [])]
            codes = requested or sorted(p.stem for p in directory.glob("*.json") if _PRODUCT.fullmatch(p.stem))
            for code in codes:
                record = self._read(self._path("runs", run_id, _code(code) + ".json"))
                if record.get("status") == "ok":
                    self.load_arrays(run_id, code, _record=record)
                products.append(self._product_summary(record, content_hash(record)))
            extra = clean(summary or {})
            for key in ("definition_snapshot", "request_snapshot", "definition_hash", "request_hash", "created_at"):
                extra.pop(key, None)
            record = {**pending, **extra, "id": run_id, "status": "completed",
                      "products": products, "immutable": True, "completed_at": utc_now()}
            self._write(path, record)
        return self.get_run(run_id)

    def list_runs(self, definition_id: str | None = None, offset: int = 0, limit: int = 50):
        self._page(offset, limit)
        items = []
        for path in self._path("runs").glob("*/manifest.json"):
            record = self._read(self._path("runs", _id(path.parent.name), "manifest.json"))
            if definition_id and record["definition_snapshot"].get("id") != definition_id:
                continue
            items.append({key: record.get(key) for key in ("id", "name", "created_at", "status", "products")})
        items.sort(key=lambda item: item["created_at"], reverse=True)
        return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit}

    @staticmethod
    def _page(offset, limit):
        if type(offset) is not int or type(limit) is not int or offset < 0 or not 1 <= limit <= MAX_BARS:
            raise ValidationError("TIMING_PAGE_INVALID", "分页范围无效，每页最多 12000 条。")

    def get_run(self, run_id: str, product_code: str | None = None, offset: int = 0, limit: int = MAX_BARS):
        self._page(offset, limit)
        record = self._read(self._path("runs", _id(run_id), "manifest.json"))
        if product_code is None:
            product_code = next((p["product_id"] for p in record["products"] if p.get("status") == "ok"), None)
        if product_code is not None:
            detail = self.get_product(run_id, product_code, offset, limit)
            record["products"] = [detail if p["product_id"] == product_code else p for p in record["products"]]
        return record

    def get_product(self, run_id: str, code: str, offset: int = 0, limit: int = 1200):
        self._page(offset, limit)
        manifest = self._read(self._path("runs", _id(run_id), "manifest.json"))
        expected = next((p for p in manifest["products"] if p["product_id"] == _code(code)), None)
        if expected is None:
            raise NotFoundError("TIMING_PRODUCT_NOT_FOUND", "该研究没有所选产品。")
        result = self._read(self._path("runs", run_id, code + ".json"))
        if expected["result_hash"] != content_hash(result):
            raise ValidationError("TIMING_RESULT_CHANGED", "产品结果与冻结运行清单不一致。")
        totals = {}
        for field in ("curve", "trades"):
            if field in result:
                totals[field] = len(result[field])
                result[field] = result[field][offset:offset + limit]
        for channel in result.get("channels", []):
            channel["values"] = channel["values"][offset:offset + limit]
        result["pagination"] = {"offset": offset, "limit": limit, "totals": totals}
        result["detail_loaded"] = True
        if "execution" not in result and "execution" in manifest:
            result["execution"] = manifest["execution"]
        return result

    def load_arrays(self, run_id: str, code: str, *, _record=None):
        record = _record if _record is not None else self._read(self._path("runs", _id(run_id), _code(code) + ".json"))
        manifest_path = self._path("runs", _id(run_id), "manifest.json")
        if manifest_path.exists():
            manifest = self._read(manifest_path)
            expected = next((item for item in manifest["products"] if item["product_id"] == _code(code)), None)
            if expected is None or expected["result_hash"] != content_hash(record):
                raise ValidationError("TIMING_RESULT_CHANGED", "冻结输入对应的产品结果与运行清单不一致。")
        if not record.get("input_checksum"):
            raise ValidationError("TIMING_INPUT_MISSING", "该产品没有冻结计算输入。")
        arrays = self._load_arrays_path(self._array_path(run_id, code))
        if _array_hash(arrays) != record["input_checksum"]:
            raise ValidationError("TIMING_INPUT_CHANGED", "冻结输入校验和不一致，已停止重放。")
        return arrays

    def create_release(self, run_id: str, name: str | None = None, note: str = ""):
        run_id = _id(run_id)
        run = self._read(self._path("runs", run_id, "manifest.json"))
        successful = [item for item in run["products"] if item.get("status") == "ok"]
        if run.get("status") != "completed" or not successful:
            raise ValidationError("TIMING_RELEASE_UNAVAILABLE", "至少需要一个已完成且成功的产品研究结果。")
        for item in successful:
            self.get_product(run_id, item["product_id"], limit=1)
            self.load_arrays(run_id, item["product_id"])
        record = {"id": "timing-release-" + uuid.uuid4().hex, "run_id": run_id,
                  "name": name or run["name"], "note": note, "created_at": utc_now(), "immutable": True,
                  "usage": "research_only", "execution_authorized": False,
                  "definition_hash": run["definition_hash"], "request_hash": run["request_hash"],
                  "run_hash": content_hash(run), "products": successful}
        with self._locked():
            self._write(self._path("releases", record["id"] + ".json"), record)
        return record

    def list_releases(self):
        return sorted([self._read(self._path("releases", p.name)) for p in self._path("releases").glob("*.json")],
                      key=lambda item: item["created_at"], reverse=True)

    def get_release(self, release_id: str):
        return self._read(self._path("releases", _id(release_id) + ".json"))

    def create_binding(self, release_id: str, context_type: str, context_id: str = "workspace", note: str = ""):
        if context_type not in {"product_research", "pre_investment"} or not isinstance(context_id, str) or not 1 <= len(context_id) <= 120:
            raise ValidationError("TIMING_BINDING_INVALID", "研究结果只能引用到产品研究或投前决策。")
        release = self.get_release(release_id)
        run = self._read(self._path("runs", _id(release["run_id"]), "manifest.json"))
        if content_hash(run) != release["run_hash"]:
            raise ValidationError("TIMING_RELEASE_CHANGED", "引用版本与冻结研究不一致。")
        # A previously published manifest does not prove its input files are
        # still intact when a new downstream reference is created.
        for product in release["products"]:
            self.get_product(release["run_id"], product["product_id"], limit=1)
            self.load_arrays(release["run_id"], product["product_id"])
        identity = {"release_id": release_id, "context": context_type, "context_id": context_id}
        record = {**identity, "id": "timing-binding-" + uuid.uuid4().hex, "note": note,
                  "created_at": utc_now(), "usage": "research_only", "execution_authorized": False,
                  "release_hash": content_hash(release)}
        with self._locked():
            for existing in self.list_bindings(context_type, context_id):
                if existing["release_id"] == release_id:
                    return existing
            self._write(self._path("bindings", record["id"] + ".json"), record)
        return record

    def list_bindings(self, context_type: str | None = None, context_id: str | None = None):
        result = [self._read(self._path("bindings", p.name)) for p in self._path("bindings").glob("*.json")]
        return [item for item in result if (context_type is None or item["context"] == context_type)
                and (context_id is None or item["context_id"] == context_id)]
