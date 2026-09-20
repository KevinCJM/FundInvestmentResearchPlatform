"""Small optimistic draft store; published CMA artifacts remain immutable."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

from backend.custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from backend.custom_indicators.repository import AtomicJsonStore, utc_now
from backend.data_storage import guard_path
from .cma_center_contracts import CmaDraftWrite


class CmaDraftStore:
    def __init__(self, root: Path):
        self.document = AtomicJsonStore(Path(root) / "cma_drafts.json")

    def read(self) -> list[dict]:
        guard_path(self.document.path)
        path = self.document.path
        if not path.exists():
            return []
        try:
            if path.is_symlink() or path.stat().st_size > 16_000_000:
                raise ValueError("invalid file")
            items = self.document.read_unlocked()["items"]
            if not isinstance(items, list):
                raise ValueError("invalid items")
            for item in items:
                if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                        or type(item.get("revision")) is not int or item["revision"] < 1):
                    raise ValueError("invalid draft")
                CmaDraftWrite.model_validate({k: item[k] for k in
                    ("name", "editable_definition", "copied_from_id") if k in item})
            return items
        except (ValueError, TypeError, KeyError, OSError) as exc:
            raise ValidationError("CMA_DRAFT_STORE_INVALID", "LTCMA 草稿记录无效，请检查存储。") from exc

    def list(self) -> list[dict]:
        return [{k: item[k] for k in ("id", "name", "revision", "created_at", "updated_at", "copied_from_id")}
                for item in self.read()]

    def get(self, identifier: str) -> dict:
        item = next((x for x in self.read() if x["id"] == identifier), None)
        if item is None:
            raise NotFoundError("CMA_DRAFT_NOT_FOUND", "LTCMA 草稿不存在或已删除。")
        return copy.deepcopy(item)

    def save(self, body: CmaDraftWrite, identifier: str | None = None) -> dict:
        guard_path(self.document.path, write=True)
        with self.document.locked():
            items = self.read()
            now = utc_now()
            if identifier is None:
                if body.expected_revision is not None:
                    raise ValidationError("CMA_DRAFT_REVISION", "新草稿不能带已有版本号。")
                item = {"id": "cma-draft-" + uuid.uuid4().hex, "revision": 1, "created_at": now}
                items.append(item)
            else:
                item = next((x for x in items if x["id"] == identifier), None)
                if item is None:
                    raise NotFoundError("CMA_DRAFT_NOT_FOUND", "LTCMA 草稿不存在或已删除。")
                if item["revision"] != body.expected_revision:
                    raise ConflictError("CMA_DRAFT_CONFLICT", "草稿已被修改，请重新加载后再保存。")
                item["revision"] += 1
            item.update(body.model_dump(mode="json", exclude={"expected_revision"}), updated_at=now)
            self._write(items)
            return copy.deepcopy(item)

    def delete(self, identifier: str, revision: int) -> dict:
        guard_path(self.document.path, write=True)
        with self.document.locked():
            items = self.read()
            item = next((x for x in items if x["id"] == identifier), None)
            if item is None:
                raise NotFoundError("CMA_DRAFT_NOT_FOUND", "LTCMA 草稿不存在或已删除。")
            if item["revision"] != revision:
                raise ConflictError("CMA_DRAFT_CONFLICT", "草稿已被修改，请重新加载后操作。")
            self._write([x for x in items if x["id"] != identifier])
        return {"id": identifier, "deleted": True}

    def _write(self, items: list[dict]) -> None:
        if len(json.dumps(items, allow_nan=False).encode("utf-8")) > 16_000_000:
            raise ValidationError("CMA_DRAFT_CAPACITY", "LTCMA 草稿库达到容量上限。")
        self.document.write_unlocked({"items": items})
