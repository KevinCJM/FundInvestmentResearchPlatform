"""Versioned event facts and research windows; no market data or numerical work."""
from __future__ import annotations

import copy
import hashlib
import json
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, Mapping
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError as ModelError, field_validator, model_validator
from custom_indicators.errors import ConflictError, NotFoundError, ValidationError
from custom_indicators.repository import AtomicJsonStore, utc_now

CATEGORIES = {
    "financial": "金融市场", "geopolitical": "地缘与战争", "political": "政治与政策",
    "trade": "贸易与制裁", "health": "公共卫生", "social": "社会事件",
    "disaster": "自然灾害", "supply_chain": "能源与供应链", "technology": "科技冲击",
}
IDENTIFIER = r"^[A-Za-z][A-Za-z0-9_-]{0,63}$"


class EventSource(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=1, max_length=200)
    url: str = Field(min_length=1, max_length=2000)
    published_at: date | None = None

    @field_validator("url")
    @classmethod
    def safe_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("来源只接受不含凭据的 http(s) 链接")
        return value


class EventWindow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=IDENTIFIER)
    label: str = Field(min_length=1, max_length=100)
    start_date: date
    end_date: date
    rationale: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="after")
    def date_order(self):
        if self.start_date > self.end_date:
            raise ValueError("研究窗口的开始日期不能晚于结束日期")
        if self.start_date.year < 1700 or self.end_date.year > 2200:
            raise ValueError("日期须在1700—2200年之间")
        return self


class EventDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=100)
    name_en: str = Field(default="", max_length=150)
    description: str = Field(default="", max_length=2000)
    categories: list[str] = Field(default_factory=list, max_length=9)
    regions: list[str] = Field(default_factory=list, max_length=20)
    fact_start: date | None = None
    fact_end: date | None = None
    date_precision: Literal["day", "month", "year", "unknown"] = "unknown"
    status: Literal["ongoing", "closed", "unknown"] = "unknown"
    verification: Literal["unreviewed", "verified"] = "unreviewed"
    known_at: datetime | None = None
    sources: list[EventSource] = Field(default_factory=list, max_length=20)
    windows: list[EventWindow] = Field(min_length=1, max_length=20)
    color: str = Field(default="#7c3aed", pattern=r"^#[0-9A-Fa-f]{6}$")
    archived: bool = False

    @field_validator("categories")
    @classmethod
    def known_categories(cls, value):
        if len(value) != len(set(value)) or set(value) - set(CATEGORIES):
            raise ValueError("事件类别无效或重复")
        return value

    @field_validator("regions")
    @classmethod
    def bounded_regions(cls, value):
        if any(not item.strip() or len(item) > 80 for item in value) or len(value) != len(set(value)):
            raise ValueError("地区不能为空、重复或超过80个字符")
        return value

    @model_validator(mode="after")
    def consistent(self):
        if self.fact_start and self.fact_end and self.fact_start > self.fact_end:
            raise ValueError("事实开始日期不能晚于结束日期")
        if self.status == "ongoing" and self.fact_end is not None:
            raise ValueError("进行中事件不能填写已结束日期；可单独设置研究截至日")
        if self.verification == "verified" and not self.sources:
            raise ValueError("标为已核验前必须填写可追溯来源")
        if self.known_at and self.known_at.tzinfo is None:
            raise ValueError("信息可得时间须包含时区")
        if len({w.id for w in self.windows}) != len(self.windows):
            raise ValueError("研究窗口编号不能重复")
        return self


class EventSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(pattern=IDENTIFIER)
    revision: int = Field(ge=1, strict=True)
    window_id: str = Field(pattern=IDENTIFIER)


def suggested_categories(label: str, description: str) -> list[str]:
    """Editable topic suggestions only; import never claims factual verification."""
    text = (label + " " + description).casefold()
    keywords = {
        "financial": ("金融", "银行", "债", "货币", "外汇", "流动性", "爆仓", "评级", "ftx", "taper", "信用", "市场结构"),
        "geopolitical": ("战争", "军事", "冲突", "地缘", "克里米亚", "以伊"),
        "political": ("政治", "公投", "脱欧", "戒严", "选举", "阿拉伯之春"),
        "trade": ("关税", "贸易", "制裁"), "health": ("疫情", "公共卫生", "sars", "covid", "埃博拉"),
        "social": ("社会", "抱团", "散户"), "disaster": ("地震", "海啸", "核事故", "灾难"),
        "supply_chain": ("供应链", "航运", "能源", "大宗商品", "运河", "石油"), "technology": ("科技", "互联网", "deepseek", "ai模型"),
    }
    return [category for category, terms in keywords.items() if any(term in text for term in terms)]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _parse(draft: Mapping[str, Any]) -> dict:
    try:
        return EventDraft.model_validate(draft).model_dump(mode="json")
    except ModelError as exc:
        errors = [{"path": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in exc.errors(include_input=False, include_url=False)]
        raise ValidationError("INVALID_LIBRARY_EVENT", errors[0]["message"], "event", errors) from None


class EventLibraryService:
    def __init__(self, directory: Path):
        self.store = AtomicJsonStore(directory / "historical_event_library.json")

    def _entries(self) -> list[dict]:
        with self.store.locked():
            return self.store.read_unlocked()["items"]

    @staticmethod
    def _version(entry: dict, revision: int | None) -> dict:
        candidates = [entry["current"], *entry.get("history", [])]
        result = next((x for x in candidates if revision is None or x["revision"] == revision), None)
        if result is None:
            raise NotFoundError("EVENT_REVISION_NOT_FOUND", "未找到该事件修订版本。")
        return copy.deepcopy(result)

    def get(self, event_id: str, revision: int | None = None) -> dict:
        entry = next((e for e in self._entries() if e["current"]["id"] == event_id), None)
        if entry is None:
            raise NotFoundError("EVENT_NOT_FOUND", "未找到该历史事件。")
        return self._version(entry, revision)

    def history(self, event_id: str) -> dict:
        entry = next((e for e in self._entries() if e["current"]["id"] == event_id), None)
        if entry is None:
            raise NotFoundError("EVENT_NOT_FOUND", "未找到该历史事件。")
        return {"items": [entry["current"], *reversed(entry.get("history", []))]}

    def list(self, *, query: str = "", category: str = "", region: str = "", verification: str = "",
             start: str = "", end: str = "", archived: bool = False, offset: int = 0, limit: int = 50) -> dict:
        if start and end and start > end:
            raise ValidationError("EVENT_DATE_RANGE", "筛选开始日期不能晚于结束日期。", "start")
        q = query.strip().casefold()
        items = [entry["current"] for entry in self._entries()]
        items = [e for e in items if (archived or not e["archived"])
                 and (not category or category in e["categories"])
                 and (not region or any(region.casefold() in r.casefold() for r in e["regions"]))
                 and (not verification or e["verification"] == verification)
                 and (not q or q in (e["name"] + " " + e["name_en"] + " " + e["description"]).casefold())
                 and (not start and not end or any((not start or w["end_date"] >= start) and (not end or w["start_date"] <= end) for w in e["windows"]))]
        items.sort(key=lambda e: (e["windows"][0]["start_date"], e["id"]), reverse=True)
        return {"items": items[offset:offset + limit], "total": len(items), "offset": offset, "limit": limit,
                "categories": CATEGORIES, "time_filter_basis": "research_window"}

    @staticmethod
    def _record(fields: dict, event_id: str, revision: int, created_at: str | None = None, provenance: dict | None = None) -> dict:
        now = utc_now()
        return {**fields, "id": event_id, "revision": revision, "created_at": created_at or now,
                "updated_at": now, "content_hash": _digest(fields), "provenance": provenance or {"kind": "manual"}}

    def create(self, draft: Mapping[str, Any]) -> dict:
        record = self._record(_parse(draft), f"event-{uuid.uuid4().hex}", 1)
        with self.store.locked():
            data = self.store.read_unlocked()
            if len(data["items"]) >= 5000:
                raise ValidationError("EVENT_LIBRARY_LIMIT", "当前事件库最多5000项。")
            data["items"].append({"current": record, "history": []})
            self.store.write_unlocked(data)
        return record

    def update(self, event_id: str, revision: int, draft: Mapping[str, Any]) -> dict:
        fields = _parse(draft)
        with self.store.locked():
            data = self.store.read_unlocked()
            entry = next((e for e in data["items"] if e["current"]["id"] == event_id), None)
            if entry is None:
                raise NotFoundError("EVENT_NOT_FOUND", "未找到该历史事件。")
            previous = entry["current"]
            if previous["revision"] != revision:
                raise ConflictError("EVENT_REVISION_CONFLICT", "事件已被其他操作更新，请重新读取后修改。", field="revision")
            current = self._record(fields, event_id, revision + 1, previous["created_at"], previous["provenance"])
            entry["history"].append(previous)
            entry["current"] = current
            self.store.write_unlocked(data)
        return current

    def packs(self) -> dict:
        items = [e["current"] for e in self._entries() if not e["current"]["archived"]]
        packs = [{"id": "all", "name": "全部历史事件", "count": len(items)}]
        packs.extend({"id": key, "name": label + "事件包", "count": sum(key in e["categories"] for e in items)} for key, label in CATEGORIES.items())
        return {"items": packs}

    def pack(self, pack_id: str) -> dict:
        if pack_id != "all" and pack_id not in CATEGORIES:
            raise NotFoundError("EVENT_PACK_NOT_FOUND", "未找到事件包。")
        items = self.list(category="" if pack_id == "all" else pack_id, limit=5001)["items"]
        if len(items) > 100:
            raise ValidationError("EVENT_SELECTION_LIMIT", "事件包超过100项，请按类别或时间筛选后选择。")
        return {"selections": [{"event_id": e["id"], "revision": e["revision"], "window_id": e["windows"][0]["id"]} for e in items],
                "labels": {f"{e['id']}:{e['revision']}:{e['windows'][0]['id']}": e["name"] + " · " + e["windows"][0]["label"] for e in items}}

    def resolve(self, selections: list[dict], *, allow_archived: bool = False) -> list[dict]:
        if not 1 <= len(selections) <= 100:
            raise ValidationError("EVENT_SELECTION_LIMIT", "请选择1—100个事件窗口。")
        entries = {e["current"]["id"]: e for e in self._entries()}
        result, seen = [], set()
        for raw in selections:
            try:
                selection = EventSelection.model_validate(raw)
            except ModelError:
                raise ValidationError("INVALID_EVENT_REFERENCE", "事件引用须包含有效ID、精确修订号和窗口ID。") from None
            identity = (selection.event_id, selection.revision, selection.window_id)
            if identity in seen:
                continue
            seen.add(identity)
            entry = entries.get(selection.event_id)
            if entry is None:
                raise NotFoundError("EVENT_NOT_FOUND", "引用的事件不存在。")
            event = self._version(entry, selection.revision)
            if not allow_archived and entry["current"]["archived"]:
                raise ConflictError("EVENT_ARCHIVED", "事件已归档，不能新加入；既有冻结引用仍然有效。")
            window = next((w for w in event["windows"] if w["id"] == selection.window_id), None)
            if window is None:
                raise NotFoundError("EVENT_WINDOW_NOT_FOUND", "该修订中不存在所选研究窗口。")
            result.append({"id": "library_" + _digest(identity)[:32], "label": event["name"],
                           "start_date": window["start_date"], "end_date": window["end_date"], "color": event["color"],
                           "description": (window["label"] + "：" + window["rationale"])[:500],
                           "library_reference": {**selection.model_dump(), "content_hash": event["content_hash"]}})
        return result

    def verify_definition(self, definition: Mapping[str, Any]) -> None:
        linked = []
        for node in definition.get("graph", {}).get("nodes", []):
            if node.get("type") != "annotation.manual_events":
                continue
            for event in node.get("parameters", {}).get("events", []):
                if not isinstance(event, dict) or "library_reference" not in event:
                    continue
                ref = event["library_reference"]
                if not isinstance(ref, dict) or set(ref) != {"event_id", "revision", "window_id", "content_hash"}:
                    raise ValidationError("INVALID_EVENT_REFERENCE", "事件库引用格式错误。")
                linked.append((node["id"], event, ref))
        # Resolve at most once per bounded block instead of rereading the entire
        # library for every event. No repository access for plain manual rows.
        for offset in range(0, len(linked), 100):
            block = linked[offset:offset + 100]
            resolved = self.resolve([{k: v for k, v in ref.items() if k != "content_hash"} for _, _, ref in block], allow_archived=True)
            canonical = {(item["library_reference"]["event_id"], item["library_reference"]["revision"], item["library_reference"]["window_id"]): item for item in resolved}
            for node_id, event, ref in block:
                expected = canonical[(ref["event_id"], ref["revision"], ref["window_id"])]
                if any(event.get(k) != expected[k] for k in expected):
                    raise ValidationError("EVENT_SNAPSHOT_MISMATCH", "事件内容与锁定库版本不一致；请重新选择或转为人工副本。", f"graph.nodes.{node_id}.parameters.events")

    def import_definition(self, definition: Mapping[str, Any]) -> dict:
        source_id, revision = definition.get("id"), definition.get("revision")
        if not source_id or not revision:
            raise ValidationError("SAVED_EVENT_DEFINITION_REQUIRED", "请先保存人工事件情景，再导入事件库。")
        events = [e for n in definition.get("graph", {}).get("nodes", []) if n.get("type") == "annotation.manual_events" for e in n.get("parameters", {}).get("events", [])]
        if not events or len(events) > 100:
            raise ValidationError("EVENT_IMPORT_LIMIT", "该版本需要包含1—100个人工事件。")
        imported, skipped = [], 0
        with self.store.locked():
            data = self.store.read_unlocked()
            seen = {e["current"]["id"] for e in data["items"]}
            for e in events:
                event_id = "import_" + _digest([source_id, e["id"]])[:32]
                if event_id in seen:
                    skipped += 1
                    continue
                fields = _parse({"name": e["label"], "description": e.get("description", ""), "color": e.get("color", "#7c3aed"),
                                 "categories": suggested_categories(e["label"], e.get("description", "")),
                                 "windows": [{"id": "research", "label": "原研究窗口", "start_date": e["start_date"], "end_date": e["end_date"],
                                              "rationale": e.get("description") or "从已保存人工情景导入；事实日期与来源有待核验。"}]})
                current = self._record(fields, event_id, 1, provenance={"kind": "regime_definition", "definition_id": source_id, "revision": revision, "manual_event_id": e["id"], "categories_basis": "unreviewed_keyword_suggestion"})
                data["items"].append({"current": current, "history": []})
                seen.add(event_id)
                imported.append(event_id)
            if len(data["items"]) > 5000:
                raise ValidationError("EVENT_LIBRARY_LIMIT", "导入后超过事件库5000项上限。")
            if imported:
                self.store.write_unlocked(data)
        return {"imported": len(imported), "skipped": skipped, "event_ids": imported, "verification": "unreviewed"}
