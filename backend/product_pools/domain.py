"""Pure validation and identity helpers shared by product-pool services."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any, Optional

from .constants import ELIGIBLE_RESEARCH_STATUSES, ELIGIBLE_USAGE_STATUSES
from .errors import ProductPoolValidationError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def today() -> date:
    return datetime.now(timezone.utc).date()


def content_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_date(value: Any, *, field: str, required: bool = True) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ProductPoolValidationError("DATE_REQUIRED", "日期不能为空。", field=field)
        return None
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ProductPoolValidationError(
            "INVALID_DATE",
            "日期必须使用 YYYY-MM-DD 格式。",
            field=field,
        ) from exc


def trimmed(value: Any, *, field: str, maximum: int, required: bool = False) -> str:
    text = str(value or "").strip()
    if required and not text:
        raise ProductPoolValidationError("FIELD_REQUIRED", "该字段不能为空。", field=field)
    if len(text) > maximum:
        raise ProductPoolValidationError(
            "FIELD_TOO_LONG",
            f"该字段不能超过 {maximum} 个字符。",
            field=field,
        )
    return text


def product_key(kind: Any, product_id: Any) -> tuple[str, str]:
    return str(kind or "").strip().lower(), str(product_id or "").strip()


def member_eligibility(member: dict[str, Any], on_date: date) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    research_status = str(member.get("research_status") or "candidate")
    usage_status = str(member.get("usage_status") or "normal")
    if research_status not in ELIGIBLE_RESEARCH_STATUSES:
        reasons.append(f"研究状态为 {research_status}")
    if usage_status not in ELIGIBLE_USAGE_STATUSES:
        reasons.append(f"使用状态为 {usage_status}")
    valid_until = member.get("valid_until")
    if valid_until:
        try:
            if date.fromisoformat(str(valid_until)) < on_date:
                reasons.append("产品准入有效期已结束")
        except ValueError:
            reasons.append("产品有效期无效")
    return not reasons, reasons


__all__ = [
    "content_hash",
    "member_eligibility",
    "parse_date",
    "product_key",
    "today",
    "trimmed",
    "utc_now",
]
