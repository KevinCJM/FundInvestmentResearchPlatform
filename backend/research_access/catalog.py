"""Versioned catalog snapshots derived from the existing indicator service.

The complete catalog participates in search, the reported ``total`` and the
revision fingerprint; only the entries returned to a caller are limited.  A
single request therefore never silently loses the tail of the catalog.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Optional

_SUMMARY_FIELDS = (
    "id",
    "name",
    "description",
    "revision",
    "context_kind",
    "result_kind",
    "indicator_type",
    "unit",
    "display_format",
    "source",
    "category",
    "rolling_series_compatibility",
    "annual_risk_free_rate_percent",
    "methodology",
)


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _summary(item: dict[str, Any]) -> dict[str, Any]:
    summary = {field: item.get(field) for field in _SUMMARY_FIELDS if item.get(field) is not None}
    presentation = item.get("presentation") or {}
    if isinstance(presentation, dict) and presentation.get("category"):
        summary["category"] = presentation["category"]
    return summary


def build_catalog(service: Any) -> dict[str, Any]:
    """Build one complete catalog snapshot; never computes indicator values."""

    meta = service.meta()
    listing = service.list_indicators()
    items = [_summary(item) for item in listing.get("items", [])]
    identity = {
        "items": sorted((item.get("id"), item.get("revision")) for item in items),
        "dsl_version": meta.get("dsl_version"),
        "operator_registry_version": meta.get("operator_registry_version"),
        "variable_registry_version": meta.get("variable_registry_version"),
        "engine_version": meta.get("engine_version"),
    }
    version = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()[:16]
    return {
        "version": version,
        "engine_version": meta.get("engine_version"),
        "dsl_version": meta.get("dsl_version"),
        "operator_registry_version": meta.get("operator_registry_version"),
        "variable_registry_version": meta.get("variable_registry_version"),
        "periods": [entry.get("id") for entry in meta.get("periods", [])],
        "items": items,
        "total": len(items),
    }


def matched_items(
    catalog: dict[str, Any],
    *,
    query: str = "",
    context_kind: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Every matching entry, best match first; never truncated by a page size."""

    terms = [term for term in re.split(r"[\s,，]+", query.casefold().strip()) if term]
    matches = []
    for item in catalog.get("items", []):
        if context_kind and item.get("context_kind") not in (None, context_kind):
            continue
        if context_kind and item.get("domains") and context_kind not in item["domains"]:
            continue
        text = ' '.join(str(item.get(key) or '') for key in ("id", "name", "label", "description", "signature", "mathematical_essence")).casefold()
        score = sum(term in text for term in terms)
        if terms and not score:
            continue
        matches.append((score, dict(item)))
    matches.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in matches]


def search_catalog(
    catalog: dict[str, Any],
    *,
    query: str = "",
    context_kind: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return matched_items(catalog, query=query, context_kind=context_kind)[:limit]
