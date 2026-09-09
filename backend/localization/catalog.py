"""Load immutable presentation catalogs from the shared locale directory."""
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path

from .contracts import I18nError

LOCALES = ("zh-CN", "en-US")
DEFAULT_LOCALE = "zh-CN"
LOCALE_LABELS = {"zh-CN": "简体中文", "en-US": "English"}
CATALOG_ROOT = Path(__file__).resolve().parents[2] / "locales"
PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z][A-Za-z0-9_]*)\s*\}\}")


def placeholders(value: str) -> tuple[str, ...]:
    return tuple(sorted(set(PLACEHOLDER.findall(value))))


def _load(name: str) -> dict:
    return json.loads((CATALOG_ROOT / f"{name}.json").read_text(encoding="utf-8"))


class Catalog:
    def __init__(self):
        self.tables = {
            "system": {**_load("system"), **{f"navigation.routes.{key}": value for key, value in _load("navigation").items()}},
            "business": _load("business"),
        }
        for scope, entries in self.tables.items():
            for key, values in entries.items():
                if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,239}", key):
                    raise ValueError(f"Invalid key in {scope}: {key}")
                if not values.get(DEFAULT_LOCALE) or set(values) - set(LOCALES):
                    raise ValueError(f"Invalid default locale: {key}")
                if any(placeholders(text) != placeholders(values[DEFAULT_LOCALE]) for text in values.values()):
                    raise ValueError(f"Inconsistent placeholders: {key}")
        self.version = hashlib.sha256(json.dumps(self.tables, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def validate_change(self, change) -> None:
        key, value = change.key, change.value
        if key in self.tables["system"] or key.startswith("system."):
            raise I18nError("I18N_SYSTEM_READ_ONLY", "系统翻译只读。", 403, key)
        defaults = self.tables["business"].get(key)
        if defaults is None:
            raise I18nError("I18N_INVALID_ENTRY", "未登记的业务词条。", field=key)
        if value is None:
            return
        limit = 1600 if key.endswith(".description") else 120
        if not value.strip() or len(value) > limit:
            raise I18nError("I18N_INVALID_ENTRY", f"译文长度应为 1 至 {limit} 个字符。", field=key)
        if re.search(r"<\s*/?\s*[A-Za-z!][^>]*>|\$t\(", value) or any(ord(char) < 32 and char not in "\n\t" for char in value):
            raise I18nError("I18N_INVALID_ENTRY", "译文必须为纯文本。", field=key)
        remainder = PLACEHOLDER.sub("", value)
        if placeholders(value) != placeholders(defaults[DEFAULT_LOCALE]) or "{{" in remainder or "}}" in remainder:
            raise I18nError("I18N_INVALID_ENTRY", "译文占位符与默认词条不一致。", field=key)

    def bundle(self, locale: str, overrides: dict, fallback_locale: str = DEFAULT_LOCALE) -> dict:
        resources, fallback_keys = {}, {}
        for scope, table in self.tables.items():
            resources[scope], fallback_keys[scope] = {}, []
            for key, values in table.items():
                custom = overrides.get(locale, {}).get(key) if scope == "business" else None
                resources[scope][key] = custom if custom is not None else values.get(locale, values.get(fallback_locale, values[DEFAULT_LOCALE]))
                if custom is None and locale not in values:
                    fallback_keys[scope].append(key)
        return {"locale": locale, "catalog_version": self.version, "resources": resources, "fallback_keys": fallback_keys}

    def matrix_rows(self, scope: str, languages: list[dict], overrides: dict) -> list[dict]:
        """A fallback is a preview, never a translation in an empty language cell."""
        rows = []
        for key, defaults in sorted(self.tables[scope].items()):
            cells = {}
            for language in languages:
                locale = language["id"]
                custom = overrides.get(locale, {}).get(key) if scope == "business" else None
                default = defaults.get(locale)
                value = custom if custom is not None else default
                fallback_locale = language["fallback_locale"]
                if fallback_locale not in defaults:
                    fallback_locale = DEFAULT_LOCALE
                cells[locale] = {
                    "value": value, "default_value": default, "override_value": custom,
                    "source": "custom" if custom is not None else "builtin" if default is not None else "missing",
                    "fallback_value": defaults[fallback_locale] if default is None else None,
                    "fallback_locale": fallback_locale if default is None else None,
                }
            module, _, code = key.partition(".")
            rows.append({"key": key, "code": code or key, "scope": scope, "module": module,
                         "cells": cells, "customizable": scope == "business",
                         "max_length": 1600 if key.endswith(".description") else 120,
                         "placeholders": list(placeholders(defaults[DEFAULT_LOCALE])),
                         "usage": [module]})
        return rows

    def rows(self, scope: str, locale: str, overrides: dict, fallback_locale: str = DEFAULT_LOCALE) -> list[dict]:
        rows = []
        for key, values in self.tables[scope].items():
            custom = overrides.get(locale, {}).get(key) if scope == "business" else None
            module = key.split(".", 1)[0]
            rows.append({
                "key": key, "scope": scope, "module": module,
                "label": values[DEFAULT_LOCALE], "translations": dict(values),
                "default_value": values.get(locale, values.get(fallback_locale, values[DEFAULT_LOCALE])),
                "override_value": custom,
                "effective_value": custom if custom is not None else values.get(locale, values.get(fallback_locale, values[DEFAULT_LOCALE])),
                "customizable": scope == "business", "missing": locale not in values and custom is None,
                "max_length": 1600 if key.endswith(".description") else 120,
                "placeholders": list(placeholders(values[DEFAULT_LOCALE])),
                "usage": ["indicator_metadata", "indicator_graph"] if module in {"variables", "operators", "parameters", "valueTypes", "axes", "nodeKinds"} else [module],
            })
        return sorted(rows, key=lambda row: row["key"])


@lru_cache(maxsize=1)
def default_catalog() -> Catalog:
    return Catalog()
