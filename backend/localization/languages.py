"""Bounded language registry. Display configuration never changes calculation IDs."""
from __future__ import annotations

import re
from copy import deepcopy

from .errors import I18nError

BUILTIN_LOCALES = ("zh-CN", "en-US")
DEFAULT_LOCALE = "zh-CN"
MAX_LANGUAGES = 24
# Supported tag subset: language, optional script, optional region. No extensions,
# private-use tags or i18next's special modes may become workspace languages.
TAG = re.compile(r"^([A-Za-z]{2,3})(?:-([A-Za-z]{4}))?(?:-([A-Za-z]{2}|[0-9]{3}))?$")
BUILTIN_LANGUAGES = (
    {"id": "zh-CN", "label": "简体中文", "fallback_locale": "zh-CN", "enabled": True},
    {"id": "en-US", "label": "English", "fallback_locale": "zh-CN", "enabled": True},
)


def normalize_locale(value: str) -> str:
    if not isinstance(value, str) or not (match := TAG.fullmatch(value.strip())):
        raise ValueError("Use a language code such as ja-JP or zh-Hant-TW.")
    language, script, region = match.groups()
    return "-".join(part for part in (language.lower(), script.title() if script else None, region.upper() if region else None) if part)


def default_languages() -> list[dict]:
    return deepcopy(list(BUILTIN_LANGUAGES))


def validate_languages(items: list[dict], default_locale: str = DEFAULT_LOCALE) -> list[dict]:
    if not isinstance(items, list) or not 2 <= len(items) <= MAX_LANGUAGES:
        raise I18nError("I18N_INVALID_LOCALE", "语言数量应为 2 至 24。", field="locales")
    normalized, seen = [], set()
    for item in items:
        try:
            locale = normalize_locale(item["id"])
            label = item["label"]
            fallback = item["fallback_locale"]
            enabled = item["enabled"]
        except (KeyError, TypeError, ValueError) as exc:
            raise I18nError("I18N_INVALID_LOCALE", "语言配置不正确。", field="locales") from exc
        if locale in seen:
            raise I18nError("I18N_DUPLICATE_LOCALE", "语言代码不能重复。", field=locale)
        if not isinstance(label, str) or not label.strip() or len(label) > 60 or re.search(r"[<>\x00-\x1f]", label):
            raise I18nError("I18N_INVALID_LOCALE", "请填写 1 至 60 个字符的纯文本语言名称。", field=locale)
        if fallback not in BUILTIN_LOCALES or type(enabled) is not bool:
            raise I18nError("I18N_INVALID_LOCALE", "回退语言必须是内置语言。", field=locale)
        seen.add(locale)
        normalized.append({"id": locale, "label": label.strip(), "fallback_locale": fallback, "enabled": enabled})
    by_id = {item["id"]: item for item in normalized}
    for builtin in BUILTIN_LANGUAGES:
        if by_id.get(builtin["id"]) != builtin:
            raise I18nError("I18N_BUILTIN_LOCALE_LOCKED", "内置语言不能删除、停用或修改其配置。", 403, builtin["id"])
    if default_locale not in by_id or not by_id[default_locale]["enabled"]:
        raise I18nError("I18N_DEFAULT_LOCALE_DISABLED", "请先更换工作区默认语言，再停用此语言。", field="default_locale")
    return normalized


def find_language(items: list[dict], locale: str, *, enabled: bool = False) -> dict:
    try:
        locale = normalize_locale(locale)
    except ValueError as exc:
        raise I18nError("I18N_INVALID_LOCALE", "语言代码格式不正确。", field="locale") from exc
    item = next((item for item in items if item["id"] == locale), None)
    if item is None:
        raise I18nError("I18N_LOCALE_UNKNOWN", "请先添加该语言列，再填写或导入译文。", field=locale)
    if enabled and not item["enabled"]:
        raise I18nError("I18N_LOCALE_DISABLED", "此语言尚未启用。", field=locale)
    return item
