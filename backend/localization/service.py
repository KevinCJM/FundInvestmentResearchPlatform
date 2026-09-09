"""Versioned workspace presentation settings with a read-only system namespace."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path

from custom_indicators.repository import AtomicJsonStore, utc_now
from .catalog import DEFAULT_LOCALE, Catalog, default_catalog
from .contracts import Change, I18nError
from .languages import BUILTIN_LOCALES, default_languages, find_language, validate_languages


class LocalizationService:
    def __init__(self, root: Path, catalog: Catalog | None = None):
        self.catalog = catalog or default_catalog()
        self.business = AtomicJsonStore(root / "i18n_business_overrides.json")
        self.preferences = AtomicJsonStore(root / "i18n_preferences.json")

    @staticmethod
    def _check_revision(actual: int, expected: int):
        if actual != expected:
            raise I18nError("I18N_REVISION_CONFLICT", "翻译配置已被其他操作更新，请刷新后核对。", 409, "expected_revision")

    def _preferences_unlocked(self) -> dict:
        data = self.preferences.read_unlocked()
        if not self.preferences.path.exists():
            return {"schema_version": 2, "revision": 0, "default_locale": DEFAULT_LOCALE, "locales": default_languages(), "items": []}
        if data.get("schema_version") not in (1, 2) or type(data.get("revision")) is not int or data["revision"] < 0:
            raise I18nError("I18N_STORAGE_INVALID", "语言配置文件格式不兼容。", 500)
        languages = validate_languages(data.get("locales", default_languages()), data.get("default_locale", DEFAULT_LOCALE))
        return {**data, "schema_version": 2, "locales": languages}

    def _business_unlocked(self, languages: list[dict]) -> dict:
        payload = self.business.read_unlocked()
        if not self.business.path.exists():
            return {"schema_version": 1, "revision": 0, "overrides": {}, "items": [{"revision": 0, "overrides": {}, "at": None, "action": "initial", "reason": "", "change_count": 0}]}
        if payload.get("schema_version") != 1 or type(payload.get("revision")) is not int or payload["revision"] < 0 or not isinstance(payload.get("overrides"), dict) or not isinstance(payload.get("items"), list):
            raise I18nError("I18N_STORAGE_INVALID", "业务翻译文件格式不兼容。", 500)
        # Files are not executable, but invalid external edits must fail closed.
        for locale, entries in payload["overrides"].items():
            language = find_language(languages, locale)
            if language["id"] != locale or not isinstance(entries, dict):
                raise I18nError("I18N_STORAGE_INVALID", "业务翻译文件格式不兼容。", 500)
            for key, value in entries.items():
                if not isinstance(value, str):
                    raise I18nError("I18N_STORAGE_INVALID", "业务翻译文件格式不兼容。", 500)
                self.catalog.validate_change(Change(key=key, locale=locale, value=value))
        return payload

    def state(self) -> dict:
        # All operations acquire locks in this order, including language writes.
        with self.preferences.locked(), self.business.locked():
            preferences = self._preferences_unlocked()
            data = self._business_unlocked(preferences["locales"])
        return {
            "revision": data["revision"], "overrides": deepcopy(data["overrides"]),
            "preferences_revision": preferences.get("revision", 0),
            "default_locale": preferences.get("default_locale", DEFAULT_LOCALE),
            "locales": [{**item, "builtin": item["id"] in BUILTIN_LOCALES, "system_pack": item["id"] in BUILTIN_LOCALES} for item in preferences["locales"]],
            "catalog_version": self.catalog.version,
        }

    def set_preferences(self, request) -> dict:
        with self.preferences.locked():
            current = self._preferences_unlocked()
            revision = current["revision"]
            self._check_revision(revision, request.expected_revision)
            find_language(current["locales"], request.default_locale, enabled=True)
            if current.get("default_locale", DEFAULT_LOCALE) == request.default_locale:
                return {"preferences_revision": revision, "default_locale": request.default_locale}
            updated = {**current, "revision": revision + 1, "default_locale": request.default_locale, "updated_at": utc_now()}
            self.preferences.write_unlocked(updated)
        return {"preferences_revision": updated["revision"], "default_locale": updated["default_locale"]}

    def set_languages(self, request) -> dict:
        with self.preferences.locked():
            current = self._preferences_unlocked()
            self._check_revision(current["revision"], request.expected_revision)
            languages = validate_languages([item.model_dump() for item in request.locales], current["default_locale"])
            if {item["id"] for item in current["locales"]} - {item["id"] for item in languages}:
                raise I18nError("I18N_LOCALE_DELETE_FORBIDDEN", "请停用语言，不要删除语言代码；已有译文将被保留。", field="locales")
            if languages != current["locales"]:
                current = {**current, "revision": current["revision"] + 1, "locales": languages, "updated_at": utc_now()}
                self.preferences.write_unlocked(current)
        return {"preferences_revision": current["revision"], "default_locale": current["default_locale"], "locales": current["locales"]}

    def bundle(self, locale: str) -> dict:
        state = self.state()
        language = find_language(state["locales"], locale, enabled=True)
        return {**self.catalog.bundle(language["id"], state["overrides"], language["fallback_locale"]), "revision": state["revision"], "preferences_revision": state["preferences_revision"], "default_locale": state["default_locale"], "fallback_locale": language["fallback_locale"]}

    def list_entries(self, scope: str, locale: str, query: str = "", module: str = "", status: str = "all", page: int = 1, page_size: int = 50) -> dict:
        state = self.state()
        language = find_language(state["locales"], locale)
        locale = language["id"]
        rows = self.catalog.rows(scope, locale, state["overrides"], language["fallback_locale"])
        modules = sorted({row["module"] for row in rows})
        coverage = {"total": len(rows), "missing": sum(row["missing"] for row in rows)}
        query = query.strip().casefold()
        filtered = [row for row in rows if (
            (not module or row["module"] == module)
            and (status != "customized" or row["override_value"] is not None)
            and (status != "missing" or row["missing"])
            and (not query or query in " ".join([row["key"], *row["translations"].values(), row["effective_value"]]).casefold())
        )]
        offset = (page - 1) * page_size
        return {"scope": scope, "locale": locale, "revision": state["revision"], "catalog_version": self.catalog.version, "items": filtered[offset:offset + page_size], "total": len(filtered), "page": page, "page_size": page_size, "modules": modules, "coverage": coverage}

    def matrix(self, scope: str, query: str = "", module: str = "", status: str = "all", page: int = 1, page_size: int = 50, sort_by: str = "code", sort_dir: str = "asc") -> dict:
        state = self.state()
        rows = self.catalog.matrix_rows(scope, state["locales"], state["overrides"])
        modules = sorted({row["module"] for row in rows})
        coverage = {item["id"]: {"total": len(rows), "missing": sum(row["cells"][item["id"]]["source"] == "missing" for row in rows)} for item in state["locales"]}
        query = query.strip().casefold()
        filtered = [row for row in rows if (
            (not module or row["module"] == module)
            and (status != "customized" or any(cell["source"] == "custom" for cell in row["cells"].values()))
            and (status != "missing" or any(cell["source"] == "missing" for cell in row["cells"].values()))
            and (not query or query in " ".join([row["key"], row["code"], *[cell["value"] or "" for cell in row["cells"].values()], *[cell["fallback_value"] or "" for cell in row["cells"].values()], *self.catalog.tables[scope][row["key"]].values()]).casefold())
        )]
        locale_ids = {item["id"] for item in state["locales"]}
        if sort_by not in {"code", "key"} | locale_ids:
            raise I18nError("I18N_INVALID_SORT", "排序字段必须是系统代码或已登记的语言列。", field="sort_by")
        if sort_dir not in {"asc", "desc"}:
            raise I18nError("I18N_INVALID_SORT", "排序方向必须是 asc 或 desc。", field="sort_dir")
        reverse = sort_dir == "desc"
        if sort_by in locale_ids:
            present = [row for row in filtered if row["cells"][sort_by]["value"] is not None]
            missing = [row for row in filtered if row["cells"][sort_by]["value"] is None]
            present.sort(key=lambda row: (row["cells"][sort_by]["value"].casefold(), row["key"].casefold()), reverse=reverse)
            missing.sort(key=lambda row: row["key"].casefold())
            filtered = present + missing
        else:
            filtered.sort(key=lambda row: ((row["code"] if sort_by == "code" else row["key"]).casefold(), row["key"].casefold()), reverse=reverse)
        offset = (page - 1) * page_size
        return {"scope": scope, "revision": state["revision"], "preferences_revision": state["preferences_revision"], "catalog_version": self.catalog.version, "locales": state["locales"], "items": filtered[offset:offset + page_size], "total": len(filtered), "page": page, "page_size": page_size, "modules": modules, "coverage": coverage, "sort_by": sort_by, "sort_dir": sort_dir}

    def _validate_changes(self, changes, languages: list[dict]) -> list[Change]:
        seen = set()
        validated = []
        for change in changes:
            find_language(languages, change.locale)
            identity = (change.locale, change.key)
            if identity in seen:
                raise I18nError("I18N_INVALID_ENTRY", "同一语言的词条不能重复提交。", field=change.key)
            seen.add(identity)
            self.catalog.validate_change(change)
            validated.append(change)
        return validated

    @staticmethod
    def _diff(current: dict, changes: list[Change]) -> list[dict]:
        return [{"key": change.key, "locale": change.locale, "before": current.get(change.locale, {}).get(change.key), "after": change.value}
                for change in changes if current.get(change.locale, {}).get(change.key) != change.value]

    def _commit_unlocked(self, current: dict, changes: list[Change], action: str, reason: str) -> dict:
        diff = self._diff(current["overrides"], changes)
        if not diff:
            return {"revision": current["revision"], "changes": [], "overrides": deepcopy(current["overrides"])}
        overrides = deepcopy(current["overrides"])
        for change in diff:
            entries = overrides.setdefault(change["locale"], {})
            if change["after"] is None:
                entries.pop(change["key"], None)
            else:
                entries[change["key"]] = change["after"]
            if not entries:
                overrides.pop(change["locale"], None)
        revision = current["revision"] + 1
        history = [*current["items"], {"revision": revision, "overrides": deepcopy(overrides), "at": utc_now(), "action": action, "reason": reason, "change_count": len(diff)}][-50:]
        self.business.write_unlocked({"schema_version": 1, "revision": revision, "overrides": overrides, "items": history})
        return {"revision": revision, "changes": diff, "overrides": overrides}

    def update(self, request) -> dict:
        with self.preferences.locked(), self.business.locked():
            languages = self._preferences_unlocked()["locales"]
            changes = self._validate_changes(request.changes, languages)
            current = self._business_unlocked(languages)
            self._check_revision(current["revision"], request.expected_revision)
            return self._commit_unlocked(current, changes, "edit", request.reason)

    def export(self) -> dict:
        state = self.state()
        return {"format_version": 1, "scope": "business", "catalog_version": self.catalog.version, "entries": [
            {"locale": locale, "key": key, "value": value}
            for locale, entries in sorted(state["overrides"].items()) for key, value in sorted(entries.items())
        ]}

    def _import_digest(self, package, expected_revision: int, preferences_revision: int) -> str:
        contents = {"catalog": self.catalog.version, "revision": expected_revision, "preferences_revision": preferences_revision, "package": package.model_dump()}
        return hashlib.sha256(json.dumps(contents, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def validate_import(self, request) -> dict:
        with self.preferences.locked(), self.business.locked():
            preferences = self._preferences_unlocked()
            changes = self._validate_changes(request.package.entries, preferences["locales"])
            current = self._business_unlocked(preferences["locales"])
            self._check_revision(current["revision"], request.expected_revision)
            diff = self._diff(current["overrides"], changes)
            digest = self._import_digest(request.package, request.expected_revision, preferences["revision"])
        return {"valid": True, "revision": current["revision"], "changes": diff, "confirmation_digest": digest, "catalog_changed": request.package.catalog_version not in (None, self.catalog.version)}

    def apply_import(self, request) -> dict:
        # Digest confirms the reviewed content; it is not an authorization token.
        # Every write independently revalidates all entries and the current revision.
        with self.preferences.locked(), self.business.locked():
            preferences = self._preferences_unlocked()
            expected = self._import_digest(request.package, request.expected_revision, preferences["revision"])
            if not hmac.compare_digest(expected, request.confirmation_digest):
                raise I18nError("I18N_INVALID_IMPORT", "导入内容、语言配置或目录已变化，请重新校验。", 409)
            changes = self._validate_changes(request.package.entries, preferences["locales"])
            current = self._business_unlocked(preferences["locales"])
            self._check_revision(current["revision"], request.expected_revision)
            return self._commit_unlocked(current, changes, "import", "")

    def history(self) -> dict:
        with self.preferences.locked(), self.business.locked():
            current = self._business_unlocked(self._preferences_unlocked()["locales"])
        return {"revision": current["revision"], "items": [{key: value for key, value in entry.items() if key != "overrides"} for entry in reversed(current["items"])]}

    def restore(self, request) -> dict:
        with self.preferences.locked(), self.business.locked():
            languages = self._preferences_unlocked()["locales"]
            current = self._business_unlocked(languages)
            self._check_revision(current["revision"], request.expected_revision)
            target = next((item for item in current["items"] if item["revision"] == request.target_revision), None)
            if target is None:
                raise I18nError("I18N_HISTORY_NOT_FOUND", "此历史修订不存在或已超出保留范围。", 404)
            target_values = target["overrides"]
            changes = [Change(locale=locale, key=key, value=target_values.get(locale, {}).get(key))
                       for locale in sorted(set(target_values) | set(current["overrides"]))
                       for key in sorted(set(target_values.get(locale, {})) | set(current["overrides"].get(locale, {})))]
            self._validate_changes(changes, languages)
            return self._commit_unlocked(current, changes, "restore", str(request.target_revision))
