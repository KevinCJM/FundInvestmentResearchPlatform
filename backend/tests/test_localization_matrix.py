"""Offline matrix, language-registry and namespace protection regression."""
from copy import deepcopy

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localization.contracts import BusinessUpdate, Change, I18nError, LanguagesUpdate, PreferencesUpdate, RestoreRequest
from localization.languages import default_languages
from localization.service import LocalizationService
from services.localization_routes import localization_service, router


@pytest.fixture
def service(tmp_path):
    return LocalizationService(tmp_path)


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[localization_service] = lambda: service
    return TestClient(app)


def add_language(service, code="ja-JP", fallback="en-US"):
    items = [{key: item[key] for key in ("id", "label", "fallback_locale", "enabled")} for item in service.state()["locales"]]
    items.append({"id": code, "label": "日本語", "fallback_locale": fallback, "enabled": True})
    return service.set_languages(LanguagesUpdate(expected_revision=service.state()["preferences_revision"], locales=items))


def update(service, locale="ja-JP", key="axes.asset", value="資産", revision=None):
    return service.update(BusinessUpdate(expected_revision=service.state()["revision"] if revision is None else revision, changes=[Change(locale=locale, key=key, value=value)]))


def test_matrix_is_key_first_and_fallback_is_not_a_translation(service):
    add_language(service)
    result = service.matrix("business", module="axes")
    asset = next(item for item in result["items"] if item["key"] == "axes.asset")
    assert asset["code"] == "asset"
    assert asset["cells"]["zh-CN"]["value"] == "资产"
    assert asset["cells"]["en-US"]["value"] == "Asset"
    assert asset["cells"]["ja-JP"] == {"value": None, "default_value": None, "override_value": None, "source": "missing", "fallback_value": "Asset", "fallback_locale": "en-US"}
    assert result["coverage"]["ja-JP"]["missing"] == result["coverage"]["ja-JP"]["total"]
    update(service)
    assert service.matrix("business", query="資産")["items"][0]["key"] == "axes.asset"
    assert service.bundle("ja-JP")["resources"]["business"]["axes.asset"] == "資産"
    assert service.bundle("ja-JP")["resources"]["system"]["common.save"] == "Save"
    assert service.matrix("system")["items"][0]["customizable"] is False


def test_normalized_codes_persist_and_duplicates_are_rejected(service, tmp_path):
    add_language(service, "JA-jp")
    update(service, "ja-jp")
    reloaded = LocalizationService(tmp_path)
    assert reloaded.state()["locales"][-1]["id"] == "ja-JP"
    assert reloaded.state()["overrides"]["ja-JP"]["axes.asset"] == "資産"
    with pytest.raises(I18nError) as caught:
        add_language(service, "ja-jp")
    assert caught.value.code == "I18N_DUPLICATE_LOCALE"


@pytest.mark.parametrize("code", ["cimode", "../../data", "zh_CN", "ja-JP-u-ca-japanese", "x-private", "<script>", ""])
def test_invalid_codes_rejected_by_transport(client, code):
    result = client.put("/api/i18n/languages", json={"expected_revision": 0, "locales": [*default_languages(), {"id": code, "label": "Invalid", "fallback_locale": "en-US", "enabled": True}]})
    assert result.status_code == 422


def test_unknown_language_is_not_accepted_implicitly(client):
    response = client.patch("/api/i18n/business", json={"expected_revision": 0, "changes": [{"key": "axes.asset", "locale": "ja-JP", "value": "資産"}]})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "I18N_LOCALE_UNKNOWN"


def test_default_disable_and_builtin_protection(service):
    add_language(service)
    service.set_preferences(PreferencesUpdate(expected_revision=1, default_locale="ja-JP"))
    items = [{key: item[key] for key in ("id", "label", "fallback_locale", "enabled")} for item in service.state()["locales"]]
    items[-1]["enabled"] = False
    with pytest.raises(I18nError) as caught:
        service.set_languages(LanguagesUpdate(expected_revision=2, locales=items))
    assert caught.value.code == "I18N_DEFAULT_LOCALE_DISABLED"
    service.set_preferences(PreferencesUpdate(expected_revision=2, default_locale="zh-CN"))
    update(service)
    service.set_languages(LanguagesUpdate(expected_revision=3, locales=items))
    assert service.state()["overrides"]["ja-JP"]["axes.asset"] == "資産"
    with pytest.raises(I18nError):
        service.bundle("ja-JP")
    assert service.matrix("business", query="axes.asset")["items"][0]["cells"]["ja-JP"]["value"] == "資産"
    items[0]["enabled"] = False
    with pytest.raises(I18nError) as locked:
        service.set_languages(LanguagesUpdate(expected_revision=4, locales=items))
    assert locked.value.status == 403


def test_language_and_business_revisions_are_independent_and_optimistic(service):
    add_language(service)
    assert service.state()["revision"] == 0
    update(service)
    assert service.state()["preferences_revision"] == 1
    with pytest.raises(I18nError) as conflict:
        service.set_languages(LanguagesUpdate(expected_revision=0, locales=default_languages()))
    assert conflict.value.status == 409
    with pytest.raises(I18nError):
        update(service, revision=0)


def test_legacy_preferences_migrate_without_overwriting_business(service):
    with service.preferences.locked():
        service.preferences.write_unlocked({"schema_version": 1, "revision": 3, "default_locale": "en-US", "items": []})
    update(service, "zh-CN", value="资产名称")
    add_language(service)
    assert service.state()["default_locale"] == "en-US"
    assert service.state()["overrides"]["zh-CN"]["axes.asset"] == "资产名称"
    assert service.state()["preferences_revision"] == 4


def test_custom_history_restore_and_reset_do_not_cross_locales(service):
    add_language(service)
    update(service)
    update(service, "zh-CN", value="专属中文")
    service.restore(RestoreRequest(expected_revision=2, target_revision=1))
    assert service.state()["overrides"] == {"ja-JP": {"axes.asset": "資産"}}
    update(service, value=None)
    assert service.bundle("ja-JP")["resources"]["business"]["axes.asset"] == "Asset"


def test_import_and_namespace_validation_remain_atomic(service, client):
    add_language(service)
    pack = {"format_version": 1, "scope": "business", "entries": [{"key": "axes.asset", "locale": "ja-JP", "value": "資産"}]}
    request = {"expected_revision": 0, "package": pack}
    preview = client.post("/api/i18n/business/import/validate", json=request).json()
    assert service.state()["revision"] == 0
    result = client.post("/api/i18n/business/import/apply", json={**request, "confirmation_digest": preview["confirmation_digest"]})
    assert result.status_code == 200
    assert client.get("/api/i18n/business/export").json()["entries"] == pack["entries"]
    before = deepcopy(service.state())
    response = client.patch("/api/i18n/business", json={"expected_revision": 1, "changes": [*pack["entries"], {"key": "common.save", "locale": "ja-JP", "value": "改菜单"}]})
    assert response.status_code == 403
    assert service.state() == before
    request["expected_revision"] = 1
    preview = client.post("/api/i18n/business/import/validate", json=request).json()
    add_language(service, "fr-FR")
    assert client.post("/api/i18n/business/import/apply", json={**request, "confirmation_digest": preview["confirmation_digest"]}).status_code == 409


def test_matrix_sorting_and_search_cover_code_and_language_columns(client, service):
    add_language(service)
    update(service, "ja-JP", "axes.asset", "資産")
    ascending = client.get("/api/i18n/matrix?scope=business&module=axes&sort_by=code&sort_dir=asc").json()
    descending = client.get("/api/i18n/matrix?scope=business&module=axes&sort_by=code&sort_dir=desc").json()
    assert [row["code"] for row in ascending["items"]] == sorted(row["code"] for row in ascending["items"])
    assert [row["code"] for row in descending["items"]] == list(reversed([row["code"] for row in ascending["items"]]))
    english = client.get("/api/i18n/matrix?scope=business&module=axes&sort_by=en-US&sort_dir=asc").json()
    present = [row["cells"]["en-US"]["value"] for row in english["items"] if row["cells"]["en-US"]["value"] is not None]
    assert present == sorted(present, key=str.casefold)
    assert client.get("/api/i18n/matrix?scope=business&q=axes.asset").json()["items"][0]["key"] == "axes.asset"
    assert client.get("/api/i18n/matrix?scope=business&q=資産").json()["items"][0]["key"] == "axes.asset"
    invalid = client.get("/api/i18n/matrix?scope=business&sort_by=unknown-column")
    assert invalid.status_code == 422
    assert invalid.json()["detail"]["code"] == "I18N_INVALID_SORT"


def test_matrix_pagination_and_language_order(client, service):
    add_language(service)
    items = service.state()["locales"]
    reordered = [{key: item[key] for key in ("id", "label", "fallback_locale", "enabled")} for item in reversed(items)]
    assert client.put("/api/i18n/languages", json={"expected_revision": 1, "locales": reordered}).status_code == 200
    first = client.get("/api/i18n/matrix?scope=business&page_size=2").json()
    second = client.get("/api/i18n/matrix?scope=business&page_size=2&page=2").json()
    assert [item["id"] for item in first["locales"]] == ["ja-JP", "en-US", "zh-CN"]
    assert not {row["key"] for row in first["items"]} & {row["key"] for row in second["items"]}
    assert first["total"] > 2
    assert client.get("/api/i18n/matrix?page_size=999").status_code == 422


def test_fallback_metadata_survives_pending_reset_and_legacy_catalog_agrees(service):
    add_language(service)
    update(service, "en-US", value="Private English name")
    update(service)
    cell = service.matrix("business", query="axes.asset")["items"][0]["cells"]["ja-JP"]
    assert cell["value"] == "資産"
    assert cell["default_value"] is None
    assert cell["fallback_value"] == "Asset"
    update(service, value=None)
    old_catalog = service.list_entries("business", "ja-JP", query="axes.asset")["items"][0]
    assert old_catalog["missing"]
    assert old_catalog["effective_value"] == "Asset"
    assert service.bundle("ja-JP")["resources"]["business"]["axes.asset"] == "Asset"


def test_matrix_search_keeps_default_names_discoverable_after_overrides(service):
    update(service, "zh-CN", key="variables.periods_per_year.label", value="每年观察期数")
    keys = {row["key"] for row in service.matrix("business", query="年化因子")["items"]}
    assert "variables.periods_per_year.label" in keys
    assert "variables.periods_per_year.label" in {row["key"] for row in service.matrix("business", query="每年观察期数")["items"]}
