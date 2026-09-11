"""Offline localization safety, concurrency, round-trip and cache-boundary tests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from localization.catalog import Catalog, CATALOG_ROOT
from localization.contracts import BusinessUpdate, Change, I18nError
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
    with TestClient(app) as client:
        yield client


def change(key="variables.periods_per_year.label", value="每年观察期数", locale="zh-CN"):
    return {"key": key, "value": value, "locale": locale}


def update(client, changes=None, revision=0, **extras):
    return client.patch("/api/i18n/business", json={"scope": "business", "expected_revision": revision, "changes": changes if changes is not None else [change()], **extras})


def test_catalogs_are_separate_and_system_is_immutable(client):
    before = (CATALOG_ROOT / "system.json").read_bytes()
    system = client.get("/api/i18n/catalog?scope=system&locale=en-US&page_size=200").json()
    business = client.get("/api/i18n/catalog?scope=business").json()
    assert system["items"] and business["items"]
    assert all(not row["customizable"] for row in system["items"])
    assert all(row["customizable"] for row in business["items"])
    rejected = update(client, [change("navigation.routes.settings", "改菜单")])
    assert rejected.status_code == 403
    assert rejected.json()["detail"]["code"] == "I18N_SYSTEM_READ_ONLY"
    assert (CATALOG_ROOT / "system.json").read_bytes() == before
    assert client.get("/api/i18n/settings").json()["revision"] == 0


def test_scope_spoofing_and_extra_fields_rejected(client):
    assert update(client, scope="system").status_code == 422
    assert update(client, resources={"system": {"common.save": "改写"}}).status_code == 422
    assert client.patch("/api/i18n/system", json={}).status_code == 404


def test_override_reset_locale_isolation_and_restart(client, service):
    result = update(client).json()
    assert result["revision"] == 1
    zh = client.get("/api/i18n/bundle?locale=zh-CN").json()
    en = client.get("/api/i18n/bundle?locale=en-US").json()
    assert zh["resources"]["business"]["variables.periods_per_year.label"] == "每年观察期数"
    assert en["resources"]["business"]["variables.periods_per_year.label"] == "Periods per year"
    assert zh["resources"]["system"]["navigation.routes.settings"] == "设置"
    assert LocalizationService(service.business.path.parent).state()["revision"] == 1
    assert update(client, [change(value=None)], 1).json()["revision"] == 2
    assert client.get("/api/i18n/bundle").json()["resources"]["business"]["variables.periods_per_year.label"] == "年化因子"
    assert update(client, [change(value=None)], 2).json()["revision"] == 2


@pytest.mark.parametrize("entry", [
    change("common.save"), change("variables.unknown.label"), change("system.common.save"),
    change("constructor"), change("business:variables.returns.label"),
    change(value=""), change(value="   "), change(value="a" * 121),
    change(value="<img src=x>"), change(value="$t(system:common.save)"),
    change(value="{{unexpected}}"), change(value="{{- unsafe}}"),
    change(value="bad\x00text"), change(locale="xx"),
])
def test_invalid_entries_fail_atomically(client, entry):
    result = update(client, [change("variables.returns.label", "不会写入"), entry])
    assert result.status_code in (403, 422)
    state = client.get("/api/i18n/settings").json()
    assert state["revision"] == 0 and state["overrides"] == {}


def test_duplicates_and_query_limits(client):
    assert update(client, [change(), change()]).status_code == 422
    assert client.get("/api/i18n/catalog?page_size=201").status_code == 422
    assert client.get("/api/i18n/catalog?scope=unknown").status_code == 422
    assert client.get("/api/i18n/bundle?locale=fr").status_code == 422
    result = client.get("/api/i18n/catalog?scope=business&q=periods_per_year").json()
    assert result["total"] == 2
    assert all("periods_per_year" in row["key"] for row in result["items"])


def test_placeholder_contract_and_cross_language_fallback():
    catalog = Catalog()
    catalog.tables = deepcopy(catalog.tables)
    catalog.tables["business"]["test.message"] = {"zh-CN": "数量 {{count}}"}
    catalog.validate_change(Change(key="test.message", locale="en-US", value="Count {{count}}"))
    with pytest.raises(I18nError):
        catalog.validate_change(Change(key="test.message", locale="en-US", value="Count"))
    en = catalog.bundle("en-US", {"zh-CN": {"test.message": "中文覆盖 {{count}}"}})
    assert en["resources"]["business"]["test.message"] == "数量 {{count}}"
    assert "test.message" in en["fallback_keys"]["business"]


def test_import_preview_is_read_only_and_apply_is_review_bound(client):
    package = {"format_version": 1, "scope": "business", "entries": [change()]}
    request = {"expected_revision": 0, "package": package}
    preview = client.post("/api/i18n/business/import/validate", json=request)
    assert preview.status_code == 200
    assert len(preview.json()["changes"]) == 1
    assert client.get("/api/i18n/settings").json()["revision"] == 0
    digest = preview.json()["confirmation_digest"]
    modified = deepcopy(request)
    modified["package"]["entries"][0]["value"] = "其他内容"
    assert client.post("/api/i18n/business/import/apply", json={**modified, "confirmation_digest": digest}).status_code == 409
    applied = client.post("/api/i18n/business/import/apply", json={**request, "confirmation_digest": digest})
    assert applied.status_code == 200 and applied.json()["revision"] == 1
    assert client.post("/api/i18n/business/import/apply", json={**request, "confirmation_digest": digest}).status_code == 409
    exported = client.get("/api/i18n/business/export")
    assert "attachment" in exported.headers["content-disposition"]
    assert exported.json()["scope"] == "business"
    assert exported.json()["entries"] == [change()]


def test_system_keys_cannot_be_imported(client):
    package = {"scope": "business", "entries": [change("common.save", "破坏保存")], "format_version": 1}
    response = client.post("/api/i18n/business/import/validate", json={"expected_revision": 0, "package": package})
    assert response.status_code == 403
    assert client.get("/api/i18n/settings").json()["revision"] == 0


def test_history_restore_creates_revision_and_leaves_preferences_alone(client):
    update(client)
    prefs = client.put("/api/i18n/preferences", json={"default_locale": "en-US", "expected_revision": 0}).json()
    assert prefs["preferences_revision"] == 1
    assert client.put("/api/i18n/preferences", json={"default_locale": "zh-CN", "expected_revision": 0}).status_code == 409
    update(client, [change(value="年内观察数")], revision=1)
    restored = client.post("/api/i18n/business/restore", json={"expected_revision": 2, "target_revision": 1})
    assert restored.json()["revision"] == 3
    state = client.get("/api/i18n/settings").json()
    assert state["overrides"]["zh-CN"]["variables.periods_per_year.label"] == "每年观察期数"
    assert state["default_locale"] == "en-US"
    history = client.get("/api/i18n/business/history").json()
    assert [item["revision"] for item in history["items"]] == [3, 2, 1, 0]
    assert all("overrides" not in item for item in history["items"])
    assert client.post("/api/i18n/business/restore", json={"expected_revision": 3, "target_revision": 100}).status_code == 404


def test_concurrent_services_use_single_optimistic_lock(service):
    other = LocalizationService(service.business.path.parent)
    def submit(instance):
        try:
            return instance.update(BusinessUpdate(expected_revision=0, changes=[Change(**change())]))["revision"]
        except I18nError as error:
            return error.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, [service, other]))
    assert sorted(map(str, results)) == ["1", "I18N_REVISION_CONFLICT"]


def test_bundle_etag_changes_only_with_presentation_state(client):
    first = client.get("/api/i18n/bundle")
    assert client.get("/api/i18n/bundle", headers={"If-None-Match": first.headers["etag"]}).status_code == 304
    update(client)
    second = client.get("/api/i18n/bundle", headers={"If-None-Match": first.headers["etag"]})
    assert second.status_code == 200 and second.headers["etag"] != first.headers["etag"]


def test_oversize_body_is_rejected_before_parsing(client):
    response = client.post("/api/i18n/business/import/validate", content=b" " * (1024 * 1024 + 1), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_no_financial_files_written(client, service):
    update(client)
    files = {item.name for item in service.business.path.parent.iterdir()}
    assert files <= {"i18n_business_overrides.json", "i18n_business_overrides.json.lock", "i18n_preferences.json.lock"}
    assert not (service.business.path.parent / "custom_indicators.json").exists()
