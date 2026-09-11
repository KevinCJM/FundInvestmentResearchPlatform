"""Offline regression for source contracts, wire samples and write boundaries."""
from __future__ import annotations

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
sys.path.insert(0, str(ROOT / "backend")) if str(ROOT / "backend") not in sys.path else None

from backend.data_sources import runtime, service
from backend.data_sources.batches import capture_batch, recent_batches
from backend.data_sources.mapping import decode_response, map_table, preview, validate_mapping
from backend.data_sources.models import CenterError, DownloadPolicy, InterfaceConfig, ResponseFormat, SourceConfig
from backend.data_sources.presets import default_interfaces, default_source
from backend.data_sources.quota import SharedQuota
from backend.data_sources.store import SourceStore
from backend.data_sources.transport import public_address, TransientSourceError
from backend.services import data_source_routes as routes

PRESETS = {item.api_name: item for item in default_interfaces()}

def test_recent_batches_use_time_index(tmp_path):
    store = SourceStore(tmp_path)
    with store.connection() as db:
        plan = db.execute('EXPLAIN QUERY PLAN SELECT result FROM source_run ORDER BY created_at DESC LIMIT 30').fetchall()
    assert any('source_run_created' in row[3] for row in plan)
    assert not any('TEMP B-TREE' in row[3] for row in plan)


def test_waiting_quota_does_not_write_or_count_expired_leases(tmp_path):
    store = SourceStore(tmp_path)
    quota = SharedQuota(store)
    policy = DownloadPolicy(requests_per_minute=1, min_interval_seconds=0, max_concurrency=1)
    levels = [('source:s', policy), ('api:s:a', policy)]
    assert quota.reserve(levels, 1, 1000, 'first') == 0
    with store.connection() as db:
        db.execute("INSERT INTO source_quota VALUES ('unrelated',0,1)")
        db.execute("INSERT INTO source_lease VALUES ('expired','source:s',1001)")
    assert quota.reserve(levels, 1, 1002, 'waiting') > 0
    with store.connection() as db:
        # Waiting is read-only, even if housekeeping has expired rows.
        assert db.execute("SELECT COUNT(*) FROM source_quota WHERE quota_key='unrelated'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM source_lease WHERE lease_id='waiting'").fetchone()[0] == 0
    assert quota.reserve(levels, 1, 1062, 'next') == 0
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM source_quota WHERE quota_key='unrelated'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM source_lease WHERE lease_id='expired'").fetchone()[0] == 0


def test_concurrent_quota_never_over_reserves(tmp_path):
    store = SourceStore(tmp_path)
    quota = SharedQuota(store)
    policy = DownloadPolicy(requests_per_minute=4, min_interval_seconds=0, max_concurrency=4)
    levels = [('source:s', policy), ('api:s:a', policy)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        waits = list(pool.map(lambda i: quota.reserve(levels, 1, 1000, str(i)), range(32)))
    assert waits.count(0) == 4
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_quota').fetchone()[0] == 8
        assert db.execute('SELECT COUNT(*) FROM source_lease').fetchone()[0] == 8


@pytest.mark.parametrize('sqlite_code,expected', [
    (sqlite3.SQLITE_BUSY, 'SOURCE_DB_BUSY'),
    (sqlite3.SQLITE_BUSY | (2 << 8), 'SOURCE_DB_BUSY'),
    (sqlite3.SQLITE_LOCKED, 'SOURCE_DB_BUSY'),
    (sqlite3.SQLITE_FULL, 'SOURCE_DB_FULL'),
    (sqlite3.SQLITE_IOERR, 'SOURCE_DB_IO'),
    (sqlite3.SQLITE_CANTOPEN, 'SOURCE_DB_OPEN'),
    (sqlite3.SQLITE_READONLY, 'SOURCE_DB_READONLY'),
    (sqlite3.SQLITE_ERROR, 'SOURCE_DB_OPERATIONAL'),
])
def test_database_error_is_safe_and_transaction_rolled_back(tmp_path, sqlite_code, expected):
    store = SourceStore(tmp_path)
    error = sqlite3.OperationalError('private credential / sensitive SQL must not leak')
    error.sqlite_errorcode = sqlite_code
    with pytest.raises(CenterError) as caught:
        with store.connection() as db:
            db.execute("INSERT INTO source_quota VALUES ('rollback',1,1)")
            raise error
    assert caught.value.code == expected
    assert 'private' not in caught.value.message
    assert 'sensitive' not in caught.value.message
    with store.connection() as db:
        assert db.execute('SELECT COUNT(*) FROM source_quota').fetchone()[0] == 0


def test_real_database_busy_has_bounded_safe_error(tmp_path, monkeypatch):
    store = SourceStore(tmp_path)
    connect = sqlite3.connect
    locker = connect(store.path)
    locker.execute('BEGIN IMMEDIATE')
    monkeypatch.setattr(sqlite3, 'connect', lambda path, **kwargs: connect(path, timeout=0.02))
    try:
        with pytest.raises(CenterError) as caught:
            with store.connection() as db:
                db.execute('BEGIN IMMEDIATE')
        assert caught.value.code == 'SOURCE_DB_BUSY'
    finally:
        locker.rollback()
        locker.close()


@pytest.fixture
def store(tmp_path):
    result = SourceStore(tmp_path)
    result.seed()
    return result


@pytest.fixture
def client(monkeypatch, store):
    monkeypatch.setenv("DATA_SOURCE_CENTER_ENABLED", "true")
    monkeypatch.setattr(routes, "get_store", lambda: store)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def sample_for(config):
    row = {f.name: 1.25 if f.data_type == "number" else "20260828" if f.data_type == "date" else "example" for f in config.source_fields}
    row.update(ts_code="510300.SH", symbol="600000.SH", con_code="600000.SH", index_code="000300.SH", l3_code="801010.SI", month="202608", quarter="2026Q2", level="L3", is_new="Y", src="SW2021", status="L", list_status="L", is_open=1, birth_year=1970)
    return row


@pytest.mark.parametrize("api", tuple(PRESETS))
def test_each_tushare_preset_has_valid_mapping_and_a_wire_sample(api):
    config = PRESETS[api]
    validation = validate_mapping(config)
    assert validation["valid"], validation
    assert config.mappings, api
    row = sample_for(config)
    if api in {"ths_member", "dc_member", "tdx_member"}:
        # These wires do not provide an entry date. Mapping must stay incomplete
        # instead of promoting observation/capture time to historical membership.
        assert not validation["ready"]
        table, errors = map_table([row], config.mappings[0], config.source_id, "fixture")
        assert table.num_rows == 0 and errors
        assert "effective_from" in errors[0]["message"]
        assert "没有真实纳入日" in config.notes
        return
    if api == "fund_manager":
        assert not validation["ready"]
        assert all(item["code"] == "IDENTITY_LOOKUP_REQUIRED" for item in validation["warnings"])
        config = config.model_copy(deep=True)
        for identity in config.mappings[0].identities:
            if identity.resolution == "lookup":
                identity.value_map = {str(row[identity.source_field]): "confirmed-" + identity.target_field}
    for mapping in config.mappings:
        table, errors = map_table([row], mapping, config.source_id, "fixture")
        assert not errors, (api, errors)
        assert table.num_rows == 1
        assert table.schema.metadata[b"status"] == b"mapped_candidate"


def test_tushare_paging_preserves_existing_full_universe_capacity():
    assert PRESETS["fund_basic"].pagination.page_size == 15000
    assert PRESETS["fund_nav"].pagination.page_size == 10000
    assert PRESETS["fund_manager"].pagination.page_size == 5000
    assert PRESETS["stock_basic"].policy.requests_per_minute <= 50
    assert PRESETS["index_member_all"].policy.max_rows_per_request <= 2000


@pytest.mark.parametrize("api,volume,turnover", [("fund_daily", 100, 1000), ("index_daily", 100, 1000), ("sw_daily", 10000, 10000), ("ci_daily", 10000, 10000), ("dc_daily", 1, 1)])
def test_quote_units_and_vendor_return_names(api, volume, turnover):
    config = PRESETS[api]
    row = sample_for(config)
    row.update(vol=12, amount=34, pct_change=2, pct_chg=2)
    table, errors = map_table([row], config.mappings[0], "tushare", "fixture")
    assert not errors
    result = table.to_pylist()[0]
    assert result["volume"] == 12 * volume
    assert result["turnover_amount"] == Decimal(34 * turnover)
    assert result["return_decimal"] == 0.02


def test_nav_date_only_availability_is_end_of_local_day():
    table, errors = map_table([sample_for(PRESETS["fund_nav"])], PRESETS["fund_nav"].mappings[0], "tushare", "fixture")
    assert not errors
    record = table.to_pylist()[0]
    assert str(record["available_at"]) == "2026-08-28 15:59:59.999999+00:00"
    assert record["is_retrospective_adjustment"] is True
    assert record["total_assets"] is None  # total_netasset is not gross total assets.


def test_fund_holding_stock_weight_is_not_nav_weight():
    row = sample_for(PRESETS["fund_portfolio"])
    row.update(stk_mkv_ratio=25, stk_float_ratio=0.3)
    table, errors = map_table([row], PRESETS["fund_portfolio"].mappings[0], "tushare", "fixture")
    assert not errors
    record = table.to_pylist()[0]
    assert record["portfolio_weight"] is None
    assert record["floating_share_weight"] == 0.003
    assert record["coverage_scope"] == "PARTIAL"


def test_macro_rate_is_decimal_and_money_uses_base_currency():
    table, errors = map_table([{"date": "20260828", "on": 3}], PRESETS["shibor"].mappings[0], "tushare", "fixture")
    assert not errors
    assert table.to_pylist()[0]["value"] == 0.03
    assert table.to_pylist()[0]["unit"] == "decimal"
    assert table.to_pylist()[0]["available_at"] is None
    table, errors = map_table([{"quarter": "2026Q2", "gdp": 2}], PRESETS["cn_gdp"].mappings[0], "tushare", "fixture")
    assert not errors
    assert table.to_pylist()[0]["value"] == 200000000


def test_decode_formats_and_malformed_rows():
    rows = [{"code": "x", "value": 1}]
    assert decode_response({"data": rows}, ResponseFormat(records_path="data")) == rows
    assert decode_response({"f": ["code", "value"], "r": [["x", 1]]}, ResponseFormat(format="json_columns", records_path="r", columns_path="f")) == rows
    assert decode_response("code,value\nx,1\n", ResponseFormat(format="csv")) == [{"code": "x", "value": "1"}]
    with pytest.raises(CenterError):
        decode_response({"f": ["x", "x"], "r": [[1, 2]]}, ResponseFormat(format="json_columns", records_path="r", columns_path="f"))
    with pytest.raises(CenterError):
        decode_response("a,b\n1\n", ResponseFormat(format="csv"))
    with pytest.raises(CenterError):
        decode_response(rows * 2, ResponseFormat(), max_rows=1)


def test_no_internal_target_or_generated_field_can_be_mapped():
    config = PRESETS["fund_daily"].model_copy(deep=True)
    config.mappings[0].target_table = "governance.data_source"
    assert not validate_mapping(config)["valid"]
    config = PRESETS["fund_daily"].model_copy(deep=True)
    config.mappings[0].fields[0].target_field = "source_id"
    assert not validate_mapping(config)["valid"]
    config.mappings[0].contract_version = "obsolete"
    assert not validate_mapping(config)["valid"]


@pytest.mark.parametrize("table_id", [
    "master.organization_identifier",
    "master.person_identifier",
    "master.instrument_identifier",
    "portfolio.external_account_identifier",
])
def test_identifier_tables_cannot_be_import_targets(client, store, table_id):
    config = PRESETS["fund_daily"].model_copy(deep=True)
    config.mappings[0].target_table = table_id
    targets = client.get("/api/data-sources/catalog").json()["targets"]["tables"]
    assert table_id not in {table["table_id"] for table in targets}
    assert not validate_mapping(config)["valid"]
    report = preview(config, {})
    assert not report["valid"] and not report["ready"]
    assert report["tables"] == []
    with pytest.raises(CenterError) as error:
        map_table([], config.mappings[0], "tushare", "fixture")
    assert error.value.code == "INVALID_IMPORT_TARGET"

    previous = store.get("interface", config.id)
    response = client.put("/api/data-sources/config/interface", json={
        "config": config.model_dump(mode="json"),
        "expected_revision": previous["revision"],
    })
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "INVALID_MAPPING"
    assert store.get("interface", config.id) == previous

    batch = capture_batch(store, config, [sample_for(config)], {}, "fixture")
    assert batch["status"] == "REJECTED"
    assert batch["published"] is False
    assert not list((store.root / "mapped_candidates").rglob("*.parquet"))


def test_code_resolution_is_configuration_not_a_separate_import():
    records = []
    for api in ("fund_basic", "fund_daily"):
        config = PRESETS[api].model_copy(deep=True)
        identity = config.mappings[0].identities[0]
        identity.resolution = "lookup"
        identity.value_map = {"510300.SH": "platform-instrument-001"}
        assert validate_mapping(config)["ready"]
        table, errors = map_table([sample_for(config)], config.mappings[0], "tushare", "fixture")
        assert not errors
        assert table.schema.metadata[b"table_id"].decode() in {
            "master.instrument", "market.quote_daily",
        }
        records.append(table.to_pylist()[0])
    assert records[0]["instrument_id"] == records[1]["instrument_id"] == "platform-instrument-001"
    assert records[0]["canonical_name"] == "example"
    assert records[1]["close"] == 1.25


def test_invalid_values_do_not_report_preview_ready():
    config = PRESETS["fund_daily"].model_copy(deep=True)
    config.response = ResponseFormat()
    row = sample_for(config)
    row["close"] = "not-a-number"
    report = preview(config, [row])
    assert not report["valid"]
    assert not report["ready"]
    assert report["tables"][0]["accepted_rows"] == 0
    assert report["tables"][0]["rejected_rows"] == 1


def test_seed_does_not_overwrite_user_config_and_revision_conflicts_fail(store):
    entry = store.get("source", "tushare")
    changed = SourceConfig.model_validate(entry["config"])
    changed.name = "My Tushare"
    store.save(changed, entry["revision"])
    store.seed()
    assert store.get("source", "tushare")["config"]["name"] == "My Tushare"
    with pytest.raises(CenterError, match="配置已被修改"):
        store.save(changed, entry["revision"])
    with pytest.raises(CenterError):
        store.delete("source", "tushare", entry["revision"] + 1)


def test_configuration_and_sample_routes(client, store, monkeypatch):
    payload = client.get("/api/data-sources/catalog").json()
    assert len(payload["interfaces"]) == 46
    assert {item['config']['id'] for item in payload['sources']} == {'tushare', 'akshare'}
    assert all(t["source_mappable"] for t in payload["targets"]["tables"])
    assert not any(t["table_id"].startswith("governance.") for t in payload["targets"]["tables"])
    source = {**payload["templates"]["source"], "id": "vendor", "name": "Vendor", "enabled": True, "auth_mode": "bearer"}
    saved = client.put("/api/data-sources/config/source", json={"config": source, "expected_revision": 0})
    assert saved.status_code == 200
    assert client.put("/api/data-sources/config/source", json={"config": source, "expected_revision": 0}).status_code == 409
    assert client.put("/api/data-sources/credentials/vendor", json={"value": "very-secret-value"}).status_code == 200
    assert "very-secret-value" not in client.get("/api/data-sources/catalog").text
    source_config = store.get("source", "vendor")["config"]
    source_config["auth_mode"] = "bearer"
    store.save(SourceConfig.model_validate(source_config), 1)
    config = PRESETS["fund_daily"].model_dump(mode="json")
    config.update(id="vendor.quotes", source_id="vendor", api_name="", method="GET", path="/quotes", response={"format": "json_records", "records_path": "", "columns_path": "", "delimiter": ","})
    assert client.put("/api/data-sources/config/interface", json={"config": config, "expected_revision": 0}).status_code == 200
    calls = []
    def fake_request(url, method, params, headers, policy):
        calls.append((url, method))
        assert headers["Authorization"] == "Bearer very-secret-value"
        return json.dumps([sample_for(InterfaceConfig.model_validate(config))])
    monkeypatch.setattr(runtime, "request", fake_request)
    assert client.post("/api/data-sources/interfaces/vendor.quotes/sample", json={"expected_revision": 1}).status_code == 422
    result = client.post("/api/data-sources/interfaces/vendor.quotes/sample", json={"expected_revision": 1, "confirm": True, "params": {}})
    assert result.status_code == 200, result.text
    assert result.json()["requests"] == 1
    assert result.json()["download_complete"] is False
    assert len(calls) == 1


def test_configuration_rejects_secret_inputs_without_echo(client):
    config = PRESETS["fund_daily"].model_dump(mode="json")
    config["params"] = {"token": "sensitive-fixture"}
    response = client.post("/api/data-sources/validate", json={"config": config})
    assert response.status_code == 422
    assert "sensitive-fixture" not in response.text
    response = client.put("/api/data-sources/config/source", json={}, headers={"Origin": "https://untrusted.example"})
    assert response.status_code == 403


def test_read_only_disallows_changes_and_network(client, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE_CENTER_ENABLED", "false")
    config = default_source().model_dump(mode="json")
    assert client.put("/api/data-sources/config/source", json={"config": config, "expected_revision": 1}).status_code == 403
    assert client.post("/api/data-sources/interfaces/tushare.fund_daily/sample", json={"expected_revision": 1, "confirm": True}).status_code == 403
    assert client.get("/api/data-sources/catalog").status_code == 200


def test_shared_quota_reserves_source_and_api_atomically(store, monkeypatch):
    monkeypatch.setattr('backend.data_sources.quota.time.monotonic', lambda:0.)
    quota = SharedQuota(store)
    policy = DownloadPolicy(requests_per_minute=2, max_rows_per_request=3, rows_per_minute=6, min_interval_seconds=0, max_concurrency=2)
    levels = [("source:s", policy), ("api:s:a", policy)]
    assert quota.reserve(levels, 3, 1000, "a") == 0
    assert SharedQuota(store).reserve(levels, 3, 1001, "b") == 0
    assert quota.reserve(levels, 3, 1002, "c") >= 58
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM source_quota").fetchone()[0] == 4
    assert quota.reserve(levels, 3, 1062, "c") == 0


def test_transport_blocks_private_and_mixed_dns(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, '', ('127.0.0.1', 443))])
    with pytest.raises(CenterError, match="禁止访问"):
        public_address("localhost", 443)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, '', ('8.8.8.8', 443)), (2, 1, 6, '', ('10.0.0.1', 443))])
    with pytest.raises(CenterError):
        public_address("rebind.example", 443)


def test_pmi_alias_candidate_has_new_identity_without_rewriting_raw_or_old_batch(store, monkeypatch):
    import pyarrow.parquet as pq
    from backend.data_sources import tushare_facts
    rows = [{'MONTH': '202608', 'PMI010000': 49.5, 'CREATE_BY': 'vendor'}]
    original = deepcopy(rows)
    config = PRESETS['cn_pmi']
    with monkeypatch.context() as previous:
        previous.setattr(tushare_facts, 'RESPONSE_CONTRACT_VERSIONS', {})
        old = capture_batch(store, config, rows, {}, 'same-configuration')
    assert old['status'] == 'REJECTED'
    current = capture_batch(store, config, rows, {}, 'same-configuration')
    assert current['status'] == 'VALIDATED_CANDIDATE'
    assert current['batch_id'] != old['batch_id']
    assert current['batch_format_version'] == 3
    assert current['response_contract'] == 'pmi-uppercase-v1'
    assert current['raw_checksum'] == old['raw_checksum']
    assert rows == original
    directory = store.root / 'mapped_candidates' / 'tushare' / current['batch_id']
    assert json.loads((directory / 'raw.json').read_text()) == original
    result = pq.read_table(store.root / current['tables'][0]['artifact']).to_pylist()[0]
    assert result['value'] == 49.5 and result['available_at'] is None
    assert capture_batch(store, config, rows, {}, 'same-configuration') == current
    assert {item['batch_id']: item for item in recent_batches(store)}[old['batch_id']] == old


def test_pmi_alias_collision_cannot_produce_candidate(store):
    with pytest.raises(CenterError) as error:
        capture_batch(store, PRESETS['cn_pmi'], [{'MONTH': '202608', 'month': '202607'}], {}, 'collision')
    assert error.value.code == 'SOURCE_FIELD_COLLISION'
    assert not recent_batches(store)
    assert not list((store.root / 'mapped_candidates').rglob('*.parquet'))
    assert len(list((store.root / 'mapped_candidates').rglob('raw.json'))) == 1


def test_candidates_are_idempotent_and_never_publish_active_data(store):
    config = PRESETS["fund_daily"]
    before = store.root / "tushare_active.json"
    before.write_text('unchanged-active-manifest')
    rows = [sample_for(config)]
    first = capture_batch(store, config, rows, {}, "configuration-1")
    second = capture_batch(store, config, rows, {}, "configuration-1")
    assert first == second
    assert first["status"] == "VALIDATED_CANDIDATE"
    assert first["published"] is False
    assert before.read_text() == 'unchanged-active-manifest'
    assert len(recent_batches(store)) == 1
    rejected = capture_batch(store, config, [{**rows[0], "close": "invalid"}], {}, "configuration-1")
    assert rejected["status"] == "REJECTED"
    assert before.read_text() == 'unchanged-active-manifest'


def test_configured_client_downloads_and_captures_using_frozen_configuration(store, monkeypatch):
    from backend.data_sources.runtime import ConfiguredTushareClient
    client = ConfiguredTushareClient("fixture-secret", root=store.root, capture=True)
    frozen_hash = client.configuration_hash
    calls = []
    row = sample_for(PRESETS["fund_daily"])
    def response(url, method, body, headers, policy):
        calls.append(body["api_name"])
        assert body["token"] == "fixture-secret"
        assert policy.requests_per_minute == 240
        return json.dumps({"code": 0, "data": {"fields": list(row), "items": [list(row.values())]}})
    monkeypatch.setattr(runtime, "request", response)
    changed = default_source().model_copy(deep=True)
    changed.name = "New configuration while client remains frozen"
    store.save(changed, 1)
    frame = client.fund_daily(ts_code="510300.SH", trade_date="20260828")
    assert len(frame) == 1
    assert calls == ["fund_daily"]
    assert client.source.name == "Tushare"
    assert client.configuration_hash == frozen_hash
    assert recent_batches(store)[0]["status"] == "VALIDATED_CANDIDATE"


def test_new_tushare_endpoint_cannot_bypass_entitlement_on_second_save(store, monkeypatch):
    monkeypatch.setenv("DATA_SOURCE_CENTER_ENABLED", "true")
    config = PRESETS["fund_daily"].model_dump(mode="json")
    config.update(id="tushare.other_api", api_name="other_api", enabled=False)
    service.save(store, "interface", config, 0)
    config["enabled"] = True
    with pytest.raises(CenterError) as error:
        service.save(store, "interface", config, 1)
    assert error.value.code == "ENTITLEMENT_REQUIRED"


def test_saved_source_configuration_changes_refresh_fingerprint_and_checkpoint(store, monkeypatch):
    from backend.services import data_refresh
    from backend.data_sources.legacy_bridge import configuration_fingerprint
    import importlib.util
    import types
    spec = importlib.util.spec_from_file_location("source_center_downloader_fixture", ROOT / "T01_get_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(data_refresh, "DATA_DIR", store.root)
    before = data_refresh.refresh_request_fingerprint(["etf"], "incremental")
    args = types.SimpleNamespace(start_date="20260101", end_date="20260828", source_configuration_hash=configuration_fingerprint(store.root))
    checkpoint = module.history_checkpoint_dir(store.root / "nav.parquet", args)
    config = InterfaceConfig.model_validate(store.get("interface", "tushare.fund_nav")["config"])
    config.pagination.page_size = 500
    store.save(config, 1)
    after = data_refresh.refresh_request_fingerprint(["etf"], "incremental")
    assert after != before
    args.source_configuration_hash = configuration_fingerprint(store.root)
    assert module.history_checkpoint_dir(store.root / "nav.parquet", args) != checkpoint
    monkeypatch.delenv("TUSHARE_FUND_NAV_PAGE_SIZE", raising=False)
    command = data_refresh.build_refresh_command(["etf"], "incremental")
    assert command[command.index("--fund-nav-page-size") + 1] == "500"


def test_non_transient_errors_are_not_retried(store, monkeypatch):
    calls = []
    def reject(*args, **kwargs):
        calls.append(True)
        raise CenterError("SOURCE_PERMISSION_OR_PARAMS", "Rejected")
    monkeypatch.setattr(runtime, "fetch_once", reject)
    with pytest.raises(CenterError):
        runtime.fetch_with_retry(store, default_source(), PRESETS["fund_daily"])
    assert calls == [True]


def test_network_retry_uses_read_window_and_keeps_attempt_budget(store, monkeypatch):
    calls, waits = [], []
    def fail(*args, **kwargs):
        calls.append(True)
        raise TransientSourceError('SOURCE_CONNECTION', '网络暂时中断', 502)
    monkeypatch.setattr(runtime, 'fetch_once', fail)
    monkeypatch.setattr(runtime.time, 'sleep', waits.append)
    monkeypatch.setattr(runtime.random, 'uniform', lambda *args: 0)
    with pytest.raises(CenterError):
        runtime.fetch_with_retry(store, default_source(), PRESETS['fund_daily'])
    assert len(calls) == 3 and waits == [30, 60]
