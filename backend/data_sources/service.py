"""Application service for configuration, validation and explicit sampling."""
from __future__ import annotations
import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

from pydantic import ValidationError

try:
    from backend.data_model.catalog import get_data_model_catalog
    from backend.services.refresh_runtime import InterProcessFileLock
except ModuleNotFoundError:
    from data_model.catalog import get_data_model_catalog
    from services.refresh_runtime import InterProcessFileLock

from .credentials import configured, save_credential
from .batches import recent_batches
from .mapping import preview, validate_mapping
from .models import CenterError, InterfaceConfig, SourceConfig
from .runtime import effective_policy, fetch_with_retry
from .store import SourceStore


def enabled() -> bool:
    default = "false" if os.getenv("APP_ENV", "development").lower() == "production" else "true"
    return os.getenv("DATA_SOURCE_CENTER_ENABLED", default).lower() in {"1", "true", "yes", "on"}


def parse_config(kind: str, payload: Any):
    try:
        return (SourceConfig if kind == "source" else InterfaceConfig).model_validate(payload)
    except ValidationError as exc:
        # Pydantic's default errors contain input values, including credentials.
        messages = [".".join(map(str, item["loc"])) + ": " + str(item["msg"]) for item in exc.errors(include_input=False, include_url=False, include_context=False)]
        raise CenterError("INVALID_CONFIG", "；".join(messages[:12]), 422) from None


@contextmanager
def mutation_lock(store: SourceStore) -> Iterator[None]:
    if not enabled():
        raise CenterError("SOURCE_CENTER_READ_ONLY", "当前环境未开启数据源配置和采样。", 403)
    lock = InterProcessFileLock(store.root / ".tushare_refresh.lock")
    if not lock.acquire(owner="data-source-center"):
        raise CenterError("DATA_TASK_RUNNING", "已有数据任务运行；结束后才能修改配置或采样。", 409)
    try:
        yield
    finally:
        lock.release()


def catalog(store: SourceStore) -> dict[str, Any]:
    from .request_schema import request_fields
    from .etl_graph import graph_schemas
    from .task_catalog import task_catalog
    store.seed()
    sources = store.list("source")
    for source in sources:
        source["credential_configured"] = configured(store, source["config"]["id"])
        source['credential_required'] = source['config']['transport'] == 'tushare' or source['config']['auth_mode'] != 'none'
    interfaces = store.list("interface")
    by_id = {item["config"]["id"]: SourceConfig.model_validate(item["config"]) for item in sources}
    for item in interfaces:
        config = InterfaceConfig.model_validate(item["config"])
        item['request_fields'] = request_fields(by_id[config.source_id], config)
        item["validation"] = validate_mapping(config)
        item["effective_policy"] = effective_policy(by_id[config.source_id].policy, config.policy).model_dump(mode="json")
    return {"sources": sources, "interfaces": interfaces, "etl_tasks": task_catalog(store)['tasks'], "graph_schemas": graph_schemas(), "targets": get_data_model_catalog(), "editing_enabled": enabled(), "batches": recent_batches(store),
            "templates": {"source": SourceConfig(id="custom", name="新数据源", base_url="https://example.com", enabled=False).model_dump(mode="json"),
                          "interface": InterfaceConfig(id="custom.endpoint", source_id="custom", name="新接口").model_dump(mode="json")},
            "boundary": "配置与映射中心；标准化结果先进入独立候选数据，不自动替换既有研究数据。"}


def save(store: SourceStore, kind: str, payload: Any, expected_revision: int) -> dict[str, Any]:
    config = parse_config(kind, payload)
    with mutation_lock(store):
        if isinstance(config, InterfaceConfig):
            parent = SourceConfig.model_validate(store.get("source", config.source_id)["config"])
            policy = effective_policy(parent.policy, config.policy)
            if config.pagination.mode != "none" and config.pagination.page_size > policy.max_rows_per_request:
                raise CenterError("PAGE_SIZE_EXCEEDS_SOURCE", "分页大小超过来源与接口共同生效的单次行数限制。", 422)
            validation = validate_mapping(config)
            if not validation["valid"]:
                raise CenterError("INVALID_MAPPING", "；".join(item["message"] for item in validation["errors"][:12]), 422)
            if parent.transport == "tushare":
                from .presets import API_SPECS
                if not config.api_name:
                    raise CenterError("API_NAME_REQUIRED", "请填写接口 API 名称。")
                if config.enabled and config.api_name not in API_SPECS and not config.entitlement_confirmed:
                    raise CenterError("ENTITLEMENT_REQUIRED", "请先核实该接口的账户权限后启用。")
                if parent.auth_mode == "none" and config.method != "POST":
                    raise CenterError("CREDENTIAL_QUERY_FORBIDDEN", "原生 Token 认证必须使用 POST；GET 接口请先选择请求头认证。")
            if parent.transport == "akshare":
                from .akshare_adapter import validate_sdk_config
                validate_sdk_config(config)
        else:
            for item in store.list("interface"):
                if item["config"]["source_id"] == config.id:
                    child = InterfaceConfig.model_validate(item["config"])
                    policy = effective_policy(config.policy, child.policy)
                    if child.pagination.mode != "none" and child.pagination.page_size > policy.max_rows_per_request:
                        raise CenterError("PAGE_SIZE_EXCEEDS_SOURCE", "请先降低下属接口的分页大小，再降低数据源行数上限。", 422)
        return store.save(config, expected_revision)


def set_credential(store: SourceStore, source_id: str, value: str | None) -> dict[str, bool]:
    with mutation_lock(store):
        save_credential(store, source_id, value)
    return {"credential_configured": configured(store, source_id)}


def sample(store: SourceStore, identifier: str, params: dict[str, Any], expected_revision: int) -> dict[str, Any]:
    with mutation_lock(store):
        item = store.get("interface", identifier)
        if item["revision"] != expected_revision:
            raise CenterError("REVISION_CONFLICT", "接口已更新，请刷新后重新采样。", 409)
        interface = InterfaceConfig.model_validate(item["config"])
        # Reuse parameter validation, including credential-key rejection.
        interface = parse_config("interface", {**interface.model_dump(), "params": {**interface.params, **params}})
        source = SourceConfig.model_validate(store.get("source", interface.source_id)["config"])
        _, rows = fetch_with_retry(store, source, interface, sample=True)
        draft = interface.model_copy(update={"response": interface.response.model_copy(update={"format": "json_records", "records_path": "", "columns_path": ""})})
        result = preview(draft, [{**interface.params, **row} for row in rows[:100]])
        result.update({"requests": 1, "received_rows": len(rows), "sampled_rows": min(len(rows), 100), "source_preview": rows[:20], "interface_revision": item["revision"], "download_complete": False})
        return result
