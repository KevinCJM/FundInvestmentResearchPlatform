"""Frozen configuration and shared acquisition boundary."""
from __future__ import annotations
import hashlib
import json
import random
import time
from functools import partial
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .credentials import destination_matches, read_credential
from .mapping import decode_response
from .models import CenterError, DownloadPolicy, InterfaceConfig, SourceConfig
from .quota import SharedQuota
from .store import DEFAULT_ROOT, SourceStore
from .transport import TransientSourceError, request
from backend.data_storage import guard_path


def effective_policy(source: DownloadPolicy, interface: DownloadPolicy) -> DownloadPolicy:
    result = interface.model_dump()
    for key in ("requests_per_minute", "max_rows_per_request", "max_concurrency", "connect_timeout_seconds", "read_timeout_seconds", "max_attempts", "max_response_bytes", "max_runtime_seconds"):
        result[key] = min(getattr(source, key), getattr(interface, key))
    for key in ("min_interval_seconds", "backoff_seconds", "rate_limit_wait_seconds"):
        result[key] = max(getattr(source, key), getattr(interface, key))
    caps = [value for value in (source.rows_per_minute, interface.rows_per_minute) if value is not None]
    result["rows_per_minute"] = min(caps) if caps else None
    return DownloadPolicy.model_validate(result)


def fetch_once(store: SourceStore, source: SourceConfig, interface: InterfaceConfig, params: dict[str, Any] | None = None, *, credential: str | None = None, sample: bool = False, check=None) -> tuple[Any, list[dict[str, Any]]]:
    guard_path(store.root, write=True)
    if source.id != interface.source_id or not source.enabled or not interface.enabled:
        raise CenterError("SOURCE_DISABLED", "数据源或接口未启用。")
    policy = effective_policy(source.policy, interface.policy)
    merged = {**interface.params, **(params or {})}
    if source.transport == "akshare":
        from .akshare_adapter import fetch_sdk
        rows = fetch_sdk(store, source, interface, merged, sample=sample)
        return rows, decode_response(rows, interface.response, policy.max_rows_per_request)
    headers = dict(interface.headers)
    secret = None
    if credential and not destination_matches(store, source.id):
        raise CenterError("CREDENTIAL_DESTINATION_CHANGED", "来源地址或认证方式已变更，请重新保存凭据后下载。")
    if source.transport == "tushare":
        from .presets import API_SPECS
        if interface.api_name not in API_SPECS and not interface.entitlement_confirmed:
            raise CenterError("ENTITLEMENT_REQUIRED", "请先核实该接口的账户权限后启用。")
        if source.auth_mode == "none" and interface.method != "POST":
            raise CenterError("CREDENTIAL_QUERY_FORBIDDEN", "原生 Token 认证必须使用 POST，避免凭据进入 URL；GET 接口请改用请求头认证。")
        secret = credential or read_credential(store, source.id)
        fields = merged.pop("fields", "")
        body = {"api_name": interface.api_name, "params": merged, "fields": fields}
        if source.auth_mode == "none":
            body["token"] = secret
        else:
            headers["Authorization" if source.auth_mode == "bearer" else source.auth_header] = "Bearer " + secret if source.auth_mode == "bearer" else secret
        url, method = source.base_url.rstrip("/") + interface.path, interface.method
    else:
        body, url, method = merged, source.base_url.rstrip("/") + interface.path, interface.method
        if source.auth_mode != "none":
            secret = credential or read_credential(store, source.id)
            headers["Authorization" if source.auth_mode == "bearer" else source.auth_header] = "Bearer " + secret if source.auth_mode == "bearer" else secret
    quota = SharedQuota(store)
    with quota.acquire(source.id, interface.api_name or interface.id, source.policy, interface.policy,
                       policy.max_rows_per_request, **({'check': check} if check else {})):
        guard_path(store.root, write=True)
        raw = request(url, method, body, headers, policy)
    if source.transport == "tushare":
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise CenterError("SOURCE_RESPONSE_INVALID", "Tushare 返回格式无效。", 502) from exc
        if not isinstance(raw, dict) or raw.get("code") != 0:
            message = str(raw.get("msg", "")) if isinstance(raw, dict) else ""
            if any(marker in message for marker in ("每分钟", "访问频次", "访问频率")):
                raise TransientSourceError("SOURCE_RATE_LIMIT", "Tushare 返回限流。", 429)
            raise CenterError("SOURCE_PERMISSION_OR_PARAMS", "Tushare 拒绝请求，请核对权限、参数与字段。", 502)
    if source.transport == "tushare" and isinstance(raw.get("data"), dict) and isinstance(raw["data"].get("items"), list) and len(raw["data"]["items"]) > policy.max_rows_per_request:
        raise CenterError("SOURCE_ROW_CAP", "接口返回超过配置的单次行数上限，需要缩小分片。")
    rows = decode_response(raw, interface.response, policy.max_rows_per_request)
    if secret:
        def clean(value):
            if isinstance(value, str):
                return value.replace(secret, "[REDACTED]")
            if isinstance(value, dict):
                return {key: clean(child) for key, child in value.items()}
            if isinstance(value, list):
                return [clean(child) for child in value]
            return value
        rows = clean(rows)
    return raw, rows


def fetch_with_retry(store: SourceStore, source: SourceConfig, interface: InterfaceConfig, params: dict[str, Any] | None = None, *, sample: bool = False) -> tuple[Any, list[dict[str, Any]]]:
    policy = effective_policy(source.policy, interface.policy)
    started = time.monotonic()
    attempts = 1 if sample else policy.max_attempts
    for attempt in range(attempts):
        remaining = policy.max_runtime_seconds - (time.monotonic() - started)
        if remaining < 0.1:
            raise CenterError("SOURCE_RUNTIME_LIMIT", "请求已超过任务时限。")
        bounded = source.model_copy(update={"policy": source.policy.model_copy(update={
            "connect_timeout_seconds": min(source.policy.connect_timeout_seconds, remaining),
            "read_timeout_seconds": min(source.policy.read_timeout_seconds, remaining),
            "max_runtime_seconds": max(1, int(remaining)),
        })})
        try:
            return fetch_once(store, bounded, interface, params, sample=sample)
        except TransientSourceError as exc:
            delay = policy.backoff_seconds * 2**attempt
            if exc.code in {'SOURCE_CONNECTION', 'SOURCE_DNS', 'SOURCE_TIMEOUT'}:
                delay = max(delay, policy.read_timeout_seconds * 2**attempt)
            if exc.code in {"SOURCE_RATE_LIMIT", "SOURCE_RETRYABLE"}:
                delay = max(delay, policy.rate_limit_wait_seconds, getattr(exc, "retry_after_seconds", 0))
            delay += random.uniform(0, 0.25)
            if attempt + 1 == attempts or time.monotonic() - started + delay >= policy.max_runtime_seconds:
                raise
            time.sleep(delay)
    raise CenterError("SOURCE_UNAVAILABLE", "数据源请求失败。", 502)


class ConfiguredTushareClient:
    """The old downloader owns slicing/retries; this client owns actual I/O."""
    def __init__(self, credential: str, root: Path = DEFAULT_ROOT, *, capture: bool = False,
                 source_id: str = 'tushare', on_batch=None) -> None:
        self.store = SourceStore(root)
        self.store.seed()
        with self.store.connection() as db:
            frozen = db.execute("SELECT * FROM source_config WHERE source_id=? ORDER BY kind,id", (source_id,)).fetchall()
        decoded = [self.store.decode(row) for row in frozen]
        entry = next((item["config"] for item in decoded if item["config"]["id"] == source_id and "transport" in item["config"]), None)
        if entry is None:
            raise CenterError("SOURCE_NOT_CONFIGURED", "原市场下载流程的数据源已删除，请使用已保存接口的下载入口。")
        self.source = SourceConfig.model_validate(entry)
        self.interfaces = {item["config"]["id"].removeprefix(source_id + '.'): InterfaceConfig.model_validate(item["config"]) for item in decoded if "api_name" in item["config"]}
        self.configuration_hash = hashlib.sha256(json.dumps([{"config": item["config"], "revision": item["revision"]} for item in decoded], sort_keys=True).encode()).hexdigest()
        self.capture = capture
        self.on_batch = on_batch
        self.credential = credential
        self.started_at = time.monotonic()
        self._request_check = None

    @contextmanager
    def request_guard(self, check):
        """One dataset owns this client; checks also interrupt shared quota waits."""
        previous = self._request_check
        self._request_check = check
        try:
            yield
        finally:
            self._request_check = previous

    def __getattr__(self, api: str):
        if api not in self.interfaces:
            raise CenterError("API_NOT_CONFIGURED", "该 Tushare 接口尚未配置。")
        interface = self.interfaces[api]
        operation = partial(self._call, interface)
        operation.download_policy = effective_policy(self.source.policy, interface.policy)
        operation.pagination_config = interface.pagination
        return operation

    def _call(self, interface: InterfaceConfig, **params: Any):
        import pandas as pd
        if time.monotonic() - self.started_at > self.source.policy.max_runtime_seconds:
            raise CenterError("SOURCE_RUNTIME_LIMIT", "数据源任务超过配置的最长运行时间。")
        # The legacy coordinator names its parameters; the actual request uses
        # the saved interface parameter names and pagination protocol.
        params = dict(params)
        for old, new in (("start_date", interface.start_param), ("end_date", interface.end_param)):
            if old in params and old != new:
                params[new] = params.pop(old)
        if "offset" in params and interface.pagination.mode == "page":
            offset = int(params.pop("offset"))
            size = int(params.get("limit", interface.pagination.page_size))
            if size <= 0 or offset < 0 or offset % size:
                raise CenterError("PAGE_ALIGNMENT", "旧下载分片与当前页码大小不对齐，请使用接口下载或修正分页配置。")
            params[interface.pagination.cursor_param] = offset // size + 1
        elif "offset" in params and interface.pagination.cursor_param != "offset":
            params[interface.pagination.cursor_param] = params.pop("offset")
        if "limit" in params and interface.pagination.limit_param != "limit":
            params[interface.pagination.limit_param] = params.pop("limit")
        checks = {'check': self._request_check} if self._request_check is not None else {}
        _, rows = fetch_once(self.store, self.source, interface, params, credential=self.credential, **checks)
        if self.capture:
            from .batches import capture_batch
            batch = capture_batch(self.store, interface, rows, {**interface.params, **params}, self.configuration_hash)
            if self.on_batch is not None:
                self.on_batch(batch)
        return pd.DataFrame.from_records(rows)
