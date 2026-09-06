"""Strict, source-independent configuration models."""
from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_PATTERN = r"^[a-z][a-z0-9_.-]{0,99}$"
FIELD_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]{0,99}$"
SOURCE_FIELD_PATTERN = r"^[^\x00-\x1f\x7f]{1,100}$"


class CenterError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class DownloadPolicy(StrictModel):
    requests_per_minute: int = Field(default=60, ge=1, le=10000)
    rows_per_minute: int | None = Field(default=None, ge=1, le=10000000)
    min_interval_seconds: float = Field(default=0.2, ge=0, le=3600)
    max_rows_per_request: int = Field(default=5000, ge=1, le=100000)
    max_concurrency: int = Field(default=1, ge=1, le=32)
    connect_timeout_seconds: float = Field(default=5, ge=0.1, le=60)
    read_timeout_seconds: float = Field(default=30, ge=0.1, le=300)
    max_attempts: int = Field(default=3, ge=1, le=6)
    backoff_seconds: float = Field(default=2, ge=0, le=60)
    rate_limit_wait_seconds: float = Field(default=60, ge=1, le=600)
    max_response_bytes: int = Field(default=8388608, ge=1024, le=33554432)
    max_runtime_seconds: int = Field(default=3600, ge=1, le=86400)

    @model_validator(mode="after")
    def consistent_limits(self) -> "DownloadPolicy":
        if self.rows_per_minute is not None and self.max_rows_per_request > self.rows_per_minute:
            raise ValueError("单次行数上限不能超过每分钟行数上限。")
        return self


class SourceConfig(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    transport: Literal["tushare", "http", "akshare"] = "http"
    base_url: str = Field(default="", max_length=500)
    enabled: bool = True
    auth_mode: Literal["none", "bearer", "header"] = "none"
    auth_header: str = Field(default="X-API-Key", pattern=r"^[A-Za-z][A-Za-z0-9-]{0,63}$")
    policy: DownloadPolicy = Field(default_factory=DownloadPolicy)
    notes: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def valid_endpoint(self) -> "SourceConfig":
        parsed = urlsplit(self.base_url)
        if self.transport != "akshare" and (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("请填写不含凭据和查询参数的 HTTPS 来源地址。")
        if self.transport == "akshare" and (self.base_url or self.auth_mode != "none"):
            raise ValueError("SDK 来源使用受控函数，不配置请求地址或认证凭据。")
        if self.auth_header.lower() in {"host", "content-length", "connection", "transfer-encoding"}:
            raise ValueError("协议控制头不能用作凭据头。")
        return self


class SourceField(StrictModel):
    name: str = Field(pattern=SOURCE_FIELD_PATTERN)
    data_type: Literal["string", "number", "integer", "boolean", "date", "datetime", "json"] = "string"
    description: str = Field(default="", max_length=1000)
    unit: str = Field(default="", max_length=100)


class ResponseFormat(StrictModel):
    format: Literal["json_records", "json_columns", "csv"] = "json_records"
    records_path: str = Field(default="", max_length=200, pattern=r"^$|^[A-Za-z_][A-Za-z0-9_.]*$")
    columns_path: str = Field(default="", max_length=200, pattern=r"^$|^[A-Za-z_][A-Za-z0-9_.]*$")
    delimiter: str = Field(default=",", min_length=1, max_length=1)


class FieldMapping(StrictModel):
    target_field: str = Field(pattern=FIELD_PATTERN)
    source_field: str | None = Field(default=None, pattern=SOURCE_FIELD_PATTERN)
    operation: Literal["copy", "scale", "constant", "enum", "date", "timestamp", "period_end", "capture_date"] = "copy"
    factor: float = 1.0
    constant: str | int | float | bool | None = None
    enum_map: dict[str, str | int | bool] = Field(default_factory=dict)
    timezone: str = Field(default="Asia/Shanghai", max_length=60)
    date_format: str | None = Field(default=None, max_length=30)

    @model_validator(mode="after")
    def valid_operation(self) -> "FieldMapping":
        if self.operation not in {"constant", "capture_date"} and not self.source_field:
            raise ValueError("非固定值映射必须指定来源字段。")
        if self.operation == "enum" and not self.enum_map:
            raise ValueError("枚举转换必须提供对应关系。")
        if len(self.enum_map) > 500:
            raise ValueError("枚举映射不能超过 500 项。")
        return self


class IdentityBinding(StrictModel):
    target_field: str = Field(pattern=FIELD_PATTERN)
    source_field: str | None = Field(default=None, pattern=SOURCE_FIELD_PATTERN)
    key_fields: list[str] = Field(default_factory=list, max_length=10)
    constant: str | None = Field(default=None, max_length=200)
    namespace: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.:-]+$")
    resolution: Literal["namespace", "lookup"] = "namespace"
    key_transform: Literal["none", "cn_etf_code", "cn_fund_code"] = "none"
    value_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exact_key(self) -> "IdentityBinding":
        if sum([self.source_field is not None, bool(self.key_fields), self.constant is not None]) != 1:
            raise ValueError("身份解析必须选择来源字段、组合键或固定代码之一。")
        if self.constant is not None and not self.constant.strip():
            raise ValueError("固定身份代码不能为空。")
        import re
        if any(not re.fullmatch(SOURCE_FIELD_PATTERN, key) for key in self.key_fields):
            raise ValueError("身份组合键包含无效字段。")
        if len(self.value_map) > 5000 or any(not value.strip() for value in self.value_map.values()):
            raise ValueError("身份对照表过大或存在空内部标识。")
        return self


class DatasetMapping(StrictModel):
    target_table: str = Field(pattern=ID_PATTERN)
    contract_version: str = "1.2.0"
    enabled: bool = True
    fields: list[FieldMapping] = Field(default_factory=list, max_length=300)
    identities: list[IdentityBinding] = Field(default_factory=list, max_length=30)


class Pagination(StrictModel):
    mode: Literal["none", "offset", "page"] = "none"
    cursor_param: str = Field(default="offset", pattern=FIELD_PATTERN)
    limit_param: str = Field(default="limit", pattern=FIELD_PATTERN)
    page_size: int = Field(default=1000, ge=1, le=100000)
    max_pages: int = Field(default=20, ge=1, le=1000)


class InterfaceConfig(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    source_id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=120)
    enabled: bool = False
    api_name: str = Field(default="", pattern=r"^[a-z][a-z0-9_]{0,99}$|^$")
    method: Literal["GET", "POST"] = "GET"
    path: str = Field(default="", max_length=500)
    params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    response: ResponseFormat = Field(default_factory=ResponseFormat)
    source_fields: list[SourceField] = Field(default_factory=list, max_length=300)
    policy: DownloadPolicy = Field(default_factory=DownloadPolicy)
    pagination: Pagination = Field(default_factory=Pagination)
    start_param: str = Field(default="start_date", pattern=FIELD_PATTERN)
    end_param: str = Field(default="end_date", pattern=FIELD_PATTERN)
    incremental_field: str | None = Field(default=None, pattern=SOURCE_FIELD_PATTERN)
    mappings: list[DatasetMapping] = Field(default_factory=list, max_length=30)
    notes: str = Field(default="", max_length=4000)
    entitlement_confirmed: bool = False

    @model_validator(mode="after")
    def valid_config(self) -> "InterfaceConfig":
        import json
        import re
        from zoneinfo import ZoneInfo

        if self.path and (not self.path.startswith("/") or self.path.startswith("//") or any(x in self.path for x in ("?", "#", "\\", ".."))):
            raise ValueError("接口路径必须位于来源地址下，不允许查询参数或目录跳转。")
        def check_keys(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if str(key).lower().replace("-", "_").endswith(("token", "password", "secret", "authorization", "api_key", "apikey", "cookie")):
                        raise ValueError("请通过专用凭据入口保存认证信息。")
                    check_keys(child)
            elif isinstance(value, list):
                for child in value:
                    check_keys(child)
        check_keys(self.params)
        check_keys(self.headers)
        if len(json.dumps(self.params, ensure_ascii=False, allow_nan=False)) > 32000 or len(self.headers) > 30:
            raise ValueError("参数或请求头超过限制。")
        for name, value in self.headers.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]{0,63}", name) or "\r" in value or "\n" in value:
                raise ValueError("请求头格式无效。")
            if name.lower() in {"host", "content-length", "connection", "transfer-encoding"}:
                raise ValueError("不允许覆盖协议控制头。")
        names = [item.name for item in self.source_fields]
        if len(names) != len(set(names)):
            raise ValueError("来源字段不能重名。")
        if self.pagination.page_size > self.policy.max_rows_per_request:
            raise ValueError("分页大小不能超过单次行数上限。")
        if self.response.format == "json_columns" and not self.response.columns_path:
            raise ValueError("列式 JSON 必须指定字段名数组路径。")
        for mapping in self.mappings:
            for binding in mapping.fields:
                try:
                    ZoneInfo(binding.timezone)
                except (KeyError, ValueError) as exc:
                    raise ValueError("映射时区无效。") from exc
        return self
