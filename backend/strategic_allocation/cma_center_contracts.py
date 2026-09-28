"""Lifecycle contracts for the independent LTCMA workspace."""
from __future__ import annotations

from typing import Any, Literal
from datetime import date
import json

from pydantic import Field, field_validator, model_validator

from .common_contracts import Contract, Fingerprint, Identifier, UpstreamRef, Usability, VersionInfo
from .contracts import PublishCmaRequest
from .cma_model_contracts import CmaWindow, CmaVersionRef, StatisticalCmaContext
from .reference_contracts import ExplicitConfirm


class CmaCenterPublish(PublishCmaRequest):
    confirm: Literal[True] | None = None
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=120,
                                      pattern=r"^[A-Za-z0-9_.:-]+$")
    copied_from_id: Identifier | None = None

    @field_validator("confirm", mode="before")
    @classmethod
    def strict_confirm(cls, value):
        if value is not None and value is not True:
            raise ValueError("确认必须是明确的 true。")
        return value

    @model_validator(mode="after")
    def explicit_publication(self):
        if self.idempotency_key is not None and self.confirm is not True:
            raise ValueError("LTCMA 发布须明确确认研究假设及限制。")
        if self.request.schema_version == "2.0" and (self.confirm is not True or self.idempotency_key is None):
            raise ValueError("LTCMA 2.0 发布需要明确确认和幂等操作键。")
        return self


class CmaCenterUpdate(CmaCenterPublish):
    confirm: Literal[True]
    idempotency_key: str = Field(min_length=8, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$")
    expected_content_hash: Fingerprint
    copied_from_id: None = None


class CmaRetire(ExplicitConfirm):
    content_hash: Fingerprint
    reason: str = Field(min_length=5, max_length=2000)


class CmaDraftWrite(Contract):
    name: str = Field(min_length=1, max_length=120)
    editable_definition: dict[str, Any]
    copied_from_id: Identifier | None = None
    editing_ref: CmaVersionRef | None = None
    expected_revision: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def bounded_json(self):
        if self.editing_ref is not None and self.copied_from_id is not None:
            raise ValueError("修改原方案与复制新研究不能同时指定。")
        try:
            encoded = json.dumps(self.editable_definition, allow_nan=False, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("草稿只能包含有限 JSON 数值，未填写项使用 null。") from exc
        if len(encoded.encode("utf-8")) > 128_000:
            raise ValueError("LTCMA 草稿超过 128 KB。")
        return self


class CmaDraftDelete(Contract):
    expected_revision: int = Field(ge=1, strict=True)


class CmaSampleRequest(Contract):
    """Inspect the likelihood window before selecting any NIW prior strength."""
    alloc_name: Identifier | None = None
    strategic_universe_id: Identifier | None = None
    model: StatisticalCmaContext

    @model_validator(mode="after")
    def one_scope(self):
        if bool(self.alloc_name) == bool(self.strategic_universe_id):
            raise ValueError("请选择一个产品大类或战略资产范围。")
        return self


class CmaSampleSummary(Contract):
    requested_start: date
    requested_end: date
    actual_start: date
    actual_end: date
    observations: int = Field(ge=1)
    # 跨过缺口的收益期不是单日收益，整段排除；数量随样本一起返回，不静默截短。
    excluded_return_periods: int = Field(ge=0)
    observation_frequency: Literal["daily"]
    periods_per_year: Literal[252]
    source_hash: Fingerprint
    return_panel_hash: Fingerprint


class CmaHistorySummary(Contract):
    window: CmaWindow
    start_date: date | None
    end_date: date | None
    observations: int | None
    source_names: list[str]


class ScopeFacts(Contract):
    currency: str
    asset_ids: list[str]
    asset_currencies: list[str]
    roles: list[str]
    liquidities: list[str]
    proxies: list[list[Any] | None]


class CmaListItem(Contract):
    id: Identifier
    name: str
    content_hash: Fingerprint
    created_at: str
    method: str
    as_of: date
    currency: str
    retired: bool
    schema_version: Literal["1.0", "2.0"]
    moment_semantics: str | None
    alloc_name: str | None
    strategic_universe_id: str | None
    implementation_mapping_id: str | None
    asset_ids: list[str]
    scope_name: str | None
    history: CmaHistorySummary | None = None
    downstream_eligible: bool = True
    downstream_reason: str | None = None
    scope_facts: ScopeFacts | None = None
    scope_fingerprint: Fingerprint | None = None
    research_proxy_facts: list[Any] | None = None
    return_basis: str | None = None
    fee_basis: str | None = None
    fx_hedging_basis: str | None = None
    research_path: Literal["strategy_first", "product_first"] | None = None
    version: VersionInfo
    upstream: list[UpstreamRef]
    usable: Usability


class CmaListResponse(Contract):
    items: list[CmaListItem]
    total: int
    offset: int
    limit: int


class CmaDraftView(Contract):
    id: Identifier
    name: str
    revision: int
    created_at: str
    updated_at: str
    editable_definition: dict[str, Any]
    copied_from_id: Identifier | None
    editing_ref: CmaVersionRef | None = None
