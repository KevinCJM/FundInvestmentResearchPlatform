"""Lifecycle contracts for the independent LTCMA workspace."""
from __future__ import annotations

from typing import Any, Literal
from datetime import date
import json

from pydantic import Field, field_validator, model_validator

from .common_contracts import Contract, Fingerprint, Identifier
from .contracts import PublishCmaRequest
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


class CmaRetire(ExplicitConfirm):
    content_hash: Fingerprint
    reason: str = Field(min_length=5, max_length=2000)


class CmaDraftWrite(Contract):
    name: str = Field(min_length=1, max_length=120)
    editable_definition: dict[str, Any]
    copied_from_id: Identifier | None = None
    expected_revision: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def bounded_json(self):
        try:
            encoded = json.dumps(self.editable_definition, allow_nan=False, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            raise ValueError("草稿只能包含有限 JSON 数值，未填写项使用 null。") from exc
        if len(encoded.encode("utf-8")) > 128_000:
            raise ValueError("LTCMA 草稿超过 128 KB。")
        return self


class CmaDraftDelete(Contract):
    expected_revision: int = Field(ge=1, strict=True)


class CmaListItem(Contract):
    id: Identifier
    name: str
    content_hash: Fingerprint
    created_at: str
    method: str
    as_of: date
    currency: str
    horizon_years: int
    retired: bool
    schema_version: Literal["1.0", "2.0"]
    moment_semantics: str | None
    alloc_name: str | None
    strategic_universe_id: str | None
    implementation_mapping_id: str | None
    asset_ids: list[str]
    scope_name: str | None


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
