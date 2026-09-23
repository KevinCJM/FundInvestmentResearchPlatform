"""External contracts for the indicator-center agent.

Every request/response model forbids unknown fields. Server-owned facts such as
hashes, tokens, revisions and preview state are never accepted from callers.
"""

from __future__ import annotations

import json
import math
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


AgentPage = Literal[
    "indicator-studio",
    "product-detail",
    "product-research",
    "product-compare",
    "holding-diagnosis",
    "evaluation-plan",
]
AgentScope = Literal["indicator_center", "product_research"]
ViewState = Literal["unknown", "inherit", "explicit", "off"]


class AgentError(Exception):
    """Agent-owned domain error mapped to a stable HTTP status by routes."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int,
        field: Optional[str] = None,
        diagnostics: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.field = field
        self.diagnostics = diagnostics

    def detail(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field:
            payload["field"] = self.field
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload


class EvaluationTarget(Contract):
    kind: Literal["etf", "fund"]
    product_id: str = Field(min_length=1, max_length=100)


class SingleProductContext(Contract):
    context_kind: Literal["single_product"]
    targets: list[EvaluationTarget] = Field(default_factory=list, max_length=10)
    period: str = Field(default="1Y", min_length=1, max_length=12)
    as_of: Optional[str] = Field(default=None, min_length=8, max_length=10)


class PortfolioContext(Contract):
    context_kind: Literal["portfolio"]
    run_id: str = Field(min_length=1, max_length=100)


CalculationContext = Annotated[
    SingleProductContext | PortfolioContext,
    Field(discriminator="context_kind"),
]


class PageContext(Contract):
    page: AgentPage
    page_instance_id: str = Field(min_length=1, max_length=100)
    context_revision: int = Field(default=0, ge=0)
    view_state: ViewState = "inherit"
    calculation: CalculationContext

    @property
    def context_kind(self) -> str:
        return self.calculation.context_kind


# Curated page-evidence sections each page may submit. Unknown pages/sections fail closed.
PAGE_SNAPSHOT_SECTIONS: dict[str, tuple[str, ...]] = {
    "indicator-studio": ("editing", "results", "series"),
    "product-research": ("request", "results"), "product-compare": ("request", "results"),
    "holding-diagnosis": ("request", "results"),
}
MAX_PAGE_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_PAGE_SNAPSHOT_DEPTH = 12


def _bounded_json(value: Any) -> None:
    """Reject non-finite numbers, unsupported types, excessive depth or size."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > MAX_PAGE_SNAPSHOT_DEPTH:
            raise ValueError("页面快照嵌套层级过深。")
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("页面快照包含非有限数值。")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 200:
                    raise ValueError("页面快照字段名无效。")
                stack.append((child, depth + 1))
        elif isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if len(item) > 200_000:
                raise ValueError("页面快照文本过长。")
        elif item is None or isinstance(item, (bool, int)):
            continue
        else:
            raise ValueError("页面快照包含不支持的数据类型。")
    if len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")) > MAX_PAGE_SNAPSHOT_BYTES:
        raise ValueError("页面快照超过传输上限。")


class PageSnapshotEnvelope(Contract):
    """Immutable, versioned copy of what the user's page showed when the message was sent."""

    version: Literal[1]
    snapshot_id: str = Field(pattern=r"^snap-[0-9a-f]{32}$")
    captured_at: Optional[str] = Field(default=None, max_length=40)
    page: AgentPage
    sections: dict[str, Any]

    @field_validator("sections")
    @classmethod
    def _known_sections(cls, value: dict[str, Any], info) -> dict[str, Any]:
        allowed = PAGE_SNAPSHOT_SECTIONS.get(info.data.get("page"), ())
        if not value:
            raise ValueError("页面快照没有任何分区。")
        unknown = sorted(set(value) - set(allowed))
        if unknown:
            raise ValueError(f"页面快照包含未允许的分区：{', '.join(unknown)}。")
        return value

    @model_validator(mode="after")
    def _bounded(self) -> "PageSnapshotEnvelope":
        _bounded_json(self.model_dump(exclude={"captured_at"}))
        return self


class AgentSessionCreate(Contract):
    page_context: PageContext
    scope: Optional[AgentScope] = None


class AgentMessageRequest(Contract):
    message_id: str = Field(min_length=1, max_length=64)
    expected_session_revision: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=4000)
    page_context: PageContext
    page_snapshot: Optional[PageSnapshotEnvelope] = None
    resume_from_run_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    edit_of_message_id: Optional[str] = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def _snapshot_belongs_to_page(self) -> "AgentMessageRequest":
        if self.page_snapshot is not None and self.page_snapshot.page != self.page_context.page:
            raise ValueError("页面快照与页面上下文不一致。")
        if self.page_snapshot is not None and self.page_snapshot.page != 'indicator-studio':
            from .research_pages import parse_request
            try:
                frozen = parse_request(self.page_snapshot.page, self.page_snapshot.model_dump())
            except AgentError:
                raise ValueError('页面冻结请求不符合该页面的登记契约。') from None
            if self.page_snapshot.page == 'holding-diagnosis':
                if self.page_context.context_kind != 'portfolio' or self.page_context.calculation.run_id != frozen.run_id:
                    raise ValueError('页面快照与组合运行对象不一致。')
            elif self.page_context.context_kind != 'single_product':
                raise ValueError('页面快照与计算域不一致。')
        return self


class AgentCancelRequest(Contract):
    request_id: str = Field(min_length=1, max_length=64)


class AgentInvalidateRequest(AgentCancelRequest):
    page_context: PageContext


class AgentScopeRequest(Contract):
    scope: AgentScope


class CommitPreviewRequest(Contract):
    draft_revision: int = Field(ge=0)
    definition: dict[str, Any]
    target: Optional[EvaluationTarget] = None
    page_context: PageContext


class CommitRequest(Contract):
    request_id: str = Field(min_length=1, max_length=64)
    confirmation_id: str = Field(min_length=1, max_length=64)
    definition_hash: str = Field(min_length=64, max_length=64)
    draft_revision: int = Field(ge=0)
    confirmed: bool = False


class MemoryRequest(Contract):
    proposal_id: str = Field(min_length=1, max_length=64)
    decision: Literal["accept", "reject"]
    speaker: str = Field(default="human", min_length=1, max_length=40)
    replace_memory_id: Optional[str] = None
    expected_version: Optional[int] = Field(default=None, ge=1)


class MemoryRevokeRequest(Contract):
    request_id: str = Field(min_length=1, max_length=64)
    memory_id: str = Field(pattern=r"^memory-[0-9a-f]{32}$")
    expected_version: int = Field(ge=1)
