"""External contracts for platform and page agents.

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


ResearchPage = Literal[
    "platform-agent",
    "indicator-studio",
    "product-detail",
    "product-research",
    "product-compare",
    "holding-diagnosis",
    "evaluation-plan",
    "scenario-algorithms", "historical-regimes", "published-scenarios", "global-events",
]
ResearchScope = Literal["platform", "indicator_center", "product_research", "scenario_center"]
ViewState = Literal["unknown", "inherit", "explicit", "off"]


class ResearchError(Exception):
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


class PlatformContext(Contract):
    context_kind: Literal["platform"]


class ScenarioContext(Contract):
    context_kind: Literal['scenario']
    workspace: Literal['graph', 'events', 'published', 'stress']
    mode: Literal['realtime', 'retrospective'] = 'realtime'
    purpose: str = Field(default='research', max_length=80)
    as_of: str | None = Field(default=None, max_length=10)
    definition_id: str | None = Field(default=None, max_length=120)
    definition_revision: int | None = Field(default=None, ge=1)


CalculationContext = Annotated[
    SingleProductContext | PortfolioContext | PlatformContext | ScenarioContext,
    Field(discriminator="context_kind"),
]


class PageContext(Contract):
    page: ResearchPage
    page_instance_id: str = Field(min_length=1, max_length=100)
    context_revision: int = Field(default=0, ge=0)
    view_state: ViewState = "inherit"
    calculation: CalculationContext

    @model_validator(mode="after")
    def _platform_domain(self):
        if (self.page == "platform-agent") != (self.context_kind == "platform"):
            raise ValueError("平台助手与业务计算上下文不能混用。")
        return self

    @property
    def context_kind(self) -> str:
        return self.calculation.context_kind


# Curated page-evidence sections each page may submit. Unknown pages/sections fail closed.
PAGE_SNAPSHOT_SECTIONS: dict[str, tuple[str, ...]] = {
    "indicator-studio": ("editing", "results", "series"),
    "product-research": ("request", "results"), "product-compare": ("request", "results"),
    "holding-diagnosis": ("request", "results"),
}
PAGE_SNAPSHOT_SECTIONS.update({page: ('editing', 'request', 'results') for page in ('scenario-algorithms','historical-regimes','published-scenarios','global-events')})
MAX_PAGE_SNAPSHOT_BYTES = 2 * 1024 * 1024
MAX_PAGE_SNAPSHOT_DEPTH = 12


def _bounded_json(value: Any, *, max_depth: int = MAX_PAGE_SNAPSHOT_DEPTH) -> None:
    """Reject non-finite numbers, unsupported types, excessive depth or size."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if depth > max_depth:
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
    page: ResearchPage
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
