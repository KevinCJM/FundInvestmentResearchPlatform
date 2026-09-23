"""Scope, page and tool whitelists for the agent.

Two scopes exist: the indicator center and product research. Tools declare
which calculation domain they accept so a portfolio page can never receive a
single-product target and a single-product page can never run portfolio math.
"""

from __future__ import annotations

from .contracts import AgentError, PageContext
from .tools import TOOL_REGISTRY

SCOPE_INDICATOR_CENTER = "indicator_center"
SCOPE_PRODUCT_RESEARCH = "product_research"

SCOPE_LABELS = {
    SCOPE_INDICATOR_CENTER: "指标中心",
    SCOPE_PRODUCT_RESEARCH: "产品研究",
}

PAGE_SCOPES: dict[str, str] = {
    "indicator-studio": SCOPE_INDICATOR_CENTER,
    "evaluation-plan": SCOPE_INDICATOR_CENTER,
    "product-detail": SCOPE_PRODUCT_RESEARCH,
    "product-research": SCOPE_PRODUCT_RESEARCH,
    "product-compare": SCOPE_PRODUCT_RESEARCH,
    "holding-diagnosis": SCOPE_PRODUCT_RESEARCH,
}

SCOPE_TOOLS = {scope: tuple(name for name, tool in TOOL_REGISTRY.items() if scope in tool.scopes)
               for scope in SCOPE_LABELS}


def scope_for_page(page: str) -> str:
    scope = PAGE_SCOPES.get(page)
    if scope is None:
        raise AgentError(
            "AGENT_PAGE_NOT_SUPPORTED",
            "该页面尚未接入智能体。",
            status_code=422,
            field="page_context.page",
        )
    return scope


def validate_scope_for_page(scope: str, page: str) -> None:
    if scope not in SCOPE_TOOLS:
        raise AgentError(
            "AGENT_SCOPE_UNKNOWN",
            "未知的智能体工作域。",
            status_code=422,
            field="scope",
        )
    if PAGE_SCOPES.get(page) != scope:
        raise AgentError(
            "AGENT_SCOPE_PAGE_MISMATCH",
            "该工作域不包含当前页面。",
            status_code=422,
            field="scope",
        )


def allowed_tools(scope: str, context_kind: str) -> tuple[str, ...]:
    return tuple(name for name, tool in TOOL_REGISTRY.items()
                 if scope in tool.scopes and context_kind in tool.domains)


def require_tool(scope: str, context_kind: str, tool: str) -> None:
    """Fail closed on any tool outside the scope/domain whitelist."""

    if tool not in SCOPE_TOOLS.get(scope, ()):
        raise AgentError(
            "AGENT_TOOL_NOT_ALLOWED",
            "该工具不在当前工作域的白名单内。",
            status_code=409,
            field="tool",
        )
    if context_kind not in TOOL_REGISTRY[tool].domains:
        message = ("该组合工具只读取组合运行快照，不接受单产品上下文。"
                   if TOOL_REGISTRY[tool].domains == ("portfolio",) else "该工具只允许在单产品计算域使用。")
        raise AgentError("AGENT_TOOL_DOMAIN_MISMATCH", message, status_code=409, field="tool")


def validate_page_context(page_context: PageContext, *, pit_off: bool, allow_authoring: bool = False) -> str:
    """Validate page/scope/view-state and return the resolved scope."""

    scope = scope_for_page(page_context.page)
    if page_context.view_state == "unknown" and not allow_authoring:
        raise AgentError(
            "AGENT_CONTEXT_CHANGED",
            "页面研究口径尚未确认，请先在页面选择 PIT 口径后重试。",
            status_code=409,
            field="page_context.view_state",
        )
    if page_context.view_state == "off" and not pit_off:
        raise AgentError(
            "AGENT_CONTEXT_CHANGED",
            "关闭 PIT 口径需要页面显式提交 x-pit-off 请求头。",
            status_code=409,
            field="page_context.view_state",
        )
    return scope
