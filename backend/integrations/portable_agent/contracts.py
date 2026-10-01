"""Host protocol DTOs; execution and conversation types live in portable-web-agent."""
from typing import Literal
from pydantic import Field, model_validator
from research_access.contracts import Contract, PageContext, PageSnapshotEnvelope, _bounded_json

ID = r'^[a-zA-Z0-9_-]{1,64}$'


class ContextInput(Contract):
    page_context: PageContext
    page_snapshot: PageSnapshotEnvelope | None = None
    capability_id: str | None = Field(default=None, max_length=100)
    intent_parameters: dict = Field(default_factory=dict)
    authoring_id: str | None = Field(default=None, pattern=ID)

    @model_validator(mode='after')
    def consistent(self):
        if self.page_snapshot and self.page_snapshot.page != self.page_context.page:
            raise ValueError('Page evidence must belong to the declared page')
        if self.page_snapshot:
            from research_access.contracts import ResearchError
            from research_access.research_pages import REQUESTS, parse_request
            snapshot = self.page_snapshot
            if snapshot.page in REQUESTS:
                try:
                    frozen = parse_request(snapshot.page, snapshot.model_dump())
                except ResearchError:
                    raise ValueError('页面冻结请求不符合该页面的登记契约。') from None
                if snapshot.page == 'holding-diagnosis':
                    if self.page_context.context_kind != 'portfolio' or self.page_context.calculation.run_id != frozen.run_id:
                        raise ValueError('页面快照与组合运行对象不一致。')
                elif self.page_context.context_kind != 'single_product':
                    raise ValueError('页面快照与计算域不一致。')
            elif self.page_context.context_kind == 'scenario':
                editing = snapshot.sections.get('editing')
                if (not isinstance(editing, dict)
                        or editing.get('mode') != self.page_context.calculation.mode
                        or editing.get('as_of') != self.page_context.calculation.as_of):
                    raise ValueError('情景页面快照与当前研究模式或研究日不一致。')
        _bounded_json(self.model_dump())
        return self


class BootstrapInput(Contract):
    context_ref: str = Field(pattern=r'^ctx-[0-9a-f]{32}$')


class AgentRelease(Contract):
    source_commit: str = Field(pattern=r'^[0-9a-f]{40}$')
    image_digest: str = Field(pattern=r'^[^\s@]+@sha256:[0-9a-f]{64}$')
    protocol_major: Literal[2]
    widget_version: str = Field(min_length=1, max_length=64)
    manifest_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    required_capabilities: list[str] = Field(min_length=1, max_length=40)


class AdoptionInput(Contract):
    operation_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    session_id: str = Field(pattern=ID)
    run_id: str = Field(pattern=ID)
    context_ref: str = Field(pattern=r'^ctx-[0-9a-f]{32}$')


class ToolInput(Contract):
    protocol_version: Literal[2]
    application: str = Field(max_length=48)
    subject: str = Field(min_length=1, max_length=200)
    session_id: str = Field(pattern=ID)
    run_id: str = Field(pattern=ID)
    operation_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    context: dict
    arguments: dict

    @model_validator(mode='after')
    def bounded(self):
        _bounded_json(self.model_dump())
        return self


class AuthorityInput(Contract):
    application: str = Field(max_length=48)
    subject: str = Field(min_length=1, max_length=200)
    context: dict
    scopes: list[str] = Field(default_factory=list, max_length=100)
    action: str = Field(max_length=40)
    operation_id: str | None = Field(default=None, pattern=ID)
    session_id: str | None = Field(default=None, pattern=ID)
    run_id: str | None = Field(default=None, pattern=ID)
    identity_only: bool = False
    source_ref: str | None = Field(default=None, pattern=ID)


class AdmissionInput(Contract):
    application: str = Field(max_length=48)
    subject: str = Field(min_length=1, max_length=200)
    context: dict
    phase: Literal['input', 'prepare', 'before', 'after'] = 'prepare'
    purpose: Literal['primary', 'summary'] = 'primary'
    messages: list[dict] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list, max_length=80)
    wire: dict | None = None
    payload_hash: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    receipt_id: str | None = Field(default=None, pattern=ID)
    run_id: str | None = Field(default=None, pattern=ID)

    @model_validator(mode='after')
    def bounded(self):
        # Tool JSON Schemas nest more deeply than page evidence. Message/argument data keeps its own stricter gates.
        _bounded_json(self.model_dump(), max_depth=64)
        return self


class AuthoringInput(Contract):
    context_ref: str = Field(pattern=r'^ctx-[0-9a-f]{32}$')
    request_id: str = Field(pattern=ID)


class ConfirmationInput(Contract):
    context_ref: str = Field(pattern=r'^ctx-[0-9a-f]{32}$')
    expected_revision: int = Field(ge=0)
    definition_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    target: dict | None = None


class CommitInput(ConfirmationInput):
    confirmation_id: str = Field(pattern=ID)
    request_id: str = Field(pattern=ID)
    confirmed: Literal[True]
