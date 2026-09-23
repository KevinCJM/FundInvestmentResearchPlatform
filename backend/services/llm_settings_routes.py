"""LLM API settings routes.

GET/PUT ``/api/settings/llm`` store the provider configuration under
``DATA_DIR``.  The API key is write-only from the HTTP surface: responses only
say whether a key is configured and show a tail mask, never the key itself.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.contracts import AgentError
from agent.llm_settings import LlmSettingsStore, ReasoningEffort
from custom_indicators.errors import IndicatorDomainError
from services.custom_indicator_routes import StableValidationRoute

router = APIRouter(tags=["settings"], route_class=StableValidationRoute)


class LlmSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Optional[str] = Field(default=None, min_length=1, max_length=300)
    base_url: Optional[str] = Field(default=None, min_length=8, max_length=300, pattern=r"^https?://")
    model: Optional[str] = Field(default=None, min_length=1, max_length=300)
    # Omit to keep the stored key; send an empty string to clear it.
    api_key: Optional[str] = Field(default=None, max_length=512)
    reasoning_effort: Optional[ReasoningEffort] = None
    timeout_seconds: Optional[int] = Field(default=None, ge=10, le=1800)
    context_window_tokens: Optional[int] = Field(default=None, ge=0, le=2_000_000, strict=True)

    @field_validator('context_window_tokens')
    @classmethod
    def context_capacity(cls, value):
        if value is not None and value != 0 and value < 8192:
            raise ValueError('上下文窗口须为 0（自动）或至少 8192 tokens')
        return value

    @field_validator('provider', 'model', 'base_url')
    @classmethod
    def nonblank(cls, value):
        if value is not None and not value.strip():
            raise ValueError('不能为空')
        return value.strip() if value is not None else value


class LlmProfileUpdate(LlmSettingsUpdate):
    name: str = Field(min_length=1, max_length=80, pattern=r'\S')


class LlmProfileCreate(LlmProfileUpdate):
    provider: str = Field(default='openai-compatible', min_length=1, max_length=300)
    base_url: str = Field(min_length=8, max_length=300, pattern=r'^https?://')
    model: str = Field(min_length=1, max_length=300)


class LlmActiveUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile_id: Optional[str] = Field(max_length=80)


def _call(fn):
    try:
        return fn()
    except (AgentError, IndicatorDomainError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail()) from exc


@router.get("/api/settings/llm")
def get_llm_settings():
    return _call(lambda: LlmSettingsStore().public())


@router.put("/api/settings/llm")
def update_llm_settings(request: LlmSettingsUpdate):
    def run():
        store = LlmSettingsStore()
        store.update(
            provider=request.provider,
            base_url=request.base_url,
            model=request.model,
            api_key=request.api_key,
            reasoning_effort=request.reasoning_effort,
            timeout_seconds=request.timeout_seconds,
            context_window_tokens=request.context_window_tokens,
        )
        return store.public()

    return _call(run)


@router.post('/api/settings/llm/profiles', status_code=201)
def create_llm_profile(request: LlmProfileCreate):
    return _call(lambda: LlmSettingsStore().save_profile(create=True, **request.model_dump()))


@router.put('/api/settings/llm/profiles/{profile_id}')
def update_llm_profile(profile_id: str, request: LlmProfileUpdate):
    return _call(lambda: LlmSettingsStore().save_profile(profile_id, **request.model_dump()))


@router.put('/api/settings/llm/active')
def activate_llm_profile(request: LlmActiveUpdate):
    return _call(lambda: LlmSettingsStore().activate(request.profile_id))
