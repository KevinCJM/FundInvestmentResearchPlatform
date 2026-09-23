"""Atomic DATA_DIR-backed storage for the LLM API configuration.

The API key is written with the existing ``AtomicJsonStore`` (0600 temp file,
fsync, atomic replace) and is never returned to callers: read responses expose
only ``configured`` and a tail mask.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

from .storage import AgentJsonStore
from .contracts import AgentError

SETTINGS_SCHEMA_VERSION = 2
DEFAULT_PROVIDER = "openai-compatible"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
MAX_API_KEY_LENGTH = 512
MAX_TEXT_LENGTH = 300
ReasoningEffort = Literal['default', 'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max']
# Never reveal a tail for short keys: a 4-char tail of an 8-char key is half the secret.
MASKED_TAIL_MIN_LENGTH = 8


def resolve_agent_data_dir() -> Path:
    """Resolve the agent storage root the same way indicator storage does."""

    override = os.getenv("CUSTOM_INDICATOR_DATA_DIR")
    if override:
        return Path(override)
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def mask_api_key(api_key: str) -> str:
    if len(api_key) > MASKED_TAIL_MIN_LENGTH:
        return "••••" + api_key[-4:]
    return "••••"


class LlmSettingsStore:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = Path(path) if path is not None else resolve_agent_data_dir() / "llm_settings.json"
        self.store = AgentJsonStore(self.path)

    def _document(self):
        payload = self.store.read_unlocked()
        if payload.get("schema_version") == SETTINGS_SCHEMA_VERSION:
            return payload
        # Read existing installations through the one current profile format.
        previous = payload.get("settings") or {}
        profiles = [{"id": "default", "name": "默认 API", **previous}] if previous else []
        return {"schema_version": SETTINGS_SCHEMA_VERSION, "profiles": profiles,
                "active_profile_id": "default" if profiles else None}

    @staticmethod
    def _settings(profile):
        return {"provider": profile.get("provider") or DEFAULT_PROVIDER,
                "base_url": profile.get("base_url") or DEFAULT_BASE_URL,
                "model": profile.get("model") or DEFAULT_MODEL,
                "reasoning_effort": profile.get("reasoning_effort") or "default",
                "context_window_tokens": profile.get("context_window_tokens") or 0,
                "timeout_seconds": float(profile.get("timeout_seconds") or 60.0),
                "api_key": str(profile.get("api_key") or "")}

    @classmethod
    def _public(cls, payload):
        def redacted(profile):
            settings = cls._settings(profile)
            key = settings.pop("api_key")
            return {**settings, "configured": bool(key), "api_key_masked": mask_api_key(key) if key else None}
        profiles = [{"id": p["id"], "name": p.get("name", "默认 API"), **redacted(p)} for p in payload["profiles"]]
        active = next((p for p in payload["profiles"] if p["id"] == payload.get("active_profile_id")), {})
        return {**redacted(active), "profiles": profiles, "active_profile_id": payload.get("active_profile_id")}

    def read(self) -> dict[str, Any]:
        """The runner receives only the explicitly active API, never a fallback."""
        with self.store.locked():
            payload = self._document()
            active = next((p for p in payload["profiles"] if p["id"] == payload.get("active_profile_id")), {})
            return self._settings(active)

    def save_profile(self, profile_id=None, *, create=False, activate=False, **fields):
        with self.store.locked():
            payload = self._document()
            if not create and profile_id is None:
                profile_id = payload.get("active_profile_id")
                create = profile_id is None
            if create:
                profile = {"id": uuid.uuid4().hex, "name": "默认 API"}
                payload["profiles"].append(profile)
            else:
                profile = next((p for p in payload["profiles"] if p["id"] == profile_id), None)
                if profile is None:
                    raise AgentError("LLM_PROFILE_NOT_FOUND", "未找到该 API 配置，请刷新后重试。", status_code=404)
            for field, value in fields.items():
                if value is not None:
                    profile[field] = value.strip() if isinstance(value, str) else value
            if activate:
                payload["active_profile_id"] = profile["id"]
            self.store.write_unlocked(payload)
            return self._public(payload)

    def update(self, **fields):
        """Compatibility for callers configuring the active API through PUT /llm."""
        return self.save_profile(activate=True, **fields)

    def activate(self, profile_id):
        with self.store.locked():
            payload = self._document()
            if profile_id is not None:
                profile = next((p for p in payload["profiles"] if p["id"] == profile_id), None)
                if profile is None:
                    raise AgentError("LLM_PROFILE_NOT_FOUND", "未找到该 API 配置。", status_code=404)
                if not self._settings(profile)["api_key"]:
                    raise AgentError("LLM_PROFILE_INCOMPLETE", "请先保存该 API 的密钥。", status_code=400)
            payload["active_profile_id"] = profile_id
            self.store.write_unlocked(payload)
            return self._public(payload)

    def public(self) -> dict[str, Any]:
        with self.store.locked():
            return self._public(self._document())
