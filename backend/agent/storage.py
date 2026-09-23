"""Agent JSON storage: reuse the indicator repository's atomic writer.

``AtomicJsonStore`` owns the file lock, path guard, temporary file, fsync and
atomic replace.  Agent documents are plain objects instead of the indicator
repository's ``items`` list, so only ``read_unlocked`` is specialized.
"""

from __future__ import annotations

import json
from typing import Any

from backend.data_storage import guard_path
from custom_indicators.repository import AtomicJsonStore

from .contracts import AgentError


class AgentJsonStore(AtomicJsonStore):
    def read_unlocked(self) -> dict[str, Any]:
        guard_path(self.path)
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AgentError(
                "AGENT_STORAGE_CORRUPT",
                "智能体本地数据无法读取，请检查数据文件。",
                status_code=500,
            ) from exc
        if not isinstance(payload, dict):
            raise AgentError(
                "AGENT_STORAGE_CORRUPT",
                "智能体本地数据格式无效。",
                status_code=500,
            )
        return payload
