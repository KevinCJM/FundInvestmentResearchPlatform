"""TAA versions reuse the managed, immutable research artifact store."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from backend.custom_indicators.errors import ValidationError
from backend.sensitivity.repository import ArtifactRepository


class TacticalAllocationRepository:
    def __init__(self, data_dir: Path):
        self.artifacts = ArtifactRepository(Path(data_dir) / "tactical_allocation" / "artifacts")

    def _list(self, kind: str, artifact_type: str) -> list[dict[str, Any]]:
        return [item for summary in self.artifacts.list(kind)
                if (item := self.artifacts.get(summary["id"], kind)).get("artifact_type") == artifact_type]

    def _get(self, identifier: str, kind: str, artifact_type: str) -> dict[str, Any]:
        item = self.artifacts.get(identifier, kind)
        if item.get("artifact_type") != artifact_type:
            raise ValidationError("TAA_VERSION_TYPE", "所选版本类型不匹配，请重新选择。")
        return item

    def list_baselines(self) -> list[dict[str, Any]]:
        return self._list("series", "saa_baseline")

    def get_baseline(self, identifier: str) -> dict[str, Any]:
        return self._get(identifier, "series", "saa_baseline")

    @staticmethod
    def _fields(fields: dict[str, Any]) -> dict[str, Any]:
        reserved = {"id", "kind", "schema_version", "created_at", "immutable", "arrays", "content_hash", "transient", "preview_hash"}
        return {key: value for key, value in fields.items() if key not in reserved}

    def save_baseline(self, fields: dict[str, Any]) -> dict[str, Any]:
        return self.artifacts.save("series", {**self._fields(fields), "artifact_type": "saa_baseline"})

    def list_decisions(self) -> list[dict[str, Any]]:
        return self._list("run", "taa_decision")

    def get_decision(self, identifier: str) -> dict[str, Any]:
        return self._get(identifier, "run", "taa_decision")

    def save_decision(self, fields: dict[str, Any], arrays: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        return self.artifacts.save("run", {**self._fields(fields), "artifact_type": "taa_decision"}, arrays)

    def decision_arrays(self, identifier: str) -> dict[str, np.ndarray]:
        self.get_decision(identifier)
        arrays = self.artifacts.arrays(identifier)
        if "returns" not in arrays:
            raise ValidationError("TAA_SNAPSHOT_MISSING", "研究版本缺少冻结收益输入，请重新计算并保存后再交接。")
        return arrays
