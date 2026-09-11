"""Application service for versioned scenario simulation and stress testing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from compute_policy import validate_execution_audit
from custom_indicators.errors import IndicatorDomainError, ValidationError
from custom_indicators.repository import utc_now

from .contracts import APPLICATION_TARGETS, meta_contract, normalize_definition
from .engine import execute
from .numba_kernels import (
    metric_deltas_kernel,
    normalize_transition_counts_kernel,
    prepare_historical_state_samples_kernel,
    scenario_stress_numba_status,
)
from .repository import ScenarioDefinitionRepository, ScenarioRunRepository


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HISTORICAL_ARTIFACT_DIRECTORY = "historical_regime_v2_artifacts"
MAX_HISTORICAL_EVALUATION_ARTIFACT_BYTES = 512 * 1024 * 1024


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _content_hash(value: Any) -> str:
    encoded = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sha256_digest(value: Any, field: str) -> str:
    text = str(value or "")
    if (
        not text.startswith("sha256:")
        or len(text) != 71
        or any(character not in "0123456789abcdef" for character in text[7:])
    ):
        raise ValidationError("INVALID_CONTENT_LOCK", f"{field} 必须是 sha256 内容锁。", field)
    return text


def _raw_sha256(value: Any, field: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValidationError("INVALID_CONTENT_LOCK", f"{field} 必须是 64 位 sha256 内容哈希。", field)
    return text


def _simple_return(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError("INVALID_STATE_RETURN", "状态收益必须是大于 -100% 的有限数值。", field)
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("INVALID_STATE_RETURN", "状态收益必须是大于 -100% 的有限数值。", field) from exc
    if not math.isfinite(numeric) or numeric <= -1.0:
        raise ValidationError("INVALID_STATE_RETURN", "状态收益必须是大于 -100% 的有限数值。", field)
    return numeric


def _business_definition(definition: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "description",
        "method",
        "horizon",
        "initial_nav",
        "factors",
        "assets",
        "portfolios",
        "mapping",
        "scenario",
        "limits",
        "usage_intent",
        "schema_version",
    )
    return {key: copy.deepcopy(definition.get(key)) for key in keys}


def _stored_run_snapshot_hash(run: dict[str, Any]) -> str:
    """Recreate the hash of the analytical payload before repository metadata."""

    analytical = copy.deepcopy(run)
    for key in ("id", "created_at", "immutable", "publications", "content_hash"):
        analytical.pop(key, None)
    # Publications append bindings as governance metadata; the analytical run
    # was originally hashed with an empty bindings list.
    analytical["application_bindings"] = []
    return _content_hash(analytical)


class ScenarioStressService:
    def __init__(
        self,
        workspace_data_dir: Optional[Path] = None,
        market_data_dir: Optional[Path] = None,
    ) -> None:
        configured = os.getenv("SCENARIO_STRESS_DATA_DIR") or os.getenv("CUSTOM_INDICATOR_DATA_DIR")
        self.workspace_data_dir = workspace_data_dir or (Path(configured) if configured else DEFAULT_DATA_DIR)
        self.market_data_dir = market_data_dir or DEFAULT_DATA_DIR
        self.definitions = ScenarioDefinitionRepository(self.workspace_data_dir / "scenario_stress_definitions.json")
        self.runs = ScenarioRunRepository(self.workspace_data_dir / "scenario_stress_runs.json")

    @staticmethod
    def meta() -> dict[str, Any]:
        payload = meta_contract()
        payload["compute_audit"] = scenario_stress_numba_status()
        return payload

    def list_definitions(self, include_archived: bool = False) -> list[dict[str, Any]]:
        return self.definitions.list(include_archived)

    def get_definition(self, definition_id: str, revision: Optional[int] = None) -> dict[str, Any]:
        return self.definitions.get(definition_id, revision)

    def create_definition(self, fields: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_definition(fields)
        for key in ("id", "revision", "created_at", "updated_at", "archived", "archived_at"):
            normalized.pop(key, None)
        return self.definitions.create(normalized)

    def update_definition(self, definition_id: str, revision: int, fields: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_definition(fields)
        for key in ("id", "revision", "created_at", "updated_at", "archived", "archived_at"):
            normalized.pop(key, None)
        return self.definitions.update(definition_id, revision, normalized)

    def archive_definition(self, definition_id: str, revision: int) -> dict[str, Any]:
        return self.definitions.archive(definition_id, revision)

    def _resolve_definition(self, requested: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if not isinstance(requested, dict):
            raise ValidationError("INVALID_DEFINITION", "run.definition 必须是情景定义或版本引用。", "definition")
        definition_id = requested.get("id")
        if definition_id:
            revision = requested.get("revision")
            only_reference = not any(key in requested for key in ("name", "method", "type", "assets", "scenario"))
            if only_reference:
                persisted = self.get_definition(str(definition_id), int(revision) if revision is not None else None)
                if persisted.get("archived"):
                    raise ValidationError("SCENARIO_DEFINITION_ARCHIVED", "情景定义已归档，不能发起新运行。", "definition.id")
                return persisted, "repository_reference"
            supplied = normalize_definition(requested)
            trial = copy.deepcopy(supplied)
            for key in ("id", "revision", "created_at", "updated_at", "archived", "archived_at"):
                trial.pop(key, None)
            try:
                persisted = self.get_definition(str(definition_id), int(revision) if revision is not None else None)
            except IndicatorDomainError:
                return trial, "inline_trial"
            if _content_hash(_business_definition(supplied)) == _content_hash(_business_definition(persisted)):
                if persisted.get("archived"):
                    raise ValidationError("SCENARIO_DEFINITION_ARCHIVED", "情景定义已归档，不能发起新运行。", "definition.id")
                return persisted, "repository_reference"
            return trial, "inline_trial"
        return normalize_definition(requested), "inline_trial"

    def _load_evaluation_artifact(
        self,
        run: dict[str, Any],
        target_ids: set[str],
        expected_checksum: str,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        manifest = (run.get("artifact_manifest") or {}).get("evaluation_targets")
        if not isinstance(manifest, dict):
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_REQUIRED",
                "历史情景运行没有持久化的评价目标制品，不能估计状态收益分布。",
                "scenario.transition.asset_return_source",
            )
        artifact_id = str(manifest.get("artifact_id") or "")
        prefix = "regime-output-sha256-"
        digest = artifact_id.removeprefix(prefix)
        manifest_checksum = _sha256_digest(
            manifest.get("checksum"),
            "artifact_manifest.evaluation_targets.checksum",
        )
        if (
            not artifact_id.startswith(prefix)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or manifest_checksum != f"sha256:{digest}"
            or manifest_checksum != expected_checksum
            or manifest.get("format") != "npz"
            or manifest.get("schema_version") != "regime-node-output-v1"
        ):
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_LOCK_MISMATCH",
                "评价目标制品与定义中锁定的内容版本不一致。",
                "scenario.transition.asset_return_source.evaluation_artifact_checksum",
            )
        artifact_root = (self.workspace_data_dir / HISTORICAL_ARTIFACT_DIRECTORY).resolve()
        path = (artifact_root / f"{digest}.npz").resolve()
        if path.parent != artifact_root or not path.is_file():
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_NOT_FOUND",
                "锁定的评价目标制品不存在。",
                "scenario.transition.asset_return_source.evaluation_artifact_checksum",
            )
        size_bytes = int(path.stat().st_size)
        if size_bytes > MAX_HISTORICAL_EVALUATION_ARTIFACT_BYTES:
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_TOO_LARGE",
                "评价目标制品超过情景压测读取上限。",
                "scenario.transition.asset_return_source.evaluation_artifact_checksum",
            )
        if manifest.get("size_bytes") is not None:
            try:
                declared_size = int(manifest["size_bytes"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(
                    "HISTORICAL_EVALUATION_ARTIFACT_SIZE_MISMATCH",
                    "评价目标制品大小清单无效。",
                    "artifact_manifest.evaluation_targets.size_bytes",
                ) from exc
            if declared_size != size_bytes:
                raise ValidationError(
                    "HISTORICAL_EVALUATION_ARTIFACT_SIZE_MISMATCH",
                    "评价目标制品大小与不可变清单不一致。",
                    "artifact_manifest.evaluation_targets.size_bytes",
                )
        if _sha256_file(path) != digest:
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_CHECKSUM_MISMATCH",
                "评价目标制品校验失败。",
                "artifact_manifest.evaluation_targets.checksum",
            )

        catalog = manifest.get("arrays")
        if not isinstance(catalog, list):
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_INVALID",
                "评价目标制品缺少数组目录。",
                "artifact_manifest.evaluation_targets.arrays",
            )
        entries: dict[str, dict[str, Any]] = {}
        for item in catalog:
            if not isinstance(item, dict) or item.get("port") != "value":
                continue
            target_id = str(item.get("node_id") or "")
            if target_id in entries:
                raise ValidationError(
                    "HISTORICAL_EVALUATION_ARTIFACT_INVALID",
                    "评价目标制品包含重复目标。",
                    "artifact_manifest.evaluation_targets.arrays",
                )
            entries[target_id] = item
        if not target_ids.issubset(entries):
            raise ValidationError(
                "HISTORICAL_EVALUATION_TARGET_MISSING",
                "锁定制品未包含全部已选择的评价目标。",
                "scenario.transition.asset_return_source.asset_target_map",
                diagnostics=[{"missing_target_ids": sorted(target_ids - set(entries))}],
            )

        results = run.get("evaluation_results")
        if not isinstance(results, dict):
            raise ValidationError(
                "HISTORICAL_EVALUATION_RESULTS_REQUIRED",
                "历史情景运行没有状态条件评价结果。",
                "evaluation_results",
            )
        state_ids = {str(item) for item in (run.get("transition") or {}).get("states", [])}
        loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        try:
            with np.load(path, allow_pickle=False) as payload:
                for target_id in sorted(target_ids):
                    target_result = results.get(target_id)
                    if not isinstance(target_result, dict):
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_TARGET_MISSING",
                            "历史情景运行缺少已选择的评价目标。",
                            "scenario.transition.asset_return_source.asset_target_map",
                            diagnostics=[{"target_id": target_id}],
                        )
                    metric_states = {
                        str(item.get("state_id"))
                        for item in target_result.get("conditional_metrics", [])
                        if isinstance(item, dict)
                    }
                    if metric_states != state_ids:
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_METRICS_MISMATCH",
                            "评价目标的状态条件表现与转移状态字典不一致。",
                            f"evaluation_results.{target_id}.conditional_metrics",
                        )
                    entry = entries[target_id]
                    if target_result.get("artifact") != entry:
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_TARGET_LINEAGE_MISMATCH",
                            "评价目标摘要与持久化数组目录不一致。",
                            f"evaluation_results.{target_id}.artifact",
                        )
                    array_prefix = str(entry.get("array_prefix") or "")
                    values_key = f"{array_prefix}_values"
                    dates_key = f"{array_prefix}_dates"
                    available_key = f"{array_prefix}_available"
                    if any(key not in payload.files for key in (values_key, dates_key, available_key)):
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_ARTIFACT_INVALID",
                            "评价目标制品缺少声明的数值或日期数组。",
                            f"evaluation_results.{target_id}.artifact",
                        )
                    raw_values = payload[values_key]
                    raw_dates = payload[dates_key]
                    raw_available = payload[available_key]
                    if (
                        str(entry.get("dtype") or "") != "float64"
                        or raw_values.dtype != np.dtype(np.float64)
                        or raw_dates.dtype != np.dtype(np.int64)
                        or raw_available.dtype != np.dtype(np.int64)
                    ):
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_ARTIFACT_DTYPE_MISMATCH",
                            "评价目标制品的数据类型与固定计算签名不一致。",
                            f"evaluation_results.{target_id}.artifact",
                        )
                    values = np.ascontiguousarray(raw_values, dtype=np.float64)
                    dates = np.ascontiguousarray(raw_dates, dtype=np.int64)
                    available = np.ascontiguousarray(raw_available, dtype=np.int64)
                    declared_shape = entry.get("shape")
                    if (
                        values.ndim != 1
                        or dates.ndim != 1
                        or available.ndim != 1
                        or values.shape != dates.shape
                        or values.shape != available.shape
                        or declared_shape != list(values.shape)
                    ):
                        raise ValidationError(
                            "HISTORICAL_EVALUATION_ARTIFACT_SHAPE_MISMATCH",
                            "评价目标制品的时序维度与清单不一致。",
                            f"evaluation_results.{target_id}.artifact",
                        )
                    loaded[target_id] = (values.copy(), dates.copy())
        except ValidationError:
            raise
        except (OSError, ValueError, KeyError) as exc:
            raise ValidationError(
                "HISTORICAL_EVALUATION_ARTIFACT_INVALID",
                "评价目标制品无法安全读取。",
                "artifact_manifest.evaluation_targets",
            ) from exc
        return loaded

    def _historical_asset_distribution(
        self,
        definition: dict[str, Any],
        run: dict[str, Any],
        inline: dict[str, Any],
        state_ids: list[str],
        by_id: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        source = inline.get("asset_return_source")
        if not isinstance(source, dict):
            raise ValidationError(
                "INVALID_HISTORICAL_ASSET_RETURN_SOURCE",
                "asset_return_source 必须是显式配置对象。",
                "scenario.transition.asset_return_source",
            )
        allowed_keys = {
            "kind",
            "run_content_hash",
            "evaluation_artifact_checksum",
            "asset_target_map",
            "sampling",
            "minimum_observations_per_state",
            "inline_policy",
        }
        unknown_keys = sorted(set(source) - allowed_keys)
        if unknown_keys:
            raise ValidationError(
                "UNKNOWN_HISTORICAL_ASSET_RETURN_SOURCE_FIELD",
                "asset_return_source 包含不支持的字段。",
                "scenario.transition.asset_return_source",
                diagnostics=[{"unknown_fields": unknown_keys}],
            )
        if source.get("kind") != "historical_evaluation_targets":
            raise ValidationError(
                "INVALID_HISTORICAL_ASSET_RETURN_SOURCE",
                "asset_return_source.kind 必须是 historical_evaluation_targets。",
                "scenario.transition.asset_return_source.kind",
            )
        if str(run.get("schema_version") or "") != "2.0":
            raise ValidationError(
                "HISTORICAL_EVALUATION_REQUIRES_V2",
                "只有历史情景 v2 运行提供可锁定的多评价目标制品。",
                "scenario.transition.asset_return_source",
            )
        expected_run_hash = _raw_sha256(
            source.get("run_content_hash"),
            "scenario.transition.asset_return_source.run_content_hash",
        )
        if expected_run_hash != str(run.get("content_hash") or ""):
            raise ValidationError(
                "HISTORICAL_RUN_CONTENT_LOCK_MISMATCH",
                "asset_return_source 锁定的运行内容与所选发布版本不一致。",
                "scenario.transition.asset_return_source.run_content_hash",
            )
        expected_artifact_checksum = _sha256_digest(
            source.get("evaluation_artifact_checksum"),
            "scenario.transition.asset_return_source.evaluation_artifact_checksum",
        )
        sampling = str(source.get("sampling") or "empirical_bootstrap")
        if sampling != "empirical_bootstrap":
            raise ValidationError(
                "UNSUPPORTED_HISTORICAL_DISTRIBUTION_SAMPLING",
                "当前仅支持 empirical_bootstrap 状态条件联合分布抽样。",
                "scenario.transition.asset_return_source.sampling",
            )
        inline_policy = str(source.get("inline_policy") or "forbid")
        if inline_policy not in {"forbid", "override"}:
            raise ValidationError(
                "INVALID_HISTORICAL_INLINE_POLICY",
                "inline_policy 必须是 forbid 或 override。",
                "scenario.transition.asset_return_source.inline_policy",
            )
        minimum = source.get("minimum_observations_per_state", 5)
        if isinstance(minimum, bool):
            raise ValidationError(
                "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
                "每状态最少观测数必须是 2 至 20000 的整数。",
                "scenario.transition.asset_return_source.minimum_observations_per_state",
            )
        raw_minimum = minimum
        try:
            minimum = int(raw_minimum)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
                "每状态最少观测数必须是 2 至 20000 的整数。",
                "scenario.transition.asset_return_source.minimum_observations_per_state",
            ) from exc
        if (
            isinstance(raw_minimum, (float, np.floating))
            and float(raw_minimum) != float(minimum)
        ) or minimum < 2 or minimum > 20_000:
            raise ValidationError(
                "INVALID_HISTORICAL_MINIMUM_OBSERVATIONS",
                "每状态最少观测数必须是 2 至 20000 的整数。",
                "scenario.transition.asset_return_source.minimum_observations_per_state",
            )

        asset_ids = [str(item["id"]) for item in definition["assets"]]
        raw_mapping = source.get("asset_target_map")
        if not isinstance(raw_mapping, dict):
            raise ValidationError(
                "INVALID_HISTORICAL_ASSET_TARGET_MAP",
                "asset_target_map 必须显式映射资产到评价目标及收益转换。",
                "scenario.transition.asset_return_source.asset_target_map",
            )
        unknown_assets = sorted(set(raw_mapping) - set(asset_ids))
        if unknown_assets:
            raise ValidationError(
                "HISTORICAL_ASSET_TARGET_DIMENSION_MISMATCH",
                "asset_target_map 包含资产字典之外的字段。",
                "scenario.transition.asset_return_source.asset_target_map",
                diagnostics=[{"unknown_assets": unknown_assets}],
            )
        target_ids: set[str] = set()
        normalized_mapping: dict[str, dict[str, str]] = {}
        return_modes = np.zeros(len(asset_ids), dtype=np.int64)
        for asset_index, asset_id in enumerate(asset_ids):
            item = raw_mapping.get(asset_id)
            if item is None:
                continue
            if not isinstance(item, dict) or set(item) - {"target_id", "return_transform"}:
                raise ValidationError(
                    "INVALID_HISTORICAL_ASSET_TARGET",
                    "每个资产映射只能包含 target_id 与 return_transform。",
                    f"scenario.transition.asset_return_source.asset_target_map.{asset_id}",
                )
            target_id = str(item.get("target_id") or "").strip()
            transform = str(item.get("return_transform") or "")
            if not target_id or transform not in {"simple_return", "forward_value"}:
                raise ValidationError(
                    "INVALID_HISTORICAL_ASSET_TARGET",
                    "资产映射必须指定 target_id，return_transform 必须是 simple_return 或 forward_value。",
                    f"scenario.transition.asset_return_source.asset_target_map.{asset_id}",
                )
            normalized_mapping[asset_id] = {
                "target_id": target_id,
                "return_transform": transform,
            }
            target_ids.add(target_id)
            return_modes[asset_index] = 0 if transform == "simple_return" else 1
        if not target_ids:
            raise ValidationError(
                "HISTORICAL_EVALUATION_TARGET_REQUIRED",
                "历史评价目标分布至少需要选择一个评价目标；全量手工收益请使用内联模式。",
                "scenario.transition.asset_return_source.asset_target_map",
            )

        state_count = len(state_ids)
        overrides = np.full((state_count, len(asset_ids)), np.nan, dtype=np.float64)
        override_fields: list[str] = []
        for state_index, state_id in enumerate(state_ids):
            state_returns = by_id[state_id].get("asset_returns")
            if state_returns is None:
                state_returns = {}
            if not isinstance(state_returns, dict):
                raise ValidationError(
                    "INVALID_STATE_RETURNS",
                    "asset_returns 必须是资产到简单收益率的对象。",
                    f"scenario.transition.states.{state_index}.asset_returns",
                )
            unknown = sorted(set(state_returns) - set(asset_ids))
            if unknown:
                raise ValidationError(
                    "STATE_RETURN_DIMENSION_MISMATCH",
                    "状态收益包含资产字典之外的字段。",
                    f"scenario.transition.states.{state_index}.asset_returns",
                    diagnostics=[{"unknown_assets": unknown}],
                )
            for asset_index, asset_id in enumerate(asset_ids):
                value = state_returns.get(asset_id)
                if value is None:
                    continue
                if inline_policy == "forbid":
                    raise ValidationError(
                        "HISTORICAL_DISTRIBUTION_INLINE_MIX_FORBIDDEN",
                        "已选择历史评价目标分布；如需覆盖，必须显式设置 inline_policy=override。",
                        f"scenario.transition.states.{state_index}.asset_returns.{asset_id}",
                    )
                overrides[state_index, asset_index] = _simple_return(
                    value,
                    f"scenario.transition.states.{state_index}.asset_returns.{asset_id}",
                )
                override_fields.append(f"{state_id}.{asset_id}")

        missing_sources = []
        for asset_index, asset_id in enumerate(asset_ids):
            if asset_id in normalized_mapping:
                continue
            if inline_policy != "override" or not np.all(np.isfinite(overrides[:, asset_index])):
                missing_sources.append(asset_id)
        if missing_sources:
            raise ValidationError(
                "HISTORICAL_ASSET_RETURN_SOURCE_INCOMPLETE",
                "每项资产必须映射评价目标，或在 override 模式下为全部状态显式提供收益。",
                "scenario.transition.asset_return_source.asset_target_map",
                diagnostics=[{"missing_assets": missing_sources}],
            )

        loaded = self._load_evaluation_artifact(run, target_ids, expected_artifact_checksum)
        series = run.get("series")
        if not isinstance(series, list) or len(series) < 3:
            raise ValidationError(
                "HISTORICAL_STATE_SERIES_REQUIRED",
                "历史情景运行缺少可用于状态条件分布的状态序列。",
                "series_artifact",
            )
        try:
            series_dates = np.ascontiguousarray(
                np.asarray([item.get("observation_date") for item in series], dtype="datetime64[ns]").view(np.int64),
                dtype=np.int64,
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "HISTORICAL_STATE_SERIES_DATE_INVALID",
                "历史情景状态序列日期无效。",
                "series_artifact",
            ) from exc
        state_index = {state_id: index for index, state_id in enumerate(state_ids)}
        state_codes = np.ascontiguousarray(
            [state_index.get(str(item.get("state_id")), -1) for item in series],
            dtype=np.int64,
        )
        levels = np.full((len(series), len(asset_ids)), np.nan, dtype=np.float64)
        for asset_index, asset_id in enumerate(asset_ids):
            mapping = normalized_mapping.get(asset_id)
            if mapping is None:
                continue
            values, dates = loaded[mapping["target_id"]]
            if values.shape[0] != len(series) or not np.array_equal(dates, series_dates):
                raise ValidationError(
                    "HISTORICAL_EVALUATION_AXIS_MISMATCH",
                    "评价目标制品与最终状态序列不在同一时序轴上。",
                    f"scenario.transition.asset_return_source.asset_target_map.{asset_id}",
                )
            levels[:, asset_index] = values

        grouped, offsets, counts, means, volatilities, status, bad_row, bad_asset = prepare_historical_state_samples_kernel(
            np.ascontiguousarray(levels),
            state_codes,
            np.ascontiguousarray(return_modes),
            np.ascontiguousarray(overrides),
            np.int64(state_count),
        )
        if int(status) == 1:
            raise ValidationError(
                "HISTORICAL_STATE_DISTRIBUTION_INVALID",
                "历史评价目标无法形成固定维度的状态条件收益样本。",
                "scenario.transition.asset_return_source",
                diagnostics=[{"row": int(bad_row), "asset_index": int(bad_asset)}],
            )
        if int(status) == 2:
            raise ValidationError(
                "HISTORICAL_STATE_RETURN_BELOW_NEGATIVE_ONE",
                "历史评价目标产生了小于或等于 -100% 的简单收益率。",
                "scenario.transition.asset_return_source",
                diagnostics=[{"row": int(bad_row), "asset_index": int(bad_asset)}],
            )
        insufficient = {
            state_ids[index]: int(count)
            for index, count in enumerate(counts)
            if int(count) < minimum
        }
        if insufficient:
            raise ValidationError(
                "HISTORICAL_STATE_DISTRIBUTION_INSUFFICIENT",
                "至少一个状态没有足够的完整跨资产收益样本。",
                "scenario.transition.asset_return_source.minimum_observations_per_state",
                diagnostics=[{"minimum": minimum, "observations_by_state": {state_ids[index]: int(value) for index, value in enumerate(counts)}, "insufficient": insufficient}],
            )
        valid_count = int(offsets[-1])
        resolved_states = []
        for state_index_value, state_id in enumerate(state_ids):
            resolved_states.append(
                {
                    **copy.deepcopy(by_id[state_id]),
                    "id": state_id,
                    "asset_returns": {
                        asset_id: float(means[state_index_value, asset_index])
                        for asset_index, asset_id in enumerate(asset_ids)
                    },
                }
            )
        distribution = {
            "kind": "historical_evaluation_targets",
            "sampling": sampling,
            "asset_ids": asset_ids,
            "grouped_samples": np.ascontiguousarray(grouped[:valid_count], dtype=np.float64),
            "state_offsets": np.ascontiguousarray(offsets, dtype=np.int64),
            "means": np.ascontiguousarray(means, dtype=np.float64),
            "volatilities": np.ascontiguousarray(volatilities, dtype=np.float64),
        }
        audit = {
            "kind": "historical_evaluation_targets",
            "sampling": sampling,
            "minimum_observations_per_state": minimum,
            "observations_by_state": {state_ids[index]: int(value) for index, value in enumerate(counts)},
            "asset_target_map": normalized_mapping,
            "inline_policy": inline_policy,
            "inline_overrides": sorted(override_fields),
            "evaluation_artifact_checksum": expected_artifact_checksum,
            "return_alignment": "state_at_period_start_to_forward_1_period_return",
            "joint_sampling": "complete_cross_asset_observation",
            "missing_value_policy": "exclude_incomplete_joint_observation_never_fill_zero",
            "means": {
                state_id: {asset_id: float(means[state_index_value, asset_index]) for asset_index, asset_id in enumerate(asset_ids)}
                for state_index_value, state_id in enumerate(state_ids)
            },
            "volatilities": {
                state_id: {asset_id: float(volatilities[state_index_value, asset_index]) for asset_index, asset_id in enumerate(asset_ids)}
                for state_index_value, state_id in enumerate(state_ids)
            },
        }
        return {"runtime": distribution, "audit": audit}, resolved_states

    def _resolve_historical_transition(
        self,
        definition: dict[str, Any],
    ) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
        if definition["method"] != "regime_conditioned":
            return definition, None
        reference = definition["scenario"].get("historical_run_ref")
        if reference is None:
            return definition, None
        if not isinstance(reference, dict) or not reference.get("run_id") or not reference.get("publication_id"):
            raise ValidationError(
                "INVALID_HISTORICAL_RUN_REFERENCE",
                "historical_run_ref 必须同时包含 run_id 与 publication_id。",
                "scenario.historical_run_ref",
            )
        # Import lazily so this module remains independently testable and does
        # not create a circular route/service dependency.
        from historical_regimes.service import HistoricalRegimeService

        historical_service = HistoricalRegimeService(self.workspace_data_dir, self.market_data_dir)
        run = historical_service.get_run(str(reference["run_id"]))
        publication = next(
            (item for item in run.get("publications", []) if item.get("id") == str(reference["publication_id"])),
            None,
        )
        if publication is None:
            raise ValidationError(
                "HISTORICAL_RUN_NOT_PUBLISHED",
                "引用的历史情景运行没有匹配的发布记录。",
                "scenario.historical_run_ref.publication_id",
            )
        if publication.get("run_content_hash") != run.get("content_hash"):
            raise ValidationError(
                "HISTORICAL_RUN_LINEAGE_MISMATCH",
                "历史情景发布记录与运行快照不一致，已阻断引用。",
                "scenario.historical_run_ref",
            )
        if _stored_run_snapshot_hash(run) != run.get("content_hash"):
            raise ValidationError(
                "HISTORICAL_RUN_SNAPSHOT_TAMPERED",
                "历史情景运行快照完整性校验失败，已阻断引用。",
                "scenario.historical_run_ref",
            )
        # V2 runs keep the large point-in-time state sequence in a
        # content-addressed Parquet artifact.  Integrity must be checked on the
        # immutable JSON snapshot first; downstream state lookup may hydrate it
        # only after that check has passed.
        if str(run.get("schema_version") or "") == "2.0":
            from historical_regimes.v2_service import hydrate_v2_run_snapshot

            run = hydrate_v2_run_snapshot(
                run,
                workspace_data_dir=self.workspace_data_dir,
            )
        historical_transition = run.get("transition") or {}
        state_ids = historical_transition.get("states")
        probabilities = historical_transition.get("probabilities")
        if not isinstance(state_ids, list) or not isinstance(probabilities, list) or len(state_ids) < 2:
            raise ValidationError(
                "HISTORICAL_TRANSITION_UNAVAILABLE",
                "历史情景运行没有可用的状态转移矩阵。",
                "scenario.historical_run_ref",
            )
        inline = definition["scenario"].get("transition")
        if not isinstance(inline, dict):
            raise ValidationError(
                "STATE_RETURN_PARAMETERS_REQUIRED",
                "引用历史状态时仍需配置各状态的资产收益参数。",
                "scenario.transition",
            )
        asset_return_source = inline.get("asset_return_source")
        inline_states = inline.get("states")
        if not isinstance(inline_states, list):
            if asset_return_source is None:
                raise ValidationError(
                    "STATE_RETURN_PARAMETERS_REQUIRED",
                    "scenario.transition.states 必须提供各历史状态的 asset_returns。",
                    "scenario.transition.states",
                )
            labels = {
                str(item.get("id")): str(item.get("label") or item.get("id"))
                for item in run.get("states", [])
                if isinstance(item, dict) and item.get("id")
            }
            inline_states = [
                {"id": str(state_id), "label": labels.get(str(state_id), str(state_id))}
                for state_id in state_ids
            ]
        by_id: dict[str, dict[str, Any]] = {}
        for position, item in enumerate(inline_states):
            if not isinstance(item, dict) or not str(item.get("id") or ""):
                raise ValidationError(
                    "INVALID_TRANSITION_STATE",
                    "每个状态必须包含 id。",
                    f"scenario.transition.states.{position}",
                )
            state_id = str(item["id"])
            if state_id in by_id:
                raise ValidationError(
                    "DUPLICATE_TRANSITION_STATE",
                    "状态 id 不能重复。",
                    f"scenario.transition.states.{position}.id",
                )
            by_id[state_id] = item
        if set(by_id) != set(str(item) for item in state_ids):
            raise ValidationError(
                "HISTORICAL_STATE_DIMENSION_MISMATCH",
                "资产收益参数的状态必须与历史运行的状态字典完全一致。",
                "scenario.transition.states",
                diagnostics=[{"expected_states": state_ids, "provided_states": sorted(by_id)}],
            )
        try:
            matrix = np.ascontiguousarray(probabilities, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "HISTORICAL_TRANSITION_INVALID",
                "历史运行中的转移矩阵无效。",
                "scenario.historical_run_ref",
            ) from exc
        if matrix.shape != (len(state_ids), len(state_ids)):
            raise ValidationError(
                "HISTORICAL_TRANSITION_INVALID",
                "历史运行中的转移矩阵无效。",
                "scenario.historical_run_ref",
            )
        matrix, imputed_mask, transition_status = normalize_transition_counts_kernel(matrix)
        if int(transition_status) != 0:
            raise ValidationError(
                "HISTORICAL_TRANSITION_INVALID",
                "历史运行中的转移矩阵无效。",
                "scenario.historical_run_ref",
            )
        imputed_self_loops = [
            str(state_ids[index])
            for index, flag in enumerate(imputed_mask)
            if int(flag) == 1
        ]
        historical_causality = run.get("causality") if isinstance(run.get("causality"), dict) else {}
        causality_gate_passed = bool(
            run.get("mode") == "realtime"
            and historical_causality.get("is_causal")
            and not historical_causality.get("repaints")
            and not historical_causality.get("uses_future_data")
        )
        publication_usage = str(publication.get("usage") or "")
        allowed_downstream_usages = {"research_display"}
        if publication_usage == "product_research":
            allowed_downstream_usages.add("product_research")
        elif publication_usage == "formal_backtest" and causality_gate_passed:
            allowed_downstream_usages.update({"portfolio_backtest", "risk_monitoring"})
        elif publication_usage == "taa" and causality_gate_passed:
            allowed_downstream_usages.update({"taa", "risk_monitoring"})
        historical_distribution = None
        resolved_states = [copy.deepcopy(by_id[str(state_id)]) for state_id in state_ids]
        distribution_audit = None
        if asset_return_source is not None:
            historical_distribution, resolved_states = self._historical_asset_distribution(
                definition,
                run,
                inline,
                [str(item) for item in state_ids],
                by_id,
            )
            distribution_audit = historical_distribution["audit"]
        initial_state = inline.get("initial_state")
        if not initial_state:
            initial_state = next(
                (
                    item.get("state_id")
                    for item in reversed(run.get("series", []))
                    if item.get("state_id") in set(state_ids)
                ),
                None,
            )
        resolved_transition = {
            **copy.deepcopy(inline),
            "states": resolved_states,
            "matrix": matrix.tolist(),
            "initial_state": initial_state,
            **(
                {"_historical_asset_distribution": historical_distribution["runtime"]}
                if historical_distribution is not None
                else {}
            ),
            "source": {
                "kind": "published_historical_regime",
                "run_id": run["id"],
                "publication_id": publication["id"],
                "run_content_hash": run["content_hash"],
                "definition_id": run.get("definition_id"),
                "definition_revision": run.get("definition_revision"),
                "publication_usage": publication.get("usage"),
                "imputed_self_loop_states": imputed_self_loops,
                "historical_mode": run.get("mode"),
                "historical_causality": {
                    "is_causal": historical_causality.get("is_causal"),
                    "uses_future_data": historical_causality.get("uses_future_data"),
                    "repaints": historical_causality.get("repaints"),
                },
                "causality_gate_passed": causality_gate_passed,
                "allowed_downstream_usages": sorted(allowed_downstream_usages),
                "asset_return_distribution": distribution_audit,
            },
        }
        execution_definition = copy.deepcopy(definition)
        execution_definition["scenario"]["_resolved_transition"] = resolved_transition
        return execution_definition, copy.deepcopy(resolved_transition["source"])

    def _execute(
        self,
        definition: dict[str, Any],
        definition_source: str,
        *,
        batch: bool,
    ) -> dict[str, Any]:
        execution_definition, historical_reference = self._resolve_historical_transition(definition)
        analytical = execute(execution_definition)
        all_complete = all(item["coverage"]["status"] == "complete" for item in analytical["results"])
        eligible_usages = list(APPLICATION_TARGETS) if all_complete else ["research_display", "product_research"]
        publication_blockers: list[str] = []
        if definition_source != "repository_reference":
            eligible_usages = []
            publication_blockers.append("试算结果没有已保存的定义版本，不能发布或绑定。")
        if not all_complete:
            publication_blockers.append("存在覆盖率降级；正式回测、TAA 与风险监控发布已阻断。")
        if historical_reference:
            allowed_by_historical_publication = set(historical_reference["allowed_downstream_usages"])
            removed = [usage for usage in eligible_usages if usage not in allowed_by_historical_publication]
            eligible_usages = [usage for usage in eligible_usages if usage in allowed_by_historical_publication]
            if removed:
                publication_blockers.append(
                    "历史状态引用仅继承其发布用途与因果等级；未授权的正式回测、TAA 或风险监控用途已阻断。"
                )
        definition_id = definition.get("id") if definition_source == "repository_reference" else None
        definition_revision = int(definition.get("revision")) if definition_source == "repository_reference" else None
        definition_hash = _content_hash(_business_definition(definition))
        data_snapshot = copy.deepcopy(analytical["data_snapshot"])
        data_snapshot["fingerprint"] = _content_hash(
            {
                "scenario": definition["scenario"],
                "method": definition["method"],
                "historical_reference": historical_reference,
            }
        )
        if historical_reference:
            data_snapshot["historical_regime_reference"] = historical_reference
        diagnostics = list(analytical["diagnostics"])
        if definition_source != "repository_reference":
            diagnostics.append(
                {
                    "code": "UNVERSIONED_TRIAL",
                    "level": "warning",
                    "message": "本次为未保存试算；请保存定义并按版本重跑后再发布。",
                }
            )
        payload = {
            "definition_id": definition_id,
            "definition_revision": definition_revision,
            "definition_source": definition_source,
            "definition_snapshot_hash": definition_hash,
            "definition": copy.deepcopy(definition),
            "name": definition["name"],
            "method": definition["method"],
            "probabilistic": analytical["probabilistic"],
            "batch": batch,
            "batch_size": len(analytical["results"]),
            "schema_version": "1.0",
            "data_snapshot": data_snapshot,
            "results": analytical["results"],
            "mapping_diagnostics": analytical["mapping_diagnostics"],
            "compute_audit": analytical["compute_audit"],
            "diagnostics": diagnostics,
            "governance": {
                "publish_eligible_usages": eligible_usages,
                "publication_blockers": publication_blockers,
                "coverage_gate": "passed" if all_complete else "research_only",
                "probability_semantics": "simulation_frequency" if analytical["probabilistic"] else "not_applicable",
            },
            "application_bindings": [],
        }
        payload["content_hash"] = _content_hash(payload)
        return self.runs.create(_json_safe(payload))

    def run(self, requested_definition: dict[str, Any]) -> dict[str, Any]:
        definition, definition_source = self._resolve_definition(requested_definition)
        return self._execute(definition, definition_source, batch=False)

    def batch_run(self, requested_definition: dict[str, Any]) -> dict[str, Any]:
        definition, definition_source = self._resolve_definition(requested_definition)
        if len(definition["portfolios"]) < 2:
            raise ValidationError(
                "BATCH_REQUIRES_MULTIPLE_PORTFOLIOS",
                "批量运行要求定义中至少包含 2 个组合。",
                "definition.portfolios",
            )
        return self._execute(definition, definition_source, batch=True)

    def list_runs(self, definition_id: Optional[str] = None) -> list[dict[str, Any]]:
        return self.runs.list(definition_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.runs.get(run_id)

    def publish(self, run_id: str, usage: str | list[str], note: str = "") -> dict[str, Any]:
        run = self.runs.get(run_id)
        usages = [usage] if isinstance(usage, str) else list(usage or [])
        usages = list(dict.fromkeys(str(item) for item in usages))
        if not usages or any(item not in APPLICATION_TARGETS for item in usages):
            raise ValidationError("INVALID_PUBLICATION_USAGE", "usage 包含不支持的应用目标。", "usage")
        if run.get("definition_source") != "repository_reference" or int(run.get("definition_revision") or 0) <= 0:
            raise ValidationError(
                "UNVERSIONED_RUN_PUBLICATION_BLOCKED",
                "试算结果没有已保存的定义版本；请先保存定义并按该版本重新运行。",
                "run_id",
            )
        if _stored_run_snapshot_hash(run) != run.get("content_hash"):
            raise ValidationError(
                "SCENARIO_RUN_SNAPSHOT_TAMPERED",
                "情景运行快照完整性校验失败，不能发布。",
                "run_id",
            )
        try:
            persisted = self.definitions.get(str(run["definition_id"]), int(run["definition_revision"]))
        except IndicatorDomainError as exc:
            raise ValidationError("RUN_DEFINITION_VERSION_MISSING", "运行引用的定义版本已不存在，不能发布。", "run_id") from exc
        if _content_hash(_business_definition(persisted)) != run.get("definition_snapshot_hash"):
            raise ValidationError("RUN_DEFINITION_LINEAGE_MISMATCH", "运行快照与已保存定义版本不一致，不能发布。", "run_id")
        eligible = set(run.get("governance", {}).get("publish_eligible_usages", []))
        blocked = [item for item in usages if item not in eligible]
        if blocked:
            raise ValidationError(
                "SCENARIO_PUBLICATION_BLOCKED",
                f"该运行不能发布到 {', '.join(blocked)}；请先满足版本与覆盖率门禁。",
                "usage",
                diagnostics=[{"requested": usages, "eligible": sorted(eligible), "governance": run.get("governance")}],
            )
        if len(note) > 500:
            raise ValidationError("PUBLICATION_NOTE_TOO_LONG", "发布说明不能超过 500 个字符。", "note")
        publications = [
            {
                "id": f"scenario-publication-{uuid.uuid4().hex}",
                "usage": item,
                "published_at": utc_now(),
                "note": note,
                "run_id": run_id,
                "definition_revision": int(run["definition_revision"]),
                "run_content_hash": run["content_hash"],
                "portfolio_ids": [result["portfolio_id"] for result in run.get("results", [])],
                "gate": "version_and_coverage_passed",
            }
            for item in usages
        ]
        updated = self.runs.add_publications(run_id, publications)
        return {
            "run_id": run_id,
            "publication": publications[0],
            "publications": updated.get("publications", []),
            "application_bindings": updated.get("application_bindings", []),
        }

    def compare(self, run_ids: list[str], reference_run_id: Optional[str] = None) -> dict[str, Any]:
        unique_ids = list(dict.fromkeys(str(item) for item in run_ids))
        if len(unique_ids) < 2 or len(unique_ids) > 8:
            raise ValidationError("INVALID_COMPARE_RUNS", "运行比较需要 2 至 8 个不同运行。", "run_ids")
        if reference_run_id and reference_run_id not in unique_ids:
            raise ValidationError("INVALID_REFERENCE_RUN", "reference_run_id 必须包含在 run_ids 中。", "reference_run_id")
        runs = [self.runs.get(run_id) for run_id in unique_ids]
        reference = next((run for run in runs if run["id"] == reference_run_id), runs[0])

        metric_names = (
            "terminal_return",
            "max_drawdown",
            "var_95",
            "es_95",
            "loss_probability",
            "coverage_ratio",
            "breach_count",
        )

        def metrics_by_portfolio(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
            return {
                result["portfolio_id"]: {
                    "terminal_return": result["summary"].get("terminal_return"),
                    "max_drawdown": result["summary"].get("max_drawdown"),
                    "var_95": result["summary"].get("var_95"),
                    "es_95": result["summary"].get("es_95"),
                    "loss_probability": result["summary"].get("loss_probability"),
                    "coverage_ratio": result["coverage"].get("ratio"),
                    "breach_count": result["summary"].get("breach_count"),
                }
                for result in run.get("results", [])
            }

        reference_metrics = metrics_by_portfolio(reference)
        summaries: list[dict[str, Any]] = []
        for run in runs:
            current = metrics_by_portfolio(run)
            deltas: dict[str, dict[str, Optional[float]]] = {}
            comparable_portfolios = sorted(set(current) & set(reference_metrics))
            current_values = np.zeros((len(comparable_portfolios), len(metric_names)), dtype=np.float64)
            reference_values = np.zeros_like(current_values)
            current_available = np.zeros(current_values.shape, dtype=np.uint8)
            reference_available = np.zeros(current_values.shape, dtype=np.uint8)
            for portfolio_index, portfolio_id in enumerate(comparable_portfolios):
                for metric_index, metric in enumerate(metric_names):
                    value = current[portfolio_id].get(metric)
                    reference_value = reference_metrics[portfolio_id].get(metric)
                    if value is not None:
                        current_values[portfolio_index, metric_index] = float(value)
                        current_available[portfolio_index, metric_index] = 1
                    if reference_value is not None:
                        reference_values[portfolio_index, metric_index] = float(reference_value)
                        reference_available[portfolio_index, metric_index] = 1
            delta_values, delta_available = metric_deltas_kernel(
                np.ascontiguousarray(current_values),
                np.ascontiguousarray(current_available),
                np.ascontiguousarray(reference_values),
                np.ascontiguousarray(reference_available),
            )
            for portfolio_index, portfolio_id in enumerate(comparable_portfolios):
                deltas[portfolio_id] = {
                    metric: (
                        float(delta_values[portfolio_index, metric_index])
                        if int(delta_available[portfolio_index, metric_index]) == 1
                        else None
                    )
                    for metric_index, metric in enumerate(metric_names)
                }
            summaries.append(
                {
                    "run_id": run["id"],
                    "name": run.get("name"),
                    "method": run.get("method"),
                    "probabilistic": run.get("probabilistic"),
                    "metrics": current,
                    "deltas_to_reference": deltas,
                }
            )
        return {
            "run_ids": unique_ids,
            "reference_run_id": reference["id"],
            "runs": summaries,
            "compared_at": utc_now(),
            "execution": validate_execution_audit(scenario_stress_numba_status()),
        }
