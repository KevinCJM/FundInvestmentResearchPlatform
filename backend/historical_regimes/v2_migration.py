"""Explicit, non-destructive migration from classic definitions to Regime Graph v2."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from custom_indicators.errors import ValidationError


_STATE_COLORS = ("#16a34a", "#64748b", "#dc2626", "#7c3aed", "#0284c7")


def _content_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def migrate_v1_definition(v1: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a new v2 draft; the caller decides whether to persist it.

    Families whose v1 semantics cannot be represented without inventing source
    bindings fail closed instead of silently changing the research definition.
    """

    family = str((v1.get("algorithm") or {}).get("family") or "")
    if family in {"merrill_clock", "ensemble"}:
        raise ValidationError(
            "V1_MIGRATION_REQUIRES_MANUAL_GRAPH",
            "该经典定义包含专用多序列或嵌套算法，请从对应 v2 模板复制后手工绑定数据。",
            "algorithm.family",
        )
    formula = (v1.get("features") or {}).get("formula")
    if formula:
        raise ValidationError(
            "V1_FORMULA_MIGRATION_REQUIRES_REBINDING",
            "经典公式引用原始表字段，迁移前必须在 v2 公式节点中显式绑定输入端口。",
            "features.formula",
        )
    target = copy.deepcopy(v1.get("target") or v1.get("data") or {})
    kind = str(target.pop("kind", ""))
    if kind not in {"inline", "index", "relative", "indicator"}:
        raise ValidationError(
            "V1_SOURCE_MIGRATION_UNSUPPORTED",
            "经典定义的数据源无法自动迁移。",
            "target.kind",
        )
    nodes: list[dict[str, Any]] = [
        {
            "id": "source",
            "type": f"source.{kind}",
            "type_version": 1,
            "parameters": target,
        }
    ]
    source_ref = {"node_id": "source", "port": "value"}
    params = copy.deepcopy((v1.get("algorithm") or {}).get("parameters") or {})
    features = copy.deepcopy(v1.get("features") or {})
    warnings: list[dict[str, Any]] = []

    if family in {"causal_filter", "relative_strength", "hmm", "markov", "gmm"}:
        transform_type = "transform.diff" if family == "relative_strength" else "transform.return"
        nodes.append(
            {
                "id": "feature",
                "type": transform_type,
                "type_version": 1,
                "parameters": {"window": 1},
                "inputs": {"value": source_ref},
            }
        )
        feature_ref = {"node_id": "feature", "port": "value"}
        filter_name = str(features.get("filter") or "ema").lower()
        if filter_name in {"zero_phase", "zero_phase_butterworth", "filtfilt"}:
            raise ValidationError(
                "V1_NONCAUSAL_FILTER_MIGRATION_REQUIRES_REVIEW",
                "经典零相位滤波需要未来数据，请在 v2 中手工选择非因果研究节点。",
                "features.filter",
            )
        filter_type = "filter.kalman" if filter_name == "kalman" else "filter.sma" if filter_name == "sma" else "filter.ema"
        filter_parameters = (
            {
                "process_variance": float(features.get("process_variance", 1e-5)),
                "measurement_variance": float(features.get("measurement_variance", 1e-2)),
            }
            if filter_type == "filter.kalman"
            else {"window": int(features.get("window", 20))}
        )
        nodes.append(
            {
                "id": "filtered",
                "type": filter_type,
                "type_version": 1,
                "parameters": filter_parameters,
                "inputs": {"value": feature_ref},
            }
        )
        feature_ref = {"node_id": "filtered", "port": "value"}
    else:
        feature_ref = source_ref

    if family in {"causal_filter", "relative_strength"}:
        nodes.append(
            {
                "id": "classifier",
                "type": "model.hysteresis",
                "type_version": 1,
                "parameters": {
                    "upper_enter": float(params.get("upper", params.get("bull_enter", 0.001))),
                    "upper_exit": float(params.get("positive_exit", 0.0)),
                    "lower_enter": float(params.get("lower", params.get("bear_enter", -0.001))),
                    "lower_exit": float(params.get("negative_exit", 0.0)),
                },
                "inputs": {"value": feature_ref},
            }
        )
        state_ref = {"node_id": "classifier", "port": "state"}
        confirmation = int(params.get("confirmation", 1))
        minimum_duration = int(params.get("min_duration", 1))
        if confirmation > 1 or minimum_duration > 1:
            nodes.append(
                {
                    "id": "confirmed",
                    "type": "post.confirmation",
                    "type_version": 1,
                    "parameters": {
                        "confirmation": confirmation,
                        "min_duration": minimum_duration,
                    },
                    "inputs": {"state": state_ref},
                }
            )
            state_ref = {"node_id": "confirmed", "port": "state"}
    elif family == "turning_point":
        nodes.append(
            {
                "id": "classifier",
                "type": "model.turning_point",
                "type_version": 1,
                "parameters": {
                    "window": int(params.get("window", 20)),
                    "min_move": float(params.get("min_move", 0.08)),
                },
                "inputs": {"value": feature_ref},
            }
        )
        state_ref = {"node_id": "classifier", "port": "state"}
    elif family == "change_point":
        nodes.append(
            {
                "id": "classifier",
                "type": "model.change_point",
                "type_version": 1,
                "parameters": {
                    "window": int(params.get("window", 20)),
                    "threshold": float(params.get("threshold", 1.5)),
                    "confirmation": int(params.get("confirmation", 2)),
                },
                "inputs": {"value": feature_ref},
            }
        )
        state_ref = {"node_id": "classifier", "port": "state"}
    elif family in {"hmm", "markov", "gmm"}:
        nodes.append(
            {
                "id": "features",
                "type": "feature.matrix",
                "type_version": 1,
                "parameters": {},
                "inputs": {"feature_1": feature_ref},
            }
        )
        nodes.append(
            {
                "id": "classifier",
                "type": f"model.{family}",
                "type_version": 1,
                "parameters": {
                    "components": int(params.get("states", 3)),
                    "initial_train_size": int(params.get("initial_train_size", 60)),
                    "iterations": int(params.get("iterations", 60)),
                },
                "inputs": {"features": {"node_id": "features", "port": "features"}},
            }
        )
        state_ref = {"node_id": "classifier", "port": "state"}
    else:
        raise ValidationError(
            "V1_ALGORITHM_MIGRATION_UNSUPPORTED",
            "经典定义的算法暂不支持自动迁移。",
            "algorithm.family",
        )

    nodes.append(
        {
            "id": "temporal",
            "type": "output.temporal",
            "type_version": 1,
            "parameters": {},
            "inputs": {"state": state_ref},
        }
    )
    states = []
    for index, item in enumerate(v1.get("states") or []):
        state = copy.deepcopy(item)
        state.setdefault("role", "neutral")
        state.setdefault("color", _STATE_COLORS[index % len(_STATE_COLORS)])
        state.setdefault("order", index + 1)
        states.append(state)
    source_lineage = {
        "id": v1.get("id"),
        "revision": v1.get("revision"),
        "content_hash": _content_hash(v1),
    }
    definition = {
        "schema_version": "2.0",
        "name": f"{v1.get('name', '历史情景')} · v2 副本",
        "description": str(v1.get("description") or ""),
        "template_id": "migrated-v1",
        "source_v1": source_lineage,
        "graph": {
            "nodes": nodes,
            "outputs": {
                "state": state_ref,
                "recognition_index": {"node_id": "temporal", "port": "recognition_index"},
                "effective_index": {"node_id": "temporal", "port": "effective_index"},
                "reason_code": {"node_id": "temporal", "port": "reason_code"},
            },
            "exposed_node_ids": [node["id"] for node in nodes if node["id"] not in {"source", "temporal"}],
        },
        "states": states,
        "evaluation_targets": [],
        "validation": copy.deepcopy(v1.get("validation") or {}),
        "usage_intent": str(v1.get("usage_intent") or "research_display"),
    }
    warnings.append(
        {
            "code": "V1_MIGRATION_REQUIRES_RESULT_REVIEW",
            "message": "已保留主要数据源、算法参数和状态字典；请先试算并与经典版本比较后再发布。",
        }
    )
    return definition, warnings


__all__ = ["migrate_v1_definition"]
