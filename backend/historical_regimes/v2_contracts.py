"""Typed contracts and semantic validation for Regime Graph v2."""

from __future__ import annotations

import hashlib
import json
import copy
import math
from collections import deque
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError as PydanticValidationError, model_validator

from compute_policy import ComputePolicyError, validate_execution_audit
from custom_indicators.errors import ValidationError

from .v2_registry import (
    CONFIDENCE,
    INDEX_SERIES,
    NODE_REGISTRY,
    PROBABILITIES,
    REASON_CODES,
    REGISTRY_VERSION,
    STATE_CODES,
)


class GraphPortRefV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    port: str = Field(default="value", min_length=1, max_length=64)


class GraphEdgeV2(BaseModel):
    """A canonical graph edge; target.port names the target node input slot."""

    model_config = ConfigDict(extra="forbid")

    source: GraphPortRefV2
    target: GraphPortRefV2


class RegimeGraphNodeV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    type: str = Field(min_length=1, max_length=100)
    type_id: str | None = Field(default=None, min_length=1, max_length=100)
    type_version: int = Field(default=1, ge=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, GraphPortRefV2] = Field(default_factory=dict)
    label: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def freeze_type_identity(self) -> "RegimeGraphNodeV2":
        if self.type_id is not None and self.type_id != self.type:
            raise ValueError("graph node 的 type 与 type_id 必须一致")
        self.type_id = self.type
        return self


class RegimeGraphV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[RegimeGraphNodeV2] = Field(min_length=1, max_length=128)
    edges: list[GraphEdgeV2] = Field(default_factory=list, max_length=256)
    outputs: dict[str, GraphPortRefV2]
    exposed_node_ids: list[str] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def validate_unique_node_ids(self) -> "RegimeGraphV2":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph.nodes 中的节点 id 必须唯一")
        if "state" not in self.outputs:
            raise ValueError("graph.outputs 必须声明 state 根输出")
        allowed_outputs = {
            "state",
            "probabilities",
            "confidence",
            "recognition_index",
            "effective_index",
            "reason_code",
        }
        unknown_outputs = sorted(set(self.outputs) - allowed_outputs)
        if unknown_outputs:
            raise ValueError(
                "graph.outputs 包含不支持的输出：" + ", ".join(unknown_outputs)
            )
        if len(self.exposed_node_ids) != len(set(self.exposed_node_ids)):
            raise ValueError("graph.exposed_node_ids 不能重复")

        node_map = {node.id: node for node in self.nodes}
        input_edges = {
            (reference.node_id, reference.port, node.id, input_name)
            for node in self.nodes
            for input_name, reference in node.inputs.items()
        }
        supplied_edges: set[tuple[str, str, str, str]] = set()
        target_slots: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.target.node_id not in node_map:
                raise ValueError(f"graph.edges 的目标节点 {edge.target.node_id} 不存在")
            target_slot = (edge.target.node_id, edge.target.port)
            if target_slot in target_slots:
                raise ValueError(
                    f"graph.edges 的目标输入 {edge.target.node_id}.{edge.target.port} 只能连接一次"
                )
            target_slots.add(target_slot)
            supplied_edges.add(
                (
                    edge.source.node_id,
                    edge.source.port,
                    edge.target.node_id,
                    edge.target.port,
                )
            )

        if self.edges and input_edges and supplied_edges != input_edges:
            raise ValueError("graph.edges 与 graph.nodes[].inputs 必须完全一致")
        if self.edges and not input_edges:
            for source_node_id, source_port, target_node_id, target_port in supplied_edges:
                node_map[target_node_id].inputs[target_port] = GraphPortRefV2(
                    node_id=source_node_id,
                    port=source_port,
                )

        canonical_edges = sorted(
            (
                (reference.node_id, reference.port, node.id, input_name)
                for node in self.nodes
                for input_name, reference in node.inputs.items()
            ),
            key=lambda item: (item[2], item[3], item[0], item[1]),
        )
        self.edges = [
            GraphEdgeV2(
                source=GraphPortRefV2(node_id=source_node_id, port=source_port),
                target=GraphPortRefV2(node_id=target_node_id, port=target_port),
            )
            for source_node_id, source_port, target_node_id, target_port in canonical_edges
        ]
        return self


class RegimeStateV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    label: str = Field(min_length=1, max_length=100)
    role: str = Field(default="neutral", max_length=64)
    color: str = Field(default="#64748b", pattern=r"^#[0-9A-Fa-f]{6}$")
    order: int = Field(ge=0, le=100)


class EvaluationTargetV2(BaseModel):
    """A display/analysis series kept separate from classification inputs."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=100)
    source: dict[str, Any]
    primary: bool = False


class RegimeDefinitionV2(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    created_at: str | None = None
    updated_at: str | None = None
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    template_id: str | None = Field(default=None, max_length=100)
    source_v1: dict[str, Any] | None = None
    graph: RegimeGraphV2
    states: list[RegimeStateV2] = Field(min_length=2, max_length=12)
    evaluation_targets: list[EvaluationTargetV2] = Field(default_factory=list, max_length=20)
    validation: dict[str, Any] = Field(default_factory=dict)
    usage_intent: Literal["research_display", "product_research", "formal_backtest", "taa"] = "research_display"

    @model_validator(mode="after")
    def validate_identifiers(self) -> "RegimeDefinitionV2":
        state_ids = [state.id for state in self.states]
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("states.id 必须唯一")
        target_ids = [target.id for target in self.evaluation_targets]
        if len(target_ids) != len(set(target_ids)):
            raise ValueError("evaluation_targets.id 必须唯一")
        if sum(1 for target in self.evaluation_targets if target.primary) > 1:
            raise ValueError("evaluation_targets 最多只能有一个 primary=true")
        allowed_validation = {
            "walk_forward",
            "folds",
            "stability_perturbation",
            "sensitivity_candidates",
            "min_classified_ratio",
            "min_walk_forward_classified_ratio",
            "min_parameter_agreement",
            "max_prefix_revision_rate",
            "max_label_flip_rate",
        }
        unknown_validation = sorted(set(self.validation) - allowed_validation)
        if unknown_validation:
            raise ValueError(
                "validation 包含不支持的字段：" + ", ".join(unknown_validation)
            )
        if "walk_forward" in self.validation and type(self.validation["walk_forward"]) is not bool:
            raise ValueError("validation.walk_forward 必须是布尔值")
        integer_ranges = {
            "folds": (2, 12, 4),
            "sensitivity_candidates": (1, 8, 4),
        }
        for field_name, (minimum, maximum, default) in integer_ranges.items():
            value = self.validation.get(field_name, default)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"validation.{field_name} 必须是整数")
            if value < minimum or value > maximum:
                raise ValueError(
                    f"validation.{field_name} 必须在 {minimum} 到 {maximum} 之间"
                )
        perturbation = self.validation.get("stability_perturbation", 0.1)
        if isinstance(perturbation, bool) or not isinstance(perturbation, (int, float)):
            raise ValueError("validation.stability_perturbation 必须是数值")
        if not 0.0 <= float(perturbation) <= 0.5:
            raise ValueError("validation.stability_perturbation 必须在 0 到 0.5 之间")
        rate_defaults = {
            "min_classified_ratio": 0.5,
            "min_walk_forward_classified_ratio": 0.25,
            "min_parameter_agreement": 0.5,
            "max_prefix_revision_rate": 0.0,
            "max_label_flip_rate": 0.5,
        }
        for field_name, default in rate_defaults.items():
            value = self.validation.get(field_name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"validation.{field_name} 必须是数值")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"validation.{field_name} 必须在 0 到 1 之间")
        return self


def _diagnostics_from_pydantic(exc: PydanticValidationError) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in exc.errors(include_url=False):
        diagnostics.append(
            {
                "code": "SCHEMA_VALIDATION_ERROR",
                "path": ".".join(str(part) for part in item.get("loc", ())),
                "message": str(item.get("msg") or "字段不合法"),
                "severity": "error",
            }
        )
    return diagnostics


def parse_definition_v2(payload: Mapping[str, Any]) -> RegimeDefinitionV2:
    try:
        return RegimeDefinitionV2.model_validate(payload)
    except PydanticValidationError as exc:
        diagnostics = _diagnostics_from_pydantic(exc)
        raise ValidationError(
            "INVALID_REGIME_GRAPH_V2",
            "历史情景图谱定义未通过结构校验。",
            "definition",
            diagnostics,
        ) from exc


def _parameter_diagnostics(node: RegimeGraphNodeV2, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    schema = metadata.get("parameter_schema") or {}
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    diagnostics: list[dict[str, Any]] = []
    for name in sorted(required - set(node.parameters)):
        diagnostics.append(
            {
                "code": "MISSING_NODE_PARAMETER",
                "path": f"graph.nodes.{node.id}.parameters.{name}",
                "message": f"节点 {node.id} 缺少参数 {name}。",
                "severity": "error",
            }
        )
    if schema.get("additionalProperties") is False:
        for name in sorted(set(node.parameters) - set(properties)):
            diagnostics.append(
                {
                    "code": "UNKNOWN_NODE_PARAMETER",
                    "path": f"graph.nodes.{node.id}.parameters.{name}",
                    "message": f"节点 {node.id} 不支持参数 {name}。",
                    "severity": "error",
                }
            )
    for name, value in node.parameters.items():
        field_schema = properties.get(name)
        if not isinstance(field_schema, Mapping):
            continue
        expected = field_schema.get("type")
        invalid_type = (
            (expected == "integer" and (isinstance(value, bool) or not isinstance(value, int)))
            or (expected == "number" and (isinstance(value, bool) or not isinstance(value, (int, float))))
            or (expected == "string" and not isinstance(value, str))
            or (expected == "array" and not isinstance(value, list))
            or (expected == "object" and not isinstance(value, dict))
        )
        if invalid_type:
            diagnostics.append(
                {
                    "code": "INVALID_NODE_PARAMETER_TYPE",
                    "path": f"graph.nodes.{node.id}.parameters.{name}",
                    "message": f"节点参数 {name} 类型不正确。",
                    "severity": "error",
                }
            )
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)):
                diagnostics.append(
                    {
                        "code": "NONFINITE_NODE_PARAMETER",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 必须是有限数值。",
                        "severity": "error",
                    }
                )
                continue
            if "minimum" in field_schema and value < field_schema["minimum"]:
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_BELOW_MINIMUM",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 不能小于 {field_schema['minimum']}。",
                        "severity": "error",
                    }
                )
            if "maximum" in field_schema and value > field_schema["maximum"]:
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_ABOVE_MAXIMUM",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 不能大于 {field_schema['maximum']}。",
                        "severity": "error",
                    }
                )
        if "enum" in field_schema and value not in field_schema["enum"]:
            diagnostics.append(
                {
                    "code": "INVALID_NODE_PARAMETER_VALUE",
                    "path": f"graph.nodes.{node.id}.parameters.{name}",
                    "message": f"节点参数 {name} 不在允许值中。",
                    "severity": "error",
                }
            )
        if isinstance(value, str):
            if "minLength" in field_schema and len(value) < int(field_schema["minLength"]):
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_TOO_SHORT",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 长度不足。",
                        "severity": "error",
                    }
                )
            if "maxLength" in field_schema and len(value) > int(field_schema["maxLength"]):
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_TOO_LONG",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 超过最大长度。",
                        "severity": "error",
                    }
                )
        if isinstance(value, list):
            if "minItems" in field_schema and len(value) < int(field_schema["minItems"]):
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_ARRAY_TOO_SHORT",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 的元素数量不足。",
                        "severity": "error",
                    }
                )
            if "maxItems" in field_schema and len(value) > int(field_schema["maxItems"]):
                diagnostics.append(
                    {
                        "code": "NODE_PARAMETER_ARRAY_TOO_LONG",
                        "path": f"graph.nodes.{node.id}.parameters.{name}",
                        "message": f"节点参数 {name} 的元素数量过多。",
                        "severity": "error",
                    }
                )
            item_schema = field_schema.get("items")
            if isinstance(item_schema, Mapping):
                item_type = item_schema.get("type")
                for item_index, item_value in enumerate(value):
                    item_invalid = (
                        item_type == "integer"
                        and (isinstance(item_value, bool) or not isinstance(item_value, int))
                    ) or (
                        item_type == "number"
                        and (
                            isinstance(item_value, bool)
                            or not isinstance(item_value, (int, float))
                            or not math.isfinite(float(item_value))
                        )
                    )
                    if item_invalid:
                        diagnostics.append(
                            {
                                "code": "INVALID_NODE_PARAMETER_ITEM",
                                "path": f"graph.nodes.{node.id}.parameters.{name}.{item_index}",
                                "message": f"节点参数 {name} 包含非法元素。",
                                "severity": "error",
                            }
                        )
                        continue
                    if isinstance(item_value, (int, float)) and not isinstance(item_value, bool):
                        if "minimum" in item_schema and item_value < item_schema["minimum"]:
                            diagnostics.append(
                                {
                                    "code": "NODE_PARAMETER_ITEM_BELOW_MINIMUM",
                                    "path": f"graph.nodes.{node.id}.parameters.{name}.{item_index}",
                                    "message": f"节点参数 {name} 的元素低于最小值。",
                                    "severity": "error",
                                }
                            )
                        if "maximum" in item_schema and item_value > item_schema["maximum"]:
                            diagnostics.append(
                                {
                                    "code": "NODE_PARAMETER_ITEM_ABOVE_MAXIMUM",
                                    "path": f"graph.nodes.{node.id}.parameters.{name}.{item_index}",
                                    "message": f"节点参数 {name} 的元素超过最大值。",
                                    "severity": "error",
                                }
                            )
    return diagnostics


def _topological_order(definition: RegimeDefinitionV2) -> tuple[list[str], list[dict[str, Any]]]:
    nodes = {node.id: node for node in definition.graph.nodes}
    indegree = {node_id: 0 for node_id in nodes}
    consumers: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    diagnostics: list[dict[str, Any]] = []
    edge_count = 0
    for node in definition.graph.nodes:
        for input_name, reference in node.inputs.items():
            edge_count += 1
            if reference.node_id not in nodes:
                diagnostics.append(
                    {
                        "code": "UNKNOWN_INPUT_NODE",
                        "path": f"graph.nodes.{node.id}.inputs.{input_name}",
                        "message": f"输入引用了不存在的节点 {reference.node_id}。",
                        "severity": "error",
                    }
                )
                continue
            indegree[node.id] += 1
            consumers[reference.node_id].append(node.id)
    if edge_count > 256:
        diagnostics.append(
            {
                "code": "TOO_MANY_GRAPH_EDGES",
                "path": "graph.nodes",
                "message": "图谱最多允许 256 条连线。",
                "severity": "error",
            }
        )
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for consumer in consumers[node_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                queue.append(consumer)
    if len(order) != len(nodes):
        diagnostics.append(
            {
                "code": "GRAPH_CYCLE",
                "path": "graph.nodes",
                "message": "图谱存在循环依赖。",
                "severity": "error",
            }
        )
    return order, diagnostics


def graph_structure_hash(definition: RegimeDefinitionV2) -> str:
    payload = {
        "registry_version": REGISTRY_VERSION,
        "nodes": [
            {
                "id": node.id,
                "type": node.type,
                "type_id": node.type_id,
                "type_version": node.type_version,
                "inputs": {
                    name: reference.model_dump(mode="json")
                    for name, reference in sorted(node.inputs.items())
                },
            }
            for node in sorted(definition.graph.nodes, key=lambda item: item.id)
        ],
        "edges": [edge.model_dump(mode="json") for edge in definition.graph.edges],
        "outputs": {
            name: reference.model_dump(mode="json")
            for name, reference in sorted(definition.graph.outputs.items())
        },
        "state_count": len(definition.states),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def definition_content_hash(definition: RegimeDefinitionV2) -> str:
    payload = definition.model_dump(mode="json", exclude={"id", "revision", "created_at", "updated_at"})
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def inspect_definition_v2(definition: RegimeDefinitionV2) -> dict[str, Any]:
    nodes = {node.id: node for node in definition.graph.nodes}
    diagnostics: list[dict[str, Any]] = []
    order, topology_diagnostics = _topological_order(definition)
    diagnostics.extend(topology_diagnostics)
    inferred: dict[str, dict[str, str]] = {}
    execution_lanes: set[str] = set()

    for node in definition.graph.nodes:
        metadata = NODE_REGISTRY.get(node.type)
        if metadata is None:
            diagnostics.append(
                {
                    "code": "UNKNOWN_NODE_TYPE",
                    "path": f"graph.nodes.{node.id}.type",
                    "message": f"未注册节点类型 {node.type}。",
                    "severity": "error",
                }
            )
            continue
        diagnostics.extend(_parameter_diagnostics(node, metadata))
        if int(node.type_version) != int(metadata["version"]):
            diagnostics.append(
                {
                    "code": "UNKNOWN_NODE_TYPE_VERSION",
                    "path": f"graph.nodes.{node.id}.type_version",
                    "message": f"节点 {node.type} 不支持版本 {node.type_version}。",
                    "severity": "error",
                }
            )
        declared_inputs = {item["name"]: item for item in metadata["inputs"]}
        required_inputs = {name for name, item in declared_inputs.items() if item.get("required", True)}
        for name in sorted(required_inputs - set(node.inputs)):
            diagnostics.append(
                {
                    "code": "MISSING_NODE_INPUT",
                    "path": f"graph.nodes.{node.id}.inputs.{name}",
                    "message": f"节点 {node.id} 缺少输入端口 {name}。",
                    "severity": "error",
                }
            )
        for input_name, reference in node.inputs.items():
            expected = declared_inputs.get(input_name)
            if expected is None:
                diagnostics.append(
                    {
                        "code": "UNKNOWN_NODE_INPUT",
                        "path": f"graph.nodes.{node.id}.inputs.{input_name}",
                        "message": f"节点 {node.id} 不支持输入端口 {input_name}。",
                        "severity": "error",
                    }
                )
                continue
            upstream = nodes.get(reference.node_id)
            upstream_meta = NODE_REGISTRY.get(upstream.type) if upstream is not None else None
            upstream_outputs = {
                item["name"]: item for item in (upstream_meta or {}).get("outputs", [])
            }
            actual = upstream_outputs.get(reference.port)
            if upstream is not None and actual is None:
                diagnostics.append(
                    {
                        "code": "UNKNOWN_OUTPUT_PORT",
                        "path": f"graph.nodes.{node.id}.inputs.{input_name}",
                        "message": f"节点 {reference.node_id} 没有输出端口 {reference.port}。",
                        "severity": "error",
                    }
                )
            elif actual is not None and actual["type"] != expected["type"]:
                diagnostics.append(
                    {
                        "code": "PORT_TYPE_MISMATCH",
                        "path": f"graph.nodes.{node.id}.inputs.{input_name}",
                        "message": f"端口需要 {expected['type']}，实际为 {actual['type']}。",
                        "severity": "error",
                    }
                )
        inferred[node.id] = {item["name"]: item["type"] for item in metadata["outputs"]}
        execution_lanes.add(str(metadata["njit_policy"]["execution_lane"]))
        if node.type == "model.external_optimized":
            declaration = node.parameters.get("declaration")
            if isinstance(declaration, Mapping):
                try:
                    validate_execution_audit(declaration)
                except ComputePolicyError as exc:
                    diagnostics.append(
                        {
                            "code": "INVALID_THIRD_PARTY_MODEL_DECLARATION",
                            "path": f"graph.nodes.{node.id}.parameters.declaration",
                            "message": str(exc),
                            "severity": "error",
                        }
                    )

        if node.type == "model.threshold":
            lower_value = node.parameters.get("lower", -0.001)
            upper_value = node.parameters.get("upper", 0.001)
            if (
                isinstance(lower_value, (int, float))
                and not isinstance(lower_value, bool)
                and isinstance(upper_value, (int, float))
                and not isinstance(upper_value, bool)
                and float(lower_value) >= float(upper_value)
            ):
                diagnostics.append(
                    {
                        "code": "INVALID_THRESHOLDS",
                        "path": f"graph.nodes.{node.id}.parameters",
                        "message": "下阈值必须小于上阈值。",
                        "severity": "error",
                    }
                )
        if node.type in {"transform.clip"}:
            lower_value = node.parameters.get("lower", -3.0)
            upper_value = node.parameters.get("upper", 3.0)
            if (
                isinstance(lower_value, (int, float))
                and not isinstance(lower_value, bool)
                and isinstance(upper_value, (int, float))
                and not isinstance(upper_value, bool)
                and float(lower_value) >= float(upper_value)
            ):
                diagnostics.append(
                    {
                        "code": "INVALID_CLIP_BOUNDS",
                        "path": f"graph.nodes.{node.id}.parameters",
                        "message": "截尾下界必须小于上界。",
                        "severity": "error",
                    }
                )
        if node.type in {"model.hysteresis", "post.hysteresis"}:
            lower_enter = node.parameters.get("lower_enter", -0.0015)
            lower_exit = node.parameters.get("lower_exit", -0.0002)
            upper_exit = node.parameters.get("upper_exit", 0.0002)
            upper_enter = node.parameters.get("upper_enter", 0.0015)
            values = (lower_enter, lower_exit, upper_exit, upper_enter)
            if all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in values
            ) and not (
                float(lower_enter)
                < float(lower_exit)
                <= float(upper_exit)
                < float(upper_enter)
            ):
                diagnostics.append(
                    {
                        "code": "INVALID_HYSTERESIS_BOUNDS",
                        "path": f"graph.nodes.{node.id}.parameters",
                        "message": "滞回阈值必须满足 lower_enter < lower_exit <= upper_exit < upper_enter。",
                        "severity": "error",
                    }
                )
        if node.type == "model.ensemble":
            member_count = sum(
                1
                for name in ("state_1", "state_2", "state_3", "state_4")
                if name in node.inputs
            )
            weights = node.parameters.get("weights", [0.5, 0.5])
            if isinstance(weights, list) and len(weights) != member_count:
                diagnostics.append(
                    {
                        "code": "ENSEMBLE_WEIGHT_COUNT_MISMATCH",
                        "path": f"graph.nodes.{node.id}.parameters.weights",
                        "message": "集成权重数量必须与已连接的成员数量一致。",
                        "severity": "error",
                    }
                )
            if (
                isinstance(weights, list)
                and weights
                and all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in weights
                )
                and sum(float(value) for value in weights) <= 0.0
            ):
                diagnostics.append(
                    {
                        "code": "ENSEMBLE_WEIGHTS_NOT_POSITIVE",
                        "path": f"graph.nodes.{node.id}.parameters.weights",
                        "message": "集成权重合计必须大于 0。",
                        "severity": "error",
                    }
                )
        if node.type == "post.component_map":
            mapping = node.parameters.get("mapping")
            if isinstance(mapping, list):
                invalid_mapping = [
                    value
                    for value in mapping
                    if isinstance(value, int)
                    and not isinstance(value, bool)
                    and (value < -1 or value >= len(definition.states))
                ]
                if invalid_mapping:
                    diagnostics.append(
                        {
                            "code": "COMPONENT_MAPPING_OUT_OF_RANGE",
                            "path": f"graph.nodes.{node.id}.parameters.mapping",
                            "message": "状态映射只能使用 -1 或已定义状态的编码。",
                            "severity": "error",
                        }
                    )
        if node.type == "feature.formula":
            variables = node.parameters.get("variables", {})
            if isinstance(variables, Mapping):
                for alias, port_name in variables.items():
                    if not str(alias).isidentifier() or str(port_name) not in node.inputs:
                        diagnostics.append(
                            {
                                "code": "INVALID_FORMULA_VARIABLE_MAPPING",
                                "path": f"graph.nodes.{node.id}.parameters.variables",
                                "message": "公式变量必须是合法标识符并映射到已连接输入端口。",
                                "severity": "error",
                            }
                        )
                        break

    for output_name, reference in definition.graph.outputs.items():
        actual = inferred.get(reference.node_id, {}).get(reference.port)
        expected = {
            "state": STATE_CODES,
            "probabilities": PROBABILITIES,
            "confidence": CONFIDENCE,
            "recognition_index": INDEX_SERIES,
            "effective_index": INDEX_SERIES,
            "reason_code": REASON_CODES,
        }.get(output_name)
        if actual is None:
            diagnostics.append(
                {
                    "code": "INVALID_GRAPH_OUTPUT",
                    "path": f"graph.outputs.{output_name}",
                    "message": "图谱输出引用的节点或端口不存在。",
                    "severity": "error",
                }
            )
        elif expected is not None and actual != expected:
            diagnostics.append(
                {
                    "code": "GRAPH_OUTPUT_TYPE_MISMATCH",
                    "path": f"graph.outputs.{output_name}",
                    "message": f"输出 {output_name} 必须是 {expected}，实际为 {actual}。",
                    "severity": "error",
                }
            )
    for node_id in definition.graph.exposed_node_ids:
        if node_id not in nodes:
            diagnostics.append(
                {
                    "code": "UNKNOWN_EXPOSED_NODE",
                    "path": "graph.exposed_node_ids",
                    "message": f"需要展示的节点 {node_id} 不存在。",
                    "severity": "error",
                }
            )

    source_count = sum(
        1
        for node in definition.graph.nodes
        if node.type.startswith("source.") and node.type != "source.constant"
    )
    if source_count == 0:
        diagnostics.append(
            {
                "code": "MISSING_SOURCE_NODE",
                "path": "graph.nodes",
                "message": "图谱至少需要一个数据源节点。",
                "severity": "error",
            }
        )
    model_types = {node.type for node in definition.graph.nodes if node.type.startswith("model.")}
    required_states = 4 if "model.quadrant" in model_types else 3
    if model_types and "model.external_optimized" not in model_types and len(definition.states) < required_states:
        diagnostics.append(
            {
                "code": "INSUFFICIENT_STATE_DEFINITIONS",
                "path": "states",
                "message": f"当前模型至少需要 {required_states} 个状态定义。",
                "severity": "error",
            }
        )

    axis_groups: dict[tuple[str, str], str] = {}
    for node_id in order:
        node = nodes[node_id]
        metadata = NODE_REGISTRY.get(node.type)
        if metadata is None:
            continue
        if node.type.startswith("source.") and node.type != "source.constant":
            axis = f"source:{node.id}"
        elif node.type in {
            "align.strict_intersection",
            "align.pit_asof",
            "align.resample",
            "align.cross_section",
        }:
            axis = f"aligned:{node.id}"
        else:
            input_axes = {
                axis_groups[(reference.node_id, reference.port)]
                for reference in node.inputs.values()
                if (reference.node_id, reference.port) in axis_groups
            }
            if len(input_axes) > 1:
                diagnostics.append(
                    {
                        "code": "EXPLICIT_ALIGNMENT_REQUIRED",
                        "path": f"graph.nodes.{node.id}.inputs",
                        "message": "来自不同时间轴的输入必须先经过显式对齐节点，禁止隐式交集或补值。",
                        "severity": "error",
                    }
                )
            axis = next(iter(input_axes), f"derived:{node.id}")
        for output in metadata["outputs"]:
            axis_groups[(node.id, output["name"])] = axis

    state_reference = definition.graph.outputs.get("state")
    state_axis = (
        axis_groups.get((state_reference.node_id, state_reference.port))
        if state_reference is not None
        else None
    )
    if state_axis is not None:
        for output_name, reference in definition.graph.outputs.items():
            if output_name == "state":
                continue
            output_axis = axis_groups.get((reference.node_id, reference.port))
            if output_axis is not None and output_axis != state_axis:
                diagnostics.append(
                    {
                        "code": "FINAL_OUTPUT_AXIS_MISMATCH",
                        "path": f"graph.outputs.{output_name}",
                        "message": (
                            f"最终输出 {output_name} 与 state 不在同一时间轴；"
                            "请先显式对齐。"
                        ),
                        "severity": "error",
                    }
                )

    direct_dependencies: dict[str, list[str]] = {}
    transitive_dependencies: dict[str, list[str]] = {}
    for node_id in order:
        node = nodes[node_id]
        direct = sorted({reference.node_id for reference in node.inputs.values()})
        direct_dependencies[node_id] = direct
        inherited: set[str] = set(direct)
        for dependency_id in direct:
            inherited.update(transitive_dependencies.get(dependency_id, []))
        transitive_dependencies[node_id] = sorted(inherited)

    decision_roots: dict[str, set[str]] = {}
    for node_id in order:
        node = nodes[node_id]
        if node.type.startswith("model."):
            # A downstream model such as an ensemble becomes the single
            # decision root; its member models must not leak separate outputs.
            decision_roots[node_id] = {node_id}
            continue
        roots: set[str] = set()
        for dependency_id in direct_dependencies.get(node_id, []):
            roots.update(decision_roots.get(dependency_id, set()))
        decision_roots[node_id] = roots

    state_reference = definition.graph.outputs.get("state")
    state_decision_roots = (
        decision_roots.get(state_reference.node_id, set())
        if state_reference is not None
        else set()
    )
    for output_name, reference in definition.graph.outputs.items():
        if output_name == "state":
            continue
        output_roots = decision_roots.get(reference.node_id, set())
        if state_decision_roots and output_roots and output_roots != state_decision_roots:
            diagnostics.append(
                {
                    "code": "MULTIPLE_STATE_ROOTS",
                    "path": f"graph.outputs.{output_name}",
                    "message": (
                        f"最终输出 {output_name} 来自另一条状态判定根；"
                        "状态、概率、置信度和时间语义必须共享同一模型根。"
                    ),
                    "severity": "error",
                }
            )

    symbolic_shapes: dict[str, list[str]] = {}
    for node_id in order:
        metadata = NODE_REGISTRY.get(nodes[node_id].type)
        if metadata is None:
            continue
        axis = next(
            (
                axis_groups[(node_id, output["name"])]
                for output in metadata["outputs"]
                if (node_id, output["name"]) in axis_groups
            ),
            f"derived:{node_id}",
        )
        time_dimension = f"T[{axis}]"
        for output in metadata["outputs"]:
            value_type = output["type"]
            if value_type == PROBABILITIES:
                shape = [time_dimension, f"S={len(definition.states)}"]
            elif value_type == "matrix<time,feature>":
                feature_count = sum(
                    1 for name in nodes[node_id].inputs if name.startswith("feature_")
                )
                shape = [time_dimension, f"F={max(feature_count, 1)}"]
            else:
                shape = [time_dimension]
            symbolic_shapes[f"{node_id}.{output['name']}"] = shape

    required_node_ids: set[str] = set()
    pending = [reference.node_id for reference in definition.graph.outputs.values()]
    while pending:
        node_id = pending.pop()
        if node_id in required_node_ids or node_id not in nodes:
            continue
        required_node_ids.add(node_id)
        pending.extend(reference.node_id for reference in nodes[node_id].inputs.values())
    noncausal_nodes = sorted(
        node_id
        for node_id in required_node_ids
        if NODE_REGISTRY.get(nodes[node_id].type, {}).get("causal") is not True
    )
    repaint_nodes = sorted(
        node_id
        for node_id in required_node_ids
        if NODE_REGISTRY.get(nodes[node_id].type, {}).get("repaints") is True
    )
    non_realtime_nodes = sorted(
        node_id
        for node_id in required_node_ids
        if NODE_REGISTRY.get(nodes[node_id].type, {}).get("supports_realtime") is not True
    )

    cost_units_by_class = {
        "io_bound": 1.0,
        "linear": 1.0,
        "adapter_declared": 10.0,
    }
    node_costs: list[dict[str, Any]] = []
    total_relative_units = 0.0
    for node_id in order:
        metadata = NODE_REGISTRY.get(nodes[node_id].type)
        if metadata is None:
            continue
        declared = copy.deepcopy(metadata.get("cost_estimate") or {})
        relative_units = float(cost_units_by_class.get(str(declared.get("class")), 2.0))
        total_relative_units += relative_units
        node_costs.append(
            {
                "node_id": node_id,
                "node_type": nodes[node_id].type,
                "relative_units_per_observation": relative_units,
                **declared,
            }
        )

    warnings: list[dict[str, Any]] = []
    if not definition.evaluation_targets:
        warnings.append(
            {
                "code": "NO_EVALUATION_TARGET",
                "path": "evaluation_targets",
                "message": "尚未配置独立评价标的；预览将使用首个分类输入作为展示序列。",
                "severity": "warning",
            }
        )
    if "isolated_model_train_or_infer" in execution_lanes:
        warnings.append(
            {
                "code": "THIRD_PARTY_MODEL_ISOLATED",
                "path": "graph.nodes",
                "message": "第三方模型只允许在隔离训练/推理通道执行，前后处理仍使用 NJIT。",
                "severity": "warning",
            }
        )
    legacy_resample_nodes = [
        node.id
        for node in definition.graph.nodes
        if node.type == "align.resample"
        and "frequency" not in node.parameters
        and "every" in node.parameters
    ]
    if legacy_resample_nodes:
        warnings.append(
            {
                "code": "LEGACY_POSITIONAL_RESAMPLE_DEPRECATED",
                "path": "graph.nodes",
                "message": "固定 every/offset 抽样仅保留兼容，请改用日/周/月/季/年日历频率转换。",
                "severity": "warning",
                "node_ids": legacy_resample_nodes,
            }
        )
    return {
        "valid": not diagnostics,
        "schema_version": "2.0",
        "registry_version": REGISTRY_VERSION,
        "graph_hash": graph_structure_hash(definition),
        "definition_hash": definition_content_hash(definition),
        "topological_order": order,
        "inferred": {
            "ports": inferred,
            "symbolic_shapes": symbolic_shapes,
            "axis_groups": {
                f"{node_id}.{port}": axis
                for (node_id, port), axis in axis_groups.items()
            },
            "state_count": len(definition.states),
        },
        "dependencies": {
            "direct": direct_dependencies,
            "transitive": transitive_dependencies,
            "required_for_outputs": sorted(required_node_ids),
            "decision_roots": {
                node_id: sorted(root_ids)
                for node_id, root_ids in decision_roots.items()
            },
        },
        "causality": {
            "causal": not noncausal_nodes,
            "repaints": bool(repaint_nodes),
            "realtime_supported": not non_realtime_nodes,
            "noncausal_node_ids": noncausal_nodes,
            "repaint_node_ids": repaint_nodes,
            "non_realtime_node_ids": non_realtime_nodes,
        },
        "cost_estimate": {
            "model": "relative_linearized_cost_v1",
            "total_relative_units_per_observation": total_relative_units,
            "nodes": node_costs,
        },
        "execution_lanes": sorted(execution_lanes),
        "errors": diagnostics,
        "warnings": warnings,
    }


def validate_definition_v2(definition: RegimeDefinitionV2) -> dict[str, Any]:
    inspection = inspect_definition_v2(definition)
    if not inspection["valid"]:
        raise ValidationError(
            "INVALID_REGIME_GRAPH_V2",
            "历史情景图谱未通过语义校验。",
            "definition.graph",
            inspection["errors"],
        )
    return inspection


def validate_graph_fragment_v2(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize a self-contained user subgraph.

    A fragment has no business-level state root of its own, so it is wrapped in
    a temporary definition only for the shared structural, port, parameter,
    topology and time-axis checks.  The two definition-only diagnostics are
    intentionally ignored; every graph diagnostic remains fail-closed.
    """

    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValidationError(
            "EMPTY_REGIME_GRAPH_ASSET",
            "用户子图至少需要一个节点。",
            "graph.nodes",
        )
    final_node = raw_nodes[-1]
    if not isinstance(final_node, Mapping):
        raise ValidationError(
            "INVALID_REGIME_GRAPH_ASSET_NODES",
            "用户子图节点必须是对象。",
            "graph.nodes",
        )
    final_type = str(final_node.get("type") or final_node.get("type_id") or "")
    metadata = NODE_REGISTRY.get(final_type) or {}
    output_ports = metadata.get("outputs") or []
    if not output_ports:
        raise ValidationError(
            "REGIME_SUBGRAPH_OUTPUT_REQUIRED",
            "用户子图末端节点必须至少提供一个输出端口。",
            "graph.nodes",
        )
    temporary = parse_definition_v2(
        {
            "schema_version": "2.0",
            "name": "用户子图结构校验",
            "graph": {
                "nodes": copy.deepcopy(raw_nodes),
                "edges": copy.deepcopy(payload.get("edges") or []),
                "outputs": {
                    "state": {
                        "node_id": str(final_node.get("id") or ""),
                        "port": str(output_ports[0]["name"]),
                    }
                },
                "exposed_node_ids": [],
            },
            "states": [
                {"id": "fragment_0", "label": "状态 0", "order": 0},
                {"id": "fragment_1", "label": "状态 1", "order": 1},
                {"id": "fragment_2", "label": "状态 2", "order": 2},
                {"id": "fragment_3", "label": "状态 3", "order": 3},
            ],
            "evaluation_targets": [],
            "validation": {},
            "usage_intent": "research_display",
        }
    )
    inspection = inspect_definition_v2(temporary)
    definition_only_codes = {
        "GRAPH_OUTPUT_TYPE_MISMATCH",
        "MISSING_SOURCE_NODE",
    }
    errors = [
        item
        for item in inspection["errors"]
        if item.get("code") not in definition_only_codes
    ]
    if errors:
        raise ValidationError(
            "INVALID_REGIME_SUBGRAPH",
            "用户子图未通过拓扑、端口或参数校验。",
            "graph",
            errors,
        )
    return {
        "nodes": [node.model_dump(mode="json") for node in temporary.graph.nodes],
        "edges": [edge.model_dump(mode="json") for edge in temporary.graph.edges],
    }


__all__ = [
    "EvaluationTargetV2",
    "GraphPortRefV2",
    "RegimeDefinitionV2",
    "RegimeGraphNodeV2",
    "RegimeGraphV2",
    "RegimeStateV2",
    "definition_content_hash",
    "graph_structure_hash",
    "inspect_definition_v2",
    "parse_definition_v2",
    "validate_definition_v2",
    "validate_graph_fragment_v2",
]
