"""Project an unfinished draft onto one output and its dependencies; never save it."""

import copy
from typing import Any, Mapping

from pydantic import ValidationError as PydanticValidationError

from custom_indicators.errors import ValidationError
from .v2_contracts import GraphPortRefV2, RegimeDefinitionV2, _diagnostics_from_pydantic
from .v2_templates import MARKET_STATES
from .v2_registry import NODE_REGISTRY, PORT_LABELS


def preview_output_context(definition: RegimeDefinitionV2, outputs: Mapping[str, Any], node_id: str) -> dict[str, Any]:
    """Describe frozen upstream outputs without resolving sources or executing a graph."""
    nodes = {node.id: node for node in definition.graph.nodes}
    distances = {node_id: 0}
    pending = [node_id]
    for current in pending:
        references = list(nodes[current].inputs.values())
        references.extend(edge.source for edge in definition.graph.edges if edge.target.node_id == current)
        for ref in references:
            if ref.node_id not in distances:
                distances[ref.node_id] = distances[current] + 1
                pending.append(ref.node_id)

    def label(key: str) -> str:
        node = nodes[key]
        return (node.label or "").strip() or NODE_REGISTRY[node.type].get("label") or key

    upstream = []
    for key in pending[1:]:
        for port in NODE_REGISTRY[nodes[key].type]["outputs"]:
            value = outputs.get(key, {}).get(port["name"])
            if value is None:
                continue
            supported = value.values.ndim == 1 and port["type"] in {"series<float64>", "confidence<time>"}
            upstream.append({
                "node_id": key, "node_label": label(key), "port": port["name"],
                "port_label": port.get("label") or PORT_LABELS.get(port["name"], port["name"]),
                "value_type": port["type"], "distance": distances[key], "plottable": supported,
                "unavailable_reason": None if supported else "此输出不是连续数值序列，请单独查看节点数据。",
            })
    return {"node_label": label(node_id), "upstream_outputs": upstream}


def node_preview_definition(payload: Mapping[str, Any], target: Mapping[str, Any]) -> RegimeDefinitionV2:
    try:
        reference = GraphPortRefV2.model_validate(target)
        graph = payload.get("graph") or {}
        nodes = graph.get("nodes") or []
        if not isinstance(nodes, list) or not 1 <= len(nodes) <= 128:
            raise ValueError("请先添加节点；计算图最多支持 128 个节点。")
        node_map = {node["id"]: node for node in nodes}
        if len(node_map) != len(nodes):
            raise ValueError("节点 ID 重复，无法确定预览目标。")
        edges = graph.get("edges") or []
        needed: set[str] = set()
        pending = [reference.node_id]
        while pending:
            node_id = pending.pop()
            if node_id in needed:
                continue
            if node_id not in node_map:
                raise ValueError(f"预览所需节点 {node_id} 不存在。")
            needed.add(node_id)
            pending.extend(item["node_id"] for item in (node_map[node_id].get("inputs") or {}).values())
            pending.extend(edge["source"]["node_id"] for edge in edges if edge["target"]["node_id"] == node_id)
        selected = [copy.deepcopy(node) for node in nodes if node["id"] in needed]
        # State settings only affect classifiers; unfinished unrelated settings do not block a price preview.
        states = payload.get("states") if any(node.get("type", node.get("type_id", "")).startswith("model.") for node in selected) else MARKET_STATES
        projected = {
            "schema_version": "2.0", "name": "节点预览", "states": copy.deepcopy(states or MARKET_STATES),
            "graph": {
                "nodes": selected,
                "edges": [copy.deepcopy(edge) for edge in edges if edge["target"]["node_id"] in needed],
                "outputs": {"preview": reference.model_dump()},
            },
        }
        return RegimeDefinitionV2.model_validate(projected, context={"node_preview": True})
    except PydanticValidationError as exc:
        raise ValidationError("INVALID_NODE_PREVIEW", "所选节点及上游配置不完整。", "preview_target", _diagnostics_from_pydantic(exc)) from exc
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValidationError("INVALID_NODE_PREVIEW", str(exc), "preview_target") from exc
