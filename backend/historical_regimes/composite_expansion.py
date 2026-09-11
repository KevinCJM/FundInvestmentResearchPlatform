"""Lossless, definition-only expansion of registered regime composites.

No source resolution, plan preparation, computation or persistence is allowed
here. Incomplete inputs are valid editor drafts; broken references are not.
"""
from __future__ import annotations

import copy
import hashlib
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from custom_indicators.errors import ValidationError
from .granular_registry import COMPOSITE_STEPS
from .v2_contracts import inspect_definition_v2, parse_definition_v2
from .v2_registry import NODE_REGISTRY


class ExpandCompositeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition: dict[str, Any]
    node_id: str = Field(min_length=1, max_length=64)
    mode: Literal["realtime", "retrospective"] = "realtime"


class _Expansion:
    def __init__(self, node, reserved):
        self.node = node
        self.reserved = set(reserved)
        self.nodes: list[dict[str, Any]] = []
        self.digest = hashlib.sha256(node.id.encode()).hexdigest()[:8]

    def add(self, key, node_type, label, inputs=None, parameters=None, port="value"):
        stem = f"{self.node.id[:24]}_{key}_{self.digest}"
        identifier = stem
        counter = 1
        while identifier in self.reserved:
            identifier = f"{stem}_{counter}"
            counter += 1
        self.reserved.add(identifier)
        self.nodes.append({"id": identifier, "type": node_type,
                           "type_version": NODE_REGISTRY[node_type]["version"],
                           "label": label, "parameters": parameters or {},
                           "inputs": {name: ref for name, ref in (inputs or {}).items() if ref is not None}})
        return {"node_id": identifier, "port": port}

    @staticmethod
    def port(reference, port):
        return {"node_id": reference["node_id"], "port": port}

    def source(self, name):
        reference = self.node.inputs.get(name)
        return reference.model_dump(mode="json") if reference else None

    def encoding(self, state):
        ref = self.add("encoding", "state.encode", "确定性状态编码", {"state": state}, port="probabilities")
        return {"probabilities": ref, "confidence": self.port(ref, "confidence")}

    def threshold(self):
        value = self.source("value")
        upper = self.node.parameters.get("upper", .001)
        lower = self.node.parameters.get("lower", -.001)
        above = self.add("above", "condition.compare", "达到上界（含等号）", {"value": value},
                         {"operator": "ge", "threshold": upper}, "condition")
        below = self.add("below", "condition.compare", "达到下界（含等号）", {"value": value},
                         {"operator": "le", "threshold": lower}, "condition")
        negative = self.add("lower_state", "state.select", "下界条件选态", {"condition": below},
                            {"true_code": 2, "false_code": 1}, "state")
        state = self.add("state", "state.select", "上界条件选态", {"condition": above, "when_false": negative},
                         {"true_code": 0, "false_code": -1}, "state")
        score = value or self.add("score", "transform.identity", "原始分类输入")
        timing = self.add("timing", "output.temporal", "识别时点与原因", {"state": state}, port="recognition_index")
        return {"state": state, "score": score, **self.encoding(state),
                "recognition_index": timing, "reason_code": self.port(timing, "reason_code")}

    def quadrant(self):
        growth, inflation = self.source("growth"), self.source("inflation")
        growth_high = self.add("growth", "condition.compare", "增长达到界线", {"value": growth},
                               {"operator": "ge", "threshold": self.node.parameters.get("growth_threshold", 0.)}, "condition")
        inflation_high = self.add("inflation", "condition.compare", "通胀达到界线", {"value": inflation},
                                  {"operator": "ge", "threshold": self.node.parameters.get("inflation_threshold", 0.)}, "condition")
        high = self.add("high_growth", "state.select", "高增长分支", {"condition": inflation_high},
                        {"true_code": 1, "false_code": 0}, "state")
        low = self.add("low_growth", "state.select", "低增长分支", {"condition": inflation_high},
                       {"true_code": 2, "false_code": 3}, "state")
        state = self.add("state", "state.select", "按增长条件选态",
                         {"condition": growth_high, "when_true": high, "when_false": low},
                         {"true_code": -1, "false_code": -1}, "state")
        score = self.add("score", "math.subtract", "增长减通胀（原得分）", {"left": growth, "right": inflation})
        return {"state": state, "score": score, **self.encoding(state)}

    def peak_trough(self):
        value = self.source("value")
        parameters = self.node.parameters
        windows = {key: parameters.get(key, parameters.get(old, default)) for key, old, default in (
            ("left_window", "window", 8), ("right_window", "window", 8),
            ("head_window", "endpoint_window", 6), ("tail_window", "endpoint_window", 6))}
        pivots = self.add("pivots", "pivot.local_extrema", "候选峰谷定位", {"value": value}, windows, "pivot")
        filtered = self.add("filtered", "pivot.ps_filter", "PS 联合约束筛选", {"value": value, "pivot": pivots},
                            {key: parameters[key] for key in ("min_phase", "min_cycle", "amplitude_exception") if key in parameters}, "pivot")
        segments = self.add("segments", "segment.between_pivots", "完整相邻峰谷分段", {"pivot": filtered}, port="start")
        boundaries = {"start": segments, "end": self.port(segments, "end")}
        phase = self.add("phase", "segment.phase_direction", "完整波段方向", {"pivot": filtered, **boundaries}, port="phase")
        change = self.add("change", "segment.change", "完整波段涨跌幅", {"value": value, **boundaries})
        line = self.add("line", "segment.boundary_line", "峰谷边界连线", {"value": value, **boundaries})
        sideways = self.add("state", "post.peak_sideways", "小波段震荡合并与选态", {"value": value, "phase": phase, **boundaries},
                            {key: parameters[key] for key in ("sideways_enabled", "small_swing_threshold", "sideways_max_range",
                             "sideways_max_efficiency", "sideways_min_duration") if key in parameters}, "state")
        result = {"state": sideways, "pivot": self.port(filtered, "marker"),
                  "phase_start_index": self.port(segments, "start_index"), "phase_end_index": self.port(segments, "end_index"),
                  "phase_return": change, "boundary_line": line}
        result.update({item["name"]: self.port(sideways, item["name"])
                       for item in NODE_REGISTRY["model.peak_trough"]["outputs"] if item["name"].startswith("sideways_")})
        return result


_DRAFT_ERRORS = {"MISSING_NODE_INPUT", "MISSING_SOURCE_NODE"}


def _inspect_draft(payload, placeholder):
    temporary = copy.deepcopy(payload)
    temporary["graph"].setdefault("outputs", {})
    if "state" not in temporary["graph"]["outputs"]:
        temporary["graph"]["outputs"]["state"] = placeholder
    definition = parse_definition_v2(temporary)
    inspection = inspect_definition_v2(definition)
    nodes = {node.id: node for node in definition.graph.nodes}
    axes = inspection["inferred"]["axis_groups"]
    errors = []
    for item in inspection["errors"]:
        if item.get("code") in _DRAFT_ERRORS:
            continue
        if item.get("code") == "EXPLICIT_ALIGNMENT_REQUIRED":
            target = nodes.get(item.get("node_id"))
            input_axes = {axes.get(f"{ref.node_id}.{ref.port}", "") for ref in target.inputs.values()} if target else set()
            known = {axis for axis in input_axes if axis and not axis.startswith("derived:")}
            # Unconnected editor inputs have provisional derived axes. Do not
            # weaken alignment checks between two genuinely different axes.
            if any(axis.startswith("derived:") for axis in input_axes) and len(known) <= 1:
                continue
        errors.append(item)
    if errors:
        raise ValidationError("INVALID_COMPOSITE_DRAFT", "草稿存在非法参数、类型或连线，请先修正后再展开。", "definition.graph", errors)
    return definition


def expand_composite(definition: Mapping[str, Any], node_id: str, mode: str = "realtime") -> dict[str, Any]:
    if mode not in {"realtime", "retrospective"}:
        raise ValidationError("INVALID_REGIME_MODE", "请选择实时或事后模式。", "mode")
    payload = copy.deepcopy(dict(definition))
    graph = payload.get("graph")
    if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list):
        raise ValidationError("INVALID_COMPOSITE_DRAFT", "请提供计算图草稿。", "definition.graph")
    matches = [node for node in graph["nodes"] if isinstance(node, Mapping) and node.get("id") == node_id]
    if len(matches) != 1:
        raise ValidationError("COMPOSITE_NODE_NOT_FOUND", "请选择一个确实存在的组合节点。", "node_id")
    node_type = matches[0].get("type")
    if node_type not in COMPOSITE_STEPS:
        raise ValidationError("NODE_NOT_EXPANDABLE", "此节点没有经过等价验证的展开契约。", "node_id")
    if mode == "realtime" and not NODE_REGISTRY[node_type]["supports_realtime"]:
        raise ValidationError("NON_CAUSAL_REALTIME_GRAPH", "此组合使用事后信息，请先切换到事后模式。", "mode")
    original = _inspect_draft(payload, {"node_id": node_id, "port": "state"})
    node = next(item for item in original.graph.nodes if item.id == node_id)
    minimum_states = 4 if node.type == "model.quadrant" else 3 if node.type == "model.threshold" else 2
    if len(original.states) < minimum_states:
        raise ValidationError("COMPOSITE_STATE_COUNT", f"此组合至少需要 {minimum_states} 个状态，请先配置状态。", "states")
    builder = _Expansion(node, [item.id for item in original.graph.nodes])
    methods = {"model.threshold": builder.threshold, "model.quadrant": builder.quadrant, "model.peak_trough": builder.peak_trough}
    outputs = methods[node.type]()
    if node.label:
        next(item for item in builder.nodes if item["id"] == outputs["state"]["node_id"])["label"] = node.label
    if set(outputs) != {item["name"] for item in NODE_REGISTRY[node.type]["outputs"]}:
        raise RuntimeError("Composite expansion must map every original output.")

    def replace(reference):
        if reference["node_id"] != node_id:
            return reference
        if reference.get("port", "value") not in outputs:
            raise ValidationError("UNKNOWN_OUTPUT_PORT", "展开引用了不存在的输出端口。", "definition.graph")
        return copy.deepcopy(outputs[reference["port"]])

    # Canonical inputs also cover drafts supplied with edges only.
    nodes = []
    for old in original.graph.nodes:
        if old.id == node_id:
            nodes.extend(builder.nodes)
        else:
            current = old.model_dump(mode="json")
            current["inputs"] = {name: replace(ref) for name, ref in current["inputs"].items()}
            nodes.append(current)
    graph["nodes"] = nodes
    graph["outputs"] = {name: replace(ref) for name, ref in (graph.get("outputs") or {}).items()}
    graph["edges"] = [{"source": ref, "target": {"node_id": item["id"], "port": name}}
                      for item in nodes for name, ref in item["inputs"].items()]
    # Exposure is an execution contract, not the set of visible canvas nodes.
    # Preserve every exposed old output, but do not execute unused new outputs.
    exposed = []
    for identifier in graph.get("exposed_node_ids", []):
        exposed.extend([ref["node_id"] for ref in outputs.values()] if identifier == node_id else [identifier])
    graph["exposed_node_ids"] = list(dict.fromkeys(exposed))
    _inspect_draft(payload, outputs["state"])
    return {"contract_version": 1, "definition": payload, "replaced_node_id": node_id,
            "inserted_node_ids": [item["id"] for item in builder.nodes],
            "primary_node_id": outputs["state"]["node_id"], "output_map": outputs,
            "steps": [item["label"] for item in builder.nodes]}
