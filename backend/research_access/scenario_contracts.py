"""Bounded graph authoring and read projections; business identity stays with the host."""
import unicodedata
from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator
from .contracts import Contract
from . import views


class Ref(Contract):
    node_id: str = Field(min_length=1, max_length=64)
    port: str = Field(default="value", min_length=1, max_length=64)

class Edge(Contract):
    source: Ref
    target: Ref

class Node(Contract):
    id: str = Field(min_length=1, max_length=64)
    type: str = Field(min_length=1, max_length=100)
    type_id: str | None = None
    type_version: int = Field(default=1, ge=1)
    label: str | None = Field(default=None, max_length=100)
    parameters: dict[str, StrictBool | StrictInt | StrictFloat | StrictStr] = Field(default_factory=dict, max_length=64)
    inputs: dict[str, Ref] = Field(default_factory=dict, max_length=32)

    @field_validator("parameters")
    @classmethod
    def registered_parameters(cls, value, info):
        from historical_regimes.v2_registry import NODE_REGISTRY
        node_type = info.data.get("type")
        schema = NODE_REGISTRY.get(node_type, {}).get("parameter_schema", {}).get("properties", {})
        for key in value:
            # Names alone never confer permission: require the actual scalar schema.
            if schema.get(key, {}).get("type") not in {"number", "integer", "boolean", "string"}:
                raise ValueError(f"参数 {key} 不是已登记的单值配置；请在编辑器处理数据或复合参数。")
        if node_type == "source.inline" or str(node_type).startswith("annotation."):
            raise ValueError("内联数据与人工事件记录不进入 AI；请使用引用已有数据的数据源。")
        return value

class Graph(Contract):
    nodes: list[Node] = Field(min_length=1, max_length=128)
    edges: list[Edge] = Field(default_factory=list, max_length=256)
    outputs: dict[str, Ref] = Field(max_length=8)
    exposed_node_ids: list[str] = Field(default_factory=list, max_length=64)

class State(Contract):
    id: str = Field(min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=100)
    role: str = Field(default="neutral", max_length=100)
    color: str = Field(default="#64748b", max_length=20)
    order: int = 0

class Definition(Contract):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    graph: Graph
    states: list[State] = Field(min_length=2, max_length=12)

class ReadArgs(Contract):
    definition_id: str = Field(min_length=1, max_length=100)
    revision: int = Field(ge=1)
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=8, ge=1, le=16)

def definition_view(value):
    """An exact, bounded authoring DTO, never a generic graph/parameter pass-through."""
    return Definition.model_validate(value).model_dump(exclude_unset=True)

def authoring_definition(value):
    """Project business-owned definitions without exposing study/run/source receipts."""
    result = {key: value[key] for key in Definition.model_fields if key in value}
    graph = result.get("graph") or {}
    result["graph"] = {key: graph[key] for key in Graph.model_fields if key in graph}
    result["graph"]["nodes"] = [{key: node[key] for key in Node.model_fields if key in node}
                                 for node in graph.get("nodes", [])]
    return definition_view(result)

def page_view(value):
    payload, redactions = views._counter_view({"mode": views.STR, "as_of": views.STR,
        "selected_node_id": views.STR, "editor_pending": views.BOOL})(value, views.Projection())
    if isinstance(value, dict) and isinstance(value.get("definition"), dict):
        try:
            payload["definition"] = authoring_definition(value["definition"])
        except (ValueError, TypeError, KeyError):
            payload["notice"] = "当前定义含内联数据、复合参数或未登记配置，不能交给 AI 修改；可在编辑器检查，或从目录模板新建。"
            redactions += 1
    return payload, redactions

PARAMETER = {"name": views.STR, "type": views.STR, "description": views.STR,
             "minimum": views.NUM, "maximum": views.NUM, "enum": [views.STR]}

CATALOG = {"kind": views.STR, "total": views.INT, "next_offset": views.INT,
    "items": [{"id": views.STR, "name": views.STR, "label": views.STR,
               "description": views.STR, "category": views.STR,
               "inputs": [{"name": views.STR, "type": views.STR}],
               "outputs": [{"name": views.STR, "type": views.STR}], "parameters": [PARAMETER]}]}

CATALOG["items"][0].update({"status": views.STR, "regime_node_type": views.STR,
                           "revision": views.INT, "default_mode": views.STR,
                           "fields": [{"name": views.STR, "label": views.STR}]})

REF_VIEW = {"node_id": views.STR, "port": views.STR}

READ_VIEW = {
    "id": views.STR, "revision": views.INT, "name": views.STR,
    "description": views.STR, "default_mode": views.STR,
    "total": views.INT, "next_offset": views.INT,
    "states": [{"id": views.STR, "label": views.STR, "role": views.STR,
                "color": views.STR, "order": views.INT}],
    "outputs": {"*": REF_VIEW},
    "edges": [{"source": REF_VIEW, "target": REF_VIEW}],
}

def saved_definition_view(value, facts):
    """Read registered settings, never observations or business qualification receipts.

    Explanation is read-only: one unsupported parameter must not hide every node.
    The stricter, all-or-nothing authoring DTO remains unchanged.
    """
    from historical_regimes.v2_registry import NODE_REGISTRY
    counter = views.Counter()
    result = views.project({k: value[k] for k in READ_VIEW if k in value}, READ_VIEW, "algorithm", counter)
    result["items"] = []
    scalar_types = {"number": views.NUM, "integer": views.INT, "string": views.STR, "boolean": views.BOOL}
    for node in value.get("items", []):
        metadata = NODE_REGISTRY.get(node.get("type"), {})
        properties = metadata.get("parameter_schema", {}).get("properties", {})
        if node.get("type") == "source.inline" or str(node.get("type", "")).startswith("annotation."):
            properties = {}
        params_schema = {k: scalar_types[spec["type"]] for k, spec in properties.items()
                         if spec.get("type") in scalar_types}
        schema = {"id": views.STR, "type": views.STR, "type_version": views.INT,
                  "label": views.STR, "inputs": {"*": REF_VIEW}, "parameters": params_schema}
        row = views.project({k: node[k] for k in schema if k in node}, schema, "node", counter)
        omitted = set(node.get("parameters", {})) - set(row.get("parameters", {}))
        if omitted:
            row["notice"] = "部分数据或复合参数未提供；不能据此推断其内容，请在编辑器核对。"
        contract = {k: metadata[k] for k in ("label", "description", "causal", "repaints", "supports_realtime") if k in metadata}
        row["contract"] = views.project(contract, {"label": views.STR, "description": views.STR,
            "causal": views.BOOL, "repaints": views.BOOL, "supports_realtime": views.BOOL}, "contract", counter)
        result["items"].append(row)
    return result, counter.dropped

def _search_text(value):
    # UI/user names differ in spaces, middle dots and full-width punctuation.
    return "".join(c for c in unicodedata.normalize("NFKC", str(value)).casefold() if c.isalnum())

def catalog_view(value, facts):
    projected, redactions = views._counter_view(CATALOG)(value, facts)
    for output, item in zip(projected.get("items", []), value.get("items", [])):
        if "binding_parameters" in item:
            try:
                node = Node(id="source", type=item["regime_node_type"], parameters=item["binding_parameters"])
                output["source_node"] = node.model_dump(exclude_unset=True)
            except (ValueError, TypeError, KeyError):
                output["note"] = "该数据源需要在研究数据面板配置后使用。"
                redactions += 1
    return projected, redactions
