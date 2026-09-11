"""Bounded, versioned authoring contracts; never executable client plans."""
from __future__ import annotations

from typing import Annotated, Literal, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, model_validator

from cal_indicators.typed_operators import TYPED_DSL_VERSION

NodeId = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_-]*$")]
OutputId = Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Number = StrictBool | StrictInt | StrictFloat


class GraphModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class NodeInput(GraphModel):
    source: Literal["node"]
    node_id: NodeId
    port_id: OutputId = "value"


class ConstantInput(GraphModel):
    source: Literal["constant"]
    value: Number


GraphInput = Annotated[NodeInput | ConstantInput, Field(discriminator="source")]


class VariableNode(GraphModel):
    id: NodeId
    kind: Literal["variable"]
    variable_id: OutputId
    label: str = Field(default="", max_length=80)


class ParameterNode(GraphModel):
    id: NodeId
    kind: Literal["parameter"]
    parameter_id: OutputId
    label: str = Field(default="", max_length=80)


class ConstantNode(GraphModel):
    id: NodeId
    kind: Literal["constant"]
    value: Number
    label: str = Field(default="", max_length=80)


class OperatorNode(GraphModel):
    id: NodeId
    kind: Literal["operator"]
    operator_id: OutputId
    arity: int | None = Field(default=None, ge=1, le=16)
    arguments: dict[OutputId, GraphInput] = Field(default_factory=dict, max_length=16)
    label: str = Field(default="", max_length=80)


AuthoringNode = Annotated[VariableNode | ConstantNode | ParameterNode | OperatorNode, Field(discriminator="kind")]


class GraphOutput(GraphModel):
    id: OutputId
    node_id: NodeId | None = None
    port_id: OutputId = "value"
    label: str = Field(default="", max_length=80)
    unit: str = Field(default="", max_length=20)
    display_format: Literal["number", "percent"] = "number"
    precision: int = Field(default=4, ge=0, le=8)
    output_measure: str = Field(default="auto", max_length=120)
    direction: Literal["neutral", "higher_better", "lower_better"] = "neutral"
    description: str = Field(default="", max_length=500)


class AuthoringGraph(GraphModel):
    graph_version: Literal[1] = 1
    nodes: list[AuthoringNode] = Field(default_factory=list, max_length=128)
    outputs: list[GraphOutput] = Field(min_length=1, max_length=8)


class GraphContext(GraphModel):
    parameter_contract_version: Literal["1.0"] | None = None
    parameter_schema: list[dict[str, Any]] = Field(default_factory=list, max_length=16)
    context_kind: Literal["single_product", "portfolio"] = "single_product"
    result_kind: Literal["scalar", "time_series"] = "scalar"
    dsl_version: str = Field(default=TYPED_DSL_VERSION, max_length=40)
    operator_registry_version: str | None = Field(default=None, max_length=40)
    numeric_kernel_version: str | None = Field(default=None, max_length=40)
    variable_registry_version: str | None = Field(default=None, max_length=80)
    data_contract_version: str | None = Field(default=None, max_length=80)
    context_schema_version: str | None = Field(default=None, max_length=80)


class FormulaResolveRequest(GraphContext):
    source_kind: Literal["formula"]
    draft_revision: int = Field(default=0, ge=0)
    expressions: dict[OutputId, Annotated[str, Field(max_length=4000)]] = Field(min_length=1, max_length=8)


class CanvasResolveRequest(GraphContext):
    source_kind: Literal["graph"]
    draft_revision: int = Field(default=0, ge=0)
    graph: AuthoringGraph


GraphResolveRequest = Annotated[FormulaResolveRequest | CanvasResolveRequest, Field(discriminator="source_kind")]


class GraphPosition(GraphModel):
    x: float = Field(ge=-1_000_000, le=1_000_000)
    y: float = Field(ge=-1_000_000, le=1_000_000)


class GraphViewport(GraphPosition):
    zoom: float = Field(ge=0.1, le=2)


class EditorStateUpdate(GraphModel):
    expected_editor_revision: int = Field(ge=0)
    graph: AuthoringGraph
    positions: dict[NodeId, GraphPosition] = Field(default_factory=dict, max_length=136)
    viewport: GraphViewport | None = None

    @model_validator(mode="after")
    def check_position_ids(self):
        ids = {node.id for node in self.graph.nodes} | {f"output_{item.id}" for item in self.graph.outputs}
        if set(self.positions) - ids:
            raise ValueError("布局含有不属于当前画布的节点。")
        return self
