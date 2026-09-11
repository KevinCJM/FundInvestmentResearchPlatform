"""Storage-neutral graph references and bounded, editor-only layout metadata."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GraphPortRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    node_id: str = Field(min_length=1, max_length=128)
    port: str = Field(default="value", min_length=1, max_length=64)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: GraphPortRef
    target: GraphPortRef


class CanvasPosition(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    x: float = Field(ge=-1_000_000, le=1_000_000)
    y: float = Field(ge=-1_000_000, le=1_000_000)


class CanvasViewport(CanvasPosition):
    zoom: float = Field(ge=0.05, le=4)


class CanvasLayout(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    positions: dict[str, CanvasPosition] = Field(default_factory=dict, max_length=128)
    viewport: CanvasViewport | None = None
