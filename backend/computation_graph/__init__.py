"""Domain-neutral graph contracts and dependency analysis. No execution or I/O."""
from .contracts import CanvasLayout, GraphEdge, GraphPortRef
from .topology import Topology, dependency_order

__all__ = ["CanvasLayout", "GraphEdge", "GraphPortRef", "Topology", "dependency_order"]
