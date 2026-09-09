"""Dependency checks extracted from the historical-regime graph validator."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, Literal


@dataclass(frozen=True)
class Topology:
    order: list[str]
    missing: list[tuple[str, str]]
    duplicates: list[str]
    cyclic: bool


def dependency_order(node_ids: Iterable[str], edges: Iterable[tuple[str, str]], *,
                     tie_break: Literal["fifo", "node_order"] = "fifo") -> Topology:
    """Edges are (source, target). FIFO preserves the regime compiler's order.

    node_order is used for side-effecting ETL: when several nodes are ready,
    preserve the user's original list order. Parallel edges are counted, not
    silently discarded. The caller applies domain-specific diagnostics/limits.
    """
    ids = list(node_ids)
    unique = list(dict.fromkeys(ids))
    seen: set[str] = set()
    duplicates = []
    for identifier in ids:
        if identifier in seen:
            duplicates.append(identifier)
        seen.add(identifier)
    indegree = dict.fromkeys(unique, 0)
    consumers: dict[str, list[str]] = {key: [] for key in unique}
    missing = []
    for source, target in edges:
        if source not in indegree or target not in indegree:
            missing.append((source, target))
            continue
        indegree[target] += 1
        consumers[source].append(target)
    ranks = {key: index for index, key in enumerate(unique)}
    queue = deque(key for key in unique if indegree[key] == 0)
    heap = [ranks[key] for key in queue]
    order = []
    while heap if tie_break == "node_order" else queue:
        node_id = unique[heappop(heap)] if tie_break == "node_order" else queue.popleft()
        order.append(node_id)
        for consumer in consumers[node_id]:
            indegree[consumer] -= 1
            if indegree[consumer] == 0:
                if tie_break == "node_order":
                    heappush(heap, ranks[consumer])
                else:
                    queue.append(consumer)
    return Topology(order, missing, duplicates, len(order) != len(unique))
