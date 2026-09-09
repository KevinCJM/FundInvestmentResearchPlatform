"""One scoring/row-assembly contract for scalar and named-scalar plans."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from .parallel_engine import score_plan_matrix


def score_result_rows(
    plan: dict[str, Any],
    values_by_target: list[list[dict[str, Any] | None]],
    target_names: list[str],
) -> tuple[list[dict[str, Any]], int, float]:
    """Package the existing fixed-signature NJIT strict-complete scoring kernel.

    Only selected plan items participate. An unselected sibling output is not a
    missing scoring item; a selected missing result is never replaced by zero.
    """
    raw_matrix = np.full((len(plan["targets"]), len(plan["indicators"])), np.nan, dtype=np.float64)
    concrete_values: list[list[dict[str, Any]]] = []
    for row_index, slots in enumerate(values_by_target):
        row_values = []
        for metric_index, value in enumerate(slots):
            if value is None:
                raise RuntimeError("评价方案批量执行未填充完整的指标结果。")
            row_values.append(value)
            if value["value"] is not None:
                raw_matrix[row_index, metric_index] = float(value["value"])
        concrete_values.append(row_values)
    weights = np.asarray([float(item["weight"]) for item in plan["indicators"]], dtype=np.float64)
    lower_better = np.asarray([item["direction"] == "lower_better" for item in plan["indicators"]], dtype=np.int8)
    normalized, contributions, scores, complete_mask, ranks, ranked_indices, effective_weights, total_weight = score_plan_matrix(raw_matrix, weights, lower_better)
    rows = []
    for row_index, target in enumerate(plan["targets"]):
        values = concrete_values[row_index]
        complete = bool(complete_mask[row_index])
        for metric_index, value in enumerate(values):
            value["effective_weight"] = float(effective_weights[metric_index])
            if complete:
                value["normalized_score"] = float(normalized[row_index, metric_index])
                value["weighted_contribution"] = float(contributions[row_index, metric_index])
        rows.append({
            "rank": int(ranks[row_index]) if complete else None,
            "target": {**target, "name": target_names[row_index]},
            "score": round(float(scores[row_index]), 6) if complete else None,
            "status": "ranked" if complete else "excluded",
            "missing_indicators": [value["indicator_name"] for value in values if value["value"] is None],
            "exclusion_reasons": [warning for value in values if value["value"] is None for warning in value.get("warnings", [])],
            "values": values,
        })
    rows.sort(key=lambda row: (row["rank"] is None, row["rank"] or math.inf))
    return rows, int(ranked_indices.size), float(total_weight)
