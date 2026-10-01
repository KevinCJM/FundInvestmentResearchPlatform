"""Conservative model-admission proof from the versioned compiler DAG.

This does not evaluate formulas. Unknown operators, selectors and masked reductions
remain unavailable to the model; the unchanged business service/UI still supports them.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from cal_indicators.typed_operators import get_typed_operator_registry
from custom_indicators.variable_registry import get_variable, variable_semantic_role

DERIVED_ROLES = frozenset({"ordinary_return", "log_return", "ordinary_return_matrix",
                          "log_return_matrix", "realized_portfolio_return", "benchmark_return"})
# No selectors or masked reducers: without selected-sample evidence they could
# reduce a large window to exactly one raw observation.
REDUCERS = frozenset({"sum", "mean", "product", "std", "variance", "median", "quantile",
                     "min_value", "max_value", "skewness", "excess_kurtosis",
                     "mean_absolute_deviation", "linear_slope", "linear_r_squared",
                     "correlation", "covariance", "count_true", "length"})
RAW_COUNTS = frozenset({"count_true", "length"})
ARITHMETIC = frozenset({"add", "subtract", "multiply", "divide", "power", "negate", "absolute",
                        "sqrt", "log", "exp", "minimum", "maximum"})
TRANSFORMS = frozenset({"drawdown_series", "difference", "new_high_mask"})
COMPARISONS = frozenset({"greater_than", "greater_equal", "less_than", "less_equal", "equal", "not_equal"})


def _unproven(code: str, *, definition_ref: Optional[str] = None) -> dict[str, Any]:
    return {"status": "unproven", "code": code, "definition_ref": definition_ref}


def definition_proof(service: Any, definition: Any, *, context_kind: str,
                     definition_ref: Optional[str] = None) -> dict[str, Any]:
    if not isinstance(definition, dict):
        return _unproven("DEFINITION_UNAVAILABLE", definition_ref=definition_ref)
    outputs = definition.get("series_outputs") if definition.get("result_kind") == "time_series" else None
    if outputs:
        channels = {item["id"]: _resolve(service, {**definition, "expression": item.get("expression")},
                                        context_kind, definition_ref)
                    for item in outputs if isinstance(item, dict) and isinstance(item.get("id"), str)}
        return {"status": "channels", "channels": channels, "definition_ref": definition_ref,
                "parameters": [item.get("id") for item in definition.get("parameter_schema", [])]}
    return _resolve(service, definition, context_kind, definition_ref)


def _resolve(service, definition, context_kind, definition_ref):
    expression = str(definition.get("expression") or "").strip()
    if not expression:
        return _unproven("EMPTY_EXPRESSION", definition_ref=definition_ref)
    fields = {key: definition[key] for key in ("dsl_version", "operator_registry_version", "parameter_schema")
              if definition.get(key) is not None}
    try:
        inferred = service.infer({**fields, "expression": expression, "context": context_kind})
        registry = get_typed_operator_registry(inferred["operator_registry_version"])
    except Exception:
        return _unproven("DEFINITION_UNRESOLVED", definition_ref=definition_ref)
    dag = inferred.get("dag") or {}
    nodes = {node["id"]: node for node in dag.get("nodes", []) if isinstance(node, dict) and "id" in node}
    root = (dag.get("roots") or {}).get("result")
    parameters = {item.get("id"): item for item in definition.get("parameter_schema", []) if isinstance(item, dict)}
    facts = {}
    aggregates = set()
    minimum_parameters = set()
    blocked_reasons = set()

    def state(index):
        if index in facts:
            return facts[index]
        node = nodes.get(index, {})
        operator = (node.get("operator") or {}).get("id")
        inputs = node.get("inputs") or []
        children = [state(child) for child in inputs]
        kind = (node.get("inferred_type") or {}).get("kind")
        result = "unknown"
        if node.get("kind") == "constant":
            result = "constant"
        elif node.get("kind") == "variable":
            variable = get_variable(str(node.get("label")))
            if node.get("label") in parameters:
                result = "constant"
            elif variable is not None:
                result = ("constant" if variable.kind == "scalar" and variable.source_field is None
                          else "derived" if variable_semantic_role(variable.variable_id) in DERIVED_ROLES
                          else "raw")
        elif operator in registry and children and "unknown" not in children:
            if operator in ARITHMETIC:
                result = ("unknown" if "raw" in children or "mask" in children else "derived" if "derived" in children
                          else "aggregate" if "aggregate" in children else "constant")
            elif operator in TRANSFORMS and children[0] in {"raw", "derived"}:
                result = "derived"
            elif operator in COMPARISONS:
                result = "mask"
            elif operator in REDUCERS and kind == "scalar" and any(c in {"raw", "derived", "mask"} for c in children):
                if "mask" in children and operator not in {"count_true", "length"}:
                    facts[index] = "unknown"
                    return "unknown"
                # The result receipt's observation_count/window_rows are axis
                # lengths, not finite input counts. Without that proof even
                # mean(volume) could pass one remaining raw observation through.
                if not ("raw" in children and operator not in RAW_COUNTS):
                    result = "aggregate"
                    aggregates.add(operator)
                else:
                    blocked_reasons.add("FINITE_RAW_SAMPLE_COUNT_UNPROVEN")
            elif operator == "rolling_apply" and children[0] == "aggregate":
                window = nodes.get(inputs[1], {}) if len(inputs) > 1 else {}
                if window.get("kind") == "constant":
                    try:
                        valid = float(window.get("label")) >= 2
                    except (TypeError, ValueError):
                        valid = False
                else:
                    name = window.get("label")
                    valid = name in parameters
                    if valid:
                        minimum_parameters.add(name)
                if valid:
                    if len(inputs) in {3, 5}:
                        minimum = nodes.get(inputs[-1], {})
                        if minimum.get("kind") == "constant":
                            try:
                                valid = float(minimum.get("label")) >= 2
                            except (TypeError, ValueError):
                                valid = False
                        else:
                            name = minimum.get("label")
                            valid = name in parameters
                            if valid:
                                minimum_parameters.add(name)
                if valid:
                    result = "derived"
        facts[index] = result
        return result

    resolved = state(root)
    shape = inferred.get("shape")
    if resolved not in {"aggregate", "derived"} or (shape == "scalar" and resolved != "aggregate"):
        return _unproven("FINITE_RAW_SAMPLE_COUNT_UNPROVEN" if blocked_reasons else
                         "SINGLE_OBSERVATION_OR_IDENTITY" if shape == "scalar" else "NOT_A_DERIVED_RESULT",
                         definition_ref=definition_ref)
    return {"status": "approved", "code": "registered_derivation", "shape": shape,
            "aggregates": sorted(aggregates), "dependencies": inferred.get("dependencies", []),
            "registry_version": inferred.get("operator_registry_version"), "definition_ref": definition_ref,
            "parameters": list(parameters), "minimum_parameters": sorted(minimum_parameters)}


def enough_samples(row: dict, proof: Optional[dict]) -> bool:
    """Check service axis cardinality and runtime parameters, not finite counts.

    Raw-valued reducers are rejected structurally above. Derived-channel finite
    counts are computed separately from its actual result values by the NJIT
    summary; neither observation_count nor coverage.window_rows proves them.
    """
    if not proof or proof.get("status") != "approved":
        return False
    count = (row.get("window") or {}).get("observation_count")
    if not isinstance(count, int) or isinstance(count, bool) or count < 2:
        return False
    for name in proof.get("minimum_parameters", []):
        value = (row.get("parameters") or {}).get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 2:
            return False
    return True


def saved_definition_proof(service: Any, indicator_id: str, revision: Optional[int]) -> dict[str, Any]:
    reference = f"saved:{indicator_id}@{revision}" if revision else f"saved:{indicator_id}"
    try:
        definition = service.get_indicator(indicator_id, revision)
    except Exception:
        return _unproven("SAVED_DEFINITION_NOT_FOUND", definition_ref=reference)
    return definition_proof(service, definition, context_kind=str(definition.get("context_kind") or "single_product"),
                            definition_ref=reference)


def draft_definition_ref(definition: Any) -> str:
    digest = hashlib.sha256(json.dumps(definition, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"inline:{digest}"
