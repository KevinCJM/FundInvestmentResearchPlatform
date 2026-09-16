"""Shared bounded variants using the sole graph executor and frozen causal folds."""

from contextlib import contextmanager
import time
import numpy as np
from custom_indicators.errors import ValidationError, IndicatorDomainError
from ..v2_contracts import inspect_definition_v2, definition_content_hash
from ..v2_registry import NODE_REGISTRY
from ..v2_contracts import validate_definition_v2
from ..v2_service import _required_node_ids
from .diagnostic_kernels import compare_kernel, integer_step_kernel
from .execution import LATENT
from .report import axis, finite
from .kernels import align_kernel

LIMITS = {
    "observations": 20000,
    "nodes": 64,
    "models": 4,
    "iterations": 100,
    "seconds": 120,
    "numeric_work": 100000000,
}
WINDOWS = {
    "window",
    "period",
    "periods",
    "confirmation",
    "min_duration",
    "left_window",
    "right_window",
    "head_window",
    "tail_window",
    "min_phase",
    "min_cycle",
    "sideways_min_duration",
}
# Same stable perturbation implementation is shared with existing experiments.
PARAMETERS = (
    "upper",
    "lower",
    "upper_enter",
    "lower_enter",
    "threshold",
    "min_move",
    "left_window",
    "right_window",
    "min_phase",
    "min_cycle",
    "head_window",
    "tail_window",
    "amplitude_exception",
    "small_swing_threshold",
    "sideways_max_range",
    "sideways_max_efficiency",
    "sideways_min_duration",
    "window",
    "period",
    "periods",
    "confirmation",
    "min_duration",
    "process_variance",
    "measurement_variance",
    "value",
)


EXPERIMENT_PARAMETERS = (
    "upper",
    "lower",
    "upper_enter",
    "lower_enter",
    "threshold",
    "min_move",
    "window",
    "periods",
    "confirmation",
    "min_duration",
    "process_variance",
    "measurement_variance",
)


def active_parameter(node, name, schema):
    """Keep explicit structural literals fixed; absent roles preserve old policy."""
    if schema.get("deprecated"):
        return False
    if node.type.startswith("source.") and node.type != "source.constant":
        return False
    if name == "value" and node.type != "source.constant":
        return False
    if node.type == "source.constant" and node.parameters.get("parameter_role", "tunable") == "structural":
        return False
    if node.type == "model.range_threshold" and name in {"upper", "lower"}:
        return name + "_bound" not in node.inputs
    if node.type == "condition.compare" and name == "threshold":
        return "bound" not in node.inputs
    return True


def changed_parameter(definition, index, name, value):
    """Copy only the edited parameter dictionary; shared source data stays readonly."""
    nodes = list(definition.graph.nodes)
    node = nodes[index]
    nodes[index] = node.model_copy(
        update={"parameters": {**node.parameters, name: value}}
    )
    return definition.model_copy(
        update={"graph": definition.graph.model_copy(update={"nodes": nodes})}
    )


def perturb_definitions(
    definition,
    perturbation,
    maximum_candidates,
    *,
    minimum_integer_step=False,
    validate_candidates=True
):
    from ..numba_kernels import perturb_scalar_kernel

    if perturbation <= 0:
        return []
    candidates = []
    for index, node in enumerate(definition.graph.nodes):
        properties = (
            NODE_REGISTRY[node.type].get("parameter_schema", {}).get("properties", {})
        )
        names = PARAMETERS if minimum_integer_step else EXPERIMENT_PARAMETERS
        for name in names:
            if name not in node.parameters and not (
                minimum_integer_step and "default" in properties.get(name, {})
            ):
                continue
            if minimum_integer_step and (
                name not in properties
                or not active_parameter(node, name, properties[name])
            ):
                continue
            value = node.parameters.get(name, properties.get(name, {}).get("default"))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            schema = properties.get(name, {})
            integer = schema.get("type") == "integer"
            changed = float(
                perturb_scalar_kernel(
                    np.float64(value),
                    np.float64(perturbation),
                    np.float64(schema.get("minimum", -1e300)),
                    np.uint8(integer),
                )
            )
            if integer and minimum_integer_step:
                changed = float(integer_step_kernel(float(value), changed))
            if changed == value or changed > schema.get("maximum", float("inf")):
                continue
            parsed = changed_parameter(
                definition, index, name, int(changed) if integer else changed
            )
            if validate_candidates:
                validate_definition_v2(parsed)
            candidates.append(
                (
                    parsed,
                    {
                        "node_id": node.id,
                        "parameter": name,
                        "base_value": value,
                        "candidate_value": int(changed) if integer else changed,
                        "perturbation": perturbation,
                    },
                )
            )
            if len(candidates) >= maximum_candidates:
                return candidates
    return candidates


def budget_definition(definition):
    required = _required_node_ids(definition)
    nodes = [n for n in definition.graph.nodes if n.id in required]
    latent = [n for n in nodes if n.type in LATENT]
    if (
        not 1 <= int(definition.validation.get("folds", 4)) <= 8
        or len(nodes) > LIMITS["nodes"]
        or len(latent) > LIMITS["models"]
        or any(
            n.parameters.get("iterations", 60) > LIMITS["iterations"] for n in latent
        )
        or len(definition.evaluation_targets) > 8
    ):
        raise ValidationError(
            "RELIABILITY_BUDGET", "诊断超过节点、模型、评价对象或迭代预算。"
        )
    return required


def check_inputs(graph, definition, mode, cutoff, cache):
    required = budget_definition(definition)
    validate_definition_v2(definition)
    bundles, _ = graph._resolve_sources(definition, required, mode, cutoff, cache)
    if any(len(b.frame) > LIMITS["observations"] for b in bundles.values()):
        raise ValidationError("RELIABILITY_BUDGET", "每条输入最多20000个观测。")
    observations = max((len(b.frame) for b in bundles.values()), default=0)
    multiplier = (
        int(definition.validation.get("folds", 4)) + 6 if mode == "realtime" else 1
    )
    if estimated_work(definition, observations, multiplier) > LIMITS["numeric_work"]:
        raise ValidationError("RELIABILITY_BUDGET", "诊断超过有界数值工作量预算。")


def estimated_work(definition, observations, multiplier=1):
    # Conservative admission accounting on dimensions, not financial computation.
    nodes = [
        n for n in definition.graph.nodes if n.id in _required_node_ids(definition)
    ]
    features = max(
        (len(n.inputs) for n in nodes if n.type == "feature.matrix"), default=1
    )
    model_cost = sum(
        int(n.parameters.get("iterations", 60))
        * int(n.parameters.get("components", 3)) ** 2
        * features
        for n in nodes
        if n.type in LATENT
    )
    return observations * (len(nodes) + model_cost) * multiplier


@contextmanager
def prepared_plan(graph, definition, compile_token=None):
    if compile_token:
        yield graph._validate_plan(definition, compile_token)
        return
    preparation_hash = graph._preparation_hash(definition)
    with graph._lock:
        plan = graph.prepare(definition.model_dump(mode="json"), persist_manifest=False)
        token = plan["compile_token"]
        record = graph._plans[token]
        record["diagnostic_leases"] = record.get("diagnostic_leases", 0) + 1
    try:
        yield plan
    finally:
        with graph._lock:
            record = graph._plans.get(token)
            if record is not None:
                record["diagnostic_leases"] -= 1
                if not record["diagnostic_leases"] and not record.get(
                    "persistent", True
                ):
                    graph._plans.pop(token, None)
                    if graph._plans_by_graph_hash.get(preparation_hash) == token:
                        graph._plans_by_graph_hash.pop(preparation_hash, None)


def compare_points(base, candidate, states):
    dates, bd = axis(base)
    _, cd = axis(candidate)
    indices = align_kernel(bd, cd)
    codes = {s.id: i for i, s in enumerate(states)}
    left = np.asarray([codes.get(p.get("state_id"), -1) for p in base], np.int64)
    right = np.asarray(
        [
            codes.get(candidate[i].get("state_id"), -1) if i >= 0 else -1
            for i in indices
        ],
        np.int64,
    )
    numbers = compare_kernel(left, right, len(states))
    return {
        "agreement": finite(numbers[1]),
        "comparable_observations": int(numbers[0]),
        "classification_coverage": finite(numbers[2]),
        "boundary_distance": finite(numbers[3]),
        "boundary_matched_events": int(numbers[4]),
        "comparison_observations": len(dates),
        "boundary_distance_unit": "observation_steps",
        "agreement_reason": None if numbers[0] else "no_paired_classified_observations",
        "boundary_distance_reason": (
            None if numbers[4] else "no_matching_ordered_boundaries"
        ),
    }


def variants(graph, definition, policy):
    # Read defaults from metadata; share every unchanged source/node with baseline.
    required = _required_node_ids(definition)
    groups = {"parameter": [], "window": [], "seed": [], "truncation": []}
    for candidate, desc in perturb_definitions(
        definition,
        policy.perturbation,
        256,
        minimum_integer_step=True,
        validate_candidates=False,
    ):
        if desc["node_id"] not in required:
            continue
        node = next(n for n in candidate.graph.nodes if n.id == desc["node_id"])
        if node.type.startswith("source.") and node.type != "source.constant":
            continue
        kind = "window" if desc["parameter"] in WINDOWS else "parameter"
        if not (policy.windows if kind == "window" else policy.parameters):
            continue
        groups[kind].append(
            (
                candidate,
                {
                    "kind": kind,
                    "changes": [
                        {
                            "node_id": desc["node_id"],
                            "parameter": desc["parameter"],
                            "before": desc["base_value"],
                            "after": desc["candidate_value"],
                        }
                    ],
                },
            )
        )
    if policy.seeds:
        for index, node in enumerate(definition.graph.nodes):
            if (
                node.id not in required
                or node.type not in LATENT
                or node.parameters.get("initialization_strategy", "quantile")
                != "random"
            ):
                continue
            for seed in (17, 43):
                if node.parameters.get("random_seed", 0) == seed:
                    continue
                changed = changed_parameter(definition, index, "random_seed", seed)
                groups["seed"].append(
                    (
                        changed,
                        {
                            "kind": "seed",
                            "changes": [
                                {
                                    "node_id": node.id,
                                    "parameter": "random_seed",
                                    "before": node.parameters.get("random_seed", 0),
                                    "after": seed,
                                }
                            ],
                        },
                    )
                )
    if policy.truncation:
        groups["truncation"].append((definition, {"kind": "truncation", "changes": []}))
    # Round robin reserves representation for each enabled, applicable family.
    selected = []
    for i in range(policy.max_variants):
        for group in groups.values():
            if i < len(group):
                selected.append(group[i])
                if len(selected) == policy.max_variants:
                    return selected
    return selected


def run_diagnostics(
    graph,
    definition,
    policy,
    baseline,
    mode,
    cutoff,
    *,
    cache=None,
    request=None,
    reference=None,
    folds=None,
    started=None
):
    from .execution import replay

    started = time.monotonic() if started is None else started
    cache = {} if cache is None else cache
    applicable_seeds = any(
        n.type in LATENT
        and n.parameters.get("initialization_strategy", "quantile") == "random"
        for n in definition.graph.nodes
        if n.id in _required_node_ids(definition)
    )
    result = {
        "status": "disabled",
        "variants": [],
        "seed_status": (
            "disabled"
            if not policy.seeds
            else "pending" if applicable_seeds else "not_applicable"
        ),
        "limits": {**LIMITS, "max_variants": policy.max_variants},
        "fixed_structural_parameters": [
            {"node_id": n.id, "parameter": "value", "value": n.parameters.get("value", 0.0),
             "reason": "explicit_structural_role"}
            for n in definition.graph.nodes
            if n.type == "source.constant" and n.parameters.get("parameter_role") == "structural"
            and n.id in _required_node_ids(definition)
        ],
        "reason": "disabled_by_policy",
    }
    if not policy.enabled:
        result["seed_status"] = "disabled"
        return result
    multiplier = len(folds or []) + 6 if mode == "realtime" else 1
    work = estimated_work(definition, len(baseline), multiplier)
    for candidate, descriptor in variants(graph, definition, policy):
        entry = {
            **descriptor,
            "status": "failed",
            "reason": None,
            "agreement": None,
            "comparable_observations": 0,
            "classification_coverage": None,
            "boundary_distance": None,
            "boundary_distance_unit": "observation_steps",
            "graph_hash": inspect_definition_v2(candidate)["graph_hash"],
            "definition_hash": definition_content_hash(candidate),
        }
        result["variants"].append(entry)
        candidate_cutoff = cutoff
        compared = baseline
        if entry["kind"] == "truncation":
            kept = len(baseline) * 9 // 10
            if kept < 5:
                entry["reason"] = "insufficient_prefix"
                continue
            compared = baseline[:kept]
            candidate_cutoff = compared[-1]["data_available_at"]
            entry.update(retained_observations=kept, as_of=candidate_cutoff)
        work += estimated_work(candidate, len(baseline), multiplier)
        if (
            time.monotonic() - started > LIMITS["seconds"]
            or work > LIMITS["numeric_work"]
        ):
            entry.update(status="budget_exceeded", reason="diagnostic_budget_exceeded")
            continue
        try:
            check_inputs(graph, candidate, mode, candidate_cutoff, cache)
            with prepared_plan(graph, candidate) as plan:
                if mode == "realtime":
                    variant_request = request.model_copy(
                        update={"compile_token": plan["compile_token"]}
                    )
                    points, lineage, _ = replay(
                        graph,
                        candidate,
                        variant_request,
                        reference,
                        source_cache=cache,
                        frozen_folds=folds,
                        cutoff_override=candidate_cutoff,
                    )
                    entry["prediction_method"] = lineage["prediction_method"]
                    entry["folds"] = lineage["folds"]
                    entry["data_snapshots"] = lineage["data_snapshots"]
                else:
                    execution = graph._execute_graph(
                        None,
                        candidate,
                        mode,
                        candidate_cutoff,
                        plan=plan,
                        source_cache=cache,
                    )
                    points = execution["series"]
                    entry["data_snapshots"] = execution["result"]["data_snapshots"]
            if time.monotonic() - started > LIMITS["seconds"]:
                entry.update(
                    status="budget_exceeded", reason="diagnostic_time_budget_exceeded"
                )
                continue
            entry.update(
                compare_points(compared, points, definition.states), status="completed"
            )
        except (IndicatorDomainError, ValueError) as exc:
            entry["reason"] = getattr(exc, "code", "invalid_variant")
    rows = result["variants"]
    result["status"] = (
        "not_applicable"
        if not rows
        else "completed" if all(r["status"] == "completed" for r in rows) else "partial"
    )
    result["reason"] = "no_applicable_variants" if not rows else None
    if applicable_seeds and policy.seeds:
        seeds = [r for r in rows if r["kind"] == "seed"]
        result["seed_status"] = (
            "completed"
            if seeds and all(r["status"] == "completed" for r in seeds)
            else "not_executed"
        )
    return result
