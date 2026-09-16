import copy
import numpy as np
import pytest
from historical_regimes.reliability import kernels
from historical_regimes.reliability.diagnostics import variants, compare_points
from historical_regimes.reliability.diagnostic_kernels import compare_kernel
from historical_regimes.reliability.contracts import StabilityPolicy
from historical_regimes.v2_contracts import parse_definition_v2
from test_regime_reliability_service import case
from test_historical_regime_v2 import _definition


def test_comparison_oracle_semantic_axis_and_boundaries():
    kernels.warm()
    left = np.array([0, 0, 1, 1, 2, 2, 0, -1, 0, 1, 2], np.int64)
    right = np.array([0, 1, 1, 2, 2, 2, 0, -1, 0, 0, 1], np.int64)
    paired = (left >= 0) & (right >= 0)
    numbers = compare_kernel(left, right, 3)
    assert numbers[1] == np.mean(left[paired] == right[paired])
    events = lambda a: [
        (i, (a[i - 1], a[i]))
        for i in range(1, len(a))
        if a[i] >= 0 and a[i - 1] >= 0 and a[i] != a[i - 1]
    ]
    distances = []
    for a, b in ((left, right), (right, left)):
        for i, pair in events(a):
            matches = [abs(i - j) for j, other in events(b) if other == pair]
            if matches:
                distances.append(min(matches))
    assert numbers[3] == np.mean(distances)
    assert compare_kernel(left, np.where(left >= 0, (left + 1) % 3, -1), 3)[1] == 0
    view = np.repeat(left, 2)[::2]
    view.flags.writeable = False
    np.testing.assert_allclose(compare_kernel(view, right, 3), numbers)
    with pytest.raises(ValueError):
        compare_kernel(left, right[:2], 3)


def test_variants_do_not_change_sources_or_states(case):
    graph, model, _, _ = case
    definition = parse_definition_v2(model)
    selected = variants(graph, definition, StabilityPolicy())
    assert {d["kind"] for _, d in selected} >= {"parameter", "window", "truncation"}
    for candidate, desc in selected:
        assert candidate.states == definition.states
        assert candidate.graph.nodes[0] is definition.graph.nodes[0]
        assert len(desc["changes"]) <= 1
    assert len(selected) <= 6


def test_reliability_stability_real_plan_and_frozen_folds(case, monkeypatch):
    graph, model, ref, request = case
    model["graph"] = {
        "nodes": [
            model["graph"]["nodes"][0],
            {
                "id": "features",
                "type": "feature.matrix",
                "inputs": {"feature_1": {"node_id": "source", "port": "value"}},
            },
            {
                "id": "model",
                "type": "model.gmm",
                "parameters": {
                    "components": 3,
                    "initial_train_size": 30,
                    "iterations": 4,
                    "initialization_strategy": "random",
                    "random_seed": 17,
                },
                "inputs": {"features": {"node_id": "features", "port": "features"}},
            },
        ],
        "outputs": {
            "state": {"node_id": "model", "port": "state"},
            "probabilities": {"node_id": "model", "port": "probabilities"},
        },
    }
    saved = graph.update_definition(model["id"], 1, model)
    graph.prepare(saved)
    observed = []
    execute = graph._execute_graph

    def wrapped(job, definition, mode, as_of, **kwargs):
        plan = kwargs["plan"]
        assert plan["preparation_hash"] == graph._preparation_hash(definition)
        observed.append(kwargs.get("latent_training_as_of"))
        return execute(job, definition, mode, as_of, **kwargs)

    monkeypatch.setattr(graph, "_execute_graph", wrapped)
    preview = graph.reliability.preview({**request, "revision": 2})
    stability = preview["report"]["stability"]["parameter_sensitivity"]
    assert stability["status"] == "completed"
    assert stability["seed_status"] == "completed"
    baseline = preview["report"]["lineage"]["folds"]
    for variant in stability["variants"]:
        assert variant["prediction_method"] == "expanding_walk_forward"
        for fold in variant["folds"]:
            original = next(
                f for f in baseline if f["training_as_of"] == fold["training_as_of"]
            )
            assert fold["train_end"] == original["train_end"]
            assert fold["test_end"] == original["test_end"]
            assert fold["test_as_of"] <= original["test_as_of"]
            assert fold["model_audits"]["model"]["walk_forward_refit"]
    assert any(observed)
    assert any(
        p["probability_evidence"] is not None for p in preview["report"]["points"]
    )


def test_disabling_diagnostics_skips_variant_execution(case):
    graph, _, _, request = case
    request["policy"]["stability"] = {"enabled": False}
    result = graph.reliability.preview(request)
    assert (
        result["report"]["stability"]["parameter_sensitivity"]["status"] == "disabled"
    )
    assert all(p["probability_evidence"] is None for p in result["report"]["points"])


def test_variant_failure_is_traced_without_discarding_baseline(case):
    graph, model, _, request = case
    node = next(n for n in model["graph"]["nodes"] if n["type"] == "model.threshold")
    node["parameters"].update(upper=0.01, lower=0.0099)
    saved = graph.update_definition(model["id"], 1, model)
    graph.prepare(saved)
    request["revision"] = 2
    out = graph.reliability.preview(request)["report"]["stability"][
        "parameter_sensitivity"
    ]
    assert out["status"] == "partial"
    bad = next(
        v
        for v in out["variants"]
        if any(c["parameter"] == "lower" for c in v["changes"])
    )
    assert bad["status"] == "failed"
    assert bad["agreement"] is None and bad["reason"]


def test_budget_checked_before_graph_execution(case):
    from historical_regimes.reliability.diagnostics import budget_definition

    graph, model, _, _ = case
    definition = parse_definition_v2(model)
    definition.validation["folds"] = 1000000
    with pytest.raises(Exception, match="预算"):
        budget_definition(definition)


def test_existing_experiment_candidates_get_matching_plans(case, monkeypatch):
    graph, model, _, _ = case
    model = copy.deepcopy(model)
    next(n for n in model["graph"]["nodes"] if n["type"] == "filter.ema")["parameters"][
        "window"
    ] = 20
    parsed = parse_definition_v2(model)
    baseline_plan = graph.prepare(model)
    execute = graph._execute_graph
    observed_definitions = set()
    from historical_regimes.v2_contracts import definition_content_hash

    def checked(job, definition, mode, as_of, **kwargs):
        plan = kwargs["plan"]
        assert plan["preparation_hash"] == graph._preparation_hash(definition)
        observed_definitions.add(definition_content_hash(definition))
        return execute(job, definition, mode, as_of, **kwargs)

    monkeypatch.setattr(graph, "_execute_graph", checked)
    execution = checked(None, parsed, "realtime", "2020-08-27", plan=baseline_plan)
    graph._validation_reports(
        parsed, "realtime", "2020-08-27", baseline_plan, execution["series"]
    )
    assert len(observed_definitions) > 1
