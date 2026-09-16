"""Offline tests for the editable CSI300 reference-recognition baseline."""
import calendar
import copy
from datetime import date

import numpy as np
import pytest

from historical_regimes.v2_contracts import parse_definition_v2
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_templates import instantiate_template_v2
from historical_regimes.temporal_audit import audit_execution
from test_historical_regime_v2_p1 import _upload_source

TEMPLATE = "csi300-maintrend-sma9-realtime-v2"
CURRENT_TEMPLATE = "csi300-maintrend-sma9-realtime-v3"
LEGACY_TEMPLATE = "csi300-maintrend-sma10-realtime-v1"


def test_template_is_composed_causal_and_keeps_reference_unchanged():
    reference = instantiate_template_v2("market-trend-reference-csi300-v1")
    before = copy.deepcopy(reference)
    draft = instantiate_template_v2(TEMPLATE)
    model = parse_definition_v2(draft)
    assert model.default_mode == "realtime"
    assert model.study.reference is None
    assert {node.type for node in model.graph.nodes} == {
        "source.index", "align.resample", "filter.sma", "math.divide",
        "source.constant", "math.subtract", "model.range_threshold",
    }
    assert next(n for n in model.graph.nodes if n.id == "average").parameters["window"] == 9
    assert next(n for n in model.graph.nodes if n.id == "upper").parameters["value"] == .04
    assert next(n for n in model.graph.nodes if n.id == "lower").parameters["value"] == -.04
    legacy = instantiate_template_v2(LEGACY_TEMPLATE)
    assert next(n for n in legacy["graph"]["nodes"] if n["id"] == "average")["parameters"]["window"] == 10
    assert next(n for n in legacy["graph"]["nodes"] if n["id"] == "upper")["parameters"]["value"] == .05
    assert legacy["description"] == (
        "闭合月末价格相对10月单边均线高于5%为牛、低于-5%为熊，其余为震荡。"
        "识别沪深300主趋势事后参考，不使用未来峰谷；需选择精确历史参考后运行校验。"
        "固定参数研究基线，不预先保证通过；规则命中不代表100%可信。"
    )
    assert {s.id for s in model.states} == {"bull", "bear", "sideways"}
    assert reference == before


def test_new_realtime_identity_keeps_math_but_drops_ltcma_authority():
    legacy = instantiate_template_v2(TEMPLATE)
    current = instantiate_template_v2(CURRENT_TEMPLATE)
    assert current["usage_intent"] == "taa"
    assert "LTCMA" in current["description"] and "不作为" in current["description"]
    assert "CMA研究" not in current["description"]
    for node_id in ("average", "upper", "lower"):
        left = next(node for node in legacy["graph"]["nodes"] if node["id"] == node_id)
        right = next(node for node in current["graph"]["nodes"] if node["id"] == node_id)
        assert left["type"] == right["type"]
        assert left["parameters"] == right["parameters"]


def test_bounded_candidates_are_real_editable_graphs_without_reference_inputs(tmp_path):
    from backend.scripts.csi300_recognition_search import candidates
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    choices = candidates()
    assert len(choices) == 72 and len({key for key, _ in choices}) == 72
    for key, draft in choices:
        model = parse_definition_v2(draft)
        inference = graph.infer(draft, "realtime")
        assert inference["valid"], (key, inference.get("errors"))
        graph._validate_realtime_graph(model, "realtime")
        assert model.study.reference is None
        assert not any(n.type.startswith(("pivot.", "segment.")) for n in model.graph.nodes)
        assert {n["id"] for n in draft["graph"]["nodes"]} == set(draft["graph"]["exposed_node_ids"])
        if "average" in draft["graph"]["channel_metadata"]:
            average = next(n for n in model.graph.nodes if n.id == "average")
            assert str(average.parameters["window"]) in draft["graph"]["channel_metadata"]["average"]["label"]


def test_structural_literal_not_perturbed_but_tunable_one_is(tmp_path):
    from historical_regimes.reliability.contracts import StabilityPolicy
    from historical_regimes.reliability.diagnostics import variants
    from historical_regimes.v2_contracts import validate_definition_v2
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    draft = instantiate_template_v2(TEMPLATE)
    definition = parse_definition_v2(draft)
    policy = StabilityPolicy(max_variants=12)
    changed = [d for _, entry in variants(graph, definition, policy) for d in entry["changes"]]
    assert not any(d["node_id"] == "one" for d in changed)
    assert {"upper", "lower"} <= {d["node_id"] for d in changed}
    assert next(n for n in definition.graph.nodes if n.id == "one").parameters["value"] == 1.
    # A literal value of 1 is NOT automatically structural: explicit role only.
    node = next(n for n in draft["graph"]["nodes"] if n["id"] == "one")
    node["parameters"].pop("parameter_role")
    legacy = parse_definition_v2(draft)
    changed = [d for _, entry in variants(graph, legacy, policy) for d in entry["changes"]]
    assert any(d["node_id"] == "one" for d in changed)
    assert "parameter_role" not in next(n for n in legacy.model_dump(mode="json")["graph"]["nodes"] if n["id"] == "one")["parameters"]
    node["parameters"]["parameter_role"] = "made-up-role"
    with pytest.raises(Exception):
        validate_definition_v2(parse_definition_v2(draft))


@pytest.fixture
def executed(tmp_path):
    graph = RegimeGraphV2Service(tmp_path, tmp_path)
    values = np.array([100.] * 12 + [110., 120., 130., 140., 150., 160.] +
                      [150., 135., 120., 105., 90., 80.] + [80.] * 12)
    rows = []
    for i, value in enumerate(values):
        year, month = 2010 + i // 12, i % 12 + 1
        day = date(year, month, calendar.monthrange(year, month)[1]).isoformat()
        rows.append({"observation_date": day, "available_at": day, "value": float(value)})
    rows.append({"observation_date": "2013-01-01", "available_at": "2013-01-01", "value": 80.})
    draft = instantiate_template_v2(TEMPLATE)
    node = _upload_source(graph, rows)
    node["id"] = "market"
    draft["graph"]["nodes"][0] = node
    draft["evaluation_targets"] = []
    model = parse_definition_v2(graph.create_definition(draft))
    plan = graph.prepare(model.model_dump(mode="json"))
    result = graph._execute_graph(None, model, "realtime", "2013-01-01", plan=plan)
    return graph, model, plan, result, values


def test_real_executor_matches_independent_sma_math(executed):
    _, _, _, result, values = executed
    series = result["series"]
    assert len(series) == 36
    for i, row in enumerate(series):
        if i < 8:
            assert row["state_id"] == "unclassified"
        else:
            distance = values[i] / values[i-8:i+1].mean() - 1.
            expected = "bull" if distance > .04 else "bear" if distance < -.04 else "sideways"
            assert row["state_id"] == expected
    assert {row["state_id"] for row in series} == {"unclassified", "bull", "bear", "sideways"}
    assert result["result"]["diagnostics"]["python_fallback"] == 0


def test_parameter_role_does_not_change_numerical_output(executed):
    graph, model, _, result, _ = executed
    old = model.model_dump(mode="json")
    next(n for n in old["graph"]["nodes"] if n["id"] == "one")["parameters"].pop("parameter_role")
    old_model = parse_definition_v2(old)
    plan = graph.prepare(old)
    old_result = graph._execute_graph(None, old_model, "realtime", "2013-01-01", plan=plan)
    assert old_result["series"] == result["series"]
    for node in ("monthly", "average", "relative", "one", "distance"):
        np.testing.assert_array_equal(old_result["node_outputs"][node]["value"].values,
                                      result["node_outputs"][node]["value"].values)


def test_prefix_no_repainting_and_no_incomplete_month(executed):
    graph, model, plan, result, _ = executed
    shorter = graph._execute_graph(None, model, "realtime", "2012-06-15", plan=plan)
    assert shorter["series"][-1]["observation_date"] == "2012-05-31"
    end = len(shorter["series"])
    assert [p["state_id"] for p in shorter["series"]] == [p["state_id"] for p in result["series"][:end]]
    audit = audit_execution(graph, model, "realtime", "2013-01-01", plan, result)
    assert audit["verified"] and audit["realtime_supported"]
