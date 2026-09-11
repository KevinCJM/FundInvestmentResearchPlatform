"""Real, editable ETF adaptations: graph, parameter search and causal features."""
import copy
from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

from backend.timing_research.catalog import build_catalog
from backend.timing_research.contracts import Definition
from backend.timing_research.etf_templates import etf_templates
from backend.timing_research.graph import GraphRuntime


def _bars(size=600):
    dates = np.arange(np.datetime64("2020-01-01", "D").astype(np.int64),
                      np.datetime64("2020-01-01", "D").astype(np.int64) + size)
    close = 100 + 15 * np.sin(np.arange(size) / 15) + np.arange(size) / 100
    values = {"dates": dates, "close": close, "open": close * 1.001,
              "high": close * 1.02, "low": close * 0.98,
              "volume": 1000 + 20 * np.cos(np.arange(size) / 11)}
    for value in values.values():
        value.setflags(write=False)
    return SimpleNamespace(**values)


def _templates():
    return {item["definition"]["adaptation"]["source_experiments"][0]: item for item in etf_templates()}


def test_each_call_returns_independent_editable_drafts():
    left, right = etf_templates(), etf_templates()
    assert left == right
    left[0]["definition"]["nodes"][0]["label"] = "changed"
    assert left != right and etf_templates() == right


def test_exact_requested_source_inventory_is_complete_without_duplicates():
    expected = {"A2552", "A160S-106", "A160-MOMO-141", "A2536-Momo-095", "A2536-Momo-096",
                "A2536-Momo-053", "ActionLearner-001", "A2067", "A2074", "A2076", "A2140",
                "A2143", "A2276", "A2296"}
    assert set(_templates()) == expected
    assert len(etf_templates()) == len(expected)
    assert len({item["id"] for item in etf_templates()}) == len(expected)


@pytest.mark.parametrize("item", etf_templates(), ids=lambda value: value["id"])
def test_template_contract_and_real_references(item):
    definition = Definition.model_validate(item["definition"])
    nodes = {node.id: node for node in definition.nodes}
    assert len(nodes) == len(definition.nodes)
    assert len(nodes) <= 128
    for node in nodes.values():
        assert node.op in build_catalog()
        for reference in node.inputs.values():
            assert reference.split(".")[0] in nodes
    assert definition.adaptation.version == "etf-v1"
    assert definition.adaptation.preserved and definition.adaptation.changed
    assert "ETF" in definition.name
    if definition.training:
        count = sum(action.entry is not None for action in definition.training.actions)
        count *= prod(len(axis.choices) for axis in definition.training.search_space)
        assert count <= 108
        for axis in definition.training.search_space:
            for choice in axis.choices:
                for patch in choice:
                    assert patch.node in nodes
                    assert patch.parameter in nodes[patch.node].parameters
    required = {definition.entry.split(".")[0]}
    if definition.exit:
        required.add(definition.exit.split(".")[0])
    if definition.training:
        required.update(ref.split(".")[0] for ref in definition.training.state_refs)
        required.update(action.entry.split(".")[0] for action in definition.training.actions if action.entry)
    pending = list(required)
    while pending:
        for ref in nodes[pending.pop()].inputs.values():
            identifier = ref.split(".")[0]
            if identifier not in required:
                required.add(identifier)
                pending.append(identifier)
    assert required == set(nodes), "Template contains a decorative/disconnected branch"


def test_main_template_search_spaces_keep_all_108_candidates():
    items = _templates()
    for name in ("A2552", "A160S-106"):
        training = items[name]["definition"]["training"]
        assert prod(len(axis["choices"]) for axis in training["search_space"]) * sum(
            action["entry"] is not None for action in training["actions"]) == 108
    repair = {node["id"]: node for node in items["A2552"]["definition"]["nodes"]}
    assert repair["prior_return5"]["parameters"]["periods"] == 1
    assert repair["prior_peak20"]["parameters"]["periods"] == 1
    assert repair["slope_nonpositive"]["parameters"]["operator"] == "le"
    assert repair["slope_positive"]["parameters"]["operator"] == "gt"


def test_reversal_uses_prior_high_low_and_real_cross_sectional_breadth():
    definition = _templates()["A160S-106"]["definition"]
    nodes = {node["id"]: node for node in definition["nodes"]}
    assert nodes["high60"]["inputs"] == {"a": "high.value"}
    assert nodes["prior_high60"]["parameters"]["periods"] == 1
    assert nodes["prior_low20"]["parameters"]["periods"] == 1
    assert nodes["market_breadth250"]["op"] == "breadth"
    assert nodes["market"]["op"] == "basket_source"
    assert nodes["recovery_confirm"]["op"] == "confirm"
    assert nodes["opening_confirm"]["parameters"]["operator"] == "ge"


@pytest.fixture(scope="module")
def runtime():
    return GraphRuntime(build_catalog())


@pytest.mark.parametrize("item", etf_templates(), ids=lambda value: value["id"])
def test_every_template_prepares_and_executes_readonly_full_axis(runtime, item):
    definition = Definition.model_validate(item["definition"])
    before = copy.deepcopy(item)
    prepared = runtime.prepare(definition)
    bars = _bars()
    panel = np.vstack((bars.close, bars.close * 1.1))
    panel.setflags(write=False)
    baskets = {"market": panel, "category": panel}
    channels = runtime.evaluate(prepared, bars, baskets=baskets)
    assert channels[definition.entry].shape == bars.dates.shape
    assert channels[definition.entry].dtype == np.int64
    assert set(np.unique(channels[definition.entry])) <= {-1, 0, 1}
    assert all(not array.flags.writeable for array in channels.values())
    assert np.shares_memory(channels["close.value"], bars.close)
    if definition.training:
        for action in definition.training.actions:
            if action.entry:
                assert action.entry in channels
        for state in definition.training.state_refs:
            assert state in channels
    assert item == before


def test_repair_feature_values_match_explicit_reference(runtime):
    definition = Definition.model_validate(_templates()["A2552"]["definition"])
    bars = _bars()
    channels = runtime.evaluate(runtime.prepare(definition), bars)
    for index in (20, 30, 99, 350):
        assert channels["prior_return5.value"][index] == pytest.approx(bars.close[index - 1] / bars.close[index - 6] - 1)
        assert channels["prior_drawdown20.value"][index] == pytest.approx(
            bars.close[index - 1] / max(bars.close[index - 20:index]) - 1)


@pytest.mark.parametrize("item", etf_templates(), ids=lambda value: value["id"])
def test_future_prices_cannot_change_any_prefix_feature_or_signal(runtime, item):
    definition = Definition.model_validate(item["definition"])
    prepared = runtime.prepare(definition)
    short, long = _bars(400), _bars(600)
    short_panel = np.vstack((short.close, short.close * 1.1))
    long_panel = np.vstack((long.close, long.close * 1.1))
    short_panel.setflags(write=False)
    long_panel.setflags(write=False)
    prefix = runtime.evaluate(prepared, short, baskets={"market": short_panel, "category": short_panel})
    complete = runtime.evaluate(prepared, long, baskets={"market": long_panel, "category": long_panel})
    assert prefix.keys() == complete.keys()
    for channel, values in prefix.items():
        np.testing.assert_allclose(values, complete[channel][..., :400], rtol=1e-12, atol=1e-12,
                                   equal_nan=True, err_msg=f"Noncausal channel {channel}")


def test_missing_price_retains_dates_and_resets_alpha_beta(runtime):
    bars = _bars()
    owner = bars.close.copy()
    owner[300] = np.nan
    owner.setflags(write=False)
    bars.close = owner
    definition = Definition.model_validate(_templates()["A2552"]["definition"])
    channels = runtime.evaluate(runtime.prepare(definition), bars)
    assert channels["first_positive.value"][300] == -1
    assert channels["first_positive.value"][301] == -1
    assert np.isnan(channels["filter.slope"][300])
    assert channels["filter.slope"][301] == 0
    assert np.isnan(channels["filter.innovation"][301])
    assert channels["prior_return5.value"].shape == owner.shape
    assert np.shares_memory(channels["close.value"], owner)


def test_zero_volume_does_not_form_infinite_quality_score(runtime):
    bars = _bars()
    volume = bars.volume.copy()
    volume[350] = 0.0
    volume.setflags(write=False)
    bars.volume = volume
    panel = np.vstack((bars.close, bars.close * 1.1))
    panel.setflags(write=False)
    definition = Definition.model_validate(_templates()["A2536-Momo-095"]["definition"])
    channels = runtime.evaluate(runtime.prepare(definition), bars, baskets={"category": panel})
    assert np.isfinite(channels["quality_score.value"][350])
    assert channels["liquidity_gate.value"][350] == 0
    assert channels[definition.entry][350] == 0


def test_calendar_and_negative_veto_adaptations_have_explicit_real_rules():
    items = _templates()
    assert items["A2074"]["definition"]["training"]["mode"] == "month"
    assert items["A2076"]["definition"]["training"]["mode"] == "quarter"
    veto = items["A2296"]["definition"]
    nodes = {node["id"]: node for node in veto["nodes"]}
    assert nodes["combined_veto"]["op"] == "any"
    assert nodes["veto_clear"]["op"] == "not"
    assert nodes["vetoed_entry"]["inputs"]["right"] == "veto_clear.value"
    assert "HMM" in "".join(veto["adaptation"]["changed"])
    assert not any("hmm" in node["op"].lower() for node in veto["nodes"])


@pytest.mark.parametrize("mode,expected_mfi", [("flat", 50.0), ("up", 100.0), ("down", 0.0)])
def test_zero_range_and_directional_money_flow_have_explicit_finite_limits(runtime, mode, expected_mfi):
    bars = _bars()
    size = bars.dates.size
    price = (np.full(size, 100.0) if mode == "flat" else
             100.0 + (1 if mode == "up" else -1) * np.arange(size) / 20.0)
    price.setflags(write=False)
    bars.close = bars.open = bars.high = bars.low = price
    volume = np.full(size, 1000.0)
    volume.setflags(write=False)
    bars.volume = volume
    panel = np.vstack((price, price * 1.1))
    panel.setflags(write=False)
    definition = Definition.model_validate(_templates()["A160S-106"]["definition"])
    channels = runtime.evaluate(runtime.prepare(definition), bars, baskets={"market": panel})
    np.testing.assert_allclose(channels["cmf20.value"][20:], 0.0)
    np.testing.assert_allclose(channels["mfi14.value"][15:], expected_mfi)
    assert np.isnan(channels["mfi14.value"][:14]).all()
    assert not any(np.isinf(array).any() for array in channels.values())
    if mode == "flat":
        np.testing.assert_allclose(channels["position20.value"][20:], 0.0)
        np.testing.assert_array_equal(channels["recovery_position.value"][20:], 0)


def test_zero_range_first_positive_location_keeps_a2552_half_default(runtime):
    bars = _bars()
    bars.high = bars.low = bars.close
    definition = Definition.model_validate(_templates()["A2552"]["definition"])
    channels = runtime.evaluate(runtime.prepare(definition), bars)
    np.testing.assert_allclose(channels["close_position.value"], 0.5)


def test_zero_activity_quality_is_finite_but_never_means_liquid(runtime):
    bars = _bars()
    volume = np.zeros(bars.dates.size)
    volume.setflags(write=False)
    bars.volume = volume
    panel = np.vstack((bars.close, bars.close * 1.1))
    panel.setflags(write=False)
    definition = Definition.model_validate(_templates()["A2536-Momo-095"]["definition"])
    channels = runtime.evaluate(runtime.prepare(definition), bars, baskets={"category": panel})
    np.testing.assert_allclose(channels["volume_ratio20.value"][20:], 0.0)
    np.testing.assert_array_equal(channels["liquidity_gate.value"][20:], 0)
    assert np.isfinite(channels["quality_score.value"][21:]).all()
