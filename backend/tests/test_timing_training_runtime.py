"""Real service policy integration, fixed basket lineage and no future labels."""
from dataclasses import replace
from collections import Counter
import numpy as np
import pytest

from backend.timing_research.contracts import Definition, RunRequest
from backend.timing_research.service import TimingResearchService
from backend.timing_research.etf_templates import etf_templates
from backend.timing_research.training_runtime import prepare_candidates
from test_timing_service import fixture_bars, completed


def _definition(mode="global"):
    return Definition.model_validate({"name": "train-only fixture", "nodes": [
        {"id": "price", "label": "Price", "op": "source"},
        {"id": "entry", "label": "Positive", "op": "compare", "inputs": {"left": "price.value"}, "parameters": {"threshold": 0.0}},
        {"id": "state", "label": "Environment", "op": "compare", "inputs": {"left": "price.value"}, "parameters": {"threshold": 120.0}},
    ], "entry": "entry.value", "training": {"mode": mode, "state_refs": ["state.value"] if mode == "state" else [],
        "actions": [{"id": "long", "label": "Long", "entry": "entry.value"}, {"id": "cash", "label": "Cash", "entry": None}],
        "min_trades": 2, "confidence": 0.0, "risk_penalty": 0.0},
        "execution": {"take_profit": .5, "stop_loss": .5, "max_holding_bars": 5, "fee_bps": 0., "slippage_bps": 0.}})


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    value = TimingResearchService(tmp_path_factory.mktemp("training-runtime"), loader=lambda *args: fixture_bars())
    value.warm()
    yield value
    value.close()


def _request(service, definition, baskets=None):
    prepared = service.prepare(definition)
    return RunRequest(definition=definition, compile_token=prepared["compile_token"], targets=[{"product_id": "510300.SH"}],
        start_date="2020-01-01", end_date="2021-12-31", holdout_start="2021-01-01", context_baskets=baskets or {})


@pytest.mark.parametrize("mode,states", [("global", 1), ("state", 2), ("month", 12), ("quarter", 4)])
def test_frozen_policy_runs_all_supported_modes(service, mode, states):
    request = _request(service, _definition(mode))
    run = completed(service, service.submit(request))
    product = run["products"][0]
    assert product["status"] == "ok", product
    audit = product["training"]
    assert len(audit["selection"]) == states
    assert audit["freeze_date"] == "2020-12-31"
    assert product["execution"]["request_time_compilation"] == 0
    assert product["summary"]["in_sample"]["trade_count"] == 0
    assert product["summary"]["in_sample"]["total_return"] == 0
    assert any(channel["id"] == "training.entry" for channel in product["channels"])
    assert all(trade["entry_date"] >= "2021-01-01" for trade in product["trades"])


def test_future_market_outcomes_cannot_change_frozen_policy(service):
    request = _request(service, _definition())
    graph = service.graph.get(request.definition, request.compile_token)
    bars = fixture_bars()
    result, arrays = service._product(request, graph, "510300.SH", load=lambda code: bars)
    index = int(np.searchsorted(bars.dates, np.datetime64("2021-01-01").astype(np.int64)))
    updates = {}
    for field in ("open", "high", "low", "close"):
        values = getattr(bars, field).copy()
        values[index:] *= 2.0
        values.setflags(write=False)
        updates[field] = values
    other, changed = service._product(request, graph, "510300.SH", load=lambda code: replace(bars, **updates))
    assert result["training"] == other["training"]
    np.testing.assert_array_equal(arrays["entry"][:index], changed["entry"][:index])


def test_missing_training_evidence_stays_cash_not_test_fallback(service):
    definition = _definition()
    definition.training.min_trades = 10000
    request = _request(service, definition)
    result, arrays = service._product(request, service.graph.get(definition, request.compile_token), "510300.SH")
    assert result["training"]["selection"][0]["action_id"] is None
    assert result["summary"]["out_of_sample"]["trade_count"] == 0
    assert np.max(arrays["entry"]) == 0


def test_search_cannot_mutate_shared_state_or_exit(service):
    definition = _definition("state")
    raw = definition.model_dump(mode="json")
    raw["training"]["search_space"] = [{"label": "invalid", "choices": [[{"node": "state", "parameter": "threshold", "value": 0}]]}]
    with pytest.raises(Exception, match="共享状态"):
        service.prepare(Definition.model_validate(raw))


def test_search_references_and_candidate_budget_are_validated(service):
    raw = _definition().model_dump(mode="json")
    raw["training"]["search_space"] = [{"label": "invalid", "choices": [[{"node": "missing", "parameter": "threshold", "value": 0}]]}]
    with pytest.raises(Exception, match="不存在"):
        service.prepare(Definition.model_validate(raw))
    raw["training"]["search_space"] = [{"label": "axis", "choices": [[{"node": "entry", "parameter": "threshold", "value": n}] for n in range(12)]}] * 2
    with pytest.raises(Exception, match="108"):
        Definition.model_validate(raw)


def test_basket_is_explicit_missing_dates_unknown_and_inputs_loaded_once(service, monkeypatch):
    item = next(item for item in etf_templates() if item["definition"]["adaptation"]["source_experiments"] == ["A160S-106"])
    definition = Definition.model_validate(item["definition"])
    definition.training.search_space = []
    request = _request(service, definition)
    with pytest.raises(Exception, match="至少 2"):
        service.submit(request)
    calls = Counter()
    bars = fixture_bars()
    def load(base, code, *args):
        calls[code] += 1
        if code == "510500.SH":
            price = bars.close.copy(); price[400] = np.nan; price.setflags(write=False)
            return replace(bars, close=price)
        return bars
    monkeypatch.setattr(service, "loader", load)
    request.context_baskets = {"market": ["510300.SH", "510500.SH"]}
    run = completed(service, service.submit(request))
    result = run["products"][0]
    assert result["status"] == "ok", result
    assert calls == {"510300.SH": 1, "510500.SH": 1}
    assert len(result["lineage"]["context_baskets"]["market"]["members"]) == 2
    arrays = service.repository.load_arrays(run["id"], "510300.SH")
    assert np.isnan(arrays["basket_market"][1, 400])
    assert not arrays["basket_market"].flags.writeable
    assert len(arrays["dates"]) == len(bars.dates)


@pytest.mark.parametrize("source", ["A2552", "A160S-106"])
def test_full_search_has_108_prepared_candidates_and_token_survives(service, source):
    item = next(item for item in etf_templates() if item["definition"]["adaptation"]["source_experiments"] == [source])
    definition = Definition.model_validate(item["definition"])
    prepared = service.prepare(definition)
    assert len(service._training_plans[prepared["compile_token"]]) == 108
    assert service.graph.get(definition, prepared["compile_token"])


def test_old_definition_without_training_remains_valid():
    raw = _definition().model_dump(mode="json")
    del raw["training"]
    del raw["adaptation"]
    assert Definition.model_validate(raw).training is None
