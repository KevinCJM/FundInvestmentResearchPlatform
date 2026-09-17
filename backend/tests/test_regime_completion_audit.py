"""Independent oracles for remaining regime diagnostics; no production data."""
import numpy as np
import pytest

from historical_regimes.reliability import diagnostic_kernels as diagnostic
from historical_regimes.reliability import kernels


@pytest.fixture(scope="module", autouse=True)
def warm_runtime():
    kernels.warm()


def runs(labels):
    start = 0
    while start < len(labels):
        end = start + 1
        while end < len(labels) and labels[end] == labels[start]:
            end += 1
        yield start, end, int(labels[start])
        start = end


@pytest.mark.parametrize("state_count", [2, 3, 6])
@pytest.mark.parametrize("size", [0, 1, 13, 61])
def test_quality_matches_independent_interval_oracle(state_count, size):
    rng = np.random.default_rng(726 + state_count + size)
    owner = rng.integers(-1, state_count + 1, size=size * 2, dtype=np.int64)
    price_owner = np.exp(rng.normal(4, .1, size=size * 2))
    labels = owner[::2]
    prices = price_owner[::2]
    if size > 5:
        prices[5] = np.nan
    labels.flags.writeable = False
    prices.flags.writeable = False
    before = owner.copy(), price_owner.copy()
    summary, stats = diagnostic.quality_kernel(labels, prices, state_count)
    expected_segments = 0
    for state in range(state_count):
        intervals = [(a, b) for a, b, code in runs(labels) if code == state]
        lengths = [b - a for a, b in intervals]
        expected_segments += len(lengths)
        assert stats[state, 0] == sum(lengths)
        assert stats[state, 1] == len(lengths)
        if lengths:
            np.testing.assert_allclose(stats[state, 2:6], [min(lengths), np.median(lengths), max(lengths), np.mean(lengths)])
        else:
            assert np.isnan(stats[state, 2:6]).all()
        returns = [prices[b - 1] / prices[a] - 1 for a, b in intervals
                   if b - a >= 2 and np.isfinite(prices[a:b]).all() and (prices[a:b] > 0).all()]
        assert stats[state, 6] == len(returns)
        if returns:
            assert stats[state, 7] == pytest.approx(np.mean(returns))
        else:
            assert np.isnan(stats[state, 7])
    valid = (labels >= 0) & (labels < state_count)
    assert summary[0] == valid.sum()
    assert summary[1] == size - valid.sum()
    assert summary[4] == expected_segments
    assert summary[5] == sum(valid[i] and valid[i - 1] and labels[i] != labels[i - 1] for i in range(1, size))
    np.testing.assert_equal(owner, before[0])
    np.testing.assert_equal(price_owner, before[1])
    if size:
        assert np.shares_memory(labels, owner)
        assert np.shares_memory(prices, price_owner)


def test_sensitivity_distinguishes_coverage_agreement_and_boundaries():
    base = np.array([0, 0, 1, 1, 2, 2, 0, 0], np.int64)
    candidate = np.array([0, -1, 0, 1, 1, 2, 2, 0], np.int64)
    result = diagnostic.compare_kernel(base, candidate, 3)
    valid = candidate >= 0
    assert result[0] == valid.sum()
    assert result[1] == pytest.approx((base[valid] == candidate[valid]).mean())
    assert result[2] == pytest.approx(valid.mean())
    events = lambda x: [(i, (x[i - 1], x[i])) for i in range(1, len(x))
                        if x[i - 1] >= 0 and x[i] >= 0 and x[i - 1] != x[i]]
    distances = []
    for first, second in ((events(base), events(candidate)), (events(candidate), events(base))):
        for position, pair in first:
            matching = [abs(position - other) for other, other_pair in second if pair == other_pair]
            if matching:
                distances.append(min(matching))
    assert result[3] == pytest.approx(np.mean(distances))
    assert result[4] == len(distances)


def test_historical_peak_quality_probes_actual_turning_point_parameters():
    from types import SimpleNamespace
    from historical_regimes.reliability.diagnostics import variants, perturb_definitions
    from historical_regimes.v2_contracts import parse_definition_v2
    from historical_regimes.v2_templates import instantiate_template_v2
    policy = SimpleNamespace(perturbation=.1, max_variants=12, parameters=True,
                             windows=True, seeds=True, truncation=True)
    graph = SimpleNamespace(_perturbed_definitions=perturb_definitions)
    definition = parse_definition_v2(instantiate_template_v2('peak-trough-ps-v2'))
    probes = variants(graph, definition, policy)
    changed = {change['parameter'] for _, probe in probes for change in probe['changes']}
    assert changed & {'left_window', 'right_window', 'min_phase', 'min_cycle'}, changed


def test_editable_threshold_constants_are_model_parameters_not_data_sources():
    from types import SimpleNamespace
    from historical_regimes.reliability.diagnostics import variants, perturb_definitions
    from historical_regimes.v2_contracts import parse_definition_v2
    from historical_regimes.v2_templates import instantiate_template_v2
    policy = SimpleNamespace(perturbation=.1, max_variants=12, parameters=True,
                             windows=False, seeds=False, truncation=False)
    graph = SimpleNamespace(_perturbed_definitions=perturb_definitions)
    definition = parse_definition_v2(instantiate_template_v2('peak-trough-daily-v2'))
    probes = variants(graph, definition, policy)
    changed = {change['node_id'] for _, probe in probes for change in probe['changes']}
    assert changed & {'upper', 'lower'}, changed


def test_probability_evidence_uses_selected_class_not_max_and_preserves_input():
    labels = np.array([0, 1, -1, 1], np.int64)
    probabilities = np.array([[.2, .7, .1], [.3, .3, .4], [.2, .7, .1], [.4, .4, .4]])
    labels.flags.writeable = False
    probabilities.flags.writeable = False
    result = diagnostic.evidence_kernel(labels, probabilities)
    assert result[0, 0] == .2
    assert result[0, 1] == .7
    assert result[0, 2] == .2
    assert result[0, 3] == pytest.approx(.5)
    assert result[0, 4] == pytest.approx(-np.sum(probabilities[0] * np.log(probabilities[0])))
    assert result[1, 3] == pytest.approx(.1)
    assert np.isnan(result[2:]).all()
    assert kernels.audit()["python_fallback"] == 0
