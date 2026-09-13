"""Multi-fold causality, maturity gaps and shared-memory regression."""
from datetime import date, timedelta

import numpy as np
import pytest

from backend.custom_indicators.errors import ValidationError
from backend.tactical_allocation import numeric
from backend.tactical_allocation.contracts import PreviewRequest, WalkForwardConfig
from backend.tactical_allocation import walk_forward as wf


@pytest.fixture(scope="module", autouse=True)
def warm():
    numeric.warm_tactical_allocation_kernels()
    assert wf.warm_walk_forward_kernels()["complete"]


def example(size=120):
    days = [str(date(2020, 1, 1) + timedelta(days=i)) for i in range(size + 1)]
    returns = np.zeros((size, 2))
    returns[:, 0] = .002
    returns[60:, 0] = -.002
    available = np.broadcast_to(np.arange(18263, 18263 + size, dtype=np.int64)[:, None], returns.shape).copy()
    data = {"returns": returns, "dates": days[1:], "period_starts": days[:-1], "available_at": available}
    signals = {"probabilities": np.ones((size, 1)), "use_signal": np.ones(size, dtype=np.uint8),
               "state_tilts": np.array([[.1, -.1]])}
    request = PreviewRequest(baseline_id="baseline", start_date=days[0], end_date=days[-1], as_of=days[-1],
        train_end_date=days[60], signal_mode="manual", manual_tilts={"equity": .1, "bonds": -.1},
        max_tracking_error=1., walk_forward=WalkForwardConfig(training_periods=40, validation_periods=20))
    args = (request, data, signals, np.array([.5, .5]), np.zeros(2), np.ones(2), np.full(2, .2), {})
    return args


def test_fold_selection_never_uses_own_holdout():
    args = example()
    first = wf.evaluate_walk_forward(*args)
    args[1]["returns"][40:] = [.05, -.04]
    changed = wf.evaluate_walk_forward(*args)
    a, b = first["folds"][0], changed["folds"][0]
    assert a["selected_id"] == b["selected_id"]
    assert a["training"] == b["training"]
    assert a["validation"] != b["validation"]
    assert first["completed_folds"] == 4
    assert first["primary_selection_changed"] is False
    assert first["holdout_used_for_selection"] is False
    assert first["independently_funded_intervals"] is True
    assert "nav" not in first


def test_immature_tail_is_purged_without_compressing_holdout_axis(monkeypatch):
    args = example()
    data = args[1]
    cutoff = (date.fromisoformat(data["period_starts"][40]) - date(1970, 1, 1)).days
    data["available_at"][39] = cutoff + 10
    original = numeric.evaluate_candidates
    observed = []

    def captured(*pos, **kwargs):
        assert np.shares_memory(pos[0], data["returns"])
        observed.append((pos[8], kwargs["validation_start_index"]))
        return original(*pos, **kwargs)

    monkeypatch.setattr(numeric, "evaluate_candidates", captured)
    result = wf.evaluate_walk_forward(*args)
    assert observed[0] == (39, 40)
    first = result["folds"][0]
    assert first["training_observations"] == 39
    assert first["purged_training_periods"] == 1
    assert first["validation_observations"] == 20
    assert first["decision_cutoff"] == data["period_starts"][40]
    data["returns"][39] = [.8, -.8]
    repeated = wf.evaluate_walk_forward(*args)["folds"][0]
    assert repeated["training"] == first["training"]
    assert repeated["validation"] == first["validation"]


def test_interior_unknown_blocks_fold_instead_of_assuming_available():
    args = example()
    args[1]["available_at"][20] = -1
    result = wf.evaluate_walk_forward(*args)
    assert result["folds"][0]["status"] == "blocked"
    assert result["folds"][0]["unknown_training_values"] == 2
    assert result["blocked_folds"] == 2
    assert result["folds"][-1]["status"] == "complete"


def test_expanding_and_rolling_have_declared_different_training_windows():
    args = example()
    rolling = wf.evaluate_walk_forward(*args)
    args[0].walk_forward.window_mode = "expanding"
    expanding = wf.evaluate_walk_forward(*args)
    assert rolling["folds"][-1]["training_observations"] == 40
    assert expanding["folds"][-1]["training_observations"] == 100
    assert len({f["train_start"] for f in expanding["folds"]}) == 1


def test_fixed_hypothesis_infeasible_fold_is_blocked_without_aborting_later_folds():
    args = list(example())
    request = args[0].model_copy(update={"search": False, "max_tracking_error": .01})
    alternating = np.where(np.arange(40) % 2 == 0, .03, -.03)
    args[1]["returns"][:] = 0.0
    args[1]["returns"][:40, 0] = alternating
    args[0] = request
    result = wf.evaluate_walk_forward(*args)
    assert result["folds"][0]["status"] == "blocked"
    assert result["folds"][0]["selected_id"] == "scale-1"
    assert "保留该假设" in result["folds"][0]["reasons"][0]
    assert result["blocked_folds"] >= 1
    assert any(fold["status"] == "complete" for fold in result["folds"][1:])


def test_noncontiguous_readonly_input_and_no_signature_growth():
    args = list(example())
    raw = np.repeat(args[1]["returns"], 2, axis=1)
    raw.setflags(write=False)
    args[1]["returns"] = raw[:, ::2]
    args[1]["available_at"].setflags(write=False)
    signatures = list(wf.mature_training_kernel.signatures)
    first = wf.evaluate_walk_forward(*args)
    second = wf.evaluate_walk_forward(*args)
    assert first == second
    assert list(wf.mature_training_kernel.signatures) == signatures
    assert np.shares_memory(raw, args[1]["returns"])


def test_short_tail_is_reported_and_excess_folds_are_rejected():
    args = example(125)
    result = wf.evaluate_walk_forward(*args)
    assert result["excluded_tail_observations"] == 5
    assert result["completed_folds"] == 4
    with pytest.raises(ValidationError, match="40"):
        wf.evaluate_walk_forward(*example(900))


def test_explicit_gap_rejects_negative_or_overlapping_validation():
    args = example()
    with pytest.raises(ValueError, match="Validation"):
        numeric.evaluate_candidates(args[1]["returns"], args[2]["probabilities"], args[2]["use_signal"],
            args[3], args[2]["state_tilts"], args[4], args[5], args[6], 40, validation_start_index=39)


def test_cold_worker_fails_closed(monkeypatch):
    monkeypatch.setattr(wf, "_WARMED_PID", None)
    with pytest.raises(RuntimeError, match="预热"):
        wf.evaluate_walk_forward(*example())
