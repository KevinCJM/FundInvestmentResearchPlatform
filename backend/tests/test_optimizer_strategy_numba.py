from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import optimizer
import strategy
from compute_policy import validate_execution_audit


def _nav_frame() -> pd.DataFrame:
    generator = np.random.default_rng(90210)
    returns = generator.normal(
        loc=np.asarray([0.0005, 0.0002, 0.00035]),
        scale=np.asarray([0.009, 0.006, 0.011]),
        size=(90, 3),
    )
    nav = np.vstack([np.ones(3), np.cumprod(1.0 + returns, axis=0)])
    return pd.DataFrame(
        nav,
        index=pd.date_range("2024-01-02", periods=nav.shape[0], freq="B"),
        columns=["权益", "债券", "商品"],
    )


def _assert_constraints(weights: list[float]) -> None:
    assert sum(weights) == pytest.approx(1.0, abs=1e-7)
    assert all(0.10 - 1e-7 <= value <= 0.75 + 1e-7 for value in weights)
    assert 0.45 - 1e-7 <= weights[0] + weights[1] <= 0.85 + 1e-7


def test_optimizer_and_strategy_audits_are_policy_compliant() -> None:
    optimizer_audit = optimizer.warm_optimizer_numba_kernels()
    strategy_audit = strategy.warm_strategy_numba_kernels()

    assert validate_execution_audit(optimizer_audit)["python_fallback"] == 0
    assert validate_execution_audit(strategy_audit)["python_fallback"] == 0
    assert optimizer_audit["fully_warmed"] is True
    assert strategy_audit["fully_warmed"] is True
    assert len(optimizer_audit["fingerprint"]) == 64
    assert len(strategy_audit["fingerprint"]) == 64
    assert all(
        len(dispatcher.nopython_signatures) == 1
        for dispatcher in optimizer.OPTIMIZER_NUMBA_KERNELS + strategy.STRATEGY_NUMBA_KERNELS
    )
    assert all(
        dispatcher._can_compile is False
        for dispatcher in optimizer.OPTIMIZER_NUMBA_KERNELS + strategy.STRATEGY_NUMBA_KERNELS
    )


def test_5000_point_frontier_is_njit_and_respects_all_constraints() -> None:
    nav = _nav_frame()
    return_values, status = strategy.nav_to_returns_kernel(
        np.ascontiguousarray(nav.to_numpy(dtype=np.float64))
    )
    assert status == 0
    returns = pd.DataFrame(return_values, columns=nav.columns)
    result = optimizer.calculate_efficient_frontier_exploration(
        returns,
        {"metric": "annual", "days": 252},
        {"metric": "annual_vol", "days": 252},
        single_limits=[(0.10, 0.75)] * 3,
        group_limits={(0, 1): (0.45, 0.85)},
        rounds=[{"samples": 5000, "step": 1.0, "buckets": 80}],
        seed=17,
    )

    assert len(result["scatter"]) == 5000
    assert result["frontier"]
    for point in result["scatter"]:
        _assert_constraints(point["weights"])
    assert result["execution"]["python_fallback"] == 0


@pytest.mark.parametrize(
    ("target", "extra"),
    [
        ("min_risk", {}),
        ("max_return", {}),
        ("max_sharpe", {}),
        ("max_sharpe_traditional", {}),
        ("risk_min_given_return", {"target_return": -0.50}),
        ("return_max_given_risk", {"target_risk": 10.0}),
    ],
)
def test_every_target_is_selected_inside_fixed_signature_njit(
    target: str,
    extra: dict[str, float],
) -> None:
    weights = strategy.compute_target_weights(
        _nav_frame(),
        {"metric": "annual", "days": 252},
        {"metric": "annual_vol", "days": 252},
        target,
        single_limits=[(0.10, 0.75)] * 3,
        group_limits={(0, 1): (0.45, 0.85)},
        **extra,
    )
    _assert_constraints(weights)


def test_risk_budget_is_njit_and_invalid_inputs_fail_explicitly() -> None:
    assert strategy.equal_weights(3) == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    with pytest.raises(ValueError, match="至少需要一个资产"):
        strategy.equal_weights(0)
    assert strategy.normalize_explicit_weights([2.0, 1.0]) == pytest.approx([2 / 3, 1 / 3])
    assert strategy.scale_weights_percent([0.6, 0.4], 0.25) == [75.0, 50.0]
    thirds = strategy.scale_weights_percent([1 / 3, 1 / 3, 1 / 3], 0.0)
    assert thirds == [33.34, 33.33, 33.33]
    assert sum(thirds) == 100.0
    with pytest.raises(ValueError, match="总和必须大于 0"):
        strategy.normalize_explicit_weights([0.0, 0.0])
    with pytest.raises(ValueError, match="最大杠杆"):
        strategy.scale_weights_percent([0.5, 0.5], -0.1)
    weights = strategy.compute_risk_budget_weights(
        _nav_frame(), {"metric": "vol"}, [0.50, 0.30, 0.20]
    )
    assert sum(weights) == pytest.approx(1.0, abs=1e-9)
    with pytest.raises(ValueError, match="总和大于 0"):
        strategy.compute_risk_budget_weights(
            _nav_frame(), {"metric": "vol"}, [0.0, 0.0, 0.0]
        )
    with pytest.raises(ValueError, match="没有可行组合"):
        strategy.compute_target_weights(
            _nav_frame(),
            {"metric": "annual"},
            {"metric": "annual_vol"},
            "min_risk",
            single_limits=[(0.6, 0.8)] * 3,
        )


def test_local_refinement_is_deterministic_constrained_and_non_worsening() -> None:
    nav = _nav_frame()
    return_values, status = strategy.nav_to_returns_kernel(
        np.ascontiguousarray(nav.to_numpy(dtype=np.float64))
    )
    assert status == 0
    returns = pd.DataFrame(return_values, columns=nav.columns)
    kwargs = dict(
        asset_returns=returns,
        return_config={"metric": "annual", "days": 252},
        risk_config={"metric": "annual_vol", "days": 252},
        single_limits=[(0.10, 0.75)] * 3,
        group_limits={(0, 1): (0.45, 0.85)},
        rounds=[{"samples": 1000, "step": 1.0, "buckets": 40}],
        use_local_refine=True,
        refine_iterations=30,
        risk_free_rate=0.01,
        seed=19,
    )
    first = optimizer.calculate_efficient_frontier_exploration(**kwargs)
    second = optimizer.calculate_efficient_frontier_exploration(**kwargs)
    assert first == second
    assert first["refinement"]["algorithm"] == "bounded_pairwise_pattern_search_njit"
    assert first["refinement"]["global_optimum_claim"] is False
    assert all(item["non_worsening"] for item in first["refinement"]["items"])
    for key in ("max_sharpe", "min_variance", "max_return"):
        _assert_constraints(first[key]["weights"])
    assert first["execution"]["python_fallback"] == 0
    assert "refine_special_candidates_kernel" in first["execution"]["kernel_signatures"]


def test_quantized_refinement_never_worsens_the_original_sampled_candidate() -> None:
    rng = np.random.default_rng(1461)
    values = rng.normal(0.0, 0.01, (80, 3)) + rng.uniform(0.0, 0.001, 3)
    lows = rng.uniform(0.001, 0.2, 3)
    highs = rng.uniform(0.5, 0.9, 3)
    kwargs = dict(
        asset_returns=pd.DataFrame(values, columns=["A", "B", "C"]),
        return_config={"metric": "annual", "days": 252},
        risk_config={"metric": "annual_vol", "days": 252},
        single_limits=list(zip(lows, highs, strict=True)),
        group_limits={(0, 1): (0.431, 0.763)},
        rounds=[{"samples": 100, "step": 1.0, "buckets": 40}],
        quantize_step=0.005,
        risk_free_rate=0.0,
        seed=42,
    )
    original = optimizer.calculate_efficient_frontier_exploration(
        **kwargs, use_local_refine=False, refine_iterations=20,
    )
    refined = optimizer.calculate_efficient_frontier_exploration(
        **kwargs, use_local_refine=True, refine_iterations=20,
    )

    def independent_sharpe(point: dict) -> float:
        portfolio = values @ np.asarray(point["weights"], dtype=np.float64)
        return float(portfolio.mean() * 252 / (portfolio.std(ddof=1) * np.sqrt(252)))

    before = independent_sharpe(original["max_sharpe"])
    after = independent_sharpe(refined["max_sharpe"])
    item = refined["refinement"]["items"][0]
    assert item["before_score"] == pytest.approx(before, rel=1e-12, abs=1e-12)
    assert item["non_worsening"] is True
    assert after >= before - 1e-12


def test_representative_selector_uses_one_finite_set_and_stable_tie_rules() -> None:
    risks = np.ascontiguousarray([0.0, 0.10, 0.10, 0.20], dtype=np.float64)
    returns = np.ascontiguousarray([1.0, 0.03, 0.04, 0.05], dtype=np.float64)
    max_sharpe, min_risk, max_return = optimizer.representative_indices_kernel(
        risks, returns, np.int64(4), np.float64(0.01)
    )
    assert max_sharpe == 2  # zero-risk point is ineligible for Sharpe selection
    assert min_risk == 0
    assert max_return == 0

    tied_returns = np.ascontiguousarray([0.03, 0.03], dtype=np.float64)
    tied_risks = np.ascontiguousarray([0.10, 0.10], dtype=np.float64)
    assert optimizer.representative_indices_kernel(
        tied_risks, tied_returns, np.int64(2), np.float64(0.0)
    ) == (0, 0, 0)
    zero_risks = np.ascontiguousarray([0.0, 0.0], dtype=np.float64)
    assert optimizer.representative_indices_kernel(
        zero_risks, tied_returns, np.int64(2), np.float64(0.0)
    ) == (-1, 0, 0)


def test_final_representatives_are_selected_from_the_complete_returned_candidate_set() -> None:
    cases = [
        (17, 42, None, None, 0.01),
        (1, 19, None, None, 0.0),
        (1461, 19, 0.005, {(0, 1): (0.431, 0.763)}, 0.005),
    ]
    for data_seed, optimizer_seed, quantize_step, group_limits, risk_free_rate in cases:
        rng = np.random.default_rng(data_seed)
        values = rng.normal(0.0, 0.01, (120, 3)) + rng.uniform(-0.001, 0.002, 3)
        kwargs = dict(
            asset_returns=pd.DataFrame(values, columns=["A", "B", "C"]),
            return_config={"metric": "annual", "days": 252},
            risk_config={"metric": "annual_vol", "days": 252},
            rounds=[{"samples": 100, "step": 1.0, "buckets": 40}],
            quantize_step=quantize_step,
            group_limits=group_limits,
            risk_free_rate=risk_free_rate,
            seed=optimizer_seed,
        )
        for refine_iterations in (0, 1, 20):
            result = optimizer.calculate_efficient_frontier_exploration(
                **kwargs,
                use_local_refine=refine_iterations > 0,
                refine_iterations=refine_iterations,
            )
            risks = np.asarray([point["value"][0] for point in result["scatter"]], dtype=np.float64)
            expected_returns = np.asarray([point["value"][1] for point in result["scatter"]], dtype=np.float64)
            valid_sharpe = np.where(risks > 1e-12, (expected_returns - risk_free_rate) / risks, -np.inf)
            assert result["max_sharpe"]["value"] == pytest.approx(
                result["scatter"][int(np.argmax(valid_sharpe))]["value"], abs=1e-12
            )
            assert result["min_variance"]["value"] == pytest.approx(
                result["scatter"][int(np.argmin(risks))]["value"], abs=1e-12
            )
            assert result["max_return"]["value"] == pytest.approx(
                result["scatter"][int(np.argmax(expected_returns))]["value"], abs=1e-12
            )


def test_refined_candidates_are_included_when_rebuilding_the_frontier() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 0.01, (160, 3)) + np.array([0.001, 0.0001, 0.0005])
    result = optimizer.calculate_efficient_frontier_exploration(
        pd.DataFrame(values, columns=["A", "B", "C"]),
        {"metric": "annual", "days": 252},
        {"metric": "annual_vol", "days": 252},
        rounds=[{"samples": 100, "step": 1.0, "buckets": 40}],
        use_local_refine=True,
        refine_iterations=20,
        seed=42,
    )
    assert result["refined_candidates"] >= 1
    for point in result["frontier"]:
        risk, expected_return = point["value"]
        for candidate in result["scatter"]:
            candidate_risk, candidate_return = candidate["value"]
            dominates = (
                candidate_risk <= risk + 1e-12
                and candidate_return >= expected_return - 1e-12
                and (candidate_risk < risk - 1e-9 or candidate_return > expected_return + 1e-9)
            )
            assert not dominates, f"candidate {candidate['value']} dominates returned frontier point {point['value']}"
    assert result["accepted_candidates"] == len(result["scatter"])
    assert result["accepted_candidates"] == result["sampled_candidates"] + result["refined_candidates"]


def test_requests_do_not_add_signatures_and_no_scipy_solver_remains() -> None:
    optimizer.warm_optimizer_numba_kernels()
    strategy.warm_strategy_numba_kernels()
    dispatchers = optimizer.OPTIMIZER_NUMBA_KERNELS + strategy.STRATEGY_NUMBA_KERNELS
    before = [tuple(dispatcher.nopython_signatures) for dispatcher in dispatchers]

    strategy.compute_target_weights(
        _nav_frame(),
        {"metric": "ewm", "alpha": 0.92},
        {"metric": "es", "confidence": 0.95},
        "max_sharpe",
    )

    assert before == [tuple(dispatcher.nopython_signatures) for dispatcher in dispatchers]
    source = inspect.getsource(optimizer) + inspect.getsource(strategy.compute_target_weights)
    assert "scipy" not in source.lower()
    assert "minimize(" not in source
