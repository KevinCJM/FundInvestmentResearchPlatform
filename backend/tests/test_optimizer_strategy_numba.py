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
