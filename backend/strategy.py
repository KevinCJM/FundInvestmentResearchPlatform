from __future__ import annotations

import hashlib
import inspect
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np
import pandas as pd
from numba import njit, types
from numba.core.registry import CPUDispatcher

try:
    from backend.compute_policy import validate_execution_audit, validate_execution_graph
    from backend.optimizer import (
        optimizer_numba_status,
        select_target_weights,
        warm_optimizer_numba_kernels,
    )
    from backend.trading_calendar import get_trading_days
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from compute_policy import validate_execution_audit, validate_execution_graph
    from optimizer import (
        optimizer_numba_status,
        select_target_weights,
        warm_optimizer_numba_kernels,
    )
    from trading_calendar import get_trading_days


STRATEGY_NUMBA_KERNEL_VERSION = "2.0.0"
_FLOAT64_1D = types.float64[::1]
_FLOAT64_2D = types.float64[:, ::1]


@njit((types.int64,), cache=False, nogil=True)
def equal_weights_kernel(asset_count: int) -> tuple[np.ndarray, int]:
    if asset_count <= 0:
        return np.empty(0, dtype=np.float64), 1
    return np.full(asset_count, 1.0 / asset_count, dtype=np.float64), 0


@njit((_FLOAT64_1D,), cache=False, nogil=True)
def normalize_explicit_weights_kernel(
    raw_weights: np.ndarray,
) -> tuple[np.ndarray, int]:
    output = raw_weights.copy()
    if output.size == 0:
        return output, 1
    total = 0.0
    for index in range(output.size):
        value = output[index]
        if not np.isfinite(value) or value < 0.0:
            return output, 2
        total += value
    if total <= 0.0 or not np.isfinite(total):
        return output, 3
    output /= total
    return output, 0


@njit((_FLOAT64_1D, types.float64), cache=False, nogil=True)
def scale_weights_percent_kernel(
    normalized_weights: np.ndarray,
    max_leverage: float,
) -> tuple[np.ndarray, int]:
    output = normalized_weights.copy()
    if output.size == 0:
        return output, 1
    if not np.isfinite(max_leverage) or max_leverage < 0.0:
        return output, 2
    total = 0.0
    for value in output:
        if not np.isfinite(value) or value < 0.0:
            return output, 3
        total += value
    if abs(total - 1.0) > 1e-7:
        return output, 4
    multiplier = 100.0 * (1.0 + max_leverage)
    if not np.isfinite(multiplier) or multiplier > 9.0e13:
        return output, 2
    target_cents = int(np.round(multiplier * 100.0))
    allocated_cents = np.empty(output.size, dtype=np.int64)
    remainders = np.empty(output.size, dtype=np.float64)
    allocated_total = 0
    for index in range(output.size):
        ideal_cents = output[index] / total * target_cents
        cents = int(np.floor(ideal_cents))
        allocated_cents[index] = cents
        remainders[index] = ideal_cents - cents
        allocated_total += cents
    remaining = target_cents - allocated_total
    for _ in range(remaining):
        largest_index = 0
        largest_remainder = remainders[0]
        for index in range(1, output.size):
            if remainders[index] > largest_remainder:
                largest_index = index
                largest_remainder = remainders[index]
        allocated_cents[largest_index] += 1
        remainders[largest_index] = -1.0
    for index in range(output.size):
        output[index] = allocated_cents[index] / 100.0
    return output, 0


@njit((_FLOAT64_2D,), cache=False, nogil=True)
def nav_to_returns_kernel(nav_values: np.ndarray) -> tuple[np.ndarray, int]:
    row_count, asset_count = nav_values.shape
    output = np.empty((max(row_count - 1, 0), asset_count), dtype=np.float64)
    if row_count < 2 or asset_count == 0:
        return output, 1
    for row_index in range(row_count - 1):
        for asset_index in range(asset_count):
            previous = nav_values[row_index, asset_index]
            current = nav_values[row_index + 1, asset_index]
            if not np.isfinite(previous) or not np.isfinite(current) or previous <= 0.0 or current <= 0.0:
                return output, 2
            output[row_index, asset_index] = current / previous - 1.0
    return output, 0


@njit((_FLOAT64_2D, _FLOAT64_1D, types.int64, types.float64), cache=False, nogil=True)
def risk_budget_weights_kernel(
    asset_returns: np.ndarray,
    raw_budgets: np.ndarray,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, int, float]:
    """Solve covariance risk budgets with cyclic coordinate descent."""

    row_count, asset_count = asset_returns.shape
    weights = np.zeros(asset_count, dtype=np.float64)
    if row_count < 2 or asset_count == 0:
        return weights, 1, 0, np.inf
    if raw_budgets.size != asset_count:
        return weights, 2, 0, np.inf
    budget_total = 0.0
    budgets = np.empty(asset_count, dtype=np.float64)
    for asset_index in range(asset_count):
        value = raw_budgets[asset_index]
        if not np.isfinite(value) or value < 0.0:
            return weights, 2, 0, np.inf
        budgets[asset_index] = value
        budget_total += value
    if budget_total <= 0.0:
        return weights, 2, 0, np.inf
    budgets /= budget_total

    means = np.zeros(asset_count, dtype=np.float64)
    for row_index in range(row_count):
        for asset_index in range(asset_count):
            value = asset_returns[row_index, asset_index]
            if not np.isfinite(value):
                return weights, 3, 0, np.inf
            means[asset_index] += value
    means /= row_count
    covariance = np.zeros((asset_count, asset_count), dtype=np.float64)
    for row_index in range(row_count):
        for left in range(asset_count):
            left_delta = asset_returns[row_index, left] - means[left]
            for right in range(asset_count):
                covariance[left, right] += left_delta * (asset_returns[row_index, right] - means[right])
    covariance /= row_count - 1
    trace = 0.0
    for asset_index in range(asset_count):
        trace += max(covariance[asset_index, asset_index], 0.0)
    ridge = max(trace / max(asset_count, 1) * 1e-10, 1e-14)
    for asset_index in range(asset_count):
        covariance[asset_index, asset_index] += ridge
        weights[asset_index] = np.sqrt(budgets[asset_index] / covariance[asset_index, asset_index]) if budgets[asset_index] > 0.0 else 0.0

    completed_iterations = 0
    converged = False
    for iteration in range(max(maximum_iterations, 1)):
        largest_change = 0.0
        for asset_index in range(asset_count):
            diagonal = covariance[asset_index, asset_index]
            cross = 0.0
            for other_index in range(asset_count):
                if other_index != asset_index:
                    cross += covariance[asset_index, other_index] * weights[other_index]
            discriminant = cross * cross + 4.0 * diagonal * budgets[asset_index]
            if diagonal <= 0.0 or discriminant < 0.0 or not np.isfinite(discriminant):
                return weights, 3, iteration, np.inf
            updated = (-cross + np.sqrt(discriminant)) / (2.0 * diagonal)
            if updated < 0.0:
                updated = 0.0
            change = abs(updated - weights[asset_index])
            if change > largest_change:
                largest_change = change
            weights[asset_index] = updated
        completed_iterations = iteration + 1
        if largest_change <= tolerance:
            converged = True
            break
    if not converged:
        return weights, 4, completed_iterations, np.inf

    weight_total = np.sum(weights)
    if not np.isfinite(weight_total) or weight_total <= 0.0:
        return weights, 3, completed_iterations, np.inf
    weights /= weight_total
    marginal = covariance @ weights
    portfolio_variance = np.dot(weights, marginal)
    if not np.isfinite(portfolio_variance) or portfolio_variance <= 0.0:
        return weights, 3, completed_iterations, np.inf
    maximum_error = 0.0
    for asset_index in range(asset_count):
        contribution_share = weights[asset_index] * marginal[asset_index] / portfolio_variance
        error = abs(contribution_share - budgets[asset_index])
        if error > maximum_error:
            maximum_error = error
    if maximum_error > 1e-5:
        return weights, 5, completed_iterations, maximum_error
    return weights, 0, completed_iterations, maximum_error


STRATEGY_NUMBA_KERNELS: tuple[CPUDispatcher, ...] = (
    equal_weights_kernel,
    normalize_explicit_weights_kernel,
    scale_weights_percent_kernel,
    nav_to_returns_kernel,
    risk_budget_weights_kernel,
)
for _dispatcher in STRATEGY_NUMBA_KERNELS:
    _dispatcher.disable_compile()
_STRATEGY_WARMED = False


def _strategy_kernel_fingerprint(dispatcher: CPUDispatcher) -> str:
    return hashlib.sha256(inspect.getsource(dispatcher.py_func).encode("utf-8")).hexdigest()


def strategy_numba_status(*, warmed: Optional[bool] = None) -> dict[str, Any]:
    signatures = {
        dispatcher.py_func.__name__: [str(signature) for signature in dispatcher.nopython_signatures]
        for dispatcher in STRATEGY_NUMBA_KERNELS
    }
    is_warmed = _STRATEGY_WARMED if warmed is None else bool(warmed)
    fingerprints = {
        dispatcher.py_func.__name__: _strategy_kernel_fingerprint(dispatcher)
        for dispatcher in STRATEGY_NUMBA_KERNELS
    }
    aggregate_fingerprint = hashlib.sha256(
        "|".join(f"{name}:{fingerprints[name]}" for name in sorted(fingerprints)).encode("utf-8")
    ).hexdigest()
    return {
        "version": STRATEGY_NUMBA_KERNEL_VERSION,
        "kernel_version": STRATEGY_NUMBA_KERNEL_VERSION,
        "numba_version": numba.__version__,
        "warmed": is_warmed,
        "fully_warmed": is_warmed and all(signatures.values()),
        "kernel_coverage": f"{sum(bool(value) for value in signatures.values())}/{len(signatures)}",
        "kernel_signatures": signatures,
        "compiled_signatures": signatures,
        "kernel_fingerprints": fingerprints,
        "fingerprint": aggregate_fingerprint,
        "backend": "numba_njit_fixed_signature",
        "execution_backend": "numba_njit_fixed_signature",
        "nopython": all(bool(dispatcher.nopython_signatures) for dispatcher in STRATEGY_NUMBA_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
    }


@lru_cache(maxsize=1)
def warm_strategy_numba_kernels() -> dict[str, Any]:
    global _STRATEGY_WARMED
    nav = np.ascontiguousarray(
        [[1.0, 1.0], [1.01, 0.995], [1.02, 1.005], [1.015, 1.012]],
        dtype=np.float64,
    )
    _, equal_status = equal_weights_kernel(np.int64(2))
    if equal_status != 0:
        raise RuntimeError("strategy equal-weight NJIT warmup failed")
    normalized, normalize_status = normalize_explicit_weights_kernel(
        np.ascontiguousarray([2.0, 1.0], dtype=np.float64)
    )
    if normalize_status != 0:
        raise RuntimeError("strategy explicit-weight NJIT warmup failed")
    _, scale_status = scale_weights_percent_kernel(normalized, np.float64(0.2))
    if scale_status != 0:
        raise RuntimeError("strategy weight-scaling NJIT warmup failed")
    returns, status = nav_to_returns_kernel(nav)
    if status != 0:
        raise RuntimeError("strategy NAV-to-return NJIT warmup failed")
    budgets = np.ascontiguousarray([0.5, 0.5], dtype=np.float64)
    _, risk_status, _, _ = risk_budget_weights_kernel(returns, budgets, np.int64(10000), np.float64(1e-11))
    if risk_status != 0:
        raise RuntimeError(f"strategy risk-budget NJIT warmup failed: {risk_status}")
    if any(len(dispatcher.nopython_signatures) != 1 for dispatcher in STRATEGY_NUMBA_KERNELS):
        raise RuntimeError("strategy NJIT kernels must each expose exactly one fixed signature")
    _STRATEGY_WARMED = True
    optimizer_audit = warm_optimizer_numba_kernels()
    strategy_audit = validate_execution_audit(strategy_numba_status())
    validate_execution_graph(strategy_audit, optimizer_audit)
    strategy_audit["dependencies"] = [optimizer_audit]
    return strategy_audit


def strategy_execution_audit() -> dict[str, Any]:
    audit = validate_execution_audit(strategy_numba_status())
    audit["dependencies"] = [validate_execution_audit(optimizer_numba_status())]
    return audit


def equal_weights(asset_count: int) -> List[float]:
    weights, status = equal_weights_kernel(np.int64(asset_count))
    if status != 0:
        raise ValueError("等权策略至少需要一个资产")
    strategy_execution_audit()
    return [float(value) for value in weights]


def normalize_explicit_weights(raw_weights: List[float] | np.ndarray) -> List[float]:
    values = np.ascontiguousarray(raw_weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("固定权重必须是一维数组")
    normalized, status = normalize_explicit_weights_kernel(values)
    if status == 1:
        raise ValueError("固定权重至少需要一个资产")
    if status == 2:
        raise ValueError("固定权重必须是有限的非负数")
    if status == 3:
        raise ValueError("固定权重总和必须大于 0，禁止以等权代替")
    strategy_execution_audit()
    return [float(value) for value in normalized]


def scale_weights_percent(
    normalized_weights: List[float] | np.ndarray,
    max_leverage: float,
) -> List[float]:
    values = np.ascontiguousarray(normalized_weights, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("权重必须是一维数组")
    scaled, status = scale_weights_percent_kernel(values, np.float64(max_leverage))
    if status == 1:
        raise ValueError("权重至少需要一个资产")
    if status == 2:
        raise ValueError("最大杠杆必须是有限的非负数")
    if status == 3:
        raise ValueError("权重必须是有限的非负数")
    if status == 4:
        raise ValueError("杠杆缩放前的权重和必须为 1")
    strategy_execution_audit()
    return [float(value) for value in scaled]


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Expect DatetimeIndex for NAV/returns frame')
    return df.sort_index()


def _to_returns(nav_wide: pd.DataFrame) -> pd.DataFrame:
    nav_wide = _ensure_datetime_index(nav_wide)
    values = np.ascontiguousarray(nav_wide.to_numpy(dtype=np.float64))
    returns, status = nav_to_returns_kernel(values)
    if status == 1:
        raise ValueError("净值数据至少需要两个观察值和一个资产")
    if status == 2:
        raise ValueError("净值数据包含缺失、非有限或非正值")
    return pd.DataFrame(returns, index=nav_wide.index[1:], columns=nav_wide.columns)


def _risk_parity_weights(returns: pd.DataFrame, budgets: Optional[List[float]] = None) -> np.ndarray:
    if budgets is None:
        raise ValueError("必须逐项提供风险预算，禁止以等权静默代替")
    values = np.ascontiguousarray(returns.to_numpy(dtype=np.float64))
    budget_values = np.ascontiguousarray(budgets, dtype=np.float64)
    if budget_values.ndim != 1:
        raise ValueError("风险预算必须是一维数组")
    weights, status, _, error = risk_budget_weights_kernel(
        values, budget_values, np.int64(10000), np.float64(1e-11)
    )
    if status == 1:
        raise ValueError("风险预算至少需要两个收益观察值和一个资产")
    if status == 2:
        raise ValueError("风险预算必须非负、总和大于 0，且数量与资产一致")
    if status == 3:
        raise ValueError("风险预算协方差矩阵无效，无法求解")
    if status == 4:
        raise ValueError("风险预算求解未在最大迭代次数内收敛")
    if status == 5:
        raise ValueError(f"风险预算贡献误差超限：{float(error):.6g}")
    strategy_execution_audit()
    return np.ascontiguousarray(weights, dtype=np.float64)


def compute_risk_budget_weights(nav_wide: pd.DataFrame, risk_cfg: Dict[str, Any], budgets: List[float], *, window_len: Optional[int] = None, window_mode: Optional[str] = None) -> List[float]:
    nav_wide = _ensure_datetime_index(nav_wide)
    if window_len and window_len > 0:
        # 取消 firstN 固定窗口：仅支持 all 与 rollingN
        nav_wide = nav_wide.tail(max(2, window_len))
    returns = _to_returns(nav_wide)
    w = _risk_parity_weights(returns, budgets)
    return [float(x) for x in w]


def compute_target_weights(
    nav_wide: pd.DataFrame,
    return_cfg: Dict[str, Any],
    risk_cfg: Dict[str, Any],
    target: str,
    *,
    window_len: Optional[int] = None,
    window_mode: Optional[str] = None,
    single_limits: Optional[List[Tuple[float, float]]] = None,
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]] = None,
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    target_risk: Optional[float] = None,
    use_exploration: bool = True,
) -> List[float]:
    nav_wide = _ensure_datetime_index(nav_wide)
    if window_len and window_len > 0:
        nav_wide = nav_wide.tail(max(2, window_len))
    returns = _to_returns(nav_wide)
    if returns.shape[1] == 0:
        raise ValueError("收益矩阵没有可优化资产")
    weights = select_target_weights(
        np.ascontiguousarray(returns.to_numpy(dtype=np.float64)),
        return_cfg,
        risk_cfg,
        target,
        single_limits=single_limits,
        group_limits=group_limits,
        risk_free_rate=risk_free_rate,
        target_return=target_return,
        target_risk=target_risk,
        candidate_count=5000,
        seed=42,
    )
    strategy_execution_audit()
    return [float(value) for value in weights]


def _gen_rebalance_dates(index: pd.DatetimeIndex, mode: str, N: Optional[int] = None, which: Optional[str] = None, unit: Optional[str] = None, fixed_interval: Optional[int] = None) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
    try:
        trading_idx = get_trading_days(idx[0], idx[-1], exchange="SSE")
    except Exception:
        trading_idx = pd.DatetimeIndex([])
    if mode == 'fixed':
        k = int(fixed_interval or 20)
        return list(idx[::k])
    # Group by week/month/year
    if mode == 'weekly':
        key = idx.to_period('W')
    elif mode == 'monthly':
        key = idx.to_period('M')
    elif mode == 'yearly':
        key = idx.to_period('Y')
    else:
        return list(idx)

    groups = {}
    for t, p in zip(idx, key):
        groups.setdefault(p, []).append(t)

    out: List[pd.Timestamp] = []
    N = int(N or 1)
    which = (which or 'nth').lower()  # 'nth'|'first'|'last'
    unit = (unit or 'trading').lower()  # 'trading'|'natural'
    for _, arr in groups.items():
        arr = sorted(arr)
        if which == 'first':
            out.append(arr[0])
        elif which == 'last':
            out.append(arr[-1])
        else:
            if unit == 'natural':
                base = arr[0]
                cand = base + pd.Timedelta(days=N - 1)
                pick = next((t for t in arr if t >= cand), arr[-1])
                out.append(pick)
            else:
                period_start = pd.Timestamp(arr[0]).normalize()
                period_end = pd.Timestamp(arr[-1]).normalize()
                if trading_idx.size:
                    mask = trading_idx[(trading_idx >= period_start) & (trading_idx <= period_end)]
                else:
                    mask = pd.DatetimeIndex([])
                if mask.size:
                    t_index = min(max(N - 1, 0), mask.size - 1)
                    target = mask[t_index]
                    pick = next((t for t in arr if t >= target), arr[-1])
                    out.append(pick)
                else:
                    idxn = min(max(N - 1, 0), len(arr) - 1)
                    out.append(arr[idxn])
    return out


def backtest_portfolio(nav_wide: pd.DataFrame, strategies: List[Dict[str, Any]], start_date: Optional[str] = None) -> Dict[str, Any]:
    """Compatibility entrypoint backed by the fixed-signature NJIT engine."""

    try:
        from backend.backtest_engine import backtest_portfolio as njit_backtest_portfolio
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from backtest_engine import backtest_portfolio as njit_backtest_portfolio

    return njit_backtest_portfolio(nav_wide, strategies, start_date=start_date)
