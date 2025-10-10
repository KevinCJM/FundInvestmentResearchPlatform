from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimizer import calculate_risk, calculate_return, calculate_efficient_frontier_exploration


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Expect DatetimeIndex for NAV/returns frame')
    return df.sort_index()


def _to_returns(nav_wide: pd.DataFrame) -> pd.DataFrame:
    nav_wide = _ensure_datetime_index(nav_wide)
    rets = nav_wide.pct_change().dropna()
    return rets


def _risk_parity_weights(returns: pd.DataFrame, budgets: Optional[List[float]] = None) -> np.ndarray:
    """Risk parity via covariance fixed-point iteration, supporting target budgets."""
    X = returns.to_numpy()
    n = X.shape[1]
    if X.shape[0] < 3 or n == 0:
        return np.full(max(n, 1), 1.0 / max(n, 1))
    S = np.cov(X, rowvar=False, ddof=1)
    eps = 1e-12
    if budgets is None or sum(budgets) <= 0:
        b = np.ones(n, dtype=float) / n
    else:
        b = np.array([max(0.0, float(x)) for x in budgets], dtype=float)
        if b.sum() <= 0:
            b = np.ones(n, dtype=float) / n
        else:
            b = b / b.sum()
    # numerical regularization if needed
    try:
        np.linalg.cholesky(S + 1e-12 * np.eye(n))
    except np.linalg.LinAlgError:
        S = S + 1e-6 * np.eye(n)
    w = np.maximum(b.copy(), eps)
    w /= w.sum()
    for _ in range(5000):
        Sw = S @ w
        denom = np.maximum(Sw, eps)
        w_new = b / denom
        w_new = np.maximum(w_new, eps)
        w_new /= w_new.sum()
        if np.linalg.norm(w_new - w, ord=1) < 1e-8:
            w = w_new
            break
        w = 0.7 * w_new + 0.3 * w
    return w / max(1e-12, w.sum())


def compute_risk_budget_weights(nav_wide: pd.DataFrame, risk_cfg: Dict[str, Any], budgets: List[float], *, window_len: Optional[int] = None, window_mode: Optional[str] = None) -> List[float]:
    nav_wide = _ensure_datetime_index(nav_wide)
    if window_len and window_len > 0:
        # 取消 firstN 固定窗口：仅支持 all 与 rollingN
        nav_wide = nav_wide.tail(max(2, window_len))
    returns = _to_returns(nav_wide)
    # 目前采用基于协方差的风险平价，支持目标预算
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
    # Simple targets: min_risk, max_return. Extendable.
    nav_wide = _ensure_datetime_index(nav_wide)
    if window_len and window_len > 0:
        nav_wide = nav_wide.tail(max(2, window_len))
    returns = _to_returns(nav_wide)
    X = returns.to_numpy()
    n = X.shape[1]
    if n == 0:
        return []
    # bounds and constraints
    n = X.shape[1]
    if single_limits is None:
        single_limits = [(0.0, 1.0) for _ in range(n)]
    bounds = tuple((float(a), float(b)) for a, b in single_limits)
    cons: Tuple[dict, ...] = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    for idx_tuple, (lo, hi) in (group_limits or {}).items():
        idx = list(idx_tuple)
        cons += (
            {'type': 'ineq', 'fun': lambda w, i=idx, l=lo: np.sum(w[i]) - l},
            {'type': 'ineq', 'fun': lambda w, i=idx, h=hi: h - np.sum(w[i])},
        )

    results = None
    if use_exploration:
        results = calculate_efficient_frontier_exploration(
            asset_returns=returns,
            return_config=return_cfg,
            risk_config=risk_cfg,
            single_limits=single_limits,
            group_limits=group_limits or {},
            rounds=[{"samples": 1500, "step": 0.5, "buckets": 30}, {"samples": 2500, "step": 0.25, "buckets": 40}],
            quantize_step=None,
            use_slsqp_refine=True,
            refine_count=20,
            risk_free_rate=risk_free_rate,
        )

    if target == 'min_risk':
        if use_exploration and results:
            if results.get('min_variance') and results['min_variance'].get('weights'):
                return [float(x) for x in results['min_variance']['weights']]
            pts = results.get('scatter', []) + results.get('frontier', [])
            if pts:
                idx = int(np.argmin([p['value'][0] for p in pts]))
                return [float(x) for x in pts[idx].get('weights', [1.0/n]*n)]
        # fast path: SLSQP directly minimize risk under constraints
        from scipy.optimize import minimize
        def risk_of(w: np.ndarray) -> float:
            return calculate_risk(X @ w, risk_cfg)
        res = minimize(lambda w: risk_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':400,'ftol':1e-9})
        w = res.x if res.success else np.full(n, 1.0/n)
        return [float(x) for x in w / max(1e-12, w.sum())]
    elif target == 'max_return':
        if use_exploration and results:
            pts = results.get('scatter', []) + results.get('frontier', [])
            if pts:
                idx = int(np.argmax([p['value'][1] for p in pts]))
                return [float(x) for x in pts[idx].get('weights', [1.0/n]*n)]
        # fast path: SLSQP directly maximize return
        from scipy.optimize import minimize
        def ret_of(w: np.ndarray) -> float:
            return calculate_return(X @ w, return_cfg)
        res = minimize(lambda w: -ret_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter':400,'ftol':1e-9})
        w = res.x if res.success else np.full(n, 1.0/n)
        return [float(x) for x in w / max(1e-12, w.sum())]
    elif target == 'max_sharpe_traditional':
        from scipy.optimize import minimize
        # This target uses a fixed traditional Sharpe Ratio definition.
        days = int(return_cfg.get('days', 252)) # Still need days for annualization
        
        def neg_traditional_sharpe(w: np.ndarray) -> float:
            p = X @ w
            # Hardcoded traditional metrics
            r = calculate_return(p, {'metric': 'annual', 'days': days})
            v = calculate_risk(p, {'metric': 'annual_vol', 'days': days})
            if v <= 1e-12:
                return 1e6
            # Traditional formula with risk-free rate
            return - (r - float(risk_free_rate)) / v
            
        w0 = np.full(n, 1.0/n)
        res = minimize(neg_traditional_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
        w = res.x if res.success else w0
        return [float(x) for x in w / max(1e-12, w.sum())]
    elif target == 'max_sharpe':
        from scipy.optimize import minimize
        def neg_sharpe(w: np.ndarray) -> float:
            p = X @ w
            r = calculate_return(p, return_cfg)
            v = calculate_risk(p, risk_cfg)
            if v <= 1e-12:
                return 1e6
            return - r / v
        w0 = np.full(n, 1.0/n)
        res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 500, 'ftol': 1e-9})
        w = res.x if res.success else w0
        return [float(x) for x in w / max(1e-12, w.sum())]
    elif target == 'risk_min_given_return':
        # minimize risk subject to return == target_return
        from scipy.optimize import minimize
        if target_return is None:
            raise ValueError('需要提供目标收益率')
        def risk_of(w: np.ndarray) -> float:
            return calculate_risk(X @ w, risk_cfg)
        def ret_of(w: np.ndarray) -> float:
            return calculate_return(X @ w, return_cfg)
        # find feasible range for return using SLSQP on bounds/constraints
        res_max = minimize(lambda w: -ret_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons)
        res_min = minimize(lambda w: ret_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons)
        rmax = ret_of(res_max.x) if res_max.success else None
        rmin = ret_of(res_min.x) if res_min.success else None
        if (rmax is None) or (rmin is None) or not (rmin - 1e-9 <= target_return <= rmax + 1e-9):
            raise ValueError(f'目标收益不在可行范围内 [{rmin:.6f}, {rmax:.6f}]')
        cons_rt = cons + ({'type': 'eq', 'fun': lambda w: ret_of(w) - float(target_return)},)
        res = minimize(lambda w: risk_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons_rt, options={'maxiter': 800, 'ftol': 1e-9})
        w = res.x if res.success else np.full(n, 1.0/n)
        return [float(x) for x in w / max(1e-12, w.sum())]
    elif target == 'return_max_given_risk':
        # maximize return subject to risk <= target_risk
        from scipy.optimize import minimize
        if target_risk is None:
            raise ValueError('需要提供目标风险值')
        def risk_of(w: np.ndarray) -> float:
            return calculate_risk(X @ w, risk_cfg)
        def ret_of(w: np.ndarray) -> float:
            return calculate_return(X @ w, return_cfg)
        # find feasible range for risk
        res_minr = minimize(lambda w: risk_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons)
        res_maxr = minimize(lambda w: -risk_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons)
        rmin = risk_of(res_minr.x) if res_minr.success else None
        rmax = risk_of(res_maxr.x) if res_maxr.success else None
        if (rmax is None) or (rmin is None) or not (rmin - 1e-9 <= target_risk <= rmax + 1e-9):
            raise ValueError(f'目标风险不在可行范围内 [{rmin:.6f}, {rmax:.6f}]')
        cons_rk = cons + ({'type': 'ineq', 'fun': lambda w: float(target_risk) - risk_of(w)},)
        res = minimize(lambda w: -ret_of(w), np.full(n, 1.0/n), method='SLSQP', bounds=bounds, constraints=cons_rk, options={'maxiter': 800, 'ftol': 1e-9})
        w = res.x if res.success else np.full(n, 1.0/n)
        return [float(x) for x in w / max(1e-12, w.sum())]
    else:
        return [float(x) for x in np.full(n, 1.0 / n)]


def _gen_rebalance_dates(index: pd.DatetimeIndex, mode: str, N: Optional[int] = None, which: Optional[str] = None, unit: Optional[str] = None, fixed_interval: Optional[int] = None) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
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
                # approximate: use calendar within the period
                base = arr[0]
                cand = base + pd.Timedelta(days=N - 1)
                # pick nearest trading day in arr not before cand
                pick = next((t for t in arr if t >= cand), arr[-1])
                out.append(pick)
            else:
                idxn = min(max(N - 1, 0), len(arr) - 1)
                out.append(arr[idxn])
    return out


def backtest_portfolio(nav_wide: pd.DataFrame, strategies: List[Dict[str, Any]], start_date: Optional[str] = None) -> Dict[str, Any]:
    """Backtest portfolio NAV series.
    - No rebal: invest initial proportions into each class, portfolio NAV = sum_i w_i * (NAV_i / NAV_i(start)).
    - With rebal: at each rebalance date, reset base to that date and invest current total NAV by weights.
    """
    nav_wide = _ensure_datetime_index(nav_wide)
    if start_date:
        nav_wide = nav_wide[nav_wide.index >= pd.to_datetime(start_date)]
    idx = nav_wide.index

    def portfolio_nav_static(nav: pd.DataFrame, weights: np.ndarray, rebal_dates: Optional[List[pd.Timestamp]] = None) -> pd.Series:
        weights = weights / max(1e-12, weights.sum())
        if not rebal_dates:
            base = nav.iloc[0]
            rel = nav.divide(base, axis=1)
            return (rel * weights).sum(axis=1)
        # ensure rebal dates on index and sorted
        rset = sorted([pd.Timestamp(d) for d in rebal_dates if d in nav.index])
        if not rset or rset[0] != nav.index[0]:
            rset = [nav.index[0]] + rset
        out = pd.Series(index=nav.index, dtype=float)
        cur_val = 1.0
        for i, d0 in enumerate(rset):
            d1 = rset[i + 1] if i + 1 < len(rset) else nav.index[-1]
            seg = nav.loc[d0:d1]
            base = seg.iloc[0]
            rel = seg.divide(base, axis=1)
            part = (rel * weights).sum(axis=1)
            out.loc[seg.index] = cur_val * part.values
            cur_val = float(out.loc[seg.index[-1]])
        return out

    series_out: Dict[str, List[float]] = {}
    for s in strategies:
        name = s.get('name') or s.get('type') or 'strategy'
        w = np.asarray(s.get('weights') or [], dtype=float)
        if w.size == 0:
            w = np.full(nav_wide.shape[1], 1.0 / max(1, nav_wide.shape[1]))
        rebal = s.get('rebalance') or {}
        rebal_enabled = bool(rebal.get('enabled', False))
        rebal_dates: Optional[List[pd.Timestamp]] = None
        if rebal_enabled:
            mode = str(rebal.get('mode', 'monthly'))
            which = str(rebal.get('which', 'nth'))
            N = int(rebal.get('N', 1))
            unit = str(rebal.get('unit', 'trading'))
            fixed_interval = int(rebal.get('fixedInterval', 20)) if mode == 'fixed' else None
            rebal_dates = _gen_rebalance_dates(nav_wide.index, mode, N=N, which=which, unit=unit, fixed_interval=fixed_interval)
        nav = portfolio_nav_static(nav_wide, w, rebal_dates=rebal_dates)
        series_out[name] = [float(x) for x in nav.values]
    return {"dates": [d.strftime('%Y-%m-%d') for d in idx], "series": series_out}
