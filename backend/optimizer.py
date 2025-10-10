
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize

# --- 收益指标计算 ---

@jit(nopython=True)
def get_log_returns(nav_series: np.ndarray) -> np.ndarray:
    return np.log(nav_series[1:] / nav_series[:-1])

@jit(nopython=True)
def get_simple_returns(nav_series: np.ndarray) -> np.ndarray:
    return nav_series[1:] / nav_series[:-1] - 1

def calculate_return(returns: np.ndarray, config: Dict[str, Any]) -> float:
    metric = config.get('metric')
    if metric == 'mean':
        return np.mean(returns)
    elif metric == 'cumulative':
        return np.sum(returns)
    elif metric == 'annual' or metric == 'annual_mean':
        days = int(config.get('days', 252))
        return np.mean(returns) * days
    elif metric == 'ewm':
        alpha = float(config.get('alpha', 0.94))
        # Numba 不支持 ewm，这里在外部用 pandas 计算然后传入
        # 此处仅为示意，实际 EWM 计算在主函数中处理
        return np.mean(returns) # Placeholder
    return 0.0

# --- 风险指标计算 ---

def calculate_risk(returns: np.ndarray, config: Dict[str, Any]) -> float:
    metric = config.get('metric')
    if metric == 'vol':
        return np.std(returns, ddof=1)
    elif metric == 'annual_vol':
        days = int(config.get('days', 252))
        return np.std(returns, ddof=1) * np.sqrt(days)
    elif metric == 'ewm_vol':
        # 指数加权波动率：使用衰减因子 λ (默认0.94)。pandas ewm 的 alpha = 1-λ。
        try:
            lam = float(config.get('alpha', 0.94))
        except Exception:
            lam = 0.94
        alpha = max(min(1.0 - lam, 0.9999), 1e-6)
        w = int(config.get('window', 60) or 0)
        s = pd.Series(returns if w <= 0 else returns[-max(2, w):])
        # 使用方差再开方，避免负数；bias=False 使用样本方差
        var = s.ewm(alpha=alpha, adjust=False).var(bias=False).iloc[-1]
        vol = float(np.sqrt(max(0.0, float(var))))
        days = int(config.get('days', 252))
        return vol * np.sqrt(days)
    elif metric == 'downside_vol':
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2:
            return 0.0
        return np.std(downside_returns, ddof=1)
    elif metric == 'var' or metric == 'es':
        confidence = float(config.get('confidence', 95)) / 100.0
        q = np.quantile(returns, 1 - confidence)
        if metric == 'var':
            return -q # VaR 为正数
        else: # ES
            return -np.mean(returns[returns <= q])
    elif metric == 'max_drawdown':
        nav = np.concatenate((np.array([1.0]), 1.0 + returns)).cumprod()
        peak = np.maximum.accumulate(nav)
        drawdown = (nav - peak) / peak
        return -np.min(drawdown)
    return 0.0

# --- Numba 加速的组合计算 ---

@jit(nopython=True)
def compute_portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> tuple[float, float]:
    port_return = np.sum(mean_returns * weights)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_vol

@jit(nopython=True)
def generate_random_portfolios(n_portfolios: int, n_assets: int, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
    results = np.zeros((2, n_portfolios))
    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        
        port_return = np.sum(mean_returns * weights)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        results[0, i] = port_vol
        results[1, i] = port_return
    return results

# --- 主函数：计算有效前沿 ---

def calculate_efficient_frontier(asset_returns: pd.DataFrame, return_config: Dict[str, Any], risk_config: Dict[str, Any], n_portfolios: int = 10000, risk_free_rate: float = 0.0):
    
    # 1. 计算每个资产的预期收益和风险 (转换为 numpy)
    mean_returns_annual_np = (asset_returns.mean() * 252).to_numpy()
    cov_matrix_annual_np = (asset_returns.cov() * 252).to_numpy()
    n_assets = len(mean_returns_annual_np)

    # 2. 蒙特卡洛模拟 (可配置空间)
    asset_names = list(asset_returns.columns)
    scatter_points = []
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        portfolio_daily_returns = asset_returns.dot(weights)
        
        ret_val = calculate_return(portfolio_daily_returns.to_numpy(), return_config)
        risk_val = calculate_risk(portfolio_daily_returns.to_numpy(), risk_config)
        # 仅返回坐标，避免载荷过大
        scatter_points.append({"value": (float(risk_val), float(ret_val))})

    # 3. 优化求解 (有效前沿)
    def minimize_volatility(weights):
        return compute_portfolio_performance(weights, mean_returns_annual_np, cov_matrix_annual_np)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # 使用简化的年化收益率作为优化目标
    frontier_target_returns = np.linspace(mean_returns_annual_np.min(), mean_returns_annual_np.max(), 100)
    frontier_points = []

    for r in frontier_target_returns:
        cons_with_ret = (constraints, {'type': 'eq', 'fun': lambda x: compute_portfolio_performance(x, mean_returns_annual_np, cov_matrix_annual_np)[0] - r})
        result = minimize(minimize_volatility, n_assets * [1. / n_assets], method='SLSQP', bounds=bounds, constraints=cons_with_ret)
        if result.success:
            # 使用优化得到的权重，重新计算前端需要的指标
            optimal_weights = result.x
            portfolio_daily_returns = asset_returns.dot(optimal_weights).to_numpy()
            risk_val = calculate_risk(portfolio_daily_returns, risk_config)
            ret_val = calculate_return(portfolio_daily_returns, return_config)
            frontier_points.append({
                "value": (float(risk_val), float(ret_val)),
                "weights": [float(w) for w in optimal_weights],
            })

    # 4. 找到特殊点 (最大夏普, 最小方差)
    def neg_sharpe_ratio(weights):
        p_return, p_vol = compute_portfolio_performance(weights, mean_returns_annual_np, cov_matrix_annual_np)
        # 避免除以零
        if p_vol < 1e-8: return np.inf
        return -(p_return - risk_free_rate) / p_vol

    max_sharpe_res = minimize(neg_sharpe_ratio, n_assets * [1. / n_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    max_sharpe_weights = max_sharpe_res.x
    
    min_vol_res = minimize(minimize_volatility, n_assets * [1. / n_assets], method='SLSQP', bounds=bounds, constraints=constraints)
    min_vol_weights = min_vol_res.x

    # 转换特殊点的坐标为前端选择的指标
    max_sharpe_daily_ret = asset_returns.dot(max_sharpe_weights).to_numpy()
    max_sharpe_point = {
        "value": (
            float(calculate_risk(max_sharpe_daily_ret, risk_config)),
            float(calculate_return(max_sharpe_daily_ret, return_config)),
        ),
        "weights": [float(w) for w in max_sharpe_weights],
    }

    min_vol_daily_ret = asset_returns.dot(min_vol_weights).to_numpy()
    min_vol_point = {
        "value": (
            float(calculate_risk(min_vol_daily_ret, risk_config)),
            float(calculate_return(min_vol_daily_ret, return_config)),
        ),
        "weights": [float(w) for w in min_vol_weights],
    }

    return {
        "asset_names": asset_names,
        "scatter": scatter_points,
        "frontier": sorted(frontier_points, key=lambda o: o["value"][0]),
        "max_sharpe": max_sharpe_point,
        "min_variance": min_vol_point,
    }


# ---------------- Exploration-based efficient frontier (random walks) ---------------- #

def _quantize_weights(w: np.ndarray, step: Optional[float]) -> np.ndarray:
    if not step or step <= 0:
        return w
    q = np.round(w / step) * step
    q = np.maximum(q, 0.0)
    s = q.sum()
    if s <= 0:
        # fallback to uniform
        q = np.full_like(w, 1.0 / len(w))
    else:
        q = q / s
    return q


def _repair_single_bounds(w: np.ndarray, single_limits: List[Tuple[float, float]]) -> Optional[np.ndarray]:
    n = len(w)
    lo = np.array([float(a) for a, _ in single_limits], dtype=float)
    hi = np.array([float(b) for _, b in single_limits], dtype=float)
    if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
        return None  # infeasible
    # clamp
    w = np.minimum(np.maximum(w, lo), hi)
    # redistribute remaining mass to free variables proportionally to (hi - w)
    deficit = 1.0 - w.sum()
    for _ in range(10):
        if abs(deficit) <= 1e-12:
            break
        if deficit > 0:
            room = hi - w
            room[room < 1e-16] = 0.0
            total_room = room.sum()
            if total_room <= 1e-16:
                # spread equally among feasibles
                idx = room > 0
                if not np.any(idx):
                    return None
                w[idx] += deficit / float(idx.sum())
            else:
                w += deficit * (room / total_room)
        else:  # excess mass
            excess = w - lo
            excess[excess < 1e-16] = 0.0
            total_ex = excess.sum()
            if total_ex <= 1e-16:
                return None
            w -= (-deficit) * (excess / total_ex)
        w = np.minimum(np.maximum(w, lo), hi)
        deficit = 1.0 - w.sum()
    if not (np.all(w >= lo - 1e-8) and np.all(w <= hi + 1e-8) and abs(w.sum() - 1.0) < 1e-6):
        return None
    return w


def _check_group_limits(w: np.ndarray, group_limits: Dict[Tuple[int, ...], Tuple[float, float]]) -> bool:
    for idxs, (lo, hi) in group_limits.items():
        s = float(np.sum(w[list(idxs)]))
        if s < lo - 1e-9 or s > hi + 1e-9:
            return False
    return True


def _random_weight(n: int, single_limits: List[Tuple[float, float]], group_limits: Dict[Tuple[int, ...], Tuple[float, float]], rng: np.random.RandomState) -> Optional[np.ndarray]:
    # Try up to some attempts
    for _ in range(200):
        w = rng.dirichlet(np.ones(n)).astype(float)
        w = _repair_single_bounds(w, single_limits)
        if w is None:
            return None
        if _check_group_limits(w, group_limits):
            return w
    return None


def _walk_from(seed: np.ndarray, step: float, single_limits: List[Tuple[float, float]], group_limits: Dict[Tuple[int, ...], Tuple[float, float]], rng: np.random.RandomState) -> Optional[np.ndarray]:
    n = len(seed)
    noise = rng.dirichlet(np.ones(n)).astype(float)
    w = (1.0 - step) * seed + step * noise
    w = _repair_single_bounds(w, single_limits)
    if w is None:
        return None
    if not _check_group_limits(w, group_limits):
        return None
    return w


def calculate_efficient_frontier_exploration(
    asset_returns: pd.DataFrame,
    return_config: Dict[str, Any],
    risk_config: Dict[str, Any],
    *,
    single_limits: Optional[List[Tuple[float, float]]] = None,
    group_limits: Optional[Dict[Tuple[int, ...], Tuple[float, float]]] = None,
    rounds: Optional[List[Dict[str, Any]]] = None,
    quantize_step: Optional[float] = None,
    use_slsqp_refine: bool = False,
    refine_count: int = 0,
    risk_free_rate: float = 0.0,
    seed: int = 42,
):
    rng = np.random.RandomState(int(seed))
    asset_names = list(asset_returns.columns)
    n_assets = len(asset_names)
    # Defaults
    if single_limits is None:
        single_limits = [(0.0, 1.0) for _ in range(n_assets)]
    if group_limits is None:
        group_limits = {}
    if not rounds:
        rounds = [{"samples": 100, "step": 0.99, "buckets": 50}]

    # Collect all candidates across rounds
    candidates: List[Tuple[np.ndarray, float, float]] = []  # (w, risk, ret)

    # Round 0: random feasible weights
    seeds: List[np.ndarray] = []
    r0 = rounds[0]
    for _ in range(int(r0.get("samples", 100))):
        w = _random_weight(n_assets, single_limits, group_limits, rng)
        if w is None:
            continue
        w = _quantize_weights(w, quantize_step)
        Rt = asset_returns.to_numpy() @ w
        r = calculate_return(Rt, return_config)
        v = calculate_risk(Rt, risk_config)
        seeds.append(w)
        candidates.append((w, float(v), float(r)))

    # Subsequent rounds: local random walks with bucketing selection
    for ridx, rconf in enumerate(rounds):
        if ridx == 0:
            continue
        samples = int(rconf.get("samples", 100))
        step = float(rconf.get("step", 0.5))
        buckets = max(1, int(rconf.get("buckets", 50)))
        new_cands: List[Tuple[np.ndarray, float, float]] = []
        if not seeds:
            break
        for _ in range(samples):
            base = seeds[rng.randint(0, len(seeds))]
            w = _walk_from(base, step, single_limits, group_limits, rng)
            if w is None:
                continue
            w = _quantize_weights(w, quantize_step)
            Rt = asset_returns.to_numpy() @ w
            r = calculate_return(Rt, return_config)
            v = calculate_risk(Rt, risk_config)
            new_cands.append((w, float(v), float(r)))
        if not new_cands:
            continue
        # Bucketing by return
        rets = np.array([c[2] for c in new_cands])
        risks = np.array([c[1] for c in new_cands])
        rmin, rmax = float(rets.min()), float(rets.max())
        if rmax <= rmin:
            keep_idx = np.argsort(risks)[: min(len(new_cands), buckets)]
        else:
            bins = np.linspace(rmin, rmax, buckets + 1)
            digit = np.digitize(rets, bins) - 1
            keep_idx = []
            for b in range(buckets):
                idx = np.where(digit == b)[0]
                if idx.size == 0:
                    continue
                # pick min risk in bucket
                i = idx[np.argmin(risks[idx])]
                keep_idx.append(int(i))
        seeds = [new_cands[i][0] for i in keep_idx]
        for i in keep_idx:
            candidates.append(new_cands[i])

    if not candidates:
        return {
            "asset_names": asset_names,
            "scatter": [],
            "frontier": [],
            "max_sharpe": None,
            "min_variance": None,
        }

    # Build arrays
    Ws = np.vstack([c[0] for c in candidates])
    risks = np.array([c[1] for c in candidates], dtype=float)
    rets = np.array([c[2] for c in candidates], dtype=float)

    # Optional SLSQP refine using risk grid around current risk range
    if use_slsqp_refine and refine_count > 0:
        try:
            from scipy.optimize import minimize
            # bounds and constraints
            bounds = single_limits
            cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
            for idx_tuple, (lo, hi) in (group_limits or {}).items():
                idx = list(idx_tuple)
                cons += (
                    {'type': 'ineq', 'fun': lambda w, i=idx, l=lo: np.sum(w[i]) - l},
                    {'type': 'ineq', 'fun': lambda w, i=idx, h=hi: h - np.sum(w[i])},
                )

            Rt_mat = asset_returns.to_numpy()

            def risk_of(w: np.ndarray) -> float:
                return calculate_risk(Rt_mat @ w, risk_config)

            def neg_ret(w: np.ndarray) -> float:
                return -calculate_return(Rt_mat @ w, return_config)

            rmin, rmax = float(np.min(risks)), float(np.max(risks))
            grid = np.linspace(rmin, rmax, int(max(2, refine_count)))
            refined = []
            for rt in grid:
                # hot start: nearest candidate in risk
                j = int(np.argmin(np.abs(risks - rt)))
                w0 = Ws[j]
                cons_rt = cons + ({'type': 'ineq', 'fun': lambda w, t=rt: t - risk_of(w)},)
                res = minimize(neg_ret, w0, method='SLSQP', bounds=bounds, constraints=cons_rt, options={'maxiter': 300, 'ftol': 1e-9})
                if res.success:
                    w = res.x
                    r = risk_of(w)
                    a = calculate_return(Rt_mat @ w, return_config)
                    refined.append({"value": (float(r), float(a)), "weights": [float(x) for x in w]})
            # refined remains as separate list; to be merged below
            pass
        except Exception:
            pass

    # Merge all points: candidates + refined (if any)
    all_points: List[Dict[str, Any]] = [
        {"value": (float(risks[i]), float(rets[i])), "weights": [float(x) for x in Ws[i]]}
        for i in range(len(candidates))
    ]
    try:
        refined  # type: ignore[name-defined]
        if isinstance(refined, list) and refined:
            all_points.extend(refined)  # refined items already in {value, weights} form
    except NameError:
        pass

    # Scatter: include all points (no filtering)
    scatter = list(all_points)

    # Recompute arrays over all_points for metrics
    all_risks = np.array([p["value"][0] for p in all_points], dtype=float)
    all_rets = np.array([p["value"][1] for p in all_points], dtype=float)

    # Frontier by risk rounding to 4 decimals: for each rounded risk, keep max return point
    risk_rounded = np.round(all_risks, 4)
    best_idx: Dict[float, int] = {}
    for i, r4 in enumerate(risk_rounded):
        if r4 not in best_idx or all_rets[i] > all_rets[best_idx[r4]]:
            best_idx[r4] = i
    # Enforce strictly increasing return along increasing risk
    frontier: List[Dict[str, Any]] = []
    last_ret = -np.inf
    for r4, idx in sorted(best_idx.items(), key=lambda kv: kv[0]):
        cur_ret = all_rets[idx]
        if cur_ret > last_ret:
            frontier.append(all_points[idx])
            last_ret = cur_ret

    # Max Sharpe and Min Variance among all points
    sharpe = np.full(len(all_points), -np.inf)
    mask_pos = all_risks > 1e-12
    sharpe[mask_pos] = (all_rets[mask_pos] - float(risk_free_rate)) / all_risks[mask_pos]
    ms_idx = int(np.argmax(sharpe)) if np.any(mask_pos) else int(np.argmin(all_risks))
    mv_idx = int(np.argmin(all_risks))

    max_sharpe = all_points[ms_idx]
    min_var = all_points[mv_idx]

    # Max return point
    mr_idx = int(np.argmax(all_rets))
    max_return = all_points[mr_idx]

    return {
        "asset_names": asset_names,
        "scatter": scatter,
        "frontier": frontier,
        "max_sharpe": max_sharpe,
        "min_variance": min_var,
        "max_return": max_return,
    }
