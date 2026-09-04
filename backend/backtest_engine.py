from __future__ import annotations

"""
Backtest engine: static and rebalanced portfolio NAVs from class NAV wide table.

Public functions
- backtest_portfolio(nav_wide, strategies, start_date=None) -> {dates, series}
- gen_rebalance_dates(index, mode, N=None, which=None, unit=None, fixed_interval=None) -> list[pd.Timestamp]

Script usage
python backend/backtest_engine.py payload.json
  where payload.json matches backend/backtest_api_spec.json#backtest.request
"""

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from trading_calendar import get_trading_days

try:
    from backend.backtest_numba import (
        backtest_numba_execution_audit,
        annual_metrics_kernel,
        cumulative_returns_kernel,
        portfolio_metrics_kernel,
        portfolio_segment_path_kernel,
        uniform_weights_kernel,
    )
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from backtest_numba import (
        backtest_numba_execution_audit,
        annual_metrics_kernel,
        cumulative_returns_kernel,
        portfolio_metrics_kernel,
        portfolio_segment_path_kernel,
        uniform_weights_kernel,
    )


def slice_fit_data(nav: pd.DataFrame, up_to: pd.Timestamp, window_mode: Optional[str], data_len: Optional[int]) -> pd.DataFrame:
    """Return fitting window ending at ``up_to`` according to window_mode/data_len."""

    win = nav.loc[:up_to]
    mode = (window_mode or 'all').lower()
    if mode != 'rollingn':
        return win
    n = max(2, int(data_len or 0))
    return win.tail(n)


def ensure_valid_rebalance_window(
    nav: pd.DataFrame,
    candidate_dates: List[pd.Timestamp],
    model: Optional[Dict[str, Any]] = None,
) -> Tuple[List[pd.Timestamp], pd.Timestamp]:
    """Return valid rebalance dates and the first viable rebalance timestamp."""

    model = model or {}
    nav_sorted = nav.sort_index()
    window_mode = (model.get('window_mode') or 'all').lower()
    data_len = model.get('data_len')
    required = max(2, int(data_len or 0)) if window_mode == 'rollingn' else 2

    valid_dates: List[pd.Timestamp] = []
    first_idx: Optional[pd.Timestamp] = None
    for d in candidate_dates:
        if d not in nav_sorted.index:
            continue
        sub = nav_sorted.loc[:d]
        if window_mode == 'rollingn':
            if len(sub) < required:
                continue
        else:
            if len(sub) < 2:
                continue
        if first_idx is None:
            first_idx = d
        valid_dates.append(d)

    if not valid_dates or first_idx is None:
        nav_len = len(nav_sorted)
        if nav_len < required:
            raise ValueError('可用样本不足，无法计算任一调仓窗口，请检查 window_mode/data_len 配置。')
        fallback_idx = nav_sorted.index[required - 1]
        follow_ups = [d for d in candidate_dates if d > fallback_idx]
        deduped = []
        for d in [fallback_idx] + follow_ups:
            if d not in deduped:
                deduped.append(d)
        return deduped, fallback_idx

    valid_dates = [first_idx] + [d for d in valid_dates if d > first_idx]
    return valid_dates, first_idx


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Expect DatetimeIndex for NAV/returns frame')
    return df.sort_index()


def gen_rebalance_dates(
    index: pd.DatetimeIndex,
    mode: str,
    N: Optional[int] = None,
    which: Optional[str] = None,
    unit: Optional[str] = None,
    fixed_interval: Optional[int] = None,
) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
    try:
        trading_idx = get_trading_days(idx[0], idx[-1], exchange="SSE")
    except Exception:
        trading_idx = pd.DatetimeIndex([])
    if mode == 'fixed':
        k = int(fixed_interval or 20)
        k = max(1, k)
        return list(idx[::k])
    if mode not in {'weekly', 'monthly', 'yearly'}:
        return []
    if mode == 'weekly':
        key = idx.to_period('W')
    elif mode == 'monthly':
        key = idx.to_period('M')
    else:
        key = idx.to_period('Y')

    N = int(N or 1)
    which = (which or 'nth').lower()  # 'nth'|'first'|'last'
    unit = (unit or 'trading').lower()  # 'trading'|'natural'

    groups: Dict[pd.Period, List[pd.Timestamp]] = {}
    for t, p in zip(idx, key):
        groups.setdefault(p, []).append(t)

    out: List[pd.Timestamp] = []
    for _, arr in groups.items():
        arr = sorted(arr)
        if which == 'first':
            out.append(arr[0])
        elif which == 'last':
            out.append(arr[-1])
        else:
            # 'nth'
            if unit == 'natural':
                base = arr[0]
                target = base + pd.Timedelta(days=max(N, 1) - 1)
                pick = next((t for t in arr if t >= target), arr[-1])
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
                    i = min(max(N - 1, 0), len(arr) - 1)
                    out.append(arr[i])
    return out


def backtest_portfolio(
    nav_wide: pd.DataFrame,
    strategies: List[Dict[str, Any]],
    start_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Backtest portfolios.
    Strategy item: { name, weights: [..], rebalance?: {enabled, mode, which, N, unit, fixedInterval}}
    """
    nav_wide = _ensure_datetime_index(nav_wide)
    if start_date:
        nav_wide = nav_wide[nav_wide.index >= pd.to_datetime(start_date)]
    idx = nav_wide.index

    def _compute_model_weights(nav: pd.DataFrame, s: Dict[str, Any], up_to: pd.Timestamp) -> Optional[np.ndarray]:
        model = s.get('model') or {}
        data_len = model.get('data_len', None)
        window_mode = model.get('window_mode') or 'all'
        nav_fit = slice_fit_data(nav, up_to, window_mode, data_len)
        stype = s.get('type')
        if stype == 'risk_budget':
            # collect budgets from classes
            cls = s.get('classes') or []
            budgets = [float((c.get('budget') or 0.0)) for c in cls]
            risk_cfg = {k: model.get(k) for k in ('metric','days','window','confidence')}
            # normalize risk_cfg metric key
            if not risk_cfg.get('metric'):
                risk_cfg['metric'] = model.get('risk_metric', 'vol')
            w = compute_risk_budget_weights(nav_fit, risk_cfg, budgets, window_len=None)
            return np.asarray(w, dtype=float)
        if stype == 'target':
            ret_cfg = {
                'metric': model.get('return_metric') or 'annual',
                'days': int(model.get('days') or 252),
                'alpha': model.get('ret_alpha'),
                'window': model.get('ret_window'),
            }
            risk_cfg = {
                'metric': model.get('risk_metric') or 'vol',
                'days': model.get('risk_days'),
                'alpha': model.get('risk_alpha'),
                'window': model.get('risk_window'),
                'confidence': model.get('risk_confidence'),
            }
            # constraints mapping
            asset_names = list(nav_fit.columns)
            single_limits = []
            sl = (model.get('constraints') or {}).get('single_limits', {})
            for nm in asset_names:
                v = sl.get(nm, {})
                lo = float(v.get('lo', 0.0)) if isinstance(v, dict) else 0.0
                hi = float(v.get('hi', 1.0)) if isinstance(v, dict) else 1.0
                single_limits.append((lo, hi))
            group_limits: Dict[Tuple[int, ...], Tuple[float, float]] = {}
            for g in (model.get('constraints') or {}).get('group_limits', []) or []:
                assets = g.get('assets', [])
                idxs = tuple(i for i, nm in enumerate(asset_names) if nm in assets)
                if idxs:
                    group_limits[idxs] = (float(g.get('lo', 0.0)), float(g.get('hi', 1.0)))
            w = compute_target_weights(
                nav_fit,
                return_cfg=ret_cfg,
                risk_cfg=risk_cfg,
                target=str(model.get('target') or 'min_risk'),
                window_len=None,
                single_limits=single_limits,
                group_limits=group_limits,
                risk_free_rate=float(model.get('risk_free_rate') or 0.0),
                target_return=model.get('target_return'),
                target_risk=model.get('target_risk'),
            )
            return np.asarray(w, dtype=float)
        return None

    def _static_or_rebalanced(nav: pd.DataFrame, s: Dict[str, Any]):
        base_weights = np.asarray(s.get('weights') or [], dtype=float)
        rb = s.get('rebalance') or {}
        rebal_dates: List[pd.Timestamp] = []
        if rb.get('enabled'):
            mode = str(rb.get('mode', 'monthly'))
            which = str(rb.get('which', 'nth'))
            N = int(rb.get('N', 1))
            unit = str(rb.get('unit', 'trading'))
            fixed_interval = int(rb.get('fixedInterval', 20)) if mode == 'fixed' else None
            rebal_dates = gen_rebalance_dates(nav.index, mode, N=N, which=which, unit=unit, fixed_interval=fixed_interval)
        recalc = bool(rb.get('recalc', False))
        markers: List[Dict[str, Any]] = []
        nav_values = np.ascontiguousarray(nav.to_numpy(dtype=np.float64))
        raw_precomputed = s.get('precomputed_weights') or {}
        precomputed_lookup: Dict[pd.Timestamp, np.ndarray] = {}
        for key, value in raw_precomputed.items():
            ts = key if isinstance(key, pd.Timestamp) else pd.to_datetime(key)
            precomputed_lookup[ts] = np.asarray(value, dtype=float)
        if not rebal_dates:
            if nav_values.shape[0] == 0:
                return pd.Series(dtype=float, index=nav.index), []
            if base_weights.size != nav_values.shape[1]:
                raise ValueError('策略权重数量必须与资产数量一致。')
            series_np, _ = portfolio_segment_path_kernel(
                nav_values,
                np.ascontiguousarray(base_weights, dtype=np.float64),
                0,
                nav_values.shape[0] - 1,
                1.0,
                0.0,
            )
            series = pd.Series(series_np, index=nav.index, dtype=float)
            return series, []
        # ensure the first valid date has enough samples
        rset = sorted([d for d in rebal_dates if d in nav.index])
        if not rset:
            raise ValueError('未找到可用的调仓日期。')
        rset, first_idx = ensure_valid_rebalance_window(nav, rset, s.get('model'))
        full_nav = nav.sort_index()
        nav = full_nav.loc[first_idx:]
        nav_values_trim = np.ascontiguousarray(nav.to_numpy(dtype=np.float64))
        index_lookup = {ts: idx for idx, ts in enumerate(nav.index)}

        series_np = np.full(nav_values_trim.shape[0], np.nan, dtype=np.float64)
        current_val = 1.0
        for i, d0 in enumerate(rset):
            d1 = rset[i + 1] if i + 1 < len(rset) else nav.index[-1]
            start_idx = index_lookup[d0]
            end_idx = index_lookup[d1]
            w_seg = base_weights
            if recalc:
                w_calc = precomputed_lookup.get(d0)
                if w_calc is None:
                    history = full_nav.loc[:d0]
                    w_calc = _compute_model_weights(history, s, d0)
                if w_calc is not None:
                    w_seg = w_calc
            seg_index = slice(start_idx, end_idx + 1)
            segment_path, weights_full = portfolio_segment_path_kernel(
                nav_values_trim,
                np.ascontiguousarray(w_seg, dtype=np.float64),
                start_idx,
                end_idx,
                current_val,
                current_val,
            )
            marker_weights = [float(x) for x in weights_full]
            series_np[seg_index] = segment_path
            markers.append({
                'date': d0.date().isoformat(),
                'weights': marker_weights,
                'value': float(segment_path[0])
            })
            current_val = float(segment_path[-1])
        out = pd.Series(series_np, index=nav.index, dtype=float)
        return out, markers

    series_out: Dict[str, List[float]] = {}
    marker_out: Dict[str, List[Dict[str, Any]]] = {}
    for s in strategies:
        name = s.get('name') or s.get('type') or 'strategy'
        # normalize classes order to match nav_wide columns if provided as dicts
        if s.get('classes'):
            # align weights array from classes if available
            name_to_weight = {c.get('name'): c.get('weight') for c in s.get('classes')}
            w_arr = [float(name_to_weight.get(col, 0.0) or 0.0) for col in nav_wide.columns]
            s['weights'] = w_arr
        if not s.get('weights'):
            s['weights'] = uniform_weights_kernel(nav_wide.shape[1]).tolist()
        series, markers = _static_or_rebalanced(nav_wide, s)
        series_full = series.reindex(idx)
        series_out[name] = [None if pd.isna(x) else float(x) for x in series_full]
        marker_out[name] = markers

    def _safe_float(val: float) -> Optional[float]:
        return float(val) if np.isfinite(val) else None

    metrics_rows: List[Dict[str, Optional[float]]] = []
    ann_factor = 252.0
    strategy_names = list(series_out)
    nav_columns = [
        np.asarray(
            [np.nan if value is None else value for value in series_out[name]],
            dtype=np.float64,
        )
        for name in strategy_names
    ]
    nav_matrix = np.ascontiguousarray(
        np.column_stack(nav_columns)
        if nav_columns
        else np.empty((len(idx), 0), dtype=np.float64)
    )
    cumulative_values = cumulative_returns_kernel(nav_matrix)
    annual_years, annual_values = annual_metrics_kernel(
        nav_matrix,
        np.ascontiguousarray(idx.year.to_numpy(dtype=np.int64)),
        ann_factor,
    )
    annual_metric_names = (
        "cumulative",
        "volatility",
        "annualReturn",
        "annualVolatility",
        "sharpe",
        "maxDrawdown",
        "calmar",
    )
    annual_payload: Dict[str, Any] = {
        "years": [int(year) for year in annual_years],
        "series": {},
    }
    for strategy_index, name in enumerate(strategy_names):
        annual_payload["series"][name] = {}
        for year_index, year in enumerate(annual_years):
            base = strategy_index * len(annual_metric_names)
            annual_payload["series"][name][str(int(year))] = {
                metric_name: _safe_float(annual_values[year_index, base + metric_index])
                for metric_index, metric_name in enumerate(annual_metric_names)
            }

    for strategy_index, name in enumerate(strategy_names):
        nav_array = nav_matrix[:, strategy_index]
        metric_values = portfolio_metrics_kernel(nav_array, ann_factor)
        metrics_rows.append({
            "name": name,
            "cumulative_return": _safe_float(cumulative_values[strategy_index]),
            "annual_return": _safe_float(metric_values[0]),
            "annual_vol": _safe_float(metric_values[1]),
            "sharpe": _safe_float(metric_values[2]),
            "var99": _safe_float(metric_values[3]),
            "es99": _safe_float(metric_values[4]),
            "max_drawdown": _safe_float(metric_values[5]),
            "calmar": _safe_float(metric_values[6]),
        })

    return {
        "dates": [d.strftime('%Y-%m-%d') for d in idx],
        "series": series_out,
        "markers": marker_out,
        "asset_names": list(nav_wide.columns),
        "metrics": metrics_rows,
        "annual_metrics": annual_payload,
        "execution": backtest_numba_execution_audit(),
    }


def load_nav_wide_from_parquet(data_dir: Path, alloc_name: str) -> pd.DataFrame:
    p = data_dir / 'asset_nv.parquet'
    df = pd.read_parquet(p)
    df = df[df['asset_alloc_name'] == alloc_name]
    nav_wide = df.pivot_table(index='date', columns='asset_name', values='nv').sort_index()
    nav_wide.index = pd.to_datetime(nav_wide.index)
    return nav_wide


def run_from_payload(payload: Dict[str, Any], data_dir: Optional[Path] = None) -> Dict[str, Any]:
    data_dir = data_dir or Path(__file__).resolve().parents[1] / 'data'
    alloc = payload['alloc_name']
    nav_wide = load_nav_wide_from_parquet(Path(data_dir), alloc)
    return backtest_portfolio(nav_wide, payload['strategies'], start_date=payload.get('start_date'))

from strategy import compute_risk_budget_weights, compute_target_weights

def _list_allocations_from_parquet(data_dir: Path) -> List[str]:
    p = data_dir / 'asset_nv.parquet'
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=['asset_alloc_name'])
    arr = sorted(df['asset_alloc_name'].dropna().unique().tolist())
    return [str(x) for x in arr]


def _default_start_from_nav(nav_wide: pd.DataFrame) -> Optional[str]:
    try:
        # For each asset, first valid date; take max
        first_dates = {}
        for col in nav_wide.columns:
            s = nav_wide[col].dropna()
            if not s.empty:
                first_dates[col] = s.index[0]
        if not first_dates:
            return None
        dt = max(first_dates.values())
        return dt.date().isoformat()
    except Exception:
        return None


if __name__ == '__main__':
    import sys
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    if len(sys.argv) >= 2:
        payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        out = run_from_payload(payload, data_dir=data_dir)
        print(json.dumps(out, ensure_ascii=False))
        sys.exit(0)

    # Built-in testcases (no CLI args): auto-detect an allocation and run several scenarios
    allocs = _list_allocations_from_parquet(data_dir)
    if not allocs:
        print('[ERROR] data/asset_nv.parquet 不存在或没有可用的配置(alloc_name)。')
        print('请先通过前端保存一个大类配置，或使用 CLI 方式传入 payload.json。')
        sys.exit(2)

    alloc_name = allocs[0]
    nav_wide = load_nav_wide_from_parquet(data_dir, alloc_name)
    start_date = _default_start_from_nav(nav_wide) or (nav_wide.index[0].date().isoformat())
    n = nav_wide.shape[1]
    if n == 0:
        print('[ERROR] 该配置无资产列。')
        sys.exit(2)
    # 等权权重
    w = uniform_weights_kernel(n).tolist()

    test_payloads = [
        {
            'alloc_name': alloc_name,
            'start_date': start_date,
            'strategies': [
                {'name': '静态等权', 'weights': w, 'rebalance': {'enabled': False}},
                {'name': '每月第1个交易日再平衡', 'weights': w, 'rebalance': {'enabled': True, 'mode': 'monthly', 'which': 'nth', 'N': 1, 'unit': 'trading'}},
                {'name': '固定区间20日', 'weights': w, 'rebalance': {'enabled': True, 'mode': 'fixed', 'fixedInterval': 20}},
            ],
        }
    ]

    for i, payload in enumerate(test_payloads, 1):
        print(f'\n[TEST {i}] alloc={payload["alloc_name"]}, start={payload["start_date"]}')
        out = run_from_payload(payload, data_dir=data_dir)
        # 打印每个策略最后一个净值，便于快速核对
        last_vals = {k: (v[-1] if v else None) for k, v in out.get('series', {}).items()}
        summary = {"dates": f"{out.get('dates', [])[:1]} ... {out.get('dates', [])[-1:]}", "last_nav": last_vals}
        print(json.dumps(summary, ensure_ascii=False))

        # 简单一致性对比：静态 vs 每月再平衡 的最终净值是否一致（通常不一致）
        s_names = list(out.get('series', {}).keys())
        if len(s_names) >= 2:
            s0, s1 = s_names[0], s_names[1]
            v0 = out['series'][s0][-1]
            v1 = out['series'][s1][-1]
            diff = abs(v0 - v1)
            print(f"对比: '{s0}'(无再平衡) 最终净值={v0:.6f} vs '{s1}'(再平衡) 最终净值={v1:.6f} -> 差异={diff:.6e}")
