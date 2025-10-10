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


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Expect DatetimeIndex for NAV/returns frame')
    return df.sort_index()


def _to_returns(nav_wide: pd.DataFrame) -> pd.DataFrame:
    nav_wide = _ensure_datetime_index(nav_wide)
    return nav_wide.pct_change().dropna()


def gen_rebalance_dates(
    index: pd.DatetimeIndex,
    mode: str,
    N: Optional[int] = None,
    which: Optional[str] = None,
    unit: Optional[str] = None,
    fixed_interval: Optional[int] = None,
) -> List[pd.Timestamp]:
    idx = pd.DatetimeIndex(index).sort_values()
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

    def _slice_fit_data(nav: pd.DataFrame, up_to: pd.Timestamp, use_all: bool, data_len: Optional[int], window_mode: Optional[str] = None) -> pd.DataFrame:
        win = nav.loc[:up_to]
        if use_all or (window_mode == 'all') or not data_len or data_len <= 0:
            return win
        # 仅 rollingN：使用最近 N 个交易日
        return win.tail(max(2, int(data_len)))

    def _compute_model_weights(nav: pd.DataFrame, s: Dict[str, Any], up_to: pd.Timestamp) -> Optional[np.ndarray]:
        model = s.get('model') or {}
        use_all = bool(model.get('use_all_data', True))
        data_len = model.get('data_len', None)
        window_mode = model.get('window_mode') or ('all' if use_all else 'rollingN')
        nav_fit = _slice_fit_data(nav, up_to, use_all, data_len, window_mode=window_mode)
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
        base_weights = base_weights / max(1e-12, base_weights.sum())
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
        if not rebal_dates:
            # no rebalance: static weights
            base = nav.iloc[0]
            rel = nav.divide(base, axis=1)
            series = (rel * base_weights).sum(axis=1)
            markers.append({'date': nav.index[0].date().isoformat(), 'weights': base_weights.tolist(), 'value': float(series.iloc[0])})
            return series, markers
        # ensure first day included
        rset = sorted([d for d in rebal_dates if d in nav.index])
        if not rset or rset[0] != nav.index[0]:
            rset = [nav.index[0]] + rset
        out = pd.Series(index=nav.index, dtype=float)
        cur_val = 1.0
        for i, d0 in enumerate(rset):
            d1 = rset[i + 1] if i + 1 < len(rset) else nav.index[-1]
            seg = nav.loc[d0:d1]
            w_seg = base_weights
            if recalc:
                w_calc = _compute_model_weights(nav.loc[:d0], s, d0)
                if w_calc is not None and np.isfinite(w_calc).all() and w_calc.sum() > 0:
                    w_seg = (w_calc / w_calc.sum())
            base = seg.iloc[0]
            rel = seg.divide(base, axis=1)
            part = (rel * w_seg).sum(axis=1)
            markers.append({'date': d0.date().isoformat(), 'weights': w_seg.tolist(), 'value': float(cur_val * float(part.iloc[0]))})
            out.loc[seg.index] = cur_val * part.values
            cur_val = float(out.loc[seg.index[-1]])
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
            s['weights'] = [1.0 / max(1, nav_wide.shape[1]) for _ in nav_wide.columns]
        series, markers = _static_or_rebalanced(nav_wide, s)
        series_out[name] = [float(x) for x in series.values]
        marker_out[name] = markers
    return {"dates": [d.strftime('%Y-%m-%d') for d in idx], "series": series_out, "markers": marker_out, "asset_names": list(nav_wide.columns)}


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
    w = np.full(n, 1.0 / n).tolist()

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
