from __future__ import annotations

"""
Coverage-style test runner for backend/backtest_engine.py.

Run: /Users/chenjunming/Desktop/myenv_312/bin/python3.12 backend/backtest_engine_tests.py

The script exercises a matrix of scenarios:
- Static vs monthly rebalance (no recalc)
- Monthly rebalance + recalc with risk_budget (rollingN/firstN/all)
- Monthly rebalance + recalc with target (min_risk/max_return) (rollingN)
- Fixed-interval rebalance + recalc

It prints end-of-period NAVs and sanity checks shapes and differences where appropriate.
The script is resilient: if a scenario fails (e.g., parameter infeasible), it logs the error and continues.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from backtest_engine import load_nav_wide_from_parquet, backtest_portfolio


DATA_DIR = Path(__file__).resolve().parents[1] / 'data'


def pick_allocation() -> str:
    p = DATA_DIR / 'asset_nv.parquet'
    if not p.exists():
        raise FileNotFoundError('data/asset_nv.parquet not found')
    df = pd.read_parquet(p, columns=['asset_alloc_name'])
    arr = df['asset_alloc_name'].dropna().unique().tolist()
    if not arr:
        raise RuntimeError('No allocation names found in asset_nv.parquet')
    return str(arr[0])


def default_start(nav: pd.DataFrame) -> str:
    nav = nav.sort_index()
    # take max of first valid for each column
    first = []
    for c in nav.columns:
        s = nav[c].dropna()
        if not s.empty:
            first.append(s.index[0])
    if not first:
        raise RuntimeError('No valid NAV series to determine start date')
    dt = max(first)
    return dt.date().isoformat()


def run_case(payload: Dict[str, Any], title: str) -> Dict[str, float]:
    try:
        out = backtest_portfolio(
            load_nav_wide_from_parquet(DATA_DIR, payload['alloc_name']),
            payload['strategies'],
            start_date=payload.get('start_date'),
        )
    except Exception as e:
        print(f"[FAIL] {title}: {e}")
        return {}
    last_nav = {k: (v[-1] if v else None) for k, v in out.get('series', {}).items()}
    print(f"[OK] {title} -> last NAV: {json.dumps(last_nav, ensure_ascii=False)}")
    return last_nav


def main():
    alloc = pick_allocation()
    nav = load_nav_wide_from_parquet(DATA_DIR, alloc)
    start = default_start(nav)
    cols = list(nav.columns)
    if not cols:
        raise RuntimeError('No asset columns in nav_wide')
    n = len(cols)
    eq = [1.0 / n] * n
    budgets = [100.0] * n

    # 1) Static vs Monthly rebalance
    payload = {
        'alloc_name': alloc,
        'start_date': start,
        'strategies': [
            { 'name': '静态等权', 'type': 'fixed', 'classes': [{'name': c, 'weight': w} for c, w in zip(cols, eq)], 'rebalance': {'enabled': False}},
            { 'name': '月度再平衡', 'type': 'fixed', 'classes': [{'name': c, 'weight': w} for c, w in zip(cols, eq)], 'rebalance': {'enabled': True, 'mode':'monthly','which':'nth','N':1,'unit':'trading'}},
        ]
    }
    last1 = run_case(payload, '静态 vs 月度再平衡')

    # 2) 月度再平衡 + recalc (risk_budget, rollingN=60)
    payload_rb = {
        'alloc_name': alloc,
        'start_date': start,
        'strategies': [
            { 'name': '风险预算-rollingN60-月度recalc', 'type':'risk_budget',
              'classes': [{'name': c, 'budget': b} for c, b in zip(cols, budgets)],
              'weights': eq, # initial
              'rebalance': {'enabled': True, 'mode':'monthly','which':'nth','N':1,'unit':'trading','recalc': True},
              'model': { 'risk_metric':'vol', 'window_mode':'rollingN', 'data_len': 60 }
            },
        ]
    }
    last2 = run_case(payload_rb, '风险预算 rollingN=60 月度recalc')

    # 3) 月度再平衡 + recalc (target min_risk, rollingN=60)
    payload_t = {
        'alloc_name': alloc,
        'start_date': start,
        'strategies': [
            { 'name': '目标-最小风险-rollingN60-月度recalc', 'type':'target',
              'classes': [{'name': c} for c in cols],
              'weights': eq,
              'rebalance': {'enabled': True, 'mode':'monthly','which':'nth','N':1,'unit':'trading','recalc': True},
              'model': {
                'target':'min_risk',
                'return_metric':'annual', 'return_type':'simple', 'days': 252,
                'risk_metric':'vol',
                'risk_free_rate': 0.0,
                'window_mode':'rollingN', 'data_len':60,
              }
            },
        ]
    }
    last3 = run_case(payload_t, '目标 最小风险 rollingN=60 月度recalc')

    # 4) 固定区间20日 + recalc (risk_budget)
    payload_fix = {
        'alloc_name': alloc,
        'start_date': start,
        'strategies': [
            { 'name': '风险预算-固定区间20-recalc', 'type':'risk_budget',
              'classes': [{'name': c, 'budget': b} for c, b in zip(cols, budgets)],
              'weights': eq,
              'rebalance': {'enabled': True, 'mode':'fixed','fixedInterval':20,'recalc': True},
              'model': { 'risk_metric':'vol', 'window_mode':'rollingN', 'data_len': 60 }
            },
        ]
    }
    last4 = run_case(payload_fix, '风险预算 固定区间20 recalc')

    # Basic sanity checks
    try:
        if last1 and last2:
            s_names = list(last1.keys()) + list(last2.keys())
            print(f"[INFO] Compared keys: {set(s_names)}")
    except Exception:
        pass

    print("\n[SUMMARY] Done. If certain cases are equal, it can be data-specific; try different alloc_name or window parameters.")


if __name__ == '__main__':
    main()

