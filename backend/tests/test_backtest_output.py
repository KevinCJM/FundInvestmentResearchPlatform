import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine import backtest_portfolio


def _make_nav(days: int = 12) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=days, freq='D')
    data = {
        'A': np.linspace(100, 120, days),
        'B': np.linspace(90, 110, days)
    }
    return pd.DataFrame(data, index=idx)


def test_static_strategy_has_no_markers_and_dates_align():
    nav = _make_nav()
    strat = {
        'name': 'static',
        'type': 'fixed',
        'weights': [0.6, 0.4],
        'classes': [{'name': 'A', 'weight': 60.0}, {'name': 'B', 'weight': 40.0}]
    }
    res = backtest_portfolio(nav, [strat])
    assert 'static' in res['series']
    assert len(res['markers']['static']) == 0
    assert len(res['dates']) == len(res['series']['static'])


def test_rebalanced_strategy_markers_match_dates():
    nav = _make_nav(40)
    strat = {
        'name': 'rebal',
        'type': 'fixed',
        'weights': [0.5, 0.5],
        'classes': [{'name': 'A', 'weight': 50.0}, {'name': 'B', 'weight': 50.0}],
        'rebalance': {'enabled': True, 'mode': 'monthly', 'which': 'nth', 'N': 1, 'unit': 'trading', 'recalc': False}
    }
    res = backtest_portfolio(nav, [strat])
    markers = res['markers']['rebal']
    assert markers, 'expect markers for rebalanced strategy'
    dates = res['dates']
    series = res['series']['rebal']
    assert len(dates) == len(series)
    for marker in markers:
        assert marker['date'] in dates
        idx = dates.index(marker['date'])
        assert np.isclose(series[idx], marker['value'], atol=1e-8)
