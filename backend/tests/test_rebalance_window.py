import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine import backtest_portfolio


def _make_nav(days: int = 40) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=days, freq='D')
    data = {
        'ClassA': np.linspace(100, 140, days),
        'ClassB': np.linspace(90, 110, days)
    }
    df = pd.DataFrame(data, index=idx)
    return df


def test_rebalance_skips_until_enough_samples_rollingn():
    nav = _make_nav()
    strategies = [{
        'name': 'target-strategy',
        'type': 'target',
        'weights': [0.5, 0.5],
        'classes': [{'name': 'ClassA'}, {'name': 'ClassB'}],
        'rebalance': {'enabled': True, 'mode': 'monthly', 'which': 'nth', 'N': 1, 'unit': 'trading', 'recalc': True},
        'model': {
            'window_mode': 'rollingN',
            'data_len': 20,
            'return_metric': 'cumulative',
            'return_type': 'simple',
            'risk_metric': 'vol',
            'target': 'min_risk',
        }
    }]

    result = backtest_portfolio(nav, strategies)
    markers = result['markers']['target-strategy']
    assert markers, 'should have markers for rebalance'
    first_date = markers[0]['date']
    # 20 日窗口意味着首个有效调仓应在 2024-01-20 之后
    assert first_date >= '2024-01-20'


def test_rebalance_all_mode_requires_two_rows():
    nav = _make_nav(5)
    strategies = [{
        'name': 'target-all',
        'type': 'target',
        'weights': [0.5, 0.5],
        'classes': [{'name': 'ClassA'}, {'name': 'ClassB'}],
        'rebalance': {'enabled': True, 'mode': 'weekly', 'which': 'nth', 'N': 1, 'unit': 'trading', 'recalc': True},
        'model': {
            'window_mode': 'all',
            'return_metric': 'cumulative',
            'risk_metric': 'vol',
            'target': 'min_risk',
        }
    }]

    result = backtest_portfolio(nav, strategies)
    markers = result['markers']['target-all']
    assert markers[0]['date'] == nav.index[1].date().isoformat()


def test_rebalance_raises_when_no_valid_window():
    nav = _make_nav(5)
    strategies = [{
        'name': 'invalid',
        'type': 'target',
        'weights': [0.5, 0.5],
        'classes': [{'name': 'ClassA'}, {'name': 'ClassB'}],
        'rebalance': {'enabled': True, 'mode': 'weekly', 'which': 'nth', 'N': 1, 'unit': 'trading', 'recalc': True},
        'model': {
            'window_mode': 'rollingN',
            'data_len': 50,
            'return_metric': 'cumulative',
            'risk_metric': 'vol',
            'target': 'min_risk',
        }
    }]

    with pytest.raises(ValueError):
        backtest_portfolio(nav, strategies)
