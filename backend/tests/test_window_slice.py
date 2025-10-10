import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest_engine import slice_fit_data


def _make_nav():
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    return pd.DataFrame({'A': range(10), 'B': range(10, 20)}, index=dates)


def test_slice_all_mode_returns_full_window():
    nav = _make_nav()
    up_to = nav.index[6]
    out = slice_fit_data(nav, up_to, 'all', None)
    assert list(out.index) == list(nav.index[:7])


def test_slice_rollingn_uses_tail_n_rows():
    nav = _make_nav()
    up_to = nav.index[-1]
    out = slice_fit_data(nav, up_to, 'rollingN', 4)
    assert list(out.index) == list(nav.index[-4:])


def test_slice_rollingn_enforces_minimum_two_rows():
    nav = _make_nav()
    up_to = nav.index[1]
    out = slice_fit_data(nav, up_to, 'rollingN', 1)
    assert len(out) == 2
    assert list(out.index) == list(nav.index[:2])


def test_slice_mode_none_defaults_to_all():
    nav = _make_nav()
    up_to = nav.index[4]
    out = slice_fit_data(nav, up_to, None, 5)
    assert list(out.index) == list(nav.index[:5])


def test_slice_rollingn_case_insensitive():
    nav = _make_nav()
    up_to = nav.index[-2]
    out = slice_fit_data(nav, up_to, 'ROLLINGn', 3)
    assert list(out.index) == list(nav.index[-4:-1])


def test_slice_rollingn_with_none_data_len_uses_two_rows():
    nav = _make_nav()
    up_to = nav.index[5]
    out = slice_fit_data(nav, up_to, 'rollingN', None)
    assert len(out) == 2
    assert list(out.index) == list(nav.index[4:6])
