import numpy as np
import pandas as pd
import pytest

from backend.research_input_checks import (
    ResearchInputError, require_return_quality, return_quality,
    return_scale_breaks_kernel, training_readiness_kernel, warm_research_input_checks,
)


def test_scale_rule_boundaries_readonly_views_and_no_new_signature():
    warm_research_input_checks()
    base = np.array([[0., 0.], [-.9, 9.], [-.899999, 8.999999], [np.nan, np.inf]])
    base.setflags(write=False)
    view = base[::1, ::-1]
    assert np.shares_memory(base, view)
    before = tuple(return_scale_breaks_kernel.signatures)
    first, counts = return_scale_breaks_kernel(view)
    np.testing.assert_array_equal(first, [1, 1])
    np.testing.assert_array_equal(counts, [1, 1])
    assert tuple(return_scale_breaks_kernel.signatures) == before
    assert np.isnan(base[3, 0]) and base[1, 0] == -.9
    assert return_quality(np.empty((0, 2)), [], ['a', 'b'])['status'] == 'clear'


def test_quality_reports_asset_date_and_rejects_without_repairing():
    values = np.array([[.001, .001], [.002, -.99]])
    with pytest.raises(ResearchInputError, match='2026-06-02') as error:
        require_return_quality(values, ['2026-06-01', '2026-06-02'], ['股票', '国债'])
    assert error.value.diagnostics[0]['asset_id'] == '国债'
    assert error.value.diagnostics[0]['value'] == -.99
    assert values[1, 1] == -.99


def test_clock_only_suggested_split_has_mature_training_and_holdout():
    days = np.arange(100, 161, dtype=np.int64)
    available = np.repeat(days[:, None], 2, axis=1)
    available[39, 0] = 145
    future, unknown, earliest, proposed = training_readiness_kernel(available, days, 139)
    assert (future, unknown, earliest) == (1, 0, 145)
    assert proposed == 138  # earlier mature boundary, not a fabricated future label
    count = np.searchsorted(days, proposed, side='right')
    assert count >= 20 and len(days) - count >= 20
    assert (available[:count] <= proposed).all()
    available[:] = 200
    assert training_readiness_kernel(available, days, 139)[3] == -1
    available[:] = -1
    assert training_readiness_kernel(available, days, 139)[1] == 80
    with pytest.raises(ValueError):
        training_readiness_kernel(available[:4], days, 139)


def test_fit_date_conflict_and_actual_scale_break_fail_before_metrics(tmp_path):
    from backend.fit import ClassSpec, ETFSpec, compute_classes_nav
    from backend.fit_numba import warm_fit_numba_kernels
    warm_fit_numba_kernels()
    classes = [ClassSpec('b', '国债', [ETFSpec('511010.SH', '国债ETF', 100.)])]
    with pytest.raises(ResearchInputError, match='研究日'):
        compute_classes_nav(tmp_path, classes, pd.Timestamp('2020-01-01'), as_of='2014-12-31')
    pd.DataFrame({'ts_code': ['511010.SH'] * 3, 'name': ['国债ETF'] * 3,
                  'date': pd.date_range('2026-06-01', periods=3),
                  'adj_nav': [145.947686, 1.459430, 1.460],
                  'ann_date': ['20260602', '20260602', '20260603']}).to_parquet(tmp_path / 'etf_daily_df.parquet')
    with pytest.raises(ResearchInputError, match='2026-06-02'):
        compute_classes_nav(tmp_path, classes, pd.Timestamp('2026-06-01'))
