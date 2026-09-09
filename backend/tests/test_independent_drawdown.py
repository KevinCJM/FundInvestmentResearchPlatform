"""Every independent drawdown result refers to the last equal deepest episode."""
from __future__ import annotations
import numpy as np
import pytest
from cal_indicators.drawdown_interval import last_drawdown_interval_kernel
from cal_indicators.typed_dsl import compose_typed_expression, TypedDslError
from cal_indicators.typed_numba_plan import compile_numba_batch_plan
from custom_indicators.variable_registry import variable_types

INTERVAL = 'last_drawdown_interval(drawdown_series(adjusted_nav))'
EXPRESSIONS = (
    'negate(min_value(drawdown_series(adjusted_nav)))',
    f'value_at(observation_dates, interval_start({INTERVAL}))',
    f'value_at(observation_dates, interval_trough({INTERVAL}))',
    f'days_between(value_at(observation_dates, interval_start({INTERVAL})), value_at(observation_dates, interval_trough({INTERVAL})))',
    f'value_at(observation_dates, interval_recovery({INTERVAL}))',
)


def test_last_equal_maximum_selects_one_interval():
    actual = last_drawdown_interval_kernel(np.array([0., -.2, 0., 0., -.2, -.2, 0.]))
    np.testing.assert_equal(actual, [3., 5., 6., 1.])


def test_last_maximum_is_not_the_longest_episode():
    dd = np.array([0., -.1, -.1, -.1, 0., -.3, 0.])
    np.testing.assert_equal(last_drawdown_interval_kernel(dd), [4., 5., 6., 1.])


@pytest.mark.parametrize('values', [[], [0., np.nan], [0., np.inf], [0., .1], [0., -1.1], [-.1, -.2]])
def test_bad_interval_input_fails_closed(values):
    actual = last_drawdown_interval_kernel(np.array(values, dtype=np.float64))
    assert actual[3] == -1
    assert np.isnan(actual[:3]).all()


def test_unrecovered_and_no_event_are_not_fake_dates():
    actual = last_drawdown_interval_kernel(np.array([0., -.2, -.1]))
    assert actual[:2] == (0., 1.) and np.isnan(actual[2]) and actual[3] == 1
    flat = last_drawdown_interval_kernel(np.zeros(4))
    assert np.isnan(flat[:3]).all() and flat[3] == 0


@pytest.mark.parametrize('parallel', [False, True])
def test_independent_batch_has_one_drawdown_and_one_interval(parallel):
    plans = tuple(compose_typed_expression(source, variable_types=variable_types('single_product')) for source in EXPRESSIONS)
    batch = compile_numba_batch_plan(plans, ({},)*len(plans), ('adjusted_nav', 'observation_dates'))
    nav = np.array([1., .8, 1., 1., .8, .8, 1.])
    dates = np.array([20000.,20001.,20002.,20003.,20007.,20009.,20010.])
    output = np.full((1,5),np.nan)
    statuses = np.full((1,5),-1,dtype=np.int16)
    before = tuple(batch.serial_dispatcher.signatures), tuple(batch.parallel_dispatcher.signatures)
    batch.compute(np.array([nav, dates]),np.array([0],dtype=np.int64),np.array([7],dtype=np.int64),np.array([10.]),output,statuses,parallel=parallel)
    np.testing.assert_allclose(output[0], [.2,20003.,20009.,6.,20010.])
    assert (statuses==0).all()
    audit=batch.metadata()
    assert audit['operator_call_sites']['drawdown_series']==1
    assert audit['operator_call_sites']['last_drawdown_interval']==1
    assert audit['python_fallback']==0
    assert before==(tuple(batch.serial_dispatcher.signatures),tuple(batch.parallel_dispatcher.signatures))


def test_dates_are_not_ordinary_score_numbers():
    with pytest.raises(TypedDslError):
        compose_typed_expression(f'value_at(observation_dates, interval_start({INTERVAL})) * 2',variable_types=variable_types('single_product'))
