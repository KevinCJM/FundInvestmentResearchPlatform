"""Causal common-observation clocks, deliberately not an exchange calendar."""
from datetime import date
import numpy as np
from numba import njit, types, int64, uint8

I = types.Array(int64, 1, 'A', readonly=True)
U = types.Array(uint8, 1, 'A', readonly=True)


@njit(uint8[::1](I), cache=True, nogil=True)
def period_mask_kernel(keys):
    out = np.ones(keys.size, dtype=np.uint8)
    for i in range(1, keys.size):
        out[i] = np.uint8(keys[i] != keys[i - 1])
    return out


@njit(uint8[::1](U, U), cache=True, nogil=True)
def decision_signal_flags_kernel(flags, decisions):
    if flags.size != decisions.size:
        raise ValueError('TAA_CLOCK_AXIS')
    out = np.empty(flags.size, dtype=np.uint8)
    for i in range(flags.size):
        out[i] = flags[i] * decisions[i]
    return out


for kernel in (period_mask_kernel, decision_signal_flags_kernel):
    kernel.disable_compile()


def clock_plan(dates, policy):
    """Dates are boundary metadata; frequency keys use no future observations."""
    values = [date.fromisoformat(str(d)[:10]) for d in dates]
    if not values or any(b <= a for a, b in zip(values, values[1:])):
        raise ValueError('TAA clocks require strictly increasing common dates.')
    def mask(frequency):
        keys = np.asarray([(d.toordinal() if frequency == 'daily' else
                            d.toordinal() - d.weekday() if frequency == 'weekly' else
                            d.year * 12 + d.month if frequency == 'monthly' else
                            d.year * 4 + (d.month - 1) // 3) for d in values], dtype=np.int64)
        out = period_mask_kernel(keys)
        out.setflags(write=False)
        return out
    return {'decisions': mask(policy.decision_frequency), 'executions': mask(policy.execution_frequency)}


def warm_clock_kernels():
    threshold_kernel(np.zeros(2), .01)
    period_mask_kernel(np.array([1, 1, 2], dtype=np.int64))
    decision_signal_flags_kernel(np.ones(3, dtype=np.uint8), np.ones(3, dtype=np.uint8))
    return all(len(k.signatures) == 1 and k.nopython_signatures for k in (period_mask_kernel, decision_signal_flags_kernel))


def application_status(request, data, plan, recommendation, decision_index):
    """Current application checks use supplied holdings, never simulated holdings."""
    policy = request.decision_policy
    if policy is None:
        return None
    current_index = len(data['dates'])
    reasons = []
    state = 'maintain'
    if request.current_weights is None or request.current_weights_as_of != request.as_of:
        reasons.append('缺少研究日的实际持仓快照，不能判断阈值是否触发或生成可交接交易。')
    if str(request.as_of) not in data['dates']:
        reasons.append('研究日不在已观测共同净值日期中，不能认证本次执行机会。')
    if decision_index < 0 or not plan['executions'][-1] or current_index - decision_index < policy.execution_lag:
        state = 'waiting_execution'
        reasons.append('目标已形成，尚未到执行机会或执行滞后尚未满足。')
    if policy.min_holding_periods:
        if request.last_execution_date is None:
            reasons.append('最小持有期检查需要实际最近执行日期。')
        else:
            elapsed = len([d for d in data['dates'] if str(request.last_execution_date) < d <= str(request.as_of)])
            if elapsed < policy.min_holding_periods:
                reasons.append('实际持仓尚未满足最小持有期。')
                state = 'waiting_execution'
    threshold_met = None
    if request.current_weights is not None and request.current_weights_as_of == request.as_of:
        threshold_met = threshold_kernel(recommendation['trade_deltas'], policy.deviation_threshold)
        if not threshold_met:
            reasons.append('实际持仓偏离尚未达到权重阈值，维持持仓。')
        elif not reasons:
            state = 'adjustment_proposal'
    if request.current_weights is None or request.current_weights_as_of != request.as_of:
        state = 'ineligible'
    return {'evaluated_as_of': str(request.as_of), 'state': state, 'eligible': not reasons, 'reasons': reasons, 'threshold_triggered': threshold_met,
            'actual_execution': False, 'calendar': 'observed_common_dates_only',
            'decision_date': (data['period_starts'] + [str(request.as_of)])[decision_index] if decision_index >= 0 else None,
            'execution_opportunity': bool(plan['executions'][-1]),
            'latest_signal_is_new_decision': bool(plan['decisions'][-1])}


@njit(types.boolean(types.Array(types.float64, 1, 'A', readonly=True), types.float64), cache=True, nogil=True)
def threshold_kernel(deltas, threshold):
    largest = 0.0
    for d in deltas:
        largest = max(largest, abs(d))
    return largest > 1e-14 and largest >= threshold


threshold_kernel.disable_compile()


def validate_clock_application(preview):
    """Also callable by the shared product bridge; never infer old-run metadata."""
    if preview.get('request', {}).get('decision_policy'):
        metadata = preview.get('application')
        if not metadata or metadata.get('eligible') is not True:
            from backend.custom_indicators.errors import ValidationError
            raise ValidationError('TAA_EXECUTION_INELIGIBLE', '；'.join((metadata or {}).get('reasons') or ['缺少新策略的执行资格证据，请重新研究。']))
        if preview['request'].get('current_weights_as_of') != str(date.today()):
            from backend.custom_indicators.errors import ValidationError
            raise ValidationError('TAA_CURRENT_HOLDINGS_STALE', '实际持仓快照不是今天，不能把历史阈值判断用于当前交接；请更新持仓并重新研究。')



def simulation_clock(request, data, signals, include_current=False):
    dates = data['period_starts'] + ([str(request.as_of)] if include_current else [])
    plan = clock_plan(dates, request.decision_policy)
    epoch = date(1970, 1, 1)
    def day(value):
        return (date.fromisoformat(str(value)[:10]) - epoch).days
    plan['period_days'] = np.asarray([day(d) for d in dates], dtype=np.int64)
    if 'expiry_days' in signals:
        plan['valid_until'] = signals['expiry_days'][:len(dates)]
    elif request.signal_mode == 'manual':
        plan['valid_until'] = np.full(len(dates), 2**62, dtype=np.int64)
    else:
        timing = signals['audit']['signal_timing']
        plan['valid_until'] = np.asarray([
            day(row.get('window_end') or row.get('signal_date')) + request.max_signal_age_days
            if row.get('window_end') or row.get('signal_date') else -1
            for row in timing[:len(dates)]], dtype=np.int64)
    for value in plan.values():
        value.setflags(write=False)
    return plan
