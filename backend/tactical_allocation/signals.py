"""Documented dated scores and causal multi-window momentum composition.

Missing positive-weight components invalidate the whole composite. Neutral zero
is available; no price-derived value/carry proxy and no missing reweighting.
"""
from datetime import date, timedelta
import numpy as np
from numba import njit, types, float64, int64, uint8
from . import numeric

M = types.Array(float64, 2, 'A', readonly=True)
C = types.Array(float64, 3, 'A', readonly=True)
IM = types.Array(int64, 2, 'A', readonly=True)
I = types.Array(int64, 1, 'A', readonly=True)
F = types.Array(float64, 1, 'A', readonly=True)
MO = float64[:, ::1]


@njit(MO(M, IM, int64), cache=True, nogil=True)
def momentum_scores_kernel(returns, windows, lookback):
    out = np.zeros((windows.shape[0], returns.shape[1]), dtype=np.float64)
    logs = np.zeros(returns.shape[1])
    cursor = 0
    for row in range(windows.shape[0]):
        end = windows[row, 1]
        while cursor < end:
            for a in range(returns.shape[1]):
                logs[a] += np.log1p(returns[cursor, a])
                if cursor >= lookback:
                    logs[a] -= np.log1p(returns[cursor - lookback, a])
            cursor += 1
        if windows[row, 4] != 3:
            continue
        mean = 0.0
        for a in range(logs.size):
            out[row, a] = np.expm1(logs[a])
            if not np.isfinite(out[row, a]):
                raise ValueError('TAA_MOMENTUM_OVERFLOW')
            mean += out[row, a]
        mean /= logs.size
        scale = 0.0
        for a in range(logs.size):
            out[row, a] -= mean
            scale = max(scale, abs(out[row, a]))
        if scale > 1e-12:
            for a in range(logs.size):
                out[row, a] /= scale
        else:
            out[row] = 0.0
    return out


@njit(types.Tuple((MO, int64[:, ::1]))(M, IM, I, int64), cache=True, nogil=True)
def dated_scores_kernel(values, dates, cutoffs, max_age):
    out = np.zeros((cutoffs.size, values.shape[1]))
    audit = np.full((cutoffs.size, 3), -1, dtype=np.int64)  # row, status, expiry
    cursor, chosen = 0, -1
    for t in range(cutoffs.size):
        while cursor < dates.shape[0] and dates[cursor, 1] <= cutoffs[t]:
            chosen = cursor
            cursor += 1
        audit[t, 0] = chosen
        audit[t, 1] = 1  # unavailable
        if chosen >= 0:
            expiry = min(dates[chosen, 2], dates[chosen, 0] + max_age)
            audit[t, 2] = expiry
            audit[t, 1] = 2 if cutoffs[t] > expiry else 3
            if audit[t, 1] == 3:
                for a in range(values.shape[1]):
                    out[t, a] = values[chosen, a]
    return out, audit


@njit(types.Tuple((MO, uint8[::1], uint8[::1]))(C, IM, F, float64), cache=True, nogil=True)
def composite_kernel(scores, statuses, weights, max_tilt):
    c, n, a = scores.shape
    out, active, known = np.zeros((n, a)), np.zeros(n, dtype=np.uint8), np.ones(n, dtype=np.uint8)
    total = 0.0
    for k in range(c):
        if not np.isfinite(weights[k]) or weights[k] < 0:
            raise ValueError('TAA_COMPOSITE_WEIGHT')
        total += weights[k]
    if abs(total - 1.0) > 1e-8:
        raise ValueError('TAA_COMPOSITE_WEIGHT')
    for t in range(n):
        for k in range(c):
            if weights[k] > 0 and statuses[k, t] != 3:
                known[t] = 0
        if not known[t]:
            continue
        mean = 0.0
        for j in range(a):
            for k in range(c):
                out[t, j] += weights[k] * scores[k, t, j]
            mean += out[t, j]
        mean /= a
        scale = 1.0
        for j in range(a):
            out[t, j] -= mean
            scale = max(scale, abs(out[t, j]))
        for j in range(a):
            out[t, j] *= max_tilt / scale
            if abs(out[t, j]) > 1e-12:
                active[t] = 1
    return out, active, known


KERNELS = (momentum_scores_kernel, dated_scores_kernel, composite_kernel)
for kernel in KERNELS:
    kernel.disable_compile()


def build_composite(request, data, assets):
    epoch = date(1970, 1, 1)
    day = lambda d: (date.fromisoformat(str(d)[:10]) - epoch).days
    cutoffs = np.asarray([day(d) for d in data['period_starts']] + [day(request.as_of)], dtype=np.int64)
    starts, ends = cutoffs[:-1], np.asarray([day(d) for d in data['dates']], dtype=np.int64)
    count, n, a = len(request.signal_components), len(cutoffs), len(assets)
    scores, statuses = np.zeros((count, n, a)), np.zeros((count, n), dtype=np.int64)
    audits, expiries, observations = [], [], []
    for k, component in enumerate(request.signal_components):
        rows = []
        if component.kind == 'momentum':
            windows = numeric.momentum_windows_kernel(data['available_at'], starts, ends, component.lookback,
                                                       day(request.as_of), component.max_age_days)
            scores[k] = momentum_scores_kernel(data['returns'], windows, component.lookback)
            statuses[k] = windows[:, 4]
            for t, window in enumerate(windows):
                end = int(window[1])
                observed = data['dates'][end - 1] if end >= 1 else None
                rows.append({'cutoff': str(epoch + timedelta(days=int(cutoffs[t]))), 'observed_on': observed,
                             'available_on': str(epoch + timedelta(days=int(window[2]))) if window[2] >= 0 else None,
                             'expires_on': str(date.fromisoformat(observed) + timedelta(days=component.max_age_days)) if observed else None,
                             'status': int(window[4])})
        else:
            ordered = sorted(component.observations, key=lambda r: r.available_on)
            if any(set(row.values) != set(assets) for row in ordered):
                from backend.custom_indicators.errors import ValidationError
                raise ValidationError('TAA_SIGNAL_ASSET_AXIS', '外部标准化信号必须完整覆盖同一资产轴，不能缺项或增项。')
            values = np.asarray([[row.values[asset] for asset in assets] for row in ordered], dtype=np.float64)
            dates = np.asarray([[day(row.observed_on), day(row.available_on), day(row.expires_on)] for row in ordered], dtype=np.int64)
            scores[k], audit = dated_scores_kernel(values, dates, cutoffs, component.max_age_days)
            statuses[k] = audit[:, 1]
            for t, (index, status, expiry) in enumerate(audit):
                row = ordered[int(index)] if index >= 0 else None
                rows.append({'cutoff': str(epoch + timedelta(days=int(cutoffs[t]))),
                             'observed_on': str(row.observed_on) if row else None,
                             'available_on': str(row.available_on) if row else None,
                             'expires_on': str(epoch + timedelta(days=int(expiry))) if expiry >= 0 else None,
                             'status': int(status), 'observation_index': int(index)})
        audits.append({'id': component.id, 'kind': component.kind, 'weight': component.weight,
                       'source': component.source, 'methodology': component.methodology, 'unit': component.unit,
                       'timing': rows})
        if component.weight > 0 and rows[-1]['expires_on']:
            expiries.append(rows[-1]['expires_on'])
            observations.append(rows[-1]['observed_on'])
    directions, active, known = composite_kernel(scores, statuses, np.asarray([c.weight for c in request.signal_components]), request.max_abs_tilt)
    directions.setflags(write=False)
    result = {'probabilities': np.ones((n - 1, 1)), 'use_signal': active[:-1], 'state_tilts': np.zeros((1, a)),
            'direct_tilts': directions[:-1], 'current_direct_tilt': directions[-1],
            'current_probabilities': np.ones(1), 'current_use_signal': int(active[-1]),
            'knowledge_verified': known[:-1], 'current_knowledge_verified': int(known[-1]),
            'current_date': min(observations) if observations else None,
            'current_expires_on': min(expiries) if expiries else None, 'confidence': None,
            'fallback_reason': None if known[-1] else '组合存在尚不可得或已过期分量；不重分配其权重，当前回归 SAA。',
            'expiry_days': np.asarray([min((day(audit['timing'][t]['expires_on']) if audit['timing'][t]['expires_on'] else -1) for audit in audits if audit['weight'] > 0) for t in range(n)], dtype=np.int64),
            'audit': {'method': 'causal_composite/1.0.0', 'components': audits,
                      'missing_policy': 'all_positive_weight_components_required',
                      'momentum_standardization': 'center window returns, divide by maximum absolute centered value; neutral equal returns are available',
                      'composition': 'weighted scores, center, divide by max(1, max_abs), multiply by max_abs_tilt'}}
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return result


def warm_signal_kernels():
    momentum_scores_kernel(np.zeros((3, 2)), np.array([[0, 2, 2, 0, 3]], dtype=np.int64), 2)
    dated_scores_kernel(np.zeros((1, 2)), np.array([[1, 2, 4]], dtype=np.int64), np.array([2, 5]), 3)
    composite_kernel(np.zeros((1, 3, 2)), np.full((1, 3), 3, dtype=np.int64), np.ones(1), .1)
    return all(len(k.signatures) == 1 and k.nopython_signatures for k in KERNELS)
