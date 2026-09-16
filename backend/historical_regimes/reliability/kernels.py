"""Fixed readonly arbitrary-stride kernels. Only warm() compiles, never requests."""
import threading
import os
import numpy as np
from numba import njit, types

I = types.Array(types.int64, 1, "A", readonly=True)
F = types.Array(types.float64, 2, "A", readonly=True)
_LOCK = threading.Lock()
_READY = False
_WARMED_PID = None


@njit(cache=True)
def align_kernel(reference_dates, prediction_dates):
    indices = np.full(len(reference_dates), -1, np.int64)
    j = 0
    for i in range(len(reference_dates)):
        while j < len(prediction_dates) and prediction_dates[j] < reference_dates[i]:
            j += 1
        if j < len(prediction_dates) and prediction_dates[j] == reference_dates[i]:
            indices[i] = j
    return indices


@njit(cache=True)
def classification_kernel(y, pred, k):
    if len(y) != len(pred) or k < 2 or k > 12:
        raise ValueError("Invalid classification dimensions")
    confusion = np.zeros((k, k + 1), np.int64)
    for i in range(len(y)):
        if 0 <= y[i] < k:
            col = pred[i] if 0 <= pred[i] < k else k
            confusion[y[i], col] += 1
    per = np.full((k, 5), np.nan)
    total = confusion.sum()
    hits = 0
    recalls = 0.0
    f1s = 0.0
    present = 0
    for c in range(k):
        support = confusion[c].sum()
        predicted = confusion[:, c].sum()
        tp = confusion[c, c]
        hits += tp
        per[c, 0] = support
        if predicted > 0:
            per[c, 1] = tp / predicted
        if support > 0:
            per[c, 2] = tp / support
            recalls += per[c, 2]
            present += 1
        if support + predicted > 0:
            per[c, 3] = 2.0 * tp / (support + predicted)
            per[c, 4] = tp / (support + predicted - tp)
        if support > 0:
            f1s += per[c, 3]
    summary = np.full(5, np.nan)
    if total > 0:
        summary[0] = hits / total
        summary[3] = (total - confusion[:, k].sum()) / total
        accepted = total - confusion[:, k].sum()
        if accepted > 0:
            summary[4] = 1.0 - hits / accepted
    if present > 0:
        summary[1] = recalls / present
        summary[2] = f1s / present
    return confusion, per, summary


@njit(cache=True)
def state_evidence_kernel(y, pred, k):
    """State-level evidence on the exact evaluation axis.

    Complete episodes exclude the open head/tail and any unknown-separated run.
    Precision answers the realtime-recognition question: when this state is
    accepted, how often does it agree with the historical reference?
    """
    if len(y) != len(pred) or len(y) > 20000 or k < 2 or k > 12:
        raise ValueError("Invalid state evidence dimensions")
    counts = np.zeros((k, 4), np.int64)
    metrics = np.full((k, 2), np.nan)
    for i in range(len(y)):
        if 0 <= y[i] < k:
            counts[y[i], 0] += 1
        if 0 <= pred[i] < k:
            counts[pred[i], 1] += 1
            if y[i] == pred[i]:
                counts[pred[i], 2] += 1
    start = 0
    n = len(y)
    while start < n:
        end = start + 1
        while end < n and y[end] == y[start]:
            end += 1
        state = y[start]
        if (0 <= state < k and start > 0 and end < n
                and 0 <= y[start - 1] < k and 0 <= y[end] < k):
            counts[state, 3] += 1
        start = end
    for state in range(k):
        if counts[state, 1] > 0:
            metrics[state, 0] = counts[state, 2] / counts[state, 1]
        if counts[state, 0] > 0:
            metrics[state, 1] = counts[state, 2] / counts[state, 0]
    return counts, metrics


@njit(cache=True)
def events_kernel(y, pred, tolerance):
    # Contiguous valid segments, with unknown labels splitting the axis.
    n = len(y)
    if n != len(pred) or n > 20000 or tolerance < 0 or tolerance > 60:
        raise ValueError("Invalid event dimensions or tolerance")
    masked_pred = np.empty(n, np.int64)
    for i in range(n):
        masked_pred[i] = pred[i] if y[i] >= 0 else -1
    pred = masked_pred
    ys = np.empty((n, 3), np.int64)
    ps = np.empty((n, 3), np.int64)
    ny = 0
    npred = 0
    for which in range(2):
        labels = y if which == 0 else pred
        start = 0
        while start < n:
            end = start + 1
            while end < n and labels[end] == labels[start]:
                end += 1
            if labels[start] >= 0:
                if which == 0:
                    ys[ny] = (start, end, labels[start])
                    ny += 1
                else:
                    ps[npred] = (start, end, labels[start])
                    npred += 1
            start = end
    # At most 2*n overlapping segment pairs. Greedy greatest overlap first,
    # deterministic ties; each reference/prediction segment is used once.
    candidates = np.empty((2*n, 3), np.int64)
    count = 0
    a = 0
    b = 0
    while a < ny and b < npred:
        overlap = min(ys[a, 1], ps[b, 1]) - max(ys[a, 0], ps[b, 0])
        if overlap > 0 and ys[a, 2] == ps[b, 2]:
            candidates[count] = (a, b, overlap)
            count += 1
        if ys[a, 1] <= ps[b, 1]:
            a += 1
        else:
            b += 1
    used_y = np.zeros(ny, np.uint8)
    used_p = np.zeros(npred, np.uint8)
    ious = np.zeros(ny)
    matches = 0
    order = np.argsort(-candidates[:count, 2], kind="mergesort")
    for ix in order:
        a, b, overlap = candidates[ix]
        if used_y[a] == 0 and used_p[b] == 0:
            used_y[a] = 1
            used_p[b] = 1
            matches += 1
            ious[a] = overlap / (ys[a, 1] - ys[a, 0] + ps[b, 1] - ps[b, 0] - overlap)
    complete = 0
    for a in range(ny):
        if ys[a, 0] > 0 and ys[a, 1] < n and y[ys[a, 0]-1] >= 0 and y[ys[a, 1]] >= 0:
            complete += 1
    transitions_y = 0
    transitions_p = 0
    event_used = np.zeros(n, np.uint8)
    delays = np.empty(n)
    matched = 0
    for i in range(1, n):
        if pred[i] >= 0 and pred[i-1] >= 0 and pred[i] != pred[i-1]:
            transitions_p += 1
        if y[i] < 0 or y[i-1] < 0 or y[i] == y[i-1]:
            continue
        transitions_y += 1
        best = -1
        distance = tolerance + 1
        for j in range(max(1, i-tolerance), min(n, i+tolerance+1)):
            if (event_used[j] == 0 and pred[j-1] == y[i-1] and pred[j] == y[i]
                    and abs(j-i) < distance):
                best = j
                distance = abs(j-i)
        if best >= 0:
            event_used[best] = 1
            delays[matched] = best-i
            matched += 1
    summary = np.array([ny, npred, matches, ny-matches, npred-matches, complete,
                        transitions_y, transitions_p, matched, transitions_y-matched,
                        transitions_p-matched], dtype=np.int64)
    stats = np.full(3, np.nan)
    if ny > 0:
        stats[0] = ious.sum() / ny
    if matched > 0:
        stats[1] = np.median(delays[:matched])
        stats[2] = np.percentile(delays[:matched], 90)
    return summary, stats


@njit(types.Array(types.float64, 2, "C")(I, F, F, types.float64, types.int64), cache=True)
def apply_kernel(pred, raw, counts, temperature, method):
    n, k = raw.shape
    if len(pred) != n or counts.shape != (k,k) or k < 2 or k > 12 or method < 0 or method > 1:
        raise ValueError("Invalid calibration dimensions")
    if (pred >= k).any():
        raise ValueError("Prediction code outside state axis")
    q = np.full((n, k), np.nan)
    for i in range(n):
        if pred[i] < 0:
            continue
        if method == 0:
            row = counts[pred[i]]
            if row.sum() > 0:
                q[i] = row / row.sum()
        elif (np.isfinite(temperature) and temperature > 0 and np.isfinite(raw[i]).all()
              and (raw[i] >= 0).all() and (raw[i] <= 1).all() and abs(raw[i].sum()-1.0) <= 1e-8):
            weights = np.maximum(raw[i], 1e-12) ** (1.0/temperature)
            q[i] = weights / weights.sum()
    return q


apply_kernel.disable_compile()


@njit(cache=True)
def calibrate_kernel(y, pred, raw, train_end, method):
    n, k = raw.shape
    if len(y) != n or len(pred) != n or train_end < 0 or train_end > n or k < 2 or k > 12:
        raise ValueError("Invalid calibration sample boundaries")
    if (y >= k).any() or (pred >= k).any():
        raise ValueError("Label code outside state axis")
    counts = np.zeros((k, k))
    base = np.zeros(k)
    for i in range(train_end):
        if 0 <= y[i] < k:
            base[y[i]] += 1
            if 0 <= pred[i] < k:
                counts[pred[i], y[i]] += 1
    if base.sum() > 0:
        base /= base.sum()
    else:
        base[:] = np.nan
    temperature = np.nan
    if method == 1:
        best_loss = np.inf
        for step in range(41):
            t = np.exp(np.log(.25) + step / 40.0 * np.log(16.0))
            loss = 0.0
            samples = 0
            for i in range(train_end):
                if (y[i] < 0 or pred[i] < 0 or not np.isfinite(raw[i]).all()
                        or (raw[i] < 0).any() or (raw[i] > 1).any() or abs(raw[i].sum()-1.0) > 1e-8):
                    continue
                weights = np.maximum(raw[i], 1e-12) ** (1.0/t)
                loss -= np.log(max(weights[y[i]]/weights.sum(), 1e-12))
                samples += 1
            if samples > 0 and loss / samples < best_loss:
                best_loss = loss / samples
                temperature = t
    q = apply_kernel(pred, raw, counts, temperature, method)
    return q, counts, base, temperature


@njit(cache=True)
def calibration_support_kernel(y, pred, raw, train_end, method, minimum_samples, minimum_class_samples):
    """Count only usable calibration pairs, on both reference and prediction axes."""
    n, k = raw.shape
    if (len(y) != n or len(pred) != n or not 0 <= train_end <= n
            or not 2 <= k <= 12 or method not in (0, 1)
            or minimum_samples < 1 or minimum_class_samples < 1):
        raise ValueError("Invalid calibration support contract")
    support = np.zeros((2, k), np.int64)
    pairs = 0
    for i in range(train_end):
        if not (0 <= y[i] < k and 0 <= pred[i] < k):
            continue
        if method == 1 and (not np.isfinite(raw[i]).all() or (raw[i] < 0).any()
                            or (raw[i] > 1).any() or abs(raw[i].sum() - 1.0) > 1e-8):
            continue
        support[0, y[i]] += 1
        support[1, pred[i]] += 1
        pairs += 1
    sufficient = pairs >= minimum_samples
    for c in range(k):
        sufficient = sufficient and support[0, c] >= minimum_class_samples and support[1, c] >= minimum_class_samples
    return pairs, support, sufficient


@njit(cache=True)
def probability_kernel(y, pred, q, bins):
    n, k = q.shape
    if len(y) != n or len(pred) != n or bins < 2 or bins > 30:
        raise ValueError("Invalid probability sample dimensions")
    totals = np.zeros(4)
    buckets = np.zeros((bins, 3))
    for i in range(n):
        if y[i] < 0 or y[i] >= k or pred[i] < 0 or pred[i] >= k:
            continue
        valid = True
        mass = 0.0
        for c in range(k):
            if not np.isfinite(q[i,c]) or q[i,c] < 0 or q[i,c] > 1:
                valid = False
            mass += q[i,c]
        if not valid or abs(mass-1.0) > 1e-8:
            continue
        totals[0] += 1
        for c in range(k):
            totals[1] += (q[i,c] - (1.0 if y[i] == c else 0.0)) ** 2
        totals[2] -= np.log(max(q[i,y[i]], 1e-12))
        confidence = q[i,pred[i]]
        bucket = min(bins-1, int(confidence*bins))
        buckets[bucket,0] += 1
        buckets[bucket,1] += confidence
        buckets[bucket,2] += (1.0 if pred[i] == y[i] else 0.0)
    if totals[0] > 0:
        totals[1:3] /= totals[0]
    else:
        totals[1:4] = np.nan
    for b in range(bins):
        if buckets[b,0] > 0:
            buckets[b,1:3] /= buckets[b,0]
            totals[3] += buckets[b,0] / totals[0] * abs(buckets[b,1]-buckets[b,2])
        else:
            buckets[b,1:3] = np.nan
    return totals, buckets


@njit(cache=True)
def map_probability_kernel(raw, mapping, k):
    if len(mapping) != raw.shape[1] or k < 2 or k > 12 or (mapping < 0).any() or (mapping >= k).any():
        raise ValueError("Invalid probability mapping axis")
    out = np.zeros((raw.shape[0], k))
    for i in range(raw.shape[0]):
        mass = 0.0
        valid = True
        for j in range(raw.shape[1]):
            value = raw[i,j]
            if not np.isfinite(value) or value < 0 or value > 1:
                valid = False
            mass += value
            out[i,mapping[j]] += value
        if not valid or abs(mass-1.0) > 1e-8:
            out[i,:] = np.nan
    return out


@njit(cache=True)
def decision_kernel(pred, q, floor):
    if len(pred) != q.shape[0]:
        raise ValueError("Invalid decision sample dimensions")
    accepted = np.full(len(pred), -1, np.int64)
    for i in range(len(pred)):
        if 0 <= pred[i] < q.shape[1] and np.isfinite(q[i, pred[i]]) and q[i, pred[i]] >= floor:
            accepted[i] = pred[i]
    return accepted


@njit(cache=True)
def sample_kernel(y, pred, indices):
    if len(y) != len(pred) or len(y) != len(indices):
        raise ValueError("Invalid sample dimensions")
    counts = np.zeros(4, np.int64)
    for i in range(len(y)):
        if indices[i] >= 0:
            counts[0] += 1
        else:
            counts[3] += 1
        if y[i] < 0:
            counts[1] += 1
        if pred[i] < 0:
            counts[2] += 1
    return counts


KERNELS = ((calibration_support_kernel, (I, I, F, types.int64, types.int64, types.int64, types.int64)), (decision_kernel,(I,F,types.float64)), (sample_kernel,(I,I,I)), (apply_kernel, (I,F,F,types.float64,types.int64)), (align_kernel, (I,I)), (classification_kernel, (I,I,types.int64)), (state_evidence_kernel, (I,I,types.int64)),
           (events_kernel, (I,I,types.int64)), (calibrate_kernel, (I,I,F,types.int64,types.int64)),
           (probability_kernel, (I,I,F,types.int64)), (map_probability_kernel, (F,I,types.int64)))


from . import diagnostic_kernels, bootstrap
KERNELS = (*KERNELS, *diagnostic_kernels.KERNELS, *bootstrap.KERNELS)


def warm():
    global _READY, _WARMED_PID
    with _LOCK:
        _READY = False
        for kernel, signature in KERNELS:
            if not kernel.signatures:
                kernel.compile(signature)
            kernel.disable_compile()
        labels = np.empty(0, np.int64)
        raw = np.empty((0, 2), np.float64)
        counts = np.zeros((2, 2), np.float64)
        mapping = np.array([0, 1], np.int64)
        for array in (labels, raw, counts, mapping):
            array.flags.writeable = False
        align_kernel(labels, labels)
        classification_kernel(labels, labels, 2)
        state_evidence_kernel(labels, labels, 2)
        events_kernel(labels, labels, 1)
        calibrate_kernel(labels, labels, raw, 0, 0)
        calibration_support_kernel(labels, labels, raw, 0, 0, 2, 1)
        apply_kernel(labels, raw, counts, 1.0, 0)
        probability_kernel(labels, labels, raw, 2)
        map_probability_kernel(raw, mapping, 2)
        decision_kernel(labels, raw, .6)
        sample_kernel(labels, labels, labels)
        diagnostic_kernels.smoke()
        bootstrap.smoke()
        _WARMED_PID = os.getpid()
        _READY = True
    return audit()


def audit():
    if not _READY or _WARMED_PID != os.getpid() or any(len(k.nopython_signatures) != 1 or len(k.signatures) != 1 or k._can_compile for k, _ in KERNELS):
        raise RuntimeError("RELIABILITY_RUNTIME_NOT_READY")
    return {"complete": True, "request_time_compilation": 0, "python_fallback": 0,
            "kernel_signatures": {k.py_func.__name__: [str(s) for s in k.nopython_signatures] for k,_ in KERNELS}}
