"""Fixed-signature exposure paths and additive, wealth-linked return contributions."""
from __future__ import annotations

import numpy as np
from numba import int64, njit, types

from .numba_kernels import F2, F3, F1, I1, I2, attribution_kernel


@njit(I2(I2), cache=True, nogil=True)
def return_availability_kernel(nav_available):
    result = np.full(nav_available.shape, np.iinfo(np.int64).max, dtype=np.int64)
    for t in range(1, nav_available.shape[0]):
        for a in range(nav_available.shape[1]):
            result[t, a] = max(nav_available[t, a], nav_available[t - 1, a])
    return result


@njit(types.Tuple((F3, F3, F2, F2))(
    F2, F2, F1, I2, I1, F2, F2, int64, int64, int64, int64, int64, int64
), cache=True, nogil=True)
def exposure_path_kernel(returns, factors, rf, available, days, fixed_beta, fixed_stats,
                         split, window, minimum, step, mode, model):
    """meta: status, paired observations, fit-start index, fit-end index, fit R².

    Rolling windows retain calendar positions and end strictly before the return
    being explained. A failed scheduled fit invalidates the entire next block.
    """
    n, assets = returns.shape
    k = factors.shape[1]
    betas = np.full((n, assets, k + 1), np.nan)
    meta = np.full((n, assets, 5), np.nan)
    current_beta = fixed_beta.copy()
    current_stats = fixed_stats.copy()
    fit_start, fit_end = 0, split - 1
    if mode == 1:
        current_beta[:] = np.nan
        current_stats[:] = np.nan
        current_stats[:, 7] = 4.0
        fit_start, fit_end = -1, -1
    for t in range(n):
        if mode == 1 and t >= window and (t - window) % step == 0:
            fit_start, fit_end = t - window, t - 1
            sample = returns[fit_start:t].copy()
            for s in range(window):
                for a in range(assets):
                    # Date-only NAV must have been announced on an earlier day.
                    if available[fit_start + s, a] >= days[t]:
                        sample[s, a] = np.nan
            current_beta, current_stats = attribution_kernel(
                sample, factors[fit_start:t].copy(), rf[fit_start:t].copy(), window, model)
            for a in range(assets):
                if current_stats[a, 0] < max(minimum, max(30, 5 * k)):
                    current_beta[a, :] = np.nan
                    current_stats[a, 7] = 1.0
        for a in range(assets):
            betas[t, a] = current_beta[a]
            meta[t, a, 0] = current_stats[a, 7]
            meta[t, a, 1] = current_stats[a, 0]
            meta[t, a, 2] = fit_start
            meta[t, a, 3] = fit_end
            meta[t, a, 4] = current_stats[a, 2]
    return betas, meta, current_beta, current_stats


@njit(types.Tuple((F3, F3, I2))(F2, F2, F1, F3, F3), cache=True, nogil=True)
def daily_contributions_kernel(returns, factors, rf, betas, fit_meta):
    """Columns: each factor, RF, intercept, residual. Total-return RF is structural 0.

    Status: 0 complete; 1/2/3 fitting errors; 4 warmup; 5 missing return;
    6 missing factor/RF; 7 nonfinite arithmetic or failed reconciliation.
    """
    n, assets = returns.shape
    k = factors.shape[1]
    values = np.full((n, assets, k + 3), np.nan)
    checks = np.full((n, assets, 2), np.nan)
    status = np.zeros((n, assets), dtype=np.int64)
    for t in range(n):
        for a in range(assets):
            fit_status = int(fit_meta[t, a, 0])
            if fit_status != 0:
                status[t, a] = fit_status
                continue
            actual = returns[t, a]
            if not np.isfinite(actual) or actual <= -1.0:
                status[t, a] = 5
                continue
            valid = np.isfinite(rf[t])
            for j in range(k):
                valid = valid and np.isfinite(factors[t, j])
            if not valid:
                status[t, a] = 6
                continue
            fitted = rf[t] + betas[t, a, k]
            for j in range(k):
                values[t, a, j] = betas[t, a, j] * factors[t, j]
                fitted += values[t, a, j]
            values[t, a, k] = rf[t]
            values[t, a, k + 1] = betas[t, a, k]
            values[t, a, k + 2] = actual - fitted
            total = 0.0
            for j in range(k + 3):
                valid = valid and np.isfinite(values[t, a, j])
                total += values[t, a, j]
            error = actual - total
            if not valid or not np.isfinite(error) or abs(error) > 1e-10 * max(1.0, abs(actual)):
                values[t, a, :] = np.nan
                status[t, a] = 7
            else:
                checks[t, a, 0] = total
                checks[t, a, 1] = error
    return values, checks, status


@njit(types.Tuple((F3, F2))(F2, F3, I2, int64, int64), cache=True, nogil=True)
def link_contributions_kernel(returns, components, status, start, end):
    """Link an explicit interval, never skipping gaps or silently moving its start.

    Curves: linked components, actual total, component total, error.
    Summary: linked components, days, valid_days, actual_days, total_return,
    contribution_sum, reconciliation_error, model_r2, residual_volatility,
    coverage, status (0 complete / 1 incomplete / 2 empty / 3 arithmetic failure).
    """
    assets, m = returns.shape[1], components.shape[2]
    length = max(0, end - start)
    curves = np.full((length, assets, m + 3), np.nan)
    summary = np.full((assets, m + 10), np.nan)
    for a in range(assets):
        wealth = 1.0
        linked = np.zeros(m)
        actual_alive, contribution_alive = True, True
        arithmetic_error = False
        paired, observed = 0, 0
        mean, sst, residual_mean, residual_m2, sse = 0.0, 0.0, 0.0, 0.0, 0.0
        for t in range(start, end):
            actual = returns[t, a]
            actual_valid = np.isfinite(actual) and actual > -1.0
            if actual_valid:
                observed += 1
            else:
                actual_alive = False
            complete = status[t, a] == 0 and actual_valid
            for j in range(m):
                complete = complete and np.isfinite(components[t, a, j])
            if complete:
                paired += 1
                y = actual - components[t, a, m - 3]  # R² on the dependent-return basis.
                delta = y - mean
                mean += delta / paired
                sst += delta * (y - mean)
                residual = components[t, a, m - 1]
                delta = residual - residual_mean
                residual_mean += delta / paired
                residual_m2 += delta * (residual - residual_mean)
                sse += residual * residual
            else:
                contribution_alive = False
            if actual_alive and contribution_alive:
                for j in range(m):
                    linked[j] += wealth * components[t, a, j]
                    if not np.isfinite(linked[j]):
                        contribution_alive = False
                        arithmetic_error = True
            if actual_alive:
                wealth *= 1.0 + actual
                if not np.isfinite(wealth):
                    actual_alive = False
                    contribution_alive = False
                    arithmetic_error = True
                else:
                    curves[t - start, a, m] = wealth - 1.0
            if actual_alive and contribution_alive:
                total = 0.0
                for j in range(m):
                    total += linked[j]
                error = wealth - 1.0 - total
                if not np.isfinite(total) or abs(error) > 1e-9 * max(1.0, abs(wealth)):
                    contribution_alive = False
                    arithmetic_error = True
                else:
                    curves[t - start, a, :m] = linked
                    curves[t - start, a, m + 1] = total
                    curves[t - start, a, m + 2] = error
        summary[a, m] = length
        summary[a, m + 1] = paired
        summary[a, m + 2] = observed
        summary[a, m + 8] = paired / length if length else np.nan
        summary[a, m + 9] = (2.0 if not length else 3.0 if arithmetic_error else
                              0.0 if actual_alive and contribution_alive else 1.0)
        if length and actual_alive:
            summary[a, m + 3] = wealth - 1.0
        if length and actual_alive and contribution_alive:
            summary[a, :m] = linked
            summary[a, m + 4] = curves[-1, a, m + 1]
            summary[a, m + 5] = curves[-1, a, m + 2]
        if paired >= 2:
            if sst > 1e-20:
                summary[a, m + 6] = 1.0 - sse / sst
            summary[a, m + 7] = np.sqrt(max(0.0, residual_m2) / (paired - 1.0) * 252.0)
    return curves, summary


KERNELS = (return_availability_kernel, exposure_path_kernel,
           daily_contributions_kernel, link_contributions_kernel)


def warm_attribution_kernels():
    t = np.arange(70, dtype=np.float64)
    x = np.ascontiguousarray(np.column_stack((np.sin(t) * .01, np.cos(t) * .01)))
    returns = np.ascontiguousarray((x[:, 0] * .6 + x[:, 1] * .4 + .0001).reshape(-1, 1))
    days = np.arange(70, dtype=np.int64)
    available = return_availability_kernel(days.reshape(-1, 1).copy())
    rf = np.zeros(70)
    for model in (0, 1):
        beta, stats = attribution_kernel(returns, x, rf, 50, model)
        for mode in (0, 1):
            path, meta, _, _ = exposure_path_kernel(
                returns, x, rf, available, days, beta, stats, 50, 40, 30, 5, mode, model)
            components, _, status = daily_contributions_kernel(returns, x, rf, path, meta)
            link_contributions_kernel(returns, components, status, 0, 70)
    return {"complete": all(len(k.nopython_signatures) == 1 and len(k.signatures) == 1 for k in KERNELS)}
