"""Fixed-signature model preparation, projection and valuation kernels.

Regression and factor projection reuse the existing numerical implementations.
Read-only input signatures also accept writable request-owned arrays without
compiling new signatures. No shared input is modified.
"""
from __future__ import annotations

import numpy as np
from numba import float64, int64, njit, types

from backend.compute_policy import validate_execution_audit
from backend.factor_research.numba_kernels import attribution_kernel
from backend.scenario_stress.numba_kernels import factor_to_asset_kernel

F1 = types.Array(float64, 1, "C")
F2 = types.Array(float64, 2, "C")
I1 = types.Array(int64, 1, "C")
I2 = types.Array(int64, 2, "C")
R1 = types.Array(float64, 1, "C", readonly=True)
R2 = types.Array(float64, 2, "C", readonly=True)
RI = types.Array(int64, 1, "C", readonly=True)
ENGINE_VERSION = "published-sensitivity-1.1"
_READY = False


@njit(types.Tuple((F1, I1, F1))(RI, RI, RI, R1, RI, RI, int64, int64, int64), cache=True, nogil=True)
def resample_transform_kernel(days, periods, available, values, grid, expected_days, cutoff, transform, require_endpoint):
    """Sample a complete calendar grid, then transform; a gap never bridges time."""
    if days.size != periods.size or days.size != available.size or days.size != values.size or grid.size != expected_days.size:
        raise ValueError("invalid resampling dimensions")
    count = grid.size
    sampled = np.full(count, np.nan)
    sampled_days = np.full(count, -1, dtype=np.int64)
    knowledge = np.full(count, -1, dtype=np.int64)
    for row in range(days.size):
        if days[row] > cutoff or available[row] > cutoff:
            continue
        index = np.searchsorted(grid, periods[row])
        if index < count and grid[index] == periods[row] and days[row] >= sampled_days[index]:
            sampled[index] = values[row]
            sampled_days[index] = days[row]
            knowledge[index] = available[row]
    if require_endpoint:
        for index in range(count):
            if sampled_days[index] != expected_days[index]:
                sampled[index] = np.nan
                knowledge[index] = -1
    output = np.full(count, np.nan)
    result_knowledge = np.full(count, -1, dtype=np.int64)
    for index in range(count):
        if not np.isfinite(sampled[index]):
            continue
        if transform == 3:
            output[index] = sampled[index]
            result_knowledge[index] = knowledge[index]
        elif index > 0 and np.isfinite(sampled[index - 1]):
            if transform == 0:
                if sampled[index] <= 0 or sampled[index - 1] <= 0:
                    continue
                output[index] = sampled[index] / sampled[index - 1] - 1.0
            else:
                output[index] = (sampled[index] - sampled[index - 1]) * (100.0 if transform == 1 else 1.0)
            result_knowledge[index] = max(knowledge[index], knowledge[index - 1])
    return output, result_knowledge, sampled


@njit(F2(R2, int64), cache=True, nogil=True)
def lag_features_kernel(values, lags):
    if lags < 0:
        raise ValueError("negative lag")
    rows, columns = values.shape
    output = np.full((rows, columns * (lags + 1)), np.nan)
    for row in range(rows):
        for lag in range(lags + 1):
            if row < lag:
                continue
            for column in range(columns):
                output[row, lag * columns + column] = values[row - lag, column]
    return output


@njit(types.Tuple((F2, F1, F1, int64))(R2, int64), cache=True, nogil=True)
def standardize_kernel(values, split):
    rows, columns = values.shape
    if split < 0 or split > rows or columns == 0:
        raise ValueError("invalid standardization window")
    means = np.zeros(columns)
    scales = np.zeros(columns)
    count = 0
    for row in range(split):
        valid = True
        for column in range(columns):
            valid = valid and np.isfinite(values[row, column])
        if valid:
            count += 1
            for column in range(columns):
                means[column] += values[row, column]
    output = np.full(values.shape, np.nan)
    if count < 2:
        return output, means, scales, 1
    for column in range(columns):
        means[column] /= count
    for row in range(split):
        valid = True
        for column in range(columns):
            valid = valid and np.isfinite(values[row, column])
        if valid:
            for column in range(columns):
                scales[column] += (values[row, column] - means[column]) ** 2
    for column in range(columns):
        scales[column] = np.sqrt(scales[column] / (count - 1))
        if not np.isfinite(scales[column]) or scales[column] <= 1e-14:
            return output, means, scales, 2
    for row in range(rows):
        for column in range(columns):
            output[row, column] = (values[row, column] - means[column]) / scales[column]
    return output, means, scales, 0


@njit(I2(R2, R2, RI, int64), cache=True, nogil=True)
def sample_metadata_kernel(inputs, responses, dates, split):
    """Count actual joint samples even when the regression cannot be solved."""
    if inputs.shape[0] != responses.shape[0] or dates.size != inputs.shape[0] or split < 0 or split > dates.size:
        raise ValueError("invalid sample metadata dimensions")
    result = np.zeros((responses.shape[1], 4), dtype=np.int64)
    result[:, 2:] = -1
    for row in range(dates.size):
        valid = True
        for column in range(inputs.shape[1]):
            valid = valid and np.isfinite(inputs[row, column])
        if not valid:
            continue
        part = 0 if row < split else 1
        for target in range(responses.shape[1]):
            if np.isfinite(responses[row, target]):
                result[target, part] += 1
                result[target, part + 2] = dates[row]
    return result


@njit(F2(R2, R1, R1), cache=True, nogil=True)
def restore_coefficients_kernel(coefficients, means, scales):
    output = np.full(coefficients.shape, np.nan)
    columns = scales.size
    for target in range(coefficients.shape[0]):
        intercept = coefficients[target, columns]
        for column in range(columns):
            output[target, column] = coefficients[target, column] / scales[column]
            intercept -= output[target, column] * means[column]
        output[target, columns] = intercept
    return output


@njit(I1(R2, R2, int64, int64, float64), cache=True, nogil=True)
def validation_status_kernel(coefficients, stats, min_train, min_validation, minimum_r2):
    status = np.zeros(coefficients.shape[0], dtype=np.int64)
    for target in range(coefficients.shape[0]):
        if stats[target, 0] < min_train or stats[target, 1] < min_validation:
            status[target] = 1
        elif stats[target, 7] != 0 or not np.isfinite(stats[target, 3]):
            status[target] = 2
        elif stats[target, 3] < minimum_r2:
            status[target] = 3
        for column in range(coefficients.shape[1]):
            if not np.isfinite(coefficients[target, column]):
                status[target] = 2
    return status


@njit(F2(R2, R2, int64), cache=True, nogil=True)
def transmission_kernel(shocks, coefficients, lags):
    """Conditional deviation response. Intercepts are not extra scenario shocks."""
    rows, inputs = shocks.shape
    if lags < 0 or coefficients.shape[1] != inputs * (lags + 1) + 1:
        raise ValueError("invalid transmission dimensions")
    output = np.zeros((rows, coefficients.shape[0]))
    for row in range(rows):
        for target in range(coefficients.shape[0]):
            for lag in range(lags + 1):
                if row >= lag:
                    for column in range(inputs):
                        output[row, target] += coefficients[target, lag * inputs + column] * shocks[row - lag, column]
    return output


@njit(F2(R2, RI), cache=True, nogil=True)
def display_shocks_kernel(values, unit_codes):
    """The only input boundary: percent return → decimal; bp/pp/points unchanged."""
    output = np.empty(values.shape)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            output[row, column] = values[row, column] / (100.0 if unit_codes[column] == 0 else 1.0)
    return output


@njit(F2(R2, RI, RI), cache=True, nogil=True)
def reference_responses_kernel(coefficients, input_units, output_units):
    """Readable sensitivity to +1% / +100bp / +1pp / +1 point, first lag only."""
    output = np.empty((coefficients.shape[0], input_units.size))
    for target in range(coefficients.shape[0]):
        for column in range(input_units.size):
            shock = 0.01 if input_units[column] == 0 else 100.0 if input_units[column] == 1 else 1.0
            output[target, column] = coefficients[target, column] * shock * (100.0 if output_units[target] == 0 else 1.0)
    return output


@njit(types.Tuple((F1, int64))(R1, R1, float64, int64), cache=True, nogil=True)
def cashflow_metrics_kernel(times, amounts, yield_percent, compounding):
    result = np.full(4, np.nan)
    if compounding <= 0 or not np.isfinite(yield_percent):
        return result, 1
    rate = yield_percent / 100.0
    base = 1.0 + rate / compounding
    if base <= 0 or times.size == 0 or times.size != amounts.size:
        return result, 1
    price = 0.0
    first = 0.0
    second = 0.0
    for index in range(times.size):
        time = times[index]
        amount = amounts[index]
        if not np.isfinite(time) or not np.isfinite(amount) or time <= 0 or amount <= 0:
            return result, 1
        present = amount * base ** (-compounding * time)
        price += present
        first += time * present / base
        second += time * (time + 1.0 / compounding) * present / (base * base)
    if not np.isfinite(price) or price <= 0 or not np.isfinite(first) or not np.isfinite(second):
        return result, 2
    result[0], result[1], result[2], result[3] = price, first / price, second / price, rate
    return result, 0


@njit(types.Tuple((F1, int64))(R1, R1, float64, int64, float64), cache=True, nogil=True)
def cashflow_shock_kernel(times, amounts, yield_percent, compounding, shock_bp):
    initial, status = cashflow_metrics_kernel(times, amounts, yield_percent, compounding)
    shocked, next_status = cashflow_metrics_kernel(times, amounts, yield_percent + shock_bp / 100.0, compounding)
    result = np.full(3, np.nan)
    if status or next_status:
        return result, 1
    change = shock_bp / 10000.0
    result[0] = shocked[0] / initial[0] - 1.0
    result[1] = -initial[1] * change + 0.5 * initial[2] * change * change
    result[2] = shocked[0]
    return result, 0


@njit(types.Tuple((F2, F1, F2, F2, int64))(R2, R1, int64), cache=True, nogil=True)
def wealth_impact_kernel(returns, weights, mode):
    """mode 0 buys and holds; mode 1 resets weights each period, zero costs explicit."""
    rows, assets = returns.shape
    if rows == 0 or assets == 0 or weights.size != assets or (mode != 0 and mode != 1):
        raise ValueError("invalid impact dimensions or holding policy")
    path = np.full((rows, 3), np.nan)
    contribution = np.zeros(assets)
    daily = np.zeros((rows, assets))
    beginnings = np.zeros((rows, assets))
    holdings = weights.copy()
    nav = 1.0
    peak = 1.0
    for row in range(rows):
        prior = nav
        change = 0.0
        for asset in range(assets):
            start = holdings[asset] if mode == 0 else prior * weights[asset]
            beginnings[row, asset] = start
            if start == 0.0:
                continue
            value = returns[row, asset]
            if not np.isfinite(value) or value <= -1.0:
                return path, contribution, daily, beginnings, 1
            delta = start * value
            daily[row, asset] = delta
            contribution[asset] += delta
            change += delta
            holdings[asset] += delta
        nav = prior + change
        if not np.isfinite(nav) or nav <= 0:
            return path, contribution, daily, beginnings, 2
        peak = max(peak, nav)
        path[row, 0] = change / prior
        path[row, 1] = nav
        path[row, 2] = 1.0 - nav / peak
    return path, contribution, daily, beginnings, 0


@njit(F1(R2, R2, R2, R2), cache=True, nogil=True)
def linked_factor_contributions_kernel(shocks, betas, beginnings, actual_returns):
    """Last column is explicit nonlinear revaluation remainder, not invented alpha."""
    totals = np.zeros(shocks.shape[1] + 1)
    for row in range(shocks.shape[0]):
        for asset in range(betas.shape[0]):
            wealth = beginnings[row, asset]
            if wealth == 0.0:
                continue
            explained = 0.0
            for factor in range(shocks.shape[1]):
                value = betas[asset, factor] * shocks[row, factor]
                totals[factor] += wealth * value
                explained += value
            totals[-1] += wealth * (actual_returns[row, asset] - explained)
    return totals


@njit(types.Tuple((F1, int64))(R1, R1), cache=True, nogil=True)
def ending_weights_kernel(beginning, last_returns):
    path, contributions, _, _, status = wealth_impact_kernel(last_returns.reshape(1, last_returns.size), beginning, 0)
    result = np.full(beginning.size, np.nan)
    if status:
        return result, status
    for asset in range(beginning.size):
        result[asset] = (beginning[asset] + contributions[asset]) / path[0, 1]
    return result, 0


@njit(F1(R2, R1, R1, float64), cache=True, nogil=True)
def impact_summary_kernel(path, contributions, factor_contributions, notional):
    if path.shape[0] == 0 or path.shape[1] != 3:
        raise ValueError("empty or invalid impact path")
    result = np.zeros(6)
    result[0] = path[-1, 1] - 1.0
    result[1] = path[-1, 1]
    result[2] = 0.0
    for row in range(path.shape[0]):
        result[2] = max(result[2], path[row, 2])
    result[3] = result[0] * notional
    result[4] = -result[0]
    result[5] = -result[0]
    for value in contributions:
        result[4] += value
    for value in factor_contributions:
        result[5] += value
    return result


KERNELS = (
    resample_transform_kernel, lag_features_kernel, standardize_kernel,
    restore_coefficients_kernel, sample_metadata_kernel, validation_status_kernel, transmission_kernel,
    display_shocks_kernel, reference_responses_kernel, cashflow_metrics_kernel,
    cashflow_shock_kernel, wealth_impact_kernel, linked_factor_contributions_kernel,
    ending_weights_kernel, impact_summary_kernel,
)
for dispatcher in KERNELS:
    dispatcher.disable_compile()


def execution_audit():
    used = (*KERNELS, attribution_kernel, factor_to_asset_kernel)
    if not _READY or any(not fn.nopython_signatures or len(fn.signatures) != len(fn.nopython_signatures) for fn in used):
        raise RuntimeError("敏感性计算内核未完成预热，禁止请求期编译或 Python 回退")
    return validate_execution_audit({
        "execution_backend": "numba_njit_fixed_signature", "engine_version": ENGINE_VERSION,
        "nopython": True, "python_fallback": 0, "python_operator_calls": 0,
        "request_time_compilation": 0,
        "kernel_signatures": {fn.py_func.__name__: [str(s) for s in fn.nopython_signatures] for fn in used},
    })


def warm_sensitivity_kernels():
    global _READY
    grid = np.arange(50, dtype=np.int64)
    values = np.arange(50, dtype=np.float64) + 100.0
    resample_transform_kernel(grid, grid, grid, values, grid, grid, np.int64(50), np.int64(0), np.int64(1))
    x = np.ascontiguousarray(values.reshape(-1, 1))
    features = lag_features_kernel(x, np.int64(1))
    standardized, mean, scale, _ = standardize_kernel(features, np.int64(40))
    coefficients, stats = attribution_kernel(x, standardized, np.zeros(50), np.int64(40), np.int64(1))
    restored = restore_coefficients_kernel(coefficients, mean, scale)
    sample_metadata_kernel(features, x, grid, np.int64(40))
    validation_status_kernel(restored, stats, np.int64(30), np.int64(10), np.float64(0.0))
    transmission_kernel(x, restored, np.int64(1))
    units = np.zeros(1, dtype=np.int64)
    display_shocks_kernel(x, units)
    reference_responses_kernel(restored, units, units)
    times, amounts = np.array([1.0, 2.0]), np.array([3.0, 103.0])
    cashflow_metrics_kernel(times, amounts, np.float64(3), np.int64(1))
    cashflow_shock_kernel(times, amounts, np.float64(3), np.int64(1), np.float64(100))
    shocks = np.array([[0.01], [-0.02]])
    beta = np.ones((1, 1))
    ret, _, _, _ = factor_to_asset_kernel(shocks, beta, np.zeros(1), np.ones(1, dtype=np.uint8))
    _, _, _, beginnings, _ = wealth_impact_kernel(ret, np.ones(1), np.int64(0))
    wealth_impact_kernel(ret, np.ones(1), np.int64(1))
    linked_factor_contributions_kernel(shocks, beta, beginnings, ret)
    ending_weights_kernel(np.ones(1), np.array([0.01]))
    path, contribution, _, _, _ = wealth_impact_kernel(ret, np.ones(1), np.int64(0))
    factors = linked_factor_contributions_kernel(shocks, beta, beginnings, ret)
    impact_summary_kernel(path, contribution, factors, np.float64(1.0))
    _READY = True
    return {"complete": True, **execution_audit()}
