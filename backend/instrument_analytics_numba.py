from __future__ import annotations

"""Fixed-signature nopython kernels for instrument and benchmark analytics.

Pandas is intentionally kept outside this module.  Callers may parse files,
dates and labels in Python, but every ordinary numerical operation in the
instrument analytics call graph is executed by one of the eagerly compiled
kernels below.
"""

import hashlib

import numpy as np
from numba import boolean, float64, int64, njit, types, uint8


INSTRUMENT_ANALYTICS_ENGINE_VERSION = "instrument-analytics-njit-1.0.0"
INSTRUMENT_ANALYTICS_KERNEL_VERSION = "instrument-statistics-quality-3"

NAT_DAY = np.int64(-9_223_372_036_854_775_808)

_F1 = float64[::1]
_I1 = int64[::1]
_I2 = int64[:, ::1]
_U1 = uint8[::1]

_SUM_COUNT_RESULT = types.Tuple((float64, int64))
_YEAR_RESULT = types.Tuple((_I1, _I1, _F1, _U1))
_CANDLE_RESULT = types.Tuple((_F1, int64))
_PRODUCT_COMPARE_RESULT = types.Tuple((_F1, _F1, _F1, _F1))
_CONTIGUOUS_BLOCKS_RESULT = types.Tuple((_I1, _I1))
_RETURN_SERIES_RESULT = types.Tuple((_F1, _F1, int64))
_DATA_QUALITY_MASKS_RESULT = types.Tuple((_U1, _U1, _U1, _U1, _U1))
_QUALITY_RESULT = types.Tuple(
    (int64, int64, int64, int64, int64, float64, int64, int64)
)


@njit(_RETURN_SERIES_RESULT(_F1), cache=False, nogil=True)
def simple_log_returns_kernel(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Derive simple and log returns from one positive finite value series."""

    output_size = values.size - 1 if values.size > 0 else 0
    simple_returns = np.empty(output_size, dtype=np.float64)
    log_returns = np.empty(output_size, dtype=np.float64)
    if values.size < 2:
        return simple_returns, log_returns, 1
    for index in range(1, values.size):
        previous = values[index - 1]
        current = values[index]
        if (
            not np.isfinite(previous)
            or not np.isfinite(current)
            or previous <= 0.0
            or current <= 0.0
        ):
            return simple_returns, log_returns, 2
        ratio = current / previous
        if not np.isfinite(ratio) or ratio <= 0.0:
            return simple_returns, log_returns, 2
        simple_returns[index - 1] = ratio - 1.0
        log_returns[index - 1] = np.log(ratio)
    return simple_returns, log_returns, 0


@njit(
    _DATA_QUALITY_MASKS_RESULT(
        _F1,
        _I1,
        _F1,
        _U1,
        _F1,
        int64,
        float64,
    ),
    cache=False,
    nogil=True,
)
def data_quality_masks_kernel(
    latest_nav: np.ndarray,
    latest_date_ns: np.ndarray,
    anomaly_counts: np.ndarray,
    active_mask: np.ndarray,
    stale_days: np.ndarray,
    future_limit_ns: int,
    stale_active_days: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Classify all numerical data-quality conditions in one NJIT pass."""

    size = latest_nav.size
    if not (
        latest_date_ns.size == size
        and anomaly_counts.size == size
        and active_mask.size == size
        and stale_days.size == size
    ):
        raise ValueError("data quality arrays must have the same length")
    missing_nav = np.zeros(size, dtype=np.uint8)
    invalid_nav = np.zeros(size, dtype=np.uint8)
    future_date = np.zeros(size, dtype=np.uint8)
    anomaly = np.zeros(size, dtype=np.uint8)
    stale_active = np.zeros(size, dtype=np.uint8)
    for index in range(size):
        nav_value = latest_nav[index]
        if np.isnan(nav_value):
            missing_nav[index] = 1
        elif not np.isfinite(nav_value) or nav_value <= 0.0:
            invalid_nav[index] = 1
        date_value = latest_date_ns[index]
        if date_value != NAT_DAY and date_value > future_limit_ns:
            future_date[index] = 1
        anomaly_value = anomaly_counts[index]
        if np.isfinite(anomaly_value) and anomaly_value > 0.0:
            anomaly[index] = 1
        stale_value = stale_days[index]
        if (
            active_mask[index] != 0
            and np.isfinite(stale_value)
            and stale_value > stale_active_days
        ):
            stale_active[index] = 1
    return missing_nav, invalid_nav, future_date, anomaly, stale_active


@njit(_SUM_COUNT_RESULT(_F1), cache=False, nogil=True)
def finite_sum_count_kernel(values: np.ndarray) -> tuple[float, int]:
    total = 0.0
    count = 0
    for value in values:
        if np.isfinite(value):
            total += value
            count += 1
    return total, count


@njit(float64(_F1), cache=False, nogil=True)
def positive_finite_sum_kernel(values: np.ndarray) -> float:
    """Sum only finite, strictly positive observations."""

    total = 0.0
    for value in values:
        if np.isfinite(value) and value > 0.0:
            total += value
    return total


@njit(int64(_I1, int64, int64), cache=False, nogil=True)
def int_range_count_kernel(values: np.ndarray, lower: int, upper: int) -> int:
    """Count integer observations inside the inclusive [lower, upper] range."""

    if lower > upper:
        return 0
    count = 0
    for value in values:
        if lower <= value <= upper:
            count += 1
    return count


@njit(int64(_I1, int64), cache=False, nogil=True)
def int_less_than_count_kernel(values: np.ndarray, target: int) -> int:
    count = 0
    for value in values:
        if value < target:
            count += 1
    return count


@njit(_CONTIGUOUS_BLOCKS_RESULT(_U1), cache=False, nogil=True)
def contiguous_blocks_kernel(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return true positions and internal split offsets for contiguous blocks."""

    position_count = 0
    for value in mask:
        if value != 0:
            position_count += 1
    positions = np.empty(position_count, dtype=np.int64)
    output_index = 0
    for index in range(mask.size):
        if mask[index] != 0:
            positions[output_index] = index
            output_index += 1
    boundary_count = 0
    for index in range(1, positions.size):
        if positions[index] != positions[index - 1] + 1:
            boundary_count += 1
    split_boundaries = np.empty(boundary_count, dtype=np.int64)
    output_index = 0
    for index in range(1, positions.size):
        if positions[index] != positions[index - 1] + 1:
            split_boundaries[output_index] = index
            output_index += 1
    return positions, split_boundaries


@njit(float64(_F1, int64), cache=False, nogil=True)
def finite_mean_kernel(values: np.ndarray, minimum_count: int) -> float:
    total = 0.0
    count = 0
    for value in values:
        if np.isfinite(value):
            total += value
            count += 1
    if count == 0 or count < minimum_count:
        return np.nan
    return total / count


@njit(float64(_F1, int64), cache=False, nogil=True)
def numeric_stat_kernel(values: np.ndarray, operation: int) -> float:
    """Return finite-only mean=0, sum=1 or linear median=2."""

    count = 0
    total = 0.0
    for value in values:
        if np.isfinite(value):
            count += 1
            total += value
    if count == 0:
        return np.nan
    if operation == 1:
        return total
    if operation == 0:
        return total / count
    if operation != 2:
        raise ValueError("unsupported numeric statistic")
    compact = np.empty(count, dtype=np.float64)
    position = 0
    for value in values:
        if np.isfinite(value):
            compact[position] = value
            position += 1
    compact.sort()
    middle = count // 2
    if count % 2:
        return compact[middle]
    return (compact[middle - 1] + compact[middle]) / 2.0


@njit(_U1(_F1, float64, float64, int64), cache=False, nogil=True)
def numeric_comparison_mask_kernel(
    values: np.ndarray,
    target: float,
    input_scale: float,
    operation: int,
) -> np.ndarray:
    """Compare scaled finite values: gte=0, lte=1, gt=2, lt=3, eq=4."""

    output = np.zeros(values.size, dtype=np.uint8)
    for index in range(values.size):
        raw_value = values[index]
        if not np.isfinite(raw_value):
            continue
        value = raw_value * input_scale
        if not np.isfinite(value):
            continue
        matched = False
        if operation == 0:
            matched = value >= target
        elif operation == 1:
            matched = value <= target
        elif operation == 2:
            matched = value > target
        elif operation == 3:
            matched = value < target
        elif operation == 4:
            difference = abs(value - target)
            tolerance = 1e-9 + 1e-9 * abs(target)
            matched = difference <= tolerance
        else:
            raise ValueError("unsupported numeric comparison")
        if matched:
            output[index] = 1
    return output


@njit(_U1(_F1), cache=False, nogil=True)
def finite_mask_kernel(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.size, dtype=np.uint8)
    for index in range(values.size):
        if np.isfinite(values[index]):
            output[index] = 1
    return output


@njit(_U1(_F1), cache=False, nogil=True)
def positive_finite_mask_kernel(values: np.ndarray) -> np.ndarray:
    output = np.zeros(values.size, dtype=np.uint8)
    for index in range(values.size):
        value = values[index]
        if np.isfinite(value) and value > 0.0:
            output[index] = 1
    return output


@njit(_U1(_F1, _F1), cache=False, nogil=True)
def positive_pair_mask_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size != right.size:
        raise ValueError("paired arrays must have the same length")
    output = np.zeros(left.size, dtype=np.uint8)
    for index in range(left.size):
        left_value = left[index]
        right_value = right[index]
        if (
            np.isfinite(left_value)
            and left_value > 0.0
            and np.isfinite(right_value)
            and right_value > 0.0
        ):
            output[index] = 1
    return output


@njit(_I1(_F1), cache=False, nogil=True)
def fee_bucket_counts_kernel(values: np.ndarray) -> np.ndarray:
    """Return counts for <=.25, (.25,.5], (.5,1], >1 and unavailable."""

    counts = np.zeros(5, dtype=np.int64)
    for value in values:
        if not np.isfinite(value):
            counts[4] += 1
        elif value <= 0.25:
            counts[0] += 1
        elif value <= 0.50:
            counts[1] += 1
        elif value <= 1.00:
            counts[2] += 1
        else:
            counts[3] += 1
    return counts


@njit(_I1(_I1), cache=False, nogil=True)
def status_counts_kernel(status_codes: np.ndarray) -> np.ndarray:
    """Count encoded active=0, issuing=1, inactive=2 and unknown=3."""

    counts = np.zeros(4, dtype=np.int64)
    for code in status_codes:
        if 0 <= code <= 2:
            counts[code] += 1
        else:
            counts[3] += 1
    return counts


@njit(int64(_U1), cache=False, nogil=True)
def count_true_kernel(mask: np.ndarray) -> int:
    count = 0
    for value in mask:
        if value != 0:
            count += 1
    return count


@njit(_I1(_I1, int64), cache=False, nogil=True)
def encoded_category_counts_kernel(
    category_codes: np.ndarray,
    category_count: int,
) -> np.ndarray:
    """Count Python-mapped text categories without performing math in Python."""

    if category_count < 0:
        raise ValueError("category count must be non-negative")
    counts = np.zeros(category_count, dtype=np.int64)
    for code in category_codes:
        if code < 0:
            continue
        if code >= category_count:
            raise ValueError("category code is outside the declared range")
        counts[code] += 1
    return counts


@njit(int64(_I1), cache=False, nogil=True)
def encoded_unique_count_kernel(category_codes: np.ndarray) -> int:
    """Count distinct non-negative Python-mapped text category codes."""

    maximum_code = -1
    for code in category_codes:
        if code > maximum_code:
            maximum_code = code
    if maximum_code < 0:
        return 0
    seen = np.zeros(maximum_code + 1, dtype=np.uint8)
    unique_count = 0
    for code in category_codes:
        if code >= 0 and seen[code] == 0:
            seen[code] = 1
            unique_count += 1
    return unique_count


@njit(float64(int64, int64), cache=False, nogil=True)
def coverage_ratio_kernel(covered: int, total: int) -> float:
    if total <= 0:
        return np.nan
    value = covered / total
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@njit(_I1(_I2), cache=False, nogil=True)
def aggregate_count_rows_kernel(rows: np.ndarray) -> np.ndarray:
    if rows.ndim != 2:
        raise ValueError("count matrix must be two-dimensional")
    output = np.zeros(rows.shape[1], dtype=np.int64)
    for row in range(rows.shape[0]):
        for column in range(rows.shape[1]):
            output[column] += rows[row, column]
    return output


@njit(_YEAR_RESULT(_I1, _F1), cache=False, nogil=True)
def yearly_event_aggregation_kernel(
    years: np.ndarray,
    issue_amounts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if years.size != issue_amounts.size:
        raise ValueError("year and issue-amount arrays must have the same length")
    if years.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.uint8),
        )

    order = np.argsort(years)
    unique_count = 1
    for position in range(1, order.size):
        if years[order[position]] != years[order[position - 1]]:
            unique_count += 1

    output_years = np.empty(unique_count, dtype=np.int64)
    counts = np.zeros(unique_count, dtype=np.int64)
    totals = np.zeros(unique_count, dtype=np.float64)
    has_total = np.zeros(unique_count, dtype=np.uint8)
    output_index = -1
    previous_year = np.int64(0)
    for position in range(order.size):
        source_index = order[position]
        year = years[source_index]
        if position == 0 or year != previous_year:
            output_index += 1
            output_years[output_index] = year
            previous_year = year
        counts[output_index] += 1
        amount = issue_amounts[source_index]
        if np.isfinite(amount):
            totals[output_index] += amount
            has_total[output_index] = 1
    return output_years, counts, totals, has_total


@njit(_F1(_F1, _F1, _F1, _F1, int64), cache=False, nogil=True)
def nav_metrics_kernel(
    one_month_nav: np.ndarray,
    three_month_nav: np.ndarray,
    one_year_nav: np.ndarray,
    three_year_nav: np.ndarray,
    three_year_elapsed_days: int,
) -> np.ndarray:
    """Return 1m/3m/1y/3y returns, vol, drawdown, Sharpe and Calmar."""

    output = np.empty(8, dtype=np.float64)
    for index in range(output.size):
        output[index] = np.nan

    windows = (one_month_nav, three_month_nav, one_year_nav, three_year_nav)
    for window_index in range(4):
        values = windows[window_index]
        if values.size >= 2:
            first = values[0]
            latest = values[values.size - 1]
            if np.isfinite(first) and first > 0.0 and np.isfinite(latest):
                result = latest / first - 1.0
                if np.isfinite(result):
                    output[window_index] = result

    return_count = one_year_nav.size - 1
    if return_count >= 60:
        returns = np.empty(return_count, dtype=np.float64)
        valid_count = 0
        for index in range(1, one_year_nav.size):
            previous = one_year_nav[index - 1]
            current = one_year_nav[index]
            if np.isfinite(previous) and previous != 0.0 and np.isfinite(current):
                value = current / previous - 1.0
                if np.isfinite(value):
                    returns[valid_count] = value
                    valid_count += 1
        if valid_count >= 60:
            total = 0.0
            for index in range(valid_count):
                total += returns[index]
            mean = total / valid_count
            variance_sum = 0.0
            for index in range(valid_count):
                delta = returns[index] - mean
                variance_sum += delta * delta
            if valid_count > 1:
                daily_std = np.sqrt(variance_sum / (valid_count - 1))
                if np.isfinite(daily_std):
                    output[4] = daily_std * np.sqrt(252.0)
                    if daily_std > 1e-12:
                        output[6] = mean / daily_std * np.sqrt(252.0)

    if three_year_nav.size - 1 >= 180:
        peak = three_year_nav[0]
        max_drawdown = 0.0
        for value in three_year_nav:
            if value > peak:
                peak = value
            if peak > 0.0:
                drawdown = value / peak - 1.0
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
        output[5] = max_drawdown
        if (
            three_year_elapsed_days > 0
            and abs(max_drawdown) > 1e-12
            and three_year_nav[0] > 0.0
        ):
            annualized = (
                (three_year_nav[three_year_nav.size - 1] / three_year_nav[0])
                ** (365.25 / three_year_elapsed_days)
                - 1.0
            )
            if np.isfinite(annualized):
                output[7] = annualized / abs(max_drawdown)
    return output


@njit(
    _PRODUCT_COMPARE_RESULT(_F1, _I1, int64, float64, float64),
    cache=False,
    nogil=True,
)
def product_compare_analysis_kernel(
    nav_values: np.ndarray,
    date_days: np.ndarray,
    rolling_window_days: int,
    management_fee: float,
    custody_fee: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute one comparison window and its chart series.

    Metric order is cumulative return, annualized return, annualized
    volatility, maximum drawdown, total fee, return-to-fee, Sharpe and Calmar.
    Return and risk values use percentage points, while ratios are unitless.
    """

    if nav_values.size != date_days.size:
        raise ValueError("产品净值与日期数量不一致")
    if nav_values.size < 2:
        raise ValueError("所选区间至少需要两个净值观察值")
    if rolling_window_days < 2:
        raise ValueError("滚动波动窗口至少为 2 天")
    if (
        (not np.isnan(management_fee) and not np.isfinite(management_fee))
        or (not np.isnan(custody_fee) and not np.isfinite(custody_fee))
    ):
        raise ValueError("管理费与托管费必须是有限数")
    if (
        (np.isfinite(management_fee) and management_fee < 0.0)
        or (np.isfinite(custody_fee) and custody_fee < 0.0)
    ):
        raise ValueError("管理费与托管费不能为负数")

    observation_count = nav_values.size
    normalized = np.empty(observation_count, dtype=np.float64)
    drawdown = np.empty(observation_count, dtype=np.float64)
    rolling_volatility = np.empty(observation_count, dtype=np.float64)
    returns = np.empty(observation_count - 1, dtype=np.float64)
    metrics = np.empty(8, dtype=np.float64)
    for index in range(metrics.size):
        metrics[index] = np.nan
    for index in range(rolling_volatility.size):
        rolling_volatility[index] = np.nan

    base_value = nav_values[0]
    previous_day = date_days[0]
    if not np.isfinite(base_value) or base_value <= 0.0:
        raise ValueError("产品净值必须是有限正数")
    for index in range(observation_count):
        value = nav_values[index]
        day = date_days[index]
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("产品净值必须是有限正数")
        if index > 0 and day <= previous_day:
            raise ValueError("产品日期必须严格递增")
        normalized[index] = value / base_value
        if index > 0:
            returns[index - 1] = value / nav_values[index - 1] - 1.0
        previous_day = day

    metrics[0] = (normalized[observation_count - 1] - 1.0) * 100.0
    elapsed_days = date_days[observation_count - 1] - date_days[0]
    if elapsed_days <= 0:
        raise ValueError("所选区间日期跨度必须大于 0")
    annualized_growth = normalized[observation_count - 1] ** (365.0 / elapsed_days)
    if np.isfinite(annualized_growth):
        metrics[1] = (annualized_growth - 1.0) * 100.0

    return_count = returns.size
    if return_count > 1:
        total_return = 0.0
        for value in returns:
            if not np.isfinite(value):
                raise ValueError("产品收益率必须为有限数")
            total_return += value
        mean_return = total_return / return_count
        variance_sum = 0.0
        for value in returns:
            difference = value - mean_return
            variance_sum += difference * difference
        daily_volatility = np.sqrt(max(variance_sum / (return_count - 1), 0.0))
        metrics[2] = daily_volatility * np.sqrt(252.0) * 100.0
        if metrics[2] > 1e-12 and np.isfinite(metrics[1]):
            metrics[6] = metrics[1] / metrics[2]

    peak = normalized[0]
    maximum_drawdown = 0.0
    for index in range(observation_count):
        value = normalized[index]
        if value > peak:
            peak = value
        current_drawdown = (value / peak - 1.0) * 100.0
        drawdown[index] = current_drawdown
        if current_drawdown < maximum_drawdown:
            maximum_drawdown = current_drawdown
    metrics[3] = maximum_drawdown
    if maximum_drawdown < -1e-12 and np.isfinite(metrics[1]):
        metrics[7] = metrics[1] / abs(maximum_drawdown)

    has_management_fee = np.isfinite(management_fee)
    has_custody_fee = np.isfinite(custody_fee)
    if has_management_fee or has_custody_fee:
        total_fee = 0.0
        if has_management_fee:
            total_fee += management_fee
        if has_custody_fee:
            total_fee += custody_fee
        metrics[4] = total_fee
        if abs(total_fee) > 1e-12:
            metrics[5] = metrics[0] / total_fee

    for nav_index in range(2, observation_count):
        start_return_index = nav_index - rolling_window_days + 1
        if start_return_index < 1:
            start_return_index = 1
        sample_count = nav_index - start_return_index + 1
        if sample_count < 2:
            continue
        sample_total = 0.0
        for return_nav_index in range(start_return_index, nav_index + 1):
            sample_total += returns[return_nav_index - 1]
        sample_mean = sample_total / sample_count
        sample_variance_sum = 0.0
        for return_nav_index in range(start_return_index, nav_index + 1):
            difference = returns[return_nav_index - 1] - sample_mean
            sample_variance_sum += difference * difference
        rolling_volatility[nav_index] = (
            np.sqrt(max(sample_variance_sum / (sample_count - 1), 0.0))
            * np.sqrt(252.0)
            * 100.0
        )

    return metrics, normalized, drawdown, rolling_volatility


@njit(_CANDLE_RESULT(_F1, _F1, _F1, _F1), cache=False, nogil=True)
def candle_metrics_kernel(
    close: np.ndarray,
    amount: np.ndarray,
    volume: np.ndarray,
    unit_nav_by_row: np.ndarray,
) -> tuple[np.ndarray, int]:
    if not (
        close.size == amount.size
        and close.size == volume.size
        and close.size == unit_nav_by_row.size
    ):
        raise ValueError("candle arrays must have the same length")
    output = np.empty(5, dtype=np.float64)
    for index in range(output.size):
        output[index] = np.nan
    if close.size == 0:
        return output, -1

    output[0] = close[close.size - 1]
    common_index = -1
    for reverse_index in range(close.size):
        index = close.size - 1 - reverse_index
        unit_nav = unit_nav_by_row[index]
        if np.isfinite(unit_nav) and unit_nav > 0.0:
            premium = close[index] / unit_nav - 1.0
            if np.isfinite(premium):
                output[1] = unit_nav
                output[2] = premium
                common_index = index
                break

    tail_start = close.size - 20 if close.size >= 20 else 0
    amount_sum = 0.0
    volume_sum = 0.0
    amount_count = 0
    volume_count = 0
    for index in range(tail_start, close.size):
        amount_value = amount[index]
        if np.isfinite(amount_value):
            amount_sum += amount_value
            amount_count += 1
        volume_value = volume[index]
        if np.isfinite(volume_value):
            volume_sum += volume_value
            volume_count += 1
    if amount_count >= 20:
        output[3] = amount_sum / amount_count
    if volume_count >= 20:
        output[4] = volume_sum / volume_count
    return output, common_index


@njit(_F1(_F1, _F1), cache=False, nogil=True)
def latest_share_metrics_kernel(
    total_share: np.ndarray,
    unit_nav: np.ndarray,
) -> np.ndarray:
    if total_share.size != unit_nav.size:
        raise ValueError("share and NAV arrays must have the same length")
    output = np.empty(3, dtype=np.float64)
    for index in range(output.size):
        output[index] = np.nan
    for reverse_index in range(total_share.size):
        index = total_share.size - 1 - reverse_index
        share = total_share[index]
        nav = unit_nav[index]
        if np.isfinite(share) and share > 0.0 and np.isfinite(nav) and nav > 0.0:
            size = share * nav
            if np.isfinite(size):
                output[0] = share
                output[1] = nav
                output[2] = size
                return output
    return output


@njit(_I1(_I1), cache=False, nogil=True)
def stale_days_kernel(date_days: np.ndarray) -> np.ndarray:
    output = np.empty(date_days.size, dtype=np.int64)
    latest = NAT_DAY
    for value in date_days:
        if value != NAT_DAY and value > latest:
            latest = value
    if latest == NAT_DAY:
        for index in range(output.size):
            output[index] = -1
        return output
    for index in range(date_days.size):
        value = date_days[index]
        if value == NAT_DAY:
            output[index] = -1
        else:
            difference = latest - value
            output[index] = difference if difference >= 0 else 0
    return output


@njit(_U1(_I1, int64), cache=False, nogil=True)
def fresh_date_mask_kernel(date_days: np.ndarray, maximum_stale_days: int) -> np.ndarray:
    stale = stale_days_kernel(date_days)
    output = np.zeros(date_days.size, dtype=np.uint8)
    for index in range(date_days.size):
        if stale[index] >= 0 and stale[index] <= maximum_stale_days:
            output[index] = 1
    return output


@njit(
    boolean(int64, int64, _F1, _I1, uint8),
    cache=False,
    nogil=True,
    inline="always",
)
def _rank_before(
    left: int,
    right: int,
    values: np.ndarray,
    code_rank: np.ndarray,
    ascending: int,
) -> bool:
    left_value = values[left]
    right_value = values[right]
    if left_value == right_value:
        return code_rank[left] <= code_rank[right]
    if ascending != 0:
        return left_value < right_value
    return left_value > right_value


@njit(_I1(_F1, _I1, uint8), cache=False, nogil=True)
def ranking_order_kernel(
    values: np.ndarray,
    code_rank: np.ndarray,
    ascending: int,
) -> np.ndarray:
    if values.size != code_rank.size:
        raise ValueError("ranking arrays must have the same length")
    finite_count = 0
    for value in values:
        if np.isfinite(value):
            finite_count += 1
    indexes = np.empty(finite_count, dtype=np.int64)
    position = 0
    for index in range(values.size):
        if np.isfinite(values[index]):
            indexes[position] = index
            position += 1
    if indexes.size < 2:
        return indexes

    scratch = np.empty_like(indexes)
    width = 1
    while width < indexes.size:
        start = 0
        while start < indexes.size:
            middle = min(start + width, indexes.size)
            end = min(start + 2 * width, indexes.size)
            left = start
            right = middle
            destination = start
            while left < middle and right < end:
                if _rank_before(indexes[left], indexes[right], values, code_rank, ascending):
                    scratch[destination] = indexes[left]
                    left += 1
                else:
                    scratch[destination] = indexes[right]
                    right += 1
                destination += 1
            while left < middle:
                scratch[destination] = indexes[left]
                left += 1
                destination += 1
            while right < end:
                scratch[destination] = indexes[right]
                right += 1
                destination += 1
            start += 2 * width
        temporary = indexes
        indexes = scratch
        scratch = temporary
        width *= 2
    return indexes


@njit(
    boolean(int64, int64, _F1, _I1, uint8),
    cache=False,
    nogil=True,
    inline="always",
)
def _sort_before_with_missing(
    left: int,
    right: int,
    values: np.ndarray,
    stable_rank: np.ndarray,
    ascending: int,
) -> bool:
    left_finite = np.isfinite(values[left])
    right_finite = np.isfinite(values[right])
    if left_finite and not right_finite:
        return True
    if not left_finite and right_finite:
        return False
    if not left_finite and not right_finite:
        return stable_rank[left] <= stable_rank[right]
    if values[left] == values[right]:
        return stable_rank[left] <= stable_rank[right]
    if ascending != 0:
        return values[left] < values[right]
    return values[left] > values[right]


@njit(_I1(_F1, _I1, uint8), cache=False, nogil=True)
def numeric_sort_order_kernel(
    values: np.ndarray,
    stable_rank: np.ndarray,
    ascending: int,
) -> np.ndarray:
    if values.size != stable_rank.size:
        raise ValueError("sort arrays must have the same length")
    indexes = np.arange(values.size, dtype=np.int64)
    if indexes.size < 2:
        return indexes
    scratch = np.empty_like(indexes)
    width = 1
    while width < indexes.size:
        start = 0
        while start < indexes.size:
            middle = min(start + width, indexes.size)
            end = min(start + 2 * width, indexes.size)
            left = start
            right = middle
            destination = start
            while left < middle and right < end:
                if _sort_before_with_missing(
                    indexes[left], indexes[right], values, stable_rank, ascending
                ):
                    scratch[destination] = indexes[left]
                    left += 1
                else:
                    scratch[destination] = indexes[right]
                    right += 1
                destination += 1
            while left < middle:
                scratch[destination] = indexes[left]
                left += 1
                destination += 1
            while right < end:
                scratch[destination] = indexes[right]
                right += 1
                destination += 1
            start += 2 * width
        temporary = indexes
        indexes = scratch
        scratch = temporary
        width *= 2
    return indexes


@njit(
    _U1(_F1, _F1, _F1, float64, float64, float64),
    cache=False,
    nogil=True,
)
def adjusted_nav_anomaly_mask_kernel(
    adjusted_nav: np.ndarray,
    accumulated_nav: np.ndarray,
    unit_nav: np.ndarray,
    dislocation_threshold: float,
    reference_stable_threshold: float,
    extreme_threshold: float,
) -> np.ndarray:
    if not (
        adjusted_nav.size == accumulated_nav.size
        and adjusted_nav.size == unit_nav.size
    ):
        raise ValueError("NAV arrays must have the same length")
    output = np.zeros(adjusted_nav.size, dtype=np.uint8)
    previous_index = -1
    for index in range(adjusted_nav.size):
        current = adjusted_nav[index]
        if not (np.isfinite(current) and current > 0.0):
            continue
        if previous_index < 0:
            previous_index = index
            continue
        previous = adjusted_nav[previous_index]
        adjusted_return = current / previous - 1.0
        reference_stable = False
        previous_accumulated = accumulated_nav[previous_index]
        current_accumulated = accumulated_nav[index]
        if (
            np.isfinite(previous_accumulated)
            and previous_accumulated > 0.0
            and np.isfinite(current_accumulated)
            and current_accumulated > 0.0
        ):
            reference_return = current_accumulated / previous_accumulated - 1.0
            reference_stable = abs(reference_return) <= reference_stable_threshold
        previous_unit = unit_nav[previous_index]
        current_unit = unit_nav[index]
        if (
            np.isfinite(previous_unit)
            and previous_unit > 0.0
            and np.isfinite(current_unit)
            and current_unit > 0.0
        ):
            reference_return = current_unit / previous_unit - 1.0
            if abs(reference_return) <= reference_stable_threshold:
                reference_stable = True
        if (
            abs(adjusted_return) > extreme_threshold
            or (abs(adjusted_return) > dislocation_threshold and reference_stable)
        ):
            output[index] = 1
        previous_index = index
    return output


@njit(
    _QUALITY_RESULT(
        _I1,
        int64,
        int64,
        _I1,
        _I1,
        int64,
        float64,
        int64,
    ),
    cache=False,
    nogil=True,
)
def period_window_quality_kernel(
    observed_days: np.ndarray,
    target_day: int,
    effective_day: int,
    expected_days: np.ndarray,
    anomaly_days: np.ndarray,
    start_tolerance_days: int,
    required_coverage: float,
    maximum_missing_run: int,
) -> tuple[int, int, int, int, int, float, int, int]:
    """Evaluate a complete period window.

    Result fields are complete, reason code, anchor day, observations, expected,
    coverage, longest missing run and anomaly count.  Reason codes map to the
    public Python strings in :mod:`series_quality`.
    """

    anchor_day = NAT_DAY
    for value in observed_days:
        if value <= target_day and value > anchor_day:
            anchor_day = value
    if anchor_day == NAT_DAY:
        return 0, 1, NAT_DAY, 0, 0, np.nan, 0, 0

    observation_count = 0
    for value in observed_days:
        if anchor_day <= value <= effective_day:
            observation_count += 1
    if target_day - anchor_day > start_tolerance_days:
        return 0, 2, anchor_day, observation_count, 0, np.nan, 0, 0

    expected_count = expected_days.size
    present_count = 0
    longest_missing = 0
    current_missing = 0
    observed_index = 0
    for expected in expected_days:
        while observed_index < observed_days.size and observed_days[observed_index] < expected:
            observed_index += 1
        if observed_index < observed_days.size and observed_days[observed_index] == expected:
            present_count += 1
            current_missing = 0
        else:
            current_missing += 1
            if current_missing > longest_missing:
                longest_missing = current_missing

    coverage = np.nan
    if expected_count > 0:
        coverage = present_count / expected_count
        if coverage > 1.0:
            coverage = 1.0

    anomaly_count = 0
    for value in anomaly_days:
        if anchor_day < value <= effective_day:
            anomaly_count += 1

    reason = 0
    if anomaly_count > 0:
        reason = 3
    elif observation_count < 2:
        reason = 4
    elif expected_count > 0 and coverage < required_coverage:
        reason = 5
    elif longest_missing > maximum_missing_run:
        reason = 6
    return (
        1 if reason == 0 else 0,
        reason,
        anchor_day,
        observation_count,
        expected_count,
        coverage,
        longest_missing,
        anomaly_count,
    )


_PUBLIC_KERNELS = (
    simple_log_returns_kernel,
    data_quality_masks_kernel,
    finite_sum_count_kernel,
    positive_finite_sum_kernel,
    int_range_count_kernel,
    int_less_than_count_kernel,
    contiguous_blocks_kernel,
    finite_mean_kernel,
    numeric_stat_kernel,
    numeric_comparison_mask_kernel,
    finite_mask_kernel,
    positive_finite_mask_kernel,
    positive_pair_mask_kernel,
    fee_bucket_counts_kernel,
    status_counts_kernel,
    count_true_kernel,
    encoded_category_counts_kernel,
    encoded_unique_count_kernel,
    coverage_ratio_kernel,
    aggregate_count_rows_kernel,
    yearly_event_aggregation_kernel,
    nav_metrics_kernel,
    product_compare_analysis_kernel,
    candle_metrics_kernel,
    latest_share_metrics_kernel,
    stale_days_kernel,
    fresh_date_mask_kernel,
    _rank_before,
    ranking_order_kernel,
    _sort_before_with_missing,
    numeric_sort_order_kernel,
    adjusted_nav_anomaly_mask_kernel,
    period_window_quality_kernel,
)

for _kernel in _PUBLIC_KERNELS:
    _kernel.disable_compile()


def instrument_analytics_numba_execution_audit() -> dict[str, object]:
    signatures = {
        kernel.py_func.__name__: [str(signature) for signature in kernel.signatures]
        for kernel in _PUBLIC_KERNELS
    }
    compiled_count = sum(1 for values in signatures.values() if len(values) == 1)
    material = "|".join(
        [INSTRUMENT_ANALYTICS_ENGINE_VERSION, INSTRUMENT_ANALYTICS_KERNEL_VERSION]
        + [f"{name}:{','.join(values)}" for name, values in sorted(signatures.items())]
    )
    return {
        "engine": INSTRUMENT_ANALYTICS_ENGINE_VERSION,
        "backend": "numba_njit_fixed_signature",
        "kernel_version": INSTRUMENT_ANALYTICS_KERNEL_VERSION,
        "kernel_coverage": f"{compiled_count}/{len(signatures)}",
        "kernel_signatures": signatures,
        "kernel_fingerprint": hashlib.sha256(material.encode("utf-8")).hexdigest(),
        "nopython": all(bool(kernel.nopython_signatures) for kernel in _PUBLIC_KERNELS),
        "object_mode": 0,
        "python_fallback": 0,
    }


def warm_instrument_analytics_numba_kernels() -> dict[str, object]:
    try:
        from backend.compute_policy import validate_execution_audit
    except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
        from compute_policy import validate_execution_audit

    values = np.ascontiguousarray(np.array([0.1, 0.3, np.nan, 1.2], dtype=np.float64))
    integers = np.ascontiguousarray(np.array([0, 1, 2, 3], dtype=np.int64))
    mask = np.ascontiguousarray(np.array([1, 0, 1], dtype=np.uint8))
    simple_log_returns_kernel(
        np.ascontiguousarray(np.array([1.0, 1.01, 1.02], dtype=np.float64))
    )
    data_quality_masks_kernel(
        values,
        np.ascontiguousarray(
            np.array([NAT_DAY, 1, 2, 3], dtype=np.int64)
        ),
        values,
        np.ascontiguousarray(np.array([1, 1, 0, 1], dtype=np.uint8)),
        values,
        2,
        7.0,
    )
    finite_sum_count_kernel(values)
    positive_finite_sum_kernel(values)
    int_range_count_kernel(integers, 1, 3)
    int_less_than_count_kernel(integers, 2)
    contiguous_blocks_kernel(mask)
    finite_mean_kernel(values, 1)
    numeric_stat_kernel(values, 0)
    numeric_comparison_mask_kernel(values, 0.25, 1.0, 0)
    finite_mask_kernel(values)
    positive_finite_mask_kernel(values)
    positive_pair_mask_kernel(values, values)
    fee_bucket_counts_kernel(values)
    status_counts_kernel(integers)
    count_true_kernel(mask)
    encoded_category_counts_kernel(integers, 4)
    encoded_unique_count_kernel(integers)
    coverage_ratio_kernel(2, 3)
    aggregate_count_rows_kernel(np.ascontiguousarray(np.array([[1, 2], [3, 4]], dtype=np.int64)))
    yearly_event_aggregation_kernel(integers + 2020, values)
    nav = np.ascontiguousarray(np.array([1.0, 1.01, 1.02], dtype=np.float64))
    nav_metrics_kernel(nav, nav, nav, nav, 2)
    product_compare_analysis_kernel(nav, integers[:3], 2, 0.5, 0.1)
    candle_metrics_kernel(nav, nav, nav, nav)
    latest_share_metrics_kernel(nav, nav)
    stale_days_kernel(integers)
    fresh_date_mask_kernel(integers, 7)
    _rank_before(0, 1, values, integers, np.uint8(0))
    ranking_order_kernel(values, integers, np.uint8(0))
    _sort_before_with_missing(0, 1, values, integers, np.uint8(0))
    numeric_sort_order_kernel(values, integers, np.uint8(0))
    adjusted_nav_anomaly_mask_kernel(nav, nav, nav, 0.20, 0.10, 1.0)
    period_window_quality_kernel(integers, 0, 3, integers, integers[:0], 10, 0.8, 10)
    audit = validate_execution_audit(instrument_analytics_numba_execution_audit())
    if not audit["nopython"] or audit["python_fallback"] != 0:
        raise RuntimeError("产品研究分析 NJIT 内核未进入 nopython 模式")
    return audit


__all__ = [
    "NAT_DAY",
    "adjusted_nav_anomaly_mask_kernel",
    "aggregate_count_rows_kernel",
    "candle_metrics_kernel",
    "contiguous_blocks_kernel",
    "count_true_kernel",
    "coverage_ratio_kernel",
    "data_quality_masks_kernel",
    "encoded_category_counts_kernel",
    "encoded_unique_count_kernel",
    "fee_bucket_counts_kernel",
    "finite_mask_kernel",
    "finite_mean_kernel",
    "finite_sum_count_kernel",
    "fresh_date_mask_kernel",
    "instrument_analytics_numba_execution_audit",
    "int_less_than_count_kernel",
    "int_range_count_kernel",
    "latest_share_metrics_kernel",
    "nav_metrics_kernel",
    "numeric_comparison_mask_kernel",
    "numeric_sort_order_kernel",
    "numeric_stat_kernel",
    "period_window_quality_kernel",
    "positive_finite_mask_kernel",
    "positive_finite_sum_kernel",
    "positive_pair_mask_kernel",
    "product_compare_analysis_kernel",
    "ranking_order_kernel",
    "simple_log_returns_kernel",
    "stale_days_kernel",
    "status_counts_kernel",
    "warm_instrument_analytics_numba_kernels",
    "yearly_event_aggregation_kernel",
]
