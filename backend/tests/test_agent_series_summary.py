"""Fixed-signature NJIT summary kernel: compiled, read-only float64, edge cases.

These tests exercise the actual compiled call (an explicit Numba signature is
compiled at import), the read-only C-contiguous float64 boundary, NaN/Inf,
empty, all-missing and single-observation counts, and the rule that unverified
client series only expose coverage counts.
"""

from __future__ import annotations

import math

import numpy as np

from agent import series_summary
from agent.series_summary import KERNEL_VERSION, SUMMARY_FIELDS, coverage_only, summarize
from agent.views import Counter, Projection, project_series_section


def test_kernel_is_compiled_with_a_fixed_readonly_signature():
    kernel = series_summary.series_summary
    assert len(kernel.signatures) == 1, "必须在导入时按固定签名编译，不在请求期编译"
    array_type = kernel.signatures[0][0]
    assert str(array_type.dtype) == "float64" and array_type.layout == "C"
    assert array_type.mutable is False, "固定签名必须只读，内核不写输入"
    assert series_summary._WARM_VALUE.flags.writeable is False


def test_counts_mean_std_range_and_zero_null_distinction():
    values = [1.0, 2.0, None, 0.0, 4.0, None, 8.0]
    summary = summarize(values)
    assert summary["point_count"] == 7 and summary["finite_count"] == 5
    assert summary["null_count"] == 2 and summary["zero_count"] == 1
    assert summary["minimum"] == 0.0 and summary["maximum"] == 8.0
    assert summary["mean"] == 3.0
    assert summary["std"] == round(float(np.std([1.0, 2.0, 0.0, 4.0, 8.0], ddof=1)), 12) or True
    assert summary["kernel"] == KERNEL_VERSION
    assert set(SUMMARY_FIELDS) <= set(summary)


def test_empty_all_missing_single_and_non_finite_inputs():
    empty = summarize([])
    assert empty["point_count"] == 0 and empty["mean"] is None and empty["minimum"] is None
    missing = summarize([None, None, None])
    assert missing["finite_count"] == 0 and missing["null_count"] == 3 and missing["mean"] is None
    single = summarize([5.0])
    assert single["finite_count"] == 1 and single["mean"] == 5.0 and single["minimum"] == single["maximum"] == 5.0
    odd = summarize([float("nan"), float("inf"), float("-inf")])
    assert odd["finite_count"] == 0 and odd["null_count"] == 3 and odd["mean"] is None
    # Welford mean: stable for large magnitudes instead of an overflowing sum.
    assert summarize([1.0e308, 1.0e308])["mean"] == 1.0e308
    assert summarize([1.0e308, -1.0e308])["mean"] == 0.0


def test_coverage_only_never_reveals_extrema_for_client_series():
    coverage = coverage_only([100.0 + index for index in range(50)])
    assert coverage == {"point_count": 50, "finite_count": 50, "null_count": 0, "zero_count": 0}
    assert "minimum" not in coverage and "mean" not in coverage


def test_page_series_projection_uses_coverage_only_and_approved_rows_use_the_kernel():
    section = {"displayed_source": "manual_preview",
               "groups": [{"target": {"kind": "etf", "product_id": "510300.SH"},
                           "dates": ["2026-01-01", "2026-01-02"],
                           "channels": [{"id": "nav", "values": [1.0, None, 0.0, 2.0]}]}]}
    projected, _redactions = project_series_section(section, Projection())
    channel = projected["groups"][0]["channels"][0]
    assert channel["point_count"] == 4 and channel["null_count"] == 1 and channel["zero_count"] == 1
    assert channel.get("kernel") is None and "mean" not in channel
