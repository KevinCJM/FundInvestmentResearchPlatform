from __future__ import annotations

import numpy as np
import pytest

from historical_regimes import result_overview
from historical_regimes.result_overview import (
    build_result_overview,
    overview_segments_kernel,
    warm_result_overview_kernel,
)


@pytest.fixture(autouse=True)
def warm_overview():
    warm_result_overview_kernel()


def _overview(series, **kwargs):
    return build_result_overview(
        run_kind="preview",
        run_id="preview-test",
        definition={
            "states": [
                {"id": "bull", "label": "牛市", "color": "#16a34a", "order": 1, "role": "positive"},
                {"id": "bear", "label": "熊市", "color": "#dc2626", "order": 2, "role": "negative"},
            ],
            "graph": {"outputs": {"state": {"node_id": "classifier", "port": "state"}}},
        },
        series=series,
        definition_hash="definition-hash",
        graph_hash="graph-hash",
        mode="realtime",
        as_of=None,
        series_endpoint="/api/historical-regimes/preview-runs/preview-test/series",
        **kwargs,
    )


def _rows():
    return [
        {"observation_date": "2026-09-04", "state_id": "bull", "value": 100.0,
         "recognized_at": "2026-09-05", "effective_date": "2026-09-07"},
        {"observation_date": "2026-09-07", "state_id": "bear", "value": np.nan},
        {"observation_date": "2026-09-08", "state_id": None, "value": np.inf},
        {"observation_date": "2026-09-09", "state_id": np.nan, "value": None},
        {"observation_date": "2026-09-10", "state_id": np.inf, "value": 99.0},
        {"observation_date": "2026-09-11", "state_id": "bull", "value": 102.0},
    ]


def test_unknown_ranges_and_single_day_segments_preserve_locked_axis():
    overview = _overview(_rows())
    assert [(segment["start_index"], segment["end_index"]) for segment in overview["segments"]] == [
        (0, 0), (1, 1), (5, 5)
    ]
    assert overview["segments"][0]["start_date"] == overview["segments"][0]["end_date"] == "2026-09-04"
    assert overview["segments"][0]["confirmed_at"] == "2026-09-05"
    assert overview["segments"][0]["effective_start"] == "2026-09-07"
    unknown = overview["unknown_intervals"]
    assert len(unknown) == 1
    assert (unknown[0]["start_index"], unknown[0]["end_index"]) == (2, 4)
    assert unknown[0]["reason"] == "unknown"
    assert overview["summary"]["total"] == 6
    assert overview["summary"]["classified"] == 3
    assert overview["summary"]["unknown"] == 3
    assert overview["summary"]["state_counts"] == {"bull": 2, "bear": 1}
    assert overview["summary"]["switch_count"] == 1
    # A missing display price does not invent an unknown classification.
    assert overview["segments"][1]["state_id"] == "bear"
    assert overview["capabilities"]["effective"]["available"] is False


def test_empty_result_does_not_invent_dates_or_samples():
    overview = _overview([])
    assert overview["segments"] == overview["unknown_intervals"] == []
    assert overview["date_range"] == {"start": None, "end": None}
    assert overview["summary"]["total"] == 0
    assert overview["summary"]["classified"] == overview["summary"]["unknown"] == 0


def test_frozen_metadata_and_evaluations_are_detached():
    snapshots = {"source": {"snapshot_id": "frozen", "generation": 7}}
    result = {"evaluation_results": {"benchmark": {"name": "基准", "conditional_metrics": []}}}
    overview = _overview(_rows(), data_snapshots=snapshots, result=result)
    snapshots["source"]["generation"] = 8
    result["evaluation_results"]["benchmark"]["name"] = "changed"
    assert overview["data_snapshots"]["source"]["generation"] == 7
    assert overview["evaluation_results"]["benchmark"]["name"] == "基准"


@pytest.mark.parametrize("codes", [
    [], [-1], [0], [0, 1, 0], [-1, -1, 0, -1, 1, 1, -1], [5, 0, -20, 1],
])
def test_kernel_matches_controlled_reference_with_fixed_dtype(codes):
    values = np.asarray(codes, dtype=np.int64)
    before = tuple(overview_segments_kernel.signatures)
    segments, counts, switches, classified, unknown = overview_segments_kernel(values, np.int64(2))
    normalized = [code if 0 <= code < 2 else -1 for code in codes]
    reference = []
    for index, code in enumerate(normalized):
        if not reference or reference[-1][2] != code:
            reference.append([index, index, code, 1])
        else:
            reference[-1][1] = index
            reference[-1][3] += 1
    np.testing.assert_array_equal(segments, np.asarray(reference, dtype=np.int64).reshape(-1, 4))
    np.testing.assert_array_equal(counts, [normalized.count(0), normalized.count(1)])
    assert classified == sum(code >= 0 for code in normalized)
    assert unknown == normalized.count(-1)
    assert switches == sum(
        left >= 0 and right >= 0 and left != right
        for left, right in zip(normalized, normalized[1:])
    )
    repeat = overview_segments_kernel(values, np.int64(2))
    np.testing.assert_array_equal(repeat[0], segments)
    assert segments.dtype == counts.dtype == np.dtype("int64")
    assert tuple(overview_segments_kernel.signatures) == before
    assert overview_segments_kernel._can_compile is False
    assert len(overview_segments_kernel.nopython_signatures) == 1


def test_serializer_enters_njit_kernel_without_compiling(monkeypatch):
    calls = []
    dispatcher = overview_segments_kernel

    def tracked(codes, count):
        calls.append((codes.dtype, codes.flags.c_contiguous, type(count)))
        return dispatcher(codes, count)

    monkeypatch.setattr(result_overview, "overview_segments_kernel", tracked)
    _overview(_rows())
    assert calls == [(np.dtype("int64"), True, np.int64)]
    assert dispatcher._can_compile is False


def test_unwarmed_runtime_fails_closed(monkeypatch):
    monkeypatch.setattr(result_overview, "_WARMED", False)
    with pytest.raises(RuntimeError, match="not ready"):
        _overview(_rows())
