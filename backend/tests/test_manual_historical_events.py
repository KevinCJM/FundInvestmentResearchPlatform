from __future__ import annotations

import copy
from datetime import date, timedelta

import numpy as np
import pytest

from custom_indicators.errors import ValidationError
from historical_regimes.manual_event_numba import MANUAL_EVENT_KERNELS, manual_event_state_kernel, manual_event_summary_kernel
from historical_regimes.v2_contracts import inspect_definition_v2, parse_definition_v2
from historical_regimes.v2_numba import regime_graph_numba_status
from historical_regimes.v2_registry import NODE_REGISTRY
from historical_regimes.v2_service import RegimeGraphV2Service
from historical_regimes.v2_templates import get_template_v2, instantiate_template_v2


def _dates(count: int = 10) -> np.ndarray:
    start = np.datetime64("2020-01-01", "ns").astype(np.int64)
    day = np.int64(86_400_000_000_000)
    return np.ascontiguousarray(start + np.arange(count, dtype=np.int64) * day)


def _events() -> list[dict[str, str]]:
    return [
        {"id": "event_a", "label": "事件 A", "start_date": "2020-01-03", "end_date": "2020-01-06", "color": "#7c3aed", "description": "第一段"},
        {"id": "event_b", "label": "事件 B", "start_date": "2020-01-05", "end_date": "2020-01-08", "color": "#dc2626", "description": "与 A 重叠"},
        {"id": "outside", "label": "样本外事件", "start_date": "2019-01-01", "end_date": "2019-02-01", "color": "#2563eb", "description": "保留但无覆盖"},
    ]


def _definition() -> dict:
    raw = instantiate_template_v2("manual-historical-events-v1")
    assert raw is not None
    start = date(2020, 1, 1)
    raw["graph"]["nodes"][0] = {
        "id": "market",
        "type": "source.inline",
        "parameters": {
            "frequency": "daily",
            "rows": [
                {
                    "observation_date": (start + timedelta(days=index)).isoformat(),
                    "available_at": (start + timedelta(days=index)).isoformat(),
                    "value": 100.0 + index,
                }
                for index in range(10)
            ],
        },
    }
    raw["graph"]["nodes"][1]["parameters"]["events"] = _events()
    return raw


def test_manual_event_kernels_preserve_overlap_without_dense_event_matrix() -> None:
    dates = _dates()
    starts = np.ascontiguousarray(np.array([dates[2], dates[4], dates[0] - 400 * 86_400_000_000_000], dtype=np.int64))
    ends = np.ascontiguousarray(np.array([dates[5], dates[7], dates[0] - 300 * 86_400_000_000_000], dtype=np.int64))
    states, counts = manual_event_state_kernel(dates, starts, ends)
    np.testing.assert_array_equal(counts, [0, 0, 1, 1, 2, 2, 1, 1, 0, 0])
    np.testing.assert_array_equal(states, [1, 1, 0, 0, 0, 0, 0, 0, 1, 1])
    observations, first, last, summary = manual_event_summary_kernel(dates, counts, starts, ends)
    np.testing.assert_array_equal(observations, [4, 4, 0])
    np.testing.assert_array_equal(first, [2, 4, -1])
    np.testing.assert_array_equal(last, [5, 7, -1])
    np.testing.assert_array_equal(summary, [6, 2, 2, 3])
    assert all(dispatcher.nopython_signatures and not dispatcher._can_compile for dispatcher in MANUAL_EVENT_KERNELS.values())


def test_manual_event_definition_allows_overlap_and_rejects_bad_event_contracts() -> None:
    raw = _definition()
    assert inspect_definition_v2(parse_definition_v2(raw))["valid"]
    template = get_template_v2("manual-historical-events-v1")
    assert template is not None and template["default_mode"] == "retrospective"
    assert template["supported_modes"] == ["retrospective"]
    schema = NODE_REGISTRY["annotation.manual_events"]
    assert schema["supports_realtime"] is False and schema["causal"] is False and schema["repaints"] is False

    cases = [
        (lambda item: item.update(start_date="2020-02-01", end_date="2020-01-01"), "MANUAL_EVENT_DATE_ORDER"),
        (lambda item: item.update(color="purple"), "INVALID_MANUAL_EVENT_COLOR"),
        (lambda item: item.update(start_date="2020/01/01"), "INVALID_MANUAL_EVENT_DATE"),
    ]
    for mutate, expected in cases:
        candidate = copy.deepcopy(raw)
        mutate(candidate["graph"]["nodes"][1]["parameters"]["events"][0])
        errors = inspect_definition_v2(parse_definition_v2(candidate))["errors"]
        assert any(item["code"] == expected for item in errors)

    duplicate = copy.deepcopy(raw)
    duplicate["graph"]["nodes"][1]["parameters"]["events"][1]["id"] = "event_a"
    assert any(item["code"] == "DUPLICATE_MANUAL_EVENT_ID" for item in inspect_definition_v2(parse_definition_v2(duplicate))["errors"])


def test_manual_event_execution_returns_frozen_multilabel_events_and_overlap_summary(tmp_path) -> None:
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    parsed = parse_definition_v2(_definition())
    execution = service._execute_graph(None, parsed, "retrospective", None)
    result = execution["result"]
    assert result["result_kind"] == "manual_events"
    assert result["manual_event_summary"] == {
        "event_count": 3,
        "covered_observations": 6,
        "overlap_observations": 2,
        "max_concurrent_events": 2,
    }
    events = result["manual_events"]
    assert [(item["id"], item["covered_observations"]) for item in events] == [("event_a", 4), ("event_b", 4), ("outside", 0)]
    assert events[0]["first_observation_date"] == "2020-01-03" and events[0]["last_observation_date"] == "2020-01-06"
    assert events[2]["first_observation_index"] is None and events[2]["last_observation_index"] is None
    assert result["state_counts"] == {"event": 6, "normal": 4}
    assert [row["features"]["event_count"] for row in execution["series"]] == [0, 0, 1, 1, 2, 2, 1, 1, 0, 0]
    assert all(not row["executable"] and row["effective_date"] is None for row in execution["series"])
    assert all(row["recognized_at"] == "2020-01-10" for row in execution["series"])
    assert "同时落入 2 个" in execution["series"][4]["reasons"][0]
    audit = regime_graph_numba_status()
    assert audit["complete"] is True
    kernel_ids = {item["kernel_id"] for item in audit["kernels"]}
    assert {"manual_event_state", "manual_event_summary"} <= kernel_ids


def test_manual_event_definition_revisions_preserve_overlapping_event_lists(tmp_path) -> None:
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    created = service.create_definition(_definition())
    assert created["revision"] == 1
    assert [event["id"] for event in service.get_definition(created["id"], 1)["graph"]["nodes"][1]["parameters"]["events"]] == ["event_a", "event_b", "outside"]

    changed = copy.deepcopy(created)
    changed["graph"]["nodes"][1]["parameters"]["events"][1]["label"] = "事件 B · 修订"
    updated = service.update_definition(created["id"], 1, changed)
    assert updated["revision"] == 2
    assert service.get_definition(created["id"], 1)["graph"]["nodes"][1]["parameters"]["events"][1]["label"] == "事件 B"
    assert service.get_definition(created["id"], 2)["graph"]["nodes"][1]["parameters"]["events"][1]["label"] == "事件 B · 修订"


def test_manual_events_realtime_is_rejected_before_source_io(tmp_path, monkeypatch) -> None:
    service = RegimeGraphV2Service(tmp_path, tmp_path)
    parsed = parse_definition_v2(_definition())
    monkeypatch.setattr(service, "_resolve_sources", lambda *args, **kwargs: pytest.fail("unexpected source I/O"))
    with pytest.raises(ValidationError) as exc_info:
        service._execute_graph(None, parsed, "realtime", None)
    assert exc_info.value.code == "NON_CAUSAL_REALTIME_GRAPH"
