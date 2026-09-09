"""Actual fixed-NJIT scalar/date/channel snapshot paths over disposable data."""
from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest

from custom_indicators.errors import ValidationError
from custom_indicators.service import CustomIndicatorService
from custom_indicators.snapshot_execution import configured_snapshot_values, last_finite_snapshot_value
from custom_indicators.rolling_series import normalize_rolling_source
from test_custom_indicator_time_series import _write_market_data

TARGET = {"kind": "etf", "product_id": "510300.SH"}


def test_last_finite_reducer_preserves_zero_window_and_frozen_signature():
    signatures = tuple(last_finite_snapshot_value.signatures)
    values = np.array([9., np.nan, 0., np.inf, np.nan])
    assert last_finite_snapshot_value(values, 1, 5) == (0., 2)
    assert math.isnan(last_finite_snapshot_value(values, 3, 5)[0])
    assert last_finite_snapshot_value(values, 3, 5)[1] == -1
    assert last_finite_snapshot_value(np.empty(0), 0, 0)[1] == -1
    with pytest.raises(ValueError):
        last_finite_snapshot_value(values, 0, 6)
    assert tuple(last_finite_snapshot_value.signatures) == signatures
    assert last_finite_snapshot_value.nopython_signatures


def test_mixed_scalar_date_and_two_channels_use_separate_paths(tmp_path, monkeypatch):
    _write_market_data(tmp_path)
    service = CustomIndicatorService(tmp_path, tmp_path)
    try:
        series_id = "builtin-kdj-series"
        series = service.get_indicator(series_id)
        channel_ids = [item["id"] for item in series["series_outputs"]][:2]
        scalar_id = "builtin-total-return-v2"
        date_id = "builtin-maximum-drawdown-start-date"
        config = service.update_snapshot_config(service.get_snapshot_config()["revision"], [
            {"indicator_id": scalar_id, "indicator_revision": 1, "period": "ALL", "field": "ret"},
            {"indicator_id": date_id, "indicator_revision": 1, "period": "ALL", "field": "event_date"},
            *[{"indicator_id": series_id, "indicator_revision": 1, "period": "ALL", "field": f"channel_{index}",
               "channel_id": channel_id, "reducer": "last_finite"} for index, channel_id in enumerate(channel_ids)],
        ])
        expected = service.evaluate_series(indicator_instances=[{"indicator_id": series_id}], target=TARGET, period="ALL")["results"][0]
        expected_values = {item["id"]: next(value for value in reversed(item["values"]) if value is not None) for item in expected["channels"]}
        calls = []
        original = service.series_service.evaluate
        def recording(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)
        monkeypatch.setattr(service.series_service, "evaluate", recording)
        output = pd.DataFrame({"instrument_type": ["etf"], "ts_code": ["510300.SH"]})
        snapshot, metadata = configured_snapshot_values(output, service)
        assert snapshot.loc[0, "ret"] > 0
        assert isinstance(snapshot.loc[0, "event_date"], str)
        for index, channel_id in enumerate(channel_ids):
            assert snapshot.loc[0, f"channel_{index}"] == pytest.approx(expected_values[channel_id])
            assert snapshot.loc[0, f"channel_{index}__value_date"] == expected["dates"][-1]
        assert len(calls) == 1 and calls[0]["_snapshot_only"]
        assert metadata["config_revision"] == config["revision"]
        assert metadata["configured_count"] == 4
        assert sum(metadata["status_counts"].values()) == 4
        assert metadata["items"][2]["channel_id"] == channel_ids[0]
        assert metadata["items"][3]["channel_id"] == channel_ids[1]
        assert all("output_id" not in item for item in metadata["items"])
    finally:
        service.close_compute_engine()


def test_snapshot_last_finite_precedes_display_tail_and_is_not_warmup(tmp_path):
    frame = _write_market_data(tmp_path, count=5020)
    nav = pd.read_parquet(tmp_path / "etf_daily_df.parquet")
    nav["adj_nav"] = 1.0
    nav.loc[:9, "adj_nav"] = 2.0
    nav.to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    service = CustomIndicatorService(tmp_path, tmp_path)
    try:
        draft = copy.deepcopy(service.get_indicator("builtin-close-moving-average-series"))
        draft.update(name="Only early finite values", axis_anchor="adjusted_nav", history_policy="lookback",
                     minimum_observations=1, lookback_observations=1, fixed_parameters=[], parameter_schema=[])
        draft["series_outputs"] = [{"id": "value", "label": "Value", "expression": "divide(adjusted_nav, adjusted_nav - 1)",
                                    "unit": "", "precision": 2, "display_format": "number", "output_measure": "auto"}]
        draft["expression"] = draft["series_outputs"][0]["expression"]
        saved = service.create_indicator(draft)
        instance = [{"indicator_id": saved["id"], "indicator_revision": saved["revision"]}]
        chart = service.series_service.evaluate(indicator_instances=instance, target=TARGET, period="ALL")
        assert chart["results"][0]["status"] == "unavailable"
        full = service.series_service.evaluate(indicator_instances=instance, target=TARGET, period="ALL", _snapshot_only=True)
        reduced = full["results"][0]["snapshot_channels"][0]
        assert reduced["value"] == 2.0
        assert reduced["value_date"] == frame.iloc[9]["date"].strftime("%Y-%m-%d")
        assert full["results"][0]["channels"] == []
        assert full["results"][0]["dates"] == []
        # The last finite value from years ago must not leak into a 1W window.
        recent = service.series_service.evaluate(indicator_instances=instance, target=TARGET, period="1W", _snapshot_only=True)
        assert recent["results"][0]["snapshot_channels"][0]["value"] is None
        assert full["execution"]["request_time_compilation"] == 0
    finally:
        service.close_compute_engine()


def test_removed_child_result_is_rejected_not_silently_rebound():
    with pytest.raises(ValidationError) as failure:
        normalize_rolling_source({"output_id": "alpha"})
    assert failure.value.code == "REMOVED_SCALAR_OUTPUT_REFERENCE"
