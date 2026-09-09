"""Snapshot and rolling consumers must identify a named result explicitly."""
from copy import deepcopy

import pandas as pd
import pytest

from custom_indicators.errors import ValidationError
from custom_indicators.service import CustomIndicatorService
from test_scalar_indicator_outputs import bundle_draft, service


def test_snapshot_columns_and_presentations_are_output_specific(service):
    definition = service.create_indicator(bundle_draft())
    config = service.get_snapshot_config()
    items = [{"indicator_id": definition["id"], "indicator_revision": 1, "output_id": key, "period": "ALL"} for key in ("mean", "twice")]
    updated = service.update_snapshot_config(config["revision"], items)
    assert len({item["field"] for item in updated["items"]}) == 2
    assert [item["presentation"]["output_id"] for item in updated["items"]] == ["mean", "twice"]
    assert [item["presentation"]["direction"] for item in updated["items"]] == ["higher_better", "neutral"]
    from services.instrument_analytics import _configured_snapshot_values
    frame, metadata = _configured_snapshot_values(
        pd.DataFrame([{"instrument_type": "etf", "ts_code": "510050.SH"}]),
        market_data_dir=service.market_data_dir, workspace_data_dir=service.workspace_data_dir,
    )
    fields = {item["output_id"]: item["field"] for item in metadata["items"]}
    assert frame.loc[0, fields["mean"]] > 0
    assert frame.loc[0, fields["twice"]] == pytest.approx(2 * frame.loc[0, fields["mean"]])
    assert frame.loc[0, fields["mean"] + "__status"] == "ok"
    assert metadata["configured_count"] == 2


def test_snapshot_requires_output_and_rejects_time_series_reducer(service):
    definition = service.create_indicator(bundle_draft())
    config = service.get_snapshot_config()
    item = {"indicator_id": definition["id"], "indicator_revision": 1, "period": "ALL"}
    with pytest.raises(ValidationError) as missing:
        service.update_snapshot_config(config["revision"], [item])
    assert missing.value.code == "OUTPUT_REQUIRED"
    with pytest.raises(ValidationError) as reducer:
        service.update_snapshot_config(config["revision"], [{**item, "output_id": "mean", "reducer": "last_finite"}])
    assert reducer.value.code == "SNAPSHOT_SCALAR_CHANNEL_NOT_ALLOWED"


def test_rolling_source_locks_output_and_revision(service):
    definition = service.create_indicator(bundle_draft())
    derived = service.derive_rolling_series(indicator_id=definition["id"], indicator_revision=1, output_id="twice", window_observations=5)
    source = derived["definition"]["rolling_source"]
    assert source["output_id"] == "twice"
    assert source["indicator_revision"] == 1
    assert "rolling_mean" in derived["definition"]["series_outputs"][0]["expression"]
    assert derived["validation"]["valid"]
    # A named output with neutral presentation is still a valid time-series source.
    saved = service.create_indicator(derived["definition"])
    assert saved["rolling_source"]["output_id"] == "twice"
    altered = deepcopy(derived["definition"])
    altered["rolling_source"]["output_id"] = "mean"
    assert not service.validate(altered)["valid"]
    changed = bundle_draft()
    changed["scalar_outputs"][1]["expression"] = "mean(returns) * 9"
    service.update_indicator(definition["id"], 1, changed)
    assert service.validate(derived["definition"])["valid"]


def test_new_facade_issues_its_own_inline_token_without_recompiling(service, monkeypatch):
    draft = bundle_draft()
    first = service.validate(draft)
    other = CustomIndicatorService(service.workspace_data_dir, service.market_data_dir)
    import custom_indicators.scalar_bundle_service as module
    monkeypatch.setattr(module, "compile_scalar_bundle", lambda *_args, **_kwargs: pytest.fail("duplicate compilation"))
    second = other.validate(draft)
    assert second["valid"]
    assert first["compile_token"] != second["compile_token"]
    result = other.evaluate(indicator_ids=[], inline_definition=draft, compile_token=second["compile_token"],
                            targets=[{"kind": "etf", "product_id": "510050.SH"}], period="ALL")
    assert result["results"][0]["outputs"][0]["value"] is not None
