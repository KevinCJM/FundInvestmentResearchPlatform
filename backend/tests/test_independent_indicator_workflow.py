"""Public independent metric workflow, same-event dates and actual shared execution."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from custom_indicators.drawdown_indicator import independent_drawdown_indicators
from custom_indicators.errors import ValidationError
from custom_indicators.service import CustomIndicatorService
from test_custom_indicator_service import _write_market_data


@pytest.fixture
def service(tmp_path):
    _write_market_data(tmp_path, 9)
    pd.DataFrame({"ts_code": "510050.SH", "date": pd.bdate_range("2026-01-02", periods=9),
                  "adj_nav": [1, .8, 1, 1, .9, .8, .9, .8, 1]}).to_parquet(tmp_path / "etf_daily_df.parquet", index=False)
    owner = CustomIndicatorService(tmp_path, tmp_path)
    yield owner
    owner.close_compute_engine()


def all_ids():
    return [item["id"] for item in independent_drawdown_indicators()]


def evaluate(service, ids=None):
    return service.evaluate(indicator_ids=ids or all_ids(), inline_definition=None,
                            targets=[{"kind": "etf", "product_id": "510050.SH"}], period="ALL", prefer_snapshot=False)


def test_prepare_and_evaluate_same_interval_with_real_calendar_dates(service):
    prepared = service.prepare_evaluation(indicator_ids=all_ids())
    assert prepared["prepared"]
    response = evaluate(service)
    values = [item["value"] for item in response["results"]]
    assert values == [pytest.approx(.2), "2026-01-07", "2026-01-13", 6., "2026-01-14", 1., 7.]
    assert all(item["result_kind"] == "scalar" for item in response["results"])
    assert all(not item.get("outputs") for item in response["results"])
    sharing = response["execution"]["shared_plans"][0]
    assert sharing["operator_call_sites"]["drawdown_series"] == 1
    assert sharing["operator_call_sites"]["last_drawdown_interval"] == 1
    assert response["execution"]["request_time_compilation"] == 0


def test_subset_reuses_prepared_superset_and_cache_does_not_execute(service, monkeypatch):
    service.prepare_evaluation(indicator_ids=all_ids())
    ids = [all_ids()[0], all_ids()[2]]
    first = evaluate(service, ids)
    assert len(first["results"]) == 2
    def unexpected(**kwargs):
        raise AssertionError("A result cache hit must not execute numerical batches")
    monkeypatch.setattr(service, "_run_fused_typed_groups", unexpected)
    second = evaluate(service, ids)
    assert second["results"] == first["results"]
    assert second["execution"]["executed_batches"] == 0
    assert second["cache"] == {"hits": 2, "misses": 0}


def test_date_custom_authoring_and_math_notation(service):
    source = independent_drawdown_indicators()[1]
    saved = service.create_indicator({**source, "name": "自定义最大回撤起点"})
    assert saved["output_measure"] == "date"
    assert saved["display_format"] == "date" and saved["direction"] == "neutral"
    assert "last_drawdown_interval" not in saved["display_latex"]
    assert "date_at" not in saved["display_latex"]
    assert r"\mathcal{I}" in saved["display_latex"]
    assert r"\mathbf{d}" in saved["display_latex"]
    source.update(name="不得给数值伪造日期", expression="mean(returns)")
    with pytest.raises(ValidationError, match="日期"):
        service.create_indicator(source)


def test_retired_multi_scalar_definition_is_not_accepted(service):
    with pytest.raises(ValidationError) as exc:
        service.create_indicator({"name": "旧组合", "result_kind": "scalar_bundle", "scalar_outputs": []})
    assert exc.value.code == "INVALID_RESULT_KIND"
    assert all(item["result_kind"] != "scalar_bundle" for item in service.list_indicators()["items"])


@pytest.mark.parametrize("field,value", [("scalar_outputs", []), ("output_id", None)])
@pytest.mark.parametrize("model_name", ["IndicatorDraft", "EvaluateRequest"])
def test_retired_request_keys_are_not_silently_ignored(field, value, model_name):
    from pydantic import ValidationError as RequestError
    from services import custom_indicator_routes as routes
    payload = {"name": "Independent", "expression": "1", "indicator_ids": ["x"],
               "targets": [{"kind": "etf", "product_id": "510050.SH"}], "period": "ALL", field: value}
    with pytest.raises(RequestError, match="独立指标"):
        getattr(routes, model_name).model_validate(payload)


def test_date_cannot_be_added_to_scoring_even_with_explicit_direction(service):
    with pytest.raises(ValidationError) as exc:
        service.create_plan({"name": "错误评分", "product_kind": "etf", "targets": [{"kind": "etf", "product_id": "510050.SH"}],
                             "indicators": [{"indicator_id": all_ids()[1], "period": "ALL", "weight": 100, "direction": "higher_better"}]})
    assert exc.value.code == "DATE_NOT_SCORABLE"


def test_no_recovery_does_not_block_depth_or_trough(service):
    path = service.market_data_dir / "etf_daily_df.parquet"
    data = pd.read_parquet(path)
    data.loc[data.index[-1], "adj_nav"] = .9
    data.to_parquet(path, index=False)
    service.prepare_evaluation(indicator_ids=all_ids())
    response = evaluate(service)
    assert response["results"][0]["value"] == pytest.approx(.2)
    assert response["results"][2]["value"] == "2026-01-13"
    assert response["results"][4]["value"] is None
    assert any(item["code"] == "DRAWDOWN_NOT_RECOVERED" for item in response["results"][4]["warnings"])
