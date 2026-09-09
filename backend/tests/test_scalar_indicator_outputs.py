from __future__ import annotations

import copy
import numpy as np
import pytest

from custom_indicators.service import CustomIndicatorService
from custom_indicators.errors import ValidationError
from test_custom_indicator_service import _write_market_data


def bundle_draft():
    return {
        "name": "收益摘要", "description": "共享数据与公式", "expression": "",
        "result_kind": "scalar_bundle", "output_contract": "scalar_bundle",
        "dsl_version": "2.3.0", "operator_registry_version": "2.3.0",
        "scalar_outputs": [
            {"id": "mean", "label": "平均收益", "expression": "mean(returns)", "display_format": "percent", "direction": "higher_better", "precision": 2},
            {"id": "twice", "label": "两倍平均收益", "expression": "mean(returns) * 2", "direction": "neutral", "precision": 4},
            {"id": "bad", "label": "除零示例", "expression": "mean(returns) / 0", "direction": "neutral"},
        ],
    }


@pytest.fixture
def service(tmp_path):
    _write_market_data(tmp_path)
    return CustomIndicatorService(tmp_path, tmp_path)


def test_bundle_crud_preview_reference_and_version(service):
    draft = bundle_draft()
    validation = service.validate(draft)
    assert validation["valid"], validation
    assert set(validation["dag"]["roots"]) == {"mean", "twice", "bad"}
    created = service.create_indicator(draft)
    assert created["result_kind"] == "scalar_bundle"
    target = [{"kind": "etf", "product_id": "510050.SH"}]
    result = service.evaluate(indicator_ids=[created["id"]], inline_definition=None, targets=target, period="ALL")["results"][0]
    mean, twice, bad = result["outputs"]
    assert mean["value"] is not None
    assert twice["value"] == pytest.approx(2 * mean["value"])
    assert bad["value"] is None
    assert result["status"] == "warning"
    assert mean["presentation"]["value_scale"] == 100
    assert twice["presentation"]["direction"] == "neutral"
    selected = service.evaluate(indicator_ids=[], inline_definition=None, indicator_refs=[{"indicator_id": created["id"], "output_id": "mean"}], targets=target, period="ALL")
    assert len(selected["results"]) == 1
    assert selected["results"][0]["value"] == mean["value"]
    changed = copy.deepcopy(draft)
    changed["scalar_outputs"].reverse()
    changed["scalar_outputs"][0]["label"] = "新的显示名"
    service.update_indicator(created["id"], 1, changed)
    old = service.indicators.get(created["id"], 1)
    assert old["scalar_outputs"][0]["id"] == "mean"
    assert old["scalar_outputs"][-1]["label"] == "除零示例"


def test_bundle_inline_token_and_all_outputs_validated(service):
    draft = bundle_draft()
    validation = service.validate(draft)
    assert validation["valid"], validation
    target = [{"kind": "etf", "product_id": "510050.SH"}]
    result = service.evaluate(indicator_ids=[], inline_definition=draft, compile_token=validation["compile_token"], targets=target, period="ALL")
    assert len(result["results"][0]["outputs"]) == 3
    draft["scalar_outputs"][1]["expression"] = "mean(returns) * 3"
    with pytest.raises(ValidationError, match="重新校验"):
        service.evaluate(indicator_ids=[], inline_definition=draft, compile_token=validation["compile_token"], targets=target, period="ALL")
    draft["scalar_outputs"][1]["expression"] = "returns"
    invalid = service.validate(draft)
    assert not invalid["valid"]
    assert invalid["diagnostics"][0]["output_id"] == "twice"


def test_missing_field_does_not_block_nav_output(service):
    draft = bundle_draft()
    draft["scalar_outputs"][1]["expression"] = "mean(volume)"
    created = service.create_indicator(draft)
    result = service.evaluate(indicator_ids=[created["id"]], inline_definition=None, targets=[{"kind": "fund", "product_id": "000001.OF"}], period="ALL")["results"][0]
    assert result["outputs"][0]["value"] is not None
    assert result["outputs"][1]["value"] is None
    assert result["outputs"][1]["status"] == "unavailable"


def test_output_id_retirement_and_explicit_selection(service):
    draft = bundle_draft()
    created = service.create_indicator(draft)
    short = copy.deepcopy(draft)
    short["scalar_outputs"] = short["scalar_outputs"][:1]
    updated = service.update_indicator(created["id"], 1, short)
    assert updated["result_kind"] == "scalar_bundle"
    assert "twice" in updated["retired_output_ids"]
    with pytest.raises(ValidationError):
        service.update_indicator(created["id"], 2, draft)
    with pytest.raises(ValidationError):
        service.scalar_service.evaluate_references([{"indicator_id": created["id"]}], [], "ALL", None)
