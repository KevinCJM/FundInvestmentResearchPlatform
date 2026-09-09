"""Common formula authoring exposes named outputs without a bespoke wizard API."""
from __future__ import annotations

import copy

import pytest

from custom_indicators.errors import ValidationError
from custom_indicators.typed_service import compose_expression, infer_expression, typed_product_meta
from custom_indicators.formula_source import canonical_formula_source
from custom_indicators.drawdown_indicator import drawdown_analysis_builtin
from test_drawdown_indicator_workflow import service  # noqa: F401


def compose(input_source="variable", value="adjusted_nav"):
    return compose_expression({
        "operator_id": "drawdown_analysis", "context": "single_product",
        "arguments": [{"parameter": "values", "source": input_source, "value": value}],
    })


def test_common_composer_returns_authoritative_named_projections():
    result = compose()
    assert result["shape"] == "record"
    ports = result["output_ports"]
    assert [port["id"] for port in ports] == ["max_drawdown", "decline_periods", "recovery_periods", "longest_underwater_periods"]
    assert ports[0]["display_format"] == "percent"
    assert ports[1]["direction"] == "neutral"
    for port in ports:
        assert port["expression"] == f"{result['expression']}.{port['id']}"
        assert canonical_formula_source(port["editable_latex"]) == port["expression"]
    assert len([node for node in result["dag"]["nodes"] if (node.get("operator") or {}).get("id") == "drawdown_analysis"]) == 1
    assert result["python_fallback"] == 0


def test_nested_input_and_plain_scalar_use_the_same_inference_contract():
    result = compose("expression", "adjusted_nav * 2")
    assert len(result["output_ports"]) == 4
    assert all("adjusted_nav * 2" in port["expression"] for port in result["output_ports"])
    scalar = infer_expression(result["output_ports"][0]["expression"], "single_product", scalar_required=True)
    assert scalar["shape"] == "scalar" and "output_ports" not in scalar
    with pytest.raises(ValidationError):
        compose(value="returns")


def test_shared_composer_defaults_and_input_constraints_are_metadata():
    operator = next(item for item in typed_product_meta()["operators"] if item["name"] == "drawdown_analysis")
    assert operator["parameters"][0]["default"] == "adjusted_nav"
    assert set(operator["parameters"][0]["excluded_semantic_dimensions"]) == {"return_decimal", "rate_decimal"}


def test_composed_outputs_save_and_execute_as_one_shared_calculation(service):
    result = compose()
    draft = drawdown_analysis_builtin()
    draft["name"] = "从常规向导创建回撤"
    draft["scalar_outputs"] = [{
        key: port[key] for key in ("id", "label", "expression", "description", "unit", "display_format", "precision", "direction")
    } for port in result["output_ports"]]
    saved = service.create_indicator(draft)
    args = dict(indicator_ids=[saved["id"]], inline_definition=None, targets=[{"kind": "etf", "product_id": "510050.SH"}], period="ALL")
    first = service.evaluate(**args)
    changed = copy.deepcopy(saved)
    changed["scalar_outputs"] = changed["scalar_outputs"][:2]
    for output, port in zip(changed["scalar_outputs"], compose("expression", "adjusted_nav * 2")["output_ports"]):
        output["expression"] = port["expression"]
    updated = service.update_indicator(saved["id"], 1, changed)
    assert [item["id"] for item in updated["scalar_outputs"]] == ["max_drawdown", "decline_periods"]
    assert service.scalar_service._require_warmed(updated).groups[0].compiled.metadata()["multi_output_call_sites"] == 1
    second = service.evaluate(**args)
    assert second["results"][0]["outputs"][0]["value"] == pytest.approx(first["results"][0]["outputs"][0]["value"])
    assert len(service.indicators.get(saved["id"], 1)["scalar_outputs"]) == 4
