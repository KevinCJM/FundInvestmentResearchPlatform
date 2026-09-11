"""Regression coverage for source -> composer -> validation round trips."""
from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cal_indicators.typed_operators import get_typed_operator_registry
from custom_indicators.formula_source import canonical_formula_source, editable_formula_latex
from custom_indicators.service import CustomIndicatorService
from custom_indicators.typed_service import compose_expression
from services import custom_indicator_routes


SCREENSHOT_LATEX = (
    r"\left(\frac{\left(\operatorname{rolling_mean}\left(\mathbf{r},15.0\right)-r_f\right)}"
    r"{\operatorname{rolling_std}\left(\mathbf{r},15.0\right)}"
    r"\cdot \sqrt{p_{\mathrm{year}}}\right)"
)


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> CustomIndicatorService:
    root: Path = tmp_path_factory.mktemp("formula-roundtrip")
    return CustomIndicatorService(root, root)


@pytest.fixture
def client(service: CustomIndicatorService, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(custom_indicator_routes, "indicator_service", service)
    app = FastAPI()
    app.include_router(custom_indicator_routes.router)
    return TestClient(app)


def series_draft(service: CustomIndicatorService, expression: str) -> dict:
    draft = copy.deepcopy(service.get_indicator("builtin-rolling-5d-annualized-sharpe-series"))
    draft["rolling_source"]["detached"] = True
    draft["expression"] = expression
    draft["series_outputs"][0]["expression"] = expression
    return draft


def test_screenshot_latex_is_accepted_by_series_validation(client: TestClient, service: CustomIndicatorService) -> None:
    response = client.post("/api/custom-indicators/validate", json=series_draft(service, SCREENSHOT_LATEX))
    assert response.status_code == 200
    result = response.json()
    assert result["valid"], result["diagnostics"]
    output = result["output_inferences"]["value"]
    assert "risk_free_rate_per_observation" in output["python_expression"]
    assert "periods_per_year" in output["python_expression"]
    assert output["display_latex"]
    # The old screenshot omitted ddof: importing it must preserve the explicit
    # source semantics (population std), not silently turn it into sample std.
    assert r"\sigma_{t,15}" in output["display_latex"]


def test_composer_returns_canonical_executable_source_not_latex() -> None:
    result = compose_expression({
        "operator_id": "subtract",
        "context": "single_product",
        "arguments": [
            {"parameter": "lhs", "source": "expression", "value": "mean(returns)"},
            {"parameter": "rhs", "source": "variable", "value": "risk_free_rate_per_observation"},
        ],
    })
    assert result["expression"] == "mean(returns) - risk_free_rate_per_observation"
    ast.parse(result["expression"], mode="eval")
    assert "\\" not in result["expression"]
    assert result["display_latex"] != result["expression"]


def recompose_dag(
    client: TestClient,
    dag: dict,
    root_key: str,
    *,
    dsl_version: str,
    operator_registry_version: str,
) -> dict:
    nodes = {str(node["id"]): node for node in dag["nodes"]}
    incoming: dict[str, list] = {}
    for edge in dag["edges"]:
        incoming.setdefault(str(edge["target"]), []).append(edge)
    memo = {}

    def compose(node_id):
        key = str(node_id)
        if key in memo:
            return memo[key]
        node = nodes[key]
        arguments = []
        edges = sorted(incoming.get(key, []), key=lambda edge: edge.get("order", 0))
        operator = node.get("operator_id") or node.get("operator", {}).get("id") or node["label"]
        parameter_names = get_typed_operator_registry(operator_registry_version)[operator].argument_names(len(edges))
        for index, edge in enumerate(edges):
            child = nodes[str(edge["source"])]
            if child["kind"] == "variable":
                source, value = "variable", child["label"]
            elif child["kind"] == "constant":
                source, value = "constant", float(child.get("formula_fragment") or child["label"])
            else:
                source, value = "expression", compose(child["id"])["expression"]
            arguments.append({"parameter": edge.get("parameter") or parameter_names[index], "source": source, "value": value})
        operator = node.get("operator_id") or node.get("operator", {}).get("id") or node["label"]
        response = client.post("/api/custom-indicators/compose", json={
            "context": "single_product", "operator_id": operator,
            "dsl_version": dsl_version, "operator_registry_version": operator_registry_version,
            "arguments": arguments,
        })
        assert response.status_code == 200, response.text
        result = response.json()
        ast.parse(result["expression"], mode="eval")
        assert "\\" not in result["expression"]
        memo[key] = result
        return result

    return compose(dag["roots"][root_key])


@pytest.mark.parametrize("indicator_id", [
    "builtin-annualized-sharpe-v2",
    "builtin-close-moving-average-series",
    "builtin-bollinger-bands-series",
    "builtin-volume-moving-average-series",
    "builtin-kdj-series",
    "builtin-rolling-5d-annualized-sharpe-series",
])
@pytest.mark.parametrize("source_field", ["expression", "editable_latex"])
def test_catalog_compose_validate_roundtrip_all_channels(client: TestClient, service: CustomIndicatorService, indicator_id: str, source_field: str) -> None:
    definition = client.get(f"/api/custom-indicators/{indicator_id}").json()
    initial = client.post("/api/custom-indicators/validate", json=definition).json()
    assert initial["valid"], initial["diagnostics"]
    assert definition["editable_latex"] == editable_formula_latex(definition["expression"])
    for channel in definition.get("series_outputs") or []:
        assert channel["editable_latex"] == editable_formula_latex(channel["expression"])
    updated = copy.deepcopy(definition)
    if definition.get("result_kind", "scalar") == "scalar":
        updated["expression"] = recompose_dag(
            client,
            initial["dag"],
            "result",
            dsl_version=definition["dsl_version"],
            operator_registry_version=definition["operator_registry_version"],
        )[source_field]
    else:
        for channel in updated["series_outputs"]:
            channel["expression"] = recompose_dag(
                client,
                initial["dag"],
                channel["id"],
                dsl_version=definition["dsl_version"],
                operator_registry_version=definition["operator_registry_version"],
            )[source_field]
        updated["expression"] = updated["series_outputs"][0]["expression"]
    if updated.get("result_kind", "scalar") == "scalar":
        assert canonical_formula_source(updated["expression"]) == canonical_formula_source(definition["expression"])
    else:
        for before, after in zip(definition["series_outputs"], updated["series_outputs"], strict=True):
            assert canonical_formula_source(before["expression"]) == canonical_formula_source(after["expression"])
    checked = client.post("/api/custom-indicators/validate", json=updated).json()
    assert checked["valid"], checked["diagnostics"]
    assert checked["dependencies"] == initial["dependencies"]
    # A second pass must be exactly stable, not alternate between DSL/LaTeX.
    for root in checked["dag"]["roots"]:
        again = recompose_dag(
            client,
            checked["dag"],
            root,
            dsl_version=updated["dsl_version"],
            operator_registry_version=updated["operator_registry_version"],
        )
        expected = updated["expression"] if root == "result" else next(
            output["expression"] for output in updated["series_outputs"] if output["id"] == root
        )
        assert again[source_field] == expected
    if updated.get("rolling_source"):
        assert updated["rolling_source"]["detached"] is False
        saved = client.post("/api/custom-indicators", json={**updated, "name": "往返校验夏普"})
        assert saved.status_code in {200, 201}, saved.text


def test_edited_fifteen_observation_sharpe_preserves_sample_std(client: TestClient, service: CustomIndicatorService) -> None:
    expression = (
        "(rolling_mean(returns, 15) - risk_free_rate_per_observation) / "
        "rolling_std(returns, 15, 1) * sqrt(periods_per_year)"
    )
    draft = series_draft(service, expression)
    result = client.post("/api/custom-indicators/validate", json=draft).json()
    assert result["valid"], result["diagnostics"]
    composed = recompose_dag(
        client,
        result["dag"],
        "value",
        dsl_version=draft["dsl_version"],
        operator_registry_version=draft["operator_registry_version"],
    )
    assert composed["expression"] == (
        "(mean(rolling_window(returns, 15)) - risk_free_rate_per_observation) / "
        "std(rolling_window(returns, 15), 1) * sqrt(periods_per_year)"
    )
    assert r"s_{t,15}" in composed["display_latex"]
    assert "rolling_mean" not in composed["expression"]
    assert "rolling_std" not in composed["expression"]


@pytest.mark.parametrize("expression", [
    "rolling_mean(unknown_returns, 5)",
    "rolling_mean(returns, 5) + unknown_rate",
    "__import__('os')",
    "returns.__class__",
    "returns[0]",
])
def test_canonicalization_does_not_bypass_typed_validation(client: TestClient, service: CustomIndicatorService, expression: str) -> None:
    result = client.post("/api/custom-indicators/validate", json=series_draft(service, expression))
    assert result.status_code == 200
    assert result.json()["valid"] is False
    assert result.json()["diagnostics"]


@pytest.mark.parametrize("symbol", ["r_f", "r_{f}"])
def test_latex_variable_aliases_do_not_match_identifier_substrings(symbol: str) -> None:
    assert canonical_formula_source(symbol) == "risk_free_rate_per_observation"
    assert canonical_formula_source("custom_r_f + r_future") == "custom_r_f + r_future"


@pytest.mark.parametrize(("operator", "left", "right", "expected"), [
    ("multiply", "mean(returns) - 1", "2", "(mean(returns) - 1) * 2"),
    ("subtract", "2", "mean(returns) - 1", "2 - (mean(returns) - 1)"),
    ("divide", "mean(returns) + 1", "mean(returns) - 1", "(mean(returns) + 1) / (mean(returns) - 1)"),
])
def test_composed_dsl_preserves_nested_precedence(operator, left, right, expected) -> None:
    names = get_typed_operator_registry()[operator].argument_names(2)
    result = compose_expression({"operator_id": operator, "context": "single_product", "arguments": [
        {"parameter": names[0], "source": "expression", "value": left},
        {"parameter": names[1], "source": "expression", "value": right},
    ]})
    assert result["expression"] == expected
    assert canonical_formula_source(result["editable_latex"]) == expected


@pytest.mark.parametrize("expression", [
    "(rolling_mean(returns, 15) - risk_free_rate_per_observation) / rolling_std(returns, 15, 1) * sqrt(periods_per_year)",
    "rolling_std(returns, 15, 1, 10)",
    "recursive_smooth(rolling_mean(returns, 3, 1), 3, 50)",
    "returns - (returns - 1)",
    "(returns + 1) * (returns - 1)",
    "returns / (returns / 2)",
    "-(returns ** 2)",
    "(-returns) ** 2",
    "(returns ** 2) ** 3",
    "returns ** (2 ** 3)",
    "returns * 1e-20",
])
def test_editable_latex_is_lossless_and_idempotent(expression: str) -> None:
    latex = editable_formula_latex(expression)
    assert canonical_formula_source(latex) == canonical_formula_source(expression)
    assert editable_formula_latex(latex) == latex


def test_sharpe_editor_source_uses_latex_not_dsl() -> None:
    latex = editable_formula_latex(
        "(mean(returns) - risk_free_rate_per_observation) / std(returns, 1) * sqrt(periods_per_year)"
    )
    assert r"\frac{" in latex
    assert "r_f" in latex
    assert r"\sqrt{p_{\mathrm{year}}}" in latex
    assert "risk_free_rate_per_observation" not in latex
    assert "returns" not in latex
    assert r"\operatorname{std}\left(\mathbf{r},1\right)" in latex
