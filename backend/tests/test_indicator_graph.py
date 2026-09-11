"""Offline authoring contract, conversion and isolated layout regression."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as ModelValidationError

from custom_indicators.errors import ConflictError
from custom_indicators.graph_contracts import AuthoringGraph, CanvasResolveRequest, EditorStateUpdate, FormulaResolveRequest
from custom_indicators.graph_service import IndicatorGraphService
from custom_indicators.repository import IndicatorRepository
from custom_indicators.service import CustomIndicatorService


@pytest.fixture
def graph_service(tmp_path):
    service = SimpleNamespace(
        workspace_data_dir=tmp_path,
        _normalize_definition=CustomIndicatorService._normalize_definition,
        indicators=IndicatorRepository(tmp_path / "indicators.json", []),
    )
    return IndicatorGraphService(service)


def formula(service, expression="mean(returns)", **kwargs):
    expressions = expression if isinstance(expression, dict) else {"result": expression}
    return service.resolve(FormulaResolveRequest(source_kind="formula", expressions=expressions, **kwargs))


def canvas(service, graph, **kwargs):
    return service.resolve(CanvasResolveRequest(source_kind="graph", graph=AuthoringGraph.model_validate(graph), **kwargs))


def assert_valid(result):
    assert result["valid"], result.get("diagnostics")
    return result


@pytest.mark.parametrize("expression", [
    "mean(returns)", "(mean(returns) - 2) / (3 - mean(returns))",
    "mean(returns) / mean(returns)", "-mean(returns)", "2", "mean(returns) ** 2",
])
def test_scalar_roundtrip_and_shared_named_arguments(graph_service, expression):
    first = assert_valid(formula(graph_service, expression))
    second = assert_valid(canvas(graph_service, first["graph"]))
    assert second["definition_fingerprint"] == first["definition_fingerprint"]
    assert len(second["editor_to_compiled"]) == len(second["graph"]["nodes"])
    if expression == "mean(returns) / mean(returns)":
        operators = [node for node in first["graph"]["nodes"] if node["kind"] == "operator"]
        assert len([node for node in operators if node["operator_id"] == "mean"]) == 1
        divide = next(node for node in operators if node["operator_id"] == "divide")
        assert len(divide["arguments"]) == 2
        assert len({item["node_id"] for item in divide["arguments"].values()}) == 1


def test_multiple_series_outputs_share_one_window_and_one_mean(graph_service):
    first = assert_valid(formula(graph_service, {
        "average": "rolling_mean(market_close, 20)",
        "deviation": "market_close / rolling_mean(market_close, 20) - 1",
    }, result_kind="time_series"))
    windows = [node for node in first["graph"]["nodes"] if node.get("operator_id") == "rolling_window"]
    means = [node for node in first["graph"]["nodes"] if node.get("operator_id") == "mean"]
    assert len(windows) == len(means) == 1
    by_id = {node["id"]: node for node in first["graph"]["nodes"]}
    assert all(item["source"] == "node" for item in windows[0]["arguments"].values())
    assert any(by_id[item["node_id"]].get("value") == 20 for item in windows[0]["arguments"].values())
    assert next(iter(means[0]["arguments"].values()))["node_id"] == windows[0]["id"]
    second = assert_valid(canvas(graph_service, first["graph"], result_kind="time_series"))
    assert first["definition_fingerprint"] == second["definition_fingerprint"]


def test_equal_literals_are_independent_authoring_nodes(graph_service):
    resolved = assert_valid(formula(graph_service, "product(returns + 1) - 1"))
    constants = [node for node in resolved["graph"]["nodes"] if node["kind"] == "constant"]
    assert len(constants) == 2
    assert constants[0]["id"] != constants[1]["id"]
    assert constants[0]["value"] == constants[1]["value"] == 1
    # A graph reload is deterministic, but equal literals are not coupled edits.
    assert formula(graph_service, "product(returns + 1) - 1")["graph"] == resolved["graph"]
    constants[0]["value"] = 2
    changed = assert_valid(canvas(graph_service, resolved["graph"]))
    assert changed["expressions"]["result"] == "product(returns + 2) - 1"


@pytest.mark.parametrize("value", [0, 2.5])
def test_fixed_window_node_rejects_invalid_values(graph_service, value):
    resolved = assert_valid(formula(graph_service, {"value": "rolling_mean(returns, 5)"}, result_kind="time_series"))
    constant = next(node for node in resolved["graph"]["nodes"] if node["kind"] == "constant")
    constant["value"] = value
    result = canvas(graph_service, resolved["graph"], result_kind="time_series")
    assert not result["valid"]
    assert result["diagnostics"][0]["editor_node_id"]


def test_fixed_window_cannot_be_replaced_by_dynamic_variable(graph_service):
    graph = assert_valid(formula(graph_service, {"value": "rolling_mean(returns, 5)"}, result_kind="time_series"))["graph"]
    constant = next(node for node in graph["nodes"] if node["kind"] == "constant")
    constant_id = constant["id"]
    graph["nodes"] = [node if node["id"] != constant_id else {"id": constant_id, "kind": "variable", "variable_id": "periods_per_year"} for node in graph["nodes"]]
    result = canvas(graph_service, graph, result_kind="time_series")
    assert not result["valid"]
    assert result["diagnostics"][0]["code"] == "SERIES_CONFIGURATION_MUST_BE_CONSTANT"


def test_legacy_inline_graphs_keep_the_same_formula_fingerprint(graph_service):
    resolved = assert_valid(formula(graph_service, "product(returns + 1) - 1"))
    graph = deepcopy(resolved["graph"])
    by_id = {node["id"]: node for node in graph["nodes"]}
    for node in graph["nodes"]:
        if node["kind"] == "operator":
            for name, binding in node["arguments"].items():
                source = by_id[binding["node_id"]]
                if source["kind"] == "constant":
                    node["arguments"][name] = {"source": "constant", "value": source["value"]}
    graph["nodes"] = [node for node in graph["nodes"] if node["kind"] != "constant"]
    result = assert_valid(canvas(graph_service, graph))
    assert result["definition_fingerprint"] == resolved["definition_fingerprint"]


def test_resolve_never_compiles_or_reads_market_data(graph_service, monkeypatch):
    import custom_indicators.typed_service as typed
    import cal_indicators.typed_numba_plan as numba_plan
    def forbidden(*args, **kwargs):
        raise AssertionError("authoring must not compile NJIT")
    monkeypatch.setattr(typed, "compile_numba_plan", forbidden)
    monkeypatch.setattr(numba_plan, "compile_numba_plan", forbidden)
    first = assert_valid(formula(graph_service, draft_revision=37))
    second = assert_valid(canvas(graph_service, first["graph"], draft_revision=38))
    assert second["draft_revision"] == 38
    assert second["compile_status"] == "not_requested"
    assert "compile_token" not in second


@pytest.mark.parametrize("kind,code", [
    ("duplicate", "DUPLICATE_NODE"), ("unknown", "UNKNOWN_NODE"),
    ("cycle", "GRAPH_CYCLE"), ("unused", "UNUSED_NODE"),
    ("missing_output", "OUTPUT_NOT_CONNECTED"), ("missing_input", "MISSING_ARGUMENT"),
])
def test_invalid_graphs_are_localized(graph_service, kind, code):
    graph = assert_valid(formula(graph_service))["graph"]
    operator = next(node for node in graph["nodes"] if node["kind"] == "operator")
    if kind == "duplicate":
        graph["nodes"].append(deepcopy(graph["nodes"][0]))
    elif kind in {"unknown", "cycle"}:
        name = next(iter(operator["arguments"]))
        operator["arguments"][name] = {"source": "node", "node_id": operator["id"] if kind == "cycle" else "missing"}
    elif kind == "unused":
        graph["nodes"].append({"id": "unused", "kind": "constant", "value": 42})
    elif kind == "missing_output":
        graph["outputs"][0]["node_id"] = None
    else:
        operator["arguments"] = {}
    result = canvas(graph_service, graph)
    assert not result["valid"]
    assert result["diagnostics"][0]["code"] == code
    assert result["graph"] is not None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "20"])
def test_constants_reject_non_finite_and_strings(value):
    with pytest.raises(ModelValidationError):
        AuthoringGraph.model_validate({"nodes": [{"id": "n", "kind": "constant", "value": value}], "outputs": [{"id": "result", "node_id": "n"}]})


def test_root_type_and_protocol_rejected(graph_service):
    assert not formula(graph_service, "returns")["valid"]
    assert not formula(graph_service, "mean(returns)", result_kind="time_series")["valid"]
    assert not formula(graph_service, operator_registry_version="missing")["valid"]
    assert not formula(graph_service, numeric_kernel_version="missing")["valid"]
    assert not formula(graph_service, dsl_version="1.0.0")["valid"]


def test_layout_revision_does_not_change_definition(graph_service):
    resolved = assert_valid(formula(graph_service))
    definition = graph_service.indicators.indicators.create({
        "name": "测试均值", "expression": resolved["expressions"]["result"],
        "dsl_version": "2.3.0", "result_kind": "scalar", "context_kind": "single_product",
    })
    indicator_id = definition["id"]
    assert graph_service.read_state(indicator_id, 1)["editor_revision"] == 0
    request = EditorStateUpdate(expected_editor_revision=0, graph=resolved["graph"], positions={})
    stored = graph_service.save_state(indicator_id, 1, request)
    assert stored["editor_revision"] == 1
    assert graph_service.indicators.indicators.get(indicator_id)["revision"] == 1
    with pytest.raises(ConflictError) as conflict:
        graph_service.save_state(indicator_id, 1, request)
    assert conflict.value.code == "EDITOR_REVISION_CONFLICT"
    request.graph = AuthoringGraph.model_validate(assert_valid(formula(graph_service, "sum(returns)"))["graph"])
    request.expected_editor_revision = 1
    with pytest.raises(ConflictError) as mismatch:
        graph_service.save_state(indicator_id, 1, request)
    assert mismatch.value.code == "EDITOR_DEFINITION_MISMATCH"


def test_route_source_is_exclusive_and_errors_stable(graph_service, monkeypatch):
    from services import custom_indicator_routes as routes
    monkeypatch.setattr(routes, "indicator_service", graph_service.indicators)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    request = {"source_kind": "formula", "expressions": {"result": "mean(returns)"}, "draft_revision": 7}
    response = client.post("/api/custom-indicators/graph/resolve", json=request)
    assert response.status_code == 200
    assert_valid(response.json())
    response = client.post("/api/custom-indicators/graph/resolve", json={**request, "graph": {"nodes": [], "outputs": [{"id": "result"}]}})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "REQUEST_VALIDATION_ERROR"
