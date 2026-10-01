from __future__ import annotations
from pathlib import Path
from typing import Any

DEFINITION: dict[str, Any] = {
    "name": "AI 累计收益",
    "description": "智能体测试定义",
    "expression": r"\left(\prod\left(\mathbf{r}+1\right)\right)-1",
    "periods": ["1Y"],
    "dsl_version": "2.1.0",
    "context_kind": "single_product",
    "result_kind": "scalar",
}

COMPILE_TOKEN = "a" * 64

class FakePortfolioRuns:
    def get(self, run_id: str) -> dict[str, Any]:
        return {
            "id": run_id,
            "created_at": "2026-09-01T00:00:00+00:00",
            "immutable": True,
            "requested_as_of": "2026-08-29",
            "effective_as_of": "2026-08-29",
        }

class FakeIndicatorService:
    """Spy implementing only the surface the agent is allowed to call."""

    def __init__(self, market_data_dir: Path) -> None:
        self.market_data_dir = market_data_dir
        self.validate_calls: list[dict[str, Any]] = []
        self.evaluate_calls: list[dict[str, Any]] = []
        self.evaluate_series_calls: list[dict[str, Any]] = []
        self.evaluate_portfolio_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.portfolio_runs = FakePortfolioRuns()

    def meta(self) -> dict[str, Any]:
        return {
            "engine_version": "test-engine",
            "dsl_version": "2.1.0",
            "operator_registry_version": "op-registry-1",
            "variable_registry_version": "var-registry-1",
            "periods": [{"id": "1Y"}],
        }

    def list_indicators(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "items": [
                {
                    "id": "indicator-demo",
                    "name": "演示指标",
                    "revision": 1,
                    "context_kind": "single_product",
                    "result_kind": "scalar",
                    "source": "custom",
                    "indicator_type": "other",
                }
            ],
            "total": 1,
        }

    def validate(self, fields: dict[str, Any]) -> dict[str, Any]:
        self.validate_calls.append(dict(fields))
        expression = str(fields.get("expression") or "")
        if "unknown" in expression:
            return {
                "valid": False,
                "diagnostics": [{"code": "UNKNOWN_FUNCTION", "message": "未知函数", "field": "expression"}],
                "dependencies": [],
            }
        return {
            "valid": True,
            "diagnostics": [],
            "dependencies": ["returns"],
            "display_latex": r"\operatorname{mean}(r)",
            "editable_latex": expression,
            "compile_token": COMPILE_TOKEN,
        }

    def infer(self, fields: dict[str, Any]) -> dict[str, Any]:
        return {"expression": fields.get("expression")}

    def availability(self, **kwargs: Any) -> dict[str, Any]:
        return {"targets": kwargs.get("targets"), "items": []}

    def evaluate(self, **kwargs: Any) -> dict[str, Any]:
        self.evaluate_calls.append(dict(kwargs))
        return {"results": [{"indicator": "demo", "value": 0.123}], "execution": {"nopython": True}}

    def evaluate_series(self, **kwargs: Any) -> dict[str, Any]:
        self.evaluate_series_calls.append(dict(kwargs))
        return {"series": []}

    def evaluate_portfolio(self, **kwargs: Any) -> dict[str, Any]:
        self.evaluate_portfolio_calls.append(dict(kwargs))
        return {"results": []}

    def list_plans(self, kind: str | None = None) -> dict[str, Any]:
        return {
            "items": [
                {
                    "id": "plan-1",
                    "name": "方案",
                    "revision": 1,
                    "product_kind": "etf",
                    "indicators": [{"indicator_id": "indicator-demo"}],
                    "targets": [{"kind": "etf", "product_id": "510300.SH"}],
                }
            ],
            "total": 1,
        }

    def create_indicator(self, fields: dict[str, Any]) -> dict[str, Any]:
        self.create_calls.append(dict(fields))
        return {**fields, "id": f"indicator-fake-{len(self.create_calls)}", "revision": 1, "source": "custom"}

def single_context(page: str = "product-detail") -> dict[str, Any]:
    return {
        "page": page,
        "page_instance_id": "instance-1",
        "context_revision": 3,
        "view_state": "inherit",
        "calculation": {
            "context_kind": "single_product",
            "targets": [{"kind": "etf", "product_id": "510300.SH"}],
            "period": "1Y",
        },
    }

def authoring_context():
    context = single_context('indicator-studio')
    context['calculation']['targets'] = []
    return context

SENTINEL = 987654.321

APPROVED_DEFINITION = {
    "name": "AI 平均收益",
    "description": "智能体测试定义",
    "expression": "mean(returns)",
    "periods": ["1Y"],
    "dsl_version": "2.4.0",
    "context_kind": "single_product",
    "result_kind": "scalar",
}
