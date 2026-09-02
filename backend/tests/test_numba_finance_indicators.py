from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cal_indicators import latex_excutor
from cal_indicators.indicator_runtime import IndicatorRuntime, load_callable_map
from cal_indicators.latex_excutor import DAGBuildError


def test_in_memory_latex_runtime_computes_scalar_indicator() -> None:
    runtime = IndicatorRuntime.from_definition(
        "累计收益率",
        r"\left(\prod\left(\mathbf{r}+1\right)\right)-1",
        ["1W"],
    )
    returns = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    result = runtime.compute_period("1W", {"returns": returns})

    assert result["累计收益率"] == pytest.approx(float(np.prod(1.0 + returns) - 1.0))


def test_redundant_mathbf_wrapper_on_variable_is_normalized() -> None:
    runtime = IndicatorRuntime.from_definition(
        "对数收益波动率",
        r"\operatorname{std}\left(\mathbf{\mathbf{\ell}},1\right)",
        ["1M"],
    )
    log_returns = np.array([0.01, -0.02, 0.03], dtype=np.float64)

    result = runtime.compute_period("1M", {"log_returns": log_returns})

    assert runtime.executor.parser.to_python(
        r"\operatorname{std}\left(\mathbf{\mathbf{\ell}},1\right)"
    ) == "sequence_std(log_returns,1)"
    assert result["对数收益波动率"] == pytest.approx(float(np.std(log_returns, ddof=1)))


def test_runtime_builds_operator_registry_when_generated_dsl_is_missing(tmp_path: Path) -> None:
    mapping = load_callable_map(tmp_path / "missing-numba-finance-math-dsl.json")

    assert "sequence_mean" in mapping
    assert "sequence_std" in mapping
    assert "sqrt" in mapping


def test_topological_execution_does_not_require_networkx(monkeypatch) -> None:
    monkeypatch.setattr(latex_excutor, "nx", None)
    runtime = IndicatorRuntime.from_definition(
        "平均收益",
        r"\overline{\mathbf{r}}",
        ["1M"],
    )

    result = runtime.compute_period(
        "1M",
        {"returns": np.array([0.01, 0.02, 0.03], dtype=np.float64)},
    )

    assert result["平均收益"] == pytest.approx(0.02)


@pytest.mark.parametrize(
    "expression, expected_message",
    [
        ("unknown_variable + 1", "未知变量"),
        ("unknown_function(returns)", "未知函数"),
        ("sequence_mean(1)", "需要 vector"),
        (r"\mathbf{r}+1", "最终结果必须是标量"),
        ("returns.__class__", "不支持的 AST 节点"),
    ],
)
def test_restricted_expression_policy_rejects_unsafe_or_invalid_dag(
    expression: str,
    expected_message: str,
) -> None:
    with pytest.raises(DAGBuildError, match=expected_message):
        IndicatorRuntime.from_definition("非法指标", expression, ["1M"])


def test_expression_policy_enforces_depth_limit() -> None:
    with pytest.raises(DAGBuildError, match="表达式深度"):
        IndicatorRuntime.from_definition(
            "过深公式",
            "((((returns + 1) + 1) + 1) + 1)",
            ["1M"],
            max_depth=3,
        )
