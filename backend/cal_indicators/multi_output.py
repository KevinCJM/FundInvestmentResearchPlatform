"""Static named-port contracts shared by the compiler and authoring UI."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .typed_types import TypedDslError, ValueType
from .multi_output_kernels import drawdown_analysis_kernel


@dataclass(frozen=True)
class ScalarPort:
    id: str
    label: str
    description: str
    measure: str = "count"
    unit: str = "期"
    display_format: str = "number"
    precision: int = 0
    direction: str = "neutral"
    missing_message: str = "此结果不可计算。"
    allow_missing: bool = False

    def value_type(self) -> ValueType:
        return ValueType.scalar(semantic_dimension=self.measure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "label": self.label, "description": self.description,
            "type": self.value_type().to_dict(), "output_measure": self.measure,
            "unit": self.unit, "display_format": self.display_format,
            "precision": self.precision, "direction": self.direction,
            "missing_message": self.missing_message,
        }


DRAWDOWN_PORTS = (
    ScalarPort("max_drawdown", "最大回撤", "窗口内从历史峰值下跌的最大幅度，正数表示损失幅度。", "return_decimal", "%", "percent", 2, "lower_better"),
    ScalarPort("decline_periods", "最大回撤下跌期数", "最大回撤对应峰值到谷底的观察间隔数；不是自然日。"),
    ScalarPort("recovery_periods", "最大回撤恢复期数", "最大回撤谷底到首次回到原峰值的观察间隔数；未恢复时不填零。", missing_message="最大回撤尚未恢复；恢复期数暂不可得。", allow_missing=True),
    ScalarPort("longest_underwater_periods", "最长水下期数", "所有回撤段中，峰值至恢复或窗口截止的最长观察间隔数，包含尚未恢复段的已观察长度。"),
)

# A valid operator can have an unavailable optional statistic (for example,
# time to recovery). This is different from invalid inputs or a failed operator.
STATUS_OUTPUT_UNAVAILABLE = 7

# Multi-output kernels are an explicit registry, not arbitrary Python plugins.
MULTI_OUTPUT_PORTS: dict[str, tuple[ScalarPort, ...]] = {"drawdown_analysis": DRAWDOWN_PORTS}
MULTI_OUTPUT_KERNELS = {"drawdown_analysis": drawdown_analysis_kernel}


def drawdown_record_type(inputs: tuple[ValueType, ...]) -> ValueType:
    source = inputs[0]
    if source.kind != "series" or not source.is_numeric:
        raise TypedDslError("TYPE_MISMATCH", "最大回撤分析需要净值/价格序列，不能使用标量或结果组。")
    if source.semantic_dimension in {"return_decimal", "rate_decimal"}:
        raise TypedDslError("SEMANTIC_MISMATCH", "请使用净值序列，而不是收益率序列；收益率应先转换成净值。")
    return ValueType("record", fields=tuple((port.id, port.value_type()) for port in DRAWDOWN_PORTS))


def multi_output_specs(version: str):
    # Lazy import keeps the registry independent of the numeric runtime.
    from .typed_operators import OperatorSignature, TypedOperatorSpec
    return (TypedOperatorSpec(
        "drawdown_analysis", version, "path", (OperatorSignature(("series<time>[L]",), "record", "one scan; named scalar ports"),),
        "一次扫描净值，同时得到最大回撤、下跌期数、恢复期数和最长水下期数。",
        drawdown_record_type, drawdown_analysis_kernel,
        cost_model="scan", cost=lambda inputs, output: str(inputs[0].shape[0]),
        latex_template=r"\operatorname{drawdown_analysis}(p)",
    ),)
