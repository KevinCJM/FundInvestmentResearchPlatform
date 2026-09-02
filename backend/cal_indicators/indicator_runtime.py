"""
根据 LaTeX 指标定义执行算子 DAG 的运行时。

该模块加载 latex_excutor 构建的拓扑结构与 numba_finance_math DSL，
根据指定版本与周期，在给定上下文下依序计算指标数值。
"""
import json
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = BASE_DIR.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from cal_indicators import numba_finance_math as math_ops  # noqa: E402
from cal_indicators.generate_math_operator_dsl import build_operators  # noqa: E402
from cal_indicators.latex_excutor import (  # noqa: E402
    DAGBuildError,
    DAGNode,
    ExpressionPolicy,
    LatexExecutor,
)


def _binary_op(label: str) -> Any:
    if label == "add":
        return np.add
    if label == "subtract":
        return np.subtract
    if label == "multiply":
        return np.multiply
    if label == "divide":
        return np.divide
    if label == "power":
        return np.power
    raise ValueError(f"未知的二元运算符: {label}")


BINARY_LABELS = {"add", "subtract", "multiply", "divide", "power"}
UNARY_LABELS = {"negate"}


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    arity: int
    input_types: tuple[str, ...]
    output_type: str
    signature: str
    description: str
    latex: str
    is_ufunc: bool = False


def _dsl_type(raw: object) -> str:
    token = str(raw or "")
    if "Tuple" in token or "tuple" in token.lower():
        return "tuple"
    if "[:]" in token or "array" in token.lower():
        return "vector"
    return "scalar"


@lru_cache(maxsize=4)
def load_operator_entries(dsl_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """从可选产物加载算子；产物缺失时直接从受控源码构建。"""

    if dsl_path is not None and dsl_path.exists():
        with dsl_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return list(payload.get("operators", []))
    operators, _ = build_operators()
    return operators


@lru_cache(maxsize=4)
def load_operator_specs(dsl_path: Optional[Path] = None) -> Dict[str, OperatorSpec]:
    specs: Dict[str, OperatorSpec] = {}
    for operator in load_operator_entries(dsl_path):
        name = str(operator["name"])
        inputs = tuple(_dsl_type(item.get("type")) for item in operator.get("inputs", []))
        output = _dsl_type(operator.get("output", {}).get("type"))
        spec = OperatorSpec(
            name=name,
            arity=len(inputs),
            input_types=inputs,
            output_type=output,
            signature=str(operator.get("signature", "")),
            description=str(operator.get("description", "")),
            latex=str(operator.get("latex", "")),
            is_ufunc=bool(operator.get("is_ufunc", False)),
        )
        specs[name] = spec
        for alias in operator.get("aliases", []):
            specs[str(alias)] = spec
    specs["sqrt"] = OperatorSpec(
        name="sqrt",
        arity=1,
        input_types=("scalar",),
        output_type="scalar",
        signature="float64(float64)",
        description="计算非负标量的平方根。",
        latex=r"\sqrt{x}",
        is_ufunc=True,
    )
    return specs


@lru_cache(maxsize=4)
def load_callable_map(dsl_path: Optional[Path] = None) -> Dict[str, Any]:
    mapping: Dict[str, Any] = {}
    for operator in load_operator_entries(dsl_path):
        name = operator["name"]
        func = getattr(math_ops, name, None)
        if callable(func):
            mapping[name] = func
        for alias in operator.get("aliases", []):
            if callable(func):
                mapping[alias] = func
    mapping["sqrt"] = np.sqrt
    return mapping


PERIOD_UNIT_FACTORS = {
    "W": 1.0 / 52.0,
    "M": 1.0 / 12.0,
    "Y": 1.0,
}

BASE_FREQUENCY_FACTORS = {
    "日": 1.0 / 252.0,
    "周": 1.0 / 52.0,
    "月": 1.0 / 12.0,
}

RE_NUM_UNIT = re.compile(r"^(?P<num>\d+)(?P<unit>[WMY])$", re.IGNORECASE)
RE_UNIT_NUM = re.compile(r"^(?P<unit>[WMY])(?P<num>\d+)$", re.IGNORECASE)


def _normalize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return tuple(_normalize_value(v) for v in value)
    return value


def _parse_constant(label: str) -> Any:
    try:
        if "." in label:
            return float(label)
        return int(label)
    except ValueError:
        if label.lower() == "true":
            return True
        if label.lower() == "false":
            return False
        raise


class IndicatorRuntime:
    """
    指标运行时类，用于根据表达式节点图计算指标值。

    该类通过构建表达式执行器（LatexExecutor）来解析和计算指标表达式，
    并支持按周期（period）对多个指标进行批量计算。
    """

    def __init__(
            self,
            version: Optional[str] = None,
            profile_payload: Optional[Dict[str, object]] = None,
            dsl_path: Optional[Path] = None,
            max_nodes: int = 128,
            max_depth: int = 20,
    ) -> None:
        """初始化 IndicatorRuntime 实例。"""
        if not version and profile_payload is None:
            raise ValueError("必须指定指标版本名称或内存指标定义。")
        self.operator_specs = load_operator_specs(dsl_path)
        self.callables = load_callable_map(dsl_path)
        self.executor = LatexExecutor(version=version, profile_payload=profile_payload)
        variable_names = set(self.executor.variables) or {
            "returns",
            "log_returns",
            "annual_risk_free_rate_decimal",
            "risk_free_rate_per_period",
        }
        policy = ExpressionPolicy(
            allowed_variables=variable_names,
            function_arity={name: spec.arity for name, spec in self.operator_specs.items()},
            max_nodes=max_nodes,
            max_depth=max_depth,
        )
        self.executor.build(policy=policy)
        self._validate_types()
        self.version = self.executor.version
        self.metadata = self.executor.metadata

    @classmethod
    def from_definition(
            cls,
            name: str,
            expression: str,
            periods: List[str],
            metadata: Optional[Dict[str, object]] = None,
            **kwargs: Any,
    ) -> "IndicatorRuntime":
        profile: Dict[str, object] = {
            "metadata": dict(metadata or {}),
            "indicators": [
                {
                    "name": name,
                    "dsl_expression": expression,
                    "periods": periods or ["__default__"],
                }
            ],
        }
        return cls(profile_payload=profile, **kwargs)

    @property
    def available_versions(self) -> List[str]:
        return self.executor.available_versions

    @property
    def available_periods(self) -> List[str]:
        return list(self.executor.roots.keys())

    def _variable_types(self) -> Mapping[str, str]:
        variable_types = {
            name: "vector" if str(entry.get("value_type")) == "vector" else "scalar"
            for name, entry in self.executor.variables.items()
        }
        if variable_types:
            return variable_types
        return {
            "returns": "vector",
            "log_returns": "vector",
            "annual_risk_free_rate_decimal": "scalar",
            "risk_free_rate_per_period": "scalar",
        }

    def _validate_types(self) -> None:
        variables = self._variable_types()
        for period in self.executor.roots:
            inferred: Dict[int, str] = {}
            for node in self.executor.topo_order_for_period(period):
                if node.kind == "constant":
                    inferred[node.node_id] = "scalar"
                elif node.kind == "variable":
                    inferred[node.node_id] = variables[node.label]
                elif node.kind == "unary":
                    input_type = inferred[node.inputs[0]]
                    if input_type == "tuple":
                        raise DAGBuildError("tuple 不能参与一元运算")
                    inferred[node.node_id] = input_type
                elif node.kind == "binary":
                    input_types = [inferred[input_id] for input_id in node.inputs]
                    if "tuple" in input_types:
                        raise DAGBuildError("tuple 不能参与二元运算")
                    inferred[node.node_id] = "vector" if "vector" in input_types else "scalar"
                elif node.kind == "call":
                    spec = self.operator_specs[node.label]
                    actual_types = [inferred[input_id] for input_id in node.inputs]
                    for index, (actual, expected) in enumerate(zip(actual_types, spec.input_types), start=1):
                        vectorized_scalar = spec.is_ufunc and expected == "scalar" and actual == "vector"
                        if actual != expected and not vectorized_scalar:
                            raise DAGBuildError(
                                f"函数 {node.label} 第 {index} 个参数需要 {expected}，实际为 {actual}"
                            )
                    if spec.is_ufunc and "vector" in actual_types:
                        inferred[node.node_id] = "vector"
                    else:
                        inferred[node.node_id] = spec.output_type
                else:  # pragma: no cover - builder guarantees node kinds
                    raise DAGBuildError(f"未知节点类型: {node.kind}")
            for name, root_id in self.executor.roots[period].items():
                if inferred[root_id] != "scalar":
                    raise DAGBuildError(f"指标 {name} 的最终结果必须是标量")

    def _annual_rate_decimal(self) -> float:
        rate_percent = self.metadata.get("annual_risk_free_rate_percent")
        try:
            return float(rate_percent) / 100.0 if rate_percent is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _base_period_fraction(self) -> float:
        freq = str(self.metadata.get("data_frequency", "日"))
        return BASE_FREQUENCY_FACTORS.get(freq, 1.0)

    def _period_fraction(self, period: str) -> float:
        if not period or period == "__default__":
            return self._base_period_fraction()
        token = period.upper()
        match = RE_NUM_UNIT.match(token)
        if not match:
            match = RE_UNIT_NUM.match(token)
        if match:
            unit = match.group("unit").upper()
            try:
                num = float(match.group("num"))
            except (TypeError, ValueError):
                num = 1.0
            factor = PERIOD_UNIT_FACTORS.get(unit, 1.0)
            return num * factor
        return self._base_period_fraction()

    def _risk_free_rate_for_period(self, period: str) -> float:
        annual = self._annual_rate_decimal()
        fraction = self._period_fraction(period)
        if fraction <= 0.0:
            return 0.0
        return (1.0 + annual) ** fraction - 1.0

    def _evaluate_nodes(
            self, order: Iterable[DAGNode], context: Dict[str, Any]
    ) -> Dict[int, Any]:
        """
        根据给定的节点顺序和上下文计算每个节点的值。

        参数:
            order (Iterable[DAGNode]): 按拓扑排序排列的节点列表，确保依赖关系正确。
            context (Dict[str, Any]): 变量名到值的映射，用于提供变量节点的值。

        返回:
            Dict[int, Any]: 节点 ID 到其计算结果的映射。

        异常:
            KeyError: 当变量节点在上下文中找不到对应值时抛出。
            ValueError: 当遇到未知的节点类型、运算符或函数时抛出。
        """
        cache: Dict[int, Any] = {}

        # 遍历所有节点，按照拓扑顺序计算每个节点的值
        for node in order:
            if node.kind == "constant":
                # 处理常量节点：解析标签并缓存值
                cache[node.node_id] = _parse_constant(node.label)
            elif node.kind == "variable":
                # 处理变量节点：从上下文中获取值
                if node.label not in context:
                    raise KeyError(f"缺少变量 `{node.label}` 的值")
                cache[node.node_id] = context[node.label]
            elif node.kind == "unary":
                # 处理一元运算节点：应用指定的一元运算符
                if node.label not in UNARY_LABELS:
                    raise ValueError(f"未知的一元运算符: {node.label}")
                value = cache[node.inputs[0]]
                cache[node.node_id] = np.negative(value)
            elif node.kind == "binary":
                # 处理二元运算节点：使用指定的二元运算符处理两个操作数
                if node.label not in BINARY_LABELS:
                    raise ValueError(f"未知的二元运算符: {node.label}")
                op = _binary_op(node.label)
                lhs = cache[node.inputs[0]]
                rhs = cache[node.inputs[1]]
                cache[node.node_id] = op(lhs, rhs)
            elif node.kind == "call":
                # 处理函数调用节点：调用注册的函数并传入参数
                if node.label not in self.callables:
                    raise ValueError(f"未知的函数调用: {node.label}")
                func = self.callables[node.label]
                args = [cache[input_id] for input_id in node.inputs]
                cache[node.node_id] = func(*args)
            else:
                # 不支持的节点类型
                raise ValueError(f"未知的节点类型: {node.kind}")

        return cache

    def compute_period(
            self, period: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        计算指定周期内的所有指标值。

        参数:
            period (str): 周期标识符，用于确定要计算哪些指标。
            context (Dict[str, Any]): 提供给表达式中变量的值。

        返回:
            Dict[str, Any]: 指标名称到其计算结果的映射。
        """
        # 获取当前周期下所有节点的拓扑排序
        context_local = dict(context)
        context_local["annual_risk_free_rate_decimal"] = self._annual_rate_decimal()
        context_local["risk_free_rate_per_period"] = self._risk_free_rate_for_period(period)

        order = self.executor.topo_order_for_period(period)

        # 执行节点计算，得到所有节点的结果缓存
        values = self._evaluate_nodes(order, context_local)

        # 收集该周期下所有根节点（即指标表达式的最终结果节点）的值
        return {
            indicator_name: _normalize_value(values[root_id])
            for indicator_name, root_id in self.executor.roots[period].items()
        }


def demo(version: str, period: str) -> None:
    runtime = IndicatorRuntime(version=version)
    print("可用指标版本:", runtime.available_versions)
    print("当前使用版本:", runtime.version)
    print("可用周期:", runtime.available_periods)
    returns = np.array(
        [0.01, -0.005, 0.007, 0.012, -0.003], dtype=np.float64
    )
    context = {
        "returns": returns,
    }
    outputs = runtime.compute_period(period, context)
    print(f"周期 {period} 指标结果：")
    for name, value in outputs.items():
        if isinstance(value, np.ndarray):
            value_repr = value.tolist()
        elif isinstance(value, (float, np.floating)):
            value_repr = float(value)
        elif isinstance(value, tuple):
            value_repr = tuple(float(v) if isinstance(v, np.floating) else v for v in value)
        else:
            value_repr = value
        print(f"  {name}: {value_repr}")


if __name__ == "__main__":
    demo("指标计算示例模板", "1W")
