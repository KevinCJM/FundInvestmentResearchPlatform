"""
根据 LaTeX 指标定义执行算子 DAG 的运行时。

该模块加载 latex_excutor 构建的拓扑结构与 numba_finance_math DSL，
根据指定版本与周期，在给定上下文下依序计算指标数值。
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = BASE_DIR.parent

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from cal_indicators import numba_finance_math as math_ops  # noqa: E402
from cal_indicators.latex_excutor import DAGNode, LatexExecutor  # noqa: E402


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


def load_callable_map(dsl_path: Path) -> Dict[str, Any]:
    with dsl_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping: Dict[str, Any] = {}
    for operator in payload.get("operators", []):
        name = operator["name"]
        func = getattr(math_ops, name, None)
        if callable(func):
            mapping[name] = func
        for alias in operator.get("aliases", []):
            if callable(func):
                mapping[alias] = func
    mapping["sqrt"] = np.sqrt
    return mapping


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

    def __init__(self, version: str) -> None:
        """初始化 IndicatorRuntime 实例。"""
        if not version:
            raise ValueError("必须指定指标版本名称。")
        self.executor = LatexExecutor(version=version)
        self.executor.build()
        self.callables = load_callable_map(BASE_DIR / "numba_finance_math_dsl.json")
        self.version = self.executor.version
        self.metadata = self.executor.metadata

    @property
    def available_versions(self) -> List[str]:
        return self.executor.available_versions

    @property
    def available_periods(self) -> List[str]:
        return list(self.executor.roots.keys())

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
        order = self.executor.topo_order_for_period(period)

        # 执行节点计算，得到所有节点的结果缓存
        values = self._evaluate_nodes(order, context)

        # 收集该周期下所有根节点（即指标表达式的最终结果节点）的值
        return {
            indicator_name: _normalize_value(values[root_id])
            for indicator_name, root_id in self.executor.roots[period].items()
        }


def demo() -> None:
    runtime = IndicatorRuntime(version="指标计算示例模板")
    print("可用指标版本:", runtime.available_versions)
    print("当前使用版本:", runtime.version)
    returns = np.array(
        [0.01, -0.005, 0.007, 0.012, -0.003], dtype=np.float64
    )
    metadata = runtime.metadata
    context = {
        "returns": returns,
        "risk_free_rate_per_period": metadata.get("annual_risk_free_rate_per_period", 0.0001),
    }
    period = runtime.available_periods[0]
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
    demo()
