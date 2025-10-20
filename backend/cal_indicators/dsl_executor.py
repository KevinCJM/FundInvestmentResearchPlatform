import ast
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = CURRENT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from cal_indicators import numba_finance_indicators as indicator_ops  # noqa: E402
from cal_indicators import numba_finance_math as math_ops  # noqa: E402


@dataclass
class Node:
    """
    表示 DSL 计算图中的一个节点。

    kind:
        - input: 外部输入变量
        - constant: 常量值
        - call: 函数调用
    """

    node_id: int
    kind: str
    label: str
    inputs: List[int] = field(default_factory=list)
    func: Optional[Callable[..., Any]] = None
    value: Optional[Any] = None


@dataclass
class ExecutionPlan:
    """封装解析得到的 DAG、拓扑序和根节点。"""

    root_id: int
    nodes: Dict[int, Node]
    adjacency: Dict[int, List[int]]
    topo_order: List[int]


class DSLExecutionError(Exception):
    """DSL 解析或执行出现问题时抛出。"""


class DSLExecutor:
    """
    负责解析 DSL 表达式，构建 DAG，生成执行计划并完成计算。

    expression 语法与 Python 调用表达式接近，例如：
        sub(sequence_prod(add(returns, 1.0)), 1.0)
    """

    DEFAULT_ALIASES = {
        "sub": "subtract",
        "mul": "multiply",
        "div": "divide",
        "neg": "negate",
        "abs": "absolute",
        "max": "maximum",
        "min": "minimum",
        "pow": "power",
    }

    def __init__(
        self,
        modules: Optional[Iterable[Any]] = None,
        aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        if modules is None:
            modules = (math_ops, indicator_ops, np)
        self.modules = tuple(modules)
        self.aliases = dict(self.DEFAULT_ALIASES)
        if aliases:
            self.aliases.update(aliases)
        self._function_cache: Dict[str, Callable[..., Any]] = {}

    def _resolve_function(self, name: str) -> Callable[..., Any]:
        lookup_name = self.aliases.get(name, name)
        if lookup_name in self._function_cache:
            return self._function_cache[lookup_name]
        for module in self.modules:
            if hasattr(module, lookup_name):
                attr = getattr(module, lookup_name)
                if callable(attr):
                    self._function_cache[lookup_name] = attr
                    return attr
        raise DSLExecutionError(f"未找到算子 `{name}`，请确认是否已在允许的模块中定义。")

    def _literal_constant(self, node: ast.AST) -> Optional[Any]:
        try:
            return ast.literal_eval(node)
        except Exception:
            return None

    def _build_graph(
        self,
        node: ast.AST,
        nodes: Dict[int, Node],
        edges: Dict[int, Set[int]],
        id_gen: Iterable[int],
        cache: Dict[Tuple[str, str], int],
    ) -> int:
        literal = self._literal_constant(node)
        if literal is not None:
            cache_key = ("const", repr(literal))
            if cache_key in cache:
                return cache[cache_key]
            node_id = next(id_gen)
            nodes[node_id] = Node(node_id=node_id, kind="constant", label=str(literal), value=literal)
            cache[cache_key] = node_id
            return node_id

        if isinstance(node, ast.Name):
            cache_key = ("input", node.id)
            if cache_key in cache:
                return cache[cache_key]
            node_id = next(id_gen)
            nodes[node_id] = Node(node_id=node_id, kind="input", label=node.id)
            cache[cache_key] = node_id
            return node_id

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise DSLExecutionError("DSL 中暂不支持属性调用或复杂函数引用。")
            func_name = node.func.id
            func = self._resolve_function(func_name)
            arg_ids: List[int] = []
            for arg in node.args:
                arg_id = self._build_graph(arg, nodes, edges, id_gen, cache)
                arg_ids.append(arg_id)

            node_id = next(id_gen)
            nodes[node_id] = Node(
                node_id=node_id,
                kind="call",
                label=func_name,
                func=func,
                inputs=list(arg_ids),
            )

            for arg_id in arg_ids:
                edges[arg_id].add(node_id)
            return node_id

        raise DSLExecutionError(f"暂不支持的语法节点：{ast.dump(node)}")

    def _topological_sort(self, nodes: Dict[int, Node]) -> List[int]:
        order: List[int] = []
        visited: Set[int] = set()

        def dfs(node_id: int) -> None:
            if node_id in visited:
                return
            visited.add(node_id)
            for dep_id in nodes[node_id].inputs:
                dfs(dep_id)
            order.append(node_id)

        for node_id in list(nodes):
            dfs(node_id)
        return order

    def plan(self, expression: str) -> ExecutionPlan:
        parsed = ast.parse(expression, mode="eval")
        nodes: Dict[int, Node] = {}
        edges: Dict[int, Set[int]] = defaultdict(set)
        id_gen = count()
        cache: Dict[Tuple[str, str], int] = {}

        root_id = self._build_graph(parsed.body, nodes, edges, id_gen, cache)

        adjacency: Dict[int, List[int]] = {node_id: [] for node_id in nodes}
        for node_id, children in edges.items():
            adjacency[node_id] = sorted(children)
        topo_order = self._topological_sort(nodes)

        return ExecutionPlan(root_id=root_id, nodes=nodes, adjacency=adjacency, topo_order=topo_order)

    def execute(self, plan: ExecutionPlan, variables: Dict[str, Any]) -> Any:
        values: Dict[int, Any] = {}
        for node_id in plan.topo_order:
            node = plan.nodes[node_id]
            if node.kind == "input":
                if node.label not in variables:
                    raise DSLExecutionError(f"缺少输入变量 `{node.label}`。")
                values[node_id] = variables[node.label]
            elif node.kind == "constant":
                values[node_id] = node.value
            elif node.kind == "call":
                args = [values[arg_id] for arg_id in node.inputs]
                try:
                    values[node_id] = node.func(*args)
                except Exception as exc:
                    raise DSLExecutionError(
                        f"执行算子 `{node.label}` 时出错：{exc}"
                    ) from exc
            else:
                raise DSLExecutionError(f"未知的节点类型 `{node.kind}`。")
        return values[plan.root_id]


DEFAULT_DSL_FILE = CURRENT_DIR / "indicator_dsl.json"


def load_examples(filepath: Optional[str] = None) -> List[Tuple[str, str]]:
    if filepath is None:
        filepath = str(DEFAULT_DSL_FILE)
    with open(filepath, "r", encoding="utf-8") as f:
        payload = json.load(f)
    examples = []
    for item in payload.get("indicators", []):
        examples.append((item["name"], item["dsl_expression"]))
    return examples


def demo() -> None:
    executor = DSLExecutor()
    examples = load_examples()

    returns = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02], dtype=np.float64)
    context = {
        "returns": returns,
        "ddof": 1,
        "periods_per_year": 252.0,
        "risk_free_rate_per_period": 0.0001,
    }

    for name, expression in examples:
        plan = executor.plan(expression)
        result = executor.execute(plan, context)
        print(f"{name}: {result}")
        print(f"  拓扑顺序: {[plan.nodes[i].label for i in plan.topo_order]}")


if __name__ == "__main__":
    demo()
