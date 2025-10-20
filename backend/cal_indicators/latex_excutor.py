"""
解析指标 LaTeX 表达式并构建可复用的 DAG 拓扑。

支持多版本指标配置，提供节点、周期与拓扑序列，供运行时执行。
"""
import ast
import json
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


class LatexParseError(Exception):
    """指示 LaTeX 表达式解析失败。"""


class DAGBuildError(Exception):
    """指示 DAG 构建过程中发生的错误。"""


def _extract_braced(expr: str, start: int) -> Tuple[str, int]:
    if start >= len(expr) or expr[start] != "{":
        raise LatexParseError("预期在位置 %d 处出现 '{'" % start)
    depth = 0
    for index in range(start, len(expr)):
        char = expr[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return expr[start + 1: index], index + 1
    raise LatexParseError("未找到与位置 %d 匹配的 '}'" % start)


def _replace_frac(expr: str) -> str:
    token = "\\frac"
    while token in expr:
        idx = expr.index(token)
        numerator, after_num = _extract_braced(expr, idx + len(token))
        denominator, after_den = _extract_braced(expr, after_num)
        replacement = f"({numerator})/({denominator})"
        expr = expr[:idx] + replacement + expr[after_den:]
    return expr


def _replace_sqrt(expr: str) -> str:
    token = "\\sqrt"
    while token in expr:
        idx = expr.index(token)
        inside, after_inside = _extract_braced(expr, idx + len(token))
        replacement = f"sqrt({inside})"
        expr = expr[:idx] + replacement + expr[after_inside:]
    return expr


class LatexExpressionParser:
    """将受限的 LaTeX 表达式转换为 Python 表达式字符串，再解析为 AST。"""

    def __init__(self) -> None:
        self.variable_map = {
            "\\mathbf{r}": "returns",
            "\\mathbf{\\ell}": "log_returns",
            "r_{f}": "risk_free_rate_per_period",
        }
        self.special_tokens = [
            ("\\overline{\\mathbf{r}}", "sequence_mean(returns)"),
            ("\\overline{\\mathbf{\\ell}}", "sequence_mean(log_returns)"),
            ("\\operatorname{std}", "sequence_std"),
            ("\\operatorname{cumret}", "cumulative_simple_returns"),
            ("\\operatorname{net}", "cumulative_net_value"),
            ("\\operatorname{drawdown}", "sequence_drawdown"),
            ("\\operatorname{gmean}", "sequence_geometric_mean"),
            ("\\operatorname{CAGR}", "sequence_compound_annual_growth_rate"),
            ("\\operatorname{MDD}", "max_drawdown_rate_and_recovery_days"),
        ]

    def to_python(self, latex: str) -> str:
        expr = latex.strip()
        expr = expr.replace("\\left", "").replace("\\right", "")
        expr = expr.replace("\\,", "").replace("\\ ", "")
        expr = _replace_frac(expr)
        expr = _replace_sqrt(expr)

        for token, replacement in self.special_tokens:
            expr = expr.replace(token, replacement)

        expr = expr.replace("\\prod", "sequence_prod")
        expr = expr.replace("\\sum", "sequence_sum")
        expr = expr.replace("\\times", "*")
        expr = expr.replace("\\cdot", "*")

        for token, name in self.variable_map.items():
            expr = expr.replace(token, name)

        expr = expr.replace("{", "(").replace("}", ")")
        expr = expr.replace("\\", "")

        return expr

    def parse(self, latex: str) -> ast.AST:
        python_expr = self.to_python(latex)
        try:
            parsed = ast.parse(python_expr, mode="eval")
        except SyntaxError as exc:  # pragma: no cover
            raise LatexParseError(f"无法解析表达式: {latex}") from exc
        return parsed.body


@dataclass
class DAGNode:
    node_id: int
    period: str
    kind: str
    label: str
    inputs: List[int] = field(default_factory=list)
    raw: str = ""


class DAGBuilder:
    def __init__(self) -> None:
        self.nodes: Dict[int, DAGNode] = {}
        self.adjacency: Dict[int, List[int]] = {}
        self._id_gen = count()
        self._cache: Dict[Tuple[str, str], int] = {}

    def _record_edge(self, child: int, parent: int) -> None:
        self.adjacency.setdefault(child, [])
        if parent not in self.adjacency[child]:
            self.adjacency[child].append(parent)

    def _next_id(self) -> int:
        return next(self._id_gen)

    def add_expression(self, period: str, node: ast.AST) -> int:
        return self._build(node, period)

    def _cache_key(self, period: str, node: ast.AST) -> Tuple[str, str]:
        return period, ast.dump(node)

    def _build(self, node: ast.AST, period: str) -> int:
        key = self._cache_key(period, node)
        if key in self._cache:
            return self._cache[key]

        if isinstance(node, ast.Constant):
            node_id = self._next_id()
            self.nodes[node_id] = DAGNode(
                node_id=node_id,
                period="__const__",
                kind="constant",
                label=str(node.value),
                raw=str(node.value),
            )
            self._cache[key] = node_id
            return node_id

        if isinstance(node, ast.Name):
            node_id = self._next_id()
            self.nodes[node_id] = DAGNode(
                node_id=node_id,
                period=period,
                kind="variable",
                label=node.id,
                raw=node.id,
            )
            self._cache[key] = node_id
            return node_id

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand_id = self._build(node.operand, period)
            node_id = self._next_id()
            self.nodes[node_id] = DAGNode(
                node_id=node_id,
                period=period,
                kind="unary",
                label="negate",
                inputs=[operand_id],
                raw=ast.dump(node),
            )
            self._record_edge(operand_id, node_id)
            self._cache[key] = node_id
            return node_id

        if isinstance(node, ast.BinOp):
            left_id = self._build(node.left, period)
            right_id = self._build(node.right, period)
            op_map = {
                ast.Add: "add",
                ast.Sub: "subtract",
                ast.Mult: "multiply",
                ast.Div: "divide",
                ast.Pow: "power",
            }
            op_type = type(node.op)
            if op_type not in op_map:
                raise DAGBuildError(f"不支持的二元运算符: {op_type}")
            label = op_map[op_type]
            node_id = self._next_id()
            self.nodes[node_id] = DAGNode(
                node_id=node_id,
                period=period,
                kind="binary",
                label=label,
                inputs=[left_id, right_id],
                raw=ast.dump(node),
            )
            self._record_edge(left_id, node_id)
            self._record_edge(right_id, node_id)
            self._cache[key] = node_id
            return node_id

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            arg_ids = [self._build(arg, period) for arg in node.args]
            node_id = self._next_id()
            self.nodes[node_id] = DAGNode(
                node_id=node_id,
                period=period,
                kind="call",
                label=func_name,
                inputs=list(arg_ids),
                raw=ast.dump(node),
            )
            for child in arg_ids:
                self._record_edge(child, node_id)
            self._cache[key] = node_id
            return node_id

        raise DAGBuildError(f"不支持的 AST 节点: {ast.dump(node)}")

    def to_networkx(self) -> "nx.DiGraph":  # type: ignore[type-arg]
        if nx is None:
            raise RuntimeError("networkx 未安装，无法导出 DAG。")
        graph = nx.DiGraph()
        for node in self.nodes.values():
            graph.add_node(
                node.node_id,
                label=node.label,
                kind=node.kind,
                period=node.period,
            )
        for child, parents in self.adjacency.items():
            for parent in parents:
                graph.add_edge(child, parent)
        return graph


@dataclass
class IndicatorDescriptor:
    name: str
    expression: str
    periods: List[str]
    notes: str = ""


class LatexExecutor:
    def __init__(
            self,
            indicator_file: Optional[Path] = None,
            variable_file: Optional[Path] = None,
            version: Optional[str] = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent
        self.indicator_path = indicator_file or (base_dir / "indicator_latex.json")
        self.variable_path = variable_file or (base_dir / "variable_latex.json")
        self.version = version
        self.available_versions: List[str] = []
        self.parser = LatexExpressionParser()
        self.builder: DAGBuilder = DAGBuilder()
        self.metadata: Dict[str, object] = {}
        self.indicators: List[IndicatorDescriptor] = []
        self.variables: Dict[str, Dict[str, object]] = {}
        self.roots: Dict[str, Dict[str, int]] = {}
        self._load()

    def _load(self) -> None:
        with self.indicator_path.open("r", encoding="utf-8") as f:
            raw_payload = json.load(f)

        if not isinstance(raw_payload, dict):
            raise ValueError("indicator_latex.json 必须是 dict 格式。")

        if "metadata" in raw_payload and "indicators" in raw_payload:
            profiles = {"__default__": raw_payload}
        else:
            profiles = raw_payload

        if not profiles:
            raise ValueError("indicator_latex.json 未包含任何指标定义。")

        self.available_versions = list(profiles.keys())

        if self.version is None:
            self.version = self.available_versions[0]
        elif self.version not in profiles:
            raise ValueError(f"未找到指定的指标版本: {self.version}")

        selected_profile = profiles[self.version]
        self.metadata = selected_profile.get("metadata", {})
        self.indicators = [
            IndicatorDescriptor(
                name=item["name"],
                expression=item["dsl_expression"],
                periods=item.get("periods", []),
                notes=item.get("notes", ""),
            )
            for item in selected_profile.get("indicators", [])
        ]

        if self.variable_path.exists():
            with self.variable_path.open("r", encoding="utf-8") as f:
                var_payload = json.load(f)
            for entry in var_payload.get("variables", []):
                self.variables[entry["name"]] = entry

    def build(self) -> None:
        self.builder = DAGBuilder()
        self.roots = {}
        for descriptor in self.indicators:
            try:
                ast_root = self.parser.parse(descriptor.expression)
            except LatexParseError as exc:
                raise LatexParseError(f"指标 {descriptor.name} 解析失败: {exc}") from exc
            for period in descriptor.periods or ["__default__"]:
                root_id = self.builder.add_expression(period, ast_root)
                self.roots.setdefault(period, {})
                self.roots[period][descriptor.name] = root_id

    def export_period_dot(self, period: str, output_path: Path) -> None:
        if nx is None:
            raise RuntimeError("networkx 未安装，无法导出 DOT。")
        graph = self.builder.to_networkx()
        if period not in self.roots:
            raise ValueError(f"未发现周期 {period} 的指标。")
        selected_nodes: Set[int] = set()
        for root_id in self.roots[period].values():
            selected_nodes.add(root_id)
            selected_nodes.update(nx.ancestors(graph, root_id))
        subgraph = graph.subgraph(selected_nodes)
        try:
            from networkx.drawing.nx_pydot import write_dot
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("缺少 pydot，无法写入 DOT 文件。") from exc
        write_dot(subgraph, str(output_path))

    def topo_order_for_period(self, period: str) -> List[DAGNode]:
        if nx is None:
            raise RuntimeError("networkx 未安装，无法执行拓扑排序。")
        graph = self.builder.to_networkx()
        if period not in self.roots:
            raise ValueError(f"未发现周期 {period} 的指标。")
        selected: Set[int] = set()
        for root_id in self.roots[period].values():
            selected.add(root_id)
            selected.update(nx.ancestors(graph, root_id))
        subgraph = graph.subgraph(selected)
        order: List[DAGNode] = []
        for node_id in nx.topological_sort(subgraph):
            order.append(self.builder.nodes[node_id])
        return order


def main(version: str) -> None:
    """构建并展示指定版本的指标 DAG 信息。"""
    if not version:
        raise ValueError("必须显式传入指标版本名称，例如 main('指标计算示例模板')")
    executor = LatexExecutor(version=version)
    executor.build()
    print("可用指标版本:", executor.available_versions)
    print("当前使用版本:", executor.version)
    print("公共参数:", json.dumps(executor.metadata, ensure_ascii=False, indent=2))
    print("变量列表:", list(executor.variables.keys()))
    for period, mapping in executor.roots.items():
        print(f"\n周期 {period} 包含指标: {list(mapping.keys())}")
        try:
            order = executor.topo_order_for_period(period)
            readable = " -> ".join(f"{node.label}[{node.kind}]" for node in order)
            print(f"拓扑顺序: {readable}")
        except RuntimeError:
            print("缺少 networkx，无法计算拓扑顺序。")


if __name__ == "__main__":
    raise RuntimeError("请从代码中调用 main('指标版本名称')，而不是直接运行该脚本。")
