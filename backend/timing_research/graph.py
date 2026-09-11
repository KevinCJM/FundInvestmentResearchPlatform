"""Typed authoring and fixed-plan orchestration; no Python numerical operators."""
from __future__ import annotations

import ast
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import threading
from collections import OrderedDict
import numpy as np

from cal_indicators.typed_dsl import compose_typed_expression, compose_typed_series_bundle, ValueType, TypedDslError
from cal_indicators.typed_numba_plan import compile_numba_series_plan, NumbaPlanCompileError
from computation_graph.series_operators import SERIES_OPERATOR_LABELS
from computation_graph.causal_series import (
    bind_series_context, causal_violations, outside_value_nodes, series_variable_types,
)
from cal_indicators.typed_operators import TYPED_DSL_VERSION, TYPED_OPERATOR_REGISTRY_VERSION
from custom_indicators.errors import ValidationError
from historical_regimes.condition_numba import condition_compare_kernel, condition_logic_kernel, COMPARISON_OPCODES, CONDITION_KERNELS
from historical_regimes.v2_numba import unary_transform_kernel
from .catalog import expressions
from .contracts import Definition
from . import numeric
from . import panel
from .learning import priority_quota_kernel, learning_execution_audit

MAX_BARS = 12_000
MAX_RUNTIME_COST = 200_000_000
MAX_WORKSPACE_BYTES = 64 * 1024 * 1024


def definition_hash(definition: Definition) -> str:
    raw = json.dumps(definition.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(("timing-graph/1|" + raw).encode()).hexdigest()


def fail(message, field=None):
    raise ValidationError("TIMING_GRAPH_INVALID", message, field)


def _parameters(step, meta):
    declared = {item["name"]: item for item in meta["parameters"]}
    if set(step.parameters) - set(declared):
        fail(f"{step.label} 包含未支持的参数。", step.id)
    for name, spec in declared.items():
        value = step.parameters.get(name, spec["default"])
        if spec["type"] in {"integer", "number"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                fail(f"{step.label}：{spec['label']}必须为有限数值。", step.id)
            if spec["type"] == "integer" and int(value) != value:
                fail(f"{step.label}：{spec['label']}必须为整数。", step.id)
            if "minimum" in spec and value < spec["minimum"] or "maximum" in spec and value > spec["maximum"]:
                fail(f"{step.label}：{spec['label']}超出允许范围。", step.id)
        elif not isinstance(value, str) or len(value) > 1000:
            fail(f"{step.label}：文本参数无效或过长。", step.id)
        if spec.get("options") and value not in {x["value"] for x in spec["options"]}:
            fail(f"{step.label}：{spec['label']}请选择目录中的选项。", step.id)


def _reference_protocol(definition):
    item = definition if definition and definition.get('result_kind') == 'time_series' else {}
    return dict(dsl_version=item.get('dsl_version', TYPED_DSL_VERSION),
                operator_registry_version=item.get('operator_registry_version', TYPED_OPERATOR_REGISTRY_VERSION))


def _formula_plan(expression, inputs, *, definition=None):
    maximum = 4000 if definition is not None else 1000
    if len(expression) > maximum:
        fail(f"单个公式最多 {maximum} 字符，请拆分步骤。")
    # Window/lag parameters must be fixed positive constants, never future offsets.
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
                "lag", "difference", "rolling_mean", "rolling_std", "rolling_min", "rolling_max", "rolling_window", "rolling_apply", "recursive_smooth",
            }:
                if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                    fail("窗口与滞后期必须填写固定正整数。")
                value = node.args[1].value
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or int(value) != value or not 1 <= value <= 5000:
                    fail("窗口与滞后期必须在 1～5000 之间，不能引用未来数据。")
        plan = compose_typed_expression(expression, variable_types=series_variable_types(inputs, definition),
                                        output_contract="series", max_nodes=128, max_depth=20, **_reference_protocol(definition))
    except (SyntaxError, TypedDslError) as exc:
        fail(getattr(exc, "message", "公式语法无效，请检查输入和括号。"))
    allowed = set(SERIES_OPERATOR_LABELS) | {"divide_or_default", "power", "reciprocal", "sign"}
    violations = causal_violations(plan, allowed)
    if violations:
        fail(f"算子或上下文 {violations[0]} 未被合法滚动窗口约束，不能把全样本结果用于历史信号。")
    if any(node.operator_id in {'lag', 'difference'} for node in outside_value_nodes(plan)):
        fail("为保留完整交易日轴，请添加独立的滞后或差分步骤，再在公式中引用其输出。")
    if plan.output_type.dtype != "float64" or len(plan.output_type.shape) != 1:
        fail("数值计算必须输出一维浮点序列，条件请使用比较步骤。")
    return plan


@dataclass(frozen=True)
class PreparedGraph:
    definition: Definition
    digest: str
    order: tuple[str, ...]
    ports: dict[str, str]
    formulas: dict[str, object]
    metadata: dict[str, dict]
    cost_per_bar: int
    workspace_arrays: int
    panel_groups: dict[str, str]


class GraphRuntime:
    def __init__(self, catalog):
        self.catalog = catalog
        self._plans: OrderedDict[str, PreparedGraph] = OrderedDict()
        self._lock = threading.RLock()

    def prepare(self, definition: Definition) -> PreparedGraph:
        digest = definition_hash(definition)
        with self._lock:
            if digest in self._plans:
                return self._plans[digest]
            prepared = self._prepare(definition.model_copy(deep=True), digest)
            self._plans[digest] = prepared
            while len(self._plans) > 128:
                self._plans.popitem(last=False)
            return prepared

    def _prepare(self, definition, digest):
        nodes = {node.id: node for node in definition.nodes}
        if len(nodes) != len(definition.nodes):
            fail("计算步骤 ID 不能重复。")
        ports, metadata, formulas = {}, {}, {}
        for step in definition.nodes:
            if step.op not in self.catalog:
                fail(f"{step.label} 的算子不可用，请重新选择。", step.id)
            meta = copy.deepcopy(self.catalog[step.op])
            metadata[step.id] = meta
            _parameters(step, meta)
            expected = {p["name"]: p for p in meta["inputs"]}
            if set(step.inputs) - set(expected):
                fail(f"{step.label} 包含未知输入。", step.id)
            for name, port in expected.items():
                if port["required"] and not step.inputs.get(name):
                    fail(f"{step.label} 尚未连接 {port['label']}。", step.id)
            for port in meta["outputs"]:
                ports[f"{step.id}.{port['name']}"] = port["type"]
        for step in definition.nodes:
            expected = {p["name"]: p for p in metadata[step.id]["inputs"]}
            for name, reference in step.inputs.items():
                if reference not in ports or ports[reference] != expected[name]["type"]:
                    fail(f"{step.label}：{name} 的连接不存在或类型不匹配。", step.id)
        for label, ref in (("入场", definition.entry), ("退出", definition.exit)):
            if ref is not None and ports.get(ref) != "condition":
                fail(f"请选择一个条件输出作为{label}规则。", label)
        order, visiting, visited = [], set(), set()

        def visit(identifier):
            if identifier in visiting:
                fail("步骤之间存在循环连接，请检查输入来源。")
            if identifier in visited:
                return
            visiting.add(identifier)
            for ref in nodes[identifier].inputs.values():
                visit(ref.split(".")[0])
            visiting.remove(identifier)
            visited.add(identifier)
            order.append(identifier)

        # Validate every node, including a disconnected draft branch.
        for identifier in nodes:
            visit(identifier)
        panel_groups = {}
        for identifier in order:
            step = nodes[identifier]
            groups = {panel_groups[ref] for ref in step.inputs.values() if ref in panel_groups}
            if len(groups) > 1:
                fail("不同成员篮子不能按位置直接连线运算；请先分别聚合为时间序列。", identifier)
            if step.op == "basket_source":
                groups = {str(step.parameters.get("group", "market"))}
            for output in metadata[identifier]["outputs"]:
                if output["type"] in {"panel", "condition_panel"}:
                    panel_groups[f"{identifier}.{output['name']}"] = next(iter(groups))
        used, requested = set(), set()

        def mark(ref):
            requested.add(ref)
            identifier = ref.split(".")[0]
            if identifier not in used:
                used.add(identifier)
                for upstream in nodes[identifier].inputs.values():
                    mark(upstream)

        mark(definition.entry)
        if definition.exit:
            mark(definition.exit)
        if definition.training:
            for ref in [*definition.training.state_refs, *(a.entry for a in definition.training.actions if a.entry)]:
                if ports.get(ref) != "condition":
                    fail("训练候选与市场状态必须连接现有条件输出。", ref)
                mark(ref)
        order = tuple(x for x in order if x in used)
        internal_nodes, cost_per_bar, workspace_arrays = 0, len(order) * 20, len(order) * 3
        for identifier, step in nodes.items():
            step_expressions = expressions(step, metadata[identifier])
            indicator_definition = metadata[identifier].get('_reference', {}).get('_indicator_definition')
            # Validate disconnected draft formulas too, but execute only actual
            # dependencies and share one multi-output plan per connected step.
            for output, expression in step_expressions.items():
                plan = _formula_plan(expression, step.inputs, definition=indicator_definition)
                internal_nodes += len(plan.nodes)
                if internal_nodes > 512:
                    fail("总计算预算超过 512 个数学步骤，请简化公式。")
            wanted = {output: expression for output, expression in step_expressions.items()
                      if f"{identifier}.{output}" in requested}
            if wanted:
                try:
                    bundle = compose_typed_series_bundle(wanted, variable_types=series_variable_types(step.inputs, indicator_definition),
                        max_nodes=128, max_depth=20, **_reference_protocol(indicator_definition))
                    bundled_nodes = {node.node_id: node for node in bundle.nodes}
                    for node in bundle.nodes:
                        cost_per_bar += 1
                        workspace_arrays += int(node.inferred_type.rank > 0)
                        if node.operator_id == 'rolling_apply':
                            from cal_indicators.rolling_scope import dependency_nodes
                            width = int(float(bundled_nodes[node.inputs[1]].label))
                            cost_per_bar += width * len(dependency_nodes(bundle.nodes, node.inputs[0]))
                        if node.operator_id in {"min_value", "max_value"} and node.inputs:
                            window = bundled_nodes[node.inputs[0]]
                            if window.operator_id == "rolling_window":
                                # Shared rolling extrema scan a window per row;
                                # account for that work before numerical execution.
                                width = bundled_nodes[window.inputs[1]]
                                cost_per_bar += int(float(width.label))
                    compiled = compile_numba_series_plan(bundle)
                except (TypedDslError, NumbaPlanCompileError) as exc:
                    fail(f"{step.label} 无法准备为固定签名数值计划：{getattr(exc, 'message', str(exc))}", step.id)
                for output in wanted:
                    formulas[f"{identifier}.{output}"] = compiled
        return PreparedGraph(definition, digest, order, ports, formulas, metadata, cost_per_bar, workspace_arrays, panel_groups)

    def get(self, definition, token):
        digest = definition_hash(definition)
        with self._lock:
            prepared = self._plans.get(digest)
        if token != digest or prepared is None:
            raise ValidationError("TIMING_PREPARE_REQUIRED", "算法已变化或服务已重启，请重新准备后运行。")
        return prepared

    @staticmethod
    def evaluate(prepared: PreparedGraph, bars, baskets=None):
        if definition_hash(prepared.definition) != prepared.digest:
            fail("已准备算法定义发生变化，请重新准备。")
        if not isinstance(bars.dates, np.ndarray) or bars.dates.dtype != np.int64 or bars.dates.ndim != 1 or not 0 < bars.dates.size <= MAX_BARS:
            fail("研究日期轴需要 1～12000 条 int64 日频观察。")
        panel_width = max((value.shape[0] for value in (baskets or {}).values()), default=1)
        panel_arrays = sum(port["type"] in {"panel", "condition_panel"} for meta in prepared.metadata.values() for port in meta["outputs"]) * 3
        workspace_arrays = prepared.workspace_arrays + panel_arrays * (panel_width - 1)
        if prepared.cost_per_bar * bars.dates.size * panel_width > MAX_RUNTIME_COST or workspace_arrays * bars.dates.size * 8 > MAX_WORKSPACE_BYTES:
            fail("该算法超过单次计算或内存预算，请减少滚动窗口、计算步骤或研究区间。")
        values = {}
        steps = {node.id: node for node in prepared.definition.nodes}
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int64)
        empty_f.setflags(write=False)
        empty_i.setflags(write=False)
        month_ids = None
        date_context = None
        for identifier in prepared.order:
            step = steps[identifier]
            metadata = prepared.metadata[identifier]
            params = {p["name"]: step.parameters.get(p["name"], p["default"]) for p in metadata["parameters"]}
            inputs = {name: values[ref] for name, ref in step.inputs.items()}
            outputs = {}
            if step.op == "source":
                source = getattr(bars, params["field"])
                if not isinstance(source, np.ndarray):
                    fail(f"{step.label} 需要标准数值数组。")
                # A header-only view avoids changing the owner's writeable flag.
                outputs["value"] = source.view()
            elif step.op == "basket_source":
                group = params["group"]
                if not baskets or group not in baskets:
                    fail(f"请先配置{ '市场' if group == 'market' else '资产类别'} ETF 环境篮子（至少 2 只）。")
                outputs["value"] = baskets[group].view()
            elif step.op == "panel_lag":
                outputs["value"] = panel.panel_lag_kernel(inputs["values"], np.int64(params["periods"]))
            elif step.op == "panel_compare":
                outputs["value"] = panel.panel_compare_kernel(inputs["left"], inputs.get("right", np.empty((0, 0), dtype=np.float64)), float(params["threshold"]), np.int64(COMPARISON_OPCODES[params["operator"]]))
            elif step.op in {"cross_mean", "breadth"}:
                kernel = panel.cross_mean_kernel if step.op == "cross_mean" else panel.breadth_kernel
                outputs["value"] = kernel(inputs["value"])
            elif metadata.get("_aligned_temporal"):
                outputs["value"] = unary_transform_kernel(inputs["values"], np.int64(4 if metadata["_aligned_temporal"] == "lag" else 2), np.int64(params["periods"]))
            elif step.op == "compare":
                outputs["value"] = condition_compare_kernel(inputs["left"], inputs.get("right", empty_f), float(params["threshold"]), np.int64(COMPARISON_OPCODES[params["operator"]]))
            elif step.op in {"all", "any", "not"}:
                outputs["value"] = condition_logic_kernel(inputs.get("left", inputs.get("value")), inputs.get("right", empty_i), np.int64({"all": 0, "any": 1, "not": 2}[step.op]))
            elif step.op in {"first", "confirm", "cooldown"}:
                outputs["value"] = numeric.condition_event_kernel(inputs["value"], np.int64({"first": 0, "confirm": 1, "cooldown": 2}[step.op]), np.int64(params.get("window", 1)))
            elif step.op == "condition_value":
                outputs["value"] = numeric.condition_values_kernel(inputs["value"])
            elif step.op == "alpha_beta":
                outputs = dict(zip(("level", "slope", "innovation"), numeric.alpha_beta_kernel(inputs["value"], float(params["alpha"]), float(params["beta"]))))
            elif step.op == "priority_quota":
                if month_ids is None:
                    month_ids = np.asarray([int(str(np.datetime64(int(day), "D"))[:7].replace("-", "")) for day in bars.dates], dtype=np.int64)
                    month_ids.setflags(write=False)
                outputs["value"] = priority_quota_kernel(inputs["core"], inputs["supplement"], month_ids, np.int64(params["max_core_before_supp"]), np.int64(params["max_supp_per_month"]), np.int64(params["cooldown"]))
            else:
                compiled = next((prepared.formulas[f"{identifier}.{port['name']}"] for port in metadata["outputs"]
                                 if f"{identifier}.{port['name']}" in prepared.formulas), None)
                if compiled is None:
                    fail(f"{step.label} 未找到已准备的数值计划。")
                if any(not array.flags.c_contiguous for array in inputs.values()):
                    fail(f"{step.label} 需要在数据读取边界规范化为连续数组；计算节点不会复制行情。")
                if date_context is None:
                    date_context = bars.dates.astype(np.float64)
                    date_context.setflags(write=False)
                indicator_definition = metadata.get('_reference', {}).get('_indicator_definition')
                try:
                    if step.op == "panel_formula":
                        shape = next(iter(inputs.values())).shape
                        if any(array.shape != shape for array in inputs.values()):
                            fail("篮子公式的成员和日期轴必须完全一致。", step.id)
                        output = np.empty(shape, dtype=np.float64)
                        # At most 12 member plans; numerical windows stay in NJIT.
                        for member in range(output.shape[0]):
                            arguments = bind_series_context(compiled.context_names,
                                {name: array[member] for name, array in inputs.items()}, date_context, indicator_definition)
                            output[member] = compiled.compute(arguments)[0]
                        outputs = {"value": output}
                    else:
                        arguments = bind_series_context(compiled.context_names, inputs, date_context, indicator_definition)
                        outputs = dict(zip(compiled.channel_names, compiled.compute(arguments)))
                except (TypeError, ValueError, ZeroDivisionError, FloatingPointError, OverflowError) as exc:
                    fail(f"{step.label} 计算失败，请检查窗口、除数或数据范围：{str(exc)}", step.id)
            for name, array in outputs.items():
                kind = prepared.ports[f"{identifier}.{name}"]
                expected_dtype = np.dtype("float64" if kind in {"series", "panel"} else "int64")
                is_panel = kind in {"panel", "condition_panel"}
                if not isinstance(array, np.ndarray) or array.dtype != expected_dtype or array.ndim != (2 if is_panel else 1) or array.shape[-1] != bars.dates.size or (is_panel and not 2 <= array.shape[0] <= 12):
                    fail(f"{step.label} 的输出类型或日期轴不符合契约。")
                if kind == "series" and numeric.series_status_kernel(array) != 0:
                    fail(f"{step.label} 产生无穷值，请检查除数和公式。")
                if kind == "panel" and any(numeric.series_status_kernel(row) != 0 for row in array):
                    fail(f"{step.label} 产生无穷值，请检查篮子公式。")
                array.setflags(write=False)
                values[f"{identifier}.{name}"] = array
        return values

    @staticmethod
    def audit(prepared):
        audit = numeric.timing_execution_audit()
        signatures = dict(audit["kernel_signatures"])
        for key, plan in prepared.formulas.items():
            signatures[f"formula:{key}"] = list(plan.compiled_signatures)
        for key, kernel in CONDITION_KERNELS.items():
            signatures[key] = [str(s) for s in kernel.signatures]
        signatures["aligned_temporal"] = [str(s) for s in unary_transform_kernel.signatures]
        signatures.update(panel.warm_panel_kernels()["kernel_signatures"])
        signatures.update(learning_execution_audit()["kernel_signatures"])
        return {**audit, "kernel_signatures": signatures, "definition_hash": prepared.digest,
                "input_contract": "readonly_float64_int64_arrays", "source_node_copies": 0,
                "cost_per_observation": prepared.cost_per_bar, "workspace_arrays_upper_bound": prepared.workspace_arrays,
                "limits": {"bars": MAX_BARS, "runtime_cost": MAX_RUNTIME_COST, "workspace_bytes": MAX_WORKSPACE_BYTES}}
