"""Authoring metadata derives mathematics from the shared Indicator registry."""
from __future__ import annotations

import copy
import ast
from types import SimpleNamespace
from computation_graph.series_operators import series_operator_specs, SERIES_OPERATOR_LABELS, PARAMETER_LABELS
from historical_regimes.indicator_nodes import register_indicator_nodes, typed_node_expressions
from custom_indicators.variable_registry import get_variable
from custom_indicators.errors import ValidationError
from computation_graph.causal_series import SYSTEM_CONTEXT_NAMES
from cal_indicators.typed_dsl import TypedDslError
from .contracts import Definition


def port(name, kind="series", required=True, label=None):
    return {"name": name, "label": label or name, "type": kind, "required": required}


def parameter(name, label, default, minimum=None, maximum=None, options=None):
    item = {"name": name, "label": label, "type": "string" if isinstance(default, str) else "integer" if isinstance(default, int) else "number", "default": default}
    for key, value in (("minimum", minimum), ("maximum", maximum), ("options", options)):
        if value is not None:
            item[key] = value
    return item


def operator(identifier, label, inputs, parameters=(), output="series", category="condition", description="", outputs=None, granularity="primitive"):
    return {"id": identifier, "label": label, "description": description, "category": category,
            "granularity": granularity, "inputs": list(inputs), "outputs": outputs or [port("value", output)],
            "parameters": list(parameters), "causal": True,
            "missing_policy": "preserve_unknown" if output == "condition" else "preserve_nan",
            "knowledge_time": "current_or_prior_completed_session", "axis_contract": "same_product_same_trading_axis"}


def _timing_reference(item):
    """Keep the saved revision and expression binding; check timing causality.

    The legacy regime validator may reject newer rolling-window lowering. Its
    version identity is reused, while this domain checks its own allowed DAG.
    """
    from .graph import _formula_plan
    candidate = copy.deepcopy(item)
    definition = candidate.get("_indicator_definition", {})
    if (definition.get("result_kind", "scalar") not in {"scalar", "time_series"}
            or definition.get("context_kind", "single_product") != "single_product"
            or not str(definition.get("dsl_version", "")).startswith("2.")):
        return None
    candidate["available"] = True
    try:
        node = SimpleNamespace(type=candidate["id"], parameters={}, inputs={})
        bound = typed_node_expressions(node, {candidate["id"]: candidate})
        inputs = {}
        for expression in bound.values():
            tree = ast.parse(expression, mode="eval")
            calls = {id(item.func) for item in ast.walk(tree) if isinstance(item, ast.Call)}
            for symbol in (item.id for item in ast.walk(tree) if isinstance(item, ast.Name) and id(item) not in calls):
                if symbol in SYSTEM_CONTEXT_NAMES:
                    continue
                variable = get_variable(symbol)
                if variable is None or variable.kind != "series" or variable.axes not in {("T",), ("time",)}:
                    return None
                inputs[symbol] = port(symbol, label=variable.label)
        if not 1 <= len(inputs) <= 4:
            return None
        for expression in bound.values():
            _formula_plan(expression, inputs, definition=definition)
        candidate["inputs"] = list(inputs.values())
        return candidate
    except (ValueError, TypeError, KeyError, SyntaxError, ValidationError, TypedDslError):
        return None


def build_catalog(indicator_service=None):
    fields = [{"value": field, "label": label} for field, label in
              (("close", "收盘价"), ("open", "开盘价"), ("high", "最高价"), ("low", "最低价"), ("volume", "成交量"))]
    registry = {
        "source": operator("source", "产品行情", [], [parameter("field", "数据字段", "close", options=fields)], category="source", description="引用同一产品的只读行情数组；成交量不做价格复权。"),
        "formula": operator("formula", "自定义数值公式", [port("a"), port("b", required=False), port("c", required=False), port("d", required=False)],
                            [parameter("expression", "公式（输入使用 a / b / c / d）", "a")], category="math", description="复用指标中心因果数值算子，禁止全样本统计和未来数据。"),
        "compare": operator("compare", "数值比较", [port("left"), port("right", required=False)],
                            [parameter("operator", "比较方式", "gt", options=[{"value": x, "label": y} for x, y in (("gt", ">"), ("ge", "≥"), ("lt", "<"), ("le", "≤"), ("eq", "="), ("ne", "≠"))]), parameter("threshold", "阈值（未连接右侧时）", 0.0)], output="condition", description="任何输入缺失时保留未知，不当作不满足。"),
        "all": operator("all", "同时满足", [port("left", "condition"), port("right", "condition")], output="condition"),
        "any": operator("any", "至少满足一个", [port("left", "condition"), port("right", "condition")], output="condition"),
        "not": operator("not", "条件取反", [port("value", "condition")], output="condition"),
        "condition_value": operator("condition_value", "条件数值化", [port("value", "condition")], category="transform", description="满足=1，不满足=0，未知=空；用于统计条件比例。"),
        "alpha_beta": operator("alpha_beta", "Alpha-Beta 低滞后滤波", [port("value")], [parameter("alpha", "水平修正", 0.4, 0.001, 1), parameter("beta", "斜率修正", 0.08, 0.001, 1)], category="filter", outputs=[port("level", label="平滑水平"), port("slope", label="趋势斜率"), port("innovation", label="创新值")], granularity="coupled", description="共享水平/斜率递推状态；缺失重置，首点斜率初始化为 0，创新值从第二个有效点开始输出。"),
    }
    registry.update({
        "basket_source": operator("basket_source", "ETF 环境篮子", [], [parameter("group", "篮子用途", "market", options=[{"value": "market", "label": "市场环境"}, {"value": "category", "label": "资产类别"}])], output="panel", category="source", description="显式选择的固定 ETF 成员 × 交易日矩阵；缺失不删日期，也不改变成员权重。"),
        "panel_formula": operator("panel_formula", "篮子逐成员计算", [port("a", "panel"), port("b", "panel", required=False)], [parameter("expression", "逐成员公式", "rolling_mean(a,250,250)")], output="panel", category="math", description="对每个成员复用指标中心同一个已准备数值计划；成员输入为零拷贝行视图。"),
        "panel_lag": operator("panel_lag", "篮子同轴滞后", [port("values", "panel")], [parameter("periods", "滞后交易日", 1, 1, 2500)], output="panel", category="math", description="复用同轴滞后内核，首端保留空值。"),
        "panel_compare": operator("panel_compare", "篮子逐成员比较", [port("left", "panel"), port("right", "panel", required=False)], registry["compare"]["parameters"], output="condition_panel", category="condition", description="复用三值比较，缺失成员条件为未知。"),
        "cross_mean": operator("cross_mean", "篮子等权均值", [port("value", "panel")], category="math", description="复用跨轴均值内核；所有固定成员有效才输出，不以缺失样本重新加权。"),
        "breadth": operator("breadth", "篮子条件占比", [port("value", "condition_panel")], category="transform", description="当日满足条件的成员比例；任何成员未知则结果为空。不是单只 ETF 的历史时间比例。"),
        "priority_quota": operator("priority_quota", "主信号优先与有限补位", [port("core", "condition"), port("supplement", "condition")], [parameter("max_core_before_supp", "当月主信号低于此数才补位", 3, 1, 100), parameter("max_supp_per_month", "每月最多补位", 2, 0, 100), parameter("cooldown", "补位冷却交易日", 5, 0, 2500)], output="condition", category="event", granularity="coupled", description="只累计当时已发候选，主信号优先；月配额重置，冷却跨月延续。不读取未来月末数量，不代表跨产品资金配额。"),
    })
    for identifier in ("basket_source", "panel_formula", "panel_lag", "panel_compare", "cross_mean", "breadth"):
        registry[identifier]["axis_contract"] = "fixed_basket_members_by_same_trading_axis"
    registry["panel_compare"]["missing_policy"] = "preserve_unknown"
    for key, label in (("first", "首次满足"), ("confirm", "连续确认"), ("cooldown", "信号冷却")):
        params = [] if key == "first" else [parameter("window", "观察数", 2 if key == "confirm" else 5, 1, 2500)]
        registry[key] = operator(key, label, [port("value", "condition")], params, output="condition", category="event", description="在完整交易日轴处理条件事件，未知不会形成信号。")
        registry[key].update(
            semantic_kind="condition_series" if key == "confirm" else "condition_pulse",
            temporal_semantics={
                "first": "仅已知不满足到满足的当日输出1；未知到满足仍未知。",
                "confirm": "连续满足达到观察数后保持1；不满足或未知重置计数。",
                "cooldown": "发出1后跳过指定交易日数；未知保留-1，不缩短日期轴。",
            }[key],
            composition_rule="逐日三值条件或脉冲，可显式组合；不是状态编码或区间边界。",
        )
    for spec in series_operator_specs():
        signature = max(spec.signatures, key=lambda item: len(item.inputs))
        names = spec.argument_names(len(signature.inputs))
        inputs, params = [], []
        for name, value_type in zip(names, signature.inputs):
            if value_type.startswith("scalar") and "series" not in value_type:
                default = {"window": 20, "periods": 1, "ddof": 1, "min_periods": 20, "initial": 0.0, "lower": -1.0, "upper": 1.0}.get(name, 0.0)
                params.append(parameter(name, PARAMETER_LABELS.get(name, name), default,
                                        0 if name == "ddof" else 1 if name in {"window", "periods", "min_periods"} else None,
                                        5000 if name in {"window", "periods", "min_periods"} else None))
            else:
                inputs.append(port(name))
        identifier = "indicator." + spec.operator_id
        registry[identifier] = {**operator(identifier, SERIES_OPERATOR_LABELS[spec.operator_id], inputs, params, category="math", description=spec.description),
                                "_arguments": list(names), "_operator": spec.operator_id}
        if spec.operator_id in {"lag", "difference"}:
            registry[identifier].update(
                _aligned_temporal=spec.operator_id,
                description="保留完整交易日轴；前几期为空，缺失不压缩。复用情景中心同轴滞后/差分内核。",
                temporal_contract="preserve_axis_nan_prefix/1",
            )
    if indicator_service is not None:
        raw = {}
        register_indicator_nodes(indicator_service, raw)
        for identifier, item in raw.items():
            item = _timing_reference(item)
            if item is None:
                continue
            inputs = [port(x["name"], label=x.get("label")) for x in item["inputs"]]
            params = [parameter(name, x.get("title") or x.get("label") or name, x.get("default", 20), x.get("minimum"), x.get("maximum"))
                      for name, x in item["parameter_schema"]["properties"].items()]
            registry[identifier] = {**operator(identifier, item["label"], inputs, params, category="indicator", outputs=[port(x["name"], label=x.get("label")) for x in item["outputs"]]),
                                    "_reference": item, "indicator_reference": item["indicator_reference"]}
    return registry


def expressions(step, metadata):
    if metadata.get("_aligned_temporal"):
        return {}
    if step.op in {"formula", "panel_formula"}:
        return {"value": str(step.parameters.get("expression", "a"))}
    if "_reference" in metadata:
        node = SimpleNamespace(type=step.op, parameters=step.parameters, inputs=step.inputs)
        return typed_node_expressions(node, {step.op: metadata["_reference"]})
    if "_operator" in metadata:
        defaults = {p["name"]: p["default"] for p in metadata["parameters"]}
        args = [name if name in step.inputs else repr(step.parameters.get(name, defaults.get(name))) for name in metadata["_arguments"]]
        return {"value": f"{metadata['_operator']}({','.join(args)})"}
    return {}


def _step(identifier, label, op, inputs=None, **parameters):
    return {"id": identifier, "label": label, "op": op, "inputs": inputs or {}, "parameters": parameters}


def templates():
    price = _step("price", "ETF 收盘价", "source", field="close")
    ma = _step("ma", "20 日均线", "formula", {"a": "price.value"}, expression="rolling_mean(a,20,20)")
    above = _step("above", "价格高于均线", "compare", {"left": "price.value", "right": "ma.value"}, operator="gt")
    baseline = [price, ma, above, _step("enter", "首次站上均线", "first", {"value": "above.value"}), _step("leave", "价格低于均线", "not", {"value": "above.value"})]
    repair = [price,
        _step("previous", "前一日收盘价", "indicator.lag", {"values": "price.value"}, periods=1),
        _step("positive", "当日上涨", "compare", {"left": "price.value", "right": "previous.value"}, operator="gt"),
        _step("first_positive", "首次转正", "first", {"value": "positive.value"}),
        _step("previous5", "5 日前收盘价", "indicator.lag", {"values": "price.value"}, periods=5),
        _step("return5", "5 日收益", "formula", {"a": "price.value", "b": "previous5.value"}, expression="a/b-1"),
        _step("prior_return", "此前 5 日收益", "indicator.lag", {"values": "return5.value"}, periods=1),
        _step("weak", "前期下跌背景", "compare", {"left": "prior_return.value"}, operator="le", threshold=-0.02),
        _step("log_price", "对数价格", "formula", {"a": "price.value"}, expression="log(a)"),
        _step("filter", "低滞后趋势", "alpha_beta", {"value": "log_price.value"}, alpha=0.4, beta=0.08),
        _step("slope_change", "斜率改善", "indicator.difference", {"values": "filter.slope"}, periods=1),
        _step("repair", "斜率改善为正", "compare", {"left": "slope_change.value"}, operator="gt", threshold=0.0),
        _step("innovation", "创新值为正", "compare", {"left": "filter.innovation"}, operator="gt", threshold=0.0),
        _step("trend_confirm", "趋势与创新确认", "all", {"left": "repair.value", "right": "innovation.value"}),
        _step("background", "下跌后的首次转正", "all", {"left": "weak.value", "right": "first_positive.value"}),
        _step("enter", "修复入场", "all", {"left": "background.value", "right": "trend_confirm.value"})]
    reversal = [price,
        _step("ma", "250 日均线", "formula", {"a": "price.value"}, expression="rolling_mean(a,250,250)"),
        _step("above", "在长期均线上方", "compare", {"left": "price.value", "right": "ma.value"}, operator="gt"),
        _step("above_value", "均线上方标记", "condition_value", {"value": "above.value"}),
        _step("ratio", "250 日均线上方比例", "formula", {"a": "above_value.value"}, expression="rolling_mean(a,250,250)"),
        _step("weak", "长期弱势", "compare", {"left": "ratio.value"}, operator="le", threshold=0.2),
        _step("previous20", "20 日前收盘价", "indicator.lag", {"values": "price.value"}, periods=20),
        _step("return20", "20 日收益", "formula", {"a": "price.value", "b": "previous20.value"}, expression="a/b-1"),
        _step("decline", "前期回撤背景", "compare", {"left": "return20.value"}, operator="le", threshold=-0.08),
        _step("daily", "单日变化", "indicator.difference", {"values": "price.value"}, periods=1),
        _step("positive", "日变化为正", "compare", {"left": "daily.value"}, operator="gt", threshold=0.0),
        _step("confirm", "连续两日反转", "confirm", {"value": "positive.value"}, window=2),
        _step("context", "弱势与下跌背景", "all", {"left": "weak.value", "right": "decline.value"}),
        _step("enter", "长期弱势反转", "all", {"left": "context.value", "right": "confirm.value"})]
    result = []
    for identifier, label, description, nodes, exit_ref in (
        ("trend", "均线趋势基线", "首次站上均线入场，跌回均线后退出。", baseline, "leave.value"),
        ("repair", "低滞后修复", "参考 A2552 的固定规则结构；不含原实验训练选择与股票绩效。", repair, None),
        ("reversal", "长期弱势反转", "参考 A160S 的弱势门控结构；简化模板，未复刻原量价过滤与训练策略。", reversal, None),
    ):
        definition = Definition(name=label, description=description, nodes=copy.deepcopy(nodes), entry="enter.value", exit=exit_ref)
        result.append({"id": identifier, "label": label, "description": description, "definition": definition.model_dump(mode="json")})
    from .etf_templates import etf_templates
    return result + etf_templates()
