"""Domain-only condition/phase contracts and operator granularity metadata."""
from __future__ import annotations

import copy

CONDITION = "condition<int64>"
PHASE_CODES = "phase_codes<int64>"

COMPOSITE_STEPS = {
    "model.threshold": ["上界比较（含等号）", "下界比较（含等号）", "条件选择状态"],
    "model.quadrant": ["增长条件", "通胀条件", "四象限条件选择"],
    "model.peak_trough": ["峰谷定位", "PS 联合约束筛选", "相邻峰谷分段", "阶段方向", "区间统计", "震荡波段合并"],
}
PENDING_COMPOSITES = {
    "feature.trend_metrics": "包含独立的尺度、偏离、斜率、效率和风险证据；尚未提供等价展开，保留现有计算契约。",
    "model.trend_regime": "候选条件、状态递推和阶段解释仍组合执行；尚未提供等价展开。",
    "model.turning_point": "转折检测与区间分类仍组合执行；尚未提供等价展开。",
    "model.change_point": "突变统计与持续确认仍组合执行；尚未提供等价展开。",
    "model.ensemble": "成员加权、共识与拒识仍组合执行；尚未提供等价展开。",
}
COUPLED_REASONS = {
    "pivot.ps_filter": "阶段、周期和首尾约束相互影响，删点后必须重新检查；每轮至少删一点，直至稳定。",
    "post.peak_sideways": "从最早起点选择最长合格完整波段，振幅、路径效率和峰谷结构联合约束。",
    "model.hysteresis": "进入和退出门槛依赖前一活动状态。",
    "post.hysteresis": "进入和退出门槛依赖前一活动状态。",
    "post.confirmation": "连续候选计数、活动状态和最短保持期共同决定切换。",
    "post.min_duration": "活动状态和持续期决定何时允许切换。",
    "post.merge_short_regimes": "依据完整前后区间选择性合并，不能作为无状态逐点过滤。",
    "filter.kalman": "状态估计与协方差按时间联合递推。",
    "filter.kama": "方向效率决定逐期递推增益。",
    "filter.super_smoother": "二阶递推共同决定当前滤波值。",
    "filter.ema": "当前平滑值依赖前一平滑值。",
    "indicator.recursive_smooth": "递推状态属于同一平滑问题。",
    "model.hmm": "转移与观测参数共同拟合并推断隐状态。",
    "model.markov": "状态转移和分量参数联合拟合。",
    "model.gmm": "分量责任度与参数联合迭代拟合。",
    "model.external_optimized": "仅允许显式登记的隔离模型拟合与推断。",
}


def granularity_metadata(node):
    identifier = node["id"]
    if identifier in COMPOSITE_STEPS:
        return {"kind": "composite", "label": "组合模板", "expandable": True,
                "reason": "添加时展开为真实计算步骤；已有节点可显式展开，原版本不自动迁移。",
                "steps": COMPOSITE_STEPS[identifier], "contract_version": 1}
    if identifier in PENDING_COMPOSITES:
        return {"kind": "composite", "label": "组合算法（整体执行）", "expandable": False,
                "reason": PENDING_COMPOSITES[identifier], "contract_version": 1}
    if identifier in COUPLED_REASONS:
        return {"kind": "coupled", "label": "耦合内核", "expandable": False,
                "reason": COUPLED_REASONS[identifier], "contract_version": 1}
    kind = "source" if node.get("category") == "source" else (
        "indicator" if node.get("category") == "indicator_calculation" else "primitive")
    return {"kind": kind, "label": {"source": "数据输入", "indicator": "指标计算", "primitive": "基础算子"}[kind],
            "expandable": False, "reason": node.get("description") or node["label"], "contract_version": 1}


def register_granular_nodes(registry, numeric_node, port):
    series, states = "series<float64>", "state_codes<int64>"
    pivots, start_type, end_type = "pivots<time>", "segment_start<time>", "segment_end<time>"

    def parameter(kind, default, title, **constraints):
        return {"type": kind, "default": default, "title": title, **constraints}

    def add(identifier, label, category, inputs, outputs, parameters, kernels, description, *, causal=True):
        item = numeric_node(identifier, label, category, inputs, outputs,
                            {"type": "object", "properties": parameters, "additionalProperties": False}, causal=causal)
        item.update(kernel_id=kernels[0], kernel_dependencies=[*kernels, "maximum_int64"], execution_handler="granular",
                    description=description, missing_policy="preserve_unavailable", granularity=granularity_metadata({"id": identifier, "label": label, "description": description}))
        registry[identifier] = item
        return item

    add("condition.compare", "数值比较", "condition", [port("value", series), port("bound", series, required=False)],
        [port("condition", CONDITION)], {
            "operator": parameter("string", "ge", "比较方式", enum=["ge", "gt", "le", "lt", "eq", "ne"],
                                  enum_labels=["大于等于 ≥", "大于 >", "小于等于 ≤", "小于 <", "等于 =", "不等于 ≠"]),
            "threshold": parameter("number", 0.0, "比较界线", description="连接界线输入时以该输入为准，否则使用此常量。"),
        }, ["condition_compare"], "逐期比较两个数值；缺失或无穷值输出未知条件，不当作不满足。")
    add("condition.valid", "有效数值判断", "condition", [port("value", series)], [port("condition", CONDITION)], {},
        ["condition_valid"], "有限数值为满足，NaN 或无穷值为不满足；用于明确的数据质量条件。")
    for identifier, label in (("condition.all", "条件同时满足"), ("condition.any", "条件至少满足一个")):
        add(identifier, label, "condition", [port("left", CONDITION), port("right", CONDITION)], [port("condition", CONDITION)], {},
            ["condition_logic"], "组合两个条件；任一条件未知时保留未知，未知不等于假。")
    add("condition.not", "条件取反", "condition", [port("condition", CONDITION)], [port("condition", CONDITION)], {},
        ["condition_logic"], "满足与不满足互换；未知仍为未知。")
    add("state.select", "按条件选择状态", "postprocess", [port("condition", CONDITION),
        port("when_true", states, required=False), port("when_false", states, required=False)], [port("state", states)], {
            "true_code": parameter("integer", 0, "条件满足时的状态", minimum=-1, maximum=11, state_code=True),
            "false_code": parameter("integer", 1, "条件不满足时的状态", minimum=-1, maximum=11, state_code=True),
        }, ["select_state"], "条件满足或不满足时选择对应状态；可连接另一状态结果。未知条件始终未分类；未选分支不影响数值。")
    encoding = add("state.encode", "确定性状态编码", "output", [port("state", states)],
        [port("probabilities", "probabilities<time,state>"), port("confidence", "confidence<time>")], {},
        ["state_probabilities", "state_confidence"], "把明确状态转为 0/1 隶属编码与已分类指示值；1 不代表预测正确率或统计置信度。")
    encoding["outputs"][0]["label"] = "状态隶属编码（非预测概率）"
    encoding["outputs"][1]["label"] = "已分类指示值（非统计置信度）"
    encoding["probability_semantics"] = "deterministic_membership"

    ps_parameters = registry["model.peak_trough"]["parameter_schema"]["properties"]
    filtered = add("pivot.ps_filter", "PS 峰谷约束筛选", "segmentation", [port("value", series), port("pivot", pivots)],
        [port("pivot", pivots), port("marker", series), port("pivot_price", series)],
        {key: copy.deepcopy(ps_parameters[key]) for key in ("min_phase", "min_cycle", "amplitude_exception")},
        ["ps_filter_pivots", "peak_trough_remove", "peak_trough_alternate", "retrospective_dating_timing"],
        "对候选峰谷联合检查首尾、最短阶段与完整周期；输出保留峰谷，不做市场分类。", causal=False)
    filtered["knowledge_scope"] = "full_input"
    filtered["outputs"][1]["label"] = "峰谷数值标记（1 峰，-1 谷）"
    boundaries = [port("start", start_type), port("end", end_type)]
    add("segment.phase_direction", "完整波段方向", "segmentation", [port("pivot", pivots), *copy.deepcopy(boundaries)],
        [{**port("phase", PHASE_CODES), "label": "完整波段方向"}], {}, ["phase_direction"], "谷到峰为上行，峰到谷为下行；这是波段方向，不是牛熊震荡标签。", causal=False)
    add("segment.boundary_line", "峰谷边界连线", "segmentation", [port("value", series), *copy.deepcopy(boundaries)],
        [port("value", series)], {}, ["boundary_line"], "只连接完整区间的首尾价格；包含末端拐点，不延伸未完成尾段。", causal=False)
    sideways_outputs = [copy.deepcopy(item) for item in registry["model.peak_trough"]["outputs"]
                        if item["name"] == "state" or item["name"].startswith("sideways_")]
    add("post.peak_sideways", "小波段震荡合并", "postprocess", [port("value", series), {**port("phase", PHASE_CODES), "label": "完整波段方向"}, *copy.deepcopy(boundaries)],
        sideways_outputs,
        {key: copy.deepcopy(ps_parameters[key]) for key in ("sideways_enabled", "small_swing_threshold", "sideways_max_range", "sideways_max_efficiency", "sideways_min_duration")},
        ["peak_trough_sideways"], "按小幅反向波段、整段振幅、路径效率及峰谷结构合并震荡；未完成和缺失不填充。", causal=False)

    # The same boundaries may be exposed as ordinary index evidence, but cannot
    # be wired back into typed segmentation inputs without their nominal type.
    registry["segment.between_pivots"]["outputs"].extend([
        {**port("start_index", "index<time>"), "label": "起点索引（证据）"},
        {**port("end_index", "index<time>"), "label": "终点索引（证据）"},
    ])
