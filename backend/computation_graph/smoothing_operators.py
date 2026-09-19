"""Single numeric smoothing contract, adapted into the existing graph registry."""


def _integer(default, minimum, maximum, title, description):
    return dict(type="integer", default=default, minimum=minimum, maximum=maximum,
                title=title, description=description)


SMOOTHING_OPERATORS = {
    "filter.butterworth_zero_phase": {
        "label": "Butterworth 零相位滤波（事后）", "kernel_id": "butterworth_zero_phase",
        "causal": False, "knowledge_scope": "full_input", "minimum_samples": 10, "kernel_version": "smoothing/1.1",
        "description": "二阶低通前向、反向各一次，抵消相位滞后；不保证原始峰谷日期不变。连续有效段超过9点才输出；首尾使用9点奇对称延拓，端点及新增数据后的结果可能修订。",
        "parameters": {"period": _integer(20, 3, 5000, "截止周期（观测数）", "单向滤波截止频率为1/周期；双向后幅度响应平方，截止处为−6 dB。不是移动均线窗口。")},
        "granularity": "coupled", "reason": "双向递推和边界初态共同定义同一个零相位滤波；不能拆成普通因果节点。",
        "complexity": "O(T)", "missing_policy": "split_finite_segments_minimum_10",
    },
    "filter.savitzky_golay_centered": {
        "label": "Savitzky–Golay 居中平滑（事后）", "kernel_id": "savitzky_golay_centered",
        "causal": False, "knowledge_scope": "full_input", "minimum_samples": 3,
        "description": "对完整居中窗口拟合局部多项式，取中心值；需右侧半窗数据。首尾与含NaN/Inf的窗口留空，不外推；保留局部形状但不保证峰谷时间不变。系统保守按完整研究快照记录可得时点。",
        "parameters": {"window": _integer(21, 3, 501, "居中窗口（奇数）", "必须为奇数；左右各需(window−1)/2个观测。"),
                       "polyorder": _integer(3, 0, 5, "多项式阶数", "必须小于窗口；0阶等价居中均值，更高阶不等于更强去噪。")},
        "granularity": "primitive", "reason": "一个局部多项式投影只输出平滑值，峰谷检测及分类保持独立。",
        "complexity": "O(T × window + window × polyorder²)", "missing_policy": "complete_finite_centered_window",
    },
    "filter.ehlers_error_correcting": {
        "label": "Ehlers 误差校正低延迟滤波（实时）", "kernel_id": "ehlers_error_correcting",
        "causal": True, "minimum_samples": 2,
        "description": "Ehlers/Way Zero Lag (Well, Almost)：在有限增益网格中最小化当期跟踪误差，仅用当期与历史数据。低延迟与平滑度存在取舍，不是严格零相位；缺失重置，连续满周期后输出。",
        "parameters": {"period": _integer(20, 2, 5000, "EMA 周期", "α=2/(周期+1)，首个有效值初始化EMA与校正值；连续满周期后输出。"),
                       "gain_limit": _integer(50, 0, 100, "增益上限（十分之一）", "50表示在−5至5内以0.1步长搜索，包含两端；误差并列取较小增益。0仍为两级EMA，不是单级EMA。")},
        "granularity": "coupled", "reason": "EMA状态、前期校正值和当期离散增益选择共同递推；无独立可替换的分类或统计步骤。",
        "complexity": "O(T × (2 × gain_limit + 1))", "missing_policy": "reset_and_rewarm",
    },
}


def register_smoothing_operators(registry, numeric_node, port):
    for identifier, spec in SMOOTHING_OPERATORS.items():
        node = numeric_node(identifier, spec["label"], "filter", [port("value", "series<float64>")],
                            [port("value", "series<float64>")],
                            {"type": "object", "properties": spec["parameters"], "additionalProperties": False},
                            causal=spec["causal"])
        node.update({key: spec[key] for key in ("description", "kernel_id", "minimum_samples", "missing_policy")})
        node.update(kernel_version=spec.get("kernel_version", "smoothing/1"), kernel_dependencies=[spec["kernel_id"], "smoothing_available", "retrospective_dating_timing"], axis_contract="same_observation_axis_equal_sample_spacing",
                    cost_estimate={"class": "windowed" if not spec["complexity"] == "O(T)" else "linear", "expression": spec["complexity"], "unit": "observations"},
                    granularity={"kind": spec["granularity"], "label": "耦合内核" if spec["granularity"] == "coupled" else "基础算子",
                                 "expandable": False, "reason": spec["reason"], "contract_version": 1})
        if "knowledge_scope" in spec:
            node["knowledge_scope"] = spec["knowledge_scope"]
        registry[identifier] = node
