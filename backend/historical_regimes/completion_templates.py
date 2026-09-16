"""New editable identities, composed exclusively from existing operators."""

import copy


def completion_templates():
    from .v2_templates import MARKET_STATES, _ref

    source = {
        "id": "market",
        "type": "source.index",
        "parameters": {
            "ts_code": "000300.SH",
            "name": "沪深300",
            "source_api": "index_daily",
            "field": "close",
        },
    }
    returns = {
        "id": "returns",
        "type": "transform.return",
        "parameters": {"window": 1},
        "inputs": {"value": _ref("market")},
    }
    items = []
    for model in ("hmm", "gmm"):
        identity = (
            "historical_hmm_risk_v1"
            if model == "hmm"
            else "historical_gmm_volatility_v1"
        )
        name = "HMM 历史风险状态" if model == "hmm" else "GMM 历史波动分组"
        # Existing scatter maps the highest first-feature mean to state index 0.
        states = [
            {"id": sid, "label": label, "role": role, "color": color, "order": order}
            for order, (sid, label, role, color) in enumerate(
                (
                    ("high_risk", "高波动风险", "negative", "#dc2626"),
                    ("normal_risk", "中等波动风险", "neutral", "#64748b"),
                    ("low_risk", "低波动风险", "positive", "#16a34a"),
                ),
                1,
            )
        ]
        nodes = [
            source,
            returns,
            {
                "id": "volatility",
                "type": "rolling.std",
                "parameters": {"window": 20},
                "inputs": {"value": _ref("returns")},
            },
            {
                "id": "features",
                "type": "feature.matrix",
                "inputs": {"feature_1": _ref("volatility")},
            },
            {
                "id": "model",
                "type": "model." + model,
                "parameters": {
                    "components": 3,
                    "iterations": 40,
                    "initial_train_size": 60,
                    "initialization_strategy": "random",
                    "random_seed": 17,
                },
                "inputs": {"features": _ref("features", "features")},
            },
        ]
        item = _item(
            identity,
            name,
            "risk",
            states,
            nodes,
            {
                "state": _ref("model", "state"),
                "probabilities": _ref("model", "probabilities"),
            },
            "对20期收益波动率按拟合均值排序；风险仅指相对波动，不代表未来损失概率。",
        )
        if model == "gmm":
            # Retained only for immutable historical compatibility. On actual
            # CSI300 data its realtime replay matched the retrospective labels
            # poorly, so it is not offered for new authoring.
            item["authoring_hidden"] = True
        items.append(item)

    # Causal recognition companion for the existing HMM volatility reference.
    # The fixed thresholds were selected on CSI300 observations through
    # 2018-12-31 only, then checked on 2019-2022 and a 2023-2026 holdout.
    # This deliberately avoids refitting a latent model at decision time: the
    # reference may use full-sample smoothing, while recognition is a simple
    # point-in-time rule on the same 20-day realized-volatility feature.
    volatility_states = [
        {"id": "high_risk", "label": "高波动风险", "role": "negative", "color": "#dc2626", "order": 1},
        {"id": "normal_risk", "label": "中等波动风险", "role": "neutral", "color": "#64748b", "order": 2},
        {"id": "low_risk", "label": "低波动风险", "role": "positive", "color": "#16a34a", "order": 3},
    ]
    realtime_nodes = [
        copy.deepcopy(source),
        copy.deepcopy(returns),
        {
            "id": "volatility",
            "type": "rolling.std",
            "parameters": {"window": 20},
            "inputs": {"value": _ref("returns")},
        },
        {
            "id": "high_condition",
            "type": "condition.compare",
            "label": "高波动条件",
            "parameters": {"operator": "gt", "threshold": 0.014},
            "inputs": {"value": _ref("volatility")},
        },
        {
            "id": "low_condition",
            "type": "condition.compare",
            "label": "低波动条件",
            "parameters": {"operator": "lt", "threshold": 0.0094},
            "inputs": {"value": _ref("volatility")},
        },
        {
            "id": "low_or_normal",
            "type": "state.select",
            "label": "低波动否则中波动",
            "parameters": {"true_code": 2, "false_code": 1},
            "inputs": {"condition": _ref("low_condition", "condition")},
        },
        {
            "id": "classifier",
            "type": "state.select",
            "label": "高波动否则沿用低/中判断",
            "parameters": {"true_code": 0, "false_code": 1},
            "inputs": {
                "condition": _ref("high_condition", "condition"),
                "when_false": _ref("low_or_normal", "state"),
            },
        },
    ]
    items.append(
        {
            "id": "csi300-volatility-hmm-reference-recognition-v1",
            "name": "沪深300波动率状态 · 实时识别",
            "version": 1,
            "default_mode": "realtime",
            "description": (
                "以20日收益波动率实时识别高/中/低波动状态。高于1.40%为高波动、"
                "低于0.94%为低波动，其余为中等波动；阈值只用2010-2018年的"
                "沪深300 HMM 事后参考拟合，并在2019-2022及2023-2026留出段检查。"
                "需绑定精确的HMM历史参考后使用识别有效性验证；不使用未来数据。"
            ),
            "tags": ["实时识别", "沪深300", "波动率状态", "HMM参考", "固定阈值", "TAA"],
            "definition": {
                "schema_version": "2.0",
                "name": "沪深300波动率状态 · 实时识别 v1",
                "description": (
                    "20日已实现波动率固定阈值分类，高>1.40%、低<0.94%、中间为中等波动。"
                    "阈值在2010-2018 HMM历史参考上确定；实时执行只依赖当时已知收盘价。"
                ),
                "template_id": "csi300-volatility-hmm-reference-recognition-v1",
                "default_mode": "realtime",
                "study": {"purpose": "realtime_recognition", "family": "risk"},
                "graph": {
                    "nodes": realtime_nodes,
                    "outputs": {"state": _ref("classifier", "state"), "volatility": _ref("volatility")},
                    "channel_metadata": {"volatility": {"label": "20日收益波动率（日频标准差）"}},
                    "exposed_node_ids": [node["id"] for node in realtime_nodes],
                },
                "states": volatility_states,
                "evaluation_targets": [
                    {"id": "csi300", "name": "沪深300", "source": {"kind": "index", "ts_code": "000300.SH", "source_api": "index_daily", "field": "close"}, "primary": True}
                ],
                "validation": {"walk_forward": True, "folds": 4},
                "usage_intent": "taa",
            },
        }
    )

    # Drawdown Cycle: the retrospective reference uses complete local monthly
    # peak/trough segments, while the realtime companion is a causal state
    # machine using only the rolling high and the active trough.
    drawdown_states = [
        {"id": "normal", "label": "正常", "role": "positive", "color": "#16a34a", "order": 1},
        {"id": "recovery", "label": "修复", "role": "neutral", "color": "#64748b", "order": 2},
        {"id": "stress", "label": "压力", "role": "negative", "color": "#dc2626", "order": 3},
    ]
    drawdown_reference_nodes = [
        copy.deepcopy(source),
        {"id": "monthly", "type": "align.resample", "parameters": {"frequency": "monthly", "aggregation": "last"}, "inputs": {"value": _ref("market")}},
        {"id": "pivots", "type": "pivot.local_extrema", "parameters": {"left_window": 2, "right_window": 2, "head_window": 1, "tail_window": 1}, "inputs": {"value": _ref("monthly")}},
        {"id": "segments", "type": "segment.between_pivots", "inputs": {"pivot": _ref("pivots", "pivot")}},
        {"id": "phase", "type": "segment.phase_direction", "inputs": {"pivot": _ref("pivots", "pivot"), "start": _ref("segments", "start"), "end": _ref("segments", "end")}},
        {"id": "change", "type": "segment.change", "inputs": {"value": _ref("monthly"), "start": _ref("segments", "start"), "end": _ref("segments", "end")}},
        {"id": "cycle", "type": "post.drawdown_cycle_reference", "parameters": {"stress_drawdown": .12}, "inputs": {"phase": _ref("phase", "phase"), "change": _ref("change"), "start": _ref("segments", "start"), "end": _ref("segments", "end")}},
    ]
    drawdown_reference_description = (
        "月频局部峰谷划分完整波段：完整峰到谷跌幅达到12%时，自峰后首月到谷底标记压力；"
        "紧随其后的完整谷到峰波段标记修复；其他完整波段标记正常。首尾未完成区间不回填，"
        "因此只作为事后参考，不产生实时信号。"
    )
    items.append({
        "id": "csi300-drawdown-cycle-reference-v1",
        "name": "沪深300回撤周期 · 事后参考",
        "version": 1,
        "default_mode": "retrospective",
        "tags": ["历史参考", "沪深300", "回撤周期", "压力修复", "月频"],
        "description": drawdown_reference_description,
        "definition": {
            "schema_version": "2.0",
            "name": "沪深300回撤周期 · 事后参考 v1",
            "description": drawdown_reference_description,
            "template_id": "csi300-drawdown-cycle-reference-v1",
            "default_mode": "retrospective",
            "study": {"purpose": "historical_reference", "family": "risk"},
            "graph": {"nodes": drawdown_reference_nodes, "outputs": {"state": _ref("cycle", "state")}, "exposed_node_ids": [node["id"] for node in drawdown_reference_nodes]},
            "states": copy.deepcopy(drawdown_states),
            "evaluation_targets": [{"id": "csi300", "name": "沪深300", "source": {"kind": "index", "ts_code": "000300.SH", "source_api": "index_daily", "field": "close"}, "primary": True}],
            "validation": {"walk_forward": False, "folds": 4},
            "usage_intent": "research_display",
        },
    })
    drawdown_realtime_nodes = [
        copy.deepcopy(source),
        {"id": "monthly", "type": "align.resample", "parameters": {"frequency": "monthly", "aggregation": "last"}, "inputs": {"value": _ref("market")}},
        {"id": "cycle", "type": "model.drawdown_cycle_realtime", "parameters": {"lookback": 9, "stress_drawdown": .12, "recovery_rebound": .05, "recovery_exit_drawdown": .08}, "inputs": {"value": _ref("monthly")}},
    ]
    drawdown_realtime_description = (
        "闭合月末只使用当时及过去信息：相对最近9个月高点回撤达到12%进入压力；"
        "自活动谷底反弹达到5%进入修复；相对滚动高点剩余回撤收窄到8%以内回到正常。"
        "用于识别对应回撤周期历史参考，不使用未来峰谷。"
    )
    items.append({
        "id": "csi300-drawdown-cycle-realtime-v1",
        "name": "沪深300回撤周期 · 实时识别",
        "version": 1,
        "default_mode": "realtime",
        "tags": ["实时识别", "沪深300", "回撤周期", "压力修复", "月频", "TAA"],
        "description": drawdown_realtime_description,
        "definition": {
            "schema_version": "2.0",
            "name": "沪深300回撤周期 · 实时识别 v1",
            "description": drawdown_realtime_description,
            "template_id": "csi300-drawdown-cycle-realtime-v1",
            "default_mode": "realtime",
            "study": {"purpose": "realtime_recognition", "family": "risk"},
            "graph": {"nodes": drawdown_realtime_nodes, "outputs": {"state": _ref("cycle", "state")}, "exposed_node_ids": [node["id"] for node in drawdown_realtime_nodes]},
            "states": copy.deepcopy(drawdown_states),
            "evaluation_targets": [{"id": "csi300", "name": "沪深300", "source": {"kind": "index", "ts_code": "000300.SH", "source_api": "index_daily", "field": "close"}, "primary": True}],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "taa",
        },
    })

    nodes = [
        source,
        {
            "id": "change",
            "type": "model.change_point",
            "parameters": {"window": 20, "threshold": 1.5, "confirmation": 2},
            "inputs": {"value": _ref("market")},
        },
    ]
    states = copy.deepcopy(MARKET_STATES)
    for state, sid, label in zip(
        states,
        ("increasing_change", "stable_change", "decreasing_change"),
        ("均值变化上移", "均值变化稳定", "均值变化下移"),
    ):
        state.update(id=sid, label=label)
    change_item = _item(
        "historical_window_mean_change_v1",
        "窗口均值变化状态",
        "custom",
        states,
        nodes,
        {"state": _ref("change", "state")},
        "价格一阶差分的前后窗口均值差除以局部标准差；不是 BOCPD 或 PELT。",
    )
    change_item["authoring_hidden"] = True
    items.append(change_item)
    nodes = [source, returns]
    for tag, window in (("fast", 10), ("slow", 40)):
        nodes.extend(
            [
                {
                    "id": tag,
                    "type": "rolling.mean",
                    "parameters": {"window": window},
                    "inputs": {"value": _ref("returns")},
                },
                {
                    "id": tag + "_state",
                    "type": "model.threshold",
                    "parameters": {"upper": 0.001, "lower": -0.001},
                    "inputs": {"value": _ref(tag)},
                },
            ]
        )
    nodes.append(
        {
            "id": "consensus",
            "type": "model.ensemble",
            "parameters": {"weights": [0.5, 0.5], "consensus_threshold": 0.75},
            "inputs": {
                "state_1": _ref("fast_state", "state"),
                "state_2": _ref("slow_state", "state"),
            },
        }
    )
    trend_item = _item(
        "historical_trend_ensemble_v1",
        "历史趋势多窗口共识",
        "market_trend",
        MARKET_STATES,
        nodes,
        {"state": _ref("consensus", "state")},
        "10期与40期平均收益趋势共识；分歧保留未分类，共识权重不是概率证据。",
    )
    trend_item["authoring_hidden"] = True
    items.append(trend_item)
    return copy.deepcopy(items)


def _item(identity, name, family, states, nodes, outputs, description):
    return {
        "id": identity,
        "name": name,
        "version": 1,
        "default_mode": "retrospective",
        "description": description,
        "tags": ["历史参考", "可编辑计算步骤"],
        "definition": {
            "schema_version": "2.0",
            "name": name,
            "description": description,
            "template_id": identity,
            "study": {"purpose": "historical_reference", "family": family},
            "graph": {
                "nodes": nodes,
                "outputs": outputs,
                "exposed_node_ids": [n["id"] for n in nodes],
            },
            "states": states,
            "evaluation_targets": [],
            "validation": {"walk_forward": True, "folds": 4},
            "usage_intent": "research_display",
        },
    }
