"""Bounded preparation and orchestration for ETF-only frozen policy research.

Feature plans are prepared explicitly, never inside execution. Candidate OHLC
and per-member inputs are reused; only a bounded signal matrix and small score
table survive between candidates. Fitting and application use separate kernels.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np

from custom_indicators.errors import ValidationError
from .contracts import Definition
from .learning import encode_states_kernel, score_trades_kernel, choose_actions_kernel, route_actions_kernel


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    graph: object
    parameters: tuple


def prepare_candidates(definition, runtime):
    training = definition.training
    if training is None:
        return ()
    nodes = {node.id: node for node in definition.nodes}
    protected = set()
    def protect(reference):
        identifier = reference.split(".")[0]
        if identifier in protected:
            return
        protected.add(identifier)
        for ref in nodes[identifier].inputs.values():
            protect(ref)
    for reference in [*training.state_refs, *([definition.exit] if definition.exit else [])]:
        protect(reference)
    # Validate each axis in isolation; conflicting writes make order ambiguous.
    axis_keys = set()
    for axis in training.search_space:
        keys = set()
        for choice in axis.choices:
            choice_keys = set()
            for patch in choice:
                if patch.node not in nodes or patch.parameter not in {p["name"] for p in runtime.catalog[nodes[patch.node].op]["parameters"]}:
                    raise ValidationError("TIMING_SEARCH_INVALID", "参数搜索绑定了不存在的步骤或参数。")
                if patch.node in protected or nodes[patch.node].op in {"source", "basket_source"}:
                    raise ValidationError("TIMING_SEARCH_INVALID", "搜索只能修改候选入场参数，不能改变数据源、共享状态或退出规则。")
                if (patch.node, patch.parameter) in choice_keys:
                    raise ValidationError("TIMING_SEARCH_INVALID", "同一搜索方案不能重复设置一个参数。")
                choice_keys.add((patch.node, patch.parameter))
            if keys and keys != choice_keys:
                raise ValidationError("TIMING_SEARCH_INVALID", "同一搜索轴的每个方案必须绑定相同参数。")
            keys = choice_keys
        if keys & axis_keys:
            raise ValidationError("TIMING_SEARCH_INVALID", "不同搜索轴不能同时修改同一个参数。")
        axis_keys |= keys
    candidates = []
    combinations = product(*(axis.choices for axis in training.search_space))
    for index, choices in enumerate(combinations):
        patches = tuple(patch for choice in choices for patch in choice)
        for action in training.actions:
            if action.entry is None:
                continue
            variant = definition.model_copy(deep=True)
            variant.training = None
            variant.entry = action.entry
            for patch in patches:
                next(node for node in variant.nodes if node.id == patch.node).parameters[patch.parameter] = patch.value
            # All original draft nodes validate, only actual candidate inputs run.
            prepared = runtime.prepare(variant)
            candidates.append(Candidate(f"{action.id}:{index}", action.label + (f" · 参数组 {index + 1}" if training.search_space else ""), prepared, patches))
    return tuple(candidates)


def _day(day):
    return str(np.datetime64(int(day), "D"))


def _number(value):
    return float(value) if np.isfinite(value) else None


def fit_and_route(training, candidates, runtime, bars, baskets, channels, start, split, simulate):
    """Freeze at split-1 close; training cannot read any OOS OHLC or label."""
    size = len(bars.dates)
    fit_end = split - training.embargo_bars
    if fit_end <= start:
        raise ValidationError("TIMING_TRAIN_TOO_SHORT", "训练区间不足以容纳隔离期，请扩大训练期或减少隔离交易日。")
    width = max((value.shape[0] for value in baskets.values()), default=1)
    if sum(candidate.graph.cost_per_bar for candidate in candidates) * size * width > 2_000_000_000:
        raise ValidationError("TIMING_TRAIN_BUDGET", "参数搜索超过工作量预算，请缩短区间或减少候选。")
    retained = sum(array.nbytes for array in channels.values()) + sum(array.nbytes for array in baskets.values())
    candidate_peak = max((candidate.graph.workspace_arrays + 3 * (width - 1) * sum(p["type"] in {"panel", "condition_panel"} for meta in candidate.graph.metadata.values() for p in meta["outputs"])) * size * 8 for candidate in candidates)
    if retained + len(candidates) * size * 8 + candidate_peak > 128 * 1024 * 1024:
        raise ValidationError("TIMING_TRAIN_MEMORY", "训练超过 128 MiB 工作区预算，请减少篮子成员、步骤或区间。")
    if training.mode in {"month", "quarter"}:
        months = [int(_day(day)[5:7]) for day in bars.dates]
        states = np.asarray([month - 1 if training.mode == "month" else (month - 1) // 3 for month in months], dtype=np.int64)
        labels = [f"{i + 1}月" if training.mode == "month" else f"第{i + 1}季度" for i in range(12 if training.mode == "month" else 4)]
    else:
        conditions = np.empty((len(training.state_refs), size), dtype=np.int64)
        for index, ref in enumerate(training.state_refs):
            conditions[index] = channels[ref]
        states = encode_states_kernel(conditions)
        labels = ["全局"] if not training.state_refs else [" / ".join(f"{ref}={'满足' if state & (1 << k) else '不满足'}" for k, ref in enumerate(training.state_refs)) for state in range(2 ** len(training.state_refs))]
    states.setflags(write=False)
    signals = np.empty((len(candidates), size), dtype=np.int64)
    scores = np.empty((len(labels), len(candidates)), dtype=np.float64)
    audit_candidates = []
    for index, candidate in enumerate(candidates):
        output = runtime.evaluate(candidate.graph, bars, baskets)
        entry = output[candidate.graph.definition.entry]
        exit_ = output.get(candidate.graph.definition.exit)
        if exit_ is None:
            exit_ = np.zeros(size, dtype=np.int64)
        signals[index] = entry
        # The execution kernel receives an exclusive end before the test split.
        path, trades, count = simulate(bars, entry, exit_, start, fit_end, candidate.graph.definition.execution)
        stats = score_trades_kernel(trades, count, states, start, fit_end, len(labels), training.min_trades, training.confidence, training.risk_penalty)
        scores[:, index] = stats[:, 3]
        audit_candidates.append({"id": candidate.id, "label": candidate.label,
            "parameters": [patch.model_dump(mode="json") for patch in candidate.parameters],
            "states": [{"state": labels[s], "sample_count": int(row[0]), "mean_return": _number(row[1]), "win_rate": _number(row[2]), "utility": _number(row[3]), "stop_rate": _number(row[4])} for s, row in enumerate(stats)]})
        del output, path, trades, entry, exit_
    selected = choose_actions_kernel(scores, training.min_utility, -1)
    routed = route_actions_kernel(signals, states, selected, split)
    routed.setflags(write=False)
    selection = []
    for state, index in enumerate(selected):
        item = audit_candidates[int(index)] if index >= 0 else None
        stats = item["states"][state] if item else None
        selection.append({"state": labels[state], "action_id": item["id"] if item else None,
            "action_label": item["label"] if item else "空仓（训练证据不足或效用未过门槛）",
            "sample_count": stats["sample_count"] if stats else 0, "utility": stats["utility"] if stats else None})
    audit = {"mode": training.mode, "freeze_date": _day(bars.dates[split - 1]),
        "fit_end_date": _day(bars.dates[fit_end - 1]), "embargo_bars": training.embargo_bars,
        "min_trades": training.min_trades, "selection": selection, "candidates": audit_candidates,
        "warnings": ["这是 ETF 改编训练目标，不继承股票实验的评分公式或绩效。",
            "同一训练集比较多个候选仍有选择偏差；效用惩罚不等于独立内层验证或多重检验校正。",
            "仅训练期已平仓交易参与选择；样本外不重训、不因零信号回退。",
            "训练区间用于拟合，不回填拟合后的策略收益；图表策略从样本外开始。",
            "同月/同季模式按信号日期的自然月/季匹配历史，整段样本外冻结，不是滚动重训。"]}
    return routed, audit
