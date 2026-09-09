"""对内置算子与用户公式做因果性审计。

两层：

* :func:`audit_operators` —— 遍历算子注册表，对每个算子构造合法输入并跑 P1。
  结果落 ``baseline.json``，由 CI 守门；新增算子没有裁决即失败。
* :func:`audit_expression` —— 审计一条用户公式。根输出带时间轴时 P1 直接决定性；
  根是标量时（大多数指标如此）改为审计 DAG 里每一个带时间轴的中间节点，
  并把「消费整窗」的位置报出来交由调用方确认窗口右端。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from cal_indicators.typed_dsl import (
    DEFAULT_VARIABLE_TYPES,
    TypedExpressionPlan,
    TypedIndicatorRuntime,
    compose_typed_expression,
)
from cal_indicators.typed_operators import TypedOperatorSpec, get_typed_operator_registry
from cal_indicators.typed_types import TypedDslError, ValueType

from .probes import (
    ProbeOutcome,
    Verdict,
    default_decision_dates,
    merge_verdicts,
    run_tail_probe,
    tail_dependence,
    warmup_sensitivity,
)
from .synthetic import SyntheticPanel, synthetic_panel

BASELINE_PATH = Path(__file__).with_name("baseline.json")

#: 每个算子最多审计多少个类型组合。够覆盖有意思的形态，又不至于让全量扫描变慢。
MAX_CASES_PER_OPERATOR = 8


# --------------------------------------------------------------------------
# 候选输入
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Candidate:
    label: str
    value_type: ValueType
    values: tuple[Any, ...]

    @property
    def has_time(self) -> bool:
        return "time" in self.value_type.axes


def candidate_pool(panel: SyntheticPanel) -> tuple[_Candidate, ...]:
    """算子探测用的候选输入。

    带时间轴的排在前面：组合是按顺序枚举后截断的，把标量放前面会让 ``add``
    这类多态算子只被标量组合覆盖，时间轴上的行为反而测不到。

    同一个结构会重复出现多次（收益率 / 净值 / 无量纲），因为 ``ValueType``
    的相等性刻意忽略语义量纲，而算子的类型推断却会检查它——净值类算子只接受
    ``price_basis`` 非空的输入，收益率类算子只接受 ``return_decimal``。
    """

    variables = panel.variables
    returns = np.asarray(variables["returns"])
    nav = np.asarray(variables["adjusted_nav"])[1:]
    asset_returns = np.asarray(variables["asset_returns"])
    weights = np.asarray(variables["asset_weights"])
    covariance = np.cov(asset_returns, rowvar=False) + np.eye(panel.assets) * 1e-3
    dates = np.arange(returns.size, dtype=np.float64) + 20_000.0
    drawdowns = np.zeros(returns.size, dtype=np.float64)
    for index in range(1, returns.size):
        phase = index % 10
        if 1 <= phase <= 5:
            drawdowns[index] = -0.01 * phase
    registry = get_typed_operator_registry()
    fit_type = registry["linear_fit"].infer_output((ValueType.series(),))
    from cal_indicators.drawdown_interval import INTERVAL_TYPE

    return (
        _Candidate("series<drawdown>", ValueType.series(semantic_dimension="return_decimal"), (drawdowns,)),
        _Candidate("series<return>", ValueType.series(semantic_dimension="return_decimal"), (returns,)),
        _Candidate("series", ValueType.series(), (returns,)),
        _Candidate(
            "series<nav>",
            ValueType.series(semantic_dimension="adjusted_nav", price_basis="adjusted_nav"),
            (nav,),
        ),
        _Candidate("series<date>", ValueType.series(semantic_dimension="date"), (dates,)),
        _Candidate("matrix<return>", ValueType.matrix(semantic_dimension="return_decimal"), (asset_returns,)),
        _Candidate("matrix", ValueType.matrix(), (asset_returns,)),
        _Candidate("mask<time>", ValueType.mask(("time",), ("T",)), (returns > 0,)),
        _Candidate("mask<time,asset>", ValueType.mask(("time", "asset"), ("T", "N")), (asset_returns > 0,)),
        _Candidate("vector", ValueType.vector(), (weights,)),
        _Candidate("mask<asset>", ValueType.mask(("asset",), ("N",)), (weights > weights.mean(),)),
        _Candidate("matrix<asset,asset>", ValueType.matrix(("asset", "asset"), ("N", "N")), (covariance,)),
        # 标量的具体取值无法由类型区分（窗口长度、分位数、下界上界都是 scalar），
        # 所以给出几个备选，谁先算得通用谁。
        _Candidate("scalar", ValueType.scalar(), (0.37, 5.0, 2.0, 0.9)),
        _Candidate("scalar<count>", ValueType.scalar(semantic_dimension="count"), (5.0, 2.0, 1.0)),
        _Candidate("scalar<date>", ValueType.scalar(semantic_dimension="date"), (20_000.0, 20_005.0, 20_010.0)),
        _Candidate("state<drawdown_interval>", INTERVAL_TYPE, ((1.0, 5.0, 8.0, 1.0),)),
        _Candidate("state<linear_fit>", fit_type, ((0.1, 0.2, 0.3, 1.2, float(returns.size)),)),
    )


# --------------------------------------------------------------------------
# 算子级审计
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OperatorCase:
    signature: str
    output: str
    outcome: ProbeOutcome

    def as_dict(self) -> dict[str, Any]:
        return {"signature": self.signature, "output": self.output, **self.outcome.as_dict()}


@dataclass(frozen=True)
class OperatorFinding:
    operator_id: str
    category: str
    verdict: Verdict
    detail: str
    cases: tuple[OperatorCase, ...] = ()
    #: P3 的结论，**不并入 verdict**：预热敏感说明回测与实盘会因可得历史长度
    #: 不同而分叉，是可复现性性质，不是泄露。混进裁决会让 cumulative_sum 这类
    #: 完全因果的算子看起来有问题。
    warmup_sensitive: bool = False
    warmup_detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "category": self.category,
            "verdict": str(self.verdict),
            "detail": self.detail,
            "warmup_sensitive": self.warmup_sensitive,
            "warmup_detail": self.warmup_detail,
            "cases": [case.as_dict() for case in self.cases],
        }


def _valid_cases(
    spec: TypedOperatorSpec,
    pool: Sequence[_Candidate],
    limit: int,
) -> Iterator[tuple[tuple[_Candidate, ...], tuple[Any, ...], ValueType]]:
    """枚举该算子能接受的 (候选类型组合, 具体取值, 输出类型)。"""

    produced = 0
    for arity in sorted(spec.arities):
        if arity > 3:
            continue
        for indices in product(range(len(pool)), repeat=arity):
            combo = tuple(pool[index] for index in indices)
            try:
                output_type = spec.infer_output(tuple(item.value_type for item in combo))
            except (TypedDslError, ValueError, TypeError):
                continue
            for assignment in product(*(item.values for item in combo)):
                try:
                    spec.evaluate(*assignment)
                except Exception:  # noqa: BLE001 - 取值不合法就换一个，这是搜索不是断言
                    continue
                yield combo, assignment, output_type
                produced += 1
                break
            if produced >= limit:
                return


def audit_operator(
    spec: TypedOperatorSpec,
    pool: Sequence[_Candidate],
    *,
    input_length: int,
) -> OperatorFinding:
    cases: list[OperatorCase] = []
    warmup_detail = ""
    # 搜索合法取值时会撞上 power/log 的定义域，NumPy 的 RuntimeWarning 在这里
    # 是噪声不是信息——真正的失败会以异常或裁决的形式出现。
    with np.errstate(all="ignore"):
        for combo, assignment, output_type in _valid_cases(spec, pool, MAX_CASES_PER_OPERATOR):
            signature = f"{spec.operator_id}({', '.join(item.label for item in combo)})"
            time_indices = tuple(index for index, item in enumerate(combo) if item.has_time)
            time_axis = output_type.axes.index("time") if "time" in output_type.axes else None

            def _evaluate(args: list[Any], _spec: TypedOperatorSpec = spec) -> Any:
                return _spec.evaluate(*args)

            if not time_indices:
                outcome = ProbeOutcome(Verdict.CAUSAL, "输入不含时间轴，不存在时间方向的泄露")
            elif time_axis is None:
                outcome = tail_dependence(
                    _evaluate,
                    list(assignment),
                    time_indices=time_indices,
                    input_length=input_length,
                )
            else:
                outcome = run_tail_probe(
                    _evaluate,
                    list(assignment),
                    time_indices=time_indices,
                    input_length=input_length,
                    time_axis=time_axis,
                )
                if not warmup_detail and len(time_indices) == 1:
                    only = time_indices[0]

                    def _compute(values: np.ndarray, _args: tuple[Any, ...] = assignment, _slot: int = only) -> Any:
                        args = list(_args)
                        args[_slot] = values
                        return spec.evaluate(*args)

                    warmup = warmup_sensitivity(_compute, np.asarray(assignment[only]))
                    if warmup.verdict is Verdict.WARMUP_SENSITIVE:
                        warmup_detail = warmup.detail

            cases.append(OperatorCase(signature, str(output_type), outcome))

    if not cases:
        return OperatorFinding(
            spec.operator_id,
            spec.category,
            Verdict.UNKNOWN,
            "无法用内置数据集构造合法输入，需人工核定",
        )
    verdict = merge_verdicts([case.outcome.verdict for case in cases])
    detail = next(
        (case.outcome.detail for case in cases if case.outcome.verdict is verdict),
        "",
    )
    return OperatorFinding(
        spec.operator_id,
        spec.category,
        verdict,
        detail,
        tuple(cases),
        warmup_sensitive=bool(warmup_detail),
        warmup_detail=warmup_detail,
    )


def audit_operators(
    panel: SyntheticPanel | None = None,
    *,
    registry_version: str | None = None,
) -> tuple[OperatorFinding, ...]:
    """遍历算子注册表。"""

    panel = panel or synthetic_panel()
    pool = candidate_pool(panel)
    registry = get_typed_operator_registry(registry_version) if registry_version else get_typed_operator_registry()
    # 注册表把别名也映射到同一个 spec，直接遍历键会把算子审计两遍，
    # 基线里也会出现重复条目。以 operator_id 去重。
    specs = {spec.operator_id: spec for spec in registry.values()}
    return tuple(
        audit_operator(specs[name], pool, input_length=panel.periods)
        for name in sorted(specs)
    )


# --------------------------------------------------------------------------
# 公式级审计
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeFinding:
    node_id: int
    operator_id: str | None
    fragment: str
    output: str
    outcome: ProbeOutcome

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operator_id": self.operator_id,
            "fragment": self.fragment,
            "output": self.output,
            **self.outcome.as_dict(),
        }


@dataclass(frozen=True)
class ExpressionReport:
    expression: str
    verdict: Verdict
    output_type: str
    detail: str
    findings: tuple[NodeFinding, ...] = ()
    window_consuming: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    @property
    def blocked(self) -> bool:
        """是否应当拒绝保存。"""

        return self.verdict is Verdict.LEAK

    def as_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "verdict": str(self.verdict),
            "output_type": self.output_type,
            "detail": self.detail,
            "blocked": self.blocked,
            "window_consuming": list(self.window_consuming),
            "warnings": list(self.warnings),
            "findings": [finding.as_dict() for finding in self.findings],
        }


def _time_variables(
    requirements: Mapping[str, ValueType],
) -> tuple[tuple[str, ...], dict[int, int]]:
    """带时间轴的上下文变量，以及它们各自的尾部偏移。

    ``adjusted_nav`` 的长度符号是 ``L``（比 ``T`` 多一个起点），时点 t 之后的
    第一个下标是 ``t + 2``，所以偏移 1。
    """

    names: list[str] = []
    offsets: dict[int, int] = {}
    for name, value_type in requirements.items():
        if "time" not in value_type.axes:
            continue
        declared = DEFAULT_VARIABLE_TYPES.get(name, value_type)
        axis = declared.axes.index("time")
        if str(declared.shape[axis]) == "L":
            offsets[len(names)] = 1
        names.append(name)
    return tuple(names), offsets


def _probe_plan(
    plan: TypedExpressionPlan,
    panel: SyntheticPanel,
    decision_dates: Sequence[int] | None,
) -> ProbeOutcome:
    names, offsets = _time_variables(plan.context_requirements)
    if not names:
        return ProbeOutcome(Verdict.CAUSAL, "公式不引用任何时间序列变量")
    try:
        runtime = TypedIndicatorRuntime.from_plan(plan)
    except TypedDslError as exc:
        return ProbeOutcome(Verdict.UNKNOWN, f"无法构建运行时：{exc}")

    context = panel.context()

    def _evaluate(args: list[Any]) -> Any:
        payload = dict(context)
        payload.update(dict(zip(names, args)))
        return runtime.compute(payload)

    return run_tail_probe(
        _evaluate,
        [context[name] for name in names],
        time_indices=tuple(range(len(names))),
        input_length=panel.periods,
        tail_offsets=offsets,
        decision_dates=decision_dates,
    )


def audit_expression(
    expression: str,
    *,
    panel: SyntheticPanel | None = None,
    variable_types: Mapping[str, Any] | None = None,
    decision_dates: Sequence[int] | None = None,
    max_nodes: int = 32,
) -> ExpressionReport:
    """审计一条 typed DSL 公式。"""

    panel = panel or synthetic_panel()
    dates = tuple(decision_dates) if decision_dates is not None else default_decision_dates(panel.periods)
    try:
        plan = compose_typed_expression(
            expression, variable_types=variable_types, output_contract="any"
        )
    except TypedDslError as exc:
        return ExpressionReport(expression, Verdict.UNKNOWN, "?", f"公式无法编译：{exc}")

    by_id = {node.node_id: node for node in plan.nodes}
    root = by_id[plan.root_id]

    window_consuming: list[str] = []
    for node in plan.nodes:
        if node.operator_id is None:
            continue
        inputs_have_time = any("time" in by_id[i].inferred_type.axes for i in node.inputs)
        if inputs_have_time and "time" not in node.inferred_type.axes:
            window_consuming.append(f"#{node.node_id} {node.operator_id}: {node.formula_fragment}")

    findings: list[NodeFinding] = []
    warnings: list[str] = []

    # 带时间轴的中间节点逐个体检。根是标量时，这是唯一决定性的证据来源:
    # ``mean(...)`` 本身永远「合法」，问题藏在它里面那个读了未来的子表达式。
    audited = 0
    for node in plan.nodes:
        if node.node_id != plan.root_id:
            if node.operator_id is None or "time" not in node.inferred_type.axes:
                continue
        if audited >= max_nodes:
            warnings.append(f"节点数超过 {max_nodes}，其余节点未审计")
            break
        if node.node_id != plan.root_id and "time" not in node.inferred_type.axes:
            continue
        try:
            fragment_plan = compose_typed_expression(
                node.formula_fragment, variable_types=variable_types, output_contract="any"
            )
        except TypedDslError as exc:
            findings.append(
                NodeFinding(
                    node.node_id,
                    node.operator_id,
                    node.formula_fragment,
                    str(node.inferred_type),
                    ProbeOutcome(Verdict.UNKNOWN, f"子表达式无法单独编译：{exc}"),
                )
            )
            audited += 1
            continue
        if "time" not in fragment_plan.output_type.axes:
            # 标量子表达式（含标量根）：P1 无从比较，交给窗口画像。
            continue
        findings.append(
            NodeFinding(
                node.node_id,
                node.operator_id,
                node.formula_fragment,
                str(node.inferred_type),
                _probe_plan(fragment_plan, panel, dates),
            )
        )
        audited += 1

    verdicts = [finding.outcome.verdict for finding in findings]
    if window_consuming:
        verdicts.append(Verdict.WINDOW_CONSUMING)
    if not verdicts:
        verdicts.append(Verdict.CAUSAL)
    verdict = merge_verdicts(verdicts)

    if verdict is Verdict.LEAK:
        culprit = next(f for f in findings if f.outcome.verdict is Verdict.LEAK)
        detail = f"节点 #{culprit.node_id} `{culprit.fragment}` 读取了决策日之后的数据：{culprit.outcome.detail}"
    elif verdict is Verdict.WINDOW_CONSUMING:
        detail = (
            f"公式沿时间轴归约（{len(window_consuming)} 处），"
            "取值随窗口右端移动而变；请确认调用方传入的窗口右端不晚于决策日"
        )
    elif verdict is Verdict.UNKNOWN:
        detail = next(f.outcome.detail for f in findings if f.outcome.verdict is Verdict.UNKNOWN)
    else:
        detail = f"{len(findings)} 个带时间轴的节点全部通过尾部扰动检验"

    return ExpressionReport(
        expression,
        verdict,
        str(root.inferred_type),
        detail,
        tuple(findings),
        tuple(window_consuming),
        tuple(warnings),
    )


# --------------------------------------------------------------------------
# 基线
# --------------------------------------------------------------------------

def load_baseline(path: Path = BASELINE_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("operators", {})


def compare_to_baseline(
    findings: Sequence[OperatorFinding],
    baseline: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """返回与基线的差异。空字典表示完全一致。"""

    baseline = load_baseline() if baseline is None else baseline
    drift: dict[str, list[str]] = {"missing": [], "changed": [], "stale": []}
    seen = set()
    for finding in findings:
        seen.add(finding.operator_id)
        recorded = baseline.get(finding.operator_id)
        if recorded is None:
            drift["missing"].append(finding.operator_id)
        elif recorded.get("verdict") != str(finding.verdict):
            drift["changed"].append(
                f"{finding.operator_id}: {recorded.get('verdict')} → {finding.verdict}"
            )
    drift["stale"] = sorted(set(baseline) - seen)
    return {key: value for key, value in drift.items() if value}


def write_baseline(
    findings: Sequence[OperatorFinding],
    path: Path = BASELINE_PATH,
    *,
    notes: Mapping[str, str] | None = None,
) -> None:
    existing = load_baseline(path)
    notes = notes or {}
    payload = {
        "_comment": "算子因果性裁决基线。由 python -m causality.audit 重新生成；"
        "verdict 变化必须经人工复核后才能提交。",
        "operators": {
            finding.operator_id: {
                "verdict": str(finding.verdict),
                "category": finding.category,
                "note": notes.get(
                    finding.operator_id,
                    existing.get(finding.operator_id, {}).get("note", ""),
                ),
            }
            for finding in sorted(findings, key=lambda item: item.operator_id)
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _main() -> None:
    findings = audit_operators()
    counts: dict[str, int] = {}
    for finding in findings:
        counts[str(finding.verdict)] = counts.get(str(finding.verdict), 0) + 1
    print(f"审计算子 {len(findings)} 个：{counts}")
    for finding in findings:
        if finding.verdict in (Verdict.LEAK, Verdict.UNKNOWN, Verdict.WARMUP_SENSITIVE):
            print(f"  [{finding.verdict}] {finding.operator_id}: {finding.detail[:110]}")
    drift = compare_to_baseline(findings)
    print(f"与基线差异：{drift or '无'}")
    write_baseline(findings)
    print(f"已写入 {BASELINE_PATH}")


if __name__ == "__main__":
    _main()
