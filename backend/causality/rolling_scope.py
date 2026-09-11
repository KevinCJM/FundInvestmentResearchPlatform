"""Probe deferred rolling bodies through their actual compiled scope."""
from __future__ import annotations

import numpy as np

from cal_indicators.typed_numba_plan import compile_numba_plan
from .probes import ProbeOutcome, Verdict, run_tail_probe


def probe_scoped_plan(plan, panel, decision_dates=None):
    """Align synthetic observations once; never probe a body as global history."""
    try:
        compiled = compile_numba_plan(plan)
        size = panel.periods
        context = panel.context()
        # The chart service provides date-aligned arrays, including an explicit
        # first unavailable return. Here use a common right-aligned test axis.
        for name, kind in plan.context_requirements.items():
            if kind.rank:
                if name == "observation_dates":
                    context[name] = np.arange(size, dtype=np.float64) + 20_000.0
                elif name in context:
                    context[name] = np.ascontiguousarray(np.asarray(context[name], dtype=np.float64)[-size:])
                else:
                    return ProbeOutcome(Verdict.UNKNOWN, f"审计样本缺少输入 {name}")
            elif name not in context:
                return ProbeOutcome(Verdict.UNKNOWN, f"审计样本未指定参数 {name}")
        names = tuple(name for name, kind in plan.context_requirements.items()
                      if kind.rank and name != "observation_dates")
        def evaluate(args):
            bound = dict(context)
            bound.update(zip(names, args))
            return compiled.compute(tuple(bound[name] for name in compiled.context_names))
        baseline = evaluate([context[name] for name in names])
        if not np.isfinite(baseline).any():
            return ProbeOutcome(Verdict.UNKNOWN, "滚动审计未生成任何有限结果，不能仅凭全空序列判为因果。")
        return run_tail_probe(evaluate, [context[name] for name in names],
            time_indices=tuple(range(len(names))), input_length=size, decision_dates=decision_dates)
    except Exception as exc:
        return ProbeOutcome(Verdict.UNKNOWN, f"滚动作用域审计失败：{type(exc).__name__}: {exc}")


def audit_scope_operator(spec, input_length):
    from .audit import OperatorFinding, OperatorCase
    from .synthetic import synthetic_panel
    from .probes import merge_verdicts
    from cal_indicators.typed_dsl import compose_typed_expression
    panel = synthetic_panel()
    cases = []
    # Different graph structures, not just a mean alias. These tests execute
    # the same compiled loop used by the production time-series service.
    for body in ("mean(returns)", "std(returns, 1)",
                 "-min_value(drawdown_series(adjusted_nav))",
                 "-mean_where(returns, less_equal(returns, quantile(returns, 0.05)))"):
        expression = f"rolling_apply({body}, 10)"
        plan = compose_typed_expression(expression, output_contract="any")
        cases.append(OperatorCase(expression, str(plan.output_type), probe_scoped_plan(plan, panel)))
    verdict = merge_verdicts([case.outcome.verdict for case in cases])
    return OperatorFinding(spec.operator_id, spec.category, verdict,
        "完整区间计算图的已编译滚动作用域尾部扰动测试", tuple(cases))
