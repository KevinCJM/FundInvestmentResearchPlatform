"""未来函数 / 数据泄露的因果性审计。

PIT 封版管住**能看到哪些数据**；本模块管住**公式有没有偷看**。两者正交：
一条用了未来数据的公式，跑在最严格的 PIT 封版下，依然是错的。

判据只有一条——输出在时点 t 的取值只能是输入 ``x[0..t]`` 的函数；等价地，
改变 t 之后的输入不允许改变 t 及之前的输出。
"""

from .audit import (
    ExpressionReport,
    NodeFinding,
    OperatorCase,
    OperatorFinding,
    audit_expression,
    audit_operator,
    audit_operators,
    candidate_pool,
    compare_to_baseline,
    load_baseline,
    write_baseline,
)
from .probes import (
    ProbeOutcome,
    Verdict,
    compare_prefix,
    default_decision_dates,
    merge_verdicts,
    perturb_tail,
    run_tail_probe,
    tail_perturbation,
    warmup_sensitivity,
)
from .synthetic import (
    CAUSALITY_DATASET_VERSION,
    KNOWN_CAUSAL,
    KNOWN_LEAKY,
    SyntheticPanel,
    hostile_variants,
    synthetic_panel,
)

__all__ = [
    "CAUSALITY_DATASET_VERSION",
    "ExpressionReport",
    "KNOWN_CAUSAL",
    "KNOWN_LEAKY",
    "NodeFinding",
    "OperatorCase",
    "OperatorFinding",
    "ProbeOutcome",
    "SyntheticPanel",
    "Verdict",
    "audit_expression",
    "audit_operator",
    "audit_operators",
    "candidate_pool",
    "compare_prefix",
    "compare_to_baseline",
    "default_decision_dates",
    "hostile_variants",
    "load_baseline",
    "merge_verdicts",
    "perturb_tail",
    "run_tail_probe",
    "synthetic_panel",
    "tail_perturbation",
    "warmup_sensitivity",
    "write_baseline",
]
