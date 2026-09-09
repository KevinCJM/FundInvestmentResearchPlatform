"""Independent scalar metrics over one shared, last-maximum drawdown interval."""
from __future__ import annotations

from cal_indicators.typed_numba_kernels import NUMERIC_KERNEL_VERSION
from cal_indicators.typed_operators import TYPED_DSL_VERSION, TYPED_OPERATOR_REGISTRY_VERSION
from .periods import SUPPORTED_PERIODS
from .variable_registry import CONTEXT_SCHEMA_VERSION, DATA_CONTRACT_VERSION, VARIABLE_REGISTRY_VERSION

DRAWDOWN_RATE_ID = "builtin-last-maximum-drawdown-rate"
_INTERVAL = "last_drawdown_interval(drawdown_series(adjusted_nav))"
_START = f"value_at(observation_dates, interval_start({_INTERVAL}))"
_TROUGH = f"value_at(observation_dates, interval_trough({_INTERVAL}))"
_RECOVERY = f"value_at(observation_dates, interval_recovery({_INTERVAL}))"


def independent_drawdown_indicators():
    common = {
        "revision": 1, "source": "built_in", "read_only": True,
        "created_at": "2026-09-08T00:00:00+00:00", "updated_at": "2026-09-08T00:00:00+00:00",
        "context_kind": "single_product", "indicator_type": "path",
        "result_kind": "scalar", "output_contract": "scalar",
        "series_outputs": [], "parameter_schema": [], "fixed_parameters": [],
        "annual_risk_free_rate_percent": 0.0, "periods": list(SUPPORTED_PERIODS),
        "period_policy": "all_supported", "minimum_observations": 1,
        "applicable_product_kinds": ["etf", "fund"],
        "availability_policy": "runtime_required", "availability_status": "runtime_check",
        "dsl_version": TYPED_DSL_VERSION, "operator_registry_version": TYPED_OPERATOR_REGISTRY_VERSION,
        "numeric_kernel_version": NUMERIC_KERNEL_VERSION, "variable_registry_version": VARIABLE_REGISTRY_VERSION,
        "data_contract_version": DATA_CONTRACT_VERSION, "context_schema_version": CONTEXT_SCHEMA_VERSION,
        "template_origin": None,
        "data_basis": "真实复权净值及同轴日期；缺失不填充；并列最大回撤选择最后谷底。",
        "methodology": "先生成净值回撤序列，再选最后一次最大回撤区间。p为此前最后峰值位置，τ为最后最大回撤谷底，ρ为此后首次恢复位置。日期差按自然日计量。",
    }
    specs = (
        (DRAWDOWN_RATE_ID, "最大回撤率", "negate(min_value(drawdown_series(adjusted_nav)))", "return_decimal", "%", "percent", 2, "lower_better", "最大下跌幅度；无回撤时为0。"),
        ("builtin-maximum-drawdown-start-date", "最大回撤开始日期", _START, "date", "", "date", 0, "neutral", "最后一次最大回撤谷底前，最后一次历史峰值日期。"),
        ("builtin-maximum-drawdown-trough-date", "最大回撤谷底日期", _TROUGH, "date", "", "date", 0, "neutral", "最大回撤下跌阶段结束日期；并列最大回撤取最后谷底，不是恢复日期。"),
        ("builtin-maximum-drawdown-duration-days", "最大回撤持续天数", f"days_between({_START}, {_TROUGH})", "calendar_days", "天", "number", 0, "neutral", "所选最大回撤区间从峰值到谷底的自然日间隔，不是最长水下时间。"),
        ("builtin-maximum-drawdown-recovery-date", "最大回撤恢复日期", _RECOVERY, "date", "", "date", 0, "neutral", "所选谷底后首次恢复原峰值的日期；尚未恢复则不可用。"),
        ("builtin-maximum-drawdown-recovery-days", "最大回撤恢复天数", f"days_between({_TROUGH}, {_RECOVERY})", "calendar_days", "天", "number", 0, "neutral", "所选谷底到恢复日期的自然日间隔；尚未恢复则不可用。"),
        ("builtin-maximum-drawdown-total-days", "最大回撤完整持续天数", f"days_between({_START}, {_RECOVERY})", "calendar_days", "天", "number", 0, "neutral", "所选峰值到恢复日期的自然日间隔；尚未恢复则不可用，不以截止日伪装恢复日。"),
    )
    return [{**common, "id": key, "name": name, "description": description,
             "expression": expression, "output_measure": measure, "unit": unit,
             "display_format": display, "precision": precision, "direction": direction,
             "required_variables": ["adjusted_nav"] + (["observation_dates"] if "observation_dates" in expression else [])}
            for key, name, expression, measure, unit, display, precision, direction, description in specs]
