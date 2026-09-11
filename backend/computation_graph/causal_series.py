"""Scope-aware series validation and system-context binding for graph consumers.

This is orchestration, not another rolling implementation. Numerical work stays
in the Indicator Center's prepared Typed DAG / NJIT plan.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any
import math

import numpy as np

from cal_indicators.rolling_scope import SCALAR_CONTEXT, analyze_interval
from cal_indicators.typed_dsl import DEFAULT_VARIABLE_TYPES, TypedDslError, ValueType
from cal_indicators.typed_operators import get_typed_operator_registry

SYSTEM_CONTEXT_NAMES = SCALAR_CONTEXT | {'observation_dates'}
INTERVAL_CONTEXT_NAMES = frozenset({'observation_count', 'window_elapsed_days', 'risk_free_return_window'})


def series_variable_types(names: Iterable[str], definition: Mapping[str, Any] | None = None) -> dict[str, ValueType]:
    """User ports and protected system bindings are distinct namespaces."""
    from custom_indicators.variable_registry import variable_types
    registered = {**DEFAULT_VARIABLE_TYPES, **variable_types('single_product')}
    names = tuple(names)
    forbidden = sorted(set(names) & SYSTEM_CONTEXT_NAMES)
    if forbidden:
        raise TypedDslError('SYSTEM_CONTEXT_OVERRIDE', f'系统上下文不能作为用户输入端口：{forbidden[0]}。')
    result = {name: ValueType.series('T') for name in names}
    if definition is not None:
        from custom_indicators.variable_registry import get_variable
        # Preserve nominal return/level axes and measures of locked indicators.
        for name in names:
            variable = get_variable(name)
            if variable is not None:
                result[name] = variable.value_type()
    result.update({name: registered[name] for name in SYSTEM_CONTEXT_NAMES})
    return result


def outside_value_nodes(plan) -> tuple[Any, ...]:
    """Visit outer value uses, not the deferred body (even with shared nodes)."""
    nodes = {node.node_id: node for node in plan.nodes}
    pending = list(plan.roots.values()) if hasattr(plan, 'roots') else [plan.root_id]
    used = set()
    while pending:
        index = pending.pop()
        if index in used:
            continue
        used.add(index)
        node = nodes[index]
        pending.extend(node.inputs[1:] if node.operator_id == 'rolling_apply' else node.inputs)
    return tuple(node for node in plan.nodes if node.node_id in used)


def causal_violations(plan, allowed_operators: Iterable[str]) -> list[str]:
    """A reducer is causal only when its actual value use is window-scoped."""
    allowed = set(allowed_operators)
    nodes = {node.node_id: node for node in plan.nodes}
    registry = get_typed_operator_registry(plan.operator_registry_version)
    violations = set()
    for node in outside_value_nodes(plan):
        operator = node.operator_id
        if node.kind == 'variable' and node.label in INTERVAL_CONTEXT_NAMES:
            violations.add(node.label)  # Full-input context cannot be broadcast.
        if operator == 'rolling_apply':
            analyze_interval(plan.nodes, node.inputs[0], registry)
            continue
        if operator == 'rolling_window':
            continue  # Non-publishable intermediate; consumers are checked below.
        if operator and node.inputs and nodes[node.inputs[0]].inferred_type.kind == 'window':
            # The typed registry has already checked the consumer's window
            # signature; this is not a mean/std whitelist and not scalar lifting.
            if node.inferred_type.kind == 'series':
                continue
        if operator is not None and operator not in allowed:
            violations.add(operator)
    return sorted(violations)


def bind_series_context(context_names: Iterable[str], inputs: Mapping[str, np.ndarray],
                        dates: np.ndarray, definition: Mapping[str, Any] | None = None) -> tuple[Any, ...]:
    """Bind a real common date axis and locked constants without copying prices.

    Epoch-day int64 -> float64 conversion is a boundary allocation. Callers may
    pass a reusable float64 axis to avoid repeating it across nodes/channels.
    """
    from custom_indicators.runtime_context import risk_free_context_kernel
    if set(inputs) & SYSTEM_CONTEXT_NAMES:
        raise TypedDslError('SYSTEM_CONTEXT_OVERRIDE', '系统日期与年度配置不能由用户连线覆盖。')
    if not isinstance(dates, np.ndarray) or dates.ndim != 1 or dates.dtype not in (np.dtype('int64'), np.dtype('float64')):
        raise TypedDslError('SERIES_DATE_AXIS_INVALID', '系统日期必须使用真实观察日的一维 epoch-day 数组。')
    names = tuple(context_names)
    for name in names:
        if name in SYSTEM_CONTEXT_NAMES:
            continue
        value = inputs.get(name)
        if (not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 1
                or value.size != dates.size or not value.flags.c_contiguous):
            raise TypedDslError('SERIES_INPUT_ALIGNMENT', f'{name} 必须是在输入边界对齐的一维连续 float64 数组。')
    annual_percent = float((definition or {}).get('annual_risk_free_rate_percent') or 0.0)
    if not math.isfinite(annual_percent) or not -100 <= annual_percent <= 100:
        raise TypedDslError('INVALID_RISK_FREE_RATE', '锁定指标的年化无风险利率无效。')
    annual, per_observation, legacy, _, periods = risk_free_context_kernel(annual_percent, 0.0)
    system = dict(annual_risk_free_rate_decimal=float(annual), periods_per_year=float(periods),
                  risk_free_rate_per_observation=float(per_observation), risk_free_rate_per_period=float(legacy),
                  observation_count=0.0, window_elapsed_days=0.0, risk_free_return_window=0.0)
    if 'observation_dates' in names:
        system['observation_dates'] = dates.astype(np.float64, copy=False)
    return tuple(system[name] if name in system else inputs[name] for name in names)
