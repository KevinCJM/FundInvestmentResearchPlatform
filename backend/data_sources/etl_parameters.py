"""Bind per-run options without changing the saved ETL definition."""
from __future__ import annotations

from datetime import date

from pydantic import ValidationError

from .etl_models import EtlDefinition, EtlRunOptions
from .models import CenterError


def parse_run_options(payload: object = None) -> EtlRunOptions:
    try:
        return EtlRunOptions.model_validate({} if payload is None else payload)
    except ValidationError:
        raise CenterError("ETL_RUN_OPTIONS_INVALID", "请选择全量或增量，运行参数必须为文本键值。", 422) from None


def bind_parameters(definition: EtlDefinition, options: EtlRunOptions) -> EtlDefinition:
    """Resolve a bounded set of named values; never evaluate expressions."""
    declared = {parameter.id: parameter for parameter in definition.parameters}
    if set(options.parameters) - declared.keys():
        raise CenterError("ETL_PARAMETER_UNKNOWN", "运行请求包含未声明的参数。", 422)
    values: dict[str, str] = {}
    for identifier, parameter in declared.items():
        value = options.parameters.get(identifier, parameter.default).strip()
        if not value:
            if parameter.required:
                raise CenterError("ETL_PARAMETER_REQUIRED", f"请填写运行参数：{parameter.label}。", 422)
            continue
        if len(value) > 200:
            raise CenterError("ETL_PARAMETER_INVALID", f"运行参数过长：{parameter.label}。", 422)
        if parameter.data_type == "date":
            try:
                parsed = date.fromisoformat(value)
            except ValueError:
                raise CenterError("ETL_PARAMETER_INVALID", f"请填写有效日期：{parameter.label}。", 422) from None
            value = parsed.strftime("%Y%m%d") if parameter.date_format == "compact" else parsed.isoformat()
        values[identifier] = value
    bound = definition.model_copy(deep=True)
    for step in bound.steps:
        for key, identifier in step.parameter_bindings.items():
            # Unfilled optional parameters must not fall back to a stale value.
            step.params.pop(key, None)
            if identifier in values:
                step.params[key] = values[identifier]
    return bound


def download_mode(step_mode: str, options: EtlRunOptions) -> str:
    return options.mode if step_mode == "inherit" else step_mode
