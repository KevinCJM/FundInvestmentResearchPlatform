"""Named scalar definitions and stable, revision-bound output references."""
from __future__ import annotations

import copy
import re
from typing import Any, Callable, Mapping

from .errors import ValidationError
from .presentation import metric_presentation

SCALAR_BUNDLE = "scalar_bundle"
MAX_SCALAR_OUTPUTS = 8
OUTPUT_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,79}$")
DIRECTIONS = {"neutral", "higher_better", "lower_better"}


def is_scalar_bundle(definition: Mapping[str, Any]) -> bool:
    return definition.get("result_kind") == SCALAR_BUNDLE


def normalize_scalar_bundle(
    fields: dict[str, Any],
    previous: dict[str, Any],
    normalize_single: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    outputs = fields.get("scalar_outputs")
    if not isinstance(outputs, list) or not 1 <= len(outputs) <= MAX_SCALAR_OUTPUTS:
        raise ValidationError("INVALID_SCALAR_OUTPUTS", "请保留 1 至 8 个结果。", field="scalar_outputs")
    if fields.get("output_contract", SCALAR_BUNDLE) != SCALAR_BUNDLE:
        raise ValidationError("INVALID_OUTPUT_CONTRACT", "多结果标量必须使用 scalar_bundle 输出契约。", field="output_contract")
    if fields.get("output_schema_version", 1) != 1:
        raise ValidationError("UNSUPPORTED_OUTPUT_SCHEMA", "不支持此结果结构版本。", field="output_schema_version")
    base_fields = {**fields, "result_kind": "scalar", "output_contract": "scalar", "expression": "1", "direction": "higher_better"}
    base_previous = {**previous, "result_kind": "scalar", "output_contract": "scalar"}
    base = normalize_single(base_fields, base_previous)
    if not str(base.get("dsl_version", "")).startswith("2."):
        raise ValidationError("SCALAR_OUTPUTS_REQUIRE_TYPED", "多结果指标需要使用类型化公式。", field="dsl_version")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    retired = set(previous.get("retired_output_ids") or [])
    for index, raw in enumerate(outputs):
        field = f"scalar_outputs.{index}"
        if not isinstance(raw, dict):
            raise ValidationError("INVALID_SCALAR_OUTPUT", "结果定义必须是对象。", field=field)
        output_id = str(raw.get("id") or "").strip()
        label = str(raw.get("label") or "").strip()
        if not OUTPUT_ID.fullmatch(output_id):
            raise ValidationError("INVALID_OUTPUT_ID", "结果编码无效，请重新添加结果。", field=f"{field}.id")
        if output_id in seen or output_id in retired:
            raise ValidationError("DUPLICATE_OUTPUT_ID", "结果编码重复或已退役，不能用于另一个结果。", field=f"{field}.id")
        if not label or len(label) > 80:
            raise ValidationError("INVALID_OUTPUT_LABEL", "结果名称需要 1 至 80 个字符。", field=f"{field}.label")
        direction = str(raw.get("direction") or "neutral")
        if direction not in DIRECTIONS:
            raise ValidationError("INVALID_OUTPUT_DIRECTION", "请选择仅展示、越高越好或越低越好。", field=f"{field}.direction")
        try:
            checked = normalize_single({
                **base,
                **{key: raw[key] for key in ("expression", "display_format", "precision", "unit") if key in raw},
                "expression": str(raw.get("expression") or ""), "name": label,
                "direction": direction if direction != "neutral" else "higher_better",
                "result_kind": "scalar", "output_contract": "scalar",
            })
        except ValidationError as exc:
            raise ValidationError(exc.code, f"{label}：{exc.message}", field=f"{field}.{exc.field or 'expression'}") from exc
        normalized.append({
            "id": output_id, "label": label,
            **{key: checked[key] for key in ("expression", "display_format", "precision", "unit")},
            "description": str(raw.get("description") or "").strip()[:500],
            "direction": direction,
        })
        seen.add(output_id)
    retired.update(item["id"] for item in previous.get("scalar_outputs", []) if item["id"] not in seen)
    return {
        **base, "expression": "", "direction": "neutral",
        "result_kind": SCALAR_BUNDLE, "output_contract": SCALAR_BUNDLE,
        "output_schema_version": 1, "output_measure": SCALAR_BUNDLE,
        "scalar_outputs": normalized, "retired_output_ids": sorted(retired),
        "series_outputs": [],
    }


def project_scalar_output(definition: dict[str, Any], output_id: str | None) -> dict[str, Any]:
    """A projection is not a separately persisted indicator or a mutable alias."""
    if definition.get("result_kind") == "time_series":
        raise ValidationError("INDICATOR_RESULT_KIND_MISMATCH", "时序指标不能直接作为标量结果使用。", field="output_id")
    if not is_scalar_bundle(definition):
        if output_id not in {None, "", "value"}:
            raise ValidationError("OUTPUT_NOT_FOUND", "此单值指标没有所选结果。", field="output_id")
        return definition
    if not output_id:
        raise ValidationError("OUTPUT_REQUIRED", "请选择这个指标中的具体结果，不能直接使用整个指标组。", field="output_id")
    output = next((item for item in definition.get("scalar_outputs", []) if item["id"] == output_id), None)
    if output is None:
        raise ValidationError("OUTPUT_NOT_FOUND", "所选结果不属于此指标版本，请重新选择。", field="output_id")
    result = {
        **definition,
        **{key: value for key, value in output.items() if key not in {"id", "label"}},
        "id": definition.get("id"), "output_id": output_id,
        "parent_indicator_name": definition.get("name", ""),
        "name": f"{definition.get('name', '')} · {output['label']}",
        "result_kind": "scalar", "output_contract": "scalar",
        "scalar_outputs": [], "output_label": output["label"],
    }
    result["presentation"] = metric_presentation(result)
    return result


def apply_scalar_output_contract(definition: dict[str, Any], validation: dict[str, Any]) -> None:
    contracts = validation.get("output_inferences") or {}
    for output in definition["scalar_outputs"]:
        contract = contracts.get(output["id"], {})
        output["required_variables"] = list(contract.get("dependencies") or [])
        output["output_measure"] = str(contract.get("output_measure") or "dimensionless")
        output["semantic_dimension"] = output["output_measure"]
        for key in ("display_latex", "editable_latex", "math_notation_version"):
            if contract.get(key) is not None:
                output[key] = contract[key]
    definition["required_variables"] = list(validation.get("dependencies") or [])
    definition["output_measure"] = SCALAR_BUNDLE
    definition["compiled_plan_id"] = validation.get("compiled_plan_id")
    definition["numeric_kernel_version"] = validation.get("kernel_version")
    definition["applicable_product_kinds"] = ["portfolio"] if definition.get("context_kind") == "portfolio" else ["etf", "fund"]


def output_result(result: dict[str, Any], output_id: str | None) -> dict[str, Any]:
    if result.get("result_kind") != SCALAR_BUNDLE:
        if output_id not in {None, "", "value"}:
            raise ValidationError("OUTPUT_NOT_FOUND", "此结果没有所选输出。", field="output_id")
        return result
    if not output_id:
        raise ValidationError("OUTPUT_REQUIRED", "请选择具体结果。", field="output_id")
    output = next((item for item in result.get("outputs", []) if item["output_id"] == output_id), None)
    if output is None:
        raise ValidationError("OUTPUT_NOT_FOUND", "此版本没有所选结果。", field="output_id")
    return {
        **{key: value for key, value in result.items() if key not in {"outputs", "presentation"}},
        **copy.deepcopy(output), "result_kind": "scalar",
        "indicator_name": output["presentation"]["name"],
    }
