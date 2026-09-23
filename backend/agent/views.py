"""Strict, source-specific model views for every registered tool.

This is where the semantic admission happens: each tool declares a view that
copies only the fields its own trusted contract allows, at every nesting level.
Unknown fields (including numbers, strings and containers) are omitted; numbers
are accepted only in declared numeric slots; declared text may not carry a
serialized JSON structure.  Capacity bounding happens later, after semantics.

Scalar indicator values are additionally gated by a server-resolved derivation
proof (:mod:`agent.derivation`); a row whose definition cannot be rechecked
keeps its status/coverage metadata but loses its value with an explicit reason.
"""

from __future__ import annotations

import copy
import json
import math
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

from . import series_summary
from .data_policy import user_text_violation
from .derivation import enough_samples

OMIT = object()

# Scalar schemas.  BOOL/NUM/INT reject bool-for-number and NaN/Inf numbers.
STR = ("str",)
NUM = ("num",)
INT = ("int",)
BOOL = ("bool",)

MAX_LIST_ITEMS = 200
MAX_KEYS = 200
MAX_DECLARED_TEXT = 8000


class Counter:
    """Counts omissions and remembers registry-known field labels only."""

    __slots__ = ("dropped", "fields")

    def __init__(self) -> None:
        self.dropped = 0
        self.fields: list[str] = []

    def drop(self, label: str) -> None:
        self.dropped += 1
        if len(self.fields) < 20:
            self.fields.append(label)


def _scalar(value: Any, kind: str, path: str, counter: Counter) -> Any:
    if value is None:
        return None
    if kind == "str":
        if isinstance(value, str):
            if len(value) > MAX_DECLARED_TEXT:
                counter.drop(path)
                return OMIT
            if user_text_violation(value, allow_number=path.endswith(".expression")):
                counter.drop(path)
                return OMIT
            return value
        counter.drop(path)
        return OMIT
    if kind == "num":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            counter.drop(path)
            return OMIT
        if not math.isfinite(float(value)):
            counter.drop(path)
            return OMIT
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            counter.drop(path)
            return OMIT
        return value
    if kind == "bool":
        if isinstance(value, bool):
            return value
        counter.drop(path)
        return OMIT
    raise ValueError(f"unknown schema kind: {kind}")


def project(value: Any, schema: Any, path: str, counter: Counter) -> Any:
    """Copy only what the schema allows; everything else is omitted."""

    if isinstance(schema, tuple):
        return _scalar(value, schema[0], path, counter)
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            counter.drop(path)
            return OMIT
        result: dict[str, Any] = {}
        wildcard = schema.get("*")
        for key, child in value.items():
            if key in schema and key != "*":
                child_schema = schema[key]
                child_path = f"{path}.{key}"
            elif wildcard is not None:
                if len(result) >= MAX_KEYS:
                    counter.drop(path)
                    continue
                child_schema = wildcard
                child_path = f"{path}.*"
            else:
                counter.drop(path)
                continue
            if not isinstance(key, str) or len(key) > 120:
                counter.drop(path)
                continue
            projected = project(child, child_schema, child_path, counter)
            if projected is not OMIT:
                result[key] = projected
        return result
    if isinstance(schema, list):
        if not isinstance(value, list):
            counter.drop(path)
            return OMIT
        if len(value) > MAX_LIST_ITEMS:
            counter.drop(path)
            return {"omitted": {"code": "list_too_long", "points": len(value), "note": "列表过长，已省略。"}}
        items = []
        for index, child in enumerate(value):
            projected = project(child, schema[0], f"{path}[{index}]", counter)
            if projected is not OMIT:
                items.append(projected)
        return items
    raise ValueError(f"invalid schema: {schema!r}")


# --------------------------------------------------------------------------- #
# Declared schemas
# --------------------------------------------------------------------------- #

TARGET = {"kind": STR, "product_id": STR, "name": STR}
WARNING = {"code": STR, "message": STR, "field": STR, "severity": STR}
DIAGNOSTIC = {"code": STR, "message": STR, "field": STR, "severity": STR}
COVERAGE = {"source_rows": INT, "non_null_rows": INT, "coverage_ratio": NUM,
            "first_date": STR, "latest_date": STR, "conditional": BOOL}
LINEAGE = {"label": STR, "dataset": STR, "first_date": STR, "latest_date": STR,
           "dataset_first_date": STR, "dataset_latest_date": STR, "rows_before_as_of": INT,
           "rows_after_date_filter": INT, "rows_after_as_of": INT, "availability_field": STR,
           "availability_filter": STR, "uses_disclosure_date": BOOL, "disclosure_status": STR,
           "fingerprint": STR, "source_fields": [STR]}
WINDOW = {"requested_as_of": STR, "effective_as_of": STR, "start_date": STR, "end_date": STR,
          "observation_count": INT, "data_latest_date": STR, "common_date_hash": STR,
          "data_lineage": [LINEAGE], "coverage": {"*": COVERAGE}, "source_fingerprints": {"*": STR}}
DATA_CONTEXT = {"found_date": STR, "list_date": STR, "as_of": STR, "sources": [LINEAGE]}
TARGET_DATA = {"available_datasets": [STR], "available_variables": [STR], "data_latest_date": STR,
               "fingerprints": {"*": STR}, "identity": {"kind": STR, "product_id": STR, "name": STR}}
REQUIREMENT_ITEM = {"variable_id": STR, "label": STR, "canonical_field": STR, "status": STR,
                    "reason_code": STR, "reason": STR, "source_configured": BOOL,
                    "source_dataset": STR, "source_field": STR, "coverage_ratio": NUM,
                    "non_null_count": INT, "first_date": STR, "latest_date": STR,
                    "actual_shape": [INT], "alternative_variables": [{"variable_id": STR, "label": STR}]}
INPUT_REQUIREMENTS = {"status": STR, "required_count": INT, "available_count": INT,
                      "items": [REQUIREMENT_ITEM], "reason": {"code": STR, "message": STR}}
PARAMETER_SCHEMA_ITEM = {"id": STR, "name": STR, "label": STR, "type": STR, "default": NUM,
                         "minimum": NUM, "maximum": NUM, "exclusive_minimum": BOOL,
                         "exclusive_maximum": BOOL, "step": NUM, "description": STR, "value": NUM,
                         "source": STR, "required": BOOL}
SERIES_OUTPUT = {"id": STR, "label": STR, "expression": STR, "unit": STR, "display_format": STR,
                 "precision": INT, "output_measure": STR}
ROLLING_SOURCE = {"kind": STR, "indicator_id": STR, "indicator_revision": INT, "indicator_name": STR,
                  "definition_hash": STR, "source_definition_hash": STR, "source_dsl_version": STR,
                  "window_observations": INT, "minimum_observations": INT, "detached": BOOL,
                  "transform_version": STR, "version": STR}
# The editor/runtime definition contract (IndicatorDraft); authored numeric constants live
# only in declared numeric slots such as annual_risk_free_rate_percent and parameters.
DRAFT_DEFINITION = {
    "name": STR, "description": STR, "expression": STR, "periods": [STR], "period_policy": STR,
    "unit": STR, "display_format": STR, "precision": INT, "direction": STR, "indicator_type": STR,
    "annual_risk_free_rate_percent": NUM, "dsl_version": STR, "operator_registry_version": STR,
    "numeric_kernel_version": STR, "variable_registry_version": STR, "data_contract_version": STR,
    "context_schema_version": STR, "context_kind": STR, "result_kind": STR, "output_contract": STR,
    "output_measure": STR, "parameter_contract_version": STR,
    "parameter_schema": [PARAMETER_SCHEMA_ITEM], "fixed_parameters": [PARAMETER_SCHEMA_ITEM],
    "series_outputs": [SERIES_OUTPUT], "axis_anchor": STR, "history_policy": STR,
    "lookback_parameter": STR, "minimum_observations": INT, "methodology": STR, "data_basis": STR,
    "rolling_source": ROLLING_SOURCE,
}
VALUE_RANGE = {"bounded": BOOL, "minimum": NUM, "maximum": NUM, "reason": STR}
CHANNEL_SUMMARY = {"id": STR, "label": STR, "unit": STR, "display_format": STR, "precision": INT,
                   "output_measure": STR, "semantic_dimension": STR, "price_basis": STR,
                   "value_range": VALUE_RANGE, "point_count": INT, "finite_count": INT,
                   "null_count": INT, "zero_count": INT, "mean": NUM, "std": NUM,
                   "minimum": NUM, "maximum": NUM, "kernel": STR}
SCALAR_ROW = {
    "status": STR, "target": TARGET, "value": NUM, "value_type": STR, "unit": STR,
    "display_format": STR, "precision": INT, "presentation": {"name": STR, "unit": STR,
                                                              "precision": INT, "value_type": STR},
    "value_type": STR, "parameters": {"*": NUM}, "parameter_hash": STR, "period": STR,
    "window": WINDOW, "data_context": DATA_CONTEXT, "target_data": TARGET_DATA,
    "input_requirements": INPUT_REQUIREMENTS, "warnings": [WARNING],
    "indicator_id": STR, "indicator_revision": INT, "indicator_name": STR, "result_kind": STR,
    "series_available": BOOL, "series_omitted": {"code": STR, "points": INT, "note": STR},
    "value_omitted": {"code": STR, "reason": STR, "definition_ref": STR},
}
SERIES_ROW = {key: value for key, value in SCALAR_ROW.items()
              if key not in {"value", "value_type", "unit", "display_format", "precision", "period"}}
SERIES_ROW.update({"channels": [CHANNEL_SUMMARY], "date_range": {"start": STR, "end": STR},
                   "observation_count": INT, "axis_anchor": STR, "instance_key": STR})

CATALOG_ITEM = {"id": STR, "name": STR, "description": STR, "revision": INT, "context_kind": STR,
                "result_kind": STR, "indicator_type": STR, "unit": STR, "display_format": STR,
                "source": STR, "category": STR, "annual_risk_free_rate_percent": NUM,
                "methodology": STR}
CATALOG_PAYLOAD = {"catalog_version": STR, "dsl_version": STR, "kind": STR, "items": [CATALOG_ITEM],
                   "matched_count": INT, "total": INT, "returned_count": INT, "hint": STR}
VARIABLE_ITEM = {"name": STR, "label": STR, "description": STR, "latex": STR, "latex_template": STR,
                 "value_type": STR, "shape": STR, "unit": STR, "price_basis": STR, "semantic": STR,
                 "semantic_role": STR, "source_field": STR, "source_dataset": STR, "dtype": STR,
                 "frequency": STR, "data_basis": STR, "source": STR, "domains": [STR],
                 "product_kinds": [STR], "signature": STR, "return_type": STR,
                 "mathematical_essence": STR,
                  "parameters": [PARAMETER_SCHEMA_ITEM]}
VARIABLE_ITEM.update({"id": STR, "transform": STR, "availability_rule": STR, "missing_policy": STR,
                      "source_bindings": {kind: {"configured": BOOL, "source_dataset": STR,
                                                  "source_field": STR, "transform": STR, "reason": STR}
                                          for kind in ("etf", "fund")}})
CATALOG_PAYLOAD_VARIABLE = {**CATALOG_PAYLOAD, "items": [VARIABLE_ITEM]}
INFER_PAYLOAD = {"expression": STR, "editable_latex": STR, "latex": STR, "display_latex": STR,
                 "python_expression": STR, "normalized_expression": STR, "inferred_type": STR,
                 "shape": STR, "dependencies": [STR], "semantic_warnings": [WARNING],
                 "dsl_version": STR, "operator_registry_version": STR, "math_notation_version": STR,
                 "parameter_schema": [PARAMETER_SCHEMA_ITEM], "intermediate_kind": STR}
DRAFT_SUMMARY = {"valid": BOOL, "draft_revision": INT, "definition_hash": STR, "result_kind": STR,
                 "display_latex": STR, "editable_latex": STR, "diagnostics": [DIAGNOSTIC],
                 "compile_token_issued": BOOL}
ROLLING_DRAFT = {"valid": BOOL, "draft_revision": INT, "source": ROLLING_SOURCE, "expression": STR,
                 "result_kind": STR, "parameter_schema": [PARAMETER_SCHEMA_ITEM],
                 "annual_risk_free_rate_percent": NUM, "methodology": STR, "diagnostics": [DIAGNOSTIC]}
SEARCH_PAYLOAD = {"items": [TARGET], "note": STR}
PLAN_ITEM = {"id": STR, "name": STR, "revision": INT, "product_kind": STR, "indicator_count": INT,
             "target_count": INT}
PLANS_PAYLOAD = {"items": [PLAN_ITEM], "total": INT}
PORTFOLIO_CONTEXT = {"id": STR, "created_at": STR, "immutable": BOOL, "target_id": STR,
                     "target_revision": INT, "target_name": STR,
                     "requested_as_of": STR, "effective_as_of": STR, "actual_start_date": STR,
                     "actual_end_date": STR, "observation_count": INT, "common_date_hash": STR,
                     "data_fingerprints": {"*": STR}}
AVAILABILITY_ITEM = {"target": TARGET, "variable_id": STR, "label": STR, "status": STR,
                     "reason": {"code": STR, "message": STR}, "reason_code": STR, "coverage": COVERAGE,
                     "coverage_ratio": NUM, "non_null_count": INT, "actual_shape": [INT],
                     "source_dataset": STR, "source_field": STR, "available_count": INT,
                     "requested_count": INT, "alternative_variables": [{"variable_id": STR, "label": STR}]}
AVAILABILITY_PAYLOAD = {"target": TARGET, "targets": [TARGET], "status": STR, "items": [AVAILABILITY_ITEM],
                        "available_count": INT, "required_count": INT, "period": STR, "as_of": STR,
                        "common_date_hash": STR, "variable_ids": [STR]}


# --------------------------------------------------------------------------- #
# Projection facts and per-tool views
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Projection:
    """Server-resolved facts for one handler result; never client-supplied."""

    proof_map: Mapping[Any, dict] = field(default_factory=dict)
    verified_rows: Tuple[Any, ...] = ()
    resolutions: Tuple[dict, ...] = ()


def _counter_view(schema: Any):
    def view(payload: Any, projection: Projection) -> tuple[Any, int]:
        counter = Counter()
        projected = project(payload, schema, "result", counter)
        if projected is OMIT:
            return {}, max(1, counter.dropped)
        return projected, counter.dropped
    return view


def _evaluation_view(default_kind: str):
    def view(payload: Any, projection: Projection) -> tuple[Any, int]:
        return project_evaluation_result(payload, projection, default_kind=default_kind)
    return view


def _page_view(payload: Any, projection: Projection) -> tuple[Any, int]:
    """The page.read handler already produced a strict section projection."""

    schema = {"available": BOOL, "page": STR, "snapshot_id": STR, "captured_at": STR,
              "section": STR, "offset": INT, "limit": INT, "total_chars": INT,
              "has_more": BOOL, "next_offset": INT, "content_is_json_text": BOOL,
              "section_redactions": INT, "evidence_kind": STR, "trust": STR, "note": STR,
              "message": STR}
    counter = Counter()
    result = project(payload, schema, "page", counter)
    # Only this handler's already-projected page text is paginated. Historical
    # receipts are never admitted through this view (they require a current seal).
    if isinstance(result, dict) and isinstance(payload.get("content"), str):
        result["content"] = payload["content"]
    return ({} if result is OMIT else result), counter.dropped


VIEW_CATALOG_INDICATORS = _counter_view(CATALOG_PAYLOAD)
VIEW_CATALOG_REFERENCE = _counter_view(CATALOG_PAYLOAD_VARIABLE)
VIEW_INFER = _counter_view(INFER_PAYLOAD)
VIEW_DRAFT_SUMMARY = _counter_view(DRAFT_SUMMARY)
VIEW_ROLLING_DRAFT = _counter_view(ROLLING_DRAFT)
VIEW_SEARCH = _counter_view(SEARCH_PAYLOAD)
VIEW_PLANS = _counter_view(PLANS_PAYLOAD)
VIEW_PORTFOLIO_CONTEXT = _counter_view(PORTFOLIO_CONTEXT)
VIEW_AVAILABILITY = _counter_view(AVAILABILITY_PAYLOAD)
VIEW_PAGE_READ = _page_view
VIEW_EVALUATION_SCALAR = _evaluation_view("scalar")
VIEW_EVALUATION_SERIES = _evaluation_view("time_series")


def _channel_summary(channel: dict[str, Any], *, derived: bool, counter: Counter,
                     trusted_metadata: bool = True) -> dict[str, Any]:
    values = channel.get("values")
    metadata = {key: item for key, item in CHANNEL_SUMMARY.items()
                if key not in {"point_count", "finite_count", "null_count", "zero_count",
                               "mean", "std", "minimum", "maximum", "kernel", "value_range"}}
    if not trusted_metadata:
        metadata.pop("precision", None)
    base = project(channel, metadata, "result.channels[]", counter)
    if base is OMIT:
        base = {}
    if isinstance(values, list):
        summary = (series_summary.summarize(values) if derived else series_summary.coverage_only(values)) or {}
        if summary.get("finite_count", 0) < 2:
            summary = {key: item for key, item in summary.items()
                       if key not in {"mean", "std", "minimum", "maximum"}}
        for key in ("point_count", "finite_count", "null_count", "zero_count", "mean", "std",
                    "minimum", "maximum", "kernel"):
            if key in summary:
                base[key] = summary[key]
    return base


def _value_omitted(reason: str, definition_ref: Optional[str]) -> dict[str, Any]:
    marker = {"code": "value_unproven", "reason": reason}
    if definition_ref:
        marker["definition_ref"] = definition_ref
    return marker


def project_result_row(row: dict[str, Any], proof: Optional[dict], *, kind: str,
                       counter: Counter) -> dict[str, Any]:
    """One evaluation/service result row under its declared contract."""

    schema = SERIES_ROW if kind == "time_series" else SCALAR_ROW
    projected = project(row, schema, "result.results[]", counter)
    if projected is OMIT:
        return {}
    approved = enough_samples(row, proof)
    allowed_parameters = set((proof or {}).get("parameters", []))
    projected["parameters"] = {key: value for key, value in (projected.get("parameters") or {}).items()
                               if key in allowed_parameters}
    if kind == "time_series":
        channels = row.get("channels")
        if isinstance(channels, list):
            projected["channels"] = [
                _channel_summary(channel, derived=enough_samples(
                    row, (proof or {}).get("channels", {}).get(channel.get("id"), proof)), counter=counter)
                for channel in channels if isinstance(channel, dict)]
        dates = row.get("dates")
        if isinstance(dates, list) and dates:
            projected["date_range"] = {"start": dates[0] if isinstance(dates[0], str) else None,
                                       "end": dates[-1] if isinstance(dates[-1], str) else None}
            projected["observation_count"] = len(dates)
        if not approved:
            projected["value_omitted"] = _value_omitted(
                (proof or {}).get("code", "definition_unverified"),
                (proof or {}).get("definition_ref"))
    else:
        value = row.get("value")
        series = row.get("series")
        if series is not None:
            projected["series_available"] = bool(series)
            projected["series_omitted"] = {"code": "raw_series_hidden",
                                           "points": len(series) if isinstance(series, list) else None,
                                           "note": "完整序列仅保留在后端与页面。"}
        if approved:
            # 0 and explicit null stay distinguishable; NaN/Inf cannot be a value.
            if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)
                                 and math.isfinite(float(value))):
                projected["value"] = value
            else:
                projected.pop("value", None)
                projected["value_omitted"] = _value_omitted("non_finite_value",
                                                            (proof or {}).get("definition_ref"))
        else:
            projected.pop("value", None)
            if "value" in row and value is None:
                projected["value"] = None
            projected["value_omitted"] = _value_omitted(
                (proof or {}).get("code", "definition_unverified"),
                (proof or {}).get("definition_ref"))
    return projected


def project_evaluation_result(payload: Any, projection: Projection, *, default_kind: str
                              ) -> tuple[Any, int]:
    """Registered evaluation result: strict rows plus proven values only."""

    counter = Counter()
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return {}, 1
    rows = []
    for row in payload["results"]:
        if not isinstance(row, dict):
            counter.drop("result.results[]")
            continue
        kind = (row.get("result_kind") if row.get("result_kind") in {"scalar", "time_series"}
                else "time_series" if "channels" in row or "dates" in row else default_kind)
        proof = None
        if projection.proof_map:
            proof = (projection.proof_map.get((row.get("indicator_id"), row.get("indicator_revision")))
                     or projection.proof_map.get((None, None)))
        rows.append(project_result_row(row, proof, kind=kind, counter=counter))
    out: dict[str, Any] = {"results": rows}
    summary = project(payload.get("summary"), {"total": INT, "ok": INT, "warning": INT,
                                               "unavailable": INT, "error": INT},
                      "result.summary", counter)
    if summary is not OMIT and summary:
        out["summary"] = summary
    execution = project(payload.get("execution"), {"nopython": BOOL, "python_fallback": INT,
                                                   "python_operator_calls": INT,
                                                   "executed_batches": INT, "parallel_tasks": INT,
                                                   "compile_cache_hits": INT, "compiled_plan_ids": [STR]},
                        "result.execution", counter)
    if execution is not OMIT and execution:
        out["execution"] = execution
    return out, counter.dropped


# --------------------------------------------------------------------------- #
# Page evidence sections (client contract), all strictly filtered
# --------------------------------------------------------------------------- #

EDITING = {
    "selection": {"indicator_id": STR, "indicator_revision": INT, "name": STR, "read_only": BOOL},
    "definition": {"source": STR, **DRAFT_DEFINITION},
    "active_preview": {"source": STR, "definition": DRAFT_DEFINITION, "result_adopted": BOOL,
                       "preview_id": STR, "run_id": STR, "definition_hash": STR, "context_hash": STR,
                       "data_generation": STR, "effective_context": {"*": STR},
                       "created_at": STR, "note": STR,
                       "runtime_inputs": {"targets": [TARGET], "period": STR, "as_of": STR,
                                          "runtime_parameters": {"*": NUM}}},
    "runtime_inputs": {"targets": [TARGET], "period": STR, "as_of": STR,
                       "runtime_parameters": {"*": NUM}},
    "state": {"canvas_pending": BOOL, "parameter_pending": BOOL, "previewing": BOOL,
              "validation_valid": BOOL, "diagnostics": [DIAGNOSTIC]},
}
RESULTS = {
    "displayed_source": STR, "pending": [STR], "provenance": {"source": STR, "preview_id": STR,
                                                              "requested_at": STR, "completed_at": STR,
                                                              "run_id": STR, "definition_hash": STR,
                                                              "context_hash": STR, "note": STR},
    "frozen_request": {"definition": DRAFT_DEFINITION, "parameters": {"*": NUM},
                       "parameters_submitted": BOOL, "targets": [TARGET], "period": STR, "as_of": STR,
                       "requested_at": STR, "completed_at": STR},
    "frozen_definition": DRAFT_DEFINITION,
    "groups": [{"target": TARGET, "status": STR, "value_type": STR, "unit": STR, "precision": INT,
                "parameters": {"*": NUM}, "parameter_hash": STR, "window": WINDOW,
                "data_context": DATA_CONTEXT, "target_data": TARGET_DATA,
                "input_requirements": INPUT_REQUIREMENTS, "warnings": [WARNING],
                "series_available": BOOL}],
}
SERIES = {"displayed_source": STR,
          "groups": [{"target": TARGET, "status": STR, "parameters": {"*": NUM},
                      "parameter_hash": STR, "window": WINDOW, "data_context": DATA_CONTEXT,
                      "warnings": [WARNING], "unavailable": {"code": STR, "message": STR},
                      "dates": [STR], "channels": [CHANNEL_SUMMARY]}],
          "note": STR}


def definition_view(value: Any) -> dict[str, Any]:
    """Authoring numbers require the actual definition contract, not a client label."""
    from services.custom_indicator_contracts import ValidateRequest
    counter = Counter()
    value = project(value, DRAFT_DEFINITION, "definition", counter)
    if value is OMIT:
        return {}
    try:
        return ValidateRequest.model_validate(value).model_dump(exclude_unset=True,
            exclude={"template_origin", "rolling_transform"})
    except (ValueError, TypeError):
        return {}


def parameter_view(value: Any, definition: dict) -> dict:
    schema = {item["id"]: item for item in definition.get("parameter_schema", [])}
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items()
            if key in schema and isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(item) and schema[key]["minimum"] <= item <= schema[key]["maximum"]}


def _frozen_view(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    definition = definition_view(value.get("definition"))
    result = project(value, {"targets": [TARGET], "period": STR, "as_of": STR,
                            "parameters_submitted": BOOL, "requested_at": STR, "completed_at": STR},
                     "frozen_request", Counter())
    result["definition"] = definition
    result["parameters"] = parameter_view(value.get("parameters"), definition)
    return result


def project_results_section(value: Any, facts: Projection) -> tuple[Any, int]:
    counter = Counter()
    frame = project(value, {"displayed_source": STR, "pending": [STR],
                            "provenance": RESULTS["provenance"]}, "results", counter)
    if frame is OMIT:
        return {}, max(1, counter.dropped)
    frozen = _frozen_view(value.get("frozen_request"))
    definition = definition_view(value.get("frozen_definition")) or frozen.get("definition", {})
    frozen["definition"] = definition
    frame.update(frozen_request=frozen, frozen_definition=definition)
    groups = value.get("groups")
    frame["groups"] = []
    if isinstance(groups, list):
        for index, group in enumerate(groups[:MAX_LIST_ITEMS]):
            if not isinstance(group, dict):
                continue
            base = project(group, {"target": TARGET, "status": STR, "warnings": [WARNING],
                                   "series_available": BOOL}, "results.groups[]", counter)
            verified = facts.verified_rows[index] if index < len(facts.verified_rows) else None
            if verified:
                base["server_verified"] = verified
                base["provenance"] = {"resolution": "server_verified"}
            else:
                base["client_claim"] = {"status": base.pop("status", None),
                                        "value_present": group.get("value") is not None}
                base["provenance"] = {"resolution": "client_only"}
                base["explain_with"] = "page.recompute"
                counter.drop("results.groups[].value")
            frame["groups"].append(base)
    return frame, counter.dropped


def project_series_section(value: Any, facts: Projection) -> tuple[Any, int]:
    counter = Counter()
    frame = project(value, {"displayed_source": STR, "note": STR}, "series", counter)
    if frame is OMIT:
        return {}, max(1, counter.dropped)
    frame["groups"] = []
    for group in (value.get("groups") or [])[:MAX_LIST_ITEMS]:
        if not isinstance(group, dict):
            continue
        base = project(group, {"target": TARGET, "status": STR}, "series.groups[]", counter)
        dates = group.get("dates")
        if isinstance(dates, list) and dates:
            base["observation_count"] = len(dates)
            try:
                base["date_range"] = {"start": date.fromisoformat(dates[0]).isoformat(),
                                      "end": date.fromisoformat(dates[-1]).isoformat()}
            except (TypeError, ValueError):
                counter.drop("series.date_range")
        # No numeric labels from the page. Counts come from the actual list; source
        # classification remains client-only and values/extrema never cross.
        base["channels"] = [_channel_summary(channel, derived=False, counter=counter, trusted_metadata=False)
                            for channel in (group.get("channels") or [])[:MAX_LIST_ITEMS]
                            if isinstance(channel, dict)]
        base["provenance"] = {"resolution": "client_only"}
        frame["groups"].append(base)
    return frame, counter.dropped


def project_editing_section(value: Any, facts: Projection) -> tuple[Any, int]:
    counter = Counter()
    projected = project(value, EDITING, "editing", counter)
    if projected is OMIT:
        return {}, max(1, counter.dropped)
    definition = definition_view(projected.get("definition"))
    projected["definition"] = definition
    dirty = (value.get("definition") or {}).get("definition_dirty")
    if isinstance(dirty, bool):
        projected["definition_dirty"] = dirty
    preview = projected.get("active_preview")
    if isinstance(preview, dict):
        preview["definition"] = definition_view(preview.get("definition"))
        if isinstance(preview.get("runtime_inputs"), dict):
            preview["runtime_inputs"]["runtime_parameters"] = parameter_view(
                preview["runtime_inputs"].get("runtime_parameters"), preview["definition"])
    active = preview["definition"] if isinstance(preview, dict) else definition
    if isinstance(projected.get("runtime_inputs"), dict):
        projected["runtime_inputs"]["runtime_parameters"] = parameter_view(
            projected["runtime_inputs"].get("runtime_parameters"), active)
    return projected, counter.dropped


def project_page_section(section: str, value: Any, facts: Projection) -> tuple[Any, int]:
    if section == "results":
        return project_results_section(value, facts)
    if section == "series":
        return project_series_section(value, facts)
    return project_editing_section(value, facts)


def project_artifact_row(artifact: Any, service: Any) -> Optional[dict[str, Any]]:
    """Server-owned preview artifact: project one row under a resolved proof.

    The artifact was computed by the existing service and is scoped to the
    session; its definition is re-proven before any value is exposed.
    """

    if not isinstance(artifact, dict):
        return None
    result = artifact.get("result")
    rows = result.get("results") if isinstance(result, dict) else None
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return None
    definition = artifact.get("definition")
    proof = None
    if isinstance(definition, dict):
        from . import derivation

        proof = derivation.definition_proof(
            service, definition,
            context_kind=str(definition.get("context_kind") or "single_product"),
            definition_ref="preview:" + str(artifact.get("preview_id") or "unknown"))
    row = rows[0]
    kind = "time_series" if "channels" in row or "dates" in row else "scalar"
    return project_result_row(row, proof, kind=kind, counter=Counter())

# Task/recall state originates in SQLite, not a model-supplied status object.
SOURCE_STATEMENT = {'id': STR, 'run_id': STR, 'seq': INT, 'text': STR, 'kind': STR, 'edit_of': STR}
CONSTRAINT = {'source_message_id': STR, 'quote': STR, 'supersedes_source_message_id': STR,
              'source_verified': BOOL, 'selection_status': STR}
PLAN = {'id': STR, 'source_message_id': STR, 'capabilities': [STR], 'questions': [STR], 'status': STR,
        'constraints': [CONSTRAINT]}
MILESTONE = {'ref': STR, 'tool': STR, 'source_message_id': STR, 'current': BOOL, 'status': STR,
             'operation': STR,
             'code': STR, 'definition_hash': STR, 'result_statuses': [STR],
             'dependencies': {'catalog_version': STR, 'data_generation': STR, 'calculation_hash': STR}}
TASK_STATE = {'schema_version': INT, 'policy_version': STR, 'revision': STR,
              'session_id': STR, 'scope': STR, 'precedence': STR,
              'sources': [SOURCE_STATEMENT], 'latest_source_id': STR, 'plans': [PLAN],
              'quoted_constraints': [CONSTRAINT],
              'pending_questions': [STR], 'milestones': [MILESTONE], 'rejected_strategies': [MILESTONE],
              'goal_status': STR, 'dependencies': {'catalog_version': STR, 'data_generation': STR, 'calculation_hash': STR},
              'page_context': {'page': STR, 'page_instance_id': STR, 'context_revision': INT, 'view_state': STR,
                  'calculation': {'context_kind': STR, 'targets': [TARGET], 'period': STR, 'as_of': STR, 'run_id': STR}},
              'current_draft': {'draft_revision': INT, 'definition_hash': STR, 'valid': BOOL, 'stale': BOOL},
              'best_valid_draft': {'draft_revision': INT, 'definition_hash': STR, 'valid': BOOL, 'stale': BOOL}}
TASK_STATE.update({'source_count': INT, 'omitted_source_count': INT, 'source_read_tool': STR, 'source_note': STR,
                   'milestones_count': INT, 'rejected_strategies_count': INT,
                   'section': STR, 'offset': INT, 'total': INT, 'next_offset': INT})
VIEW_TASK_STATE = _counter_view(TASK_STATE)
VIEW_PLAN = _counter_view(PLAN)
VIEW_MEMORY_PROPOSAL = _counter_view({'proposal_id': STR, 'kind': STR, 'key': STR, 'object_id': STR,
    'source_message_id': STR, 'summary': STR, 'scope': STR, 'status': STR, 'created_at': STR})
