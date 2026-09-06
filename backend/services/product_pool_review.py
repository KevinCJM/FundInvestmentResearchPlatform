"""Read model for product-pool candidate review tables.

The mutable product-pool aggregate owns review decisions. Product master data
and metric snapshots remain external read-only evidence and are joined only for
presentation, so publishing a pool still freezes the reviewed aggregate rather
than duplicating market-data tables into the draft.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

from product_pools.errors import ProductPoolValidationError
from product_pools.repository import ProductPoolRepository
from services.instrument_analytics import (
    load_product_review_snapshot,
    snapshot_metric_definitions,
)

try:
    from backend.market_data import resolve_tushare_data_dir
except ModuleNotFoundError:  # pragma: no cover - backend/ direct execution
    from market_data import resolve_tushare_data_dir


ProductKind = Literal["etf", "fund"]
SnapshotLoader = Callable[[ProductKind, Path], tuple[pd.DataFrame, dict[str, Any]]]
MetricDefinitionLoader = Callable[[Path], dict[str, dict[str, Any]]]

MAX_BASIC_FIELDS = 10
MAX_SNAPSHOT_METRICS = 8
PERCENT_SNAPSHOT_FIELDS = {
    "return_1m",
    "return_3m",
    "return_1y",
    "return_3y",
    "annual_volatility_1y",
    "max_drawdown_3y",
    "premium_discount_latest",
}

BASIC_FIELD_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {"field": "management", "label": "管理人", "data_type": "text", "unit": None},
    {"field": "custodian", "label": "托管人", "data_type": "text", "unit": None},
    {"field": "fund_type", "label": "基金类型", "data_type": "text", "unit": None},
    {"field": "type", "label": "产品类别", "data_type": "text", "unit": None},
    {"field": "invest_type", "label": "投资类型", "data_type": "text", "unit": None},
    {"field": "qdii_type", "label": "QDII 属性", "data_type": "text", "unit": None},
    {"field": "market", "label": "交易市场", "data_type": "text", "unit": None},
    {"field": "status", "label": "产品状态", "data_type": "text", "unit": None},
    {"field": "benchmark", "label": "业绩基准", "data_type": "text", "unit": None},
    {"field": "issue_amount", "label": "发行规模", "data_type": "number", "unit": "project_normalized_wan"},
    {"field": "m_fee", "label": "管理费率", "data_type": "number", "unit": "percent_points"},
    {"field": "c_fee", "label": "托管费率", "data_type": "number", "unit": "percent_points"},
    {"field": "exp_return", "label": "预期收益率", "data_type": "number", "unit": "percent_points"},
    {"field": "duration_year", "label": "存续年限", "data_type": "number", "unit": "years"},
    {"field": "min_amount", "label": "最低认购金额", "data_type": "number", "unit": "number"},
    {"field": "list_date", "label": "上市日期", "data_type": "date", "unit": None},
    {"field": "found_date", "label": "成立日期", "data_type": "date", "unit": None},
    {"field": "issue_date", "label": "发行日期", "data_type": "date", "unit": None},
    {"field": "due_date", "label": "到期日期", "data_type": "date", "unit": None},
    {"field": "purc_startdate", "label": "申购起始日", "data_type": "date", "unit": None},
    {"field": "redm_startdate", "label": "赎回起始日", "data_type": "date", "unit": None},
)
BASIC_FIELDS_BY_ID = {item["field"]: item for item in BASIC_FIELD_DEFINITIONS}


def _serialize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
        parsed = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(parsed) else parsed.strftime("%Y-%m-%d")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip()
    return text or None


def _clean_requested_fields(
    raw_fields: list[str] | None,
    *,
    allowed: set[str],
    limit: int,
    field_name: str,
) -> list[str]:
    fields = list(dict.fromkeys(str(item).strip() for item in (raw_fields or []) if str(item).strip()))
    if len(fields) > limit:
        raise ProductPoolValidationError(
            "TOO_MANY_REVIEW_COLUMNS",
            f"候选复核表最多选择 {limit} 个{field_name}。",
            field=field_name,
        )
    invalid = [field for field in fields if field not in allowed]
    if invalid:
        raise ProductPoolValidationError(
            "INVALID_REVIEW_COLUMN",
            f"不支持的{field_name}: {', '.join(invalid)}。",
            field=field_name,
        )
    return fields


def _row_lookup(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if frame.empty:
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for record in frame.to_dict(orient="records"):
        for raw_key in (record.get("ts_code"), record.get("code")):
            key = str(raw_key or "").strip().lower()
            if key:
                lookup[key] = record
    return lookup


class ProductPoolReviewDataService:
    """Joins pool members with product master data and validated snapshots."""

    def __init__(
        self,
        repository: ProductPoolRepository,
        market_data_root: Path,
        *,
        snapshot_loader: SnapshotLoader = load_product_review_snapshot,
        definition_loader: MetricDefinitionLoader = snapshot_metric_definitions,
    ) -> None:
        self.repository = repository
        self.market_data_root = Path(market_data_root)
        self.snapshot_loader = snapshot_loader
        self.definition_loader = definition_loader

    def _data_dir(self) -> Path:
        return resolve_tushare_data_dir(self.market_data_root)

    @staticmethod
    def _member_kinds(pool: dict[str, Any]) -> set[ProductKind]:
        return {
            str(member.get("kind"))
            for member in pool.get("members", [])
            if str(member.get("kind")) in {"etf", "fund"}
        }  # type: ignore[return-value]

    def _snapshot_fields(
        self,
        definitions: dict[str, dict[str, Any]],
        member_kinds: set[ProductKind],
        states: dict[str, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for field, definition in definitions.items():
            applicable = [
                kind
                for kind in definition.get("applicable_product_kinds", ["etf", "fund"])
                if kind in {"etf", "fund"}
            ]
            relevant_kinds = [kind for kind in applicable if kind in member_kinds]
            if member_kinds and not relevant_kinds:
                continue
            presentation = definition.get("presentation") or {}
            display_format = presentation.get("display_format")
            if display_format not in {"percent", "number"}:
                display_format = "percent" if field in PERCENT_SNAPSHOT_FIELDS else "number"
            available = any(
                states.get(kind, {}).get("status") == "ready"
                and states.get(kind, {}).get("metric_availability", {}).get(field, "ready") == "ready"
                for kind in (relevant_kinds or applicable)
            )
            result.append(
                {
                    "field": field,
                    "label": str(definition.get("label") or field),
                    "source": "snapshot",
                    "data_type": "number",
                    "unit": definition.get("unit"),
                    "display_format": display_format,
                    "metric_type": definition.get("metric_type", "other"),
                    "metric_type_label": definition.get("metric_type_label", "其他指标"),
                    "description": definition.get("source", ""),
                    "applicable_product_kinds": applicable,
                    "available": available,
                }
            )
        return result

    def get_review_data(
        self,
        pool_id: str,
        *,
        basic_fields: list[str] | None = None,
        snapshot_metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        pool = self.repository.get_pool(pool_id)
        member_kinds = self._member_kinds(pool)
        data_dir = self._data_dir()

        states: dict[str, dict[str, Any]] = {}
        snapshots: dict[str, pd.DataFrame] = {}
        for kind in sorted(member_kinds):
            frame, state = self.snapshot_loader(kind, data_dir)
            snapshots[kind] = frame
            states[kind] = state

        definitions = self.definition_loader(data_dir)
        snapshot_catalog = self._snapshot_fields(definitions, member_kinds, states)
        allowed_snapshot = {item["field"] for item in snapshot_catalog}
        selected_basic = _clean_requested_fields(
            basic_fields,
            allowed=set(BASIC_FIELDS_BY_ID),
            limit=MAX_BASIC_FIELDS,
            field_name="基础信息",
        )
        selected_snapshot = _clean_requested_fields(
            snapshot_metrics,
            allowed=allowed_snapshot,
            limit=MAX_SNAPSHOT_METRICS,
            field_name="快照指标",
        )

        info_lookups: dict[str, dict[str, dict[str, Any]]] = {}
        snapshot_lookups: dict[str, dict[str, dict[str, Any]]] = {}
        for kind in sorted(member_kinds):
            info_path = data_dir / ("etf_info_df.parquet" if kind == "etf" else "fund_info_df.parquet")
            info_frame = pd.read_parquet(info_path) if info_path.exists() else pd.DataFrame()
            info_lookups[kind] = _row_lookup(info_frame)
            snapshot_lookups[kind] = _row_lookup(snapshots.get(kind, pd.DataFrame()))

        rows: list[dict[str, Any]] = []
        for member in pool.get("members", []):
            kind = str(member.get("kind") or "")
            product_id = str(member.get("product_id") or "").strip()
            code = str(member.get("code") or "").strip()
            candidates = [value.lower() for value in (product_id, code) if value]
            info = next(
                (info_lookups.get(kind, {}).get(candidate) for candidate in candidates if info_lookups.get(kind, {}).get(candidate)),
                {},
            )
            snapshot = next(
                (snapshot_lookups.get(kind, {}).get(candidate) for candidate in candidates if snapshot_lookups.get(kind, {}).get(candidate)),
                {},
            )
            snapshot_values: dict[str, Any] = {}
            snapshot_dates: dict[str, Any] = {}
            snapshot_statuses: dict[str, Any] = {}
            snapshot_warnings: dict[str, Any] = {}
            for field in selected_snapshot:
                value = _serialize(snapshot.get(field))
                snapshot_values[field] = value
                date_field = f"{field}__effective_as_of"
                snapshot_dates[field] = _serialize(
                    snapshot.get(date_field)
                    if date_field in snapshot
                    else snapshot.get("current_size_as_of") if field == "current_size" else snapshot.get("as_of")
                )
                state = states.get(kind, {})
                raw_status = _serialize(snapshot.get(f"{field}__status"))
                if raw_status:
                    snapshot_statuses[field] = raw_status
                elif state.get("status") != "ready":
                    snapshot_statuses[field] = state.get("status")
                else:
                    snapshot_statuses[field] = "available" if value is not None else "missing"
                snapshot_warnings[field] = _serialize(snapshot.get(f"{field}__warning_message"))
            rows.append(
                {
                    "key": member.get("key"),
                    "kind": kind,
                    "product_id": product_id,
                    "basic_values": {
                        field: _serialize(info.get(field)) for field in selected_basic
                    },
                    "snapshot_values": snapshot_values,
                    "snapshot_value_dates": snapshot_dates,
                    "snapshot_statuses": snapshot_statuses,
                    "snapshot_warnings": snapshot_warnings,
                }
            )

        return {
            "pool_id": pool_id,
            "pool_revision": int(pool.get("revision") or 0),
            "basic_fields": [
                {**definition, "source": "basic", "available": True, "applicable_product_kinds": ["etf", "fund"]}
                for definition in BASIC_FIELD_DEFINITIONS
            ],
            "snapshot_metric_fields": snapshot_catalog,
            "selected_basic_fields": selected_basic,
            "selected_snapshot_metrics": selected_snapshot,
            "snapshot_states": states,
            "rows": rows,
        }


__all__ = [
    "BASIC_FIELD_DEFINITIONS",
    "MAX_BASIC_FIELDS",
    "MAX_SNAPSHOT_METRICS",
    "ProductPoolReviewDataService",
]
