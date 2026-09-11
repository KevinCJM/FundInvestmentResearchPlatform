"""Snapshot orchestration for independent scalars and explicit series channels.

Only the reducer below performs numeric work. Series formulas use their normal
prewarmed plan; snapshot reduction happens before any chart display truncation.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Iterator

import numba
import numpy as np
import pandas as pd
from numba import types

from .errors import ValidationError
from .series_provider import market_data_generation

_F1 = types.Array(types.float64, 1, "C", readonly=True)


@numba.njit(types.Tuple((types.float64, types.int64))(_F1, types.int64, types.int64),
            cache=False, nogil=True)
def last_finite_snapshot_value(values, start, end):
    """Return the last finite value and its actual aligned position, or missing."""
    if start < 0 or end > values.size or end < start:
        raise ValueError("INVALID_SNAPSHOT_WINDOW")
    for position in range(end - 1, start - 1, -1):
        if math.isfinite(values[position]):
            return values[position], position
    return np.nan, -1


last_finite_snapshot_value.disable_compile()


def _records(service: Any, configured: list[dict], targets: list[dict]) -> Iterator[tuple[dict, dict]]:
    scalars: dict[str, list[dict]] = defaultdict(list)
    series: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for item in configured:
        if item.get("channel_id"):
            if item.get("reducer") != "last_finite":
                raise ValidationError("SNAPSHOT_SERIES_REDUCER_REQUIRED", "时序快照必须明确使用末个有限值归约。")
            series[(item["period"], item["indicator_id"], item["indicator_revision"])].append(item)
        else:
            scalars[item["period"]].append(item)

    for period, items in scalars.items():
        for start in range(0, len(items), 10):
            batch = items[start:start + 10]
            refs = [{"indicator_id": item["indicator_id"], "indicator_revision": item["indicator_revision"]} for item in batch]
            # Snapshot jobs have an explicit preparation stage, just like the UI.
            # Prepare the exact slice: unrelated dependency groups may otherwise
            # be chunked differently by the overall workspace warmup.
            service.prepare_evaluation(indicator_ids=[], indicator_refs=refs)
            by_id = {item["indicator_id"]: item for item in batch}
            for target_start in range(0, len(targets), 50):
                response = service.evaluate(
                    indicator_ids=[], indicator_refs=refs, inline_definition=None,
                    targets=targets[target_start:target_start + 50], period=period,
                    include_series=False, prefer_snapshot=False,
                )
                for record in response["results"]:
                    yield by_id[record["indicator_id"]], record

    for (period, indicator_id, revision), items in series.items():
        for target in targets:
            # All configured channels of this instance share one sequence plan.
            response = service.series_service.evaluate(
                indicator_instances=[{"indicator_id": indicator_id, "indicator_revision": revision}],
                target=target, period=period, _snapshot_only=True,
            )
            record = response["results"][0]
            channels = {channel["id"]: channel for channel in record.get("snapshot_channels", [])}
            for item in items:
                channel = channels.get(item["channel_id"])
                warnings = list(record.get("warnings", []))
                value = channel.get("value") if channel else None
                if value is None:
                    warnings.append({"code": "SNAPSHOT_CHANNEL_UNAVAILABLE", "message": "所选时序通道在本次区间没有有限值。"})
                actual_date = channel.get("value_date") if channel else None
                window = dict(record.get("window", {}))
                if value is not None and actual_date != window.get("end_date"):
                    warnings.append({"code": "SNAPSHOT_LAST_FINITE_BEFORE_END", "message": f"末个有限值来自 {actual_date}，不是区间末日。"})
                yield item, {
                    **record, "value": value, "value_date": actual_date,
                    "status": ("warning" if warnings else "ok") if value is not None else "unavailable",
                    "warnings": warnings, "window": window,
                }


def configured_snapshot_values(output: pd.DataFrame, service: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build snapshot columns without conflating a scalar and a series channel."""
    generation = market_data_generation(service.market_data_dir)
    config = service.get_snapshot_config()
    configured = [item for item in config.get("items", []) if item.get("status") == "ready"]
    service.warm_snapshot_numba_plans(configured)
    result = output.copy()
    for item in configured:
        field = item["field"]
        is_date = item.get("presentation", {}).get("value_type") == "date"
        result[field] = pd.Series(None, index=result.index, dtype=object) if is_date else np.nan
        result[f"{field}__status"] = "unavailable"
        result[f"{field}__observation_count"] = 0
        for suffix in ("start_date", "end_date", "effective_as_of", "value_date", "warning_code", "warning_message"):
            result[f"{field}__{suffix}"] = None
    targets = [{"kind": str(row.instrument_type), "product_id": str(row.ts_code)}
               for row in result[["instrument_type", "ts_code"]].itertuples(index=False)]
    rows = {(str(row.instrument_type), str(row.ts_code)): row.Index
            for row in result[["instrument_type", "ts_code"]].itertuples()}
    status_counts = {"ok": 0, "warning": 0, "unavailable": 0, "error": 0}
    for item, record in _records(service, configured, targets):
        field = item["field"]
        target = record["target"]
        row = rows[(str(target["kind"]), str(target["product_id"]))]
        status = record.get("status", "error")
        status_counts[status if status in status_counts else "error"] += 1
        result.at[row, f"{field}__status"] = status
        window = record.get("window") or {}
        result.at[row, f"{field}__observation_count"] = int(window.get("observation_count") or 0)
        for suffix in ("start_date", "end_date", "effective_as_of"):
            result.at[row, f"{field}__{suffix}"] = window.get(suffix)
        result.at[row, f"{field}__value_date"] = record.get("value_date")
        warnings = record.get("warnings") or []
        if warnings:
            result.at[row, f"{field}__warning_code"] = warnings[0].get("code")
            result.at[row, f"{field}__warning_message"] = warnings[0].get("message")
        value = record.get("value")
        if item.get("presentation", {}).get("value_type") == "date":
            if isinstance(value, str):
                result.at[row, field] = pd.Timestamp(value).date().isoformat()
        elif value is not None and math.isfinite(float(value)):
            result.at[row, field] = float(value)
    if generation != market_data_generation(service.market_data_dir):
        raise ValidationError("SNAPSHOT_DATA_CHANGED", "数据在快照计算期间发生变化，请重新生成快照。")
    return result, {
        "schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_generation": generation, "config_revision": config.get("revision"),
        "configured_count": len(configured),
        "items": [{key: item.get(key) for key in (
            "field", "indicator_id", "indicator_revision", "period", "channel_id", "reducer", "name", "source", "presentation",
        )} for item in configured],
        "status_counts": status_counts,
        "failures": [{"indicator_id": str(item.get("indicator_id")), "message": str(item.get("status_message") or "指标版本不存在。")}
                     for item in config.get("items", []) if item.get("status") != "ready"],
        "legacy_columns": ["amount_avg_20d", "volume_avg_20d", "premium_discount_latest", "current_size"],
        "legacy_note": "这些列是兼容属性；新增快照指标只能从指标中心配置。",
    }
