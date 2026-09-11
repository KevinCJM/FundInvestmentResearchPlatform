"""Deterministic whole-record arbitration over canonical, comparable records."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from typing import Any

import numpy as np

try:
    from backend.data_model.catalog import TABLES_BY_ID
except ModuleNotFoundError:
    from data_model.catalog import TABLES_BY_ID
from .models import CenterError
from .mapping import convert
from .resolution_models import ResolutionConfig
from .resolution_kernels import difference_flags, quality_flags, warm_resolution_kernels

LINEAGE = {"source_id", "source_batch_id", "source_record_id", "source_record_hash", "revision", "ingested_at", "recorded_at", "vintage_id", "available_at", "availability_status", "announced_at"}
POSITIVE = {"open", "high", "low", "close", "previous_close", "settlement_price", "unit_nav", "accumulated_nav", "adjusted_nav"}
NONNEGATIVE = {"volume", "turnover_amount", "net_assets", "total_assets", "total_shares"}
REASONS = {1: "REQUIRED_VALUE_MISSING", 2: "VALUE_OUT_OF_RANGE", 4: "OHLC_INCONSISTENT", 8: "SUSPICIOUS_JUMP"}


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("时点必须携带时区")
    return result.astimezone(timezone.utc)


def _day(value: Any) -> str:
    return value.isoformat()[:10] if isinstance(value, (date, datetime)) else str(value)[:10]


def _business_payload(row: dict) -> str:
    return json.dumps({k: v for k, v in row.items() if k not in LINEAGE}, sort_keys=True, default=str, allow_nan=True)


def resolve_records(table_id: str, rows: list[dict[str, Any]], config: ResolutionConfig,
                    *, as_of: str | None = None, source_issues: dict[str, str] | None = None,
                    unverified_batches: frozenset[str] = frozenset()) -> dict:
    definition = TABLES_BY_ID.get(table_id)
    if definition is None or not definition.source_mappable:
        raise CenterError("RESOLUTION_TARGET_INVALID", "取值规则只能应用于外部业务表。")
    if len(rows) > 100000:
        raise CenterError("RESOLUTION_LIMIT", "单次合并最多 100000 条候选，请按日期或标的缩小范围。")
    cutoff = _timestamp(as_of)
    rule = config.for_table(table_id)
    priority = rule.source_priority or config.default_source_priority
    numeric = [f.name for f in definition.fields if f.data_type in {"float64", "int64"} or f.data_type.startswith("decimal128")]
    fields = {f.name: f for f in definition.fields}
    compared = rule.compare_fields or [f.name for f in definition.fields if f.source_mappable and f.role in {"measure", "dimension"}]
    # Currency and adjustment are compatibility dimensions, not a tolerance.
    keys = [key for key in definition.primary_key if key not in LINEAGE]
    if "currency" in fields and "currency" not in keys:
        keys.append("currency")
    if table_id == "market.quote_daily":
        keys.append("adjustment_basis")
    if table_id == "market.nav_daily" and "adjusted_nav" in set(compared + rule.required_fields):
        keys.extend(["adjustment_method", "is_retrospective_adjustment"])
    valid_rows, excluded = [], []
    schema_errors: dict[int, list[str]] = {}
    for original in rows:
        if not isinstance(original, dict):
            raise CenterError("RESOLUTION_RECORD_INVALID", "每条标准记录必须为对象。")
        row = dict(original)
        if set(row) - fields.keys():
            raise CenterError("RESOLUTION_UNKNOWN_FIELD", "候选包含未知字段，不能绕过标准表合同。")
        if row.get("source_id") not in priority:
            excluded.append({"source_id": row.get("source_id"), "reason": "SOURCE_NOT_SELECTED"})
            continue
        if any(row.get(key) is None or row.get(key) == "" for key in keys):
            excluded.append({"source_id": row.get("source_id"), "reason": "IDENTITY_OR_BASIS_MISSING"})
            continue
        # Validation belongs at the boundary, not in the numeric kernels. Keep
        # invalid candidates visible so invalid/missing fallback policies differ.
        problems = ["LEGACY_CANDIDATE_UNVERIFIED"] if row.get("source_batch_id") in unverified_batches else []
        for name, field in fields.items():
            if name not in row and field.nullable:
                continue
            try:
                if field.data_type != "json":
                    row[name] = convert(row.get(name), field)
            except (ValueError, TypeError, ArithmeticError):
                problems.append("REQUIRED_VALUE_MISSING" if row.get(name) is None or row.get(name) == "" else "TYPE_INVALID")
        schema_errors[id(row)] = list(dict.fromkeys(problems))
        try:
            if type(row.get("revision", 1)) is not int or row.get("revision", 1) < 1:
                raise ValueError("修订序号无效")
            captured = _timestamp(row.get("ingested_at"))
            published = _timestamp(row.get("available_at"))
            availability = row.get("availability_status", "UNKNOWN")
            # Unknown/estimated publication is only safe once captured. DATE_ONLY
            # is already conservatively converted to end-of-day by the mapper.
            known_at = published if availability in {"EXACT", "DATE_ONLY"} and published else captured
            if captured is None:
                raise ValueError()
            if cutoff is not None and (known_at is None or known_at > cutoff or captured > cutoff):
                excluded.append({"source_id": row.get("source_id"), "reason": "NOT_KNOWN_AS_OF"})
                continue
            if cutoff is not None and row.get("adjusted_nav") is not None and row.get("is_retrospective_adjustment"):
                excluded.append({"source_id": row.get("source_id"), "reason": "RETROSPECTIVE_ADJUSTMENT"})
                continue
            date_key = "trade_date" if table_id == "market.quote_daily" else "valuation_date" if table_id == "market.nav_daily" else None
            if date_key and _day(row[date_key]) > (cutoff or datetime.now(timezone.utc)).date().isoformat():
                excluded.append({"source_id": row.get("source_id"), "reason": "FUTURE_OBSERVATION"})
                continue
            valid_rows.append((row, tuple(_day(row[k]) if fields[k].data_type == "date32" else row[k] for k in keys), known_at or captured, captured))
        except (ValueError, TypeError, KeyError):
            excluded.append({"source_id": row.get("source_id"), "reason": "INVALID_TIMESTAMP_OR_KEY"})
    by_key = defaultdict(lambda: defaultdict(list))
    for row, key, known, captured in valid_rows:
        by_key[key][row["source_id"]].append((known, captured, int(row.get("revision", 1)), row))

    candidates, candidate_keys, ambiguous = [], [], set()
    for key in sorted(by_key, key=lambda k: tuple(map(str, k))):
        for source in priority:
            versions = by_key[key].get(source, [])
            if not versions:
                continue
            # Old format records cannot prove their original checksum. They are
            # rejection markers only; a new verified capture supersedes them for
            # this business key without deleting historical audit artifacts.
            verified = [v for v in versions if v[3].get("source_batch_id") not in unverified_batches]
            versions = verified or versions
            newest = max((v[0], v[1], v[2]) for v in versions)
            latest = [v[3] for v in versions if v[:3] == newest]
            if len({_business_payload(v) for v in latest}) > 1:
                ambiguous.add((key, source))
            row = sorted(latest, key=lambda r: str(r.get("source_batch_id") or ""))[0]
            candidate_keys.append(key)
            candidates.append(row)

    values = np.full((len(candidates), len(numeric)), np.nan, dtype=np.float64)
    conversion_failed = set()
    for i, row in enumerate(candidates):
        for j, name in enumerate(numeric):
            if row.get(name) is not None:
                try:
                    values[i, j] = float(row[name])
                except (ValueError, TypeError, OverflowError):
                    conversion_failed.add(i)
    required = np.array([name in rule.required_fields for name in numeric], dtype=np.bool_)
    minimum = np.array([0.0 if name in NONNEGATIVE else -np.inf for name in numeric], dtype=np.float64)
    maximum = np.full(len(numeric), np.inf)
    positive = np.array([name in POSITIVE for name in numeric], dtype=np.bool_)
    for bound in rule.field_rules:
        if bound.field in numeric:
            j = numeric.index(bound.field)
            if bound.minimum is not None:
                minimum[j] = max(minimum[j], bound.minimum)
            if bound.maximum is not None:
                maximum[j] = bound.maximum
    ohlc = np.array([numeric.index(k) for k in ("open", "high", "low", "close")] if table_id == "market.quote_daily" else [], dtype=np.int64)
    jump_name = "close" if table_id == "market.quote_daily" else "unit_nav"
    previous = np.full(len(candidates), np.nan)
    history = {}
    chronological = sorted(range(len(candidates)), key=lambda i: str(candidates[i].get("trade_date") or candidates[i].get("valuation_date") or ""))
    for i in chronological:
        row = candidates[i]
        history_key = (row.get("source_id"), row.get("instrument_id"), row.get("currency"), row.get("adjustment_basis"), row.get("adjustment_method"))
        if table_id == "market.quote_daily" and row.get("previous_close") is not None:
            try:
                previous[i] = float(row["previous_close"])
            except (ValueError, TypeError):
                pass
        elif history_key in history:
            previous[i] = history[history_key]
        if jump_name in numeric:
            history[history_key] = values[i, numeric.index(jump_name)]
    flags = quality_flags(values, required, minimum, maximum, positive, ohlc, previous,
                          numeric.index(jump_name) if jump_name in numeric else -1,
                          float(rule.max_relative_jump or 0))
    grouped = defaultdict(dict)
    reasons = []
    for i, (key, row) in enumerate(zip(candidate_keys, candidates)):
        problems = [*schema_errors.get(id(row), []), *[label for bit, label in REASONS.items() if flags[i] & bit]]
        if i in conversion_failed:
            problems.append("TYPE_INVALID")
        if any(row.get(name) in (None, "") for name in rule.required_fields if name not in numeric):
            problems.append("REQUIRED_VALUE_MISSING")
        if (key, row["source_id"]) in ambiguous:
            problems.append("SAME_SOURCE_CONFLICT")
        if source_issues and row["source_id"] in source_issues:
            problems.append(source_issues[row["source_id"]])
        problems = list(dict.fromkeys(problems))
        reasons.append(problems)
        grouped[key][row["source_id"]] = i

    float_compares = [n for n in compared if fields[n].data_type == "float64"]
    exact_compares = [n for n in compared if n not in float_compares]
    overrides = {item.field: item for item in rule.field_rules}
    abs_tol = np.array([overrides[n].absolute_tolerance if n in overrides and overrides[n].absolute_tolerance is not None else rule.absolute_tolerance for n in float_compares], dtype=np.float64)
    rel_tol = np.array([overrides[n].relative_tolerance if n in overrides and overrides[n].relative_tolerance is not None else rule.relative_tolerance for n in float_compares], dtype=np.float64)
    selected, decisions = [], []
    for key, source_rows in grouped.items():
        skipped, chosen, blocked = [], None, False
        for source in priority:
            i = source_rows.get(source)
            if i is None:
                reason = (source_issues or {}).get(source, "PRIMARY_RECORD_MISSING")
                skipped.append({"source_id": source, "reasons": [reason]})
                permitted = rule.fallback_on_invalid if source in (source_issues or {}) else rule.fallback_on_missing
                if not permitted:
                    blocked = True
                    break
                continue
            if reasons[i]:
                skipped.append({"source_id": source, "reasons": reasons[i]})
                missing_only = set(reasons[i]) == {"REQUIRED_VALUE_MISSING"}
                if not (rule.fallback_on_missing if missing_only else rule.fallback_on_invalid):
                    blocked = True
                    break
                continue
            chosen = i
            break
        conflicts = []
        if chosen is not None:
            for other in source_rows.values():
                if other == chosen or reasons[other]:
                    continue
                left = np.array([values[chosen, numeric.index(n)] for n in float_compares], dtype=np.float64)
                right = np.array([values[other, numeric.index(n)] for n in float_compares], dtype=np.float64)
                different = difference_flags(left, right, abs_tol, rel_tol)
                names = [n for n, differs in zip(float_compares, different) if differs]
                names += [n for n in exact_compares if candidates[chosen].get(n) is not None and candidates[other].get(n) is not None and candidates[chosen][n] != candidates[other][n]]
                if names:
                    conflicts.append({"source_id": candidates[other]["source_id"], "fields": names,
                                      "selected_values": {n: candidates[chosen].get(n) for n in names},
                                      "other_values": {n: candidates[other].get(n) for n in names}})
        status = "BLOCKED" if blocked or chosen is None else "CONFLICT" if conflicts and rule.conflict_action == "quarantine" else "WARNING" if conflicts else "FALLBACK" if skipped else "SELECTED"
        accepted = chosen is not None and status not in {"BLOCKED", "CONFLICT"}
        if accepted:
            selected.append(candidates[chosen])
        decisions.append({"key": dict(zip(keys, key)), "status": status,
            "selected_source": candidates[chosen]["source_id"] if accepted else None,
            "selected_batch": candidates[chosen].get("source_batch_id") if accepted else None,
            "alternatives": [{"source_id": s, "batch_id": candidates[i].get("source_batch_id"), "reasons": reasons[i]} for s, i in source_rows.items()],
            "skipped": skipped, "conflicts": conflicts})
    counts = dict(Counter(item["status"] for item in decisions))
    return {"table_id": table_id, "rows": selected, "decisions": decisions, "excluded": excluded,
            "summary": {"input_rows": len(rows), "selected_rows": len(selected), "excluded_rows": len(excluded),
                        "rejected_input_rows": sum(item["reason"] not in {"SOURCE_NOT_SELECTED", "NOT_KNOWN_AS_OF", "RETROSPECTIVE_ADJUSTMENT"} for item in excluded), **counts},
            "as_of": as_of, "published": False, "execution": warm_resolution_kernels()}
