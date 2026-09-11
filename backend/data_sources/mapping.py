"""Controlled response decoding and canonical-contract mapping, without eval."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import uuid
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from typing import Any
from zoneinfo import ZoneInfo

import pyarrow as pa

try:
    from backend.data_model.catalog import SCHEMA_CATALOG_VERSION, TABLES_BY_ID
except ModuleNotFoundError:
    from data_model.catalog import SCHEMA_CATALOG_VERSION, TABLES_BY_ID

from .models import CenterError, DatasetMapping, FieldMapping, InterfaceConfig, ResponseFormat

GENERATED = {"source_id", "source_batch_id", "source_record_hash", "revision", "ingested_at", "recorded_at", "vintage_id"}


def validate_mapping(config: InterfaceConfig) -> dict[str, Any]:
    errors, warnings = [], []
    known_source = {item.name for item in config.source_fields}
    for index, mapping in enumerate(config.mappings):
        prefix = f"mappings.{index}"
        table = TABLES_BY_ID.get(mapping.target_table)
        if table is None or not table.source_mappable:
            errors.append({"field": prefix, "message": "目标必须是可外部导入的标准表。"})
            continue
        if mapping.contract_version != SCHEMA_CATALOG_VERSION:
            errors.append({"field": prefix, "message": "标准表版本不匹配，请重新选择目标表。"})
        fields = {item.name: item for item in table.fields}
        assigned = set()
        for binding in [*mapping.fields, *mapping.identities]:
            name = binding.target_field
            target = fields.get(name)
            if name in assigned:
                errors.append({"field": prefix, "message": f"目标字段 {name} 被重复映射。"})
            assigned.add(name)
            if target is None:
                errors.append({"field": prefix, "message": f"目标字段 {name} 不存在。"})
                continue
            if binding in mapping.fields and not target.source_mappable:
                errors.append({"field": prefix, "message": f"{name} 是系统维护字段，不能直接映射。"})
            if getattr(binding, "operation", None) == "constant":
                try:
                    convert(binding.constant, target)
                except (ValueError, TypeError, InvalidOperation, OverflowError, pa.ArrowException):
                    errors.append({"field": prefix, "message": f"{name} 的固定值不符合标准字段类型或枚举。"})
            if binding in mapping.identities and (target.source_mappable or target.data_type != "string" or target.role not in {"primary_key", "foreign_key"} or name in GENERATED):
                errors.append({"field": prefix, "message": f"{name} 不能使用外部标识解析。"})
            source = binding.source_field
            sources = [source] if source else getattr(binding, "key_fields", [])
            if known_source and any(name not in known_source for name in sources):
                errors.append({"field": prefix, "message": "映射引用了返回结构中未声明的来源字段。"})
            if getattr(binding, "operation", None) == "capture_date" and name not in {"valid_from", "effective_from"}:
                errors.append({"field": prefix, "message": "采集日期只能作为当前元数据的版本生效起点。"})
            if getattr(binding, "resolution", None) == "lookup" and not binding.value_map:
                warnings.append({"field": prefix, "message": f"{name} 需要人工确认身份对照。", "code": "IDENTITY_LOOKUP_REQUIRED"})
        missing = [f.name for f in table.fields if not f.nullable and f.name not in assigned and f.name not in GENERATED]
        if missing:
            warnings.append({"field": prefix, "message": "仍需配置必填字段：" + ", ".join(missing), "code": "REQUIRED_MAPPING_MISSING"})
    if not any(item.enabled for item in config.mappings):
        warnings.append({"field": "mappings", "message": "尚未配置启用的标准表映射。", "code": "MAPPING_REQUIRED"})
    return {"valid": not errors, "ready": bool(config.mappings) and not errors and not warnings, "errors": errors, "warnings": warnings}


def at_path(payload: Any, path: str) -> Any:
    for name in path.split(".") if path else []:
        if not isinstance(payload, dict) or name not in payload:
            raise CenterError("RESPONSE_PATH_MISSING", "返回数据不包含配置的数据路径。")
        payload = payload[name]
    return payload


def decode_response(payload: Any, spec: ResponseFormat, max_rows: int = 1000) -> list[dict[str, Any]]:
    if spec.format == "csv":
        if not isinstance(payload, str):
            raise CenterError("RESPONSE_FORMAT", "CSV 样本必须是文本。")
        reader = csv.DictReader(io.StringIO(payload.lstrip("\ufeff")), delimiter=spec.delimiter)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise CenterError("CSV_HEADER", "CSV 表头为空或有重复字段。")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise CenterError("CSV_WIDTH", "CSV 行与表头列数不一致。")
            rows.append(row)
            if len(rows) > max_rows:
                raise CenterError("SAMPLE_TOO_LARGE", "样本行数超过限制。")
    else:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (ValueError, RecursionError) as exc:
                raise CenterError("RESPONSE_FORMAT", "样本不是有效 JSON。") from exc
        rows = at_path(payload, spec.records_path)
        if not isinstance(rows, list) or len(rows) > max_rows:
            raise CenterError("SAMPLE_TOO_LARGE", "数据路径须指向不超过限制行数的数组。")
        if spec.format == "json_columns":
            names = at_path(payload, spec.columns_path)
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or len(names) != len(set(names)):
                raise CenterError("COLUMN_NAMES", "字段名数组无效或重复。")
            if any(not isinstance(row, list) or len(row) != len(names) for row in rows):
                raise CenterError("COLUMN_WIDTH", "字段名与数据行列数不一致。")
            rows = [dict(zip(names, row)) for row in rows]
    if any(not isinstance(row, dict) for row in rows):
        raise CenterError("RESPONSE_FORMAT", "每条样本必须为对象。")
    return rows


def _date(value: Any, fmt: str | None = None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    return datetime.strptime(text, fmt or ("%Y%m%d" if re.fullmatch(r"\d{8}", text) else "%Y-%m-%d")).date()


def transform(value: Any, binding: FieldMapping) -> Any:
    if binding.operation == "capture_date":
        return value
    if binding.operation == "constant":
        return binding.constant
    if value is None or value == "":
        return None
    if binding.operation == "copy":
        return value
    if binding.operation == "scale":
        # Exact decimal unit conversion is an ingestion boundary, not analytics.
        return Decimal(str(value)) * Decimal(str(binding.factor))
    if binding.operation == "enum":
        key = str(value)
        if key not in binding.enum_map:
            raise ValueError("来源枚举值未配置对应关系")
        return binding.enum_map[key]
    if binding.operation == "date":
        return _date(value, binding.date_format)
    if binding.operation == "period_end":
        import calendar
        text = str(value).upper().replace("-", "")
        if re.fullmatch(r"\d{4}Q[1-4]", text):
            year, month = int(text[:4]), int(text[-1]) * 3
        elif re.fullmatch(r"\d{6}", text):
            year, month = int(text[:4]), int(text[4:])
        else:
            return _date(value, binding.date_format)
        return date(year, month, calendar.monthrange(year, month)[1])
    if binding.operation == "timestamp":
        text = str(value)
        if isinstance(value, date) and not isinstance(value, datetime) or re.fullmatch(r"\d{8}|\d{4}-\d{2}-\d{2}", text):
            # Date-only announcements become available at local end-of-day.
            parsed = datetime.combine(_date(value, binding.date_format), time.max)
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(binding.timezone))
        return parsed.astimezone(timezone.utc)
    raise ValueError("不支持的转换")


def arrow_type(name: str) -> pa.DataType:
    if name == "json":
        return pa.string()
    if name == "date32":
        return pa.date32()
    if name == "timestamp[us, UTC]":
        return pa.timestamp("us", tz="UTC")
    if name == "list<string>":
        return pa.list_(pa.string())
    decimal = re.fullmatch(r"decimal128\((\d+),\s*(\d+)\)", name)
    if decimal:
        return pa.decimal128(int(decimal[1]), int(decimal[2]))
    return {"string": pa.string(), "bool": pa.bool_(), "int64": pa.int64(), "float64": pa.float64()}[name]


def convert(value: Any, field: Any) -> Any:
    if value is None or value == "":
        if not field.nullable:
            raise ValueError("必填字段没有值")
        return None
    dtype = field.data_type
    if dtype == "string":
        result = str(value)
    elif dtype == "float64":
        if isinstance(value, bool):
            raise ValueError("布尔值不是数值")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("数值不能为 NaN 或无穷")
    elif dtype == "int64":
        decimal = Decimal(str(value))
        if not decimal.is_finite() or decimal != decimal.to_integral_value():
            raise ValueError("整数类型不允许小数或非有限值")
        result = int(decimal)
        if not -(2**63) <= result < 2**63:
            raise ValueError("整数超出范围")
    elif dtype == "bool":
        allowed = {"true": True, "false": False, "1": True, "0": False}
        if str(value).lower() not in allowed:
            raise ValueError("布尔值须为 true/false 或 1/0")
        result = allowed[str(value).lower()]
    elif dtype == "date32":
        result = _date(value)
    elif dtype.startswith("timestamp"):
        result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("时间戳必须带时区，请使用时间转换")
        result = result.astimezone(timezone.utc)
    elif dtype.startswith("decimal128"):
        result = Decimal(str(value))
        if not result.is_finite():
            raise ValueError("金额不能为非有限值")
        pa.scalar(result, type=arrow_type(dtype))
    elif dtype == "json":
        result = json.dumps(value, ensure_ascii=False, allow_nan=False)
    elif dtype == "list<string>":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError("字段须为字符串数组")
        result = value
    else:
        raise ValueError("字段类型尚不支持")
    if field.enum_values and result not in field.enum_values:
        raise ValueError("值不在标准字段枚举范围")
    return result


def map_table(rows: list[dict[str, Any]], mapping: DatasetMapping, source_id: str, batch_id: str) -> tuple[pa.Table, list[dict[str, Any]]]:
    definition = TABLES_BY_ID.get(mapping.target_table)
    # Enforce the catalog boundary even when invoked without API validation.
    if definition is None or not definition.source_mappable:
        raise CenterError(
            "INVALID_IMPORT_TARGET",
            "只能导入业务数据表；代码对照和映射配置属于系统内部结构。",
            422,
        )
    fields = {f.name: f for f in definition.fields}
    timestamp = datetime.now(timezone.utc)
    output, errors, keys = [], [], set()
    for number, row in enumerate(rows, 1):
        mapped = {"source_id": source_id, "source_batch_id": batch_id, "revision": 1,
                  "ingested_at": timestamp, "recorded_at": timestamp, "vintage_id": batch_id,
                  "source_record_hash": hashlib.sha256(json.dumps(row, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()}
        try:
            for binding in mapping.fields:
                mapped[binding.target_field] = transform(timestamp.date() if binding.operation == "capture_date" else row.get(binding.source_field), binding)
            for binding in mapping.identities:
                parts = [row.get(key) for key in binding.key_fields] if binding.key_fields else [binding.constant if binding.constant is not None else row.get(binding.source_field)]
                if any(part is None or not str(part).strip() for part in parts):
                    raise ValueError("外部标识字段为空")
                if binding.key_transform != "none":
                    from .identity import normalize_external_key
                    if len(parts) != 1:
                        raise ValueError("代码转换只支持单一身份字段")
                    parts = [normalize_external_key(parts[0], binding.key_transform)]
                token = json.dumps(parts, ensure_ascii=False, default=str)
                if token is None or not str(token).strip():
                    raise ValueError("外部标识字段为空")
                # Namespace selection is explicit, never a name/substring match.
                if binding.resolution == "lookup":
                    lookup = str(parts[0]).strip() if len(parts) == 1 else token
                    if lookup not in binding.value_map:
                        raise ValueError(f"{binding.target_field} 的外部标识尚未人工确认")
                    mapped[binding.target_field] = binding.value_map[lookup]
                else:
                    mapped[binding.target_field] = str(uuid.uuid5(uuid.NAMESPACE_URL, binding.namespace + ":" + str(token).strip()))
            record = {}
            for name, field in fields.items():
                try:
                    record[name] = convert(mapped.get(name), field)
                except (ValueError, TypeError, InvalidOperation, OverflowError, pa.ArrowException) as exc:
                    raise ValueError(f"字段 {name} 不符合类型、必填、单位或枚举约束") from exc
            if record.get("available_at") is None and record.get("availability_status") not in (None, "UNKNOWN"):
                raise ValueError("缺少可得时间时，可得性状态只能为 UNKNOWN")
            key = tuple(record[name] for name in definition.primary_key)
            if key in keys:
                raise ValueError("样本内存在重复业务主键")
            keys.add(key)
            output.append(record)
        except (ValueError, TypeError, InvalidOperation, KeyError, OverflowError) as exc:
            errors.append({"row": number, "table": mapping.target_table, "message": str(exc) if type(exc) is ValueError else "字段转换失败"})
    schema = pa.schema([pa.field(f.name, arrow_type(f.data_type), nullable=f.nullable) for f in definition.fields], metadata={b"contract_version": mapping.contract_version.encode(), b"table_id": mapping.target_table.encode(), b"source_id": source_id.encode(), b"batch_id": batch_id.encode(), b"status": b"mapped_candidate"})
    return pa.Table.from_pylist(output, schema=schema), errors


def json_value(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    return value


def preview(config: InterfaceConfig, payload: Any) -> dict[str, Any]:
    validation = validate_mapping(config)
    if not validation["valid"]:
        return {**validation, "tables": [], "source_rows": 0}
    rows = decode_response(payload, config.response)
    tables, errors = [], []
    for mapping in config.mappings:
        if not mapping.enabled:
            continue
        table, failures = map_table(rows, mapping, config.source_id, "preview")
        errors.extend(failures)
        tables.append({"table_id": mapping.target_table, "accepted_rows": table.num_rows, "rejected_rows": len(failures), "columns": table.column_names,
                       "rows": [{key: json_value(value) for key, value in row.items()} for row in table.slice(0, 20).to_pylist()]})
    if not rows:
        validation["warnings"].append({"field": "sample", "code": "EMPTY_SAMPLE", "message": "样本为空，尚未验证真实数据转换。"})
    return {**validation, "valid": validation["valid"] and not errors,
            "ready": validation["ready"] and not errors and bool(rows),
            "source_rows": len(rows), "tables": tables,
            "errors": [*validation["errors"], *errors[:100]], "error_count": len(errors), "preview_only": True}
