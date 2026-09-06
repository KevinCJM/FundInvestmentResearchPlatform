"""Typed metadata primitives for the platform-owned data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


DataLayer = Literal["control", "master", "canonical", "mart"]
StorageEngine = Literal["sqlite", "parquet"]
DeliveryPhase = Literal["core", "next"]
UpdateStrategy = Literal["append", "upsert", "snapshot", "scd2", "derived"]
FieldRole = Literal[
    "primary_key",
    "foreign_key",
    "dimension",
    "measure",
    "observation_time",
    "available_time",
    "effective_time",
    "audit",
    "configuration",
]


@dataclass(frozen=True, slots=True)
class CategoryDefinition:
    category_id: str
    label: str
    description: str
    order: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FieldDefinition:
    name: str
    label: str
    data_type: str
    nullable: bool
    role: FieldRole
    description: str
    unit: str | None = None
    enum_values: tuple[str, ...] = ()
    reference: str | None = None
    source_mappable: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["enum_values"] = list(self.enum_values)
        return payload


@dataclass(frozen=True, slots=True)
class TableDefinition:
    table_id: str
    category_id: str
    label: str
    description: str
    layer: DataLayer
    storage_engine: StorageEngine
    storage_location: str
    delivery_phase: DeliveryPhase
    grain: str
    primary_key: tuple[str, ...]
    update_strategy: UpdateStrategy
    fields: tuple[FieldDefinition, ...]
    source_mappable: bool = False
    partition_by: tuple[str, ...] = ()
    sort_by: tuple[str, ...] = ()
    pit_supported: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_key"] = list(self.primary_key)
        payload["partition_by"] = list(self.partition_by)
        payload["sort_by"] = list(self.sort_by)
        # Table policy is authoritative: a control/mart field cannot become an
        # import target merely because its field-level default is permissive.
        payload["usage"] = "external_import" if self.source_mappable else "system_internal"
        payload["fields"] = [
            {**item.to_dict(), "source_mappable": self.source_mappable and item.source_mappable}
            for item in self.fields
        ]
        return payload


def field(
    name: str,
    label: str,
    data_type: str,
    description: str,
    *,
    nullable: bool = True,
    role: FieldRole = "dimension",
    unit: str | None = None,
    enum_values: tuple[str, ...] = (),
    reference: str | None = None,
    source_mappable: bool = True,
) -> FieldDefinition:
    return FieldDefinition(
        name=name,
        label=label,
        data_type=data_type,
        nullable=nullable,
        role=role,
        description=description,
        unit=unit,
        enum_values=enum_values,
        reference=reference,
        source_mappable=source_mappable,
    )


def primary_id(name: str, label: str, description: str) -> FieldDefinition:
    return field(
        name,
        label,
        "string",
        description,
        nullable=False,
        role="primary_key",
        source_mappable=False,
    )


def foreign_id(
    name: str,
    label: str,
    reference: str,
    description: str,
    *,
    nullable: bool = False,
    source_mappable: bool = False,
) -> FieldDefinition:
    return field(
        name,
        label,
        "string",
        description,
        nullable=nullable,
        role="foreign_key",
        reference=reference,
        source_mappable=source_mappable,
    )


def control_audit_fields() -> tuple[FieldDefinition, ...]:
    return (
        field(
            "created_at",
            "创建时间",
            "timestamp[us, UTC]",
            "记录首次创建时间。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
        field(
            "updated_at",
            "更新时间",
            "timestamp[us, UTC]",
            "记录最近一次修改时间。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
    )


def validity_fields() -> tuple[FieldDefinition, ...]:
    return (
        field(
            "valid_from",
            "生效起始",
            "date32",
            "该版本业务生效的起始日期。",
            nullable=False,
            role="effective_time",
        ),
        field(
            "valid_to",
            "生效结束",
            "date32",
            "该版本业务生效的结束日期；为空表示仍然有效。",
            role="effective_time",
        ),
        field(
            "recorded_at",
            "系统记录时间",
            "timestamp[us, UTC]",
            "该版本写入平台的时间。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
    )


def source_lineage_fields() -> tuple[FieldDefinition, ...]:
    return (
        foreign_id(
            "source_id",
            "数据源",
            "governance.data_source.data_source_id",
            "产生该记录的外部数据源。",
            source_mappable=False,
        ),
        foreign_id(
            "source_batch_id",
            "采集批次",
            "governance.ingestion_batch.ingestion_batch_id",
            "产生该记录的采集批次；人工导入时可为空。",
            nullable=True,
            source_mappable=False,
        ),
        field(
            "source_record_id",
            "来源记录标识",
            "string",
            "外部数据源中的原始记录标识。",
        ),
        field(
            "source_record_hash",
            "来源记录哈希",
            "string",
            "标准化前原始记录的内容哈希，用于幂等和审计。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
        field(
            "revision",
            "修订序号",
            "int64",
            "同一业务键在该数据源中的修订序号，从 1 开始。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
        field(
            "ingested_at",
            "采集时间",
            "timestamp[us, UTC]",
            "平台取得该记录的时间。",
            nullable=False,
            role="audit",
            source_mappable=False,
        ),
    )


def availability_fields(
    *,
    observation_name: str = "observation_date",
    observation_label: str = "观测日期",
    observation_description: str = "数据所描述的业务日期或报告期末。",
) -> tuple[FieldDefinition, ...]:
    return (
        field(
            observation_name,
            observation_label,
            "date32",
            observation_description,
            nullable=False,
            role="observation_time",
        ),
        field(
            "available_at",
            "最早可得时间",
            "timestamp[us, UTC]",
            "该数据在历史时点最早可被研究或交易决策使用的时间。",
            role="available_time",
        ),
        field(
            "availability_status",
            "可得性状态",
            "string",
            "说明 available_at 的精度和可信度。",
            nullable=False,
            role="available_time",
            enum_values=("EXACT", "DATE_ONLY", "ESTIMATED", "UNKNOWN"),
        ),
    )
