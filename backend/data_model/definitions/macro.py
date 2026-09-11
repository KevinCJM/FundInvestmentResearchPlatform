"""Canonical macroeconomic series, observation, and release-event contracts."""

from __future__ import annotations

from ..types import (
    CategoryDefinition,
    TableDefinition,
    field,
    foreign_id,
    primary_id,
    source_lineage_fields,
    validity_fields,
)


CATEGORY = CategoryDefinition(
    category_id="macro",
    label="宏观、利率与汇率",
    description="以长表统一 GDP、CPI、PPI、PMI、货币、社融及其他宏观序列，并保存修订和发布日期。",
    order=60,
)


def _macro_table(
    table_id: str,
    label: str,
    description: str,
    grain: str,
    primary_key: tuple[str, ...],
    fields: tuple,
    *,
    update_strategy: str,
    partition_by: tuple[str, ...] = (),
    sort_by: tuple[str, ...] = (),
) -> TableDefinition:
    name = table_id.split(".", 1)[1]
    return TableDefinition(
        table_id=table_id,
        category_id=CATEGORY.category_id,
        label=label,
        description=description,
        layer="canonical" if table_id != "macro.series" else "master",
        storage_engine="parquet",
        storage_location=f"data/canonical/v1/macro/{name}/",
        delivery_phase="next",
        grain=grain,
        primary_key=primary_key,
        update_strategy=update_strategy,  # type: ignore[arg-type]
        fields=fields,
        source_mappable=True,
        partition_by=partition_by,
        sort_by=sort_by,
        pit_supported=True,
    )


TABLES = (
    _macro_table(
        "macro.series",
        "宏观序列",
        "定义宏观、货币、信用、利率、流动性和经济日历序列的稳定内部 ID 与单位。",
        "每个宏观序列的每个有效版本一条记录",
        ("series_id", "valid_from"),
        (
            primary_id("series_id", "宏观序列 ID", "平台生成的稳定序列标识。"),
            field("canonical_name", "标准名称", "string", "平台统一序列名称。", nullable=False),
            field("short_name", "简称", "string", "图表与选择器使用的简称。"),
            field("category", "类别", "string", "经济增长、通胀、货币信用、利率、流动性等类别。", nullable=False, enum_values=("GROWTH", "INFLATION", "MONEY_CREDIT", "RATE", "LIQUIDITY", "EMPLOYMENT", "TRADE", "OTHER")),
            field("country_code", "国家或地区", "string", "序列覆盖国家或地区。", unit="ISO-3166-1"),
            field("region", "区域", "string", "省份、城市或经济区域。"),
            field("frequency", "频率", "string", "原始观测频率。", nullable=False, enum_values=("DAILY", "WEEKLY", "MONTHLY", "QUARTERLY", "ANNUAL", "EVENT", "IRREGULAR")),
            field("unit", "标准单位", "string", "平台统一数值单位。", nullable=False),
            field("seasonal_adjustment", "季调口径", "string", "季调状态。", nullable=False, enum_values=("SA", "NSA", "UNKNOWN", "NOT_APPLICABLE")),
            foreign_id("publisher_organization_id", "发布机构", "master.organization.organization_id", "统计或发布机构。", nullable=True, source_mappable=False),
            field("revision_policy", "修订策略", "string", "是否修订及正式研究采用的版本策略。", nullable=False, enum_values=("NO_REVISION", "FIRST_RELEASE", "LATEST_VINTAGE", "VERSIONED")),
            field("expected_release_lag_days", "预计发布滞后", "int64", "观测期结束到发布日期的预计天数。", unit="day"),
            field("status_code", "状态", "string", "序列是否仍在发布。", nullable=False, enum_values=("ACTIVE", "DISCONTINUED", "UNKNOWN")),
            *validity_fields(),
            *source_lineage_fields(),
        ),
        update_strategy="scd2",
        partition_by=("category",),
        sort_by=("series_id", "valid_from"),
    ),
    _macro_table(
        "macro.observation",
        "宏观观察值",
        "以统一长表保存宏观数值、参考期、发布日期、可得时间、修订和 Vintage。",
        "每个宏观序列、参考期、数据源和修订版本一条记录",
        ("series_id", "observation_period_end", "source_id", "revision"),
        (
            foreign_id("series_id", "宏观序列 ID", "macro.series.series_id", "观测所属宏观序列。", source_mappable=False),
            field("observation_period_start", "参考期起始", "date32", "统计参考期开始日期。", role="observation_time"),
            field("observation_period_end", "参考期结束", "date32", "统计参考期结束日期。", nullable=False, role="primary_key"),
            field("published_at", "发布时间", "timestamp[us, UTC]", "该修订值正式发布时间。", role="available_time"),
            field("available_at", "最早可得时间", "timestamp[us, UTC]", "历史研究中最早可使用该修订值的时间。", role="available_time"),
            field("availability_status", "可得性状态", "string", "available_at 的精度与可信度。", nullable=False, role="available_time", enum_values=("EXACT", "DATE_ONLY", "ESTIMATED", "UNKNOWN")),
            field("value", "观察值", "float64", "按 macro.series.unit 表示的数值。", nullable=False, role="measure", unit="series_unit"),
            field("unit", "单位", "string", "该记录实际单位，应与序列标准单位一致。", nullable=False),
            field("vintage_id", "Vintage ID", "string", "来源或平台生成的数据版本标识。", nullable=False, role="audit"),
            field("is_final", "是否最终值", "bool", "来源是否将该修订标记为最终值。", nullable=False),
            field("release_status", "发布状态", "string", "初值、修订值或最终值。", nullable=False, enum_values=("PRELIMINARY", "REVISED", "FINAL", "UNKNOWN")),
            *source_lineage_fields(),
        ),
        update_strategy="append",
        partition_by=("observation_period_end",),
        sort_by=("series_id", "observation_period_end", "source_id", "revision"),
    ),
    _macro_table(
        "macro.release_event",
        "宏观发布事件",
        "保存宏观数据预定发布时间、实际发布时间、参考期和发布状态。",
        "每个宏观序列和参考期的一次发布事件一条记录",
        ("release_event_id", "source_id", "revision"),
        (
            primary_id("release_event_id", "发布事件 ID", "宏观发布事件的稳定标识。"),
            foreign_id("series_id", "宏观序列 ID", "macro.series.series_id", "发布事件对应序列。", source_mappable=False),
            field("reference_period_end", "参考期结束", "date32", "本次发布对应的统计参考期。", role="observation_time"),
            field("scheduled_release_at", "计划发布时间", "timestamp[us, UTC]", "事先公布的计划发布时间。", role="available_time"),
            field("actual_release_at", "实际发布时间", "timestamp[us, UTC]", "数据实际发布的时间。", role="available_time"),
            field("available_at", "最早可得时间", "timestamp[us, UTC]", "通常等于实际发布时间。", role="available_time"),
            field("availability_status", "可得性状态", "string", "available_at 的精度。", nullable=False, role="available_time", enum_values=("EXACT", "DATE_ONLY", "ESTIMATED", "UNKNOWN")),
            field("release_status", "事件状态", "string", "发布事件状态。", nullable=False, enum_values=("SCHEDULED", "RELEASED", "DELAYED", "CANCELLED", "UNKNOWN")),
            field("title", "发布标题", "string", "来源中的发布事件标题。"),
            field("details_json", "事件详情", "json", "扩展发布信息。"),
            *source_lineage_fields(),
        ),
        update_strategy="append",
        partition_by=("reference_period_end",),
        sort_by=("series_id", "reference_period_end", "actual_release_at", "source_id", "revision"),
    ),
)
