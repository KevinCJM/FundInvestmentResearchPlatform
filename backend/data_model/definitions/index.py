"""Canonical index membership, weight, and valuation contracts."""

from __future__ import annotations

from ..types import (
    CategoryDefinition,
    TableDefinition,
    availability_fields,
    field,
    foreign_id,
    primary_id,
    source_lineage_fields,
)


CATEGORY = CategoryDefinition(
    category_id="index",
    label="指数与基准",
    description="统一不同指数供应商的成分、权重和估值数据；指数行情统一进入日行情表。",
    order=50,
)


def _index_table(
    table_id: str,
    label: str,
    description: str,
    grain: str,
    primary_key: tuple[str, ...],
    fields: tuple,
    *,
    partition_by: tuple[str, ...],
    sort_by: tuple[str, ...],
) -> TableDefinition:
    name = table_id.split(".", 1)[1]
    return TableDefinition(
        table_id=table_id,
        category_id=CATEGORY.category_id,
        label=label,
        description=description,
        layer="canonical",
        storage_engine="parquet",
        storage_location=f"data/canonical/v1/index/{name}/",
        delivery_phase="next",
        grain=grain,
        primary_key=primary_key,
        update_strategy="append",
        fields=fields,
        source_mappable=True,
        partition_by=partition_by,
        sort_by=sort_by,
        pit_supported=True,
    )


TABLES = (
    _index_table(
        "index.membership",
        "指数成分关系",
        "保存指数成分的纳入、剔除、公告和历史可得时间；未解析成分保留来源代码。",
        "每个指数、成分、有效期、数据源和修订版本一条记录",
        ("index_instrument_id", "member_key", "effective_from", "source_id", "revision"),
        (
            foreign_id("index_instrument_id", "指数标的 ID", "master.instrument.instrument_id", "指数的内部标的 ID。", source_mappable=False),
            field("member_key", "成分键", "string", "已解析 instrument_id 或来源代码形成的稳定成分键。", nullable=False, role="primary_key", source_mappable=False),
            foreign_id("member_instrument_id", "成分标的 ID", "master.instrument.instrument_id", "已解析的成分内部标的。", nullable=True, source_mappable=False),
            field("member_external_code", "成分外部代码", "string", "来源中的成分代码；未解析时保留。"),
            field("member_name", "成分名称", "string", "来源披露的成分名称。"),
            field("effective_from", "纳入日期", "date32", "成分开始生效日期。", nullable=False, role="effective_time"),
            field("effective_to", "剔除日期", "date32", "成分停止生效日期；为空表示仍在指数中。", role="effective_time"),
            field("announced_at", "公告时间", "timestamp[us, UTC]", "成分调整公告时间。", role="available_time"),
            field("available_at", "最早可得时间", "timestamp[us, UTC]", "历史研究中最早可使用该成分关系的时间。", role="available_time"),
            field("availability_status", "可得性状态", "string", "available_at 的精度。", nullable=False, role="available_time", enum_values=("EXACT", "DATE_ONLY", "ESTIMATED", "UNKNOWN")),
            field("is_current", "当前成分", "bool", "该版本是否表示当前仍为成分。", nullable=False),
            *source_lineage_fields(),
        ),
        partition_by=("effective_from",),
        sort_by=("index_instrument_id", "effective_from", "member_key", "source_id", "revision"),
    ),
    _index_table(
        "index.weight",
        "指数成分权重",
        "保存指数在指定权重日的成分权重快照。",
        "每个指数、成分、权重日、数据源和修订版本一条记录",
        ("index_instrument_id", "member_key", "weight_date", "source_id", "revision"),
        (
            foreign_id("index_instrument_id", "指数标的 ID", "master.instrument.instrument_id", "指数的内部标的 ID。", source_mappable=False),
            field("member_key", "成分键", "string", "已解析 instrument_id 或来源代码形成的稳定成分键。", nullable=False, role="primary_key", source_mappable=False),
            foreign_id("member_instrument_id", "成分标的 ID", "master.instrument.instrument_id", "已解析的成分内部标的。", nullable=True, source_mappable=False),
            field("member_external_code", "成分外部代码", "string", "来源中的成分代码。"),
            *availability_fields(
                observation_name="weight_date",
                observation_label="权重日期",
                observation_description="指数权重快照适用日期。",
            ),
            field("weight", "成分权重", "float64", "小数制权重，0.05 表示 5%。", nullable=False, role="measure", unit="decimal"),
            field("weight_method", "权重口径", "string", "自由流通市值、等权或其他权重方法。"),
            *source_lineage_fields(),
        ),
        partition_by=("weight_date",),
        sort_by=("index_instrument_id", "weight_date", "member_key", "source_id", "revision"),
    ),
    _index_table(
        "index.valuation_daily",
        "指数日估值",
        "统一指数市盈率、市净率、股息率、换手率和总市值等估值指标。",
        "每个指数、交易日、数据源和修订版本一条记录",
        ("index_instrument_id", "trade_date", "source_id", "revision"),
        (
            foreign_id("index_instrument_id", "指数标的 ID", "master.instrument.instrument_id", "指数的内部标的 ID。", source_mappable=False),
            *availability_fields(
                observation_name="trade_date",
                observation_label="交易日期",
                observation_description="估值指标对应交易日期。",
            ),
            field("pe", "市盈率", "float64", "静态市盈率。", role="measure", unit="multiple"),
            field("pe_ttm", "滚动市盈率", "float64", "TTM 市盈率。", role="measure", unit="multiple"),
            field("pb", "市净率", "float64", "市净率。", role="measure", unit="multiple"),
            field("dividend_yield", "股息率", "float64", "小数制股息率。", role="measure", unit="decimal"),
            field("turnover_rate", "换手率", "float64", "小数制指数成分换手率或市场换手率。", role="measure", unit="decimal"),
            field("total_market_value", "总市值", "decimal128(38, 6)", "指数覆盖标的总市值。", role="measure", unit="currency"),
            field("float_market_value", "流通市值", "decimal128(38, 6)", "指数覆盖标的流通市值。", role="measure", unit="currency"),
            field("currency", "币种", "string", "市值计价币种。", unit="ISO-4217"),
            *source_lineage_fields(),
        ),
        partition_by=("trade_date",),
        sort_by=("index_instrument_id", "trade_date", "source_id", "revision"),
    ),
)
