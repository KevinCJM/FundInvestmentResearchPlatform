"""Read-only registry for the platform-owned canonical data model."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from functools import lru_cache
from typing import Any, Literal

from .definitions import (
    accounting,
    fund,
    governance,
    index,
    macro,
    market,
    marts,
    master,
    operations,
    portfolio,
    research,
    security,
)
from .types import CategoryDefinition, TableDefinition


MODEL_ID = "fund-investment-research-platform-data-model"
SCHEMA_CATALOG_VERSION = "1.2.0"
CatalogScope = Literal["external", "internal", "all"]

CATEGORY_MODULES = (
    governance,
    master,
    market,
    fund,
    index,
    macro,
    marts,
    research,
    portfolio,
    operations,
    accounting,
    security,
)
CATEGORIES: tuple[CategoryDefinition, ...] = tuple(
    sorted((module.CATEGORY for module in CATEGORY_MODULES), key=lambda item: item.order)
)
TABLES: tuple[TableDefinition, ...] = tuple(
    table for module in CATEGORY_MODULES for table in module.TABLES
)
TABLES_BY_ID = {table.table_id: table for table in TABLES}

MODEL_PRINCIPLES = (
    "业务代码只使用平台内部 ID，不直接使用 Tushare、Wind 或其他来源代码。",
    "外部导入表只列出可从外部取得的业务数据，例如产品、机构、行情、净值、持仓和宏观数据。",
    "字段映射与外部代码对照通过配置管理，属于系统内部结构；不要求用户单独导入映射表。",
    "基金经理任职、指数成分、产品分类和基准等描述真实业务关系的数据，仍属于外部导入数据。",
    "系统内部表仅在高级入口中查看；任务状态、血缘和研究派生结果也不能作为外部导入目标。",
    "业务记录中的内部 ID 按导入配置解析；批次和哈希由系统生成，不允许来源直接写入。",
    "外部数据只能写入允许映射的内部表和字段，研究 Mart 禁止外部数据源直接写入。",
    "业务日期使用 date32；带时点的信息统一使用 UTC timestamp。",
    "收益率、费率、权重统一使用小数制，0.01 表示 1%。",
    "空值表示未知、未披露或不可得，不能静默转换为 0。",
    "历史研究必须使用 available_at 判断当时是否可得，不能只看 observation_date。",
    "每次正式研究锁定 data_release_id、表合同版本和输入数据校验和。",
)

TYPE_CONVENTIONS = (
    {"logical_type": "identifier", "physical_type": "string", "rule": "平台 ID 使用不可变字符串标识。"},
    {"logical_type": "business_date", "physical_type": "date32", "rule": "只表达自然日，不携带时区。"},
    {"logical_type": "event_time", "physical_type": "timestamp[us, UTC]", "rule": "所有时点统一转换为 UTC。"},
    {"logical_type": "market_number", "physical_type": "float64", "rule": "行情、净值、收益率和统计值使用 float64。"},
    {"logical_type": "money", "physical_type": "decimal128(38, 6)", "rule": "资金、资产和成交额使用定点十进制。"},
    {"logical_type": "quantity", "physical_type": "decimal128(38, 10)", "rule": "份额和持仓数量保留更高精度。"},
)


def _validate_catalog() -> None:
    category_ids = [item.category_id for item in CATEGORIES]
    if len(category_ids) != len(set(category_ids)):
        raise RuntimeError("系统数据模型存在重复 category_id。")

    table_ids = [item.table_id for item in TABLES]
    if len(table_ids) != len(set(table_ids)):
        raise RuntimeError("系统数据模型存在重复 table_id。")

    known_categories = set(category_ids)
    for table in TABLES:
        if table.category_id not in known_categories:
            raise RuntimeError(f"{table.table_id} 引用了未知分类 {table.category_id}。")
        field_names = [item.name for item in table.fields]
        if len(field_names) != len(set(field_names)):
            raise RuntimeError(f"{table.table_id} 存在重复字段。")
        known_fields = set(field_names)
        missing_primary = set(table.primary_key) - known_fields
        missing_partition = set(table.partition_by) - known_fields
        missing_sort = set(table.sort_by) - known_fields
        if missing_primary:
            raise RuntimeError(f"{table.table_id} 主键字段不存在: {sorted(missing_primary)}")
        if missing_partition:
            raise RuntimeError(f"{table.table_id} 分区字段不存在: {sorted(missing_partition)}")
        if missing_sort:
            raise RuntimeError(f"{table.table_id} 排序字段不存在: {sorted(missing_sort)}")
        for primary_key in table.primary_key:
            definition = next(item for item in table.fields if item.name == primary_key)
            if definition.nullable:
                raise RuntimeError(f"{table.table_id}.{primary_key} 为主键但允许空值。")
        if table.source_mappable and not any(item.source_mappable for item in table.fields):
            raise RuntimeError(f"{table.table_id} 声明可映射但没有可映射字段。")

    for table in TABLES:
        for definition in table.fields:
            if not definition.reference:
                continue
            target_table_id, target_field = definition.reference.rsplit(".", 1)
            target_table = TABLES_BY_ID.get(target_table_id)
            if target_table is None:
                raise RuntimeError(
                    f"{table.table_id}.{definition.name} 引用了未知表 {target_table_id}。"
                )
            if target_field not in {item.name for item in target_table.fields}:
                raise RuntimeError(
                    f"{table.table_id}.{definition.name} 引用了未知字段 {definition.reference}。"
                )


_validate_catalog()


def _matches_scope(table: TableDefinition, scope: CatalogScope) -> bool:
    if scope not in {"external", "internal", "all"}:
        raise ValueError("数据字典范围必须为 external、internal 或 all。")
    return scope == "all" or table.source_mappable == (scope == "external")


@lru_cache(maxsize=3)
def _catalog_payload(scope: CatalogScope) -> dict[str, Any]:
    visible_tables = tuple(table for table in TABLES if _matches_scope(table, scope))
    visible_categories = tuple(
        category for category in CATEGORIES
        if any(table.category_id == category.category_id for table in visible_tables)
    )
    tables_by_category = Counter(table.category_id for table in visible_tables)
    fields_by_category = Counter()
    for table in visible_tables:
        fields_by_category[table.category_id] += len(table.fields)

    categories = []
    for category in visible_categories:
        payload = category.to_dict()
        payload["table_count"] = tables_by_category[category.category_id]
        payload["field_count"] = fields_by_category[category.category_id]
        categories.append(payload)

    table_payloads = [table.to_dict() for table in visible_tables]
    return {
        "model_id": MODEL_ID,
        "schema_version": SCHEMA_CATALOG_VERSION,
        "status": "defined",
        "scope": scope,
        "description": (
            "这里列出系统可以从外部导入的业务数据，以及需要提供的字段。"
            "来源字段和代码对应关系在数据源配置中管理，不需要单独导入映射表。"
        ),
        "principles": list(MODEL_PRINCIPLES),
        "type_conventions": list(TYPE_CONVENTIONS),
        "categories": categories,
        "tables": table_payloads,
        "summary": {
            "category_count": len(visible_categories),
            "table_count": len(visible_tables),
            "field_count": sum(len(table.fields) for table in visible_tables),
            "mapping_target_table_count": sum(table.source_mappable for table in visible_tables),
            "mapping_target_field_count": sum(
                table.source_mappable and definition.source_mappable
                for table in visible_tables for definition in table.fields
            ),
            "pit_table_count": sum(table.pit_supported for table in visible_tables),
            "by_layer": dict(Counter(table.layer for table in visible_tables)),
            "by_storage_engine": dict(Counter(table.storage_engine for table in visible_tables)),
            "by_delivery_phase": dict(Counter(table.delivery_phase for table in visible_tables)),
        },
    }


def get_data_model_catalog(*, scope: CatalogScope = "external") -> dict[str, Any]:
    """Default to import targets; expose internal definitions only on opt-in."""

    return deepcopy(_catalog_payload(scope))


def get_table_definition(
    table_id: str, *, scope: CatalogScope = "external"
) -> dict[str, Any] | None:
    table = TABLES_BY_ID.get(table_id)
    return None if table is None or not _matches_scope(table, scope) else table.to_dict()
