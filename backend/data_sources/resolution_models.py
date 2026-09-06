"""Versioned source arbitration rules, not provider-specific business tables."""
from __future__ import annotations
from typing import Literal
from pydantic import Field, model_validator

from .models import ID_PATTERN, CenterError, StrictModel


class FieldRule(StrictModel):
    field: str
    minimum: float | None = None
    maximum: float | None = None
    absolute_tolerance: float | None = Field(default=None, ge=0)
    relative_tolerance: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def bounds(self):
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("字段最小值不能大于最大值。")
        return self


class TableRule(StrictModel):
    table_id: str = Field(pattern=ID_PATTERN)
    source_priority: list[str] = Field(default_factory=list, max_length=30)
    fallback_on_missing: bool = True
    fallback_on_invalid: bool = False
    conflict_action: Literal["quarantine", "prefer_priority"] = "quarantine"
    required_fields: list[str] = Field(default_factory=list, max_length=50)
    compare_fields: list[str] = Field(default_factory=list, max_length=50)
    absolute_tolerance: float = Field(default=0.000001, ge=0)
    relative_tolerance: float = Field(default=0.0001, ge=0, le=1)
    max_relative_jump: float | None = Field(default=None, gt=0, le=100)
    field_rules: list[FieldRule] = Field(default_factory=list, max_length=50)


class ResolutionConfig(StrictModel):
    default_source_priority: list[str] = Field(default_factory=lambda: ["tushare", "akshare"], min_length=1, max_length=30)
    tables: list[TableRule] = Field(default_factory=lambda: [
        TableRule(table_id="market.quote_daily", required_fields=["close"], compare_fields=["open", "high", "low", "close"]),
        TableRule(table_id="market.nav_daily", required_fields=["unit_nav"], compare_fields=["unit_nav"]),
    ], max_length=100)

    @model_validator(mode="after")
    def contracts(self):
        try:
            from backend.data_model.catalog import TABLES_BY_ID
        except ModuleNotFoundError:
            from data_model.catalog import TABLES_BY_ID
        if len(self.default_source_priority) != len(set(self.default_source_priority)):
            raise ValueError("来源优先级不可重复。")
        ids = [item.table_id for item in self.tables]
        if len(ids) != len(set(ids)):
            raise ValueError("同一业务表只能保存一份取值规则。")
        for item in self.tables:
            table = TABLES_BY_ID.get(item.table_id)
            if table is None or not table.source_mappable:
                raise ValueError("多源规则只能用于外部业务数据表。")
            fields = {f.name: f for f in table.fields if f.source_mappable}
            chosen = set(item.required_fields + item.compare_fields + [f.field for f in item.field_rules])
            if chosen - fields.keys():
                raise ValueError("规则包含未知字段或系统维护字段。")
            if len(item.source_priority) != len(set(item.source_priority)):
                raise ValueError("表级来源顺序不可重复。")
            if len(item.field_rules) != len({r.field for r in item.field_rules}):
                raise ValueError("字段规则不可重复。")
            for rule in item.field_rules:
                if fields[rule.field].data_type != "float64":
                    raise ValueError("数值容差/值域规则仅用于 float64 字段；金额与文本按精确值比较。")
        return self

    def for_table(self, table_id: str) -> TableRule:
        return next((item for item in self.tables if item.table_id == table_id), TableRule(table_id=table_id))
