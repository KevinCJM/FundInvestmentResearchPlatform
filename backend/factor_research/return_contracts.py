"""Contracts for return-producing algorithms, distinct from product scores."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .contracts import Contract


class ReturnPlanFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    method: Literal["characteristic_spread", "ff3_2x3"]
    source_run_id: str | None = Field(default=None, max_length=120)
    source_panel_id: str | None = Field(default=None, max_length=120)
    factor_key: str = Field(default="composite", min_length=1, max_length=120)
    quantiles: int = Field(default=3, ge=2, le=10)
    cost_bps: float = Field(default=0, ge=0, le=500)
    output_factor: str = Field(default="SPREAD", pattern=r"^[A-Z][A-Z0-9_]{0,31}$")

    @model_validator(mode="after")
    def consistent(self):
        if not self.name.strip():
            raise ValueError("方案名称不能为空")
        if self.method == "characteristic_spread":
            if not self.source_run_id or self.source_panel_id:
                raise ValueError("特征收益构建须且仅须选择已完成的特征运行")
            if self.output_factor in {"MKT_RF", "SMB", "HML", "RF"}:
                raise ValueError("特征收益差额不能命名为 FF3 因子或 RF")
        elif not self.source_panel_id or self.source_run_id:
            raise ValueError("FF3 构建须且仅须选择股票时点面板")
        elif self.cost_bps != 0:
            raise ValueError("FF3 输出为毛因子收益，不能在此混入交易费用")
        return self


class ReturnPlanUpdate(ReturnPlanFields):
    revision: int = Field(ge=1)


class ReturnRow(Contract):
    date: date
    values: dict[str, float | None]


class ReturnDatasetFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    source_url: str = Field(pattern=r"^https?://", max_length=500)
    market: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{1,15}$")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    frequency: Literal["daily"] = "daily"
    units: Literal["decimal_return"] = "decimal_return"
    construction: str = Field(min_length=10, max_length=2000)
    factor_names: list[str] = Field(min_length=1, max_length=8)
    dependent_return: Literal["total", "excess"] = "excess"
    rows: list[ReturnRow] = Field(min_length=30, max_length=6000)

    @field_validator("factor_names")
    @classmethod
    def names(cls, value):
        import re
        if len(set(value)) != len(value) or any(
            name == "RF" or re.fullmatch(r"[A-Z][A-Z0-9_]{0,31}", name) is None for name in value
        ):
            raise ValueError("因子列名须唯一，使用大写字母、数字或下划线；RF 为保留列")
        return value

    @model_validator(mode="after")
    def rows_match(self):
        dates = [row.date for row in self.rows]
        if dates != sorted(set(dates)) or dates[-1] > date.today():
            raise ValueError("收益日期必须升序、唯一，且不能晚于今天")
        columns = set(self.factor_names) | ({"RF"} if self.dependent_return == "excess" else set())
        for row in self.rows:
            if set(row.values) != columns:
                raise ValueError("每行列名须与 factor_names 和回归口径严格一致；超额收益须提供 RF")
            rf = row.values.get("RF")
            if rf is not None and not -0.1 <= rf <= 0.1:
                raise ValueError("RF 必须为日频小数收益，不是年化利率")
        if not self.name.strip():
            raise ValueError("数据集名称不能为空")
        return self


class FormationRow(Contract):
    date: date
    asset: str = Field(pattern=r"^[A-Za-z0-9._-]{1,32}$")
    market_cap: float = Field(gt=0, le=1e18)
    december_market_cap: float = Field(gt=0, le=1e18)
    book_equity: float = Field(gt=0, le=1e18)
    fiscal_year_end: date
    announced_date: date
    reference_member: bool = True

    @model_validator(mode="after")
    def availability(self):
        if self.fiscal_year_end.year != self.date.year - 1:
            raise ValueError("账面权益必须来自形成日前一财年")
        if not self.fiscal_year_end < self.announced_date < self.date:
            raise ValueError("财报公告日须晚于财年末并严格早于形成日")
        return self


class AssetReturnRow(Contract):
    date: date
    asset: str = Field(pattern=r"^[A-Za-z0-9._-]{1,32}$")
    return_value: float | None = Field(ge=-1, le=10)
    lagged_market_cap: float = Field(gt=0, le=1e18)
    weight_date: date


class RiskFreeRow(Contract):
    date: date
    RF: float = Field(ge=-0.1, le=0.1)


class FF3SourceFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    source_url: str = Field(pattern=r"^https?://", max_length=500)
    market: str = Field(pattern=r"^[A-Z][A-Z0-9_-]{1,15}$")
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    frequency: Literal["daily"] = "daily"
    units: Literal["decimal_return"] = "decimal_return"
    construction: str = Field(min_length=10, max_length=2000)
    calendar: list[date] = Field(min_length=31, max_length=6000)
    formations: list[FormationRow] = Field(min_length=6, max_length=30000)
    returns: list[AssetReturnRow] = Field(min_length=30, max_length=300000)
    rf: list[RiskFreeRow] = Field(min_length=30, max_length=5999)

    @model_validator(mode="after")
    def temporal_panel(self):
        days = self.calendar
        if days != sorted(set(days)) or days[-1] > date.today():
            raise ValueError("交易日历须升序、唯一且不包含未来日期")
        june_ends = {days[i] for i in range(len(days) - 1)
                     if days[i].month == 6 and days[i].day >= 25 and days[i + 1].month == 7}
        formed = {row.date for row in self.formations}
        if formed != june_ends or days[0] not in formed:
            raise ValueError("日历须从六月形成日开始，并为区间内每个六月末提供形成截面")
        keys = [(row.date, row.asset) for row in self.formations]
        if len(set(keys)) != len(keys):
            raise ValueError("同一形成日的股票不能重复")
        prior = dict(zip(days[1:], days[:-1]))
        keys = set()
        assets = {row.asset for row in self.formations}
        observed_days = set()
        for row in self.returns:
            if prior.get(row.date) != row.weight_date:
                raise ValueError("市场收益权重必须来自该收益日前一交易日市值")
            key = (row.date, row.asset)
            if key in keys:
                raise ValueError("股票收益的日期和代码不能重复")
            keys.add(key)
            assets.add(row.asset)
            observed_days.add(row.date)
        if observed_days != set(days[1:]):
            raise ValueError("交易日历中的每个收益日必须有明确的市场成员数据")
        rf_dates = [row.date for row in self.rf]
        if rf_dates != days[1:]:
            raise ValueError("RF 须按交易日历逐日完整提供，不能补零或使用年化利率")
        if len(assets) > 1000 or len(assets) * len(days) > 2_000_000:
            raise ValueError("本次研究面板限1000个资产、200万个日期×资产单元，请缩小区间或研究池")
        if not self.name.strip():
            raise ValueError("面板名称不能为空")
        return self
