"""Validated public contracts; all numerical work lives in numba_kernels."""
from __future__ import annotations

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Kind = Literal["etf", "fund", "stock"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FactorFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=600)
    operator: Literal["momentum", "volatility", "drawdown", "reversal"]
    window: int = Field(ge=2, le=504)
    skip: int = Field(default=0, ge=0, le=126)
    direction: Literal[-1, 1] = 1
    product_kinds: list[Kind] = Field(default_factory=lambda: ["etf", "fund"], min_length=1, max_length=3)

    @field_validator("name")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("名称不能为空")
        return value.strip()


class FactorUpdate(FactorFields):
    revision: int = Field(ge=1)


class FactorRef(Contract):
    factor_id: str = Field(min_length=1, max_length=120)
    revision: int = Field(ge=1)
    weight: float = Field(gt=0, le=100)


class Benchmark(Contract):
    kind: Literal["etf", "index"] = "etf"
    code: str = Field(min_length=1, max_length=32)
    label: str = Field(min_length=1, max_length=120)
    return_basis: Literal["adjusted_nav", "price_index", "total_return_index"] = "adjusted_nav"

    @model_validator(mode="after")
    def basis_matches(self):
        if (self.kind == "etf") != (self.return_basis == "adjusted_nav"):
            raise ValueError("ETF 基准须为复权净值，指数须明确价格或全收益口径")
        return self


class StudyFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    product_kind: Kind = "etf"
    asset_class: Literal["equity", "bond", "commodity", "multi_asset"] = "equity"
    market: Literal["CN"] = "CN"
    currency: Literal["CNY"] = "CNY"
    targets: list[str] = Field(min_length=3, max_length=120)
    universe_source: str = Field(default="manual_fixed", min_length=1, max_length=160)
    start_date: date
    end_date: date
    oos_date: date
    benchmark: Benchmark
    factors: list[FactorRef] = Field(min_length=1, max_length=8)
    normalization: Literal["rank", "zscore"] = "rank"
    horizon: int = Field(default=21, ge=1, le=126)
    signal_frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    ic_window: int = Field(default=12, ge=3, le=252)
    ic_min_periods: int = Field(default=6, ge=3, le=252)
    quantiles: int = Field(default=3, ge=2, le=10)
    top_n: int = Field(default=4, ge=1, le=120)
    cost_bps: float = Field(default=5, ge=0, le=500)
    model: Literal["characteristic_composite"] = "characteristic_composite"
    dataset: Literal["active_adjusted_nav"] = "active_adjusted_nav"

    @model_validator(mode="after")
    def consistent(self):
        if not self.start_date < self.oos_date < self.end_date:
            raise ValueError("样本外日期必须位于研究区间内部")
        if self.end_date > date.today():
            raise ValueError("研究截止日不能晚于今天")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("研究产品不能重复")
        if any(not code.strip() or len(code) > 32 for code in self.targets):
            raise ValueError("产品代码无效")
        if len({item.factor_id for item in self.factors}) != len(self.factors):
            raise ValueError("同一因子不能重复加入组合")
        if self.ic_min_periods > self.ic_window:
            raise ValueError("滚动 IC 最少有效截面不能超过统计窗口")
        if self.quantiles > len(self.targets) or self.top_n > len(self.targets):
            raise ValueError("分组数和 Top N 不能超过研究产品数")
        if not self.name.strip():
            raise ValueError("方案名称不能为空")
        return self


class StudyUpdate(StudyFields):
    revision: int = Field(ge=1)


class RunRequest(Contract):
    revision: int = Field(ge=1)


class ReleaseFields(Contract):
    run_id: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=80)
    effective_from: date
    effective_to: date
    note: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def dates(self):
        if self.effective_to < self.effective_from:
            raise ValueError("失效日不能早于生效日")
        return self


class FactorReturnRow(Contract):
    date: date
    MKT_RF: float = Field(ge=-2, le=2)
    SMB: float = Field(ge=-2, le=2)
    HML: float = Field(ge=-2, le=2)
    RF: float = Field(ge=-0.1, le=0.1)


class DatasetFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    source_url: str = Field(pattern=r"^https?://", max_length=500)
    market: str = Field(min_length=1, max_length=40)
    currency: str = Field(min_length=3, max_length=3)
    frequency: Literal["daily"] = "daily"
    units: Literal["decimal_return"] = "decimal_return"
    construction: str = Field(min_length=10, max_length=2000)
    rows: list[FactorReturnRow] = Field(min_length=30, max_length=6000)

    @model_validator(mode="after")
    def unique_dates(self):
        dates = [item.date for item in self.rows]
        if len(set(dates)) != len(dates) or dates != sorted(dates):
            raise ValueError("因子收益日期必须升序且唯一")
        return self


class AttributionFields(Contract):
    name: str = Field(min_length=1, max_length=80)
    product_kind: Literal["etf", "fund"] = "fund"
    targets: list[str] = Field(min_length=1, max_length=40)
    model: Literal["rbsa", "ff3", "factor_regression"] = "rbsa"
    exposure_mode: Literal["fixed", "rolling"] = "fixed"
    rolling_window: int = Field(default=126, ge=30, le=1260)
    min_observations: int = Field(default=60, ge=30, le=1260)
    refit_step: int = Field(default=21, ge=1, le=252)
    indices: list[str] = Field(default_factory=list, max_length=8)
    dataset_id: str | None = None
    market: Literal["CN"] = "CN"
    currency: Literal["CNY"] = "CNY"
    start_date: date
    end_date: date
    oos_date: date

    @model_validator(mode="after")
    def compatible(self):
        if self.exposure_mode == "rolling" and self.min_observations > self.rolling_window:
            raise ValueError("最少有效样本不能超过滚动窗口")
        if not self.start_date < self.oos_date < self.end_date <= date.today():
            raise ValueError("归因区间或样本外日期无效")
        if len(set(self.targets)) != len(self.targets) or len(set(self.indices)) != len(self.indices):
            raise ValueError("研究产品和代理指数不能重复")
        if self.model == "rbsa" and not 2 <= len(self.indices) <= 8:
            raise ValueError("RBSA 需要 2–8 个不同指数代理")
        if self.model != "rbsa" and not self.dataset_id:
            raise ValueError("因子回归必须选择明确的因子收益数据集")
        return self


class BindingFields(Contract):
    release_id: str = Field(min_length=1, max_length=120)
    context_type: Literal["product_research", "saa", "taa", "allocation", "portfolio", "post_investment", "regime"]
    context_id: str = Field(min_length=1, max_length=120)
    note: str = Field(default="", max_length=600)


class Holding(Contract):
    product_id: str = Field(min_length=1, max_length=32)
    weight: float = Field(ge=0, le=1)


class PortfolioProfile(Contract):
    holdings: list[Holding] = Field(min_length=1, max_length=120)
    as_of: date
