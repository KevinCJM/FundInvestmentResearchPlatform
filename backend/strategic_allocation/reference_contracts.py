"""Shared standalone reference-asset and proxy contracts for universal Risk Scale research."""
from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common_contracts import Contract, Fingerprint, Identifier, Number


class ExplicitConfirm(Contract):
    confirm: Literal[True]

    @field_validator("confirm", mode="before")
    @classmethod
    def explicit_true(cls, value):
        if value is not True:
            raise ValueError("需要明确的 true 确认。")
        return value


class ProxyComponent(Contract):
    kind: Literal["index", "etf", "fund"]
    series_id: str = Field(min_length=3, max_length=160)
    field: Literal["close", "close_hfq", "adj_nav"]
    weight: Number = Field(ge=0, le=1)

    @model_validator(mode="after")
    def identity(self):
        if not self.series_id.startswith(self.kind + ":"):
            raise ValueError("来源类型与序列标识不一致。")
        if self.kind == "index":
            if self.field != "close":
                raise ValueError("指数直接使用用户选择的指数值序列。")
        elif self.field not in ("close_hfq", "adj_nav"):
            raise ValueError("产品代理必须使用复权价格或复权净值，不允许使用未复权价格。")
        return self


class ReferenceAsset(Contract):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    asset_type: Literal["cash", "market"]
    rationale: str = Field(default='', max_length=2000)
    cash_return: Number | None = Field(ge=-0.5, le=1)
    components: list[ProxyComponent] = Field(max_length=30)
    rebalance: Literal["daily", "monthly", "quarterly", "yearly", "buy_and_hold"] | None

    @model_validator(mode="after")
    def asset_contract(self):
        if self.asset_type == "cash":
            if self.cash_return is None:
                raise ValueError("现金大类须明确预期年化收益率。")
            if self.components or self.rebalance is not None:
                raise ValueError("现金大类不能选择代理或再平衡规则。")
            return self
        if self.cash_return is not None:
            raise ValueError("非现金大类不能使用现金收益率。")
        if not self.components or self.rebalance is None:
            raise ValueError("非现金大类须选择至少一个代理并明确再平衡规则。")
        if abs(sum(x.weight for x in self.components) - 1) > 1e-10:
            raise ValueError("代理权重须明确合计 100%，不会自动归一化。")
        if len({(x.series_id, x.field) for x in self.components}) != len(self.components):
            raise ValueError("代理成分不能重复。")
        return self


class ReferenceInputRequest(Contract):
    name: str = Field(min_length=1, max_length=120)
    currency: Literal["CNY"] = "CNY"
    as_of: date
    calendar: Literal["SSE"] = "SSE"
    frequency: Literal["daily"] = "daily"
    periods_per_year: Literal[252] = 252
    return_basis: Literal["selected_index_and_adjusted_product_total_return"] = "selected_index_and_adjusted_product_total_return"
    fee_basis: Literal["source_embedded_no_additional_fee"] = "source_embedded_no_additional_fee"
    fx_basis: Literal["same_currency_no_conversion"] = "same_currency_no_conversion"
    assets: list[ReferenceAsset] = Field(min_length=2, max_length=30)

    @model_validator(mode="after")
    def validate_inputs(self):
        if self.as_of > date.today():
            raise ValueError("参考研究日不能位于未来。")
        if len({x.id for x in self.assets}) != len(self.assets):
            raise ValueError("资产标识不能重复。")
        if sum(x.asset_type == "cash" for x in self.assets) > 1:
            raise ValueError("一套风险标尺最多定义一个纯现金大类。")
        if not any(x.asset_type == "market" for x in self.assets):
            raise ValueError("风险标尺至少需要一个非现金大类用于构建风险收益前沿。")
        if sum(len(x.components) for x in self.assets) > 300:
            raise ValueError("代理成分超过资源上限 300。")
        return self


class ConfirmReferenceInput(ExplicitConfirm):
    request: ReferenceInputRequest
    preview_hash: Fingerprint
    confirm: Literal[True]
    idempotency_key: str = Field(min_length=8, max_length=120, pattern=r"^[A-Za-z0-9_.:-]+$")
    acknowledged_warnings: list[str] = Field(default_factory=list, max_length=40)
