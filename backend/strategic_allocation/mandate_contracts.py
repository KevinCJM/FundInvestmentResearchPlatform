"""Explicit governance and risk authorization; no market assumptions or solvers."""
from datetime import date
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .common_contracts import Contract, FrozenRef, Identifier, Number


class CapitalTarget(Contract):
    amount: Number = Field(ge=0, le=1e13)
    amount_basis: Literal["nominal", "real"] = "nominal"


class CashProtection(Contract):
    mode: Literal["payments_only", "payments_and_terminal_floor"]
    terminal_floor: CapitalTarget | None = None

    @model_validator(mode="after")
    def target(self):
        if (self.mode == "payments_and_terminal_floor") != (self.terminal_floor is not None):
            raise ValueError("期末资金保护须明确金额；仅支付保护不能残留期末底线。")
        return self


class MandatePolicy(Contract):
    """A dated, explicitly confirmed research policy, frozen inside a mandate."""
    name: Identifier
    source: str = Field(min_length=5, max_length=2000)
    reviewed_on: date
    valid_until: date | None = None
    confirmed: Literal[True]
    required_probability: Number | None = Field(default=None, ge=0.5, le=0.99)
    liquidity_months: int | None = Field(default=None, ge=1, le=36, strict=True)
    contribution_stress_ratio: Number | None = Field(default=None, ge=0, le=1)
    cash_reserve_weight: Number = Field(default=0, ge=0, le=1)

    @field_validator("confirmed", mode="before")
    @classmethod
    def explicit_confirmation(cls, value):
        if value is not True:
            raise ValueError("须明确确认本次边界政策；默认值不构成授权。")
        return value

    @model_validator(mode="after")
    def dates(self):
        if self.valid_until is not None and self.valid_until <= self.reviewed_on:
            raise ValueError("政策复核截止日须晚于核验日。")
        return self


class RiskAuthorization(Contract):
    mode: Literal["manual_level", "funding_suggestion", "explicit_numeric"]
    risk_scale_ref: FrozenRef | None = None
    authorized_max_level: int | None = Field(default=None, ge=1, le=5, strict=True)
    selected_max_level: int | None = Field(default=None, ge=1, le=5, strict=True)
    source: str = Field(default="risk_scale_selection", min_length=5, max_length=2000)

    @model_validator(mode="after")
    def levels(self):
        if self.mode == "explicit_numeric":
            if self.risk_scale_ref or self.authorized_max_level or self.selected_max_level:
                raise ValueError("明确数值授权不同时冒充风险等级授权。")
            return self
        if self.authorized_max_level is None:
            raise ValueError("请明确最高授权等级；模型不能代替机构决定授权。")
        if self.mode == "manual_level" and self.selected_max_level is None:
            raise ValueError("手动模式须明确本次采用的风险上限。")
        if self.selected_max_level is not None and self.selected_max_level > self.authorized_max_level:
            raise ValueError("本次选择不能超过已确认的最高授权等级。")
        return self
