"""Strategic identity is independent of product names and product domains."""
from datetime import date
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Contract, Currency, Fingerprint, Identifier


class StrategicAsset(Contract):
    id: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,79}$')
    name: Identifier
    currency: Currency
    role: Literal['growth', 'rates', 'inflation', 'credit', 'liquidity', 'diversifier']
    liquidity: Literal['liquid', 'illiquid']
    rationale: str = Field(min_length=3, max_length=1000)
    source: str = Field(min_length=3, max_length=1000)


class UniverseRequest(Contract):
    name: Identifier
    as_of: date
    currency: Currency = 'CNY'
    source: str = Field(min_length=3, max_length=2000)
    assets: list[StrategicAsset] = Field(min_length=1, max_length=30)

    @model_validator(mode='after')
    def axis(self):
        if self.as_of > date.today():
            raise ValueError('战略范围研究日不能在未来。')
        if len({a.id for a in self.assets}) != len(self.assets):
            raise ValueError('战略资产ID必须唯一，不根据展示名推断身份。')
        if any(a.currency != self.currency for a in self.assets):
            raise ValueError('战略资产必须声明同一本位币风险口径；本模块不自动折汇。')
        return self


class ConfirmUniverseRequest(Contract):
    request: UniverseRequest
    preview_hash: Fingerprint


class ProxyAssignment(Contract):
    strategic_asset_id: Identifier
    proxy_asset_id: Identifier
    rationale: str = Field(min_length=3, max_length=2000)


class ImplementationMapRequest(Contract):
    name: Identifier
    strategic_universe_id: Identifier
    universe_snapshot_id: Identifier
    alloc_name: Identifier
    as_of: date
    valid_until: date
    assignments: list[ProxyAssignment] = Field(default_factory=list, max_length=30)

    @model_validator(mode='after')
    def mapping(self):
        if not self.as_of < self.valid_until or self.as_of > date.today():
            raise ValueError('映射研究日不能在未来，有效期须晚于研究日。')
        if len({a.strategic_asset_id for a in self.assignments}) != len(self.assignments):
            raise ValueError('每个战略资产只能显式分配一个真实代理大类。')
        if len({a.proxy_asset_id for a in self.assignments}) != len(self.assignments):
            raise ValueError('同一代理不能重复分配到多个战略资产，避免重复预算归属。')
        return self


class ConfirmImplementationMapRequest(Contract):
    request: ImplementationMapRequest
    preview_hash: Fingerprint
