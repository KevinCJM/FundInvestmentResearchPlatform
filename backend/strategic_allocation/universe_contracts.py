"""Strategic identity is independent of product names and product domains."""
from datetime import date
from typing import Literal
from pydantic import Field, model_validator
from .contracts import Contract, Currency, Fingerprint, Identifier
from .reference_contracts import ReferenceAsset, ProxyComponent, Number


class ResearchProxy(Contract):
    """Editable research input defaults, never an implementation mapping."""
    asset_type: Literal['cash', 'market']
    cash_return: Number | None = Field(default=None, ge=-0.5, le=1)
    components: list[ProxyComponent] = Field(default_factory=list, max_length=30)
    rebalance: Literal['daily', 'monthly', 'quarterly', 'yearly', 'buy_and_hold'] | None = None
    source_labels: dict[str, str] = Field(default_factory=dict, max_length=30)

    @model_validator(mode='after')
    def valid_proxy(self):
        # A scope may be saved before market proxies are chosen. Calculation
        # still requires complete ReferenceAsset inputs at the LTCMA boundary.
        if not (self.asset_type == 'market' and not self.components and self.cash_return is None and self.rebalance is not None):
            ReferenceAsset(id='proxy', name='proxy', rationale='', **self.model_dump(exclude={'source_labels'}))
        if any(len(k) > 160 or not v.strip() or len(v) > 120 for k, v in self.source_labels.items()):
            raise ValueError('代理名称或标识长度无效。')
        if set(self.source_labels) - {item.series_id for item in self.components}:
            raise ValueError('代理名称须对应已选来源。')
        return self


class WeightLimits(Contract):
    """Scope-level class bounds; downstream only tightens mandate authority with them."""
    min_weight: Number = Field(default=0, ge=0, le=1)
    max_weight: Number = Field(default=1, ge=0, le=1)

    @model_validator(mode='after')
    def ordered(self):
        if self.min_weight > self.max_weight:
            raise ValueError('大类权重下限不能高于上限。')
        return self


class StrategicAsset(Contract):
    id: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,79}$')
    name: Identifier
    currency: Currency
    role: Literal['growth', 'rates', 'inflation', 'credit', 'liquidity', 'diversifier']
    liquidity: Literal['liquid', 'illiquid']
    # Roles remain in the frozen downstream contract, but typed scope inputs
    # derive cash membership rather than asking users to repeat the choice.
    rationale: str = Field(default='', max_length=1000)
    source: str = Field(default='', max_length=1000)
    research_proxy: ResearchProxy | None = None
    weight_limits: WeightLimits | None = None

    @model_validator(mode='before')
    @classmethod
    def classify_typed_asset(cls, value):
        if not isinstance(value, dict):
            return value
        proxy = value.get('research_proxy')
        asset_type = proxy.get('asset_type') if isinstance(proxy, dict) else getattr(proxy, 'asset_type', None)
        if asset_type not in ('cash', 'market'):
            return value  # Preserve the explicit classification of untyped history.
        value = dict(value)
        if asset_type == 'cash':
            value.update(role='liquidity', liquidity='liquid')
        else:
            if value.get('role') in (None, 'liquidity'):
                value['role'] = 'growth'
            # Non-cash does not imply unrestricted liquidity. Keep previously
            # declared restrictions when editing/copying an existing scope.
            value.setdefault('liquidity', 'liquid')
        return value


class UniverseRequest(Contract):
    name: Identifier
    as_of: date
    currency: Currency = 'CNY'
    source: str = Field(default='', max_length=2000)
    assets: list[StrategicAsset] = Field(min_length=1, max_length=30)

    @model_validator(mode='after')
    def axis(self):
        if not self.name.strip():
            raise ValueError('战略范围名称不能为空。')
        if self.as_of > date.today():
            raise ValueError('战略范围研究日不能在未来。')
        if len({a.id for a in self.assets}) != len(self.assets):
            raise ValueError('战略资产ID必须唯一，不根据展示名推断身份。')
        if any(a.currency != self.currency for a in self.assets):
            raise ValueError('战略资产必须声明同一本位币风险口径；本模块不自动折汇。')
        if self.currency != 'CNY' and any(a.research_proxy and (a.research_proxy.asset_type == 'cash' or a.research_proxy.components) for a in self.assets):
            raise ValueError('研究代理目前仅支持人民币口径；其他币种请暂不设置代理。')
        cash_assets = [a for a in self.assets if a.research_proxy and a.research_proxy.asset_type == 'cash']
        if len(cash_assets) > 1:
            raise ValueError('研究范围最多定义一个纯现金大类。')
        if sum(a.weight_limits.min_weight for a in self.assets if a.weight_limits) > 1 + 1e-12:
            raise ValueError('各大类权重下限合计超过 100%，无法同时满足。')
        if sum(len(a.research_proxy.components) for a in self.assets if a.research_proxy) > 300:
            raise ValueError('代理成分超过资源上限 300。')
        return self


class ConfirmUniverseRequest(Contract):
    request: UniverseRequest
    preview_hash: Fingerprint
    # 编辑保存：以新不可变版本替代同一名称的当前版本，旧版本保留只读。
    replaces_universe_id: Identifier | None = None
    mandate_id: Identifier | None = None


class ScopeMandateBinding(Contract):
    mandate_id: Identifier


class ProxyAssignment(Contract):
    strategic_asset_id: Identifier
    proxy_asset_id: Identifier
    rationale: str = Field(default='', max_length=2000)


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
