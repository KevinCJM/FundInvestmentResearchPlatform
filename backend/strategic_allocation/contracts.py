"""Versioned, same-currency, long-only research inputs; no client performance."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from backend.tactical_allocation.contracts import AssetLimit, GroupLimit
from .institution_contracts import InstitutionalContext
from .mandate_contracts import CapitalTarget, CashProtection, MandatePolicy, RiskAuthorization

from .common_contracts import Contract, Number, Identifier, Fingerprint, Currency
from .cma_model_contracts import CmaModelRequest


class FundingFlow(Contract):
    name: Identifier
    kind: Literal["contribution", "withdrawal"]
    amount: Number = Field(gt=0, le=1e12)
    first_month: int = Field(ge=1, le=360, strict=True)
    last_month: int = Field(ge=1, le=360, strict=True)
    every_months: Literal[1, 3, 12] = 1

    @model_validator(mode="after")
    def interval(self):
        if self.last_month < self.first_month:
            raise ValueError("现金流结束月不能早于开始月；单次支付请将两者设为同一月。")
        return self


class FundingPlan(Contract):
    total_capital: Number = Field(gt=0, le=1e12)
    outside_reserve: Number = Field(default=0, ge=0, le=1e12)
    terminal_target: Number = Field(ge=0, le=1e13)
    amount_basis: Literal["nominal", "real"] = "nominal"
    inflation: Number = Field(default=0, ge=-0.05, le=0.20)
    annual_fee: Number = Field(default=0, ge=0, le=0.10)
    required_probability: Number = Field(ge=0.5, le=0.99)
    liquidity_months: int = Field(default=12, ge=1, le=36, strict=True)
    contribution_stress_ratio: Number = Field(default=0.5, ge=0, le=1)
    drawdown_alert: Number = Field(default=0.2, gt=0, le=1)
    flows: list[FundingFlow] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def capital(self):
        if self.outside_reserve >= self.total_capital:
            raise ValueError("组合外储备必须小于总资金，须保留正的可投资本金。")
        if self.terminal_target == 0 and not any(flow.kind == "withdrawal" for flow in self.flows):
            raise ValueError("期末目标为0时，须至少定义一笔必要支付，不能把空目标评为成功。")
        return self


class CashBudget(Contract):
    """Funding facts independent of the investment success criterion."""
    total_capital: Number = Field(gt=0, le=1e12)
    outside_reserve: Number = Field(default=0, ge=0, le=1e12)
    balance_as_of: date
    source: str = Field(default="investment_objectives_cash_plan", min_length=5, max_length=2000)
    amount_basis: Literal["nominal", "real"] = "nominal"
    inflation: Number = Field(default=0, ge=-0.05, le=0.20)
    annual_fee: Number = Field(default=0, ge=0, le=0.10)
    flows: list[FundingFlow] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def capital(self):
        if self.outside_reserve >= self.total_capital:
            raise ValueError("组合外储备必须小于总资金，不能重复扣减或产生负本金。")
        return self


class BenchmarkPolicy(Contract):
    name: Identifier
    alloc_name: Identifier
    weights: dict[Identifier, Number] = Field(min_length=1, max_length=30)
    target_excess_return: Number = Field(ge=-0.5, le=1)
    max_tracking_error: Number = Field(ge=0, le=1)
    source: Literal["explicit", "risk_scale_reference"] = "explicit"

    @model_validator(mode="after")
    def full_investment(self):
        if any(value < 0 or value > 1 for value in self.weights.values()) or abs(sum(self.weights.values()) - 1) > 1e-8:
            raise ValueError("基准大类权重须非负且合计为100%。")
        return self


class MandateRequest(Contract):
    schema_version: Literal["1.0", "2.0"] = "1.0"
    name: Identifier
    as_of: date
    review_date: date | None = None
    currency: Currency = "CNY"
    horizon_years: int = Field(default=10, ge=1, le=30, strict=True)
    target_return: Number = Field(default=0.0, ge=-0.5, le=1)
    target_excess_return: Number = Field(default=0.0, ge=-0.5, le=1)
    min_cash_weight: Number = Field(default=0.0, ge=0, le=1)
    max_volatility: Number | None = Field(default=0.15, gt=0, le=2)
    min_liquid_weight: Number = Field(default=0, ge=0, le=1)
    max_illiquid_weight: Number = Field(default=0, ge=0, le=1)
    max_tracking_error: Number = Field(default=0.10, ge=0, le=1)
    risk_aversion: Number = Field(default=5, gt=0, le=1000)
    rebalance_policy: Literal["monthly", "quarterly", "annually", "threshold"] = "quarterly"
    rebalance_note: str = Field(default="", max_length=1000)
    note: str = Field(default="", max_length=2000)
    objective_kind: Literal["absolute_return", "funding_goal", "benchmark_relative"] = "absolute_return"
    funding_plan: FundingPlan | None = None
    cash_budget: CashBudget | None = None
    funding_target: CapitalTarget | None = None
    cash_protection: CashProtection | None = None
    boundary_policy: MandatePolicy | None = None
    risk_authorization: RiskAuthorization | None = None
    benchmark: BenchmarkPolicy | None = None
    # 合同里写的业绩比较基准原文。只做留痕：建模仍走上面的大类权重，两者口径存在基差。
    stated_benchmark: str = Field(default="", max_length=500)
    boundary_reason: str = Field(default="", max_length=2000)
    institutional_context: InstitutionalContext | None = None
    strategic_universe_id: Identifier | None = None
    allocation_scope: Identifier | None = None
    asset_limits: dict[str, AssetLimit] = Field(default_factory=dict, max_length=30)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def dates(self):
        if self.as_of > date.today() or self.review_date is not None and self.review_date <= self.as_of:
            raise ValueError("研究日不能在未来；填写政策复核日时须晚于研究日。")
        if self.schema_version == "1.0":
            if self.review_date is None:
                raise ValueError("旧版目标必须提供政策复核日。")
            if any(value is not None for value in (self.cash_budget, self.funding_target, self.cash_protection,
                                                   self.boundary_policy, self.risk_authorization)):
                raise ValueError("新资金与授权字段须显式使用2.0版本，不能改变旧请求的语义。")
            if self.max_volatility is None:
                raise ValueError("原数值授权必须提供有效波动上限。")
            if (self.objective_kind == "funding_goal") != (self.funding_plan is not None):
                raise ValueError("金额目标须提供资金计划；其他目标不能残留金额计划。")
        else:
            self._validate_budget_authorization()
        if self.objective_kind == "benchmark_relative":
            if self.schema_version == "1.0" and self.benchmark is None:
                raise ValueError("旧版相对目标须提供真实基准权重。")
        elif self.benchmark is not None:
            raise ValueError("非相对目标不能残留基准设置。")
        if self.objective_kind != "absolute_return" and self.target_return != 0:
            raise ValueError("非绝对收益目标不使用最低算术收益字段，请清零；所需复合收益另行计算。")
        if self.objective_kind != "benchmark_relative" and self.target_excess_return != 0:
            raise ValueError("非相对目标不使用目标超额收益字段，请清零。")
        if self.objective_kind != "benchmark_relative" and self.stated_benchmark:
            raise ValueError("非相对目标不记录业绩比较基准原文，请清空。")
        if self.funding_plan:
            months = self.horizon_years * 12
            if self.funding_plan.liquidity_months > months or any(flow.last_month > months for flow in self.funding_plan.flows):
                raise ValueError("现金流或流动性窗口超出了投资期限；不能静默截断支付计划。")
        if self.rebalance_policy == "threshold" and not self.rebalance_note:
            raise ValueError("阈值再平衡须说明触发和恢复规则；本页只记录政策，不自动交易。")
        if self.strategic_universe_id and self.allocation_scope:
            raise ValueError("战略范围与产品大类授权只能显式选择一种来源。")
        if self.institutional_context:
            context = self.institutional_context
            if context.balance_sheet and (context.balance_sheet.as_of != self.as_of or context.balance_sheet.currency != self.currency):
                raise ValueError("经济状况快照须与目标同研究日、同本位币；不自动折算或滚动。")
            if any(item.reviewed_on and item.reviewed_on > self.as_of for item in context.review_items):
                raise ValueError("人工核验日不得晚于目标研究日。")
        if (self.asset_limits or self.group_limits) and not (self.allocation_scope or self.strategic_universe_id):
            raise ValueError("资产或分组授权必须绑定所属大类方案，不能仅按资产名称复用。")
        if (self.benchmark and self.benchmark.source == "explicit" and self.allocation_scope
                and self.benchmark.alloc_name != self.allocation_scope):
            raise ValueError("相对基准与资产授权必须属于同一个大类方案。")
        seen_groups = set()
        for group in self.group_limits:
            if (group.id in seen_groups or not group.assets or len(set(group.assets)) != len(group.assets)
                    or group.lo > group.hi):
                raise ValueError("投资授权分组必须有唯一名称、非空不重复成员及有效上下界。")
            seen_groups.add(group.id)
        return self

    def _validate_budget_authorization(self):
        if self.funding_plan is not None:
            raise ValueError("2.0资金预算不能与旧funding_plan重复提供。")
        if self.risk_authorization is None:
            raise ValueError("新版目标须选择风险等级配置和最大风险等级。")
        policy, risk = self.boundary_policy, self.risk_authorization
        if policy is not None:
            if policy.reviewed_on > self.as_of:
                raise ValueError("政策核验日不能晚于目标研究日。")
            if policy.valid_until is not None and (self.as_of >= policy.valid_until
                    or self.review_date is not None and self.review_date > policy.valid_until):
                raise ValueError("目标研究日或复核日超过了政策有效期。")
        if risk.mode == "explicit_numeric":
            if self.max_volatility is None:
                raise ValueError("明确数值授权须填写波动上限。")
        elif self.max_volatility is not None:
            raise ValueError("等级模式的数值上限由服务端解析，请勿另填max_volatility。")
        if (self.objective_kind == "funding_goal") != (self.funding_target is not None):
            raise ValueError("资金目标须有期末余额条件，其他目标不能残留该条件。")
        if self.objective_kind == "funding_goal" and (self.cash_budget is None or self.cash_protection is not None):
            raise ValueError("资金目标须提供现金预算，不可重复提供现金保护成功条件。")
        if self.cash_protection is not None and self.cash_budget is None:
            raise ValueError("现金保护需要真实现金预算。")
        if risk.mode == "funding_suggestion" and (self.cash_budget is None
                or self.funding_target is None and self.cash_protection is None):
            raise ValueError("资金建议模式需要现金预算和明确的资金成功条件。")
        if self.cash_budget is None:
            return
        budget, months = self.cash_budget, self.horizon_years * 12
        if budget.balance_as_of != self.as_of:
            raise ValueError("资金余额日须与目标研究日一致，不自动滚动资金计划。")
        if any(flow.last_month > months for flow in budget.flows):
            raise ValueError("现金流不能超出投资期限。")
        if policy is not None and (policy.liquidity_months is None or policy.contribution_stress_ratio is None
                or policy.liquidity_months > months):
            raise ValueError("已提供的政策须包含有效保障窗口和压力投入比例。")
        payments = any(flow.kind == "withdrawal" for flow in budget.flows)
        target = self.funding_target or (self.cash_protection.terminal_floor if self.cash_protection else None)
        has_condition = payments or (target is not None and target.amount > 0)
        if (self.funding_target is not None or self.cash_protection is not None
                or risk.mode == "funding_suggestion") and not has_condition:
            raise ValueError("没有支付和正的期末条件，不能把空目标评为成功。")
        if has_condition and policy is not None and policy.required_probability is None:
            raise ValueError("已提供的政策须包含资金成功概率门槛。")


class MandateStudyRequest(Contract):
    definition: MandateRequest
    cma_id: Identifier | None = None
    simulation_paths: int = Field(default=2000, ge=500, le=10000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)
    uncertainty_penalty: Number = Field(default=1, ge=0, le=5)
    validation_seed: int = Field(default=104729, ge=0, le=2**32 - 1, strict=True)

    @model_validator(mode="after")
    def independent_samples(self):
        if self.definition.schema_version == "2.0" and self.seed in (self.validation_seed, self.validation_seed ^ 0x9E3779B9):
            raise ValueError("搜索、参考验证与SAA独立验证必须使用不同随机种子。")
        return self


class ConfirmMandateRequest(Contract):
    request: MandateStudyRequest
    preview_hash: Fingerprint
    acknowledge_limits: Literal[True]
    replaces_mandate_id: Identifier | None = None


class RiskReferenceRequest(Contract):
    alloc_name: Identifier
    as_of: date
    start_date: date
    end_date: date
    shrinkage: Number = Field(default=0.1, ge=0, le=1)
    periods_per_year: int = Field(default=252, ge=1, le=366, strict=True)

    @model_validator(mode="after")
    def dates(self):
        if not self.start_date < self.end_date <= self.as_of <= date.today():
            raise ValueError("日期须满足：样本开始 < 样本结束 ≤ 研究日 ≤ 今天。")
        return self


class AssetAssumption(Contract):
    id: Identifier
    role: Literal["growth", "rates", "inflation", "credit", "liquidity", "diversifier"]
    liquidity: Literal["liquid", "illiquid"]
    rationale: str = Field(min_length=3, max_length=1000)
    annual_return: Number | None = Field(default=None, ge=-0.5, le=2)
    annual_volatility: Number | None = Field(default=None, gt=0, le=3)
    mean_uncertainty: Number = Field(ge=0, le=1)


class CmaRequest(Contract):
    name: Identifier
    alloc_name: Identifier | None = None
    strategic_universe_id: Identifier | None = None
    implementation_mapping_id: Identifier | None = None
    as_of: date
    currency: Currency = "CNY"
    horizon_years: int = Field(default=10, ge=1, le=30, strict=True)
    return_basis: Literal["annual_arithmetic_total_return"] = "annual_arithmetic_total_return"
    source: str = Field(min_length=3, max_length=2000)
    basis_confirmed: Literal[True]
    assets: list[AssetAssumption] = Field(min_length=1, max_length=30)
    correlation: list[list[Number]] | None = Field(default=None, min_length=1, max_length=30)
    model: CmaModelRequest | None = None
    risk_origin: Literal["manual", "historical_reference"] = "manual"
    risk_reference: RiskReferenceRequest | None = None
    risk_reference_hash: Fingerprint | None = None

    @model_validator(mode="after")
    def shape(self):
        if bool(self.alloc_name) == bool(self.strategic_universe_id):
            raise ValueError("须明确选择真实大类方案或不可变战略范围，不能混用来源。")
        if self.implementation_mapping_id and not self.strategic_universe_id:
            raise ValueError("实施映射必须绑定独立战略范围。")
        if self.strategic_universe_id and self.risk_origin != "manual":
            raise ValueError("战略先行使用显式前瞻风险假设；历史风险参考请在真实代理研究中核验后人工引用。")
        if self.as_of > date.today():
            raise ValueError("长期假设的研究日不能位于未来。")
        count = len(self.assets)
        if len({a.id for a in self.assets}) != count:
            raise ValueError("每个资产类别只能有一条假设。")
        if self.model is None:
            if any(a.annual_return is None or a.annual_volatility is None for a in self.assets):
                raise ValueError("人工模式须填写每项资产的预期收益与波动率。")
            if self.correlation is None:
                raise ValueError("人工模式须填写完整相关矩阵。")
        else:
            model = self.model
            if (model.asset_ids != [a.id for a in self.assets] or model.as_of != self.as_of
                    or model.currency != self.currency or model.return_basis != self.return_basis):
                raise ValueError("模型须与长期假设采用完全相同的资产顺序、研究日、币种及收益口径。")
            if self.risk_origin != "manual" or self.risk_reference is not None or self.risk_reference_hash is not None:
                raise ValueError("模型风险使用其显式来源，不能沿用无关历史风险参考认证。")
        if self.correlation is not None and (len(self.correlation) != count or any(len(row) != count for row in self.correlation)):
            raise ValueError("相关矩阵行列须与资产列表完全一致。")
        if self.risk_origin == "historical_reference":
            if not self.risk_reference or not self.risk_reference_hash:
                raise ValueError("历史风险参考须保留样本区间和来源校验。")
            if self.risk_reference.alloc_name != self.alloc_name or self.risk_reference.as_of != self.as_of:
                raise ValueError("风险参考须属于同一分类与研究日。")
        elif self.risk_reference is not None or self.risk_reference_hash is not None:
            raise ValueError("人工风险假设不能保留旧历史参考认证，请清除引用。")
        return self


class PublishCmaRequest(Contract):
    request: CmaRequest
    preview_hash: Fingerprint


class PolicyRequest(Contract):
    mandate_id: Identifier
    cma_id: Identifier
    constraints: dict[str, AssetLimit] = Field(default_factory=dict)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=28)
    uncertainty_penalty: Number = Field(default=1, ge=0, le=5)
    risk_budget: dict[Identifier, Number] | None = Field(default=None, min_length=1, max_length=30)
    candidate_count: int = Field(default=2000, ge=200, le=5000, strict=True)
    seed: int = Field(default=42, ge=0, le=2**32 - 1, strict=True)


    @model_validator(mode="after")
    def budget(self):
        if self.risk_budget is not None and (any(v < 0 for v in self.risk_budget.values())
                or abs(sum(self.risk_budget.values()) - 1) > 1e-8):
            raise ValueError("风险预算须非负且合计100%；未提供时保留原四类候选。")
        return self


class PublishPolicyRequest(Contract):
    request: PolicyRequest
    preview_hash: Fingerprint
    candidate_id: Literal["minimum-risk", "nominal-utility", "robust-utility", "maximum-return", "risk-budget"]
    name: Identifier
    reason: str = Field(min_length=5, max_length=2000)
