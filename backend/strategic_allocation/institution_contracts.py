"""Optional institutional facts; no tax, legal or approval engine."""
from datetime import date
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(strict=True)]
ReviewTopic = Literal['tax', 'regulation', 'currency_hedging', 'leverage', 'special_liquidity']
REVIEW_TOPICS = ('tax', 'regulation', 'currency_hedging', 'leverage', 'special_liquidity')


class InstitutionalContract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)


class EconomicBalanceSheet(InstitutionalContract):
    as_of: date
    currency: str = Field(pattern=r'^[A-Z]{3}$')
    source: str = Field(min_length=3, max_length=2000)
    investable_assets: Number | None = Field(default=None, ge=0, le=1e15)
    outside_assets: Number | None = Field(default=None, ge=0, le=1e15)
    confirmed_liabilities: Number | None = Field(default=None, ge=0, le=1e15)
    uncalled_commitments: Number | None = Field(default=None, ge=0, le=1e15)


class ManualReviewItem(InstitutionalContract):
    topic: ReviewTopic
    status: Literal['not_assessed', 'pending', 'researcher_checked', 'not_applicable'] = 'not_assessed'
    reason: str = Field(default='', max_length=2000)
    evidence: str = Field(default='', max_length=2000)
    reviewed_on: date | None = None
    valid_until: date | None = None

    @model_validator(mode='after')
    def evidence_boundary(self):
        if self.status in {'researcher_checked', 'not_applicable'}:
            if not self.reviewed_on or not self.valid_until or len(self.reason) < 3 or len(self.evidence) < 3:
                raise ValueError('已核对或不适用须记录理由、证据、核验日与有效期；不构成独立审批。')
        if self.reviewed_on and self.valid_until and self.valid_until <= self.reviewed_on:
            raise ValueError('人工核验有效期须晚于核验日。')
        return self


class InstitutionalContext(InstitutionalContract):
    investor_type: Literal['personal', 'family_office', 'asset_manager', 'corporate_treasury']
    purpose: str = Field(min_length=3, max_length=2000)
    cash_reserve_weight: Number = Field(default=0, ge=0, le=1)
    balance_sheet: EconomicBalanceSheet | None = None
    review_items: list[ManualReviewItem] = Field(default_factory=lambda: [
        ManualReviewItem(topic=topic) for topic in REVIEW_TOPICS], min_length=5, max_length=5)

    @model_validator(mode='after')
    def complete_review_topics(self):
        if {item.topic for item in self.review_items} != set(REVIEW_TOPICS):
            raise ValueError('须逐项保留税务、监管、币种对冲、杠杆和特殊流动性的核验状态，未评估不能删除。')
        return self
