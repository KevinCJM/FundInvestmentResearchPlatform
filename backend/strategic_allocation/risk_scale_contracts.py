"""Single Pydantic/OpenAPI contract, also used to generate UI types."""
from datetime import date
from typing import Any, Literal
from pydantic import Field, model_validator
from .common_contracts import Contract, Number, Fingerprint, FrozenRef
from .reference_contracts import ExplicitConfirm
from backend.tactical_allocation.contracts import GroupLimit

AlgorithmId = Literal['frontier_shape_dp_v2', 'equal_volatility_v1', 'equal_arclength_v1', 'equal_return_v1', 'manual_volatility_bands_v1']
RiskBasisId = Literal['annualized-periodic-volatility-v1']


class Problem(Contract):
    code: str
    message: str
    field: str | None = None
    suggested_action: str = '请核对输入或重新加载后重试。'


class ErrorResponse(Contract):
    detail: Problem


class Segmentation(Contract):
    algorithm_id: AlgorithmId = 'frontier_shape_dp_v2'
    manual_caps: list[Number] | None = Field(default=None, min_length=5, max_length=5)
    adjusted_caps: list[Number] | None = Field(default=None, min_length=5, max_length=5)
    rationale: str = Field(default='', max_length=2000)

    @staticmethod
    def _validate_caps(caps: list[float]) -> None:
        if caps[0] < 0 or any(a >= b for a, b in zip(caps, caps[1:])):
            raise ValueError('风险上限须非负且严格递增。')

    @model_validator(mode='after')
    def bands(self):
        if self.algorithm_id == 'manual_volatility_bands_v1':
            if self.manual_caps is None or len(self.rationale.strip()) < 5:
                raise ValueError('人工阈值需要五个上限和依据。')
            if self.adjusted_caps is not None:
                raise ValueError('人工阈值算法不能再叠加边界微调。')
            self._validate_caps(self.manual_caps)
        else:
            if self.manual_caps is not None or self.rationale:
                raise ValueError('自动算法不能残留人工阈值或覆盖理由。')
            if self.adjusted_caps is not None:
                self._validate_caps(self.adjusted_caps)
        return self


class RiskScaleAssetLimit(Contract):
    min_weight: Number = Field(default=0, ge=0, le=1)
    max_weight: Number = Field(default=1, ge=0, le=1)


class ConstraintProfile(Contract):
    asset_limits: dict[str, RiskScaleAssetLimit] = Field(default_factory=dict, max_length=30)
    group_limits: list[GroupLimit] = Field(default_factory=list, max_length=28)
    allow_short: Literal[False] = False
    allow_leverage: Literal[False] = False
    gross_exposure: Literal[1] = 1


class RiskScaleDefinition(Contract):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default='', max_length=2000)
    scheme_id: str = Field(pattern=r'^scheme-[a-z0-9-]{1,80}$')
    base_currency: Literal['CNY', 'USD', 'HKD'] = 'CNY'
    risk_basis_id: RiskBasisId = 'annualized-periodic-volatility-v1'
    research_as_of: date
    review_due_at: date | None = None
    valid_until: date | None = None
    reference_input_ref: FrozenRef
    constraint_profile: ConstraintProfile = Field(default_factory=ConstraintProfile)
    segmentation: Segmentation = Field(default_factory=Segmentation)
    purpose: str = Field(default='', max_length=2000)

    @model_validator(mode='after')
    def clock(self):
        if self.research_as_of > date.today():
            raise ValueError('参考研究日不能在未来。')
        if self.review_due_at and self.review_due_at <= self.research_as_of:
            raise ValueError('预计复核日须晚于参考研究日。')
        if self.valid_until and self.valid_until <= self.research_as_of:
            raise ValueError('失效日须晚于参考研究日。')
        return self


class PreviewRequest(Contract):
    definition: RiskScaleDefinition
    draft_id: str | None = None
    draft_revision: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode='after')
    def revision(self):
        if (self.draft_id is None) != (self.draft_revision is None):
            raise ValueError('草稿标识与修订号必须同时提供。')
        return self


class ConfirmRequest(ExplicitConfirm):
    request: PreviewRequest
    confirm: Literal[True]
    preview_hash: Fingerprint
    idempotency_key: str = Field(min_length=8, max_length=120, pattern=r'^[A-Za-z0-9_.:-]+$')
    acknowledged_warnings: list[str] = Field(default_factory=list, max_length=40)


class DraftWrite(Contract):
    name: str = Field(min_length=1, max_length=120)
    scheme_id: str = Field(pattern=r'^scheme-[a-z0-9-]{1,80}$')
    editable_definition: dict[str, Any] = Field(default_factory=dict)


class DraftUpdate(DraftWrite):
    expected_revision: int = Field(ge=1, strict=True)


class RevisionRequest(Contract):
    expected_revision: int = Field(ge=1, strict=True)


class DraftView(DraftWrite):
    id: str
    revision: int
    created_at: str
    updated_at: str


class ActivateRequest(ExplicitConfirm):
    confirm: Literal[True]
    expected_revision: int = Field(ge=0, strict=True)


class RetireRequest(ActivateRequest):
    reason: str = Field(min_length=5, max_length=2000)
    clear_default: bool = False
    replacement_id: str | None = None

    @model_validator(mode='after')
    def replacement(self):
        if self.clear_default and self.replacement_id:
            raise ValueError('清空默认与替换默认不能同时选择。')
        return self


class CompareRequest(Contract):
    left_id: str
    right_id: str


class ClassifyRequest(Contract):
    volatility: Number | None
    authorized_level: int = Field(default=3, ge=1, le=5, strict=True)


class Metric(Contract):
    value: Number | None = None
    status: str
    reason: str | None = None
    unit: str = 'decimal'


class Level(Contract):
    level_code: Literal['C1', 'C2', 'C3', 'C4', 'C5']
    lower_bound: Number
    lower_inclusive: bool
    upper_bound: Number
    upper_inclusive: Literal[True] = True
    authorized_volatility_cap: Number
    calibration_status: str
    representative_node_id: int | None
    representative_weights: list[Number] | None
    expected_return: Metric
    volatility: Metric
    historical_es: Metric
    historical_mdd: Metric


class FrontierPoint(Contract):
    node_id: int
    expected_return: Number | None
    volatility: Number | None
    weights: list[Number] | None
    status: str


class ScaleResult(Contract):
    parameter_evidence: dict[str, Any] = Field(default_factory=dict)
    ordered_asset_ids: list[str]
    frontier: list[FrontierPoint]
    levels: list[Level]
    applied_boundaries: list[Number]
    algorithm_id: AlgorithmId
    algorithm_version: str
    fallback_reason: str | None = None
    stability: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Eligibility(Contract):
    eligible: bool
    blockers: list[Problem] = Field(default_factory=list)


class PreviewResponse(Contract):
    request_echo: PreviewRequest
    resolved_refs: dict[str, FrozenRef]
    data_fingerprints: dict[str, str]
    result: ScaleResult
    mathematical_status: str
    publication_eligibility: Eligibility
    default_eligibility: Eligibility
    warnings: list[Problem]
    limitations: list[str]
    execution_audit: dict[str, Any]
    preview_hash: str


class VersionView(Contract):
    id: str
    name: str
    artifact_type: Literal['risk_scale'] = 'risk_scale'
    content_hash: str
    scheme_id: str
    version_number: int
    created_at: str
    immutable: Literal[True] = True
    preview: PreviewResponse
    current_eligibility: Eligibility
    current_default_eligibility: Eligibility
    retired: bool
    review_due_at: date | None = None
    review_status: Literal['none', 'scheduled', 'upcoming', 'due'] = 'none'


class Summary(Contract):
    id: str
    name: str
    content_hash: str
    created_at: str
    artifact_type: str
    retired: bool = False
    scheme_id: str | None = None
    version_number: int | None = None
    base_currency: str | None = None
    risk_basis_id: str | None = None
    method: str | None = None
    review_due_at: date | None = None
    review_status: Literal['none', 'scheduled', 'upcoming', 'due'] | None = None


class CatalogResponse(Contract):
    items: list[Summary]
    drafts: list[DraftView]
    total: int
    next_offset: int | None


class StudyOption(Contract):
    id: str
    name: str
    content_hash: str
    version_number: int
    base_currency: str
    risk_basis_id: RiskBasisId
    research_as_of: date
    valid_until: date | None = None


class StudyOptionsResponse(Contract):
    as_of: date
    items: list[StudyOption]


class DefaultBinding(Contract):
    key: str
    revision: int
    version_id: str | None = None
    content_hash: str | None = None
    eligibility: Eligibility | None = None


class DefaultsResponse(Contract):
    items: list[DefaultBinding]


class ReferencePreview(Contract):
    preview_hash: str
    definition: dict[str, Any]
    ordered_asset_ids: list[str]
    quality: dict[str, Any]
    warnings: list[Problem]
    moments: dict[str, Any] | None = None
    provenance: dict[str, Any]


class ReferenceVersion(ReferencePreview):
    id: str
    content_hash: str
    created_at: str
    artifact_type: str
    immutable: Literal[True] = True


class Capabilities(Contract):
    trust_mode: Literal['single_local_trusted_workspace'] = 'single_local_trusted_workspace'
    ready: bool
    execution: dict[str, Any]
    algorithms: list[dict[str, Any]]
    limits: dict[str, Any]
    reference: dict[str, Any]
    templates: list[dict[str, Any]]


class CompareResponse(Contract):
    compatible: bool
    left: VersionView
    right: VersionView
    boundary_differences: list[Number] | None
    differences: list[str]


class ClassifyResponse(Contract):
    status: str
    level_code: str | None
    cap_satisfied: bool | None
    authorized_cap: Number
    flags: list[str]


class SourceCatalog(Contract):
    items: list[dict[str, Any]]
    total: int
    offset: int
    limit: int
    problems: list[Problem] = Field(default_factory=list)


class DeleteResponse(Contract):
    deleted: Literal[True]
    id: str
