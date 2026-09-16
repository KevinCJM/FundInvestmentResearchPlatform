"""Optional study protocol and bounded reliability requests."""

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer


class Reference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str = Field(min_length=1, max_length=120)
    publication_id: str = Field(min_length=1, max_length=120)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class Study(BaseModel):
    model_config = ConfigDict(extra="forbid")
    purpose: Literal["historical_reference", "realtime_recognition"]
    family: Literal[
        "market_trend",
        "macro_growth_inflation",
        "risk",
        "financial_conditions",
        "custom",
    ]
    reference: Reference | None = None
    state_mapping: dict[str, str] | None = Field(default=None, max_length=12)
    calibration_id: str | None = Field(default=None, max_length=120)
    qualification_id: str | None = Field(default=None, max_length=120)

    @model_serializer(mode="wrap")
    def omit_absent_qualification(self, handler):
        result = handler(self)
        if self.qualification_id is None:
            result.pop("qualification_id", None)
        return result

    @model_validator(mode="after")
    def no_recursive_reference(self):
        if self.purpose == "historical_reference" and any(
            value is not None
            for value in (self.reference, self.state_mapping, self.calibration_id, self.qualification_id)
        ):
            raise ValueError("历史参考不能递归绑定参考、映射或校准器")
        if (
            self.state_mapping is not None or self.calibration_id is not None
        ) and self.reference is None:
            raise ValueError("状态映射需要精确参考")
        if self.qualification_id is not None and (self.calibration_id is None or self.reference is None):
            raise ValueError("前瞻资格必须同时绑定精确参考和校准制品")
        return self


class StabilityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    max_variants: int = Field(default=6, ge=1, le=12)
    perturbation: float = Field(default=0.1, gt=0, le=0.5)
    parameters: bool = True
    windows: bool = True
    seeds: bool = True
    truncation: bool = True


class BootstrapPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    enabled: bool = True
    replicates: int = Field(default=200, ge=20, le=500)
    block_length: int = Field(default=10, ge=2, le=250)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    minimum_blocks: int = Field(default=5, ge=2, le=100)
    minimum_cycles: int = Field(default=3, ge=1, le=100)
    minimum_valid_replicates: int = Field(default=100, ge=10, le=500)

    @model_validator(mode="after")
    def valid_replicates(self):
        if self.minimum_valid_replicates > self.replicates:
            raise ValueError("minimum_valid_replicates cannot exceed replicates")
        return self


class QualityPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    stability: StabilityPolicy = Field(default_factory=StabilityPolicy)
    include_price_returns: bool = True
    minimum_state_episodes_for_estimation: int = Field(default=3, ge=1, le=100)


class QualityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition_id: str = Field(min_length=1, max_length=120)
    revision: int = Field(ge=1)
    mode: Literal["retrospective"] = "retrospective"
    as_of: date | None = None
    compile_token: str | None = Field(default=None, max_length=200)
    policy: QualityPolicy


class QualityConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: QualityRequest
    preview_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    stability: StabilityPolicy = Field(default_factory=StabilityPolicy)
    bootstrap: BootstrapPolicy = Field(default_factory=BootstrapPolicy)
    calibration_end: date
    validation_end: date | None = None
    test_end: date | None = None
    minimum_samples: int = Field(default=60, ge=2, le=20000)
    minimum_class_samples: int = Field(default=10, ge=1, le=20000)
    minimum_segments: int = Field(default=3, ge=1, le=1000)
    minimum_state_episodes: int = Field(default=3, ge=1, le=100)
    minimum_state_predictions: int = Field(default=5, ge=1, le=20000)
    minimum_state_precision: float = Field(default=0.65, ge=0, le=1)
    bins: int = Field(default=10, ge=2, le=30)
    transition_tolerance: int = Field(default=3, ge=0, le=60)
    confidence_floor: float = Field(default=0.6, ge=0, le=1)
    calibration_method: Literal["auto", "temperature", "class_frequency"] = "auto"

    @model_validator(mode="after")
    def time_blocks(self):
        if self.validation_end and self.validation_end <= self.calibration_end:
            raise ValueError("validation_end 必须晚于 calibration_end")
        if self.test_end and self.test_end <= (
            self.validation_end or self.calibration_end
        ):
            raise ValueError("test_end 必须晚于前一时间块")
        return self


class PreviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    definition_id: str = Field(min_length=1, max_length=120)
    revision: int = Field(ge=1)
    reference: Reference
    policy: Policy
    compile_token: str | None = Field(default=None, max_length=200)


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: PreviewRequest
    preview_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
