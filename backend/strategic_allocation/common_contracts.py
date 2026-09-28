"""Shared input primitives, independent of any particular allocation model.

Keeping these primitives below mandate/CMA/model contracts prevents circular
imports when explicit model definitions are embedded in a CMA request.
"""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Number = Annotated[float, Field(strict=True)]
Identifier = Annotated[str, Field(min_length=1, max_length=120)]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Currency = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class FrozenRef(Contract):
    """An immutable artifact identity, shared by reference and mandate inputs."""
    id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,119}$")
    content_hash: Fingerprint


VersionStatus = Literal["current", "superseded", "retired", "missing"]


class VersionInfo(Contract):
    """产物版本：系列、版本号、状态与同系列当前版本，见 docs/pre-investment/versioning.md。"""
    lineage_id: str | None
    number: int
    status: VersionStatus
    latest_id: str | None
    latest_number: int | None


class UpstreamRef(Contract):
    kind: Literal["mandate", "strategic_scope", "product_scope", "cma", "saa_policy", "taa_decision"]
    id: str | None
    name: str | None
    number: int | None
    latest_id: str | None
    latest_number: int | None
    status: VersionStatus
    equivalent: bool = False
    usable: Literal["ready", "stale", "blocked"]


class UsabilityReason(Contract):
    code: Literal["self_retired", "self_superseded", "upstream_deleted", "upstream_superseded"]
    kind: str | None = None
    name: str | None = None
    number: int | None = None
    latest_number: int | None = None


class Usability(Contract):
    status: Literal["ready", "stale", "blocked"]
    reasons: list[UsabilityReason]
