"""Shared input primitives, independent of any particular allocation model.

Keeping these primitives below mandate/CMA/model contracts prevents circular
imports when explicit model definitions are embedded in a CMA request.
"""
from typing import Annotated

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
