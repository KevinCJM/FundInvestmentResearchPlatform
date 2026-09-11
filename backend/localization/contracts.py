"""Strict transport contracts. System translations have no write contract."""
from __future__ import annotations

from typing import Annotated, Literal
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, StrictInt
from .errors import I18nError
from .languages import normalize_locale

Locale = Annotated[str, Field(min_length=2, max_length=35), AfterValidator(normalize_locale)]
Scope = Literal["system", "business"]
Key = Annotated[str, Field(min_length=1, max_length=240, pattern=r"^[A-Za-z][A-Za-z0-9_.-]*$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Change(Contract):
    key: Key
    locale: Locale
    value: str | None = Field(default=None, max_length=1600)


class BusinessUpdate(Contract):
    scope: Literal["business"] = "business"
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    changes: list[Change] = Field(min_length=1, max_length=1000)
    reason: str = Field(default="", max_length=200)


class PreferencesUpdate(Contract):
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    default_locale: Locale


class ImportPackage(Contract):
    format_version: Literal[1] = 1
    scope: Literal["business"]
    catalog_version: str | None = Field(default=None, max_length=64)
    entries: list[Change] = Field(max_length=1000)


class ImportValidate(Contract):
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    package: ImportPackage


class ImportApply(ImportValidate):
    confirmation_digest: str = Field(min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")


class RestoreRequest(Contract):
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    target_revision: Annotated[StrictInt, Field(ge=0)]


class LanguageDefinition(Contract):
    id: Locale
    label: str = Field(min_length=1, max_length=60)
    fallback_locale: Locale = "zh-CN"
    enabled: StrictBool = True


class LanguagesUpdate(Contract):
    expected_revision: Annotated[StrictInt, Field(ge=0)]
    locales: list[LanguageDefinition] = Field(min_length=2, max_length=24)
