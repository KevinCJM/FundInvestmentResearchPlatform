"""Stable domain errors exposed by the custom indicator API."""

from __future__ import annotations

from typing import Any, Optional


class IndicatorDomainError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        field: Optional[str] = None,
        diagnostics: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.field = field
        self.diagnostics = diagnostics

    def detail(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field:
            payload["field"] = self.field
        if self.diagnostics:
            payload["diagnostics"] = self.diagnostics
        return payload


class NotFoundError(IndicatorDomainError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, status_code=404)


class ConflictError(IndicatorDomainError):
    def __init__(self, code: str, message: str, field: Optional[str] = None) -> None:
        super().__init__(code, message, status_code=409, field=field)


class ValidationError(IndicatorDomainError):
    def __init__(
        self,
        code: str,
        message: str,
        field: Optional[str] = None,
        diagnostics: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        super().__init__(
            code,
            message,
            status_code=422,
            field=field,
            diagnostics=diagnostics,
        )
