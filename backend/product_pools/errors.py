"""Domain errors for product-pool workflows."""

from __future__ import annotations


class ProductPoolDomainError(Exception):
    """Stable product-pool error contract shared by downstream workflows."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status_code: int = 400,
        field: str | None = None,
        diagnostics: list[dict] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.field = field
        self.diagnostics = list(diagnostics or [])


class ProductPoolError(ProductPoolDomainError):
    """Compatibility name used by the product-pool HTTP routes."""


class ProductPoolNotFoundError(ProductPoolError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, status_code=404)


class ProductPoolConflictError(ProductPoolError):
    def __init__(self, code: str, message: str, *, field: str | None = None) -> None:
        super().__init__(code, message, status_code=409, field=field)


class ProductPoolValidationError(ProductPoolError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        field: str | None = None,
        diagnostics: list[dict] | None = None,
    ) -> None:
        super().__init__(
            code,
            message,
            status_code=422,
            field=field,
            diagnostics=diagnostics,
        )


__all__ = [
    "ProductPoolConflictError",
    "ProductPoolDomainError",
    "ProductPoolError",
    "ProductPoolNotFoundError",
    "ProductPoolValidationError",
]
