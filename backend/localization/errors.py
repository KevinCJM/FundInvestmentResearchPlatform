"""Stable presentation errors shared by transport and registry validation."""


class I18nError(Exception):
    def __init__(self, code: str, message: str, status: int = 422, field: str | None = None):
        super().__init__(message)
        self.code, self.message, self.status, self.field = code, message, status, field

    def detail(self) -> dict:
        return {"code": self.code, "message_key": f"errors.{self.code}", "message": self.message, "field": self.field}
