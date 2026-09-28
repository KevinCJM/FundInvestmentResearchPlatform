"""Read-only historical opportunity-set checks before an LTCMA exists."""
from datetime import date

from pydantic import Field, model_validator

from .common_contracts import Contract, Identifier
from .cma_model_contracts import CmaWindow
from .universe_contracts import UniverseRequest


class ScopeFeasibilityRequest(Contract):
    mandate_id: Identifier
    as_of: date
    window: CmaWindow = Field(default_factory=CmaWindow)
    strategic_definition: UniverseRequest | None = None
    product_version_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=100)
    universe_snapshot_id: Identifier | None = None
    product_excluded_keys: list[str] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def source(self):
        if sum(x is not None for x in (self.strategic_definition, self.product_version_ids,
                                      self.universe_snapshot_id)) != 1:
            raise ValueError("请选择一种研究范围来源。")
        if self.as_of > date.today():
            raise ValueError("研究日不能在未来。")
        if self.strategic_definition is not None and self.strategic_definition.as_of != self.as_of:
            raise ValueError("研究范围和初筛须使用同一研究日。")
        if self.product_excluded_keys and self.product_version_ids is None:
            raise ValueError("产品排除项仅适用于正在编辑的产品池范围。")
        if self.window.end_date is not None and self.window.end_date > self.as_of:
            raise ValueError("历史窗口不能晚于研究日。")
        return self
