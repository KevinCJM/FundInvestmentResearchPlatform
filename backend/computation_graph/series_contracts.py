"""Common numeric / categorical series output contracts for authoring clients."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .contracts import GraphPortRef


class ChannelPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str = Field(default="", max_length=80)
    unit: str = Field(default="", max_length=20)
    display_format: Literal["number", "percent"] = "number"
    precision: int = Field(default=4, ge=0, le=8)


class EnumItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80)
    code: int = Field(ge=0)
    label: str
    color: str | None = None


class SeriesOutputContract(ChannelPresentation):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str
    shape: Literal["series", "matrix"] = "series"
    value_type: Literal["number", "boolean", "enum", "index"] = "number"
    source: GraphPortRef
    axis: Literal["time"] = "time"
    enum_items: list[EnumItem] = Field(default_factory=list)
    missing: Literal["null", "unclassified"] = "null"
    interpolation: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_enum(self):
        if self.value_type == "enum":
            if not self.enum_items or self.missing != "unclassified":
                raise ValueError("枚举时序必须声明值域，并保留未识别状态。")
            if len({x.code for x in self.enum_items}) != len(self.enum_items) or len({x.id for x in self.enum_items}) != len(self.enum_items):
                raise ValueError("枚举编号和 ID 必须唯一。")
        elif self.enum_items:
            raise ValueError("只有枚举输出可以声明枚举项。")
        return self


def regime_series_outputs(definition):
    labels = {"state": "市场状态", "probabilities": "状态概率", "confidence": "置信度",
              "recognition_index": "识别时点", "effective_index": "生效时点", "reason_code": "原因码"}
    domain = [EnumItem(id=item.id, code=index, label=item.label, color=item.color)
              for index, item in enumerate(definition.states)]
    return [SeriesOutputContract(id=name, **(definition.graph.channel_metadata[name].model_dump() if name in definition.graph.channel_metadata else {"label": labels.get(name, name)}), source=ref.model_dump(),
                shape="matrix" if name == "probabilities" else "series",
                value_type="enum" if name == "state" else "index" if name.endswith("index") else "number",
                enum_items=domain if name == "state" else [],
                missing="unclassified" if name == "state" else "null").model_dump(mode="json")
            for name, ref in definition.graph.outputs.items()]
