"""Versioned, bounded authoring and research contracts."""
from __future__ import annotations

from datetime import date
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, StrictStr, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Step(Contract):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,47}$")
    label: str = Field(min_length=1, max_length=80)
    op: str = Field(min_length=1, max_length=100)
    inputs: dict[str, str] = Field(default_factory=dict, max_length=4)
    parameters: dict[str, StrictFloat | StrictInt | StrictStr] = Field(default_factory=dict, max_length=12)


class Execution(Contract):
    take_profit: float = Field(default=0.15, gt=0, le=5)
    stop_loss: float = Field(default=0.15, gt=0, lt=1)
    max_holding_bars: int = Field(default=15, ge=1, le=2500)
    cooldown_bars: int = Field(default=0, ge=0, le=2500)
    fee_bps: float = Field(default=3, ge=0, le=1000)
    slippage_bps: float = Field(default=5, ge=0, le=1000)


class Adaptation(Contract):
    source_experiments: list[str] = Field(min_length=1, max_length=16)
    version: Literal["etf-v1"] = "etf-v1"
    preserved: list[str] = Field(default_factory=list, max_length=16)
    changed: list[str] = Field(default_factory=list, max_length=16)


class TrainingAction(Contract):
    id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,47}$")
    label: str = Field(min_length=1, max_length=100)
    entry: str | None = Field(default=None, max_length=100)


class ParameterPatch(Contract):
    node: str
    parameter: str
    value: StrictFloat | StrictInt | StrictStr


class SearchAxis(Contract):
    label: str = Field(min_length=1, max_length=100)
    choices: list[list[ParameterPatch]] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def bounded_choices(self):
        if any(not choice or len(choice) > 4 for choice in self.choices):
            raise ValueError("每个参数方案需要 1～4 个参数绑定。")
        return self


class Training(Contract):
    mode: Literal["global", "state", "month", "quarter"] = "global"
    state_refs: list[str] = Field(default_factory=list, max_length=3)
    actions: list[TrainingAction] = Field(min_length=1, max_length=8)
    search_space: list[SearchAxis] = Field(default_factory=list, max_length=6)
    min_trades: int = Field(default=5, ge=2, le=10000)
    confidence: float = Field(default=1.0, ge=0, le=5)
    risk_penalty: float = Field(default=0.1, ge=0, le=5)
    min_utility: float = Field(default=0, ge=0, le=5)
    embargo_bars: int = Field(default=0, ge=0, le=250)

    @model_validator(mode="after")
    def research_budget(self):
        if len({action.id for action in self.actions}) != len(self.actions):
            raise ValueError("训练候选 ID 不能重复。")
        if self.mode == "state" and not self.state_refs:
            raise ValueError("状态选择需要连接至少一个条件输出。")
        if self.mode != "state" and self.state_refs:
            raise ValueError("只有状态选择模式使用状态条件。")
        count = sum(action.entry is not None for action in self.actions)
        if not count:
            raise ValueError("训练至少需要一个非空仓候选。")
        for axis in self.search_space:
            count *= len(axis.choices)
        if count > 108:
            raise ValueError("非空仓动作乘以参数组合最多为 108 个训练候选。")
        return self


class Definition(Contract):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=2000)
    schema_version: Literal["1.0"] = "1.0"
    nodes: list[Step] = Field(min_length=1, max_length=128)
    entry: str = Field(min_length=1, max_length=100)
    exit: str | None = Field(default=None, max_length=100)
    execution: Execution = Field(default_factory=Execution)
    adaptation: Adaptation | None = None
    training: Training | None = None


class DefinitionUpdate(Definition):
    revision: int = Field(ge=1)


class PrepareRequest(Contract):
    definition: Definition


class Target(Contract):
    kind: Literal["etf"] = "etf"
    product_id: str = Field(pattern=r"^[0-9]{6}\.(SH|SZ)$")


class RunRequest(PrepareRequest):
    compile_token: str = Field(min_length=16, max_length=128)
    targets: list[Target] = Field(min_length=1, max_length=12)
    start_date: date
    end_date: date
    holdout_start: date
    walk_forward_splits: int = Field(default=3, ge=2, le=12)
    price_basis: Literal["hfq", "qfq", "raw"] = "hfq"
    context_baskets: dict[Literal["market", "category"], list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ordered_dates(self):
        if not self.start_date < self.holdout_start <= self.end_date:
            raise ValueError("样本外开始日必须晚于研究开始日，且不晚于结束日。")
        keys = [item.product_id for item in self.targets]
        if len(keys) != len(set(keys)):
            raise ValueError("同一研究不能重复添加相同产品。")
        import re
        for codes in self.context_baskets.values():
            if len(codes) > 12 or len(codes) != len(set(codes)):
                raise ValueError("每个环境篮子最多 12 只 ETF，代码不能重复。")
            if any(not re.fullmatch(r"[0-9]{6}\.(SH|SZ)", code) for code in codes):
                raise ValueError("环境篮子需要完整 ETF 代码，例如 510300.SH。")
        return self


class CompareRequest(Contract):
    run_ids: list[str] = Field(min_length=2, max_length=8)


class ReleaseRequest(Contract):
    run_id: str
    note: str = Field(default="", max_length=500)


class BindingRequest(Contract):
    release_id: str
    context: Literal["product_research", "pre_investment"]
    note: str = Field(default="", max_length=500)
