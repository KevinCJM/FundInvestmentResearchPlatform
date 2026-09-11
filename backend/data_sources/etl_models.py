"""User-owned ordered ETL plans; data dependencies are explicit and acyclic."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import Field, model_validator
from .models import ID_PATTERN, FIELD_PATTERN, StrictModel
try:
    from backend.computation_graph import CanvasLayout, dependency_order
except ModuleNotFoundError:
    from computation_graph import CanvasLayout, dependency_order


class EtlParameter(StrictModel):
    id: str = Field(pattern=FIELD_PATTERN)
    label: str = Field(min_length=1, max_length=100)
    data_type: Literal["text", "date"] = "text"
    default: str = Field(default="", max_length=200)
    required: bool = True
    date_format: Literal["iso", "compact"] = "iso"
    description: str = Field(default="", max_length=500)


class EtlRunOptions(StrictModel):
    mode: Literal["full", "incremental", "auto_incremental"] = "incremental"
    parameters: dict[str, str] = Field(default_factory=dict, max_length=30)
    auto_baseline_run_id: str | None = Field(default=None, pattern=r'^[a-f0-9]{32}$')
    auto_baseline_scope: Literal['missing_only', 'acquisition'] = 'missing_only'
    event_update_purpose: Literal['update', 'recheck'] = 'update'
    event_revision_interval_days: int = Field(default=7, ge=1, le=90)
    event_revision_window_days: int = Field(default=90, ge=1, le=366)

    @model_validator(mode='after')
    def automatic_baseline_only(self):
        if self.auto_baseline_run_id and self.mode != 'auto_incremental':
            raise ValueError('补充基线只适用于自动增量')
        return self


class EtlStep(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    name: str = Field(min_length=1, max_length=100)
    kind: Literal["download", "map", "resolve", "snapshot", "task"]
    task_id: str | None = Field(default=None, pattern=ID_PATTERN)
    inputs: list[str] = Field(default_factory=list, max_length=40)
    after: list[str] = Field(default_factory=list, max_length=40)
    source_id: str | None = None
    interface_id: str | None = None
    interface_revision: int | None = Field(default=None, ge=1)
    mode: Literal["inherit", "full", "incremental"] = "inherit"
    params: dict[str, Any] = Field(default_factory=dict)
    parameter_bindings: dict[str, str] = Field(default_factory=dict, max_length=30)
    target_tables: list[str] = Field(default_factory=list, max_length=30)
    table_id: str | None = None
    include_history: bool = True
    history_scope: Literal["table", "matching_inputs"] = "table"
    allow_empty: bool = False
    start_date: str | None = None
    end_date: str | None = None
    as_of: str | None = None


class EtlDefinition(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)
    max_runtime_seconds: int = Field(default=3600, ge=10, le=86400)
    steps: list[EtlStep] = Field(min_length=1, max_length=40)
    parameters: list[EtlParameter] = Field(default_factory=list, max_length=30)
    graph_version: Literal[1] | None = None
    canvas: CanvasLayout | None = None

    def execution_steps(self) -> list[EtlStep]:
        if self.graph_version is None:
            return list(self.steps)
        by_id = {step.id: step for step in self.steps}
        topology = dependency_order(by_id, ((source, step.id) for step in self.steps for source in dict.fromkeys([*step.inputs, *step.after])), tie_break="node_order")
        if topology.missing or topology.cyclic or len(by_id) != len(self.steps):
            raise ValueError("计算图存在重复节点、未知引用或循环依赖。")
        return [by_id[key] for key in topology.order]

    def compiled(self) -> "EtlDefinition":
        # Presentation never becomes an execution input. Keep it in the saved
        # template, not in the bound worker definition or configuration hashes.
        return self.model_copy(update={"steps": self.execution_steps(), "canvas": None})

    @model_validator(mode="after")
    def dependencies(self):
        import re
        parameter_ids = [parameter.id for parameter in self.parameters]
        if len(parameter_ids) != len(set(parameter_ids)):
            raise ValueError("运行参数不可重名。")
        node_ids = [step.id for step in self.steps]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("步骤 ID 或依赖不可重复。")
        if self.canvas and set(self.canvas.positions) - set(node_ids):
            raise ValueError("画布位置引用了不存在的步骤。")
        if self.graph_version is None and any(step.after for step in self.steps):
            raise ValueError("控制依赖需要使用图流程版本。")
        prior = {}
        for step in self.execution_steps():
            if step.id in prior or len(step.inputs) != len(set(step.inputs)) or len(step.after) != len(set(step.after)):
                raise ValueError("步骤 ID 或依赖不可重复。")
            if any(key not in prior for key in [*step.inputs, *step.after]):
                raise ValueError(f"{step.name} 引用了未完成的前置步骤；请调整顺序或依赖。")
            expected = {"map": "download", "resolve": "map", "snapshot": "resolve"}.get(step.kind)
            if step.kind == "task":
                if not step.task_id or any(prior[key].kind != 'task' for key in step.inputs):
                    raise ValueError('数据集任务须选择已登记任务，只能引用前置数据集工作区。')
                if self.graph_version is None and len(step.inputs) > 1:
                    raise ValueError('多个工作区输入需要使用图流程版本。')
            elif step.kind == "download":
                if step.inputs or not step.interface_id or not step.source_id or not step.interface_revision:
                    raise ValueError("下载步骤须选择已保存的数据源、接口和修订，不能引用数据输入。")
            elif not step.inputs or any(prior[key].kind != expected for key in step.inputs):
                raise ValueError(f"{step.name} 必须引用前置 {expected} 步骤。")
            if step.kind == "map" and len(step.inputs) != 1:
                raise ValueError("字段映射步骤必须对应一个下载步骤。")
            if step.kind == "resolve" and not step.table_id:
                raise ValueError("多源取值步骤须选择标准业务表。")
            if step.kind == "snapshot":
                tables = [prior[key].table_id for key in step.inputs]
                if len(tables) != len(set(tables)):
                    raise ValueError("快照每张标准表只能引用一个取值结果，避免混合版本。")
                if not {"master.instrument", "market.nav_daily"}.issubset(tables):
                    raise ValueError("指标快照需要已取值的产品信息和基金净值；行情不能冒充净值。")
                if set(tables) - {"master.instrument", "market.nav_daily", "market.quote_daily", "master.trading_calendar"}:
                    raise ValueError("指标快照只读取产品、净值、日行情及交易日历。")
            if step.kind not in {"download", "task"} and (step.params or step.parameter_bindings):
                raise ValueError("只有下载步骤接受请求参数；其他步骤通过显式输入取得数据。")
            if set(step.parameter_bindings.values()) - set(parameter_ids):
                raise ValueError("步骤引用了未定义的运行参数。")
            if any(not re.fullmatch(FIELD_PATTERN, key) for key in step.parameter_bindings):
                raise ValueError("运行参数绑定包含非法接口字段。")
            if step.as_of:
                from datetime import datetime
                timestamp = datetime.fromisoformat(step.as_of.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("历史可得截止时点必须携带时区。")
            if step.start_date or step.end_date:
                from datetime import date
                first = date.fromisoformat(step.start_date) if step.start_date else None
                last = date.fromisoformat(step.end_date) if step.end_date else None
                if first and last and first > last:
                    raise ValueError("开始日期不能晚于结束日期。")
            prior[step.id] = step
        return self
