"""ETL-specific port metadata; the shared canvas knows no ETL or source names."""
from __future__ import annotations


def graph_schemas() -> list[dict]:
    specs = (
        ("download", "下载原始数据", "采集", None, "raw_batch", False),
        ("map", "字段映射", "转换", "raw_batch", "mapped_batch", False),
        ("resolve", "多源取值", "质量与取值", "mapped_batch", "resolved_table", True),
        ("snapshot", "指标快照计算", "本地计算", "resolved_table", "snapshot", True),
        ("task", "数据集任务", "批量任务", "workspace", "workspace", False),
    )
    labels = {"raw_batch": "原始批次", "mapped_batch": "标准批次", "resolved_table": "取值结果", "snapshot": "指标快照", "workspace": "数据工作区"}
    result = []
    for kind, label, category, input_type, output_type, multiple in specs:
        inputs = [] if input_type is None else [{"id": "data", "label": labels[input_type], "value_type": input_type, "required": kind != "task", "multiple": multiple}]
        inputs.append({"id": "after", "label": "等待完成", "value_type": "control", "required": False, "multiple": True})
        result.append({"id": kind, "label": label, "category": kind, "category_label": category, "inputs": inputs,
                       "outputs": [{"id": "data", "label": labels[output_type], "value_type": output_type}, {"id": "done", "label": "完成", "value_type": "control"}]})
    return result
