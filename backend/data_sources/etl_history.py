"""Select history explicitly, without importing unrelated products into a run."""
from __future__ import annotations

import json
import re

from .models import CenterError, FIELD_PATTERN


def history_batch_ids(db, step, definition, downloads: dict) -> list[str]:
    query = "SELECT id FROM source_run WHERE EXISTS (SELECT 1 FROM json_each(source_run.result,'$.tables') WHERE json_extract(value,'$.table_id')=?)"
    arguments: list = [step.table_id]
    if step.history_scope == "matching_inputs":
        by_id = {item.id: item for item in definition.steps}
        conditions = []
        for mapping_id in step.inputs:
            item = downloads[by_id[mapping_id].inputs[0]]["effective_interface"]
            clause = ["interface_id=?"]
            arguments.append(item["id"])
            # Dates change between runs; product/market/adjustment parameters do
            # not. Compare those scope values against the recorded interface.
            for key, value in sorted(item["params"].items()):
                if key in {item["start_param"], item["end_param"]}:
                    continue
                if not re.fullmatch(FIELD_PATTERN, key):
                    raise CenterError("ETL_HISTORY_SCOPE_INVALID", "同范围历史匹配只支持普通参数名，请调整参数或选择整表历史。")
                clause.append("json_extract(snapshot,?) IS ?")
                arguments.extend(["$.params." + key, json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (list, dict)) else value])
            conditions.append("(" + " AND ".join(clause) + ")")
        query += " AND (" + " OR ".join(conditions) + ")"
    rows = db.execute(query + " ORDER BY created_at,id LIMIT 1001", arguments).fetchall()
    if len(rows) > 1000:
        raise CenterError("ETL_HISTORY_LIMIT", "匹配的历史候选超过 1000 批，请先归档或缩小范围；不会静默截取。")
    return [row[0] for row in rows]
