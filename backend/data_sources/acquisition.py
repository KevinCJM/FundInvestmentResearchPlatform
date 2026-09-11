"""Bounded acquisition shared by interface downloads and ETL steps (I/O only)."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime, timedelta
from typing import Callable

from .models import CenterError, InterfaceConfig, SourceConfig
from .runtime import effective_policy, fetch_with_retry
from .store import SourceStore


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def checkpoint_key(source: SourceConfig, interface: InterfaceConfig) -> str:
    config = interface.model_dump(mode="json")
    config["params"] = {k: v for k, v in interface.params.items() if k not in {interface.start_param, interface.end_param}}
    return fingerprint({"source": source.model_dump(mode="json"), "interface": config})


def incremental_params(store: SourceStore, source: SourceConfig, interface: InterfaceConfig, mode: str) -> tuple[dict, str]:
    params, key = dict(interface.params), checkpoint_key(source, interface)
    with store.connection() as db:
        db.execute("CREATE TABLE IF NOT EXISTS source_sync_checkpoint (id TEXT PRIMARY KEY, latest_date TEXT NOT NULL)")
        row = db.execute("SELECT latest_date FROM source_sync_checkpoint WHERE id=?", (key,)).fetchone()
    if mode == "incremental" and row and interface.incremental_field:
        overlap = (date.fromisoformat(row[0]) - timedelta(days=3)).strftime("%Y%m%d")
        current = str(params.get(interface.start_param) or "").replace("-", "")
        params[interface.start_param] = max(current, overlap)
    start, end = (str(params.get(k) or "").replace("-", "") for k in (interface.start_param, interface.end_param))
    if start and end and start > end:
        raise CenterError("DATE_RANGE_INVALID", "增量断点晚于结束日期；补历史请使用全量重新请求所选范围。")
    return params, key


def advance_checkpoint(store: SourceStore, key: str, interface: InterfaceConfig, rows: list[dict]) -> None:
    if not rows or not interface.incremental_field:
        return
    dates = []
    for row in rows:
        value = str(row.get(interface.incremental_field) or "")
        try:
            dates.append(date.fromisoformat(value[:10]).isoformat() if "-" in value else datetime.strptime(value[:8], "%Y%m%d").date().isoformat())
        except ValueError:
            continue
    if dates:
        with store.connection() as db:
            db.execute("CREATE TABLE IF NOT EXISTS source_sync_checkpoint (id TEXT PRIMARY KEY, latest_date TEXT NOT NULL)")
            db.execute("INSERT INTO source_sync_checkpoint VALUES (?,?) ON CONFLICT(id) DO UPDATE SET latest_date=MAX(latest_date,excluded.latest_date)", (key, max(dates)))


def download_rows(store: SourceStore, source: SourceConfig, interface: InterfaceConfig, params: dict,
                  *, progress: Callable[[int, int], None] = lambda pages, rows: None,
                  check: Callable[[], None] = lambda: None, fetch=None) -> tuple[list[dict], int]:
    """Return only a complete bounded response; never commit truncated pages."""
    fetch = fetch or fetch_with_retry
    policy = effective_policy(source.policy, interface.policy)
    pagination = interface.pagination
    started, rows, seen_pages = time.monotonic(), [], set()
    for page in range(pagination.max_pages if pagination.mode != "none" else 1):
        check()
        remaining = int(policy.max_runtime_seconds - (time.monotonic() - started))
        if remaining < 1:
            raise CenterError("SOURCE_RUNTIME_LIMIT", "下载达到最长运行时间，未提交不完整结果。")
        request_params = dict(params)
        if pagination.mode != "none":
            request_params[pagination.cursor_param] = page * pagination.page_size if pagination.mode == "offset" else page + 1
            request_params[pagination.limit_param] = pagination.page_size
        bounded = source.model_copy(update={"policy": source.policy.model_copy(update={"max_runtime_seconds": remaining})})
        _, received = fetch(store, bounded, interface, request_params)
        check()
        if time.monotonic() - started >= policy.max_runtime_seconds:
            raise CenterError("SOURCE_RUNTIME_LIMIT", "下载达到最长运行时间，未提交不完整结果。")
        if pagination.mode != "none" and len(received) > pagination.page_size:
            raise CenterError("SOURCE_PAGINATION_INVALID", "返回行数超过请求页大小，请检查分页配置。")
        signature = fingerprint(received)
        if received and signature in seen_pages:
            raise CenterError("SOURCE_REPEATED_PAGE", "接口重复返回同一页，请检查分页游标，不能视为完整下载。")
        seen_pages.add(signature)
        rows.extend({**params, **row} for row in received)
        progress(page + 1, len(rows))
        if len(rows) > 100000:
            raise CenterError("SOURCE_SYNC_LIMIT", "单次下载超过 100000 行，请按产品或日期拆分步骤。")
        if pagination.mode == "none":
            if len(received) >= policy.max_rows_per_request:
                raise CenterError("SOURCE_TRUNCATION", "结果触及行数上限，请配置分页或缩小范围。")
            return rows, page + 1
        if len(received) < pagination.page_size:
            return rows, page + 1
    raise CenterError("SOURCE_PAGINATION_INCOMPLETE", "分页未完成，已达到最大页数；没有提交不完整结果。")
