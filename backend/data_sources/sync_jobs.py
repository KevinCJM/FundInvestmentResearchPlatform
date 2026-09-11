"""Bounded background interface sync shared by HTTP, Tushare and AKShare."""
from __future__ import annotations
import hashlib
import json
import os
import threading
import time
import uuid
from datetime import datetime, timedelta

from .batches import capture_batch
from .models import CenterError, InterfaceConfig, SourceConfig
from .runtime import effective_policy, fetch_with_retry
from .store import SourceStore, utc_now
from .resolution_store import get_policy, resolve_saved

try:
    from backend.services.refresh_runtime import InterProcessFileLock
except ModuleNotFoundError:
    from services.refresh_runtime import InterProcessFileLock


def _init(store):
    with store.connection() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS source_sync_job (id TEXT PRIMARY KEY, result TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS source_sync_checkpoint (id TEXT PRIMARY KEY, latest_date TEXT NOT NULL);
        """)


def _save(store, job):
    job["updated_at"] = utc_now()
    with store.connection() as db:
        db.execute("INSERT OR REPLACE INTO source_sync_job VALUES (?,?,?)", (job["job_id"], json.dumps(job, ensure_ascii=False, default=str), job["updated_at"]))


def list_jobs(store):
    _init(store)
    with store.connection() as db:
        values = db.execute("SELECT result FROM source_sync_job ORDER BY updated_at DESC LIMIT 10").fetchall()
    jobs = [json.loads(v[0]) for v in values]
    for job in jobs:
        if job["status"] == "RUNNING":
            try:
                os.kill(job["owner_pid"], 0)
            except ProcessLookupError:
                job.update(status="FAILED", error="服务进程已退出；可按原配置重新执行，已完成候选保留。", code="PROCESS_INTERRUPTED")
                _save(store, job)
    return jobs


def start_sync(store: SourceStore, identifier: str, expected_revision: int, params: dict, mode: str) -> dict:
    from .service import enabled, parse_config
    from .mapping import validate_mapping
    from .credentials import configured
    if not isinstance(params, dict):
        raise CenterError("INVALID_PARAMS", "下载参数必须为对象。", 422)
    if not enabled():
        raise CenterError("SOURCE_CENTER_READ_ONLY", "当前环境不允许下载。", 403)
    if mode not in {"full", "incremental"}:
        raise CenterError("MODE_INVALID", "更新方式须为全量或增量。")
    _init(store)
    lock = InterProcessFileLock(store.root / ".tushare_refresh.lock")
    if not lock.acquire(owner="interface-sync"):
        raise CenterError("DATA_TASK_RUNNING", "已有数据任务运行，请等待完成。", 409)
    try:
        saved = store.get("interface", identifier)
        if saved["revision"] != expected_revision:
            raise CenterError("REVISION_CONFLICT", "接口已修改，请重新加载后下载。", 409)
        original = InterfaceConfig.model_validate(saved["config"])
        interface = parse_config("interface", {**saved["config"], "params": {**original.params, **params}})
        source = SourceConfig.model_validate(store.get("source", interface.source_id)["config"])
        if not source.enabled or not interface.enabled:
            raise CenterError("SOURCE_DISABLED", "请先启用数据源和接口。")
        validation = validate_mapping(interface)
        if not validation["valid"] or not validation["ready"]:
            raise CenterError("MAPPING_NOT_READY", "请先补齐字段映射与身份关联，再下载此接口。")
        if (source.transport == "tushare" or source.auth_mode != "none") and not configured(store, source.id):
            raise CenterError("CREDENTIAL_REQUIRED", "请先保存数据源凭据。")
        if source.transport == "akshare":
            from .akshare_adapter import validate_sdk_config, validate_sdk_params
            validate_sdk_config(interface)
            validate_sdk_params(interface.api_name, interface.params)
        policy = effective_policy(source.policy, interface.policy)
        if interface.pagination.mode != "none" and interface.pagination.page_size > policy.max_rows_per_request:
            raise CenterError("PAGE_SIZE_EXCEEDS_SOURCE", "分页大小超过有效行数限制。")
        configuration_hash = hashlib.sha256(json.dumps({"source": source.model_dump(mode="json"), "interface": saved}, sort_keys=True).encode()).hexdigest()
        scope_params = {k: v for k, v in interface.params.items() if k not in {interface.start_param, interface.end_param}}
        checkpoint = hashlib.sha256(json.dumps([identifier, configuration_hash, scope_params], sort_keys=True).encode()).hexdigest()
        if mode == "incremental" and interface.incremental_field:
            with store.connection() as db:
                latest = db.execute("SELECT latest_date FROM source_sync_checkpoint WHERE id=?", (checkpoint,)).fetchone()
            if latest:
                overlap = (datetime.fromisoformat(latest[0]) - timedelta(days=3)).strftime("%Y%m%d")
                current = str(interface.params.get(interface.start_param) or "").replace('-', '')
                interface.params[interface.start_param] = max(current, overlap)
        start = str(interface.params.get(interface.start_param) or '').replace('-', '')
        end = str(interface.params.get(interface.end_param) or '').replace('-', '')
        if start and end and start > end:
            raise CenterError("DATE_RANGE_INVALID", "增量断点晚于所选结束日；如需补历史，请选择重新下载所选范围。")
        job = {"job_id": uuid.uuid4().hex, "interface_id": identifier, "source_id": source.id, "status": "RUNNING",
               "owner_pid": os.getpid(), "mode": mode, "pages": 0, "rows": 0, "created_at": utc_now(),
               "interface_revision": expected_revision, "configuration_hash": configuration_hash, "published": False}
        _save(store, job)
        thread = threading.Thread(target=_run, args=(store, source, interface, policy, checkpoint, job, lock), daemon=True, name="source-sync-" + job["job_id"][:8])
        try:
            thread.start()
        except Exception:
            job.update(status="FAILED", code="THREAD_START_FAILED", error="无法启动后台下载线程。")
            _save(store, job)
            raise
        return dict(job)
    except Exception:
        lock.release()
        raise


def _run(store, source, interface, policy, checkpoint, job, lock):
    started, rows, completed = time.monotonic(), [], False
    params = dict(interface.params)
    pagination = interface.pagination
    try:
        from .acquisition import download_rows
        def progress(pages, count):
            job.update(pages=pages, rows=count)
            _save(store, job)
        rows, _pages = download_rows(store, source, interface, params, progress=progress, fetch=fetch_with_retry)
        batch = capture_batch(store, interface, rows, params, job["configuration_hash"])
        job["batch"] = batch
        if batch["status"] == "REJECTED":
            raise CenterError("MAPPING_REJECTED", "数据已获取，但映射校验未通过；请修正字段映射。")
        if rows and interface.incremental_field:
            import pandas as pd
            dates = pd.to_datetime([str(row.get(interface.incremental_field) or "") for row in rows], errors="coerce")
            if dates.notna().any():
                with store.connection() as db:
                    db.execute("INSERT INTO source_sync_checkpoint VALUES (?,?) ON CONFLICT(id) DO UPDATE SET latest_date=MAX(latest_date,excluded.latest_date)", (checkpoint, dates.max().date().isoformat()))
        saved = get_policy(store)
        job["resolutions"] = []
        # Tables without overrides still inherit the global source order.
        for table in sorted({r["table_id"] for r in batch["tables"]}):
            # A large existing candidate archive must not turn a successful
            # download into a false network failure. Surface arbitration limits.
            try:
                job["resolutions"].append(resolve_saved(store, table, saved["revision"]))
            except CenterError as exc:
                job["resolutions"].append({"table_id": table, "status": "NEEDS_REVIEW", "code": exc.code, "message": exc.message, "published": False})
        job["status"] = "SUCCEEDED" if rows else "EMPTY"
        job["message"] = "下载完成；标准化与多源取值结果为独立候选，尚未发布到正式研究数据。" if rows else "来源返回空数据，请核对代码、日期及上游状态。"
    except CenterError as exc:
        job.update(status="FAILED", code=exc.code, error=exc.message)
    except Exception:
        job.update(status="FAILED", code="SYNC_FAILED", error="任务未完成，请检查数据源、映射或服务日志。")
    finally:
        job["finished_at"] = utc_now()
        try:
            _save(store, job)
        finally:
            lock.release()
