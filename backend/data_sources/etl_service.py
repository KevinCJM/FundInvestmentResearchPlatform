"""Durable sequential ETL, with explicit data lineage and resumable step boundaries."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from pydantic import ValidationError

from .acquisition import advance_checkpoint, download_rows, fingerprint, incremental_params
from .batches import capture_batch
from .credentials import configured
from .etl_models import EtlDefinition, EtlRunOptions
from .etl_parameters import bind_parameters, download_mode, parse_run_options
from .etl_history import history_batch_ids
from .etl_store import EtlStore, public_run
from .etl_collection import record_resume, resume_warnings
from .mapping import validate_mapping
from .models import CenterError, InterfaceConfig, SourceConfig
from .resolution_store import get_policy, resolve_saved
from .runtime import effective_policy
from .service import enabled, parse_config
from .store import SourceStore, utc_now
try:
    from backend.services.refresh_runtime import InterProcessFileLock, is_file_lock_held
except ModuleNotFoundError:
    from services.refresh_runtime import InterProcessFileLock, is_file_lock_held

OWNER = uuid.uuid4().hex
_ACTIVE: dict[str, threading.Thread] = {}


def parse_definition(payload) -> EtlDefinition:
    try:
        return EtlDefinition.model_validate(payload)
    except ValidationError as exc:
        messages = [str(e["msg"]) for e in exc.errors(include_input=False, include_context=False, include_url=False)]
        raise CenterError("ETL_PLAN_INVALID", "；".join(messages[:8]), 422) from None


def inspect_plan(store: SourceStore, definition: EtlDefinition, *, credentials: bool = False) -> dict:
    frozen, outputs, plan, tasks, origins = {}, {}, [], {}, {}
    for step in definition.execution_steps():
        if step.kind == 'task':
            from .task_catalog import inspect_task
            available = outputs[step.inputs[0]] if step.inputs else set()
            tasks[step.id] = inspect_task(store, step, available, require_values=credentials)
            origin = origins.get(step.inputs[0]) if step.inputs else None
            if origin and step.source_id and origin != step.source_id:
                raise CenterError('ETL_TASK_SOURCE_MIX', '数据集工作区不能混入不同来源；跨来源请使用标准映射与多源取值节点。')
            origins[step.id] = step.source_id or origin
            outputs[step.id] = available | set(tasks[step.id]['spec']['provides'])
        elif step.kind == "download":
            saved = store.get("interface", step.interface_id)
            if saved["revision"] != step.interface_revision:
                raise CenterError("REVISION_CONFLICT", f"{step.name} 的接口修订已变化，请重新选择接口。", 409)
            if saved["config"]["source_id"] != step.source_id:
                raise CenterError("ETL_SOURCE_MISMATCH", "所选接口不属于所选数据源。")
            parent = store.get("source", step.source_id)
            source = SourceConfig.model_validate(parent["config"])
            interface = parse_config("interface", {**saved["config"], "params": {**saved["config"].get("params", {}), **step.params}})
            if not source.enabled or not interface.enabled:
                raise CenterError("SOURCE_DISABLED", f"{step.name} 的数据源或接口已停用。")
            available = {m.target_table for m in interface.mappings if m.enabled}
            if set(step.target_tables) - available:
                raise CenterError("ETL_TARGET_INVALID", "下载目标必须属于该接口已启用的标准表映射。")
            if step.target_tables:
                interface.mappings = [m for m in interface.mappings if m.enabled and m.target_table in step.target_tables]
            mapping = validate_mapping(interface)
            if not mapping["valid"] or not mapping["ready"]:
                raise CenterError("MAPPING_NOT_READY", f"{step.name} 的字段映射或身份关联尚未就绪。")
            policy = effective_policy(source.policy, interface.policy)
            if interface.pagination.mode != "none" and interface.pagination.page_size > policy.max_rows_per_request:
                raise CenterError("PAGE_SIZE_EXCEEDS_SOURCE", "分页大小超过来源的有效单次行数限制。")
            if source.transport == "akshare":
                from .akshare_adapter import validate_sdk_config, validate_sdk_params
                validate_sdk_config(interface)
                validate_sdk_params(interface.api_name, interface.params, require_symbol=credentials or not step.parameter_bindings)
            if credentials and (source.transport == "tushare" or source.auth_mode != "none") and not configured(store, source.id):
                raise CenterError("CREDENTIAL_REQUIRED", f"请先保存 {source.name} 的访问凭据。")
            frozen[step.id] = {"source": parent, "interface": saved, "effective_interface": interface.model_dump(mode="json")}
            outputs[step.id] = {m.target_table for m in interface.mappings if m.enabled}
        elif step.kind == "map":
            outputs[step.id] = outputs[step.inputs[0]]
        elif step.kind == "resolve":
            if any(step.table_id not in outputs[key] for key in step.inputs):
                raise CenterError("ETL_DEPENDENCY_TABLE", f"{step.name} 的输入步骤没有提供所选业务表。")
            outputs[step.id] = {step.table_id}
        else:
            outputs[step.id] = {"instrument_metrics_snapshot"}
        plan.append({"id": step.id, "name": step.name, "kind": step.kind, "inputs": step.inputs, "after": step.after, "tables": sorted(outputs[step.id])})
    return {"downloads": frozen, "tasks": tasks, "plan": plan}


def validate(store: SourceStore, payload, options=None) -> dict:
    try:
        definition = parse_definition(payload)
        automatic = None
        if options is not None:
            parsed_options = parse_run_options(options)
            definition = bind_parameters(definition, parsed_options)
            if parsed_options.mode == 'auto_incremental':
                from .auto_incremental import plan
                automatic = plan(store, definition)
                definition = automatic['definition']
        inspection = inspect_plan(store, definition)
        if automatic:
            public = automatic['public']
            return {'valid': public['ready'], 'errors': public['errors'], 'steps': inspection['plan'], 'auto_plan': public, 'published': False}
        return {"valid": True, "errors": [], "steps": inspection["plan"], "published": False}
    except CenterError as exc:
        return {"valid": False, "errors": [{"code": exc.code, "message": exc.message}], "steps": []}


def save_workflow(store: SourceStore, identifier: str, payload, expected: int) -> dict:
    _writable()
    definition = parse_definition(payload)
    inspect_plan(store, definition)
    return EtlStore(store).save_workflow(identifier, definition, expected)


def _writable():
    if not enabled():
        raise CenterError("SOURCE_CENTER_READ_ONLY", "当前环境未开放 ETL 修改或执行。", 403)


def _lock(store: SourceStore):
    lock = InterProcessFileLock(store.root / ".tushare_refresh.lock")
    if not lock.acquire(owner="etl-workflow"):
        raise CenterError("DATA_TASK_RUNNING", "已有下载、快照或 ETL 任务运行，请等待结束。", 409)
    return lock


def execution_fingerprint() -> str:
    """Pin transforms and numerical definitions without depending on git state."""
    from .resolution_store import _checksum
    backend = Path(__file__).resolve().parents[1]
    paths = [backend / "services" / "instrument_analytics.py", backend.parent / 'T01_get_data.py',
             backend / 'data_storage.py']
    for package in ("computation_graph", "data_sources", "data_model", "custom_indicators", "cal_indicators"):
        paths.extend(sorted((backend / package).rglob("*.py")))
    return fingerprint({str(path.relative_to(backend.parent)): _checksum(path) for path in paths})


def _freeze(store: SourceStore, definition: EtlDefinition, options: EtlRunOptions | None = None) -> dict:
    frozen = inspect_plan(store, definition, credentials=True)
    options = options or EtlRunOptions()
    for step in definition.steps:
        if step.kind != "download":
            continue
        item = frozen["downloads"][step.id]
        source = SourceConfig.model_validate(item["source"]["config"])
        interface = InterfaceConfig.model_validate(item["effective_interface"])
        mode = download_mode(step.mode, options)
        params, checkpoint = incremental_params(store, source, interface, mode)
        item.update(mode=mode, request_params=params, checkpoint=checkpoint)
    frozen["execution_fingerprint"] = execution_fingerprint()
    saved = get_policy(store)
    frozen["policy"] = {"config": saved["config"], "revision": saved["revision"]}
    frozen["history"] = {}
    from .task_runtime import freeze_tasks
    freeze_tasks(store, definition, frozen, options)
    with store.connection() as db:
        for step in definition.steps:
            if step.kind == "resolve" and step.include_history:
                frozen["history"][step.id] = history_batch_ids(db, step, definition, frozen["downloads"])
    return frozen


def _launch(journal: EtlStore, run: dict, lock) -> dict:
    from .etl_executor import launch
    return launch(journal, run, lock)


def _launch_inline(journal: EtlStore, run: dict, lock) -> dict:
    """Injectable offline test harness; production always uses _launch."""
    thread = threading.Thread(target=execute, args=(journal, run, lock), daemon=True, name="etl-" + run["run_id"][:8])
    _ACTIVE[run["run_id"]] = thread
    try:
        thread.start()
    except Exception:
        run.update(status="FAILED", error="无法启动 ETL 工作线程。")
        journal.save_run(run)
        lock.release()
        _ACTIVE.pop(run["run_id"], None)
        raise
    return public_run(journal.get_run(run["run_id"]))


def start(store: SourceStore, payload: dict) -> dict:
    _writable()
    if payload.get("confirm") is not True:
        raise CenterError("CONFIRM_ETL", "请确认执行清单；下载步骤会访问真实数据源。", 422)
    try:
        run_id = uuid.UUID(str(payload.get("request_id"))).hex
    except (ValueError, TypeError, AttributeError):
        raise CenterError("ETL_REQUEST_ID_REQUIRED", "请提供有效 request_id，防止重复提交。", 422) from None
    journal = EtlStore(store)
    request_hash = fingerprint({k: v for k, v in payload.items() if k != "confirm"})
    try:
        previous = journal.get_run(run_id)
    except CenterError as exc:
        if exc.code != "ETL_RUN_NOT_FOUND":
            raise
    else:
        if previous["request_hash"] != request_hash:
            raise CenterError("ETL_REQUEST_CONFLICT", "同一 request_id 不能执行不同流程。", 409)
        return public_run(journal.interrupted(previous))
    lock = _lock(store)
    try:
        if payload.get("workflow_id"):
            if payload.get("definition") is not None:
                raise CenterError("ETL_PLAN_CONFLICT", "选择已保存流程或临时流程，不可同时提供。")
            workflow = journal.workflow(str(payload["workflow_id"]))
            if workflow["revision"] != payload.get("expected_revision"):
                raise CenterError("REVISION_CONFLICT", "流程已修改，请重新加载。", 409)
            definition = parse_definition(workflow["definition"])
        else:
            definition = parse_definition(payload.get("definition"))
        options = parse_run_options(payload.get("options"))
        template = definition.model_dump(mode="json")
        definition = bind_parameters(definition, options).compiled()
        automatic = None
        if options.mode == 'auto_incremental':
            from .auto_incremental import plan, freeze_baseline
            automatic = plan(store, definition)
            if not automatic['public']['ready']:
                raise CenterError('AUTO_PLAN_BLOCKED', '；'.join(e['message'] for e in automatic['public']['errors']), 422)
            if payload.get('auto_plan_id') != automatic['public']['plan_id']:
                raise CenterError('AUTO_PLAN_CHANGED', '请先预览自动增量计划；快照、日期或配置变化后需要重新确认。', 409)
            definition = automatic['definition']
        frozen = _freeze(store, definition, options)
        directory = store.root / "etl_runs" / run_id
        if automatic:
            baseline = freeze_baseline(journal, automatic, directory / 'baseline' / 'data')
            frozen['task_baseline'] = {'run_id': None, 'workspace': baseline}
            frozen['auto_plan'] = automatic['public']
        config_dir = directory / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_artifacts = []
        for name in ("custom_indicators.json", "snapshot_indicator_config.json"):
            source = store.root / name
            if source.is_file():
                shutil.copyfile(source, config_dir / name)
                config_artifacts.append(journal.artifact(config_dir / name))
        frozen["config_artifacts"] = config_artifacts
        run = {"run_id": run_id, "request_hash": request_hash, "workflow_id": payload.get("workflow_id"),
               "workflow_revision": payload.get("expected_revision"), "definition": definition.model_dump(mode="json"),
               "template_definition": template, "options": options.model_dump(mode="json"),
               **({'auto_plan': automatic['public']} if automatic else {}),
               "name": definition.name, "status": "RUNNING", "owner_pid": os.getpid(), "owner_instance": OWNER,
               "created_at": utc_now(), "started_at": utc_now(), "attempt": 1, "published": False, "frozen": frozen,
               "steps": [{"id": s.id, "name": s.name, "kind": s.kind, "status": "PENDING", "attempt": 0} for s in definition.steps]}
        journal.save_run(run, create=True)
        return _launch(journal, run, lock)
    except Exception:
        lock.release()
        raise


def recovery_status(store: SourceStore, run: dict, current_fingerprint: str | None = None) -> dict:
    """Cheap read-only eligibility. Full artifact validation stays under the lock."""
    blockers = []
    if run['status'] not in {'FAILED', 'CANCELLED', 'INTERRUPTED'}:
        blockers.append({'code': 'ETL_NOT_RESUMABLE', 'message': '只有失败、中断或取消的任务可以继续。'})
    if is_file_lock_held(store.root / '.tushare_refresh.lock'):
        blockers.append({'code': 'DATA_TASK_RUNNING', 'message': '后台仍有下载或数据处理进程持有任务锁。调度服务中断不代表工作进程已停止；请等待其结束，不能重复启动或删除锁文件。'})
    current_fingerprint = current_fingerprint or execution_fingerprint()
    if run.get('frozen', {}).get('execution_fingerprint') != current_fingerprint:
        blockers.append({'code': 'ETL_IMPLEMENTATION_CHANGED', 'message': '执行程序已更新，旧任务不能原地续跑。已下载文件与检查点仍保留；需恢复原执行版本，或经过兼容性核验迁移到新运行，不能直接改写旧任务的版本标识。'})
    return {'can_resume': not blockers, 'artifact_check_pending': not blockers, 'blockers': blockers,
            'warnings': resume_warnings(run)}


def resume(store: SourceStore, identifier: str, confirm: bool) -> dict:
    _writable()
    if not confirm:
        raise CenterError("CONFIRM_ETL", "继续未完成步骤可能重新访问数据源，请确认。", 422)
    journal = EtlStore(store)
    status = recovery_status(store, journal.interrupted(journal.get_run(identifier)))
    if not status['can_resume']:
        raise CenterError(status['blockers'][0]['code'], '\n'.join(item['message'] for item in status['blockers']), 409)
    lock = _lock(store)
    try:
        run = journal.interrupted(journal.get_run(identifier))
        if run["status"] not in {"FAILED", "CANCELLED", "INTERRUPTED"}:
            raise CenterError("ETL_NOT_RESUMABLE", "只有失败、中断或取消的任务可以继续。", 409)
        if run["frozen"].get("execution_fingerprint") != execution_fingerprint():
            raise CenterError("ETL_IMPLEMENTATION_CHANGED", "计算或映射代码已变化，请建立新运行，不能改写历史运行的执行契约。", 409)
        from .task_runtime import verify_sources
        from .task_workspace import read_inventory
        verify_sources(store, run['frozen'])
        for state in run['steps']:
            if state['status'] == 'SUCCEEDED' and state.get('output', {}).get('workspace'):
                read_inventory(journal, state['output']['workspace'])
        for frozen in run["frozen"]["downloads"].values():
            for kind, record in (("source", frozen["source"]), ("interface", frozen["interface"])):
                if store.get(kind, record["config"]["id"])["revision"] != record["revision"]:
                    raise CenterError("ETL_CONFIG_CHANGED", "来源或接口配置已改变，请使用新运行，不能修改历史运行契约。", 409)
        for item in run["frozen"]["config_artifacts"]:
            journal.checked_path(item)
        if run.get('recovery_receipt'):
            receipt = json.loads(journal.checked_path(run['recovery_receipt']).read_text())
            journal.checked_path(receipt['original_run'])
            if receipt['target_execution'] != run['frozen']['execution_fingerprint']:
                raise CenterError('ETL_MIGRATION_BLOCKED', '恢复清单的目标版本与运行不一致。', 409)
            for shard in receipt['shards'].get('files', []):
                journal.checked_path(shard['imported'])
            for shard in receipt['shards'].get('history', {}).get('files', []):
                journal.checked_path(shard['imported'])
            if receipt['shards'].get('day_import_receipt'):
                journal.checked_path(receipt['shards']['day_import_receipt'])
        for state in run["steps"]:
            if state["status"] == "SUCCEEDED":
                for artifact in state.get("artifacts", []):
                    journal.checked_path(artifact)
            else:
                state.update(status="PENDING", error=None)
        resumed_at = utc_now()
        record_resume(run, resumed_at)
        run.update(status="RUNNING", owner_pid=os.getpid(), owner_instance=OWNER, error=None, started_at=resumed_at, finished_at=None, attempt=run["attempt"] + 1)
        run.pop('executor', None)
        journal.request_cancel(identifier, False)
        journal.save_run(run)
        return _launch(journal, run, lock)
    except Exception:
        lock.release()
        raise


def cancel(store: SourceStore, identifier: str) -> dict:
    _writable()
    journal = EtlStore(store)
    run = journal.interrupted(journal.get_run(identifier))
    if run["status"] == "RUNNING":
        journal.request_cancel(identifier)
    return public_run(journal.get_run(identifier))


def _snapshot_process(journal, run, directory, inputs, check):
    config_dir = journal.root / "etl_runs" / run["run_id"] / "config"
    for artifact in run["frozen"]["config_artifacts"]:
        journal.checked_path(artifact)
    # Existing indicator repositories may seed built-ins; never let that mutate
    # the frozen configuration that another snapshot or resume will reuse.
    runtime_config = directory / "runtime_config"
    shutil.copytree(config_dir, runtime_config)
    payload = json.dumps({"directory": str(directory), "inputs": {k: str(v) for k, v in inputs.items()}, "config_dir": str(runtime_config)})
    with subprocess.Popen([sys.executable, "-m", "backend.data_sources.etl_snapshot"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=Path(__file__).resolve().parents[2]) as process:
        try:
            process.stdin.write(payload)
            process.stdin.close()
            process.stdin = None
            while True:
                check()
                try:
                    stdout, _ = process.communicate(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    try:
        result = json.loads(stdout)
    except ValueError:
        raise CenterError("SNAPSHOT_PROCESS_FAILED", "快照计算进程异常退出；前置下载保留，可继续未完成步骤。") from None
    if not result.get("ok"):
        raise CenterError(result.get("code", "SNAPSHOT_FAILED"), result.get("message", "快照计算失败。"))
    return result["result"]


def execute(journal: EtlStore, run: dict, lock) -> None:
    started = time.monotonic()
    definition = EtlDefinition.model_validate(run["definition"])
    states = {s["id"]: s for s in run["steps"]}
    current = None
    def check():
        if journal.get_run(run["run_id"])["cancel_requested"]:
            raise CenterError("ETL_CANCELLED", "任务已在安全边界停止；已完成步骤保留。")
        if time.monotonic() - started >= definition.max_runtime_seconds:
            raise CenterError("ETL_TIMEOUT", "流程达到最长运行时间；已完成步骤保留。")
    try:
        for step in definition.steps:
            current = states[step.id]
            if current["status"] == "SUCCEEDED":
                continue
            check()
            if run.get('executor') and execution_fingerprint() != run['frozen']['execution_fingerprint']:
                raise CenterError('ETL_IMPLEMENTATION_CHANGED', '代码已更新，当前步骤完成后已停止推进；不能混用新旧执行版本。', 409)
            current.update(status="RUNNING", started_at=utc_now(), attempt=current["attempt"] + 1)
            # An attempt never inherits another attempt's counters or timestamps.
            for key in ('rows', 'pages', 'finished_at', 'error', 'code', 'heartbeat_at'):
                current.pop(key, None)
            current['progress'] = {'phase': '准备执行', 'message': '正在校验输入和准备工作区。',
                                   'completed': None, 'total': None, 'logs': []}
            journal.save_run(run)
            directory = journal.root / "etl_runs" / run["run_id"] / step.id / str(current["attempt"])
            directory.mkdir(parents=True, exist_ok=True)
            artifacts = []
            if step.kind == 'task':
                from .task_runtime import execute_task
                remaining = max(1, int(definition.max_runtime_seconds - (time.monotonic() - started)))
                output = execute_task(journal, run, step, states, directory, check, lock, remaining)
                artifacts.append(output['workspace'])
                current['mode'] = download_mode(step.mode, parse_run_options(run.get('options')))
            elif step.kind == "download":
                frozen = run["frozen"]["downloads"][step.id]
                source = SourceConfig.model_validate(frozen["source"]["config"])
                interface = InterfaceConfig.model_validate(frozen["effective_interface"])
                # Resume must reuse the original mode/window, not a later checkpoint.
                if "request_params" in frozen:
                    params, key = dict(frozen["request_params"]), frozen["checkpoint"]
                else:
                    params, key = incremental_params(journal.sources, source, interface, download_mode(step.mode, parse_run_options(run.get("options"))))
                current["mode"] = frozen.get("mode", download_mode(step.mode, parse_run_options(run.get("options"))))
                def progress(pages, count):
                    captured_at = utc_now()
                    window = current['progress'].setdefault('collection_window', {'first_at': captured_at})
                    window['last_at'] = captured_at
                    current.update(pages=pages, rows=count)
                    current['progress'].update(phase='下载分页', message=f'已接收 {pages} 页数据。',
                                               batches=pages, received_rows=count, activity_at=utc_now())
                    current['heartbeat_at'] = utc_now()
                    journal.save_run(run)
                remaining = max(1, int(definition.max_runtime_seconds - (time.monotonic() - started)))
                bounded = source.model_copy(update={"policy": source.policy.model_copy(update={"max_runtime_seconds": min(remaining, source.policy.max_runtime_seconds)})})
                rows, pages = download_rows(journal.sources, bounded, interface, params, check=check, progress=progress)
                raw = journal.write_json(directory / "raw.json", rows)
                artifacts.append(raw)
                output = {"raw": raw, "rows": len(rows), "pages": pages, "params": params, "checkpoint": key, "download_step": step.id}
                current.update(output=output, artifacts=artifacts)
                if not rows and not step.allow_empty:
                    raise CenterError("ETL_EMPTY_DATA", "来源返回空数据，流程已停止。请核对范围，或明确开启允许空结果。")
            elif step.kind == "map":
                downloaded = states[step.inputs[0]]["output"]
                rows = json.loads(journal.checked_path(downloaded["raw"]).read_text())
                frozen = run["frozen"]["downloads"][downloaded["download_step"]]
                interface = InterfaceConfig.model_validate(frozen["effective_interface"])
                batch = capture_batch(journal.sources, interface, rows, {**downloaded["params"], "etl_capture": run["run_id"]}, fingerprint(frozen))
                output = {"batch": batch, "rows": batch["source_rows"]}
                for item in batch["tables"]:
                    if item.get("artifact"):
                        artifacts.append({"path": item["artifact"], "checksum": item["checksum"]})
                current.update(output=output, artifacts=artifacts)
                if batch["status"] == "REJECTED":
                    raise CenterError("MAPPING_REJECTED", "映射未通过。原始数据已保留，后续取值和快照未执行。")
                advance_checkpoint(journal.sources, downloaded["checkpoint"], interface, rows)
            elif step.kind == "resolve":
                ids = list(run["frozen"]["history"].get(step.id, []))
                for key in step.inputs:
                    ids.append(states[key]["output"]["batch"]["batch_id"])
                policy = run["frozen"]["policy"]
                result = resolve_saved(journal.sources, step.table_id, policy["revision"], step.start_date, step.end_date, step.as_of, batch_ids=list(dict.fromkeys(ids)), frozen_policy=policy)
                output = {"resolution": result, "table_id": step.table_id, "rows": result["summary"]["selected_rows"]}
                artifacts.append({"path": result["artifact"], "checksum": result["checksum"]})
                current.update(output=output, artifacts=artifacts)
                if result["status"] == "NEEDS_REVIEW" or not output["rows"] and not step.allow_empty:
                    raise CenterError("ETL_QUALITY_GATE", "多源取值存在冲突、阻断或无可用记录。后续步骤未执行，请查看取值明细。")
            else:
                inputs = {states[key]["output"]["table_id"]: journal.checked_path(states[key]["artifacts"][0]) for key in step.inputs}
                output = _snapshot_process(journal, run, directory, inputs, check)
                artifacts = [journal.artifact(p) for p in directory.iterdir() if p.is_file()]
            check()
            receipt = journal.write_json(directory / "result.json", output)
            current.update(status="SUCCEEDED", output=output, artifacts=[*artifacts, receipt], rows=output.get("rows", 0), finished_at=utc_now(), error=None)
            journal.save_run(run)
        warning_count = sum(s.get('output', {}).get('mapped_rejected_batches', 0) + s.get('output', {}).get('warnings', 0) for s in run['steps'])
        run.update(status='SUCCEEDED', warning_count=warning_count,
                   message=('流程采集完成，但有映射或数据告警，请检查步骤结果；不能标记标准数据就绪。' if warning_count else '流程完成，结果已保存为私有候选，尚未发布到正式研究。'))
    except Exception as exc:
        code = exc.code if isinstance(exc, CenterError) else "ETL_STEP_FAILED"
        message = exc.message if isinstance(exc, CenterError) else f"步骤执行失败（{type(exc).__name__}）；已完成步骤保留。"
        run.update(status="CANCELLED" if code == "ETL_CANCELLED" else "FAILED", code=code, error=message)
        if current is not None and current["status"] != "SUCCEEDED":
            current.update(status=run["status"], error=message, code=code, finished_at=utc_now())
        for state in run["steps"]:
            if state["status"] == "PENDING":
                state.update(status="SKIPPED", error="前置流程未完成，尚未执行。")
    finally:
        run["finished_at"] = utc_now()
        try:
            journal.save_run(run)
        finally:
            lock.release()
            _ACTIVE.pop(run["run_id"], None)
