"""One cancellable runner with transactional checkpoints and evidence-based stopping."""
from __future__ import annotations

import asyncio
import contextvars
import copy
import fcntl
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from custom_indicators.errors import IndicatorDomainError
from . import data_policy
from . import ledger, memory
from .research_runtime import research_context, catalog_version, data_generation, system_prompt
from .context import close_pending, compact_if_needed, context_messages, record_usage, measure, capacity_error, refresh_dependencies
from .contracts import AgentError, AgentMessageRequest
from .llm import LLMUnavailableError
from .progress import ProgressGuard
from .scopes import allowed_tools, require_tool
from .sessions import (AgentSessionStore, ACTIVE_STATUSES, page_snapshot_identity,
                       public_draft, stable_hash, stable_json, stop_message)
from .tools import TOOL_REGISTRY, execute_tool, parse_arguments, tool_specs, model_view
from .views import Projection
from .views import VIEW_TASK_STATE

MAX_FINAL_REPLY_CHARS = 20_000
PATCH_KEYS = ("draft", "last_valid_draft", "product_candidates", "preview", "pending_confirmation", "memory_proposals", "task_plans")


def optional_limit(name):
    value = os.getenv(name)
    if not value:
        return None
    number = int(value)
    if number < 1:
        raise ValueError(f"{name} must be positive when configured")
    return number


def execution_limits():
    return {"tool_calls_per_turn": optional_limit("AGENT_MAX_TOOL_CALLS"),
            "tool_rounds_per_turn": optional_limit("AGENT_MAX_MODEL_STEPS"), "session_events": None}


class RunStopped(BaseException):
    pass


def require_controller_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        raise AgentError("AGENT_CONTROLLER_CONTEXT", "智能体控制操作必须在所属事件循环中执行。", status_code=503) from None


class RunModel:
    def __init__(self, controller, run, llm, stop, service):
        self.controller, self.run, self.llm, self.stop = controller, run, llm, stop
        self.service = service

    @property
    def context_window_tokens(self):
        return getattr(self.llm, 'context_window_tokens', 32768)

    async def complete(self, *, system=None, messages=None, tools=(), system_seal=None, **kwargs):
        if self.stop.is_set():
            raise RunStopped()
        # One versioned, signed admission policy for every outbound request: primary
        # turns, compaction, no-progress finalization, retries and resumed history all
        # pass through here, so no path can hand unverified payloads to the adapter.
        try:
            admitted = data_policy.enforce_request(system=system, messages=messages, tools=tools,
                                                   system_seal=system_seal)
        except AgentError as exc:
            diagnostic = (exc.diagnostics or [{}])[0]
            self.controller._event(self.run, "data.rejected", status="blocked",
                                   policy_version=data_policy.POLICY_VERSION,
                                   code=diagnostic.get("code"), field=exc.field)
            raise
        if not self.controller._event(self.run, "data.admitted", **admitted):
            raise RunStopped()
        task = asyncio.create_task(self.llm.complete(system=system, messages=messages, tools=list(tools), **kwargs))
        stopped = asyncio.create_task(self.stop.wait())
        self.run["usage"]["model_steps"] += 1
        try:
            done, _ = await asyncio.wait({task, stopped}, return_when=asyncio.FIRST_COMPLETED)
            if self.stop.is_set():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise RunStopped()
            reply, failure = None, None
            try:
                reply = await task
            except Exception as exc:
                failure = exc
            if failure is None:
                usage = getattr(reply, "usage", {}) or {}
                for key in ("prompt_tokens", "completion_tokens"):
                    value = usage.get(key)
                    if isinstance(value, int):
                        self.run["usage"][key] = (self.run["usage"][key] or 0) + value
                self.run["usage"]["http_attempts"] = self.run["usage"].get("http_attempts", 0) + getattr(reply, "attempts", 1)
            # This transaction orders consumption against remote cancellation. The
            # body remains tentative until its model/tool/final checkpoint commits.
            if not self.controller._event(self.run, "model.returned", status="error" if failure else "ok"):
                raise RunStopped()
            if failure is not None:
                raise failure
            if catalog_version(self.service) != self.run["catalog_version"]:
                self.controller.cancel(self.run["session_id"], self.run["run_id"], reason="context_changed", source="catalog")
                raise RunStopped()
            return reply
        finally:
            stopped.cancel()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, stopped, return_exceptions=True)


class RunController:
    def __init__(self, store=None, *, page_services=None):
        self.loop = require_controller_loop()
        self.thread_id = threading.get_ident()
        self.store = store or AgentSessionStore()
        self.page_services = page_services or {}
        self.instance_id = uuid.uuid4().hex
        self.workspace = str(Path(__file__).resolve().parents[2])
        self.tasks = {}
        self.stops = {}
        self.futures = set()
        self.future_owners = {}
        self.closing = False
        self.capacity = max(1, int(os.getenv("AGENT_ACTIVE_RUNS", "2")))
        self.executor = ThreadPoolExecutor(max_workers=self.capacity, thread_name_prefix="agent-tool")
        self.owner_dir = self.store.root / "owners"
        self.owner_dir.mkdir(exist_ok=True)
        self.owner_file = open(self.owner_dir / f"{self.instance_id}.lock", "a+")
        os.chmod(self.owner_file.name, 0o600)
        fcntl.flock(self.owner_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.recover_owners()

    def _assert_owner(self):
        if threading.get_ident() != self.thread_id or require_controller_loop() is not self.loop:
            raise AgentError("AGENT_CONTROLLER_CONTEXT", "智能体控制操作不属于当前执行上下文。", status_code=503)

    def _dispatch_tool_callback(self, callback, work):
        try:
            self.loop.call_soon_threadsafe(callback, work)
        except RuntimeError:
            if not self.closing or not self.loop.is_closed():
                raise
        finally:
            # A stopped loop can accept callbacks it will never run. Once shutdown's
            # runners are done, retire only local resources here; durable recovery
            # stays with the queued owner callback or the next owner after restart.
            if self.closing and all(task.done() for task in tuple(self.tasks.values())):
                self.futures.discard(work)
                self.future_owners.pop(work, None)
                if not self.futures and not self.owner_file.closed:
                    self.owner_file.close()

    def recover_owners(self):
        self._assert_owner()
        for run in self.store.unfinished_runs():
            owner = run.get("owner_instance")
            if not owner or run.get("owner_workspace") != self.workspace:
                continue
            path = self.owner_dir / f"{owner}.lock"
            if not path.exists():
                continue  # Unknown ownership must not become permission to replay.
            with open(path, "a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                self.store.recover(run)

    def submit(self, session_id, request, llm, service):
        self._assert_owner()
        owner = {"owner_instance": self.instance_id, "owner_workspace": self.workspace}
        orphaned = sum(not future.done() and rid not in self.tasks for future, rid in list(self.future_owners.items()))
        if not self.store.find_message(session_id, request.message_id) and (self.closing or len(self.tasks)+orphaned >= self.capacity):
            raise AgentError("AGENT_SERVICE_BUSY", "助手正在处理其他任务，请稍后继续。", status_code=503)
        run, replayed = self.store.accept(session_id, request, owner)
        if replayed:
            return run, True
        stop = asyncio.Event()
        self.stops[run["run_id"]] = stop
        task = asyncio.create_task(self._execute(run, llm, service, stop))
        self.tasks[run["run_id"]] = task
        def finished(_):
            self.tasks.pop(run["run_id"], None)
            self.stops.pop(run["run_id"], None)
        task.add_done_callback(finished)
        return run, False

    def reconcile(self):
        """Retry unfinished local cleanup only after its runner and real tool threads exit."""
        self._assert_owner()
        for run in self.store.unfinished_runs():
            rid = run["run_id"]
            if run.get("owner_instance") != self.instance_id or rid in self.tasks:
                continue
            if any(owner == rid and not f.done() for f, owner in list(self.future_owners.items())):
                continue
            if run["status"] in ACTIVE_STATUSES:
                self.store.recover(run, reason="storage_unavailable")
            elif run.get("execution_blocked_by"):
                self.store.release_quarantine(run["session_id"], run["execution_blocked_by"])

    async def wait(self, run):
        self._assert_owner()
        task = self.tasks.get(run["run_id"])
        if task:
            await asyncio.shield(task)
        return self.store.get_run(run["session_id"], run["run_id"])

    def cancel(self, session_id, run_id, *, reason="user_cancelled", context=None, source=None):
        self._assert_owner()
        run = self.store.cancel(session_id, run_id, reason=reason, context=context, source=source)
        stop = self.stops.get(run_id)
        if stop and run.get("cancel_requested"):
            stop.set()
        return run

    def _event(self, run, kind, **data):
        self._assert_owner()
        return self.store.admit(run, events=[{"type": kind, "data": data}])

    async def _prepare_context(self, run, *, phase, **kwargs):
        compacted = []
        await compact_if_needed(run["checkpoint"], on_compacted=compacted.append, **kwargs)
        # compact_if_needed installs its candidate only after its callback returns.
        # Commit that installed context explicitly; control events never carry it.
        run["phase"] = phase
        events = [{"type": "context.compacted", "data": item} for item in compacted]
        events.append({"type": "run.phase", "data": {"status": "running", "phase": phase}})
        if not self.store.checkpoint(run, events=events):
            raise RunStopped()

    async def _tool(self, run, call, state, page_context, service, stop):
        engine = getattr(service, "compute_engine", None)
        timeout = max(.01, float(os.getenv("AGENT_TOOL_TIMEOUT_SECONDS", str(getattr(engine, "hard_timeout_seconds", 600)))))
        operation_id = f"op-{uuid.uuid4().hex}"
        receipt = {"operation_id": operation_id, "model_step": run["checkpoint"]["model_step"], "call_id": call["id"],
                   "tool": call["name"], "arguments": call["arguments"], "status": "started"}
        run["phase"] = "tool"
        if not self.store.admit(run, tool_receipt=receipt, events=[{"type": "tool.started", "data": {"tool": call["name"], "operation_id": operation_id}}]):
            raise RunStopped()
        context = contextvars.copy_context()
        started = time.monotonic()
        def execute():
            if call['name'] == 'context.read':
                require_tool(state['scope'], page_context.context_kind, call['name'])
                arguments = parse_arguments(call['name'], call['arguments']).model_dump()
                return self.store.read_context(run['session_id'], **arguments)
            return execute_tool(call['name'], call['arguments'], session=state, page_context=page_context, service=service,
                                page_snapshot=(run.get("request") or {}).get("page_snapshot"),
                                store=self.store, session_id=run["session_id"], page_services=self.page_services)
        try:
            work = self.executor.submit(context.run, execute)
        except Exception:
            # No worker owns this admitted operation; release it or leave the
            # durable fence for reconciliation if storage is temporarily down.
            try:
                self.store.release_quarantine(run["session_id"], operation_id)
            except AgentError as exc:
                if exc.code != "AGENT_STORAGE_UNAVAILABLE":
                    raise
            raise
        self.futures.add(work)
        self.future_owners[work] = run["run_id"]
        def operation_done(f):
            self._assert_owner()
            self.futures.discard(f)
            self.future_owners.pop(f, None)
            try:
                self.store.release_quarantine(run["session_id"], operation_id)
            except AgentError as exc:
                if exc.code != "AGENT_STORAGE_UNAVAILABLE":
                    raise
                # The started receipt keeps the durable fence for reconciliation.
            finally:
                if self.closing and not self.futures and not self.owner_file.closed:
                    self.owner_file.close()
        work.add_done_callback(lambda f: self._dispatch_tool_callback(operation_done, f))
        future = asyncio.wrap_future(work)
        # Observe late worker failures even if a control-state write aborts this
        # coroutine before its normal result/timeout handling is reached.
        future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        stop_task = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait({future, stop_task}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if stop_task in done and not future.done():
                self._event(run, "run.phase", status="stopping", phase="tool")
                await asyncio.wait({future}, timeout=max(0, timeout-(time.monotonic()-started)))
            if not future.done():
                run["execution_blocked_by"] = operation_id
                receipt.update(status="unknown", applied=False)
                run["stop_reason"] = "operation_timeout"
                self.store.admit(run, tool_receipt=receipt)
                raise AgentError("AGENT_OPERATION_TIMEOUT", "后台操作超时，已保留进度；当前操作退出前不能叠加计算。", status_code=504)
            self.futures.discard(work)
            try:
                result = future.result()
            except (AgentError, IndicatorDomainError) as exc:
                error = {"code": exc.code, "message": "工具未完成；请根据错误代码核对参数或数据条件。"}
                result = data_policy.seal_notice(data_policy.notice(
                    source=call["name"], status="error", ok=False,
                    message=str(error.get("message") or "工具执行失败。"), extra={"error": error}), call["name"])
            except Exception:
                result = data_policy.seal_notice(data_policy.notice(
                    source=call["name"], status="error", ok=False,
                    message="工具执行失败，请根据已有证据调整需求。",
                    extra={"error": {"code": "AGENT_TOOL_FAILED", "message": "工具执行失败，请根据已有证据调整需求。"}}), call["name"])
            receipt.update(status="completed", duration_ms=int((time.monotonic()-started)*1000))
            return result, receipt
        except asyncio.CancelledError:
            # Closing an app does not cancel a running Python thread. Fence its result
            # and keep the owner/slot until the real operation has finished.
            if not work.done():
                run["execution_blocked_by"] = operation_id
                receipt.update(status="unknown", applied=False)
                self.store.admit(run, tool_receipt=receipt)
            raise
        finally:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)

    async def _execute(self, run, llm, service, stop):
        started = time.monotonic()
        run["started_monotonic"] = started
        model = RunModel(self, run, llm, stop, service)
        if hasattr(llm, "on_retry"):
            llm.on_retry = lambda attempt: self._event(run, "run.phase", status="running", phase="waiting_retry", attempt=attempt)
        state = self.store.read(run["session_id"])
        request = AgentMessageRequest.model_validate(run["request"])
        page = request.page_context
        guard = ProgressGuard(run.get("detector_state"), continued=bool(request.resume_from_run_id),
                              seen=lambda kind, values: self.store.seen_facts(run.get("progress_space_id",run["run_id"]), kind, values))
        bindings = ExitStack()
        try:
            if data_policy.user_text_violation(request.text):
                raise data_policy._blocked("user", "structured_user_payload")
            page = bindings.enter_context(research_context(run, page, service))
            specs = tool_specs(allowed_tools(state["scope"], page.context_kind))
            checkpoint = copy.deepcopy(state.get("conversation") or {"messages": [], "summary": "", "model_step": 0})
            resume_batch = False
            if request.resume_from_run_id:
                parent = self.store.get_run(run["session_id"], request.resume_from_run_id)
                if not parent.get("checkpoint") and parent.get("request", {}).get("text"):
                    checkpoint.setdefault("messages", []).append({"role": "user", "content": parent["request"]["text"]})
                if parent.get("checkpoint"):
                    checkpoint = copy.deepcopy(parent["checkpoint"])
                    try:
                        pending_before = data_policy.pending_tool_calls(checkpoint.get("messages", []))
                    except AgentError:
                        pending_before = None
                    # Pending calls were planned against the parent's page evidence. Replaying them
                    # under different evidence would mix two page states, so only an identical
                    # snapshot content keeps the safe resume contract; otherwise the batch closes
                    # and the model replans against the new page.
                    same_evidence = (page_snapshot_identity((parent.get("request") or {}).get("page_snapshot"))
                                     == page_snapshot_identity((run.get("request") or {}).get("page_snapshot")))
                    resume_batch = (parent["status"] == "interrupted" and request.text.strip() in {"继续", "继续分析", "继续处理"}
                        and parent["request"]["page_context"] == request.page_context.model_dump()
                        and parent.get("data_generation") == run["data_generation"]
                        and parent.get("catalog_version") == run["catalog_version"]
                        and same_evidence
                        and bool(pending_before))
            checkpoint.setdefault("messages", [])
            if checkpoint.get('summary') and 'pinned_user_messages' not in checkpoint:
                # Older checkpoints kept user intent only in a generated summary.
                history = self.store.message_page(run['session_id'], limit=200)['items']
                present = {m.get('content') for m in checkpoint['messages'] if m.get('role') == 'user'}
                checkpoint['pinned_user_messages'] = [{'role': 'user', 'content': e['text']} for e in history
                    if e.get('speaker') == 'user' and e.get('id') != request.message_id and e.get('text') not in present][-4:]
            if checkpoint:
                # Policy-old checkpoints are rebuilt from current permitted evidence; opaque
                # old bodies are omitted explicitly rather than replayed to the model.
                reprojected = data_policy.scrub_checkpoint(checkpoint, view_for=model_view,
                                                           projection_for=lambda tool: Projection())
                checkpoint['data_policy_version'] = data_policy.POLICY_VERSION
            if resume_batch:
                try:
                    # Sanitizing a historical completed call does not invalidate the
                    # remaining plan, but changing that plan requires a fresh decision.
                    resume_batch = pending_before == data_policy.pending_tool_calls(checkpoint['messages'])
                except AgentError:
                    resume_batch = False
            checkpoint.pop("pending_calls", None)  # Legacy cache is never dispatch authority.
            context_key = stable_hash(llm.context_key) if hasattr(llm, 'context_key') else None
            if checkpoint.get('llm_context_key') != context_key:
                # Do not send provider-specific reasoning to a different API or model.
                for message in checkpoint['messages']:
                    message.pop('reasoning_content', None)
                for key in ('token_ratio', 'last_prompt_tokens', 'observed_input_budget', 'failed_compaction_key'):
                    checkpoint.pop(key, None)
                checkpoint.pop('summary', None)
                checkpoint.pop('summary_seal', None)
            checkpoint['llm_context_key'] = context_key
            capacity_key = stable_hash([context_key, model.context_window_tokens])
            if checkpoint.get('capacity_key') != capacity_key:
                checkpoint.pop('observed_input_budget', None)
                checkpoint.pop('failed_compaction_key', None)
            checkpoint['capacity_key'] = capacity_key
            if resume_batch:
                checkpoint["deferred_user"] = request.text
            else:
                close_pending(checkpoint["messages"])
                checkpoint.pop("deferred_user", None)
                checkpoint["messages"].append({"role": "user", "content": request.text})
            run.update(checkpoint=checkpoint, detector_state=guard.snapshot())
            self._refresh_task(run, state, page)
            guard.relevant = lambda tool, arguments: ledger.relevant(tool, arguments,
                checkpoint.get('task_state') or {}, state.get('draft'))
            if not self.store.checkpoint(run, events=[{"type": "run.started", "data": {"status": "running", "phase": "thinking"}}]):
                raise RunStopped()
            if reprojected and (reprojected["reprojected"] or reprojected["arguments_reset"]):
                self._event(run, "data.reprojected", policy_version=data_policy.POLICY_VERSION, **reprojected)
            limits = execution_limits()
            while True:
                if stop.is_set():
                    raise RunStopped()
                if (limits["tool_calls_per_turn"] is not None and run["usage"]["tool_calls"] >= limits["tool_calls_per_turn"]) or (limits["tool_rounds_per_turn"] is not None and run["usage"]["model_steps"] >= limits["tool_rounds_per_turn"]):
                    await self._finish(run, "已达到显式配置的执行上限，进度已保留。", status="paused", reason="explicit_limit")
                    return
                calls = data_policy.pending_tool_calls(checkpoint["messages"])
                if not calls:
                    if checkpoint.get("deferred_user"):
                        checkpoint["messages"].append({"role": "user", "content": checkpoint.pop("deferred_user")})
                    system = system_prompt(state, page, run["catalog_version"], (run.get("request") or {}).get("page_snapshot"))
                    if guard.state["recovering"]:
                        system += "\n" + guard.recovery_prompt()
                    def compacting():
                        run['phase'] = 'compacting'
                        self._event(run, 'run.phase', status='running', phase='compacting')
                    for attempt in range(2):
                        await self._prepare_context(run, phase="recovery" if guard.state["recovering"] else "thinking",
                            system=system, llm=model, tools=specs, force=attempt > 0, on_compacting=compacting)
                        messages = context_messages(checkpoint)
                        system_seal = data_policy.seal_text(system, "primary")
                        try:
                            reply = await model.complete(system=system, messages=messages, tools=specs,
                                                         system_seal=system_seal)
                            record_usage(checkpoint, system=system, messages=messages, tools=specs, usage=reply.usage)
                            break
                        except LLMUnavailableError as exc:
                            if exc.code != 'AGENT_LLM_CONTEXT_OVERFLOW':
                                raise
                            if attempt:
                                raise capacity_error('模型接口在整理后仍报告输入过长。进度已保留，请核对该 API 的实际上下文窗口。') from None
                            checkpoint['observed_input_budget'] = min(checkpoint.get('observed_input_budget', model.context_window_tokens),
                                                                     int(measure(checkpoint, system, specs)[0] * .8))
                    assistant_message = {'role': 'assistant', 'content': reply.content or ''}
                    if reply.reasoning_content is not None:
                        assistant_message['reasoning_content'] = reply.reasoning_content
                    if not reply.tool_calls:
                        if not str(reply.content or "").strip():
                            raise LLMUnavailableError("模型未返回有效答复。", code="AGENT_LLM_INVALID_RESPONSE")
                        # Preserve token calibration without treating the final reply
                        # as committed content before the terminal transaction.
                        if not self.store.checkpoint(run):
                            raise RunStopped()
                        await self._finish(run, str(reply.content)[:MAX_FINAL_REPLY_CHARS], reasoning_content=reply.reasoning_content)
                        return
                    calls = [{"id": call.call_id, "name": call.name, "arguments": call.arguments} for call in reply.tool_calls]
                    ids = [call["id"] for call in calls]
                    if len(set(ids)) != len(ids) or not all(ids):
                        raise LLMUnavailableError("模型工具调用标识重复或缺失。", code="AGENT_LLM_INVALID_RESPONSE")
                    checkpoint["model_step"] = checkpoint.get("model_step", 0) + 1
                    run["usage"]["rounds"] += 1
                    assistant_message['tool_calls'] = [
                        {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": stable_json(c["arguments"])}} for c in calls]
                    checkpoint["messages"].append(assistant_message)
                    if not self.store.checkpoint(run, events=[{"type": "model.completed", "_model_message": checkpoint["messages"][-1]}]):
                        raise RunStopped()
                for index, call in enumerate(calls):
                    if stop.is_set():
                        raise RunStopped()
                    if limits["tool_calls_per_turn"] is not None and run["usage"]["tool_calls"] >= limits["tool_calls_per_turn"]:
                        await self._finish(run, "已达到显式配置的执行上限，进度已保留。", status="paused", reason="explicit_limit")
                        return
                    tool = TOOL_REGISTRY.get(call["name"])
                    # Portfolio evaluations/diagnosis read an immutable run; a
                    # scenario instead rebuilds it from the current market data.
                    uses_current_data = bool(tool and "data" in tool.dependencies and not (
                        page.context_kind == "portfolio" and (call["name"] == "portfolios.eval" or
                        (call["name"] == "page.analyze" and call["arguments"].get("operation") != "scenario"))))
                    if catalog_version(service) != run["catalog_version"]:
                        self.cancel(run["session_id"], run["run_id"], reason="context_changed", source="catalog")
                        raise RunStopped()
                    if uses_current_data and data_generation(service) != run["data_generation"]:
                        self.cancel(run["session_id"], run["run_id"], reason="context_changed", source="data")
                        raise RunStopped()
                    try:
                        arguments = parse_arguments(call["name"], call["arguments"]).model_dump()
                    except AgentError:
                        arguments = call["arguments"]
                    deps = {"context": run["request"]["page_context"], "catalog": run["catalog_version"]}
                    if tool and "data" in tool.dependencies:
                        deps.update(data_generation=run["data_generation"], effective_context=run.get("effective_context"))
                    if tool and "draft" in tool.dependencies:
                        deps["definition"] = (state.get("draft") or {}).get("definition_hash")
                    if tool and "page_evidence" in tool.dependencies:
                        # The selected section's frozen content, not the random snapshot id: the same
                        # content with a new id must stay blocked, changed content must unblock.
                        section = arguments.get("section") if isinstance(arguments, dict) else None
                        deps["page_evidence"] = page_snapshot_identity((run.get("request") or {}).get("page_snapshot"), section)
                    run["usage"]["tool_calls"] += 1
                    if guard.check_call(call["name"], arguments, deps):
                        decision = guard.record_blocked(call["name"], arguments, deps)
                        blocked_error = {"code": "AGENT_NO_PROGRESS", "message": guard.recovery_prompt()}
                        result = data_policy.seal_notice(data_policy.notice(
                            source=call["name"], status="blocked", ok=False,
                            message=blocked_error["message"], extra={"error": blocked_error}), call["name"])
                        receipt = {"model_step": checkpoint["model_step"], "call_id": call["id"], "tool": call["name"], "status": "blocked"}
                        local_state = state
                    else:
                        local_state = copy.deepcopy(state)
                        local_state['_task_dependencies'] = checkpoint.get('task_state', {}).get('dependencies', {})
                        result, receipt = await self._tool(run, call, local_state, page, service, stop)
                        if catalog_version(service) != run["catalog_version"]:
                            self.cancel(run["session_id"], run["run_id"], reason="context_changed", source="catalog")
                        elif uses_current_data and data_generation(service) != run["data_generation"]:
                            self.cancel(run["session_id"], run["run_id"], reason="context_changed", source="data")
                        decision = guard.record(call["name"], arguments, result, draft=local_state.get("draft"), dependencies=deps)
                    preview_payload = local_state.pop("_preview_payload", None)
                    if preview_payload is not None and not stop.is_set():
                        local_state["preview"] = self.store.save_preview(run, preview_payload)
                        result["preview"] = local_state["preview"]
                    model_result = {k: v for k, v in result.items() if not k.startswith("_")}
                    if receipt.get('operation_id') and call['name'] != 'context.read':
                        model_result['context_ref'] = receipt['operation_id']
                    # Sign only at the last step: the signature covers exactly the content
                    # the model will see, and the same sealed body is stored as the receipt.
                    model_result = data_policy.seal(model_result, call["name"])
                    message = {"role": "tool", "tool_call_id": call["id"], "content": stable_json(model_result)}
                    checkpoint["messages"].append(message)
                    run["detector_state"] = guard.snapshot()
                    receipt["result"] = model_result
                    receipt["execution_key"] = stable_hash([call["name"], arguments, deps])
                    receipt["dependency_stamp"] = deps
                    receipt["validated_arguments"] = arguments
                    success = result.get("ok", True)
                    trace = {"round": run["usage"]["rounds"], "tool": call["name"], "status": "ok" if success else "error",
                             "error_code": (result.get("error") or {}).get("code"), "duration_ms": receipt.get("duration_ms", 0)}
                    suboperations = local_state.pop("_tool_suboperations", [])
                    if suboperations:
                        trace["suboperations"] = suboperations
                    # A response exposes a bounded trace; the complete trace is durable events.
                    run["tool_trace"] = (run.get("tool_trace", []) + [trace])[-200:]
                    patch = {k: local_state.get(k) for k in PATCH_KEYS}
                    events = [{"type": "tool.completed", "data": trace, "_model_message": message}]
                    if local_state.get("draft") != state.get("draft"):
                        events.append({"type": "draft.updated", "data": {"draft": public_draft(local_state)}})
                    if local_state.get("preview") != state.get("preview"):
                        events.append({"type": "preview.updated", "data": {"preview": local_state.get("preview")}})
                    if decision["progress"]:
                        events.append({"type": "progress.updated", "data": {"tool": call["name"], "summary": "已取得新的工具证据"}})
                    applied = self.store.checkpoint(run, events=events, patch=patch, tool_receipt=receipt, new_facts=guard.pending_facts)
                    if not applied or stop.is_set():
                        raise RunStopped()
                    state.update(patch)
                    self._refresh_task(run, state, page)
                    if decision["recovery_started"] or receipt["status"] == "blocked" or decision["pause"]:
                        for skipped in calls[index+1:]:
                            skipped_message = {"role": "tool", "tool_call_id": skipped["id"], "content": stable_json(
                                data_policy.seal_notice(
                                    data_policy.notice(source=skipped["name"], status="not_executed",
                                                       message="需要先纠偏", extra={"reason": "需要先纠偏"}),
                                    skipped["name"]))}
                            checkpoint["messages"].append(skipped_message)
                            if not self.store.checkpoint(run, tool_receipt={"model_step": checkpoint["model_step"], "call_id": skipped["id"], "tool": skipped["name"], "status": "not_executed"}, events=[{"type": "tool.completed", "data": {"tool": skipped["name"], "status": "not_executed"}, "_model_message": skipped_message}]):
                                raise RunStopped()
                        if not self.store.checkpoint(run, events=[{"type": "run.recovering", "data": {"status": "running", "phase": "recovery", "reason": decision["reason"]}}]):
                            raise RunStopped()
                        break
                if guard.state["pause"]:
                    run["phase"] = "summarizing"
                    fallback = "当前方法没有取得新进展，已暂停自动处理并保留草稿。请补充计算口径，或调整需求后继续。"
                    reasoning_content = None
                    try:
                        summary_system = "只能根据工具证据用简短中文说明已完成内容、阻碍和需要用户补充什么。不能调用工具、不能编造结果。"
                        await self._prepare_context(run, phase="summarizing", system=summary_system, llm=model)
                        final = await model.complete(system=summary_system, messages=context_messages(checkpoint), tools=[],
                                                     system_seal=data_policy.seal_text(summary_system, "finalization"))
                        if final.content and not final.tool_calls:
                            fallback = str(final.content)[:MAX_FINAL_REPLY_CHARS]
                            reasoning_content = final.reasoning_content
                    except Exception:
                        pass
                    await self._finish(run, fallback, status="paused", reason="no_progress", reasoning_content=reasoning_content)
                    return
        except RunStopped:
            current = self.store.get_run(run["session_id"], run["run_id"])
            reason = current.get("stop_reason") or "user_cancelled"
            await self._finish(run, stop_message(current), status="cancelled" if reason == "user_cancelled" else "paused", reason=reason)
        except asyncio.CancelledError:
            run.update(self.store.interrupt(run))
            raise
        except Exception as exc:
            known = isinstance(exc, (LLMUnavailableError, AgentError))
            error = {"code": exc.code if known else "AGENT_RUN_FAILED", "message": str(exc) if isinstance(exc, LLMUnavailableError) else exc.message if isinstance(exc, AgentError) else "助手处理失败，已提交进度已保留。"}
            if isinstance(exc, LLMUnavailableError) and exc.upstream_status:
                error["upstream_status"] = exc.upstream_status
            run["error"] = error
            reason = "context_capacity" if error["code"] == "AGENT_CONTEXT_CAPACITY" else "operation_timeout" if error["code"] == "AGENT_OPERATION_TIMEOUT" else "data_policy_blocked" if error["code"] == "AGENT_DATA_ADMISSION_BLOCKED" else "upstream_error"
            try:
                await self._finish(run, error["message"], status="paused" if reason in {"context_capacity", "data_policy_blocked"} else "failed", reason=reason)
            except Exception:
                # Durable last checkpoint wins if the storage device disappeared.
                pass
        finally:
            run["usage"]["duration_ms"] = int((time.monotonic()-started)*1000)
            bindings.close()
            close = getattr(llm, "aclose", None)
            if close:
                await close()

    async def _finish(self, run, text, *, status="completed", reason=None, reasoning_content=None):
        self._assert_owner()
        run["usage"]["duration_ms"] = int((time.monotonic()-run.get("started_monotonic", time.monotonic()))*1000)
        run.update(self.store.finish(run, text, status=status, reason=reason, reasoning_content=reasoning_content))

    def _refresh_task(self, run, state, page):
        run['calculation_hash'] = stable_hash(page.calculation.model_dump())
        task, evidence = ledger.rebuild(self.store, run['session_id'], page=page.model_dump(),
            page_snapshot=(run.get('request') or {}).get('page_snapshot'),
            dependencies={key: run.get(key) for key in ('catalog_version', 'data_generation', 'calculation_hash')},
            draft=state.get('draft'), best_draft=state.get('last_valid_draft'))
        if len(task['quoted_constraints']) > data_policy.MAX_LIST_ITEMS:
            raise capacity_error('已引用约束超出当前工作集容量，完整状态已保留；不能通过丢弃约束继续。')
        projected, _ = VIEW_TASK_STATE(ledger.working_view(task), Projection())
        object_ids = [target.product_id for target in getattr(page.calculation, 'targets', [])]
        preferences = memory.recall(store=self.store, session_id=run['session_id'], object_ids=object_ids)
        preference_count = len(preferences)
        terms = ledger.tokens(' '.join(item['text'] for item in task['sources'][-3:]))
        preferences.sort(key=lambda item: (item['key'].startswith('reply.'),
                                          bool(terms & ledger.tokens(item['text'])),
                                          item['accepted_at']), reverse=True)
        preferences = preferences[:5]
        selected = ledger.select_evidence(evidence, task)
        projected.update(preferences=preferences, preference_count=preference_count,
                         omitted_preference_count=preference_count-len(preferences),
                         evidence_refs=[item['ref'] for item in selected],
                         read_evidence=list({(item['tool'], item['kind']): item for item in task['read_evidence']}.values()))
        projected['revision'] = stable_hash(projected)
        refresh_dependencies(run['checkpoint'], projected['revision'])
        run['checkpoint'].update(task_state=data_policy.seal(projected, 'task.state'),
            memory_sources=preferences, selected_evidence=selected, session_id=run['session_id'],
            valid_evidence_refs=[item['ref'] for item in evidence])

    async def close(self):
        self._assert_owner()
        self.closing = True
        for rid, stop in tuple(self.stops.items()):
            task = self.tasks.get(rid)
            if task and not task.done():
                # App shutdown is an interruption, not an invented user cancellation.
                task.cancel()
        if self.tasks:
            await asyncio.gather(*tuple(self.tasks.values()), return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)
        if not self.futures:
            self.owner_file.close()
