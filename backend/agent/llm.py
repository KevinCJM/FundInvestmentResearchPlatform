"""Cancellable, bounded OpenAI-compatible HTTP calls and deterministic offline fixtures."""
from __future__ import annotations

import asyncio
import copy
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit

import httpx

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class LLMUnavailableError(Exception):
    def __init__(self, message, *, code="AGENT_LLM_UNAVAILABLE", upstream_status=None):
        super().__init__(message)
        self.code, self.upstream_status = code, upstream_status


def _wire_name(name):
    alias = name.replace(".", "_")
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", alias):
        raise LLMUnavailableError("工具名称不符合模型接口要求。", code="AGENT_LLM_REQUEST_REJECTED")
    return alias


def _http_error(status, body=b""):
    if status in {400, 413, 422}:
        # Inspect bounded error data only; never expose upstream text or credentials.
        try:
            error = json.loads(body).get('error', {})
            code = str(error.get('code', '')).lower() if isinstance(error, dict) else ''
            message = str(error.get('message', '')).lower() if isinstance(error, dict) else str(error).lower()
        except (ValueError, AttributeError, UnicodeError):
            code, message = '', ''
        if code in {'context_length_exceeded', 'context_window_exceeded', 'prompt_too_long', 'input_too_long'} or any(
            text in message for text in ('maximum context length', 'context window exceeded', 'prompt is too long', 'input token count exceeds')):
            return LLMUnavailableError('模型接口报告输入上下文过长，正在尝试整理。', code='AGENT_LLM_CONTEXT_OVERFLOW', upstream_status=status)
    if status in {401, 403}:
        code, message = "AUTH_FAILED", "模型接口鉴权失败，请检查 API Key 和模型访问权限。"
    elif status == 429:
        code, message = "RATE_LIMITED", "模型接口触发限流或额度限制，请稍后重试或检查配额。"
    elif status == 404:
        code, message = "ENDPOINT_NOT_FOUND", "模型接口地址或模型不存在，请检查配置。"
    elif 400 <= status < 500:
        code, message = "REQUEST_REJECTED", "模型接口拒绝了请求参数，请检查模型支持的思考等级及协议兼容性。"
    else:
        code, message = "UNAVAILABLE", "模型接口暂时出错，请稍后重试。"
    return LLMUnavailableError(f"{message}（HTTP {status}）", code="AGENT_LLM_"+code, upstream_status=status)


def invalid(message="模型返回结构不符合 Chat Completions 协议。"):
    return LLMUnavailableError(message, code="AGENT_LLM_INVALID_RESPONSE")


@dataclass
class LLMToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = ""


@dataclass
class LLMReply:
    content: str | None = None
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, int | None] = field(default_factory=dict)
    provider_request_id: str | None = None
    attempts: int = 1
    # Protocol context for tool continuations; never part of the public reply.
    reasoning_content: str | None = None


class FixtureLLMClient:
    name = "fixture"
    def __init__(self, replies):
        self._replies = copy.deepcopy(replies)
        self.requests = []

    async def complete(self, *, system, messages, tools, **_kwargs):
        self.requests.append(copy.deepcopy({"system": system, "messages": messages, "tools": tools}))
        if not self._replies:
            raise LLMUnavailableError("fixture 脚本已耗尽。")
        entry = self._replies.pop(0)
        return LLMReply(content=entry.get("content"), tool_calls=[LLMToolCall(str(c["name"]), dict(c.get("arguments") or {}), str(c.get("call_id") or f"fixture-{len(self.requests)}-{i}")) for i,c in enumerate(entry.get("tool_calls", []))], usage=entry.get("usage", {}), reasoning_content=entry.get("reasoning_content"))

    async def aclose(self):
        pass


class HttpLLMClient:
    name = "http"
    def __init__(self, *, base_url, api_key, model, timeout_seconds=60., transport=None, on_retry=None, reasoning_effort='default', session_id=None, context_window_tokens=0):
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.timeout_seconds = float(timeout_seconds)
        self._transport, self._client, self.on_retry = transport, None, on_retry
        self.reasoning_effort, self.session_id = reasoning_effort, session_id
        self.context_key = (self.base_url, self.model)
        endpoint = urlsplit(self.endpoint)
        known_go = (endpoint.scheme == 'https' and endpoint.hostname == 'opencode.ai'
                    and endpoint.path == '/zen/go/v1/chat/completions' and model == 'deepseek-v4.1-flash')
        self.context_window_tokens = context_window_tokens or (1_000_000 if known_go else 32768)

    @property
    def endpoint(self):
        return self.base_url if self.base_url.endswith("/chat/completions") else self.base_url + "/chat/completions"

    async def aclose(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def complete(self, *, system, messages, tools):
        aliases = {_wire_name(spec["name"]): spec["name"] for spec in tools}
        if len(aliases) != len(tools):
            raise LLMUnavailableError("工具名称转换后冲突。", code="AGENT_LLM_REQUEST_REJECTED")
        wire = copy.deepcopy(messages)
        deepseek = 'deepseek' in self.model.lower()
        for item in wire:
            if deepseek and item.get('role') == 'assistant':
                # Old conversations and server-authored stop notices have no model reasoning.
                item.setdefault('reasoning_content', '')
            for call in item.get("tool_calls", []):
                call["function"]["name"] = _wire_name(call["function"]["name"])
        payload = {"model": self.model, "temperature": 0, "messages": [{"role": "system", "content": system}, *wire]}
        if self.reasoning_effort != 'default':
            payload['reasoning_effort'] = self.reasoning_effort
            payload.pop('temperature')  # Reasoning models may reject sampling overrides.
            if deepseek:
                payload['thinking'] = {'type': 'disabled' if self.reasoning_effort == 'none' else 'enabled'}
        headers = {'Authorization': 'Bearer '+self.api_key, 'Content-Type': 'application/json'}
        endpoint = urlsplit(self.endpoint)
        if endpoint.scheme == 'https' and endpoint.hostname == 'opencode.ai' and endpoint.path.startswith('/zen/go/v1/'):
            if not self.session_id:
                raise LLMUnavailableError('OpenCode Go 请求缺少会话标识。', code='AGENT_LLM_REQUEST_REJECTED')
            headers.update({'User-Agent': 'AiFunctions/1.0', 'x-opencode-session': self.session_id})
        if tools:
            payload.update(tools=[{"type": "function", "function": {**spec, "name": _wire_name(spec["name"])}} for spec in tools], tool_choice="auto")
        # With no advertised tools, the harness additionally refuses any unsolicited call.
        if self._client is None:
            self._client = httpx.AsyncClient(transport=self._transport, timeout=self.timeout_seconds, follow_redirects=False)
        try:
            async with asyncio.timeout(self.timeout_seconds):
                for attempt in range(3):
                    retry_after = None
                    try:
                        async with self._client.stream("POST", self.endpoint, json=payload, headers=headers) as response:
                            if response.status_code >= 300:
                                error_body = bytearray()
                                async for chunk in response.aiter_bytes():
                                    remaining = 16384 - len(error_body)
                                    error_body.extend(chunk[:remaining])
                                    if len(error_body) >= 16384:
                                        break
                                error = _http_error(response.status_code, error_body)
                                if (response.status_code != 429 and response.status_code < 500) or attempt == 2:
                                    raise error
                                retry_after = response.headers.get("retry-after")
                            else:
                                body = bytearray()
                                async for chunk in response.aiter_bytes():
                                    if len(body)+len(chunk) > MAX_RESPONSE_BYTES:
                                        raise invalid("模型响应过大，已停止读取并保留进度。")
                                    body.extend(chunk)
                                try:
                                    data = json.loads(body)
                                except (ValueError, UnicodeDecodeError):
                                    raise invalid("模型接口未返回有效 JSON，请检查 Base URL。") from None
                                return _parse(data, aliases, attempt+1)
                    except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
                        if attempt == 2:
                            raise LLMUnavailableError("模型接口响应超时或连接失败，请稍后重试。", code="AGENT_LLM_TIMEOUT") from None
                    if self.on_retry:
                        result = self.on_retry(attempt+1)
                        if hasattr(result, "__await__"):
                            await result
                    delay = .25 * (2**attempt) + random.uniform(0, .1)
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            try:
                                value = parsedate_to_datetime(retry_after)
                                if value.tzinfo is None:
                                    value = value.replace(tzinfo=timezone.utc)
                                delay = max(delay, (value-datetime.now(timezone.utc)).total_seconds())
                            except (ValueError, TypeError, OverflowError):
                                pass
                    await asyncio.sleep(delay)
        except TimeoutError:
            raise LLMUnavailableError("模型接口响应超时，请稍后重试。", code="AGENT_LLM_TIMEOUT") from None
        except httpx.HTTPError:
            raise LLMUnavailableError("无法连接模型接口，请检查网络和 Base URL。") from None


def _parse(data, aliases, attempts):
    try:
        choice = data["choices"][0]
        if choice.get("finish_reason") in {"length", "content_filter", "max_tokens"}:
            raise invalid("模型未完整返回答复，已保留进度。请调整需求后继续。")
        message = choice["message"]
        if not isinstance(message, dict):
            raise ValueError
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError
        if content is not None:
            content.encode("utf-8")
        reasoning = message.get('reasoning_content')
        if reasoning is not None:
            if not isinstance(reasoning, str):
                raise ValueError
            reasoning.encode('utf-8')
        raw_calls = message.get("tool_calls") or []
        if not isinstance(raw_calls, list):
            raise ValueError
        calls, ids = [], set()
        for call in raw_calls:
            call_id, function = call["id"], call["function"]
            name = function["name"]
            if not isinstance(call_id, str) or not call_id or call_id in ids or not isinstance(name, str):
                raise ValueError
            ids.add(call_id)
            raw = function.get("arguments", "{}")
            arguments = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(arguments, dict):
                raise ValueError
            # Reject NaN/Infinity and invalid Unicode before durable JSON checkpoints.
            json.dumps(arguments, ensure_ascii=False, allow_nan=False).encode("utf-8")
            calls.append(LLMToolCall(aliases.get(name, name), arguments, call_id))
        if not calls and not str(content or "").strip():
            raise ValueError
        raw_usage = data.get("usage") or {}
        usage = {key: raw_usage.get(key) if isinstance(raw_usage.get(key), int) and not isinstance(raw_usage.get(key), bool) and raw_usage[key]>=0 else None for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        return LLMReply(content, calls, choice.get("finish_reason"), usage, data.get("id") if isinstance(data.get("id"), str) else None, attempts, reasoning)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError):
        raise invalid() from None


def build_client(settings, *, session_id=None):
    api_key = str(settings.get("api_key") or "").strip()
    if not api_key or settings.get("enabled") is False:
        return None
    return HttpLLMClient(base_url=str(settings.get("base_url") or "https://api.openai.com/v1"), api_key=api_key,
                         model=str(settings.get("model") or "gpt-4o-mini"), timeout_seconds=float(settings.get("timeout_seconds") or 60),
                         reasoning_effort=settings.get('reasoning_effort') or 'default', session_id=session_id,
                         context_window_tokens=settings.get('context_window_tokens') or 0)
