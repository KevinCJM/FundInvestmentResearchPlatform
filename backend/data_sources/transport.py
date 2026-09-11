"""Bounded HTTPS transport with DNS pinning and no redirects or proxy use."""
from __future__ import annotations
import http.client
import ipaddress
import json
import socket
import ssl
import time
from typing import Any
from urllib.parse import urlencode, urlsplit

from .models import CenterError, DownloadPolicy


class TransientSourceError(CenterError):
    retry_after_seconds: float = 0.0


def public_address(host: str, port: int) -> str:
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise TransientSourceError("SOURCE_DNS", "无法解析数据源地址。", 502) from exc
    resolved = {item[4][0] for item in addresses}
    if not resolved or any(not ipaddress.ip_address(value).is_global for value in resolved):
        raise CenterError("SOURCE_ADDRESS_FORBIDDEN", "禁止访问私网、回环、保留或链路本地地址。")
    return sorted(resolved)[0]


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self) -> None:
        connection = socket.create_connection((self.address, self.port), self.timeout)
        try:
            self.sock = self._context.wrap_socket(connection, server_hostname=self.host)
        except Exception:
            connection.close()
            raise


def request(url: str, method: str, params: dict[str, Any], headers: dict[str, str], policy: DownloadPolicy) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
        raise CenterError("SOURCE_URL_INVALID", "只允许不含凭据的 HTTPS 数据源。")
    port = parsed.port or 443
    address = public_address(parsed.hostname, port)
    connection = PinnedHTTPSConnection(parsed.hostname, port, address, policy.connect_timeout_seconds)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    data = None
    outgoing = {"Accept-Encoding": "identity", **headers}
    if method == "GET":
        if params:
            path += ("&" if "?" in path else "?") + urlencode(params, doseq=True)
    else:
        outgoing["Content-Type"] = "application/json"
        data = json.dumps(params, ensure_ascii=False, allow_nan=False).encode()
    deadline = time.monotonic() + policy.read_timeout_seconds
    try:
        connection.connect()
        connection.sock.settimeout(policy.read_timeout_seconds)
        connection.request(method, path, body=data, headers=outgoing)
        response = connection.getresponse()
        if response.status == 429 or 500 <= response.status < 600:
            error = TransientSourceError("SOURCE_RETRYABLE" if response.status == 429 else "SOURCE_TRANSIENT", "数据源限流或暂时不可用。", 502)
            retry_after = response.getheader("Retry-After", "")
            try:
                if retry_after.isdigit():
                    error.retry_after_seconds = float(retry_after)
                elif retry_after:
                    from email.utils import parsedate_to_datetime
                    error.retry_after_seconds = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
            raise error
        if not 200 <= response.status < 300:
            raise CenterError("SOURCE_REJECTED", "数据源拒绝请求；请检查地址、权限或参数。", 502)
        if response.getheader("Content-Encoding", "identity").lower() != "identity":
            raise CenterError("ENCODING_UNSUPPORTED", "暂不接受压缩响应，请使用未压缩 JSON/CSV。")
        length = response.getheader("Content-Length")
        if length and (not length.isdigit() or int(length) > policy.max_response_bytes):
            raise CenterError("SOURCE_RESPONSE_TOO_LARGE", "响应超过配置的字节上限。")
        chunks, received = [], 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TransientSourceError("SOURCE_TIMEOUT", "读取数据源超时。", 504)
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            block = response.read(min(65536, policy.max_response_bytes + 1 - received))
            if not block:
                break
            chunks.append(block)
            received += len(block)
            if received > policy.max_response_bytes:
                raise CenterError("SOURCE_RESPONSE_TOO_LARGE", "响应超过配置的字节上限。")
        return b"".join(chunks).decode("utf-8-sig")
    except ssl.SSLCertVerificationError as exc:
        raise CenterError("SOURCE_CERTIFICATE_INVALID", "数据源 TLS 证书验证失败。", 502) from exc
    except (socket.timeout, ConnectionError, OSError, http.client.HTTPException) as exc:
        raise TransientSourceError("SOURCE_CONNECTION", "数据源连接失败或超时。", 502) from exc
    finally:
        connection.close()
