"""Bounded, isolated SDK adapter; no user-supplied Python is executed."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .models import CenterError, InterfaceConfig, SourceConfig
from .store import SourceStore
from .transport import TransientSourceError

SDK_APIS = {"fund_etf_hist_em", "fund_open_fund_info_em"}


def validate_sdk_config(interface: InterfaceConfig) -> None:
    if interface.api_name not in SDK_APIS:
        raise CenterError("SDK_API_UNSUPPORTED", "当前 SDK 适配器支持 ETF 历史行情和公募基金单位净值接口。")
    if interface.path or interface.headers or interface.pagination.mode != "none":
        raise CenterError("SDK_CONFIG_INVALID", "SDK 接口由函数处理网络协议；不填写 HTTP 路径、请求头或分页。")
    validate_sdk_params(interface.api_name, interface.params, require_symbol=False)


def validate_sdk_params(api: str, params: dict[str, Any], *, require_symbol: bool = True) -> dict[str, Any]:
    if api not in SDK_APIS:
        raise CenterError("SDK_API_UNSUPPORTED", "SDK 接口未登记。")
    allowed = {"symbol", "start_date", "end_date", "period", "adjust"} if api == "fund_etf_hist_em" else {"symbol", "indicator", "period", "start_date", "end_date"}
    if set(params) - allowed:
        raise CenterError("SDK_PARAMS_INVALID", "SDK 参数包含未支持的名称。")
    symbol = params.get("symbol", "")
    if (require_symbol or symbol) and (not isinstance(symbol, str) or not re.fullmatch(r"\d{6}", symbol)):
        raise CenterError("SDK_SYMBOL_REQUIRED", "请提供六位产品代码，保留前导零。")
    for field in ("start_date", "end_date"):
        value = params.get(field)
        if value:
            from datetime import datetime
            try:
                if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
                    raise ValueError()
                datetime.strptime(value, "%Y%m%d")
            except ValueError:
                raise CenterError("SDK_DATE_INVALID", "日期使用有效的 YYYYMMDD 格式。") from None
    if params.get("start_date") and params.get("end_date") and params["start_date"] > params["end_date"]:
        raise CenterError("SDK_DATE_INVALID", "开始日期不能晚于结束日期。")
    if api == "fund_etf_hist_em":
        if params.get("period", "daily") != "daily" or params.get("adjust", "") not in {"", "qfq", "hfq"}:
            raise CenterError("SDK_SEMANTICS_INVALID", "日行情只允许 daily 周期；adjust 可为不复权、qfq、hfq。")
    elif params.get("indicator", "单位净值走势") != "单位净值走势":
        raise CenterError("SDK_SEMANTICS_INVALID", "此接入合同为单位净值，不允许混入累计收益率或其他口径。")
    return params


def fetch_sdk(store: SourceStore, source: SourceConfig, interface: InterfaceConfig, params: dict[str, Any], *, sample: bool = False) -> list[dict[str, Any]]:
    validate_sdk_config(interface)
    validate_sdk_params(interface.api_name, params)
    timeout = min(source.policy.max_runtime_seconds, interface.policy.max_runtime_seconds, source.policy.read_timeout_seconds + source.policy.connect_timeout_seconds + 20)
    payload = {"root": str(store.root.resolve()), "source": source.model_dump(mode="json"),
               "interface": interface.model_dump(mode="json"), "params": params,
               "max_http_requests": 1 if sample else min(interface.pagination.max_pages, 20)}
    try:
        result = subprocess.run([sys.executable, "-m", "backend.data_sources.akshare_worker"],
            input=json.dumps(payload), text=True, capture_output=True,
            cwd=Path(__file__).resolve().parents[2], timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise TransientSourceError("SOURCE_TIMEOUT", "SDK 请求超时，隔离进程已终止。", 504) from None
    try:
        body = json.loads(result.stdout)
    except ValueError:
        raise CenterError("SDK_PROCESS_FAILED", "SDK 进程未返回有效结果，请检查依赖版本。", 502) from None
    if not body.get("ok"):
        error = TransientSourceError if body.get("retryable") else CenterError
        raise error(body.get("code", "SDK_FAILED"), body.get("message", "SDK 数据获取失败。"), body.get("status", 502))
    return body["rows"]
