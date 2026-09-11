"""Disposable AKShare process: constrain every SDK HTTP request, not just calls."""
from __future__ import annotations
import contextlib
import io
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

from .akshare_adapter import validate_sdk_params
from .models import CenterError, InterfaceConfig, SourceConfig
from .quota import SharedQuota
from .runtime import effective_policy
from .store import SourceStore
from .transport import TransientSourceError, request


def execute(payload: dict) -> dict:
    import pandas as pd
    import requests
    try:
        import akshare as ak
    except ImportError:
        raise CenterError("SDK_NOT_INSTALLED", "服务环境未安装 AKShare，请安装 backend/requirements-akshare.txt。", 503) from None
    source = SourceConfig.model_validate(payload["source"])
    interface = InterfaceConfig.model_validate(payload["interface"])
    params = validate_sdk_params(interface.api_name, payload["params"])
    store = SourceStore(Path(payload["root"]))
    policy = effective_policy(source.policy, interface.policy)
    quota, count = SharedQuota(store), 0
    original = requests.sessions.Session.request

    def bounded_request(self, method, url, **kwargs):
        nonlocal count
        host = (urlsplit(url).hostname or "").lower()
        if not (host == "eastmoney.com" or host.endswith(".eastmoney.com")):
            raise CenterError("SDK_HOST_FORBIDDEN", "SDK 请求超出已登记的上游主机范围。")
        if method.upper() != "GET":
            raise CenterError("SDK_METHOD_UNSUPPORTED", "SDK 出现未登记的请求协议。")
        if count >= payload["max_http_requests"]:
            raise CenterError("SDK_REQUEST_CAP", "SDK 需要更多网络请求，已在请求上限处停止；未标记下载成功。")
        headers = {k: v for k, v in dict(kwargs.get("headers") or {}).items() if k.lower() not in {"accept-encoding", "host", "content-length", "connection"}}
        count += 1
        with quota.acquire(source.id, interface.api_name, source.policy, interface.policy, policy.max_rows_per_request):
            text = request(url, "GET", dict(kwargs.get("params") or {}), headers, policy)
        response = requests.Response()
        response.status_code, response.encoding, response.url = 200, "utf-8", url
        response._content = text.encode("utf-8")
        return response

    requests.sessions.Session.request = bounded_request
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if interface.api_name == "fund_etf_hist_em":
                frame = ak.fund_etf_hist_em(**params)
                date_field = "日期"
            else:
                sdk_params = {k: v for k, v in params.items() if k not in {"start_date", "end_date"}}
                frame = ak.fund_open_fund_info_em(**sdk_params)
                date_field = "净值日期"
        if frame.empty:
            return {"ok": True, "rows": [], "http_requests": count}
        if len(frame) >= policy.max_rows_per_request:
            raise CenterError("SOURCE_ROW_CAP", "SDK 结果达到单次行数上限，请缩小区间或调整已核实上限。")
        if date_field not in frame:
            raise CenterError("SDK_SCHEMA_CHANGED", "SDK 返回字段已变化，需要检查接口映射。")
        dates = pd.to_datetime(frame[date_field], errors="raise")
        mask = pd.Series(True, index=frame.index)
        if params.get("start_date"):
            mask &= dates >= pd.Timestamp(params["start_date"])
        if params.get("end_date"):
            mask &= dates <= pd.Timestamp(params["end_date"])
        frame = frame.loc[mask].copy()
        frame[date_field] = dates.loc[mask].dt.strftime("%Y-%m-%d")
        rows = json.loads(frame.to_json(orient="records", date_format="iso", force_ascii=False))
        basis = {"": "RAW", "qfq": "FORWARD", "hfq": "BACKWARD"}[params.get("adjust", "")]
        for row in rows:
            row["symbol"] = params["symbol"]
            if interface.api_name == "fund_etf_hist_em":
                row["_adjustment_basis"] = basis
        if len(json.dumps(rows).encode()) > policy.max_response_bytes:
            raise CenterError("SOURCE_RESPONSE_TOO_LARGE", "SDK 转换结果超过字节上限。")
        return {"ok": True, "rows": rows, "http_requests": count}
    finally:
        requests.sessions.Session.request = original


def main() -> None:
    try:
        result = execute(json.load(sys.stdin))
    except CenterError as exc:
        result = {"ok": False, "code": exc.code, "message": exc.message, "status": exc.status,
                  "retryable": isinstance(exc, TransientSourceError)}
    except Exception:
        result = {"ok": False, "code": "SDK_FAILED", "message": "SDK 或上游结构异常；请检查 AKShare 版本及来源可用性。", "status": 502}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
