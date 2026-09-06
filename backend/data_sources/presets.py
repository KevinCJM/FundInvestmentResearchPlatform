"""Tushare defaults tied to the existing downloader's declared API contracts."""
from __future__ import annotations
import ast
from functools import lru_cache
from pathlib import Path

from .models import DatasetMapping, DownloadPolicy, FieldMapping, IdentityBinding, InterfaceConfig, Pagination, ResponseFormat, SourceConfig, SourceField
from .tushare_facts import LOCAL_CAPS, NAMESPACES, QUOTE_UNITS, source_field_type

ROOT = Path(__file__).resolve().parents[2]
# API -> (label, declared fields constant, official document id).
API_SPECS = {
    "fund_basic": ("基金基础信息", "FUND_BASIC_FIELDS", 19),
    "etf_basic": ("ETF 基础信息", "ETF_BASIC_FIELDS", 385),
    "fund_nav": ("基金净值", "FUND_NAV_FIELDS", 119),
    "fund_daily": ("ETF 日行情", "FUND_DAILY_FIELDS", 127),
    "etf_share_size": ("ETF 份额规模", "ETF_SHARE_SIZE_FIELDS", 408),
    "fund_company": ("基金公司", "FUND_COMPANY_FIELDS", 118),
    "fund_manager": ("基金经理", "FUND_MANAGER_FIELDS", 208),
    "fund_portfolio": ("基金股票持仓披露", "FUND_PORTFOLIO_FIELDS", 121),
    "fund_div": ("基金分红", "FUND_DIVIDEND_FIELDS", 120),
    "fund_adj": ("基金复权因子", "FUND_ADJUSTMENT_FIELDS", 199),
    "mkt_idx_bmk": ("业绩基准目录", "FUND_BENCHMARK_FIELDS", 462),
    "stock_basic": ("股票基本信息", "STOCK_BASIC_FIELDS", 25),
    "trade_cal": ("交易日历", "", 26),
    "index_basic": ("指数基础信息", "INDEX_BASIC_FIELDS", 94),
    "etf_index": ("ETF 跟踪指数目录", "ETF_INDEX_FIELDS", 386),
    "index_classify": ("行业分类", "", 181),
    "index_member_all": ("行业成分", "", 335),
    "index_weight": ("指数权重", "", 96),
    "ths_index": ("同花顺指数目录", "", 259),
    "dc_index": ("东财指数目录", "", 362),
    "tdx_index": ("通达信指数目录", "", 376),
    "index_daily": ("境内指数行情", "", 95),
    "sw_daily": ("申万指数行情", "", 327),
    "ci_daily": ("中信指数行情", "", 308),
    "ths_daily": ("同花顺指数行情", "", 260),
    "dc_daily": ("东财指数行情", "", 382),
    "tdx_daily": ("通达信指数行情", "", 378),
    "index_global": ("国际指数行情", "", 211),
    "fut_index_daily": ("期货指数行情", "", 468),
    "index_dailybasic": ("指数估值", "", 128),
    "cn_gdp": ("GDP", "", 227), "cn_cpi": ("CPI", "", 228),
    "cn_ppi": ("PPI", "", 245), "cn_pmi": ("PMI", "", 325),
    "cn_m": ("货币供应量", "", 242), "sf_month": ("社会融资", "", 310),
    "shibor": ("Shibor", "", 149), "shibor_lpr": ("LPR", "", 151),
    "repo_daily": ("回购行情", "", 256), "cn_schedule": ("宏观发布日历", "", 461),
}


@lru_cache(maxsize=1)
def legacy_declarations() -> dict:
    result = {}
    for node in ast.parse((ROOT / "T01_get_data.py").read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "API_ROW_LIMITS" or name.endswith("_FIELDS"):
                try:
                    result[name] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    return result


def default_source() -> SourceConfig:
    return SourceConfig(id="tushare", name="Tushare", transport="tushare", base_url="https://api.tushare.pro", policy=DownloadPolicy(requests_per_minute=450, min_interval_seconds=0.13, max_rows_per_request=15000, max_concurrency=16, max_runtime_seconds=86400), notes="复用现有下载页的本机 Token。配额为共享安全上限，不代表账户权限；预置接口不扩展原下载范围。")


def direct(target: str, source: str | None = None, operation: str = "copy", **kwargs) -> FieldMapping:
    return FieldMapping(target_field=target, source_field=source or target, operation=operation, **kwargs)


def fixed(target: str, value) -> FieldMapping:
    return FieldMapping(target_field=target, operation="constant", constant=value)


def identity(target: str = "instrument_id", source: str = "ts_code", namespace: str = "TS_CODE") -> IdentityBinding:
    return IdentityBinding(target_field=target, source_field=source, namespace=namespace)


def default_mappings(api: str, columns: list[str]) -> list[DatasetMapping]:
    captured = FieldMapping(target_field="valid_from", operation="capture_date")
    if api == "fund_nav":
        return [DatasetMapping(target_table="market.nav_daily", identities=[identity()], fields=[
            direct("valuation_date", "nav_date", "date"), direct("available_at", "ann_date", "timestamp"),
            direct("announced_at", "ann_date", "timestamp"), fixed("availability_status", "DATE_ONLY"),
            direct("unit_nav"), direct("accumulated_nav", "accum_nav"), direct("adjusted_nav", "adj_nav"),
            direct("accumulated_dividend", "accum_div"), direct("net_assets", "net_asset"),
            fixed("currency", "CNY"), fixed("adjustment_method", "SOURCE"), fixed("is_retrospective_adjustment", True),
        ])]
    quote_apis = {"fund_daily", "index_daily", "sw_daily", "ci_daily", "ths_daily", "dc_daily", "tdx_daily", "index_global", "fut_index_daily"}
    if api in quote_apis:
        fields = [direct("trade_date", operation="date"), fixed("availability_status", "UNKNOWN"),
                  fixed("quote_type", "TRADE_PRICE" if api == "fund_daily" else "INDEX_LEVEL"),
                  fixed("adjustment_basis", "RAW" if api == "fund_daily" else "NOT_APPLICABLE")]
        fields.extend(direct(name) for name in ("open", "high", "low", "close", "change") if name in columns)
        if "pre_close" in columns:
            fields.append(direct("previous_close", "pre_close"))
        change_field = next((name for name in ("pct_chg", "pct_change") if name in columns), None)
        if change_field:
            fields.append(direct("return_decimal", change_field, "scale", factor=0.01))
        if api in QUOTE_UNITS:
            volume_factor, amount_factor = QUOTE_UNITS[api]
            if "vol" in columns:
                fields.append(direct("volume", "vol", "scale", factor=volume_factor))
            if amount_factor is not None and "amount" in columns:
                fields.append(direct("turnover_amount", "amount", "scale", factor=amount_factor))
            fields.append(fixed("currency", "CNY"))
        return [DatasetMapping(target_table="market.quote_daily", fields=fields, identities=[identity(namespace=NAMESPACES.get(api, "TS_CODE"))])]
    masters = {"fund_basic": ("name", "FUND_SHARE"), "etf_basic": ("csname", "ETF"), "stock_basic": ("name", "STOCK"), "index_basic": ("name", "INDEX"), "etf_index": ("indx_name", "INDEX"), "ths_index": ("name", "INDEX"), "dc_index": ("name", "INDEX"), "tdx_index": ("name", "INDEX"), "mkt_idx_bmk": ("name", "INDEX")}
    if api in masters:
        name, kind = masters[api]
        fields = [direct("canonical_name", name), direct("ticker", "ts_code"), fixed("instrument_type", kind),
                  fixed("currency", "CNY"), fixed("status_code", "UNKNOWN"), fixed("tradable", kind != "INDEX"), captured]
        if "list_date" in columns:
            fields.append(direct("listing_date", "list_date", "date"))
        if "found_date" in columns:
            fields.append(direct("inception_date", "found_date", "date"))
        if "setup_date" in columns:
            fields.append(direct("inception_date", "setup_date", "date"))
        status_field = "list_status" if "list_status" in columns else "status" if "status" in columns else None
        if status_field:
            fields = [item for item in fields if item.target_field != "status_code"]
            fields.append(direct("status_code", status_field, "enum", enum_map={"L": "ACTIVE", "I": "PENDING", "P": "SUSPENDED" if api == "stock_basic" else "PENDING", "G": "PENDING", "D": "DELISTED"}))
        return [DatasetMapping(target_table="master.instrument", fields=fields, identities=[identity(namespace=NAMESPACES.get(api, "TS_CODE"))])]
    if api == "fund_company":
        return [DatasetMapping(target_table="master.organization", identities=[identity("organization_id", "org_code", "TS_ORG_CODE")], fields=[
            direct("legal_name", "name"), direct("short_name", "shortname"), direct("registration_number", "credit_code"),
            *[direct(name) for name in ("province", "city", "website")], fixed("organization_type", "FUND_MANAGER"), fixed("status_code", "UNKNOWN"), captured])]
    if api == "etf_share_size":
        return [DatasetMapping(target_table="market.fund_scale", identities=[identity()], fields=[
            direct("observation_date", "trade_date", "date"), fixed("availability_status", "UNKNOWN"),
            direct("total_shares", "total_share", "scale", factor=10000), direct("net_assets", "total_size", "scale", factor=10000), fixed("currency", "CNY")])]
    if api == "trade_cal":
        return [DatasetMapping(target_table="master.trading_calendar", fields=[
            direct("calendar_code", "exchange"), direct("calendar_date", "cal_date", "date"),
            direct("is_open", operation="enum", enum_map={"0": False, "1": True}),
            direct("previous_open_date", "pretrade_date", "date")])]
    # Additional endpoint mappings are declared in a separate, domain-sized file.
    from .preset_extensions import extra_mappings
    return extra_mappings(api, columns)


@lru_cache(maxsize=1)
def default_interfaces() -> tuple[InterfaceConfig, ...]:
    declarations = legacy_declarations()
    caps = declarations.get("API_ROW_LIMITS", {})
    items = []
    for api, (label, constant, doc) in API_SPECS.items():
        columns = list(declarations.get(constant, []))
        if not columns:
            from .preset_extensions import extra_columns
            columns = extra_columns(api)
        cap = int(LOCAL_CAPS.get(api, caps.get(api, 5000)))
        policy = DownloadPolicy(requests_per_minute=45 if api == "stock_basic" else 240, min_interval_seconds=1.34 if api == "stock_basic" else 0.25, max_rows_per_request=cap, max_concurrency=16, max_runtime_seconds=86400)
        items.append(InterfaceConfig(
            id="tushare." + api, source_id="tushare", name=label, enabled=True,
            api_name=api, method="POST", response=ResponseFormat(format="json_columns", records_path="data.items", columns_path="data.fields"),
            source_fields=[SourceField(name=name, data_type=source_field_type(api, name)) for name in columns],
            policy=policy, pagination=Pagination(mode="offset" if api in {"fund_basic", "fund_nav", "fund_manager"} else "none", page_size={"fund_basic": 15000, "fund_nav": 10000, "fund_manager": 5000}.get(api, min(cap, 1000))),
            incremental_field=next((name for name in ("trade_date", "nav_date", "date", "month", "quarter") if name in columns), None),
            mappings=default_mappings(api, columns),
            notes=f"官方文档 doc_id={doc}；配额为本地安全限制，并非供应商授权承诺；部分无文档行数上限的接口使用本地防截断阈值。预置不代表独立权限已开通。未映射字段完整保留于原始批次；身份或必填缺失时拒绝对应候选表，不伪造补值。标准候选尚不替代旧研究数据。",
        ))
    return tuple(items)
