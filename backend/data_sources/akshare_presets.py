"""Initial ordinary AKShare configurations; no credentials or entitlement claims."""
from __future__ import annotations

from .models import (DatasetMapping, DownloadPolicy, IdentityBinding, InterfaceConfig,
                     Pagination, ResponseFormat, SourceConfig, SourceField)
from .presets import direct, fixed


def akshare_source() -> SourceConfig:
    return SourceConfig(
        id="akshare", name="AKShare", transport="akshare", base_url="", enabled=True,
        policy=DownloadPolicy(requests_per_minute=20, min_interval_seconds=3,
                              max_rows_per_request=20000, max_concurrency=1,
                              max_attempts=2, read_timeout_seconds=30,
                              max_runtime_seconds=3600),
        notes="开源接口库；基金数据来自东方财富。仅供符合上游条款的研究使用。当前限制是本地保守设置，不是官方无限配额。",
    )


def akshare_interfaces() -> tuple[InterfaceConfig, ...]:
    policy = akshare_source().policy
    quote_fields = ["symbol", "日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅", "_adjustment_basis"]
    quote = InterfaceConfig(
        id="akshare.etf_daily", source_id="akshare", name="ETF 日行情", enabled=True,
        api_name="fund_etf_hist_em", params={"symbol": "510300", "period": "daily", "adjust": ""},
        response=ResponseFormat(), policy=policy, pagination=Pagination(), incremental_field="日期",
        source_fields=[SourceField(name=n, data_type="date" if n == "日期" else "string" if n in {"symbol", "_adjustment_basis"} else "number") for n in quote_fields],
        mappings=[DatasetMapping(target_table="market.quote_daily",
            identities=[IdentityBinding(target_field="instrument_id", source_field="symbol", namespace="TS_CODE", key_transform="cn_etf_code")],
            fields=[direct("trade_date", "日期", "date"), fixed("availability_status", "UNKNOWN"),
                    fixed("quote_type", "TRADE_PRICE"), direct("adjustment_basis", "_adjustment_basis"),
                    fixed("currency", "CNY"), direct("open", "开盘"), direct("close", "收盘"),
                    direct("high", "最高"), direct("low", "最低"),
                    direct("volume", "成交量", "scale", factor=100), direct("turnover_amount", "成交额"),
                    direct("return_decimal", "涨跌幅", "scale", factor=0.01)])],
        notes="东方财富日 K；成交量从手转为份，成交额为元。收盘后获取完整日线；不把前/后复权价格与原始价格比较。未提供可靠历史发布时间。",
    )
    nav = InterfaceConfig(
        id="akshare.fund_nav", source_id="akshare", name="公募基金单位净值", enabled=True,
        api_name="fund_open_fund_info_em", params={"symbol": "000001", "indicator": "单位净值走势"},
        response=ResponseFormat(), policy=policy, pagination=Pagination(), incremental_field="净值日期",
        source_fields=[SourceField(name="symbol"), SourceField(name="净值日期", data_type="date"),
                       SourceField(name="单位净值", data_type="number"), SourceField(name="日增长率", data_type="number", unit="percent")],
        mappings=[DatasetMapping(target_table="market.nav_daily",
            identities=[IdentityBinding(target_field="instrument_id", source_field="symbol", namespace="TS_CODE", key_transform="cn_fund_code")],
            fields=[direct("valuation_date", "净值日期", "date"), direct("unit_nav", "单位净值"),
                    fixed("availability_status", "UNKNOWN"), fixed("currency", "CNY"),
                    fixed("adjustment_method", "NONE"), fixed("is_retrospective_adjustment", False)])],
        notes="场外基金没有交易所日 K；此接口导入披露的单位净值。SDK 返回单基金历史后在本地按请求日期截取。缺少公告日和复权净值，不推测或代填。",
    )
    return quote, nav
