"""Domain-specific mappings that cannot be reduced to renamed OHLC columns."""
from __future__ import annotations
from .models import DatasetMapping, FieldMapping, IdentityBinding
from .presets import direct, fixed, identity
try:
    from backend.data_model.catalog import TABLES_BY_ID
except ModuleNotFoundError:
    from data_model.catalog import TABLES_BY_ID

MACRO_FIELDS = {
    "cn_gdp": ("quarter", "gdp", "CNY_100million"),
    "cn_cpi": ("month", "nt_val", "index_point"),
    "cn_ppi": ("month", "ppi_yoy", "percent"),
    "cn_pmi": ("month", "pmi010000", "index_point"),
    "cn_m": ("month", "m2", "CNY_100million"),
    "sf_month": ("month", "inc_month", "CNY_100million"),
    "shibor": ("date", "on", "percent"),
    "shibor_lpr": ("date", "1y", "percent"),
}


def extra_columns(api: str) -> list[str]:
    from .tushare_facts import EXTRA_COLUMNS
    if api in EXTRA_COLUMNS:
        return EXTRA_COLUMNS[api].split()
    special = {
        "trade_cal": ["exchange", "cal_date", "is_open", "pretrade_date"],
        "index_weight": ["index_code", "con_code", "trade_date", "weight"],
        "index_member_all": ["l1_code", "l2_code", "l3_code", "ts_code", "name", "in_date", "out_date", "is_new"],
        "index_classify": ["index_code", "industry_name", "level", "industry_code", "parent_code", "src"],
        "index_dailybasic": ["ts_code", "trade_date", "pe", "pe_ttm", "pb", "turnover_rate", "total_mv", "float_mv"],
        "cn_schedule": ["publish_date", "title", "data_api"],
        "repo_daily": ["ts_code", "trade_date", "close", "open", "high", "low", "vol", "amount"],
    }
    if api in special:
        return special[api]
    if api in {"ths_index", "dc_index", "tdx_index"}:
        return ["ts_code", "name", "trade_date"]
    return ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg"]


def observation(target: str, source: str):
    name = next(f.name for f in TABLES_BY_ID[target].fields if f.role == "observation_time")
    return direct(name, source, "date")


def extra_mappings(api: str, columns: list[str]) -> list[DatasetMapping]:
    if api in MACRO_FIELDS:
        period, value, source_unit = MACRO_FIELDS[api]
        factor = 0.01 if source_unit == "percent" else 100000000 if source_unit == "CNY_100million" else 1
        unit = "decimal" if source_unit == "percent" else "CNY" if source_unit == "CNY_100million" else source_unit
        binding = IdentityBinding(target_field="series_id", constant=api + "." + value, namespace="TUSHARE_MACRO_SERIES")
        return [DatasetMapping(target_table="macro.observation", identities=[binding], fields=[
            direct("observation_period_end", period, "period_end"), direct("value", value, "scale", factor=factor), fixed("unit", unit),
            fixed("availability_status", "UNKNOWN"), fixed("is_final", False), fixed("release_status", "UNKNOWN")])]
    if api == "fund_portfolio":
        target = "fund.holding_disclosure"
        return [DatasetMapping(target_table=target, identities=[identity("fund_instrument_id"), identity("holding_key", "symbol"), identity("holding_instrument_id", "symbol")], fields=[
            observation(target, "end_date"), direct("available_at", "ann_date", "timestamp"), direct("announced_at", "ann_date", "timestamp"),
            direct("holding_external_code", "symbol"), fixed("availability_status", "DATE_ONLY"), fixed("holding_type", "STOCK"),
            direct("quantity", "amount"), direct("market_value", "mkv"), direct("floating_share_weight", "stk_float_ratio", "scale", factor=0.01), fixed("coverage_scope", "PARTIAL"), fixed("currency", "CNY")])]
    if api == "fund_div":
        target = "fund.dividend_event"
        event = IdentityBinding(target_field="dividend_event_id", key_fields=["ts_code", "ann_date", "ex_date"], namespace="TS_FUND_DIVIDEND")
        return [DatasetMapping(target_table=target, identities=[identity(), event], fields=[
            observation(target, "record_date"), direct("available_at", "ann_date", "timestamp"), direct("announced_at", "ann_date", "timestamp"),
            direct("ex_date", operation="date"), direct("pay_date", operation="date"), direct("cash_per_unit", "div_cash"), fixed("base_unit", 1),
            fixed("currency", "CNY"), fixed("availability_status", "DATE_ONLY"), fixed("event_status", "UNKNOWN")])]
    if api == "index_weight":
        return [DatasetMapping(target_table="index.weight", identities=[identity("index_instrument_id", "index_code"), identity("member_key", "con_code"), identity("member_instrument_id", "con_code")], fields=[
            observation("index.weight", "trade_date"), direct("member_external_code", "con_code"), direct("weight", operation="scale", factor=0.01), fixed("availability_status", "UNKNOWN")])]
    if api == "index_dailybasic":
        return [DatasetMapping(target_table="index.valuation_daily", identities=[identity("index_instrument_id")], fields=[
            observation("index.valuation_daily", "trade_date"), *[direct(name) for name in ("pe", "pe_ttm", "pb")],
            direct("turnover_rate", operation="scale", factor=0.01), direct("total_market_value", "total_mv"), direct("float_market_value", "float_mv"), fixed("availability_status", "UNKNOWN"), fixed("currency", "CNY")])]
    if api == "fund_adj":
        return [DatasetMapping(target_table="fund.adjustment_factor", identities=[identity()], fields=[
            direct("factor_date", "trade_date", "date"), direct("factor", "adj_factor"),
            fixed("adjustment_basis", "SOURCE"), fixed("availability_status", "UNKNOWN")])]
    if api == "fund_manager":
        person = IdentityBinding(target_field="person_id", source_field="name", namespace="TS_MANAGER_NAME", resolution="lookup")
        fund = IdentityBinding(target_field="fund_product_id", source_field="ts_code", namespace="TS_SHARE_TO_FUND", resolution="lookup")
        tenure = IdentityBinding(target_field="tenure_id", key_fields=["ts_code", "name", "begin_date"], namespace="TS_MANAGER_TENURE")
        return [DatasetMapping(target_table="master.fund_manager_tenure", identities=[person, fund, tenure], fields=[
            direct("begin_date", operation="date"), direct("end_date", operation="date"),
            direct("announced_at", "ann_date", "timestamp"), direct("available_at", "ann_date", "timestamp"),
            fixed("role_code", "OTHER"), fixed("is_primary", False)])]
    if api == "index_classify":
        return [DatasetMapping(target_table="master.classification_node", identities=[
            identity("scheme_id", "src", "TS_CLASSIFICATION"), identity("node_id", "index_code", "TS_CLASSIFICATION_NODE")], fields=[
            direct("scheme_version", "src"), direct("code", "industry_code"), direct("name", "industry_name"),
            direct("level", operation="enum", enum_map={"L1": 1, "L2": 2, "L3": 3, "1": 1, "2": 2, "3": 3}), fixed("sort_order", 0)])]
    if api == "index_member_all":
        return [DatasetMapping(target_table="index.membership", identities=[
            identity("index_instrument_id", "l3_code", "TS_SW_INDEX"), identity("member_key"), identity("member_instrument_id")], fields=[
            direct("member_external_code", "ts_code"), direct("member_name", "name"),
            direct("effective_from", "in_date", "date"), direct("effective_to", "out_date", "date"),
            direct("is_current", "is_new", "enum", enum_map={"Y": True, "N": False}), fixed("availability_status", "UNKNOWN")])]
    if api == "repo_daily":
        return [DatasetMapping(target_table="macro.observation", identities=[identity("series_id", "ts_code", "TS_REPO_RATE")], fields=[
            direct("observation_period_end", "trade_date", "date"), direct("value", "close", "scale", factor=0.01),
            fixed("unit", "decimal"), fixed("availability_status", "UNKNOWN"), fixed("is_final", False), fixed("release_status", "UNKNOWN")])]
    if api == "cn_schedule":
        return [DatasetMapping(target_table="macro.release_event", identities=[
            identity("series_id", "data_api", "TS_MACRO_RELEASE_API"),
            IdentityBinding(target_field="release_event_id", key_fields=["publish_date", "title", "data_api"], namespace="TS_MACRO_RELEASE")], fields=[
            direct("scheduled_release_at", "publish_date", "timestamp"), direct("title"),
            fixed("availability_status", "UNKNOWN"), fixed("release_status", "SCHEDULED")])]
    return []
