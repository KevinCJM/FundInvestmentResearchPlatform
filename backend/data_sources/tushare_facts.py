"""Reviewed wire fields; no dependency on a developer's home-directory catalog.

Grounding: tushare-fetcher interface catalog (2026-06-01), reviewed 2026-09-05;
fund_nav/fund_daily/etf_share_size reconfirmed against official docs 119/127/408.
These are selected supported fields, not a claim that every vendor field is mapped.
"""

EXTRA_COLUMNS = {
    "trade_cal": "exchange cal_date is_open pretrade_date",
    "index_weight": "index_code con_code trade_date weight",
    "index_member_all": "l1_code l1_name l2_code l2_name l3_code l3_name ts_code name in_date out_date is_new",
    "index_classify": "index_code industry_name parent_code level industry_code is_pub src",
    "index_dailybasic": "ts_code trade_date total_mv float_mv total_share float_share free_share turnover_rate turnover_rate_f pe pe_ttm pb",
    "cn_schedule": "month publish_date title issuing_org data_api",
    "repo_daily": "ts_code trade_date repo_maturity pre_close open high low close weight weight_r amount num",
    "ths_index": "ts_code name count exchange list_date type",
    "dc_index": "ts_code trade_date name leading leading_code pct_change leading_pct total_mv turnover_rate up_num down_num idx_type level",
    "tdx_index": "ts_code trade_date name idx_type idx_count total_share float_share total_mv float_mv",
    "index_daily": "ts_code trade_date close open high low pre_close change pct_chg vol amount",
    "sw_daily": "ts_code trade_date name open low high close change pct_change vol amount pe pb float_mv total_mv",
    "ci_daily": "ts_code trade_date open low high close pre_close change pct_change vol amount",
    "ths_daily": "ts_code trade_date close open high low pre_close avg_price change pct_change vol turnover_rate total_mv float_mv",
    "dc_daily": "ts_code trade_date close open high low change pct_change vol amount swing turnover_rate",
    "tdx_daily": "ts_code trade_date close open high low pre_close change pct_change vol amount",
    "index_global": "ts_code trade_date open close high low pre_close change pct_chg swing vol amount",
    "fut_index_daily": "ts_code trade_date close open high low pre_close change pct_chg vol amount",
    "cn_gdp": "quarter gdp gdp_yoy pi pi_yoy si si_yoy ti ti_yoy",
    "cn_cpi": "month nt_val nt_yoy nt_mom nt_accu town_val town_yoy town_mom town_accu cnt_val cnt_yoy cnt_mom cnt_accu",
    "cn_ppi": "month ppi_yoy ppi_mp_yoy ppi_mp_qm_yoy ppi_mp_rm_yoy ppi_mp_p_yoy ppi_cg_yoy ppi_cg_f_yoy ppi_cg_c_yoy ppi_cg_adu_yoy ppi_cg_dcg_yoy ppi_mom ppi_mp_mom ppi_mp_qm_mom ppi_mp_rm_mom ppi_mp_p_mom ppi_cg_mom ppi_cg_f_mom ppi_cg_c_mom ppi_cg_adu_mom ppi_cg_dcg_mom ppi_accu ppi_mp_accu ppi_mp_qm_accu ppi_mp_rm_accu ppi_mp_p_accu ppi_cg_accu ppi_cg_f_accu ppi_cg_c_accu ppi_cg_adu_accu ppi_cg_dcg_accu",
    "cn_pmi": "month pmi010000 pmi010100 pmi010200 pmi010300 pmi010400 pmi010401 pmi010402 pmi010403 pmi010500 pmi010501 pmi010502 pmi010503 pmi010600 pmi010601 pmi010602 pmi010603 pmi010700 pmi010701 pmi010702 pmi010703 pmi010800 pmi010801 pmi010802 pmi010803 pmi010900 pmi011000 pmi011100 pmi011200 pmi011300 pmi011400 pmi011500 pmi011600 pmi011700 pmi011800 pmi011900 pmi012000 pmi020100 pmi020101 pmi020102 pmi020200 pmi020201 pmi020202 pmi020300 pmi020301 pmi020302 pmi020400 pmi020401 pmi020402 pmi020500 pmi020501 pmi020502 pmi020600 pmi020601 pmi020602 pmi020700 pmi020800 pmi020900 pmi021000 pmi030000",
    "cn_m": "month m0 m0_yoy m0_mom m1 m1_yoy m1_mom m2 m2_yoy m2_mom",
    "sf_month": "month inc_month inc_cumval stk_endval",
    "shibor": "date on 1w 2w 1m 3m 6m 9m 1y",
    "shibor_lpr": "date 1y 5y",
}

# Non-interchangeable code spaces must not merge accidentally.
NAMESPACES = {
    "sw_daily": "TS_SW_INDEX", "ci_daily": "TS_CI_INDEX",
    "ths_index": "TS_THS_INDEX", "ths_daily": "TS_THS_INDEX",
    "dc_index": "TS_DC_INDEX", "dc_daily": "TS_DC_INDEX",
    "tdx_index": "TS_TDX_INDEX", "tdx_daily": "TS_TDX_INDEX",
    "index_global": "TS_GLOBAL_INDEX", "fut_index_daily": "TS_NH_INDEX",
}

# Units verified per endpoint; never apply one multiplier to all index families.
QUOTE_UNITS = {
    "fund_daily": (100, 1000), "index_daily": (100, 1000),
    "sw_daily": (10000, 10000), "ci_daily": (10000, 10000),
    "ths_daily": (100, None), "dc_daily": (1, 1),
    # TDX amount can represent futures open interest, so do not map as turnover.
    "tdx_daily": (100, None),
}

# Undocumented caps remain local safeguards, not vendor entitlement assertions.
LOCAL_CAPS = {"fund_nav": 15000, "stock_basic": 6000, "trade_cal": 15000,
              "index_basic": 15000, "etf_index": 5000, "fund_company": 5000,
              "index_member_all": 2000}


def source_field_type(api: str, name: str) -> str:
    if name.endswith("date") or name == "date":
        return "date"
    if name in {"quarter", "month", "ts_code", "exchange", "index_code", "con_code", "name", "level", "is_new", "src", "parent_code", "industry_code", "industry_name", "l1_code", "l2_code", "l3_code", "l1_name", "l2_name", "l3_name", "type", "idx_type", "title", "issuing_org", "data_api", "repo_maturity", "leading", "leading_code"}:
        return "string"
    if api in EXTRA_COLUMNS and name not in {"quarter", "month"}:
        return "number"
    if name in {"open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount", "unit_nav", "accum_nav", "accum_div", "adj_nav", "net_asset", "total_netasset", "total_share", "total_size", "nav", "mkv", "m_fee", "c_fee", "mgt_fee", "weight", "adj_factor", "div_cash", "base_unit", "birth_year", "stk_float_ratio", "stk_mkv_ratio"}:
        return "number"
    return "string"
