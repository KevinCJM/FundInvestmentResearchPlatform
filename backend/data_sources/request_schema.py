"""Request metadata belongs to connector contracts, never response-column guesses."""
from __future__ import annotations
from .models import RequestField

# Input parameters declared by the existing connector APIs. User definitions
# override these defaults, including an explicit empty list for JSON-only input.
CODE_APIS = {'fund_basic', 'etf_basic', 'fund_nav', 'fund_daily', 'etf_share_size',
             'fund_manager', 'fund_portfolio', 'fund_div', 'fund_adj', 'stock_basic',
             'index_basic', 'index_daily', 'sw_daily', 'ci_daily', 'ths_daily', 'dc_daily',
             'tdx_daily', 'index_global', 'fut_index_daily', 'index_dailybasic'}
RANGE_APIS = {'trade_cal', 'fund_nav', 'fund_daily', 'etf_share_size', 'fund_adj',
              'index_daily', 'sw_daily', 'ci_daily', 'ths_daily', 'dc_daily', 'tdx_daily',
              'index_global', 'fut_index_daily', 'index_dailybasic', 'index_weight',
              'shibor', 'shibor_lpr', 'repo_daily'}


def request_fields(source, interface) -> list[dict]:
    if interface.request_fields is not None:
        return [f.model_dump() for f in interface.request_fields]
    result = []
    if source.transport == 'tushare':
        if interface.api_name in CODE_APIS:
            result.append(RequestField(name='ts_code', label='产品代码（ts_code）', description='留空时由接口协议决定范围；全部标的采集应使用数据集任务。'))
        if interface.api_name == 'trade_cal':
            result.append(RequestField(name='exchange', label='交易所代码'))
        dates = interface.api_name in RANGE_APIS
    elif source.transport == 'akshare':
        result.append(RequestField(name='symbol', label='产品代码（symbol）', required=True, description='六位产品代码，保留前导零。'))
        dates = True
    else:
        dates = False
    if dates:
        result.extend(RequestField(name=name, label=label, data_type='date') for name, label in
                      ((interface.start_param, '开始日期'), (interface.end_param, '结束日期')))
    return [f.model_dump() for f in result]
