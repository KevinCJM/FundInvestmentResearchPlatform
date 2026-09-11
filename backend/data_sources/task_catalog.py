"""Registered dataset operations. Provider knowledge stays behind this catalog."""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from .models import CenterError

ROOT = Path(__file__).resolve().parents[2]
# Acquisition adapters own these endpoint groups; the browser never interprets them.
ACTION_APIS = {
    'etf_info': ['fund_basic', 'etf_basic'], 'fund_info': ['fund_basic'],
    'fund_company': ['fund_company'], 'calendar': ['trade_cal'], 'nav': ['fund_nav'],
    'etf_share': ['etf_share_size'], 'candle': ['fund_daily'], 'fund_nav': ['fund_nav'],
    'fund_manager': ['fund_manager'], 'fund_scale': [], 'fund_portfolio': ['fund_portfolio'],
    'fund_dividend': ['fund_div'], 'fund_adjustment': ['fund_adj'], 'fund_benchmark': ['mkt_idx_bmk'],
    'stock_basic': ['stock_basic'], 'index_info': ['index_basic'], 'etf_index': ['etf_index'],
    'index_catalog': ['index_basic', 'etf_index', 'index_classify', 'ths_index', 'dc_index', 'tdx_index'],
    'index_domestic': ['index_daily'], 'index_industry': ['sw_daily', 'ci_daily'],
    'index_concept': ['ths_daily', 'dc_daily', 'tdx_daily'], 'index_global': ['index_global'],
    'index_futures': ['fut_index_daily'], 'index_valuation': ['index_dailybasic'],
    'index_constituents': ['index_member_all', 'index_weight', 'ci_index_member', 'ths_member', 'dc_member', 'tdx_member'], 'index_coverage': [],
    'macro_cycle': ['cn_gdp', 'cn_cpi', 'cn_ppi', 'cn_pmi'], 'macro_money_credit': ['cn_m', 'sf_month'],
    'macro_rates': ['shibor', 'shibor_lpr', 'repo_daily'], 'macro_release_calendar': ['cn_schedule'],
}
REQUIRES = {
    'nav': ['etf_info', 'calendar'], 'etf_share': ['etf_info', 'calendar'],
    'candle': ['etf_info', 'calendar'], 'fund_nav': ['fund_info', 'calendar'],
    'fund_manager': ['fund_info'], 'fund_scale': ['fund_nav'], 'fund_portfolio': ['fund_info'],
    'fund_dividend': ['fund_info'], 'fund_adjustment': ['etf_info', 'calendar'],
    **{name: ['index_catalog', 'calendar'] for name in (
        'index_domestic', 'index_industry', 'index_concept', 'index_global',
        'index_futures', 'index_valuation', 'index_constituents', 'index_coverage')},
}
DATE_FIELDS = [
    {'name': 'start_date', 'label': '历史开始日期', 'data_type': 'date', 'required': True, 'default': '20100101', 'description': '指定数据范围的起点；并非只下载单个产品。'},
    {'name': 'end_date', 'label': '本次截止日期', 'data_type': 'date', 'required': True, 'default': '', 'description': '选择已经完成披露的日期。'},
]


@lru_cache(maxsize=1)
def task_specs() -> dict[str, dict]:
    labels = None
    for node in ast.parse((ROOT / 'T01_get_data.py').read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ACTION_LABELS' for t in node.targets):
            labels = ast.literal_eval(node.value)
    if labels is None or set(labels) != set(ACTION_APIS):
        raise CenterError('TASK_REGISTRY_DRIFT', '数据集目录与采集器不一致，请同步任务合同。')
    tasks = {}
    for action, label in labels.items():
        category = ('指数与基准' if action.startswith('index_') or action == 'etf_index' else
                    '宏观与利率' if action.startswith('macro_') else
                    '公募基金' if action.startswith('fund_') and action not in {'fund_info', 'fund_company', 'fund_adjustment'} else
                    'ETF' if action in {'nav', 'candle', 'etf_share', 'fund_adjustment'} else '基础信息')
        identifier = 'tushare.' + action
        tasks[identifier] = {
            'id': identifier, 'name': label, 'category': category, 'transport': 'tushare',
            'handler': 'tushare_dataset', 'action': action, 'api_slots': ACTION_APIS[action],
            'requires': REQUIRES.get(action, []), 'provides': [action],
            'parameters': DATE_FIELDS if ACTION_APIS[action] else [],
            'output_type': 'market_files_v1', 'network': bool(ACTION_APIS[action]),
            'description': '遍历该数据集全部适用标的，复用采集器的分页、日期分片和断点；未设置标的数量限制。' if ACTION_APIS[action] else '使用明确的前置工作区计算本地派生数据，不访问外部接口。',
        }
        if action == 'macro_cycle':
            tasks[identifier]['description'] = ('美林时钟增长与通胀输入：PMI、CPI、GDP、PPI。'
                '每次核验完整历史快照并保留修订，运行日期不裁剪快照；'
                '历史发布日期未知时，仅用于事后研究。')
    tasks['local.analytics_snapshot'] = {
        'id': 'local.analytics_snapshot', 'name': '指标分析快照', 'category': '本地计算',
        'transport': None, 'handler': 'analytics_snapshot', 'action': '', 'api_slots': [],
        'requires': ['etf_info', 'fund_info', 'nav', 'fund_nav', 'calendar'], 'provides': ['analytics_snapshot'],
        'parameters': [], 'output_type': 'market_files_v1', 'network': False,
        'description': '只读取前置私有工作区及冻结指标配置，使用现有 NJIT 计算器；不自动发布。',
    }
    return tasks


def get_task(identifier: str) -> dict:
    spec = task_specs().get(identifier)
    if spec is None:
        raise CenterError('ETL_TASK_UNKNOWN', '未知或未登记的数据集任务。', 422)
    return spec


def task_catalog(store) -> dict:
    from .auto_incremental import supported
    sources = store.list('source')
    result = []
    for spec in task_specs().values():
        public = {k: v for k, v in spec.items() if k not in {'handler', 'action', 'transport'}}
        public['source_ids'] = [s['config']['id'] for s in sources if s['config']['transport'] == spec['transport']]
        public['requires_source'] = spec['transport'] is not None
        public['auto_incremental_supported'] = supported(spec['action'])
        result.append(public)
    return {'tasks': result, 'version': 1}


def inspect_task(store, step, capabilities: set[str], *, require_values: bool = False) -> dict:
    from datetime import datetime
    from .credentials import configured
    spec = get_task(step.task_id)
    if set(spec['requires']) - capabilities:
        names = ', '.join(set(spec['requires']) - capabilities)
        raise CenterError('ETL_TASK_DEPENDENCY', f'{step.name} 缺少前置数据集：{names}。', 422)
    fields = {p['name']: p for p in spec['parameters']}
    if set(step.params) - fields.keys() or set(step.parameter_bindings) - fields.keys():
        raise CenterError('ETL_TASK_PARAMS', '任务含未登记参数；不能传入目录、命令或产品数量限制。', 422)
    for name, field in fields.items():
        value = step.params.get(name, field['default'])
        if not value and require_values:
            raise CenterError('ETL_TASK_PARAMS', f'请填写 {field["label"]}。', 422)
        if value:
            try:
                datetime.strptime(str(value), '%Y%m%d')
            except ValueError:
                raise CenterError('ETL_TASK_PARAMS', '数据集日期必须是有效 YYYYMMDD。', 422) from None
    if step.params.get('start_date') and step.params.get('end_date') and step.params['start_date'] > step.params['end_date']:
        raise CenterError('ETL_TASK_PARAMS', '开始日期不能晚于截止日期。', 422)
    record = {'spec': spec, 'source': None, 'interfaces': {}}
    if spec['transport']:
        source = store.get('source', step.source_id)
        if source['config']['transport'] != spec['transport'] or not source['config']['enabled']:
            raise CenterError('ETL_TASK_SOURCE', '所选来源不支持此任务或已停用。', 422)
        if require_values and not configured(store, step.source_id):
            raise CenterError('CREDENTIAL_REQUIRED', '请先保存数据源凭据。', 422)
        record['source'] = source
        for api in spec['api_slots']:
            entry = store.get('interface', step.source_id + '.' + api)
            if not entry['config']['enabled']:
                raise CenterError('ETL_TASK_INTERFACE_DISABLED', f'请启用接口 {entry["config"]["name"]}。', 422)
            from .presets import API_SPECS
            config = entry['config']
            if config['api_name'] not in API_SPECS and not config.get('entitlement_confirmed'):
                raise CenterError('ENTITLEMENT_REQUIRED', '被修改的接口尚未确认权限，请先核验来源配置。', 422)
            narrowed = {k for k, v in config.get('params', {}).items() if v not in (None, '')} & {'ts_code', 'symbol', 'code', 'limit', 'offset', 'page', 'nav_date', 'trade_date'}
            if narrowed:
                raise CenterError('ETL_TASK_NARROWED', f'{config["name"]} 的默认参数限制了数据范围，请为全数据任务使用无单产品或单日限制的接口配置。', 422)
            record['interfaces'][api] = entry
    elif step.source_id:
        raise CenterError('ETL_TASK_SOURCE', '本地计算任务不使用外部来源。', 422)
    return record
