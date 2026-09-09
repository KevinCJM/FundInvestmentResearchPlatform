"""Explicitly requested workflow examples; never seed or overwrite user plans."""
from __future__ import annotations

from .etl_models import EtlDefinition, EtlParameter, EtlStep
from .models import CenterError
from .store import SourceStore


def tushare_all_data_workflow(store: SourceStore, source_id: str = 'tushare') -> EtlDefinition:
    from .task_catalog import task_specs
    source = store.get('source', source_id)
    if source['config']['transport'] != 'tushare':
        raise CenterError('ETL_TEMPLATE_SOURCE', '所选来源没有此采集适配器。')
    steps = []
    previous = None
    for spec in task_specs().values():
        identifier = 'dataset_' + (spec['action'] or 'analytics_snapshot')
        bindings = {p['name']: p['name'] for p in spec['parameters']}
        steps.append(EtlStep(id=identifier, name=spec['name'], kind='task', task_id=spec['id'],
                             source_id=source_id if spec['transport'] else None,
                             inputs=[previous] if previous else [], parameter_bindings=bindings,
                             mode='inherit'))
        previous = identifier
    return EtlDefinition(name=source['config']['name'] + ' 全数据同步',
        description='覆盖项目已接入的全部数据集：基础信息、全部 ETF、公募基金、指数及宏观数据，最后计算私有指标快照。不是供应商全站所有 API。无需单产品代码；可自由增删任务、重排依赖及修改快照位置。标准映射候选需单独核验，不自动发布。',
        max_runtime_seconds=86400, steps=steps, parameters=[
            EtlParameter(id='start_date', label='历史开始日期', data_type='date', date_format='compact', default='2010-01-01'),
            EtlParameter(id='end_date', label='本次截止日期', data_type='date', date_format='compact', description='选择已完成披露的日期。全量和增量共用此流程。'),
        ])


def template_catalog(store: SourceStore) -> list[dict]:
    result = []
    for source in store.list('source'):
        if source['config']['transport'] == 'tushare':
            definition = tushare_all_data_workflow(store, source['config']['id'])
            result.append({'id': source['config']['id'] + '.all-data', 'name': definition.name,
                           'description': definition.description, 'definition': definition.model_dump(mode='json')})
    return result


def tushare_fund_workflow(store: SourceStore) -> EtlDefinition:
    """Use saved interfaces; products and dates are supplied at each run."""
    source = store.get("source", "tushare")
    if source["config"]["transport"] != "tushare":
        raise CenterError("ETL_TEMPLATE_SOURCE", "此示例需要一个使用 Tushare 协议的数据源。")
    steps: list[EtlStep] = []

    def download(identifier, api, name, table, *, code=None, dates=False, mode="inherit", params=None):
        saved = store.get("interface", "tushare." + api)
        interface = saved["config"]
        if not any(mapping["enabled"] and mapping["target_table"] == table for mapping in interface["mappings"]):
            raise CenterError("ETL_TEMPLATE_MAPPING", f"{name} 的已保存接口未映射到所需业务表，请先完善接口。")
        bindings = {"ts_code": code} if code else {}
        if dates:
            bindings.update({interface["start_param"]: "start_date", interface["end_param"]: "end_date"})
        steps.append(EtlStep(id=identifier, name=name, kind="download", source_id="tushare",
                             interface_id=interface["id"], interface_revision=saved["revision"],
                             mode=mode, params=params or {}, parameter_bindings=bindings, target_tables=[table]))
        mapped = identifier + "_map"
        steps.append(EtlStep(id=mapped, name="映射 · " + name, kind="map", inputs=[identifier]))
        return mapped

    def resolve(identifier, name, table, inputs, *, history=False):
        steps.append(EtlStep(id=identifier, name=name, kind="resolve", table_id=table,
                             inputs=inputs, include_history=history, history_scope="matching_inputs"))
        return identifier

    calendar = download("calendar", "trade_cal", "上交所交易日历", "master.trading_calendar",
                        dates=True, mode="full", params={"exchange": "SSE"})
    calendar_result = resolve("calendar_result", "整理交易日历", "master.trading_calendar", [calendar])
    etf = download("etf_info", "etf_basic", "ETF 基础信息", "master.instrument", code="etf_code", mode="full")
    fund = download("fund_info", "fund_basic", "场外基金基础信息", "master.instrument",
                    code="fund_code", mode="full", params={"market": "O"})
    instruments = resolve("instruments", "整理产品信息", "master.instrument", [etf, fund])
    etf_nav = download("etf_nav", "fund_nav", "ETF 净值", "market.nav_daily", code="etf_code", dates=True, params={"market": "E"})
    fund_nav = download("fund_nav", "fund_nav", "场外基金净值", "market.nav_daily", code="fund_code", dates=True, params={"market": "O"})
    nav = resolve("nav_result", "合并净值与同范围历史", "market.nav_daily", [etf_nav, fund_nav], history=True)
    quote = download("etf_quote", "fund_daily", "ETF 日行情", "market.quote_daily", code="etf_code", dates=True)
    quote_result = resolve("quote_result", "合并行情与同范围历史", "market.quote_daily", [quote], history=True)
    steps.append(EtlStep(id="snapshot", name="计算基金研究指标快照", kind="snapshot", inputs=[calendar_result, instruments, nav, quote_result]))
    return EtlDefinition(
        name="Tushare ETF 与公募基金数据同步",
        description="按运行时指定的一只 ETF、一只场外基金及日期范围同步，不默认拉全市场。日历和产品信息每次全量刷新；净值与行情跟随本次全量/增量模式。先下载映射，完成取值后计算候选指标快照，不自动发布。可编辑或复制步骤扩展范围。",
        max_runtime_seconds=3600,
        parameters=[
            EtlParameter(id="etf_code", label="ETF 代码", description="填写完整代码，例如 510300.SH。未填写时不会下载全市场。"),
            EtlParameter(id="fund_code", label="场外基金代码", description="填写完整份额代码，例如 000001.OF；不合并不同份额类别。"),
            EtlParameter(id="start_date", label="历史开始日期", data_type="date", date_format="compact", default="2010-01-01", description="全量或首次增量从此日起；后续增量使用成功断点并回查 3 天。"),
            EtlParameter(id="end_date", label="本次截止日期", data_type="date", date_format="compact", description="选择已完成披露的日期；本次日期固定后，中断恢复不会自动延长。"),
        ],
        steps=steps,
    )
