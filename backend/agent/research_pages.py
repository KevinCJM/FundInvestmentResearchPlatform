"""Frozen page requests -> existing read-only services -> registered model views.

Runtime callbacks come from already-loaded API modules. Importing agent common
code never imports a business router or creates another service/store.
"""
from __future__ import annotations

import sys
import re
from collections import defaultdict
from datetime import date
from typing import Annotated, Literal, Optional

from pydantic import Field, ValidationError, field_validator, model_validator

from .contracts import AgentError, Contract, EvaluationTarget
from . import data_policy, views
from .sessions import stable_hash
from .views import STR, INT, NUM, BOOL

Text = Annotated[str, Field(max_length=120)]
Date = Annotated[str, Field(pattern=r'^\d{4}-\d{2}-\d{2}$')]
ParameterNumber = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class IndicatorRef(Contract):
    indicator_id: Text
    indicator_revision: int = Field(ge=1)
    period: Text = '1Y'
    parameters: dict[str, ParameterNumber] = Field(default_factory=dict, max_length=16)


class Condition(Contract):
    field: Text
    operator: Literal['gte', 'lte', 'gt', 'lt', 'eq']
    value: Annotated[str, Field(min_length=1, max_length=80)]


class ProductListRequest(Contract):
    kind: Literal['etf', 'fund']
    q: Annotated[str, Field(max_length=80)] = ''
    fund_type: list[Text] = Field(default_factory=list, max_length=30)
    fund_category: list[Text] = Field(default_factory=list, max_length=30)
    invest_type: list[Text] = Field(default_factory=list, max_length=30)
    market: list[Text] = Field(default_factory=list, max_length=30)
    status: list[Text] = Field(default_factory=list, max_length=30)
    management: list[Text] = Field(default_factory=list, max_length=30)
    custodian: list[Text] = Field(default_factory=list, max_length=30)
    qdii_type: list[Text] = Field(default_factory=list, max_length=30)
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=10, ge=1, le=200)
    sort_by: Text = 'issue_amount'
    sort_dir: Literal['asc', 'desc'] = 'desc'
    conditions: list[Condition] = Field(default_factory=list, max_length=20)
    snapshot_metrics: list[Text] = Field(default_factory=list, max_length=8)
    as_of: Optional[Date] = None
    targets: list[EvaluationTarget] = Field(default_factory=list, max_length=10)
    batch_offset: int = Field(default=0, ge=0)
    visible_count: int = Field(default=0, ge=0)
    selected_count: int = Field(default=0, ge=0)
    selection_mode: Literal['current_page', 'selected', 'all_matching'] = 'current_page'
    excluded_ids: list[Text] = Field(default_factory=list, max_length=200)
    indicators: list[IndicatorRef] = Field(default_factory=list, max_length=10)
    view_mode: Literal['overview', 'metrics'] = 'overview'

    @model_validator(mode='after')
    def batch_contract(self):
        if any(target.kind != self.kind for target in self.targets):
            raise ValueError('batch kind mismatch')
        return self


class CompareRange(Contract):
    start_date: Optional[Date] = None
    end_date: Optional[Date] = None

    @model_validator(mode='after')
    def dates(self):
        for value in (self.start_date, self.end_date):
            if value: date.fromisoformat(value)
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('invalid range')
        return self


class CompareRanges(Contract):
    performance: CompareRange
    risk: CompareRange
    efficiency: CompareRange


class CompareTarget(EvaluationTarget):
    management_fee: Optional[float] = Field(default=None, allow_inf_nan=False)
    custody_fee: Optional[float] = Field(default=None, allow_inf_nan=False)


class CompareRequest(Contract):
    targets: list[CompareTarget] = Field(min_length=1, max_length=10)
    ranges: CompareRanges
    rolling_window_days: int = Field(default=30, ge=2, le=252)
    indicators: list[IndicatorRef] = Field(default_factory=list, max_length=10)
    as_of: Optional[Date] = None
    # The matrix has its own explicit cutoff; the comparison follows the page PIT.
    metrics_as_of: Optional[Date] = None
    source: Literal['actual', 'demo'] = 'actual'


class HoldingRequest(Contract):
    run_id: Annotated[str, Field(pattern=r'^run-[0-9a-f]{32}$')]
    indicators: list[IndicatorRef] = Field(default_factory=list, max_length=10)
    scenario: Optional[CompareRange] = None


REQUESTS = {'product-research': ProductListRequest, 'product-compare': CompareRequest, 'holding-diagnosis': HoldingRequest}
OPERATIONS = {'product-research': {'catalog', 'metrics'}, 'product-compare': {'comparison', 'metrics'},
              'holding-diagnosis': {'diagnosis', 'scenario', 'metrics'}}


def _error(code='AGENT_PAGE_REQUEST_INVALID', message='页面冻结请求无效，请核对页面条件后重发。'):
    return AgentError(code, message, status_code=409, field='page_snapshot')


def parse_request(page, snapshot):
    if not isinstance(snapshot, dict) or snapshot.get('page') != page or page not in REQUESTS:
        raise _error()
    raw = (snapshot.get('sections') or {}).get('request')
    try:
        parsed = REQUESTS[page].model_validate(raw)
    except (ValidationError, ValueError, TypeError):
        raise _error() from None
    # Free-form names cannot be a second structured payload. Declared condition
    # values and runtime parameters are checked against business contracts later.
    safe = parsed.model_dump(exclude={'conditions', 'indicators', 'targets', 'q'})
    data_policy.check_text_fields(safe)
    if hasattr(parsed, 'q') and data_policy.user_text_violation(parsed.q, allow_number=True):
        raise _error()
    for target in getattr(parsed, 'targets', []):
        if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,100}', target.product_id):
            raise _error()
    for reference in parsed.indicators:
        data_policy.check_text_fields(reference.model_dump(exclude={'parameters'}))
    for condition in getattr(parsed, 'conditions', []):
        data_policy.check_text_fields(condition.model_dump(exclude={'value'}))
    return parsed


def request_view(page, snapshot):
    request = parse_request(page, snapshot)
    result = request.model_dump()
    if isinstance(request, CompareRequest):
        result['targets'] = [{**target.model_dump(include={'kind', 'product_id'}),
                              'fees_withheld': target.management_fee is not None or target.custody_fee is not None}
                             for target in request.targets]
    if 'conditions' in result:
        result['conditions'] = [{'field': item.field, 'operator': item.operator, 'value_withheld': True}
                                for item in request.conditions]
    result['indicators'] = [{'indicator_id': item.indicator_id, 'indicator_revision': item.indicator_revision,
                             'period': item.period, 'parameters_withheld': bool(item.parameters)}
                            for item in request.indicators]
    result['trust'] = 'frozen_user_request_not_result_proof'
    return result


# Last-displayed requests are client claims, not numerical evidence. In particular,
# fee/parameter values and serialized query/PIT blobs must not regain admission here.
_DISPLAY_INDICATOR = {'indicator_id': STR, 'indicator_revision': INT, 'period': STR}
_DISPLAY_REQUEST = {
    'targets': [{'kind': STR, 'product_id': STR}], 'as_of': STR,
    'ranges': {name: {'start_date': STR, 'end_date': STR} for name in ('performance', 'risk', 'efficiency')},
    'rolling_window_days': INT, 'run_id': STR, 'indicator_ids': [STR],
    'start_date': STR, 'end_date': STR,
}
_DISPLAY_REQUEST['requests'] = [{**_DISPLAY_REQUEST, 'period': STR, 'indicator_refs': [_DISPLAY_INDICATOR]}]
_DISPLAY_REFERENCES = {'product-research': ('list', 'metrics'),
                       'product-compare': ('comparison', 'metrics'),
                       'holding-diagnosis': ('metrics', 'scenario_result')}


def results_view(page, snapshot):
    section = (snapshot.get('sections') or {}).get('results')
    refs = section.get('refs') if isinstance(section, dict) else None
    refs = refs if isinstance(refs, dict) else {}
    projected = {}
    for name in _DISPLAY_REFERENCES.get(page, ()):
        record = refs.get(name)
        if not isinstance(record, dict):
            continue
        status = record.get('status')
        frozen = record.get('frozen_request')
        projected[name] = {
            'status': status if isinstance(status, str) and status in {
                'loading', 'error', 'not_requested', 'stale', 'ready', 'pending'} else 'unknown',
            'frozen_request': _project(frozen, _DISPLAY_REQUEST) if isinstance(frozen, dict) else None,
            'request_hash': stable_hash(frozen) if isinstance(frozen, dict) else None,
            **_project(record, {'indicators': [_DISPLAY_INDICATOR], 'resolved_indicators': [_DISPLAY_INDICATOR]}),
        }
    return {'source': 'unverified_client_display', 'trust': 'client_display_reference_not_result_proof',
            'refs': projected,
            'note': '这些状态和原请求引用只是页面声明，不证明数值。loading/error/stale/pending/unknown 时先说明显示结果尚未对应当前请求；'
                    'page.analyze只按request分区重算，不能冒充旧图表的原结果。完整查询、费率、参数值和客户端结果数值不下发。'}


def runtime_callbacks():
    """Resolve existing router callbacks without importing/initializing routers."""
    instruments = sys.modules.get('services.instrument_routes') or sys.modules.get('backend.services.instrument_routes')
    portfolio = sys.modules.get('services.portfolio_routes') or sys.modules.get('backend.services.portfolio_routes')
    callbacks = {}
    if instruments is not None:
        callbacks['catalog'] = instruments.instrument_products
        def compare(target, parameters):
            detail = instruments.instrument_product_detail(target['product_id'], kind=target['kind'], include_timeseries=False)
            if not isinstance(detail, dict):
                raise _error('AGENT_PRODUCT_UNAVAILABLE', '产品数据不可用。')
            fees = detail.get('metrics') or {}
            if (target['management_fee'] != fees.get('m_fee') or target['custody_fee'] != fees.get('c_fee')):
                raise _error('AGENT_PAGE_FEES_CHANGED', '页面费用与当前产品资料不一致，请刷新比较条件。')
            body = instruments.ProductCompareAnalysisRequest(**parameters,
                management_fee=fees.get('m_fee'), custody_fee=fees.get('c_fee'))
            return instruments.instrument_product_compare_analysis(target['product_id'], body, kind=target['kind'])
        callbacks['comparison'] = compare
    if portfolio is not None:
        callbacks.update(run=portfolio.portfolio_service.get_run, diagnosis=portfolio.portfolio_service.diagnose,
                         scenario=portfolio.portfolio_service.scenario)
    return callbacks


def _callback(callbacks, name):
    callback = (callbacks or {}).get(name)
    if not callable(callback):
        raise _error('AGENT_PAGE_SERVICE_UNAVAILABLE', '此页面分析服务尚未挂载。')
    return callback


def _project(payload, schema):
    counter = views.Counter()
    result = views.project(payload, schema, 'research', counter)
    return {} if result is views.OMIT else result


RUN_METADATA = {'id': STR, 'created_at': STR, 'immutable': BOOL, 'target_id': STR, 'target_revision': INT,
                'target_name': STR, 'requested_as_of': STR, 'effective_as_of': STR, 'actual_start_date': STR,
                'actual_end_date': STR, 'observation_count': INT, 'common_date_hash': STR}
COMPARE_UNITS = {'cumulativeReturn': 'percent', 'annualizedReturn': 'percent', 'volatility': 'percent',
                 'maxDrawdown': 'percent', 'totalFee': 'percent', 'returnToFee': 'ratio',
                 'sharpeRatio': 'ratio', 'calmarRatio': 'ratio'}
PORTFOLIO_UNITS = {'cumulative_return': 'decimal_return', 'annual_return': 'decimal_return',
                   'annual_volatility': 'decimal_return', 'sharpe_ratio': 'ratio', 'max_drawdown': 'decimal_return',
                   'var_99': 'decimal_return', 'es_99': 'decimal_return'}


def _portfolio_summary(snapshot, diagnosis=None):
    result = {'snapshot': _project(snapshot, RUN_METADATA), 'source': 'immutable_portfolio_run',
              'pit': {'source': 'snapshot_requested_and_effective_dates', 'run_mode': 'unknown', 'data_release_id': None}}
    observations = snapshot.get('observation_count')
    if isinstance(observations, int) and observations >= 2:
        result['metrics'] = _project(snapshot.get('summary') or {}, {key: NUM for key in PORTFOLIO_UNITS})
        result['metric_units'] = PORTFOLIO_UNITS
    else:
        result['metrics_omitted'] = 'insufficient_observations'
    if diagnosis is not None:
        result['concentration'] = _project(diagnosis.get('concentration_summary') or {},
            {key: NUM for key in ('max_weight', 'top3_weight', 'hhi', 'effective_holdings')})
        components = diagnosis.get('components') or []
        result['component_count'] = len(components)
        result['components'] = [_project(row, {'kind': STR, 'product_id': STR, 'name': STR, 'current_weight': NUM,
                                               'period_return': NUM, 'simple_return_contribution': NUM})
                                for row in components[:10]]
        result['risk_contributions'] = [_project(row, {'product_id': STR, 'name': STR, 'risk_contribution': NUM,
                                                       'marginal_risk': NUM, 'component_risk': NUM})
                                        for row in (diagnosis.get('risk_contributions') or [])[:10]]
        result['omitted_components'] = max(0, len(components)-10)
    result['warnings'] = _project(snapshot, {'warnings': [views.WARNING]}).get('warnings', [])
    result['data_identity'] = {'fingerprints': _project(snapshot.get('data_fingerprints') or {}, {'*': STR}),
                               'common_date_hash': snapshot.get('common_date_hash')}
    return result


def _single_context(request, page_context, service):
    from pit.context import resolve_request_context, view_override
    if page_context.context_kind != 'single_product':
        raise _error('AGENT_TOOL_DOMAIN_MISMATCH', '此页面只允许单产品研究上下文。')
    resolved = resolve_request_context(service.market_data_dir, request.as_of)
    current = view_override()
    if resolved.as_of != page_context.calculation.as_of or (current is not None and resolved != current):
        raise _error('AGENT_CONTEXT_CHANGED', '冻结研究日与当前有效研究口径不一致。')
    targets = [{key: getattr(target, key) for key in ('kind', 'product_id')} for target in request.targets]
    if targets != [target.model_dump() for target in page_context.calculation.targets]:
        raise _error('AGENT_CONTEXT_CHANGED', '冻结批次与当前页面计算对象不一致。')
    return targets, resolved.as_of


def _evaluate_metrics(service, request, *, targets=None, as_of=None, run=None):
    from .tools import _proof_map_for_rows
    groups = defaultdict(list)
    for reference in request.indicators:
        definition = service.get_indicator(reference.indicator_id, reference.indicator_revision)
        if int(definition.get('revision', 0)) != reference.indicator_revision:
            raise _error('AGENT_INDICATOR_VERSION_CHANGED', '指标冻结版本不可用。')
        if views.parameter_view(reference.parameters, views.definition_view(definition)) != reference.parameters:
            raise _error('AGENT_PAGE_PARAMETERS_INVALID', '指标参数与冻结版本的契约不一致。')
        groups[reference.period].append(reference)
    evaluations = []
    for period, references in groups.items():
        if run is not None:
            # Existing portfolio API resolves current saved definitions. Refuse a
            # historical mismatch rather than silently switching the requested version.
            for reference in references:
                if service.get_indicator(reference.indicator_id).get('revision') != reference.indicator_revision:
                    raise _error('AGENT_INDICATOR_VERSION_CHANGED', '组合接口当前版本与页面锁定版本不一致。')
                if reference.parameters:
                    raise _error('AGENT_PAGE_PARAMETERS_INVALID', '当前组合指标接口不接受运行参数覆盖。')
            output = service.evaluate_portfolio(run_id=run['id'], indicator_ids=[r.indicator_id for r in references], inline_definition=None)
        else:
            output = service.evaluate(indicator_ids=[], inline_definition=None,
                indicator_refs=[reference.model_dump(exclude={'period'}) for reference in references],
                targets=targets, period=period, as_of=as_of, include_series=False, prefer_snapshot=False)
        locked = {(reference.indicator_id, reference.indicator_revision) for reference in references}
        if any((row.get('indicator_id'), row.get('indicator_revision')) not in locked for row in output.get('results', [])):
            raise _error('AGENT_INDICATOR_VERSION_CHANGED', '计算结果与冻结指标版本不一致。')
        projected, _ = views.project_evaluation_result(output, views.Projection(
            proof_map=_proof_map_for_rows(service, output.get('results'))), default_kind='scalar')
        evaluations.append({'period': period if run is None else 'frozen_run', **projected})
    return evaluations


def analyze(operation, page_context, snapshot, service, callbacks, *, target_offset=0, target_limit=3,
            indicator_offset=0, indicator_limit=3):
    page = page_context.page
    if operation not in OPERATIONS.get(page, set()):
        raise _error('AGENT_PAGE_OPERATION_NOT_ALLOWED', '该操作不适用于当前页面。')
    request = parse_request(page, snapshot)
    result = {'page': page, 'operation': operation, 'provenance': 'existing_service_current_data_frozen_request'}
    result['pagination'] = {'indicator_count': len(request.indicators), 'indicator_offset': indicator_offset,
        'next_indicator_offset': indicator_offset+indicator_limit if indicator_offset+indicator_limit < len(request.indicators) else None}
    request.indicators = request.indicators[indicator_offset:indicator_offset+indicator_limit]
    if page == 'holding-diagnosis':
        if page_context.context_kind != 'portfolio' or page_context.calculation.run_id != request.run_id:
            raise _error('AGENT_CONTEXT_CHANGED', '页面快照与组合运行对象不一致。')
        run = _callback(callbacks, 'run')(request.run_id)
        if not isinstance(run, dict) or run.get('id') != request.run_id or run.get('immutable') is not True:
            raise _error('AGENT_PORTFOLIO_RUN_INVALID', '只有真实不可变运行快照可用于诊断。')
        if operation == 'metrics':
            result['evaluations'] = _evaluate_metrics(service, request, run=run)
        elif operation == 'scenario':
            if request.scenario is None or not request.scenario.start_date or not request.scenario.end_date:
                raise _error('AGENT_SCENARIO_REQUEST_REQUIRED', '请先在页面明确情景起止日期。')
            scenario = _callback(callbacks, 'scenario')(request.run_id, **request.scenario.model_dump())
            if scenario.get('source_run_id') != request.run_id or scenario.get('locked_target_revision') != run.get('target_revision'):
                raise _error('AGENT_CONTEXT_CHANGED', '情景来源与锁定策略不一致。')
            result['portfolio'] = _portfolio_summary(scenario)
            result.update(source_run_id=request.run_id, locked_target_revision=run['target_revision'],
                          provenance='current_data_scenario_locked_strategy_not_snapshot_replay')
        else:
            diagnosis = _callback(callbacks, 'diagnosis')(request.run_id, [])
            if diagnosis.get('run_id') != request.run_id:
                raise _error('AGENT_CONTEXT_CHANGED', '诊断结果不属于请求的运行。')
            result['portfolio'] = _portfolio_summary(run, diagnosis)
        return result
    targets, as_of = _single_context(request, page_context, service)
    total_targets = len(targets)
    targets = targets[target_offset:target_offset+target_limit]
    request.targets = request.targets[target_offset:target_offset+target_limit]
    result['pagination'].update(target_count=total_targets, target_offset=target_offset,
        next_target_offset=target_offset+target_limit if target_offset+target_limit < total_targets else None)
    result['effective_as_of'] = as_of
    result['batch'] = {'count': len(targets), 'targets': targets, 'scope': 'explicit_current_batch_only'}
    if page == 'product-research':
        result['batch'].update(visible_count=request.visible_count, selected_count=request.selected_count,
                               selection_mode=request.selection_mode, offset=request.batch_offset,
                               selection_provenance='client_selection_claim; only_explicit_targets_are_evaluated')
        if operation == 'catalog':
            fields = ('kind', 'q', 'fund_type', 'fund_category', 'invest_type', 'market', 'status', 'management',
                      'custodian', 'page', 'page_size', 'sort_by', 'sort_dir', 'snapshot_metrics', 'qdii_type')
            arguments = request.model_dump(include=set(fields))
            arguments['conditions'] = [f'{item.field}|{item.operator}|{item.value}' for item in request.conditions]
            output = _callback(callbacks, 'catalog')(**arguments)
            if not isinstance(output, dict):
                raise _error('AGENT_PRODUCT_UNAVAILABLE', '产品列表当前不可用。')
            result['catalog'] = _project(output, CATALOG_SCHEMA)
            result['catalog']['items'] = result['catalog'].get('items', [])[request.batch_offset:request.batch_offset+10]
            result['catalog']['returned_batch_limit'] = 10
            result['catalog']['page_item_count'] = len(output.get('items') or [])
            result['catalog']['note'] = '列表摘要覆盖全部筛选结果，条目仅本页明确偏移的最多10个；近12月上市数以当前日为基准。快照行情值不进入模型，指标按冻结版本重算。'
        elif targets:
            result['evaluations'] = _evaluate_metrics(service, request, targets=targets, as_of=as_of)
        else:
            result.update(status='unavailable', code='NO_CURRENT_BATCH')
    else:
        if request.source != 'actual':
            raise _error('AGENT_DEMO_NOT_EVIDENCE', '演示比较不能作为真实计算来源；请先选择实际产品。')
        if operation == 'metrics':
            from pit.context import resolve_request_context, set_view_override, reset_view_override
            metric_context = resolve_request_context(service.market_data_dir, request.metrics_as_of)
            token = set_view_override(metric_context)
            try:
                result['evaluations'] = _evaluate_metrics(service, request, targets=targets, as_of=metric_context.as_of)
                result['effective_as_of'] = metric_context.as_of
            finally:
                reset_view_override(token)
        else:
            parameters = request.model_dump(include={'ranges', 'rolling_window_days'})
            result['comparisons'] = []
            for target in request.targets:
                output = _callback(callbacks, 'comparison')(target.model_dump(), parameters)
                if output.get('product_id') != target.product_id:
                    raise _error('AGENT_CONTEXT_CHANGED', '比较结果不属于当前产品。')
                ranges = {}
                for name in ('performance', 'risk', 'efficiency'):
                    row = (output.get('ranges') or {}).get(name) or {}
                    window = _project(row.get('window') or {}, views.WINDOW)
                    ranges[name] = {'window': window}
                    if window.get('observation_count', 0) >= 2:
                        ranges[name]['metrics'] = _project(row.get('metrics') or {}, {key: NUM for key in COMPARE_UNITS})
                    else:
                        ranges[name]['status'] = 'insufficient_observations'
                result['comparisons'].append({'target': {'kind': target.kind, 'product_id': target.product_id},
                    'ranges': ranges, 'metric_units': COMPARE_UNITS,
                    'price_basis': 'etf_close_with_adjusted_nav_fallback' if target.kind == 'etf' else 'fund_adjusted_nav',
                    'source_resolution': 'existing_loader_precedence; exact_etf_dataset_not_reported' if target.kind == 'etf' else 'existing_fund_adjusted_nav_loader',
                    'pit': 'observation_date_cutoff_only; announcement_pit_not_proven',
                    'fees': {'management_fee': target.management_fee, 'custody_fee': target.custody_fee}})
    return result


CATALOG_SCHEMA = {'page': INT, 'page_size': INT, 'total': INT, 'sort_by': STR, 'sort_dir': STR, 'kind': STR,
    'items': [{'ts_code': STR, 'code': STR, 'name': STR, 'instrument_type': STR, 'management': STR,
               'custodian': STR, 'fund_type': STR, 'type': STR, 'invest_type': STR, 'market': STR, 'status': STR,
               'm_fee': NUM, 'c_fee': NUM, 'list_date': STR, 'found_date': STR}],
    'summary': {key: INT for key in ('universe_total', 'filtered_total', 'active_count', 'recent_listings_12m', 'unique_managements')},
    'pit': {'as_of': STR, 'snapshot_is_hindsight': BOOL, 'warnings': [STR]}}
CATALOG_SCHEMA['summary'].update({key: NUM for key in ('active_rate', 'avg_m_fee', 'avg_c_fee', 'avg_exp_return',
    'avg_duration_year', 'total_issue_amount', 'median_issue_amount')})

ANALYSIS_SCHEMA = {'page': STR, 'operation': STR, 'provenance': STR, 'effective_as_of': STR, 'status': STR, 'code': STR,
    'pagination': {'target_count': INT, 'target_offset': INT, 'next_target_offset': INT,
                   'indicator_count': INT, 'indicator_offset': INT, 'next_indicator_offset': INT},
    'source_run_id': STR, 'locked_target_revision': INT,
    'batch': {'count': INT, 'targets': [views.TARGET], 'scope': STR, 'visible_count': INT, 'selected_count': INT,
              'selection_mode': STR, 'selection_provenance': STR, 'offset': INT},
    'catalog': {**CATALOG_SCHEMA, 'returned_batch_limit': INT, 'page_item_count': INT, 'note': STR},
    'evaluations': [{'period': STR, 'results': [views.SCALAR_ROW], 'summary': {'total': INT, 'ok': INT, 'warning': INT, 'unavailable': INT, 'error': INT}}],
    'comparisons': [{'target': views.TARGET,
        'ranges': {name: {'window': views.WINDOW, 'metrics': {key: NUM for key in COMPARE_UNITS}, 'status': STR}
                   for name in ('performance', 'risk', 'efficiency')},
        'metric_units': {key: STR for key in COMPARE_UNITS}, 'price_basis': STR, 'source_resolution': STR, 'pit': STR,
        'fees': {'management_fee': NUM, 'custody_fee': NUM}}],
    'portfolio': {'snapshot': RUN_METADATA, 'source': STR, 'pit': {'source': STR, 'run_mode': STR, 'data_release_id': STR},
        'data_identity': {'fingerprints': {'*': STR}, 'common_date_hash': STR},
        'metrics': {key: NUM for key in PORTFOLIO_UNITS}, 'metric_units': {key: STR for key in PORTFOLIO_UNITS},
        'metrics_omitted': STR, 'component_count': INT, 'omitted_components': INT, 'warnings': [views.WARNING],
        'concentration': {key: NUM for key in ('max_weight', 'top3_weight', 'hhi', 'effective_holdings')},
        'components': [{'kind': STR, 'product_id': STR, 'name': STR, 'current_weight': NUM, 'period_return': NUM, 'simple_return_contribution': NUM}],
        'risk_contributions': [{'product_id': STR, 'name': STR, 'risk_contribution': NUM, 'marginal_risk': NUM, 'component_risk': NUM}]}}


def analysis_view(payload, projection):
    counter = views.Counter()
    result = views.project(payload, ANALYSIS_SCHEMA, 'analysis', counter)
    return ({} if result is views.OMIT else result), counter.dropped
