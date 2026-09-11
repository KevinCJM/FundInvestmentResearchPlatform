import type { TimingCatalog, TimingRun } from '../../services/timingResearch'

export const timingExecutionFixture = { backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { timing: ['readonly float64[::1] -> float64[::1]'] } }
export const timingCatalogFixture: TimingCatalog = {
  operators: [
    { id: 'source', label: '行情字段', description: '读取真实行情字段。', category: 'source', granularity: 'primitive', inputs: [], outputs: [{ name: 'value', label: '数值', type: 'series' }], parameters: [{ name: 'field', label: '行情字段', type: 'string', default: 'close', options: [{ value: 'close', label: '收盘价' }, { value: 'open', label: '开盘价' }] }] },
    { id: 'compare', label: '大于阈值', description: '根据阈值判断条件。', category: 'condition', granularity: 'primitive', inputs: [{ name: 'input', label: '比较序列', type: 'series', required: true }], outputs: [{ name: 'condition', label: '条件', type: 'condition' }], parameters: [{ name: 'threshold', label: '阈值', type: 'number', default: 1, minimum: 0 }] },
  ],
  templates: [{ id: 'trend', label: '均线趋势', description: '可编辑的趋势模板。', definition: { name: '均线趋势研究', description: '观察趋势条件。', nodes: [{ id: 'price', label: '收盘价格', op: 'source', inputs: {}, parameters: { field: 'close' } }, { id: 'signal', label: '买入信号', op: 'compare', inputs: { input: 'price.value' }, parameters: { threshold: 1 } }], entry: 'signal.condition', exit: null, execution: { take_profit: .15, stop_loss: .15, max_holding_bars: 15, cooldown_bars: 0, fee_bps: 3, slippage_bps: 2 } } }],
}
const metrics = { observations: 5, total_return: .04, buy_hold_return: .02, excess_return: .02, max_drawdown: -.01, trade_count: 1, win_rate: 1, exposure: .5 }
export const timingRunFixture: TimingRun = {
  id: 'timing-run-test', name: '均线趋势研究', created_at: '2026-09-10T09:00:00', execution: timingExecutionFixture,
  definition_snapshot: timingCatalogFixture.templates[0].definition,
  request_snapshot: { definition: timingCatalogFixture.templates[0].definition, targets: [{ kind: 'etf', product_id: '510300.SH' }], start_date: '2020-01-01', end_date: '2026-09-03', holdout_start: '2024-01-01', walk_forward_splits: 3, price_basis: 'hfq' },
  restrictions: ['当前历史快照用于研究，尚未认证历史 PIT。'],
  products: [{ product_id: '510300.SH', status: 'ok', detail_loaded: true, summary: { all: metrics, in_sample: { ...metrics, total_return: null }, out_of_sample: metrics }, yearly: [{ period: '2024', ...metrics }], monthly: [{ period: '2024-01', ...metrics }], walk_forward: [{ period: '第1段', ...metrics }], warnings: [], signal_quality: [],
    curve: [{ date: '2024-01-02', close: 1, nav: 1, buy_hold_nav: 1, position: 0, action: 0, reason: 'none' }, { date: '2024-01-03', close: 1.01, nav: 1, buy_hold_nav: 1.01, position: 1, action: 1, reason: 'entry_signal' }, { date: '2024-01-04', close: 1.04, nav: 1.04, buy_hold_nav: 1.04, position: 0, action: -1, reason: 'exit_signal' }],
    trades: [{ signal_date: '2024-01-02', entry_date: '2024-01-03', exit_date: '2024-01-04', entry_price: 1, exit_price: 1.04, net_return: .04, holding_days: 2, reason: 'exit_signal', max_adverse_excursion: -.01, max_favorable_excursion: .04 }],
    channels: [{ id: 'price.value', label: '收盘价格', type: 'series', values: [1, 1.01, 1.04] }, { id: 'signal.condition', label: '买入信号', type: 'condition', values: [-1, 1, 0] }],
  }],
}

// Offline UI fixtures only; none of these values are research evidence.
export const timingTrainingCatalogFixture: TimingCatalog = {
  ...timingCatalogFixture,
  limits: { steps: 128 },
  operators: [...timingCatalogFixture.operators,
    { id: 'basket_source', label: '参考篮子行情', description: '独立选取的 ETF 篮子。', category: 'source', granularity: 'primitive', inputs: [], outputs: [{ name: 'value', label: '篮子价格', type: 'panel' }], parameters: [{ name: 'group', label: '篮子类型', type: 'string', default: 'market', options: [{ value: 'market', label: '市场参考' }, { value: 'category', label: '同类参考' }] }] },
    { id: 'cross_mean', label: '横截面均值', description: '将篮子矩阵转换为数值序列。', category: 'panel', granularity: 'primitive', inputs: [{ name: 'value', label: '篮子输入', type: 'panel', required: true }], outputs: [{ name: 'value', label: '均值', type: 'series' }], parameters: [] },
  ],
  templates: [{ id: 'etf-repair', label: 'ETF 弱月选择改编', description: '训练期选择候选动作，样本外冻结。', definition: {
    ...structuredClone(timingCatalogFixture.templates[0].definition), name: 'ETF 弱月选择改编',
    nodes: [...structuredClone(timingCatalogFixture.templates[0].definition.nodes), { id: 'basket', label: '独立市场篮子', op: 'basket_source', inputs: {}, parameters: { group: 'market' } }, { id: 'basket_mean', label: '篮子均价', op: 'cross_mean', inputs: { value: 'basket.value' }, parameters: {} }],
    adaptation: { source_experiments: ['A2074'], version: 'etf-v1', preserved: ['弱状态时替代或空仓。'], changed: ['使用 ETF 候选规则，不读取 A 股母池。', '训练期按同自然月选方案，样本外冻结不滚动重训。'] },
    training: { mode: 'month', state_refs: [], actions: [{ id: 'trend', label: '趋势候选', entry: 'signal.condition' }, { id: 'cash', label: '空仓', entry: null }], search_space: [{ label: '阈值对照', choices: [[{ node: 'signal', parameter: 'threshold', value: 1 }], [{ node: 'signal', parameter: 'threshold', value: 1.02 }]] }], min_trades: 5, confidence: 1, risk_penalty: .1, min_utility: 0, embargo_bars: 0 },
  } }],
}
