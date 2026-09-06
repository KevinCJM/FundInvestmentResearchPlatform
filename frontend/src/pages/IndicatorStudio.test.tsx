import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import IndicatorStudio from './IndicatorStudio'
import type { IndicatorDefinition, IndicatorMeta, IndicatorOperator, IndicatorVariable, SnapshotIndicatorConfig } from '../services/customIndicators'

type MockGraphOption = { series: Array<{ layout: string; edgeSymbol: string[]; data: Array<{ id: unknown; y: number; name: string }>; links: Array<{ source: unknown; target: unknown; parameter?: string }> }> }

vi.mock('echarts-for-react', () => ({
  default: ({ option, 'aria-label': ariaLabel }: { option: MockGraphOption; 'aria-label'?: string }) => {
    const graph = option.series[0]
    return <div aria-label={ariaLabel} data-testid="dag-chart" data-layout={graph.layout} data-directed={String((graph.data[0]?.y ?? 0) < (graph.data[graph.data.length - 1]?.y ?? 0))} data-arrow={graph.edgeSymbol[1]} data-link-count={graph.links.length} data-node-id-type={typeof graph.data[0]?.id} data-edge-id-type={typeof graph.links[0]?.source} data-edge-parameter={graph.links[0]?.parameter || ''} />
  },
}))

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { typed_indicator_plan: ['fixed'] },
}

const variables: IndicatorVariable[] = [
  { name: 'returns', label: '复权净值普通收益率', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{r}', shape: 'series', semantic: '由相邻复权净值计算的普通收益率。', semantic_role: 'ordinary_return', measure: 'return_decimal', price_basis: 'adjusted_nav', source: '真实复权净值派生', source_dataset: 'fund_nav', source_field: 'adj_nav', data_basis: '复权净值', frequency: '交易日', unit: '小数', domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'returns', category_label: '收益与变化', availability: 'available' },
  { name: 'log_returns', label: '复权净值对数收益率', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{\\ell}', shape: 'series', semantic: '由相邻复权净值计算的对数收益率。', semantic_role: 'log_return', measure: 'return_decimal', price_basis: 'adjusted_nav', source: '真实复权净值派生', source_dataset: 'fund_nav', source_field: 'adj_nav', data_basis: '复权净值', frequency: '交易日', unit: '小数', domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'returns', category_label: '收益与变化', availability: 'available' },
  { name: 'adjusted_nav', label: '复权净值', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{p}_{\\mathrm{adj}}', shape: 'series', semantic: '产品在计算窗口内的复权净值。', semantic_role: 'adjusted_nav_level', measure: 'adjusted_nav', price_basis: 'adjusted_nav', source: 'Tushare 本地 Parquet', source_dataset: 'fund_nav', source_field: 'adj_nav', data_basis: '后复权', frequency: '交易日', unit: '净值', domains: ['single_product'], category_id: 'price', category_label: '净值与价格', availability: 'available' },
  { name: 'market_open', label: '开盘价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{o}', shape: 'series', semantic: 'ETF 未复权日 K 开盘价。', semantic_role: 'raw_open_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', source_dataset: 'candle', source_field: 'open', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_high', label: '最高价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{h}', shape: 'series', semantic: 'ETF 未复权日 K 最高价。', semantic_role: 'raw_high_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', source_dataset: 'candle', source_field: 'high', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_low', label: '最低价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{l}', shape: 'series', semantic: 'ETF 未复权日 K 最低价。', semantic_role: 'raw_low_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', source_dataset: 'candle', source_field: 'low', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_close', label: '收盘价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{c}', shape: 'series', semantic: 'ETF 未复权日 K 收盘价。', semantic_role: 'raw_close_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', source_dataset: 'candle', source_field: 'close', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'volume', label: '成交量', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{v}', shape: 'series', semantic_role: 'trading_volume', measure: 'volume', source: 'Tushare ETF 日线', source_dataset: 'candle', source_field: 'vol', unit: '原始单位', domains: ['single_product'], product_kinds: ['etf'], category_id: 'trading', category_label: '成交与流动性' },
  { name: 'risk_free_rate_per_observation', label: '单观察期无风险收益率', value_type: 'scalar', dtype: 'float64', latex: 'r_f', shape: 'scalar', semantic: '按当前观察频率换算的无风险收益率。', source: '指标配置', domains: ['single_product', 'portfolio'], category_id: 'configuration', category_label: '基准与配置' },
  { name: 'asset_returns', label: '多资产普通收益矩阵', value_type: 'matrix<time,asset>', dtype: 'float64', latex: '\\mathbf{R}', shape: 'matrix', domains: ['portfolio'], semantic_role: 'ordinary_return_matrix', category_id: 'portfolio', category_label: '组合上下文' },
  { name: 'asset_weights', label: '资产权重向量', value_type: 'vector<asset>', dtype: 'float64', latex: '\\mathbf{w}', shape: 'vector', domains: ['portfolio'], semantic_role: 'weight', category_id: 'portfolio', category_label: '组合上下文' },
]

const operators: IndicatorOperator[] = [
  { name: 'sequence_std', label: '全元素标准差', signature: 'std(values, ddof) → scalar', latex_template: '\\operatorname{std}', return_type: 'scalar', output_shape: 'scalar', mathematical_essence: '计算数值序列全部元素的标准差。', semantic: '纯数学离散程度归约。', category_id: 'statistics', category_label: '统计归约', version: '2.1.0', cost_estimate: 'O(T)', parameters: [{ name: 'values', label: '输入值', description: '待归约的一维数值序列。', allowed_shapes: ['series'] }, { name: 'ddof', label: '自由度修正', description: '样本标准差通常为 1。', allowed_shapes: ['scalar'], default: 1, optional: true }] },
  { name: 'identity_series', label: '逐元素绝对值', signature: 'abs(values) → same(values)', latex_template: '\\operatorname{abs}', return_type: 'series<time>', output_shape: 'series', mathematical_essence: '逐元素计算绝对值。', semantic: '保持时间轴不变。', category_id: 'transform', category_label: '逐元素变换', parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['series'] }] },
  { name: 'add', label: '逐元素加法', signature: 'A + B → same(A,B)', latex_template: 'A+B', return_type: 'same(first)', output_shape: 'unknown', mathematical_essence: '两个兼容数值输入逐元素相加。', semantic: '只允许标量广播。', category_id: 'arithmetic', category_label: '基础运算', parameters: [{ name: 'lhs', label: '输入 A', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }, { name: 'rhs', label: '输入 B', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }] },
  { name: 'subtract', label: '逐元素减法', signature: 'A - B → same(A,B)', latex_template: 'A-B', return_type: 'same(first)', output_shape: 'unknown', mathematical_essence: '两个兼容数值输入逐元素相减。', semantic: '只允许标量广播。', category_id: 'arithmetic', category_label: '基础运算', cost_estimate: 'elementwise', parameters: [{ name: 'lhs', label: '输入 A', description: '允许类型：scalar | series<T> | vector<N> | matrix<A,B>。', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }, { name: 'rhs', label: '输入 B', description: '允许类型：scalar | series<T> | vector<N> | matrix<A,B>。', allowed_shapes: ['scalar', 'series', 'vector', 'matrix'] }] },
  { name: 'product', label: '累乘', signature: 'product(values) → scalar', latex_template: '\\prod_i x_i', display_latex_template: '\\prod_i x_i', return_type: 'scalar', output_shape: 'scalar', mathematical_essence: '将输入序列的所有数值依次相乘。', semantic: '对全部元素执行乘法归约。', category_id: 'reduction', category_label: '统计归约', parameters: [{ name: 'values', label: '待处理数值', allowed_shapes: ['series', 'vector'] }] },
  { name: 'covariance', label: '协方差', signature: 'matrix<T,N> → matrix<N,N> / series<T>, series<T> → scalar', latex_template: '\\operatorname{Cov}', return_type: 'matrix<asset,asset> | scalar', output_shape: 'unknown', domains: ['single_product', 'portfolio'], mathematical_essence: '计算协方差矩阵或两条序列的协方差。', semantic: '按输入签名推导输出。', category_id: 'statistics', category_label: '统计计算', parameters: [{ name: 'asset_returns', label: '多资产收益矩阵', allowed_shapes: ['matrix'] }], parameter_sets: [{ arity: 1, parameters: [{ name: 'asset_returns', label: '多资产收益矩阵', allowed_shapes: ['matrix'] }], output: 'matrix<asset,asset>' }, { arity: 2, parameters: [{ name: 'lhs', label: '输入 A', allowed_shapes: ['series'] }, { name: 'rhs', label: '输入 B', allowed_shapes: ['series'] }], output: 'scalar' }] },
  { name: 'diag', label: '对角构造或提取', signature: 'vector<N> → matrix<N,N> / matrix<N,N> → vector<N>', latex_template: '\\operatorname{diag}', return_type: 'matrix<asset,asset> | vector<asset>', output_shape: 'unknown', domains: ['portfolio'], mathematical_essence: '从向量构造对角矩阵，或提取矩阵对角线。', semantic: '输入可以是资产向量或方阵。', category_id: 'linear_algebra', category_label: '线性代数', parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['vector'] }], parameter_sets: [{ arity: 1, parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['vector'] }], output: 'matrix<asset,asset>' }, { arity: 1, parameters: [{ name: 'values', label: '输入值', allowed_shapes: ['matrix'] }], output: 'vector<asset>' }] },
  { name: 'matvec', label: '矩阵向量乘', signature: 'matrix<time,asset> × vector<asset> → series<time>', latex_template: '\\operatorname{matvec}', return_type: 'series<time>', output_shape: 'series', domains: ['portfolio'], mathematical_essence: '矩阵与资产向量相乘。', semantic: '资产轴必须对齐。', category_id: 'linear_algebra', category_label: '线性代数', parameters: [{ name: 'matrix', label: '矩阵', allowed_shapes: ['matrix'] }, { name: 'vector', label: '向量', allowed_shapes: ['vector'] }] },
]

const rollingMeanOperator: IndicatorOperator = {
  name: 'rolling_mean',
  label: '滚动平均值',
  signature: 'rolling_mean(values, window) → series<time>',
  latex_template: '\\operatorname{rolling\\_mean}(x,w)',
  display_latex_template: '\\left(\\frac{1}{w}\\sum_{i=t-w+1}^{t}x_i\\right)_{t=1}^{T}',
  return_type: 'series<time>',
  output_shape: 'series',
  mathematical_essence: '按固定窗口计算时间序列的滚动算术平均值。',
  semantic: '保持时间轴不变，窗口未满时返回缺失值。',
  category_id: 'rolling',
  category_label: '滚动与时序',
  domains: ['single_product'],
  parameters: [
    { name: 'values', label: '输入序列', allowed_shapes: ['series'] },
    {
      name: 'window',
      label: '窗口期数',
      allowed_shapes: ['scalar'],
      source_policy: 'fixed_constant',
      constant_kind: 'integer',
      default: 20,
      minimum: 1,
      maximum: 20_000,
    },
  ],
}

const meta: IndicatorMeta = {
  engine_version: 'test', workspace_scope: 'shared', limits: {}, variables, operators,
  periods: [{ value: '1M', label: '近 1 月', description: '自然月窗口' }, { value: '1Y', label: '近 1 年', description: '自然年窗口' }],
  templates: [], predefined_calculations: [], indicator_types: [{ id: 'risk', label: '风险型指标' }, { id: 'return', label: '收益型指标' }, { id: 'risk_adjusted', label: '风险调整指标' }, { id: 'technical', label: '技术与时序指标' }, { id: 'other', label: '其他指标' }],
}

const builtIn: IndicatorDefinition = {
  id: 'builtin-volatility', revision: 1, source: 'built_in', read_only: true, created_at: '2026-01-01', updated_at: '2026-01-01',
  name: '收益波动率', description: '收益率序列的样本标准差。', expression: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', periods: ['1M'], unit: '%', display_format: 'percent', precision: 2, direction: 'lower_better', annual_risk_free_rate_percent: 1.5,
  display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)',
  context_kind: 'single_product', dsl_version: '2.0.0', operator_registry_version: '2.0.0', output_contract: 'scalar',
  category_id: 'risk', category_label: '风险型指标',
  indicator_type: 'risk',
}

const portfolioBuiltIn: IndicatorDefinition = {
  ...builtIn,
  id: 'builtin-portfolio-volatility',
  name: '组合波动率',
  description: '锁定组合运行的波动率。',
  expression: '\\operatorname{std}\\left(\\operatorname{matvec}\\left(\\mathbf{R},\\mathbf{w}\\right),1\\right)',
  context_kind: 'portfolio',
}

const reusableBuiltIn: IndicatorDefinition = {
  ...builtIn,
  id: 'builtin-total-return-v2',
  name: '累计收益率',
  description: '窗口内普通收益率逐期复合后的累计收益。',
  expression: 'product(returns + 1) - 1',
  display_latex: '\\prod_t(1+r_t)-1',
  dsl_version: '2.1.0',
  operator_registry_version: '2.1.0',
  variable_registry_version: '2.1.0',
  data_contract_version: 'tushare-eod-v2',
  context_schema_version: 'typed-context-v2',
  indicator_type: 'return',
  category_id: 'return',
  category_label: '收益型指标',
  output_measure: 'return_decimal',
}

const rollingSharpeSource: IndicatorDefinition = {
  ...reusableBuiltIn,
  id: 'builtin-annualized-sharpe-v2',
  name: '年化夏普比率',
  description: '平均超额收益与样本标准差之比，再乘以年化因子的平方根。',
  expression: '((mean(returns) - risk_free_rate_per_observation) / std(returns, 1)) * sqrt(periods_per_year)',
  display_latex: '\\frac{\\bar r-r_f}{s_r}\\sqrt{p_{year}}',
  indicator_type: 'risk_adjusted',
  category_id: 'risk_adjusted',
  category_label: '风险调整指标',
  output_measure: 'dimensionless',
  unit: '',
  display_format: 'number',
  precision: 3,
  direction: 'higher_better',
  rolling_series_compatibility: {
    supported: true,
    protocol_version: '1.0.0',
    rewritten_reductions: ['mean', 'std'],
  },
}

const workspaceIndicator: IndicatorDefinition = {
  ...builtIn,
  id: 'workspace-volatility',
  source: 'custom',
  read_only: false,
  name: '我的波动指标',
  revision: 3,
}

const timeSeriesBuiltIn: IndicatorDefinition = {
  ...builtIn,
  id: 'builtin-close-moving-average-series',
  name: '20 日收盘价均线',
  description: '对真实收盘价执行固定 20 个交易日窗口的简单移动平均。',
  expression: 'rolling_mean(market_close, 20)',
  result_kind: 'time_series',
  output_contract: 'series_bundle',
  output_measure: 'series_bundle',
  indicator_type: 'technical',
  category_id: 'technical',
  category_label: '技术与时序指标',
  unit: '元',
  display_format: 'number',
  precision: 4,
  direction: 'higher_better',
  annual_risk_free_rate_percent: 0,
  dsl_version: '2.3.0',
  operator_registry_version: '2.3.0',
  variable_registry_version: '2.1.0',
  data_contract_version: 'tushare-eod-v2',
  context_schema_version: 'typed-context-v2',
  parameter_schema: [],
  fixed_parameters: [],
  series_outputs: [{
    id: 'ma',
    label: '20 日收盘价均线',
    expression: 'rolling_mean(market_close, 20)',
    unit: '元',
    display_format: 'number',
    precision: 4,
    output_measure: 'auto',
    inferred_output_measure: 'raw_market_price',
    resolved_output_measure: 'raw_market_price',
    output_measure_source: 'inferred',
    semantic_dimension: 'raw_market_price',
    price_basis: 'raw_market',
    value_range: null,
  }],
  axis_anchor: 'market_close',
  history_policy: 'lookback',
  history_inference_source: 'typed_dag',
  lookback_parameter: null,
  lookback_observations: 20,
  minimum_observations: 20,
  methodology: '在每个时点使用截至当日最近 20 个有限收盘价计算简单移动平均。',
  data_basis: 'ETF 未复权日 K 收盘价。',
  applicable_product_kinds: ['etf'],
  required_variables: ['market_close'],
  display_latex: null,
}

const rollingSharpeTimeSeriesBuiltIn: IndicatorDefinition = {
  ...timeSeriesBuiltIn,
  id: 'builtin-rolling-5d-annualized-sharpe-series',
  name: '5 日滚动年化夏普比率',
  description: '对每个时点最近 5 个有效收益观察值计算样本标准差口径的年化夏普比率。',
  expression: '(rolling_mean(returns, 5) - risk_free_rate_per_observation) / rolling_std(returns, 5, 1) * sqrt(periods_per_year)',
  indicator_type: 'risk_adjusted',
  category_id: 'risk_adjusted',
  category_label: '风险调整指标',
  unit: '',
  display_format: 'number',
  precision: 3,
  annual_risk_free_rate_percent: 1.5,
  series_outputs: [{
    id: 'value',
    label: '5 日滚动年化夏普比率',
    expression: '(rolling_mean(returns, 5) - risk_free_rate_per_observation) / rolling_std(returns, 5, 1) * sqrt(periods_per_year)',
    unit: '',
    display_format: 'number',
    precision: 3,
    output_measure: 'auto',
    inferred_output_measure: 'dimensionless',
    resolved_output_measure: 'dimensionless',
    output_measure_source: 'inferred',
    semantic_dimension: 'dimensionless',
    price_basis: null,
    value_range: null,
  }],
  axis_anchor: 'adjusted_nav',
  lookback_observations: 5,
  minimum_observations: 5,
  methodology: '最近 5 个普通收益率的平均超额收益除以样本标准差，再乘年化因子平方根。',
  data_basis: '复权净值普通收益率；日期对齐，缺失不填充。',
  applicable_product_kinds: ['etf', 'fund'],
  required_variables: ['returns', 'risk_free_rate_per_observation', 'periods_per_year'],
  rolling_source: {
    kind: 'rolling_scalar',
    transform_version: '1.0.0',
    indicator_id: rollingSharpeSource.id,
    indicator_revision: rollingSharpeSource.revision,
    indicator_name: rollingSharpeSource.name,
    definition_hash: 'd'.repeat(64),
    source_dsl_version: rollingSharpeSource.dsl_version || '2.1.0',
    window_observations: 5,
    minimum_observations: 5,
    detached: false,
  },
  rolling_transform: {
    version: '1.0.0',
    window_observations: 5,
    source_expression: rollingSharpeSource.expression,
    generated_expression: '(rolling_mean(returns, 5) - risk_free_rate_per_observation) / rolling_std(returns, 5, 1) * sqrt(periods_per_year)',
    series_variables: ['returns'],
    reduction_mappings: ['mean', 'std'],
  },
  display_latex: null,
}

const rollingSharpeSeriesDag = {
  nodes: [
    { id: 'returns', label: 'returns', kind: 'variable', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: '\\mathbf{r}' },
    { id: 'window', label: '5', kind: 'constant', value_type: 'scalar<count>', shape: 'scalar' as const, symbolic_shape: [], latex_fragment: '5', formula_fragment: '5' },
    { id: 'ddof', label: '1', kind: 'constant', value_type: 'scalar<count>', shape: 'scalar' as const, symbolic_shape: [], latex_fragment: '1', formula_fragment: '1' },
    { id: 'mean', label: 'rolling_mean', operator_id: 'rolling_mean', kind: 'call', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: '\\overline{r}_{t,5}' },
    { id: 'rf', label: 'risk_free_rate_per_observation', kind: 'variable', value_type: 'scalar<rate_decimal>', shape: 'scalar' as const, symbolic_shape: [], latex_fragment: 'r_f' },
    { id: 'std', label: 'rolling_std', operator_id: 'rolling_std', kind: 'call', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: 's_{t,5}' },
    { id: 'root', label: 'multiply', operator_id: 'multiply', kind: 'binary', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: '\\frac{\\overline{r}_{t,5}-r_f}{s_{t,5}}\\sqrt{p_{\\mathrm{year}}}' },
  ],
  edges: [
    { source: 'returns', target: 'mean', parameter: 'values', order: 0 },
    { source: 'window', target: 'mean', parameter: 'window', order: 1 },
    { source: 'returns', target: 'std', parameter: 'values', order: 0 },
    { source: 'window', target: 'std', parameter: 'window', order: 1 },
    { source: 'ddof', target: 'std', parameter: 'ddof', order: 2 },
    { source: 'mean', target: 'root', parameter: 'lhs', order: 0 },
    { source: 'std', target: 'root', parameter: 'rhs', order: 1 },
  ],
  roots: { value: 'root' },
}

const reusableDag = {
  nodes: [
    { id: 'returns', label: 'returns', kind: 'variable', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: '\\mathbf{r}' },
    { id: 'one-a', label: '1', kind: 'constant', value_type: 'scalar', shape: 'scalar' as const, symbolic_shape: [], latex_fragment: '1', formula_fragment: '1' },
    { id: 'add', label: 'add', operator_id: 'add', kind: 'binary', value_type: 'series<time>', shape: 'series' as const, symbolic_shape: ['T'], latex_fragment: '\\mathbf{r}+1' },
    { id: 'product', label: 'product', operator: { id: 'product', version: '2.2.0' }, kind: 'call', inferred_type: { kind: 'scalar', display: 'scalar', shape: [] }, latex_fragment: '\\prod_i(1+r_i)' },
    { id: 'one-b', label: '1', kind: 'constant', value_type: 'scalar', shape: 'scalar' as const, symbolic_shape: [], latex_fragment: '1', formula_fragment: '1' },
    { id: 'result', label: 'subtract', operator: { id: 'subtract', version: '2.2.0' }, kind: 'binary', inferred_type: { kind: 'scalar', display: 'scalar', shape: [] }, latex_fragment: '\\prod_i(1+r_i)-1' },
  ],
  edges: [
    { source: 'returns', target: 'add', parameter: 'lhs', order: 0 },
    { source: 'one-a', target: 'add', parameter: 'rhs', order: 1 },
    { source: 'add', target: 'product', parameter: 'values', order: 0 },
    { source: 'product', target: 'result', parameter: 'lhs', order: 0 },
    { source: 'one-b', target: 'result', parameter: 'rhs', order: 1 },
  ],
  roots: { result: 'result' },
}

const timeSeriesDag = {
  nodes: [
    {
      id: 'market-close',
      label: 'market_close',
      kind: 'variable',
      value_type: 'series<time>',
      shape: 'series' as const,
      symbolic_shape: ['T'],
      formula_fragment: 'market_close',
      latex_fragment: '\\mathbf{c}',
    },
    {
      id: 'window',
      label: '20',
      kind: 'constant',
      value_type: 'scalar',
      shape: 'scalar' as const,
      symbolic_shape: [],
      formula_fragment: '20',
      latex_fragment: '20',
    },
    {
      id: 'ma-root',
      label: 'rolling_mean',
      operator: { id: 'rolling_mean', version: '2.3.0' },
      operator_id: 'rolling_mean',
      kind: 'call',
      inferred_type: {
        kind: 'series',
        display: 'series<time>',
        shape: ['T'],
        semantic_dimension: 'raw_market_price',
        price_basis: 'raw_market',
      },
      value_type: 'series<time>',
      shape: 'series' as const,
      symbolic_shape: ['T'],
      formula_fragment: 'rolling_mean(market_close, 20)',
      latex_fragment: '\\left(\\frac{1}{N_t}\\sum_{i=\\max(1,t-20+1)}^t c_i\\right)_{t=1}^T',
      inputs: ['market-close', 'window'],
      arguments: [
        { name: 'values', input_node_id: 'market-close' },
        { name: 'window', input_node_id: 'window' },
      ],
    },
  ],
  edges: [
    { source: 'market-close', target: 'ma-root', parameter: 'values', order: 0 },
    { source: 'window', target: 'ma-root', parameter: 'window', order: 1 },
  ],
  roots: { ma: 'ma-root' },
}

function json(body: unknown) {
  return { ok: true, status: 200, json: async () => body }
}

function apiError(message: string) {
  return { ok: false, status: 422, json: async () => ({ detail: { code: 'COMPOSE_FAILED', message } }) }
}

async function renderStudio(initialEntry = '/indicator-studio') {
  await act(async () => {
    render(<MemoryRouter initialEntries={[initialEntry]}><IndicatorStudio /></MemoryRouter>)
    await Promise.resolve()
    await Promise.resolve()
  })
  await screen.findByRole('button', { name: /收益波动率/ })
  await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/custom-indicators/meta', expect.anything()))
}

function setupUser() {
  const user = userEvent.setup()
  return {
    click: async (...args: Parameters<typeof user.click>) => { await act(async () => { await user.click(...args) }) },
    selectOptions: async (...args: Parameters<typeof user.selectOptions>) => { await act(async () => { await user.selectOptions(...args) }) },
    clear: async (...args: Parameters<typeof user.clear>) => { await act(async () => { await user.clear(...args) }) },
    type: async (...args: Parameters<typeof user.type>) => { await act(async () => { await user.type(...args) }) },
    keyboard: async (...args: Parameters<typeof user.keyboard>) => { await act(async () => { await user.keyboard(...args) }) },
  }
}

type TestUser = ReturnType<typeof setupUser>

async function openCatalog(user: TestUser) {
  await user.click(screen.getByRole('button', { name: '浏览公式构建资源' }))
  return screen.findByRole('dialog', { name: '变量、算子与已有指标' })
}

async function chooseCombobox(user: TestUser, name: string, query: string) {
  await user.click(screen.getByRole('combobox', { name }))
  const search = await screen.findByLabelText(`搜索${name}`)
  await user.clear(search)
  await user.type(search, query)
  await user.keyboard('{Enter}')
}

describe('IndicatorStudio', () => {
  let catalog: IndicatorDefinition[]
  let activeMeta: IndicatorMeta
  let composeFailure = false
  let validateAsSeries = false
  let snapshotConfig: SnapshotIndicatorConfig

  beforeEach(() => {
    catalog = [builtIn, reusableBuiltIn, rollingSharpeSource, portfolioBuiltIn]
    activeMeta = meta
    composeFailure = false
    validateAsSeries = false
    snapshotConfig = {
      schema_version: 1,
      revision: 1,
      max_items: 30,
      updated_at: null,
      snapshot: null,
      items: [{ indicator_id: reusableBuiltIn.id, indicator_revision: 1, period: '1M', field: 'return_1m', name: reusableBuiltIn.name, status: 'ready' }],
    }
    vi.stubGlobal('confirm', vi.fn(() => true))
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/custom-indicators/meta') return json(activeMeta)
      if (url === '/api/custom-indicators?context_kind=single_product') {
        const items = catalog.filter((item) => (item.context_kind ?? 'single_product') === 'single_product')
        return json({ items, total: items.length })
      }
      if (url === '/api/custom-indicators/snapshot-config') {
        if (init?.method === 'PUT') {
          const body = JSON.parse(String(init.body))
          snapshotConfig = {
            ...snapshotConfig,
            revision: snapshotConfig.revision + 1,
            items: body.items.map((item: { indicator_id: string; indicator_revision: number; period: string }) => ({
              ...item,
              field: `metric_${item.period.toLowerCase()}`,
              name: catalog.find((indicator) => indicator.id === item.indicator_id)?.name,
              status: 'ready',
            })),
          }
        }
        return json(snapshotConfig)
      }
      if (url === '/api/custom-indicators') {
        if (init?.method === 'POST') {
          const body = JSON.parse(String(init.body))
          const created = { ...builtIn, ...body, id: 'custom-volatility', source: 'custom', read_only: false, revision: 1 }
          catalog = [...catalog, created]
          return json(created)
        }
        return json({ items: catalog, total: catalog.length })
      }
      if (url === '/api/custom-indicators/derive-rolling-series') {
        const body = JSON.parse(String(init?.body || '{}'))
        const source = catalog.find((item) => item.id === body.indicator_id) ?? rollingSharpeSource
        const window = Number(body.window_observations)
        const expression = `(rolling_mean(returns, ${window}) - risk_free_rate_per_observation) / rolling_std(returns, ${window}, 1) * sqrt(252)`
        const definition = {
          ...timeSeriesBuiltIn,
          name: `${window} 日滚动${source.name}`,
          description: `对每个时点最近 ${window} 个有效观察值应用锁定的“${source.name}”标量公式。`,
          expression,
          indicator_type: source.indicator_type,
          category_id: source.indicator_type,
          category_label: source.category_label,
          direction: source.direction,
          annual_risk_free_rate_percent: source.annual_risk_free_rate_percent,
          unit: source.unit,
          precision: source.precision,
          axis_anchor: 'adjusted_nav',
          lookback_observations: window,
          minimum_observations: window,
          fixed_parameters: [{ id: 'window_observations', label: '滚动观察数', type: 'integer', value: window, source: 'rolling_source' }],
          series_outputs: [{
            id: 'value',
            label: `${window} 日滚动${source.name}`,
            expression,
            unit: source.unit,
            display_format: source.display_format,
            precision: source.precision,
            output_measure: 'auto',
            inferred_output_measure: 'dimensionless',
            resolved_output_measure: 'dimensionless',
            output_measure_source: 'inferred',
            semantic_dimension: 'dimensionless',
            price_basis: null,
            value_range: null,
          }],
          required_variables: ['returns', 'risk_free_rate_per_observation'],
          applicable_product_kinds: ['etf', 'fund'],
          rolling_source: {
            kind: 'rolling_scalar',
            transform_version: '1.0.0',
            indicator_id: source.id,
            indicator_revision: source.revision,
            indicator_name: source.name,
            definition_hash: 'b'.repeat(64),
            source_dsl_version: source.dsl_version,
            window_observations: window,
            minimum_observations: window,
            detached: false,
          },
          rolling_transform: { version: '1.0.0', rewritten_reductions: ['mean', 'std'], source_variables: ['returns'] },
        }
        return json({
          definition,
          validation: {
            valid: true,
            diagnostics: [],
            dependencies: ['returns', 'risk_free_rate_per_observation'],
            dag: null,
            result_kind: 'time_series',
            output_contract: 'series_bundle',
            output_channels: definition.series_outputs,
            output_inferences: {},
            parameter_schema: [],
            fixed_parameters: definition.fixed_parameters,
            history_policy: 'lookback',
            history_inference_source: 'typed_dag',
            lookback_observations: window,
            minimum_observations: window,
            compile_token: 'c'.repeat(64),
            compile_token_scope: 'current_process_warm_cache',
            execution: fixedExecution,
          },
          source: {
            indicator_id: source.id,
            indicator_revision: source.revision,
            indicator_name: source.name,
            window_observations: window,
          },
        })
      }
      if (url === '/api/custom-indicators/validate' && validateAsSeries) return json({
        valid: false,
        diagnostics: [{ code: 'OUTPUT_CONTRACT_MISMATCH', message: '输出契约要求 scalar，实际推断为 series<time>[T]。', expected: 'scalar', actual: 'series<time>[T]' }],
        dependencies: ['returns'],
        python_expression: 'returns',
        dag: null,
      })
      if (url === '/api/custom-indicators/validate') {
        const body = JSON.parse(String(init?.body || '{}'))
        if (body.result_kind === 'time_series') {
          const output = body.series_outputs?.[0]
          if (output?.id === 'value' && String(output.expression).includes('rolling_mean(returns, 5)')) return json({
            valid: true,
            diagnostics: [],
            dependencies: ['returns', 'risk_free_rate_per_observation', 'periods_per_year'],
            python_expression: null,
            dag: rollingSharpeSeriesDag,
            result_kind: 'time_series',
            output_contract: 'series_bundle',
            output_channels: body.series_outputs,
            output_inferences: {
              value: {
                id: 'value',
                label: '5 日滚动年化夏普比率',
                expression: output.expression,
                latex: output.expression,
                display_latex: '\\frac{\\mu_{t,5}\\left(\\mathbf r\\right)-r_f}{s_{t,5}\\left(\\mathbf r\\right)}\\cdot\\sqrt{p_{\\mathrm{year}}}',
                math_notation_version: '1.4.0',
                python_expression: output.expression,
                inferred_type: 'series<time>',
                shape: 'series',
                semantic_dimension: 'dimensionless',
                price_basis: null,
                output_measure: 'auto',
                inferred_output_measure: 'dimensionless',
                resolved_output_measure: 'dimensionless',
                output_measure_source: 'inferred',
                value_range: null,
                dependencies: ['returns', 'risk_free_rate_per_observation', 'periods_per_year'],
                root_id: 'root',
              },
            },
            parameter_schema: [],
            fixed_parameters: body.fixed_parameters ?? [],
            history_policy: 'lookback',
            history_inference_source: 'typed_dag',
            lookback_observations: 5,
            minimum_observations: 5,
            compile_token: 'e'.repeat(64),
            compile_token_scope: 'current_process_warm_cache',
            execution: fixedExecution,
          })
          return json({
          valid: true,
          diagnostics: [],
          dependencies: ['market_close'],
          python_expression: null,
          dag: timeSeriesDag,
          result_kind: 'time_series',
          output_contract: 'series_bundle',
          output_channels: body.series_outputs,
          output_inferences: {
            ma: {
              id: 'ma',
              label: '20 日收盘价均线',
              expression: 'rolling_mean(market_close, 20)',
              latex: 'rolling_mean(market_close, 20)',
              display_latex: '\\mu_{t,20}\\left(\\mathbf c\\right)',
              math_notation_version: '1.4.0',
              python_expression: 'rolling_mean(market_close, 20)',
              inferred_type: 'series<time>',
              shape: 'series',
              semantic_dimension: 'raw_market_price',
              price_basis: 'raw_market',
              output_measure: 'auto',
              inferred_output_measure: 'raw_market_price',
              resolved_output_measure: 'raw_market_price',
              output_measure_source: 'inferred',
              value_range: null,
              dependencies: ['market_close'],
              root_id: 'ma-root',
            },
          },
          parameter_schema: [],
          fixed_parameters: [],
          history_policy: 'lookback',
          history_inference_source: 'typed_dag',
          lookback_observations: 20,
          minimum_observations: 20,
          compile_token: 'a'.repeat(64),
          compile_token_scope: 'current_process_warm_cache',
          execution: fixedExecution,
        })
        }
        return json({
        valid: true, diagnostics: [], dependencies: ['returns'], python_expression: 'sequence_std(returns, 1)',
        display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)',
        dag: {
          nodes: [
            { id: 'input', label: 'returns', kind: 'variable', value_type: 'series<time>', shape: 'series', symbolic_shape: ['T'] },
            { id: 'root', label: 'sequence_std', kind: 'call', value_type: 'scalar', shape: 'scalar', symbolic_shape: [] },
          ],
          edges: [{ source: 'input', target: 'root' }],
          roots: { result: 'root' },
        },
      })
      }
      if (url === '/api/custom-indicators/compose') {
        if (composeFailure) return apiError('服务端拒绝展开，请检查参数。')
        const body = JSON.parse(String(init?.body))
        if (body.indicator_id === reusableBuiltIn.id) return json({ expression: reusableBuiltIn.expression, latex: reusableBuiltIn.expression, display_latex: reusableBuiltIn.display_latex, inferred_type: 'scalar', shape: 'scalar', semantic_warnings: [], dependencies: ['returns'], dag: reusableDag, indicator_origin: { indicator_id: reusableBuiltIn.id, indicator_revision: reusableBuiltIn.revision, name: reusableBuiltIn.name, source: reusableBuiltIn.source } })
        if (body.operator_id === 'identity_series') return json({ expression: '\\operatorname{abs}\\left(\\mathbf{r}\\right)', latex: '\\operatorname{abs}\\left(\\mathbf{r}\\right)', display_latex: '\\left\\lvert\\mathbf{r}\\right\\rvert', inferred_type: 'series<time>', shape: 'series', semantic_warnings: [] })
        if (body.operator_id === 'sequence_std') return json({ expression: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', latex: '\\operatorname{std}\\left(\\mathbf{r},1\\right)', display_latex: '\\operatorname{Std}_{\\mathrm{ddof}=1}\\left(\\mathbf{r}\\right)', inferred_type: 'scalar', shape: 'scalar', semantic_warnings: [] })
        return json({ expression: '\\left(\\mathbf{r}+1\\right)', latex: '\\left(\\mathbf{r}+1\\right)', display_latex: '\\left(\\mathbf{r}\\right)+\\left(1\\right)', inferred_type: 'series<time>', shape: 'series', semantic_warnings: [] })
      }
      if (url === '/api/custom-indicators/variables/availability') {
        const body = JSON.parse(String(init?.body))
        const requestedTargets = body.targets || [{ kind: body.kind, product_id: body.product_id }]
        return json({
          targets: requestedTargets.map((target: { kind: string; product_id: string }) => ({
            target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id },
          })),
          period: body.period,
          as_of: body.as_of || null,
          items: body.variable_ids.map((variableId: string) => {
            const definition = activeMeta.variables.find((variable) => variable.name === variableId)
            const targetStatuses = requestedTargets.map((target: { kind: 'etf' | 'fund'; product_id: string }) => {
              const applicable = !definition?.product_kinds || definition.product_kinds.includes(target.kind)
              const kindLabel = target.kind === 'etf' ? 'ETF' : '场外公募基金'
              return {
                target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id },
                status: applicable ? 'available' : 'source_unavailable',
                reason: applicable ? null : { code: 'SOURCE_UNAVAILABLE_FOR_PRODUCT', message: `${kindLabel}没有可供计算“${definition?.label || variableId}”的真实数据源。` },
              }
            })
            const availableCount = targetStatuses.filter((item) => item.status === 'available').length
            const status = availableCount === targetStatuses.length ? 'available' : availableCount ? 'partial' : 'source_unavailable'
            return {
              variable_id: variableId,
              status,
              coverage: { coverage_ratio: 0.98 },
              actual_shape: availableCount ? (variableId === 'risk_free_rate_per_observation' ? [] : [250]) : null,
              reason: status === 'available' ? null : { code: 'VARIABLE_AVAILABILITY_SUMMARY', message: `所选产品中 ${availableCount}/${targetStatuses.length} 个可使用该变量。` },
              target_statuses: targetStatuses,
              window: { requested_as_of: body.as_of || null, effective_as_of: '2026-08-28', start_date: '2025-08-28', end_date: '2026-08-28', observation_count: 249, data_latest_date: '2026-08-28' },
            }
          }),
        })
      }
      if (url === '/api/portfolio-runs') return json({ items: [{ id: 'run-001', target_name: '稳健组合', target_revision: 7, effective_as_of: '2026-08-28', window: { start_date: '2024-01-02', end_date: '2026-08-28' } }] })
      if (url.startsWith('/api/instruments/search')) {
        const query = new URL(url, 'http://localhost').searchParams.get('q') || ''
        const second = query === '512960.SH'
        const code = second ? '512960.SH' : '510300.SH'
        return json({ items: [{ code, ts_code: code, name: second ? '证券ETF' : '沪深300ETF', management: '测试', found_date: '2026-01-01', instrument_type: 'etf' }], total: 1, page: 1, page_size: 30, kind: 'etf' })
      }
      if (url === '/api/custom-indicators/export-excel') {
        return {
          ok: true,
          status: 200,
          headers: new Headers({
            'Content-Disposition': "attachment; filename*=UTF-8''indicator-calculation.xlsx",
          }),
          blob: async () => new Blob(['xlsx-content'], {
            type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          }),
        }
      }
      if (url === '/api/custom-indicators/evaluate') {
        const body = JSON.parse(String(init?.body))
        const targets = body.targets as Array<{ kind: 'etf' | 'fund'; product_id: string }>
        return json({
          results: targets.map((target) => ({ indicator_id: null, indicator_revision: null, indicator_name: '收益波动率', target: { ...target, name: target.product_id === '510300.SH' ? '沪深300ETF' : target.product_id }, period: body.period, value: 0.1234, status: 'ok', warnings: [], window: { requested_as_of: body.as_of || null, effective_as_of: '2026-08-28', start_date: '2025-08-28', end_date: '2026-08-28', observation_count: 250, data_latest_date: '2026-08-28' } })),
          summary: { total: targets.length, ok: targets.length, warning: 0, error: 0 },
          cache: { hits: 0, misses: targets.length },
          execution: fixedExecution,
        })
      }
      if (url === '/api/custom-indicators/evaluate-portfolio') return json({ results: [{ indicator_id: null, indicator_revision: null, indicator_name: '组合波动率', target: { kind: 'portfolio', product_id: 'run-001', name: '稳健组合' }, period: 'snapshot', value: 0.087, status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: '2026-08-28', start_date: '2024-01-02', end_date: '2026-08-28', observation_count: 640, data_latest_date: '2026-08-28' } }], summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 }, execution: fixedExecution })
      return json({})
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('解析与校验合并为资源按钮下方的单一操作', async () => {
    const user = setupUser()
    await renderStudio()

    const browseButton = screen.getByRole('button', { name: '浏览公式构建资源' })
    const combinedButton = screen.getByRole('button', { name: '解析并校验公式' })
    expect(browseButton.compareDocumentPosition(combinedButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.queryByRole('button', { name: '解析公式' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '校验公式' })).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(combinedButton)
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/validate')).toHaveLength(1))
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/custom-indicators/infer')).toBe(false)
  })

  it('为指标配置数据刷新后的快照预计算区间', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /累计收益率/ }))
    await user.click(screen.getByText('快照加速'))
    const oneYear = screen.getByRole('checkbox', { name: /近 1 年（1Y）/ })
    expect(oneYear).not.toBeChecked()
    await user.click(oneYear)
    await user.click(screen.getByRole('button', { name: '保存快照配置' }))

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/custom-indicators/snapshot-config',
      expect.objectContaining({ method: 'PUT' }),
    ))
    expect(await screen.findByText(/下一次数据刷新后生成预计算结果/)).toBeInTheDocument()
  })

  it('定义时不选择周期，保存不提交 periods，单产品预览仍显式选择运行周期', async () => {
    const user = setupUser()
    await renderStudio()

    expect(screen.queryByText('可用周期')).not.toBeInTheDocument()
    expect(screen.getByText(/指标定义默认支持全部计算周期/)).toBeInTheDocument()
    expect(screen.getByLabelText('计算周期')).toHaveTextContent('近 1 月')
    expect(screen.getByLabelText('计算周期')).toHaveTextContent('近 1 年')

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getAllByRole('button', { name: '复制为新指标' })[0])
    await waitFor(() => expect(fetch).toHaveBeenCalledWith('/api/custom-indicators', expect.objectContaining({ method: 'POST' })))
    const createCall = vi.mocked(fetch).mock.calls.find(([url, init]) => url === '/api/custom-indicators' && init?.method === 'POST')
    const validateCall = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/validate')
    expect(JSON.parse(String((createCall?.[1] as RequestInit).body))).not.toHaveProperty('periods')
    expect(JSON.parse(String((validateCall?.[1] as RequestInit).body))).not.toHaveProperty('periods')
  })

  it('校验与预览保留深链中的多个产品，并在一次请求中批量计算', async () => {
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH%2C512960.SH')

    expect(await screen.findByText('2 / 10')).toBeInTheDocument()
    expect(screen.queryByText(/仅保留第一个已选产品/)).not.toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/custom-indicators/variables/availability')).toBe(false)
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/evaluate')
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).targets).toEqual([
        { kind: 'etf', product_id: '510300.SH' },
        { kind: 'etf', product_id: '512960.SH' },
      ])
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).include_series).toBe(false)
    })
    expect(await screen.findByText('预览完成：2 个成功，0 个需关注。')).toBeInTheDocument()
    expect(screen.getAllByText('512960.SH').length).toBeGreaterThan(0)
  })

  it('在预览按钮下方下载包含直接入参与 Excel 公式的工作簿', async () => {
    const nativeUrl = URL
    const createObjectURL = vi.fn(() => 'blob:indicator-excel')
    const revokeObjectURL = vi.fn()
    class DownloadURL extends nativeUrl {
      static createObjectURL = createObjectURL
      static revokeObjectURL = revokeObjectURL
    }
    vi.stubGlobal('URL', DownloadURL)
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH')
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))

    const previewButton = screen.getByRole('button', { name: '预览指标' })
    const downloadButton = screen.getByRole('button', { name: '下载 Excel 计算逻辑' })
    expect(previewButton.compareDocumentPosition(downloadButton) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    await user.click(downloadButton)

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/export-excel')
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body))).toMatchObject({
        targets: [{ kind: 'etf', product_id: '510300.SH' }],
        period: '1Y',
      })
    })
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    expect(anchorClick).toHaveBeenCalled()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:indicator-excel')
    expect(await screen.findByText(/Excel 已生成：包含 1 个产品/)).toBeInTheDocument()
    anchorClick.mockRestore()
  })

  it('可以锁定标量指标版本并生成固定观察数的滚动时序指标草稿', async () => {
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /年化夏普比率/ }))
    await user.click(screen.getByRole('button', { name: '生成滚动时序指标' }))
    const rollingPanel = screen.getByRole('region', { name: '标量指标滚动派生' })
    const sourceSelect = within(rollingPanel).getByLabelText('滚动来源标量指标')
    expect(sourceSelect).toHaveValue('builtin-annualized-sharpe-v2')
    expect(within(sourceSelect).getByRole('option', { name: '年化夏普比率 · v1' })).toBeInTheDocument()
    expect(within(rollingPanel).getByLabelText('滚动观察数')).toHaveValue(5)

    await user.click(within(rollingPanel).getByRole('button', { name: '生成滚动公式' }))
    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/derive-rolling-series')
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body))).toEqual({
        indicator_id: 'builtin-annualized-sharpe-v2',
        indicator_revision: 1,
        window_observations: 5,
      })
    })

    expect(screen.getByLabelText('名称')).toHaveValue('5 日滚动年化夏普比率')
    expect(await screen.findByText(/来源定义哈希、版本和生成公式已锁定/)).toBeInTheDocument()
    expect(screen.getByText(/已从“年化夏普比率”v1 生成 5 日滚动时序草稿/)).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    expect(screen.getByLabelText('“5 日滚动年化夏普比率”公式源码（DSL / LaTeX）')).toHaveValue(
      '(rolling_mean(returns, 5) - risk_free_rate_per_observation) / rolling_std(returns, 5, 1) * sqrt(252)',
    )
  })

  it('5 日滚动年化夏普点击校验后展示数学 LaTeX', async () => {
    catalog = [...catalog, rollingSharpeTimeSeriesBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(
      screen.getByRole('button', { name: /5 日滚动年化夏普比率/ }),
    )
    await user.click(
      screen.getByRole('button', { name: '解析并校验全部通道' }),
    )

    expect(
      await screen.findByText(/时序定义校验通过，共 1 个输出通道/),
    ).toBeInTheDocument()
    const preview = screen.getByTestId('formula-preview')
    expect(preview.querySelector('.katex')).not.toBeNull()
    expect(preview).not.toHaveTextContent('rolling_mean')
    expect(preview).not.toHaveTextContent('rolling_std')
    expect(preview).not.toHaveTextContent('NaN')
    expect(preview).not.toHaveTextContent('∑')
    expect(preview).toHaveTextContent('μ')
    expect(screen.getByText('dimensionless / 无')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '查看完整计算说明' }))
    const explanation = screen.getByRole('region', { name: '公式计算说明' })
    expect(within(explanation).getByText('使用“滚动平均值”')).toBeInTheDocument()
    expect(within(explanation).getByText('使用“滚动标准差”')).toBeInTheDocument()
    expect(explanation).toHaveTextContent('自由度修正')
    expect(explanation).not.toHaveTextContent('rolling_mean')
    expect(explanation).not.toHaveTextContent('rolling_std')
  })

  it('时序指标支持构建向导、LaTeX、嵌套说明和公式驱动 Excel 导出', async () => {
    catalog = [...catalog, timeSeriesBuiltIn]
    activeMeta = {
      ...meta,
      operators: [...operators, rollingMeanOperator],
      indicator_types: meta.indicator_types,
    }
    const nativeUrl = URL
    const createObjectURL = vi.fn(() => 'blob:series-indicator-excel')
    const revokeObjectURL = vi.fn()
    class DownloadURL extends nativeUrl {
      static createObjectURL = createObjectURL
      static revokeObjectURL = revokeObjectURL
    }
    vi.stubGlobal('URL', DownloadURL)
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
    const user = setupUser()

    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH')
    await user.click(screen.getAllByRole('button', { name: /收盘价均线/ })[0])

    expect(screen.getByRole('tab', { name: '构建向导' })).toBeInTheDocument()
    expect(screen.getByRole('tab', { name: '高级公式模式' })).toBeInTheDocument()
    expect(screen.queryByLabelText('时序图表区域')).not.toBeInTheDocument()
    expect(screen.getByText(/图表位置由具体使用页面决定/)).toBeInTheDocument()
    const measureSelect = screen.getByLabelText('输出通道 1 输出量纲')
    expect(measureSelect).toHaveValue('auto')
    expect(within(measureSelect).getByRole('option', { name: '自动推断' })).toBeInTheDocument()
    expect(within(measureSelect).getByRole('option', { name: '原始市价' })).toBeInTheDocument()
    expect(within(measureSelect).getByRole('option', { name: '虚拟净值（起点 1）' })).toBeInTheDocument()
    expect(within(measureSelect).getByRole('option', { name: '0～1 区间' })).toBeInTheDocument()
    expect(within(measureSelect).getByRole('option', { name: '-1～1 区间' })).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '解析并校验全部通道' }))
    expect(await screen.findByText(/时序定义校验通过，共 1 个输出通道；当前通道已解析为可编辑的嵌套计算步骤/)).toBeInTheDocument()
    expect(screen.getByText('计算逻辑已解析')).toBeInTheDocument()
    const formulaPreview = screen.getByTestId('formula-preview')
    expect(formulaPreview.querySelector('.katex')).not.toBeNull()
    expect(formulaPreview).not.toHaveTextContent('rolling_mean')
    expect(screen.getAllByText('原始市价').length).toBeGreaterThan(0)
    expect(screen.getByText('raw_market_price / raw_market')).toBeInTheDocument()
    expect(screen.getAllByText('20 个').length).toBeGreaterThanOrEqual(2)

    await user.click(screen.getByRole('button', { name: '查看完整计算说明' }))
    const explanation = screen.getByRole('region', { name: '公式计算说明' })
    expect(within(explanation).getAllByText('滚动平均值').length).toBeGreaterThan(0)
    expect(explanation).toHaveTextContent('窗口期数：常量 20')
    expect(within(explanation).queryByText(/运行参数/)).not.toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '浏览公式构建资源' }))
    const composer = await screen.findByRole('dialog', { name: '编辑“20 日收盘价均线”的计算逻辑' })
    expect(within(composer).getByLabelText('窗口期数 有限常量')).toHaveValue(20)
    expect(within(composer).getByLabelText('窗口期数 输入来源')).toBeDisabled()
    expect(within(composer).getByText(/必须随指标版本固定/)).toBeInTheDocument()
    await user.click(within(composer).getByRole('button', { name: '关闭参数配置' }))

    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    expect(screen.getByLabelText('“20 日收盘价均线”公式源码（DSL / LaTeX）')).toHaveValue(
      'rolling_mean(market_close, 20)',
    )

    await user.click(screen.getByRole('button', { name: '下载 Excel 计算逻辑' }))
    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/export-excel')
      const body = JSON.parse(String((calls.at(-1)?.[1] as RequestInit).body))
      expect(body).toMatchObject({
        period: '1Y',
        targets: [{ kind: 'etf', product_id: '510300.SH' }],
        inline_definition: {
          result_kind: 'time_series',
          output_contract: 'series_bundle',
          axis_anchor: 'market_close',
          parameter_schema: [],
          history_policy: 'lookback',
          lookback_parameter: null,
          series_outputs: [{
            expression: 'rolling_mean(market_close, 20)',
            output_measure: 'auto',
          }],
        },
      })
      expect(body).not.toHaveProperty('parameters')
      expect(body.inline_definition).not.toHaveProperty('chart_panel')
    })
    expect(await screen.findByText(/逐步 Excel 公式和各时序通道结果/)).toBeInTheDocument()
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:series-indicator-excel')
    anchorClick.mockRestore()
  })

  it('指标中心不提供滚动曲线入口，预览只请求单窗口结果', async () => {
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=etf&ids=510300.SH')
    expect(screen.queryByRole('checkbox', { name: /同时计算滚动曲线/ })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/evaluate')
      const body = JSON.parse(String((calls[calls.length - 1]?.[1] as RequestInit).body))
      expect(body.include_series).toBe(false)
    })
  })

  it('搜索添加产品时追加选择而不是替换已有产品', async () => {
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.type(screen.getByLabelText('搜索产品'), '512960.SH')
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))

    expect(await screen.findByText('2 / 10')).toBeInTheDocument()
    expect(screen.getByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.getAllByText('证券ETF').length).toBeGreaterThan(0)
  })

  it('产品搜索下拉在焦点离开搜索控件后自动收起', async () => {
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: '搜索' }))
    expect(await screen.findByRole('listbox', { name: '搜索产品结果' })).toBeInTheDocument()

    await user.click(screen.getByLabelText('计算周期'))
    expect(screen.queryByRole('listbox', { name: '搜索产品结果' })).not.toBeInTheDocument()
  })

  it('产品搜索下拉允许操作内部结果，并支持 Esc 收起后返回输入框', async () => {
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    expect(screen.getByRole('listbox', { name: '搜索产品结果' })).toBeInTheDocument()

    await user.keyboard('{Escape}')
    expect(screen.queryByRole('listbox', { name: '搜索产品结果' })).not.toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: '搜索产品' })).toHaveFocus()
  })

  it('深链产品超过十个时只保留前十个', async () => {
    const ids = Array.from({ length: 12 }, (_, index) => `TEST${String(index + 1).padStart(2, '0')}.SH`)
    await renderStudio(`/indicator-studio?kind=etf&ids=${encodeURIComponent(ids.join(','))}`)

    expect(await screen.findByText('10 / 10')).toBeInTheDocument()
    expect(screen.getByText('校验与预览最多选择 10 个产品，已保留前 10 个。')).toBeInTheDocument()
    expect(screen.queryByText('TEST11.SH')).not.toBeInTheDocument()
    expect(screen.queryByText('TEST12.SH')).not.toBeInTheDocument()
  })

  it('指标库支持按关键词、来源和分类筛选', async () => {
    const user = setupUser()
    await renderStudio()

    await user.type(screen.getByLabelText('搜索指标'), '不存在的指标')
    expect(screen.queryByRole('button', { name: /收益波动率/ })).not.toBeInTheDocument()
    await user.clear(screen.getByLabelText('搜索指标'))
    await user.selectOptions(screen.getByLabelText('指标分类'), 'risk')
    expect(screen.getByRole('button', { name: /收益波动率/ })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('指标来源'), 'custom')
    expect(screen.getByText('当前筛选条件下没有指标。')).toBeInTheDocument()
  })

  it('大型目录通过资源类型、分类和条目三级选择，并支持键盘搜索', async () => {
    const extraVariables: IndicatorVariable[] = Array.from({ length: 240 }, (_, index) => ({
      name: `market_field_${index}`,
      label: `成交变量 ${index}`,
      value_type: 'series<time>',
      dtype: 'float64',
      latex: `x_{${index}}`,
      shape: 'series',
      semantic: `第 ${index} 个原始成交数据字段。`,
      source: 'Tushare 日线数据',
      category_id: index % 2 ? 'trading' : 'price',
      category_label: index % 2 ? '成交与流动性' : '净值与价格',
      domains: ['single_product'],
    }))
    activeMeta = { ...meta, variables: [...variables, ...extraVariables] }
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)

    expect(within(dialog).getByLabelText('资源类型')).toHaveValue('variables')
    await chooseCombobox(user, '资源分类', '成交与流动性')
    await chooseCombobox(user, '选择变量', '成交变量 217')
    expect(within(dialog).getByRole('heading', { name: '成交变量 217' })).toBeInTheDocument()
    expect(within(dialog).getByText(/Tushare 日线数据/)).toBeInTheDocument()
    expect(within(dialog).getAllByText('时间序列').length).toBeGreaterThan(0)
    expect(within(dialog).queryByText(/series<time>|\[T\]/)).not.toBeInTheDocument()
  })

  it('构建目录不按深链产品禁用变量，收益率名称明确复权净值口径', async () => {
    const user = setupUser()
    await renderStudio('/indicator-studio?kind=fund&ids=005300.OF')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/custom-indicators/variables/availability')).toBe(false)
    await user.click(screen.getByText('浏览公式构建资源'))
    const dialog = (await screen.findByText('变量、算子与已有指标')).closest<HTMLElement>('[role="dialog"]') as HTMLElement
    const chooseCatalogOption = async (label: string, query: string) => {
      const combobox = within(dialog).getAllByLabelText(label).find((element) => element.getAttribute('role') === 'combobox') as HTMLElement
      if (combobox.getAttribute('aria-expanded') !== 'true') await user.click(combobox)
      const search = within(dialog).getByLabelText(`搜索${label}`)
      await user.click(search)
      await user.type(search, query)
      await user.keyboard('{Enter}')
    }

    await chooseCatalogOption('资源分类', '净值与价格')
    expect(within(dialog).getByLabelText('资源分类')).toHaveTextContent('净值与价格（5）')
    const variableCombobox = within(dialog).getAllByLabelText('选择变量').find((element) => element.getAttribute('role') === 'combobox') as HTMLElement
    await user.click(variableCombobox)
    const priceOptions = variableCombobox.parentElement?.querySelector<HTMLElement>('[role="listbox"]')
    expect(priceOptions).not.toBeNull()
    expect(within(priceOptions).getByText('开盘价')).toBeInTheDocument()
    expect(within(priceOptions).getByText('最高价')).toBeInTheDocument()
    expect(within(priceOptions).getByText('最低价')).toBeInTheDocument()
    expect(within(priceOptions).getByText('收盘价')).toBeInTheDocument()
    await user.click(within(priceOptions).getByText('开盘价'))
    expect(within(dialog).getByText('开盘价', { selector: 'h4' })).toBeInTheDocument()
    expect(within(dialog).getByText('ETF')).toBeInTheDocument()
    expect(within(dialog).queryByText(/场外公募基金没有可供计算“开盘价”的真实数据源/)).not.toBeInTheDocument()
    expect(within(dialog).queryByText('所选产品可用性')).not.toBeInTheDocument()
    expect(within(dialog).getByRole('button', { name: '设为当前公式' })).toBeEnabled()

    await chooseCatalogOption('资源分类', '收益与变化')
    const returnCombobox = within(dialog).getAllByLabelText('选择变量').find((element) => element.getAttribute('role') === 'combobox') as HTMLElement
    await user.click(returnCombobox)
    const returnOptions = returnCombobox.parentElement?.querySelector<HTMLElement>('[role="listbox"]')
    expect(returnOptions).not.toBeNull()
    expect(within(returnOptions).getByText('复权净值普通收益率')).toBeInTheDocument()
    expect(within(returnOptions).getByText('复权净值对数收益率')).toBeInTheDocument()
  })

  it('输出结果与输出契约错误只展示中文数据类型', async () => {
    validateAsSeries = true
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '解析并校验公式' }))
    expect((await screen.findAllByText('时间序列')).length).toBeGreaterThan(0)
    expect(screen.getByText('能否作为指标')).toBeInTheDocument()
    expect(screen.getByText(/还需要使用求和、平均值、标准差等归约计算/)).toBeInTheDocument()
    expect(screen.queryByLabelText('推导后的数学公式')).not.toBeInTheDocument()
    expect(screen.queryByText(/series<time>|\[T\]/)).not.toBeInTheDocument()

    const editorPanel = within(document.getElementById('indicator-panel-editor')!)
    expect(await editorPanel.findByText('校验未通过')).toBeInTheDocument()
    expect(editorPanel.getByText(/指标结果类型不符合要求/)).toBeInTheDocument()
    expect(editorPanel.getByText(/指标最终结果必须是单个有限数值/)).toBeInTheDocument()
    expect(editorPanel.getByText(/需要 有限标量/)).toBeInTheDocument()
    expect(editorPanel.getByText(/当前是 时间序列/)).toBeInTheDocument()
    expect(screen.queryByText(/OUTPUT_CONTRACT_MISMATCH|series<time>|\[T\]|输出契约要求 scalar/)).not.toBeInTheDocument()
  })

  it('算子目录与参数配置只展示中文数据要求', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素减法')

    expect(within(catalogDialog).getByLabelText('算子输入输出说明')).toHaveTextContent('输入 A支持有限标量、时间序列、资产向量、矩阵')
    expect(within(catalogDialog).getByText('逐元素计算')).toBeInTheDocument()
    expect(within(catalogDialog).queryByText(/same\(|scalar|series<|vector<|matrix<|elementwise/)).not.toBeInTheDocument()

    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    const composer = await screen.findByRole('dialog', { name: '配置 逐元素减法' })
    expect(within(composer).getByText(/输入 A支持有限标量/)).toBeInTheDocument()
    expect(within(composer).queryByText(/same\(|scalar|series<|vector<|matrix<|elementwise/)).not.toBeInTheDocument()
  })

  it('目录包含变量、计算算子和已有指标，不出现预定义计算或公式片段', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    const resourceType = within(dialog).getByLabelText('资源类型')
    expect(resourceType).toHaveTextContent('变量')
    expect(resourceType).toHaveTextContent('计算算子')
    expect(resourceType).toHaveTextContent('已有指标')
    expect(resourceType).not.toHaveTextContent('预定义计算')
    expect(resourceType).not.toHaveTextContent('公式片段')
    expect(within(dialog).queryByText('公式片段')).not.toBeInTheDocument()
  })

  it('已有指标按锁定版本展开为独立公式', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.selectOptions(within(dialog).getByLabelText('资源类型'), 'indicators')
    await chooseCombobox(user, '资源分类', '收益型指标')
    await chooseCombobox(user, '选择已有指标', '累计收益率')

    expect(within(dialog).getByText('内置指标 · 锁定 v1')).toBeInTheDocument()
    expect(within(dialog).getByText(/原指标以后更新不会改变本草稿/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: '查看计算说明' }))
    const explanation = await within(dialog).findByRole('region', { name: '公式计算说明' })
    expect(within(explanation).getByText('输入变量与符号')).toBeInTheDocument()
    expect(within(explanation).getByText('算子与数学符号')).toBeInTheDocument()
    expect(within(explanation).getByText('累乘')).toBeInTheDocument()
    expect(within(explanation).getByText('逐元素加法')).toBeInTheDocument()
    expect(within(explanation).getByLabelText('累乘的数学符号').querySelector('.katex')).not.toBeNull()
    await user.click(within(dialog).getByRole('button', { name: '展开为当前公式' }))

    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url, init]) => {
        if (url !== '/api/custom-indicators/compose') return false
        const body = JSON.parse(String((init as RequestInit).body))
        return body.indicator_id === reusableBuiltIn.id
      })
      expect(call).toBeDefined()
      expect(JSON.parse(String((call?.[1] as RequestInit).body))).toMatchObject({
        indicator_id: reusableBuiltIn.id,
        indicator_revision: 1,
        context: 'single_product',
        dsl_version: '2.3.0',
        operator_registry_version: '2.3.0',
      })
    })
    expect(screen.getByText(/“累计收益率” v1 已按锁定版本展开为当前公式/)).toBeInTheDocument()
    expect(screen.getByText(/公式已由结构化构建器生成/)).toBeInTheDocument()
    expect(screen.queryByText(reusableBuiltIn.expression)).not.toBeInTheDocument()
    expect(screen.getByTestId('formula-preview').querySelector('.katex')).not.toBeNull()
  })

  it('公式源码与数学排版分离，预览不把下划线标识符渲染成下标', async () => {
    const payoffIndicator: IndicatorDefinition = {
      ...builtIn,
      id: 'builtin-payoff-ratio',
      name: '平均盈亏比',
      expression: 'mean_where(returns, greater_than(returns, 0)) / absolute(mean_where(returns, less_than(returns, 0)))',
      display_latex: '\\frac{\\mathbb{E}[\\mathbf{r}\\mid\\mathbf{r}>0]}{\\left\\lvert\\mathbb{E}[\\mathbf{r}\\mid\\mathbf{r}<0]\\right\\rvert}',
    }
    catalog = [builtIn, payoffIndicator, portfolioBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /平均盈亏比/ }))
    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    expect(screen.getByLabelText('受限公式源码（DSL / LaTeX）')).toHaveValue(payoffIndicator.expression)
    const preview = screen.getByTestId('formula-preview')
    expect(preview.innerHTML).not.toContain('mean_where')
    expect(preview.innerHTML).not.toContain('greater_than')

    await user.type(screen.getByLabelText('受限公式源码（DSL / LaTeX）'), '+1')
    expect(screen.getByTestId('formula-preview')).toHaveTextContent('请先解析并校验公式以生成数学排版')
    expect(screen.getByTestId('formula-preview').innerHTML).not.toContain('mean_where')
  })

  it('非法数学排版不会以红色 LaTeX 源码泄露到预览区', async () => {
    const invalidNotation = {
      ...builtIn,
      display_latex: String.raw`\left(\mathbf{r}\right)_t_i`,
    }
    catalog = [invalidNotation, reusableBuiltIn, portfolioBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    const preview = screen.getByTestId('formula-preview')
    expect(preview).toHaveTextContent('公式排版不可用')
    expect(preview.querySelector('.katex-error')).toBeNull()
  })

  it('算子参数仅提供兼容变量、有限常量和嵌套算子，并明确应用方式', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 全元素标准差' })
    const source = within(composer).getByLabelText('输入值 输入来源')
    expect(source).toHaveTextContent('兼容变量')
    expect(source).toHaveTextContent('有限数值常量')
    expect(source).toHaveTextContent('嵌套算子')
    expect(source).toHaveTextContent('已有标量指标')
    expect(source).not.toHaveTextContent('当前公式')
    expect(within(composer).getByLabelText('应用方式')).toHaveValue('replace_formula')

    await user.selectOptions(source, 'operator')
    expect(within(composer).getByLabelText('输入值 嵌套算子')).toHaveValue('identity_series')
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))

    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')).toHaveLength(2))
    const composeCalls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')
    const parentBody = JSON.parse(String((composeCalls[1][1] as RequestInit).body))
    expect(parentBody.arguments[0]).toMatchObject({ parameter: 'values', source: 'expression' })
    expect(await screen.findByText('输出结果')).toBeInTheDocument()
  })

  it('引导构建允许分别切换输入，并在参数组合层校验类型与语义量纲', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素加法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素加法' })
    const firstInput = within(composer).getByLabelText('输入 A 变量')
    const secondInput = within(composer).getByLabelText('输入 B 变量')
    expect(within(secondInput).getByRole('option', { name: /复权净值普通收益率/ })).not.toBeDisabled()
    expect(within(secondInput).getByRole('option', { name: /成交量/ })).not.toBeDisabled()

    await user.selectOptions(firstInput, 'volume')
    expect(within(composer).getByText(/输入 A 与 输入 B.*不兼容/)).toBeInTheDocument()
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeDisabled()

    await user.selectOptions(secondInput, 'volume')
    expect(within(composer).queryByText(/输入 A 与 输入 B.*不兼容/)).not.toBeInTheDocument()

    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))
    const composeCall = vi.mocked(fetch).mock.calls.find(([url], index) => (
      url === '/api/custom-indicators/compose'
      && index >= 0
    ))
    const composeBody = JSON.parse(String((composeCall?.[1] as RequestInit).body))
    expect(composeBody.arguments).toEqual([
      { parameter: 'lhs', source: 'variable', value: 'volume' },
      { parameter: 'rhs', source: 'variable', value: 'volume' },
    ])
    expect(screen.queryByText(/缺少参数/)).not.toBeInTheDocument()
  })

  it('算子参数可以复用已有标量指标并在提交前展开', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素加法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素加法' })
    await user.selectOptions(within(composer).getByLabelText('输入 A 输入来源'), 'indicator')
    await user.selectOptions(within(composer).getByLabelText('输入 B 输入来源'), 'indicator')
    expect(within(composer).getByLabelText('输入 A 已有指标')).toHaveValue(`${reusableBuiltIn.id}@1`)
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
    await user.click(within(composer).getByRole('button', { name: '展开到公式' }))

    await waitFor(() => {
      const calls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')
      expect(calls).toHaveLength(3)
      const parent = JSON.parse(String((calls[2][1] as RequestInit).body))
      expect(parent.arguments).toEqual([
        { parameter: 'lhs', source: 'expression', value: reusableBuiltIn.expression },
        { parameter: 'rhs', source: 'expression', value: reusableBuiltIn.expression },
      ])
    })
  })

  it('逐元素减法允许先选择净值，再完成同口径配对', async () => {
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '基础运算')
    await chooseCombobox(user, '选择计算算子', '逐元素减法')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 逐元素减法' })
    const firstInput = within(composer).getByLabelText('输入 A 变量')
    const secondInput = within(composer).getByLabelText('输入 B 变量')
    expect(within(firstInput).getByRole('option', { name: /^复权净值 ·/ })).not.toBeDisabled()
    expect(within(secondInput).getByRole('option', { name: /^复权净值 ·/ })).not.toBeDisabled()

    await user.selectOptions(firstInput, 'adjusted_nav')
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeDisabled()
    expect(within(composer).getByText(/输入 A 与 输入 B.*不兼容/)).toBeInTheDocument()

    await user.selectOptions(secondInput, 'adjusted_nav')
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
  })

  it('单产品指标中心只提供产品可用的算子签名', async () => {
    const user = setupUser()
    await renderStudio()
    let catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计计算')
    await chooseCombobox(user, '选择计算算子', '协方差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    let composer = await screen.findByRole('dialog', { name: '配置 协方差' })
    expect(within(composer).getByLabelText('输入 A 变量')).toHaveValue('returns')
    expect(within(composer).getByLabelText('输入 B 变量')).toHaveValue('returns')
    expect(within(composer).queryByLabelText('多资产收益矩阵 变量')).not.toBeInTheDocument()
    await user.click(within(composer).getByRole('button', { name: '取消' }))

    catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await user.click(within(catalogDialog).getByRole('combobox', { name: '资源分类' }))
    expect(within(catalogDialog).queryByRole('option', { name: /线性代数/ })).not.toBeInTheDocument()
  })

  it('公式编辑会清除旧推导、校验、DAG 与运行结果，并提示重新校验', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '解析并校验公式' }))
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))
    expect((await screen.findAllByText('12.34%')).length).toBeGreaterThan(0)
    expect(screen.getByText('输出结果')).toBeInTheDocument()
    expect(screen.getByText('校验通过')).toBeInTheDocument()
    expect(screen.getByTestId('dag-chart')).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: '高级公式模式' }))
    await user.type(screen.getByLabelText('受限公式源码（DSL / LaTeX）'), '+1')

    expect(screen.queryByText('输出结果')).not.toBeInTheDocument()
    expect(screen.queryByText('校验通过')).not.toBeInTheDocument()
    expect(screen.queryByTestId('dag-chart')).not.toBeInTheDocument()
    expect(screen.queryByText('12.34%')).not.toBeInTheDocument()
    expect(screen.getByText('公式已更改，需要重新解析并校验。')).toBeInTheDocument()
  })

  it('DAG 始终使用单一 result 根，运行周期只更新展示上下文', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '解析并校验公式' }))

    expect(await screen.findByText('校验通过')).toBeInTheDocument()
    expect(screen.getByText('已识别公式的计算结构')).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '公式解析摘要' })).toHaveTextContent('1 个输入变量 · 1 个计算步骤')
    expect(screen.queryByRole('region', { name: '公式计算说明' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '查看完整计算说明' }))
    const explanation = screen.getByRole('region', { name: '公式计算说明' })
    expect(within(explanation).getByText('输入变量与符号')).toBeInTheDocument()
    expect(within(explanation).getByText('算子与数学符号')).toBeInTheDocument()
    expect(within(explanation).getByText('逐步计算')).toBeInTheDocument()
    expect(within(explanation).getByLabelText('全元素标准差的数学符号').querySelector('.katex')).not.toBeNull()
    expect(within(explanation).getByText('使用“全元素标准差”')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '编辑计算步骤' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '浏览公式构建资源' }))
    const parsedComposer = await screen.findByRole('dialog', { name: '编辑“收益波动率”的计算逻辑' })
    expect(within(parsedComposer).getByLabelText('自由度修正 有限常量')).toHaveValue(1)
    expect(within(parsedComposer).getByRole('button', { name: '应用逻辑修改' })).toBeInTheDocument()
    expect(within(parsedComposer).getByRole('button', { name: '改用其他构建资源' })).toBeInTheDocument()
    await user.click(within(parsedComposer).getByRole('button', { name: '取消' }))
    const chart = screen.getByTestId('dag-chart')
    expect(chart).toHaveAttribute('data-layout', 'none')
    expect(chart).toHaveAttribute('data-directed', 'true')
    expect(chart).toHaveAttribute('data-arrow', 'arrow')
    expect(chart).toHaveAttribute('data-link-count', '1')
    expect(chart).toHaveAttribute('data-edge-parameter', '输入值')
    expect(screen.getByText('预览周期：1Y')).toBeInTheDocument()
    const table = screen.getByRole('table', { name: 'DAG 节点、输入参数和输出类型数据表' })
    expect(table).toHaveTextContent('输入值 ← 复权净值普通收益率')
    expect(table).toHaveTextContent('全元素标准差')
    expect(screen.getByRole('button', { name: '适配并复位图谱' })).toBeInTheDocument()
    const nodeDetails = screen.getByRole('article', { name: 'DAG 节点详情' })
    expect(nodeDetails).toHaveTextContent('全元素标准差')
    expect(nodeDetails).toHaveTextContent('理论规模：单个数值')
    expect(table).toHaveTextContent('理论规模：随计算窗口变化的时间点数量')
    await user.click(within(table).getByRole('button', { name: '复权净值普通收益率' }))
    expect(nodeDetails).toHaveTextContent('复权净值普通收益率')

    await user.selectOptions(screen.getByLabelText('计算周期'), '1M')
    expect(screen.getByText('预览周期：1M')).toBeInTheDocument()
    expect(screen.getByTestId('dag-chart')).toHaveAttribute('data-link-count', '1')
  })

  it('已有工作区指标从统一入口载入当前逻辑，修改后可另存为新指标', async () => {
    catalog = [builtIn, reusableBuiltIn, workspaceIndicator, portfolioBuiltIn]
    const user = setupUser()
    await renderStudio()

    await user.click(screen.getByText('我的波动指标'))
    expect(screen.getAllByText('保存修改').length).toBeGreaterThan(0)
    expect(screen.getAllByText('另存为新指标').length).toBeGreaterThan(0)
    await user.click(screen.getByText('浏览公式构建资源'))

    const builder = await screen.findByRole('dialog', { name: '编辑“我的波动指标”的计算逻辑' })
    const ddof = within(builder).getByLabelText('自由度修正 有限常量')
    await user.clear(ddof)
    await user.type(ddof, '0')
    await user.click(within(builder).getByRole('button', { name: '应用逻辑修改' }))

    await waitFor(() => {
      const composeCalls = vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/compose')
      const body = JSON.parse(String((composeCalls[composeCalls.length - 1]?.[1] as RequestInit).body))
      expect(body.arguments).toContainEqual({ parameter: 'ddof', source: 'constant', value: 0 })
    })
    await user.click(screen.getAllByText('另存为新指标')[0])

    await waitFor(() => {
      const createCalls = vi.mocked(fetch).mock.calls.filter(([url, init]) => url === '/api/custom-indicators' && init?.method === 'POST')
      const body = JSON.parse(String((createCalls[createCalls.length - 1]?.[1] as RequestInit).body))
      expect(body.name).toBe('我的波动指标 副本')
    })
    expect(screen.getByText(/已另存为新指标/)).toBeInTheDocument()
  })

  it('指标中心固定为单产品上下文，不展示组合域和组合指标', async () => {
    setupUser()
    await renderStudio()
    expect(screen.queryByText('计算域')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('组合运行域')).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /组合波动率/ })).not.toBeInTheDocument()
    expect(screen.getByLabelText('计算周期')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/portfolio-runs')).toBe(false)
  })

  it('构建目录只展示变量定义，产品可计算性留到预览，并传递 as-of', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /收益波动率/ }))
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))
    await user.type(screen.getByLabelText('历史截止日'), '2026-08-28')

    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/custom-indicators/variables/availability')).toBe(false)

    await user.click(screen.getByRole('button', { name: '浏览公式构建资源' }))
    const currentLogic = await screen.findByRole('dialog', { name: '编辑“收益波动率”的计算逻辑' })
    await user.click(within(currentLogic).getByRole('button', { name: '改用其他构建资源' }))
    const dialog = await screen.findByRole('dialog', { name: '变量、算子与已有指标' })
    await chooseCombobox(user, '资源分类', '收益与变化')
    await chooseCombobox(user, '选择变量', '复权净值普通收益率')
    expect(within(dialog).getAllByText('由相邻复权净值计算的普通收益率。').length).toBeGreaterThan(0)
    expect(within(dialog).queryByText(/所选产品中/)).not.toBeInTheDocument()
    expect(within(dialog).queryByText('所选产品可用性')).not.toBeInTheDocument()
    expect(within(dialog).queryByText('实际数据规模 / 覆盖率')).not.toBeInTheDocument()
    expect(within(dialog).getByText(/实际可计算性将在选择产品并执行预览后/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: '设为当前公式' }))
    await user.click(screen.getByRole('button', { name: '预览指标' }))
    await waitFor(() => {
      const call = vi.mocked(fetch).mock.calls.find(([url]) => url === '/api/custom-indicators/evaluate')
      expect(JSON.parse(String((call?.[1] as RequestInit).body)).as_of).toBe('2026-08-28')
    })
  })

  it('构建阶段不请求产品可用性，算子继续提供定义上兼容的变量', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: '搜索' }))
    await user.click(await screen.findByRole('button', { name: '添加' }))

    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/custom-indicators/variables/availability')).toBe(false)
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))

    const composer = await screen.findByRole('dialog', { name: '配置 全元素标准差' })
    expect(within(composer).getByLabelText('输入值 输入来源')).toHaveTextContent('兼容变量')
    expect(within(composer).getByRole('option', { name: /复权净值普通收益率/ })).not.toBeDisabled()
    expect(within(composer).queryByText(/当前产品变量中没有满足该参数类型与语义约束的输入/)).not.toBeInTheDocument()
    expect(within(composer).getByRole('button', { name: '展开到公式' })).toBeEnabled()
  })

  it('资源分类只展示定义数量，并隐藏当前域不可用的分类', async () => {
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.selectOptions(within(dialog).getByLabelText('资源类型'), 'operators')
    await user.click(within(dialog).getByRole('combobox', { name: '资源分类' }))

    const options = within(dialog).getAllByRole('option')
    expect(options.some((option) => option.textContent?.includes('/'))).toBe(false)
    expect(within(dialog).getByRole('option', { name: '基础运算（2）' })).toBeInTheDocument()
    expect(within(dialog).queryByRole('option', { name: /线性代数/ })).not.toBeInTheDocument()
  })

  it('meta 明确标记 unavailable 的变量在未选择产品时不展示对应分类', async () => {
    activeMeta = {
      ...meta,
      variables: variables.map((variable) => variable.name === 'volume'
        ? { ...variable, availability: 'unavailable' }
        : variable),
    }
    const user = setupUser()
    await renderStudio()
    const dialog = await openCatalog(user)
    await user.click(within(dialog).getByRole('combobox', { name: '资源分类' }))
    await user.type(await within(dialog).findByLabelText('搜索资源分类'), '成交与流动性')

    expect(within(dialog).queryByRole('option', { name: /成交与流动性/ })).not.toBeInTheDocument()
  })

  it('移动端三区 Tab 支持方向键、Home/End 与正确的 aria 关联', async () => {
    await renderStudio()
    const libraryTab = screen.getByRole('tab', { name: '指标库' })
    const editorTab = screen.getByRole('tab', { name: '编辑' })
    const previewTab = screen.getByRole('tab', { name: '预览' })
    expect(libraryTab).toHaveAttribute('aria-controls', 'indicator-panel-library')
    expect(editorTab).toHaveAttribute('aria-controls', 'indicator-panel-editor')
    expect(previewTab).toHaveAttribute('aria-controls', 'indicator-panel-preview')

    fireEvent.keyDown(previewTab, { key: 'Enter' })
    expect(previewTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(previewTab, { key: 'ArrowLeft' })
    expect(editorTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(editorTab, { key: 'Home' })
    expect(libraryTab).toHaveAttribute('aria-selected', 'true')
    fireEvent.keyDown(libraryTab, { key: 'End' })
    expect(previewTab).toHaveAttribute('aria-selected', 'true')
  })

  it('桌面工作台一次只展示定义或预览，并支持键盘切换', async () => {
    await renderStudio()
    const tablist = screen.getByRole('tablist', { name: '指标工作台' })
    const editorTab = within(tablist).getByRole('tab', { name: /定义与公式/ })
    const previewTab = within(tablist).getByRole('tab', { name: /校验与预览/ })
    const editorPanel = document.getElementById('indicator-panel-editor')
    const previewPanel = document.getElementById('indicator-panel-preview')

    expect(editorTab).toHaveAttribute('aria-selected', 'true')
    expect(editorPanel).toHaveClass('md:block')
    expect(previewPanel).toHaveClass('md:hidden')

    fireEvent.keyDown(editorTab, { key: 'ArrowRight' })
    expect(previewTab).toHaveAttribute('aria-selected', 'true')
    expect(editorPanel).toHaveClass('md:hidden')
    expect(previewPanel).toHaveClass('md:block')

    fireEvent.keyDown(previewTab, { key: 'Home' })
    expect(editorTab).toHaveAttribute('aria-selected', 'true')
  })

  it('在校验与预览中切换指标时保留当前页签并更新当前指标', async () => {
    const user = setupUser()
    await renderStudio()
    const tablist = screen.getByRole('tablist', { name: '指标工作台' })
    const editorTab = within(tablist).getByRole('tab', { name: /定义与公式/ })
    const previewTab = within(tablist).getByRole('tab', { name: /校验与预览/ })

    await user.click(previewTab)
    await user.click(screen.getByRole('button', { name: /累计收益率/ }))

    expect(previewTab).toHaveAttribute('aria-selected', 'true')
    expect(editorTab).toHaveAttribute('aria-selected', 'false')
    expect(document.getElementById('indicator-panel-preview')).toHaveClass('md:block')
    expect(screen.getByLabelText('当前预览指标')).toHaveTextContent('累计收益率')
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/validate')).toHaveLength(1))
    expect(await screen.findByRole('heading', { name: '层级计算 DAG' })).toBeInTheDocument()
    expect(screen.getByText(/已保留预览条件.*公式解析和校验通过/)).toBeInTheDocument()
  })

  it('从定义页进入校验与预览时自动解析当前指标', async () => {
    const user = setupUser()
    await renderStudio()
    await user.click(screen.getByRole('button', { name: /累计收益率/ }))

    const previewTab = within(screen.getByRole('tablist', { name: '指标工作台' })).getByRole('tab', { name: /校验与预览/ })
    await user.click(previewTab)

    await waitFor(() => expect(vi.mocked(fetch).mock.calls.filter(([url]) => url === '/api/custom-indicators/validate')).toHaveLength(1))
    expect(await screen.findByRole('heading', { name: '层级计算 DAG' })).toBeInTheDocument()
  })

  it('算子展开失败会在抽屉内给出明确错误并允许重试', async () => {
    composeFailure = true
    const user = setupUser()
    await renderStudio()
    const catalogDialog = await openCatalog(user)
    await user.selectOptions(within(catalogDialog).getByLabelText('资源类型'), 'operators')
    await chooseCombobox(user, '资源分类', '统计归约')
    await chooseCombobox(user, '选择计算算子', '全元素标准差')
    await user.click(within(catalogDialog).getByRole('button', { name: '配置算子参数' }))
    await user.click(screen.getByRole('button', { name: '展开到公式' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('服务端拒绝展开，请检查参数。')
    expect(screen.getByRole('dialog', { name: '配置 全元素标准差' })).toBeInTheDocument()
  })
})
