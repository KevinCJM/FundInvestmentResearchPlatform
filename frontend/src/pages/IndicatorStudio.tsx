import SeriesOutputFields from '../components/computation-graph/SeriesOutputFields'
import IndicatorParameterEditor from '../components/indicator-parameters/IndicatorParameterEditor'
import IndicatorParameterInputs from '../components/indicator-parameters/IndicatorParameterInputs'
import React, { useEffect, useId, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import IndicatorGraphEditor, { type IndicatorGraphEditorHandle } from '../components/indicator-graph/IndicatorGraphEditor'
import katex from 'katex'
import ScalarOutputEditor from '../components/indicator-outputs/ScalarOutputEditor'
import { businessText, systemText, useI18n } from '../i18n/runtime'
import { localizeIndicatorMeta } from '../i18n/indicatorMetadata'
import 'katex/dist/katex.min.css'
import {
  evaluationStatusLabel,
  formatIndicatorDiagnostic,
  humanizeIndicatorMessage,
  humanizeIndicatorTechnicalText,
} from '../utils/indicatorDiagnostics'
import {
  CustomIndicatorApiError,
  composeCustomIndicator,
  createCustomIndicator,
  deleteCustomIndicator,
  deriveRollingSeriesIndicator,
  evaluateCustomIndicators,
  evaluateTimeSeriesIndicators,
  exportCustomIndicatorExcel,
  getCustomIndicatorMeta,
  getSnapshotIndicatorConfig,
  indicatorsForContext,
  listCustomIndicators,
  searchInstruments,
  updateCustomIndicator,
  updateSnapshotIndicatorConfig,
  validateCustomIndicator,
  type EvaluationResult,
  type EvaluationTarget,
  type IndicatorResultKind,
  type IndicatorDefinition,
  type IndicatorDag,
  type IndicatorDagNode,
  type IndicatorDraft,
  type IndicatorContextDomain,
  type IndicatorMeta,
  type IndicatorOperator,
  type IndicatorOperatorParameter,
  type IndicatorShape,
  type IndicatorVariable,
  type InferenceResponse,
  type InstrumentSearchItem,
  type ProductKind,
  type SeriesOutputDefinition,
  type SeriesOutputInference,
  type SeriesOutputMeasureId,
  type SeriesOutputMeasureOption,
  type SnapshotIndicatorConfig,
  type TimeSeriesIndicatorResult,
  type ValidationResponse,
} from '../services/customIndicators'
import { MetricUnavailableReason, MetricValue } from '../components/metrics/MetricDisplay'
import { SearchDropdown } from '../components/FilterDropdown'

const FALLBACK_PERIODS = [
  { value: '1W', label: '近 1 周', description: '最近 5 个收益观察值' },
  { value: '1M', label: '近 1 月', description: '自然月窗口' },
  { value: '3M', label: '近 3 月', description: '自然月窗口' },
  { value: '6M', label: '近 6 月', description: '自然月窗口' },
  { value: '1Y', label: '近 1 年', description: '自然年窗口' },
  { value: '2Y', label: '近 2 年', description: '自然年窗口' },
  { value: '3Y', label: '近 3 年', description: '自然年窗口' },
  { value: '5Y', label: '近 5 年', description: '自然年窗口' },
  { value: '10Y', label: '近 10 年', description: '自然年窗口' },
  { value: '20Y', label: '近 20 年', description: '自然年窗口' },
  { value: '30Y', label: '近 30 年', description: '自然年窗口' },
  { value: 'W1', label: '上周', description: '上一个完整自然周（周一至周日）' },
  { value: 'W2', label: '上上周', description: '截止日前第二个完整自然周' },
  { value: 'M1', label: '上月', description: '上一个完整自然月' },
  { value: 'M2', label: '上上月', description: '截止日前第二个完整自然月' },
  { value: 'Y1', label: '去年', description: '上一个完整自然年度' },
  { value: 'Y2', label: '前年', description: '截止日前第二个完整自然年度' },
  { value: 'ALL', label: '成立以来', description: '首个真实净值点至有效截止日' },
]

const FALLBACK_SERIES_OUTPUT_MEASURES: SeriesOutputMeasureOption[] = [
  { id: 'auto', label: '自动推断', description: '根据公式量纲与可证明范围自动选择。', semantic_dimensions: ['*'], range: null, default_unit: '', default_display_format: 'number' },
  { id: 'raw_market_price', label: '原始市价', description: '开盘价、最高价、最低价、收盘价及同量纲结果。', semantic_dimensions: ['raw_market_price'], range: null, default_unit: '元', default_display_format: 'number' },
  { id: 'adjusted_nav', label: '复权净值', description: '复权后的单位净值或财富水平。', semantic_dimensions: ['adjusted_nav'], range: null, default_unit: '净值', default_display_format: 'number' },
  { id: 'reported_nav', label: '披露净值', description: '基金披露的单位净值。', semantic_dimensions: ['reported_nav'], range: null, default_unit: '净值', default_display_format: 'number' },
  { id: 'virtual_nav', label: '虚拟净值（起点 1）', description: '由收益累计得到的无量纲财富指数。', semantic_dimensions: ['dimensionless'], range: null, default_unit: '', default_display_format: 'number' },
  { id: 'normalized', label: '归一化值', description: '无物理单位且不承诺固定边界的相对水平。', semantic_dimensions: ['dimensionless'], range: null, default_unit: '', default_display_format: 'number' },
  { id: 'bounded_0_1', label: '0～1 区间', description: '明确限制在 [0, 1] 的无量纲值。', semantic_dimensions: ['dimensionless'], range: [0, 1], default_unit: '', default_display_format: 'number' },
  { id: 'bounded_minus1_1', label: '-1～1 区间', description: '明确限制在 [-1, 1] 的无量纲值。', semantic_dimensions: ['dimensionless'], range: [-1, 1], default_unit: '', default_display_format: 'number' },
  { id: 'oscillator_0_100', label: '0～100 摆动值', description: 'K、D、RSI 等技术指标常用刻度。', semantic_dimensions: ['dimensionless'], range: [0, 100], default_unit: '', default_display_format: 'number' },
  { id: 'return_decimal', label: '收益率', description: '内部为小数，展示时通常使用百分比。', semantic_dimensions: ['return_decimal'], range: null, default_unit: '%', default_display_format: 'percent' },
  { id: 'rate_decimal', label: '利率', description: '内部为小数的利率或费率。', semantic_dimensions: ['rate_decimal'], range: null, default_unit: '%', default_display_format: 'percent' },
  { id: 'volume', label: '成交量', description: '份额、手数或成交数量。', semantic_dimensions: ['volume'], range: [0, null], default_unit: '份', default_display_format: 'number' },
  { id: 'currency_amount', label: '金额', description: '成交额、资产规模等货币金额。', semantic_dimensions: ['currency_amount'], range: null, default_unit: '元', default_display_format: 'number' },
  { id: 'count', label: '计数', description: '观察数、次数等离散计数。', semantic_dimensions: ['count'], range: [0, null], default_unit: '', default_display_format: 'number' },
  { id: 'calendar_days', label: '日历天数', description: '以自然日为单位的期限或间隔。', semantic_dimensions: ['calendar_days'], range: [0, null], default_unit: '天', default_display_format: 'number' },
  { id: 'dimensionless', label: '其他无量纲值', description: '没有物理单位且不声明固定范围。', semantic_dimensions: ['dimensionless'], range: null, default_unit: '', default_display_format: 'number' },
  { id: 'derived', label: '复合量纲', description: '乘方、乘除产生的派生量纲。', semantic_dimensions: ['derived:*'], range: null, default_unit: '', default_display_format: 'number' },
]

const EMPTY_DRAFT: IndicatorDraft = {
  name: '未命名指标',
  description: '',
  expression: '',
  unit: '',
  display_format: 'number',
  precision: 3,
  direction: 'higher_better',
  indicator_type: 'return',
  annual_risk_free_rate_percent: 1.5,
  dsl_version: '2.4.0',
  operator_registry_version: '2.4.0',
  numeric_kernel_version: '2.2.0',
  variable_registry_version: '2.1.0',
  context_schema_version: 'typed-context-v2',
  data_contract_version: 'tushare-eod-v2',
  period_policy: 'all_supported',
  context_kind: 'single_product',
  result_kind: 'scalar',
  output_contract: 'scalar',
  output_measure: 'dimensionless',
  template_origin: null,
}

type MobileTab = 'library' | 'editor' | 'preview'
type WorkspaceTab = 'editor' | 'preview'
type CatalogTab = 'variables' | 'operators' | 'indicators'
type EditorMode = 'canvas' | 'guided' | 'advanced'
type ComposerSource = 'variable' | 'constant' | 'operator' | 'indicator' | 'omitted'
type ComposerApplyMode = 'replace_formula' | 'replace_selection' | 'insert_cursor'

type ComposerItem = {
  intermediateKind?: 'linear_fit' | 'drawdown_interval'
  id: string
  label: string
  kind: 'operator'
  signature: string
  essence: string
  semantic: string
  outputShape: IndicatorShape
  outputType?: string
  domains: IndicatorContextDomain[]
  parameters: IndicatorOperatorParameter[]
  parameterChoices?: IndicatorOperatorParameter[][]
  categoryId: string
  categoryLabel: string
  aliases: string[]
  tags: string[]
  examples: string[]
  displayTemplate?: string
  costEstimate?: string | number
  version?: string
  executionBackend?: string
}

type ComposerArgument = {
  parameter: IndicatorOperatorParameter
  source: ComposerSource
  value: string
  nested?: ComposerNode
}

type ComposerNode = {
  item: ComposerItem
  arguments: ComposerArgument[]
}

type ComposerState = {
  node: ComposerNode
  applyMode: ComposerApplyMode
  selectionStart: number
  selectionEnd: number
  origin: 'catalog' | 'current_formula'
  seriesOutputId?: string
} | null

const MOBILE_TABS: [MobileTab, string][] = [
  ['library', '指标库'],
  ['editor', '编辑'],
  ['preview', '预览'],
]

type StudioTarget = EvaluationTarget & { name: string }

const MAX_PREVIEW_TARGETS = 10
const STUDIO_CONTEXT: IndicatorContextDomain = 'single_product'

const INDICATOR_PROTOCOL_FIELDS = [
  'variable_registry_version',
  'data_contract_version',
  'context_schema_version',
] as const

function indicatorReferenceKey(indicator: IndicatorDefinition) {
  return `${indicator.id}@${indicator.revision}`
}

function indicatorIsComposable(
  indicator: IndicatorDefinition,
  draft: IndicatorDraft,
  contextDomain: IndicatorContextDomain,
) {
  if ((indicator.context_kind ?? 'single_product') !== contextDomain) return false
  if ((indicator.output_contract ?? 'scalar') !== 'scalar') return false
  if (!indicator.expression.trim() || !String(indicator.dsl_version || '').startsWith('2.')) return false
  // A referenced indicator is expanded into plain formula text and then
  // compiled against the current draft protocol.  Therefore typed v2.0/v2.1
  // formulas remain reusable in v2.2 without carrying an executable reference.
  return INDICATOR_PROTOCOL_FIELDS.every((field) => (
    !draft[field]
    || !indicator[field]
    || String(draft[field]) === String(indicator[field])
  ))
}

function composableProtocolPriority(indicator: IndicatorDefinition, draft: IndicatorDraft) {
  const dsl = String(indicator.dsl_version || '')
  const operatorRegistry = String(indicator.operator_registry_version || '')
  if (dsl === draft.dsl_version && operatorRegistry === draft.operator_registry_version) return 3
  if (dsl.startsWith('2.2') && operatorRegistry.startsWith('2.2')) return 2
  if (dsl.startsWith('2.1') && operatorRegistry.startsWith('2.1')) return 1
  return 0
}

function indicatorCategory(indicator: IndicatorDefinition) {
  return {
    id: indicator.category_id || indicator.indicator_type || 'other',
    label: indicator.category_label || indicator.presentation?.category_label || '其他指标',
  }
}

function parameterAcceptsIndicator(parameter: IndicatorOperatorParameter) {
  if (!acceptedShapes(parameter).some((shape) => shape === 'scalar' || shape === 'unknown')) return false
  return !['ddof', 'periods', 'probability'].includes(parameter.name)
}

const FALLBACK_VARIABLES: IndicatorVariable[] = [
  { name: 'returns', label: '复权净值普通收益率', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{r}', shape: 'series', semantic: '由相邻复权净值计算的普通收益率，即 adjusted_nav[t] / adjusted_nav[t-1] - 1。', semantic_role: 'ordinary_return', measure: 'return_decimal', price_basis: 'adjusted_nav', source: '真实复权净值派生', data_basis: '复权净值', frequency: '交易日', unit: '小数', domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'returns', category_label: '收益与变化' },
  { name: 'log_returns', label: '复权净值对数收益率', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{\\ell}', shape: 'series', semantic: '由相邻复权净值计算的对数收益率，即 log(adjusted_nav[t] / adjusted_nav[t-1])。', semantic_role: 'log_return', measure: 'return_decimal', price_basis: 'adjusted_nav', source: '真实复权净值派生', data_basis: '复权净值', frequency: '交易日', unit: '小数', domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'returns', category_label: '收益与变化' },
  { name: 'adjusted_nav', label: '复权净值', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{p}_{\\mathrm{adj}}', shape: 'series', semantic: '产品在计算窗口内的真实复权净值。', semantic_role: 'adjusted_nav_level', measure: 'adjusted_nav', price_basis: 'adjusted_nav', source: 'Tushare 本地 Parquet', data_basis: '复权净值', frequency: '交易日', unit: '净值', domains: ['single_product'], product_kinds: ['etf', 'fund'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_open', label: '开盘价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{o}', shape: 'series', semantic: 'ETF 未复权日 K 开盘价；场外基金没有日内开盘价。', semantic_role: 'raw_open_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_high', label: '最高价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{h}', shape: 'series', semantic: 'ETF 未复权日 K 最高价；场外基金没有日内最高价。', semantic_role: 'raw_high_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_low', label: '最低价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{l}', shape: 'series', semantic: 'ETF 未复权日 K 最低价；场外基金没有日内最低价。', semantic_role: 'raw_low_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'market_close', label: '收盘价', value_type: 'series<time>', dtype: 'float64', latex: '\\mathbf{c}', shape: 'series', semantic: 'ETF 未复权日 K 收盘价；场外基金没有交易所收盘价。', semantic_role: 'raw_close_price', measure: 'raw_market_price', price_basis: 'raw_market', source: 'Tushare ETF 日线行情', data_basis: '未复权日 K', frequency: '交易日', unit: '价格', domains: ['single_product'], product_kinds: ['etf'], category_id: 'price', category_label: '净值与价格' },
  { name: 'risk_free_rate_per_observation', label: '单观察期无风险收益率', value_type: 'scalar', dtype: 'float64', latex: 'r_{f}', shape: 'scalar', semantic: '由年化无风险利率按当前观察频率换算。', source: '指标配置', domains: ['single_product', 'portfolio'], category_id: 'configuration', category_label: '基准与配置' },
]

function inferShape(variable: IndicatorVariable): IndicatorShape {
  if (variable.dtype === 'bool' || /mask/i.test(variable.value_type)) return 'mask'
  if (variable.shape) return variable.shape
  if (['returns', 'log_returns', 'benchmark_returns'].includes(variable.name)) return 'series'
  if (['asset_returns', 'asset_log_returns', 'weight_path'].includes(variable.name)) return 'matrix'
  if (variable.name === 'asset_weights') return 'vector'
  if (/risk_free|periods_per_year/i.test(variable.name)) return 'scalar'
  if (/window/i.test(variable.value_type)) return 'window'
  if (/matrix|covariance|square/i.test(variable.value_type)) return 'matrix'
  if (/series|returns|time/i.test(variable.value_type)) return 'series'
  if (/vector|array/i.test(variable.value_type)) return 'vector'
  return /tuple/i.test(variable.value_type) ? 'tuple' : 'scalar'
}

function acceptedShapes(parameter: IndicatorOperatorParameter): IndicatorShape[] {
  return parameter.allowed_shapes || (parameter.shape
    ? [parameter.shape]
    : ['scalar', 'series', 'vector', 'matrix', 'window', 'mask', 'tuple', 'unknown'])
}

function supportsVariableDomain(variable: IndicatorVariable, contextDomain: IndicatorContextDomain) {
  if (variable.domains) return variable.domains.includes(contextDomain)
  return contextDomain === 'portfolio'
    ? ['asset_returns', 'asset_log_returns', 'asset_weights', 'weight_path', 'benchmark_returns'].includes(variable.name)
    : !['asset_returns', 'asset_log_returns', 'asset_weights', 'weight_path', 'benchmark_returns'].includes(variable.name)
}

function isCompatibleVariable(
  variable: IndicatorVariable,
  parameter: IndicatorOperatorParameter,
  contextDomain: IndicatorContextDomain,
) {
  if (!acceptedShapes(parameter).includes(inferShape(variable))) return false
  if (!supportsVariableDomain(variable, contextDomain)) return false
  if (parameter.excluded_semantic_dimensions?.includes(variable.measure || variable.semantic || '')) return false
  const roles = parameter.allowed_semantic_roles ?? []
  const role = variable.semantic_role || variable.semantic
  return roles.length === 0 || !role || roles.includes(role)
}

function normalizedVariableMeasure(variable: IndicatorVariable) {
  if (variable.measure) return variable.measure
  const role = variable.semantic_role || ''
  if (/ordinary_return|log_return|benchmark_return|quote_return/.test(role)) return 'return_decimal'
  if (/risk_free_rate/.test(role)) return 'rate_decimal'
  if (/raw_.*price|raw_price/.test(role)) return 'raw_market_price'
  if (/volume/.test(role)) return 'volume'
  if (/turnover|net_asset|dividend/.test(role)) return 'currency_amount'
  if (/weight/.test(role)) return 'dimensionless'
  return ''
}

function measuresCanCombine(left: IndicatorVariable, right: IndicatorVariable) {
  const leftMeasure = normalizedVariableMeasure(left)
  const rightMeasure = normalizedVariableMeasure(right)
  if (!leftMeasure || !rightMeasure || leftMeasure === rightMeasure) {
    return !left.price_basis || !right.price_basis || left.price_basis === right.price_basis
  }
  const returnAndRate = new Set([leftMeasure, rightMeasure])
  if (returnAndRate.size === 2 && returnAndRate.has('return_decimal') && returnAndRate.has('rate_decimal')) return true
  return false
}

function isCompatibleWithComposerSiblings(
  variable: IndicatorVariable,
  argumentIndex: number,
  node: ComposerNode,
  variables: IndicatorVariable[],
) {
  const binaryAligned = new Set([
    'add', 'subtract', 'multiply', 'divide', 'minimum', 'maximum',
    'equal', 'not_equal', 'less_than', 'less_equal', 'greater_than', 'greater_equal',
    'dot', 'outer',
  ])
  const semanticAligned = new Set([
    'add', 'subtract', 'minimum', 'maximum',
    'equal', 'not_equal', 'less_than', 'less_equal', 'greater_than', 'greater_equal',
  ])
  const branchIndexes = node.item.id === 'where' ? [1, 2] : binaryAligned.has(node.item.id) ? [0, 1] : []
  if (!branchIndexes.includes(argumentIndex)) return true
  const siblingIndex = branchIndexes.find((index) => index !== argumentIndex)
  if (siblingIndex === undefined) return true
  const sibling = node.arguments[siblingIndex]
  if (!sibling || sibling.source !== 'variable') return true
  const siblingVariable = variables.find((item) => item.name === sibling.value)
  if (!siblingVariable) return true
  const candidateShape = inferShape(variable)
  const siblingShape = inferShape(siblingVariable)
  if (node.item.id === 'dot' || node.item.id === 'outer') {
    if (candidateShape !== siblingShape || candidateShape === 'scalar') return false
  } else if (candidateShape !== siblingShape && candidateShape !== 'scalar' && siblingShape !== 'scalar') {
    return false
  }
  if ((semanticAligned.has(node.item.id) || node.item.id === 'where') && !measuresCanCombine(variable, siblingVariable)) {
    return false
  }
  return true
}

function parameterRequiresFixedConstant(
  parameter: IndicatorOperatorParameter,
) {
  return parameter.source_policy === 'fixed_constant'
}

function initialComposerArgument(
  parameter: IndicatorOperatorParameter,
  variables: IndicatorVariable[],
  contextDomain: IndicatorContextDomain,
): ComposerArgument {
  if (parameterRequiresFixedConstant(parameter)) {
    const fallback = parameter.constant_kind === 'integer' ? 1 : 0
    return {
      parameter,
      source: 'constant',
      value: String(typeof parameter.default === 'number' ? parameter.default : fallback),
    }
  }
  if (parameter.optional && parameter.default == null) {
    return { parameter, source: 'omitted', value: '' }
  }
  if (typeof parameter.default === 'number') {
    return { parameter, source: 'constant', value: String(parameter.default) }
  }
  const compatibleVariables = variables.filter((variable) => isCompatibleVariable(variable, parameter, contextDomain))
  const defaultVariable = compatibleVariables.find((variable) => variable.name === parameter.default)
  const variable = defaultVariable ?? compatibleVariables[0]
  if (variable) return { parameter, source: 'variable', value: variable.name }
  return { parameter, source: acceptedShapes(parameter).includes('scalar') ? 'constant' : 'variable', value: '' }
}

function initialComposerNode(
  item: ComposerItem,
  variables: IndicatorVariable[],
  contextDomain: IndicatorContextDomain,
): ComposerNode {
  return {
    item,
    arguments: item.parameters.map((parameter) => initialComposerArgument(parameter, variables, contextDomain)),
  }
}

function composerNodeFromDag(
  dag: IndicatorDag | null | undefined,
  operators: ComposerItem[],
  rootKey?: string,
): ComposerNode | null {
  if (!dag) return null
  const nodeById = new Map(dag.nodes.map((node) => [String(node.id), node]))
  const operatorById = new Map(operators.flatMap((operator) => [
    [operator.id, operator] as const,
    ...operator.aliases.map((alias) => [alias, operator] as const),
  ]))
  const incoming = new Map<string, typeof dag.edges>()
  dag.edges.forEach((edge) => {
    const target = String(edge.target)
    incoming.set(target, [...(incoming.get(target) ?? []), edge].sort((left, right) => (left.order ?? 0) - (right.order ?? 0)))
  })

  const build = (nodeId: string | number, visiting: Set<string>): ComposerNode | null => {
    const key = String(nodeId)
    if (visiting.has(key)) return null
    const node = nodeById.get(key)
    if (!node) return null
    const operatorId = node.operator?.id || node.operator_id || node.label
    const catalogItem = operatorById.get(operatorId)
    if (!catalogItem) return null
    const nextVisiting = new Set(visiting).add(key)
    const namedInputs = new Map((node.arguments ?? []).map((argument) => [argument.name, argument.input_node_id]))
    const orderedInputs = node.inputs?.length
      ? node.inputs
      : (incoming.get(key) ?? []).map((edge) => edge.source)
    // A catalog's preferred overload is for NEW operators, not an existing
    // DAG. Reconstruct its actual arity so ddof/min_periods cannot disappear.
    const choices = catalogItem.parameterChoices ?? [catalogItem.parameters]
    const parameters = choices.find((choice) => (
      choice.length === orderedInputs.length
      && choice.every((parameter, index) => {
        const child = nodeById.get(String(orderedInputs[index]))
        const shape = child ? dagNodeShape(child) : 'unknown'
        const allowed = acceptedShapes(parameter)
        return shape === 'unknown' || allowed.includes('unknown') || allowed.includes(shape)
      })
    ))
    if (!parameters && orderedInputs.length > catalogItem.parameters.length) return null
    const item = { ...catalogItem, parameters: parameters ?? catalogItem.parameters }
    const argumentsForNode = item.parameters.map((parameter, index): ComposerArgument => {
      const childId = namedInputs.get(parameter.name) ?? orderedInputs[index]
      if (childId === undefined) {
        if (typeof parameter.default === 'number') {
          return { parameter, source: 'constant', value: String(parameter.default) }
        }
        return parameter.optional
          ? { parameter, source: 'omitted', value: '' }
          : { parameter, source: 'variable', value: '' }
      }
      const child = nodeById.get(String(childId))
      if (!child) return { parameter, source: 'variable', value: '' }
      if (child.kind === 'variable') return { parameter, source: 'variable', value: child.label }
      if (child.kind === 'constant') return { parameter, source: 'constant', value: child.formula_fragment || child.label }
      const nested = build(childId, nextVisiting)
      return nested
        ? { parameter, source: 'operator', value: nested.item.id, nested }
        : { parameter, source: 'variable', value: '' }
    })
    return { item, arguments: argumentsForNode }
  }

  const rootId = (rootKey ? dag.roots[rootKey] : undefined)
    ?? dag.roots.result
    ?? Object.values(dag.roots)[0]
  return rootId === undefined ? null : build(rootId, new Set())
}

function dagForRoot(
  dag: IndicatorDag | null | undefined,
  rootKey: string,
): IndicatorDag | null {
  if (!dag) return null
  const rootId = dag.roots[rootKey]
  if (rootId === undefined) return null
  const incoming = new Map<string, Array<string | number>>()
  dag.edges.forEach((edge) => {
    const target = String(edge.target)
    incoming.set(target, [...(incoming.get(target) ?? []), edge.source])
  })
  const reachable = new Set<string>()
  const pending: Array<string | number> = [rootId]
  while (pending.length) {
    const nodeId = pending.pop()
    if (nodeId === undefined || reachable.has(String(nodeId))) continue
    reachable.add(String(nodeId))
    pending.push(...(incoming.get(String(nodeId)) ?? []))
  }
  return {
    nodes: dag.nodes.filter((node) => reachable.has(String(node.id))),
    edges: dag.edges.filter((edge) => reachable.has(String(edge.source)) && reachable.has(String(edge.target))),
    roots: { result: rootId },
  }
}

function inferenceForSeriesOutput(
  validation: ValidationResponse | null,
  outputId: string,
): InferenceResponse | null {
  const output = validation?.output_inferences?.[outputId]
  if (!output) return null
  return {
    expression: output.expression,
    python_expression: output.python_expression || undefined,
    editable_latex: output.editable_latex,
    latex: output.latex || output.expression,
    display_latex: output.display_latex || undefined,
    math_notation_version: output.math_notation_version || undefined,
    inferred_type: output.inferred_type,
    shape: output.shape,
    semantic_warnings: [],
    dependencies: output.dependencies,
    dag: dagForRoot(validation?.dag, outputId) || undefined,
  }
}

function composedExecutableSource(response: InferenceResponse): string {
  const source = response.python_expression?.trim() || response.expression?.trim()
  if (!source || source.includes('\\')) {
    throw new Error('构建服务未返回可执行 DSL 源码，请刷新页面并确认后端已更新。')
  }
  return source
}

function composedEditableLatex(response: InferenceResponse): string {
  const source = response.editable_latex?.trim()
  if (!source) throw new Error('构建服务未返回可编辑 LaTeX，请确认后端已更新。')
  return source
}

function composerNodeCount(node: ComposerNode): number {
  return 1 + node.arguments.reduce(
    (total, argument) => total + (argument.nested ? composerNodeCount(argument.nested) : 0),
    0,
  )
}

function isCompatibleNestedOperator(
  item: ComposerItem,
  parameter: IndicatorOperatorParameter,
  contextDomain: IndicatorContextDomain,
) {
  if (!item.domains.includes(contextDomain)) return false
  const expected = acceptedShapes(parameter)
  if (parameter.intermediate_kind && item.intermediateKind !== parameter.intermediate_kind) return false
  return item.outputShape === 'unknown' || expected.includes('unknown') || expected.includes(item.outputShape)
}

function isComposerNodeValid(
  node: ComposerNode,
  variables: IndicatorVariable[],
  indicators: IndicatorDefinition[],
  contextDomain: IndicatorContextDomain,
): boolean {
  return node.arguments.every((argument, index) => (
    isComposerArgumentValid(argument, variables, indicators, contextDomain, node, index)
  ))
}

function updateComposerNodeArgument(
  node: ComposerNode,
  path: number[],
  nextArgument: ComposerArgument,
): ComposerNode {
  const [argumentIndex, ...nestedPath] = path
  return {
    ...node,
    arguments: node.arguments.map((argument, index) => {
      if (index !== argumentIndex) return argument
      if (nestedPath.length === 0) return nextArgument
      if (!argument.nested) return argument
      return { ...argument, nested: updateComposerNodeArgument(argument.nested, nestedPath, nextArgument) }
    }),
  }
}

function firstInvalidComposerArgument(
  node: ComposerNode,
  variables: IndicatorVariable[],
  indicators: IndicatorDefinition[],
  contextDomain: IndicatorContextDomain,
): ComposerArgument | null {
  for (const [index, argument] of node.arguments.entries()) {
    if (!isComposerArgumentValid(argument, variables, indicators, contextDomain, node, index)) return argument
    if (argument.source === 'operator' && argument.nested) {
      const nestedInvalid = firstInvalidComposerArgument(argument.nested, variables, indicators, contextDomain)
      if (nestedInvalid) return nestedInvalid
    }
  }
  return null
}

function firstComposerSiblingIssue(
  node: ComposerNode,
  variables: IndicatorVariable[],
): string | null {
  for (const [index, argument] of node.arguments.entries()) {
    if (argument.source === 'variable' && argument.value) {
      const variable = variables.find((item) => item.name === argument.value)
      if (variable && !isCompatibleWithComposerSiblings(variable, index, node, variables)) {
        const sibling = node.arguments.find((_, siblingIndex) => siblingIndex !== index && (
          node.item.id === 'where'
            ? [1, 2].includes(siblingIndex)
            : siblingIndex < 2
        ))
        const currentLabel = argument.parameter.label || argument.parameter.name
        const siblingLabel = sibling?.parameter.label || sibling?.parameter.name || '另一输入'
        return `${currentLabel} 与 ${siblingLabel} 的类型、轴、语义量纲或价格口径不兼容；请为两个参数选择可配对的变量。`
      }
    }
    if (argument.source === 'operator' && argument.nested) {
      const nestedIssue = firstComposerSiblingIssue(argument.nested, variables)
      if (nestedIssue) return nestedIssue
    }
  }
  return null
}

function isComposerArgumentValid(
  argument: ComposerArgument,
  variables: IndicatorVariable[],
  indicators: IndicatorDefinition[],
  contextDomain: IndicatorContextDomain,
  node?: ComposerNode,
  argumentIndex?: number,
) {
  if (parameterRequiresFixedConstant(argument.parameter)) {
    if (argument.source !== 'constant' || !argument.value.trim()) return false
    const value = Number(argument.value)
    if (!Number.isFinite(value)) return false
    if (argument.parameter.constant_kind === 'integer' && !Number.isInteger(value)) return false
    if (argument.parameter.minimum !== undefined && value < argument.parameter.minimum) return false
    if (argument.parameter.maximum !== undefined && value > argument.parameter.maximum) return false
    return true
  }
  if (argument.source === 'omitted') return Boolean(argument.parameter.optional)
  if (!argument.value.trim()) return false
  if (argument.source === 'constant') return Number.isFinite(Number(argument.value))
  if (argument.source === 'operator') {
    return Boolean(
      argument.nested
      && isCompatibleNestedOperator(argument.nested.item, argument.parameter, contextDomain)
      && isComposerNodeValid(argument.nested, variables, indicators, contextDomain),
    )
  }
  if (argument.source === 'indicator') {
    return parameterAcceptsIndicator(argument.parameter)
      && indicators.some((indicator) => indicatorReferenceKey(indicator) === argument.value)
  }
  const variable = variables.find((item) => item.name === argument.value)
  return Boolean(
    variable
    && isCompatibleVariable(variable, argument.parameter, contextDomain)
    && (!node || argumentIndex === undefined || isCompatibleWithComposerSiblings(variable, argumentIndex, node, variables)),
  )
}

function shapeLabel(shape: IndicatorShape, valueType = '') {
  if (shape === 'unknown' && /same\s*\(/i.test(valueType)) return '与输入相同的数据类型'
  if (shape === 'unknown' && /one_dimensional/i.test(valueType)) return '一维数据（按所选轴计算）'
  if (shape === 'unknown' && valueType.includes('|')) {
    const alternatives = [
      /scalar/i.test(valueType) ? '有限标量' : '',
      /series/i.test(valueType) ? '时间序列' : '',
      /vector/i.test(valueType) ? '资产向量' : '',
      /matrix<time\s*,\s*asset>/i.test(valueType) ? '时间—资产矩阵' : /matrix/i.test(valueType) ? '矩阵' : '',
    ].filter(Boolean)
    if (alternatives.length > 1) return `依入参推导（${alternatives.join(' / ')}）`
  }
  const matrixLabel = /matrix<time\s*,\s*asset>|matrix<asset\s*,\s*time>/i.test(valueType)
    ? '时间—资产矩阵'
    : /matrix<asset\s*,\s*asset>|square|covariance|\[n\s*,\s*n\]|n_n/i.test(valueType)
      ? '资产方阵'
      : '矩阵'
  const maskLabel = /mask<time\s*,\s*asset>|mask<asset\s*,\s*time>/i.test(valueType)
    ? '时间—资产布尔掩码'
    : /mask<time>/i.test(valueType)
      ? '时间序列布尔掩码'
      : /mask<asset>/i.test(valueType)
        ? '资产布尔掩码'
        : '布尔掩码'
  return ({ record: '计算中间结果（需提取字段）', scalar: '有限标量', series: '时间序列', vector: '资产向量', matrix: matrixLabel, window: '滚动窗口集合（中间结果）', mask: maskLabel, tuple: '结构化值（兼容）', unknown: '数据类型待推导' } as Record<IndicatorShape, string>)[shape]
}

function valueTypeText(value: unknown) {
  if (typeof value !== 'object' || value === null) return String(value || '')
  const record = value as Record<string, unknown>
  return String(record.display || record.kind || '')
}

function shapeFromValueType(value: unknown): IndicatorShape {
  const raw = valueTypeText(value)
  if (/mask|bool/i.test(raw)) return 'mask'
  if (/matrix/i.test(raw)) return 'matrix'
  if (/series/i.test(raw)) return 'series'
  if (/vector/i.test(raw)) return 'vector'
  if (/tuple/i.test(raw)) return 'tuple'
  if (/scalar/i.test(raw)) return 'scalar'
  return 'unknown'
}

function humanizeTechnicalTypes(value: unknown, fallbackShape: IndicatorShape = 'unknown'): string {
  if (value === null || value === undefined || value === '') return shapeLabel(fallbackShape)
  if (Array.isArray(value)) return value.map((item) => humanizeTechnicalTypes(item)).join(' / ')
  if (typeof value === 'object') {
    const record = value as Record<string, unknown>
    if (record.display) return humanizeTechnicalTypes(record.display, fallbackShape)
    if (record.kind) return humanizeTechnicalTypes(record.kind, fallbackShape)
    return '由输入数据类型决定'
  }
  return humanizeIndicatorTechnicalText(value)
}

function userFacingErrorMessage(error: unknown, fallback = '请求失败，请稍后重试。') {
  if (error instanceof CustomIndicatorApiError) return formatIndicatorDiagnostic(error.code, error.message)
  const message = error instanceof Error ? error.message : fallback
  return humanizeIndicatorMessage(message, fallback)
}

function actualDataSizeLabel(actualShape: number[] | null | undefined, shape: IndicatorShape, valueType = '') {
  if (actualShape === null || actualShape === undefined) return '尚未形成运行窗口'
  if (actualShape.length === 0) return '单个数值'
  if (actualShape.length === 1) {
    if (shape === 'series' || /<time>/i.test(valueType)) return `${actualShape[0]} 个时间点`
    if (shape === 'vector' || /<asset>/i.test(valueType)) return `${actualShape[0]} 个资产`
    return `${actualShape[0]} 个元素`
  }
  if (actualShape.length === 2 && /asset\s*,\s*asset/i.test(valueType)) return `${actualShape[0]} × ${actualShape[1]}（资产 × 资产）`
  if (actualShape.length === 2 && /time\s*,\s*asset|asset\s*,\s*time/i.test(valueType)) return `${actualShape[0]} × ${actualShape[1]}（时间点 × 资产）`
  return `${actualShape.join(' × ')} 个元素`
}

function symbolicDataSizeLabel(symbolicShape: string | Array<string | number> | null | undefined, shape: IndicatorShape, valueType = '') {
  const shapeText = Array.isArray(symbolicShape)
    ? `[${symbolicShape.map((axis) => String(axis)).join(',')}]`
    : typeof symbolicShape === 'string'
      ? symbolicShape
      : ''
  const normalized = shapeText.replace(/\s+/g, '').toUpperCase()
  if (!normalized || normalized === '[]') return shape === 'scalar' ? '单个数值' : '由输入数据规模决定'
  if (normalized === '[T]') return '随计算窗口变化的时间点数量'
  if (normalized === '[L]') return '包含窗口边界点的净值或价格观察数'
  if (normalized === '[N]') return '随组合成分变化的资产数量'
  if (normalized === '[T,N]' || normalized === '[N,T]') return '时间点数量 × 资产数量'
  if (normalized === '[N,N]') return '资产数量 × 资产数量'
  return /scalar/i.test(valueType) ? '单个数值' : '由输入数据规模决定'
}

function diagnosticMessage(code: string, message: string) {
  return formatIndicatorDiagnostic(code, message)
}

function nodeKindLabel(kind: string) {
  return ({ variable: '输入变量', constant: '数值常量', call: '计算算子', binary: '基础运算', unary: '一元运算', comparison: '比较运算' } as Record<string, string>)[kind] || '计算节点'
}

function dagNodeOperatorId(node: IndicatorDagNode) {
  return node.operator_id || node.operator?.id || node.label
}

function dagNodeInferredType(node: IndicatorDagNode) {
  return typeof node.inferred_type === 'object' && node.inferred_type !== null
    ? node.inferred_type
    : null
}

function dagNodeShape(node: IndicatorDagNode): IndicatorShape {
  if (node.shape) return node.shape
  const kind = dagNodeInferredType(node)?.kind
  if (kind && ['scalar', 'series', 'vector', 'matrix', 'window', 'mask', 'tuple'].includes(kind)) return kind as IndicatorShape
  if (node.kind === 'variable') return 'series'
  if (node.kind === 'constant') return 'scalar'
  return 'unknown'
}

function dagNodeValueType(node: IndicatorDagNode) {
  if (node.value_type) return node.value_type
  if (typeof node.inferred_type === 'string') return node.inferred_type
  const inferred = dagNodeInferredType(node)
  return inferred?.display || inferred?.kind || ''
}

function dagNodeSymbolicShape(node: IndicatorDagNode) {
  return node.symbolic_shape ?? dagNodeInferredType(node)?.shape
}

function scopeLabel(scope: string) {
  return ({ single_product: '单产品', portfolio: '组合', etf: 'ETF', fund: '场外基金' } as Record<string, string>)[scope] || scope
}

function measureLabel(measure: string | undefined) {
  return ({ return_decimal: '收益率', rate_decimal: '利率', adjusted_nav: '复权净值', reported_nav: '披露净值', raw_market_price: '市场价格', volume: '成交量', currency_amount: '金额', count: '数量', calendar_days: '日历天数', dimensionless: '无量纲数值' } as Record<string, string>)[measure || 'dimensionless'] || '数值'
}

const OPERATOR_DISPLAY_LABELS: Record<string, string> = {
  rolling_apply: '滚动计算',
  rolling_window: '滚动窗口',
  rolling_mean: '滚动平均值（历史兼容）',
  rolling_std: '滚动标准差（历史兼容）',
  rolling_min: '滚动最小值',
  rolling_max: '滚动最大值',
  recursive_smooth: '递归平滑',
  divide_or_default: '安全除法',
}

function operatorDisplayLabel(operatorId: string | undefined, fallback?: string) {
  const original = fallback || (operatorId ? OPERATOR_DISPLAY_LABELS[operatorId] : '') || systemText('graph.operators')
  return operatorId ? businessText(`operators.${operatorId}.label`, original) : original
}

const OPERATOR_PARAMETER_LABELS: Record<string, string[]> = {
  rolling_apply: ['区间计算内容', '窗口期数', '观察日期（自动绑定）', '年度配置（自动绑定）'],
  rolling_window: ['待处理数值', '窗口期数', '最少有效观察数'],
  rolling_mean: ['待处理数值', '窗口期数', '最少有效观察数'],
  rolling_std: ['待处理数值', '窗口期数', '自由度修正', '最少有效观察数'],
  rolling_min: ['待处理数值', '窗口期数', '最少有效观察数'],
  rolling_max: ['待处理数值', '窗口期数', '最少有效观察数'],
  recursive_smooth: ['待处理数值', '平滑期数', '递归初始值'],
  divide_or_default: ['分子', '分母', '分母为零时的默认值'],
}

function parameterLabel(parameter: string | undefined, fallback: string) {
  if (!parameter) return fallback
  if (fallback && fallback !== parameter && /[^\u0000-\u007f]/.test(fallback)) return fallback
  return ({
    lhs: '输入 A', rhs: '输入 B', left: '输入 A', right: '输入 B',
    values: '待处理数值', value: '待处理数值', x: '输入数值', y: '参考数值',
    numerator: '分子', denominator: '分母', default: '分母为零时的默认值',
    lower: '允许的最小值', upper: '允许的最大值', threshold: '比较阈值',
    q: '分位点', ddof: '自由度修正', axis: '计算维度', periods: '间隔期数',
    window: '窗口期数', min_periods: '最少有效观察数', initial: '递归初始值',
    matrix: '输入矩阵', vector: '输入向量', mask: '判断条件', condition: '判断条件',
  } as Record<string, string>)[parameter] || fallback
}

function operatorParameterLabel(
  operatorId: string | undefined,
  index: number,
  parameter: string | undefined,
  fallback: string,
) {
  const contextual = operatorId ? OPERATOR_PARAMETER_LABELS[operatorId]?.[index] : undefined
  return parameterLabel(parameter, contextual || fallback)
}

function parameterDescription(parameter: IndicatorOperatorParameter) {
  const description = humanizeTechnicalTypes(parameter.description || '')
  if (description && !/^允许类型\s*[:：]/i.test(description)) return description
  const label = parameterLabel(parameter.name, parameter.label || parameter.name)
  return `${label}在当前数学运算中的数值。`
}

function operatorContractDescription(item: ComposerItem) {
  const inputs = item.parameters.map((parameter) => {
    const types = acceptedShapes(parameter).map((shape) => shapeLabel(shape)).join('、') || '由上下文决定'
    return `${parameterLabel(parameter.name, parameter.label || parameter.name)}支持${types}`
  }).join('；')
  return `${inputs || '无需输入参数'}；返回${shapeLabel(item.outputShape, item.outputType)}。`
}

function costEstimateLabel(cost: string | number | undefined) {
  if (cost === null || cost === undefined || cost === '') return '由系统根据输入规模估算'
  if (typeof cost === 'number') return `约 ${cost} 个基础计算单元`
  const normalized = cost.replace(/\s+/g, '').toLowerCase()
  if (normalized === 'elementwise') return '逐元素计算'
  if (normalized === 'reduction') return '单次遍历归约'
  if (/o\([^)]*\^?3[^)]*\)/.test(normalized)) return '随矩阵维度呈立方增长'
  if (/o\([^)]*\^?2[^)]*\)/.test(normalized)) return '随输入规模呈平方增长'
  if (/^o\(/.test(normalized)) return '随输入规模线性增长'
  return humanizeTechnicalTypes(cost)
}

function executionBackendLabel(backend: string | undefined) {
  if (backend === 'numba_njit_fixed_signature') return '高性能编译计算（已固定输入输出类型）'
  if (backend === 'numpy_blas_lapack') return '高性能线性代数计算'
  return '数组向量化计算'
}

function MathNotation({ latex, label }: { latex: string; label: string }) {
  try {
    const markup = katex.renderToString(latex, { throwOnError: true, displayMode: true })
    return <div aria-label={label} className="overflow-x-auto text-slate-900" dangerouslySetInnerHTML={{ __html: markup }} />
  } catch {
    return <p aria-label={label} className="text-sm text-slate-500">数学排版暂不可用</p>
  }
}

function mathFormulaForDisplay(preferred?: string | null, fallback?: string | null) {
  const display = preferred?.trim()
  if (display) return display
  const legacyLatex = fallback?.trim()
  return legacyLatex?.includes('\\') ? legacyLatex : null
}

function operatorOptionContract(item: ComposerItem) {
  const inputs = item.parameters.map((parameter) => {
    const shapes = acceptedShapes(parameter).map((shape) => shapeLabel(shape)).join('/')
    return `${parameter.label || parameter.name}:${shapes}`
  }).join(' + ')
  return `${inputs || '无参数'} → ${shapeLabel(item.outputShape, item.outputType)}`
}

function variableCategory(variable: IndicatorVariable) {
  if (variable.category_id || variable.category || variable.category_label) {
    return {
      id: variable.category_id || variable.category || variable.category_label || 'other',
      label: variable.category_label || variable.category || '其他变量',
    }
  }
  const token = `${variable.name} ${variable.semantic_role || ''} ${variable.source || ''}`.toLowerCase()
  if (/weight|asset_|portfolio|benchmark/.test(token)) return { id: 'portfolio', label: '组合上下文' }
  if (/risk_free|periods_per_year|annualization|config/.test(token)) return { id: 'configuration', label: '基准与配置' }
  if (/return|change|pct/.test(token)) return { id: 'returns', label: '收益与变化' }
  if (/open|high|low|close|nav|price/.test(token)) return { id: 'price', label: '净值与价格' }
  if (/volume|amount|turnover|liquid/.test(token)) return { id: 'trading', label: '成交与流动性' }
  if (/share|size|aum|scale/.test(token)) return { id: 'scale', label: '规模与份额' }
  if (/flow|money|margin|northbound/.test(token)) return { id: 'flow', label: '资金与持仓' }
  return { id: 'other', label: '其他变量' }
}

function operatorCategory(operator: IndicatorOperator) {
  if (operator.category_id || operator.category || operator.category_label) {
    return {
      id: operator.category_id || operator.category || operator.category_label || 'other',
      label: operator.category_label || operator.category || '其他算子',
    }
  }
  const name = operator.name.toLowerCase()
  if (/matmul|matvec|dot|outer|transpose|trace|solve|diag|quadratic/.test(name)) return { id: 'linear_algebra', label: '线性代数' }
  if (/covariance|correlation|variance|std/.test(name)) return { id: 'statistics', label: '统计计算' }
  if (/cumulative|last/.test(name)) return { id: 'cumulative', label: '累计与时序' }
  if (/mean|sum|product|min_value|max_value|_time|_asset/.test(name)) return { id: 'reduction', label: '归约与轴计算' }
  if (/add|subtract|multiply|divide|power/.test(name)) return { id: 'arithmetic', label: '基础运算' }
  return { id: 'transform', label: '逐元素变换' }
}

function mergeParameterContracts(
  primary: IndicatorOperatorParameter[],
  overload: IndicatorOperatorParameter[],
) {
  return primary.map((parameter, index) => {
    const alternative = overload[index]
    if (!alternative) return parameter
    const roles = parameter.allowed_semantic_roles ?? []
    const alternativeRoles = alternative.allowed_semantic_roles ?? []
    return {
      ...parameter,
      allowed_shapes: [...new Set([...acceptedShapes(parameter), ...acceptedShapes(alternative)])],
      allowed_types: [...new Set([...(parameter.allowed_types ?? []), ...(alternative.allowed_types ?? [])])],
      allowed_semantic_roles: roles.length && alternativeRoles.length
        ? [...new Set([...roles, ...alternativeRoles])]
        : undefined,
    }
  })
}

function operatorParameterChoices(operator: IndicatorOperator, fallback: IndicatorOperatorParameter[]) {
  const preferred = operator.parameters?.length ? operator.parameters : fallback
  const choices: IndicatorOperatorParameter[][] = []
  for (const parameters of [preferred, ...(operator.parameter_sets ?? []).map((item) => item.parameters)]) {
    if (!parameters.length) continue
    const key = parameters.map((parameter) => parameter.name).join('|')
    const existingIndex = choices.findIndex((item) => item.map((parameter) => parameter.name).join('|') === key)
    if (existingIndex >= 0) choices[existingIndex] = mergeParameterContracts(choices[existingIndex], parameters)
    else choices.push(parameters)
  }
  return choices.length ? choices : [fallback]
}

function operatorParametersForContext(
  operator: IndicatorOperator,
  fallback: IndicatorOperatorParameter[],
  variables: IndicatorVariable[],
  contextDomain: IndicatorContextDomain,
) {
  const choices = operatorParameterChoices(operator, fallback)
  return choices.find((parameters) => parameters.every((parameter) => (
    parameter.optional
    || parameter.default !== null && parameter.default !== undefined
    || variables.some((variable) => isCompatibleVariable(variable, parameter, contextDomain))
  ))) ?? choices[0]
}

function operatorToComposer(operator: IndicatorOperator, variables: IndicatorVariable[], contextDomain: IndicatorContextDomain): ComposerItem {
  const fallbackParameters: IndicatorOperatorParameter[] = (operator.input_shapes ?? ['scalar'])
    .map((shape, index) => ({ name: `input_${index + 1}`, label: `输入 ${index + 1}`, shape }))
  const parameters = operatorParametersForContext(operator, fallbackParameters, variables, contextDomain)
  const category = operatorCategory(operator)
  return {
    id: operator.name, label: operatorDisplayLabel(operator.name, operator.label), kind: 'operator', signature: operator.signature,
    essence: operator.mathematical_essence || operator.signature || '受控数学算子。',
    semantic: operator.semantic || operator.mathematical_essence || '使用白名单函数对输入值进行计算。', outputShape: operator.output_shape || (/returns|vector|array/i.test(operator.return_type) ? 'vector' : 'scalar'),
    outputType: operator.return_type,
    domains: operator.domains || ['single_product', 'portfolio'], parameters,
    parameterChoices: operatorParameterChoices(operator, fallbackParameters),
    categoryId: category.id,
    categoryLabel: category.label,
    aliases: operator.aliases || [],
    tags: operator.tags || [],
    examples: operator.examples || [],
    intermediateKind: operator.intermediate_kind,
    displayTemplate: operator.display_latex_template || operator.latex_template,
    costEstimate: operator.cost_estimate,
    version: operator.version,
    executionBackend: operator.execution_backend,
  }
}

function editableDraft(source: IndicatorDraft, validation?: ValidationResponse): IndicatorDraft {
  const { editable_latex: latex, ...draft } = source
  const outputs = draft.series_outputs?.map(({ editable_latex: channelLatex, ...output }) => ({
    ...output,
    expression: channelLatex || validation?.output_inferences?.[output.id]?.editable_latex || output.expression,
  }))
  return {
    ...draft,
    expression: draft.result_kind === 'time_series'
      ? outputs?.[0]?.expression || draft.expression
      : latex || validation?.editable_latex || draft.expression,
    ...(outputs ? { series_outputs: outputs } : {}),
  }
}

function asDraft(indicator: IndicatorDefinition): IndicatorDraft {
  const { id: _id, revision: _revision, source: _source, read_only: _readOnly, created_at: _createdAt, updated_at: _updatedAt, display_latex: _displayLatex, math_notation_version: _mathNotationVersion, ...draft } = indicator
  return editableDraft(draft)
}

const newSeriesOutput = (): SeriesOutputDefinition => ({
  id: '',
  label: '',
  expression: '',
  unit: '',
  display_format: 'number',
  precision: 4,
  output_measure: 'auto',
})

function normalizeDraft(draft: IndicatorDraft): IndicatorDraft {
  const { periods: _legacyPeriods, ...definition } = draft
  const resultKind: IndicatorResultKind = draft.result_kind ?? 'scalar'
  if (resultKind === 'time_series') {
    const seriesOutputs = (draft.series_outputs ?? []).map((item) => ({
      ...item,
      id: item.id.trim(),
      label: item.label.trim() || item.id.trim(),
      expression: item.expression.trim(),
      unit: item.unit.trim(),
      precision: Math.min(8, Math.max(0, Number.isFinite(item.precision) ? item.precision : 4)),
      output_measure: item.output_measure || 'auto',
    }))
    const firstOutput = seriesOutputs[0]
    return {
      ...definition,
      name: draft.name.trim() || '未命名时序指标',
      description: draft.description.trim(),
      expression: firstOutput?.expression ?? draft.expression.trim(),
      unit: firstOutput?.unit ?? draft.unit.trim(),
      display_format: firstOutput?.display_format ?? draft.display_format,
      precision: firstOutput?.precision ?? draft.precision,
      direction: draft.direction || 'higher_better',
      indicator_type: draft.indicator_type || 'technical',
      annual_risk_free_rate_percent: Number.isFinite(draft.annual_risk_free_rate_percent)
        ? draft.annual_risk_free_rate_percent
        : 0,
      dsl_version: draft.dsl_version || '2.4.0',
      operator_registry_version: draft.operator_registry_version || '2.4.0',
      context_kind: 'single_product',
      result_kind: 'time_series',
      output_contract: 'series_bundle',
      output_measure: 'series_bundle',
      parameter_schema: draft.parameter_contract_version === '1.0' ? draft.parameter_schema ?? [] : [],
      fixed_parameters: draft.fixed_parameters ?? [],
      series_outputs: seriesOutputs,
      axis_anchor: draft.axis_anchor || 'market_close',
      history_policy: draft.history_policy || 'lookback',
      history_inference_source: draft.history_inference_source || 'typed_dag',
      lookback_parameter: null,
      lookback_observations: Math.max(1, Math.round(draft.lookback_observations ?? 1)),
      minimum_observations: Math.max(1, Math.round(draft.minimum_observations ?? 1)),
      methodology: draft.methodology?.trim() || draft.description.trim(),
      data_basis: draft.data_basis?.trim() || '真实数据、日期轴对齐、缺失不填充',
      template_origin: draft.template_origin ?? null,
      rolling_source: draft.rolling_source ?? null,
      rolling_transform: draft.rolling_transform ?? null,
    }
  }
  return {
    ...definition,
    name: draft.name.trim() || '未命名指标',
    description: draft.description.trim(),
    expression: draft.expression.trim(),
    precision: Math.min(8, Math.max(0, Number.isFinite(draft.precision) ? draft.precision : 2)),
    annual_risk_free_rate_percent: Number.isFinite(draft.annual_risk_free_rate_percent)
      ? draft.annual_risk_free_rate_percent
      : 0,
    dsl_version: draft.dsl_version || '2.4.0',
    operator_registry_version: draft.operator_registry_version || '2.4.0',
    context_kind: draft.context_kind || 'single_product',
    result_kind: 'scalar',
    output_contract: 'scalar',
    output_measure: draft.output_measure || 'dimensionless',
    parameter_contract_version: null,
    parameter_schema: [],
    fixed_parameters: [],
    series_outputs: [],
    axis_anchor: null,
    history_policy: null,
    history_inference_source: null,
    lookback_parameter: null,
    lookback_observations: undefined,
    template_origin: draft.template_origin ?? null,
    rolling_source: null,
    rolling_transform: null,
  }
}

function convertDraftResultKind(
  draft: IndicatorDraft,
  resultKind: IndicatorResultKind,
  metadata: IndicatorMeta | null,
): IndicatorDraft {
  if (resultKind === 'scalar') {
    return normalizeDraft({
      ...draftForCurrentRegistries(metadata, STUDIO_CONTEXT),
      name: draft.name,
      description: draft.description,
      result_kind: 'scalar',
    })
  }
  const output = newSeriesOutput()
  const normalized = normalizeDraft({
    ...draftForCurrentRegistries(metadata, STUDIO_CONTEXT),
    name: draft.name,
    description: draft.description,
    result_kind: 'time_series',
    output_contract: 'series_bundle',
    indicator_type: 'technical',
    expression: '',
    parameter_schema: [],
    series_outputs: [output],
    axis_anchor: null,
    history_policy: null,
    history_inference_source: null,
    lookback_parameter: null,
    lookback_observations: undefined,
    minimum_observations: undefined,
  })
  return {
    ...normalized,
    expression: '',
    axis_anchor: null,
    history_policy: null,
    history_inference_source: null,
    lookback_observations: undefined,
    minimum_observations: undefined,
  }
}

function draftForCurrentRegistries(
  metadata: IndicatorMeta | null,
  contextKind: IndicatorContextDomain,
): IndicatorDraft {
  return {
    ...EMPTY_DRAFT,
    dsl_version: metadata?.dsl_version || EMPTY_DRAFT.dsl_version,
    operator_registry_version: metadata?.operator_registry_version || EMPTY_DRAFT.operator_registry_version,
    numeric_kernel_version: metadata?.numeric_kernel_version || EMPTY_DRAFT.numeric_kernel_version,
    variable_registry_version: metadata?.variable_registry_version || EMPTY_DRAFT.variable_registry_version,
    context_schema_version: metadata?.context_schema_version || EMPTY_DRAFT.context_schema_version,
    data_contract_version: metadata?.data_contract_version || EMPTY_DRAFT.data_contract_version,
    context_kind: contextKind,
  }
}

function displayValue(result: EvaluationResult, indicator: IndicatorDraft): string {
  if (indicator.display_format === 'date') return typeof result.value === 'string' ? result.value : '不可计算'
  if (typeof result.value !== 'number' || !Number.isFinite(result.value)) return '不可计算'
  const value = result.value
  const digits = indicator.precision
  return indicator.display_format === 'percent'
    ? `${(value * 100).toFixed(digits)}%`
    : value.toFixed(digits)
}

function targetFromItem(item: InstrumentSearchItem): StudioTarget | null {
  const productId = item.code ?? item.ts_code
  if (!productId) return null
  const kind = item.instrument_type === 'fund' ? 'fund' : 'etf'
  return { kind, product_id: productId, name: item.name || productId }
}

function ApiMessage({ error }: { error: unknown }) {
  if (!error) return null
  const message = userFacingErrorMessage(error)
  return <p role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{message}</p>
}

export default function IndicatorStudio() {
  const [searchParams] = useSearchParams()
  const { s, b, version: translationVersion } = useI18n()
  const [rawMeta, setMeta] = useState<IndicatorMeta | null>(null)
  const meta = useMemo(() => localizeIndicatorMeta(rawMeta), [rawMeta, translationVersion])
  const [indicators, setIndicators] = useState<IndicatorDefinition[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [draft, setDraft] = useState<IndicatorDraft>(EMPTY_DRAFT)
  const [baseline, setBaseline] = useState(JSON.stringify(EMPTY_DRAFT))
  const [validation, setValidation] = useState<ValidationResponse | null>(null)
  const [results, setResults] = useState<EvaluationResult[]>([])
  const [seriesResults, setSeriesResults] = useState<TimeSeriesIndicatorResult[]>([])
  const [runtimeParameters, setRuntimeParameters] = useState<Record<string, number>>({})
  const [parameterPending, setParameterPending] = useState(false)
  const [activeSeriesOutputId, setActiveSeriesOutputId] = useState('')
  const [targets, setTargets] = useState<StudioTarget[]>([])
  const [searchKind, setSearchKind] = useState<ProductKind | 'all'>((searchParams.get('kind') as ProductKind) || 'all')
  const [searchText, setSearchText] = useState('')
  const [searchResults, setSearchResults] = useState<InstrumentSearchItem[]>([])
  const [period, setPeriod] = useState(searchParams.get('period') || '1Y')
  const [asOf, setAsOf] = useState(searchParams.get('as_of') || '')
  const [mobileTab, setMobileTab] = useState<MobileTab>('library')
  const [workspaceTab, setWorkspaceTab] = useState<WorkspaceTab>(() => searchParams.get('ids') ? 'preview' : 'editor')
  const [catalogTab, setCatalogTab] = useState<CatalogTab>('variables')
  const [editorMode, setEditorMode] = useState<EditorMode>('guided')
  const [canvasPending, setCanvasPending] = useState(false)
  const [canvasSession, setCanvasSession] = useState(0)
  const [canvasVisited, setCanvasVisited] = useState(false)
  const [libraryCollapsed, setLibraryCollapsed] = useState(false)
  const [basicInfoOpen, setBasicInfoOpen] = useState(false)
  const canvasPendingRef = useRef(false)
  const canvasEditorRef = useRef<IndicatorGraphEditorHandle>(null)
  const [catalogOpen, setCatalogOpen] = useState(false)
  const [catalogError, setCatalogError] = useState<string | null>(null)
  const [insertingIndicatorId, setInsertingIndicatorId] = useState<string | null>(null)
  const [composer, setComposer] = useState<ComposerState>(null)
  const [guidedTree, setGuidedTree] = useState<ComposerNode | null>(null)
  const [composerLoading, setComposerLoading] = useState(false)
  const [formulaBuilderOpening, setFormulaBuilderOpening] = useState(false)
  const [composerError, setComposerError] = useState<string | null>(null)
  const [inference, setInference] = useState<InferenceResponse | null>(null)
  const [snapshotConfig, setSnapshotConfig] = useState<SnapshotIndicatorConfig | null>(null)
  const [snapshotPeriods, setSnapshotPeriods] = useState<string[]>([])
  const [snapshotSeriesChannel, setSnapshotSeriesChannel] = useState('')
  const [snapshotSaving, setSnapshotSaving] = useState(false)
  const [snapshotError, setSnapshotError] = useState<string | null>(null)
  const [rollingSourceIndicatorId, setRollingSourceIndicatorId] = useState('')
  const [rollingWindowObservations, setRollingWindowObservations] = useState<number | ''>('')
  const [rollingDeriving, setRollingDeriving] = useState(false)
  const [rollingDerivationError, setRollingDerivationError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [validating, setValidating] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const [excelExporting, setExcelExporting] = useState(false)
  const [searching, setSearching] = useState(false)
  const [indicatorQuery, setIndicatorQuery] = useState('')
  const [indicatorSourceFilter, setIndicatorSourceFilter] = useState<'all' | 'built_in' | 'custom'>('all')
  const [indicatorResultKindFilter, setIndicatorResultKindFilter] = useState<'all' | IndicatorResultKind>('all')
  const [indicatorCategoryFilter, setIndicatorCategoryFilter] = useState('all')
  const [message, setMessage] = useState<string | null>(null)
  const [error, setError] = useState<unknown>(null)
  const expressionRef = useRef<HTMLTextAreaElement>(null)
  const validationRequestRef = useRef(0)
  const previewRequestRef = useRef(0)
  const previewContextRef = useRef('')
  const composerRequestRef = useRef(0)

  const selectedIndicator = indicators.find((item) => item.id === selectedId) ?? null
  const isTimeSeries = (draft.result_kind ?? 'scalar') === 'time_series'
  const hasNamedOutputs = isTimeSeries
  const seriesOutputs: SeriesOutputDefinition[] = draft.series_outputs ?? []
  const activeSeriesOutputIndex = Math.max(
    0,
    seriesOutputs.findIndex((item) => item.id === activeSeriesOutputId),
  )
  const activeSeriesOutput = seriesOutputs[activeSeriesOutputIndex] ?? null
  const currentExpression = hasNamedOutputs
    ? activeSeriesOutput?.expression ?? ''
    : draft.expression
  const hasDefinitionFormula = !loading && Boolean(meta) && !canvasPending && !parameterPending && (hasNamedOutputs
    ? seriesOutputs.length > 0 && seriesOutputs.every((item) => item.expression.trim())
    : Boolean(draft.expression.trim()))
  const periods = meta?.periods?.length ? meta.periods : FALLBACK_PERIODS
  const seriesOutputMeasures = meta?.series_output_measures?.length
    ? meta.series_output_measures
    : FALLBACK_SERIES_OUTPUT_MEASURES
  const definitionDirty = baseline !== JSON.stringify(normalizeDraft(draft))
  const isDirty = canvasPending || parameterPending || definitionDirty
  const canvasPendingChanged = (pending: boolean) => {
    canvasPendingRef.current = pending
    setCanvasPending(pending)
    if (pending) {
      validationRequestRef.current += 1
      previewRequestRef.current += 1
      setPreviewing(false)
      setValidating(false)
      setValidation(null)
      setInference(null)
      setResults([])
      setSeriesResults([])
    }
  }
  const resetCanvasSession = () => {
    canvasPendingRef.current = false
    setCanvasPending(false)
    setCanvasSession(value => value + 1)
  }
  const changeEditorMode = (mode: EditorMode) => {
    if (mode === editorMode) return
    if (canvasPendingRef.current) {
      if (!window.confirm('画布尚未应用。放弃画布修改并切换编辑方式？')) return
      resetCanvasSession()
    }
    setEditorMode(mode)
    if (mode === 'canvas') setCanvasVisited(true)
    setLibraryCollapsed(mode === 'canvas')
    setCatalogOpen(false)
    setComposer(null)
  }
  const activePeriod = periods.some((item) => item.value === period) ? period : periods[0]?.value || ''
  previewContextRef.current = JSON.stringify({ draft, selectedId, targets, activePeriod, asOf, canvasPending, runtimeParameters })
  const parameterSchemaKey = JSON.stringify(draft.parameter_schema ?? [])
  useEffect(() => {
    setRuntimeParameters({})
    setSeriesResults([])
    previewRequestRef.current += 1
    setPreviewing(false)
  }, [selectedId, parameterSchemaKey])
  const applyRuntimeParameters = (values: Record<string, number>) => {
    previewRequestRef.current += 1
    setPreviewing(false)
    setSeriesResults([])
    setRuntimeParameters(values)
  }
  const contextIndicators = indicatorsForContext(indicators, STUDIO_CONTEXT)
  const rollingScalarIndicators = contextIndicators
    .filter((item) => (
      (item.result_kind ?? 'scalar') === 'scalar'
      && (item.dsl_version ?? '').startsWith('2.')
      && (item.output_contract ?? 'scalar') === 'scalar'
      && item.rolling_series_compatibility?.supported === true
    ))
    .sort((left, right) => (
      left.id === 'builtin-annualized-sharpe-v2' ? -1
        : right.id === 'builtin-annualized-sharpe-v2' ? 1
          : left.name.localeCompare(right.name, 'zh-CN')
    ))
  const effectiveRollingSourceIndicatorId = rollingScalarIndicators.some(
    (item) => item.id === rollingSourceIndicatorId,
  )
    ? rollingSourceIndicatorId
    : ''
  const rollingSource = rollingScalarIndicators.find((item) => item.id === effectiveRollingSourceIndicatorId)
  const rollingWindowValid = rollingWindowObservations !== '' && Number.isInteger(rollingWindowObservations) && rollingWindowObservations >= 1 && rollingWindowObservations <= 5000
  const normalizedIndicatorQuery = indicatorQuery.trim().toLowerCase()
  const visibleIndicators = contextIndicators.filter((item) => (
    (indicatorSourceFilter === 'all' || item.source === indicatorSourceFilter)
    && (indicatorResultKindFilter === 'all' || (item.result_kind ?? 'scalar') === indicatorResultKindFilter)
    && (indicatorCategoryFilter === 'all' || item.category_id === indicatorCategoryFilter)
    && (!normalizedIndicatorQuery || `${item.name} ${item.description} ${item.expression}`.toLowerCase().includes(normalizedIndicatorQuery))
  ))
  const customIndicators = visibleIndicators.filter((item) => item.source === 'custom')
  const builtInIndicators = visibleIndicators.filter((item) => item.source === 'built_in')
  const indicatorCategories = meta?.indicator_types?.length
    ? meta.indicator_types
    : meta?.indicator_categories?.length
      ? meta.indicator_categories
    : [...new Map(contextIndicators.filter((item) => item.category_id).map((item) => [item.category_id as string, item.category_label || item.category_id as string])).entries()].map(([id, label]) => ({ id, label }))
  const variables = useMemo(
    () => (meta?.variables?.length ? meta.variables : FALLBACK_VARIABLES)
      .filter((variable) => supportsVariableDomain(variable, STUDIO_CONTEXT)),
    [meta],
  )
  const formulaVariables = variables
  const composerVariables = useMemo(
    () => formulaVariables.filter((variable) => (
      variable.availability !== 'unavailable' && variable.availability !== 'not_applicable'
    )),
    [formulaVariables],
  )
  const operatorItems = useMemo(
    () => (meta?.operators ?? [])
      .map((operator) => operatorToComposer(operator, composerVariables, STUDIO_CONTEXT))
      .filter((operator) => operator.domains.includes(STUDIO_CONTEXT)),
    [composerVariables, meta?.operators],
  )
  const composableIndicators = contextIndicators
    .filter((indicator) => indicatorIsComposable(indicator, draft, STUDIO_CONTEXT))
    .sort((left, right) => (
      composableProtocolPriority(right, draft) - composableProtocolPriority(left, draft)
      || left.name.localeCompare(right.name, 'zh-CN')
    ))
  const composerReady = composer
    ? isComposerNodeValid(composer.node, composerVariables, composableIndicators, STUDIO_CONTEXT)
    : false
  const resourceLabels = useMemo(() => new Map<string, string>([
    ...formulaVariables.map((variable) => [variable.name, variable.label] as const),
    ...operatorItems.map((operator) => [operator.id, operator.label] as const),
  ]), [formulaVariables, operatorItems])

  const validatedSeriesInference = useMemo(
    () => hasNamedOutputs && activeSeriesOutput
      ? inferenceForSeriesOutput(validation, activeSeriesOutput.id)
      : null,
    [activeSeriesOutput, hasNamedOutputs, validation],
  )
  const displayedInference = hasNamedOutputs
    ? validatedSeriesInference ?? inference
    : inference
  const validatedSeriesTree = useMemo(
    () => hasNamedOutputs && activeSeriesOutput && validation?.valid
      ? composerNodeFromDag(validation.dag, operatorItems, activeSeriesOutput.id)
      : null,
    [activeSeriesOutput, hasNamedOutputs, operatorItems, validation],
  )
  const displayedGuidedTree = hasNamedOutputs
    ? guidedTree ?? validatedSeriesTree
    : guidedTree
  const activeFormulaDag = hasNamedOutputs
    ? displayedInference?.dag ?? dagForRoot(validation?.dag, activeSeriesOutput?.id ?? '')
    : validation?.dag ?? displayedInference?.dag ?? null
  const displayLatex = displayedInference?.display_latex
    || (!isTimeSeries ? validation?.display_latex : null)
    || (!isTimeSeries && selectedIndicator
      && (selectedIndicator.expression === draft.expression || selectedIndicator.editable_latex === draft.expression)
      ? selectedIndicator.display_latex
      : null)
  const formulaMarkup = useMemo(() => {
    try {
      const formula = displayLatex
        || (currentExpression
          ? '\\text{请先解析并校验公式以生成数学排版}'
          : '\\text{等待输入公式}')
      return { __html: katex.renderToString(formula, { throwOnError: true, displayMode: true }) }
    } catch {
      return { __html: '<span>公式排版不可用</span>' }
    }
  }, [currentExpression, displayLatex])

  const refreshCatalog = async () => {
    const response = await listCustomIndicators({ contextKind: STUDIO_CONTEXT })
    setIndicators(response.items)
    return response.items
  }

  useEffect(() => {
    let active = true
    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const [metadata, catalog] = await Promise.all([
          getCustomIndicatorMeta(),
          listCustomIndicators({ contextKind: STUDIO_CONTEXT }),
        ])
        if (!active) return
        setMeta(metadata)
        setIndicators(catalog.items)
        const initialDraft = draftForCurrentRegistries(metadata, 'single_product')
        // The user may already be typing while the registry request is in
        // flight. Initialize technical defaults, never replace their draft.
        setDraft(current => {
          const next = { ...current }
          for (const key of ['dsl_version', 'operator_registry_version', 'numeric_kernel_version', 'variable_registry_version', 'context_schema_version', 'data_contract_version'] as const) {
            if (current[key] === EMPTY_DRAFT[key]) next[key] = initialDraft[key]
          }
          return next
        })
        setBaseline(JSON.stringify(normalizeDraft(initialDraft)))
        const availablePeriods = metadata.periods.map((item) => item.value)
        setPeriod(current => availablePeriods.includes(current) ? current : availablePeriods[0] || '1Y')
      } catch (loadError) {
        if (active) setError(loadError)
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, []) // load once; the selected formula is local state

  useEffect(() => {
    if (!hasNamedOutputs) {
      if (activeSeriesOutputId) setActiveSeriesOutputId('')
      return
    }
    const outputs = draft.series_outputs ?? []
    if (!outputs.some(item => item.id === activeSeriesOutputId)) setActiveSeriesOutputId(outputs[0]?.id ?? '')
  }, [activeSeriesOutputId, draft.series_outputs, hasNamedOutputs])

  useEffect(() => {
    let active = true
    void getSnapshotIndicatorConfig()
      .then((response) => {
        if (active) setSnapshotConfig({
          schema_version: response.schema_version ?? 1,
          revision: response.revision ?? 1,
          max_items: response.max_items ?? 30,
          items: Array.isArray(response.items) ? response.items : [],
          updated_at: response.updated_at ?? null,
          snapshot: response.snapshot ?? null,
        })
      })
      .catch((failure) => {
        if (active) setSnapshotError(userFacingErrorMessage(failure, '快照指标配置暂时无法读取。'))
      })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (!selectedIndicator || !snapshotConfig) {
      setSnapshotPeriods([])
      return
    }
    const matchingItems = snapshotConfig.items
      .filter((item) => item.indicator_id === selectedIndicator.id && item.indicator_revision === selectedIndicator.revision)
    setSnapshotPeriods(matchingItems.map((item) => item.period))
    if ((selectedIndicator.result_kind ?? 'scalar') === 'time_series') {
      setSnapshotSeriesChannel(
        matchingItems.find((item) => item.channel_id)?.channel_id
        || selectedIndicator.series_outputs?.[0]?.id
        || '',
      )
    } else {
      setSnapshotSeriesChannel('')
    }
  }, [selectedIndicator, snapshotConfig])

  useEffect(() => {
    const ids = (searchParams.get('ids') || '').split(',').map((id) => id.trim()).filter(Boolean)
    if (!ids.length) return
    const kind = (searchParams.get('kind') as ProductKind) || 'etf'
    const previewIds = [...new Set(ids)].slice(0, MAX_PREVIEW_TARGETS)
    setTargets(previewIds.map((product_id) => ({ kind, product_id, name: product_id })))
    setResults([])
    setSeriesResults([])
    if (ids.length > MAX_PREVIEW_TARGETS) {
      setMessage(`校验与预览最多选择 ${MAX_PREVIEW_TARGETS} 个产品，已保留前 ${MAX_PREVIEW_TARGETS} 个。`)
    }
    void searchInstruments({ kind, query: previewIds.join(' '), pageSize: MAX_PREVIEW_TARGETS })
      .then((response) => {
        const names = new Map(response.items.map((item) => [item.code ?? item.ts_code, item.name]))
        setTargets((current) => current.map((target) => ({ ...target, name: names.get(target.product_id) || target.name })))
      })
      .catch(() => undefined)
  }, [searchParams])

  const selectDraft = (indicator: IndicatorDefinition) => {
    if (isDirty && !window.confirm('当前未保存的修改将被替换，是否继续？')) return
    resetCanvasSession()
    validationRequestRef.current += 1
    setValidating(false)
    const keepPreviewContext = workspaceTab === 'preview'
    const next = normalizeDraft(asDraft(indicator))
    setSelectedId(indicator.id)
    const productDraft = normalizeDraft({ ...next, context_kind: STUDIO_CONTEXT })
    setDraft(productDraft)
    setBaseline(JSON.stringify(productDraft))
    setValidation(null)
    setInference(null)
    setGuidedTree(null)
    setResults([])
    setSeriesResults([])
    const loadedMessage = indicator.read_only ? '已载入内置指标。可编辑后另存为工作区自定义指标。' : `已载入版本 ${indicator.revision}。`
    const previewMessage = `${loadedMessage} 已保留预览条件。`
    setMessage(keepPreviewContext ? previewMessage : loadedMessage)
    if (mobileTab === 'library') setMobileTab(workspaceTab)
    if (keepPreviewContext && productDraft.expression.trim()) void validateDefinition(productDraft, previewMessage)
  }

  const createNew = () => {
    if (isDirty && !window.confirm('当前未保存的修改将被替换，是否继续？')) return
    resetCanvasSession()
    validationRequestRef.current += 1
    setValidating(false)
    const next = draftForCurrentRegistries(meta, STUDIO_CONTEXT)
    setSelectedId(null)
    setDraft(next)
    setBaseline(JSON.stringify(next))
    setValidation(null)
    setInference(null)
    setGuidedTree(null)
    setResults([])
    setSeriesResults([])
    setEditorMode('guided')
    setMessage('已创建新指标草稿。请从变量、数学算子或已有指标开始构建公式。')
    setMobileTab('editor')
    setWorkspaceTab('editor')
  }

  const deriveFromScalarIndicator = async () => {
    const source = rollingScalarIndicators.find((item) => item.id === effectiveRollingSourceIndicatorId)
    if (!source) {
      setRollingDerivationError('请选择一个可滚动转换的单产品标量指标。')
      return
    }
    const windowObservations = Number(rollingWindowObservations)
    if (!rollingWindowValid) {
      setRollingDerivationError('滚动观察数必须是 1 至 5000 的整数，不能四舍五入或留空。')
      return
    }
    if (isDirty && !window.confirm('滚动派生会替换当前未保存公式，是否继续？')) return
    try {
      setRollingDeriving(true)
      setRollingDerivationError(null)
      setError(null)
      const response = await deriveRollingSeriesIndicator({
        indicator_id: source.id,
        indicator_revision: source.revision,
        window_observations: windowObservations,
      })
      resetCanvasSession()
      const next = normalizeDraft(editableDraft(response.definition, response.validation))
      const emptySeries = convertDraftResultKind(
        draftForCurrentRegistries(meta, STUDIO_CONTEXT),
        'time_series',
        meta,
      )
      validationRequestRef.current += 1
      setSelectedId(null)
      setDraft(next)
      setBaseline(JSON.stringify(emptySeries))
      setValidation(response.validation)
      setInference(null)
      setGuidedTree(null)
      setResults([])
      setSeriesResults([])
      setActiveSeriesOutputId(next.series_outputs?.[0]?.id ?? '')
      setEditorMode('guided')
      setWorkspaceTab('editor')
      setMobileTab('editor')
      const derivedMessage = `已从“${source.name}”v${source.revision} 生成 ${windowObservations} 日滚动时序草稿；保存后形成独立锁定版本。`
      if (JSON.stringify(next.series_outputs) !== JSON.stringify(response.definition.series_outputs)) {
        // Compile tokens bind source text. Obtain a token for the LaTeX draft
        // rather than reusing the derivation's canonical-DSL token.
        await validateDefinition(next, derivedMessage)
      } else {
        setMessage(derivedMessage)
      }
    } catch (failure) {
      setRollingDerivationError(userFacingErrorMessage(failure, '标量指标无法转换为滚动时序指标。'))
    } finally {
      setRollingDeriving(false)
    }
  }

  const saveSnapshotPeriods = async () => {
    if (!selectedIndicator || !snapshotConfig) return
    if ((selectedIndicator.result_kind ?? 'scalar') === 'time_series' && snapshotPeriods.length && !snapshotSeriesChannel) {
      setSnapshotError('请选择要归约为快照值的输出通道。')
      return
    }
    try {
      setSnapshotSaving(true)
      setSnapshotError(null)
      const untouched = snapshotConfig.items.filter(item => item.indicator_id !== selectedIndicator.id)
      const nextItems = [
        ...untouched.map((item) => ({
          indicator_id: item.indicator_id,
          indicator_revision: item.indicator_revision,
          period: item.period,
          channel_id: item.channel_id ?? null,
          reducer: item.reducer ?? null,
        })),
        ...snapshotPeriods.map((snapshotPeriod) => ({
          indicator_id: selectedIndicator.id,
          indicator_revision: selectedIndicator.revision,
          period: snapshotPeriod,
          channel_id: (selectedIndicator.result_kind ?? 'scalar') === 'time_series' ? snapshotSeriesChannel : null,
          reducer: (selectedIndicator.result_kind ?? 'scalar') === 'time_series' ? 'last_finite' as const : null,
        })),
      ]
      const updated = await updateSnapshotIndicatorConfig(snapshotConfig.revision, nextItems)
      setSnapshotConfig(updated)
      setMessage(snapshotPeriods.length
        ? '快照指标配置已保存；将在下一次数据刷新后生成预计算结果。'
        : '该指标已从快照加速配置中移除。')
    } catch (failure) {
      setSnapshotError(userFacingErrorMessage(failure, '快照指标配置保存失败。'))
    } finally {
      setSnapshotSaving(false)
    }
  }

  const patchDraft = (patch: Partial<IndicatorDraft>) => {
    composerRequestRef.current += 1
    previewRequestRef.current += 1
    setPreviewing(false)
    validationRequestRef.current += 1
    setValidating(false)
    setDraft((current) => {
      const next = { ...current, ...patch }
      const previousOutputs = (current.series_outputs ?? []).map((item) => ({ id: item.id, expression: item.expression }))
      const nextOutputs = (next.series_outputs ?? []).map((item) => ({ id: item.id, expression: item.expression }))
      const rollingContractChanged = (
        (Object.prototype.hasOwnProperty.call(patch, 'expression') && next.expression !== current.expression)
        || (Object.prototype.hasOwnProperty.call(patch, 'axis_anchor') && next.axis_anchor !== current.axis_anchor)
        || (Object.prototype.hasOwnProperty.call(patch, 'annual_risk_free_rate_percent')
          && next.annual_risk_free_rate_percent !== current.annual_risk_free_rate_percent)
        || (Object.prototype.hasOwnProperty.call(patch, 'series_outputs')
          && JSON.stringify(previousOutputs) !== JSON.stringify(nextOutputs))
      )
      if (
        rollingContractChanged
        && current.rolling_source
        && !current.rolling_source.detached
        && !Object.prototype.hasOwnProperty.call(patch, 'rolling_source')
      ) {
        next.rolling_source = { ...current.rolling_source, detached: true }
      }
      return next
    })
    setValidation(null)
    setInference(null)
    setGuidedTree(null)
    setResults([])
    setSeriesResults([])
    if (
      Object.prototype.hasOwnProperty.call(patch, 'expression')
      || Object.prototype.hasOwnProperty.call(patch, 'series_outputs')
      || Object.prototype.hasOwnProperty.call(patch, 'parameter_schema')
    ) setMessage('公式已更改，需要重新解析并校验。')
  }

  const changeResultKind = (resultKind: IndicatorResultKind) => {
    if ((draft.result_kind ?? 'scalar') === resultKind) return
    if (canvasPendingRef.current && !window.confirm('画布尚未应用。放弃画布修改并切换结果类型？')) return
    resetCanvasSession()
    const next = convertDraftResultKind(draft, resultKind, meta)
    validationRequestRef.current += 1
    setValidating(false)
    setDraft(next)
    setValidation(null)
    setInference(null)
    setGuidedTree(null)
    setResults([])
    setSeriesResults([])
    setEditorMode('guided')
    setMessage(resultKind === 'time_series'
      ? '已切换为时序指标。请填写日期轴、输出通道和计算公式；支持的常数输入可按需开放为可变参数。'
      : '已切换为标量指标。请重新构建最终标量公式。')
  }

  const patchSeriesOutput = (index: number, patch: Partial<SeriesOutputDefinition>) => {
    const current = draft.series_outputs ?? []
    const previous = current[index]
    if (!previous) return
    const next = current.map((item, itemIndex) => itemIndex === index ? { ...item, ...patch } : item)
    const first = next[0]
    if (patch.id && activeSeriesOutputId === previous.id) setActiveSeriesOutputId(patch.id)
    patchDraft({
      series_outputs: next,
      expression: first?.expression ?? '',
      unit: first?.unit ?? '',
      display_format: first?.display_format ?? 'number',
      precision: first?.precision ?? 4,
    })
  }

  const addSeriesOutput = () => {
    const existing = new Set((draft.series_outputs ?? []).map((item) => item.id))
    let index = 1
    while (existing.has(`channel_${index}`)) index += 1
    const item: SeriesOutputDefinition = {
      id: `channel_${index}`,
      label: '',
      expression: '',
      unit: '',
      display_format: 'number',
      precision: 4,
      output_measure: 'auto',
    }
    const next = [...(draft.series_outputs ?? []), item]
    const first = next[0]
    setActiveSeriesOutputId(item.id)
    patchDraft({
      series_outputs: next,
      expression: first.expression,
      unit: first.unit,
      display_format: first.display_format,
      precision: first.precision,
    })
  }

  const removeSeriesOutput = (index: number) => {
    const current = draft.series_outputs ?? []
    if (current.length <= 1) {
      setMessage('时序指标至少需要一个输出通道。')
      return
    }
    const removed = current[index]
    const next = current.filter((_item, itemIndex) => itemIndex !== index)
    const first = next[0]
    if (removed?.id === activeSeriesOutputId) setActiveSeriesOutputId(first.id)
    patchDraft({
      series_outputs: next,
      expression: first.expression,
      unit: first.unit,
      display_format: first.display_format,
      precision: first.precision,
    })
  }

  const selectSeriesOutput = (outputId: string) => {
    composerRequestRef.current += 1
    setActiveSeriesOutputId(outputId)
    const nextInference = inferenceForSeriesOutput(validation, outputId)
    setInference(nextInference)
    setGuidedTree(
      validation?.valid
        ? composerNodeFromDag(validation.dag, operatorItems, outputId)
        : null,
    )
    setComposer(null)
    setComposerError(null)
  }

  const patchCurrentExpression = (
    expression: string,
    seriesOutputId = activeSeriesOutput?.id,
  ) => {
    if (isTimeSeries) {
      const outputIndex = Math.max(
        0,
        (draft.series_outputs ?? []).findIndex((item) => item.id === seriesOutputId),
      )
      patchSeriesOutput(outputIndex, { expression })
      return
    }
    patchDraft({ expression, template_origin: null })
  }

  const insertExpression = (token: string) => {
    if (editorMode === 'guided') {
      patchCurrentExpression(token)
      setCatalogOpen(false)
      setMessage(isTimeSeries
        ? `变量已设为“${activeSeriesOutput?.label ?? '当前输出'}”的计算公式；可以继续嵌套数学算子。`
        : '变量已设为当前公式；可以继续用数学算子构建标量结果。')
      return
    }
    const textarea = expressionRef.current
    const start = textarea?.selectionStart ?? currentExpression.length
    const end = textarea?.selectionEnd ?? currentExpression.length
    const expression = `${currentExpression.slice(0, start)}${token}${currentExpression.slice(end)}`
    patchCurrentExpression(expression)
    window.requestAnimationFrame(() => {
      textarea?.focus()
      textarea?.setSelectionRange(start + token.length, start + token.length)
    })
    setCatalogOpen(false)
  }

  const insertIndicator = async (indicator: IndicatorDefinition) => {
    try {
      setCatalogError(null)
      setInsertingIndicatorId(indicator.id)
      const response = await composeCustomIndicator({
        indicator_id: indicator.id,
        indicator_revision: indicator.revision,
        arguments: [],
        context: STUDIO_CONTEXT,
        dsl_version: draft.dsl_version,
        operator_registry_version: draft.operator_registry_version,
        variable_registry_version: draft.variable_registry_version,
        data_contract_version: draft.data_contract_version,
        context_schema_version: draft.context_schema_version,
      })
      const expanded = composedEditableLatex(response)
      const textarea = expressionRef.current
      const replaceFullFormula = editorMode === 'guided' || !currentExpression.trim()
      const start = replaceFullFormula ? 0 : textarea?.selectionStart ?? currentExpression.length
      const end = replaceFullFormula ? currentExpression.length : textarea?.selectionEnd ?? start
      const token = replaceFullFormula ? expanded : `(${expanded})`
      const expression = `${currentExpression.slice(0, start)}${token}${currentExpression.slice(end)}`
      patchCurrentExpression(expression)
      if (replaceFullFormula) {
        setInference(response)
        setGuidedTree(composerNodeFromDag(response.dag, operatorItems))
      }
      setCatalogOpen(false)
      setMessage(
        `“${indicator.name}” v${indicator.revision} 已按锁定版本展开${replaceFullFormula ? '为当前公式' : '到高级公式'}；原指标后续更新不会影响本草稿。`,
      )
    } catch (insertError) {
      setCatalogError(userFacingErrorMessage(insertError, '已有指标展开失败，请稍后重试。'))
    } finally {
      setInsertingIndicatorId(null)
    }
  }

  const openComposer = (item: ComposerItem) => {
    composerRequestRef.current += 1
    const textarea = expressionRef.current
    const selectionStart = textarea?.selectionStart ?? currentExpression.length
    const selectionEnd = textarea?.selectionEnd ?? selectionStart
    setComposerError(null)
    setCatalogOpen(false)
    const node = initialComposerNode(item, composerVariables, STUDIO_CONTEXT)
    if (displayedGuidedTree?.item.outputShape === 'record' && node.arguments.length === 1
      && acceptedShapes(node.arguments[0].parameter).includes('record')
      && (!node.arguments[0].parameter.intermediate_kind || node.arguments[0].parameter.intermediate_kind === displayedGuidedTree.item.intermediateKind)) {
      node.arguments[0] = { ...node.arguments[0], source: 'operator', value: displayedGuidedTree.item.id, nested: displayedGuidedTree }
    }
    setComposer({
      node,
      applyMode: item.outputShape !== 'record' && editorMode === 'advanced' && selectionEnd > selectionStart ? 'replace_selection' : 'replace_formula',
      selectionStart,
      selectionEnd,
      origin: 'catalog',
      seriesOutputId: hasNamedOutputs ? activeSeriesOutput?.id : undefined,
    })
  }

  const updateComposerArgument = (path: number[], nextArgument: ComposerArgument) => {
    setComposerError(null)
    setComposer((current) => current ? {
      ...current,
      node: updateComposerNodeArgument(current.node, path, nextArgument),
    } : current)
  }

  const applyComposer = async () => {
    if (!composer) return
    const requestId = ++composerRequestRef.current
    const contextKey = previewContextRef.current
    const invalid = firstInvalidComposerArgument(composer.node, composerVariables, composableIndicators, STUDIO_CONTEXT)
    if (invalid) {
      setComposerError(
        firstComposerSiblingIssue(composer.node, composerVariables)
        || `请为“${invalid.parameter.label || invalid.parameter.name}”选择兼容变量、有限常量或嵌套算子。`,
      )
      return
    }
    try {
      setComposerLoading(true)
      setComposerError(null)
      setError(null)
      const composeNode = async (node: ComposerNode): Promise<InferenceResponse> => {
        const argumentsForRequest = []
        for (const argument of node.arguments) {
          if (argument.source === 'omitted') continue
          if (argument.source === 'operator' && argument.nested) {
            const nested = await composeNode(argument.nested)
            argumentsForRequest.push({ parameter: argument.parameter.name, source: 'expression' as const, value: composedExecutableSource(nested) })
          } else if (argument.source === 'indicator') {
            const indicator = composableIndicators.find((item) => indicatorReferenceKey(item) === argument.value)
            if (!indicator) throw new Error('所选已有指标已不可用，请重新选择。')
            const nested = await composeCustomIndicator({
              indicator_id: indicator.id,
              indicator_revision: indicator.revision,
              arguments: [],
              context: STUDIO_CONTEXT,
              dsl_version: draft.dsl_version,
              operator_registry_version: draft.operator_registry_version,
              variable_registry_version: draft.variable_registry_version,
              data_contract_version: draft.data_contract_version,
              context_schema_version: draft.context_schema_version,
            })
            argumentsForRequest.push({ parameter: argument.parameter.name, source: 'expression' as const, value: composedExecutableSource(nested) })
          } else {
            argumentsForRequest.push({
              parameter: argument.parameter.name,
              source: argument.source as 'variable' | 'constant',
              value: argument.source === 'constant' ? Number(argument.value) : argument.value,
            })
          }
        }
        return composeCustomIndicator({
          operator_id: node.item.id,
          context: STUDIO_CONTEXT,
          dsl_version: draft.dsl_version,
          operator_registry_version: draft.operator_registry_version,
          variable_registry_version: draft.variable_registry_version,
          data_contract_version: draft.data_contract_version,
          context_schema_version: draft.context_schema_version,
          arguments: argumentsForRequest,
        })
      }
      const response = await composeNode(composer.node)
      if (requestId !== composerRequestRef.current || contextKey !== previewContextRef.current) return
      const targetExpression = hasNamedOutputs && composer.seriesOutputId
        ? seriesOutputs.find((item) => item.id === composer.seriesOutputId)?.expression ?? currentExpression
        : currentExpression
      const start = composer.applyMode === 'replace_formula' ? 0 : composer.selectionStart
      const end = composer.applyMode === 'replace_formula'
        ? targetExpression.length
        : composer.applyMode === 'replace_selection'
          ? composer.selectionEnd
          : composer.selectionStart
      const executableExpression = composedExecutableSource(response)
      const previousCanonical = hasNamedOutputs && composer.seriesOutputId
        ? validation?.output_inferences?.[composer.seriesOutputId]?.python_expression
        : validation?.python_expression
      const editableSource = composedEditableLatex(response)
      const expression = composer.applyMode === 'replace_formula' && executableExpression === previousCanonical
        ? targetExpression
        : `${targetExpression.slice(0, start)}${editableSource}${targetExpression.slice(end)}`
      if (expression !== targetExpression) patchCurrentExpression(expression, composer.seriesOutputId)
      setInference(response)
      setGuidedTree(composer.node)
      setComposer(null)
      setComposerError(null)
      setMessage(response.shape === 'record'
        ? s('primitiveAuthoring.stateHint', {}, '这是共享计算的中间结果。请连接字段提取算子，再生成一个独立指标结果。')
        : composer.origin === 'current_formula'
        ? `${isTimeSeries ? `输出通道“${activeSeriesOutput?.label ?? composer.seriesOutputId}”` : '当前指标'}的计算逻辑已更新，请重新校验后保存修改或另存为新指标。`
        : `${composer.node.item.label} 已安全展开并${composer.applyMode === 'replace_formula' ? '替换完整公式' : composer.applyMode === 'replace_selection' ? '替换选中内容' : '插入光标位置'}。`)
    } catch (composeError) {
      if (requestId === composerRequestRef.current && contextKey === previewContextRef.current) setComposerError(userFacingErrorMessage(composeError, '公式展开失败，请检查参数后重试。'))
    } finally {
      setComposerLoading(false)
    }
  }

  async function validateDefinition(sourceDraft: IndicatorDraft = draft, messagePrefix = ''): Promise<ValidationResponse | null> {
    const requestId = ++validationRequestRef.current
    const normalized = normalizeDraft({ ...sourceDraft, context_kind: STUDIO_CONTEXT })
    try {
      setValidating(true)
      setError(null)
      const response = await validateCustomIndicator(normalized)
      if (requestId !== validationRequestRef.current) return null
      setValidation(response)
      const seriesDefinition = normalized.result_kind === 'time_series'
      const namedOutputs = normalized.series_outputs
      const firstInvalid = (response.diagnostics as Array<{ output_id?: string }>).find(item => item.output_id)?.output_id
      const selectedOutputId = seriesDefinition
        ? firstInvalid || (namedOutputs?.some(item => item.id === activeSeriesOutputId) ? activeSeriesOutputId : namedOutputs?.[0]?.id ?? '')
        : ''
      if (seriesDefinition && selectedOutputId) setActiveSeriesOutputId(selectedOutputId)
      const rootId = seriesDefinition
        ? response.dag?.roots[selectedOutputId]
        : response.dag?.roots.result ?? Object.values(response.dag?.roots ?? {})[0]
      const rootNode = response.dag?.nodes.find((node) => String(node.id) === String(rootId))
      const inferredType = rootNode
        ? dagNodeValueType(rootNode)
        : response.diagnostics.find((diagnostic) => diagnostic.actual !== undefined)?.actual
      const inferredShape = rootNode ? dagNodeShape(rootNode) : shapeFromValueType(inferredType)
      const nextInference = seriesDefinition
        ? inferenceForSeriesOutput(response, selectedOutputId)
        : inferredShape !== 'unknown'
          ? {
              expression: normalized.expression,
              latex: response.latex || normalized.expression,
              editable_latex: response.editable_latex,
              python_expression: response.python_expression || undefined,
              display_latex: response.display_latex || undefined,
              math_notation_version: response.math_notation_version || undefined,
              inferred_type: valueTypeText(inferredType) || inferredShape,
              shape: inferredShape,
              semantic_warnings: [],
              dependencies: response.dependencies,
              dag: response.dag || undefined,
            }
          : null
      setInference(nextInference)
      const parsed = response.valid
        ? composerNodeFromDag(
            response.dag,
            operatorItems,
            seriesDefinition ? selectedOutputId : undefined,
          )
        : null
      setGuidedTree(parsed)
      const resultMessage = response.valid
        ? seriesDefinition
            ? `时序定义校验通过，共 ${response.output_channels?.length ?? Object.keys(response.dag?.roots ?? {}).length} 个输出通道${parsed ? '；当前通道已解析为可编辑的嵌套计算步骤' : ''}。`
          : `公式解析和校验通过${parsed ? '，已同步为可编辑的计算步骤' : ''}。`
        : '公式已完成解析，但存在需要修复的校验问题。'
      setMessage(messagePrefix ? `${messagePrefix} ${resultMessage}` : resultMessage)
      return response
    } catch (validateError) {
      if (requestId !== validationRequestRef.current) return null
      setValidation(null)
      setInference(null)
      setGuidedTree(null)
      setError(validateError)
      return null
    } finally {
      if (requestId === validationRequestRef.current) setValidating(false)
    }
  }

  const validate = (): Promise<ValidationResponse | null> => {
    if (loading || !meta) return Promise.resolve(null)
    if (parameterPending) { setMessage(s('indicatorParameters.pending')); return Promise.resolve(null) }
    if (canvasPendingRef.current) { setMessage('请先应用画布修改，再校验或预览指标。'); return Promise.resolve(null) }
    return validateDefinition(draft)
  }

  const openFormulaBuilder = async () => {
    if (formulaBuilderOpening) return
    if (draft.parameter_contract_version === '1.0' && draft.parameter_schema?.length) { changeEditorMode('canvas'); return }
    if (displayedInference?.shape === 'record') {
      setCatalogError(null)
      setCatalogOpen(true)
      return
    }
    setCatalogError(null)
    setComposerError(null)
    if (!currentExpression.trim()) {
      setCatalogOpen(true)
      return
    }

    try {
      setFormulaBuilderOpening(true)
      const checked = validation?.valid ? validation : await validate()
      const currentTree = displayedGuidedTree ?? (checked?.valid
        ? composerNodeFromDag(
            checked.dag,
            operatorItems,
            hasNamedOutputs ? activeSeriesOutput?.id : undefined,
          )
        : null)
      if (!checked?.valid || !currentTree) {
        setCatalogError('当前公式暂时无法转换为结构化计算步骤。可以选择变量、算子或已有指标替换当前公式，也可以在高级公式模式中修复源码。')
        setCatalogOpen(true)
        return
      }
      setGuidedTree(currentTree)
      setComposer({
        node: currentTree,
        applyMode: 'replace_formula',
        selectionStart: 0,
        selectionEnd: currentExpression.length,
        origin: 'current_formula',
        seriesOutputId: hasNamedOutputs ? activeSeriesOutput?.id : undefined,
      })
      setMessage(`已载入“${isTimeSeries ? activeSeriesOutput?.label ?? draft.name : draft.name || '当前指标'}”的计算逻辑，可以修改参数或嵌套步骤。`)
    } finally {
      setFormulaBuilderOpening(false)
    }
  }

  const save = async (saveMode: 'auto' | 'new' = 'auto') => {
    if (parameterPending) { setMessage(s('indicatorParameters.pending')); return }
    if (canvasPendingRef.current) { setMessage('画布尚未应用，不能保存旧公式。'); return }
    const normalized = normalizeDraft({ ...draft, context_kind: STUDIO_CONTEXT })
    const checked = validation?.valid ? validation : await validate()
    if (!checked?.valid) {
      setMobileTab('editor')
      setWorkspaceTab('editor')
      return
    }
    try {
      setSaving(true)
      setError(null)
      let saved: IndicatorDefinition
      const createNewDefinition = saveMode === 'new' || !selectedIndicator || selectedIndicator.read_only
      if (!createNewDefinition && selectedIndicator) {
        saved = await updateCustomIndicator(selectedIndicator.id, normalized, selectedIndicator.revision)
      } else {
        const copyName = selectedIndicator && normalized.name === selectedIndicator.name
          ? `${normalized.name} 副本`
          : normalized.name
        saved = await createCustomIndicator({ ...normalized, name: copyName })
      }
      let layoutWarning = ''
      try { await canvasEditorRef.current?.persistLayout(saved) }
      catch (failure) { layoutWarning = ` 指标已保存，但布局未保存：${userFacingErrorMessage(failure)}` }
      const catalog = await refreshCatalog()
      setSelectedId(saved.id)
      const savedDraft = normalizeDraft(asDraft(catalog.find((item) => item.id === saved.id) || saved))
      setDraft(savedDraft)
      setBaseline(JSON.stringify(savedDraft))
      // The server may return equivalent reversible LaTeX or a new copy name.
      // Compile tokens bind exact draft source/metadata, not only the math.
      validationRequestRef.current += 1
      setValidation(null)
      setInference(null)
      setGuidedTree(null)
      setResults([])
      setSeriesResults([])
      setMessage(
        (createNewDefinition && selectedIndicator
          ? `已另存为新指标“${saved.name}”。原指标保持不变。`
          : `已保存为版本 ${saved.revision}。`) + layoutWarning,
      )
    } catch (saveError) {
      setError(saveError)
      if (saveError instanceof CustomIndicatorApiError && saveError.status === 409) {
        setMessage('保存冲突：该指标已被其他人更新。请重新载入最新版本后再合并修改。')
      }
    } finally {
      setSaving(false)
    }
  }

  const removeSelected = async () => {
    if (!selectedIndicator || selectedIndicator.read_only) return
    if (!window.confirm(`确定删除“${selectedIndicator.name}”吗？被评价方案引用的指标不能删除。`)) return
    try {
      setError(null)
      await deleteCustomIndicator(selectedIndicator.id, selectedIndicator.revision)
      await refreshCatalog()
      createNew()
      setMessage('指标已删除。')
    } catch (deleteError) {
      setError(deleteError)
    }
  }

  const lookupProducts = async () => {
    try {
      setSearching(true)
      setError(null)
      const response = await searchInstruments({ kind: searchKind, query: searchText, pageSize: 30 })
      setSearchResults(response.items)
    } catch (lookupError) {
      setError(lookupError)
    } finally {
      setSearching(false)
    }
  }

  const addTarget = (item: InstrumentSearchItem) => {
    const target = targetFromItem(item)
    if (!target) return
    if (targets.some((value) => value.kind === target.kind && value.product_id === target.product_id)) return
    if (targets.length >= MAX_PREVIEW_TARGETS) {
      setMessage(`校验与预览最多选择 ${MAX_PREVIEW_TARGETS} 个产品，请先移除一个产品。`)
      return
    }
    setTargets((current) => [...current, target])
    setResults([])
    setSeriesResults([])
    setMessage(`已添加预览产品（${targets.length + 1}/${MAX_PREVIEW_TARGETS}），请运行计算。`)
  }

  const preview = async () => {
    if (parameterPending) { setMessage(s('indicatorParameters.pending')); return }
    if (canvasPendingRef.current) { setMessage('画布尚未应用，不能预览旧公式。'); return }
    const checked = validation?.valid ? validation : await validate()
    if (!checked?.valid) {
      setMobileTab('editor')
      setWorkspaceTab('editor')
      return
    }
    if (!targets.length) {
      setMessage('请至少选择一个真实 ETF 或公募基金。')
      setMobileTab('preview')
      setWorkspaceTab('preview')
      return
    }
    const previewSequence = ++previewRequestRef.current
    const previewContext = previewContextRef.current
    const isCurrentPreview = () => previewSequence === previewRequestRef.current && previewContext === previewContextRef.current && !canvasPendingRef.current
    try {
      setPreviewing(true)
      setSeriesResults([])
      setError(null)
      const calculationPeriod = activePeriod
      if (!calculationPeriod) {
        setMessage('请选择本次预览的计算周期。')
        return
      }
      const normalized = normalizeDraft({ ...draft, context_kind: STUDIO_CONTEXT })
      if (normalized.result_kind === 'time_series') {
        const responses = await Promise.all(targets.map(({ name: _name, ...target }) => (
          evaluateTimeSeriesIndicators({
            indicator_instances: [{
              inline_definition: normalized,
              compile_token: checked.compile_token ?? undefined,
              ...(normalized.parameter_contract_version === '1.0' ? { parameters: runtimeParameters } : {}),
            }],
            target,
            period: calculationPeriod,
            as_of: asOf || undefined,
            max_points: 5000,
          })
        )))
        if (!isCurrentPreview()) return
        const timeSeriesResults = responses.flatMap((response) => response.results)
        setResults([])
        setSeriesResults(timeSeriesResults)
        const successful = timeSeriesResults.filter((result) => result.status === 'ok').length
        const attention = timeSeriesResults.length - successful
        setMessage(`时序预览完成：${successful} 个成功，${attention} 个需关注。`)
      } else {
        const response = await evaluateCustomIndicators({
          inline_definition: normalized,
          compile_token: checked.compile_token ?? undefined,
          targets: targets.map(({ name: _name, ...target }) => target),
          period: calculationPeriod,
          as_of: asOf || undefined,
          include_series: false,
        })
        if (!isCurrentPreview()) return
        setSeriesResults([])
        setResults(response.results)
        setMessage(`预览完成：${response.summary.ok} 个成功，${response.summary.warning + response.summary.error} 个需关注。`)
      }
      setMobileTab('preview')
      setWorkspaceTab('preview')
    } catch (previewError) {
      if (isCurrentPreview()) setError(previewError)
    } finally {
      if (previewSequence === previewRequestRef.current) setPreviewing(false)
    }
  }

  const downloadExcel = async () => {
    if (parameterPending) { setMessage(s('indicatorParameters.pending')); return }
    if (canvasPendingRef.current) { setMessage('请先应用画布修改，再导出计算逻辑。'); return }
    const checked = validation?.valid ? validation : await validate()
    if (!checked?.valid) {
      setMobileTab('editor')
      setWorkspaceTab('editor')
      return
    }
    if (!targets.length || !activePeriod) {
      setMessage('请先选择产品和计算周期。')
      setMobileTab('preview')
      setWorkspaceTab('preview')
      return
    }
    try {
      setExcelExporting(true)
      setError(null)
      const exportContext = previewContextRef.current
      const downloaded = await exportCustomIndicatorExcel({
        inline_definition: normalizeDraft({ ...draft, context_kind: STUDIO_CONTEXT }),
        compile_token: checked.compile_token ?? undefined,
        ...(isTimeSeries && draft.parameter_contract_version === '1.0' ? { parameters: runtimeParameters } : {}),
        targets: targets.map(({ name: _name, ...target }) => target),
        period: activePeriod,
        as_of: asOf || undefined,
      })
      if (canvasPendingRef.current || exportContext !== previewContextRef.current) return
      const objectUrl = URL.createObjectURL(downloaded.blob)
      const anchor = document.createElement('a')
      anchor.href = objectUrl
      anchor.download = downloaded.filename
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      URL.revokeObjectURL(objectUrl)
      setMessage(`Excel 已生成：包含 ${targets.length} 个产品的真实原始数据、逐步 Excel 公式和${isTimeSeries ? '各时序通道' : '指标'}结果。`)
    } catch (exportError) {
      setError(exportError)
    } finally {
      setExcelExporting(false)
    }
  }

  const statusText = loading ? '正在加载指标中心…' : message

  const activateWorkspaceTab = (tab: WorkspaceTab) => {
    setWorkspaceTab(tab)
    if (tab === 'preview' && hasDefinitionFormula && !validation && !validating) {
      void validateDefinition(draft)
    }
  }

  const activateMobileTab = (tab: MobileTab) => {
    setMobileTab(tab)
    if (tab === 'editor' || tab === 'preview') activateWorkspaceTab(tab)
  }

  const handleMobileTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, tab: MobileTab) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      activateMobileTab(tab)
      return
    }

    const currentIndex = MOBILE_TABS.findIndex(([id]) => id === tab)
    let nextIndex = currentIndex
    if (event.key === 'ArrowRight') nextIndex = (currentIndex + 1) % MOBILE_TABS.length
    else if (event.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + MOBILE_TABS.length) % MOBILE_TABS.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = MOBILE_TABS.length - 1
    else return

    event.preventDefault()
    const nextTab = MOBILE_TABS[nextIndex][0]
    activateMobileTab(nextTab)
    requestAnimationFrame(() => document.getElementById(`indicator-tab-${nextTab}`)?.focus())
  }

  const handleWorkspaceTabKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>, tab: WorkspaceTab) => {
    const tabs: WorkspaceTab[] = ['editor', 'preview']
    const currentIndex = tabs.indexOf(tab)
    let nextIndex = currentIndex
    if (event.key === 'ArrowRight') nextIndex = (currentIndex + 1) % tabs.length
    else if (event.key === 'ArrowLeft') nextIndex = (currentIndex - 1 + tabs.length) % tabs.length
    else if (event.key === 'Home') nextIndex = 0
    else if (event.key === 'End') nextIndex = tabs.length - 1
    else return

    event.preventDefault()
    const nextTab = tabs[nextIndex]
    activateWorkspaceTab(nextTab)
    requestAnimationFrame(() => document.getElementById(`workspace-tab-${nextTab}`)?.focus())
  }

  return (
    <div className={`mx-auto px-4 py-6 sm:px-6 lg:px-8 ${editorMode === 'canvas' ? 'max-w-[1920px]' : 'max-w-[1440px]'}`}>
      <div className="mb-6 flex flex-col gap-4 rounded-2xl bg-gradient-to-r from-slate-950 via-slate-900 to-violet-950 px-5 py-5 text-white shadow-lg sm:px-7 sm:py-6 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-sm font-semibold text-violet-200">工作区共享 · 统一研究指标层</p>
          <h1 className="mt-1 text-2xl font-bold tracking-tight sm:text-3xl">{b('indicator.studio.title')}</h1>
          <p className="mt-2 max-w-2xl text-sm text-slate-300">通过画布、构建向导或 LaTeX 定义计算；在校验与预览中查看 ETF 与公募基金的真实结果。</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" onClick={createNew} className="rounded-lg border border-white/25 px-4 py-2 text-sm font-semibold transition hover:bg-white/10 focus:outline-none focus:ring-2 focus:ring-white">{s('indicator.new')}</button>
          {selectedIndicator && !selectedIndicator.read_only && <button type="button" onClick={() => void save('new')} disabled={saving || loading || !hasDefinitionFormula} className="rounded-lg border border-violet-300/70 px-4 py-2 text-sm font-semibold text-violet-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-white">另存为新指标</button>}
          <button type="button" onClick={() => void save()} disabled={saving || loading || !hasDefinitionFormula} className="rounded-lg bg-violet-400 px-4 py-2 text-sm font-semibold text-slate-950 transition hover:bg-violet-300 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-white">{saving ? '保存中…' : selectedIndicator?.read_only ? '复制为新指标' : selectedIndicator ? '保存修改' : '保存新指标'}</button>
        </div>
      </div>

      <div className="mb-4 grid min-w-0 grid-cols-3 gap-1 rounded-xl border border-slate-200 bg-white p-1 md:hidden" role="tablist" aria-label="指标中心区域">
        {MOBILE_TABS.map(([id, label]) => (
          <button key={id} id={`indicator-tab-${id}`} type="button" role="tab" aria-controls={`indicator-panel-${id}`} aria-selected={mobileTab === id} tabIndex={mobileTab === id ? 0 : -1} onClick={() => activateMobileTab(id)} onKeyDown={(event) => handleMobileTabKeyDown(event, id)} className={`min-w-0 rounded-lg px-2 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${mobileTab === id ? 'bg-violet-600 text-white' : 'text-slate-600'}`}>{label}</button>
        ))}
      </div>

      <div aria-live="polite" className="mb-4 min-h-6 text-sm text-slate-600">{statusText}</div>
      <ApiMessage error={error} />

      <div className="mb-3 hidden md:block"><button type="button" onClick={() => setLibraryCollapsed(value => !value)} aria-expanded={!libraryCollapsed} aria-controls="indicator-panel-library" className="min-h-10 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-600">{libraryCollapsed ? '展开指标库' : '收起指标库'}</button></div>
      <div className={`grid min-w-0 grid-cols-[minmax(0,1fr)] gap-5 md:items-start ${libraryCollapsed ? 'md:grid-cols-1' : 'md:grid-cols-[minmax(240px,300px)_minmax(0,1fr)]'}`}>
        <aside id="indicator-panel-library" role="tabpanel" aria-labelledby="indicator-tab-library" className={`${mobileTab === 'library' ? 'block' : 'hidden'} min-w-0 rounded-2xl border border-slate-200 bg-white shadow-sm md:sticky md:top-5 md:self-start ${libraryCollapsed ? 'md:hidden' : 'md:block'}`}>
          <div className="border-b border-slate-100 px-4 py-4">
            <div className="flex items-center justify-between"><h2 className="font-semibold text-slate-900">{s('indicator.library')}</h2><span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">{visibleIndicators.length}</span></div>
            <p className="mt-1 text-xs text-slate-500">这里统一管理单产品指标；内置指标可直接使用，复制后可在工作区维护。</p>
            <div className="mt-3 grid gap-2">
              <label className="text-xs font-semibold text-slate-600">搜索指标<input type="search" aria-label="搜索指标" value={indicatorQuery} onChange={(event) => setIndicatorQuery(event.target.value)} placeholder="名称、说明或公式" className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
              <div className="grid grid-cols-2 gap-2">
                <label className="text-xs font-semibold text-slate-600">来源<select aria-label="指标来源" value={indicatorSourceFilter} onChange={(event) => setIndicatorSourceFilter(event.target.value as 'all' | 'built_in' | 'custom')} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-2 text-sm"><option value="all">全部来源</option><option value="built_in">内置指标</option><option value="custom">工作区指标</option></select></label>
                <label className="text-xs font-semibold text-slate-600">结果类型<select aria-label="指标结果类型" value={indicatorResultKindFilter} onChange={(event) => setIndicatorResultKindFilter(event.target.value as 'all' | IndicatorResultKind)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-2 text-sm"><option value="all">全部结果</option><option value="scalar">标量指标</option><option value="time_series">时序指标</option></select></label>
                <label className="col-span-2 text-xs font-semibold text-slate-600">分类<select aria-label="指标分类" value={indicatorCategoryFilter} onChange={(event) => setIndicatorCategoryFilter(event.target.value)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-2 text-sm"><option value="all">全部分类</option>{indicatorCategories.map((category) => <option key={category.id} value={category.id}>{category.label}</option>)}</select></label>
              </div>
            </div>
          </div>
          <div className="max-h-[65vh] space-y-5 overflow-y-auto p-3">
            {loading ? <p className="p-3 text-sm text-slate-500">正在读取指标库…</p> : visibleIndicators.length === 0 ? <div className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">{contextIndicators.length > 0 ? '当前筛选条件下没有指标。' : '当前还没有单产品指标，可以新建一个。'}</div> : <>
              <IndicatorGroup title="内置指标" items={builtInIndicators} selectedId={selectedId} onSelect={selectDraft} />
              <IndicatorGroup title="我的工作区指标" items={customIndicators} selectedId={selectedId} onSelect={selectDraft} empty="尚未保存自定义指标" />
            </>}
          </div>
        </aside>

        <div className={`${mobileTab === 'library' ? 'hidden' : 'block'} min-w-0 md:block`}>
          <div className="mb-5 hidden rounded-2xl border border-slate-200 bg-white p-2 shadow-sm md:block" role="tablist" aria-label="指标工作台">
            <div className="grid grid-cols-2 gap-2">
              <button id="workspace-tab-editor" type="button" role="tab" aria-controls="indicator-panel-editor" aria-selected={workspaceTab === 'editor'} tabIndex={workspaceTab === 'editor' ? 0 : -1} onClick={() => activateWorkspaceTab('editor')} onKeyDown={(event) => handleWorkspaceTabKeyDown(event, 'editor')} className={`rounded-xl px-4 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-violet-300 ${workspaceTab === 'editor' ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-50'}`}><span className="block text-sm font-semibold">{s('indicator.editor')}</span><span className={`mt-0.5 block text-xs ${workspaceTab === 'editor' ? 'text-violet-100' : 'text-slate-400'}`}>编辑指标信息、构建公式并完成校验</span></button>
              <button id="workspace-tab-preview" type="button" role="tab" aria-controls="indicator-panel-preview" aria-selected={workspaceTab === 'preview'} tabIndex={workspaceTab === 'preview' ? 0 : -1} onClick={() => activateWorkspaceTab('preview')} onKeyDown={(event) => handleWorkspaceTabKeyDown(event, 'preview')} className={`rounded-xl px-4 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-violet-300 ${workspaceTab === 'preview' ? 'bg-violet-600 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-50'}`}><span className="flex items-center justify-between gap-2 text-sm font-semibold">{s('indicator.preview')}{targets.length > 0 && <span aria-hidden="true" className={`rounded-full px-2 py-0.5 text-[11px] ${workspaceTab === 'preview' ? 'bg-white/20 text-white' : 'bg-violet-50 text-violet-700'}`}>{targets.length} 个产品</span>}</span><span className={`mt-0.5 block text-xs ${workspaceTab === 'preview' ? 'text-violet-100' : 'text-slate-400'}`}>校验定义、选择运行条件，查看真实计算结果</span></button>
            </div>
          </div>

          <section id="indicator-panel-editor" role="tabpanel" aria-labelledby="indicator-tab-editor workspace-tab-editor indicator-editor-title" className={`${mobileTab === 'editor' ? 'flex' : 'hidden'} min-w-0 flex-col gap-5 ${workspaceTab === 'editor' ? 'md:flex' : 'md:hidden'}`}>
          <details className="min-w-0" open={editorMode !== 'canvas' || basicInfoOpen} onToggle={event => { if (editorMode === 'canvas') setBasicInfoOpen(event.currentTarget.open) }}>
          <summary className={editorMode === 'canvas' ? 'cursor-pointer rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-700' : 'hidden'}><strong>{draft.name || '未命名指标'}</strong><span className="mx-3 text-slate-400">{isTimeSeries ? '时序指标' : '标量指标'} · {selectedIndicator ? `v${selectedIndicator.revision}` : '未保存'}</span><span className="font-semibold text-violet-700">基本信息与结果设置</span></summary>
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2"><div><h2 id="indicator-editor-title" className="font-semibold text-slate-900">指标定义</h2><p className="mt-1 text-xs text-slate-500">{selectedIndicator?.read_only ? '内置指标：保存时将创建工作区副本。' : selectedIndicator ? `工作区指标 · 当前版本 ${selectedIndicator.revision}` : '未保存草稿'}</p></div>{selectedIndicator && !selectedIndicator.read_only && <button type="button" onClick={() => void removeSelected()} className="rounded-lg px-3 py-2 text-sm font-medium text-rose-600 hover:bg-rose-50 focus:outline-none focus:ring-2 focus:ring-rose-300">删除</button>}</div>
            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <label className="text-sm font-medium text-slate-700">名称<input value={draft.name} onChange={(event) => patchDraft({ name: event.target.value })} maxLength={80} className="mt-1 block w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
              <label className="text-sm font-medium text-slate-700">结果类型<select aria-label="结果类型" value={isTimeSeries ? 'time_series' : 'scalar'} onChange={(event) => changeResultKind(event.target.value as IndicatorResultKind)} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="scalar">标量指标</option><option value="time_series">时序指标</option></select></label>
              <label className="text-sm font-medium text-slate-700">指标类型<select aria-label="指标类型" disabled={isTimeSeries} value={draft.indicator_type ?? 'other'} onChange={(event) => patchDraft({ indicator_type: event.target.value as IndicatorDraft['indicator_type'] })} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100 disabled:bg-slate-100">{indicatorCategories.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label>
              <label className="sm:col-span-3 text-sm font-medium text-slate-700">说明<textarea value={draft.description} onChange={(event) => patchDraft({ description: event.target.value })} rows={2} maxLength={500} className="mt-1 block w-full resize-y rounded-lg border border-slate-200 px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
            </div>
            <div className="mt-4 rounded-xl border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">
              {isTimeSeries
                ? '时序指标返回一个或多个具名通道，可在不同研究页面展示并导出 Excel。算法参数锁定在指标版本中，不直接参与标量评价排名。'
                : '标量指标用于跨产品展示、评价和排名；指标定义默认支持全部计算周期，周期只在预览或评价运行时选择。'}
            </div>
            {!isTimeSeries && selectedIndicator && <p role="status" className="mt-3 text-xs leading-5 text-slate-600">{selectedIndicator.rolling_series_compatibility?.supported ? s('rollingScope.supported') : selectedIndicator.rolling_series_compatibility?.message || s('rollingScope.unknown')}</p>}
            {!isTimeSeries && selectedIndicator?.rolling_series_compatibility?.supported && <div className="mt-3 flex justify-end"><button type="button" onClick={() => { setRollingSourceIndicatorId(selectedIndicator.id); setRollingWindowObservations(5); setRollingDerivationError(null); changeResultKind('time_series') }} className="min-h-10 rounded-lg border border-violet-200 bg-white px-4 py-2 text-sm font-semibold text-violet-700 hover:bg-violet-50 focus:outline-none focus:ring-2 focus:ring-violet-300">生成滚动时序指标</button></div>}
            {isTimeSeries && <section aria-label="标量指标滚动派生" className="mt-4 rounded-xl border border-violet-200 bg-violet-50/50 p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-sm font-semibold text-violet-950">{s('rollingScope.title', {}, '滚动计算')}</h3>
                  <p className="mt-1 text-xs leading-5 text-violet-800">{s('rollingScope.intro', {}, '选择一个区间指标，设置窗口，系统每天对最近一段数据重新计算。')}</p>
                </div>
                {draft.rolling_source && <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${draft.rolling_source.detached ? 'bg-amber-100 text-amber-800' : 'bg-emerald-100 text-emerald-800'}`}>{draft.rolling_source.detached ? '已脱离来源' : '来源已锁定'}</span>}
              </div>
              <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1fr)_10rem_auto] xl:items-end">
                <SearchableCombobox label="滚动来源标量指标" value={effectiveRollingSourceIndicatorId} options={rollingScalarIndicators.map((item) => ({ value: item.id, label: `${item.name} · v${item.revision}`, description: item.description, keywords: `${item.name} ${item.description}` }))} placeholder="搜索区间指标名称" onChange={(value) => { setRollingSourceIndicatorId(value); setRollingDerivationError(null) }} />
                <label className="text-xs font-semibold text-slate-700">{s('rollingScope.window', {}, '窗口观察数')}<input aria-label="滚动观察数" type="number" min="1" max="5000" step="1" aria-invalid={rollingWindowObservations !== '' && !rollingWindowValid} value={rollingWindowObservations} onChange={(event) => { setRollingWindowObservations(event.target.value === '' ? '' : Number(event.target.value)); setRollingDerivationError(null) }} className="mt-1 block min-h-11 w-full rounded-lg border border-violet-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>
                <button type="button" onClick={() => void deriveFromScalarIndicator()} disabled={rollingDeriving || !effectiveRollingSourceIndicatorId || !rollingWindowValid} className="min-h-11 rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-violet-300">{rollingDeriving ? '生成中…' : '生成滚动公式'}</button>
              </div>
              {rollingSource && rollingWindowValid && <p className="mt-3 text-xs leading-5 text-violet-900">{s('rollingScope.summary', { name: rollingSource.name, window: rollingWindowObservations, unit: rollingSource.rolling_series_compatibility?.window_unit_label || '观察点' }, '对最近 {{window}} 个{{unit}}逐日计算“{{name}}”。')}</p>}
              {rollingWindowObservations !== '' && !rollingWindowValid && <p role="alert" className="mt-2 text-xs text-rose-700">{s('rollingScope.invalidWindow', {}, '请输入 1 至 5000 的整数窗口。')}</p>}
              <details className="mt-3 text-xs leading-5 text-slate-600"><summary className="cursor-pointer font-semibold">{s('rollingScope.rules', {}, '窗口规则与来源版本')}</summary><p className="mt-2">{s('rollingScope.ruleText', {}, '每个窗口独立执行完整计算图，峰值等状态从窗口起点重新建立。窗口不足或包含缺失数据时留空，不补零，不读取未来数据。')}</p><p>{s('rollingScope.parameters', {}, '生成后可在“计算参数”中开放窗口调整；在其他页面修改只影响本次计算。')}</p>
              {draft.rolling_source && <p className="mt-2">当前来源：{draft.rolling_source.indicator_name} v{draft.rolling_source.indicator_revision}。{draft.rolling_source.detached ? '当前公式已被手工修改，不再保证与来源标量公式一致。' : '来源定义哈希、版本和生成公式已锁定。'}</p>}</details>
              {rollingDerivationError && <p role="alert" className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-700">{rollingDerivationError}</p>}
            </section>}
            {isTimeSeries && <TimeSeriesDefinitionFields draft={draft} variables={variables} validation={validation} onPatch={patchDraft} />}

          </div>

          </details>

          {isTimeSeries ? <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="indicator-output-settings" aria-label={s('outputAuthoring.seriesResults', {}, '时序结果')}>
            <h2 className="font-semibold text-slate-900">{s('outputAuthoring.title', {}, '输出结果')}</h2>
            <fieldset disabled={canvasPending || saving || validating || parameterPending}>
              <TimeSeriesFormulaEditor draft={draft} activeOutputId={activeSeriesOutput?.id ?? ''} measureOptions={seriesOutputMeasures} inference={activeSeriesOutput ? validation?.output_inferences?.[activeSeriesOutput.id] ?? null : null} onSelectOutput={selectSeriesOutput} onPatchOutput={patchSeriesOutput} onAddOutput={addSeriesOutput} onRemoveOutput={removeSeriesOutput} />
            </fieldset>
            {canvasPending && <p className="mt-2 text-xs text-amber-800">{s('outputAuthoring.locked')}</p>}
          </section> : <ScalarOutputEditor draft={draft} validation={validation} disabled={canvasPending || saving || validating || parameterPending} onPatch={patchDraft} />}

          <details className="order-last rounded-2xl border border-slate-200 bg-white shadow-sm">
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 p-5 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-violet-300">
              <div><h2 className="font-semibold text-slate-900">快照加速</h2><p className="mt-1 text-xs text-slate-500">选择需要在数据刷新后预计算的区间；时序指标还需明确输出通道，并取该通道末个有限值。</p></div>
              <span className="shrink-0 rounded-full bg-violet-50 px-3 py-1 text-xs font-semibold text-violet-700">{snapshotPeriods.length} 个区间</span>
            </summary>
            <div className="border-t border-slate-100 p-5 pt-4">
              {!selectedIndicator ? <p className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">请先从指标库选择一个已保存指标。未保存草稿不能作为可复现快照。</p> : <>
                <div className="rounded-xl border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">
                  快照会锁定“{selectedIndicator.name}” v{selectedIndicator.revision}。修改配置不会立即计算；下一次数据刷新或手动重建分析快照后生效。
                </div>
                {isTimeSeries && <label className="mt-4 block text-sm font-semibold text-slate-700">快照输出通道<select aria-label="快照输出通道" value={snapshotSeriesChannel} onChange={(event) => setSnapshotSeriesChannel(event.target.value)} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="">请选择输出通道</option>{(selectedIndicator.series_outputs ?? []).map((output) => <option key={output.id} value={output.id}>{output.label} · 末个有限值</option>)}</select><span className="mt-1 block text-xs font-normal text-slate-500">预计算会锁定指标版本、输出通道、区间和“末个有限值”归约规则。</span></label>}
                <fieldset className="mt-4"><legend className="text-sm font-semibold text-slate-700">预计算区间</legend><div className="mt-2 grid max-h-48 gap-2 overflow-y-auto rounded-xl border border-slate-200 p-3 sm:grid-cols-2 lg:grid-cols-3">{periods.map((item) => { const checked = snapshotPeriods.includes(item.value); return <label key={item.value} className={`flex min-h-11 cursor-pointer items-start gap-2 rounded-lg border px-3 py-2 text-sm ${checked ? 'border-violet-300 bg-violet-50 text-violet-900' : 'border-slate-100 bg-white text-slate-600'}`}><input type="checkbox" checked={checked} onChange={(event) => setSnapshotPeriods((current) => event.target.checked ? [...new Set([...current, item.value])] : current.filter((value) => value !== item.value))} className="mt-0.5" /><span><span className="block font-medium">{item.label}（{item.value}）</span><span className="mt-0.5 block text-xs text-slate-400">{item.description}</span></span></label> })}</div></fieldset>
                <div className="mt-4 flex flex-wrap items-center justify-between gap-3"><p className="text-xs text-slate-500">工作区当前共配置 {snapshotConfig?.items.length ?? 0} / {snapshotConfig?.max_items ?? 30} 个“指标 + 区间”快照。{snapshotConfig?.snapshot_status === 'ready' ? '当前快照已使用此配置和最新数据。' : snapshotConfig?.snapshot_status === 'stale' ? '配置或数据已变化，需要重新生成快照。' : '尚未生成指标快照。'}</p><button type="button" onClick={() => void saveSnapshotPeriods()} disabled={!snapshotConfig || snapshotSaving} className="rounded-lg bg-violet-600 px-4 py-2 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-violet-300">{snapshotSaving ? '保存中…' : '保存快照配置'}</button></div>
              </>}
              {snapshotError && <p role="alert" className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{snapshotError}</p>}
            </div>
          </details>

          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" data-testid="indicator-calculation-editor">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h2 className="font-semibold text-slate-900">{s('outputAuthoring.logic', {}, '计算逻辑')}</h2>
                <p className="mt-1 text-xs text-slate-500">画布适合搭建与修改计算流程；构建向导和高级公式仍可使用，三种方式共用同一份指标定义。</p>
              </div>
            </div>
            <div className="mt-4 inline-flex flex-wrap rounded-lg border border-slate-200 bg-slate-50 p-1" role="tablist" aria-label="公式编辑方式">
              <button id="formula-mode-canvas" type="button" role="tab" aria-controls="formula-canvas-panel" aria-selected={editorMode === 'canvas'} onClick={() => changeEditorMode('canvas')} className={`rounded-md px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${editorMode === 'canvas' ? 'bg-white text-violet-700 shadow-sm' : 'text-slate-600'}`}>{s('graph.title')}</button>
              <button id="formula-mode-guided" type="button" role="tab" aria-controls="formula-guided-panel" aria-selected={editorMode === 'guided'} onClick={() => changeEditorMode('guided')} className={`rounded-md px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${editorMode === 'guided' ? 'bg-white text-violet-700 shadow-sm' : 'text-slate-600'}`}>{s('indicator.guided')}</button>
              <button id="formula-mode-advanced" type="button" role="tab" aria-controls="formula-advanced-panel" aria-selected={editorMode === 'advanced'} onClick={() => changeEditorMode('advanced')} className={`rounded-md px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${editorMode === 'advanced' ? 'bg-white text-violet-700 shadow-sm' : 'text-slate-600'}`}>{s('indicator.formula')}</button>
            </div>
            <div id="formula-canvas-panel" role="tabpanel" aria-labelledby="formula-mode-canvas" hidden={editorMode !== 'canvas'}>
              {canvasVisited && <IndicatorGraphEditor key={`${canvasSession}:${selectedId ?? 'new'}:${draft.result_kind ?? 'scalar'}`} ref={canvasEditorRef} draft={draft} indicator={selectedIndicator} operators={meta?.operators ?? []} variables={variables} indicators={contextIndicators} active={editorMode === 'canvas' && workspaceTab === 'editor'} ready={Boolean(meta) && !loading} disabled={saving || validating} definitionDirty={definitionDirty} onPendingChange={canvasPendingChanged} onApply={patchDraft} />}
            </div>
            {editorMode !== 'canvas' && (editorMode === 'guided' ? <div id="formula-guided-panel" role="tabpanel" aria-labelledby="formula-mode-guided" className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4">
              {currentExpression.trim() ? <>
                <div className="flex items-center justify-between gap-3"><p className="text-xs font-semibold text-slate-600">{hasNamedOutputs ? `当前结果：${activeSeriesOutput?.label || '未命名结果'}` : '当前计算逻辑'}</p><span className="text-xs text-slate-400">已生成</span></div>
                <p className="mt-2 rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-600">公式已由结构化构建器生成。下方使用数学符号展示计算逻辑；如需直接查看或编辑 LaTeX 源码，可切换到高级公式模式。</p>
                {displayedGuidedTree ? <div className="mt-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2"><p className="text-xs font-semibold text-emerald-800">已识别公式的计算结构</p><p className="mt-0.5 text-xs text-emerald-700">最后一步：{displayedGuidedTree.item.label} · 共 {composerNodeCount(displayedGuidedTree)} 个计算步骤。点击下方入口可查看并修改完整嵌套逻辑。</p></div> : <p className="mt-2 text-xs text-slate-500">点击下方入口时，系统会先解析当前公式，再展示可编辑的变量、参数和嵌套计算步骤；无法解析时可在高级公式模式中修复源码。</p>}
              </> : <div className="rounded-lg border border-dashed border-violet-200 bg-white p-4 text-center">
                <p className="font-semibold text-slate-800">从变量、数学算子或已有指标开始</p>
                <p className="mt-1 text-xs leading-5 text-slate-500">目录会解释数据与类型契约；时序运行参数也可作为标量输入，算子配置会屏蔽不兼容组合。</p>
              </div>}
              <button type="button" onClick={() => void openFormulaBuilder()} disabled={formulaBuilderOpening || (isTimeSeries && !activeSeriesOutput)} aria-busy={formulaBuilderOpening} className="mt-3 w-full rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-wait disabled:bg-violet-400 focus:outline-none focus:ring-2 focus:ring-violet-300">{formulaBuilderOpening ? '正在载入当前计算逻辑…' : '浏览公式构建资源'}</button>
            </div> : <div id="formula-advanced-panel" role="tabpanel" aria-labelledby="formula-mode-advanced" className="mt-4">
              <label className="block text-sm font-semibold text-slate-700" htmlFor="indicator-expression">{hasNamedOutputs ? `“${activeSeriesOutput?.label || '当前结果'}”公式源码（LaTeX）` : '受限公式源码（LaTeX）'}</label>
              <p className="mt-1 text-xs text-slate-500">使用 LaTeX 编写变量、分式和根式；窗口、自由度等参数完整保留。下方预览使用简洁数学符号，系统在内部转换为 DSL 进行校验与计算。</p>
              <textarea id="indicator-expression" ref={expressionRef} value={currentExpression} onChange={(event) => patchCurrentExpression(event.target.value)} maxLength={isTimeSeries ? 4000 : 1000} spellCheck={false} rows={6} className="mt-2 block w-full rounded-xl border border-slate-300 bg-slate-950 p-4 font-mono text-sm leading-6 text-emerald-200 focus:border-violet-400 focus:outline-none focus:ring-2 focus:ring-violet-200" />
              <div className="mt-2 flex items-center justify-between gap-3"><button type="button" onClick={() => void openFormulaBuilder()} disabled={formulaBuilderOpening || (isTimeSeries && !activeSeriesOutput)} aria-busy={formulaBuilderOpening} className="rounded-lg border border-violet-200 px-3 py-2 text-sm font-semibold text-violet-700 hover:bg-violet-50 disabled:cursor-wait disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-violet-300">{formulaBuilderOpening ? '正在载入当前计算逻辑…' : '浏览公式构建资源'}</button><span className="text-xs text-slate-400">{currentExpression.length} / {isTimeSeries ? 4000 : 1000}</span></div>
            </div>)}
            {editorMode !== 'canvas' && <>
            {isTimeSeries && <IndicatorParameterEditor key={`${selectedId ?? 'new'}-${canvasSession}`} draft={draft} disabled={canvasPending || saving || validating} onPendingChange={setParameterPending} onPatch={patchDraft} />}
            <button type="button" onClick={() => void validate()} disabled={loading || !meta || validating || !hasDefinitionFormula} className="mt-3 w-full rounded-lg border border-violet-200 bg-violet-50 px-4 py-2.5 text-sm font-semibold text-violet-700 hover:bg-violet-100 disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-violet-300">{validating ? '解析与校验中…' : isTimeSeries ? '解析并校验全部通道' : '解析并校验公式'}</button>
            <div className="mt-4 rounded-xl border border-violet-100 bg-violet-50/50 p-4"><p className="text-xs font-semibold uppercase tracking-wide text-violet-700">数学排版预览{isTimeSeries && activeSeriesOutput ? ` · ${activeSeriesOutput.label}` : ''}</p><div data-testid="formula-preview" className="mt-3 overflow-x-auto text-slate-900" dangerouslySetInnerHTML={formulaMarkup} /></div>
            {displayedInference && <InferencePanel inference={displayedInference} resourceLabels={resourceLabels} />}
            {activeFormulaDag && <FormulaExplanationDisclosure dag={activeFormulaDag} variables={formulaVariables} operators={operatorItems} />}
            {validation && <ValidationPanel validation={validation} resourceLabels={resourceLabels} />}
            </>}
          </div>
        </section>

        <section id="indicator-panel-preview" role="tabpanel" aria-labelledby="indicator-tab-preview workspace-tab-preview indicator-preview-title" className={`${mobileTab === 'preview' ? 'block' : 'hidden'} min-w-0 space-y-5 ${workspaceTab === 'preview' ? 'md:block' : 'md:hidden'}`}>
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-5 xl:grid-cols-[minmax(320px,0.72fr)_minmax(0,1.28fr)] xl:items-start">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h2 id="indicator-preview-title" className="font-semibold text-slate-900">校验与预览</h2><p className="mt-1 text-xs text-slate-500">最多选择 {MAX_PREVIEW_TARGETS} 个真实产品并分别计算同一指标；不会合并为组合，也不会使用模拟数据。</p><div aria-label="当前预览指标" aria-live="polite" className="mt-3 flex flex-wrap items-center gap-2 rounded-xl border border-violet-100 bg-violet-50 px-3 py-2 text-sm"><span className="text-slate-500">当前指标</span><strong className="text-violet-800">{draft.name}</strong><span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-500">{selectedIndicator?.read_only ? '内置指标' : selectedIndicator ? `工作区 v${selectedIndicator.revision}` : '未保存草稿'}</span></div>
              <div className="mt-4 flex gap-2"><select aria-label="产品类型" value={searchKind} onChange={(event) => { setSearchKind(event.target.value as ProductKind | 'all'); setSearchResults([]) }} className="rounded-lg border border-slate-200 bg-white px-2 text-sm focus:border-violet-500 focus:outline-none"><option value="all">全部</option><option value="etf">ETF</option><option value="fund">公募基金</option></select><SearchDropdown label="搜索产品" value={searchText} items={searchResults} onChange={setSearchText} onSearch={lookupProducts} loading={searching} placeholder="名称或代码" className="flex-1" getItemKey={(item, index) => `${item.code ?? item.ts_code ?? index}-${item.instrument_type ?? ''}`} renderItem={(item) => { const target = targetFromItem(item); const selected = Boolean(target && targets.some((value) => value.kind === target.kind && value.product_id === target.product_id)); const atLimit = targets.length >= MAX_PREVIEW_TARGETS; return <div className="flex min-h-11 items-center justify-between gap-3 border-b border-slate-100 px-3 py-2 last:border-0"><span className="min-w-0 text-sm text-slate-700"><span className="font-medium">{item.name || target?.product_id}</span><span className="ml-2 text-xs text-slate-400">{target?.product_id}</span></span><button type="button" onClick={() => addTarget(item)} disabled={!target || selected || atLimit} title={!selected && atLimit ? `最多选择 ${MAX_PREVIEW_TARGETS} 个产品` : undefined} className="shrink-0 rounded-md px-2 py-1 text-xs font-semibold text-violet-700 hover:bg-violet-50 disabled:text-slate-300 focus:outline-none focus:ring-2 focus:ring-violet-300">{selected ? '已添加' : atLimit ? '已达上限' : '添加'}</button></div> }} /></div>
              <div className="mt-4"><p className="text-sm font-medium text-slate-700">已选产品 <span aria-live="polite" className="text-slate-400">{targets.length} / {MAX_PREVIEW_TARGETS}</span></p><div className="mt-2 flex flex-wrap gap-2">{targets.length ? targets.map((target) => <span key={`${target.kind}-${target.product_id}`} className="inline-flex items-center gap-1 rounded-full bg-slate-100 py-1 pl-3 pr-1 text-xs text-slate-700"><span>{target.name}</span><span className="text-slate-400">{target.product_id !== target.name ? target.product_id : ''}</span><button type="button" aria-label={`移除 ${target.name}`} onClick={() => { setTargets((current) => current.filter((item) => item.kind !== target.kind || item.product_id !== target.product_id)); setResults([]); setMessage('预览产品已移除，请重新计算。') }} className="rounded-full px-1.5 py-0.5 text-slate-400 hover:bg-white hover:text-rose-600">×</button></span>) : <p className="text-sm text-slate-400">尚未选择产品</p>}</div></div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2"><label className="text-sm font-medium text-slate-700">计算周期<select aria-label="计算周期" value={activePeriod} onChange={(event) => { setPeriod(event.target.value); setResults([]); setSeriesResults([]); setMessage('预览周期已更改，请重新计算。') }} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100">{periods.map((item) => <option key={item.value} value={item.value}>{item.label}（{item.value}）</option>)}</select></label><label className="text-sm font-medium text-slate-700">历史截止日（可选）<input aria-label="历史截止日" type="date" value={asOf} onChange={(event) => { setAsOf(event.target.value); setResults([]); setSeriesResults([]); setMessage('历史截止日已更改，请重新计算。') }} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label></div>
              {isTimeSeries && (draft.parameter_contract_version === '1.0' && draft.parameter_schema?.length
                ? <IndicatorParameterInputs schema={draft.parameter_schema} values={runtimeParameters} onApply={applyRuntimeParameters} disabled={parameterPending || canvasPending || saving} />
                : <p className="mt-3 rounded-lg border border-sky-100 bg-sky-50 px-3 py-2 text-xs text-sky-800">{s('indicatorParameters.fixedHint')}</p>)}
              <p className="mt-2 text-xs text-slate-500">选择产品并执行预览后，系统会根据实际数据判断是否可计算，并在结果中说明数据缺失、样本不足等原因。</p><button type="button" onClick={() => void preview()} disabled={previewing || excelExporting || periods.length === 0 || !targets.length || !hasDefinitionFormula} className="mt-3 w-full rounded-lg bg-violet-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300 focus:outline-none focus:ring-2 focus:ring-violet-300">{previewing ? '计算中…' : '预览指标'}</button><button type="button" onClick={() => void downloadExcel()} disabled={excelExporting || previewing || periods.length === 0 || !targets.length || !hasDefinitionFormula} className="mt-2 w-full rounded-lg border border-violet-200 bg-white px-4 py-2.5 text-sm font-semibold text-violet-700 hover:bg-violet-50 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-300 focus:outline-none focus:ring-2 focus:ring-violet-300">{excelExporting ? '正在生成 Excel…' : '下载 Excel 计算逻辑'}</button>
          </div>

          <div className="min-w-0 space-y-5">
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-label="指标定义校验">
              <h2 className="font-semibold text-slate-900">{validation?.valid === false ? '当前公式未通过校验' : '指标定义校验'}</h2>
              <p role="status" className="mt-2 text-sm text-slate-600">{canvasPending ? '画布尚未应用，请返回定义与公式完成修改。' : validating ? '正在校验公式、类型与计算计划…' : validation?.valid ? '指标定义有效；产品数据是否充足，以实际预览结果为准。' : '请校验当前定义，再选择产品运行。'}</p>
              {validation?.valid === false && <ul className="mt-2 space-y-1 text-sm text-rose-700">{validation.diagnostics.map((item, index) => <li key={`${item.code}-${index}`}>{diagnosticMessage(item.code, item.message)}</li>)}</ul>}
              <div className="mt-3 flex flex-wrap gap-2"><button type="button" onClick={() => void validate()} disabled={!hasDefinitionFormula || validating} className="min-h-10 rounded-lg border border-violet-200 px-3 py-2 text-sm font-semibold text-violet-700 disabled:opacity-40">校验指标定义</button><button type="button" onClick={() => { setMobileTab('editor'); activateWorkspaceTab('editor') }} className="min-h-10 rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-600">返回定义与公式修改</button></div>
            </section>
            {isTimeSeries
              ? <TimeSeriesResultsPanel results={seriesResults} />
              : <ResultsPanel results={results} draft={draft} contextDomain={STUDIO_CONTEXT} />}
          </div>
          </div>
          <p className="px-1 text-xs text-slate-500">指标定义和评价方案在当前工作区共享。数据缺失、样本不足或无效数值会明确标为不可计算。</p>
          <Link to="/product-research/products" className="inline-flex px-1 text-sm font-semibold text-violet-700 underline decoration-violet-300 underline-offset-4 hover:text-violet-900">返回产品研究</Link>
        </section>
        </div>
      </div>

      <CatalogDrawer open={catalogOpen} onClose={() => setCatalogOpen(false)}>
        <TypedCatalog tabsValue={catalogTab} onTabChange={setCatalogTab} variables={formulaVariables} operators={operatorItems} indicators={composableIndicators} contextDomain={STUDIO_CONTEXT} onInsertVariable={insertExpression} onInsertIndicator={insertIndicator} onExplainIndicator={(indicator) => composeCustomIndicator({ indicator_id: indicator.id, indicator_revision: indicator.revision, arguments: [], context: STUDIO_CONTEXT, dsl_version: draft.dsl_version, operator_registry_version: draft.operator_registry_version, variable_registry_version: draft.variable_registry_version, data_contract_version: draft.data_contract_version, context_schema_version: draft.context_schema_version })} onOpenComposer={openComposer} variableActionLabel={editorMode === 'guided' ? '设为当前公式' : '插入到光标'} indicatorActionLabel={editorMode === 'guided' ? '展开为当前公式' : '插入锁定版本公式'} insertingIndicatorId={insertingIndicatorId} actionError={catalogError} />
      </CatalogDrawer>
      <ComposerDrawer composer={composer} indicatorName={hasNamedOutputs ? activeSeriesOutput?.label ?? draft.name : draft.name} variables={composerVariables} operators={operatorItems} indicators={composableIndicators} contextDomain={STUDIO_CONTEXT} loading={composerLoading} canApply={composerReady} error={composerError} onClose={() => { composerRequestRef.current += 1; setComposer(null); setComposerError(null) }} onBrowseResources={() => { composerRequestRef.current += 1; setComposer(null); setComposerError(null); setCatalogError(null); setCatalogOpen(true) }} onChange={updateComposerArgument} onApplyModeChange={(applyMode) => setComposer((current) => current ? { ...current, applyMode } : current)} onApply={() => void applyComposer()} />
      <div className="sticky bottom-0 z-10 mt-5 flex gap-2 border-t border-slate-200 bg-white/95 p-3 backdrop-blur md:hidden"><button type="button" onClick={() => void save()} disabled={saving || !hasDefinitionFormula} className="flex-1 rounded-lg bg-violet-600 py-2 text-sm font-semibold text-white disabled:bg-slate-300">{selectedIndicator?.read_only ? '复制为新指标' : selectedIndicator ? '保存修改' : '保存新指标'}</button><button type="button" onClick={() => void preview()} disabled={previewing || !hasDefinitionFormula || !targets.length || !activePeriod} className="flex-1 rounded-lg bg-slate-900 py-2 text-sm font-semibold text-white disabled:bg-slate-300">预览</button></div>
    </div>
  )
}

type SearchableOption = {
  value: string
  label: string
  description?: string
  keywords?: string
  disabled?: boolean
}

function SearchableCombobox({ label, value, options, placeholder, onChange }: { label: string; value: string; options: SearchableOption[]; placeholder: string; onChange: (value: string) => void }) {
  const virtualThreshold = 80
  const virtualWindowSize = 24
  const virtualRowHeight = 60
  const id = useId()
  const containerRef = useRef<HTMLDivElement | null>(null)
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [highlightedIndex, setHighlightedIndex] = useState(0)
  const [virtualStart, setVirtualStart] = useState(0)
  const selected = options.find((option) => option.value === value)
  const normalizedQuery = query.trim().toLowerCase()
  const filtered = options.filter((option) => !normalizedQuery || `${option.label} ${option.description || ''} ${option.keywords || ''}`.toLowerCase().includes(normalizedQuery))
  const safeIndex = Math.max(0, Math.min(highlightedIndex, Math.max(0, filtered.length - 1)))
  const virtualized = filtered.length > virtualThreshold
  const windowStart = virtualized ? Math.max(0, Math.min(virtualStart, filtered.length - virtualWindowSize)) : 0
  const visibleOptions = virtualized
    ? filtered.slice(windowStart, windowStart + virtualWindowSize)
    : filtered

  useEffect(() => {
    if (!open) return
    const closeOutside = (event: MouseEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', closeOutside)
    return () => document.removeEventListener('mousedown', closeOutside)
  }, [open])

  useEffect(() => {
    if (!virtualized) {
      setVirtualStart(0)
      return
    }
    setVirtualStart((current) => {
      if (safeIndex < current) return safeIndex
      if (safeIndex >= current + virtualWindowSize) {
        return Math.max(0, safeIndex - virtualWindowSize + 1)
      }
      return current
    })
  }, [safeIndex, virtualized])

  const moveHighlight = (direction: 1 | -1) => {
    if (!filtered.length) return
    let next = safeIndex
    for (let count = 0; count < filtered.length; count += 1) {
      next = (next + direction + filtered.length) % filtered.length
      if (!filtered[next].disabled) break
    }
    setHighlightedIndex(next)
  }

  const choose = (option: SearchableOption | undefined) => {
    if (!option || option.disabled) return
    onChange(option.value)
    setOpen(false)
    setQuery('')
  }

  return <div ref={containerRef} className="relative">
    <label id={`${id}-label`} className="block text-xs font-semibold text-slate-600">{label}</label>
    <button type="button" role="combobox" aria-labelledby={`${id}-label`} aria-expanded={open} aria-controls={`${id}-listbox`} aria-activedescendant={open && filtered[safeIndex] ? `${id}-option-${safeIndex}` : undefined} onClick={() => { setOpen((current) => !current); setQuery(''); setVirtualStart(0); setHighlightedIndex(Math.max(0, options.findIndex((option) => option.value === value))) }} onKeyDown={(event) => {
      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        if (!open) setOpen(true)
        else moveHighlight(event.key === 'ArrowDown' ? 1 : -1)
      } else if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault()
        if (!open) setOpen(true)
        else choose(filtered[safeIndex])
      } else if (event.key === 'Escape') setOpen(false)
    }} className="mt-1 flex min-h-11 w-full items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2 text-left text-sm text-slate-700 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100">
      <span className="min-w-0 truncate">{selected?.label || placeholder}</span><span aria-hidden="true" className="text-slate-400">⌄</span>
    </button>
    {open && <div className="absolute z-20 mt-1 w-full rounded-xl border border-slate-200 bg-white p-2 shadow-xl">
      <input autoFocus type="search" aria-label={`搜索${label}`} value={query} onChange={(event) => { setQuery(event.target.value); setHighlightedIndex(0); setVirtualStart(0) }} onKeyDown={(event) => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); moveHighlight(event.key === 'ArrowDown' ? 1 : -1) }
        else if (event.key === 'Home') { event.preventDefault(); setHighlightedIndex(0) }
        else if (event.key === 'End') { event.preventDefault(); setHighlightedIndex(Math.max(0, filtered.length - 1)) }
        else if (event.key === 'Enter') { event.preventDefault(); choose(filtered[safeIndex]) }
        else if (event.key === 'Escape') { event.preventDefault(); setOpen(false) }
      }} placeholder={placeholder} className="block w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" />
      {virtualized && <p className="mt-2 px-2 text-[11px] text-slate-400" role="status">大目录按需渲染：当前 {windowStart + 1}–{Math.min(filtered.length, windowStart + visibleOptions.length)} / {filtered.length}</p>}
      <ul id={`${id}-listbox`} role="listbox" aria-labelledby={`${id}-label`} onScroll={(event) => { if (virtualized) setVirtualStart(Math.floor(event.currentTarget.scrollTop / virtualRowHeight)) }} className="mt-2 max-h-64 overflow-auto py-1">
        {virtualized && windowStart > 0 && <li role="presentation" aria-hidden="true" style={{ height: windowStart * virtualRowHeight }} />}
        {visibleOptions.map((option, relativeIndex) => { const index = windowStart + relativeIndex; return <li key={option.value} id={`${id}-option-${index}`} role="option" aria-setsize={filtered.length} aria-posinset={index + 1} aria-selected={option.value === value} aria-disabled={option.disabled || undefined} onMouseDown={(event) => { event.preventDefault(); choose(option) }} onMouseEnter={() => setHighlightedIndex(index)} style={virtualized ? { minHeight: virtualRowHeight } : undefined} className={`rounded-lg px-3 py-2 ${option.disabled ? 'cursor-not-allowed text-slate-300' : 'cursor-pointer'} ${index === safeIndex ? 'bg-violet-50 text-violet-900' : 'text-slate-700'}`}><span className="block text-sm font-medium">{option.label}</span>{option.description && <span className="mt-0.5 block text-xs text-slate-400">{option.description}</span>}</li> })}
        {virtualized && windowStart + visibleOptions.length < filtered.length && <li role="presentation" aria-hidden="true" style={{ height: (filtered.length - windowStart - visibleOptions.length) * virtualRowHeight }} />}
        {!filtered.length && <li className="px-3 py-4 text-center text-sm text-slate-400">没有匹配项</li>}
      </ul>
    </div>}
  </div>
}

function CatalogDrawer({ open, onClose, children }: { open: boolean; onClose: () => void; children: React.ReactNode }) {
  const dialogRef = useRef<HTMLElement | null>(null)
  const closeRef = useRef<HTMLButtonElement | null>(null)
  useEffect(() => {
    if (!open) return
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    requestAnimationFrame(() => closeRef.current?.focus())
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialogRef.current) return
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), [href], [tabindex]:not([tabindex="-1"])')]
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', trapFocus)
    return () => { document.removeEventListener('keydown', trapFocus); previous?.focus() }
  }, [open])
  if (!open) return null
  return <div className="fixed inset-0 z-[100] flex justify-end bg-slate-950/40" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="catalog-drawer-title" onKeyDown={(event) => { if (event.key === 'Escape') onClose() }} className="flex h-full w-full flex-col bg-white shadow-2xl sm:max-w-2xl">
      <div className="flex items-start justify-between border-b border-slate-200 px-4 py-4 sm:px-5"><div><p className="text-xs font-semibold text-violet-700">公式构建资源</p><h2 id="catalog-drawer-title" className="mt-1 text-lg font-bold text-slate-900">变量、算子与已有指标</h2><p className="mt-1 text-xs text-slate-500">按资源类型、分类、条目逐级定位；已有指标会按所选版本展开为独立公式。</p></div><button ref={closeRef} type="button" onClick={onClose} aria-label="关闭资源目录" className="rounded-lg px-3 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-violet-300">关闭</button></div>
      <div className="min-h-0 flex-1 overflow-auto p-4 sm:p-5">{children}</div>
    </section>
  </div>
}

function TypedCatalog({ tabsValue, onTabChange, variables, operators, indicators, contextDomain, onInsertVariable, onInsertIndicator, onExplainIndicator, onOpenComposer, variableActionLabel, indicatorActionLabel, insertingIndicatorId, actionError }: { tabsValue: CatalogTab; onTabChange: (tab: CatalogTab) => void; variables: IndicatorVariable[]; operators: ComposerItem[]; indicators: IndicatorDefinition[]; contextDomain: IndicatorContextDomain; onInsertVariable: (token: string) => void; onInsertIndicator: (indicator: IndicatorDefinition) => Promise<void> | void; onExplainIndicator: (indicator: IndicatorDefinition) => Promise<InferenceResponse>; onOpenComposer: (item: ComposerItem) => void; variableActionLabel: string; indicatorActionLabel: string; insertingIndicatorId: string | null; actionError: string | null }) {
  const [selectedCategoryIds, setSelectedCategoryIds] = useState<Record<CatalogTab, string>>({ variables: '', operators: '', indicators: '' })
  const [selectedItemIds, setSelectedItemIds] = useState<Record<CatalogTab, string>>({ variables: '', operators: '', indicators: '' })
  const [indicatorExplanations, setIndicatorExplanations] = useState<Record<string, InferenceResponse>>({})
  const [visibleExplanationId, setVisibleExplanationId] = useState('')
  const [explainingIndicatorId, setExplainingIndicatorId] = useState('')
  const [indicatorExplanationError, setIndicatorExplanationError] = useState<{ id: string; message: string } | null>(null)
  const catalogLabels: Record<CatalogTab, string> = { variables: '变量', operators: '计算算子', indicators: '已有指标' }
  const rawEntries = tabsValue === 'variables'
    ? variables.map((variable) => {
      const category = variableCategory(variable)
      const contextSupported = supportsVariableDomain(variable, contextDomain)
        && variable.availability !== 'unavailable'
        && variable.availability !== 'not_applicable'
      const description = variable.description || variable.semantic || `${variable.label}在计算窗口内的数值。`
      return { id: variable.name, categoryId: category.id, categoryLabel: category.label, contextSupported, supported: contextSupported, label: variable.label, description: [shapeLabel(inferShape(variable), variable.value_type), description].join(' · '), keywords: `${variable.name} ${variable.description || ''} ${variable.semantic || ''} ${(variable.aliases || []).join(' ')} ${(variable.tags || []).join(' ')}` }
    })
    : tabsValue === 'operators'
      ? operators.map((operator) => ({ id: operator.id, categoryId: operator.categoryId, categoryLabel: operator.categoryLabel, contextSupported: operator.domains.includes(contextDomain), supported: operator.domains.includes(contextDomain), label: operator.label, description: operatorOptionContract(operator), keywords: `${operator.id} ${operator.essence} ${operator.semantic} ${operator.signature} ${operator.aliases.join(' ')} ${operator.tags.join(' ')}` }))
      : indicators.map((indicator) => {
        const category = indicatorCategory(indicator)
        const source = indicator.source === 'built_in' ? '内置' : '工作区'
        return { id: indicatorReferenceKey(indicator), categoryId: category.id, categoryLabel: category.label, contextSupported: true, supported: true, label: indicator.name, description: `${source} · 版本 ${indicator.revision} · 有限标量`, keywords: `${indicator.id} ${indicator.name} ${indicator.description} ${indicator.expression} ${source} ${category.label}` }
      })
  const catalogEntries = rawEntries.filter((entry) => tabsValue === 'variables' ? entry.contextSupported : entry.supported)
  const categoryMap = new Map<string, { label: string; count: number }>()
  catalogEntries.forEach((entry) => {
    // Different registry families may share one product-facing label. Group by
    // that label so the catalog does not expose duplicate UI categories.
    const current = categoryMap.get(entry.categoryLabel) ?? { label: entry.categoryLabel, count: 0 }
    categoryMap.set(entry.categoryLabel, { ...current, count: current.count + 1 })
  })
  const categories = [...categoryMap.entries()].map(([value, item]) => ({ value, label: `${item.label}（${item.count}）`, keywords: item.label }))
  const requestedCategory = selectedCategoryIds[tabsValue]
  const selectedCategory = categories.some((item) => item.value === requestedCategory) ? requestedCategory : categories[0]?.value || ''
  const entries = catalogEntries.filter((entry) => entry.categoryLabel === selectedCategory)
  const requestedItem = selectedItemIds[tabsValue]
  const selectedId = entries.some((entry) => entry.id === requestedItem) ? requestedItem : entries.find((entry) => entry.supported)?.id || entries[0]?.id || ''
  const selectedVariable = tabsValue === 'variables' ? variables.find((variable) => variable.name === selectedId) ?? null : null
  const selectedComposer = tabsValue === 'operators' ? operators.find((operator) => operator.id === selectedId) ?? null : null
  const selectedIndicator = tabsValue === 'indicators' ? indicators.find((indicator) => indicatorReferenceKey(indicator) === selectedId) ?? null : null
  const selectedIndicatorKey = selectedIndicator ? indicatorReferenceKey(selectedIndicator) : ''
  const selectedIndicatorExplanation = selectedIndicatorKey ? indicatorExplanations[selectedIndicatorKey] : undefined
  const selectedSupported = selectedVariable
    ? supportsVariableDomain(selectedVariable, contextDomain) && selectedVariable.availability !== 'unavailable' && selectedVariable.availability !== 'not_applicable'
    : selectedComposer
      ? selectedComposer.domains.includes(contextDomain)
      : Boolean(selectedIndicator)

  return <section aria-label="类型化计算目录">
    <div className="flex items-start justify-between gap-3"><div><h3 className="text-sm font-semibold text-slate-800">三级资源目录</h3><p className="mt-1 text-xs text-slate-500">目录规模增长后仍可按分类和关键词定位，不需要滚动浏览卡片墙。</p></div><span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-500">{rawEntries.filter((entry) => entry.supported).length} 项资源</span></div>
    <div className="mt-4 grid gap-3">
      <label className="text-xs font-semibold text-slate-600">资源类型<select aria-label="资源类型" value={tabsValue} onChange={(event) => onTabChange(event.target.value as CatalogTab)} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="variables">变量</option><option value="operators">计算算子</option><option value="indicators">已有指标</option></select></label>
      <SearchableCombobox label="资源分类" value={selectedCategory} options={categories} placeholder="搜索分类" onChange={(value) => setSelectedCategoryIds((current) => ({ ...current, [tabsValue]: value }))} />
      <SearchableCombobox label={`选择${catalogLabels[tabsValue]}`} value={selectedId} options={entries.map((entry) => ({ value: entry.id, label: entry.label, description: entry.description, keywords: entry.keywords, disabled: tabsValue !== 'variables' && !entry.supported }))} placeholder={`搜索${catalogLabels[tabsValue]}名称、别名或含义`} onChange={(value) => setSelectedItemIds((current) => ({ ...current, [tabsValue]: value }))} />
    </div>
    {!entries.length && <p className="mt-4 rounded-lg border border-dashed border-slate-200 bg-slate-50 p-3 text-sm text-slate-500" role="status">此分类暂无{catalogLabels[tabsValue]}。</p>}
    {selectedVariable && <article className={`mt-4 rounded-xl border p-4 ${selectedSupported ? 'border-slate-200 bg-white' : 'border-amber-200 bg-amber-50/40'}`}>
      <div className="flex items-start justify-between gap-3"><div className="min-w-0"><h4 className="text-base font-semibold text-slate-900">{selectedVariable.label}</h4><div className="mt-1 max-w-40 text-violet-700">{selectedVariable.latex ? <MathNotation latex={selectedVariable.latex} label={`${selectedVariable.label}的数学符号`} /> : <span className="text-xs text-slate-400">暂未配置数学符号</span>}</div></div><ShapeBadge shape={inferShape(selectedVariable)} valueType={selectedVariable.value_type} /></div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{selectedVariable.description || selectedVariable.semantic || `${shapeLabel(inferShape(selectedVariable), selectedVariable.value_type)}输入。`}</p>
      <dl className="mt-4 grid gap-x-4 gap-y-3 text-xs sm:grid-cols-2">
        <div><dt className="font-semibold text-slate-700">变量含义</dt><dd className="mt-1 text-slate-500">{selectedVariable.description || selectedVariable.semantic || `${selectedVariable.label}在当前计算窗口内的数值。`}</dd></div>
        <div><dt className="font-semibold text-slate-700">数据类型</dt><dd className="mt-1 text-slate-500">{shapeLabel(inferShape(selectedVariable), selectedVariable.value_type)}</dd></div>
        <div><dt className="font-semibold text-slate-700">数据来源</dt><dd className="mt-1 text-slate-500">{selectedVariable.source || '由指标运行上下文提供'}</dd></div>
        <div><dt className="font-semibold text-slate-700">口径 / 频率 / 单位</dt><dd className="mt-1 text-slate-500">{[selectedVariable.data_basis, selectedVariable.frequency, selectedVariable.unit].filter(Boolean).join(' · ') || '由运行上下文决定'}</dd></div>
        <div><dt className="font-semibold text-slate-700">适用范围</dt><dd className="mt-1 text-slate-500">{(selectedVariable.product_kinds || selectedVariable.domains || [contextDomain]).map(scopeLabel).join(' / ')}</dd></div>
      </dl>
      <p className="mt-4 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-500">变量可直接用于公式构建；实际可计算性将在选择产品并执行预览后，根据该产品的数据覆盖情况判断。</p>
      <button type="button" disabled={!selectedSupported} onClick={() => onInsertVariable(selectedVariable.latex || selectedVariable.name)} className="mt-4 w-full rounded-lg bg-violet-600 px-3 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300">{selectedSupported ? variableActionLabel : '当前计算域不可用'}</button>
    </article>}
    {selectedComposer && <article className={`mt-4 rounded-xl border p-4 ${selectedSupported ? 'border-slate-200 bg-white' : 'border-slate-100 bg-slate-50 opacity-60'}`}>
      <div className="flex items-start justify-between gap-3"><div><h4 className="text-base font-semibold text-slate-900">{selectedComposer.label}</h4><p className="mt-1 text-xs text-slate-500">{humanizeTechnicalTypes(selectedComposer.essence)}</p></div><ShapeBadge shape={selectedComposer.outputShape} valueType={selectedComposer.outputType} /></div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{humanizeTechnicalTypes(selectedComposer.semantic)}</p><p className="mt-3 rounded-lg bg-slate-50 p-3 text-xs leading-5 text-slate-600" aria-label="算子输入输出说明">{operatorContractDescription(selectedComposer)}</p>
      <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[440px] text-left text-xs"><caption className="sr-only">{selectedComposer.label} 参数要求</caption><thead><tr className="border-b border-slate-200 text-slate-500"><th className="pb-2 pr-3">参数</th><th className="pb-2 pr-3">支持的数据</th><th className="pb-2">参数含义</th></tr></thead><tbody>{selectedComposer.parameters.map((parameter) => <tr key={parameter.name} className="border-b border-slate-100 align-top last:border-0"><td className="py-2 pr-3 font-semibold text-slate-700">{parameterLabel(parameter.name, parameter.label || parameter.name)}{parameter.optional ? '（可选）' : ''}</td><td className="py-2 pr-3 text-slate-500">{acceptedShapes(parameter).map((shape) => shapeLabel(shape)).join(' / ')}</td><td className="py-2 text-slate-500">{parameterDescription(parameter)}</td></tr>)}</tbody></table></div>
      <dl className="mt-4 grid gap-2 text-xs text-slate-500 sm:grid-cols-2"><div><dt className="font-semibold text-slate-700">返回结果</dt><dd className="mt-1">{shapeLabel(selectedComposer.outputShape, selectedComposer.outputType)}</dd></div><div><dt className="font-semibold text-slate-700">计算特点</dt><dd className="mt-1">{costEstimateLabel(selectedComposer.costEstimate)}</dd></div><div><dt className="font-semibold text-slate-700">所属分类</dt><dd className="mt-1">{selectedComposer.categoryLabel}</dd></div><div><dt className="font-semibold text-slate-700">执行方式</dt><dd className="mt-1">{executionBackendLabel(selectedComposer.executionBackend)}</dd></div>{selectedComposer.displayTemplate && <div className="sm:col-span-2"><dt className="font-semibold text-slate-700">数学符号示例</dt><dd className="mt-1 rounded-lg border border-violet-100 bg-violet-50/40 px-3 py-2"><MathNotation latex={selectedComposer.displayTemplate} label={`${selectedComposer.label}的数学符号示例`} /></dd></div>}</dl>
      <button type="button" disabled={!selectedSupported} onClick={() => onOpenComposer(selectedComposer)} className="mt-4 w-full rounded-lg bg-violet-600 px-3 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300">{selectedSupported ? '配置算子参数' : '不适用于单产品指标'}</button>
    </article>}
    {selectedIndicator && <article className="mt-4 rounded-xl border border-slate-200 bg-white p-4">
      <div className="flex items-start justify-between gap-3"><div><h4 className="text-base font-semibold text-slate-900">{selectedIndicator.name}</h4><p className="mt-1 text-xs text-slate-500">{selectedIndicator.source === 'built_in' ? '内置指标' : '工作区指标'} · 锁定 v{selectedIndicator.revision}</p></div><ShapeBadge shape="scalar" valueType="scalar" /></div>
      <p className="mt-3 text-sm leading-6 text-slate-600">{selectedIndicator.description || '已保存的标量指标公式。'}</p>
      <div className="mt-3 rounded-lg border border-violet-100 bg-violet-50/40 px-3 py-2">{mathFormulaForDisplay(selectedIndicator.display_latex, selectedIndicator.expression) ? <MathNotation latex={mathFormulaForDisplay(selectedIndicator.display_latex, selectedIndicator.expression)!} label={`${selectedIndicator.name}的数学公式`} /> : <p className="text-xs text-slate-500">该兼容指标暂未提供数学符号排版；可在高级公式模式中查看源码。</p>}</div>
      <dl className="mt-4 grid gap-3 text-xs sm:grid-cols-2">
        <div><dt className="font-semibold text-slate-700">指标类型</dt><dd className="mt-1 text-slate-500">{indicatorCategory(selectedIndicator).label}</dd></div>
        <div><dt className="font-semibold text-slate-700">方向 / 单位</dt><dd className="mt-1 text-slate-500">{selectedIndicator.direction === 'higher_better' ? '越高越好' : '越低越好'} · {selectedIndicator.unit || '无单位'}</dd></div>
        <div><dt className="font-semibold text-slate-700">计算语义</dt><dd className="mt-1 text-slate-500">{selectedIndicator.catalog_status === 'compatibility' ? '兼容旧版指标逻辑' : '当前类型化指标逻辑'}，已锁定版本</dd></div>
        <div><dt className="font-semibold text-slate-700">输出契约</dt><dd className="mt-1 text-slate-500">有限标量 · {measureLabel(selectedIndicator.output_measure)}</dd></div>
      </dl>
      <p className="mt-4 rounded-lg border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">插入时由服务端按当前 revision 重新校验并展开为普通公式，不建立可变引用；原指标以后更新不会改变本草稿。</p>
      <div className="mt-4 grid gap-2 sm:grid-cols-2">
        <button type="button" aria-expanded={visibleExplanationId === selectedIndicatorKey} disabled={explainingIndicatorId !== ''} onClick={() => {
          if (selectedIndicatorExplanation) {
            setVisibleExplanationId((current) => current === selectedIndicatorKey ? '' : selectedIndicatorKey)
            return
          }
          setExplainingIndicatorId(selectedIndicatorKey)
          setIndicatorExplanationError(null)
          void onExplainIndicator(selectedIndicator)
            .then((response) => {
              setIndicatorExplanations((current) => ({ ...current, [selectedIndicatorKey]: response }))
              setVisibleExplanationId(selectedIndicatorKey)
            })
            .catch((explanationError) => setIndicatorExplanationError({ id: selectedIndicatorKey, message: userFacingErrorMessage(explanationError, '指标计算说明解析失败，请稍后重试。') }))
            .finally(() => setExplainingIndicatorId(''))
        }} className="rounded-lg border border-violet-200 bg-white px-3 py-2.5 text-sm font-semibold text-violet-700 hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-50">{explainingIndicatorId === selectedIndicatorKey ? '正在解析…' : visibleExplanationId === selectedIndicatorKey ? '收起计算说明' : '查看计算说明'}</button>
        <button type="button" disabled={insertingIndicatorId !== null} onClick={() => void onInsertIndicator(selectedIndicator)} className="rounded-lg bg-violet-600 px-3 py-2.5 text-sm font-semibold text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:bg-slate-300">{insertingIndicatorId === selectedIndicator.id ? '正在展开锁定版本…' : indicatorActionLabel}</button>
      </div>
      {indicatorExplanationError?.id === selectedIndicatorKey && <p role="alert" className="mt-3 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{indicatorExplanationError.message}</p>}
      {visibleExplanationId === selectedIndicatorKey && selectedIndicatorExplanation?.dag && <FormulaExplanation compact dag={selectedIndicatorExplanation.dag} variables={variables} operators={operators} />}
      {visibleExplanationId === selectedIndicatorKey && selectedIndicatorExplanation && !selectedIndicatorExplanation.dag && <p role="status" className="mt-3 rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">该兼容指标可以展开计算，但当前版本没有可展示的逐步解析信息。</p>}
    </article>}
    {actionError && <p role="alert" className="mt-4 rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{actionError}</p>}
  </section>
}

function ShapeBadge({ shape, valueType }: { shape: IndicatorShape; valueType?: string }) {
  const tone: Record<IndicatorShape, string> = { window: 'bg-violet-100 text-violet-800', record: 'bg-sky-100 text-sky-800', scalar: 'bg-slate-100 text-slate-700', series: 'bg-sky-100 text-sky-800', vector: 'bg-cyan-100 text-cyan-800', matrix: 'bg-indigo-100 text-indigo-800', mask: 'bg-emerald-100 text-emerald-800', tuple: 'bg-amber-100 text-amber-800', unknown: 'bg-slate-100 text-slate-500' }
  return <span className={`shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold ${tone[shape]}`}>{shapeLabel(shape, valueType)}</span>
}

function InferencePanel({ inference, resourceLabels }: { inference: InferenceResponse; resourceLabels: Map<string, string> }) {
  const { s } = useI18n()
  const scalarOutput = inference.shape === 'scalar'
  const dependencies = (inference.dependencies ?? []).map((dependency) => resourceLabels.get(dependency) || '公式所需数据')
  return <section className="mt-4 rounded-xl border border-sky-200 bg-sky-50 p-4" role="status" aria-labelledby="formula-output-title">
    <div className="flex flex-wrap items-center gap-2"><h3 id="formula-output-title" className="font-semibold text-sky-900">{systemText('outputAuthoring.inference', {}, '类型推断')}</h3><ShapeBadge shape={inference.shape} valueType={inference.inferred_type} /></div>
    <dl className="mt-3 grid gap-3 rounded-lg bg-white/70 px-3 py-3 text-xs sm:grid-cols-2">
      <div><dt className="font-semibold text-slate-700">结果形式</dt><dd className="mt-1 text-slate-600">{shapeLabel(inference.shape, inference.inferred_type)}</dd></div>
      <div><dt className="font-semibold text-slate-700">能否作为指标</dt><dd className="mt-1 text-slate-600">{scalarOutput ? '可以。最终得到一个数值。' : inference.shape === 'record' ? s('primitiveAuthoring.stateHint', {}, '这是共享计算的中间结果。请连接字段提取算子，再生成一个独立指标结果。') : '暂时不可以。还需要使用求和、平均值、标准差等归约计算得到单个数值。'}</dd></div>
      {dependencies.length > 0 && <div className="sm:col-span-2"><dt className="font-semibold text-slate-700">需要的数据</dt><dd className="mt-1 text-slate-600">{dependencies.join('、')}</dd></div>}
    </dl>
    {inference.semantic_warnings.length > 0 && <ul className="mt-2 space-y-1 text-xs text-amber-800">{inference.semantic_warnings.map((warning, index) => <li key={`${warning.code}-${index}`}>{diagnosticMessage(warning.code, warning.message)}</li>)}</ul>}
  </section>
}

function FormulaExplanationDisclosure({ dag, variables, operators }: { dag: IndicatorDag; variables: IndicatorVariable[]; operators: ComposerItem[] }) {
  const [open, setOpen] = useState(false)
  const disclosureId = useId()
  const dagKey = `${Object.values(dag.roots).join(',')}:${dag.nodes.map((node) => node.id).join(',')}:${dag.edges.length}`
  const variableCount = dag.nodes.filter((node) => node.kind === 'variable').length
  const calculationCount = dag.nodes.filter((node) => node.kind !== 'variable' && node.kind !== 'constant').length

  useEffect(() => setOpen(false), [dagKey])

  return <section className="mt-4 rounded-xl border border-violet-200 bg-violet-50/40 p-4" aria-label="公式解析摘要">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div>
        <h3 className="text-sm font-semibold text-slate-900">计算逻辑已解析</h3>
        <p className="mt-1 text-xs leading-5 text-slate-600">{variableCount} 个输入变量 · {calculationCount} 个计算步骤。完整说明包含变量符号、算子符号和逐步计算。</p>
      </div>
      <button type="button" aria-expanded={open} aria-controls={disclosureId} onClick={() => setOpen((current) => !current)} className="rounded-lg border border-violet-200 bg-white px-3 py-2 text-xs font-semibold text-violet-700 hover:bg-violet-50 focus:outline-none focus:ring-2 focus:ring-violet-300">{open ? '收起完整计算说明' : '查看完整计算说明'}</button>
    </div>
    {open && <div id={disclosureId}><FormulaExplanation compact dag={dag} variables={variables} operators={operators} /></div>}
  </section>
}

function FormulaExplanation({ dag, variables, operators, compact = false }: { dag: IndicatorDag; variables: IndicatorVariable[]; operators: ComposerItem[]; compact?: boolean }) {
  const nodeById = new Map(dag.nodes.map((node) => [String(node.id), node]))
  const variableById = new Map(variables.map((variable) => [variable.name, variable]))
  const operatorById = new Map(operators.flatMap((operator) => [
    [operator.id, operator] as const,
    [operator.label, operator] as const,
    ...operator.aliases.map((alias) => [alias, operator] as const),
  ]))
  const incoming = new Map<string, typeof dag.edges>()
  dag.edges.forEach((edge) => {
    const target = String(edge.target)
    incoming.set(target, [...(incoming.get(target) ?? []), edge].sort((left, right) => (left.order ?? 0) - (right.order ?? 0)))
  })
  const depthCache = new Map<string, number>()
  const depthOf = (nodeId: string, visiting = new Set<string>()): number => {
    const cached = depthCache.get(nodeId)
    if (cached !== undefined) return cached
    if (visiting.has(nodeId)) return 0
    const next = new Set(visiting).add(nodeId)
    const parents = incoming.get(nodeId) ?? []
    const depth = parents.length ? Math.max(...parents.map((edge) => depthOf(String(edge.source), next))) + 1 : 0
    depthCache.set(nodeId, depth)
    return depth
  }
  dag.nodes.forEach((node) => depthOf(String(node.id)))
  const nodeName = (node: IndicatorDagNode | undefined) => {
    if (!node) return '上一计算结果'
    if (node.kind === 'variable') return variableById.get(node.label)?.label || node.label
    if (node.kind === 'constant') return `常量 ${node.formula_fragment || node.label}`
    return operatorDisplayLabel(
      dagNodeOperatorId(node),
      operatorById.get(dagNodeOperatorId(node))?.label || node.label,
    )
  }
  const variableNodes = dag.nodes.filter((node) => node.kind === 'variable')
  const calculationNodes = dag.nodes
    .filter((node) => node.kind !== 'variable' && node.kind !== 'constant')
    .sort((left, right) => (depthCache.get(String(left.id)) ?? 0) - (depthCache.get(String(right.id)) ?? 0))
  const usedOperators = [...new Map(calculationNodes.map((node) => {
    const operator = operatorById.get(dagNodeOperatorId(node))
    return [operator?.id || dagNodeOperatorId(node), { node, operator }] as const
  })).values()]
  const rootId = dag.roots.result ?? Object.values(dag.roots)[0]
  const rootNode = nodeById.get(String(rootId))
  const wrapperClass = compact ? 'mt-4 rounded-xl border border-violet-100 bg-violet-50/30 p-3' : 'mt-4 rounded-xl border border-violet-200 bg-white p-4 shadow-sm'
  return <section className={wrapperClass} aria-label="公式计算说明">
    {dag.nodes.some(node => dagNodeOperatorId(node) === 'rolling_apply') && <p className="mb-3 rounded-lg bg-violet-100 p-3 text-xs leading-5 text-violet-900">{systemText('rollingScope.graphHint')}</p>}
    <div><h3 className="font-semibold text-slate-900">公式计算说明</h3><p className="mt-1 text-xs leading-5 text-slate-500">数学符号、中文名称和计算步骤一一对应；先认变量，再认算子，最后按顺序理解完整计算。</p></div>
    <div className="mt-4">
      <h4 className="text-xs font-semibold text-slate-700">输入变量与符号</h4>
      {variableNodes.length ? <ul className="mt-2 grid gap-2 sm:grid-cols-2">{variableNodes.map((node) => {
        const variable = variableById.get(node.label)
        const symbol = variable?.latex || node.latex_fragment
        const shape = variable ? inferShape(variable) : dagNodeShape(node)
        return <li key={node.id} className="rounded-lg border border-sky-100 bg-sky-50/60 p-3"><div className="flex items-center gap-3"><div className="min-w-20 rounded-md bg-white px-2 py-1 text-center text-sky-900">{symbol ? <MathNotation latex={symbol} label={`${nodeName(node)}的数学符号`} /> : <span className="text-xs">输入</span>}</div><div className="min-w-0"><p className="text-sm font-semibold text-slate-800">{nodeName(node)}</p><p className="mt-0.5 text-xs text-slate-500">{shapeLabel(shape, dagNodeValueType(node))}</p></div></div><p className="mt-2 text-xs leading-5 text-slate-600">{variable?.description || variable?.semantic || '由当前运行上下文提供的真实输入数据。'}</p></li>
      })}</ul> : <p className="mt-2 text-xs text-slate-500">该公式不依赖外部变量。</p>}
    </div>
    <div className="mt-4">
      <h4 className="text-xs font-semibold text-slate-700">算子与数学符号</h4>
      <ul className="mt-2 grid gap-2 sm:grid-cols-2">{usedOperators.map(({ node, operator }) => {
        const symbol = mathFormulaForDisplay(operator?.displayTemplate, node.latex_fragment || node.formula_fragment)
        return <li key={operator?.id || node.id} className="rounded-lg border border-violet-100 bg-violet-50/60 p-3"><div className="flex items-center gap-3"><div className="min-w-24 rounded-md bg-white px-2 py-1 text-center text-violet-900">{symbol ? <MathNotation latex={symbol} label={`${nodeName(node)}的数学符号`} /> : <span className="text-xs">数学运算</span>}</div><div className="min-w-0"><p className="text-[10px] font-semibold uppercase tracking-wide text-violet-600">对应算子</p><p className="text-sm font-semibold text-slate-800">{nodeName(node)}</p></div></div><p className="mt-2 text-xs leading-5 text-slate-600">{humanizeTechnicalTypes(operator?.essence || operator?.semantic || '对输入数据执行受控数学计算。')}</p></li>
      })}</ul>
    </div>
    <div className="mt-4">
      <h4 className="text-xs font-semibold text-slate-700">逐步计算</h4>
      <ol className="mt-2 space-y-2">{calculationNodes.map((node, index) => {
        const operator = operatorById.get(dagNodeOperatorId(node))
        const inputs = incoming.get(String(node.id)) ?? []
        const stepFormula = mathFormulaForDisplay(node.latex_fragment, node.formula_fragment) || operator?.displayTemplate
        return <li key={node.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3"><div className="flex items-start gap-3"><span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-violet-600 text-xs font-bold text-white">{index + 1}</span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-2"><p className="text-sm font-semibold text-slate-800">使用“{nodeName(node)}”</p><span className="text-xs text-slate-400">得到{shapeLabel(dagNodeShape(node), dagNodeValueType(node))}</span></div>{stepFormula && <div className="mt-2 rounded-md bg-white px-2 py-1"><MathNotation latex={stepFormula} label={`第 ${index + 1} 步的数学公式`} /></div>}{inputs.length > 0 && <ul className="mt-2 space-y-1 text-xs text-slate-600">{inputs.map((edge, inputIndex) => {
          const source = nodeById.get(String(edge.source))
          const parameter = operatorParameterLabel(
            operator?.id || dagNodeOperatorId(node),
            inputIndex,
            edge.parameter || edge.parameter_name || edge.input_name || operator?.parameters[inputIndex]?.name,
            operator?.parameters[inputIndex]?.label || `输入 ${inputIndex + 1}`,
          )
          return <li key={`${edge.source}-${inputIndex}`}><span className="font-semibold text-violet-700">{parameter}</span>：{nodeName(source)}</li>
        })}</ul>}</div></div></li>
      })}</ol>
    </div>
    <p className="mt-4 rounded-lg border border-emerald-100 bg-emerald-50 px-3 py-2 text-xs leading-5 text-emerald-800"><span className="font-semibold">最终结果：</span>{rootNode ? `${nodeName(rootNode)}输出${shapeLabel(dagNodeShape(rootNode), dagNodeValueType(rootNode))}` : '按上述步骤得到指标结果'}。</p>
  </section>
}

function ComposerNodeFields({ node, path, variables, operators, indicators, contextDomain, depth, onChange }: { node: ComposerNode; path: number[]; variables: IndicatorVariable[]; operators: ComposerItem[]; indicators: IndicatorDefinition[]; contextDomain: IndicatorContextDomain; depth: number; onChange: (path: number[], argument: ComposerArgument) => void }) {
  return <div className="space-y-4">
    {node.arguments.map((argument, index) => {
      const argumentPath = [...path, index]
      const needs = acceptedShapes(argument.parameter)
      const fixedConstant = parameterRequiresFixedConstant(argument.parameter)
      const compatibleVariables = fixedConstant ? [] : variables.filter((variable) => (
        isCompatibleVariable(variable, argument.parameter, contextDomain)
      ))
      const compatibleOperators = fixedConstant ? [] : operators.filter((operator) => isCompatibleNestedOperator(operator, argument.parameter, contextDomain))
      const compatibleIndicators = fixedConstant ? [] : parameterAcceptsIndicator(argument.parameter) ? indicators : []
      const constantAllowed = fixedConstant || needs.includes('scalar') || needs.includes('unknown')
      const nestedAllowed = !fixedConstant && depth < 3 && compatibleOperators.length > 0
      const label = parameterLabel(argument.parameter.name, argument.parameter.label || argument.parameter.name)
      const source = fixedConstant ? 'constant' : argument.source
      return <fieldset key={`${argument.parameter.name}-${argumentPath.join('-')}`} className={`rounded-xl border p-3 ${depth > 0 ? 'border-violet-100 bg-violet-50/30' : 'border-slate-200'}`}>
        <legend className="px-1 text-sm font-semibold text-slate-800">{label} <span className="font-normal text-slate-400">· {needs.map((shape) => shapeLabel(shape)).join(' / ')}</span></legend>
        <p className="mt-1 text-xs leading-5 text-slate-500">{parameterDescription(argument.parameter)}</p>
        {fixedConstant && <p className="mt-2 rounded-lg border border-sky-100 bg-sky-50 px-2 py-1.5 text-xs text-sky-800">该配置决定指标公式及所需历史，必须随指标版本固定。需要其他数值时，请保存为另一个指标或新版本。</p>}
        <label className="mt-3 block text-xs font-semibold text-slate-600">输入来源<select aria-label={`${label} 输入来源`} disabled={fixedConstant} value={source} onChange={(event) => {
          const nextSource = event.target.value as ComposerSource
          if (nextSource === 'variable') {
            const variable = compatibleVariables[0]
            onChange(argumentPath, { ...argument, source: nextSource, value: variable?.name || '', nested: undefined })
          } else if (nextSource === 'constant') {
            const value = typeof argument.parameter.default === 'number' ? String(argument.parameter.default) : argument.value || ''
            onChange(argumentPath, { ...argument, source: nextSource, value, nested: undefined })
          } else if (nextSource === 'operator') {
            const nestedItem = compatibleOperators[0]
            onChange(argumentPath, { ...argument, source: nextSource, value: nestedItem?.id || '', nested: nestedItem ? initialComposerNode(nestedItem, variables, contextDomain) : undefined })
          } else if (nextSource === 'indicator') {
            const indicator = compatibleIndicators[0]
            onChange(argumentPath, { ...argument, source: nextSource, value: indicator ? indicatorReferenceKey(indicator) : '', nested: undefined })
          } else {
            onChange(argumentPath, { ...argument, source: nextSource, value: '', nested: undefined })
          }
        }} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100 disabled:bg-slate-100">
          <option value="variable" disabled={fixedConstant || compatibleVariables.length === 0}>兼容变量</option>
          <option value="constant" disabled={!constantAllowed}>{fixedConstant ? '定义级固定常量' : '有限数值常量'}</option>
          <option value="operator" disabled={!nestedAllowed}>嵌套算子{depth >= 3 ? '（已达层级上限）' : ''}</option>
          <option value="indicator" disabled={fixedConstant || compatibleIndicators.length === 0}>已有标量指标</option>
          {argument.parameter.optional && !fixedConstant && <option value="omitted">省略可选参数</option>}
        </select></label>
        {source === 'variable' && <>
          <label className="mt-3 block text-xs font-semibold text-slate-600">选择兼容变量<select aria-label={`${label} 变量`} value={argument.value} onChange={(event) => onChange(argumentPath, { ...argument, value: event.target.value })} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="">选择兼容变量</option>{variables.map((variable) => {
            const compatible = isCompatibleVariable(variable, argument.parameter, contextDomain)
            return <option key={variable.name} value={variable.name} disabled={!compatible}>{variable.label} · {shapeLabel(inferShape(variable), variable.value_type)}{compatible ? '' : '（类型或语义不兼容）'}</option>
          })}</select></label>
          {compatibleVariables.length === 0 && <p className="mt-2 text-xs text-amber-700">当前变量中没有满足该参数类型与语义约束的输入，可改用嵌套算子构造。</p>}
        </>}
        {source === 'constant' && <label className="mt-3 block text-xs font-semibold text-slate-600">{fixedConstant ? '定义级固定常量' : '有限数值常量'}<input aria-label={`${label} 有限常量`} type="number" step="any" value={argument.value} onChange={(event) => onChange(argumentPath, { ...argument, source: 'constant', value: event.target.value, nested: undefined })} className="mt-1 block min-h-11 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100" /></label>}
        {source === 'indicator' && <div className="mt-3 rounded-xl border border-sky-200 bg-sky-50/40 p-3">
          <label className="block text-xs font-semibold text-sky-800">选择已有标量指标<select aria-label={`${label} 已有指标`} value={argument.value} onChange={(event) => onChange(argumentPath, { ...argument, value: event.target.value, nested: undefined })} className="mt-1 block min-h-11 w-full rounded-lg border border-sky-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="">选择同协议指标</option>{compatibleIndicators.map((indicator) => <option key={indicatorReferenceKey(indicator)} value={indicatorReferenceKey(indicator)}>{indicator.name} · {indicator.source === 'built_in' ? '内置' : '工作区'} v{indicator.revision}</option>)}</select></label>
          <p className="mt-2 text-xs leading-5 text-sky-700">提交时按锁定版本安全展开为表达式，不建立运行时指标依赖。</p>
        </div>}
        {source === 'operator' && <div className="mt-3 rounded-xl border border-violet-200 bg-white p-3">
          <label className="block text-xs font-semibold text-violet-800">嵌套数学算子<select aria-label={`${label} 嵌套算子`} value={argument.nested?.item.id || argument.value} onChange={(event) => {
            const nestedItem = compatibleOperators.find((operator) => operator.id === event.target.value)
            onChange(argumentPath, { ...argument, value: nestedItem?.id || '', nested: nestedItem ? initialComposerNode(nestedItem, variables, contextDomain) : undefined })
          }} className="mt-1 block min-h-11 w-full rounded-lg border border-violet-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="">选择输出类型兼容的算子</option>{compatibleOperators.map((operator) => <option key={operator.id} value={operator.id}>{operator.label} · {operatorOptionContract(operator)}</option>)}</select></label>
          {argument.nested && <div className="mt-3 border-l-2 border-violet-200 pl-3"><div className="mb-3 flex items-center justify-between gap-2"><p className="text-xs font-semibold text-violet-800">第 {depth + 1} 层 · {argument.nested.item.label}</p><ShapeBadge shape={argument.nested.item.outputShape} valueType={argument.nested.item.outputType} /></div><ComposerNodeFields node={argument.nested} path={argumentPath} variables={variables} operators={operators} indicators={indicators} contextDomain={contextDomain} depth={depth + 1} onChange={onChange} /></div>}
        </div>}
        {source === 'omitted' && <p className="mt-3 rounded-lg bg-slate-50 px-3 py-2 text-xs text-slate-500">不会提交该参数，服务端使用注册表声明的默认行为。</p>}
      </fieldset>
    })}
  </div>
}

function ComposerDrawer({ composer, indicatorName, variables, operators, indicators, contextDomain, loading, canApply, error, onClose, onBrowseResources, onChange, onApplyModeChange, onApply }: { composer: ComposerState; indicatorName: string; variables: IndicatorVariable[]; operators: ComposerItem[]; indicators: IndicatorDefinition[]; contextDomain: IndicatorContextDomain; loading: boolean; canApply: boolean; error: string | null; onClose: () => void; onBrowseResources: () => void; onChange: (path: number[], argument: ComposerArgument) => void; onApplyModeChange: (mode: ComposerApplyMode) => void; onApply: () => void }) {
  const dialogRef = useRef<HTMLElement | null>(null)
  const closeRef = useRef<HTMLButtonElement | null>(null)
  const composerOpen = Boolean(composer)
  const siblingIssue = composer ? firstComposerSiblingIssue(composer.node, variables) : null
  useEffect(() => {
    if (!composerOpen) return
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null
    requestAnimationFrame(() => closeRef.current?.focus())
    const trapFocus = (event: KeyboardEvent) => {
      if (event.key !== 'Tab' || !dialogRef.current) return
      const focusable = [...dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [href]')]
      if (!focusable.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', trapFocus)
    return () => { document.removeEventListener('keydown', trapFocus); previous?.focus() }
  }, [composerOpen])
  if (!composer) return null
  const editingCurrentFormula = composer.origin === 'current_formula'

  return <div className="fixed inset-0 z-[100] flex justify-end bg-slate-950/35" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section ref={dialogRef} role="dialog" aria-modal="true" aria-labelledby="composer-title" onKeyDown={(event) => { if (event.key === 'Escape') onClose() }} className="flex h-full w-full max-w-xl flex-col bg-white shadow-2xl">
      <div className="flex items-start justify-between border-b border-slate-200 p-5"><div><p className="text-xs font-semibold text-violet-700">{editingCurrentFormula ? '当前指标计算逻辑' : '计算算子'}</p><h2 id="composer-title" className="mt-1 text-lg font-bold text-slate-900">{editingCurrentFormula ? `编辑“${indicatorName || '当前指标'}”的计算逻辑` : `配置 ${composer.node.item.label}`}</h2><p className="mt-1 text-sm text-slate-600">{editingCurrentFormula ? `最终计算步骤：${composer.node.item.label}。修改下方参数或嵌套步骤后应用到当前公式。` : composer.node.item.essence}</p></div><button ref={closeRef} type="button" onClick={onClose} className="rounded-md px-2 py-1 text-sm text-slate-500 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-violet-300" aria-label="关闭参数配置">关闭</button></div>
      <div className="flex-1 space-y-4 overflow-auto p-5">
        <div className="rounded-xl border border-violet-100 bg-violet-50/60 p-3"><p className="text-xs font-semibold text-violet-800">输入与返回结果</p><p className="mt-1 text-xs leading-5 text-violet-900">{operatorContractDescription(composer.node.item)}</p><p className="mt-2 text-xs text-violet-800">嵌套算子会先安全展开，再作为上级算子的输入；系统会在插入前检查数据类型和含义是否兼容。</p></div>
        <ComposerNodeFields node={composer.node} path={[]} variables={variables} operators={operators} indicators={indicators} contextDomain={contextDomain} depth={0} onChange={onChange} />
        {editingCurrentFormula ? <div className="rounded-xl border border-slate-200 bg-slate-50 p-3"><p className="text-sm font-semibold text-slate-700">需要重新选择根步骤？</p><p className="mt-1 text-xs leading-5 text-slate-500">可以返回资源目录，从变量、计算算子或已有指标重新构建当前公式。</p><button type="button" onClick={onBrowseResources} className="mt-3 min-h-11 w-full rounded-lg border border-violet-200 bg-white px-3 py-2 text-sm font-semibold text-violet-700 hover:bg-violet-50 focus:outline-none focus:ring-2 focus:ring-violet-300">改用其他构建资源</button></div> : composer.node.item.outputShape === 'record' ? <p className="rounded-xl border border-sky-100 bg-sky-50 p-3 text-sm text-sky-900">{systemText('primitiveAuthoring.stateHint', {}, '这是共享计算的中间结果。请连接字段提取算子，再生成一个独立指标结果。')}</p> : <label className="block rounded-xl border border-slate-200 p-3 text-sm font-semibold text-slate-700">应用方式<select aria-label="应用方式" value={composer.applyMode} onChange={(event) => onApplyModeChange(event.target.value as ComposerApplyMode)} className="mt-2 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="replace_formula">替换完整公式</option><option value="replace_selection" disabled={composer.selectionEnd <= composer.selectionStart}>替换高级模式选中内容</option><option value="insert_cursor">插入高级模式光标位置</option></select><span className="mt-2 block text-xs font-normal text-slate-500">选择后再执行展开，系统不会隐式猜测插入位置。</span></label>}
        {error && <p id="composer-error" role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
        {!canApply && !error && <p id="composer-help" role="status" className="rounded-xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-800">{siblingIssue || '请完成所有必填参数后再展开；不兼容的参数组合不会被提交。'}</p>}
      </div>
      <div className="flex gap-3 border-t border-slate-200 p-5"><button type="button" onClick={onClose} className="flex-1 rounded-lg border border-slate-200 py-2 text-sm font-semibold text-slate-700">取消</button><button type="button" onClick={onApply} disabled={loading || !canApply} aria-describedby={!canApply ? 'composer-help' : error ? 'composer-error' : undefined} className="flex-1 rounded-lg bg-violet-600 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">{loading ? '应用中…' : editingCurrentFormula ? '应用逻辑修改' : composer.node.item.outputShape === 'record' ? systemText('outputAuthoring.apply', {}, '应用计算') : '展开到公式'}</button></div>
    </section>
  </div>
}

function IndicatorGroup({ title, items, selectedId, onSelect, empty }: { title: string; items: IndicatorDefinition[]; selectedId: string | null; onSelect: (item: IndicatorDefinition) => void; empty?: string }) {
  return <section><h3 className="px-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3><div className="mt-2 space-y-1">{items.map((item) => <button key={item.id} type="button" onClick={() => onSelect(item)} className={`w-full rounded-xl px-3 py-3 text-left transition focus:outline-none focus:ring-2 focus:ring-violet-300 ${item.id === selectedId ? 'bg-violet-100 text-violet-900' : 'hover:bg-slate-50 text-slate-700'}`}><span className="flex items-center justify-between gap-2"><span className="truncate text-sm font-semibold">{item.name}</span><span className="flex shrink-0 items-center gap-1"><span className={`rounded-full px-2 py-0.5 text-[11px] ${(item.result_kind ?? 'scalar') === 'time_series' ? 'bg-sky-100 text-sky-700' : 'bg-emerald-100 text-emerald-700'}`}>{(item.result_kind ?? 'scalar') === 'time_series' ? '时序' : '标量'}</span><span className={`rounded-full px-2 py-0.5 text-[11px] ${item.read_only ? 'bg-slate-200 text-slate-600' : 'bg-violet-50 text-violet-700'}`}>{item.read_only ? '内置' : `v${item.revision}`}</span></span></span><span className="mt-1 block line-clamp-2 text-xs text-slate-500">{item.description || item.expression}</span></button>)}{!items.length && empty && <p className="px-2 py-3 text-sm text-slate-400">{empty}</p>}</div></section>
}

function TimeSeriesDefinitionFields({
  draft,
  variables,
  validation,
  onPatch,
}: {
  draft: IndicatorDraft
  variables: IndicatorVariable[]
  validation: ValidationResponse | null
  onPatch: (patch: Partial<IndicatorDraft>) => void
}) {
  const axisVariables = variables.filter((variable) => inferShape(variable) === 'series')
  const historyPolicy = validation?.history_policy ?? draft.history_policy
  const lookbackObservations = validation?.lookback_observations ?? draft.lookback_observations
  const minimumObservations = validation?.minimum_observations ?? draft.minimum_observations
  const fixedParameters = validation?.fixed_parameters ?? draft.fixed_parameters ?? []
  return <div className="mt-4 space-y-4">
    <div className="rounded-xl border border-sky-100 bg-sky-50 px-3 py-2 text-xs leading-5 text-sky-800">这里只定义计算逻辑和时间轴。可在下方“计算参数”中开放窗口、平滑周期等输入；未开放的输入保持固定。图表位置由具体使用页面决定。</div>
    <label className="block max-w-md text-sm font-medium text-slate-700">日期轴变量<select aria-label="日期轴变量" value={draft.axis_anchor ?? ''} onChange={(event) => onPatch({ axis_anchor: event.target.value || null })} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-3 py-2 focus:border-violet-500 focus:outline-none focus:ring-2 focus:ring-violet-100"><option value="">请选择日期轴变量</option>{axisVariables.map((variable) => <option key={variable.name} value={variable.name}>{variable.label}</option>)}</select></label>
    <section className="rounded-xl border border-slate-200 bg-slate-50 p-4" aria-label="自动推断的历史计算契约">
      <div><h3 className="text-sm font-semibold text-slate-800">历史计算契约</h3><p className="mt-1 text-xs text-slate-500">按默认参数从完整计算图自动推算；使用页面修改参数后，系统会重新计算历史需求。</p></div>
      <dl className="mt-3 grid gap-3 sm:grid-cols-3">
        <div className="rounded-lg bg-white p-3"><dt className="text-xs font-semibold text-slate-500">历史策略</dt><dd className="mt-1 text-sm font-semibold text-slate-800">{historyPolicy ? historyPolicy === 'full_history' ? '完整历史递归' : '有限回看' : '校验后自动推断'}</dd></div>
        <div className="rounded-lg bg-white p-3"><dt className="text-xs font-semibold text-slate-500">计算前置观察数</dt><dd className="mt-1 text-sm font-semibold text-slate-800">{lookbackObservations ? `${lookbackObservations} 个` : '校验后自动推断'}</dd></div>
        <div className="rounded-lg bg-white p-3"><dt className="text-xs font-semibold text-slate-500">最少观察数</dt><dd className="mt-1 text-sm font-semibold text-slate-800">{minimumObservations ? `${minimumObservations} 个` : '校验后自动推断'}</dd></div>
      </dl>
      {fixedParameters.length > 0 && <div className="mt-3 rounded-lg border border-amber-100 bg-amber-50 p-3"><p className="text-xs font-semibold text-amber-800">已从旧定义固化的参数</p><p className="mt-1 text-xs text-amber-700">{fixedParameters.map((item) => `${item.label}=${item.value}`).join('，')}。这些数值已经写入公式，不再接受运行时覆盖。</p></div>}
    </section>
  </div>
}

function TimeSeriesFormulaEditor({
  draft,
  activeOutputId,
  measureOptions,
  inference,
  onSelectOutput,
  onPatchOutput,
  onAddOutput,
  onRemoveOutput,
}: {
  draft: IndicatorDraft
  activeOutputId: string
  measureOptions: SeriesOutputMeasureOption[]
  inference: SeriesOutputInference | null
  onSelectOutput: (outputId: string) => void
  onPatchOutput: (index: number, patch: Partial<SeriesOutputDefinition>) => void
  onAddOutput: () => void
  onRemoveOutput: (index: number) => void
}) {
  const outputs = draft.series_outputs ?? []
  const activeIndex = Math.max(0, outputs.findIndex((item) => item.id === activeOutputId))
  const output = outputs[activeIndex]
  const selectedMeasure = measureOptions.find((item) => item.id === output?.output_measure)
  const inferredMeasure = measureOptions.find((item) => item.id === inference?.inferred_output_measure)
  const resolvedMeasure = measureOptions.find((item) => item.id === inference?.resolved_output_measure)
  const rangeLabel = inference?.value_range
    ? `${inference.value_range[0] ?? '-∞'} ～ ${inference.value_range[1] ?? '+∞'}`
    : '未证明固定边界'
  return <div className="mt-4 space-y-3">
    <div className="rounded-xl border border-violet-100 bg-violet-50 px-3 py-2 text-xs leading-5 text-violet-900">每个输出通道只定义计算逻辑和结果口径，必须返回 <code>series&lt;time&gt;</code>。图表放在主价格图、成交量区还是独立副图，由使用该指标的页面决定，不写入指标定义。</div>
    <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label="时序输出通道">
      {outputs.map((item, index) => <button key={`${item.id}-${index}`} type="button" role="tab" aria-selected={index === activeIndex} onClick={() => onSelectOutput(item.id)} className={`rounded-lg border px-3 py-2 text-sm font-semibold focus:outline-none focus:ring-2 focus:ring-violet-300 ${index === activeIndex ? 'border-violet-500 bg-violet-600 text-white' : 'border-slate-200 bg-white text-slate-600 hover:border-violet-300'}`}>{item.label || item.id || `输出通道 ${index + 1}`}</button>)}
      <button type="button" onClick={onAddOutput} disabled={outputs.length >= 8} className="rounded-lg border border-dashed border-violet-300 bg-white px-3 py-2 text-sm font-semibold text-violet-700 hover:bg-violet-50 disabled:opacity-50">添加输出通道</button>
    </div>
    {output ? <fieldset className="rounded-xl border border-slate-200 bg-slate-50 p-4"><legend className="px-1 text-sm font-semibold text-slate-800">当前输出通道</legend><SeriesOutputFields output={output} index={activeIndex} onChange={patch => onPatchOutput(activeIndex, patch)}><label className="text-xs font-semibold text-slate-600">输出量纲<select aria-label={`输出通道 ${activeIndex + 1} 输出量纲`} value={output.output_measure} onChange={(event) => onPatchOutput(activeIndex, { output_measure: event.target.value as SeriesOutputMeasureId })} className="mt-1 block w-full rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm">{measureOptions.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}</select><span className="mt-1 block font-normal text-slate-400">{selectedMeasure?.description || '由系统根据公式推断。'}</span></label></SeriesOutputFields><div className="mt-3 rounded-lg border border-violet-100 bg-white p-3"><p className="text-xs font-semibold text-violet-800">系统推断</p>{inference ? <dl className="mt-2 grid gap-2 text-xs sm:grid-cols-2 lg:grid-cols-4"><div><dt className="text-slate-500">推断口径</dt><dd className="mt-0.5 font-semibold text-slate-800">{inferredMeasure?.label || inference.inferred_output_measure || '其他无量纲值'}</dd></div><div><dt className="text-slate-500">最终口径</dt><dd className="mt-0.5 font-semibold text-slate-800">{resolvedMeasure?.label || inference.resolved_output_measure || selectedMeasure?.label || '待定'}</dd></div><div><dt className="text-slate-500">底层语义 / 价格基准</dt><dd className="mt-0.5 font-semibold text-slate-800">{inference.semantic_dimension || 'dimensionless'} / {inference.price_basis || '无'}</dd></div><div><dt className="text-slate-500">可证明范围</dt><dd className="mt-0.5 font-semibold text-slate-800">{rangeLabel}</dd></div></dl> : <p className="mt-1 text-xs text-slate-500">解析并校验全部通道后显示量纲、价格基准和可证明范围。</p>}</div><div className="mt-3 flex justify-end"><button type="button" onClick={() => onRemoveOutput(activeIndex)} disabled={outputs.length <= 1} className="rounded-lg border border-rose-100 bg-white px-3 py-2 text-xs font-semibold text-rose-600 hover:bg-rose-50 disabled:cursor-not-allowed disabled:opacity-40">移除当前通道</button></div></fieldset> : <p className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">请先添加一个输出通道。</p>}
  </div>
}

function InsertionPalette({ title, items, fallback, onInsert }: { title: string; items: { key: string; label: string; token: string }[]; fallback: { key: string; label: string; token: string }[]; onInsert: (token: string) => void }) {
  const list = items.length ? items : fallback
  return <div><p className="text-xs font-semibold text-slate-500">{title}</p><div className="mt-2 flex max-h-24 flex-wrap gap-2 overflow-auto">{list.map((item) => <button key={item.key} type="button" title={`插入 ${item.token}`} onClick={() => onInsert(item.token)} className="rounded-md border border-slate-200 bg-white px-2 py-1 text-xs text-slate-700 hover:border-violet-300 hover:bg-violet-50 focus:outline-none focus:ring-2 focus:ring-violet-200">{item.label}</button>)}</div></div>
}

function ValidationPanel({ validation, resourceLabels }: { validation: ValidationResponse; resourceLabels: Map<string, string> }) {
  return <div className={`mt-4 rounded-xl border p-4 ${validation.valid ? 'border-emerald-200 bg-emerald-50' : 'border-rose-200 bg-rose-50'}`} role={validation.valid ? 'status' : 'alert'}><p className={`font-semibold ${validation.valid ? 'text-emerald-800' : 'text-rose-800'}`}>{validation.valid ? '校验通过' : '校验未通过'}</p>{validation.dependencies.length > 0 && <p className="mt-1 text-xs text-slate-600">所需数据：{validation.dependencies.map((dependency) => resourceLabels.get(dependency) || '公式所需变量').join('、')}</p>}<ul className="mt-2 space-y-2 text-sm text-slate-700">{validation.diagnostics.map((item, index) => <li key={`${item.code}-${index}`}><span>{diagnosticMessage(item.code, item.message)}</span>{(item.node_id !== undefined || item.expected !== undefined || item.actual !== undefined) && <span className="mt-0.5 block text-xs text-slate-500">{item.node_id !== undefined ? '问题位于公式中的一个计算步骤' : '数据要求'}{item.expected !== undefined ? ` · 需要 ${humanizeTechnicalTypes(item.expected)}` : ''}{item.actual !== undefined ? ` · 当前是 ${humanizeTechnicalTypes(item.actual)}` : ''}</span>}</li>)}</ul></div>
}

function formatTimeSeriesChannelValue(
  value: number | null | undefined,
  channel: TimeSeriesIndicatorResult['channels'][number],
) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '—'
  const scaled = channel.display_format === 'percent' ? value * 100 : value
  const formatted = scaled.toFixed(channel.precision)
  const suffix = channel.display_format === 'percent'
    ? '%'
    : channel.unit
      ? ` ${channel.unit}`
      : ''
  return `${formatted}${suffix}`
}

function TimeSeriesResultsPanel({ results }: { results: TimeSeriesIndicatorResult[] }) {
  if (!results.length) return <div className="rounded-2xl border border-dashed border-slate-200 bg-white p-5 text-sm text-slate-500">选择产品并预览后显示具名时序通道。</div>
  const computedCount = results.filter((result) => result.status === 'ok' || result.status === 'warning').length
  return <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="series-preview-results-title"><div className="flex flex-wrap items-center justify-between gap-2"><div><h2 id="series-preview-results-title" className="font-semibold text-slate-900">时序结果预览</h2><p className="mt-1 text-xs text-slate-500">日期轴与通道值按日期显式对齐；缺失点保留为空，不补零。</p></div><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">可计算 {computedCount}/{results.length} 个产品</span></div><div className="mt-4 space-y-5">{results.map((result, resultIndex) => {
    const option = result.dates.length && result.channels.length ? {
      animation: false,
      tooltip: { trigger: 'axis' },
      legend: { top: 0, data: result.channels.map((channel) => channel.label) },
      grid: { left: 48, right: 24, top: 44, bottom: 54 },
      xAxis: { type: 'category', boundaryGap: false, data: result.dates },
      yAxis: { type: 'value', scale: true },
      dataZoom: [
        { type: 'inside', start: Math.max(0, 100 - 252 / Math.max(result.dates.length, 1) * 100), end: 100 },
        { type: 'slider', height: 18, bottom: 10, start: Math.max(0, 100 - 252 / Math.max(result.dates.length, 1) * 100), end: 100 },
      ],
      series: result.channels.map((channel) => ({
        name: channel.label,
        type: 'line',
        showSymbol: false,
        connectNulls: false,
        data: channel.values,
      })),
    } : null
    const parameterText = Object.entries(result.parameters).map(([key, value]) => `${key}=${value}`).join('，')
    return <article key={`${result.target.kind}-${result.target.product_id}-${result.indicator_id ?? resultIndex}-${result.parameter_hash ?? resultIndex}`} className="rounded-xl border border-slate-200 bg-slate-50 p-4"><div className="flex flex-wrap items-start justify-between gap-3"><div><h3 className="font-semibold text-slate-800">{result.target.name}</h3><p className="mt-1 text-xs text-slate-500">{result.target.product_id} · {result.period} · {result.channels.length} 个通道</p></div><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${result.status === 'ok' ? 'bg-emerald-100 text-emerald-800' : result.status === 'warning' ? 'bg-amber-100 text-amber-800' : 'bg-rose-100 text-rose-800'}`}>{evaluationStatusLabel(result.status)}</span></div><p className="mt-2 text-xs text-slate-500">实际窗口 {result.window.start_date || '—'} 至 {result.window.end_date || '—'} · {result.window.observation_count} 个观察值{parameterText ? ` · ${parameterText}` : ''}</p>{option ? <ReactECharts option={option} style={{ height: 360 }} notMerge lazyUpdate aria-label={`${result.target.name}时序指标图`} /> : <p className="mt-3 rounded-lg border border-dashed border-slate-200 bg-white p-4 text-sm text-slate-500">没有可绘制的时序数据。</p>}{result.warnings.length > 0 && <ul className="mt-2 space-y-1 text-xs text-amber-800">{result.warnings.map((warning, index) => <li key={`${warning.code}-${index}`}>{diagnosticMessage(warning.code, warning.message)}</li>)}</ul>}<div className="mt-3 max-h-56 overflow-auto rounded-lg border border-slate-200 bg-white"><table className="w-full min-w-[520px] text-left text-xs"><caption className="sr-only">{result.target.name}时序指标数据表</caption><thead className="sticky top-0 bg-slate-50"><tr><th className="px-3 py-2">日期</th>{result.channels.map((channel) => <th key={channel.id} className="px-3 py-2">{channel.label}</th>)}</tr></thead><tbody>{result.dates.map((date, dateIndex) => <tr key={date} className="border-t border-slate-100"><td className="px-3 py-2 text-slate-500">{date}</td>{result.channels.map((channel) => <td key={channel.id} className="px-3 py-2 text-slate-700">{formatTimeSeriesChannelValue(channel.values[dateIndex], channel)}</td>)}</tr>)}</tbody></table></div></article>
  })}</div></section>
}

function ResultsPanel({ results, draft, contextDomain }: { results: EvaluationResult[]; draft: IndicatorDraft; contextDomain: IndicatorContextDomain }) {
  if (!results.length) return <div className="rounded-2xl border border-dashed border-slate-200 bg-white p-5 text-sm text-slate-500">选择产品并预览后显示真实净值计算结果。</div>
  const computedCount = results.filter(result => result.value !== null).length
  return <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby="preview-results-title">
    <div className="flex flex-wrap items-center justify-between gap-2"><h2 id="preview-results-title" className="font-semibold text-slate-900">结果预览</h2><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">有可用结果 {computedCount}/{results.length} 个产品</span></div>
    <div className="mt-3 space-y-3">{results.map((result, index) => <article key={`${result.target.kind}-${result.target.product_id}-${index}`} className="rounded-xl border border-slate-100 bg-slate-50 p-3">
      <div className="flex items-start justify-between gap-3"><div><p className="font-semibold text-slate-800">{result.target.name}</p><p className="text-xs text-slate-500">{result.target.product_id} · {contextDomain === 'portfolio' ? '快照窗口' : result.period}</p></div>{<strong className={result.value === null ? 'text-amber-700' : 'text-violet-700'}>{result.presentation ? <MetricValue value={result.value} presentation={result.presentation} /> : displayValue(result, draft)}</strong>}</div>
      {<>
        <p className="mt-2 text-xs text-slate-500">实际窗口 {result.window.start_date || '—'} 至 {result.window.end_date || '—'} · {result.window.observation_count} 个观测 · 数据最新 {result.window.data_latest_date || result.target_data?.data_latest_date || '—'}</p>
        <MetricUnavailableReason result={result} />
        {result.warnings.length > 0 && !result.input_requirements && <ul className="mt-2 space-y-1 text-xs text-amber-800">{result.warnings.map(warning => <li key={warning.code}>{diagnosticMessage(warning.code, warning.message)}</li>)}</ul>}
      </>}
    </article>)}</div>
    <div className="sr-only"><table><caption>指标结果数据表</caption><thead><tr><th>产品</th><th>结果</th><th>指标值</th><th>状态</th></tr></thead><tbody>{results.map((result, index) => <tr key={index}><td>{result.target.name}</td><td>{result.presentation?.name || result.indicator_name}</td><td>{result.presentation ? <MetricValue value={result.value} presentation={result.presentation} /> : displayValue(result, draft)}</td><td>{evaluationStatusLabel(result.status)}</td></tr>)}</tbody></table></div>
  </section>
}
