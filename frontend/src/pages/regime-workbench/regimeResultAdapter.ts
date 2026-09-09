import type { RegimeSeriesPage, RegimeEvaluationResults, RegimeEvaluationConditionalMetric } from '../../services/regimeGraph'

export interface RegimeResultState {
  id: string
  label: string
  color: string
  order: number
  role?: string
}

export interface RegimeResultInterval {
  id: string
  state_id: string
  label: string
  start_date: string
  end_date: string
  start_index: number
  end_index: number
  observations: number
  confirmed_at: string | null
  effective_start: string | null
  reason?: string
}

interface ResultCapability {
  available: boolean
  reason?: string
}

export interface RegimeResultOverview {
  schema_version: string
  run_kind: 'preview' | 'saved'
  run_id: string
  definition_id: string | null
  definition_revision: number | null
  definition_hash: string
  graph_hash: string
  mode: string
  as_of: string | null
  created_at?: string
  data_snapshots: Record<string, unknown>
  frequency: string | null
  calendar: string | null
  time_basis: 'observation'
  date_range: { start: string | null; end: string | null }
  complete: true
  states: RegimeResultState[]
  segments: RegimeResultInterval[]
  unknown_intervals: RegimeResultInterval[]
  summary: {
    total: number
    classified: number
    unknown: number
    state_counts: Record<string, number>
    switch_count: number
    denominator: 'all_observations'
  }
  primary_series: {
    source_kind: 'final_series'
    run_id: string
    node_id: null
    port: string
    value_column: 'value'
    date_column: 'observation_date'
    unit: string | null
    endpoint: string
    label: string
    total: number
  }
  evaluation_results?: RegimeEvaluationResults
  numeric_channels?: Record<string, { label: string; unit?: string; display_format?: 'number' | 'percent'; precision?: number }>
  capabilities: {
    observation: ResultCapability
    effective: ResultCapability
    probabilities: ResultCapability
    confidence: ResultCapability
    evidence: ResultCapability
  }
}

export interface RegimeResultPoint {
  observation_date: string
  state_id: string
  value: number | null
  confirmed_at: string | null
  effective_from: string | null
  confidence: number | null
  probabilities: Record<string, number | null>
  raw: Record<string, unknown>
}

export interface RegimeResultData {
  overview: RegimeResultOverview
  points: RegimeResultPoint[]
  intervals: RegimeResultInterval[]
}

export function regimeFeatureNumber(point: RegimeResultPoint | undefined, name: string): number | null {
  const features = point?.raw.features
  if (!isRecord(features)) return null
  const value = features[name]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

const invalid = (message: string): never => { throw new Error('情景结果不完整：' + message) }

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value)
}
function record(value: unknown, label: string): Record<string, unknown> {
  return isRecord(value) ? value : invalid(label)
}
function string(value: unknown, label: string): string {
  return typeof value === 'string' && value.length > 0 ? value : invalid(label)
}
function optionalString(value: unknown): string | null {
  return typeof value === 'string' && value.length > 0 ? value : null
}
function count(value: unknown, label: string): number {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : invalid(label)
}
function nullableNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}
function list(value: unknown, label: string): unknown[] {
  return Array.isArray(value) ? value : invalid(label)
}
function capability(value: unknown): ResultCapability {
  const source = record(value, '缺少展示能力信息')
  return { available: source.available === true, ...(typeof source.reason === 'string' ? { reason: source.reason } : {}) }
}
function evaluationResults(value: unknown): RegimeEvaluationResults | undefined {
  if (value == null) return undefined
  return Object.fromEntries(Object.entries(record(value, '评估结果格式错误')).map(([id, value]) => {
    const target = record(value, '评估对象格式错误')
    const metrics = target.conditional_metrics == null ? undefined : list(target.conditional_metrics, '条件表现格式错误').map(value => {
      const metric = record(value, '条件表现行格式错误')
      const row: RegimeEvaluationConditionalMetric = { state_id: string(metric.state_id, '条件表现缺少状态'), state_label: string(metric.state_label, '条件表现缺少状态名称') }
      const numericKeys = ['observations', 'return_observations', 'mean_period_return', 'return', 'annualized_return', 'volatility', 'max_drawdown', 'sharpe', 'positive_rate', 'win_rate'] as const
      for (const key of numericKeys) if (key in metric) row[key] = nullableNumber(metric[key])
      if (typeof metric.return_alignment === 'string') row.return_alignment = metric.return_alignment
      return row
    })
    return [id, {
      id: string(target.id ?? id, '评估对象缺少身份'), name: string(target.name, '评估对象缺少名称'), primary: target.primary === true,
      ...(target.source == null ? {} : { source: record(target.source, '评估数据源格式错误') }),
      ...(target.snapshot == null ? {} : { snapshot: record(target.snapshot, '评估快照格式错误') }),
      ...(metrics == null ? {} : { conditional_metrics: metrics }),
    }]
  }))
}

function parseInterval(value: unknown, unknownInterval: boolean): RegimeResultInterval {
  const row = record(value, '区间格式错误')
  const start = count(row.start_index, '区间缺少起点序号')
  const end = count(row.end_index, '区间缺少终点序号')
  const observations = count(row.observations, '区间缺少样本数')
  if (end < start || observations !== end - start + 1) return invalid('区间长度与样本数不一致')
  return {
    id: string(row.id, '区间缺少身份'),
    state_id: unknownInterval ? 'unclassified' : string(row.state_id, '区间缺少状态'),
    label: string(row.label, '区间缺少名称'),
    start_date: string(row.start_date, '区间缺少起始日期'),
    end_date: string(row.end_date, '区间缺少结束日期'),
    start_index: start, end_index: end, observations,
    confirmed_at: optionalString(row.confirmed_at),
    effective_start: optionalString(row.effective_start),
    ...(typeof row.reason === 'string' ? { reason: row.reason } : {}),
  }
}

/** This boundary accepts server snapshots only; it never reads the editable graph. */
export function adaptRegimeOverview(value: unknown, runId: string, runKind: 'preview' | 'saved'): RegimeResultOverview {
  const data = record(value, '缺少结果总览')
  if (data.run_id !== runId || data.run_kind !== runKind) return invalid('运行身份与请求不一致')
  if (data.complete !== true || data.time_basis !== 'observation') return invalid('时间轴尚未完整提供')
  const summary = record(data.summary, '缺少全样本统计')
  const total = count(summary.total, '缺少全样本数量')
  const classified = count(summary.classified, '缺少已分类数量')
  const unknown = count(summary.unknown, '缺少未分类数量')
  if (classified + unknown !== total || summary.denominator !== 'all_observations') return invalid('统计分母不一致')
  const stateCounts = Object.fromEntries(Object.entries(record(summary.state_counts, '缺少状态数量')).map(([id, n]) => [id, count(n, '状态数量错误')]))
  const states = list(data.states, '缺少冻结状态字典').map(value => {
    const state = record(value, '状态字典格式错误')
    return {
      id: string(state.id, '状态缺少身份'), label: string(state.label, '状态缺少名称'),
      color: typeof state.color === 'string' && /^#(?:[0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i.test(state.color) ? state.color : '#64748b',
      order: typeof state.order === 'number' && Number.isFinite(state.order) ? state.order : 0,
      ...(typeof state.role === 'string' ? { role: state.role } : {}),
    }
  }).sort((a, b) => a.order - b.order)
  if (new Set(states.map(state => state.id)).size !== states.length) return invalid('状态身份重复')
  if (Object.values(stateCounts).reduce((sum, n) => sum + n, 0) !== classified) return invalid('各状态数量不等于已分类数量')
  const series = record(data.primary_series, '缺少主对照走势')
  if (series.source_kind !== 'final_series' || series.run_id !== runId || series.value_column !== 'value' || series.date_column !== 'observation_date' || series.node_id != null) return invalid('主对照走势引用错误')
  if (series.total !== total) return invalid('走势数量与总览不一致')
  const range = record(data.date_range, '缺少日期范围')
  const caps = record(data.capabilities, '缺少能力信息')
  const segments = list(data.segments, '缺少完整情景区间').map(value => parseInterval(value, false))
  const unknownIntervals = list(data.unknown_intervals, '缺少未分类区间').map(value => parseInterval(value, true))
  const allIntervals = [...segments, ...unknownIntervals].sort((a, b) => a.start_index - b.start_index)
  let expected = 0
  for (const interval of allIntervals) {
    if (interval.start_index !== expected || interval.end_index >= total) return invalid('情景区间有缺口或重叠')
    if (interval.state_id !== 'unclassified' && !states.some(state => state.id === interval.state_id)) return invalid('区间引用了未冻结的状态')
    expected = interval.end_index + 1
  }
  if (expected !== total) return invalid('情景区间未覆盖全样本')
  const intervalCounts: Record<string, number> = {}
  for (const interval of segments) intervalCounts[interval.state_id] = (intervalCounts[interval.state_id] ?? 0) + interval.observations
  if (states.some(state => (intervalCounts[state.id] ?? 0) !== (stateCounts[state.id] ?? 0)) || unknownIntervals.reduce((sum, interval) => sum + interval.observations, 0) !== unknown) return invalid('区间覆盖与统计摘要不一致')
  return {
    schema_version: string(data.schema_version, '缺少结果版本'), run_kind: runKind, run_id: runId,
    definition_id: optionalString(data.definition_id),
    definition_revision: (data.definition_revision ?? data.revision) == null ? null : count(data.definition_revision ?? data.revision, '定义修订号错误'),
    definition_hash: string(data.definition_hash, '缺少定义快照'),
    graph_hash: string(data.graph_hash, '缺少计算图快照'),
    mode: string(data.mode, '缺少识别模式'), as_of: optionalString(data.as_of),
    ...(typeof data.created_at === 'string' ? { created_at: data.created_at } : {}),
    data_snapshots: record(data.data_snapshots ?? {}, '数据快照格式错误'),
    frequency: optionalString(data.frequency), calendar: optionalString(data.calendar), time_basis: 'observation',
    date_range: { start: optionalString(range.start), end: optionalString(range.end) }, complete: true,
    states, segments, unknown_intervals: unknownIntervals,
    summary: { total, classified, unknown, state_counts: stateCounts, switch_count: count(summary.switch_count, '缺少切换次数'), denominator: 'all_observations' },
    primary_series: {
      source_kind: 'final_series', run_id: runId, node_id: null,
      port: string(series.port, '缺少主序列端口'), value_column: 'value', date_column: 'observation_date',
      unit: optionalString(series.unit), endpoint: string(series.endpoint, '缺少主序列读取地址'),
      label: string(series.label, '缺少主对照名称'), total,
    },
    evaluation_results: evaluationResults(data.evaluation_results),
    numeric_channels: Object.fromEntries(Object.entries(record(data.numeric_channels ?? {}, '数值通道设置错误')).map(([id, value]) => {
      const channel = record(value, '数值通道格式错误')
      return [id, { label: optionalString(channel.label) || id, unit: optionalString(channel.unit) || '',
        display_format: channel.display_format === 'percent' ? 'percent' as const : 'number' as const,
        precision: typeof channel.precision === 'number' && Number.isInteger(channel.precision) && channel.precision >= 0 && channel.precision <= 8 ? channel.precision : 4 }]
    })),
    capabilities: {
      observation: capability(caps.observation), effective: capability(caps.effective),
      probabilities: capability(caps.probabilities), confidence: capability(caps.confidence), evidence: capability(caps.evidence),
    },
  }
}

export function adaptRegimeFormalOverview(detail: unknown, runId: string): RegimeResultOverview {
  const run = record(detail, '缺少正式运行详情')
  if (run.id !== runId) return invalid('正式运行身份不一致')
  if (!run.overview) return invalid('此历史版本缺少完整展示快照，请重新运行；现有运行仍可在版本详情查看')
  return adaptRegimeOverview(run.overview, runId, 'saved')
}

export function adaptRegimeResult(overview: RegimeResultOverview, rows: unknown[]): RegimeResultData {
  if (rows.length !== overview.summary.total) return invalid('主序列尚未完整加载')
  const points = rows.map((value): RegimeResultPoint => {
    const row = record(value, '序列行格式错误')
    const probabilities = row.probabilities && typeof row.probabilities === 'object'
      ? Object.fromEntries(Object.entries(record(row.probabilities, '概率格式错误')).map(([id, p]) => [id, nullableNumber(p)]))
      : {}
    return {
      observation_date: string(row.observation_date, '序列缺少观测日期'),
      state_id: optionalString(row.state_id) ?? 'unclassified',
      value: nullableNumber(row.value), confirmed_at: optionalString(row.recognized_at ?? row.confirmed_at),
      effective_from: optionalString(row.effective_date ?? row.effective_from), confidence: nullableNumber(row.confidence),
      probabilities, raw: row,
    }
  })
  for (let index = 1; index < points.length; index += 1) {
    if (points[index].observation_date <= points[index - 1].observation_date) return invalid('观测日期重复或顺序错误')
  }
  const intervals = [...overview.segments, ...overview.unknown_intervals].sort((a, b) => a.start_index - b.start_index)
  for (const interval of intervals) {
    if (points[interval.start_index]?.observation_date !== interval.start_date || points[interval.end_index]?.observation_date !== interval.end_date) return invalid('区间日期与主序列不一致')
    for (let index = interval.start_index; index <= interval.end_index; index += 1) {
      if (points[index].state_id !== interval.state_id) return invalid('区间状态与主序列不一致')
    }
  }
  if (points.length && (points[0].observation_date !== overview.date_range.start || points[points.length - 1].observation_date !== overview.date_range.end)) return invalid('日期范围与主序列不一致')
  return { overview, points, intervals }
}

export async function loadCompleteRegimeSeries(
  overview: RegimeResultOverview,
  readPage: (offset: number, limit: number, signal: AbortSignal) => Promise<RegimeSeriesPage>,
  signal: AbortSignal,
): Promise<unknown[]> {
  const rows: unknown[] = []
  while (rows.length < overview.summary.total) {
    if (signal.aborted) throw new DOMException('已取消结果加载', 'AbortError')
    const page = await readPage(rows.length, 5000, signal)
    if (signal.aborted) throw new DOMException('已取消结果加载', 'AbortError')
    if (page.total !== overview.summary.total || page.offset !== rows.length || !Array.isArray(page.items) || !page.items.length || page.items.length > 5000) return invalid('主序列分页缺失或数量发生变化')
    if (page.run_id && page.run_id !== overview.run_id) return invalid('主序列来自其他运行')
    rows.push(...page.items)
    if (rows.length > overview.summary.total) return invalid('主序列返回了重复或超量数据')
  }
  return rows
}
