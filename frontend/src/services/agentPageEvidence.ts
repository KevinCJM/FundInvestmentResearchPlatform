import type { AgentPreview, PageEvidenceSnapshot } from './agent'
import type { EvaluationResult, IndicatorDraft, InputRequirements, TimeSeriesIndicatorResult } from './customIndicators'

/** Keep the envelope far below the server's 2 MiB transport bound; oversize sections state their omission. */
export const MAX_PAGE_EVIDENCE_CHARS = 512 * 1024
const SERIES_SAMPLE_POINTS = 3
const MAX_REQUIREMENT_ITEMS = 24
const MAX_RUNTIME_TARGETS = 10

type Target = { kind: string; product_id: string; name?: string }

export type FrozenPreviewRequest = {
  definition: Record<string, unknown>
  /** Overrides actually sent with this request; `null` means the request carried none. */
  parameters: Record<string, number> | null
  parameters_submitted: boolean
  targets: Target[]
  period: string
  as_of: string | null
  requested_at: string
  completed_at: string | null
}

/* ------------------------------------------------------------------------- *
 * Frozen research-page requests (product-research / product-compare / holding-diagnosis)
 *
 * Field names mirror `backend/agent/research_pages.py` exactly: the server
 * validates the `request` section as a strict model and rejects unknown fields.
 * The page freezes the request it actually submitted, never a re-read of the
 * mutable page; the `results` section stays a reference-level statement and
 * never carries raw rows, series or chart arrays.
 * ------------------------------------------------------------------------- */

export type FrozenResearchTarget = { kind: 'etf' | 'fund'; product_id: string }
export type FrozenIndicatorRef = {
  indicator_id: string
  /** Locked page revision; a later revision is a different definition, not a newer number for this request. */
  indicator_revision: number
  /** Each indicator keeps its own period; there is no single page period for the batch. */
  period: string
  parameters?: Record<string, number>
}
export type FrozenCondition = { field: string; operator: 'gte' | 'lte' | 'gt' | 'lt' | 'eq'; value: string }
export type FrozenRange = { start_date: string | null; end_date: string | null }
export type FrozenSelectionMode = 'current_page' | 'selected' | 'all_matching'

export type ProductResearchFrozenRequest = {
  kind: 'etf' | 'fund'
  q?: string
  fund_type?: string[]
  /** The page's `type` filter is the original API's `type`/`fund_category` alias. */
  fund_category?: string[]
  invest_type?: string[]
  market?: string[]
  status?: string[]
  management?: string[]
  custodian?: string[]
  qdii_type?: string[]
  page?: number
  page_size?: number
  sort_by?: string
  sort_dir?: 'asc' | 'desc'
  conditions?: FrozenCondition[]
  snapshot_metrics?: string[]
  as_of: string | null
  /** Only the explicit current analysis batch; the selection counts are claims, not evaluated rows. */
  targets: FrozenResearchTarget[]
  batch_offset?: number
  visible_count?: number
  selected_count?: number
  selection_mode?: FrozenSelectionMode
  excluded_ids?: string[]
  indicators?: FrozenIndicatorRef[]
  view_mode?: 'overview' | 'metrics'
}

export type ProductCompareFrozenRequest = {
  targets: Array<FrozenResearchTarget & { management_fee: number | null; custody_fee: number | null }>
  ranges: { performance: FrozenRange; risk: FrozenRange; efficiency: FrozenRange }
  rolling_window_days: number
  indicators?: FrozenIndicatorRef[]
  as_of: string | null
  metrics_as_of?: string | null
  source: 'actual' | 'demo'
}

export type HoldingDiagnosisFrozenRequest = {
  run_id: string
  indicators?: FrozenIndicatorRef[]
  /** Only when the user explicitly requested this exact scenario; otherwise null. */
  scenario?: FrozenRange | null
}

/** Reference-level copy of what the page displayed; the backend keeps it as the original page record. */
export type ResearchDisplayReference = { source: string; refs: Record<string, unknown> }

export type PageResultRecord = {
  key: string
  request: Record<string, unknown>
  indicators?: FrozenIndicatorRef[]
  resolved_indicators?: FrozenIndicatorRef[]
  completed: boolean
}

/** Preserve the exact request separately from the last result, with no numeric observations. */
export function pageResultReference(record: PageResultRecord | null, key: string, loading: boolean, error: unknown) {
  return {
    status: loading ? 'loading' : error ? 'error' : !record ? 'not_requested' : record.key !== key ? 'stale' : record.completed ? 'ready' : 'pending',
    frozen_request: record?.request ?? null,
    indicators: record?.indicators ?? [],
    resolved_indicators: record?.resolved_indicators ?? [],
  }
}

const DISPLAY_REFERENCE_NOTE = '页面完整结果由原业务接口加载；这里只登记显示来源与引用，不包含原始数值、完整数组或图表曲线，数值真假以服务端核验为准。'

/** Stable short id for a page instance key; URL identifiers can exceed the server's 100-char bound. */
export function shortStableId(value: string): string {
  const round = (input: string) => {
    let hash = 5381
    for (let index = 0; index < input.length; index += 1) hash = ((hash * 33) ^ input.charCodeAt(index)) >>> 0
    return hash.toString(16).padStart(8, '0')
  }
  return `${round(value)}${round(`page-instance:${value}`)}`
}

/** Freeze the page's own indicator selection: id + locked revision + that indicator's period. */export function frozenIndicatorRefs(
  selectedIds: string[],
  periodsByIndicator: Record<string, string | undefined>,
  definitions: Array<{ id: string; revision: number }>,
  fallbackPeriod: string,
  max = 10,
): FrozenIndicatorRef[] {
  return selectedIds.flatMap(indicatorId => {
    const definition = definitions.find(item => item.id === indicatorId)
    return definition
      ? [{ indicator_id: indicatorId, indicator_revision: definition.revision, period: periodsByIndicator[indicatorId] || fallbackPeriod }]
      : []
  }).slice(0, max)
}

function researchSnapshot(page: 'product-research' | 'product-compare' | 'holding-diagnosis',
  request: Record<string, unknown>, displayed: ResearchDisplayReference): PageEvidenceSnapshot {
  return {
    version: 1,
    snapshot_id: newPageSnapshotId(),
    captured_at: new Date().toISOString(),
    page,
    sections: {
      request,
      results: { source: 'unverified_client_display', displayed_source: displayed.source, refs: displayed.refs, note: DISPLAY_REFERENCE_NOTE },
    },
  }
}

export function buildProductResearchEvidence(input: { request: ProductResearchFrozenRequest; displayed: ResearchDisplayReference }): PageEvidenceSnapshot {
  return researchSnapshot('product-research', input.request as unknown as Record<string, unknown>, input.displayed)
}

export function buildProductCompareEvidence(input: { request: ProductCompareFrozenRequest; displayed: ResearchDisplayReference }): PageEvidenceSnapshot {
  return researchSnapshot('product-compare', input.request as unknown as Record<string, unknown>, input.displayed)
}

export function buildHoldingDiagnosisEvidence(input: { request: HoldingDiagnosisFrozenRequest; displayed: ResearchDisplayReference }): PageEvidenceSnapshot {
  return researchSnapshot('holding-diagnosis', input.request as unknown as Record<string, unknown>, input.displayed)
}

export type IndicatorStudioEvidenceInput = {
  selection: { indicator_id: string | null; indicator_revision: number | null; name?: string; read_only?: boolean }
  /** The editor's own definition, always; never a preview definition relabelled as the editor's. */
  draft: IndicatorDraft
  /**
   * The definition that currently governs the page's controls and calculation, when it is an
   * adopted AI preview (it survives clearing the displayed result). Null means the editor governs.
   */
  active_preview_definition: Record<string, unknown> | null
  definition_dirty: boolean
  validation: { valid: boolean; diagnostics: Array<{ code?: string; message?: string }> } | null
  canvas_pending: boolean
  parameter_pending: boolean
  previewing: boolean
  targets: Target[]
  period: string
  as_of: string
  runtime_parameters: Record<string, number>
  manual_results: EvaluationResult[]
  manual_series_results: TimeSeriesIndicatorResult[]
  manual_request: FrozenPreviewRequest | null
  agent_preview: AgentPreview | null
}

export function newPageSnapshotId() {
  const uuid = globalThis.crypto?.randomUUID?.()
  const raw = uuid ? uuid.replace(/-/g, '') : Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('')
  return `snap-${raw.slice(0, 32).padEnd(32, '0')}`
}

/** Only overrides the request actually carries may be reported as submitted. */
export const submittedParameterOverrides = (
  definition: { parameter_contract_version?: string | null },
  runtimeParameters: Record<string, number>,
): Record<string, number> | null => definition.parameter_contract_version === '1.0' ? { ...runtimeParameters } : null

/**
 * Freeze one message's page copy. A live page object must never travel with an outgoing message,
 * so a failed clone degrades to an explicit omission instead of a shared mutable reference.
 */
export function freezePageSnapshot(snapshot: PageEvidenceSnapshot): PageEvidenceSnapshot | undefined {
  try { return structuredClone(snapshot) } catch { /* fall through to a JSON-safe copy */ }
  try { return JSON.parse(JSON.stringify(snapshot)) as PageEvidenceSnapshot } catch { /* not JSON-safe either */ }
  if (snapshot.page !== 'indicator-studio') return undefined
  return {
    version: 1, snapshot_id: newPageSnapshotId(), captured_at: new Date().toISOString(), page: 'indicator-studio',
    sections: { editing: { omitted: { code: 'page_evidence_freeze_failed', message: '页面证据无法安全冻结，本条消息未携带页面快照。' } } },
  }
}

const plain = (value: unknown): Record<string, unknown> => JSON.parse(JSON.stringify(value ?? {})) as Record<string, unknown>

const targetOf = (target: { kind?: unknown; product_id?: unknown; name?: unknown } | undefined) => ({
  kind: typeof target?.kind === 'string' ? target.kind : null,
  product_id: typeof target?.product_id === 'string' ? target.product_id : null,
  name: typeof target?.name === 'string' ? target.name : null,
})

const warningsOf = (warnings: Array<{ code?: string; message?: string }> | undefined) =>
  (warnings || []).slice(0, 20).map(warning => ({ code: warning.code || null, message: warning.message || null }))

const windowOf = (window: EvaluationResult['window'] | TimeSeriesIndicatorResult['window']) => ({
  requested_as_of: window?.requested_as_of ?? null,
  effective_as_of: window?.effective_as_of ?? null,
  start_date: window?.start_date ?? null,
  end_date: window?.end_date ?? null,
  observation_count: window?.observation_count ?? null,
  data_latest_date: window?.data_latest_date ?? null,
})

function requirementsOf(requirements: InputRequirements | null | undefined) {
  if (!requirements) return null
  const items = requirements.items || []
  return {
    status: requirements.status,
    required_count: requirements.required_count,
    available_count: requirements.available_count,
    reason: requirements.reason ? { code: requirements.reason.code || null, message: requirements.reason.message || null } : null,
    items: items.slice(0, MAX_REQUIREMENT_ITEMS).map(item => ({
      variable_id: item.variable_id,
      status: item.status,
      reason_code: item.reason_code ?? null,
      reason: item.reason ?? null,
      source_dataset: item.source_dataset ?? null,
    })),
    omitted_items: Math.max(0, items.length - MAX_REQUIREMENT_ITEMS),
  }
}

function sampledSeries(dates: string[], values: Array<number | null>) {
  const size = values.length <= SERIES_SAMPLE_POINTS * 2 ? values.length : SERIES_SAMPLE_POINTS
  const point = (index: number) => ({ date: dates[index] ?? null, value: values[index] ?? null })
  const head = Array.from({ length: size }, (_, index) => point(index))
  const tail = values.length <= SERIES_SAMPLE_POINTS * 2 ? [] : Array.from({ length: size }, (_, index) => point(values.length - size + index))
  return { head, tail, omitted_values: Math.max(0, values.length - head.length - tail.length) }
}

function scalarGroup(result: EvaluationResult) {
  const series = result.series || []
  return {
    target: targetOf(result.target),
    status: result.status,
    value: result.value ?? null,
    value_type: result.value_type ?? null,
    unit: result.presentation?.unit ?? null,
    precision: result.presentation?.precision ?? null,
    parameters: result.parameters ?? null,
    parameter_hash: result.parameter_hash ?? null,
    window: windowOf(result.window),
    data_context: result.data_context ?? null,
    target_data: result.target_data ?? null,
    input_requirements: requirementsOf(result.input_requirements),
    warnings: warningsOf(result.warnings),
    series_available: series.length > 0,
    series: series.length ? sampledSeries(series.map(item => item.date), series.map(item => item.value)) : null,
  }
}

function seriesGroup(result: TimeSeriesIndicatorResult) {
  const dates = result.dates || []
  return {
    target: targetOf(result.target),
    status: result.status,
    parameters: result.parameters ?? null,
    parameter_hash: result.parameter_hash ?? null,
    window: windowOf(result.window),
    data_context: result.data_context ?? null,
    observation_count: dates.length,
    date_range: { start: dates[0] ?? null, end: dates[dates.length - 1] ?? null },
    warnings: warningsOf(result.warnings),
    series_available: (result.channels || []).length > 0,
    channels: (result.channels || []).map(channel => {
      const values = channel.values || []
      const nullCount = typeof channel.null_count === 'number' ? channel.null_count : values.filter(value => value === null).length
      return {
        id: channel.id,
        label: channel.label,
        unit: channel.unit,
        output_measure: channel.output_measure ?? null,
        semantic_dimension: channel.semantic_dimension ?? null,
        price_basis: channel.price_basis ?? null,
        value_range: channel.value_range ?? null,
        point_count: values.length,
        null_count: nullCount,
        sample: sampledSeries(dates, values),
      }
    }),
  }
}

function fullSeriesGroup(result: TimeSeriesIndicatorResult) {
  return {
    target: targetOf(result.target),
    status: result.status,
    parameters: result.parameters ?? null,
    definition_hash: null,
    dates: result.dates || [],
    channels: (result.channels || []).map(channel => ({
      id: channel.id, label: channel.label, unit: channel.unit, output_measure: channel.output_measure ?? null,
      values: (channel.values || []).slice(),
    })),
  }
}

function fullScalarGroup(result: EvaluationResult) {
  const series = result.series || []
  return {
    target: targetOf(result.target),
    status: result.status,
    parameters: result.parameters ?? null,
    definition_hash: null,
    dates: series.map(point => point.date),
    channels: series.length ? [{
      id: 'value', label: result.presentation?.name || result.indicator_name, unit: result.presentation?.unit ?? null,
      output_measure: null, values: series.map(point => point.value),
    }] : [],
    ...(series.length ? {} : { unavailable: { code: 'series_not_on_page', message: '该标量结果未包含曲线数据（本次预览未请求序列）。' } }),
  }
}

function definitionOf(definition: IndicatorDraft | Record<string, unknown> | null | undefined) {
  return plain(definition)
}

function editingSection(input: IndicatorStudioEvidenceInput) {
  const preview = input.agent_preview
  const previewDefinition = input.active_preview_definition
  return {
    selection: {
      indicator_id: input.selection.indicator_id,
      indicator_revision: input.selection.indicator_revision,
      name: input.selection.name ?? null,
      read_only: input.selection.read_only ?? null,
    },
    // The editor definition and the adopted preview definition are different facts and stay separate.
    definition: { source: 'editor', definition_dirty: input.definition_dirty, ...definitionOf(input.draft) },
    // The active preview keeps governing the controls and their inputs even after its displayed
    // result was cleared by a parameter or period change; only the result provenance disappears.
    active_preview: previewDefinition ? {
      source: 'adopted_agent_preview',
      definition: definitionOf(previewDefinition),
      result_adopted: preview !== null,
      preview_id: preview?.preview_id ?? null,
      run_id: preview?.run_id ?? null,
      definition_hash: preview?.definition_hash ?? null,
      context_hash: preview?.context_hash ?? null,
      data_generation: preview?.data_generation ?? null,
      effective_context: preview?.effective_context ?? null,
      created_at: preview?.created_at ?? null,
      runtime_inputs: {
        targets: input.targets.slice(0, MAX_RUNTIME_TARGETS).map(targetOf),
        period: input.period || null,
        as_of: input.as_of || null,
        runtime_parameters: input.runtime_parameters,
      },
      note: preview
        ? '这是页面当前采纳的 AI 试算定义及其运行输入。'
        : '页面仍按这份 AI 试算定义和当前输入计算，但原试算结果已被后续修改清除，尚未重新计算。',
    } : null,
    runtime_inputs: {
      targets: input.targets.slice(0, MAX_RUNTIME_TARGETS).map(targetOf),
      period: input.period || null,
      as_of: input.as_of || null,
      runtime_parameters: input.runtime_parameters,
    },
    state: {
      canvas_pending: input.canvas_pending,
      parameter_pending: input.parameter_pending,
      previewing: input.previewing,
      validation_valid: input.validation ? input.validation.valid : null,
      diagnostics: (input.validation?.diagnostics || []).slice(0, 8).map(item => ({ code: item.code ?? null, message: item.message ?? null })),
    },
  }
}

function pendingReasons(input: IndicatorStudioEvidenceInput) {
  const reasons: string[] = []
  if (input.canvas_pending) reasons.push('画布修改尚未应用，编辑器中的定义不是已计算的定义。')
  if (input.parameter_pending) reasons.push('参数修改尚未应用。')
  if (input.previewing) reasons.push('页面正在计算中，结果尚未返回。')
  if (!reasons.length) reasons.push('页面当前没有已显示的计算结果（尚未预览，或结果已被后续修改清除）。')
  return reasons
}

function resultsSection(input: IndicatorStudioEvidenceInput) {
  if (input.agent_preview) {
    const preview = input.agent_preview
    const groups = preview.result_kind === 'time_series'
      ? (preview.result.results as TimeSeriesIndicatorResult[]).map(seriesGroup)
      : (preview.result.results as EvaluationResult[]).map(scalarGroup)
    return {
      displayed_source: 'agent_preview',
      pending: [] as string[],
      provenance: {
        source: 'agent_preview',
        preview_id: preview.preview_id,
        run_id: preview.run_id ?? null,
        definition_hash: preview.definition_hash,
        context_hash: preview.context_hash ?? null,
        data_generation: preview.data_generation ?? null,
        effective_context: preview.effective_context ?? null,
        created_at: preview.created_at ?? null,
        note: '这是 AI 试算并被页面采纳的结果，不是编辑器当前定义的自动重算。',
      },
      frozen_definition: definitionOf(preview.definition),
      frozen_request: {
        targets: [targetOf(preview.result.results?.[0]?.target ?? preview.target)],
        period: preview.period,
        as_of: preview.as_of ?? null,
      },
      groups,
    }
  }
  const manualSeries = input.manual_series_results || []
  const manualScalar = input.manual_results || []
  if (manualSeries.length || manualScalar.length) {
    const groups = manualSeries.length ? manualSeries.map(seriesGroup) : manualScalar.map(scalarGroup)
    const request = input.manual_request
    return {
      displayed_source: 'manual_preview',
      pending: [] as string[],
      provenance: {
        source: 'manual_preview',
        requested_at: request?.requested_at ?? null,
        completed_at: request?.completed_at ?? null,
        note: '这是页面预览按钮发起的请求结果，使用该请求冻结的定义与参数；服务器实际生效的参数在各分组 parameters 中。',
      },
      frozen_request: request ? {
        definition: request.definition,
        parameters: request.parameters,
        parameters_submitted: request.parameters_submitted,
        targets: request.targets.slice(0, MAX_RUNTIME_TARGETS).map(targetOf),
        period: request.period,
        as_of: request.as_of,
        requested_at: request.requested_at,
        completed_at: request.completed_at,
      } : null,
      groups,
    }
  }
  return { displayed_source: null, pending: pendingReasons(input), provenance: null, frozen_request: null, groups: [] as unknown[] }
}

function seriesSection(input: IndicatorStudioEvidenceInput) {
  const note = '这是页面实际持有的完整数组，用于查看 results 摘要省略的中间点；不包含页面上没有的数据。'
  if (input.agent_preview) {
    const preview = input.agent_preview
    const groups = preview.result_kind === 'time_series'
      ? (preview.result.results as TimeSeriesIndicatorResult[]).map(fullSeriesGroup)
      : (preview.result.results as EvaluationResult[]).map(fullScalarGroup)
    return { displayed_source: 'agent_preview', groups, note }
  }
  if (input.manual_series_results.length) return { displayed_source: 'manual_preview', groups: input.manual_series_results.map(fullSeriesGroup), note }
  if (input.manual_results.length) return { displayed_source: 'manual_preview', groups: input.manual_results.map(fullScalarGroup), note }
  return { displayed_source: null, groups: [] as unknown[], note: '页面当前没有已显示的结果；没有可读的时序数据。' }
}

function seriesStats(section: Record<string, unknown>) {
  const groups = Array.isArray(section.groups) ? section.groups as Array<Record<string, unknown>> : []
  let points = 0, channels = 0
  for (const group of groups) {
    const rows = Array.isArray(group.channels) ? group.channels as Array<Record<string, unknown>> : []
    channels += rows.length
    for (const channel of rows) points += Array.isArray(channel.values) ? channel.values.length : 0
  }
  return { groups: groups.length, channels, points }
}

export function buildIndicatorStudioEvidence(input: IndicatorStudioEvidenceInput): PageEvidenceSnapshot {
  const snapshot: PageEvidenceSnapshot = {
    version: 1,
    snapshot_id: newPageSnapshotId(),
    captured_at: new Date().toISOString(),
    page: 'indicator-studio',
    sections: { editing: editingSection(input), results: resultsSection(input), series: seriesSection(input) },
  }
  const size = (value: unknown) => JSON.stringify(value).length
  if (size(snapshot) <= MAX_PAGE_EVIDENCE_CHARS) return snapshot
  // Extreme page data must never be silently discarded or silently delivered: each dropped section states why.
  const oversized = (name: string, originalChars: number, extra: Record<string, unknown> = {}) => ({
    omitted: { code: `${name === 'series' ? 'page_evidence_series_too_large' : 'page_evidence_too_large'}`, original_chars: originalChars, name, ...extra,
      message: `${name} 分区过大，未随消息提交；请让用户在页面上缩小结果范围后重发。` },
  })
  const series = snapshot.sections.series as Record<string, unknown>
  const originalSeries = size(series)
  snapshot.sections.series = oversized('series', originalSeries, seriesStats(series))
  if (size(snapshot) <= MAX_PAGE_EVIDENCE_CHARS) return snapshot
  snapshot.sections.results = oversized('results', size(snapshot.sections.results))
  if (size(snapshot) <= MAX_PAGE_EVIDENCE_CHARS) return snapshot
  snapshot.sections.editing = oversized('editing', size(snapshot.sections.editing))
  return snapshot
}
