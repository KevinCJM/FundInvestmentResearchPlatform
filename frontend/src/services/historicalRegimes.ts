import {
  assertCompliantExecutionGraph,
  assertFixedNjitExecution,
  type FixedNjitExecutionAudit,
} from '../utils/fixedNjitExecution'

export type RegimeMode = 'realtime' | 'retrospective'
export type AlgorithmFamily =
  | 'causal_filter'
  | 'turning_point'
  | 'hmm'
  | 'markov'
  | 'gmm'
  | 'change_point'
  | 'merrill_clock'
  | 'relative_strength'
  | 'ensemble'

export type PublicationUsage = 'research_display' | 'product_research' | 'formal_backtest' | 'taa'

export interface RegimeStateDefinition {
  id: string
  label: string
  color: string
  description?: string
}

export interface HistoricalRegimeDefinition {
  id?: string
  revision?: number
  status?: 'draft' | 'validated' | 'published' | 'deprecated' | string
  name: string
  description: string
  template_id: string
	target: {
		kind: 'inline' | 'index' | 'relative' | 'indicator'
    series_id?: string
    name?: string
    frequency?: string
    source_api?: string
    ts_code?: string
    code?: string
    field?: string
    start_date?: string
    end_date?: string
    calendar?: string
    price_basis?: string
    availability_mode?: 'point_in_time' | 'latest'
    rows?: Array<Record<string, unknown>>
    points?: Array<Record<string, unknown>>
    series?: Array<Record<string, unknown>>
    numerator?: Record<string, unknown>
    denominator?: Record<string, unknown>
		transform?: 'ratio' | 'log_ratio'
		indicator_id?: string
		indicator_revision?: number
		indicator_name?: string
		product_kind?: 'etf' | 'fund'
		product_id?: string
		period?: string
		indicator_dsl_version?: string
		indicator_compiled_plan_id?: string
  }
  data?: Record<string, unknown>
  features: Record<string, number | string | boolean | null>
  algorithm: {
    family: AlgorithmFamily
    parameters: Record<string, number | string | boolean | null | Record<string, unknown> | unknown[]>
    params?: Record<string, number | string | boolean | null | Record<string, unknown> | unknown[]>
  }
  states: RegimeStateDefinition[]
  validation: {
    train_start?: string
    train_end?: string
    test_start?: string
    test_end?: string
    walk_forward?: boolean
    sensitivity_pct?: number
    minimum_segment?: number
    folds?: number
    stability_perturbation?: number
  }
  usage_intent?: PublicationUsage
  template?: string
  application_bindings?: RegimeApplicationBinding[]
  created_at?: string
  updated_at?: string
}

export interface RegimeTemplate {
  id: string
  name: string
  description: string
  category?: string
  definition?: HistoricalRegimeDefinition
  default_definition?: HistoricalRegimeDefinition
  [key: string]: unknown
}

export interface RegimeDataSource {
  id: string
  name?: string
  label?: string
  frequency?: string
  description?: string
  point_in_time?: boolean
}

export interface RegimeIndicatorCatalogItem {
	id: string
	revision: number
	name: string
	description?: string
	source?: string
	dsl_version: string
	operator_registry_version?: string
	compiled_plan_id?: string
	applicable_product_kinds: Array<'etf' | 'fund'>
	periods: string[]
	execution_backend: 'numba_njit_fixed_signature' | string
	njit_required: true
}

export interface RegimeIndicatorPeriod {
	value: string
	label: string
	description?: string
	kind?: string
	group?: string
}

export interface RegimeFeatureOption {
  id: string
  name?: string
  label?: string
  description?: string
  source?: string
  transforms?: string[]
  causal?: boolean
  repaints?: boolean
}

export interface RegimeAlgorithmOption {
  id?: AlgorithmFamily
  family?: AlgorithmFamily
  name?: string
  label?: string
  description?: string
  causal?: boolean
  supports_realtime?: boolean
  njit_supported?: boolean
  execution_backend?: string
  python_fallback?: number
  parameters?: Array<{ key: string; label: string; type?: 'number' | 'select' | 'boolean'; default?: number | string | boolean; min?: number; max?: number; step?: number; options?: string[]; unit?: string }>
}

export interface RegimeApplicationTarget {
  id: PublicationUsage
  name?: string
  label?: string
  description?: string
}

export interface RegimeFormulaLanguage {
  id?: string
  allowlist_version?: string
  evaluator_version?: string
  dsl_version?: string
  operator_registry_version?: string
  numeric_kernel_version?: string
  engine_version?: string
  execution_backend?: string
  njit_required?: boolean
  python_fallback?: number
  python_operator_calls?: number
  operators?: string[]
  operator_catalog?: Array<{ id: string; njit_supported: boolean; execution_backend?: string }>
  functions?: Array<string | { id?: string; name?: string; signature?: string; causal?: boolean; njit_supported?: boolean; execution_backend?: string; kernel_version?: string }>
  variable_rule?: string
  limits?: Record<string, number>
}

export interface HistoricalRegimeMeta {
  schema_version?: string
  modes?: Array<{ id: RegimeMode; label: string; description?: string }>
  templates: RegimeTemplate[]
  data_sources: RegimeDataSource[]
  feature_catalog: RegimeFeatureOption[]
  algorithm_families: RegimeAlgorithmOption[]
  application_targets: RegimeApplicationTarget[]
  causality_classes?: unknown[]
  limits?: Record<string, number>
	formula_language?: RegimeFormulaLanguage
	indicator_catalog?: RegimeIndicatorCatalogItem[]
	indicator_periods?: RegimeIndicatorPeriod[]
	historical_regime_runtime?: {
		complete?: boolean
		engine_version?: string
		kernel_version?: string
		python_fallback?: number
		kernels?: RegimeNjitKernelAudit[]
	}
}

export interface RegimeNjitKernelAudit {
	kernel_id: string
	compiled_plan_id?: string
	compile_status?: string
	compiled_signatures?: string[]
	kernel_fingerprint?: string
	kernel_version?: string
	engine_version?: string
	execution_backend?: string
	njit_required?: boolean
	python_fallback?: number
	python_operator_calls?: number
}

export interface RegimeCalculationDag {
	nodes?: Array<Record<string, unknown>>
	edges?: Array<{ source: number | string; target: number | string }>
	roots?: { result?: number | string }
	[key: string]: unknown
}

export interface RegimeCalculationAudit {
	source_kind?: 'formula' | 'indicator' | string
	typed_ast?: {
		nodes?: Array<Record<string, unknown>>
		edges?: Array<{ source: number | string; target: number | string }>
		root?: number | string
		expression_hash?: string
		compiler_version?: string
	}
	dag?: RegimeCalculationDag
	plan?: {
		compiled_plan_id?: string
		compile_status?: string
		compiled_signatures?: string[]
		kernel_version?: string
		engine_version?: string
		njit_required?: boolean
		execution_backend?: string
		python_fallback?: number
		python_operator_calls?: number
		[key: string]: unknown
	}
	family?: string
	compiled_plan_id?: string
	compile_status?: string
	compiled_signatures?: string[]
	kernel_version?: string
	engine_version?: string
	execution_backend?: string
	njit_required?: boolean
	python_fallback?: number
	python_operator_calls?: number
	kernels?: RegimeNjitKernelAudit[]
	[key: string]: unknown
}

export interface RegimeSeriesPoint {
  observation_date?: string
  date: string
  effective_date?: string | null
  recognized_at?: string
  signal_date?: string
  value: number | null
  filtered_value?: number | null
  state_id: string
  state_label: string
  probabilities: Record<string, number>
  confidence: number | null
  features: Record<string, number | null>
  reasons: string[]
  is_final?: boolean
  executable?: boolean
  data_available_at?: string
}

export interface RegimeSegment {
  state_id: string
  state_label: string
  start_date: string
  end_date: string
  effective_start?: string
  recognized_at?: string
  duration_observations: number
  return: number | null
  annualized_return?: number | null
  volatility?: number | null
  max_drawdown?: number | null
  confidence: number | null
  reasons: string[]
}

export interface RegimeConditionalStat {
  state_id: string
  state_label: string
  observations?: number
  return?: number | null
  annualized_return?: number | null
  volatility?: number | null
  max_drawdown?: number | null
  sharpe?: number | null
  win_rate?: number | null
}

export interface RegimeCausality {
  classification: string
  is_causal: boolean
  uses_future_data: boolean
  repaints: boolean
  realtime_eligible: boolean
  publish_eligible_usages: PublicationUsage[]
  blockers: string[]
  warnings: string[]
}

export interface RegimeTransition {
  states: string[]
  counts: number[][]
  probabilities: number[][]
}

export interface RegimePublication {
  id: string
  usage: PublicationUsage
  published_at: string
  definition_revision: number
  run_id: string
  run_content_hash?: string
  gate?: string
  note?: string
}

export interface HistoricalRegimeEvaluationResult {
  id: string
  name: string
  primary?: boolean
  source?: Record<string, unknown>
  snapshot?: Record<string, unknown>
  conditional_metrics?: Array<Record<string, unknown>>
  artifact?: Record<string, unknown> | null
}

export interface HistoricalRegimeEvaluationArtifact {
  artifact_id?: string
  checksum?: string
  format?: string
  schema_version?: string
  size_bytes?: number
  arrays?: Array<Record<string, unknown>>
  uri?: string
  [key: string]: unknown
}

export interface HistoricalRegimeRun {
  id: string
  schema_version?: string
  definition_id: string | null
  definition_revision: number | null
  definition_source?: 'saved_version' | 'inline_trial' | string
  name: string
  mode: RegimeMode
  created_at: string
  immutable: boolean
  content_hash?: string
  definition_snapshot_hash?: string
  evaluation_results?: Record<string, HistoricalRegimeEvaluationResult>
  artifact_manifest?: {
    evaluation_targets?: HistoricalRegimeEvaluationArtifact
    [key: string]: unknown
  }
  data_snapshot?: Record<string, unknown>
  definition?: HistoricalRegimeDefinition
  algorithm?: HistoricalRegimeDefinition['algorithm']
  states: RegimeStateDefinition[]
  series: RegimeSeriesPoint[]
  segments: RegimeSegment[]
  conditional_stats: RegimeConditionalStat[]
  transition: RegimeTransition
  causality: RegimeCausality
  stability: Record<string, unknown>
  walk_forward: Record<string, unknown>
	governance?: {
		formal_gate_passed?: boolean
		publish_eligible_usages?: PublicationUsage[]
		publication_blockers?: string[]
	}
	diagnostics: Array<{ code?: string; message: string; level?: 'info' | 'warning' | 'error' | string; field?: string }>
	algorithm_diagnostics?: Record<string, unknown>
	formula_diagnostics?: Record<string, unknown> | null
	calculation_audit?: RegimeCalculationAudit | null
	calculation_audits?: RegimeCalculationAudit[]
  publications: RegimePublication[]
  application_bindings?: RegimeApplicationBinding[]
  /** List endpoints return lightweight snapshots; use getHistoricalRegimeRun for series/detail consumers. */
  series_included?: boolean
  series_detail_endpoint?: string
}

export interface RegimeComparison {
  run_ids: string[]
  reference_run_id?: string
  agreement_rate?: number | null
  disagreement_periods?: Array<{ start_date: string; end_date: string; states: Record<string, string> }>
  pairwise?: Array<{ left_run_id: string; right_run_id: string; agreement_rate?: number | null; boundary_distance?: number | null }>
  execution: FixedNjitExecutionAudit
}

export interface RegimeApplicationBinding {
  usage: PublicationUsage
  name?: string
  path?: string
  status?: string
  run_id?: string
  revision?: number
  publication_id?: string
}

export class HistoricalRegimeApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'HistoricalRegimeApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    let message = `请求失败（${response.status}）`
    try {
      const body = await response.json()
      const detail = body?.detail
      if (typeof detail?.message === 'string') message = detail.message
      else if (typeof detail === 'string') message = detail
      else if (typeof body?.message === 'string') message = body.message
      else if (Array.isArray(detail)) message = '提交内容未通过校验，请检查必填项与参数范围。'
    } catch { /* retain the stable fallback */ }
    throw new HistoricalRegimeApiError(response.status, message)
  }
  return response.json() as Promise<T>
}

function listFrom<T>(value: unknown, key = 'items'): T[] {
  if (Array.isArray(value)) return value as T[]
  if (value && typeof value === 'object') {
    const source = value as Record<string, unknown>
    if (Array.isArray(source[key])) return source[key] as T[]
  }
  return []
}

function assertHistoricalRegimeRunExecution(run: HistoricalRegimeRun): HistoricalRegimeRun {
  const audits = run.calculation_audits?.length
    ? run.calculation_audits
    : run.calculation_audit
      ? [run.calculation_audit]
      : []
  assertCompliantExecutionGraph(audits, `历史情景「${run.name || run.id}」`)
  return run
}

function normalizeMeta(value: unknown): HistoricalRegimeMeta {
  const source = (value && typeof value === 'object' ? value : {}) as Record<string, unknown>
  return {
    schema_version: typeof source.schema_version === 'string' ? source.schema_version : undefined,
    modes: listFrom<{ id: RegimeMode; label: string; description?: string }>(source.modes),
    templates: listFrom<RegimeTemplate>(source.templates),
    data_sources: listFrom<RegimeDataSource>(source.data_sources ?? source.series_catalog),
    feature_catalog: listFrom<RegimeFeatureOption>(source.feature_catalog ?? source.features),
    algorithm_families: listFrom<RegimeAlgorithmOption>(source.algorithm_families ?? source.algorithms),
    application_targets: listFrom<RegimeApplicationTarget>(source.application_targets ?? source.usages),
    causality_classes: Array.isArray(source.causality_classes) ? source.causality_classes : undefined,
    limits: source.limits && typeof source.limits === 'object' ? source.limits as Record<string, number> : undefined,
		formula_language: source.formula_language && typeof source.formula_language === 'object' ? source.formula_language as RegimeFormulaLanguage : undefined,
		indicator_catalog: listFrom<RegimeIndicatorCatalogItem>(source.indicator_catalog),
		indicator_periods: listFrom<RegimeIndicatorPeriod>(source.indicator_periods),
		historical_regime_runtime: source.historical_regime_runtime && typeof source.historical_regime_runtime === 'object'
			? source.historical_regime_runtime as HistoricalRegimeMeta['historical_regime_runtime']
			: undefined,
	}
}

export async function getHistoricalRegimeMeta(): Promise<HistoricalRegimeMeta> {
  return normalizeMeta(await request<unknown>('/api/historical-regimes/meta'))
}

export async function listHistoricalRegimeDefinitions(): Promise<HistoricalRegimeDefinition[]> {
  return listFrom<HistoricalRegimeDefinition>(await request<unknown>('/api/historical-regimes/definitions'))
}

export async function getHistoricalRegimeDefinition(id: string): Promise<HistoricalRegimeDefinition> {
  return request<HistoricalRegimeDefinition>(`/api/historical-regimes/definitions/${encodeURIComponent(id)}`)
}

export async function createHistoricalRegimeDefinition(definition: HistoricalRegimeDefinition): Promise<HistoricalRegimeDefinition> {
  return request<HistoricalRegimeDefinition>('/api/historical-regimes/definitions', {
    method: 'POST',
    body: JSON.stringify(definition),
  })
}

export async function updateHistoricalRegimeDefinition(definition: HistoricalRegimeDefinition): Promise<HistoricalRegimeDefinition> {
  if (!definition.id) throw new Error('保存修订版前需要定义 ID。')
  return request<HistoricalRegimeDefinition>(`/api/historical-regimes/definitions/${encodeURIComponent(definition.id)}`, {
    method: 'PUT',
    body: JSON.stringify(definition),
  })
}

export async function runHistoricalRegime(
  definition: HistoricalRegimeDefinition | { id: string; revision: number },
  mode: RegimeMode,
  asOf?: string,
): Promise<HistoricalRegimeRun> {
  const preparation = await request<{ required: boolean; compile_token?: string | null }>(
    '/api/historical-regimes/formulas/prepare',
    {
      method: 'POST',
      body: JSON.stringify({ definition, mode, ...(asOf ? { as_of: asOf } : {}) }),
    },
  )
  const run = await request<HistoricalRegimeRun>('/api/historical-regimes/run', {
    method: 'POST',
    body: JSON.stringify({
      definition,
      mode,
      ...(asOf ? { as_of: asOf } : {}),
      ...(preparation.compile_token ? { compile_token: preparation.compile_token } : {}),
    }),
  })
  return assertHistoricalRegimeRunExecution(run)
}

export async function listHistoricalRegimeRuns(definitionId?: string): Promise<HistoricalRegimeRun[]> {
  const query = definitionId ? `?definition_id=${encodeURIComponent(definitionId)}` : ''
  return listFrom<HistoricalRegimeRun>(await request<unknown>(`/api/historical-regimes/runs${query}`))
}

export async function getHistoricalRegimeRun(id: string): Promise<HistoricalRegimeRun> {
  return assertHistoricalRegimeRunExecution(
    await request<HistoricalRegimeRun>(`/api/historical-regimes/runs/${encodeURIComponent(id)}`),
  )
}

export async function publishHistoricalRegimeRun(id: string, usage: PublicationUsage, note = ''): Promise<{ run_id: string; publication: RegimePublication; publications: RegimePublication[]; application_bindings?: RegimeApplicationBinding[] }> {
  return request(`/api/historical-regimes/runs/${encodeURIComponent(id)}/publish`, {
    method: 'POST',
    body: JSON.stringify({ usage, ...(note ? { note } : {}) }),
  })
}

export async function compareHistoricalRegimeRuns(runIds: string[], referenceRunId?: string): Promise<RegimeComparison> {
  const result = await request<RegimeComparison>('/api/historical-regimes/compare', {
    method: 'POST',
    body: JSON.stringify({ run_ids: runIds, ...(referenceRunId ? { reference_run_id: referenceRunId } : {}) }),
  })
  assertFixedNjitExecution(result.execution, '历史情景版本比较')
  return result
}
