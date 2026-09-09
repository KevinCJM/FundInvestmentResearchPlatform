import {
  assertCompliantExecutionGraph,
  assertCompliantNumericalExecution,
} from '../utils/fixedNjitExecution'

export type RegimeMode = 'realtime' | 'retrospective'
export type RegimeRunStatus = 'queued' | 'preparing' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface RegimeGraphPortSchema {
  id: string
  name?: string
  type?: string
  label?: string
  description?: string
  type_label?: string
  value_type?: string
  required?: boolean
  multiple?: boolean
}

export interface RegimeParameterSchema {
  deprecated?: boolean
  type?: 'number' | 'integer' | 'string' | 'boolean' | 'array' | 'object'
  title?: string
  label?: string
  description?: string
  default?: unknown
  enum?: unknown[]
  enum_labels?: string[]
  option_source?: 'research_series.fields' | string
  minimum?: number
  maximum?: number
  step?: number
  placeholder?: string
}

export interface RegimeNodeSchema {
  authoring_hidden?: boolean
  indicator_reference?: { id: string; revision: number; definition_hash: string; result_kind: string }
  id: string
  type_id?: string
  type?: string
  version?: number
  type_version?: number
  label: string
  description?: string
  category: string
  category_label?: string
  phase?: string
  causal?: boolean
  repaints?: boolean
  supports_realtime?: boolean
  minimum_samples?: number
  cost_estimate?: { class?: string; expression?: string; unit?: string }
  kernel_id?: string | null
  kernel_version?: string | null
  model_version?: string | null
  formula_language?: {
    id?: string
    expression_parameter?: string
    variables?: string[]
    prepare_required?: boolean
    token_bound_to_ast?: boolean
  } | null
  inputs: RegimeGraphPortSchema[]
  outputs: RegimeGraphPortSchema[]
  parameter_schema?: {
    type?: 'object'
    properties?: Record<string, RegimeParameterSchema>
    required?: string[]
  }
  parameters?: Record<string, RegimeParameterSchema>
  execution_policy?: {
    backend?: string
    njit_required?: boolean
    third_party_exempt?: boolean
    request_time_compilation?: number
    execution_lane?: string
  }
  njit_policy?: {
    execution_backend?: string
    njit_required?: boolean
    fixed_signature?: boolean
    request_time_compilation?: number
    python_fallback?: number
    execution_lane?: string
  }
  tags?: string[]
  available?: boolean
  status?: string
  unavailable_reason?: string | null
}

export interface RegimeGraphConnection {
  node_id: string
  port: string
}

export interface RegimeGraphNode {
  id: string
  type: string
  type_version?: number
  label?: string
  parameters: Record<string, unknown>
  inputs: Record<string, RegimeGraphConnection>
  /** Editor-only layout. It is stripped before requests. */
  position?: { x: number; y: number }
}

export interface RegimeGraphOutput {
  node_id: string
  port: string
}

export interface RegimeGraphEdge {
  source: RegimeGraphOutput
  target: RegimeGraphOutput
}

export interface RegimeStateDefinition {
  id: string
  label: string
  color?: string
  role?: string
  order?: number
}

export interface RegimeGraphDefinition {
  id?: string
  revision?: number
  template_id?: string
  created_at?: string
  updated_at?: string
  schema_version: '2.0' | string
  name: string
  description: string
  graph: {
    nodes: RegimeGraphNode[]
    edges?: RegimeGraphEdge[]
    outputs: {
      [name: string]: RegimeGraphOutput | undefined
      state?: RegimeGraphOutput
      probabilities?: RegimeGraphOutput
      confidence?: RegimeGraphOutput
      recognition_index?: RegimeGraphOutput
      effective_index?: RegimeGraphOutput
      reason_code?: RegimeGraphOutput
    }
    exposed_node_ids?: string[]
    channel_metadata?: Record<string, { label: string; unit?: string; display_format?: 'number' | 'percent'; precision?: number }>
  }
  states: RegimeStateDefinition[]
  evaluation_targets: Array<Record<string, unknown>>
  validation: Record<string, unknown>
  usage_intent: string
}

export interface RegimeGraphTemplate {
  id: string
  name: string
  description?: string
  tags?: string[]
  revision?: number
  definition?: RegimeGraphDefinition
  default_mode?: RegimeMode
  supported_modes?: RegimeMode[]
}

export interface RegimeGraphIssue {
  code?: string
  message: string
  node_id?: string
  field?: string
  severity?: 'error' | 'warning' | string
}

export interface RegimeGraphInference {
  valid: boolean
  graph_hash?: string
  definition_hash?: string
  topological_order?: string[]
  errors: RegimeGraphIssue[]
  warnings: RegimeGraphIssue[]
  inferred?: {
    ports?: Record<string, Record<string, string>>
    state_count?: number
    execution_lanes?: string[]
    nodes?: Record<string, {
      inputs?: Record<string, string>
      outputs?: Record<string, string>
      causal?: boolean
      execution_backend?: string
    }>
    outputs?: Record<string, string>
    causal?: boolean
    realtime_eligible?: boolean
  }
}

export interface PreparedRegimeGraph {
  plan_id: string
  compile_token: string
  graph_hash: string
  prepared_at?: string
  runtime_audit: unknown
  formula_plans?: Record<string, {
    compile_token?: string
    compiled_plan_id?: string
    expression_hash?: string
    typed_expression?: {
      version?: string
      root?: string
      output?: unknown
      context?: unknown
      cost?: unknown
      nodes?: Array<Record<string, unknown>> | Record<string, Record<string, unknown>>
      edges?: Array<Record<string, unknown>>
    }
  }>
}

export interface RegimePreviewRun {
  id: string
  status: RegimeRunStatus
  stage?: string
  progress?: number
  message?: string
  created_at?: string
  expires_at?: string
  error?: { code?: string; message: string } | string | null
  result?: RegimePreviewResult | null
  execution?: unknown
  calculation_audits?: unknown
  runtime_audit?: unknown
}

export interface RegimeEvaluationConditionalMetric {
  state_id: string
  state_label: string
  observations?: number | null
  return_observations?: number | null
  mean_period_return?: number | null
  return?: number | null
  annualized_return?: number | null
  volatility?: number | null
  max_drawdown?: number | null
  sharpe?: number | null
  positive_rate?: number | null
  win_rate?: number | null
  return_alignment?: string
}

export interface RegimeEvaluationResult {
  id: string
  name: string
  primary: boolean
  source?: Record<string, unknown>
  snapshot?: Record<string, unknown>
  conditional_metrics?: RegimeEvaluationConditionalMetric[]
  artifact?: Record<string, unknown> | null
}

export type RegimeEvaluationResults = Record<string, RegimeEvaluationResult>

export interface RegimePreviewResult {
  schema_version?: string
  graph_hash?: string
  definition_hash?: string
  mode?: RegimeMode
  row_count?: number
  state_counts?: Record<string, number>
  evaluation_results?: RegimeEvaluationResults
  [key: string]: unknown
}

export interface RegimeSeriesRow {
  date: string
  observation_date?: string
  recognized_at?: string | null
  effective_date?: string | null
  recognition_index?: number | null
  effective_index?: number | null
  reason_code?: number | string | null
  reasons?: string[]
  value?: number | null
  state_id?: string | null
  state_label?: string | null
  confidence?: number | null
  probabilities?: Record<string, number | null>
  values?: Record<string, number | string | null>
  [key: string]: unknown
}

export interface RegimeUpstreamOutput {
  node_id: string
  node_label: string
  port: string
  port_label: string
  value_type: string
  distance: number
  plottable: boolean
  unavailable_reason?: string | null
}

export interface RegimeSeriesPage {
  run_id: string
  node_label?: string
  upstream_outputs?: RegimeUpstreamOutput[]
  value_type?: string
  node_id?: string
  port?: string
  items: RegimeSeriesRow[]
  total: number
  offset: number
  limit: number
  execution?: unknown
}

export type RegimePublicationUsage = 'research_display' | 'product_research' | 'formal_backtest' | 'taa'

export interface RegimeFormalRun {
  id: string
  schema_version?: string
  definition_id: string | null
  definition_revision: number | null
  definition_source?: string
  name: string
  mode: RegimeMode
  created_at: string
  immutable?: boolean
  content_hash?: string
  /** Frozen final-result view, validated by the result adapter before display. */
  overview?: unknown
  states?: RegimeStateDefinition[]
  series?: RegimeSeriesRow[]
  segments?: Array<Record<string, unknown>>
  conditional_metrics?: Array<Record<string, unknown>> | Record<string, unknown>
  conditional_stats?: Array<Record<string, unknown>> | Record<string, unknown>
  evaluation_results?: RegimeEvaluationResults
  transition?: Record<string, unknown>
  stability?: { status?: string; state_switches?: number; classified_ratio?: number | null; [key: string]: unknown }
  walk_forward?: { status?: string; method?: string; fold_count?: number; classified_observations?: number; folds?: Array<Record<string, unknown>>; [key: string]: unknown }
  causality?: {
    publish_eligible_usages?: RegimePublicationUsage[]
    blockers?: string[]
    warnings?: string[]
    realtime_eligible?: boolean
    [key: string]: unknown
  }
  calculation_audits?: unknown[]
  calculation_audit?: unknown
  artifact_manifest?: {
    artifact_id?: string
    checksum?: string
    path?: string
    node_outputs?: { artifact_id?: string; checksum?: string; size_bytes?: number; arrays?: Array<Record<string, unknown>>; [key: string]: unknown }
    series?: { artifact_id?: string; checksum?: string; row_count?: number; size_bytes?: number; [key: string]: unknown }
    [key: string]: unknown
  }
  publications?: Array<{ id: string; usage: RegimePublicationUsage; published_at: string; note?: string; definition_revision?: number; run_id?: string }>
  application_bindings?: Array<Record<string, unknown>>
  /** Lightweight list snapshots deliberately omit the materialized series. */
  series_included?: boolean
  series_detail_endpoint?: string
}

export interface RegimeRunComparison {
  run_ids: string[]
  reference_run_id?: string
  agreement_rate?: number | null
  disagreement_periods?: Array<{ start_date: string; end_date: string; states: Record<string, string> }>
  runs?: Array<{ run_id: string; name?: string; mode?: string; segments?: number; causality_class?: string; publish_eligible_usages?: string[] }>
  pairwise?: Array<{ left_run_id: string; right_run_id: string; common_observations?: number; agreement_rate?: number | null; boundary_distance?: number | null }>
  execution: unknown
}

export interface RegimePublicationResult {
  run_id: string
  publication: { id: string; usage: RegimePublicationUsage; published_at: string; note?: string }
  publications: Array<{ id: string; usage: RegimePublicationUsage; published_at: string; note?: string }>
  application_bindings?: Array<Record<string, unknown>>
}

export interface RegimeV1CopyResult {
  source_v1: { id: string; revision: number; content_hash?: string }
  definition: RegimeGraphDefinition
  inference: RegimeGraphInference
  warnings?: string[]
}

export type RegimeGraphAssetKind = 'template' | 'subgraph'

export interface RegimeGraphAsset {
  id: string
  kind: RegimeGraphAssetKind
  revision: number
  name: string
  description?: string
  registry_version?: string
  graph_hash?: string
  content_hash: string
  definition?: RegimeGraphDefinition
  graph?: RegimeGraphDefinition['graph']
  created_at?: string
  updated_at?: string
}

export interface RegimeGraphAssetInstantiation {
  kind: RegimeGraphAssetKind
  source: { asset_id: string; asset_revision: number; content_hash: string; registry_version?: string }
  definition?: RegimeGraphDefinition
  graph?: RegimeGraphDefinition['graph']
  inference?: RegimeGraphInference
}

export interface RegimeExperimentDimension {
  node_id: string
  parameter: string
  values: unknown[]
}

export interface RegimeExperimentCandidate {
  rank: number
  candidate_id: string
  parameter_differences: Array<{ node_id: string; parameter: string; baseline?: unknown; candidate?: unknown }>
  definition_hash?: string
  metrics: {
    agreement?: number | null
    classified_observations?: number
    classified_ratio?: number | null
    flip_rate?: number | null
    mean_boundary_distance_observations?: number | null
    [key: string]: unknown
  }
  disagreement_intervals?: Array<{ start_index?: number; end_index?: number; start_date?: string; end_date?: string; observations?: number }>
  disagreement_intervals_truncated?: boolean
  rank_value?: number | null
}

export interface RegimeBatchExperiment {
  id: string
  schema_version: string
  definition_id: string
  definition_revision: number
  mode: RegimeMode
  as_of?: string | null
  plan_id?: string
  graph_hash?: string
  parameter_grid: RegimeExperimentDimension[]
  ranking_metric: 'agreement' | 'classified_ratio' | 'low_flip_rate' | 'boundary_distance'
  baseline: { classified_observations?: number; state_switches?: number; classified_ratio?: number | null; flip_rate?: number | null; [key: string]: unknown }
  candidate_count: number
  ranking: RegimeExperimentCandidate[]
  calculation_audit: unknown
  created_at?: string
  immutable?: boolean
  content_hash?: string
}

export interface RegimeGraphDiagnostic {
  code: string
  message: string
  path?: string
  node_id?: string
}

export class RegimeGraphApiError extends Error {
  constructor(readonly status: number, message: string, readonly diagnostics: RegimeGraphDiagnostic[] = []) {
    super(message)
    this.name = 'RegimeGraphApiError'
  }
}

function errorMessage(body: unknown, fallback: string) {
  const value = body as { detail?: unknown; message?: unknown } | null
  const detail = value?.detail as { message?: unknown } | string | undefined
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && typeof detail.message === 'string') return detail.message
  if (typeof value?.message === 'string') return value.message
  return fallback
}

function errorDiagnostics(body: unknown): RegimeGraphDiagnostic[] {
  const diagnostics = (body as { detail?: { diagnostics?: unknown } } | null)?.detail?.diagnostics
  return Array.isArray(diagnostics) ? diagnostics.filter((item): item is RegimeGraphDiagnostic =>
    item && typeof item.code === 'string' && typeof item.message === 'string') : []
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  })
  if (!response.ok) {
    let body: unknown = null
    try { body = await response.json() } catch { /* keep stable fallback */ }
    const diagnostics = errorDiagnostics(body)
    const message = errorMessage(body, `请求失败（${response.status}）`)
    const reasons = [...new Set(diagnostics.map(item => item.message).filter(item => item !== message))].slice(0, 3)
    throw new RegimeGraphApiError(response.status, [message, ...reasons].join(' '), diagnostics)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

function listFrom<T>(value: unknown): T[] {
  if (Array.isArray(value)) return value as T[]
  const wrapped = value as { items?: unknown } | null
  return Array.isArray(wrapped?.items) ? wrapped.items as T[] : []
}

function assertExecution(value: unknown, label: string) {
  if (Array.isArray(value)) {
    assertCompliantExecutionGraph(value, label)
    return
  }
  const wrapped = value as { plan?: unknown } | null
  assertCompliantNumericalExecution(wrapped?.plan ?? value, label)
}

function completedRunWithAudit(run: RegimePreviewRun) {
  if (run.status !== 'completed') return run
  const result = run.result as { execution?: unknown; calculation_audits?: unknown; runtime_audit?: unknown; diagnostics?: { execution_audit?: unknown } } | null
  const audit = run.execution
    ?? run.calculation_audits
    ?? run.runtime_audit
    ?? result?.execution
    ?? result?.calculation_audits
    ?? result?.runtime_audit
    ?? result?.diagnostics?.execution_audit
  assertExecution(audit, '历史情景试算')
  return run
}

function formalRunWithAudit(run: RegimeFormalRun) {
  const audits = run.calculation_audits?.length ? run.calculation_audits : run.calculation_audit ? [run.calculation_audit] : []
  assertCompliantExecutionGraph(audits, `历史情景正式运行「${run.name || run.id}」`)
  return run
}

export function createBlankRegimeDefinition(): RegimeGraphDefinition {
  return {
    schema_version: '2.0',
    name: '未命名历史情景研究',
    description: '',
    graph: { nodes: [], edges: [], outputs: {} },
    states: [],
    evaluation_targets: [],
    validation: { walk_forward: true },
    usage_intent: 'research_display',
  }
}

export function cloneRegimeGraphDefinition(definition: RegimeGraphDefinition): RegimeGraphDefinition {
  const cloned = JSON.parse(JSON.stringify(definition)) as RegimeGraphDefinition
  cloned.graph.nodes = (cloned.graph.nodes || []).map((node) => ({ ...node, parameters: node.parameters || {}, inputs: node.inputs || {} }))
  for (const edge of cloned.graph.edges || []) {
    const target = cloned.graph.nodes.find((node) => node.id === edge.target.node_id)
    if (target && !target.inputs[edge.target.port]) target.inputs[edge.target.port] = { ...edge.source }
  }
  cloned.graph.edges = cloned.graph.nodes.flatMap((node) => Object.entries(node.inputs).map(([port, source]) => ({ source: { ...source }, target: { node_id: node.id, port } })))
  return cloned
}

export function definitionForRequest(definition: RegimeGraphDefinition): RegimeGraphDefinition {
  const cloned = cloneRegimeGraphDefinition(definition)
  cloned.graph.nodes = cloned.graph.nodes.map(({ position: _position, ...node }) => node)
  cloned.graph.edges = cloned.graph.nodes.flatMap((node) => Object.entries(node.inputs).map(([port, source]) => ({ source: { ...source }, target: { node_id: node.id, port } })))
  return cloned
}

export async function getRegimeNodeCatalog(signal?: AbortSignal) {
  return listFrom<RegimeNodeSchema>(await request<unknown>('/api/historical-regimes/nodes', { signal })).map((schema) => {
    const normalizePort = (port: RegimeGraphPortSchema) => ({ ...port, id: port.id || port.name || '', value_type: port.value_type || port.type })
    const policy = schema.execution_policy ?? (schema.njit_policy ? {
      backend: schema.njit_policy.execution_backend,
      njit_required: schema.njit_policy.njit_required,
      third_party_exempt: schema.njit_policy.execution_lane === 'isolated_model_train_or_infer',
      request_time_compilation: schema.njit_policy.request_time_compilation,
      execution_lane: schema.njit_policy.execution_lane,
    } : undefined)
    return {
      ...schema,
      id: schema.id || schema.type_id || schema.type || '',
      type: schema.type || schema.type_id,
      version: schema.version ?? schema.type_version,
      inputs: (schema.inputs || []).map(normalizePort),
      outputs: (schema.outputs || []).map(normalizePort),
      execution_policy: policy,
    }
  })
}

export async function getRegimeGraphTemplates(signal?: AbortSignal) {
  return listFrom<RegimeGraphTemplate>(await request<unknown>('/api/historical-regimes/templates/v2', { signal }))
}

export async function instantiateRegimeTemplate(templateId: string, signal?: AbortSignal) {
  const response = await request<{ definition?: RegimeGraphDefinition } | RegimeGraphDefinition>(
    `/api/historical-regimes/templates/${encodeURIComponent(templateId)}/instantiate`,
    { method: 'POST', body: JSON.stringify({}), signal },
  )
  const wrapped = response as { definition?: RegimeGraphDefinition }
  return wrapped.definition ?? response as RegimeGraphDefinition
}

export async function copyHistoricalRegimeDefinitionToV2(definitionId: string, revision: number, signal?: AbortSignal) {
  return request<RegimeV1CopyResult>(`/api/historical-regimes/definitions/${encodeURIComponent(definitionId)}/copy-to-v2?revision=${encodeURIComponent(String(revision))}`, {
    method: 'POST',
    signal,
  })
}

export async function listRegimeGraphAssets(kind?: RegimeGraphAssetKind, signal?: AbortSignal) {
  const query = kind ? `?kind=${encodeURIComponent(kind)}` : ''
  return listFrom<RegimeGraphAsset>(await request<unknown>(`/api/historical-regimes/v2/graph-assets${query}`, { signal }))
}

export async function createRegimeGraphAsset(kind: RegimeGraphAssetKind, asset: { name: string; description?: string; definition?: RegimeGraphDefinition; graph?: RegimeGraphDefinition['graph'] }, signal?: AbortSignal) {
  return request<RegimeGraphAsset>('/api/historical-regimes/v2/graph-assets', {
    method: 'POST', body: JSON.stringify({ kind, asset }), signal,
  })
}

export async function updateRegimeGraphAsset(assetId: string, revision: number, asset: { name: string; description?: string; definition?: RegimeGraphDefinition; graph?: RegimeGraphDefinition['graph'] }, signal?: AbortSignal) {
  return request<RegimeGraphAsset>(`/api/historical-regimes/v2/graph-assets/${encodeURIComponent(assetId)}`, {
    method: 'PUT', body: JSON.stringify({ revision, asset }), signal,
  })
}

export async function instantiateRegimeGraphAsset(assetId: string, revision?: number, signal?: AbortSignal) {
  const query = revision ? `?revision=${encodeURIComponent(String(revision))}` : ''
  return request<RegimeGraphAssetInstantiation>(`/api/historical-regimes/v2/graph-assets/${encodeURIComponent(assetId)}/instantiate${query}`, {
    method: 'POST', signal,
  })
}

export async function listRegimeBatchExperiments(definitionId?: string, signal?: AbortSignal) {
  const query = definitionId ? `?definition_id=${encodeURIComponent(definitionId)}` : ''
  return listFrom<RegimeBatchExperiment>(await request<unknown>(`/api/historical-regimes/v2/experiments${query}`, { signal }))
}

export async function getRegimeBatchExperiment(experimentId: string, signal?: AbortSignal) {
  const experiment = await request<RegimeBatchExperiment>(`/api/historical-regimes/v2/experiments/${encodeURIComponent(experimentId)}`, { signal })
  assertExecution(experiment.calculation_audit, '历史情景批量实验')
  return experiment
}

export async function runRegimeBatchExperiment(input: {
  definition: Pick<RegimeGraphDefinition, 'schema_version' | 'id' | 'revision'>
  compileToken: string
  mode: RegimeMode
  asOf?: string
  parameterGrid: RegimeExperimentDimension[]
  rankingMetric: RegimeBatchExperiment['ranking_metric']
}, signal?: AbortSignal) {
  if (!input.definition.id || !input.definition.revision) throw new Error('批量实验必须引用已保存的精确定义版本。')
  const experiment = await request<RegimeBatchExperiment>('/api/historical-regimes/v2/experiments', {
    method: 'POST',
    body: JSON.stringify({
      definition: { schema_version: '2.0', id: input.definition.id, revision: input.definition.revision },
      compile_token: input.compileToken,
      mode: input.mode,
      ...(input.asOf ? { as_of: input.asOf } : {}),
      parameter_grid: input.parameterGrid,
      ranking_metric: input.rankingMetric,
    }),
    signal,
  })
  assertExecution(experiment.calculation_audit, '历史情景批量实验')
  return experiment
}

export async function listRegimeGraphDefinitions(signal?: AbortSignal) {
  return listFrom<RegimeGraphDefinition>(await request<unknown>('/api/historical-regimes/v2/definitions', { signal }))
}

export async function getRegimeGraphDefinition(definitionId: string, revision?: number, signal?: AbortSignal) {
  const query = revision ? `?revision=${encodeURIComponent(String(revision))}` : ''
  return request<RegimeGraphDefinition>(`/api/historical-regimes/v2/definitions/${encodeURIComponent(definitionId)}${query}`, { signal })
}

export async function createRegimeGraphDefinition(definition: RegimeGraphDefinition, signal?: AbortSignal) {
  return request<RegimeGraphDefinition>('/api/historical-regimes/v2/definitions', {
    method: 'POST',
    body: JSON.stringify({ definition: definitionForRequest(definition) }),
    signal,
  })
}

export async function updateRegimeGraphDefinition(definition: RegimeGraphDefinition, signal?: AbortSignal) {
  if (!definition.id || !definition.revision) throw new Error('保存修订版前需要已保存的定义 ID 与 revision。')
  return request<RegimeGraphDefinition>(`/api/historical-regimes/v2/definitions/${encodeURIComponent(definition.id)}`, {
    method: 'PUT',
    body: JSON.stringify({ definition: definitionForRequest(definition), revision: definition.revision }),
    signal,
  })
}

export async function inferRegimeGraph(definition: RegimeGraphDefinition, signal?: AbortSignal) {
  return request<RegimeGraphInference>('/api/historical-regimes/infer', {
    method: 'POST',
    body: JSON.stringify({ definition: definitionForRequest(definition) }),
    signal,
  })
}

export interface RegimeAuthoringResolution {
  valid: boolean
  definition: RegimeGraphDefinition | null
  source: string
  diagnostics: Array<{ code: string; message: string; severity?: string; line?: number | null; path?: string }>
  compile_status: 'not_requested'
  display_latex?: Record<string, string>
  formula_steps?: Record<string, Array<{ node_id: string; port: string; label: string; latex: string; description: string; parameters: Array<{ label: string; value: string }> }>>
  math_notation_version?: string
  math_error?: string
}

export async function resolveRegimeAuthoring(definition: RegimeGraphDefinition, mode: RegimeMode,
  sourceKind: 'graph' | 'formula', source = '', signal?: AbortSignal) {
  return request<RegimeAuthoringResolution>('/api/historical-regimes/authoring/resolve', {
    method: 'POST', signal, body: JSON.stringify({ definition: definitionForRequest(definition), mode, source_kind: sourceKind, source, compact: true }),
  })
}

export async function prepareRegimeGraph(definition: RegimeGraphDefinition, signal?: AbortSignal, previewTarget?: RegimeGraphConnection) {
  const response = await request<PreparedRegimeGraph>('/api/historical-regimes/prepare', {
    method: 'POST',
    body: JSON.stringify({ definition: definitionForRequest(definition), preview_target: previewTarget }),
    signal,
  })
  assertExecution(response.runtime_audit, '历史情景预热计划')
  return response
}

export async function startRegimePreviewRun(
  definition: RegimeGraphDefinition,
  options: { compileToken: string; mode: RegimeMode; asOf?: string; ttlSeconds?: number; previewTarget?: RegimeGraphConnection },
  signal?: AbortSignal,
) {
  return request<RegimePreviewRun>('/api/historical-regimes/preview-runs', {
    method: 'POST',
    body: JSON.stringify({
      definition: definitionForRequest(definition),
      compile_token: options.compileToken,
      mode: options.mode,
      as_of: options.asOf || undefined,
      ttl_seconds: options.ttlSeconds,
      preview_target: options.previewTarget,
    }),
    signal,
  })
}

export async function getRegimePreviewRun(runId: string, signal?: AbortSignal) {
  const run = await request<RegimePreviewRun>(`/api/historical-regimes/preview-runs/${encodeURIComponent(runId)}`, { signal })
  return completedRunWithAudit(run)
}

export async function cancelRegimePreviewRun(runId: string, signal?: AbortSignal) {
  return request<RegimePreviewRun | void>(`/api/historical-regimes/preview-runs/${encodeURIComponent(runId)}`, {
    method: 'DELETE', signal,
  })
}

export async function getRegimePreviewOverview(runId: string, signal?: AbortSignal): Promise<unknown> {
  return request<unknown>(`/api/historical-regimes/preview-runs/${encodeURIComponent(runId)}/overview`, { signal })
}

export async function getRegimePreviewSeries(
  runId: string,
  options: { nodeId?: string; port?: string; offset?: number; limit?: number } = {},
  signal?: AbortSignal,
) {
  const params = new URLSearchParams()
  if (options.nodeId) params.set('node_id', options.nodeId)
  if (options.port) params.set('port', options.port)
  params.set('offset', String(options.offset ?? 0))
  params.set('limit', String(options.limit ?? 200))
  const page = await request<RegimeSeriesPage>(
    `/api/historical-regimes/preview-runs/${encodeURIComponent(runId)}/series?${params.toString()}`,
    { signal },
  )
  if (page.execution !== undefined) assertExecution(page.execution, '历史情景节点序列')
  const raw = page as RegimeSeriesPage & { id?: string; items?: Array<RegimeSeriesRow & { observation_date?: string }> }
  return {
    ...page,
    run_id: page.run_id || raw.id || runId,
    items: (raw.items || []).map((item) => ({ ...item, date: item.date || item.observation_date || '' })),
  }
}

export async function getAllRegimePreviewSeries(runId: string, nodeId: string, port: string, signal?: AbortSignal) {
  let offset = 0
  let total: number | undefined
  const items: RegimeSeriesRow[] = []
  while (true) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError')
    const part = await getRegimePreviewSeries(runId, { nodeId, port, offset, limit: 5000 }, signal)
    if (!Number.isInteger(part.total) || part.total < 0 || part.run_id !== runId || part.node_id !== nodeId || part.port !== port || part.offset !== offset ||
      (total !== undefined && part.total !== total) || (!part.items.length && offset < part.total) || offset + part.items.length > part.total) {
      throw new Error('节点结果分页不完整或与当前预览不一致，请重新预览。')
    }
    total = part.total
    items.push(...part.items)
    if (items.length >= total) return { ...part, offset: 0, items }
    offset = items.length
  }
}

export interface RegimeNormalizedChart {
  run_id: string
  node_id: string
  port: string
  base_index: number
  base_date: string
  base_value: number
  values: Array<number | null>
  change_pct: Array<number | null>
  execution: unknown
}

export async function getRegimeNormalizedChart(runId: string, nodeId: string, port: string, baseIndex: number, signal?: AbortSignal) {
  const params = new URLSearchParams({ node_id: nodeId, port, base_index: String(baseIndex) })
  const result = await request<RegimeNormalizedChart>(`/api/historical-regimes/preview-runs/${encodeURIComponent(runId)}/normalized-chart?${params}`, { signal })
  assertExecution(result.execution, '区间归一化')
  return result
}

export async function runSavedRegimeGraph(
  definition: Pick<RegimeGraphDefinition, 'schema_version' | 'id' | 'revision'>,
  compileToken: string,
  mode: RegimeMode,
  asOf?: string,
  signal?: AbortSignal,
) {
  if (!definition.id || !definition.revision) throw new Error('正式运行必须引用已保存的精确定义版本。')
  const run = await request<RegimeFormalRun>('/api/historical-regimes/run', {
    method: 'POST',
    body: JSON.stringify({
      definition: { schema_version: '2.0', id: definition.id, revision: definition.revision },
      mode,
      compile_token: compileToken,
      ...(asOf ? { as_of: asOf } : {}),
    }),
    signal,
  })
  return formalRunWithAudit(run)
}

export async function listRegimeFormalRuns(definitionId?: string, signal?: AbortSignal) {
  const query = definitionId ? `?definition_id=${encodeURIComponent(definitionId)}` : ''
  return listFrom<RegimeFormalRun>(await request<unknown>(`/api/historical-regimes/runs${query}`, { signal }))
}

export async function getRegimeFormalRun(runId: string, signal?: AbortSignal) {
  return formalRunWithAudit(await request<RegimeFormalRun>(`/api/historical-regimes/runs/${encodeURIComponent(runId)}`, { signal }))
}

export async function compareRegimeFormalRuns(runIds: string[], referenceRunId?: string, signal?: AbortSignal) {
  const response = await request<RegimeRunComparison>('/api/historical-regimes/compare', {
    method: 'POST',
    body: JSON.stringify({ run_ids: runIds, ...(referenceRunId ? { reference_run_id: referenceRunId } : {}) }),
    signal,
  })
  assertExecution(response.execution, '历史情景正式运行比较')
  return response
}

export async function publishRegimeFormalRun(runId: string, usage: RegimePublicationUsage, note = '', signal?: AbortSignal) {
  return request<RegimePublicationResult>(`/api/historical-regimes/runs/${encodeURIComponent(runId)}/publish`, {
    method: 'POST',
    body: JSON.stringify({ usage, ...(note ? { note } : {}) }),
    signal,
  })
}
