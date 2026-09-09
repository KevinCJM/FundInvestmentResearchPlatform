import type { ConfigRecord, InterfaceConfig } from './dataSources'

export type EtlKind = 'download' | 'map' | 'resolve' | 'snapshot' | 'task'
export interface EtlParameter { id: string; label: string; data_type: 'text' | 'date'; default: string; required: boolean; date_format: 'iso' | 'compact'; description: string }
export interface EtlRunOptions { mode: 'full' | 'incremental' | 'auto_incremental'; parameters: Record<string, string> }
export const runModeLabels = { full: '全量重取', incremental: '手动增量', auto_incremental: '自动增量' }
export interface AutoIncrementalPlan {
  plan_id: string; snapshot: string; cutoff_date: string; lookback_trade_days: number; ready: boolean
  errors: { code: string; message: string }[]
  steps: { id: string; name: string; strategy: string; latest_date: string | null; start_date: string; end_date: string; message: string }[]
}
export interface EtlStep {
  id: string; name: string; kind: EtlKind; inputs: string[]; after?: string[]; task_id?: string | null
  source_id?: string | null; interface_id?: string | null; interface_revision?: number | null
  mode: 'inherit' | 'full' | 'incremental'; params: Record<string, unknown>; target_tables: string[]
  parameter_bindings?: Record<string, string>
  table_id?: string | null; include_history: boolean; history_scope?: 'table' | 'matching_inputs'; allow_empty: boolean
  start_date?: string | null; end_date?: string | null; as_of?: string | null
}
export interface EtlDefinition {
  name: string; description: string; max_runtime_seconds: number; steps: EtlStep[]; parameters?: EtlParameter[]
  graph_version?: 1 | null
  canvas?: { version: 1; positions: Record<string, { x: number; y: number }>; viewport?: { x: number; y: number; zoom: number } | null } | null
}
export interface EtlWorkflow { id: string; revision: number; definition: EtlDefinition; updated_at: string }
export interface EtlValidation { valid: boolean; errors: { code: string; message: string }[]; steps: { id: string; name: string; kind: EtlKind; inputs: string[]; tables: string[] }[]; auto_plan?: AutoIncrementalPlan }
export interface EtlProgress {
  phase: string; message: string; completed?: number | null; total?: number | null; unit?: string
  batches?: number; received_rows?: number; activity_at?: string
  logs?: { at: string; message: string }[]
  collection_window?: { first_at: string; last_at: string }
}
export interface EtlWarning { code: string; message: string }
export interface EtlCollectionTiming {
  timezone: string; first_date: string | null; last_date: string | null; cross_date: boolean
  warnings: EtlWarning[]; boundary: string; scope: string
  windows: { run_id: string; step_id: string; name: string; attempt: number; first_at: string | null; last_at: string | null; basis: 'batch_receipts' | 'execution_window' | 'unknown' }[]
}
export interface EtlRun {
  run_id: string; name: string; status: string; created_at: string; updated_at: string; error?: string; message?: string
  attempt: number; published: false; cancel_requested?: boolean; definition?: EtlDefinition
  template_definition?: EtlDefinition; options?: EtlRunOptions; auto_plan?: AutoIncrementalPlan
  recovered_from?: string
  execution?: { mode: 'independent' | 'legacy'; state: string; heartbeat_at?: string | null; message: string }
  recovery?: { can_resume: boolean; artifact_check_pending: boolean; blockers: EtlWarning[]; warnings?: EtlWarning[]; job?: EtlRecoveryJob | null; successor?: { run_id: string; name: string; status: string } | null }
  collection_timing?: EtlCollectionTiming
  resume_events?: { attempt: number; at: string; warnings: EtlWarning[] }[]
  steps: { id: string; name: string; kind: EtlKind; status: string; rows?: number; pages?: number; error?: string; started_at?: string; finished_at?: string; output?: Record<string, unknown>; progress?: EtlProgress; heartbeat_at?: string; worker_only?: boolean; imported_from?: { run_id: string; step_id: string; execution_fingerprint: string } }[]
}
async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/data-sources/etl${path}`, { ...init, cache: 'no-store', headers: { 'Content-Type': 'application/json', ...init.headers } })
  const value = await response.json().catch(() => null)
  if (!response.ok) throw new Error(value?.detail?.message || `ETL 操作失败（HTTP ${response.status}）`)
  return value as T
}
const post = <T,>(path: string, body: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(body) })
export const listEtlWorkflows = () => request<EtlWorkflow[]>('/workflows')
export const listEtlTemplates = () => request<{ id: string; name: string; description: string; definition: EtlDefinition }[]>('/templates')
export const saveEtlWorkflow = (id: string, definition: EtlDefinition, expected_revision: number) => request<EtlWorkflow>(`/workflows/${id}`, { method: 'PUT', body: JSON.stringify({ definition, expected_revision }) })
export const deleteEtlWorkflow = (id: string, expected_revision: number) => request(`/workflows/${id}`, { method: 'DELETE', body: JSON.stringify({ expected_revision }) })
export const validateEtl = (definition: EtlDefinition, options?: EtlRunOptions) => post<EtlValidation>('/validate', { definition, ...(options ? { options } : {}) })
export const runEtl = (definition: EtlDefinition, request_id: string, options: EtlRunOptions = { mode: 'incremental', parameters: {} }, auto_plan_id?: string) => post<EtlRun>('/runs', { definition, request_id, options, confirm: true, ...(auto_plan_id ? { auto_plan_id } : {}) })
export const listEtlRuns = () => request<EtlRun[]>('/runs')
export interface EtlRecoveryJob {
  id: string; source_run_id: string; target_run_id: string
  status: 'QUEUED' | 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'INTERRUPTED'
  phase: string; message: string; code?: string; created_at: string; updated_at: string
  logs: { at: string; message: string }[]
}
export const recoverEtl = (id: string, request_id: string) => post<EtlRecoveryJob>(`/runs/${id}/recovery`, { confirm: true, request_id })
export const getEtlRun = (id: string) => request<EtlRun>(`/runs/${id}`)
export const cancelEtl = (id: string) => post<EtlRun>(`/runs/${id}/cancel`, {})
export const stepLabels: Record<EtlKind, string> = { download: '下载原始数据', map: '字段映射', resolve: '多源取值', snapshot: '指标快照计算', task: '数据集任务' }
export const stateLabels: Record<string, string> = { PENDING: '等待执行', RUNNING: '正在执行', SUCCEEDED: '已完成', FAILED: '失败', SKIPPED: '前置未完成', CANCELLED: '已取消', INTERRUPTED: '服务中断' }
export const emptyDefinition = (): EtlDefinition => ({ name: '我的 ETL 流程', description: '', max_runtime_seconds: 3600, steps: [] })
export const blankStep = (kind: EtlKind): EtlStep => ({ id: 's_' + crypto.randomUUID().replace(/-/g, '').slice(0, 12), name: stepLabels[kind], kind, inputs: [], mode: 'inherit', params: {}, target_tables: [], include_history: true, allow_empty: false })
export function downloadStep(record: ConfigRecord<InterfaceConfig>, mode: EtlStep['mode'], params: Record<string, unknown> = {}): EtlStep {
  return { ...blankStep('download'), name: record.config.name, source_id: record.config.source_id, interface_id: record.config.id, interface_revision: record.revision, mode, params: { ...record.config.params, ...params } }
}
export function quickPlan(records: ConfigRecord<InterfaceConfig>[], _mode: EtlStep['mode'], params: Record<string, Record<string, unknown>>, includeHistory: boolean): EtlDefinition {
  // Loading strategy belongs to the run, not a copy of the workflow.
  const downloads = records.map(r => downloadStep(r, 'inherit', params[r.config.id]))
  const maps = downloads.map(d => ({ ...blankStep('map'), name: `映射 · ${d.name}`, inputs: [d.id] }))
  const tableInputs = new Map<string, string[]>()
  records.forEach((record, i) => record.config.mappings.filter(m => m.enabled).forEach(mapping => {
    tableInputs.set(mapping.target_table, [...(tableInputs.get(mapping.target_table) ?? []), maps[i].id])
  }))
  const resolutions = Array.from(tableInputs, ([table, inputs]) => ({ ...blankStep('resolve'), name: `取值 · ${table}`, table_id: table, inputs: [...new Set(inputs)], include_history: includeHistory }))
  return { ...emptyDefinition(), name: '数据下载与更新', steps: [...downloads, ...maps, ...resolutions] }
}
