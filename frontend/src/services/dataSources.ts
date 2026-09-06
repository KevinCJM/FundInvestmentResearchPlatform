import type { DataModelCatalog } from './dataModel'

export interface DownloadPolicy {
  requests_per_minute: number
  rows_per_minute: number | null
  min_interval_seconds: number
  max_rows_per_request: number
  max_concurrency: number
  connect_timeout_seconds: number
  read_timeout_seconds: number
  max_attempts: number
  backoff_seconds: number
  rate_limit_wait_seconds: number
  max_response_bytes: number
  max_runtime_seconds: number
}

export interface SourceConfig {
  id: string
  name: string
  transport: 'http' | 'tushare' | 'akshare'
  base_url: string
  enabled: boolean
  auth_mode: 'none' | 'bearer' | 'header'
  auth_header: string
  policy: DownloadPolicy
  notes: string
}

export interface FieldMapping {
  target_field: string
  source_field: string | null
  operation: 'copy' | 'scale' | 'constant' | 'enum' | 'date' | 'timestamp' | 'period_end' | 'capture_date'
  factor: number
  constant: string | number | boolean | null
  enum_map: Record<string, string | number | boolean>
  timezone: string
  date_format: string | null
}

export interface IdentityBinding {
  target_field: string
  source_field: string | null
  key_fields: string[]
  constant: string | null
  namespace: string
  resolution: 'namespace' | 'lookup'
  key_transform?: 'none' | 'cn_etf_code' | 'cn_fund_code'
  value_map: Record<string, string>
}

export interface DatasetMapping {
  target_table: string
  contract_version: string
  enabled: boolean
  fields: FieldMapping[]
  identities: IdentityBinding[]
}

export interface InterfaceConfig {
  id: string
  source_id: string
  name: string
  enabled: boolean
  api_name: string
  method: 'GET' | 'POST'
  path: string
  params: Record<string, unknown>
  headers: Record<string, string>
  response: {
    format: 'json_records' | 'json_columns' | 'csv'
    records_path: string
    columns_path: string
    delimiter: string
  }
  source_fields: { name: string; data_type: 'string' | 'number' | 'integer' | 'boolean' | 'date' | 'datetime' | 'json'; description: string; unit: string }[]
  policy: DownloadPolicy
  pagination: { mode: 'none' | 'offset' | 'page'; cursor_param: string; limit_param: string; page_size: number; max_pages: number }
  start_param: string
  end_param: string
  incremental_field: string | null
  mappings: DatasetMapping[]
  notes: string
  entitlement_confirmed: boolean
}

export interface ConfigRecord<T> {
  config: T
  revision: number
  builtin: boolean
  updated_at: string
  credential_configured?: boolean
  validation?: MappingValidation
  effective_policy?: DownloadPolicy
}

export interface MappingValidation {
  valid: boolean
  ready: boolean
  errors: { field?: string; row?: number; table?: string; message: string }[]
  warnings: { field?: string; message: string; code?: string }[]
}

export interface MappingPreview extends MappingValidation {
  source_rows: number
  tables: { table_id: string; accepted_rows: number; rejected_rows: number; columns: string[]; rows: Record<string, unknown>[] }[]
  preview_only: boolean
  received_rows?: number
  sampled_rows?: number
  download_complete?: boolean
  requests?: number
  source_preview?: Record<string, unknown>[]
}

export interface CandidateBatch {
  batch_id: string
  source_id: string
  interface_id: string
  status: 'EMPTY' | 'REJECTED' | 'VALIDATED_CANDIDATE'
  source_rows: number
  created_at: string
  published: false
  tables: { table_id: string; status: string; rows?: number; rejected_rows?: number; errors: { message: string }[] }[]
}

export interface SourceCatalog {
  batches?: CandidateBatch[]
  sources: ConfigRecord<SourceConfig>[]
  interfaces: ConfigRecord<InterfaceConfig>[]
  targets: DataModelCatalog
  editing_enabled: boolean
  templates: { source: SourceConfig; interface: InterfaceConfig }
  boundary: string
}

export class SourceCenterError extends Error {
  constructor(readonly status: number, readonly code: string, message: string) {
    super(message)
    this.name = 'SourceCenterError'
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api/data-sources${path}`, {
    ...init,
    cache: 'no-store',
    headers: { 'Content-Type': 'application/json', ...init.headers },
  })
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    throw new SourceCenterError(response.status, payload?.detail?.code ?? 'REQUEST_FAILED', payload?.detail?.message ?? `请求失败（HTTP ${response.status}）`)
  }
  return payload as T
}

export const fetchSourceCatalog = (signal?: AbortSignal) => request<SourceCatalog>('/catalog', { signal })
export const saveSource = (config: SourceConfig, expected_revision: number) => request<ConfigRecord<SourceConfig>>('/config/source', { method: 'PUT', body: JSON.stringify({ config, expected_revision }) })
export const saveInterface = (config: InterfaceConfig, expected_revision: number) => request<ConfigRecord<InterfaceConfig>>('/config/interface', { method: 'PUT', body: JSON.stringify({ config, expected_revision }) })
export const deleteSourceConfig = (kind: 'source' | 'interface', id: string, expected_revision: number) => request<{ deleted: boolean }>(`/config/${kind}/${encodeURIComponent(id)}`, { method: 'DELETE', body: JSON.stringify({ expected_revision }) })
export const saveSourceCredential = (id: string, value: string | null) => request<{ credential_configured: boolean }>(`/credentials/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify({ value }) })
export const validateSourceMapping = (config: InterfaceConfig) => request<MappingValidation>('/validate', { method: 'POST', body: JSON.stringify({ config }) })
export const previewSourceMapping = (config: InterfaceConfig, sample: string) => request<MappingPreview>('/preview', { method: 'POST', body: JSON.stringify({ config, sample }) })
export const sampleSourceInterface = (id: string, expected_revision: number, params: Record<string, unknown>) => request<MappingPreview>(`/interfaces/${encodeURIComponent(id)}/sample`, { method: 'POST', body: JSON.stringify({ expected_revision, params, confirm: true }) })

export interface ResolutionFieldRule {
  field: string
  minimum: number | null
  maximum: number | null
  absolute_tolerance: number | null
  relative_tolerance: number | null
}
export interface ResolutionTableRule {
  table_id: string
  source_priority: string[]
  fallback_on_missing: boolean
  fallback_on_invalid: boolean
  conflict_action: 'quarantine' | 'prefer_priority'
  required_fields: string[]
  compare_fields: string[]
  absolute_tolerance: number
  relative_tolerance: number
  max_relative_jump: number | null
  field_rules: ResolutionFieldRule[]
}
export interface ResolutionConfig { default_source_priority: string[]; tables: ResolutionTableRule[] }
export interface ResolutionDecision {
  key: Record<string, unknown>; status: string; selected_source: string | null; selected_batch?: string | null
  skipped: { source_id: string; reasons: string[] }[]
  conflicts: { source_id: string; fields: string[]; selected_values?: Record<string, unknown>; other_values?: Record<string, unknown> }[]
}
export interface ResolutionRun {
  run_id?: string; table_id: string; policy_revision?: number; summary?: Record<string, number>; status?: string
  created_at?: string; decisions?: ResolutionDecision[]; published: false; message?: string
}
export interface ResolutionPolicyRecord { config: ResolutionConfig; revision: number; updated_at: string; runs: ResolutionRun[] }
export interface SourceSyncJob {
  job_id: string; interface_id: string; source_id: string; status: 'RUNNING' | 'SUCCEEDED' | 'FAILED' | 'EMPTY'
  mode: string; rows: number; pages: number; message?: string; error?: string; published: false
  resolutions?: ResolutionRun[]; batch?: CandidateBatch
}
export const getResolutionPolicy = () => request<ResolutionPolicyRecord>('/resolution/config')
export const saveResolutionPolicy = (config: ResolutionConfig, expected_revision: number) => request<ResolutionPolicyRecord>('/resolution/config', { method: 'PUT', body: JSON.stringify({ config, expected_revision }) })
export const runResolution = (table_id: string, expected_revision: number, start_date?: string, end_date?: string, as_of?: string) => request<ResolutionRun>('/resolution/run', { method: 'POST', body: JSON.stringify({ table_id, expected_revision, start_date: start_date || null, end_date: end_date || null, as_of: as_of || null }) })
export const previewResolution = (table_id: string, config: ResolutionConfig, rows: Record<string, unknown>[], as_of?: string) => request<ResolutionRun>('/resolution/preview', { method: 'POST', body: JSON.stringify({ table_id, config, rows, as_of: as_of || null }) })
export const listSourceSyncJobs = () => request<SourceSyncJob[]>('/sync/jobs')
export const startSourceSync = (id: string, expected_revision: number, params: Record<string, unknown>, mode: 'full' | 'incremental') => request<SourceSyncJob>(`/interfaces/${encodeURIComponent(id)}/sync`, { method: 'POST', body: JSON.stringify({ expected_revision, params, mode, confirm: true }) })
