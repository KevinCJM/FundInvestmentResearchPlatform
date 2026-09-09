import { apiErrorMessage } from '../utils/apiError'

export type PitGrade = 'A' | 'B' | 'C'
export type RunMode = 'RESEARCH' | 'STRICT_PIT'

export interface PitLagBucket {
  bucket: string
  rows: number
}

export interface PitLagProfile {
  p50: number | null
  p95: number | null
  max: number | null
  negative_rows: number
  histogram: PitLagBucket[]
}

export interface PitDatasetAudit {
  dataset_id: string
  label: string
  file: string
  event_field: string | null
  availability_field: string | null
  declared_lag_days: number
  revisable: boolean
  note: string
  present: boolean
  rows: number
  availability_coverage: number | null
  event_range: { start: string | null; end: string | null }
  availability_range: { start: string | null; end: string | null }
  lag: PitLagProfile
  grade: PitGrade
  grade_label: string
  fingerprint: string | null
  available_through: string | null
}

export interface PitAuditSummary {
  declared: number
  present: number
  missing: number
  grade_a: number
  grade_b: number
  grade_c: number
  total_rows: number
  available_through: string | null
  latest_dataset_end: string | null
  available_through_basis?: string
}

export interface PitAudit {
  datasets: PitDatasetAudit[]
  summary: PitAuditSummary
  latest_release: { id: string; name: string; created_at: string; release_fingerprint: string } | null
}

export interface DataReleaseTable {
  dataset_id: string
  label: string
  file: string
  rows: number
  bytes: number | null
  grade: PitGrade
  event_range: { start: string | null; end: string | null }
  availability_range: { start: string | null; end: string | null }
  available_through: string | null
  fingerprint: string
}

export interface DataRelease {
  id: string
  name: string
  note: string
  created_at: string
  parent_release_id: string | null
  immutable: boolean
  tables: DataReleaseTable[]
  summary: PitAuditSummary
  release_fingerprint: string
}

export interface PitContextResolution {
  context: { as_of: string | null; run_mode: RunMode; run_mode_label: string; data_release_id: string | null }
  summary: PitAuditSummary
  blocked_datasets: { dataset_id: string; label: string; grade: PitGrade; grade_label: string; reason: string }[]
  degraded_datasets: { dataset_id: string; label: string; grade: PitGrade; grade_label: string }[]
  stale_datasets: { dataset_id: string; label: string; available_through: string }[]
  usable: boolean
  release?: { id: string; name: string; created_at: string; release_fingerprint: string } | null
  release_error?: string
}

async function request<T>(path: string, init?: RequestInit, fallback = '请求失败'): Promise<T> {
  const response = await fetch(path, init)
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new Error(apiErrorMessage(body, `${fallback}（${response.status}）`))
  return body as T
}

export function fetchPitAudit(options: { refresh?: boolean; signal?: AbortSignal } = {}): Promise<PitAudit> {
  const query = options.refresh ? '?refresh=true' : ''
  return request<PitAudit>(`/api/pit/audit${query}`, { signal: options.signal }, 'PIT 体检失败')
}

export function fetchDataReleases(signal?: AbortSignal): Promise<{ releases: DataRelease[] }> {
  return request<{ releases: DataRelease[] }>('/api/pit/releases', { signal }, '读取数据版本失败')
}

export function createDataRelease(name: string, note = ''): Promise<DataRelease> {
  return request<DataRelease>(
    '/api/pit/releases',
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name, note }) },
    '封版失败',
  )
}

export function resolvePitContext(
  input: { asOf: string | null; runMode: RunMode; dataReleaseId: string | null },
  signal?: AbortSignal,
): Promise<PitContextResolution> {
  return request<PitContextResolution>(
    '/api/pit/context/resolve',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(input),
      signal,
    },
    '研究上下文校验失败',
  )
}

export const GRADE_TONE: Record<PitGrade, string> = {
  A: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  B: 'bg-amber-100 text-amber-900 border-amber-200',
  C: 'bg-rose-100 text-rose-800 border-rose-200',
}

export function formatCoverage(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(2)}%`
}

export function formatLagDays(value: number | null): string {
  if (value === null || Number.isNaN(value)) return '—'
  return Number.isInteger(value) ? String(value) : value.toFixed(1)
}

// --------------------------------------------------------------------------
// System-level PIT setting
// --------------------------------------------------------------------------

export interface PitReleaseSummary {
  id: string
  name: string
  note: string
  created_at: string
  sequence: number | null
  available_through: string | null
  grade_a: number | null
  grade_b: number | null
  grade_c: number | null
  table_count: number
  total_rows: number | null
  release_fingerprint: string
}

export interface PitEffectiveContext {
  as_of: string | null
  run_mode: RunMode
  run_mode_label: string
  data_release_id: string | null
  no_pit: boolean
  /** One short string a result footnote can print verbatim. */
  label: string
}

export interface PitSettingsPayload {
  settings: { active_release_id: string | null; run_mode: RunMode; updated_at: string | null; note: string }
  effective: PitEffectiveContext
  release: PitReleaseSummary | null
  release_error: string | null
  available_releases: PitReleaseSummary[]
  can_apply: boolean
}

export function fetchPitSettings(signal?: AbortSignal): Promise<PitSettingsPayload> {
  return request<PitSettingsPayload>('/api/pit/settings', { signal }, '读取 PIT 系统设置失败')
}

export function applyPitSettings(
  input: { activeReleaseId: string | null; runMode: RunMode; note?: string },
  signal?: AbortSignal,
): Promise<PitSettingsPayload> {
  return request<PitSettingsPayload>(
    '/api/pit/settings',
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...input, note: input.note ?? '' }),
      signal,
    },
    '应用 PIT 口径失败',
  )
}

/** Audit trail a run returns about the point-in-time cut it actually applied. */
export interface PitRunLineage {
  as_of: string | null
  as_of_applied: boolean
  run_mode: RunMode
  availability_field: string
  rows_before_cut: number
  rows_after_cut: number
  rows_dropped_by_as_of: number
  rows_without_announcement: number
  announcement_fallback: boolean
  warnings: string[]
}
