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
  /** Null while this dataset is still pending measurement. */
  grade: PitGrade | null
  pending?: boolean
  grade_label: string
  fingerprint: string | null
  available_through: string | null
}

export interface PitAuditSummary {
  declared: number
  present: number
  missing: number
  /** Declared files not measured yet; the scan runs behind the request. */
  pending?: number
  grade_a: number
  grade_b: number
  grade_c: number
  total_rows: number
  available_through: string | null
  latest_dataset_end: string | null
  available_through_basis?: string
}

export interface PitAuditScan {
  state: 'idle' | 'running' | 'ready' | 'failed'
  started_at: string | null
  finished_at: string | null
  error: string | null
  pending: string[]
}

export interface PitAudit {
  datasets: PitDatasetAudit[]
  summary: PitAuditSummary
  latest_release: { id: string; name: string; created_at: string; release_fingerprint: string } | null
  scan?: PitAuditScan
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
  /** Stamped when the口径 was edited; null on a version never edited. */
  updated_at?: string | null
  /** The day this version stands on. Null on versions sealed before it moved in. */
  as_of: string | null
  run_mode: RunMode | null
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

/** Sealing a version defines a whole口径: the day, the vintage and the mode. */
export function createDataRelease(
  input: { name: string; note?: string; asOf: string | null; runMode: RunMode },
): Promise<DataRelease> {
  return request<DataRelease>(
    '/api/pit/releases',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: input.name, note: input.note ?? '', asOf: input.asOf, runMode: input.runMode }),
    },
    '封版失败',
  )
}

/** Fix a version's口径. Its table fingerprints are not editable. */
export function updateDataRelease(
  id: string,
  input: { name: string; note?: string; asOf: string | null; runMode: RunMode },
): Promise<DataRelease> {
  return request<DataRelease>(
    `/api/pit/releases/${encodeURIComponent(id)}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: input.name, note: input.note ?? '', asOf: input.asOf, runMode: input.runMode }),
    },
    '修改数据版本失败',
  )
}

export function deleteDataRelease(id: string): Promise<{ deleted_id: string }> {
  return request<{ deleted_id: string }>(
    `/api/pit/releases/${encodeURIComponent(id)}`,
    { method: 'DELETE' },
    '删除数据版本失败',
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
  /** The day this version stands on — its own choice, else its ceiling. */
  as_of: string | null
  run_mode: RunMode | null
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
  /** 'explicit' = the user stated it; 'release' = inherited from the vintage. */
  as_of_source: 'explicit' | 'release' | null
  run_mode: RunMode
  run_mode_label: string
  data_release_id: string | null
  no_pit: boolean
  /** One short string a result footnote can print verbatim. */
  label: string
}

export interface PitSettingsPayload {
  settings: {
    active_release_id: string | null
    /** Both null once a version is applied: the version is the setting. */
    as_of: string | null
    run_mode: RunMode | null
    updated_at: string | null
    note: string
  }
  effective: PitEffectiveContext
  release: PitReleaseSummary | null
  release_error: string | null
  available_releases: PitReleaseSummary[]
  can_apply: boolean
}

export async function fetchPitSettings(signal?: AbortSignal): Promise<PitSettingsPayload> {
  const payload = await request<PitSettingsPayload>('/api/pit/settings', { signal }, '读取 PIT 系统设置失败')
  if (!payload?.settings || !payload.effective || typeof payload.effective.no_pit !== 'boolean'
    || !['RESEARCH', 'STRICT_PIT'].includes(payload.effective.run_mode)
    || (payload.effective.as_of !== null && typeof payload.effective.as_of !== 'string')
    || typeof payload.effective.label !== 'string' || !Array.isArray(payload.available_releases)) {
    throw new Error('PIT 系统设置响应不完整，无法确认当前口径')
  }
  return payload
}

export function applyPitSettings(
  input: { activeReleaseId: string | null; asOf: string | null; runMode: RunMode | null; note?: string },
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
/**
 * What the candidate set behind a run could prove about its own timing.
 *
 * A pool screened on today's numbers and replayed over 2018 is causal in every
 * individual formula and wrong as a whole, so the finding travels with the
 * result rather than being recomputed by whoever reads it.
 */
export interface PitUniverseLineage {
  source: string | null
  coverage?: string | null
  replayable: boolean
  established_at: string | null
  history_begins_at?: string | null
  clean: boolean
  findings: Array<{ code: string; label: string; message: string }>
}

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
  universe?: PitUniverseLineage | null
}

/**
 * What a saved allocation's NAV could prove about the day it stands on.
 *
 * `asset_nv` is computed, not observed: `series_as_of` is the research day the
 * whole series was built under, and a null one means full hindsight.
 */
export interface PitAllocationLineage {
  alloc_name: string
  as_of: string | null
  run_mode: RunMode | null
  series_as_of: string | null
  series_variants: string[]
  hindsight_series: boolean
  rows_dropped_by_as_of: number
  availability_available: boolean
  warnings: string[]
  universe?: PitAllocationUniverse | null
}

/**
 * A saved allocation makes two claims about time, not one.
 *
 * `established_at` is the day its class NAV was *computed*;
 * `snapshot_established_at` is the day the locked pool its products were
 * *screened* from was cut. The second can sit years after the first without any
 * formula noticing, so it is judged and shown separately.
 */
export interface PitAllocationUniverse extends PitUniverseLineage {
  snapshot_id?: string | null
  snapshot_established_at?: string | null
}

/** How firmly the candidate set behind a result could be established. */
export type PitUniverseCoverage = 'REPLAYED' | 'INTERVAL' | 'LATEST_ONLY' | 'NOT_APPLICABLE'

export interface PitUniverseView {
  kind: string
  kind_label: string
  as_of: string | null
  run_mode: RunMode
  coverage: PitUniverseCoverage
  replayable: boolean
  history_begins_at: string | null
  member_count: number
  latest_member_count: number
  excluded_by_replay: number
  sample: { code: string; name: string }[]
  warnings: string[]
}

export const UNIVERSE_COVERAGE_LABELS: Record<PitUniverseCoverage, string> = {
  REPLAYED: '域可回放 · 有当日维表快照',
  INTERVAL: '域按上市/退市日还原',
  LATEST_ONLY: '域仅最新态 · 有幸存者偏差',
  NOT_APPLICABLE: '该数据集不涉及产品域',
}

export const UNIVERSE_COVERAGE_TONE: Record<PitUniverseCoverage, string> = {
  REPLAYED: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  INTERVAL: 'bg-accent-100 text-accent-800 border-accent-200',
  LATEST_ONLY: 'bg-rose-100 text-rose-800 border-rose-200',
  NOT_APPLICABLE: 'bg-slate-100 text-slate-600 border-slate-200',
}

export function fetchPitUniverse(
  options: { kind?: string; asOf?: string | null; signal?: AbortSignal } = {},
): Promise<PitUniverseView> {
  const params = new URLSearchParams({ kind: options.kind ?? 'fund' })
  if (options.asOf) params.set('as_of', options.asOf)
  return request<PitUniverseView>(
    `/api/pit/universe?${params.toString()}`,
    { signal: options.signal },
    '读取可选产品域失败',
  )
}

/** Replay of a published pool: who it would have held on another research day. */
export interface PoolReplayMember {
  key: string
  kind: string
  product_id: string
  code: string
  name: string
}

export interface PoolReplayResult {
  version_id: string
  pool_id: string
  as_of: string
  published_as_of: string | null
  lookahead: boolean
  plans: {
    plan_id: string
    plan_name: string
    published_as_of: string | null
    replayed_as_of: string
    ranked_count: number
    selected_count: number
  }[]
  summary: {
    published_count: number
    replayed_count: number
    kept: number
    added: number
    removed: number
    manual_only: number
  }
  added: PoolReplayMember[]
  removed: PoolReplayMember[]
  manual_only: PoolReplayMember[]
}

export function replayPoolVersion(
  versionId: string,
  asOf: string,
  signal?: AbortSignal,
): Promise<PoolReplayResult> {
  return request<PoolReplayResult>(
    `/api/product-pool-versions/${encodeURIComponent(versionId)}/replay?as_of=${encodeURIComponent(asOf)}`,
    { signal },
    '回放产品池失败',
  )
}
