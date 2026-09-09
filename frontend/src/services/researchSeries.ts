import {
  assertCompliantExecutionGraph,
  assertCompliantNumericalExecution,
} from '../utils/fixedNjitExecution'

export type ResearchSeriesKind = 'index' | 'etf' | 'fund' | 'macro' | 'indicator' | 'upload' | string

export interface ResearchInlineRow {
  date?: string
  observation_date?: string
  value: number | null
  available_at?: string
  vintage?: string
  revision?: number
}

export interface ResearchSeriesCatalogItem {
  id: string
  name: string
  kind: ResearchSeriesKind
  code?: string
  status?: 'available' | 'not_downloaded' | string
  description?: string
  category?: string
  category_label?: string
  frequency?: string
  unit?: string
  source_api?: string
  dataset?: string
  default_field?: string
  fields?: Array<string | { id?: string; name?: string; label?: string; unit?: string; dtype?: string; nullable?: boolean; available?: boolean; unavailable_reason?: string | null; binding_parameters?: Record<string, unknown> }>
  coverage?: {
    start_date?: string | null
    end_date?: string | null
    first_date?: string | null
    last_date?: string | null
    observations?: number | null
    valid_observations?: number | null
    calendar_coverage_rate?: number | null
  }
  missing?: {
    count?: number | null
    rate?: number | null
    ratio?: number | null
    null_preserved?: boolean
  }
  pit?: { supported?: boolean; policy?: string; available_at_field?: string | null; available_at?: Array<string | null>; as_of?: string | null; [key: string]: unknown }
  vintage?: { supported?: boolean; values?: Array<string | null>; revisions?: Array<number | null>; default?: string | null; requested?: string | null; [key: string]: unknown }
  profile_operations?: string[]
  regime_node_type?: string
  binding_parameters?: Record<string, unknown>
  indicator_version?: { indicator_id: string; revision: number; dsl_version?: string }
  binding_supported?: boolean
  binding_reason?: string | null
  product_kinds?: string[]
  periods?: string[]
  capability?: {
    available?: boolean
    accepted_formats?: string[]
    parsing_location?: string
    transport?: string
    persisted?: boolean
    max_rows?: number
    required_fields?: string[]
    optional_fields?: string[]
  }
}

export interface ResearchSeriesCatalogResponse {
  schema_version?: string
  items: ResearchSeriesCatalogItem[]
  total: number
  offset: number
  limit: number
  snapshot?: {
    id?: string
    generation?: string
    directory?: string
    activated_at?: string
    status?: string
    validation_status?: string
  } | null
  capabilities?: Record<string, unknown>
  execution?: unknown
}

export interface ResearchSeriesProfileRequest {
  series_id: string
  field?: string
  inline_rows?: ResearchInlineRow[]
  name?: string
  frequency?: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annual' | 'irregular'
  start_date?: string
  end_date?: string
  availability_mode?: 'point_in_time' | 'latest'
  register_artifact?: boolean
  artifact_id?: string
  checksum?: string
  as_of?: string
  vintage?: string
  rolling_window?: number
  sample_limit?: number
}

export interface ResearchSeriesProfile {
  series: ResearchSeriesCatalogItem
  coverage?: ResearchSeriesCatalogItem['coverage']
  missing?: ResearchSeriesCatalogItem['missing']
  sampling?: {
    source_observations?: number
    returned_observations?: number
    computed_observations?: number
    displayed_observations?: number
    sampled?: boolean
    method?: string
    computed_before_sampling?: boolean
  }
  dates: string[]
  values: Record<string, Array<number | null>>
  distribution?: Record<string, {
    valid_count?: number | null
    missing_count?: number | null
    missing_rate?: number | null
    mean?: number | null
    std?: number | null
    min?: number | null
    p05?: number | null
    p25?: number | null
    median?: number | null
    p75?: number | null
    p95?: number | null
    max?: number | null
  }>
  pit?: ResearchSeriesCatalogItem['pit']
  vintage?: ResearchSeriesCatalogItem['vintage'] | string | null
  transform_definitions?: Record<string, string>
  warnings?: string[]
  execution: unknown
  regime_node_type?: string
  binding_parameters?: Record<string, unknown>
  binding?: { node_type: string; parameters: Record<string, unknown>; fingerprint?: string }
  artifact?: { artifact_id: string; checksum: string; uri?: string; row_count?: number; [key: string]: unknown } | null
  snapshot?: { source?: string; persisted?: boolean; fingerprint?: string; [key: string]: unknown }
}

export interface ResearchSeriesCompareSource extends ResearchSeriesProfileRequest {
  id?: string
  label?: string
}

export interface ResearchSeriesComparison {
  schema_version: string
  alignment: {
    method: string
    intersected_observations: number
    source_observations: number[]
    computed_before_sampling: boolean
  }
  sampling: {
    method: string
    computed_observations: number
    displayed_observations: number
    sample_limit: number
    computed_before_sampling: boolean
  }
  dates: string[]
  series: Array<{
    id: string
    label: string
    series: ResearchSeriesCatalogItem
    values: Array<number | null>
    standardized: Array<number | null>
    profile_observations: number
    snapshot?: ResearchSeriesProfile['snapshot']
    regime_node_type?: string
    binding_parameters?: Record<string, unknown>
    binding?: ResearchSeriesProfile['binding']
  }>
  correlation: {
    method: string
    source_ids: string[]
    matrix: Array<Array<number | null>>
    observation_counts: number[][]
  }
  scatter_pairs: Array<{
    left_id: string
    right_id: string
    observation_count: number
    displayed_observations: number
    dates: string[]
    x: number[]
    y: number[]
    standardized_x: number[]
    standardized_y: number[]
    correlation: number | null
  }>
  common_valid: {
    definition: string
    observation_count: number
    start_date: string | null
    end_date: string | null
  }
  execution: unknown
}

export class ResearchSeriesApiError extends Error {
  constructor(readonly status: number, message: string) {
    super(message)
    this.name = 'ResearchSeriesApiError'
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
    } catch { /* keep stable fallback */ }
    throw new ResearchSeriesApiError(response.status, message)
  }
  return response.json() as Promise<T>
}

export interface ResearchImportFile {
  columns: string[]
  rows: Record<string, string | number | null>[]
  sheets: string[]
  sheet: string | null
}

export function parseResearchFile(file: File, sheet?: string, signal?: AbortSignal) {
  if (file.size > 8 * 1024 * 1024) return Promise.reject(new Error('文件大小不能超过 8 MB。'))
  const query = new URLSearchParams({ filename: file.name, ...(sheet ? { sheet } : {}) })
  return request<ResearchImportFile>(`/api/research-series/parse-file?${query}`, {
    method: 'POST', body: file, signal, headers: { 'Content-Type': 'application/octet-stream' },
  })
}

export function listUploadedResearchSeries(options: { query?: string; offset?: number; limit?: number }, signal?: AbortSignal) {
  return request<ResearchSeriesCatalogResponse>(`/api/research-series/uploads?${queryString({ q: options.query, offset: options.offset, limit: options.limit })}`, { signal })
}

export function searchResearchProducts(kind: string, query: string, page: number, signal?: AbortSignal) {
  return request<{ items: Array<{ ts_code: string; name: string }>; total: number }>(
    `/api/instruments/products?${queryString({ kind, q: query, page, page_size: 50 })}`, { signal },
  )
}

function queryString(values: Record<string, string | number | undefined>) {
  const params = new URLSearchParams()
  Object.entries(values).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value))
  })
  return params.toString()
}

function assertProfileExecution(value: unknown) {
  if (Array.isArray(value)) assertCompliantExecutionGraph(value, '研究序列统计')
  else {
    const wrapped = value as { plan?: unknown } | null
    assertCompliantNumericalExecution(wrapped?.plan ?? value, '研究序列统计')
  }
}

export async function listResearchSeries(
  options: { query?: string; kind?: string; status?: string; offset?: number; limit?: number } = {},
  signal?: AbortSignal,
) {
  const query = queryString({
    q: options.query,
    kind: options.kind && options.kind !== 'all' ? options.kind : undefined,
    status: options.status && options.status !== 'all' ? options.status : undefined,
    offset: options.offset ?? 0,
    limit: options.limit ?? 200,
  })
  const response = await request<ResearchSeriesCatalogResponse | ResearchSeriesCatalogItem[]>(
    `/api/research-series/catalog?${query}`,
    { signal },
  )
  if (Array.isArray(response)) {
    return { items: response, total: response.length, offset: 0, limit: response.length } as ResearchSeriesCatalogResponse
  }
  if (response.execution !== undefined) assertProfileExecution(response.execution)
  return { ...response, items: Array.isArray(response.items) ? response.items : [] }
}

export async function getResearchSeriesProfile(payload: ResearchSeriesProfileRequest, signal?: AbortSignal) {
  const profile = await request<ResearchSeriesProfile>('/api/research-series/profile', {
    method: 'POST',
    body: JSON.stringify(payload),
    signal,
  })
  assertProfileExecution(profile.execution)
  return profile
}

export async function compareResearchSeries(
  sources: ResearchSeriesCompareSource[],
  sampleLimit = 500,
  signal?: AbortSignal,
) {
  const comparison = await request<ResearchSeriesComparison>('/api/research-series/compare', {
    method: 'POST',
    body: JSON.stringify({ sources, sample_limit: sampleLimit }),
    signal,
  })
  assertProfileExecution(comparison.execution)
  return comparison
}
