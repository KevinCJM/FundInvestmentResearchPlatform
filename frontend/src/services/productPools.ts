import type { ProductKind } from './customIndicators'

export type ProductPoolResearchStatus = 'pending' | 'approved' | 'watch' | 'rejected'
export type ProductPoolUsageStatus = 'normal' | 'limited' | 'no_new' | 'hold_only' | 'unavailable'
export type EvaluationPlanSelectionMode = 'all_ranked' | 'top_n' | 'top_percent'

export interface ProductPoolPlanBinding {
  plan_id: string
  plan_revision: number
  plan_name: string
  product_kind: ProductKind
  result_id: string
  as_of: string | null
  data_generation?: string | null
  selection_mode: EvaluationPlanSelectionMode
  selection_value: number | null
  ranked_count: number
  excluded_count: number
  imported_count: number
  attached_at: string
}

export interface ProductPoolMemberEvidence {
  source: 'evaluation_plan' | 'manual_exception'
  plan_id: string
  plan_revision: number
  plan_name: string
  result_id: string | null
  as_of: string | null
  rank: number | null
  score: number | null
  percentile: number | null
  result_status: string
  exclusion_reason: string | null
}

export interface ProductPoolMember {
  key: string
  kind: ProductKind
  product_id: string
  code: string
  name: string
  research_status: ProductPoolResearchStatus
  usage_status: ProductPoolUsageStatus
  primary_plan_id: string | null
  max_weight: number | null
  reasons: string[]
  owner: string
  review_due_date: string | null
  valid_until: string | null
  substitute_group: string
  manual_exception: boolean
  evidences: ProductPoolMemberEvidence[]
  reviewed_at?: string | null
}

export interface ProductPool {
  id: string
  revision: number
  name: string
  description: string
  purpose: string
  owner: string
  state: 'draft' | 'active' | 'archived'
  evaluation_plans: ProductPoolPlanBinding[]
  members: ProductPoolMember[]
  current_version_id: string | null
  published_at: string | null
  created_at: string
  updated_at: string
}

export interface ProductPoolMemberReviewUpdate {
  research_status: ProductPoolResearchStatus
  usage_status: ProductPoolUsageStatus
  primary_plan_id: string
  max_weight: number | null
  reasons: string[]
  owner: string
  review_due_date: string | null
  valid_until: string | null
  substitute_group: string
}

export interface ProductPoolReviewField {
  field: string
  label: string
  source: 'basic' | 'snapshot'
  data_type: 'text' | 'number' | 'date'
  unit: string | null
  display_format?: 'percent' | 'number'
  available: boolean
  applicable_product_kinds: ProductKind[]
  metric_type?: string
  metric_type_label?: string
  description?: string
}

export interface ProductPoolReviewDataRow {
  key: string
  kind: ProductKind
  product_id: string
  basic_values: Record<string, number | string | null>
  snapshot_values: Record<string, number | string | null>
  snapshot_value_dates: Record<string, string | null>
  snapshot_statuses: Record<string, string | null>
  snapshot_warnings: Record<string, string | null>
}

export interface ProductPoolReviewData {
  pool_id: string
  pool_revision: number
  basic_fields: ProductPoolReviewField[]
  snapshot_metric_fields: ProductPoolReviewField[]
  selected_basic_fields: string[]
  selected_snapshot_metrics: string[]
  snapshot_states: Record<string, Record<string, unknown>>
  rows: ProductPoolReviewDataRow[]
}

export interface ProductPoolVersion {
  id: string
  pool_id: string
  pool_name: string
  version: number
  pool_revision: number
  description: string
  purpose: string
  owner: string
  effective_from: string
  effective_to: string | null
  publication_note: string
  evaluation_plans: ProductPoolPlanBinding[]
  members: ProductPoolMember[]
  member_counts: Record<ProductPoolResearchStatus, number>
  investable_count: number
  immutable: true
  created_at: string
}

/**
 * The latest day any input behind a published version was cut.
 *
 * Not `effective_from` — a desk can set that to anything. This is what a
 * research day has to be compared against to know whether the pool knew more
 * than the day it is being used on.
 */
export function poolVersionDataAsOf(version: ProductPoolVersion): string | null {
  // Older stored versions predate the binding list; a missing field must not
  // blank the page it is printed on.
  const stamps = (version.evaluation_plans ?? [])
    .map((binding) => binding?.as_of)
    .filter((value): value is string => Boolean(value))
    .sort()
  return stamps.length > 0 ? stamps[stamps.length - 1] : null
}

export interface InvestableUniverseProduct {
  key: string
  kind: ProductKind
  product_id: string
  code: string | null
  name: string | null
  evaluation_plan_id: string
  evaluation_plan_revision: number
  evaluation_plan_name: string
  usage_status: 'normal' | 'limited'
  max_weight: number | null
  valid_until: string | null
  substitute_group: string
  reasons: string[]
  source_version_ids: string[]
  source_pool_ids: string[]
}

export interface InvestableUniverseGroup {
  evaluation_plan_id: string
  evaluation_plan_revision: number
  evaluation_plan_name: string
  products: InvestableUniverseProduct[]
  product_count: number
}

export interface InvestableUniverseSnapshot {
  id: string
  name: string
  research_date: string
  version_ids?: string[]
  version_refs?: Array<Record<string, unknown>>
  pool_ids?: string[]
  excluded_product_keys?: string[]
  groups?: InvestableUniverseGroup[]
  products?: InvestableUniverseProduct[]
  members?: InvestableUniverseSearchItem[]
  product_count?: number
  summary?: {
    pool_count: number
    member_count: number
    eligible_count: number
    restricted_count: number
    watch_count: number
  }
  content_hash?: string
  immutable: true
  created_at: string
}

export interface InvestableUniverseEvaluationSource {
  pool_id: string
  pool_name: string
  pool_version_id: string
  pool_version_number: number
  evaluation_plan_id: string
  evaluation_plan_revision: number
  evaluation_plan_name: string
  source_rank: number | null
  source_score: number | null
}

export interface InvestableUniverseSearchItem {
  kind: ProductKind
  product_id: string
  code: string | null
  name: string
  research_status: ProductPoolResearchStatus
  usage_status: ProductPoolUsageStatus
  decision_reasons: string[]
  max_weight: number | null
  valid_until: string | null
  substitute_groups: string[]
  evaluation_sources: InvestableUniverseEvaluationSource[]
  eligible: boolean
  eligibility_reasons: string[]
  warnings: string[]
}

export interface InvestableUniverseSearchResponse {
  snapshot_id: string
  snapshot_name: string
  research_date: string
  items: InvestableUniverseSearchItem[]
  total: number
  page: number
  page_size: number
}

export function investableUniverseEligibleCount(
  snapshot: InvestableUniverseSnapshot | null | undefined,
): number {
  if (!snapshot) return 0
  const summaryCount = Number(snapshot.summary?.eligible_count)
  if (Number.isFinite(summaryCount)) return summaryCount
  const productCount = Number(snapshot.product_count)
  if (Number.isFinite(productCount)) return productCount
  if (Array.isArray(snapshot.members)) {
    return snapshot.members.filter((item) => item.eligible !== false).length
  }
  if (Array.isArray(snapshot.products)) return snapshot.products.length
  return 0
}

export interface ProductPoolVersionDiff {
  version_id: string
  against_id: string
  added: ProductPoolMember[]
  removed: ProductPoolMember[]
  changed: Array<{
    key: string
    name: string
    changes: Record<string, { before: unknown; after: unknown }>
  }>
}

export class ProductPoolApiError extends Error {
  status: number
  code?: string
  field?: string

  constructor(message: string, status: number, code?: string, field?: string) {
    super(message)
    this.name = 'ProductPoolApiError'
    this.status = status
    this.code = code
    this.field = field
  }
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })
  const payload = await response.json().catch(() => null) as any
  if (!response.ok) {
    const detail = payload?.detail
    const message = typeof detail === 'string'
      ? detail
      : detail?.message ?? payload?.message ?? `请求失败（${response.status}）`
    throw new ProductPoolApiError(message, response.status, detail?.code, detail?.field)
  }
  return payload as T
}

export const listProductPools = () =>
  apiRequest<{ items: ProductPool[]; total: number }>('/api/product-pools')

export const getProductPool = (poolId: string) =>
  apiRequest<ProductPool>(`/api/product-pools/${encodeURIComponent(poolId)}`)

export const createProductPool = (input: Pick<ProductPool, 'name' | 'description' | 'purpose' | 'owner'>) =>
  apiRequest<ProductPool>('/api/product-pools', {
    method: 'POST',
    body: JSON.stringify(input),
  })

export const updateProductPool = (
  poolId: string,
  input: Pick<ProductPool, 'name' | 'description' | 'purpose' | 'owner'> & { revision: number },
) => apiRequest<ProductPool>(`/api/product-pools/${encodeURIComponent(poolId)}`, {
  method: 'PUT',
  body: JSON.stringify(input),
})

export const archiveProductPool = (poolId: string, revision: number) =>
  apiRequest<ProductPool>(`/api/product-pools/${encodeURIComponent(poolId)}?revision=${revision}`, {
    method: 'DELETE',
  })

export const attachEvaluationPlan = (
  poolId: string,
  input: {
    revision: number
    plan_id: string
    as_of?: string | null
    selection_mode: EvaluationPlanSelectionMode
    selection_value?: number | null
  },
) => apiRequest<ProductPool>(`/api/product-pools/${encodeURIComponent(poolId)}/evaluation-plans`, {
  method: 'POST',
  body: JSON.stringify(input),
})

export const removeEvaluationPlan = (poolId: string, planId: string, revision: number) =>
  apiRequest<ProductPool>(
    `/api/product-pools/${encodeURIComponent(poolId)}/evaluation-plans/${encodeURIComponent(planId)}?revision=${revision}`,
    { method: 'DELETE' },
  )

export const addManualPoolMember = (
  poolId: string,
  input: {
    revision: number
    plan_id: string
    kind?: ProductKind | null
    product_id: string
    code?: string | null
    name?: string | null
    reason: string
  },
) => apiRequest<ProductPool>(`/api/product-pools/${encodeURIComponent(poolId)}/members`, {
  method: 'POST',
  body: JSON.stringify(input),
})

export const updateProductPoolMember = (
  poolId: string,
  member: Pick<ProductPoolMember, 'kind' | 'product_id'>,
  input: ProductPoolMemberReviewUpdate & { revision: number },
) => apiRequest<ProductPool>(
  `/api/product-pools/${encodeURIComponent(poolId)}/members/${member.kind}/${encodeURIComponent(member.product_id)}`,
  { method: 'PUT', body: JSON.stringify(input) },
)

export const batchUpdateProductPoolMembers = (
  poolId: string,
  input: {
    revision: number
    items: Array<Pick<ProductPoolMember, 'kind' | 'product_id'> & ProductPoolMemberReviewUpdate>
  },
) => apiRequest<ProductPool>(
  `/api/product-pools/${encodeURIComponent(poolId)}/members/batch`,
  { method: 'PUT', body: JSON.stringify(input) },
)

export const getProductPoolReviewData = (
  poolId: string,
  options: {
    basicFields?: string[]
    snapshotMetrics?: string[]
    signal?: AbortSignal
  } = {},
) => {
  const params = new URLSearchParams()
  for (const field of options.basicFields ?? []) params.append('basic_field', field)
  for (const field of options.snapshotMetrics ?? []) params.append('snapshot_metric', field)
  const query = params.toString()
  return apiRequest<ProductPoolReviewData>(
    `/api/product-pools/${encodeURIComponent(poolId)}/review-data${query ? `?${query}` : ''}`,
    { signal: options.signal },
  )
}

export const publishProductPool = (
  poolId: string,
  input: { revision: number; effective_from: string; effective_to?: string | null; publication_note?: string },
) => apiRequest<{ pool: ProductPool; version: ProductPoolVersion }>(
  `/api/product-pools/${encodeURIComponent(poolId)}/publish`,
  { method: 'POST', body: JSON.stringify(input) },
)

export const listProductPoolVersions = (options: { poolId?: string; activeOn?: string } = {}) => {
  const params = new URLSearchParams()
  if (options.poolId) params.set('pool_id', options.poolId)
  if (options.activeOn) params.set('active_on', options.activeOn)
  const query = params.toString()
  return apiRequest<{ items: ProductPoolVersion[]; total: number }>(
    `/api/product-pool-versions${query ? `?${query}` : ''}`,
  )
}

export const getProductPoolVersion = (versionId: string) =>
  apiRequest<ProductPoolVersion>(`/api/product-pool-versions/${encodeURIComponent(versionId)}`)

export const diffProductPoolVersions = (versionId: string, againstId: string) =>
  apiRequest<ProductPoolVersionDiff>(
    `/api/product-pool-versions/${encodeURIComponent(versionId)}/diff?against=${encodeURIComponent(againstId)}`,
  )

export const createInvestableUniverseSnapshot = (input: {
  name: string
  research_date: string
  version_ids: string[]
  excluded_product_keys?: string[]
}) => apiRequest<InvestableUniverseSnapshot>('/api/investable-universe-snapshots', {
  method: 'POST',
  body: JSON.stringify(input),
})

export const getInvestableUniverseSnapshot = (snapshotId: string) =>
  apiRequest<InvestableUniverseSnapshot>(
    `/api/investable-universe-snapshots/${encodeURIComponent(snapshotId)}`,
  )

// Downstream research pages use the shorter name while the repository keeps
// the immutable-snapshot endpoint for backward compatibility.
export const getInvestableUniverse = getInvestableUniverseSnapshot

export const searchInvestableUniverseProducts = (
  snapshotId: string,
  options: {
    query?: string
    kind?: ProductKind
    eligibleOnly?: boolean
    page?: number
    pageSize?: number
    signal?: AbortSignal
  } = {},
) => {
  const params = new URLSearchParams()
  if (options.query) params.set('q', options.query)
  if (options.kind) params.set('kind', options.kind)
  params.set('eligible_only', String(options.eligibleOnly ?? true))
  params.set('page', String(options.page ?? 1))
  params.set('page_size', String(options.pageSize ?? 20))
  return apiRequest<InvestableUniverseSearchResponse>(
    `/api/investable-universes/${encodeURIComponent(snapshotId)}/products?${params.toString()}`,
    { signal: options.signal },
  )
}
