import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  attachEvaluationPlan,
  batchUpdateProductPoolMembers,
  createInvestableUniverseSnapshot,
  getProductPoolReviewData,
  ProductPoolApiError,
} from './productPools'

const jsonResponse = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
  status,
  headers: { 'Content-Type': 'application/json' },
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('productPools service', () => {
  it('attaches an evaluation plan as the pool grouping source', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: 'pool-1', revision: 2 }))
    vi.stubGlobal('fetch', fetchMock)

    await attachEvaluationPlan('pool-1', {
      revision: 1,
      plan_id: 'plan-equity',
      as_of: '2026-08-31',
      selection_mode: 'top_n',
      selection_value: 20,
    })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/product-pools/pool-1/evaluation-plans')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toEqual({
      revision: 1,
      plan_id: 'plan-equity',
      as_of: '2026-08-31',
      selection_mode: 'top_n',
      selection_value: 20,
    })
  })

  it('loads selected review columns through repeated query parameters', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ pool_id: 'pool-1', rows: [] }))
    vi.stubGlobal('fetch', fetchMock)

    await getProductPoolReviewData('pool-1', {
      basicFields: ['management', 'issue_amount'],
      snapshotMetrics: ['return_1y', 'sharpe_1y'],
    })

    expect(fetchMock.mock.calls[0][0]).toBe(
      '/api/product-pools/pool-1/review-data?basic_field=management&basic_field=issue_amount&snapshot_metric=return_1y&snapshot_metric=sharpe_1y',
    )
  })

  it('saves multiple review decisions in one atomic request', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: 'pool-1', revision: 5 }))
    vi.stubGlobal('fetch', fetchMock)
    const items = [
      {
        kind: 'etf' as const,
        product_id: '510300.SH',
        research_status: 'approved' as const,
        usage_status: 'normal' as const,
        primary_plan_id: 'plan-equity',
        max_weight: 0.3,
        reasons: ['批量复核通过'],
        owner: 'Kevin',
        review_due_date: null,
        valid_until: null,
        substitute_group: '',
      },
    ]

    await batchUpdateProductPoolMembers('pool-1', { revision: 4, items })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/product-pools/pool-1/members/batch')
    expect(init.method).toBe('PUT')
    expect(JSON.parse(init.body)).toEqual({ revision: 4, items })
  })

  it('creates an immutable investable-universe snapshot from published versions', async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: 'universe-1', immutable: true }))
    vi.stubGlobal('fetch', fetchMock)

    await createInvestableUniverseSnapshot({
      name: '投前研究域',
      research_date: '2026-09-04',
      version_ids: ['pool-version-1'],
    })

    const [url, init] = fetchMock.mock.calls[0]
    expect(url).toBe('/api/investable-universe-snapshots')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body)).toMatchObject({
      research_date: '2026-09-04',
      version_ids: ['pool-version-1'],
    })
  })

  it('keeps structured backend errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({
      detail: { code: 'PRODUCT_POOL_REVIEW_INCOMPLETE', message: '仍有产品未复核。', field: 'members' },
    }, 422)))

    await expect(createInvestableUniverseSnapshot({
      name: '投前研究域',
      research_date: '2026-09-04',
      version_ids: ['pool-version-1'],
    })).rejects.toMatchObject({
      status: 422,
      code: 'PRODUCT_POOL_REVIEW_INCOMPLETE',
      field: 'members',
      message: '仍有产品未复核。',
    } satisfies Partial<ProductPoolApiError>)
  })
})
