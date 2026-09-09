import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AutoAssetClassification from './AutoAssetClassification'
import { ResearchContextProvider } from '../app/ResearchContext'

// Deliberately NOT mocking ../services/productPools: the page tests all stub it,
// so the request path, query parameters and response mapping of the real client
// were never exercised anywhere.
vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))

const SNAPSHOT_ID = 'universe-d8999f58e242468b949e933fbcf0e096'

/** Shape copied from a live GET /api/investable-universes/{id}/products response. */
function productItem(code: string, name: string, maxWeight: number | null = null) {
  return {
    kind: 'etf',
    product_id: code,
    code,
    name,
    research_status: 'approved',
    usage_status: 'normal',
    decision_reasons: [],
    max_weight: maxWeight,
    valid_until: null,
    substitute_groups: [],
    evaluation_sources: [{
      pool_id: 'pool-1',
      pool_name: '测试001',
      pool_version_id: 'pool-version-1',
      pool_version_number: 1,
      evaluation_plan_id: 'plan-1',
      evaluation_plan_revision: 4,
      evaluation_plan_name: '沪深300被动权益ETF评价',
      source_rank: 8,
      source_score: 62.46,
    }],
    eligible: true,
    eligibility_reasons: [],
    warnings: [],
  }
}

const ITEMS = [
  productItem('159913.SZ', '交银深证300价值ETF'),
  productItem('159912.SZ', '汇添富深证300ETF', 0.05),
  productItem('510300.SH', '华泰柏瑞沪深300ETF'),
]

let productRequests: string[] = []

beforeEach(() => {
  productRequests = []
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    if (url === '/api/asset-classes/auto/meta') {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          algorithms: [{ id: 'hierarchical', label: '相关性层次聚类' }],
          features: [{ id: 'correlation', label: '收益相关性距离' }],
          linkages: [{ id: 'average', label: 'average' }],
          weight_modes: [{ id: 'inv_vol', label: '逆波动率' }],
          limits: { min_observations: 60, max_auto_k: 8, min_products: 2 },
          defaults: {},
        }),
      }
    }
    if (url.includes('/products?')) {
      productRequests.push(url)
      return {
        ok: true,
        status: 200,
        json: async () => ({
          snapshot_id: SNAPSHOT_ID,
          snapshot_name: '投前研究可投资域',
          research_date: '2026-09-04',
          items: ITEMS,
          total: ITEMS.length,
          page: 1,
          page_size: 100,
        }),
      }
    }
    if (url.includes(SNAPSHOT_ID)) {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          id: SNAPSHOT_ID,
          name: '投前研究可投资域',
          research_date: '2026-09-04',
          version_ids: ['pool-version-1'],
          members: ITEMS,
          product_count: ITEMS.length,
          summary: {
            pool_count: 1,
            member_count: ITEMS.length,
            eligible_count: ITEMS.length,
            restricted_count: 0,
            watch_count: 0,
          },
          content_hash: 'hash',
          immutable: true,
          created_at: '2026-09-04T00:00:00Z',
        }),
      }
    }
    return { ok: false, status: 404, json: async () => ({ detail: `unexpected ${url}` }) }
  }))
})

afterEach(() => {
  vi.unstubAllGlobals()
})

function renderWithUniverse() {
  return render(
    <MemoryRouter initialEntries={[`/pre-investment/saa/auto-classification?universe=${SNAPSHOT_ID}`]}>
      <ResearchContextProvider>
        <Routes>
          <Route path="/pre-investment/saa/auto-classification" element={<AutoAssetClassification />} />
        </Routes>
      </ResearchContextProvider>
    </MemoryRouter>,
  )
}

describe('AutoAssetClassification against the real investable-universe client', () => {
  it('lists the locked universe products instead of an empty picker', async () => {
    renderWithUniverse()

    expect(await screen.findByText(/已锁定可投资域/)).toBeInTheDocument()
    // The reported failure: the banner said 20 products while the picker said 0.
    expect(await screen.findByText('交银深证300价值ETF', { exact: false })).toBeInTheDocument()
    expect(screen.getByText(/匹配 3 个产品/)).toBeInTheDocument()
    expect(screen.queryByText('没有匹配的产品')).not.toBeInTheDocument()
  })

  it('requests the products endpoint the backend actually serves', async () => {
    renderWithUniverse()
    await waitFor(() => expect(productRequests.length).toBeGreaterThan(0))
    const url = productRequests[productRequests.length - 1]
    expect(url).toContain(`/api/investable-universes/${SNAPSHOT_ID}/products`)
    const params = new URL(url, 'http://localhost').searchParams
    expect(params.get('eligible_only')).toBe('true')
    // page_size is capped at 100 by the route; anything larger is a 422.
    expect(Number(params.get('page_size'))).toBeLessThanOrEqual(100)
  })

  it('shows why the picker is empty instead of failing silently', async () => {
    // The universe itself loads, so the old error banner (which only renders
    // while the universe is missing) stayed hidden and the user saw an empty
    // list with no explanation -- exactly the reported symptom.
    const original = (globalThis.fetch as any)
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes('/products?')) {
        return { ok: false, status: 404, json: async () => ({ detail: 'Not Found' }) }
      }
      return original(input, init)
    }))

    renderWithUniverse()

    expect(await screen.findByText(/已锁定可投资域/)).toBeInTheDocument()
    expect(await screen.findByText(/读取可投资域产品失败/)).toBeInTheDocument()
  })

  it('carries the pool restriction through to the picker', async () => {
    renderWithUniverse()
    expect(await screen.findByText(/限额 5\.0%/)).toBeInTheDocument()
  })
})
