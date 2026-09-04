import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ProductPoolSelection from './ProductPoolSelection'
import {
  createInvestableUniverseSnapshot,
  listProductPoolVersions,
} from '../services/productPools'

vi.mock('../services/productPools', () => ({
  createInvestableUniverseSnapshot: vi.fn(),
  listProductPoolVersions: vi.fn(),
}))

const version = {
  id: 'version-1', pool_id: 'pool-1', pool_name: '核心产品池', version: 3, pool_revision: 4,
  description: '', purpose: '长期配置', owner: 'Kevin', effective_from: '2026-09-01', effective_to: null,
  publication_note: '', evaluation_plans: [], members: [], investable_count: 10,
  member_counts: { pending: 0, approved: 9, watch: 1, rejected: 2 }, immutable: true,
  created_at: '2026-09-01T00:00:00Z',
} as any

const universe = {
  id: 'universe-1', name: '投前研究可投资域', research_date: '2026-09-04', version_ids: ['version-1'],
  groups: [{ evaluation_plan_id: 'plan-equity', evaluation_plan_revision: 2, evaluation_plan_name: '权益评价', product_count: 1, products: [{ key: 'etf:510300.SH', code: '510300.SH', name: '沪深300ETF', usage_status: 'normal' }] }],
  product_count: 1, content_hash: 'universe-hash', created_at: '2026-09-04T00:00:00Z', immutable: true,
} as any

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname}{location.search}</div>
}

describe('ProductPoolSelection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(listProductPoolVersions).mockResolvedValue({ items: [version], total: 1 })
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(universe)
  })

  it('选择有效版本并锁定不可变可投资域', async () => {
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))

    await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalledWith({
      name: '投前研究可投资域',
      research_date: expect.any(String),
      version_ids: ['version-1'],
    }))
    expect(await screen.findByText('可投资域快照已锁定')).toBeInTheDocument()
    expect(screen.getByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '进入手动构建大类' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '进入自动构建大类' })).toBeInTheDocument()
  })

  // The hand-off used to point at a route that does not exist and at a query key
  // the target pages never read, so the locked universe was silently dropped and
  // the user landed back on the dashboard.
  it.each([
    ['进入自动构建大类', '/pre-investment/saa/auto-classification'],
    ['进入手动构建大类', '/pre-investment/saa/asset-classes'],
  ])('%s 携带已锁定的可投资域跳转到 %s', async (label, path) => {
    render(
      <MemoryRouter initialEntries={['/pre-investment/product-pool']}>
        <Routes>
          <Route path="/pre-investment/product-pool" element={<ProductPoolSelection />} />
          <Route path="*" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox'))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await screen.findByText('可投资域快照已锁定')
    fireEvent.click(screen.getByRole('button', { name: label }))

    expect(await screen.findByTestId('location-probe')).toHaveTextContent(`${path}?universe=universe-1`)
  })
})
