import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ProductPoolSelection from './ProductPoolSelection'
import { readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'
import {
  createInvestableUniverseSnapshot,
  getInvestableUniverse,
  listProductPoolVersions,
} from '../services/productPools'

vi.mock('../services/productPools', async () => {
  // Keep the pure helpers real: the page prints a version's data cut-off next
  // to the research day, and a stubbed-out helper would hide that regression.
  const actual = await vi.importActual<typeof import('../services/productPools')>(
    '../services/productPools',
  )
  return {
    ...actual,
    createInvestableUniverseSnapshot: vi.fn(),
    getInvestableUniverse: vi.fn(),
    listProductPoolVersions: vi.fn(),
  }
})

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
    localStorage.clear()
    sessionStorage.clear()
    vi.mocked(listProductPoolVersions).mockResolvedValue({ items: [version], total: 1 })
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(universe)
    vi.mocked(getInvestableUniverse).mockResolvedValue(universe)
  })

  it('选择有效版本并锁定不可变可投资域', async () => {
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
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
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await screen.findByText('可投资域快照已锁定')
    fireEvent.click(screen.getByRole('button', { name: label }))

    expect(await screen.findByTestId('location-probe')).toHaveTextContent(`${path}?universe=universe-1`)
  })
  it('把评价数据截止日晚于研究日的版本标成含未来信息', async () => {
    vi.mocked(listProductPoolVersions).mockResolvedValue({
      items: [
        {
          ...version,
          evaluation_plans: [
            { plan_id: 'plan-1', plan_name: '权益评价', plan_revision: 1, product_kind: 'etf', as_of: '2026-08-31' },
          ],
        } as never,
      ],
      total: 1,
    } as never)

    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    // The research day defaults to today; a 2026-08-31 data cut is later than
    // any date this test can run on only if today is earlier, so drive the date
    // explicitly rather than depending on the clock.
    const dateInput = await screen.findByLabelText(/研究日期/)
    fireEvent.change(dateInput, { target: { value: '2020-01-01' } })

    expect(await screen.findByText(/评价数据截至 2026-08-31/)).toBeInTheDocument()
    expect(await screen.findByText(/晚于研究日 2020-01-01/)).toBeInTheDocument()
  })
  it('从发布池进入时预选目标版本，并把其他历史版本按需收起', async () => {
    vi.mocked(listProductPoolVersions).mockResolvedValue({ items: [version, { ...version, id: 'old-version', version: 2 }], total: 2 })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?version=version-1']}><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    expect(screen.queryByRole('checkbox', { name: /核心产品池 · V2/ })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: '显示历史发布版本' }))
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池 · V2/ }))
    expect(screen.getByRole('checkbox', { name: /核心产品池 · V3/ })).not.toBeChecked()
    expect(screen.getByRole('checkbox', { name: /核心产品池 · V2/ })).toBeChecked()
  })

  it('从明确产品范围恢复真实名称和日期，不套用之前的选池草稿', async () => {
    writeAllocationDraft('pool:resume', { name: '旧草稿', researchDate: '2020-01-01', selectedIds: ['old-version'], snapshotId: 'old-universe', selectionEdited: false })
    updateAllocationJourney({ universeId: 'old-universe', name: '旧研究', researchDate: '2020-01-01', poolVersionIds: ['old-version'] })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?universe=universe-1']}><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByText('可投资域快照已锁定')).toBeInTheDocument()
    expect(getInvestableUniverse).toHaveBeenCalledWith('universe-1')
    expect(getInvestableUniverse).not.toHaveBeenCalledWith('old-universe')
    expect(screen.getByLabelText(/研究日期/)).toHaveValue('2026-09-04')
    expect(screen.getByDisplayValue('投前研究可投资域')).toBeInTheDocument()
    expect(await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    expect(readAllocationJourney()).toEqual({ universeId: 'universe-1', name: '投前研究可投资域', researchDate: '2026-09-04', poolVersionIds: ['version-1'] })
  })

  it('基于已锁定范围生成新快照时切换到新身份，不把新快照存入原范围草稿', async () => {
    const next = { ...universe, id: 'universe-2', name: '新的研究范围' }
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(next)
    vi.mocked(getInvestableUniverse).mockImplementation(async id => id === next.id ? next : universe)
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?universe=universe-1']}><ProductPoolSelection /><LocationProbe /></MemoryRouter>)
    await screen.findByText('可投资域快照已锁定')
    fireEvent.change(screen.getByLabelText('研究名称'), { target: { value: next.name } })
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('?universe=universe-2'))
    expect(readAllocationDraft<{ snapshotId: string }>('pool:universe:universe-1')?.snapshotId).toBe('universe-1')
    expect(readAllocationDraft<{ snapshotId: string }>('pool:universe:universe-2')?.snapshotId).toBe('universe-2')
  })

})
