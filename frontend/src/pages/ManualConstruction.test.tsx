import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ManualConstruction from './ManualConstruction'
import { evaluateNumericControls } from '../services/businessNumeric'
import { getInvestableUniverse, searchInvestableUniverseProducts } from '../services/productPools'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/businessNumeric', () => ({ evaluateNumericControls: vi.fn() }))
vi.mock('../services/productPools', async () => {
  const actual = await vi.importActual<typeof import('../services/productPools')>('../services/productPools')
  return {
    ...actual,
    getInvestableUniverse: vi.fn(),
    searchInvestableUniverseProducts: vi.fn(),
  }
})

function LocationProbe() {
  const location = useLocation()
  const params = new URLSearchParams(location.search)
  const state = location.state as { returnTo?: string; returnLabel?: string } | null
  return (
    <div data-testid="location-probe">
      {location.pathname}|{params.get('ids')}|{params.get('kinds')}|{state?.returnTo}|{state?.returnLabel}
    </div>
  )
}

describe('ManualConstruction product navigation', () => {
  beforeEach(() => {
    vi.mocked(getInvestableUniverse).mockResolvedValue({
      id: 'universe-1',
      name: '测试可投资域',
      research_date: '2026-09-04',
      version_refs: [],
      members: [],
      summary: { pool_count: 1, member_count: 2, eligible_count: 2, restricted_count: 0, watch_count: 0 },
      content_hash: 'hash',
      created_at: '2026-09-04T00:00:00Z',
      immutable: true,
    })
    vi.mocked(searchInvestableUniverseProducts).mockResolvedValue({
      snapshot_id: 'universe-1',
      snapshot_name: '测试可投资域',
      research_date: '2026-09-04',
      total: 2,
      page: 1,
      page_size: 10,
      items: [
        {
          kind: 'etf', product_id: '159393.SZ', code: '159393.SZ', name: '万家沪深300ETF',
          research_status: 'approved', usage_status: 'normal', decision_reasons: ['通过'], max_weight: 0.6,
          valid_until: null, substitute_groups: [], eligible: true, eligibility_reasons: [], warnings: [],
          evaluation_sources: [{ pool_id: 'pool-1', pool_name: '核心池', pool_version_id: 'v1', pool_version_number: 1, evaluation_plan_id: 'plan-etf', evaluation_plan_revision: 1, evaluation_plan_name: 'ETF评价', source_rank: 1, source_score: 90 }],
        },
        {
          kind: 'fund', product_id: '024011.OF', code: '024011.OF', name: '万家沪深300ETF联接-A',
          research_status: 'approved', usage_status: 'normal', decision_reasons: ['通过'], max_weight: 0.5,
          valid_until: null, substitute_groups: [], eligible: true, eligibility_reasons: [], warnings: [],
          evaluation_sources: [{ pool_id: 'pool-1', pool_name: '核心池', pool_version_id: 'v1', pool_version_number: 1, evaluation_plan_id: 'plan-fund', evaluation_plan_revision: 1, evaluation_plan_name: '联接基金评价', source_rank: 1, source_score: 88 }],
        },
      ],
    })
    vi.mocked(evaluateNumericControls).mockImplementation(async (groups) => ({
      items: groups.map((group) => {
        const total = group.values.reduce((sum, value) => sum + value, 0)
        return { key: group.key, total, difference: total - (group.target ?? 0), within_tolerance: Math.abs(total - (group.target ?? 0)) <= (group.tolerance ?? 1e-8), positive: total > 0, normalized_shares: group.values.map((value) => total > 0 ? value / total : 0) }
      }),
      execution: {
        execution_backend: 'numba_njit_fixed_signature',
        nopython: true,
        object_mode: 0,
        python_fallback: 0,
        request_time_compilation: 0,
        kernel_signatures: { numeric_controls_kernel: ['(Array(float64, 1, C, False, aligned=True), float64, float64)'] },
      },
    }))
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/list-allocations') {
        return { ok: true, json: async () => [] }
      }
      return { ok: false, status: 404, json: async () => ({}) }
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('兼容已生成但缺少 summary 的产品池快照，不再白屏', async () => {
    vi.mocked(getInvestableUniverse).mockResolvedValue({
      id: 'universe-legacy',
      name: '旧结构可投资域',
      research_date: '2026-09-04',
      version_ids: ['version-1'],
      pool_ids: ['pool-1'],
      groups: [],
      products: [
        {
          key: 'etf:159393.SZ',
          kind: 'etf',
          product_id: '159393.SZ',
          code: '159393.SZ',
          name: '万家沪深300ETF',
          evaluation_plan_id: 'plan-etf',
          evaluation_plan_revision: 1,
          evaluation_plan_name: 'ETF评价',
          usage_status: 'normal',
          max_weight: 0.6,
          valid_until: null,
          substitute_group: '',
          reasons: ['通过'],
          source_version_ids: ['version-1'],
          source_pool_ids: ['pool-1'],
        },
        {
          key: 'fund:024011.OF',
          kind: 'fund',
          product_id: '024011.OF',
          code: '024011.OF',
          name: '万家沪深300ETF联接-A',
          evaluation_plan_id: 'plan-fund',
          evaluation_plan_revision: 1,
          evaluation_plan_name: '联接基金评价',
          usage_status: 'normal',
          max_weight: 0.5,
          valid_until: null,
          substitute_group: '',
          reasons: ['通过'],
          source_version_ids: ['version-1'],
          source_pool_ids: ['pool-1'],
        },
      ],
      product_count: 2,
      created_at: '2026-09-04T00:00:00Z',
      immutable: true,
    })

    render(
      <MemoryRouter initialEntries={['/manual-construction?universe=universe-legacy']}>
        <ManualConstruction />
      </MemoryRouter>,
    )

    const universeName = await screen.findByText('旧结构可投资域')
    expect(universeName.closest('div')).toHaveTextContent('2 只可用产品')
    expect(screen.getByRole('heading', { name: '资产大类构建模块' })).toBeInTheDocument()
    await waitFor(() => expect(searchInvestableUniverseProducts).toHaveBeenCalledWith(
      'universe-legacy',
      expect.objectContaining({ eligibleOnly: true }),
    ))
  })

  it('产品名称进入对应类别的产品研究，并可对比同一大类的全部产品', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}>
        <Routes>
          <Route path="/manual-construction" element={<ManualConstruction />} />
          <Route path="/product-research/compare" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>,
    )

    await screen.findByText(/已锁定可投资域/)
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[0])
    await user.click(await screen.findByRole('button', { name: /159393\.SZ.*万家沪深300ETF.*添加/ }))
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[0])
    await user.click(await screen.findByRole('button', { name: /024011\.OF.*万家沪深300ETF联接-A.*添加/ }))

    const etfLink = await screen.findByRole('link', { name: '万家沪深300ETF' })
    const fundLink = screen.getByRole('link', { name: '万家沪深300ETF联接-A' })
    expect(etfLink).toHaveAttribute('href', '/product-research/products/159393.SZ?kind=etf')
    expect(fundLink).toHaveAttribute('href', '/product-research/products/024011.OF?kind=fund')

    await act(async () => {
      await user.click(screen.getByRole('button', { name: '产品对比（2）' }))
    })
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent(
      '/product-research/compare|159393.SZ,024011.OF|etf,fund|/manual-construction?universe=universe-1|返回手动构建大类',
    ))
  })

  it('滚动相关性响应缺少固定签名 NJIT 证明时失败关闭', async () => {
    const execution = {
      backend: 'numba_njit_fixed_signature',
      execution_backend: 'numba_njit_fixed_signature',
      kernel_version: 'test',
      kernel_coverage: 'equal_weights',
      kernel_signatures: { equal_weights_kernel: ['(int64,)'] },
      fingerprint: 'test',
      nopython: true as const,
      njit_required: true as const,
      object_mode: 0 as const,
      python_fallback: 0 as const,
      request_time_compilation: 0 as const,
    }
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/list-allocations') return { ok: true, json: async () => [] } as Response
      if (url === '/api/strategy/equal-weights') return { ok: true, status: 200, json: async () => ({ weights: [], execution }) } as Response
      if (url === '/api/rolling-corr-classes') return { ok: true, status: 200, json: async () => ({ dates: [], series: {}, metrics: [] }) } as Response
      return { ok: false, status: 404, json: async () => ({}) } as Response
    })
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)

    const emptyOption = await screen.findByRole('option', { name: '请选择大类' })
    await user.selectOptions(emptyOption.parentElement as HTMLSelectElement, '权益类')
    await user.click(screen.getByRole('button', { name: '计算滚动相关性' }))

    await waitFor(() => expect(alertSpy).toHaveBeenCalledWith(
      '滚动相关性计算失败：大类滚动相关性未提供有效的固定签名 NJIT 执行证明',
    ))
  })
})
