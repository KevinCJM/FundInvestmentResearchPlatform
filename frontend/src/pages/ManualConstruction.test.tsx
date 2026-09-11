import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ManualConstruction from './ManualConstruction'
import { useResearchDay } from '../app/ResearchContext'
import { readAllocationDraft } from '../app/allocationJourney'
import { evaluateNumericControls } from '../services/businessNumeric'
import { getInvestableUniverse, searchInvestableUniverseProducts } from '../services/productPools'

vi.mock('../app/ResearchContext', () => ({ useResearchDay: vi.fn(() => null) }))
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
    vi.mocked(useResearchDay).mockReturnValue(null)
    localStorage.clear()
    sessionStorage.clear()
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
    await user.click(await screen.findByRole('button', { name: /159393\.SZ.*万家沪深300ETF.*选择/ }))
    await user.click(screen.getByRole('button', { name: '加入所选产品（1）' }))
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[0])
    await user.click(await screen.findByRole('button', { name: /024011\.OF.*万家沪深300ETF联接-A.*选择/ }))
    await user.click(screen.getByRole('button', { name: '加入所选产品（1）' }))

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

  it('拟合时报出所用产品池，并把口径印在结果上', async () => {
    // 后端要靠这个 id 才能判断"这个池子是不是事后筛出来的"；页面上一直显示着
    // 研究日期，却从来没往请求里放过。
    const pit = {
      as_of: '2021-09-01',
      as_of_applied: true,
      run_mode: 'RESEARCH' as const,
      availability_field: 'ann_date',
      rows_before_cut: 100,
      rows_after_cut: 90,
      rows_dropped_by_as_of: 10,
      rows_without_announcement: 0,
      announcement_fallback: false,
      warnings: [],
      universe: {
        source: 'investable_universe_snapshot',
        replayable: false,
        established_at: '2026-09-04',
        clean: false,
        findings: [{ code: 'UNIVERSE_LOOKAHEAD', label: '可投资域', message: '该决策带入了未来信息。' }],
      },
    }
    const bodies: Record<string, any> = {}
    vi.mocked(fetch).mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/list-allocations') return { ok: true, json: async () => [] } as Response
      if (url === '/api/strategy/equal-weights') {
        const count = JSON.parse(String(init?.body)).asset_count as number
        return {
          ok: true,
          status: 200,
          json: async () => ({
            weights: Array.from({ length: count }, () => (count ? 100 / count : 0)),
            execution: {
              backend: 'numba_njit_fixed_signature', execution_backend: 'numba_njit_fixed_signature',
              kernel_version: 'test', kernel_coverage: 'equal_weights',
              kernel_signatures: { equal_weights_kernel: ['(int64,)'] }, fingerprint: 'test',
              nopython: true, njit_required: true, object_mode: 0, python_fallback: 0,
              request_time_compilation: 0,
            },
          }),
        } as Response
      }
      if (url === '/api/save-allocation') {
        bodies[url] = JSON.parse(String(init?.body))
        return { ok: true, status: 200, json: async () => ({ ok: true, message: 'saved' }) } as Response
      }
      if (url === '/api/fit-classes') {
        bodies[url] = JSON.parse(String(init?.body))
        return {
          ok: true,
          status: 200,
          json: async () => ({
            dates: ['2021-01-04'], navs: { 权益类: [1] }, corr: [[1]], corr_labels: ['权益类'],
            metrics: [{ name: '权益类' }], consistency: [], annual_metrics: { years: [], rows: [] },
            execution: {
              execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0,
              python_fallback: 0, request_time_compilation: 0,
              kernel_signatures: { class_nav_corr_metrics_kernel: ['(Array(float64, 2, C), float64)'] },
            },
            pit,
          }),
        } as Response
      }
      return { ok: false, status: 404, json: async () => ({}) } as Response
    })
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => undefined)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)

    await screen.findByText(/已锁定可投资域/)
    // 默认的「权益类」是自定义权重且没有产品，合计不到 100% 会先被挡下来。
    await user.click(screen.getAllByRole('button', { name: '等权重' })[0])
    await act(async () => {
      await user.click(screen.getByRole('button', { name: '拟合大类收益率' }))
    })
    expect(alertSpy).not.toHaveBeenCalled()

    await waitFor(() => expect(bodies['/api/fit-classes']?.universe_snapshot_id).toBe('universe-1'))
    const note = await screen.findByTestId('pit-provenance')
    expect(note).toHaveTextContent('研究日 2021-09-01')
    expect(screen.getByTestId('pit-universe-finding')).toHaveTextContent('带入了未来信息')

    // 保存那条路径的字段后端早就接了，只是前端从来没传，落盘的血缘一直是 None。
    await user.click(screen.getByRole('button', { name: '保存当前大类配置' }))
    await user.type(screen.getByPlaceholderText('请输入配置名称...'), '手动大类-测试')
    await act(async () => {
      await user.click(screen.getByRole('button', { name: '保存' }))
    })
    await waitFor(() => expect(bodies['/api/save-allocation']?.universe_snapshot_id).toBe('universe-1'))
    expect(screen.getByRole('link', { name: /继续 SAA：手动大类-测试/ })).toHaveAttribute('href', '/pre-investment/saa/allocation-lab?alloc=%E6%89%8B%E5%8A%A8%E5%A4%A7%E7%B1%BB-%E6%B5%8B%E8%AF%95&universe=universe-1')
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

    await user.click(screen.getByText('滚动相关性研究', { selector: 'summary' }))
    const emptyOption = await screen.findByRole('option', { name: '请选择大类' })
    await user.selectOptions(emptyOption.parentElement as HTMLSelectElement, '权益类')
    await user.click(screen.getByRole('button', { name: '计算滚动相关性' }))

    await waitFor(() => expect(alertSpy).toHaveBeenCalledWith(
      '滚动相关性计算失败：大类滚动相关性未提供有效的固定签名 NJIT 执行证明',
    ))
  })
  it('一次批量加入产品，并在其他大类明确阻止重复归属', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)
    await screen.findByText(/已锁定可投资域/)
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[0])
    await user.click(await screen.findByRole('button', { name: /159393\.SZ.*选择/ }))
    await user.click(screen.getByRole('button', { name: /024011\.OF.*选择/ }))
    await user.click(screen.getByRole('button', { name: '加入所选产品（2）' }))
    expect(screen.getByRole('button', { name: '产品对比（2）' })).toBeInTheDocument()
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[1])
    expect(screen.getByRole('button', { name: /159393\.SZ.*已属于权益类/ })).toBeDisabled()
    expect(screen.getByRole('button', { name: /024011\.OF.*已属于权益类/ })).toBeDisabled()
  })

  it('单产品默认100%，重载恢复输入，切换范围不带入旧产品', async () => {
    const user = userEvent.setup()
    const first = render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)
    await screen.findByText(/已锁定可投资域/)
    await user.click(screen.getAllByRole('button', { name: '+ 添加新的产品' })[0])
    await user.click(await screen.findByRole('button', { name: /159393\.SZ.*选择/ }))
    await user.click(screen.getByRole('button', { name: '加入所选产品（1）' }))
    expect(screen.getByLabelText('权益类 万家沪深300ETF 类内权重')).toHaveValue(100)
    first.unmount()
    const second = render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)
    await screen.findByText(/已锁定可投资域/)
    expect(screen.getByLabelText('权益类 万家沪深300ETF 类内权重')).toHaveValue(100)
    second.unmount()
    render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-2']}><ManualConstruction /></MemoryRouter>)
    await screen.findByText(/已锁定可投资域/)
    expect(screen.queryByRole('link', { name: '万家沪深300ETF' })).not.toBeInTheDocument()
    expect(readAllocationDraft<{classes: {etfs: unknown[]}[]}>('classes:universe-1')?.classes[0].etfs).toHaveLength(1)
  })

  it('开始日晚于平台知识截止时，在工作区提示修正并保留输入', async () => {
    vi.mocked(useResearchDay).mockReturnValue('2014-12-31')
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/manual-construction?universe=universe-1']}><ManualConstruction /></MemoryRouter>)
    await screen.findByText(/已锁定可投资域/)
    await user.click(screen.getByRole('button', { name: '拟合大类收益率' }))
    expect(screen.getByRole('alert')).toHaveTextContent('开始日期 2020-01-01 晚于平台知识截止 2014-12-31')
    expect(screen.getByLabelText('选择开始日期')).toHaveValue('2020-01-01')
    expect(vi.mocked(fetch).mock.calls.some(([url]) => url === '/api/fit-classes')).toBe(false)
  })

})
