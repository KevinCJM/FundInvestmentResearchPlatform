import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import AutoAssetClassification from './AutoAssetClassification'
import { getInvestableUniverse, searchInvestableUniverseProducts } from '../services/productPools'
import { ResearchContextProvider } from '../app/ResearchContext'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/productPools', () => ({
  getInvestableUniverse: vi.fn(),
  searchInvestableUniverseProducts: vi.fn(),
  investableUniverseEligibleCount: (snapshot: any) => snapshot?.summary?.eligible_count ?? 0,
}))

const META = {
  algorithms: [
    { id: 'hierarchical', label: '相关性层次聚类' },
    { id: 'kmedoids', label: 'K-medoids（代表产品）' },
    { id: 'rule', label: '合同分类规则映射' },
    { id: 'gmm', label: '高斯混合（软分配）' },
  ],
  features: [
    { id: 'correlation', label: '收益相关性距离' },
    { id: 'denoised', label: '去噪相关性距离（RMT）' },
    { id: 'pca', label: '主成分载荷' },
  ],
  linkages: [{ id: 'average', label: 'average' }, { id: 'ward', label: 'ward' }],
  weight_modes: [{ id: 'inv_vol', label: '逆波动率' }, { id: 'equal', label: '等权' }],
  taxonomy_levels: [
    { id: 'asset_class', label: '一级·资产大类' },
    { id: 'category', label: '二级·细分类型' },
    { id: 'detail', label: '三级·风格/行业/主题' },
  ],
  block_modes: [
    { id: 'none', label: '不分层（纯统计聚类）' },
    { id: 'asset_class', label: '一级·资产大类' },
    { id: 'category', label: '二级·细分类型' },
  ],
  limits: { min_observations: 60, max_auto_k: 8, min_products: 2 },
  defaults: {},
}

const SEARCH_ITEMS = [
  { code: '510300.SH', name: '华泰柏瑞沪深300ETF', instrument_type: 'etf' },
  { code: '511010.SH', name: '国泰上证5年期国债ETF', instrument_type: 'etf' },
]

const PREVIEW = {
  algorithm: 'hierarchical',
  features: 'correlation',
  taxonomy_level: 'asset_class',
  block_by: 'none',
  k: 2,
  weight_mode: 'inv_vol',
  observations: 1200,
  start_date: '2020-01-02',
  end_date: '2026-09-01',
  classes: [
    {
      id: 'auto-0',
      name: '权益类',
      size: 1,
      medoid: '510300.SH',
      silhouette: 0.72,
      mean_corr: 0.88,
      max_class_weight: 100,
      etfs: [{
        code: '510300.SH', name: '华泰柏瑞沪深300ETF', weight: 100, instrument_type: 'etf',
        fund_type: '股票型', invest_type: '被动指数型', management: '华泰柏瑞',
        contract_label: '权益类', affinity: -0.2, is_medoid: true, max_weight: null, capped: false,
        taxonomy: { asset_class: '权益类', category: '宽基规模', detail: '大盘', path: '权益类 / 宽基规模 / 大盘', matched: '沪深300' },
      }],
    },
    {
      id: 'auto-1',
      name: '固收类',
      size: 1,
      medoid: '511010.SH',
      silhouette: 0.61,
      mean_corr: null,
      max_class_weight: 25,
      etfs: [{
        code: '511010.SH', name: '国泰上证5年期国债ETF', weight: 100, instrument_type: 'etf',
        fund_type: '债券型', invest_type: '被动指数型', management: '国泰',
        contract_label: '固收类', affinity: -0.3, is_medoid: true, max_weight: 0.1, capped: true,
        taxonomy: { asset_class: '固收类', category: '利率债', detail: '国债', path: '固收类 / 利率债 / 国债', matched: '国债' },
      }],
    },
  ],
  unassigned: [{ code: '512000.SH', name: '证券ETF', reason: 'CAPACITY', detail: '所有候选大类已达上限或亲和度不足' }],
  skipped: [],
  warnings: ['1 个产品未归类，已进入待观察池'],
  diagnostics: {
    silhouette: 0.665,
    cross_class_corr: [[1, 0.1], [0.1, 1]],
    cross_class_labels: ['权益类', '固收类'],
    significant_eigenvalues: 2,
    k_suggestions: [{ k: 2, silhouette: 0.665 }, { k: 3, silhouette: 0.4 }],
    blocks: [],
    contract_deviations: [{ code: '512000.SH', name: '证券ETF', assigned_class: '固收类', contract_label: '权益类' }],
    winsorized: [{ code: '511010.SH', name: '国泰上证5年期国债ETF', clipped: 1, max_raw_return: -0.99 }],
  },
  execution: {
    execution_backend: 'numba_njit_fixed_signature',
    nopython: true,
    object_mode: 0,
    python_fallback: 0,
    request_time_compilation: 0,
    kernel_signatures: { correlation_matrix_kernel: ['(array(float64, 2d, C),)'] },
  },
}

const FIT = {
  dates: ['2020-01-02', '2020-01-03'],
  navs: { 权益类: [1, 1.01], 固收类: [1, 1.001] },
  corr: [[1, 0.1], [0.1, 1]],
  corr_labels: ['权益类', '固收类'],
  metrics: [
    { name: '权益类', cumulative_return: 0.2, annual_return: 0.1, annual_vol: 0.18, sharpe: 0.6, var99: 0.03, es99: 0.04, max_drawdown: -0.2, calmar: 0.5 },
    { name: '固收类', cumulative_return: 0.05, annual_return: 0.02, annual_vol: 0.01, sharpe: 1.2, var99: 0.001, es99: 0.002, max_drawdown: -0.01, calmar: 2.0 },
  ],
  consistency: [{ name: '权益类', mean_corr: 0.88, pca_evr1: 0.91, max_te: 0.02 }],
  annual_metrics: { years: [], series: {} },
  // /api/fit-classes proves two lanes; the old flat mock never matched the
  // endpoint, which is why the page failed only in the browser.
  execution: {
    fit_analytics: PREVIEW.execution,
    performance_metrics: {
      execution_backend: 'numba_njit_fixed_signature',
      nopython: true,
      object_mode: 0,
      python_fallback: 0,
      request_time_compilation: 0,
      kernel_signatures: { annual_metrics_kernel: ['(array(float64, 2d, C), array(int64, 1d, C), float64)'] },
    },
  },
}

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname}</div>
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/pre-investment/saa/auto-classification?universe=universe-1']}>
      <ResearchContextProvider>
        <Routes>
          <Route path="/pre-investment/saa/auto-classification" element={<AutoAssetClassification />} />
          <Route path="*" element={<LocationProbe />} />
        </Routes>
      </ResearchContextProvider>
    </MemoryRouter>,
  )
}

let previewBody: any = null
let fitBody: any = null
let previewResponse: { ok: boolean; status: number; payload: any } = { ok: true, status: 200, payload: PREVIEW }

beforeEach(() => {
  previewBody = null
  fitBody = null
  previewResponse = { ok: true, status: 200, payload: PREVIEW }
  vi.mocked(getInvestableUniverse).mockResolvedValue({
    id: 'universe-1', name: '测试可投资域', research_date: '2026-09-04', version_refs: [], members: [],
    summary: { pool_count: 1, member_count: 2, eligible_count: 2, restricted_count: 0, watch_count: 0 },
    content_hash: 'universe-hash', created_at: '2026-09-04T00:00:00Z', immutable: true,
  })
  vi.mocked(searchInvestableUniverseProducts).mockResolvedValue({
    snapshot_id: 'universe-1', snapshot_name: '测试可投资域', research_date: '2026-09-04',
    total: SEARCH_ITEMS.length, page: 1, page_size: 100,
    items: SEARCH_ITEMS.map((item, index) => ({
      kind: 'etf' as const,
      product_id: item.code,
      code: item.code,
      name: item.name,
      research_status: 'approved' as const,
      usage_status: 'normal' as const,
      decision_reasons: ['通过'],
      max_weight: 0.6,
      valid_until: null,
      substitute_groups: [],
      evaluation_sources: [{
        pool_id: 'pool-1', pool_name: '核心池', pool_version_id: 'version-1', pool_version_number: 1,
        evaluation_plan_id: 'plan-1', evaluation_plan_revision: 1, evaluation_plan_name: 'ETF评价',
        source_rank: index + 1, source_score: 90 - index,
      }],
      eligible: true,
      eligibility_reasons: [],
      warnings: [],
    })),
  })
  vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url === '/api/asset-classes/auto/meta') {
      return { ok: true, status: 200, json: async () => META }
    }
    if (url === '/api/asset-classes/auto/preview') {
      previewBody = JSON.parse(String(init?.body))
      return { ok: previewResponse.ok, status: previewResponse.status, json: async () => previewResponse.payload }
    }
    if (url === '/api/fit-classes') {
      fitBody = JSON.parse(String(init?.body))
      return { ok: true, status: 200, json: async () => FIT }
    }
    if (url === '/api/save-allocation') {
      return { ok: true, status: 200, json: async () => ({ ok: true }) }
    }
    return { ok: false, status: 404, json: async () => ({}) }
  }))
  const store = new Map<string, string>()
  vi.stubGlobal('sessionStorage', {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => { store.set(key, value) },
    removeItem: (key: string) => { store.delete(key) },
    clear: () => { store.clear() },
  })
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function addTwoProducts(user: ReturnType<typeof userEvent.setup>) {
  const selectAll = await screen.findByLabelText('全选当前结果')
  await waitFor(() => expect(selectAll).not.toBeDisabled())
  await user.click(selectAll)
  await waitFor(() => expect(screen.getByText('已选产品（2）')).toBeInTheDocument())
}

describe('AutoAssetClassification', () => {
  it('sends the contract taxonomy options and renders the block summary', async () => {
    previewResponse = {
      ok: true,
      status: 200,
      payload: {
        ...PREVIEW,
        block_by: 'asset_class',
        diagnostics: {
          ...PREVIEW.diagnostics,
          blocks: [
            { block: '权益类', size: 12, k: 2, silhouette: 0.41 },
            { block: '固收类', size: 3, k: 1, silhouette: null },
          ],
        },
      },
    }
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.selectOptions(screen.getByLabelText('按合同分层（硬约束）'), 'asset_class')
    await user.selectOptions(screen.getByLabelText('合同分类层级'), 'category')
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))

    await waitFor(() => expect(previewBody).not.toBeNull())
    expect(previewBody.blockBy).toBe('asset_class')
    expect(previewBody.taxonomyLevel).toBe('category')

    const summary = await screen.findByText(/^合同分层（/)
    const table = summary.closest('div') as HTMLElement
    expect(within(table).getByText('权益类')).toBeInTheDocument()
    expect(within(table).getByText('0.410')).toBeInTheDocument()
    // A block that was not split reports no silhouette rather than a fake 0.
    expect(within(table).getByText('—')).toBeInTheDocument()
  })

  it('keeps the block summary hidden when the run was not layered', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await waitFor(() => expect(previewBody).not.toBeNull())
    expect(previewBody.blockBy).toBe('none')
    await screen.findByText('④ 大类映射草案')
    expect(screen.queryByText(/^合同分层（/)).not.toBeInTheDocument()
  })

  it('greys out the parameters the selected algorithm ignores', async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(screen.getByLabelText('算法')).toBeInTheDocument())

    // hierarchical is the only algorithm that reads a linkage method.
    expect(screen.getByLabelText('连接方式（层次聚类）')).not.toBeDisabled()
    expect(screen.getByLabelText('按合同分层（硬约束）')).not.toBeDisabled()
    expect(screen.getByLabelText('大类个数')).not.toBeDisabled()

    await user.selectOptions(screen.getByLabelText('算法'), 'gmm')
    expect(screen.getByLabelText('连接方式（层次聚类）')).toBeDisabled()
    expect(screen.getByText('仅层次聚类需要连接方式，该算法不读取此项')).toBeInTheDocument()
    // A mixture still clusters, so blocking and K stay live.
    expect(screen.getByLabelText('按合同分层（硬约束）')).not.toBeDisabled()
    expect(screen.getByLabelText('大类个数')).not.toBeDisabled()

    await user.selectOptions(screen.getByLabelText('算法'), 'rule')
    expect(screen.getByLabelText('连接方式（层次聚类）')).toBeDisabled()
    expect(screen.getByLabelText('按合同分层（硬约束）')).toBeDisabled()
    expect(screen.getByLabelText('大类个数')).toBeDisabled()
    expect(screen.getByLabelText('自动建议大类个数')).toBeDisabled()
    expect(screen.getByText('规则映射本身就按合同分类，再分层没有意义')).toBeInTheDocument()
    expect(screen.getByText('规则映射的大类个数由合同标签自然决定')).toBeInTheDocument()
    // The hint panel must not explain a control the user cannot reach.
    expect(screen.queryByText('分层')).not.toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('算法'), 'hierarchical')
    expect(screen.getByLabelText('连接方式（层次聚类）')).not.toBeDisabled()
    expect(screen.getByLabelText('按合同分层（硬约束）')).not.toBeDisabled()
  })

  it('refuses to run before enough products are selected', async () => {
    const user = userEvent.setup()
    renderPage()
    await waitFor(() => expect(screen.getByText('已选产品（0）')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    expect(await screen.findByText('请至少选择 2 个产品')).toBeInTheDocument()
    expect(previewBody).toBeNull()
  })

  it('rejects an inverted size range without calling the backend', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    // Controlled numeric inputs snap to their minimum when emptied, so set the
    // value directly instead of clear-then-type.
    fireEvent.change(screen.getByLabelText('每类最多产品数'), { target: { value: '1' } })
    fireEvent.change(screen.getByLabelText('每类最少产品数'), { target: { value: '3' } })
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    expect(await screen.findByText('每类最多产品数不能小于最少产品数')).toBeInTheDocument()
    expect(previewBody).toBeNull()
  })

  it('sends the selected parameters and renders classes, diagnostics and NAV metrics', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.selectOptions(screen.getByLabelText('类内权重'), 'equal')
    fireEvent.change(screen.getByLabelText('大类个数'), { target: { value: '2' } })
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))

    await waitFor(() => expect(previewBody).not.toBeNull())
    expect(previewBody.products.map((item: any) => item.code)).toEqual(['510300.SH', '511010.SH'])
    expect(previewBody.products.every((item: any) => item.kind === 'etf')).toBe(true)
    expect(previewBody).toMatchObject({ universe_snapshot_id: 'universe-1', algorithm: 'hierarchical', features: 'correlation', k: 2, weightMode: 'equal', unassignedPolicy: 'park' })

    expect(await screen.findByText('④ 大类映射草案')).toBeInTheDocument()
    const equityCard = screen.getByRole('heading', { level: 3, name: '权益类' }).closest('div')?.parentElement as HTMLElement
    expect(within(equityCard).getByText('华泰柏瑞沪深300ETF', { exact: false })).toBeInTheDocument()
    expect(within(equityCard).getByText('100.00')).toBeInTheDocument()
    expect(within(equityCard).getByTitle('代表产品')).toBeInTheDocument()

    // Diagnostics
    expect(screen.getByText('0.665')).toBeInTheDocument()
    expect(screen.getByText('K=2：0.665')).toBeInTheDocument()
    expect(screen.getByText('· 1 个产品未归类，已进入待观察池')).toBeInTheDocument()
    expect(screen.getByText(/合同「权益类」→ 归入「固收类」/)).toBeInTheDocument()
    expect(screen.getByText(/复权净值疑似异常/)).toBeInTheDocument()

    // The fit call reuses the generated classes and their intra-class weights.
    await waitFor(() => expect(fitBody).not.toBeNull())
    expect(fitBody.classes).toEqual([
      { id: 'auto-0', name: '权益类', etfs: [{ code: '510300.SH', name: '华泰柏瑞沪深300ETF', weight: 100 }] },
      { id: 'auto-1', name: '固收类', etfs: [{ code: '511010.SH', name: '国泰上证5年期国债ETF', weight: 100 }] },
    ])
    expect(await screen.findByText('⑤ 大类净值与指标')).toBeInTheDocument()
    expect(screen.getByText('横向指标对比')).toBeInTheDocument()
    expect(screen.getByText('⑥ 同类资产一致性')).toBeInTheDocument()
  })

  it('sends k as null when automatic K is requested', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByLabelText('自动建议大类个数'))
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await waitFor(() => expect(previewBody).not.toBeNull())
    expect(previewBody.k).toBeNull()
  })

  it('surfaces a backend rejection instead of rendering a stale draft', async () => {
    const user = userEvent.setup()
    previewResponse = { ok: false, status: 400, payload: { detail: '共同样本仅 12 个交易日，少于 60 日的最低要求' } }
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    expect(await screen.findByText('共同样本仅 12 个交易日，少于 60 日的最低要求')).toBeInTheDocument()
    expect(screen.queryByText('④ 大类映射草案')).not.toBeInTheDocument()
  })

  it('hands the draft to the manual workspace through session storage', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await screen.findByText('④ 大类映射草案')
    await user.click(screen.getByRole('button', { name: '在手动构建大类中打开' }))

    const handoff = JSON.parse(String(window.sessionStorage.getItem('autoClassificationImport')))
    expect(handoff.classes).toHaveLength(2)
    expect(handoff.classes[0]).toMatchObject({ name: '权益类', mode: 'custom' })
    expect(handoff.classes[0].etfs[0]).toMatchObject({ code: '510300.SH', weight: 100, instrument_type: 'etf' })
    expect(await screen.findByTestId('location-probe')).toHaveTextContent('/pre-investment/saa/asset-classes')
    expect(handoff.universe_snapshot_id).toBe('universe-1')
  })

  it('requires a name before saving the allocation', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await screen.findByText('④ 大类映射草案')
    await user.click(screen.getByRole('button', { name: '保存为大类配置' }))
    expect(await screen.findByText('配置名称不能为空')).toBeInTheDocument()
    await user.type(screen.getByLabelText('大类配置名称'), '自动五大类')
    await user.click(screen.getByRole('button', { name: '保存为大类配置' }))
    expect(await screen.findByText(/配置「自动五大类」已保存/)).toBeInTheDocument()
  })

  it('surfaces the product-pool limit and the resulting class capacity', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await screen.findByText('④ 大类映射草案')

    const bondCard = screen.getByRole('heading', { level: 3, name: '固收类' }).closest('div')?.parentElement as HTMLElement
    // The pool restriction and the fact that it bound must both be visible.
    expect(within(bondCard).getByText('10.0%')).toBeInTheDocument()
    expect(within(bondCard).getByTitle('已被产品池限额压低')).toBeInTheDocument()
    // A class whose limits cannot fill it must state the SAA ceiling.
    expect(within(bondCard).getByText(/成员限额合计 25\.0%/)).toBeInTheDocument()

    // The unrestricted class shows no limit and no ceiling note.
    const equityCard = screen.getByRole('heading', { level: 3, name: '权益类' }).closest('div')?.parentElement as HTMLElement
    expect(within(equityCard).queryByTitle('已被产品池限额压低')).not.toBeInTheDocument()
    expect(within(equityCard).queryByText(/成员限额合计/)).not.toBeInTheDocument()
  })

  it('warns that a class holding an anomalous NAV series has untrustworthy metrics', async () => {
    const user = userEvent.setup()
    renderPage()
    await addTwoProducts(user)
    await user.click(screen.getByRole('button', { name: '运行自动分类' }))
    await screen.findByText('⑤ 大类净值与指标')

    // /api/fit-classes runs on the raw series, so the class holding 511010.SH
    // must be called out next to the numbers it corrupts.
    const notice = screen.getByText('以下大类的净值与指标不可直接采信').closest('div') as HTMLElement
    expect(within(notice).getByText(/固收类/)).toBeInTheDocument()
    expect(within(notice).getByText(/511010\.SH/)).toBeInTheDocument()
    expect(within(notice).getByText(/-99\.0%/)).toBeInTheDocument()
    // The unaffected class must not be dragged into the warning.
    expect(within(notice).queryByText(/权益类/)).not.toBeInTheDocument()
  })

  it('does not expose an arbitrary-code bypass outside the investable universe', async () => {
    renderPage()
    expect(await screen.findByText(/已锁定可投资域/)).toBeInTheDocument()
    expect(screen.queryByLabelText('批量粘贴代码')).not.toBeInTheDocument()
    expect(screen.getByText(/不允许粘贴任意代码/)).toBeInTheDocument()
  })
})
