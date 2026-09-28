import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ProductPoolWorkspacePage from './ProductPoolWorkspacePage'
import ScopeMandateSummary from '../components/strategic-scope/ScopeMandateSummary'
import { readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'
import {
  createInvestableUniverseSnapshot,
  bindInvestableUniverseMandate,
  getInvestableUniverse,
  listInvestableUniverseSnapshots,
  listProductPoolVersions,
} from '../services/productPools'
import { bindUniverseMandate, confirmUniverse, getStrategicUniverse, getScopeFeasibility, previewUniverse } from '../services/strategicScope'
import { getMandate, getStrategicCatalog } from '../services/strategicAllocation'
import { mandateVersion, strategicCatalog } from '../test/strategicAllocationFixtures'
import { boundaryAssessment } from '../test/mandateBoundaryFixtures'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day, useResearchContextIdentity: () => JSON.stringify(clock) }))

vi.mock('../services/productPools', async () => {
  // Keep the pure helpers real: the page prints a version's data cut-off next
  // to the research day, and a stubbed-out helper would hide that regression.
  const actual = await vi.importActual<typeof import('../services/productPools')>(
    '../services/productPools',
  )
  return {
    ...actual,
    createInvestableUniverseSnapshot: vi.fn(),
    bindInvestableUniverseMandate: vi.fn(),
    getInvestableUniverse: vi.fn(),
    listInvestableUniverseSnapshots: vi.fn(),
    listProductPoolVersions: vi.fn(),
  }
})

vi.mock('../services/strategicScope', async () => {
  const actual = await vi.importActual<typeof import('../services/strategicScope')>(
    '../services/strategicScope',
  )
  return { ...actual, bindUniverseMandate: vi.fn(), getStrategicUniverse: vi.fn(), getScopeFeasibility: vi.fn(), previewUniverse: vi.fn(), confirmUniverse: vi.fn() }
})

const strategicScopeFixture = {
  id: 'scope-ui', name: '独立战略范围', content_hash: 'a'.repeat(64), created_at: '2026-09-12',
  preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped', implementation_gaps: ['growth'],
  definition: {
    name: '独立战略范围', as_of: '2026-09-12', currency: 'CNY', source: '',
    assets: [{ id: 'growth', name: '增长资产', currency: 'CNY', role: 'growth', liquidity: 'liquid', rationale: '', source: '' }],
  },
} as never

vi.mock('../services/strategicAllocation', async () => {
  const actual = await vi.importActual<typeof import('../services/strategicAllocation')>(
    '../services/strategicAllocation',
  )
  return { ...actual, getMandate: vi.fn(), getStrategicCatalog: vi.fn() }
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

const summary = {
  id: 'saved-1', name: '已保存范围甲', research_date: '2026-09-04', version_ids: ['version-1'],
  pool_ids: ['pool-1'], product_count: 1, content_hash: 'h'.repeat(8), immutable: true,
  created_at: '2026-09-04T00:00:00Z',
} as never

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname}{location.search}</div>
}

const enter = (search: string) => `/pre-investment/product-pool/new${search}`

describe('ProductPoolWorkspacePage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    clock.day = '2026-09-12'
    localStorage.clear()
    sessionStorage.clear()
    vi.mocked(listProductPoolVersions).mockResolvedValue({ items: [version], total: 1 })
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(bindUniverseMandate).mockImplementation(async (_id, mandate_id) => ({ mandate_id }))
    vi.mocked(bindInvestableUniverseMandate).mockImplementation(async (_id, mandate_id) => ({ mandate_id }))
    vi.mocked(getStrategicUniverse).mockImplementation(async id => ({ ...(strategicScopeFixture as object), id } as never))
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(universe)
    vi.mocked(getInvestableUniverse).mockResolvedValue(universe)
    vi.mocked(getStrategicCatalog).mockResolvedValue(strategicCatalog)
    vi.mocked(getScopeFeasibility).mockImplementation(async body => ({ status: 'undetermined', reason_code: 'DATA_MISSING',
      reasons: [{ code: 'DATA_MISSING', message: '初筛数据不足。' }], research_only: true,
      mandate: { id: body.mandate_id, content_hash: '', target_return: .03, volatility_cap: .15, min_cash_weight: 0 },
    }))
  })

  it.each(['product', 'strategic'])('%s 新建在配置内必选目标，选择后保存正确关联', async kind => {
    const isStrategic = kind === 'strategic'
    writeAllocationDraft('strategic-universe:editor:new:choose', { name: '现金研究范围', as_of: clock.day, currency: 'CNY', source: '', assets: [{ id: 'cash', name: '现金', currency: 'CNY', role: 'liquidity', liquidity: 'liquid', rationale: '', source: '', research_proxy: { asset_type: 'cash', cash_return: .01, components: [], source_labels: {}, rebalance: 'daily' } }] })
    vi.mocked(previewUniverse).mockResolvedValue({ preview_hash: 'preview' } as never)
    vi.mocked(confirmUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: 'mandate-1' } as never)
    render(<MemoryRouter initialEntries={[enter(`?new=choose${isStrategic ? '&scope=strategic' : ''}`)]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    await screen.findByRole('option', { name: /长期配置目标/ })
    const selector = screen.getByRole('combobox', { name: '投资目标与约束' })
    expect(selector).toBeRequired()
    expect(selector.closest('section')).toHaveTextContent(isStrategic ? '配置大类资产与研究代理' : '选择产品池版本')
    if (!isStrategic) fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    const save = screen.getByRole('button', { name: isStrategic ? '保存战略范围' : '生成锁定快照' })
    expect(save).toBeDisabled()
    fireEvent.change(selector, { target: { value: 'mandate-1' } })
    await waitFor(() => expect(save).toBeEnabled())
    expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-1')
    fireEvent.click(save)
    if (isStrategic) await waitFor(() => expect(confirmUniverse).toHaveBeenCalledWith(expect.anything(), 'preview', expect.any(AbortSignal), undefined, 'mandate-1'))
    else await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalledWith(expect.objectContaining({ mandate_id: 'mandate-1' })))
  })

  it.each(['product', 'strategic'])('%s 复制范围可显式选择另一个目标，已保存关联不被改写', async kind => {
    const other = { ...mandateVersion, id: 'mandate-B', name: '另一个目标' }
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, other] })
    vi.mocked(getStrategicUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: 'mandate-1' } as never)
    vi.mocked(getInvestableUniverse).mockResolvedValue({ ...universe, mandate_id: 'mandate-1' })
    render(<MemoryRouter initialEntries={[enter(kind === 'strategic' ? '?scope=strategic&strategic_universe=scope-ui&copy=1' : '?universe=universe-1&copy=universe-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    const selector = await screen.findByRole('combobox', { name: '投资目标与约束' })
    expect(selector).toBeEnabled()
    fireEvent.change(selector, { target: { value: 'mandate-B' } })
    expect(await screen.findByRole('heading', { name: '另一个目标' })).toBeVisible()
    expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-B')
    expect(bindUniverseMandate).not.toHaveBeenCalled()
    expect(bindInvestableUniverseMandate).not.toHaveBeenCalled()
  })

  it('产品范围保存期间切换目标，旧保存回包不能覆盖新选择', async () => {
    let finish!: (value: typeof universe) => void
    vi.mocked(createInvestableUniverseSnapshot).mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const other = { ...mandateVersion, id: 'mandate-B', name: '另一个目标' }
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, other] })
    render(<MemoryRouter initialEntries={[enter('?new=late&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalled())
    fireEvent.change(screen.getByRole('combobox', { name: '投资目标与约束' }), { target: { value: 'mandate-B' } })
    await act(async () => finish({ ...universe, mandate_id: 'mandate-1' }))
    expect(screen.getByRole('heading', { name: '另一个目标' })).toBeVisible()
    expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-B')
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('universe=')
    expect(screen.queryByRole('heading', { name: '已保存的产品范围' })).not.toBeInTheDocument()
  })

  it('产品选择后自动初筛，未保存也会携带已选版本与排除项；清空选择立即作废结论', async () => {
    writeAllocationDraft('pool:new:screening', { researchDate: '2026-09-12', name: '产品初筛', selectedIds: [], snapshotId: '', selectionEdited: true, excludedKeys: ['etf:excluded'] })
    render(<MemoryRouter initialEntries={[enter('?mandate=mandate-1&new=screening')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    await waitFor(() => expect(getScopeFeasibility).toHaveBeenCalledWith({ mandate_id: 'mandate-1', as_of: '2026-09-12',
      product_version_ids: ['version-1'], product_excluded_keys: ['etf:excluded'], window: { kind: '5Y' } }, expect.any(AbortSignal)))
    expect(createInvestableUniverseSnapshot).not.toHaveBeenCalled()
    await screen.findByText('初筛数据不足。')
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
    expect(screen.queryByText('初筛数据不足。')).not.toBeInTheDocument()
  })

  it('选择有效版本并锁定不可变可投资域', async () => {
    render(<MemoryRouter initialEntries={[enter('?mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))

    await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalledWith({
      name: '投前研究可投资域',
      mandate_id: 'mandate-1',
      research_date: expect.any(String),
      version_ids: ['version-1'],
    }))
    expect(await screen.findByRole('heading', { name: '已保存的产品范围' })).toBeInTheDocument()
    expect(screen.getByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '进入手动构建大类' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '进入自动构建大类' })).toBeInTheDocument()
  })

  it.each(['strategic', 'product'])('已保存 %s 范围不带目标参数也自动恢复绑定，编辑和刷新都无需重选', async kind => {
    const isStrategic = kind === 'strategic'
    vi.mocked(getStrategicUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: 'mandate-1' } as never)
    vi.mocked(getInvestableUniverse).mockResolvedValue({ ...universe, mandate_id: 'mandate-1' })
    const url = enter(isStrategic ? '?scope=strategic&strategic_universe=scope-ui&edit=1' : '?universe=universe-1&edit=universe-1')
    const first = render(<MemoryRouter initialEntries={[url]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    const panel = await screen.findByRole('region', { name: '当前投资目标与约束' })
    expect(await within(panel).findByRole('heading', { name: mandateVersion.name })).toBeVisible()
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-1'))
    expect(await screen.findByRole('button', { name: '保存修改' })).toBeEnabled()
    expect(screen.getByRole('combobox', { name: '投资目标与约束' })).toBeDisabled()
    expect(bindUniverseMandate).not.toHaveBeenCalled()
    expect(bindInvestableUniverseMandate).not.toHaveBeenCalled()
    first.unmount(); localStorage.clear(); sessionStorage.clear()
    render(<MemoryRouter initialEntries={[url]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await within(await screen.findByRole('region', { name: '当前投资目标与约束' })).findByRole('heading', { name: mandateVersion.name })).toBeVisible()
  })

  it('已绑定范围优先恢复原目标，不接受地址里误带的其他目标', async () => {
    const other = { ...mandateVersion, id: 'mandate-B', name: '其他目标' }
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, other] })
    vi.mocked(getStrategicUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: 'mandate-1' } as never)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui&edit=1&mandate=mandate-B')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    expect(await screen.findByRole('heading', { name: mandateVersion.name })).toBeVisible()
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-1'))
    expect(screen.queryByRole('heading', { name: '其他目标' })).not.toBeInTheDocument()
    expect(bindUniverseMandate).not.toHaveBeenCalled()
  })

  it('绑定目标已被替代时仍展示冻结约束，不换用最新目标', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [] })
    vi.mocked(getStrategicUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: mandateVersion.id, mandate_hash: mandateVersion.content_hash } as never)
    vi.mocked(getMandate).mockResolvedValue(mandateVersion)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui&edit=1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    const panel = await screen.findByRole('region', { name: '当前投资目标与约束' })
    expect(await within(panel).findByRole('heading', { name: mandateVersion.name })).toBeVisible()
    expect(within(panel).getByRole('status')).toHaveTextContent('该目标已停用或被新版本替代')
    expect(await screen.findByRole('button', { name: '保存修改' })).toBeDisabled()
    expect(getMandate).toHaveBeenCalledWith(mandateVersion.id, expect.any(AbortSignal))
  })

  it('旧范围只按明确目标链接补记，不用书签推测历史归属', async () => {
    updateAllocationJourney({ strategicUniverseId: 'scope-ui', mandateId: 'mandate-1' })
    const view = render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui&mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByRole('heading', { name: mandateVersion.name })).toBeVisible()
    expect(bindUniverseMandate).toHaveBeenCalledWith('scope-ui', 'mandate-1', expect.any(AbortSignal))
    view.unmount(); vi.mocked(bindUniverseMandate).mockClear()
    updateAllocationJourney({ strategicUniverseId: 'scope-ui', mandateId: 'mandate-1' })
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByText(/投资目标与约束为必填/)).toBeVisible()
    expect(bindUniverseMandate).not.toHaveBeenCalled()
  })

  it('战略配置页展示 URL 选定目标，详情默认收起，不采用旧旅程目标', async () => {
    updateAllocationJourney({ mandateId: 'mandate-1' })
    const chosen = { ...mandateVersion, id: 'mandate-B', name: '教育资金目标', definition: {
      ...mandateVersion.definition, target_return: .05, effective_target_return: .06,
      min_cash_weight: .05, effective_cash_reserve_weight: .12, note: '为未来教育支出准备',
    } }
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, chosen] })
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&new=1&mandate=mandate-B')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    const panel = await screen.findByRole('region', { name: '当前投资目标与约束' })
    expect(await within(panel).findByRole('heading', { name: '教育资金目标' })).toBeVisible()
    expect(within(panel).queryByText('长期配置目标')).not.toBeInTheDocument()
    expect(within(panel).getByText('预期年化收益').nextElementSibling).toHaveTextContent('5.00%')
    expect(within(panel).getByText('12.00%')).toBeVisible()
    const summary = within(panel).getByText('查看目标与约束详情')
    expect(summary.closest('details')).not.toHaveAttribute('open')
    fireEvent.click(summary)
    expect(within(panel).getByText('6.00%')).toBeVisible()
    expect(within(panel).getByText('备注：为未来教育支出准备')).toBeVisible()
    expect(within(panel).getByRole('link', { name: '查看完整投资目标 →' })).toHaveAttribute('href', '/pre-investment/objectives/new?view=mandate-B')
    expect(screen.getByRole('link', { name: '← 返回研究范围库' }).compareDocumentPosition(panel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(panel.compareDocumentPosition(screen.getByRole('region', { name: '独立战略范围' })) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })

  it('失效目标显示重新选择提示，不展示其他目标的数据', async () => {
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&new=1&mandate=missing')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByText(/原先选择的投资目标与约束已不存在/)).toBeVisible()
    expect(screen.queryByRole('heading', { name: '长期配置目标' })).not.toBeInTheDocument()
    const selector = screen.getByRole('combobox', { name: '投资目标与约束' })
    expect(selector).toHaveValue('')
    fireEvent.change(selector, { target: { value: 'mandate-1' } })
    expect(await screen.findByRole('heading', { name: '长期配置目标' })).toBeVisible()
  })

  it('目标目录读取失败时在配置内重试，恢复后再展示所选目标', async () => {
    vi.mocked(getStrategicCatalog).mockRejectedValue(new Error('目录暂不可用'))
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&new=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    const retry = await screen.findByRole('button', { name: '重试读取投资目标' })
    expect(screen.getByRole('combobox', { name: '投资目标与约束' })).toBeDisabled()
    vi.mocked(getStrategicCatalog).mockResolvedValue(strategicCatalog)
    fireEvent.click(retry)
    expect(await screen.findByRole('heading', { name: '长期配置目标' })).toBeVisible()
  })

  it('保存范围恢复期间目录失败必须显示错误与重试，不能出现空白工作区', async () => {
    vi.mocked(getStrategicCatalog).mockRejectedValue(new Error('目录暂不可用'))
    vi.mocked(getStrategicUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), mandate_id: 'mandate-1' } as never)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByRole('alert')).toHaveTextContent('目录暂不可用')
    vi.mocked(getStrategicCatalog).mockResolvedValue(strategicCatalog)
    fireEvent.click(screen.getByRole('button', { name: '重试读取投资目标' }))
    expect(await screen.findByRole('heading', { name: mandateVersion.name })).toBeVisible()
    expect(await screen.findByRole('combobox', { name: '投资目标与约束' })).toBeDisabled()
  })

  it('资金目标摘要使用已保存金额，展开后保留支出安排与购买力口径', () => {
    const assessment = boundaryAssessment()
    const version = { ...mandateVersion, definition: { ...assessment.definition,
      funding_target: { amount: 1_500_000, amount_basis: 'real' as const },
      cash_budget: { ...assessment.definition.cash_budget!, flows: [{ name: '学费', kind: 'withdrawal' as const, amount: 50_000, first_month: 12, last_month: 36, every_months: 12 as const }] },
    }, assessment }
    render(<MemoryRouter><ScopeMandateSummary mandate={version} loading={false} blockedReason="" backHref="/pre-investment/product-pool" researchDay={null} /></MemoryRouter>)
    expect(screen.getByText('1,500,000 CNY（当前购买力）')).toBeVisible()
    expect(screen.queryByText('预期年化收益')).not.toBeInTheDocument()
    fireEvent.click(screen.getByText('查看目标与约束详情'))
    expect(screen.getByText(/学费：支出 50,000 CNY/)).toBeVisible()
    expect(screen.getByText('1,000,000 CNY')).toBeVisible()
    expect(screen.getByText('4.20%')).toBeVisible()
  })

  it.each([
    [.020218343484683543, 'solved', '2.02%'],
    [0, 'solved', '0.00%'],
    [-1e-16, 'solved', '0.00%'],
    [-.02, 'solved', '-2.00%'],
    [5, 'above_search_bound', '超过计算上限'],
    [-.99, 'at_lower_bound', '低于或等于计算下限'],
    [null, 'solved', '未保存计算结果，请在 01 重新测算'],
  ] as const)('目录中的资金门槛 %s/%s 在详情收起时可见，求解边界不当作精确收益', (value, status, text) => {
    const assessment = boundaryAssessment()
    const version = { ...mandateVersion, definition: assessment.definition, assessment: undefined,
      funding_summary: { cashflow_required_return: value, cashflow_required_return_status: status,
        required_effective_return: value, root_status: status } }
    render(<MemoryRouter><ScopeMandateSummary mandate={version} loading={false} blockedReason="" backHref="/pre-investment/objectives" researchDay={null} /></MemoryRouter>)
    expect(screen.getByText('计划所需年化收益率')).toBeVisible()
    expect(within(screen.getByText('计划所需年化收益率').parentElement!).getByText(text, { exact: true })).toBeVisible()
    expect(screen.getByText(/它是完成计划的收益要求，不是市场收益预测/)).toBeVisible()
    expect(screen.getByText('查看目标与约束详情').closest('details')).not.toHaveAttribute('open')
    if (status !== 'solved') expect(screen.queryByText(status === 'above_search_bound' ? '500.00%' : '-99.00%')).not.toBeInTheDocument()
  })

  it('没有冻结资金计算的目标不能借用 target_return 或显示为零', () => {
    const version = { ...mandateVersion, definition: { ...boundaryAssessment().definition, target_return: .99 }, assessment: undefined }
    render(<MemoryRouter><ScopeMandateSummary mandate={version} loading={false} blockedReason="" backHref="/pre-investment/objectives" researchDay={null} /></MemoryRouter>)
    expect(screen.getByText('未保存计算结果，请在 01 重新测算')).toBeVisible()
    expect(screen.queryByText('99.00%')).not.toBeInTheDocument()
  })

  it('相对目标展示冻结基准与超额收益，未知风险不显示为零', () => {
    const version = { ...mandateVersion, definition: { ...mandateVersion.definition,
      objective_kind: 'benchmark_relative' as const, max_volatility: null,
      benchmark: { name: '稳健股债基准', alloc_name: '股债分类', weights: { equity: .4, bond: .6 }, target_excess_return: .015, max_tracking_error: .03 },
    } }
    render(<MemoryRouter><ScopeMandateSummary mandate={version} loading={false} blockedReason="" backHref="/pre-investment/product-pool" researchDay={null} /></MemoryRouter>)
    expect(screen.getByText('目标年化超额收益')).toBeVisible()
    expect(screen.getByText('1.50%')).toBeVisible()
    expect(screen.getByText('待确认')).toBeVisible()
    fireEvent.click(screen.getByText('查看目标与约束详情'))
    expect(screen.getByText('稳健股债基准')).toBeVisible()
    expect(screen.getByText('3.00%')).toBeVisible()
  })

  it('修改范围后忽略旧快照请求的迟到响应', async () => {
    let resolve!: (value: typeof universe) => void
    vi.mocked(createInvestableUniverseSnapshot).mockReturnValue(new Promise(done => { resolve = done }))
    render(<MemoryRouter initialEntries={[enter('?mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    const choice = await screen.findByRole('checkbox', { name: /核心产品池/ })
    fireEvent.click(choice)
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalledOnce())
    fireEvent.click(choice)
    await act(async () => resolve(universe))
    expect(screen.queryByText('可投资域快照已锁定')).not.toBeInTheDocument()
    expect(choice).not.toBeChecked()
    expect(readAllocationJourney().universeId).toBeUndefined()
  })

  it('目标链接进入产品流程后持久保留该目标，清除旧政策并跨快照和构建路由传递', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, { ...mandateVersion, id: 'mandate-B', name: '第二目标' }] })
    updateAllocationJourney({ mandateId: 'mandate-A', baselineId: 'policy-A', taaRunId: 'taa-A' })
    render(<MemoryRouter initialEntries={[enter('?mandate=mandate-B')]}><Routes>
      <Route path="/pre-investment/product-pool/new" element={<ProductPoolWorkspacePage />} />
      <Route path="*" element={<LocationProbe />} />
    </Routes></MemoryRouter>)
    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await screen.findByRole('heading', { name: '已保存的产品范围' })
    fireEvent.click(screen.getByRole('button', { name: '进入手动构建大类' }))
    expect(await screen.findByTestId('location-probe')).toHaveTextContent('/pre-investment/saa/asset-classes?universe=universe-1')
    expect(readAllocationJourney()).toMatchObject({ mandateId: 'mandate-B', universeId: 'universe-1' })
    expect(readAllocationJourney().baselineId).toBeUndefined()
    expect(readAllocationJourney().taaRunId).toBeUndefined()
  })

  // The hand-off used to point at a route that does not exist and at a query key
  // the target pages never read, so the locked universe was silently dropped and
  // the user landed back on the dashboard.
  it.each([
    ['进入自动构建大类', '/pre-investment/saa/auto-classification'],
    ['进入手动构建大类', '/pre-investment/saa/asset-classes'],
  ])('%s 携带已锁定的可投资域跳转到 %s', async (label, path) => {
    render(
      <MemoryRouter initialEntries={[enter('?mandate=mandate-1')]}>
        <Routes>
          <Route path="/pre-investment/product-pool/new" element={<ProductPoolWorkspacePage />} />
          <Route path="*" element={<LocationProbe />} />
        </Routes>
      </MemoryRouter>,
    )

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await screen.findByRole('heading', { name: '已保存的产品范围' })
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

    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

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
    render(<MemoryRouter initialEntries={[enter('?version=version-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    expect(screen.queryByRole('checkbox', { name: /核心产品池 · V2/ })).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: '显示历史发布版本' }))
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池 · V2/ }))
    expect(screen.getByRole('checkbox', { name: /核心产品池 · V3/ })).not.toBeChecked()
    expect(screen.getByRole('checkbox', { name: /核心产品池 · V2/ })).toBeChecked()
  })

  it('从明确产品范围恢复只读摘要，不套用之前的选池草稿', async () => {
    writeAllocationDraft('pool:resume', { name: '旧草稿', researchDate: '2020-01-01', selectedIds: ['old-version'], snapshotId: 'old-universe', selectionEdited: false })
    updateAllocationJourney({ universeId: 'old-universe', name: '旧研究', researchDate: '2020-01-01', poolVersionIds: ['old-version'] })
    render(<MemoryRouter initialEntries={[enter('?universe=universe-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    expect(await screen.findByRole('heading', { name: '已保存的产品范围' })).toBeInTheDocument()
    expect(getInvestableUniverse).toHaveBeenCalledWith('universe-1')
    expect(getInvestableUniverse).not.toHaveBeenCalledWith('old-universe')
    expect(screen.getByText('投前研究可投资域')).toBeInTheDocument()
    expect(screen.getAllByText('2026-09-04').length).toBeGreaterThan(0)
    expect(await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    expect(readAllocationJourney()).toEqual({ universeId: 'universe-1', name: '投前研究可投资域', researchDate: '2026-09-04', poolVersionIds: ['version-1'] })
  })

  it('基于已锁定范围生成新快照时切换到新身份，不把新快照存入原范围草稿', async () => {
    const next = { ...universe, id: 'universe-2', name: '新的研究范围' }
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(next)
    vi.mocked(getInvestableUniverse).mockImplementation(async id => id === next.id ? next : universe)
    render(<MemoryRouter initialEntries={[enter('?universe=universe-1&copy=universe-1&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)
    await screen.findByDisplayValue('投前研究可投资域（副本）')
    fireEvent.change(screen.getByLabelText('研究名称'), { target: { value: next.name } })
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent(/universe=universe-2/))
    expect(readAllocationDraft<{ snapshotId: string }>('pool:universe:universe-1')).toBeNull()
    expect(readAllocationDraft<{ snapshotId: string }>('pool:universe:universe-2')?.snapshotId).toBe('universe-2')
  })

  it('未选择投资目标与约束时阻止锁定快照并说明原因', async () => {
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByText(/核心产品池 · V3/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('checkbox', { name: /核心产品池/ }))
    const create = screen.getByRole('button', { name: '生成锁定快照' })
    expect(create).toBeDisabled()
    expect(create).toHaveAttribute('title', '投资目标与约束为必填：请先选择已发布的投资目标与约束，再锁定研究范围并继续后续研究。')
    expect(createInvestableUniverseSnapshot).not.toHaveBeenCalled()
  })

  it('缺少有效目标时，已锁定快照也不能进入大类构建', async () => {
    render(<MemoryRouter initialEntries={[enter('?universe=universe-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    await screen.findByRole('heading', { name: '已保存的产品范围' })

    expect(screen.getByRole('button', { name: '进入自动构建大类' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '进入手动构建大类' })).toBeDisabled()
  })

  it('失效的目标 ID 不能解锁锁定快照', async () => {
    render(<MemoryRouter initialEntries={[enter('?mandate=stale-id')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    const create = screen.getByRole('button', { name: '生成锁定快照' })
    expect(create).toBeDisabled()
    expect(create).toHaveAttribute('title', '原先选择的投资目标与约束已不存在或未发布，请重新选择。')
    expect(createInvestableUniverseSnapshot).not.toHaveBeenCalled()
  })

  it('没有已发布目标时不能锁定快照', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [] })
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    expect(screen.getByRole('button', { name: '生成锁定快照' })).toBeDisabled()
  })

  it('目标目录读取失败时不能锁定快照', async () => {
    vi.mocked(getStrategicCatalog).mockRejectedValue(new Error('目录暂时离线'))
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    expect(screen.getByRole('button', { name: '生成锁定快照' })).toBeDisabled()
  })

  it('区分尚未发布与该日无生效版本，非生效版本只作只读信息', async () => {
    vi.mocked(listProductPoolVersions).mockImplementation(async (options = {}) => options.activeOn
      ? { items: [], total: 0 }
      : { items: [version], total: 1 })
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByText(/没有已生效产品池版本。以下为已发布但当日未生效的版本/)).toBeInTheDocument()
    expect(screen.getByRole('table', { name: '已发布但当日未生效的产品池版本' })).toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: /核心产品池/ })).not.toBeInTheDocument()
  })

  it('搜索不会隐藏已选版本', async () => {
    vi.mocked(listProductPoolVersions).mockResolvedValue({ items: [version, { ...version, id: 'old-version', version: 2 }], total: 2 })
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)
    const selected = await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })
    fireEvent.click(selected)
    fireEvent.change(screen.getByLabelText('搜索产品池版本'), { target: { value: '不存在的池' } })
    expect(screen.getByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    expect(screen.queryByRole('checkbox', { name: /核心产品池 · V2/ })).not.toBeInTheDocument()
  })

  it('研究名称默认生成并收在补充区，研究日保持可见', async () => {
    render(<MemoryRouter initialEntries={[enter('')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByLabelText(/研究日期/)).toBeVisible()
    expect(screen.getByText('研究名称（默认已生成，可修改）')).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('研究名称'), { target: { value: '自定义名称' } })
    expect(screen.getByLabelText('研究名称')).toHaveValue('自定义名称')
  })

  it('编辑保存以新版本替代当前版本，取消不改动原范围', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    const next = { ...universe, id: 'universe-2', name: '投前研究可投资域' }
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(next)
    vi.mocked(getInvestableUniverse).mockImplementation(async id => ({ ...universe, id }))
    render(<MemoryRouter initialEntries={[enter('?universe=saved-1&edit=saved-1&mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByDisplayValue('投前研究可投资域')).toBeInTheDocument()
    expect(await screen.findByRole('checkbox', { name: /核心产品池 · V3/ })).toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: '保存修改' }))
    await waitFor(() => expect(createInvestableUniverseSnapshot).toHaveBeenCalledWith(expect.objectContaining({ replaces_snapshot_id: 'saved-1' })))
    await waitFor(() => expect(readAllocationJourney().universeId).toBe('universe-2'))
  })

  it('同名预检给出内联提示，改名后可保存', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    render(<MemoryRouter initialEntries={[enter('?new=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    const input = await screen.findByRole('textbox', { name: '研究名称' })
    fireEvent.change(input, { target: { value: ' 已保存范围甲 ' } })
    expect(input).toHaveAttribute('aria-invalid', 'true')
    expect(screen.getByText('研究范围名称已存在，请修改名称。')).toHaveAttribute('id', 'scope-name-error')
    expect(screen.getByRole('button', { name: '生成锁定快照' })).toBeDisabled()
    fireEvent.change(input, { target: { value: '新的研究范围' } })
    expect(input).not.toHaveAttribute('aria-invalid')
    expect(screen.queryByText('研究范围名称已存在，请修改名称。')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '生成锁定快照' })).toBeEnabled()
  })

  it('取消编辑回到只读范围且不写入', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    render(<MemoryRouter initialEntries={[enter('?universe=saved-1&edit=saved-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)

    await screen.findByRole('button', { name: '取消编辑' })
    fireEvent.click(screen.getByRole('button', { name: '取消编辑' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('?universe=saved-1'))
    expect(createInvestableUniverseSnapshot).not.toHaveBeenCalled()
    expect(await screen.findByRole('heading', { name: '已保存的产品范围' })).toBeInTheDocument()
  })

  it('产品从空库新建保存后 URL 指向新范围并保持只读视图与目标', async () => {
    const saved = { ...universe, id: 'universe-new', name: '新范围' }
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(saved)
    vi.mocked(getInvestableUniverse).mockImplementation(async id => ({ ...saved, id }))
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    render(<MemoryRouter initialEntries={[enter('?new=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('checkbox', { name: /核心产品池/ }))
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('universe=universe-new'))
    expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-1')
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('new=')
    expect(await screen.findByRole('heading', { name: '新范围' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '进入自动构建大类' })).toBeEnabled()
  })

  it('战略范围从空库保存后 URL 指向新版本并保持只读视图', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(previewUniverse).mockResolvedValue({
      definition: (strategicScopeFixture as never as { definition: unknown }).definition,
      preview_hash: 'p'.repeat(64), implementation_status: 'unmapped', implementation_gaps: [], research_only: true,
    } as never)
    vi.mocked(confirmUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), id: 'scope-new', name: '战略新范围' } as never)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&new=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('button', { name: '添加参考大类' }))
    fireEvent.change(screen.getByLabelText('大类名称'), { target: { value: '增长资产' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '保存战略范围' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('strategic_universe=scope-new'))
    expect(screen.getByTestId('location-probe')).toHaveTextContent('mandate=mandate-1')
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('new=')
    expect(await screen.findByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  })

  it('产品复制保存后 URL 指向新范围，原冻结记录不变且编辑载入新版本', async () => {
    const copy = { ...universe, id: 'universe-copy', name: '投前研究可投资域（副本）' }
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    vi.mocked(getInvestableUniverse).mockImplementation(async id => id === 'universe-copy'
      ? copy
      : { ...universe, id: 'saved-1', name: '投前研究可投资域', excluded_product_keys: ['etf:excluded'] })
    vi.mocked(createInvestableUniverseSnapshot).mockResolvedValue(copy)
    render(<MemoryRouter initialEntries={[enter('?universe=saved-1&copy=saved-1&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)

    await screen.findByDisplayValue('投前研究可投资域（副本）')
    fireEvent.click(screen.getByRole('button', { name: '生成锁定快照' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('universe=universe-copy'))
    expect(createInvestableUniverseSnapshot).toHaveBeenCalledWith(expect.objectContaining({ excluded_product_keys: ['etf:excluded'] }))
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('copy=')
    fireEvent.click(screen.getByRole('link', { name: '编辑此范围' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('edit=universe-copy'))
    expect(await screen.findByDisplayValue('投前研究可投资域（副本）')).toBeInTheDocument()
  })

  it('战略范围编辑保存发送替代身份并回到只读', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [strategicScopeFixture] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(previewUniverse).mockResolvedValue({
      definition: (strategicScopeFixture as never as { definition: unknown }).definition,
      preview_hash: 'p'.repeat(64), implementation_status: 'unmapped', implementation_gaps: [], research_only: true,
    } as never)
    vi.mocked(confirmUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), id: 'scope-new' } as never)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui&edit=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /></MemoryRouter>)

    expect(await screen.findByRole('textbox', { name: '战略范围名称' })).toHaveValue('独立战略范围')
    await waitFor(() => expect(screen.getByRole('button', { name: '保存修改' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '保存修改' }))
    await waitFor(() => expect(confirmUniverse).toHaveBeenCalledWith(expect.anything(), 'p'.repeat(64), expect.anything(), 'scope-ui', 'mandate-1'))
    expect(await screen.findByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  })

  it('战略范围复制保存后 URL 指向新版本', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [strategicScopeFixture] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(previewUniverse).mockResolvedValue({
      definition: (strategicScopeFixture as never as { definition: unknown }).definition,
      preview_hash: 'p'.repeat(64), implementation_status: 'unmapped', implementation_gaps: [], research_only: true,
    } as never)
    vi.mocked(confirmUniverse).mockResolvedValue({ ...(strategicScopeFixture as object), id: 'scope-copy-new' } as never)
    render(<MemoryRouter initialEntries={[enter('?scope=strategic&strategic_universe=scope-ui&copy=1&mandate=mandate-1')]}><ProductPoolWorkspacePage /><LocationProbe /></MemoryRouter>)

    await screen.findByDisplayValue('独立战略范围（副本）')
    await waitFor(() => expect(screen.getByRole('button', { name: '保存战略范围' })).toBeEnabled())
    fireEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('strategic_universe=scope-copy-new'))
    expect(screen.getByTestId('location-probe')).not.toHaveTextContent('copy=')
  })
})
