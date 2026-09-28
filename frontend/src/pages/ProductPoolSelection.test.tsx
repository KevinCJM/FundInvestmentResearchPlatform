import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import ProductPoolSelection from './ProductPoolSelection'
import { readAllocationJourney, updateAllocationJourney } from '../app/allocationJourney'
import { listInvestableUniverseSnapshots, retireInvestableUniverseSnapshot } from '../services/productPools'
import { retireUniverse } from '../services/strategicScope'
import { getMandate, getStrategicCatalog } from '../services/strategicAllocation'
import { mandateVersion, strategicCatalog } from '../test/strategicAllocationFixtures'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))

vi.mock('../services/productPools', async () => {
  const actual = await vi.importActual<typeof import('../services/productPools')>(
    '../services/productPools',
  )
  return { ...actual, listInvestableUniverseSnapshots: vi.fn(), retireInvestableUniverseSnapshot: vi.fn() }
})

vi.mock('../services/strategicScope', async () => {
  const actual = await vi.importActual<typeof import('../services/strategicScope')>(
    '../services/strategicScope',
  )
  return { ...actual, retireUniverse: vi.fn() }
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

const summary = {
  id: 'saved-1', name: '已保存范围甲', research_date: '2026-09-04', version_ids: ['version-1'],
  pool_ids: ['pool-1'], product_count: 1, content_hash: 'h'.repeat(8), immutable: true,
  created_at: '2026-09-04T00:00:00Z',
} as never

function LocationProbe() {
  const location = useLocation()
  return <div data-testid="location-probe">{location.pathname}{location.search}</div>
}

describe('ProductPoolSelection', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    clock.day = '2026-09-12'
    localStorage.clear()
    sessionStorage.clear()
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    vi.mocked(retireInvestableUniverseSnapshot).mockResolvedValue({ deleted: true, id: 'saved-1' })
    vi.mocked(retireUniverse).mockResolvedValue({ deleted: true, id: 'scope-ui' })
    vi.mocked(getStrategicCatalog).mockResolvedValue(strategicCatalog)
  })

  it('列表不要求选择目标，两条路径都可以直接新建范围', async () => {
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool']}><ProductPoolSelection /><LocationProbe /></MemoryRouter>)
    expect(screen.getByRole('heading', { name: '选择研究路径与范围' })).toBeVisible()
    expect(screen.queryByRole('combobox', { name: '投资目标与约束' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '新建研究范围' })).toHaveAttribute('href', expect.stringMatching(/new=\d+/))
    fireEvent.click(screen.getByRole('link', { name: '先做战略研究：独立资产范围' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent('scope=strategic'))
    expect(screen.getByRole('link', { name: '新建研究范围' })).toHaveAttribute('href', expect.stringMatching(/scope=strategic&new=\d+/))
    expect(screen.queryByRole('combobox', { name: '投资目标与约束' })).not.toBeInTheDocument()
  })

  it.each(['product', 'strategic'])('%s 列表展示每条范围绑定的目标名称，不使用当前 URL 的其他目标', async kind => {
    const second = { ...mandateVersion, id: 'mandate-other', name: '教育资金目标' }
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, mandates: [mandateVersion, second],
      strategic_universes: [{ ...(strategicScopeFixture as object), mandate_id: second.id }] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [{ ...(summary as object), mandate_id: second.id }] as never, total: 1 })
    render(<MemoryRouter initialEntries={[`/pre-investment/product-pool?mandate=mandate-1${kind === 'strategic' ? '&scope=strategic' : ''}`]}><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByRole('cell', { name: '教育资金目标' })).toBeVisible()
    expect(screen.getByRole('columnheader', { name: '投资目标与约束' })).toBeVisible()
    expect(screen.getByRole('link', { name: '继续研究' })).toHaveAttribute('href', expect.stringContaining('mandate=mandate-other'))
    expect(getMandate).not.toHaveBeenCalled()
  })

  it('历史目标不在活动目录时按精确 ID 读取名称，失败可重试', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [{ ...(summary as object), mandate_id: 'retired', mandate_hash: mandateVersion.content_hash }] as never, total: 1 })
    vi.mocked(getMandate).mockRejectedValueOnce(new Error('暂不可用')).mockResolvedValue({ ...mandateVersion, id: 'retired', name: '原资金目标' })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByRole('cell', { name: '目标名称暂不可用' })).toBeVisible()
    fireEvent.click(screen.getByRole('button', { name: '重试读取投资目标' }))
    expect(await screen.findByRole('cell', { name: '原资金目标' })).toBeVisible()
    expect(getMandate).toHaveBeenCalledWith('retired', expect.any(AbortSignal))
  })

  it('未绑定旧范围明确显示未关联，不能偷偷采用 URL 的目标', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?mandate=mandate-1']}><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByRole('cell', { name: '未关联目标' })).toBeVisible()
    expect(screen.getByRole('link', { name: '继续研究' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=saved-1')
  })

  it('目标目录失败不隐藏产品范围，提供名称读取重试', async () => {
    vi.mocked(getStrategicCatalog).mockRejectedValue(new Error('目录暂时离线'))
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [{ ...(summary as object), mandate_id: 'mandate-1' }] as never, total: 1 })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)
    expect(await screen.findByRole('alert')).toHaveTextContent('部分目标名称读取失败')
    expect(await screen.findByRole('link', { name: '已保存范围甲' })).toBeVisible()
    expect(screen.getByRole('button', { name: '重试读取投资目标' })).toBeVisible()
  })

  it('范围库可搜索，链接指向新建工作区并携带精确身份，新建为显式命令', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({
      items: [summary, { ...(summary as object), id: 'saved-2', name: '已保存范围乙', research_date: '2026-09-05' } as never],
      total: 2,
    })
    render(<MemoryRouter><ProductPoolSelection /><LocationProbe /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '已保存范围甲' })).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '已保存范围乙' })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('搜索范围名称'), { target: { value: '乙' } })
    expect(screen.queryByRole('link', { name: '已保存范围甲' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '已保存范围乙' })).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('搜索范围名称'), { target: { value: '' } })

    expect(screen.getAllByRole('link', { name: '继续研究' })[0]).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=saved-1')
    expect(screen.getAllByRole('link', { name: '复制新建' })[0]).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=saved-1&copy=saved-1')

    fireEvent.click(screen.getByRole('link', { name: '新建研究范围' }))
    await waitFor(() => expect(screen.getByTestId('location-probe')).toHaveTextContent(/\/pre-investment\/product-pool\/new\?new=\d+/))
  })

  it('范围库研究日与 PIT 不一致时，浮窗提示脱离表格滚动容器完整显示', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    const trigger = await screen.findByRole('button', { name: '研究日与当前 PIT 日期不一致' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
    fireEvent.mouseEnter(trigger)
    const tip = await screen.findByRole('tooltip')
    expect(tip.parentElement).toBe(document.body)
    expect(tip).toHaveTextContent('研究日 2026-09-04 与当前页面 PIT 日期 2026-09-12 不一致；仍可继续研究。')
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByRole('tooltip')).not.toBeInTheDocument()
  })

  it('范围库提供编辑与删除，删除需具名确认且可取消', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '编辑' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=saved-1&edit=saved-1')
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    expect(screen.getByText(/从列表移除「已保存范围甲」？已引用的历史研究保留。/)).toBeInTheDocument()

    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(retireInvestableUniverseSnapshot).not.toHaveBeenCalled()
    expect(screen.getByRole('link', { name: '已保存范围甲' })).toBeInTheDocument()

    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    fireEvent.click(screen.getByRole('button', { name: '确认移除' }))
    await waitFor(() => expect(retireInvestableUniverseSnapshot).toHaveBeenCalledWith('saved-1'))
    await waitFor(() => expect(screen.queryByRole('link', { name: '已保存范围甲' })).not.toBeInTheDocument())
  })

  it('删除失败保留列表并显示错误', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    vi.mocked(retireInvestableUniverseSnapshot).mockRejectedValue(new Error('移除请求失败'))
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    fireEvent.click(await screen.findByRole('button', { name: '删除' }))
    fireEvent.click(screen.getByRole('button', { name: '确认移除' }))
    expect(await screen.findByText('移除请求失败')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '已保存范围甲' })).toBeInTheDocument()
  })

  it('删除当前选中的范围会清掉旅程引用', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    updateAllocationJourney({ universeId: 'saved-1', name: '已保存范围甲', researchDate: '2026-09-04', poolVersionIds: ['version-1'] })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '已保存范围甲' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    fireEvent.click(screen.getByRole('button', { name: '确认移除' }))
    await waitFor(() => expect(retireInvestableUniverseSnapshot).toHaveBeenCalledWith('saved-1'))
    await waitFor(() => expect(readAllocationJourney().universeId).toBeUndefined())
  })

  it('等待期间切换选择时，迟到的删除不会清掉新选择', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({
      items: [summary, { ...(summary as object), id: 'saved-2', name: '已保存范围乙' } as never],
      total: 2,
    })
    let resolveDelete!: (value: { deleted: true; id: string }) => void
    vi.mocked(retireInvestableUniverseSnapshot).mockReturnValue(new Promise(done => { resolveDelete = done }))
    updateAllocationJourney({ universeId: 'saved-1' })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    await screen.findByRole('link', { name: '已保存范围甲' })
    fireEvent.click(screen.getAllByRole('button', { name: '删除' })[0])
    fireEvent.click(screen.getByRole('button', { name: '确认移除' }))
    // 删除请求在途时，用户已经在别处（新建工作区页）把当前研究切到了另一个范围。
    updateAllocationJourney({ universeId: 'saved-2' })
    await act(async () => resolveDelete({ deleted: true, id: 'saved-1' }))
    expect(readAllocationJourney().universeId).toBe('saved-2')
  })

  it('战略范围库提供编辑与删除，编辑自己的名字不误报', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [strategicScopeFixture] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?scope=strategic']}><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '编辑' })).toHaveAttribute('href', '/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-ui&edit=1')
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    expect(screen.getByText(/从列表移除「独立战略范围」？已引用的历史研究保留。/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '取消' }))
    expect(retireUniverse).not.toHaveBeenCalled()
  })

  it('战略范围删除成功后从列表移除', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [strategicScopeFixture] } as never)
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [], total: 0 })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?scope=strategic']}><ProductPoolSelection /></MemoryRouter>)

    await screen.findByRole('link', { name: '独立战略范围' })
    fireEvent.click(screen.getByRole('button', { name: '删除' }))
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [] } as never)
    fireEvent.click(screen.getByRole('button', { name: '确认移除' }))
    await waitFor(() => expect(retireUniverse).toHaveBeenCalledWith('scope-ui'))
    await waitFor(() => expect(screen.queryByRole('link', { name: '独立战略范围' })).not.toBeInTheDocument())
  })

  it('仅有旧旅程书签时裸 URL 只展示范围库', async () => {
    vi.mocked(listInvestableUniverseSnapshots).mockResolvedValue({ items: [summary], total: 1 })
    updateAllocationJourney({ universeId: 'saved-1', name: '旧研究', researchDate: '2026-09-04', poolVersionIds: ['version-1'] })
    render(<MemoryRouter><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '已保存范围甲' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=saved-1')
  })

  it('战略路径同理，裸 URL 只展示范围库', async () => {
    vi.mocked(getStrategicCatalog).mockResolvedValue({ ...strategicCatalog, strategic_universes: [strategicScopeFixture] } as never)
    updateAllocationJourney({ strategicUniverseId: 'scope-ui' })
    render(<MemoryRouter initialEntries={['/pre-investment/product-pool?scope=strategic']}><ProductPoolSelection /></MemoryRouter>)

    expect(await screen.findByRole('link', { name: '独立战略范围' })).toBeInTheDocument()
  })
})
