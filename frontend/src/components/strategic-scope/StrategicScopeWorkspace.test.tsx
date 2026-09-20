import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ProductPoolSelection from '../../pages/ProductPoolSelection'
import LtcmaWorkspace from '../../pages/LtcmaWorkspace'
import { ltcmaCapabilities } from '../../test/ltcmaFixtures'
import ImplementationMappingEditor from './ImplementationMappingEditor'
import { writeAllocationDraft, readAllocationJourney, updateAllocationJourney } from '../../app/allocationJourney'
import { cmaPreview, cmaVersion, strategicCatalog } from '../../test/strategicAllocationFixtures'
import type { MappingVersion, UniverseDefinition, UniverseVersion } from '../../services/strategicScope'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
const definition: UniverseDefinition = { name: '独立战略范围', as_of: '2026-09-12', currency: 'CNY', source: '离线研究来源', assets: [
  { id: 'equity', name: '增长资产', currency: 'CNY', role: 'growth', liquidity: 'liquid', rationale: '资本增长用途', source: '独立风险定义' },
  { id: 'cash', name: '储备现金', currency: 'CNY', role: 'liquidity', liquidity: 'liquid', rationale: '现金储备用途', source: '明确资金要求' },
] }
const version: UniverseVersion = { id: 'scope-one', name: definition.name, definition, content_hash: 'a'.repeat(64), created_at: '2026-09-12', preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped', implementation_gaps: ['equity', 'cash'] }
const catalog = { ...strategicCatalog, allocations: [], assumptions: [], policies: [], strategic_universes: [version], implementation_maps: [] }
const mapping: MappingVersion = {
  id: 'map-one', name: '已有映射', content_hash: 'c'.repeat(64), created_at: '2026-09-12', preview_hash: 'd'.repeat(64),
  definition: { name: '已有映射', strategic_universe_id: version.id, universe_snapshot_id: 'domain-one', alloc_name: '真实方案', as_of: '2026-09-12', valid_until: '2027-01-01', assignments: [] },
  implementation_status: 'incomplete', implementation_gaps: ['equity', 'cash'],
  coverage: [{ strategic_asset_id: 'equity', proxy_asset_id: null, status: 'missing_products' }],
}
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)
const root = '/api/strategic-allocation'
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (overrides[url]) return overrides[url](init)
    if (url === `${root}/catalog`) return response(catalog)
    if (url === `${root}/cma/capabilities`) return response(ltcmaCapabilities)
    if (url === `${root}/cma/study-options`) return response({ ...catalog, regime_runs: [] })
    if (url === `${root}/universes/scope-one`) return response(version)
    if (url === `${root}/universes/preview`) return response({ ...version, definition: JSON.parse(String(init?.body)) })
    if (url === `${root}/universes/confirm`) return response({ ...version, definition: JSON.parse(String(init?.body)).request }, 201)
    if (url === `${root}/cma/preview`) return response({ ...cmaPreview, definition: JSON.parse(String(init?.body)) })
    if (url === `${root}/cma`) return response({ ...cmaVersion, definition: JSON.parse(String(init?.body)).request }, 201)
    throw new Error(`Unexpected API: ${url}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function scopePage(url = '/pre-investment/product-pool?scope=strategic') {
  return render(<MemoryRouter initialEntries={[url]}><ProductPoolSelection /></MemoryRouter>)
}
beforeEach(() => { localStorage.clear(); sessionStorage.clear(); clock.day = '2026-09-12'; vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

it.each([false, true])('映射历史读取独立于知识时钟初始化（切换=%s），只读证据保留且复制可编辑', async changeClock => {
  clock.day = undefined
  let resolve!: (r: Response) => void
  const fetch = install({ [`${root}/implementation-maps/map-one`]: () => new Promise(done => { resolve = done }) })
  const tree = () => <MemoryRouter><ImplementationMappingEditor universe={version} catalog={catalog} domainId="domain-one" requestedMapping="map-one" onSaved={() => {}} /></MemoryRouter>
  const view = render(tree())
  await screen.findByText('正在读取不可变实施映射…')
  expect(screen.getByLabelText('映射名称')).toBeDisabled()
  if (changeClock) { clock.day = '2026-09-12'; view.rerender(tree()) }
  await act(async () => resolve(await response(mapping)))
  await screen.findByText(/只读映射/)
  expect(screen.getByLabelText('映射名称')).toHaveValue(mapping.name)
  expect(screen.getByLabelText('映射名称')).toBeDisabled()
  expect(readAllocationJourney().implementationMappingId).toBe(mapping.id)
  clock.day = '2026-09-13'; view.rerender(tree())
  expect(screen.getByText('equity：缺少产品')).toBeInTheDocument()
  expect(fetch.mock.calls.filter(([url]) => String(url).includes('implementation-maps'))).toHaveLength(1)
  await userEvent.click(screen.getByRole('button', { name: '复制映射为新研究' }))
  fireEvent.change(screen.getByLabelText('映射名称'), { target: { value: '新的映射研究' } })
  expect(screen.getByLabelText('映射名称')).toHaveValue('新的映射研究')
  expect(screen.queryByText(/只读映射/)).not.toBeInTheDocument()
  expect(mapping.definition.name).toBe('已有映射')
})

it('映射历史失败可重试，旧响应不能覆盖后来选择的版本', async () => {
  let resolveOld!: (r: Response) => void
  let attempts = 0
  const other = { ...mapping, id: 'map-two', name: '第二映射', definition: { ...mapping.definition, name: '第二映射' } }
  install({
    [`${root}/implementation-maps/map-one`]: () => new Promise(done => { resolveOld = done }),
    [`${root}/implementation-maps/map-two`]: () => ++attempts === 1 ? response({ detail: { message: '映射读取暂时失败' } }, 503) : response(other),
  })
  const tree = (id: string) => <MemoryRouter><ImplementationMappingEditor universe={version} catalog={catalog} domainId="domain-one" requestedMapping={id} onSaved={() => {}} /></MemoryRouter>
  const view = render(tree('map-one'))
  view.rerender(tree('map-two'))
  await screen.findByText('映射读取暂时失败')
  await userEvent.click(screen.getByRole('button', { name: '重试读取实施映射' }))
  await screen.findByText(/只读映射：第二映射/)
  await act(async () => resolveOld(await response(mapping)))
  expect(screen.getByLabelText('映射名称')).toHaveValue('第二映射')
  expect(readAllocationJourney().implementationMappingId).toBe('map-two')
})

it('无产品即可预览确认战略范围，保留缺口且确认前不调用保存', async () => {
  const fetch = install(); const user = userEvent.setup()
  writeAllocationDraft('strategic-universe:editor', definition); scopePage()
  await user.click(screen.getByRole('button', { name: '预览战略范围' }))
  await screen.findByText(/定义已通过校验/)
  expect(fetch.mock.calls.some(([url]) => String(url).includes('confirm'))).toBe(false)
  expect(fetch.mock.calls.some(([url]) => String(url).includes('product-pools'))).toBe(false)
  await user.click(screen.getByRole('button', { name: '确认保存战略范围' }))
  await screen.findByText(/只读战略范围/)
  expect(screen.getByLabelText(/资产1稳定ID/)).toBeDisabled()
  expect(await screen.findByRole('link', { name: /先做前瞻CMA与SAA研究/ })).toHaveAttribute('href', '/pre-investment/saa/policy?strategic_universe=scope-one')
  expect(readAllocationJourney().strategicUniverseId).toBe('scope-one')
  expect(JSON.stringify(localStorage)).not.toContain('preview_hash')
})

it('修改输入使预览及迟到确认失效；确认请求仍固定旧输入', async () => {
  let resolve!: (r: Response) => void
  const fetch = install({ [`${root}/universes/confirm`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); writeAllocationDraft('strategic-universe:editor', definition); scopePage()
  await user.click(screen.getByRole('button', { name: '预览战略范围' }))
  await screen.findByText(/定义已通过校验/)
  await user.click(screen.getByRole('button', { name: '确认保存战略范围' }))
  fireEvent.change(screen.getByLabelText('战略范围名称'), { target: { value: '新的范围' } })
  await act(async () => resolve(await response(version, 201)))
  expect(screen.queryByText(/只读战略范围/)).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '确认保存战略范围' })).toBeDisabled()
  expect(screen.getByLabelText('战略范围名称')).toHaveValue('新的范围')
  expect(readAllocationJourney().strategicUniverseId).toBeUndefined()
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/confirm'))![1]?.body))
  expect(body.request.name).toBe(definition.name)
})

it('未知知识截止日阻止计算，重复资产不能预览，原产品路径入口仍在', async () => {
  clock.day = undefined; install(); writeAllocationDraft('strategic-universe:editor', definition); scopePage()
  expect(screen.getByRole('button', { name: '预览战略范围' })).toBeDisabled()
  expect(screen.getByText(/知识截止日尚未确认，暂不能预览/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '已有产品：选择产品池' })).toBeInTheDocument()
})

it('目录失败保留表单及重试，不把失败显示为没有历史', async () => {
  const fetch = install({ [`${root}/catalog`]: () => response({ detail: { message: '目录暂时离线' } }, 503) })
  scopePage(); await screen.findByText('目录暂时离线')
  await userEvent.click(screen.getByText('读取历史范围与映射'))
  await userEvent.click(screen.getByRole('button', { name: '重试读取范围目录' }))
  await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/catalog')).length).toBe(2))
  expect(screen.getByRole('button', { name: '增加战略资产' })).toBeEnabled()
})

it('历史战略范围只读，复制编辑不会修改原版本', async () => {
  install(); const user = userEvent.setup(); scopePage('/pre-investment/product-pool?scope=strategic&strategic_universe=scope-one')
  await screen.findByText(/只读战略范围/)
  expect(screen.getByLabelText('战略范围名称')).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '复制范围为新研究' }))
  fireEvent.change(screen.getByLabelText('战略范围名称'), { target: { value: '另一版' } })
  expect(screen.getByLabelText('战略范围名称')).toBeEnabled()
  expect(version.definition.name).toBe('独立战略范围')
})

it('显式选择暂不匹配会清除旅程中的旧映射和下游引用', async () => {
  const mapping = { id: 'map-one', name: '已有映射', definition: { name: '已有映射', strategic_universe_id: version.id, universe_snapshot_id: 'domain-one', alloc_name: '真实方案', as_of: '2026-09-12', valid_until: '2027-01-01', assignments: [] }, implementation_status: 'incomplete', coverage: [], implementation_gaps: ['equity', 'cash'] }
  install({ [`${root}/catalog`]: () => response({ ...catalog, implementation_maps: [mapping] }), [`${root}/implementation-maps/map-one`]: () => response(mapping) })
  scopePage('/pre-investment/product-pool?scope=strategic&strategic_universe=scope-one&mapping=map-one')
  await screen.findByText(/只读映射/)
  act(() => { updateAllocationJourney({ baselineId: 'baseline-old', taaRunId: 'decision-old' }) })
  await userEvent.click(screen.getByText('读取历史范围与映射'))
  await userEvent.selectOptions(screen.getByLabelText('已保存的实施映射'), '')
  expect(readAllocationJourney()).toMatchObject({ strategicUniverseId: version.id })
  expect(readAllocationJourney().implementationMappingId).toBeUndefined()
  expect(readAllocationJourney().baselineId).toBeUndefined()
  expect(readAllocationJourney().taaRunId).toBeUndefined()
  expect(screen.getByRole('link', { name: /先做前瞻CMA与SAA研究/ })).not.toHaveAttribute('href', expect.stringContaining('mapping='))
})

it('独立战略进入LTCMA只发战略ID，数值从空白填写，无伪造alloc_name或历史参考', async () => {
  const fetch = install(); const user = userEvent.setup()
  render(<MemoryRouter initialEntries={['/pre-investment/ltcma/new?strategic_universe=scope-one']}><LtcmaWorkspace /></MemoryRouter>)
  await screen.findByLabelText('名称')
  await user.type(screen.getByLabelText('名称'), '战略 LTCMA 研究')
  expect(screen.getByLabelText('equity · 预期年收益（%）')).toHaveValue('')
  expect(screen.queryByRole('button', { name: '读取历史风险参考' })).not.toBeInTheDocument()
  for (const [id, ret, vol] of [['equity', '6', '15'], ['cash', '2', '1']]) {
    fireEvent.change(screen.getByLabelText(`${id} · 预期年收益（%）`), { target: { value: ret } })
    fireEvent.change(screen.getByLabelText(`${id} · 年化波动（%）`), { target: { value: vol } })
    fireEvent.change(screen.getByLabelText(new RegExp(`${id} · 均值不确定半宽`)), { target: { value: '1' } })
  }
  fireEvent.change(screen.getByLabelText('相关矩阵: equity / cash'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText(/假设依据/), { target: { value: '明确的离线研究假设' } })
  await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
  await user.click(screen.getByRole('button', { name: '计算预览' }))
  await screen.findByRole('button', { name: '确认保存版本' })
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/cma/preview'))![1]?.body))
  expect(body).toMatchObject({ alloc_name: null, strategic_universe_id: 'scope-one', implementation_mapping_id: null })
  expect(body.assets.map((a: { id: string }) => a.id)).toEqual(['equity', 'cash'])
  expect(body.assets[1].role).toBe('liquidity')
})
