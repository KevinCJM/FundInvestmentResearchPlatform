import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import ProductPoolSelection from '../../pages/ProductPoolSelection'
import ProductPoolWorkspacePage from '../../pages/ProductPoolWorkspacePage'
import LtcmaWorkspace from '../../pages/LtcmaWorkspace'
import { ltcmaCapabilities } from '../../test/ltcmaFixtures'
import ImplementationMappingEditor from './ImplementationMappingEditor'
import StrategicScopeWorkspace from './StrategicScopeWorkspace'
import { writeAllocationDraft, readAllocationDraft, readAllocationJourney, updateAllocationJourney } from '../../app/allocationJourney'
import { cmaPreview, cmaVersion, strategicCatalog } from '../../test/strategicAllocationFixtures'
import type { MappingVersion, UniverseDefinition, UniverseVersion } from '../../services/strategicScope'

const clock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../../app/ResearchContext', () => ({ useResearchDay: () => clock.day, useResearchContextIdentity: () => JSON.stringify(clock) }))
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
    if (url.split('?')[0] === `${root}/cma/study-options`) return response({ ...catalog, regime_runs: [], existing_names: [] })
    if (url === `${root}/universes/scope-one`) return response(version)
    if (url === `${root}/universes/scope-one/mandate`) return response({ mandate_id: JSON.parse(String(init?.body)).mandate_id })
    if (url === `${root}/universes/preview`) return response({ ...version, definition: JSON.parse(String(init?.body)) })
    if (url === `${root}/universes/confirm`) return response({ ...version, definition: JSON.parse(String(init?.body)).request }, 201)
    if (url === `${root}/cma/preview`) return response({ ...cmaPreview, definition: JSON.parse(String(init?.body)) })
    if (url === `${root}/cma`) return response({ ...cmaVersion, definition: JSON.parse(String(init?.body)).request }, 201)
    throw new Error(`Unexpected API: ${url}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
// 02 的投资目标在配置内选择，也支持 URL 显式带入；这里按需直接带 mandate，
// 不再走下拉框（那个下拉框已经不在这个页面上）。
function scopePage(url = '/pre-investment/product-pool/new?scope=strategic') {
  return render(<MemoryRouter initialEntries={[url]}><ProductPoolWorkspacePage /></MemoryRouter>)
}
/** 空范围库时编辑器自动打开，沿用标准工作草稿（含预置草稿的用例）。 */
function installEditorCatalog(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  return install({ [`${root}/catalog`]: () => response({ ...catalog, strategic_universes: [] }), ...overrides })
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

it('无产品也可一次保存，自动核验当前输入且保存前不写入', async () => {
  const fetch = installEditorCatalog(); const user = userEvent.setup()
  writeAllocationDraft('strategic-universe:editor', definition); scopePage('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-1')
  expect(await screen.findByLabelText('战略范围名称')).toBeRequired()
  expect(screen.getByLabelText(/战略研究日/)).toBeRequired()
  expect(screen.getByLabelText('范围备注').closest('label')).not.toHaveAttribute('data-required')
  expect(screen.getByRole('heading', { name: '大类资产与研究代理' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '预览战略范围' })).not.toBeInTheDocument()
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/universes/'))).toBe(false)
  expect(fetch.mock.calls.some(([url]) => String(url).includes('product-pools'))).toBe(false)
  await user.click(screen.getByRole('button', { name: '保存战略范围' }))
  await screen.findByText(/只读战略范围/)
  expect(screen.getByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  expect(screen.queryByLabelText('战略范围名称')).not.toBeInTheDocument()
  expect(await screen.findByRole('link', { name: '下一步' })).toHaveAttribute('href', '/pre-investment/ltcma/new?strategic_universe=scope-one&mandate=mandate-1')
  expect(screen.getByRole('region', { name: '独立战略范围' }).querySelector('a[href*="/saa/"]')).toBeNull()
  expect(readAllocationJourney().strategicUniverseId).toBe('scope-one')
  const writes = fetch.mock.calls.filter(([url]) => String(url).includes('/universes/') && String(url).endsWith('/scope-one') === false && !String(url).endsWith('/mandate'))
  expect(writes.map(([url]) => String(url).split('/').pop())).toEqual(['preview', 'confirm'])
  const checked = JSON.parse(String(writes[0][1]?.body))
  expect(JSON.parse(String(writes[1][1]?.body))).toEqual({ request: checked, preview_hash: version.preview_hash, mandate_id: 'mandate-1' })
  expect(JSON.stringify(localStorage)).not.toContain('preview_hash')
})

it('修改输入使迟到保存失效；在途请求仍固定旧输入', async () => {
  let resolve!: (r: Response) => void
  const fetch = installEditorCatalog({ [`${root}/universes/confirm`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); writeAllocationDraft('strategic-universe:editor', definition); scopePage('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-1')
  await user.click(screen.getByRole('button', { name: '保存战略范围' }))
  expect(screen.getByRole('button', { name: '正在保存…' })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('战略范围名称'), { target: { value: '新的范围' } })
  await act(async () => resolve(await response(version, 201)))
  expect(screen.queryByText(/只读战略范围/)).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '保存战略范围' })).toBeEnabled()
  expect(screen.getByLabelText('战略范围名称')).toHaveValue('新的范围')
  expect(readAllocationJourney().strategicUniverseId).toBeUndefined()
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/confirm'))![1]?.body))
  expect(body.request.name).toBe(definition.name)
})

it.each(['preview', 'confirm'])('%s 失败在保存处显示原因，保留输入并支持重试', async endpoint => {
  let attempts = 0
  const fetch = installEditorCatalog({ [`${root}/universes/${endpoint}`]: () => ++attempts === 1
    ? response({ detail: { message: '服务暂时不可用，请重试。' } }, 503)
    : response(version) })
  writeAllocationDraft('strategic-universe:editor', definition)
  scopePage('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-1')
  await waitFor(() => expect(screen.getByRole('button', { name: '保存战略范围' })).toBeEnabled())
  await userEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
  const error = await screen.findByRole('alert')
  expect(error).toHaveTextContent('保存失败：服务暂时不可用，请重试。')
  expect(screen.getByRole('button', { name: '保存战略范围' })).toHaveAccessibleDescription(error.textContent!)
  expect(screen.getByLabelText('战略范围名称')).toHaveValue(definition.name)
  if (endpoint === 'preview') expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/confirm'))).toBe(false)
  await userEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
  await screen.findByText(/只读战略范围/)
  expect(attempts).toBe(2)
  expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/preview'))).toHaveLength(2)
})

it('旧服务拒绝研究代理时说明服务问题，保留代理输入且不降级写入', async () => {
  const fetch = installEditorCatalog({ [`${root}/universes/preview`]: () => response({ detail: [0, 1, 2].map(index => ({ type: 'extra_forbidden', loc: ['body', 'assets', index, 'research_proxy'], msg: 'Extra inputs are not permitted' })) }, 422) })
  const draft: UniverseDefinition = { ...definition, assets: definition.assets.map(asset => ({ ...asset, research_proxy: { asset_type: 'market', cash_return: null, components: [], rebalance: 'daily', source_labels: {} } })) }
  writeAllocationDraft('strategic-universe:editor', draft)
  scopePage('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-1')
  await waitFor(() => expect(screen.getByRole('button', { name: '保存战略范围' })).toBeEnabled())
  await userEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
  const error = await screen.findByRole('alert')
  expect(error).toHaveTextContent('当前页面与服务版本不匹配')
  expect(error).not.toHaveTextContent(/assets|research_proxy|Extra inputs/)
  expect(screen.getByRole('button', { name: '保存战略范围' })).toHaveAccessibleDescription(error.textContent!)
  expect(readAllocationDraft<UniverseDefinition>('strategic-universe:editor')).toEqual(draft)
  expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/preview'))).toHaveLength(1)
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/confirm'))).toBe(false)
})

it.each(['draft', 'clock', 'mandate', 'unmount'])('保存校验期间 %s 改变，迟到结果不得发起写入', async changed => {
  let resolve!: (r: Response) => void
  const fetch = installEditorCatalog({ [`${root}/universes/preview`]: () => new Promise(done => { resolve = done }) })
  writeAllocationDraft('strategic-universe:editor', definition)
  const tree = (blocked = '') => <MemoryRouter><StrategicScopeWorkspace mandateBlockedReason={blocked} /></MemoryRouter>
  const view = render(tree())
  await userEvent.click(screen.getByRole('button', { name: '保存战略范围' }))
  // 重复点击不能创建第二个保存链路。
  fireEvent.click(screen.getByRole('button', { name: '正在保存…' }))
  expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/preview'))).toHaveLength(1)
  if (changed === 'draft') fireEvent.change(screen.getByLabelText('战略范围名称'), { target: { value: '新的范围' } })
  if (changed === 'clock') { clock.day = '2026-09-13'; view.rerender(tree()) }
  if (changed === 'mandate') view.rerender(tree('请重新选择投资目标。'))
  if (changed === 'unmount') view.unmount()
  await act(async () => resolve(await response(version)))
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/confirm'))).toBe(false)
  expect(readAllocationJourney().strategicUniverseId).toBeUndefined()
})

it('未知知识截止日阻止计算，保存战略范围也被拦住', async () => {
  clock.day = undefined; installEditorCatalog(); writeAllocationDraft('strategic-universe:editor', definition); scopePage()
  expect(await screen.findByRole('button', { name: '保存战略范围' })).toHaveAccessibleDescription('知识截止日尚未确认，暂时无法保存；草稿已保留。')
  expect(screen.getByRole('button', { name: '保存战略范围' })).toBeDisabled()
})

it('目录失败保留表单及重试，不把失败显示为没有历史', async () => {
  const fetch = install({ [`${root}/catalog`]: () => response({ detail: { message: '目录暂时离线' } }, 503) })
  scopePage(); await screen.findByText('目录暂时离线')
  await userEvent.click(screen.getByText('读取历史映射与重试'))
  await userEvent.click(screen.getByRole('button', { name: '重试读取范围目录' }))
  // 工作区页自己的同名预检取数与战略工作区各读一次目录，重试再读一次。
  await waitFor(() => expect(fetch.mock.calls.filter(([url]) => String(url).endsWith('/catalog')).length).toBe(3))
  expect(screen.getByRole('button', { name: '添加参考大类' })).toBeEnabled()
})

it('历史战略范围只读，复制编辑不会修改原版本', async () => {
  install(); const view = scopePage('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-one')
  await screen.findByText(/只读战略范围/)
  expect(screen.getByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  expect(screen.queryByLabelText('战略范围名称')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '复制范围为新研究' })).not.toBeInTheDocument()
  view.unmount()
  scopePage('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-one&copy=1')
  expect(await screen.findByDisplayValue('独立战略范围（副本）')).toBeEnabled()
  fireEvent.change(screen.getByLabelText('战略范围名称'), { target: { value: '另一版' } })
  expect(screen.getByLabelText('战略范围名称')).toBeEnabled()
  expect(version.definition.name).toBe('独立战略范围')
})

it('显式选择暂不匹配会清除旅程中的旧映射和下游引用', async () => {
  const mapping = { id: 'map-one', name: '已有映射', definition: { name: '已有映射', strategic_universe_id: version.id, universe_snapshot_id: 'domain-one', alloc_name: '真实方案', as_of: '2026-09-12', valid_until: '2027-01-01', assignments: [] }, implementation_status: 'incomplete', coverage: [], implementation_gaps: ['equity', 'cash'] }
  install({ [`${root}/catalog`]: () => response({ ...catalog, implementation_maps: [mapping] }), [`${root}/implementation-maps/map-one`]: () => response(mapping) })
  scopePage('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-one&mapping=map-one&mandate=mandate-1')
  await screen.findByText(/只读映射/)
  act(() => { updateAllocationJourney({ baselineId: 'baseline-old', taaRunId: 'decision-old' }) })
  await userEvent.click(screen.getByText('读取历史映射与重试'))
  await userEvent.selectOptions(screen.getByLabelText('已保存的实施映射'), '')
  expect(readAllocationJourney()).toMatchObject({ strategicUniverseId: version.id })
  expect(readAllocationJourney().implementationMappingId).toBeUndefined()
  expect(readAllocationJourney().baselineId).toBeUndefined()
  expect(readAllocationJourney().taaRunId).toBeUndefined()
  expect(screen.getByRole('link', { name: '下一步' })).not.toHaveAttribute('href', expect.stringContaining('mapping='))
})

it('独立战略进入LTCMA只发战略ID，数值从空白填写，无伪造alloc_name或历史参考', async () => {
  const fetch = install(); const user = userEvent.setup()
  render(<MemoryRouter initialEntries={['/pre-investment/ltcma/new?strategic_universe=scope-one']}><LtcmaWorkspace /></MemoryRouter>)
  await screen.findByLabelText('名称')
  await user.type(screen.getByLabelText('名称'), '战略 LTCMA 研究')
  expect(screen.getByLabelText('增长资产 · 预期年收益（%）')).toHaveValue('')
  expect(screen.queryByRole('button', { name: '读取历史风险参考' })).not.toBeInTheDocument()
  for (const [name, ret, vol] of [['增长资产', '6', '15'], ['储备现金', '2', '1']]) {
    fireEvent.change(screen.getByLabelText(`${name} · 预期年收益（%）`), { target: { value: ret } })
    fireEvent.change(screen.getByLabelText(`${name} · 年化波动（%）`), { target: { value: vol } })
    fireEvent.change(screen.getByLabelText(new RegExp(`${name} · 均值不确定半宽`)), { target: { value: '1' } })
  }
  fireEvent.change(screen.getByLabelText('相关矩阵: 增长资产 / 储备现金'), { target: { value: '0' } })
  fireEvent.change(screen.getByLabelText(/假设依据/), { target: { value: '明确的离线研究假设' } })
  await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
  await user.click(screen.getByRole('button', { name: '计算预览' }))
  await screen.findByRole('button', { name: '确认保存版本' })
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/cma/preview'))![1]?.body))
  expect(body).toMatchObject({ alloc_name: null, strategic_universe_id: 'scope-one', implementation_mapping_id: null })
  expect(body.assets.map((a: { id: string }) => a.id)).toEqual(['equity', 'cash'])
  expect(body.assets[1].role).toBe('liquidity')
})

it('未选择投资目标与约束时战略范围不能确认保存', async () => {
  const fetch = installEditorCatalog()
  writeAllocationDraft('strategic-universe:editor', definition); scopePage()
  await waitFor(() => expect(screen.getByRole('button', { name: '保存战略范围' })).toHaveAccessibleDescription(expect.stringContaining('请先选择已发布的投资目标与约束')))
  expect(screen.getByRole('button', { name: '保存战略范围' })).toBeDisabled()
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/universes/confirm'))).toBe(false)
})

it('已保存战略范围在缺少有效目标时不提供后续入口', async () => {
  install()
  scopePage('/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-one')
  await screen.findByText(/只读战略范围/)
  expect(screen.queryByRole('link', { name: '下一步' })).not.toBeInTheDocument()
})

it('PIT 开启时战略研究日跟随平台口径，关闭后才手填', async () => {
  installEditorCatalog()
  // 草稿是旧口径存下来的：进页面就要对齐到当前 PIT，而不是等用户自己发现。
  writeAllocationDraft('strategic-universe:editor', { ...definition, as_of: '2026-08-01' }); scopePage()

  const date = await screen.findByLabelText(/战略研究日/)
  await waitFor(() => expect(date).toHaveValue('2026-09-12'))
  expect(date).toBeDisabled()
  expect(screen.getByText(/研究日跟随当前研究日期/)).toBeInTheDocument()

  clock.day = null
  scopePage()
  await waitFor(() => expect(screen.getAllByText(/PIT 已关闭，可手动选择研究日/).length).toBeGreaterThan(0))
  const fields = screen.getAllByLabelText(/战略研究日/)
  const manual = fields[fields.length - 1]
  expect(manual).toBeEnabled()
  expect(screen.getAllByText(/PIT 已关闭，可手动选择研究日/).length).toBeGreaterThan(0)
})

it('新战略资产自动生成稳定ID，说明字段非必填，可空注解保存', async () => {
  const fetch = install(); const user = userEvent.setup()
  scopePage('/pre-investment/product-pool/new?scope=strategic&new=1&mandate=mandate-1')

  await waitFor(() => expect(screen.getByLabelText('战略范围名称')).toHaveValue('独立战略范围 2026-09-12'))
  await user.click(screen.getByRole('button', { name: '添加参考大类' }))
  expect(screen.queryByLabelText(/稳定ID/)).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('大类名称'), { target: { value: '初始名称' } })
  const original = readAllocationDraft<UniverseDefinition>('strategic-universe:editor:new:1')!
  const id = original.assets[0].id
  expect(id).toMatch(/^asset-[0-9a-f]{8}$/)
  // 补充说明默认折叠，不把非量化字段摆在主路径上。
  expect(screen.getByText('补充说明（非必填）').closest('details')).not.toHaveAttribute('open')
  expect(screen.getByText('备注（选填）').closest('details')).not.toHaveAttribute('open')
  expect(screen.getByLabelText('资产1备注').closest('label')).not.toHaveAttribute('data-required')
  expect(screen.getByLabelText('范围备注').closest('label')).not.toHaveAttribute('data-required')
  fireEvent.change(screen.getByLabelText('大类名称'), { target: { value: '增长资产' } })
  // 改名与改本位币都不会重新生成或改写稳定身份。
  fireEvent.change(screen.getByLabelText(/战略本位币/), { target: { value: 'usd' } })
  await user.click(screen.getByRole('button', { name: '保存战略范围' }))
  await screen.findByText(/只读战略范围/)

  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/confirm'))![1]?.body))
  expect(body.request.assets[0].id).toBe(id)
  expect(body.request.assets[0].rationale).toBe('')
  expect(body.request.assets[0].source).toBe('')
  expect(body.request.source).toBe('')
})

it('修改战略本位币会同步所有资产行，且不再逐行手填币种', async () => {
  const fetch = installEditorCatalog(); const user = userEvent.setup()
  writeAllocationDraft('strategic-universe:editor', definition); scopePage('/pre-investment/product-pool/new?scope=strategic&mandate=mandate-1')
  await screen.findByLabelText(/战略本位币/)

  expect(screen.queryByLabelText(/资产1风险计价币种/)).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText(/战略本位币/), { target: { value: 'usd' } })
  expect(screen.getByLabelText(/战略本位币/)).toHaveValue('USD')

  await user.click(screen.getByRole('button', { name: '保存战略范围' }))
  await screen.findByText(/只读战略范围/)
  const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/preview'))![1]?.body))
  expect(body.currency).toBe('USD')
  expect(body.assets.every((asset: { currency: string }) => asset.currency === 'USD')).toBe(true)
})

it('只选资产类型即可切换现金分类，非现金编辑保留原有变现限制', async () => {
  installEditorCatalog(); const user = userEvent.setup()
  const original = { ...definition, assets: [{ ...definition.assets[0], role: 'credit' as const, liquidity: 'illiquid' as const }] }
  writeAllocationDraft('strategic-universe:editor', original); scopePage()
  await screen.findByLabelText('大类名称')
  await user.click(screen.getByText('备注（选填）'))
  expect(screen.queryByLabelText(/配置用途|变现能力/)).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('大类名称'), { target: { value: '受限信用资产' } })
  const currentAsset = () => readAllocationDraft<UniverseDefinition>('strategic-universe:editor')!.assets[0]
  expect(currentAsset()).toMatchObject({ role: 'credit', liquidity: 'illiquid', research_proxy: { asset_type: 'market' } })
  await user.selectOptions(screen.getByLabelText('资产类型'), 'cash')
  expect(currentAsset()).toMatchObject({ role: 'liquidity', liquidity: 'liquid', research_proxy: { asset_type: 'cash', cash_return: 0 } })
  await user.selectOptions(screen.getByLabelText('资产类型'), 'market')
  expect(currentAsset()).toMatchObject({ role: 'growth', research_proxy: { asset_type: 'market', cash_return: null } })
  expect(original.assets[0]).toMatchObject({ role: 'credit', liquidity: 'illiquid' })
})

it('已保存、复制、返回与新建之间正确切换视图身份', async () => {
  // 编辑器独立成页后，继续研究/复制新建/新建都是从列表页真实跳转过去的，
  // 每次切身份都要先回列表页再点下一个链接——不再是同页局部刷新。
  install(); const user = userEvent.setup()
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool?scope=strategic']}><Routes>
    <Route path="/pre-investment/product-pool" element={<ProductPoolSelection />} />
    <Route path="/pre-investment/product-pool/new" element={<ProductPoolWorkspacePage />} />
  </Routes></MemoryRouter>)
  const backToList = () => user.click(screen.getByRole('link', { name: /返回研究范围库/ }))

  // 范围库 → 继续研究：只读摘要
  await user.click(await screen.findByRole('link', { name: '继续研究' }))
  expect(await screen.findByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  await backToList()

  // 复制新建（同一 ID）：可编辑副本，不残留只读摘要
  await user.click(await screen.findByRole('link', { name: '复制新建' }))
  expect(await screen.findByDisplayValue('独立战略范围（副本）')).toBeInTheDocument()
  expect(screen.queryByRole('table', { name: '只读战略资产摘要' })).not.toBeInTheDocument()
  expect(screen.queryByRole('link', { name: '下一步' })).not.toBeInTheDocument()
  // 系统身份保留在请求中，编辑器不展示内部 ID。
  expect(screen.queryByLabelText(/稳定ID/)).not.toBeInTheDocument()
  expect(screen.getAllByLabelText('大类名称')[0]).toHaveValue('增长资产')
  await backToList()

  // 返回原范围（同一 ID）：只读摘要恢复
  await user.click(await screen.findByRole('link', { name: '独立战略范围' }))
  expect(await screen.findByRole('table', { name: '只读战略资产摘要' })).toBeInTheDocument()
  await backToList()

  // 新建：全新草稿，无旧摘要与前向入口
  await user.click(await screen.findByRole('link', { name: '新建研究范围' }))
  expect(await screen.findByDisplayValue('独立战略范围 2026-09-12')).toBeInTheDocument()
  expect(screen.queryByRole('table', { name: '只读战略资产摘要' })).not.toBeInTheDocument()
  expect(screen.queryByRole('link', { name: '下一步' })).not.toBeInTheDocument()
  await backToList()

  // 新建 → 新建：两次点击各自拿到独立的新草稿 key，不会撞上前一次未保存的草稿。
  await user.click(await screen.findByRole('link', { name: '新建研究范围' }))
  expect(await screen.findByDisplayValue('独立战略范围 2026-09-12')).toBeInTheDocument()
  expect(screen.queryByRole('table', { name: '只读战略资产摘要' })).not.toBeInTheDocument()
})
