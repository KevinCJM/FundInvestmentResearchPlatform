import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LtcmaCenter from './LtcmaCenter'
import LtcmaWorkspace from './LtcmaWorkspace'
import LtcmaVersionView from './LtcmaVersionView'
import { ltcmaCapabilities, ltcmaDefinition, ltcmaItem, ltcmaOptions, ltcmaVersion } from '../test/ltcmaFixtures'
import { applyScope, changeMethod, newDraft, proxyFor } from '../components/ltcma/model'
import { cmaDraftFromDefinition } from '../services/strategicAllocation'
import type { UniverseVersion } from '../services/strategicScope'
import { strategicCatalog } from '../test/strategicAllocationFixtures'

const clock = vi.hoisted(() => ({ day: '2026-09-18' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
const response = (value: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => value } as Response)
const root = '/api/strategic-allocation'
function install(overrides: Record<string, (init: RequestInit | undefined, url: URL) => Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input), path = url.slice(root.length).split('?')[0]
    if (overrides[path]) return overrides[path](init, new URL(url, 'http://localhost'))
    if (path === '/cma/capabilities') return response(ltcmaCapabilities)
    if (path === '/cma/study-options') return response(ltcmaOptions)
    if (path === '/cma/drafts' && init?.method === 'POST') {
      const data = JSON.parse(String(init.body))
      return response({ ...data, id: 'draft-ltcma', revision: 1, created_at: '2026-09-18', updated_at: '2026-09-18' })
    }
    if (path === '/cma/drafts') return response({ items: [] })
    if (path === '/cma/cma-1') return response(init?.method === 'PATCH' ? { ...ltcmaVersion, definition: JSON.parse(String(init.body)).request } : ltcmaVersion)
    if (path === '/cma/cma-1/view') return response({ version: ltcmaVersion, retired: false })
    if (path === '/cma/cma-1/retire') return response({ id: 'cma-1', retired: true })
    if (path === '/cma/preview') return response({ ...ltcmaVersion, definition: JSON.parse(String(init?.body)) })
    if (path === '/cma' && init?.method === 'POST') return response({ ...ltcmaVersion, definition: JSON.parse(String(init.body)).request })
    if (path === '/cma') return response({ items: [ltcmaItem], offset: 0, limit: 20, total: 1 })
    throw new Error(`Unexpected API ${url}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function tree(path: string) {
  return <MemoryRouter initialEntries={[path]}><Routes>
    <Route path="/pre-investment/ltcma" element={<LtcmaCenter />} />
    <Route path="/pre-investment/ltcma/new" element={<LtcmaWorkspace />} />
    <Route path="/pre-investment/ltcma/:versionId" element={<LtcmaVersionView />} />
    <Route path="/pre-investment/saa/policy" element={<p>SAA destination</p>} />
  </Routes></MemoryRouter>
}
beforeEach(() => { clock.day = '2026-09-18'; localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

describe('confirmed CMA deletion', () => {
  // jsdom has no native dialog methods; browser tests cover modality and keyboard behavior.
  beforeEach(() => {
    Object.defineProperties(HTMLDialogElement.prototype, {
      showModal: { configurable: true, value(this: HTMLDialogElement) { this.open = true } },
      close: { configurable: true, value(this: HTMLDialogElement) { this.open = false } },
    })
  })
  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(HTMLDialogElement.prototype, 'showModal')
    Reflect.deleteProperty(HTMLDialogElement.prototype, 'close')
  })
  it('explains retained history and lets Cancel and Escape close without a mutation', async () => {
    const fetch = install(), user = userEvent.setup(); render(tree('/pre-investment/ltcma'))
    const trigger = await screen.findByRole('button', { name: '删除' })
    await user.click(trigger)
    const dialog = screen.getByRole('dialog', { name: '删除此 LTCMA 版本？' })
    expect(dialog).toHaveTextContent(ltcmaItem.name)
    expect(dialog).toHaveTextContent('已有 SAA、TAA 等研究引用及历史记录会保留')
    expect(within(dialog).getByRole('button', { name: '取消' })).toHaveFocus()
    await user.click(within(dialog).getByRole('button', { name: '取消' }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(trigger).toHaveFocus()
    await user.click(trigger); fireEvent(screen.getByRole('dialog'), new Event('cancel', { cancelable: true }))
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(fetch.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true)
  })
  it('keeps failures retryable, prevents duplicate requests and clears the retired selection after success', async () => {
    let attempts = 0, retired = false, finish!: (value: Response) => void
    const fetch = install({
      '/cma': (_, url) => response({ items: retired && url.searchParams.get('include_retired') !== 'true' ? [] : [{ ...ltcmaItem, retired }], total: retired ? 0 : 1, offset: 0, limit: 20 }),
      '/cma/cma-1/retire': () => ++attempts === 1 ? response({ detail: 'CMA_VERSION_CHANGED' }, 409) : new Promise(resolve => { finish = resolve }),
    })
    const user = userEvent.setup(); render(tree('/pre-investment/ltcma'))
    await user.click(await screen.findByRole('checkbox', { name: `选择 ${ltcmaItem.name}` }))
    await user.click(screen.getByRole('button', { name: '删除' }))
    const dialog = screen.getByRole('dialog'), confirm = within(dialog).getByRole('button', { name: '确认删除' })
    await user.click(confirm)
    expect(await within(dialog).findByRole('alert')).toHaveTextContent('CMA_VERSION_CHANGED')
    expect(screen.getByRole('region', { name: '用于 SAA 的 CMA' })).toHaveTextContent('已选 1 个 CMA')
    await user.click(confirm); await user.click(confirm)
    expect(confirm).toBeDisabled()
    expect(within(dialog).getByRole('button', { name: '取消' })).toBeDisabled()
    fireEvent(dialog, new Event('cancel', { cancelable: true })); expect(dialog).toBeVisible()
    expect(attempts).toBe(2)
    retired = true
    await act(async () => { finish(await response({ id: ltcmaItem.id, retired: true })) })
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(screen.queryByRole('link', { name: ltcmaItem.name })).not.toBeInTheDocument()
    expect(screen.getByRole('region', { name: '用于 SAA 的 CMA' })).toHaveTextContent('已选 0 个 CMA')
    expect(screen.getByRole('button', { name: '下一步：战略资产配置 →' })).toBeDisabled()
    expect(screen.getByText(`已删除“${ltcmaItem.name}”：停止新引用，历史记录已保留。`)).toBeVisible()
    const calls = fetch.mock.calls.filter(([url]) => String(url).endsWith('/retire'))
    for (const [, init] of calls) expect(JSON.parse(String(init?.body))).toMatchObject({ confirm: true, content_hash: ltcmaItem.content_hash })
    await user.click(screen.getByRole('checkbox', { name: '显示已停止引用的版本' }))
    expect(await screen.findByRole('link', { name: ltcmaItem.name })).toBeVisible()
    expect(screen.getByRole('button', { name: '删除' })).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: `选择 ${ltcmaItem.name}` })).toBeDisabled()
    expect(screen.getByRole('link', { name: '修改' })).toBeVisible()
  })
  it('returns to the previous page after deleting its last visible version', async () => {
    let retired = false
    install({
      '/cma': (_, url) => response({ items: Number(url.searchParams.get('offset')) ? [ltcmaItem] : [{ ...ltcmaItem, id: 'another', name: '上一页版本' }], total: retired ? 20 : 21, offset: Number(url.searchParams.get('offset')), limit: 20 }),
      '/cma/cma-1/retire': () => { retired = true; return response({ id: ltcmaItem.id, retired: true }) },
    })
    const user = userEvent.setup(); render(tree('/pre-investment/ltcma'))
    await user.click(await screen.findByRole('button', { name: '下一页' }))
    await screen.findByRole('link', { name: ltcmaItem.name })
    await user.click(screen.getByRole('button', { name: '删除' }))
    await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: '确认删除' }))
    expect(await screen.findByRole('link', { name: '上一页版本' })).toBeVisible()
    expect(screen.getByText('第 1 页，共 20 条')).toBeVisible()
    expect(screen.getByRole('button', { name: '下一页' })).toBeDisabled()
  })
})

describe('LTCMA independent workflow', () => {
  it('inherits a delayed bound investment horizon when starting conditional research', async () => {
    let resolveGoal!: (value: Response) => void
    const goal = { ...strategicCatalog.mandates[0], id: 'bound-goal', definition: { ...strategicCatalog.mandates[0].definition, horizon_years: 3 } }
    const scope: UniverseVersion = { id: 'bound-scope', name: '目标资产范围', created_at: clock.day!, content_hash: 'a'.repeat(64), preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped', implementation_gaps: [], mandate_id: goal.id,
      definition: { name: '目标资产范围', as_of: clock.day!, currency: 'CNY', source: '', assets: ltcmaDefinition.assets.map(asset => ({ id: asset.id, name: asset.id, role: asset.role, liquidity: asset.liquidity, currency: 'CNY', rationale: '', source: '' })) } }
    install({ '/cma/study-options': () => response({ ...ltcmaOptions, strategic_universes: [scope] }), '/mandates/bound-goal': () => new Promise(resolve => { resolveGoal = resolve }) })
    render(tree('/pre-investment/ltcma/new?strategic_universe=bound-scope'))
    fireEvent.click(await screen.findByLabelText('生成方法'))
    fireEvent.click(screen.getByRole('menuitemradio', { name: '条件情景' }))
    expect(screen.getByText('结合当前市场状态，研究指定区间的收益与风险。结果用于情景研究，暂不用于 SAA。')).toBeVisible()
    expect(screen.queryByText('研究资产的收益与风险，符合条件的长期假设可供 SAA 使用。这里不决定组合权重。')).not.toBeInTheDocument()
    expect(await screen.findByLabelText('预测区间')).toHaveValue('126')
    await act(async () => resolveGoal(await response(goal)))
    await waitFor(() => expect(screen.getByLabelText('预测区间')).toHaveValue('756'))
    fireEvent.change(screen.getByLabelText('预测区间'), { target: { value: '252' } })
    expect(screen.getByLabelText('预测区间')).toHaveValue('252')
  })
  it('drops retired horizon labels from copied versions and restored drafts without changing the original', () => {
    const original = { ...ltcmaDefinition, horizon_years: 5 }
    expect(cmaDraftFromDefinition(original)).not.toHaveProperty('horizon_years')
    expect(newDraft('2026-09-18')).not.toHaveProperty('horizon_years')
    expect(original.horizon_years).toBe(5)
  })
  it('restores saved strategic proxies and cash inputs into every statistical method without mutating history', () => {
    const scope: UniverseVersion = { id: 'scope-with-proxies', name: '配置范围', created_at: '2026-09-18', content_hash: 'a'.repeat(64), preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped', implementation_gaps: ['equity', 'cash'],
      definition: { name: '配置范围', as_of: '2026-09-18', currency: 'CNY', source: '', assets: [
        { id: 'equity', name: '中国股票', currency: 'CNY', role: 'growth', liquidity: 'liquid', rationale: '', source: '', research_proxy: { asset_type: 'market', cash_return: null, rebalance: 'monthly', components: [{ kind: 'index', series_id: 'index:index_daily:000300.SH', field: 'close', weight: 1 }], source_labels: { 'index:index_daily:000300.SH': '沪深300' } } },
        { id: 'cash', name: '现金', currency: 'CNY', role: 'liquidity', liquidity: 'liquid', rationale: '', source: '', research_proxy: { asset_type: 'cash', cash_return: .02, components: [], rebalance: null, source_labels: {} } },
      ] } }
    const original = JSON.stringify(scope)
    const manual = applyScope(newDraft('2026-09-18'), `universe:${scope.id}`, { ...ltcmaOptions, strategic_universes: [scope] })
    expect(manual.assets[0].annual_return).toBeNaN()
    expect(manual.assets[1]).toMatchObject({ annual_return: .02, annual_volatility: 0, role: 'liquidity' })
    expect(manual.correlation).toEqual([[1, 0], [0, 1]])
    for (const method of ['historical_statistics', 'bayesian_niw', 'historical_regime_occupancy'] as const) {
      const statistical = changeMethod(manual, method, scope)
      expect(statistical.model).toMatchObject({ proxy_inputs: { assets: [
        { id: 'equity', name: '中国股票', rebalance: 'monthly', components: scope.definition.assets[0].research_proxy!.components },
        { id: 'cash', asset_type: 'cash', cash_return: .02, components: [], rebalance: null },
      ] } })
      expect(statistical.implementation_mapping_id).toBeNull()
    }
    expect(JSON.stringify(scope)).toBe(original)
  })
  it('lists saved versions without recomputing or needing a mandate', async () => {
    const fetch = install(); render(tree('/pre-investment/ltcma'))
    expect(await screen.findByRole('link', { name: ltcmaVersion.name })).toHaveAttribute('href', '/pre-investment/ltcma/cma-1')
    expect(screen.getByRole('button', { name: '新建 LTCMA' })).toBeEnabled()
    expect(screen.getByRole('link', { name: '修改' })).toHaveAttribute('href', '/pre-investment/ltcma/new?edit=cma-1')
    expect(screen.getByRole('link', { name: '复制为新研究' })).toHaveAttribute('href', '/pre-investment/ltcma/new?copy=cma-1')
    expect(screen.getByRole('button', { name: '删除' })).toBeEnabled()
    expect(screen.getByText(/LTCMA（Long-Term Capital Market Assumptions，长期资本市场假设）/)).toBeVisible()
    expect(screen.getByText(/均值方差优化可能放大参数估计误差/)).toBeVisible()
    expect(fetch.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true)
  })
  it('distinguishes equal names with sample windows while hiding detailed evidence and the repeated scope column', async () => {
    const history = { window: { kind: '5Y' }, start_date: '2015-01-05', end_date: '2019-12-31', observations: 1200,
      source_names: ['沪深300', '中证国债', '黄金指数', '海外股票'] }
    const fetch = install({ '/cma': () => response({ items: [
      { ...ltcmaItem, id: 'five-years', method: 'historical_statistics', history },
      { ...ltcmaItem, id: 'one-year', method: 'historical_statistics', history: { ...history,
        window: { kind: '1Y' }, start_date: '2019-01-02', observations: 244, source_names: ['沪深300', '中证国债'] } },
    ], total: 2, offset: 0, limit: 20 }) })
    render(tree('/pre-investment/ltcma'))
    const table = await screen.findByRole('table', { name: '已确认版本' })
    expect(within(table).queryByRole('columnheader', { name: '资产范围' })).not.toBeInTheDocument()
    expect(within(table).getByText('历史统计 · 近 5 年')).toBeVisible()
    expect(within(table).getByText('历史统计 · 近 1 年')).toBeVisible()
    expect(table).not.toHaveTextContent('个日收益样本')
    expect(table).not.toHaveTextContent('2019-01-02')
    expect(table).not.toHaveTextContent('沪深300')
    expect(fetch.mock.calls).toHaveLength(3)
  })
  it.each(['复制为新研究', '修改'])('%s uses its own save endpoint and lineage', async action => {
    const fetch = install({ '/cma/study-options': () => response({ ...ltcmaOptions, existing_names: [ltcmaDefinition.name] }) }), user = userEvent.setup(); render(tree('/pre-investment/ltcma'))
    await user.click(await screen.findByRole('link', { name: action }))
    await screen.findByLabelText('名称')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent(action === '修改' ? '修改 LTCMA' : action)
    expect(screen.getByText(action === '修改' ? '保存后更新当前 LTCMA 方案，列表不会新增一份；已有研究引用保留其当时的结果。' : '复制将创建独立的新研究，请使用不同名称；原方案保持不变。')).toBeVisible()
    expect(screen.getByLabelText('名称')).toHaveValue(ltcmaDefinition.name)
    expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    if (action === '复制为新研究') {
      expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
      fireEvent.change(screen.getByLabelText('名称'), { target: { value: '独立研究' } })
    }
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    const save = await screen.findByRole('button', { name: action === '修改' ? '保存修改' : '确认保存版本' })
    expect(save).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已阅读结果和限制/ }))
    await user.click(save)
    expect(await screen.findByRole('button', { name: '用于 SAA' })).toBeEnabled()
    const published = fetch.mock.calls.find(([url, init]) => action === '修改'
      ? String(url) === `${root}/cma/cma-1` && init?.method === 'PATCH'
      : String(url) === `${root}/cma` && init?.method === 'POST')!
    const body = JSON.parse(String(published[1]?.body))
    expect(body).toMatchObject({ confirm: true, preview_hash: ltcmaVersion.preview_hash,
      ...(action === '修改' ? { expected_content_hash: ltcmaVersion.content_hash } : { copied_from_id: 'cma-1' }) })
    if (action === '修改') expect(fetch.mock.calls.some(([url, init]) => String(url) === `${root}/cma` && init?.method === 'POST')).toBe(false)
    expect(body.request).toMatchObject({ schema_version: '2.0', implementation_mapping_id: null })
    expect(body.request).not.toHaveProperty('mandate_id')
    expect(body.idempotency_key).toBeTruthy()
    await user.click(screen.getByRole('button', { name: '用于 SAA' }))
    expect(await screen.findByText('SAA destination')).toBeVisible()
  })
  it('restores edit identity from a draft and keeps failed updates retryable without creating a study', async () => {
    let stored: Record<string, unknown> = {}, attempts = 0
    const fetch = install({
      '/cma/study-options': () => response({ ...ltcmaOptions, existing_names: [ltcmaDefinition.name] }),
      '/cma/drafts': init => {
        stored = { ...JSON.parse(String(init?.body)), id: 'edit-draft', revision: 1 }
        return response(stored)
      },
      '/cma/drafts/edit-draft': () => response(stored),
      '/cma/cma-1': init => init?.method !== 'PATCH' ? response(ltcmaVersion) : ++attempts === 1
        ? response({ detail: { message: '临时保存失败，请重试' } }, 503) : response(ltcmaVersion),
    })
    const user = userEvent.setup(), first = render(tree('/pre-investment/ltcma/new?edit=cma-1'))
    await screen.findByLabelText('名称')
    await user.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('草稿已保存，可从列表继续编辑。')
    expect(stored).toMatchObject({ copied_from_id: null, editing_ref: { id: 'cma-1', content_hash: ltcmaVersion.content_hash } })
    first.unmount(); render(tree('/pre-investment/ltcma/new?draft=edit-draft'))
    await screen.findByLabelText('名称')
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('修改 LTCMA')
    await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    const save = await screen.findByRole('button', { name: '保存修改' })
    await user.click(screen.getByRole('checkbox', { name: /我已阅读结果和限制/ }))
    await user.click(save)
    expect(await screen.findByRole('alert')).toHaveTextContent('临时保存失败，请重试')
    expect(save).toBeEnabled()
    await user.click(save)
    await screen.findByRole('button', { name: '用于 SAA' })
    const updates = fetch.mock.calls.filter(([, init]) => init?.method === 'PATCH')
    expect(updates).toHaveLength(2)
    expect(updates[0][1]?.body).toEqual(updates[1][1]?.body)
    expect(fetch.mock.calls.some(([url, init]) => String(url) === `${root}/cma` && init?.method === 'POST')).toBe(false)
  })
  it('saves incomplete inputs as JSON nulls, without calculation', async () => {
    const fetch = install(), user = userEvent.setup(); render(tree('/pre-investment/ltcma/new'))
    await screen.findByLabelText('名称')
    await user.type(screen.getByLabelText('名称'), '未完成的研究')
    await user.selectOptions(screen.getByLabelText('资产范围'), `allocation:${ltcmaDefinition.alloc_name}`)
    await user.click(screen.getByRole('button', { name: '保存草稿' }))
    expect(await screen.findByText('草稿已保存，可从列表继续编辑。')).toBeVisible()
    const call = fetch.mock.calls.find(([url, init]) => String(url) === `${root}/cma/drafts` && init?.method === 'POST')!
    expect(JSON.parse(String(call[1]?.body)).editable_definition.definition.assets[0].annual_return).toBeNull()
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/preview'))).toBe(false)
  })
  it('renders statistical methods separately from the BL/scenario editor', async () => {
    install(); const user = userEvent.setup(); render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('生成方法')
    await user.click(screen.getByLabelText('生成方法'))
    await user.click(screen.getByRole('menuitemradio', { name: '贝叶斯更新（NIW）' }))
    expect(screen.getByLabelText('先验 LTCMA 版本')).toBeVisible()
    expect(screen.getByLabelText('均值先验等效日观察数')).toHaveValue('')
    expect(screen.queryByLabelText('市场风险厌恶系数 δ')).not.toBeInTheDocument()
    await user.click(screen.getByLabelText('生成方法'))
    await user.click(screen.getByRole('menuitemradio', { name: '历史情景' }))
    expect(screen.getByLabelText('已保存的事后状态研究')).toBeVisible()
    expect(screen.queryByLabelText('先验 LTCMA 版本')).not.toBeInTheDocument()
  })
  it('invalidates a pending calculation when the knowledge clock changes', async () => {
    let finish!: (value: Response) => void
    install({ '/cma/preview': () => new Promise(resolve => { finish = resolve }) })
    const user = userEvent.setup(), rendered = render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('名称'); await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    clock.day = '2026-09-01'; rendered.rerender(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await act(async () => { finish(await response(ltcmaVersion)) })
    expect(screen.queryByRole('button', { name: '确认保存版本' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
  })
  it('auto-names a study and accepts empty optional notes without changing numeric inputs', async () => {
    const blank = { ...ltcmaVersion, definition: { ...ltcmaDefinition, name: '', source: '',
      assets: ltcmaDefinition.assets.map(asset => ({ ...asset, rationale: '' })) } }
    const fetch = install({ '/cma/cma-1': () => response(blank) })
    const user = userEvent.setup(); render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('名称')
    expect(screen.getByLabelText('名称')).toBeVisible()
    expect(screen.queryByLabelText('预测期限（年）')).not.toBeInTheDocument()
    expect(screen.getByLabelText('研究日')).toBeDisabled()
    expect(screen.getByLabelText('研究日')).toHaveValue(clock.day)
    await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    await screen.findByRole('button', { name: '确认保存版本' })
    const body = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/cma/preview'))![1]?.body))
    expect(body.name).toContain('直接假设')
    expect(body.name).toContain(clock.day)
    expect(body.source).toBe('')
    expect(body).not.toHaveProperty('horizon_years')
    expect(screen.queryByText(/预测期限/)).not.toBeInTheDocument()
    expect(body.assets.map((a: { rationale: string }) => a.rationale)).toEqual(['', ''])
    expect(body.correlation).toEqual(ltcmaDefinition.correlation)
  })
  it('blocks the preview while the study name repeats a saved LTCMA', async () => {
    install({ '/cma/study-options': () => response({ ...ltcmaOptions, existing_names: ['已占用的名称'] }) })
    const user = userEvent.setup(); render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    const field = await screen.findByLabelText('名称')
    await user.clear(field); await user.type(field, ' 已占用的名称 ')
    expect(await screen.findByRole('alert')).toHaveTextContent('LTCMA 名称已存在')
    expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
    await user.clear(field); await user.type(field, '另一个名称')
    expect(screen.queryByText(/LTCMA 名称已存在/)).not.toBeInTheDocument()
  })
  it('follows PIT when loading an older draft, allows editing only when PIT is explicitly off', async () => {
    const raw = { ...ltcmaDefinition, as_of: '2020-01-01', basis_confirmed: true }
    install({ '/cma/drafts/old': () => response({ id: 'old', revision: 1, name: raw.name, copied_from_id: null, editable_definition: { definition: raw } }) })
    const rendered = render(tree('/pre-investment/ltcma/new?draft=old'))
    await screen.findByLabelText('研究日')
    expect(screen.getByLabelText('研究日')).toHaveValue('2026-09-18')
    expect(screen.getByLabelText('研究日')).toBeDisabled()
    expect(screen.getByRole('checkbox', { name: /我已核对资产范围/ })).not.toBeChecked()
    clock.day = null; rendered.rerender(tree('/pre-investment/ltcma/new?draft=old'))
    expect(screen.getByLabelText('研究日')).toBeEnabled()
    fireEvent.change(screen.getByLabelText('研究日'), { target: { value: '2026-08-01' } })
    expect(screen.getByLabelText('研究日')).toHaveValue('2026-08-01')
    clock.day = undefined; rendered.rerender(tree('/pre-investment/ltcma/new?draft=old'))
    expect(screen.getByLabelText('研究日')).toBeDisabled()
    expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeDisabled()
    clock.day = '2026-09-01'; rendered.rerender(tree('/pre-investment/ltcma/new?draft=old'))
    await waitFor(() => expect(screen.getByLabelText('研究日')).toHaveValue('2026-09-01'))
  })
  it('keeps history readable after retirement but blocks new handoff', async () => {
    const fetch = install(), user = userEvent.setup(); render(tree('/pre-investment/ltcma/cma-1'))
    await screen.findByRole('button', { name: '停止新引用' })
    await user.click(screen.getByRole('button', { name: '停止新引用' }))
    await user.type(screen.getByLabelText('停止引用的原因（至少 5 字）'), '长期假设需要重新复核')
    await user.click(screen.getByRole('checkbox', { name: '确认停止此版本的新引用' }))
    await user.click(screen.getByRole('button', { name: '确认停止此版本的新引用' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '用于 SAA' })).toBeDisabled())
    expect(screen.getByRole('table', { name: '收益与风险假设' })).toBeVisible()
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/preview'))).toBe(false)
  })
  it('clears preview after changing a confirmed basis', async () => {
    install(); const user = userEvent.setup(); render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('名称'); await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' })); await screen.findByRole('button', { name: '确认保存版本' })
    await user.click(screen.getByRole('button', { name: '返回修改输入' }))
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: 'Updated assumptions' } })
    expect(screen.getByRole('button', { name: '2. 结果与确认' })).toBeDisabled()
  })
})

describe('LTCMA draft identity', () => {
  it.each(['historical_statistics', 'bayesian_niw', 'historical_regime_occupancy'] as const)(
    'clears strategic proxy inputs when %s switches to a product allocation', method => {
      const raw = { ...cmaDraftFromDefinition(ltcmaDefinition), alloc_name: null, strategic_universe_id: 'universe-1' }
      const strategic = changeMethod(raw, method)
      expect(strategic.model).toHaveProperty('proxy_inputs')
      const product = applyScope(strategic, `allocation:${ltcmaDefinition.alloc_name}`, ltcmaOptions)
      expect(product.alloc_name).toBe(ltcmaDefinition.alloc_name)
      expect(product.strategic_universe_id).toBeNull()
      expect(product.model?.method).toBe(method)
      expect(product.model).toHaveProperty('proxy_inputs', undefined)
      expect(product.basis_confirmed).toBe(false)
      expect(product.risk_reference_hash).toBeNull()
    },
  )
  it('preserves a fixed strategic proxy axis across method changes', () => {
    const raw = { ...cmaDraftFromDefinition(ltcmaDefinition), alloc_name: null, strategic_universe_id: 'universe-1' }
    const historical = changeMethod(raw, 'historical_statistics')
    expect(historical.model?.method).toBe('historical_statistics')
    const proxies = proxyFor(historical)
    expect(proxies.name).toBeTruthy()
    expect(proxies.fee_basis).toBe('source_embedded_no_additional_fee')
    expect(proxies.assets.map(asset => asset.id)).toEqual(raw.assets.map(asset => asset.id))
    expect(newDraft('2026-09-18').assets).toEqual([])
  })
})

describe('method studies load independently of the editor', () => {
  const baseOptions = { ...ltcmaOptions, assumptions: [], regime_runs: [] }
  const selectMethod = async (name: string) => {
    fireEvent.click(screen.getByLabelText('生成方法'))
    fireEvent.click(screen.getByRole('menuitemradio', { name }))
  }
  it('requests the exact saved NIW prior even when it is absent from current choices', async () => {
    const definition = changeMethod(cmaDraftFromDefinition(ltcmaDefinition), 'bayesian_niw')
    if (definition.model?.method !== 'bayesian_niw') throw new Error('NIW fixture required')
    definition.model.prior_ref = { id: ltcmaItem.id, content_hash: ltcmaItem.content_hash }
    const fetch = install({
      '/cma/drafts/niw': () => response({ id: 'niw', revision: 1, name: '保留原先验', copied_from_id: null, editable_definition: { definition } }),
      '/cma/study-options': (_, url) => response(url.searchParams.get('section') === 'base' ? baseOptions
        : { assumptions: url.searchParams.get('selected_prior_id') === ltcmaItem.id ? [ltcmaItem] : [] }),
    })
    render(tree('/pre-investment/ltcma/new?draft=niw'))
    expect(await screen.findByLabelText('先验 LTCMA 版本')).toHaveValue(ltcmaItem.id)
    expect(fetch.mock.calls.some(([url]) => String(url).includes(`selected_prior_id=${ltcmaItem.id}`))).toBe(true)
  })
  it('loads only base options initially and preserves edits and draft saving while priors wait', async () => {
    let finish!: (value: Response) => void
    const fetch = install({ '/cma/study-options': (_, url) => url.searchParams.get('section') === 'base'
      ? response(baseOptions) : new Promise(resolve => { finish = resolve }) })
    render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('资产范围')
    expect(fetch.mock.calls.filter(([url]) => String(url).includes('study-options')).map(([url]) => new URL(String(url), 'http://localhost').searchParams.get('section'))).toEqual(['base'])
    await selectMethod('贝叶斯更新（NIW）')
    expect(screen.queryByLabelText('先验 LTCMA 版本')).not.toBeInTheDocument()
    expect(screen.queryByText('没有同资产范围、币种及收益口径的先验。请先保存一份适用 LTCMA。')).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '等待期间编辑' } })
    expect(screen.getByRole('button', { name: '保存草稿' })).toBeEnabled()
    fireEvent.click(screen.getByRole('button', { name: '保存草稿' }))
    await screen.findByText('草稿已保存，可从列表继续编辑。')
    await act(async () => finish(await response({ assumptions: [ltcmaItem] })))
    expect(await screen.findByLabelText('先验 LTCMA 版本')).toBeVisible()
    expect(screen.getByLabelText('名称')).toHaveValue('等待期间编辑')
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/preview'))).toBe(false)
  })
  it('keeps a failed method read local and retries without resetting the form', async () => {
    let attempts = 0
    const fetch = install({ '/cma/study-options': (_, url) => url.searchParams.get('section') === 'base'
      ? response(baseOptions) : ++attempts === 1 ? response({ detail: 'STUDIES_OFFLINE' }, 503) : response({ regime_runs: [] }) })
    render(tree('/pre-investment/ltcma/new?copy=cma-1')); await screen.findByLabelText('资产范围')
    fireEvent.change(screen.getByLabelText('名称'), { target: { value: '保留我的研究' } })
    await selectMethod('历史情景')
    await screen.findByRole('button', { name: '重试' })
    expect(screen.queryByLabelText('已保存的事后状态研究')).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '重试' }))
    expect(await screen.findByLabelText('已保存的事后状态研究')).toBeVisible()
    expect(screen.getByLabelText('名称')).toHaveValue('保留我的研究')
    expect(attempts).toBe(2)
    expect(fetch.mock.calls.filter(([url]) => String(url).includes('section=base'))).toHaveLength(1)
  })
  it('ignores old responses after a method or research-date switch even when transport ignores abort', async () => {
    const pending: Array<{ section: string; day: string; signal?: AbortSignal | null; finish: (value: Response) => void }> = []
    install({ '/cma/study-options': (init, url) => url.searchParams.get('section') === 'base' ? response(baseOptions)
      : new Promise(resolve => pending.push({ section: url.searchParams.get('section')!, day: url.searchParams.get('as_of')!, signal: init?.signal, finish: resolve })) })
    const view = render(tree('/pre-investment/ltcma/new?copy=cma-1')); await screen.findByLabelText('资产范围')
    await selectMethod('贝叶斯更新（NIW）')
    await selectMethod('历史情景')
    expect(pending[0].signal?.aborted).toBe(true)
    await act(async () => pending[0].finish(await response({ detail: 'OLD_METHOD_ERROR' }, 503)))
    expect(screen.queryByText('OLD_METHOD_ERROR')).not.toBeInTheDocument()
    clock.day = '2026-09-17'; view.rerender(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await waitFor(() => expect(pending[pending.length - 1]?.day).toBe('2026-09-17'))
    expect(pending[1].signal?.aborted).toBe(true)
    await act(async () => pending[1].finish(await response({ regime_runs: [{ id: 'stale', name: 'STALE_STUDY', frequency: 'daily', as_of: '2020-01-01', states: [] }] })))
    expect(screen.queryByText('STALE_STUDY')).not.toBeInTheDocument()
    await act(async () => pending[pending.length - 1]!.finish(await response({ regime_runs: [] })))
    expect(await screen.findByLabelText('已保存的事后状态研究')).toBeVisible()
    expect(screen.getByLabelText('研究日')).toHaveValue('2026-09-17')
  })
})

describe('LTCMA groups by objective, path and scope', () => {
  afterEach(() => cleanup())
  const ref = (kind: 'mandate' | 'strategic_scope', id: string, name: string, status: 'current' | 'superseded' | 'retired' = 'current') =>
    ({ kind, id, name, number: 1, latest_id: status === 'superseded' ? `${id}-v2` : status === 'current' ? id : null,
      latest_number: status === 'superseded' ? 2 : status === 'current' ? 1 : null, status, usable: 'ready' as const })
  const ready = { status: 'ready' as const, reasons: [] }
  it('shows linked group headers with versions and only allows ready items inside one group', async () => {
    const upstream = [ref('mandate', 'mandate-a', '目标甲'), ref('strategic_scope', 'scope-a', '范围甲')]
    const group = { research_path: 'strategy_first' as const, strategic_universe_id: 'scope-a', upstream, usable: ready }
    const items = [
      { ...ltcmaItem, ...group, id: 'cma-a1', name: '甲一' },
      { ...ltcmaItem, ...group, id: 'cma-a2', name: '甲二', version: { ...ltcmaItem.version, number: 3 } },
      { ...ltcmaItem, ...group, id: 'cma-x', name: '上游已删', retired: true,
        upstream: [upstream[0], ref('strategic_scope', 'scope-a', '范围甲', 'retired')],
        usable: { status: 'blocked' as const, reasons: [{ code: 'upstream_deleted' as const, kind: 'strategic_scope', name: '范围甲' }] } },
      { ...ltcmaItem, ...group, id: 'cma-b1', name: '乙一', strategic_universe_id: 'scope-b',
        upstream: [ref('mandate', 'mandate-a', '目标甲', 'superseded'), ref('strategic_scope', 'scope-b', '范围乙')],
        usable: { status: 'stale' as const, reasons: [{ code: 'upstream_superseded' as const, kind: 'mandate', name: '目标甲', number: 1, latest_number: 2 }] } },
    ]
    install({ '/cma': () => response({ items, offset: 0, limit: 20, total: items.length }) })
    render(tree('/pre-investment/ltcma'))
    const table = await screen.findByRole('table', { name: '已确认版本' })
    const headers = within(table).getAllByRole('rowheader').filter(cell => cell.getAttribute('scope') === 'rowgroup')
    expect(headers).toHaveLength(2) // 后端已按组排序；分组只看目标与范围版本 ID
    expect(within(headers[0]).getByRole('link', { name: '目标甲' })).toHaveAttribute('href', '/pre-investment/objectives/new?view=mandate-a')
    expect(within(headers[0]).getByRole('link', { name: '范围甲' })).toHaveAttribute('href', '/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-a')
    expect(headers[0]).toHaveTextContent('Strategy first')
    expect(headers[0]).toHaveTextContent('v1')
    expect(headers[1]).toHaveTextContent('已有新版本 v2')
    expect(screen.getByRole('rowheader', { name: /甲二/ })).toHaveTextContent('v3')
    await userEvent.click(screen.getByRole('checkbox', { name: /甲一/ }))
    expect(screen.getByRole('checkbox', { name: /甲二/ })).toBeEnabled()
    expect(screen.getByRole('checkbox', { name: /乙一/ })).toBeDisabled()
    expect(screen.getByText('投资目标与约束「目标甲」已有新版本 v2；请基于新版本修改后再用于下一步。')).toBeInTheDocument()
    expect(screen.getByText('研究范围「范围甲」已删除，已自动停止引用。')).toBeInTheDocument()
  })
})
