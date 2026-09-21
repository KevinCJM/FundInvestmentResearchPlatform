import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import LtcmaCenter from './LtcmaCenter'
import LtcmaWorkspace from './LtcmaWorkspace'
import LtcmaVersionView from './LtcmaVersionView'
import { ltcmaCapabilities, ltcmaDefinition, ltcmaItem, ltcmaOptions, ltcmaVersion } from '../test/ltcmaFixtures'
import { applyScope, changeMethod, newDraft, proxyFor } from '../components/ltcma/model'
import { cmaDraftFromDefinition } from '../services/strategicAllocation'

const clock = vi.hoisted(() => ({ day: '2026-09-18' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
const response = (value: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => value } as Response)
const root = '/api/strategic-allocation'
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input), path = url.slice(root.length).split('?')[0]
    if (overrides[path]) return overrides[path](init)
    if (path === '/cma/capabilities') return response(ltcmaCapabilities)
    if (path === '/cma/study-options') return response(ltcmaOptions)
    if (path === '/cma/drafts' && init?.method === 'POST') {
      const data = JSON.parse(String(init.body))
      return response({ ...data, id: 'draft-ltcma', revision: 1, created_at: '2026-09-18', updated_at: '2026-09-18' })
    }
    if (path === '/cma/drafts') return response({ items: [] })
    if (path === '/cma/cma-1') return response(ltcmaVersion)
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

describe('LTCMA independent workflow', () => {
  it('lists saved versions without recomputing or needing a mandate', async () => {
    const fetch = install(); render(tree('/pre-investment/ltcma'))
    expect(await screen.findByRole('link', { name: ltcmaVersion.name })).toHaveAttribute('href', '/pre-investment/ltcma/cma-1')
    expect(screen.getByRole('button', { name: '新建 LTCMA' })).toBeEnabled()
    expect(fetch.mock.calls.every(([, init]) => init?.method === 'GET')).toBe(true)
  })
  it('copies, previews, explicitly confirms and hands off an exact saved version', async () => {
    const fetch = install(), user = userEvent.setup(); render(tree('/pre-investment/ltcma/new?copy=cma-1'))
    await screen.findByLabelText('名称')
    expect(screen.getByLabelText('名称')).toHaveValue(ltcmaDefinition.name)
    expect(screen.getByRole('button', { name: '计算预览' })).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    const save = await screen.findByRole('button', { name: '确认保存版本' })
    expect(save).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已阅读结果和限制/ }))
    await user.click(save)
    expect(await screen.findByRole('button', { name: '用于 SAA' })).toBeEnabled()
    const published = fetch.mock.calls.find(([url, init]) => String(url) === `${root}/cma` && init?.method === 'POST')!
    const body = JSON.parse(String(published[1]?.body))
    expect(body).toMatchObject({ confirm: true, copied_from_id: 'cma-1', preview_hash: ltcmaVersion.preview_hash })
    expect(body.request).toMatchObject({ schema_version: '2.0', implementation_mapping_id: null })
    expect(body.request).not.toHaveProperty('mandate_id')
    expect(body.idempotency_key).toBeTruthy()
    await user.click(screen.getByRole('button', { name: '用于 SAA' }))
    expect(await screen.findByText('SAA destination')).toBeVisible()
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
    await user.selectOptions(screen.getByLabelText('生成方法'), 'bayesian_niw')
    expect(screen.getByLabelText('先验 LTCMA 版本')).toBeVisible()
    expect(screen.getByLabelText('均值先验等效日观察数')).toHaveValue('')
    expect(screen.queryByLabelText('市场风险厌恶系数 δ')).not.toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('生成方法'), 'historical_regime_occupancy')
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
