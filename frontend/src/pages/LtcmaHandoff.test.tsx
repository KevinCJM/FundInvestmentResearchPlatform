import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import LtcmaCenter from './LtcmaCenter'
import StrategicAllocationWorkspace from './StrategicAllocationWorkspace'
import { ltcmaCapabilities, ltcmaItem, ltcmaVersion } from '../test/ltcmaFixtures'
import { policyFrontierFixture, policyPreview, strategicCatalog } from '../test/strategicAllocationFixtures'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney } from '../app/allocationJourney'
import { cmaHandoffIssue, prepareLtcmaHandoff } from '../services/ltcmaHandoff'
import type { CmaListItem } from '../services/ltcma'

const clock = vi.hoisted(() => ({ day: '2026-09-18' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => clock.day }))
vi.mock('echarts-for-react', () => ({ default: () => <div /> }))
const root = '/api/strategic-allocation'
const items = [ltcmaItem, { ...ltcmaItem, id: 'cma-2', name: '另一份 CMA', content_hash: 'd'.repeat(64) }]
const versions = items.map(item => ({ ...ltcmaVersion, id: item.id, name: item.name, content_hash: item.content_hash }))
const catalog = { ...strategicCatalog, allocations: [{ ...strategicCatalog.allocations[0], universe_snapshot_id: 'u-1' }], assumptions: items }
const response = (value: unknown) => Promise.resolve({ ok: true, status: 200, json: async () => value } as Response)
function install(overrides: Record<string, (url: URL, init?: RequestInit) => Promise<Response>> = {}) {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost'), path = url.pathname
    if (overrides[path]) return overrides[path](url, init)
    if (path === `${root}/cma`) return response({ items, total: 2, offset: 0, limit: 20 })
    if (path === `${root}/cma/capabilities`) return response(ltcmaCapabilities)
    if (path === `${root}/cma/drafts`) return response({ items: [] })
    if (path === `${root}/catalog`) return response(catalog)
    if (path === '/api/investable-universe-snapshots/u-1') return response({ id: 'u-1', name: '股债范围', mandate_id: 'mandate-1' })
    const version = versions.find(item => path === `${root}/cma/${item.id}` || path === `${root}/cma/${item.id}/view`)
    if (version) return response(path.endsWith('/view') ? { version, retired: false } : version)
    if (path === `${root}/policy/frontier`) return response(policyFrontierFixture(JSON.parse(String(init?.body))))
    if (path === `${root}/policy/preview`) return response({ ...policyPreview, request: JSON.parse(String(init?.body)) })
    throw new Error(`Unexpected request ${path}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function Location() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }
const tree = (path = '/pre-investment/ltcma') => <MemoryRouter initialEntries={[path]}><Location /><Routes>
  <Route path="/pre-investment/ltcma" element={<LtcmaCenter />} />
  <Route path="/pre-investment/saa/policy" element={<StrategicAllocationWorkspace />} />
</Routes></MemoryRouter>
const selected = () => screen.getByRole('region', { name: '用于 SAA 的 CMA' })
async function selectBoth(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  await user.click(screen.getByRole('checkbox', { name: `选择 ${items[1].name}` }))
}
async function comparePolicies(user: ReturnType<typeof userEvent.setup>) {
  const button = await screen.findByRole('button', { name: '比较符合目标的政策候选' })
  await waitFor(() => expect(button).toBeEnabled())
  // Require a successful frontier response, not the diagnostic-error fallback.
  expect(screen.getByTestId('saa-frontier-chart')).toBeInTheDocument()
  await user.click(button)
}
beforeEach(() => { clock.day = '2026-09-18'; localStorage.clear(); sessionStorage.clear() })
afterEach(() => { vi.unstubAllGlobals() })

it('carries a single exact CMA and its bound goal, ignoring the previous browser goal', async () => {
  const fetch = install(), user = userEvent.setup()
  updateAllocationJourney({ mandateId: 'unrelated-goal', baselineId: 'old-policy' })
  render(tree())
  expect(screen.getByRole('button', { name: '下一步：战略资产配置 →' })).toBeDisabled()
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await screen.findByRole('button', { name: '比较符合目标的政策候选' })
  expect(screen.getByTestId('location')).toHaveTextContent('mandate=mandate-1')
  expect(readAllocationJourney().mandateId).toBe('mandate-1')
  expect(readAllocationJourney().baselineId).toBeUndefined()
  await comparePolicies(user)
  const request = fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))!
  expect(JSON.parse(String(request[1]?.body))).toMatchObject({ cma_id: items[0].id, mandate_id: 'mandate-1' })
})

it.each(['feasible', 'infeasible'] as const)('waits for frontier evidence after handoff and respects %s targets', async status => {
  let resolve!: (value: Response) => void
  let frontier!: ReturnType<typeof policyFrontierFixture>
  const fetch = install({ [`${root}/policy/frontier`]: (_url, init) => {
    frontier = policyFrontierFixture(JSON.parse(String(init?.body)))
    return new Promise(done => { resolve = done })
  } })
  const user = userEvent.setup(); render(tree())
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  const compare = await screen.findByRole('button', { name: '比较符合目标的政策候选' })
  await waitFor(() => expect(resolve).toBeTypeOf('function'))
  expect(compare).toBeDisabled()
  await user.click(compare)
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/policy/preview'))).toBe(false)
  await act(async () => resolve(await response({ ...frontier,
    views: frontier.views.map(view => ({ ...view, target_return: status === 'infeasible' ? .2 : view.target_return })),
    target_check: { status, reason: status === 'infeasible' ? 'target_outside' : null },
  })))
  if (status === 'infeasible') {
    expect(compare).toBeDisabled()
    expect(screen.getByText(/无法同时满足年收益至少 20.00%/)).toBeVisible()
    await user.click(compare)
    expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/policy/preview'))).toBe(false)
  } else {
    await comparePolicies(user)
    const request = fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))!
    expect(JSON.parse(String(request[1]?.body))).toMatchObject({ cma_id: items[0].id, mandate_id: 'mandate-1' })
  }
})

it('keeps selected rows through filters and pagination and allows removing hidden choices', async () => {
  install({ [`${root}/cma`]: url => response({ items: url.searchParams.get('q') ? [] : Number(url.searchParams.get('offset')) ? [items[1]] : [items[0]], total: 21, offset: Number(url.searchParams.get('offset')), limit: 20 }) })
  const user = userEvent.setup(); render(tree())
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  await user.click(screen.getByRole('button', { name: '下一页' }))
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[1].name}` }))
  fireEvent.change(screen.getByLabelText('搜索名称'), { target: { value: 'no results' } })
  await waitFor(() => expect(screen.queryByRole('table')).not.toBeInTheDocument())
  expect(selected()).toHaveTextContent('已选 2 个 CMA')
  await user.click(within(selected()).getByRole('button', { name: `移除 ${items[0].name}` }))
  expect(selected()).toHaveTextContent('已选 1 个 CMA')
})

it('imports every selected CMA, requires an explicit mode and weights, and preserves them on refresh', async () => {
  const fetch = install(), user = userEvent.setup(); const mounted = render(tree()); await selectBoth(user)
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await waitFor(() => expect(screen.getByLabelText('CMA 使用方式')).toBeEnabled())
  const mode = screen.getByLabelText('CMA 使用方式')
  expect(mode).toHaveValue('')
  expect(screen.getByRole('button', { name: '3. 政策比较' })).toBeDisabled()
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/policy/preview'))).toBe(false)
  const path = screen.getByTestId('location').textContent!
  expect(new URLSearchParams(path.split('?')[1]).getAll('cma')).toEqual(items.map(item => item.id))
  expect(new URLSearchParams(allocationJourneyPath('saa').split('?')[1]).getAll('cma')).toEqual(items.map(item => item.id))
  await user.selectOptions(mode, 'parameter_average')
  expect(screen.getByRole('button', { name: '核对权重并进入政策比较' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '确认使用等权' }))
  mounted.unmount(); render(tree(path))
  await waitFor(() => expect(screen.getByRole('button', { name: '核对权重并进入政策比较' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: '核对权重并进入政策比较' }))
  await comparePolicies(user)
  const request = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))![1]?.body))
  expect(request).toMatchObject({ mode: 'parameter_average', cma_id: null, cma_refs: items.map(item => ({ cma_id: item.id, content_hash: item.content_hash, weight: .5 })) })
})

it('can enter common constraints without assigning model weights', async () => {
  const fetch = install(), user = userEvent.setup(); render(tree()); await selectBoth(user)
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await waitFor(() => expect(screen.getByLabelText('CMA 使用方式')).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  expect(screen.queryByRole('button', { name: '确认使用等权' })).not.toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: '核对模型并进入共同配置' }))
  await comparePolicies(user)
  expect(JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))![1]?.body))).toMatchObject({ mode: 'compatible_all_models', cma_refs: items.map(item => ({ cma_id: item.id, content_hash: item.content_hash, weight: null })) })
})

it('keeps an explicit switch to one CMA on refresh instead of reapplying the multi-CMA entry URL', async () => {
  const fetch = install(), user = userEvent.setup(); const mounted = render(tree()); await selectBoth(user)
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await waitFor(() => expect(screen.getByLabelText('CMA 使用方式')).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'single')
  await user.selectOptions(screen.getByLabelText('选择已确认 LTCMA'), items[1].id)
  await screen.findByRole('button', { name: '比较符合目标的政策候选' })
  const path = screen.getByTestId('location').textContent!
  mounted.unmount(); render(tree(path))
  await comparePolicies(user)
  expect(JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))![1]?.body))).toMatchObject({ cma_id: items[1].id })
})

it('retains incoming CMA choices while an unbound legacy scope needs its objective selected', async () => {
  install({ '/api/investable-universe-snapshots/u-1': () => response({ id: 'u-1', name: '未绑定范围' }) })
  const user = userEvent.setup(); render(tree()); await selectBoth(user)
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  expect(await screen.findByText(/该范围尚未绑定投资目标/)).toBeVisible()
  await user.selectOptions(await screen.findByLabelText('投资目标版本'), 'mandate-1')
  await waitFor(() => expect(screen.getByLabelText('CMA 使用方式')).toBeEnabled())
  expect(screen.getByText('已选 2 个 CMA')).toBeVisible()
})

it('explains unavailable rows and rechecks retirement before navigating', async () => {
  install({ [`${root}/cma`]: () => response({ items: [items[0], { ...items[1], as_of: '2026-09-10' }], total: 2 }),
    [`${root}/cma/${items[0].id}/view`]: () => response({ version: versions[0], retired: true }) })
  const user = userEvent.setup(); render(tree())
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  expect(screen.getByRole('checkbox', { name: `选择 ${items[1].name}` })).toBeDisabled()
  expect(screen.getByText('研究日不同，请选择同一天的 CMA。')).toBeVisible()
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('已停止引用')
  expect(screen.getByTestId('location')).toHaveTextContent('/pre-investment/ltcma')
})

it('ignores a late handoff response after clearing the selection', async () => {
  let resolve!: (value: Response) => void
  install({ [`${root}/cma/${items[0].id}/view`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); render(tree())
  await user.click(await screen.findByRole('checkbox', { name: `选择 ${items[0].name}` }))
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await user.click(screen.getByRole('button', { name: '清空选择' }))
  await act(async () => resolve(await response({ version: versions[0], retired: false })))
  expect(screen.getByTestId('location')).toHaveTextContent('/pre-investment/ltcma')
  expect(readAllocationJourney().ltcmaId).toBeUndefined()
})

it.each([
  [{ schema_version: '1.0' }, 'handoffLegacy'], [{ alloc_name: '其他范围' }, 'scopeAssets'],
  [{ currency: 'USD' }, 'handoffCurrency'], [{ asset_ids: ['bond', 'equity'] }, 'handoffAssets'],
  [{ retired: true }, 'handoffRetired'], [{ as_of: '2099-01-01' }, 'handoffFuture'],
] as Array<[Partial<CmaListItem>, string]>)('blocks incompatible or unavailable selections %j', (patch, expected) => {
  expect(cmaHandoffIssue([items[0], { ...items[1], ...patch }], clock.day!)).toBe(expected)
})

it('resolves a strategic scope binding without selecting an implementation mapping', async () => {
  const version = { ...versions[0], definition: { ...versions[0].definition, alloc_name: null, strategic_universe_id: 's-1' } }
  install({ [`${root}/cma/cma-1/view`]: () => response({ version, retired: false }),
    [`${root}/catalog`]: () => response({ ...catalog, strategic_universes: [{ id: 's-1', name: '三大类', mandate_id: 'mandate-1' }] }) })
  const result = await prepareLtcmaHandoff([{ ...items[0], alloc_name: null, strategic_universe_id: 's-1' }], clock.day!, key => key, new AbortController().signal)
  expect(result.path).toContain('strategic_universe=s-1')
  expect(result.path).toContain('mandate=mandate-1')
  expect(result.path).not.toContain('mapping=')
})

function replacedScopeFixture() {
  const scope = { id: 's-original', name: '股债研究范围', content_hash: 'a'.repeat(64), mandate_id: 'mandate-1',
    definition: { name: '股债研究范围', as_of: ltcmaItem.as_of, currency: 'CNY', source: '',
      assets: ltcmaVersion.definition.assets.map(asset => ({ ...asset, name: asset.id, currency: 'CNY', source: '' })) } }
  const selectedItems = items.map(item => ({ ...item, alloc_name: null, strategic_universe_id: scope.id }))
  const savedVersions = versions.map(version => ({ ...version,
    definition: { ...version.definition, alloc_name: null, strategic_universe_id: scope.id },
    source_snapshot: { ...version.source_snapshot, lineage: { ...version.source_snapshot.lineage, strategic_universe_hash: scope.content_hash } } }))
  const overrides = {
    [`${root}/cma`]: () => response({ items: selectedItems, total: selectedItems.length }),
    [`${root}/catalog`]: () => response({ ...catalog, assumptions: selectedItems,
      strategic_universes: [{ ...scope, id: 's-new', content_hash: 'b'.repeat(64), mandate_id: 'other-goal',
        definition: { ...scope.definition, assets: [] } }] }),
    [`${root}/universes/${scope.id}`]: () => response(scope),
    ...Object.fromEntries(savedVersions.flatMap(version => [
      [`${root}/cma/${version.id}`, () => response(version)],
      [`${root}/cma/${version.id}/view`, () => response({ version, retired: false })],
    ])),
  }
  return { scope, selectedItems, overrides }
}

it.each([1, 2])('continues with the original scope and bound goal after replacement, including refresh (%i CMAs)', async count => {
  const { scope, overrides } = replacedScopeFixture()
  const fetch = install(overrides), user = userEvent.setup(), mounted = render(tree())
  for (const item of items.slice(0, count)) await user.click(await screen.findByRole('checkbox', { name: `选择 ${item.name}` }))
  await user.click(screen.getByRole('button', { name: '下一步：战略资产配置 →' }))
  await waitFor(() => expect(screen.getByRole('button', { name: '1. 研究范围' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: '1. 研究范围' }))
  expect(screen.getByLabelText('研究范围')).toHaveValue(`strategic:${scope.id}`)
  expect(screen.getByLabelText('投资目标版本')).toHaveValue('mandate-1')
  const path = screen.getByTestId('location').textContent!
  expect(path).toContain(`strategic_universe=${scope.id}`)
  expect(path).not.toContain('s-new')
  if (count === 2) {
    await user.selectOptions(screen.getByLabelText('CMA 使用方式'), 'compatible_all_models')
  }
  mounted.unmount(); render(tree(path))
  if (count === 2) {
    await waitFor(() => expect(screen.getByRole('button', { name: '核对模型并进入共同配置' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '核对模型并进入共同配置' }))
  }
  await comparePolicies(user)
  const request = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/policy/preview'))![1]?.body))
  expect(Object.keys(request.constraints)).toEqual(scope.definition.assets.map(asset => asset.id))
  expect(request.mandate_id).toBe('mandate-1')
  expect(count === 1 ? [request.cma_id] : request.cma_refs.map((ref: { cma_id: string }) => ref.cma_id)).toEqual(items.slice(0, count).map(item => item.id))
})

it('does not replace an unreadable original scope with a same-named current scope and allows retry', async () => {
  const { scope, selectedItems, overrides } = replacedScopeFixture()
  let failing = true
  install({ ...overrides, [`${root}/universes/${scope.id}`]: () => failing
    ? Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: 'not found' }) } as Response) : response(scope) })
  const handoff = () => prepareLtcmaHandoff(selectedItems.slice(0, 1), clock.day!, key => key, new AbortController().signal)
  await expect(handoff()).rejects.toThrow('读取研究范围失败')
  failing = false
  await expect(handoff()).resolves.toMatchObject({ journey: { strategicUniverseId: scope.id, mandateId: 'mandate-1' } })
})

it('blocks an original scope whose fingerprint differs from the frozen CMA source', async () => {
  const { scope, selectedItems, overrides } = replacedScopeFixture()
  install({ ...overrides, [`${root}/universes/${scope.id}`]: () => response({ ...scope, content_hash: 'c'.repeat(64) }) })
  await expect(prepareLtcmaHandoff(selectedItems, clock.day!, key => key, new AbortController().signal)).rejects.toThrow('handoffChanged')
})

it('checks PIT availability, selection limits and the linked goal before handoff', async () => {
  expect(cmaHandoffIssue(items, undefined)).toBe('handoffClock')
  expect(cmaHandoffIssue(Array.from({ length: 21 }, (_, index) => ({ ...items[0], id: `cma-${index}` })), clock.day!)).toBe('handoffLimit')
  install({ [`${root}/catalog`]: () => response({ ...catalog, mandates: [{ ...catalog.mandates[0], definition: { ...catalog.mandates[0].definition, currency: 'USD' } }] }) })
  await expect(prepareLtcmaHandoff(items, clock.day!, key => key, new AbortController().signal)).rejects.toThrow('handoffGoalCurrency')
})

it('reloads pending incoming versions after a PIT change without staying locked', async () => {
  let resolve!: (value: Response) => void
  let reads = 0
  install({ [`${root}/cma/cma-1/view`]: () => ++reads === 1 ? new Promise(done => { resolve = done }) : response({ version: versions[0], retired: false }) })
  const path = '/pre-investment/saa/policy?alloc=股债分类&mandate=mandate-1&cma=cma-1&cma=cma-2'
  const mounted = render(tree(path))
  await waitFor(() => expect(reads).toBe(1))
  clock.day = '2026-09-17'; mounted.rerender(tree(path))
  await waitFor(() => expect(screen.getByLabelText('CMA 使用方式')).toBeEnabled())
  expect(reads).toBe(2)
  await act(async () => resolve(await response({ version: versions[0], retired: true })))
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.getByLabelText('CMA 使用方式')).toHaveValue('')
})
