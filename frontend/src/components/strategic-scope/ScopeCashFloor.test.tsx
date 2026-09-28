import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import StrategicScopeWorkspace from './StrategicScopeWorkspace'
import { strategicCatalog } from '../../test/strategicAllocationFixtures'
import { cashFloorIssue, weightLimitsIssue, withCashFloor, type UniverseDefinition } from '../../services/strategicScope'

vi.mock('../../app/ResearchContext', () => ({ useResearchDay: () => '2026-09-12', useResearchContextIdentity: () => 'fixed' }))
const base: UniverseDefinition = { name: '现金下限范围', as_of: '2026-09-12', currency: 'CNY', source: '', assets: [
  { id: 'equity', name: '权益', currency: 'CNY', role: 'growth', liquidity: 'liquid', rationale: '', source: '' },
] }
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)
beforeEach(() => { localStorage.clear(); sessionStorage.clear() })
afterEach(() => { vi.unstubAllGlobals() })

it('按目标补齐现金大类；已显式设过边界时不改回，调低或缺失只提醒', () => {
  const seeded = withCashFloor(base, .1)
  expect(seeded.assets[0]).toMatchObject({ name: '现金', role: 'liquidity', weight_limits: { min_weight: .1, max_weight: 1 }, research_proxy: { asset_type: 'cash' } })
  expect(cashFloorIssue(seeded.assets, .1)).toBe('')
  const lowered = { ...seeded, assets: seeded.assets.map((a, i) => i === 0 ? { ...a, weight_limits: { min_weight: .05, max_weight: 1 } } : a) }
  expect(withCashFloor(lowered, .1)).toBe(lowered)
  expect(cashFloorIssue(lowered.assets, .1)).toContain('5.00%')
  expect(cashFloorIssue(lowered.assets, .1)).toContain('10.00%')
  expect(cashFloorIssue(base.assets, .1)).toContain('未配置现金大类')
  expect(cashFloorIssue(base.assets, 0)).toBe('')
  // 旧范围未设边界：沿用目标，不提示。
  expect(cashFloorIssue([{ ...seeded.assets[0], weight_limits: undefined }], .1)).toBe('')
})

it('权重边界与后端契约一致：下限高于上限或下限合计超 100% 时阻止保存', () => {
  const at = (min: number, max: number) => ({ ...base, assets: [{ ...base.assets[0], weight_limits: { min_weight: min, max_weight: max } }] })
  expect(weightLimitsIssue(at(.2, .8))).toBe('')
  expect(weightLimitsIssue(at(.6, .5))).not.toBe('')
  expect(weightLimitsIssue(withCashFloor(at(.95, 1), .1))).not.toBe('')
})

it('新建范围自动带出现金 10% 下限；调低后行内红色提醒、仍可保存', async () => {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url.endsWith('/catalog')) return response({ ...strategicCatalog, strategic_universes: [], implementation_maps: [] })
    if (url.endsWith('/universes/preview')) return response({ definition: JSON.parse(String(init?.body)), preview_hash: 'b'.repeat(64) })
    if (url.endsWith('/universes/confirm')) return response({ id: 'scope-new', name: '现金下限范围', content_hash: 'a'.repeat(64), created_at: '2026-09-12', preview_hash: 'b'.repeat(64), research_only: true, implementation_status: 'unmapped', implementation_gaps: [], definition: JSON.parse(String(init?.body)).request }, 201)
    throw new Error(`Unexpected API: ${url}`)
  })
  vi.stubGlobal('fetch', fetch)
  render(<MemoryRouter initialEntries={['/pre-investment/product-pool/new?scope=strategic&mandate=mandate-one&new=t1']}><StrategicScopeWorkspace freshKey="t1" cashFloor={.1} /></MemoryRouter>)
  const min = await screen.findByLabelText('现金 权重下限（%）')
  expect(min).toHaveValue('10')
  expect(screen.queryByRole('button', { name: '现金下限偏离投资目标' })).not.toBeInTheDocument()
  fireEvent.change(min, { target: { value: '5' } })
  fireEvent.blur(min)
  const mark = await screen.findByRole('button', { name: '现金下限偏离投资目标' })
  fireEvent.mouseEnter(mark)
  expect(await screen.findByRole('tooltip')).toHaveTextContent('现金下限 5.00% 低于投资目标要求的 10.00%')
  const save = within(screen.getByLabelText('保存战略范围')).getByRole('button', { name: '保存战略范围' })
  expect(save).toBeEnabled()
  await userEvent.click(save)
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/universes/confirm'))).toBe(true))
  const confirmed = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url).endsWith('/universes/confirm'))![1]!.body))
  expect(confirmed.request.assets.find((a: { role: string }) => a.role === 'liquidity').weight_limits).toEqual({ min_weight: .05, max_weight: 1 })
})
