import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, expect, it, vi } from 'vitest'
import SaaCenter from './SaaCenter'
import { listSaaPolicies, type SavedSaaSummary } from '../services/strategicAllocation'

vi.mock('../services/strategicAllocation', () => ({ listSaaPolicies: vi.fn() }))
afterEach(() => { cleanup(); vi.resetAllMocks() })
const item: SavedSaaSummary = {
  id: 'saa-a', name: '稳健配置', as_of: '2019-12-31', created_at: '2026-09-23', mode: 'compatible_all_models',
  mandate: { id: 'goal-a', name: '三年目标', definition: { target_return: .0772, max_volatility: .095, min_cash_weight: .1 } },
  scope: { research_path: 'strategy_first', id: 'scope-a', name: '三大类范围' },
  cmas: [{ id: 'cma-a', name: '历史统计方案' }, { id: 'cma-b', name: '长期情景方案' }],
}
const open = () => render(<MemoryRouter><SaaCenter /></MemoryRouter>)

it('shows frozen references for every model and searchable objectives and scopes', async () => {
  vi.mocked(listSaaPolicies).mockResolvedValue({ items: [item] })
  open()
  const table = await screen.findByRole('table', { name: '已保存的 SAA 方案' })
  expect(within(table).getByRole('link', { name: '三年目标' })).toHaveAttribute('href', '/pre-investment/objectives/new?view=goal-a')
  expect(within(table).getByRole('link', { name: '三大类范围' })).toHaveAttribute('href', '/pre-investment/product-pool/new?scope=strategic&strategic_universe=scope-a')
  expect(within(table).getByRole('link', { name: '长期情景方案' })).toHaveAttribute('href', '/pre-investment/ltcma/cma-b')
  expect(within(table).getByRole('link', { name: '查看方案' })).toHaveAttribute('href', '/pre-investment/saa/policy?baseline=saa-a')
  expect(table).toHaveTextContent('共同约束')
  expect(table).toHaveTextContent('7.72%')
  expect(table).toHaveTextContent('9.50%')
  expect(table).toHaveTextContent('10.00%')
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '长期情景' } })
  expect(within(table).getByRole('link', { name: '稳健配置' })).toBeVisible()
  fireEvent.change(screen.getByRole('textbox'), { target: { value: '不匹配' } })
  expect(within(table).queryByRole('link')).not.toBeInTheDocument()
  expect(table).toHaveTextContent('没有找到匹配的方案')
})

it('uses exclusive loading, error, retry and empty states and fresh new-plan tokens', async () => {
  let reject!: (reason: Error) => void
  vi.mocked(listSaaPolicies).mockImplementationOnce(() => new Promise((_, fail) => { reject = fail }))
    .mockResolvedValue({ items: [] })
  const first = open()
  expect(screen.getByText('正在读取 SAA 方案…')).toBeVisible()
  expect(screen.queryByRole('table')).not.toBeInTheDocument()
  const href = screen.getByRole('link', { name: '新建 SAA 方案' }).getAttribute('href')
  expect(href).toMatch(/\/policy\?new=.+/)
  await act(async () => reject(new Error('读取失败，请重试。')))
  expect(screen.getByText('SAA 方案暂时无法读取，请重试。')).toBeVisible()
  expect(screen.queryByText('正在读取 SAA 方案…')).not.toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: '重试' }))
  expect(await screen.findByText('还没有保存的 SAA 方案')).toBeVisible()
  expect(screen.queryByText('SAA 方案暂时无法读取，请重试。')).not.toBeInTheDocument()
  first.unmount(); open()
  await screen.findByText('还没有保存的 SAA 方案')
  expect(screen.getByRole('link', { name: '新建 SAA 方案' }).getAttribute('href')).not.toBe(href)
})

it('does not invent names or zero targets for incomplete legacy records', async () => {
  vi.mocked(listSaaPolicies).mockResolvedValue({ items: [{ ...item, mode: 'single',
    scope: { research_path: 'product_first', id: 'products', name: '旧产品范围' },
    mandate: { id: null, name: null, definition: {} }, cmas: [{ id: 'old-cma', name: null }] }] })
  open()
  const table = await screen.findByRole('table')
  expect(table).toHaveTextContent('历史记录未保存名称')
  expect(table).not.toHaveTextContent('0.00%')
  expect(table).toHaveTextContent('单一 LTCMA')
  expect(within(table).getByRole('link', { name: '旧产品范围' })).toHaveAttribute('href', '/pre-investment/product-pool/new?universe=products')
})

it('shows pinned upstream versions and why a plan needs updating', async () => {
  const ref = (kind: 'mandate' | 'cma', id: string, name: string, status: 'current' | 'superseded', usable: 'ready' | 'stale' = 'ready') =>
    ({ kind, id, name, number: 1, latest_id: id, latest_number: status === 'superseded' ? 2 : 1, status, usable })
  vi.mocked(listSaaPolicies).mockResolvedValue({ items: [{ ...item, cmas: [{ id: 'cma-a', name: '历史统计方案' }],
    version: { lineage_id: 'saa-a', number: 1, status: 'current', latest_id: 'saa-a', latest_number: 1 },
    upstream: [ref('mandate', 'goal-a', '三年目标', 'superseded'), ref('cma', 'cma-a', '历史统计方案', 'current', 'stale')],
    usable: { status: 'stale', reasons: [{ code: 'upstream_superseded', kind: 'mandate', name: '三年目标', number: 1, latest_number: 2 }] } }] })
  open()
  const table = await screen.findByRole('table', { name: '已保存的 SAA 方案' })
  expect(within(table).getByRole('link', { name: '三年目标' })).toHaveAttribute('href', '/pre-investment/objectives/new?view=goal-a')
  expect(table).toHaveTextContent('已有新版本 v2')
  expect(within(table).getAllByText('需更新').length).toBeGreaterThanOrEqual(2) // 方案状态与 LTCMA 状态
  expect(table).toHaveTextContent('投资目标与约束「三年目标」已有新版本 v2；请基于新版本修改后再用于下一步。')
})
