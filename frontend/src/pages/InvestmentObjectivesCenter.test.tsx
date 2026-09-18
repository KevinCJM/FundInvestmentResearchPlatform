import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import InvestmentObjectivesCenter from './InvestmentObjectivesCenter'
import { boundaryAssessment, boundaryStudy } from '../test/mandateBoundaryFixtures'
import { strategicCatalog } from '../test/strategicAllocationFixtures'

const root = '/api/strategic-allocation'
const assessment = boundaryAssessment(boundaryStudy())
const version = { id: 'mandate-active', name: '家庭长期目标', content_hash: 'f'.repeat(64), created_at: '2026-09-17T08:00:00Z',
  definition: assessment.definition, assessment_status: assessment.status }
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)

beforeEach(() => {
  const fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (url === `${root}/catalog`) return response({ ...strategicCatalog, mandates: [version] })
    if (url === `${root}/mandates/${version.id}` && init?.method === 'DELETE') return response({ deleted: true, id: version.id })
    throw new Error(`Unexpected request ${url} ${init?.method ?? 'GET'}`)
  })
  vi.stubGlobal('fetch', fetch)
})
afterEach(() => { vi.unstubAllGlobals() })

it('主页面先展示现有目标列表，并提供新增、修改和删除', async () => {
  const user = userEvent.setup()
  render(<MemoryRouter><InvestmentObjectivesCenter /></MemoryRouter>)
  const table = await screen.findByRole('table', { name: '已发布投资目标与约束列表' })
  expect(within(table).getByText('家庭长期目标')).toBeInTheDocument()
  expect(within(table).getByRole('link', { name: '家庭长期目标' })).toHaveAttribute('href', `/pre-investment/objectives/new?view=${version.id}`)
  expect(within(table).getByText('C3')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '添加投资目标与约束' })).toHaveAttribute('href', '/pre-investment/objectives/new?fresh=1')
  expect(within(table).getByRole('link', { name: '修改' })).toHaveAttribute('href', `/pre-investment/objectives/new?editFrom=${version.id}`)

  await user.click(within(table).getByRole('button', { name: '删除' }))
  expect(screen.getByText(/确认删除“家庭长期目标”/)).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: '确认删除' }))
  expect(await screen.findByText('投资目标与约束已从可用列表删除。')).toBeInTheDocument()
  expect(screen.queryByText('家庭长期目标')).not.toBeInTheDocument()
})
