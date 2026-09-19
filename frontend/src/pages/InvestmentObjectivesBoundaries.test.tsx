import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import InvestmentObjectivesWorkspace from './InvestmentObjectivesWorkspace'
import { writeAllocationDraft } from '../app/allocationJourney'
import { boundaryAssessment, boundaryStudy } from '../test/mandateBoundaryFixtures'
import { riskVersion } from '../test/riskScaleFixtures'
import { strategicCatalog } from '../test/strategicAllocationFixtures'
import type { MandateAssessment } from '../services/strategicAllocation'
import { checkReferenceAssessment } from '../services/mandateTypes'
import { newBoundaryMandate, mandateStepIssues } from '../components/investment-mandate/model'

vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => '2026-09-17' }))
vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="echarts" /> }))
const root = '/api/strategic-allocation'
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)
const studyOption = { id: riskVersion.id, name: riskVersion.name, content_hash: riskVersion.content_hash,
  version_number: 1, base_currency: 'CNY', risk_basis_id: 'annualized-periodic-volatility-v1', research_as_of: '2026-09-17', valid_until: null }

function install(transform?: (value: MandateAssessment) => MandateAssessment) {
  const fetch = vi.fn((url: RequestInfo | URL, init?: RequestInit) => {
    const key = String(url)
    if (key === `${root}/catalog`) return response({ ...strategicCatalog, mandates: [] })
    if (key.startsWith(`${root}/risk-scales/study-options?`)) return response({ as_of: '2026-09-17', items: [studyOption] })
    if (key === `${root}/risk-scales/${riskVersion.id}`) return response(riskVersion)
    if (key === `${root}/mandates/preview`) {
      const value = boundaryAssessment(JSON.parse(String(init?.body)))
      return response(transform ? transform(value) : value)
    }
    throw new Error(`Unexpected request: ${key}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function ready() {
  writeAllocationDraft('mandate-study:editor', boundaryStudy())
  return render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
}
async function diagnose(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
}

beforeEach(() => { localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

it('新建目标没有隐藏填入风险上限、治理政策或正式CMA', () => {
  const value = newBoundaryMandate('2026-09-17')
  expect(value.max_volatility).toBeNull()
  expect(value.boundary_policy).toBeNull()
  expect(value.review_date).toBeNull()
  expect(value.risk_authorization?.authorized_max_level).toBeNull()
  expect(value.risk_authorization?.risk_scale_ref).toBeNull()
  expect(value.asset_limits).toEqual({})
  expect(value.group_limits).toEqual([])
})

it('紧凑输入校验只要求目标、风险等级和现金约束', () => {
  const study = boundaryStudy()
  expect(mandateStepIssues(study.definition, '2026-09-17')).toEqual(['', '', ''])
  const noScale = { ...study.definition, risk_authorization: { ...study.definition.risk_authorization!, risk_scale_ref: null } }
  expect(mandateStepIssues(noScale, '2026-09-17')[2]).toContain('风险等级配置')
  expect(mandateStepIssues({ ...study.definition, min_cash_weight: 1.1 }, '2026-09-17')[2]).toContain('现金占比')
  expect(mandateStepIssues({ ...study.definition, cash_budget: { ...study.definition.cash_budget!, outside_reserve: 1_000_000 } }, '2026-09-17')[2]).toContain('组合外储备')
  expect(mandateStepIssues({ ...study.definition, cash_budget: { ...study.definition.cash_budget!, annual_fee: .11 } }, '2026-09-17')[2]).toContain('额外年费用')
  expect(mandateStepIssues({ ...study.definition, review_date: '2026-09-16' }, '2026-09-17')[0]).toContain('复核日期')
})

it.each([
  ['缺少独立验证', (value: MandateAssessment) => { value.reference_diagnosis!.validation = null }],
  ['独立验证概率矛盾', (value: MandateAssessment) => { value.reference_diagnosis!.validation!.central.probability_lower = .7 }],
  ['搜索与验证同源', (value: MandateAssessment) => { value.reference_diagnosis!.validation_seed = value.reference_diagnosis!.search_seed }],
  ['冻结风险上限被篡改', (value: MandateAssessment) => { value.definition.max_volatility = .2 }],
])('%s时前端契约拒绝把响应当成可信结果', (_, mutate) => {
  const value = boundaryAssessment(); mutate(value)
  expect(() => checkReferenceAssessment(value, () => undefined)).toThrow(/不完整|不一致/)
})

it('没有代表组合的风险档位不能用于相对基准目标', async () => {
  const broken = structuredClone(riskVersion)
  broken.preview.result.levels[2].representative_node_id = null
  broken.preview.result.levels[2].representative_weights = null
  const fetch = vi.fn((url: RequestInfo | URL) => {
    const key = String(url)
    if (key === `${root}/catalog`) return response({ ...strategicCatalog, mandates: [] })
    if (key.startsWith(`${root}/risk-scales/study-options?`)) return response({ as_of: '2026-09-17', items: [studyOption] })
    if (key === `${root}/risk-scales/${riskVersion.id}`) return response(broken)
    throw new Error(`Unexpected request: ${key}`)
  })
  vi.stubGlobal('fetch', fetch)
  const relative = boundaryStudy(); relative.definition.objective_kind = 'benchmark_relative'; relative.definition.funding_target = null
  relative.definition.target_excess_return = .01; relative.definition.cash_budget = null
  writeAllocationDraft('mandate-study:editor', relative)
  const user = userEvent.setup(); render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  // 等级改为卡片选择：缺代表组合的档位在相对基准目标下不可点击。
  expect(await screen.findByRole('button', { name: /^C3/ })).toBeDisabled()
})
