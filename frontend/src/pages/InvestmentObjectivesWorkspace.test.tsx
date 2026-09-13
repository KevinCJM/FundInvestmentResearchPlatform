import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import InvestmentObjectivesWorkspace from './InvestmentObjectivesWorkspace'
import { writeAllocationDraft } from '../app/allocationJourney'
import { assessment, fundingStudy } from '../test/mandateFixtures'
import { mandateVersion, strategicCatalog } from '../test/strategicAllocationFixtures'
import { mandateIssues, modelMonthLabel } from '../components/investment-mandate/model'
import type { MandateStudyRequest } from '../services/strategicAllocation'

const researchClock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => researchClock.day }))
vi.mock('echarts-for-react', () => ({ default: ({ option }: { option: { series: unknown[] } }) => <div data-testid="funding-chart" data-series={option.series.length} /> }))
const root = '/api/strategic-allocation'
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const fetch = vi.fn((url: RequestInfo | URL, init?: RequestInit) => {
    const key = String(url)
    if (overrides[key]) return overrides[key](init)
    if (key === `${root}/catalog`) return response(strategicCatalog)
    if (key === `${root}/mandates/preview`) return response(assessment(JSON.parse(String(init?.body))))
    if (key === `${root}/mandates/confirm`) {
      const request = JSON.parse(String(init?.body)).request as MandateStudyRequest
      return response({ ...mandateVersion, name: request.definition.name, definition: request.definition, assessment: assessment(request) }, 201)
    }
    if (key === `${root}/mandates/mandate-1`) return response(mandateVersion)
    throw new Error(`Unexpected API ${key}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function ready(request: MandateStudyRequest = fundingStudy) {
  writeAllocationDraft('mandate-study:editor', request)
  return render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
}
async function run(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await screen.findByLabelText('目标可行性诊断结果')
}
beforeEach(() => { researchClock.day = '2026-09-12'; localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

it('资金目标不要求用户先编造收益率，并可无CMA完成资金测算与输入确认', async () => {
  const fetch = install(); const user = userEvent.setup(); ready()
  expect(screen.queryByLabelText(/最低预期年收益/)).not.toBeInTheDocument()
  expect(screen.getByLabelText('总资金（CNY）')).toHaveValue('1000000')
  await run(user)
  expect(screen.getByText(/只完成输入与资金测算/)).toBeInTheDocument()
  expect(screen.getByText('900,000')).toBeInTheDocument()
  expect(screen.queryByTestId('funding-chart')).not.toBeInTheDocument()
  expect(fetch.mock.calls.some(([url]) => url === `${root}/mandates/confirm`)).toBe(false)
  await user.click(screen.getByRole('button', { name: '下一步：核对与确认' }))
  expect(screen.getByRole('button', { name: '保存新目标版本' })).toBeDisabled()
  await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
  await user.click(screen.getByRole('button', { name: '保存新目标版本' }))
  expect(await screen.findByRole('link', { name: /使用此目标进入长期配置/ })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=mandate-1')
  const call = fetch.mock.calls.find(([url]) => url === `${root}/mandates/confirm`)!
  expect(JSON.parse(String(call[1]?.body))).toMatchObject({ preview_hash: 'd'.repeat(64), acknowledge_limits: true,
    request: { definition: { objective_kind: 'funding_goal', target_return: 0, funding_plan: { outside_reserve: 100000 } } } })
})

it('概率点估计超过80%但区间下界不足时，不将目标显示为达标', async () => {
  install(); const user = userEvent.setup(); ready({ ...fundingStudy, cma_id: 'cma-1' })
  await run(user)
  expect(screen.getByText(/区间下界未达到/)).toBeInTheDocument()
  expect(screen.getByRole('table', { name: '目标诊断候选对照' })).toHaveTextContent('未达标')
  expect(screen.getByTestId('funding-chart')).toHaveAttribute('data-series', '3')
  await user.click(screen.getByText('目标未达成时，可以比较哪些调整？'))
  expect(screen.getByText(/可投资本金增加10%/)).toBeInTheDocument()
  expect(screen.getByText(/保守敏感性/)).toHaveTextContent('65.00%')
  expect(screen.getByText(/不会改写输入/)).toBeInTheDocument()
  expect(screen.getByLabelText('按采纳口径测算本金')).toHaveTextContent('975,000')
  expect(screen.getByLabelText('按采纳口径测算本金')).toHaveTextContent('80.80%')
  expect(screen.getByLabelText('近期支付缓冲')).toHaveTextContent('不是可承受回撤上限')
  await user.click(screen.getByText('逐年资金余额分位数'))
  const table = screen.getByRole('table', { name: '逐年资金余额分位数' })
  expect(within(table).getByRole('row', { name: /第10年/ })).toHaveTextContent('1,700,000')
})

it('研究日改变后不沿用旧CMA与旧诊断；迟到结果不能覆盖新输入', async () => {
  let resolve!: (value: Response) => void
  install({ [`${root}/mandates/preview`]: () => new Promise(done => { resolve = done }) })
  const user = userEvent.setup(); ready()
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await user.click(screen.getByRole('button', { name: '1. 资金与成功标准' }))
  fireEvent.change(screen.getByLabelText('总资金（CNY）'), { target: { value: '2000000' } })
  await act(async () => resolve(await response(assessment())))
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  expect(screen.queryByLabelText('目标可行性诊断结果')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '下一步：核对与确认' })).toBeDisabled()
})

it('历史输入版本只读，必须显式复制后才能修改', async () => {
  install(); const user = userEvent.setup(); ready()
  await user.click(screen.getByText(/已保存的目标/))
  await user.click(await screen.findByRole('button', { name: /长期配置目标/ }))
  expect(await screen.findByRole('button', { name: '复制为新研究' })).toBeInTheDocument()
  expect(screen.getByLabelText('目标名称')).toBeDisabled()
  expect(screen.getByRole('link', { name: /使用此目标进入长期配置/ })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=mandate-1')
  await user.click(screen.getByRole('button', { name: '复制为新研究' }))
  expect(screen.getByLabelText('目标名称')).toBeEnabled()
  expect(screen.queryByRole('link', { name: /使用此目标进入长期配置/ })).not.toBeInTheDocument()
})

it('现金流按可读日期选择期次，缩短期限不会静默截断原现金流', async () => {
  install(); const user = userEvent.setup(); ready()
  await user.click(screen.getByRole('button', { name: '添加投入或支付' }))
  fireEvent.change(screen.getByLabelText('现金流 1名称'), { target: { value: '必要支付' } })
  fireEvent.change(screen.getByLabelText('现金流 1金额（CNY）'), { target: { value: '10000' } })
  await user.selectOptions(screen.getByLabelText('现金流 1开始月'), '60')
  await user.selectOptions(screen.getByLabelText('现金流 1结束月'), '60')
  expect(screen.getByLabelText('现金流 1开始月')).toHaveValue('60')
  fireEvent.change(screen.getByLabelText('投资期限（年）'), { target: { value: '2' } })
  expect(screen.getByRole('button', { name: '下一步：风险与限制' })).toBeDisabled()
  expect(screen.getByText(/现金流须有名称/)).toBeInTheDocument()
  expect(modelMonthLabel('2020-01-31', 1)).toBe('第1月（2020-02-29）')
})

it('相对目标需填写真实基准轴及权重，超额与TAA风险预算分开', async () => {
  install(); const user = userEvent.setup(); ready()
  await user.selectOptions(screen.getByLabelText('这笔资金以什么为成功标准？'), 'benchmark_relative')
  await user.type(screen.getByLabelText('基准名称'), '股债六四基准')
  await user.selectOptions(screen.getByLabelText(/基准大类方案/), '股债分类')
  expect(screen.getByRole('button', { name: '下一步：风险与限制' })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('equity基准权重（%）'), { target: { value: '60' } })
  fireEvent.change(screen.getByLabelText('bond基准权重（%）'), { target: { value: '40' } })
  fireEvent.change(screen.getByLabelText('目标年超额收益（百分点）'), { target: { value: '1' } })
  fireEvent.change(screen.getByLabelText(/相对基准主动风险上限/), { target: { value: '3' } })
  expect(screen.getByRole('button', { name: '下一步：风险与限制' })).toBeEnabled()
  expect(screen.queryByLabelText('总资金（CNY）')).not.toBeInTheDocument()
})

it('API失败保留草稿并就近提示，不把服务错误画成零概率', async () => {
  install({ [`${root}/mandates/preview`]: () => response({ detail: { message: 'CMA与资金目标的研究日不一致。' } }, 422) })
  const user = userEvent.setup(); ready(); await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('研究日不一致')
  expect(screen.queryByTestId('funding-chart')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '运行目标诊断' })).toBeEnabled()
})

it('恢复的CMA不匹配当前研究范围时阻止诊断，不把它伪装成未选CMA', async () => {
  const fetch = install(); const user = userEvent.setup(); ready({ ...fundingStudy, cma_id: 'missing-old-cma' })
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  expect(await screen.findByText(/已选CMA不在当前日期/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '运行目标诊断' })).toBeDisabled()
  expect(fetch.mock.calls.some(([url]) => url === `${root}/mandates/preview`)).toBe(false)
})

it('缺少目标诊断的响应不能显示达标或进入保存步骤', async () => {
  const incomplete = assessment({ ...fundingStudy, cma_id: 'cma-1' })
  delete incomplete.candidates[0].goal_check
  install({ [`${root}/mandates/preview`]: () => response(incomplete) })
  const user = userEvent.setup(); ready({ ...fundingStudy, cma_id: 'cma-1' })
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('诊断缺失或口径不一致')
  expect(screen.queryByText('本次达标')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '下一步：核对与确认' })).toBeDisabled()
})

it('知识截止日改变只作废未保存预览，不清除只读版本的冻结诊断', async () => {
  const saved = { ...mandateVersion, definition: fundingStudy.definition, assessment: assessment() }
  const fetch = install({ [`${root}/mandates/mandate-1`]: () => response(saved) })
  const user = userEvent.setup(); const view = ready()
  await user.click(screen.getByText(/已保存的目标/))
  await user.click(await screen.findByRole('button', { name: /长期配置目标/ }))
  expect(await screen.findByText('确认的是目标与边界，不是收益承诺')).toBeInTheDocument()
  const before = fetch.mock.calls.length
  researchClock.day = '2026-09-10'
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  expect(screen.getByText('确认的是目标与边界，不是收益承诺')).toBeInTheDocument()
  expect(screen.getByText(/当前只读版本仍显示保存时的诊断/)).toBeInTheDocument()
  expect(fetch.mock.calls.length).toBe(before)
  expect(screen.getByRole('button', { name: '已锁定此目标版本' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: '复制为新研究' }))
  expect(screen.getByRole('button', { name: '下一步：风险与限制' })).toBeDisabled()
  expect(screen.getByText(/研究日不能超过知识截止日/)).toBeInTheDocument()
})

it('未保存诊断在知识截止日变化后失效，不能继续确认', async () => {
  install(); const user = userEvent.setup(); const view = ready()
  await run(user)
  researchClock.day = '2026-09-10'
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  expect(screen.queryByLabelText('目标可行性诊断结果')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '下一步：核对与确认' })).toBeDisabled()
})

it('空值、联合边界和资金预留冲突保持明确的输入错误', () => {
  expect(mandateIssues({ ...fundingStudy.definition, max_volatility: NaN }, '2026-09-12')[1]).toContain('最高预期年波动')
  expect(mandateIssues({ ...fundingStudy.definition, funding_plan: { ...fundingStudy.definition.funding_plan!, outside_reserve: 1_000_000 } }, '2026-09-12')[0]).toContain('组合外储备')
  expect(mandateIssues({ ...fundingStudy.definition, group_limits: [{ id: 'x', assets: [], lo: 0, hi: 1 }] }, '2026-09-12')[1]).toContain('联合约束')
  expect(mandateIssues({ ...fundingStudy.definition, allocation_scope: null, asset_limits: { equity: { min_weight: 0, max_weight: .3, max_abs_tilt: .1 } } }, '2026-09-12')[1]).toContain('绑定所属大类')
})


it('未知研究时钟阻止诊断与保存，明确无PIT仍可诊断，恢复为未知使旧结果失效', async () => {
  researchClock.day = undefined
  const fetch = install(); const user = userEvent.setup(); const view = ready()
  expect(screen.getByRole('alert')).toHaveTextContent('平台知识截止日尚未确认')
  await user.click(screen.getByRole('button', { name: '3. 量化诊断' }))
  expect(screen.getByRole('button', { name: '运行目标诊断' })).toBeDisabled()
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/mandates/preview'))).toBe(false)
  researchClock.day = null
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await screen.findByLabelText('目标可行性诊断结果')
  await user.click(screen.getByRole('button', { name: '下一步：核对与确认' }))
  await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
  expect(screen.getByRole('button', { name: '保存新目标版本' })).toBeEnabled()
  researchClock.day = undefined
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  expect(screen.queryByRole('button', { name: '保存新目标版本' })).not.toBeInTheDocument()
  expect(fetch.mock.calls.some(([url]) => String(url).endsWith('/mandates/confirm'))).toBe(false)
})
