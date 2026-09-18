import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import InvestmentObjectivesWorkspace from './InvestmentObjectivesWorkspace'
import { writeAllocationDraft, readAllocationDraft, updateAllocationJourney } from '../app/allocationJourney'
import { boundaryAssessment, boundaryStudy } from '../test/mandateBoundaryFixtures'
import { fundingStudy } from '../test/mandateFixtures'
import { riskReference, riskVersion } from '../test/riskScaleFixtures'
import { strategicCatalog } from '../test/strategicAllocationFixtures'
import type { MandateStudyRequest } from '../services/strategicAllocation'
import { compactMandateDefinition } from '../components/investment-mandate/model'

const researchClock = vi.hoisted(() => ({ day: '2026-09-17' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => researchClock.day }))
vi.mock('echarts-for-react', () => ({ default: ({ option }: { option: any }) => <div data-testid="echarts"
  data-series={String(option?.series?.length ?? 0)} data-bands={String(option?.series?.[0]?.markLine?.data?.length ?? 0)} /> }))

const root = '/api/strategic-allocation'
const response = (body: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => body } as Response)
const option = { id: riskVersion.id, name: riskVersion.name, content_hash: riskVersion.content_hash,
  version_number: riskVersion.version_number, base_currency: 'CNY', risk_basis_id: 'annualized-periodic-volatility-v1',
  research_as_of: '2026-09-17', valid_until: null }
const savedAssessment = boundaryAssessment(boundaryStudy())
const savedVersion = { id: 'boundary-saved', name: savedAssessment.definition.name, content_hash: 'a'.repeat(64),
  created_at: '2026-09-17T00:00:00Z', definition: savedAssessment.definition, assessment: savedAssessment }

type Override = (init?: RequestInit, url?: string) => Promise<Response>
function install(overrides: Record<string, Override> = {}) {
  const fetch = vi.fn((url: RequestInfo | URL, init?: RequestInit) => {
    const key = String(url)
    for (const [prefix, handler] of Object.entries(overrides)) if (key.startsWith(prefix)) return handler(init, key)
    if (key === `${root}/catalog`) return response({ ...strategicCatalog, mandates: [savedVersion] })
    if (key.startsWith(`${root}/risk-scales/study-options?`)) {
      const day = new URL(key, 'http://local').searchParams.get('as_of')!
      return response({ as_of: day, items: day >= option.research_as_of ? [option] : [] })
    }
    if (key === `${root}/risk-scales/${riskVersion.id}`) return response(riskVersion)
    if (key === `${root}/reference-inputs/${riskReference.id}`) return response(riskReference)
    if (key === `${root}/mandates/preview`) return response(boundaryAssessment(JSON.parse(String(init?.body))))
    if (key === `${root}/mandates/confirm`) {
      const request = JSON.parse(String(init?.body)).request as MandateStudyRequest
      const assessment = boundaryAssessment(request)
      return response({ ...savedVersion, name: request.definition.name, definition: assessment.definition, assessment }, 201)
    }
    if (key === `${root}/mandates/funding`) {
      const request = JSON.parse(String(init?.body)) as MandateStudyRequest
      const echo = boundaryAssessment(request)
      return response({ funding: { ...echo.funding, cashflow_required_return: .042, cashflow_required_return_status: 'solved' },
        effective_target_return: request.definition.objective_kind === 'absolute_return'
          ? Math.max(request.definition.target_return, .042) : null, execution: echo.execution })
    }
    if (key === `${root}/mandates/${savedVersion.id}`) return response(savedVersion)
    throw new Error(`Unexpected API ${key}`)
  })
  vi.stubGlobal('fetch', fetch)
  return fetch
}
function ready(request: MandateStudyRequest = boundaryStudy(), entry = '/pre-investment/objectives/new') {
  writeAllocationDraft('mandate-study:editor', request)
  return render(<MemoryRouter initialEntries={[entry]}><InvestmentObjectivesWorkspace /></MemoryRouter>)
}
async function openResult(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await screen.findByText('当前约束下可实现')
}

beforeEach(() => { researchClock.day = '2026-09-17'; localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

it('工作台收敛为两步，PIT开启时锁定研究日，并隐藏正式CMA和模型技术参数', async () => {
  install(); ready();
  const navigation = screen.getByRole('navigation', { name: '投资目标步骤' })
  expect(within(navigation).getAllByRole('button')).toHaveLength(2)
  const researchDate = screen.getByLabelText(/目标研究日/)
  expect(researchDate).toBeDisabled(); expect(researchDate).toHaveValue('2026-09-17')
  expect(screen.getAllByText('PIT 已开启 · 2026-09-17')).toHaveLength(1)
  expect(screen.getByLabelText(/^计价币种/)).toHaveAttribute('readonly')
  expect(screen.queryByLabelText(/CMA/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/模拟路径/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/风险厌恶/)).not.toBeInTheDocument()
  expect(screen.getByLabelText(/政策复核日期/)).not.toBeRequired()
})

it('旧资金计划迁移到精简契约时保留本金、外部储备和现金假设，不丢业务事实', () => {
  const migrated = compactMandateDefinition(fundingStudy.definition, '2026-09-17')
  expect(migrated.cash_budget).toMatchObject({ total_capital: 1_000_000, outside_reserve: 100_000,
    amount_basis: 'nominal', inflation: .02, annual_fee: .005, flows: [] })
  expect(migrated.funding_target).toEqual({ amount: 1_500_000, amount_basis: 'nominal' })
  expect(migrated.funding_plan).toBeNull()
})

it('风险页只选择研究日可用Risk Scale、一个最大C等级和最低现金占比，币种自动继承', async () => {
  const fetch = install(); const user = userEvent.setup(); ready()
  const scale = await screen.findByLabelText(/^风险标尺版本/)
  expect(scale).toHaveValue(riskVersion.id)
  expect(screen.getByLabelText(/^计价币种/)).toHaveValue('CNY')
  // 等级改为卡片选择：卡片本身展示波动上限、参考收益和历史回撤。
  expect(screen.getByRole('button', { name: /^C3/ })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByRole('button', { name: /^C1/ })).toHaveAttribute('aria-pressed', 'false')
  expect(screen.getByLabelText(/最低现金占比/)).toHaveValue('10')
  expect((await screen.findAllByText('6.00%')).length).toBeGreaterThan(0)
  expect(screen.queryByLabelText(/本次采用的等级上限/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/政策来源/)).not.toBeInTheDocument()
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/risk-scales/study-options?as_of=2026-09-17'))).toBe(true)
})

it('PIT关闭时研究日可编辑；修改日期会清除旧Risk Scale并重新按日期取可用版本', async () => {
  researchClock.day = null
  const fetch = install(); const user = userEvent.setup(); ready()
  const dateInput = screen.getByLabelText(/目标研究日/)
  expect(dateInput).toBeEnabled()
  fireEvent.change(dateInput, { target: { value: '2026-09-16' } })
  await waitFor(() => expect(fetch.mock.calls.some(([url]) => String(url).includes('study-options?as_of=2026-09-16'))).toBe(true))
  expect(screen.getByLabelText(/^风险标尺版本/)).toHaveValue('')
  expect(await screen.findByText('这个研究日没有可用的风险等级配置')).toBeInTheDocument()
})

it('未知PIT阻止诊断；恢复为关闭后可继续编辑', async () => {
  researchClock.day = undefined
  install(); const user = userEvent.setup(); const view = ready()
  expect(screen.getByRole('alert')).toHaveTextContent('平台知识截止日尚未确认')
  expect(screen.getByRole('button', { name: '2. 结果与确认' })).toBeDisabled()
  researchClock.day = null
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.getByLabelText(/目标研究日/)).toBeEnabled()
  expect(await screen.findByLabelText(/^风险标尺版本/)).toHaveValue(riskVersion.id)
})

it.each([
  [undefined, '2026-09-18'], ['2026-09-17', '2026-09-18'],
  ['2026-09-17', null], ['2026-09-17', undefined],
] as const)('已保存目标在PIT从%s变为%s后保留冻结诊断与只读状态', async (initial, next) => {
  researchClock.day = initial
  const fetch = install(); const user = userEvent.setup()
  const route = `/pre-investment/objectives/new?view=${savedVersion.id}`
  updateAllocationJourney({ mandateId: 'another-objective' })
  const tree = () => <MemoryRouter initialEntries={[route]}><InvestmentObjectivesWorkspace /></MemoryRouter>
  const view = render(tree())
  await screen.findByText('只读版本')
  expect(screen.getByRole('link', { name: /下一步：确定投资范围/ })).toHaveAttribute('href', `/pre-investment/product-pool?mandate=${savedVersion.id}`)
  const frozen = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')
  researchClock.day = next
  view.rerender(tree())
  expect(screen.getByText('只读版本')).toBeInTheDocument()
  expect(screen.getByText('当前约束下可实现')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '运行目标诊断' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: '保存新目标版本' })).not.toBeInTheDocument()
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')).toEqual(frozen)
  await user.click(screen.getByRole('button', { name: '1. 目标与约束' }))
  expect(screen.getByLabelText('目标名称')).toBeDisabled()
  expect(screen.getByLabelText(/目标研究日/)).toHaveValue(savedVersion.definition.as_of)
  expect(screen.getByLabelText(/^风险标尺版本/)).toHaveValue(riskVersion.id)
  expect(fetch.mock.calls.filter(([url]) => /\/mandates\/(preview|confirm)$/.test(String(url)))).toHaveLength(0)
})

it('PIT变化仍清除可编辑草稿的旧诊断与风险引用', async () => {
  install(); const user = userEvent.setup(); const view = ready()
  await openResult(user)
  researchClock.day = '2026-09-18'
  view.rerender(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
  expect(screen.queryByText('当前约束下可实现')).not.toBeInTheDocument()
  const draft = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!
  expect(draft.definition.as_of).toBe('2026-09-18')
  expect(draft.definition.risk_authorization?.risk_scale_ref).toBeNull()
})

it.each(['view', 'editFrom'])('版本读取失败的%s页在PIT恢复后只提供重试，不暴露本地草稿', async mode => {
  researchClock.day = undefined
  let unavailable = true
  const fetch = install({ [`${root}/mandates/${savedVersion.id}`]: () => unavailable
    ? response({ detail: { message: '保存版本暂时不可读' } }, 503) : response(savedVersion) })
  const user = userEvent.setup()
  const route = `/pre-investment/objectives/new?${mode}=${savedVersion.id}`
  const tree = () => <MemoryRouter initialEntries={[route]}><InvestmentObjectivesWorkspace /></MemoryRouter>
  writeAllocationDraft('mandate-study:editor', boundaryStudy())
  const view = render(tree())
  expect(await screen.findByRole('alert')).toHaveTextContent('保存版本暂时不可读')
  researchClock.day = '2026-09-18'; view.rerender(tree())
  expect(screen.queryByLabelText('目标名称')).not.toBeInTheDocument()
  expect(screen.queryByRole('navigation', { name: '投资目标步骤' })).not.toBeInTheDocument()
  expect(fetch.mock.calls.filter(([url]) => /\/mandates\/(funding|preview|confirm)$/.test(String(url)))).toHaveLength(0)
  unavailable = false
  await user.click(screen.getByRole('button', { name: '重试' }))
  if (mode === 'view') {
    expect(await screen.findByText('只读版本')).toBeInTheDocument()
    expect(screen.getByText('当前约束下可实现')).toBeInTheDocument()
    expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.as_of).toBe(savedVersion.definition.as_of)
  } else {
    expect(await screen.findByRole('heading', { name: '修改投资目标与约束' })).toBeInTheDocument()
    expect(screen.getByLabelText('目标名称')).toBeEnabled()
    expect(screen.getByLabelText(/目标研究日/)).toHaveValue('2026-09-18')
  }
})

it('相对基准目标只要求超额收益，基准自动来自所选风险等级代表组合', async () => {
  install(); const user = userEvent.setup(); ready()
  const objective = screen.getByLabelText(/投资目标类型/)
  await user.selectOptions(objective, 'benchmark_relative')
  expect(screen.getByLabelText(/目标年超额收益/)).toBeInTheDocument()
  expect(screen.queryByLabelText(/基准名称/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/基准大类方案/)).not.toBeInTheDocument()
  expect(screen.queryByLabelText(/相对基准主动风险上限/)).not.toBeInTheDocument()
  // 合同原文只留痕，同时把"建模用的是大类代理"写在脸上，不让这层近似无人知晓。
  await user.type(screen.getByLabelText(/合同业绩比较基准/), '沪深300×60%')
  expect(screen.getByText(/与逐指数口径存在基差/)).toBeInTheDocument()
  await user.selectOptions(objective, 'absolute_return')
  await user.selectOptions(objective, 'benchmark_relative')
  expect(screen.getByLabelText(/合同业绩比较基准/)).toHaveValue('')
  expect(screen.queryByText(/与逐指数口径存在基差/)).not.toBeInTheDocument()
})

it('诊断同时展示原参考前沿和现金约束前沿，C1-C5仍来自冻结Risk Scale', async () => {
  install(); const user = userEvent.setup(); ready(); await openResult(user)
  expect(screen.getAllByText('C2').length).toBeGreaterThan(0)
  expect(screen.getAllByText('10.00%').length).toBeGreaterThan(0)
  expect(screen.getByText('5.00%')).toBeInTheDocument()
  const charts = screen.getAllByTestId('echarts')
  const frontier = charts.find(node => node.getAttribute('data-series') === '2')
  expect(frontier).toBeTruthy(); expect(frontier).toHaveAttribute('data-bands', '5')
  expect(screen.getByText(/虚线：风险等级配置发布时的原始参考前沿/)).toBeInTheDocument()
  expect(screen.getByText(/实线：同一 Reference CMA 加入当前现金约束后的新前沿/)).toBeInTheDocument()
})

it('修改现金输入后迟到的旧诊断不能覆盖新输入', async () => {
  let finish!: (value: Response) => void
  install({ [`${root}/mandates/preview`]: () => new Promise(resolve => { finish = resolve }) })
  const user = userEvent.setup(); ready()
  await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await user.click(screen.getByRole('button', { name: '1. 目标与约束' }))
  fireEvent.change(screen.getByLabelText(/最低现金占比/), { target: { value: '20' } })
  await act(async () => finish(await response(boundaryAssessment())))
  expect(screen.queryByText('当前约束下可实现')).not.toBeInTheDocument()
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.min_cash_weight).toBe(.2)
})

it('从主列表修改已发布目标时直接进入可编辑副本，保存新版本会替代原版本', async () => {
  const fetch = install(); const user = userEvent.setup()
  ready(boundaryStudy(), `/pre-investment/objectives/new?editFrom=${savedVersion.id}`)
  expect(await screen.findByRole('heading', { name: '修改投资目标与约束' })).toBeInTheDocument()
  expect(screen.getByText(/正在修改“量化边界测试”/)).toBeInTheDocument()
  expect(screen.getByLabelText('目标名称')).toBeEnabled()
  fireEvent.change(screen.getByLabelText('目标名称'), { target: { value: '量化边界测试-修改版' } })
  await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  await screen.findByText('当前约束下可实现')
  await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
  await user.click(screen.getByRole('button', { name: '保存修改后的版本' }))
  expect(await screen.findByText('只读版本')).toBeInTheDocument()
  const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/mandates/confirm`)!
  expect(JSON.parse(String(call[1]?.body)).replaces_mandate_id).toBe(savedVersion.id)
})

it.each(['absolute_return', 'benchmark_relative'] as const)('编辑%s目标保留原现金保护，并随新版本提交', async kind => {
  const request = boundaryStudy()
  request.definition = { ...request.definition, objective_kind: kind, target_return: 0,
    target_excess_return: 0, funding_target: null,
    cash_protection: { mode: 'payments_and_terminal_floor', terminal_floor: { amount: 800000, amount_basis: 'real' } } }
  const assessment = boundaryAssessment(request)
  const version = { ...savedVersion, definition: assessment.definition, assessment }
  const fetch = install({ [`${root}/mandates/${version.id}`]: () => response(version) })
  const user = userEvent.setup()
  ready(request, `/pre-investment/objectives/new?editFrom=${version.id}`)
  await screen.findByRole('heading', { name: '修改投资目标与约束' })
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.cash_protection).toEqual(request.definition.cash_protection)
  fireEvent.change(screen.getByLabelText('目标名称'), { target: { value: '保留资金保护的新版本' } })
  await openResult(user)
  await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
  await user.click(screen.getByRole('button', { name: '保存修改后的版本' }))
  await screen.findByText('只读版本')
  const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/mandates/confirm`)!
  const submitted = JSON.parse(String(call[1]?.body))
  expect(submitted.replaces_mandate_id).toBe(version.id)
  expect(submitted.request.definition.cash_protection).toEqual(request.definition.cash_protection)
})

it('诊断API失败时保留草稿并显示错误，不伪造前沿结果', async () => {
  install({ [`${root}/mandates/preview`]: () => response({ detail: { message: '参考数据暂不可用。' } }, 422) })
  const user = userEvent.setup(); ready()
  await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
  await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('参考数据暂不可用')
  expect(screen.queryByText('当前约束下可实现')).not.toBeInTheDocument()
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.name).toBe('量化边界测试')
})

it('填写页底部回显考虑现金流后真正需要的收益，并判断所选风险等级够不够', async () => {
  install(); ready()
  const check = await screen.findByRole('region', { name: '这套配置需要的收益' })
  expect(await within(check).findByText(/需要扣费前年复合收益 4\.20%/)).toBeInTheDocument()
  // C3 的参考收益 3.50% 不到 4.20%，判断写在页面底部，不再只挂在上面的等级卡片上。
  expect(within(check).getByText(/高于 C3 的参考收益 3\.50%.*需要 C4/)).toBeInTheDocument()
})

it('非期末金额目标默认不打开本金与现金流计划', async () => {
  install(); const user = userEvent.setup()
  const request = boundaryStudy()
  request.definition = { ...request.definition, objective_kind: 'absolute_return', funding_target: null, cash_budget: null }
  ready(request)
  await user.selectOptions(screen.getByLabelText(/^投资目标类型/), 'benchmark_relative')
  const ledger = screen.getByRole('checkbox', { name: /这笔资金有明确本金/ })
  expect(ledger).not.toBeChecked()
  expect(screen.queryByLabelText(/^总资金/)).not.toBeInTheDocument()
})

it('切换目标类型保留本金、储备、现金流及兼容的现金保护', async () => {
  install(); const user = userEvent.setup(); const request = boundaryStudy()
  request.definition = { ...request.definition, objective_kind: 'absolute_return', funding_target: null,
    cash_budget: { ...request.definition.cash_budget!, outside_reserve: 50000,
      flows: [{ name: '必要支付', kind: 'withdrawal', amount: 10000, first_month: 3, last_month: 3, every_months: 1 }] },
    cash_protection: { mode: 'payments_and_terminal_floor', terminal_floor: { amount: 800000, amount_basis: 'real' } } }
  ready(request)
  for (const kind of ['benchmark_relative', 'absolute_return', 'funding_goal', 'absolute_return']) {
    await user.selectOptions(screen.getByLabelText(/^投资目标类型/), kind)
    const current = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition
    expect(current.cash_budget).toEqual(request.definition.cash_budget)
    if (kind === 'benchmark_relative') expect(current.cash_protection).toEqual(request.definition.cash_protection)
    if (kind === 'funding_goal') expect(current.cash_protection).toBeNull()
  }
})

it('自定义加权基准按资产名称列出权重，种子权重不带求解器残差', async () => {
  const result = riskVersion.preview.result
  const noisy = { ...riskVersion, preview: { ...riskVersion.preview, result: { ...result,
    levels: result.levels.map(level => ({ ...level, representative_weights: [.564362546525, .435637453475] })) } } }
  install({ [`${root}/risk-scales/${riskVersion.id}`]: () => response(noisy) })
  const user = userEvent.setup(); ready()
  await user.selectOptions(screen.getByLabelText(/^投资目标类型/), 'benchmark_relative')
  await user.selectOptions(screen.getByRole('option', { name: '自定义加权基准' }).closest('select')!, 'custom')
  // 权重按大类资产名称标注，不是内部 id；求解器残差按展示精度取整后合计仍是 100%。
  expect(screen.getByLabelText('现金')).toHaveValue('56.44')
  expect(screen.getByLabelText('权益')).toHaveValue('43.56')
  expect(screen.queryByLabelText('cash')).not.toBeInTheDocument()
  expect(screen.getByText(/大类权重合计 100\.00%/)).toBeInTheDocument()
  // 每个大类展开成参考输入里的指数/产品，用户看得见自己在给什么加权。
  expect(await screen.findByText('成分 000300.SH 100.00%')).toBeInTheDocument()
  expect(screen.getByText('现金资产 · 参考利率 1.00%')).toBeInTheDocument()
})

it.each(['draft', 'editFrom', 'manual'])('日期变化不自动改写现金事实，%s入口须明确确认预算', async mode => {
  researchClock.day = mode === 'manual' ? null : '2026-09-18'
  install(); const user = userEvent.setup()
  const request = boundaryStudy()
  const entry = mode === 'editFrom' ? `/pre-investment/objectives/new?editFrom=${savedVersion.id}` : '/pre-investment/objectives/new'
  ready(request, entry)
  await screen.findByLabelText('目标名称')
  if (mode === 'manual') fireEvent.change(screen.getByLabelText(/目标研究日/), { target: { value: '2026-09-18' } })
  const confirm = await screen.findByRole('button', { name: '我已核对资金和现金流，确认按新研究日使用' })
  const pending = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!
  expect(pending.definition.as_of).toBe('2026-09-18')
  expect(pending.definition.cash_budget).toEqual(request.definition.cash_budget)
  expect(screen.getByRole('button', { name: '2. 结果与确认' })).toBeDisabled()
  await user.click(confirm)
  const rolled = readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!
  expect(rolled.definition.cash_budget).toEqual({ ...request.definition.cash_budget, balance_as_of: '2026-09-18' })
  expect(screen.queryByRole('button', { name: '我已核对资金和现金流，确认按新研究日使用' })).not.toBeInTheDocument()
})

it.each(['absolute_return', 'benchmark_relative'] as const)('%s现金计划可显式选择仅支付保护并提交独立验证', async kind => {
  const fetch = install(); const user = userEvent.setup(); const request = boundaryStudy()
  request.definition = { ...request.definition, objective_kind: kind, target_return: 0, target_excess_return: 0,
    funding_target: null, cash_protection: null, cash_budget: { ...request.definition.cash_budget!,
      flows: [{ name: '必要支付', kind: 'withdrawal', amount: 10000, first_month: 1, last_month: 1, every_months: 1 }] } }
  ready(request)
  await user.selectOptions(screen.getByRole('combobox', { name: /^现金保护条件/ }), 'payments_only')
  expect(screen.queryByLabelText(/^期末至少保有/)).not.toBeInTheDocument()
  await openResult(user)
  const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/mandates/preview`)!
  const definition = JSON.parse(String(call[1]?.body)).definition
  expect(definition.cash_protection).toEqual({ mode: 'payments_only' })
  expect(definition.funding_target).toBeNull()
})

it('期限变化重算循环现金流末月，单次支付原日期保持且无效期限不改计划', async () => {
  install(); const request = boundaryStudy()
  request.definition = { ...request.definition, horizon_years: 2, cash_budget: { ...request.definition.cash_budget!, flows: [
    { name: '季度投入', kind: 'contribution', amount: 10000, first_month: 2, last_month: 23, every_months: 3 },
    { name: '单次支付', kind: 'withdrawal', amount: 5000, first_month: 6, last_month: 6, every_months: 1 },
  ] } }
  ready(request)
  const horizon = screen.getByLabelText('投资期限（年）')
  fireEvent.change(horizon, { target: { value: '5' } })
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.cash_budget!.flows.map(f => f.last_month)).toEqual([59, 6])
  fireEvent.change(horizon, { target: { value: '1.5' } })
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.cash_budget!.flows.map(f => f.last_month)).toEqual([59, 6])
  fireEvent.change(horizon, { target: { value: '2' } })
  expect(readAllocationDraft<MandateStudyRequest>('mandate-study:editor')!.definition.cash_budget!.flows.map(f => f.last_month)).toEqual([23, 6])
})
