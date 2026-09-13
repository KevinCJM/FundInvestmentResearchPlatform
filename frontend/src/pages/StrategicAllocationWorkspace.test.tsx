import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import StrategicAllocationWorkspace from './StrategicAllocationWorkspace'
import InvestmentObjectivesWorkspace from './InvestmentObjectivesWorkspace'
import TaaWalkForward from '../components/tactical-allocation/TaaWalkForward'
import { cmaDefinition, cmaPreview, cmaVersion, mandateVersion, policyBaseline, policyPreview, strategicCatalog } from '../test/strategicAllocationFixtures'
import { taaExecution } from '../test/tacticalAllocationFixtures'
import { assessment, fundingStudy } from '../test/mandateFixtures'
import { completeCma, previewCma } from '../services/strategicAllocation'
import { writeAllocationDraft } from '../app/allocationJourney'

const researchClock = vi.hoisted(() => ({ day: '2026-09-12' }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => researchClock.day }))
const response = (value: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => value } as Response)
const root = '/api/strategic-allocation'
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const mock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (overrides[url]) return overrides[url](init)
    if (url === `${root}/catalog`) return response(strategicCatalog)
    if (url === `${root}/mandates/preview`) {
      const request = JSON.parse(String(init?.body))
      return response({ request, definition: request.definition, preview_hash: 'e'.repeat(64), execution: taaExecution,
        status: 'inputs_only', funding: null, candidates: [], cma: null, blockers: [], warnings: [] })
    }
    if (url === `${root}/mandates/confirm`) {
      const { request } = JSON.parse(String(init?.body))
      return response({ ...mandateVersion, definition: request.definition, assessment: { request, definition: request.definition,
        preview_hash: 'e'.repeat(64), execution: taaExecution, status: 'inputs_only', funding: null, candidates: [], cma: null, blockers: [], warnings: [] } }, 201)
    }
    if (url === `${root}/cma/cma-1`) return response(cmaVersion)
    if (url === `${root}/cma/preview`) return response({ ...cmaPreview, definition: JSON.parse(String(init?.body)) })
    if (url === `${root}/cma`) return response({ ...cmaVersion, definition: JSON.parse(String(init?.body)).request }, 201)
    if (url === `${root}/policy/preview`) return response({ ...policyPreview, request: JSON.parse(String(init?.body)) })
    if (url === `${root}/policies`) return response(policyBaseline, 201)
    throw new Error(`Unexpected API: ${url}`)
  })
  vi.stubGlobal('fetch', mock)
  return mock
}
function policyTree() {
  return <MemoryRouter initialEntries={['/pre-investment/saa/policy?alloc=股债分类&mandate=mandate-1']}><Routes>
    <Route path="/pre-investment/saa/policy" element={<StrategicAllocationWorkspace />} />
    <Route path="/pre-investment/taa" element={<p>已到战术研究</p>} />
  </Routes></MemoryRouter>
}
function renderPolicy() {
  return render(policyTree())
}
async function loadCma(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByLabelText('或者使用已保存的长期假设')
  await waitFor(() => expect(screen.getByRole('button', { name: '填写长期假设' })).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('或者使用已保存的长期假设'), 'cma-1')
  await screen.findByRole('button', { name: '比较符合目标的政策候选' })
}

beforeEach(() => { researchClock.day = '2026-09-12'; localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

describe('真实自上而下操作顺序', () => {
  it('目标未完成时不允许保存；保存后直接携带目标版本进入政策研究', async () => {
    const fetch = install()
    const user = userEvent.setup()
    render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
    expect(screen.queryByRole('button', { name: '保存新目标版本' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '下一步：风险与限制' })).toBeDisabled()
    await user.type(screen.getByLabelText('目标名称'), '养老配置目标')
    fireEvent.change(screen.getByLabelText(/最低预期年收益/), { target: { value: '3' } })
    await user.click(screen.getByRole('button', { name: '下一步：风险与限制' }))
    fireEvent.change(screen.getByLabelText('最高预期年波动（%）'), { target: { value: '15' } })
    fireEvent.change(screen.getByLabelText(/风险与流动性边界的依据/), { target: { value: '根据必要支出及损失承受能力确定' } })
    await user.click(screen.getByRole('button', { name: '下一步：量化诊断' }))
    await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '下一步：核对与确认' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '下一步：核对与确认' }))
    expect(screen.getByRole('button', { name: '保存新目标版本' })).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
    await user.click(screen.getByRole('button', { name: '保存新目标版本' }))
    expect(await screen.findByRole('link', { name: /使用此目标进入长期配置/ })).toHaveAttribute('href', '/pre-investment/saa/policy?mandate=mandate-1')
    const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/mandates/confirm`)!
    expect(JSON.parse(String(call[1]?.body)).request.definition.target_return).toBe(.03)
    expect(screen.getByRole('button', { name: '已锁定此目标版本' })).toBeDisabled()
  })

  it('SAA 主入口明确提供历史有效前沿实验入口，但不把它混成前瞻 CMA', async () => {
    install(); renderPolicy()
    const link = await screen.findByRole('link', { name: '打开历史有效前沿与策略回测' })
    expect(link).toHaveAttribute('href', '/pre-investment/saa/allocation-lab?alloc=%E8%82%A1%E5%80%BA%E5%88%86%E7%B1%BB')
    expect(screen.getByText(/历史有效前沿属于独立实验工具/)).toBeInTheDocument()
  })

  it('选择范围后不编造预期收益、经济用途或相关性', async () => {
    install(); const user = userEvent.setup(); renderPolicy()
    await waitFor(() => expect(screen.getByRole('button', { name: '填写长期假设' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '填写长期假设' }))
    expect(screen.getByLabelText('equity预期年收益（%）')).toHaveValue('')
    expect(screen.getByLabelText('equity经济角色')).toHaveValue('')
    expect(screen.getByLabelText('equity与bond相关系数')).toHaveValue('')
    expect(screen.getByRole('button', { name: '验证长期假设' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '确认保存假设版本' })).toBeDisabled()
  })

  it('已保存 CMA → 比较 → 理由 → 确认 → TAA，不会自动采纳', async () => {
    const fetch = install(); const user = userEvent.setup(); renderPolicy()
    await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await user.click(await screen.findByRole('button', { name: '复核此候选' }))
    expect(screen.getByRole('button', { name: '确认采用此长期政策' })).toBeDisabled()
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/policies`)).toHaveLength(0)
    await user.type(screen.getByLabelText(/采纳理由与复核关注点/), '保守假设下仍符合长期目标')
    await user.click(screen.getByRole('button', { name: '确认采用此长期政策' }))
    await user.click(await screen.findByRole('button', { name: /进入 TAA，研究是否需要偏离/ }))
    expect(await screen.findByText('已到战术研究')).toBeInTheDocument()
    const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/policies`)!
    expect(JSON.parse(String(call[1]?.body))).toMatchObject({ candidate_id: 'robust-utility', preview_hash: policyPreview.preview_hash, request: { mandate_id: 'mandate-1', cma_id: 'cma-1' } })
  })

  it('修改约束立即清除旧结果，迟到计算不能覆盖新输入', async () => {
    let resolve!: (value: Response) => void
    install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    fireEvent.change(screen.getByLabelText('equity最高权重'), { target: { value: '60' } })
    await act(async () => resolve(await response(policyPreview)))
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('equity最高权重')).toHaveValue('60')
  })

  it('复核页调早知识截止日后不能采纳旧候选，输入保留', async () => {
    const fetch = install(); const user = userEvent.setup(); const view = renderPolicy()
    await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await user.click(await screen.findByRole('button', { name: '复核此候选' }))
    fireEvent.change(screen.getByLabelText(/采纳理由与复核关注点/), { target: { value: '研究时钟变化后必须重新核对' } })
    researchClock.day = '2026-09-11'
    view.rerender(policyTree())
    expect(screen.getByRole('alert')).toHaveTextContent('晚于平台知识截止')
    expect(screen.queryByRole('button', { name: '确认采用此长期政策' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeDisabled()
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('equity最高权重')).toHaveValue('100')
    expect(fetch.mock.calls.some(([url]) => String(url) === `${root}/policies`)).toBe(false)
  })

  it('比较在途时改变知识截止日，迟到结果不恢复旧候选', async () => {
    let resolve!: (value: Response) => void
    install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
    const user = userEvent.setup(); const view = renderPolicy(); await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    researchClock.day = '2026-09-11'
    view.rerender(policyTree())
    await act(async () => resolve(await response(policyPreview)))
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '复核此候选' })).not.toBeInTheDocument()
  })

  it('已保存政策在知识截止日变化后继续只读展示，不额外保存', async () => {
    const fetch = install(); const user = userEvent.setup(); const view = renderPolicy(); await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await user.click(await screen.findByRole('button', { name: '复核此候选' }))
    fireEvent.change(screen.getByLabelText(/采纳理由与复核关注点/), { target: { value: '冻结后的政策继续作为历史记录' } })
    await user.click(screen.getByRole('button', { name: '确认采用此长期政策' }))
    await screen.findByRole('button', { name: '长期政策已确认' })
    researchClock.day = '2026-09-11'
    view.rerender(policyTree())
    expect(screen.getByRole('button', { name: '长期政策已确认' })).toBeDisabled()
    expect(screen.getByText(/当前展示已保存的历史政策/)).toBeInTheDocument()
    expect(screen.getByRole('region', { name: '政策采纳确认' })).toBeInTheDocument()
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/policies`)).toHaveLength(1)
  })

  it('矩阵失败显示可操作原因，预览不执行保存', async () => {
    const fetch = install({ [`${root}/cma/preview`]: () => response({ detail: { message: '相关矩阵不是半正定矩阵，请检查相关系数。' } }, 422) })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await user.click(screen.getByRole('button', { name: '2. 长期假设' }))
    await user.click(screen.getByRole('button', { name: '验证长期假设' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('半正定')
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/cma`)).toHaveLength(0)
  })

  it('目标保存中修改输入，不把旧版本当成新目标', async () => {
    let resolve!: (value: Response) => void
    install({ [`${root}/mandates/confirm`]: () => new Promise(done => { resolve = done }) })
    writeAllocationDraft('strategic-mandate:editor', { ...mandateVersion.definition, boundary_reason: '旧目标需要重新核对风险承受能力' })
    const user = userEvent.setup(); render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
    await user.click(screen.getByRole('button', { name: '下一步：风险与限制' }))
    await user.click(screen.getByRole('button', { name: '下一步：量化诊断' }))
    await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '下一步：核对与确认' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '下一步：核对与确认' }))
    await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
    await user.click(screen.getByRole('button', { name: '保存新目标版本' }))
    await user.click(screen.getByRole('button', { name: '1. 资金与成功标准' }))
    fireEvent.change(screen.getByLabelText('目标名称'), { target: { value: '修改后的目标' } })
    await act(async () => resolve(await response(mandateVersion, 201)))
    expect(screen.queryByRole('link', { name: /使用此目标进入长期配置/ })).not.toBeInTheDocument()
    expect(screen.getByLabelText('目标名称')).toHaveValue('修改后的目标')
  })

  it('资金目标未达门槛时可以复核，但填写理由也不能采用', async () => {
    const failed = assessment({ ...fundingStudy, cma_id: 'cma-1' })
    const fetch = install({ [`${root}/policy/preview`]: () => response({ ...policyPreview, candidates: failed.candidates, mandate: failed.definition, funding: failed.funding, funding_execution: failed.funding_execution }) })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await user.click(await screen.findByRole('button', { name: '复核此候选' }))
    await user.type(screen.getByLabelText(/采纳理由与复核关注点/), '理由不能代替未通过的概率门槛')
    expect(screen.getByText(/区间下界未达到/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '确认采用此长期政策' })).toBeDisabled()
    expect(fetch.mock.calls.some(([url]) => String(url) === `${root}/policies`)).toBe(false)
  })

  it('不合格数值执行证明会阻止展示', async () => {
    install({ [`${root}/cma/preview`]: () => response({ ...cmaPreview, execution: { ...taaExecution, python_fallback: 1 } }) })
    await expect(previewCma(cmaDefinition)).rejects.toThrow()
    expect(completeCma({ ...cmaDefinition, assets: [{ ...cmaDefinition.assets[0], annual_return: NaN }] })).toBe(false)
  })

  it('多段验证开关有明确参数，不把未知训练样本显示为通过', async () => {
    const onChange = vi.fn()
    const user = userEvent.setup()
    const view = render(<TaaWalkForward value={null} disabled={false} onChange={onChange} />)
    await user.click(screen.getByRole('checkbox', { name: '同时运行多段样本外检验' }))
    expect(onChange).toHaveBeenCalledWith({ window_mode: 'rolling', training_periods: 126, validation_periods: 63 })
    view.rerender(<TaaWalkForward value={{ window_mode: 'rolling', training_periods: 126, validation_periods: 63 }} disabled={false} onChange={onChange} result={{
      config: { window_mode: 'rolling', training_periods: 126, validation_periods: 63 }, completed_folds: 0, blocked_folds: 1,
      excluded_tail_observations: 0, primary_selection_changed: false, independently_funded_intervals: true, execution: taaExecution,
      warnings: ['各段独立，不拼接实盘净值。'], folds: [{ fold: 1, status: 'blocked', reasons: ['训练收益可得时点未知'], train_start: '2020-01-01', train_end: '2020-07-01', validation_start: '2020-07-02', validation_end: '2020-10-01', training_observations: 126, validation_observations: 63, purged_training_periods: 0 }],
    }} />)
    expect(screen.getByRole('table', { name: '分段样本外结果' })).toHaveTextContent('训练收益可得时点未知')
    expect(screen.queryByText('验证未超限')).not.toBeInTheDocument()
  })
})
