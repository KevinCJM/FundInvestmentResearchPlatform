import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import StrategicAllocationWorkspace from './StrategicAllocationWorkspace'
import LtcmaWorkspace from './LtcmaWorkspace'
import { ltcmaCapabilities, ltcmaOptions } from '../test/ltcmaFixtures'
import InvestmentObjectivesWorkspace from './InvestmentObjectivesWorkspace'
import TaaWalkForward from '../components/tactical-allocation/TaaWalkForward'
import { cmaDefinition, cmaPreview, cmaVersion, mandateVersion, policyBaseline, policyPreview, strategicCatalog } from '../test/strategicAllocationFixtures'
import { boundaryAssessment, boundaryStudy } from '../test/mandateBoundaryFixtures'
import { riskVersion } from '../test/riskScaleFixtures'
import { taaExecution } from '../test/tacticalAllocationFixtures'
import { assessment, fundingStudy } from '../test/mandateFixtures'
import { completeCma, previewCma } from '../services/strategicAllocation'
import { allocationJourneyPath, writeAllocationDraft } from '../app/allocationJourney'
import { cmaSelectionReason } from '../components/strategic-allocation/LtcmaSelection'

it('uses the investment horizon only from the mandate when selecting old or new CMA', () => {
  const context = { allocationName: cmaDefinition.alloc_name!, cutoff: cmaDefinition.as_of,
    mandate: { ...mandateVersion.definition, horizon_years: 5 } }
  expect(cmaSelectionReason(cmaDefinition, context)).toBeNull()
  const { horizon_years: _oldLabel, ...withoutHorizon } = cmaDefinition
  expect(cmaSelectionReason(withoutHorizon, context)).toBeNull()
  expect(cmaSelectionReason({ ...withoutHorizon, currency: 'USD' }, context)).toBe('handoffGoalCurrency')
})

const researchClock = vi.hoisted(() => ({ day: '2026-09-12' as string | null | undefined }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => researchClock.day }))
vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="echarts" /> }))

it('waits for PIT before loading LTCMA options and preserves inputs on later clock changes', async () => {
  researchClock.day = undefined
  const fetch = install()
  const tree = () => <MemoryRouter initialEntries={['/pre-investment/ltcma/new?alloc=股债分类']}><LtcmaWorkspace /></MemoryRouter>
  const mounted = render(tree())
  expect(fetch.mock.calls.some(([url]) => String(url).includes('/cma/study-options'))).toBe(false)
  researchClock.day = '2019-12-31'
  mounted.rerender(tree())
  await screen.findByLabelText('生成方法')
  expect(fetch.mock.calls.filter(([url]) => String(url).includes('/cma/study-options')).map(([url]) => String(url)))
    .toEqual([`${root}/cma/study-options?section=base&as_of=2019-12-31`])
  fireEvent.click(screen.getByLabelText('生成方法'))
  fireEvent.click(screen.getByRole('menuitemradio', { name: '历史统计' }))
  researchClock.day = '2020-01-02'
  mounted.rerender(tree())
  await waitFor(() => expect(screen.getByLabelText('研究日')).toHaveValue('2020-01-02'))
  expect(screen.getByLabelText('生成方法')).toHaveValue('historical_statistics')
})
const response = (value: unknown, status = 200) => Promise.resolve({ ok: status < 400, status, json: async () => value } as Response)
const root = '/api/strategic-allocation'
function install(overrides: Record<string, (init?: RequestInit) => Promise<Response>> = {}) {
  const mock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    if (overrides[url]) return overrides[url](init)
    if (url === `${root}/catalog`) return response(strategicCatalog)
    if (url === `${root}/cma/capabilities`) return response(ltcmaCapabilities)
    if (url.split('?')[0] === `${root}/cma/study-options`) return response(ltcmaOptions)
    if (url.startsWith(`${root}/risk-scales/study-options?`)) return response({ as_of: '2026-09-17', items: [{
      id: riskVersion.id, name: riskVersion.name, content_hash: riskVersion.content_hash, version_number: riskVersion.version_number,
      base_currency: 'CNY', risk_basis_id: 'annualized-periodic-volatility-v1', research_as_of: '2026-09-17', valid_until: null,
    }] })
    if (url === `${root}/risk-scales/${riskVersion.id}`) return response(riskVersion)
    if (url === `${root}/mandates/preview`) {
      const request = JSON.parse(String(init?.body))
      return response(boundaryAssessment(request))
    }
    if (url === `${root}/mandates/confirm`) {
      const { request } = JSON.parse(String(init?.body))
      const assessment = boundaryAssessment(request)
      return response({ ...mandateVersion, definition: assessment.definition, assessment }, 201)
    }
    if (url === `${root}/mandates/mandate-1`) { const assessment = boundaryAssessment(boundaryStudy()); return response({ ...mandateVersion, definition: assessment.definition, assessment }) }
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
  await screen.findByLabelText('选择已确认 LTCMA')
  await waitFor(() => expect(screen.getByRole('button', { name: '选择已确认 LTCMA' })).toBeEnabled())
  await user.selectOptions(screen.getByLabelText('选择已确认 LTCMA'), 'cma-1')
  await screen.findByRole('button', { name: '比较符合目标的政策候选' })
}

beforeEach(() => { researchClock.day = '2026-09-12'; localStorage.clear(); sessionStorage.clear(); vi.clearAllMocks() })
afterEach(() => { vi.unstubAllGlobals() })

describe('真实自上而下操作顺序', () => {
  it('政策候选不可行时仍展示前沿和目标差距', async () => {
    install({
      [`${root}/policy/frontier`]: () => response({ mode: 'single', execution: policyPreview.execution,
        additional_checks: {}, views: [{ id: 'cma-1', name: '测试 CMA', target_return: .0772,
          volatility_cap: .095, cash_floor: .1,
          reference: { complete: true, points: [{ status: 'optimal_to_tolerance', volatility: .2, expected_return: .0417 }] },
          configured: { complete: true, max_return: .0376, points: [{ status: 'optimal_to_tolerance', volatility: .18, expected_return: .0376 }] },
        }] }),
      [`${root}/policy/preview`]: () => response({ detail: { code: 'SAA_NO_FEASIBLE_CANDIDATE', message: '当前目标下未找到可行候选。' } }, 422),
    })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await screen.findByText(/收益上限也只有 3.76%/)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await screen.findByText(/当前目标下未找到可行候选/)
    expect(screen.getByTestId('saa-frontier-chart')).toBeInTheDocument()
    expect(screen.getByText('7.72%')).toBeInTheDocument()
  })

  it('CMA 深链接只初始化一次，手动切换和清空不被旧链接覆盖', async () => {
    const second = { ...cmaVersion, id: 'cma-2', name: '第二份长期假设' }
    let completeSecond!: (value: Response) => void
    const fetch = install({
      [`${root}/catalog`]: () => response({ ...strategicCatalog, assumptions: [
        ...strategicCatalog.assumptions, { ...strategicCatalog.assumptions[0], id: second.id, name: second.name },
      ] }),
      [`${root}/cma/cma-2`]: () => new Promise(resolve => { completeSecond = resolve }),
    })
    render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?alloc=股债分类&mandate=mandate-1&cma=cma-1']}><StrategicAllocationWorkspace /></MemoryRouter>)
    await screen.findByRole('button', { name: '比较符合目标的政策候选' })
    fireEvent.click(screen.getByRole('button', { name: '2. 选择 LTCMA' }))
    fireEvent.change(screen.getByLabelText('选择已确认 LTCMA'), { target: { value: second.id } })
    await waitFor(() => expect(completeSecond).toBeTypeOf('function'))
    await act(async () => { completeSecond(await response(second)) })
    fireEvent.click(screen.getByRole('button', { name: '2. 选择 LTCMA' }))
    expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue(second.id)
    fireEvent.change(screen.getByLabelText('选择已确认 LTCMA'), { target: { value: '' } })
    await waitFor(() => expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue(''))
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/cma/cma-1`)).toHaveLength(1)
  })

  it('目标确认后只冻结目标/风险/现金，并进入后续产品范围而不是提前选择正式CMA', async () => {
    researchClock.day = '2026-09-17'
    const fetch = install(); const user = userEvent.setup()
    writeAllocationDraft('mandate-study:editor', boundaryStudy())
    render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
    expect(screen.queryByLabelText(/CMA/)).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
    await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
    expect(await screen.findByRole('button', { name: '保存新目标版本' })).toBeDisabled()
    await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
    await user.click(screen.getByRole('button', { name: '保存新目标版本' }))
    expect(await screen.findByRole('link', { name: /下一步：确定投资范围/ })).toHaveAttribute('href', '/pre-investment/product-pool?mandate=mandate-1')
    const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/mandates/confirm`)!
    const submitted = JSON.parse(String(call[1]?.body)).request
    expect(submitted.cma_id).toBeNull()
    expect(submitted.definition.risk_authorization.selected_max_level).toBe(3)
    expect(screen.getByText('只读版本')).toBeInTheDocument()
  })

  it('SAA 主入口明确提供历史有效前沿实验入口，但不把它混成前瞻 CMA', async () => {
    install(); renderPolicy()
    const link = await screen.findByRole('link', { name: '打开历史有效前沿与策略回测' })
    expect(link).toHaveAttribute('href', '/pre-investment/saa/allocation-lab?alloc=%E8%82%A1%E5%80%BA%E5%88%86%E7%B1%BB')
    expect(screen.getByText(/历史有效前沿用于对照研究，不替代 LTCMA/)).toBeInTheDocument()
  })

  it('选择范围后不编造预期收益、经济用途或相关性', async () => {
    install(); const user = userEvent.setup(); renderPolicy()
    await waitFor(() => expect(screen.getByRole('button', { name: '选择已确认 LTCMA' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '选择已确认 LTCMA' }))
    expect(screen.queryByLabelText('equity预期年收益（%）')).not.toBeInTheDocument()
    expect(screen.queryByLabelText('equity经济角色')).not.toBeInTheDocument()
    expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue('')
    expect(screen.queryByRole('button', { name: '验证长期假设' })).not.toBeInTheDocument()
    expect(screen.getByRole('link', { name: '新建 LTCMA' }).getAttribute('href')).toContain('/pre-investment/ltcma/new?')
  })

  it('已保存 CMA → 比较 → 理由 → 确认 → TAA，不会自动采纳', async () => {
    const fetch = install({ [`${root}/policy/preview`]: () => response({ ...policyPreview, current_application_eligible: false, application_blockers: ['战略资产 权益 缺少真实代理产品。'] }) }); const user = userEvent.setup(); renderPolicy()
    await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await screen.findByRole('table', { name: '长期政策候选比较' })
    expect(screen.queryByText(/缺少真实代理产品|当前应用条件尚未满足/)).not.toBeInTheDocument()
    // 比较结果是单独一页：设置区和比较按钮不在这一页，返回后可再回到结果。
    expect(screen.queryByRole('button', { name: '比较符合目标的政策候选' })).not.toBeInTheDocument()
    expect(screen.getAllByRole('region', { name: '目标与有效前沿' })).toHaveLength(1)
    await user.click(screen.getByRole('button', { name: '返回调整设置' }))
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '查看候选比较结果' }))
    await screen.findByRole('table', { name: '长期政策候选比较' })
    await user.click(screen.getByRole('button', { name: '复核此候选' }))
    expect(screen.queryByText(/缺少真实代理产品|当前应用条件尚未满足/)).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '确认采用此长期政策' })).toBeDisabled()
    const rationale = screen.getByLabelText('采纳理由与复核关注点')
    const policyName = screen.getByLabelText('政策版本名称')
    const confirm = screen.getByRole('button', { name: '确认采用此长期政策' })
    expect(rationale).toBeRequired()
    expect(policyName).toBeRequired()
    expect(rationale.closest('label')).toHaveAttribute('data-required', 'true')
    expect(policyName.closest('label')).toHaveAttribute('data-required', 'true')
    expect(rationale).toHaveAttribute('minlength', '5')
    expect(rationale).toHaveAttribute('maxlength', '2000')
    expect(confirm).toHaveAccessibleDescription('采纳理由至少 5 个字，还需填写 5 个字。')
    fireEvent.change(rationale, { target: { value: '  采用理由  ' } })
    expect(confirm).toBeDisabled()
    expect(confirm).toHaveAccessibleDescription('采纳理由至少 5 个字，还需填写 1 个字。')
    expect(screen.getByText('已填 4 / 2000 字')).toBeInTheDocument()
    fireEvent.change(rationale, { target: { value: '采用理由足' } })
    expect(confirm).toBeEnabled()
    fireEvent.change(policyName, { target: { value: '   ' } })
    expect(confirm).toHaveAccessibleDescription('请填写政策版本名称。')
    expect(confirm).toBeDisabled()
    fireEvent.change(policyName, { target: { value: '长期方案' } })
    fireEvent.change(rationale, { target: { value: '字'.repeat(2001) } })
    expect(confirm).toBeDisabled()
    expect(confirm).toHaveAccessibleDescription('采纳理由不能超过 2000 个字，请精简内容。')
    fireEvent.change(rationale, { target: { value: '' } })
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/policies`)).toHaveLength(0)
    await user.type(screen.getByLabelText(/采纳理由与复核关注点/), '保守假设下仍符合长期目标')
    await user.click(screen.getByRole('button', { name: '确认采用此长期政策' }))
    await user.click(await screen.findByRole('button', { name: /进入 TAA，研究是否需要偏离/ }))
    expect(await screen.findByText('已到战术研究')).toBeInTheDocument()
    const call = fetch.mock.calls.find(([url]) => String(url) === `${root}/policies`)!
    expect(JSON.parse(String(call[1]?.body))).toMatchObject({ candidate_id: 'robust-utility', preview_hash: policyPreview.preview_hash, request: { mandate_id: 'mandate-1', cma_id: 'cma-1' } })
  })

  it('uses frozen asset names throughout policy comparison while submitting original IDs and retaining policy limits', async () => {
    const fetch = install({
      [`${root}/catalog`]: () => response({ ...strategicCatalog, allocations: strategicCatalog.allocations.map(allocation => ({
        ...allocation, assets: allocation.assets.map(asset => ({ ...asset, name: `已改名-${asset.id}` })),
      })) }),
      [`${root}/cma/cma-1`]: () => response({ ...cmaVersion, source_snapshot: {
        ...cmaVersion.source_snapshot, assets: [...cmaVersion.source_snapshot.assets].reverse(),
      } }),
    })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    const constraints = within(screen.getByRole('table', { name: '政策资产约束' }))
    expect(constraints.getAllByRole('rowheader').map(cell => cell.textContent)).toEqual(['权益', '债券'])
    expect(screen.queryByRole('columnheader', { name: /战术偏离/ })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('权益最高权重'), { target: { value: '60' } })
    await user.click(screen.getByText('联合约束与候选搜索设置'))
    await user.click(screen.getByRole('button', { name: '增加联合约束' }))
    await user.click(screen.getByRole('checkbox', { name: '权益' }))
    await user.click(screen.getByRole('checkbox', { name: '增加风险预算候选' }))
    fireEvent.change(screen.getByLabelText('权益风险预算（%）'), { target: { value: '40' } })
    fireEvent.change(screen.getByLabelText('债券风险预算（%）'), { target: { value: '60' } })
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    const results = within(await screen.findByRole('table', { name: '长期政策候选比较' }))
    expect(results.getByRole('columnheader', { name: '权益' })).toBeInTheDocument()
    expect(results.getByRole('columnheader', { name: '债券' })).toBeInTheDocument()
    const submitted = JSON.parse(String(fetch.mock.calls.find(([url]) => String(url) === `${root}/policy/preview`)![1]?.body))
    expect(submitted.constraints).toEqual({
      equity: { min_weight: 0, max_weight: .6, max_abs_tilt: .1 },
      bond: { min_weight: 0, max_weight: 1, max_abs_tilt: .1 },
    })
    expect(submitted.group_limits[0].assets).toEqual(['equity'])
    expect(submitted.risk_budget).toEqual({ equity: .4, bond: .6 })
    await user.click(screen.getByText('风险贡献与证据限制'))
    expect(screen.getByText(/权益 80.00%；债券 20.00%/)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '复核此候选' }))
    expect(screen.getByText('权益', { selector: 'dt' })).toBeInTheDocument()
    expect(screen.getByText('债券', { selector: 'dt' })).toBeInTheDocument()
    expect(screen.queryByText('equity', { exact: true })).not.toBeInTheDocument()
  })

  it('修改约束立即清除旧结果，迟到计算不能覆盖新输入', async () => {
    let resolve!: (value: Response) => void
    install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    fireEvent.change(screen.getByLabelText('权益最高权重'), { target: { value: '60' } })
    await act(async () => resolve(await response(policyPreview)))
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('权益最高权重')).toHaveValue('60')
  })

  it('复核页调早知识截止日后不能采纳旧候选，输入保留', async () => {
    const fetch = install(); const user = userEvent.setup(); const view = renderPolicy()
    await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    await user.click(await screen.findByRole('button', { name: '复核此候选' }))
    fireEvent.change(screen.getByLabelText(/采纳理由与复核关注点/), { target: { value: '研究时钟变化后必须重新核对' } })
    researchClock.day = '2026-09-11'
    view.rerender(policyTree())
    expect(screen.getByRole('alert')).toHaveTextContent('晚于平台知识截止')
    expect(screen.queryByRole('button', { name: '确认采用此长期政策' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeDisabled()
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.getByLabelText('权益最高权重')).toHaveValue('100')
    expect(fetch.mock.calls.some(([url]) => String(url) === `${root}/policies`)).toBe(false)
  })

  it('比较在途时改变知识截止日，迟到结果不恢复旧候选', async () => {
    let resolve!: (value: Response) => void
    install({ [`${root}/policy/preview`]: () => new Promise(done => { resolve = done }) })
    const user = userEvent.setup(); const view = renderPolicy(); await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
    researchClock.day = '2026-09-11'
    view.rerender(policyTree())
    await act(async () => resolve(await response(policyPreview)))
    expect(screen.queryByRole('table', { name: '长期政策候选比较' })).not.toBeInTheDocument()
    expect(screen.queryByRole('button', { name: '复核此候选' })).not.toBeInTheDocument()
  })

  it('已保存政策在知识截止日变化后继续只读展示，不额外保存', async () => {
    const fetch = install(); const user = userEvent.setup(); const view = renderPolicy(); await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
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
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/pre-investment/ltcma/new?copy=cma-1']}><LtcmaWorkspace /></MemoryRouter>)
    await screen.findByLabelText('名称')
    await user.click(screen.getByRole('checkbox', { name: /我已核对资产范围/ }))
    await user.click(screen.getByRole('button', { name: '计算预览' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('半正定')
    expect(fetch.mock.calls.filter(([url]) => String(url) === `${root}/cma`)).toHaveLength(0)
  })

  it('目标保存中修改输入，不把迟到的旧版本当成当前目标', async () => {
    researchClock.day = '2026-09-17'
    let resolve!: (value: Response) => void
    install({ [`${root}/mandates/confirm`]: () => new Promise(done => { resolve = done }) })
    writeAllocationDraft('mandate-study:editor', boundaryStudy())
    const user = userEvent.setup(); render(<MemoryRouter><InvestmentObjectivesWorkspace /></MemoryRouter>)
    await user.click(screen.getByRole('button', { name: '2. 结果与确认' }))
    await user.click(screen.getByRole('button', { name: '运行目标诊断' }))
    await screen.findByText('当前约束下可实现')
    await user.click(screen.getByRole('checkbox', { name: /我已核对输入/ }))
    await user.click(screen.getByRole('button', { name: '保存新目标版本' }))
    await user.click(screen.getByRole('button', { name: '1. 目标与约束' }))
    fireEvent.change(screen.getByLabelText('目标名称'), { target: { value: '修改后的目标' } })
    await act(async () => resolve(await response(mandateVersion, 201)))
    expect(screen.queryByText('只读版本')).not.toBeInTheDocument()
    expect(screen.getByLabelText('目标名称')).toHaveValue('修改后的目标')
  })

  it('资金目标未达门槛时可以复核，但填写理由也不能采用', async () => {
    const failed = assessment({ ...fundingStudy, cma_id: 'cma-1' })
    const fetch = install({ [`${root}/policy/preview`]: () => response({ ...policyPreview, candidates: failed.candidates, mandate: failed.definition, funding: failed.funding, funding_execution: failed.funding_execution }) })
    const user = userEvent.setup(); renderPolicy(); await loadCma(user)
    await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
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


it('未知时钟禁止SAA比较，明确无PIT允许比较，重新未知后丢弃候选', async () => {
  researchClock.day = undefined
  const fetch = install(); const user = userEvent.setup(); const view = renderPolicy()
  await loadCma(user)
  expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeDisabled()
  expect(screen.getByRole('alert')).toHaveTextContent('平台知识截止日尚未确认')
  researchClock.day = null
  view.rerender(policyTree())
  await waitFor(() => expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: '比较符合目标的政策候选' }))
  await user.click(await screen.findByRole('button', { name: '复核此候选' }))
  researchClock.day = undefined
  view.rerender(policyTree())
  expect(screen.queryByRole('button', { name: '确认采用此长期政策' })).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: '比较符合目标的政策候选' })).toBeDisabled()
  expect(fetch.mock.calls.some(([url]) => String(url) === `${root}/policies`)).toBe(false)
})

it('从TAA返回精确的已保存SAA版本，不依赖其他目标的本地草稿', async () => {
  const fetch = install({ '/api/tactical-allocation/baselines/POLICY-1': () => response(policyBaseline) })
  const path = allocationJourneyPath('saa', { allocationName: '股债分类', baselineId: 'POLICY-1', universeId: 'one' })
  render(<MemoryRouter initialEntries={[path]}><StrategicAllocationWorkspace /></MemoryRouter>)
  expect(await screen.findByRole('heading', { name: policyBaseline.name })).toBeInTheDocument()
  expect(screen.getByText(policyBaseline.policy.reason)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: '使用此目标建立新政策' }).getAttribute('href')).toContain('mandate=mandate-1')
  expect(screen.getByRole('link', { name: '返回此政策的 TAA 研究' })).toHaveAttribute('href', '/pre-investment/taa?baseline=POLICY-1')
  expect(fetch.mock.calls.every(([, init]) => !init?.method || init.method === 'GET')).toBe(true)
  expect(screen.queryByRole('button', { name: '确认采用此长期政策' })).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: '← 返回 SAA 方案列表' })).toHaveAttribute('href', '/pre-investment/saa')
})

it('new-plan URLs isolate drafts while refreshing the same URL restores its inputs', async () => {
  install()
  const draft = { mandateId: 'mandate-1', allocationName: '股债分类', strategicUniverseId: '', implementationMappingId: '', savedCmaId: null,
    settings: { constraints: {}, group_limits: [], uncertainty_penalty: 1, candidate_count: 2000, seed: 42 }, policyName: '未完成方案', reason: '' }
  writeAllocationDraft('strategic-policy::', draft)
  writeAllocationDraft('strategic-policy:::new:first', draft)
  const first = render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?new=first']}><StrategicAllocationWorkspace /></MemoryRouter>)
  expect(await screen.findByLabelText('投资目标版本')).toHaveValue('mandate-1')
  first.unmount()
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?new=second']}><StrategicAllocationWorkspace /></MemoryRouter>)
  expect(await screen.findByLabelText('投资目标版本')).toHaveValue('')
})

it('SAA返回链接不会把另一个版本的响应当成所选政策', async () => {
  install({ '/api/tactical-allocation/baselines/POLICY-1': () => response({ ...policyBaseline, id: 'other' }) })
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?baseline=POLICY-1']}><StrategicAllocationWorkspace /></MemoryRouter>)
  expect(await screen.findByRole('alert')).toHaveTextContent('读取的政策与所选版本不一致')
  expect(screen.queryByRole('link', { name: '返回此政策的 TAA 研究' })).not.toBeInTheDocument()
})

it('未映射产品的已确认 SAA 仍可进入大类 TAA 研究', async () => {
  install({ '/api/tactical-allocation/baselines/POLICY-1': () => response({
    ...policyBaseline, alloc_name: null, strategic_universe_id: 'scope-no-products',
    implementation_status: 'incomplete', implementation_mapping_id: null,
    assets: policyBaseline.assets.map(asset => ({ ...asset, products: [] })),
  }) })
  render(<MemoryRouter initialEntries={['/pre-investment/saa/policy?baseline=POLICY-1']}><StrategicAllocationWorkspace /></MemoryRouter>)
  expect(await screen.findByRole('link', { name: '返回此政策的 TAA 研究' })).toHaveAttribute('href', '/pre-investment/taa?baseline=POLICY-1')
  expect(screen.queryByText(/缺少真实代理产品|尚未关联实际投资产品|映射未完成/)).not.toBeInTheDocument()
})

// The SAA entry has one scope selector; product links are a separate later step.
const scopeUniverse = {
  id: 'scope-human', name: '权益固收现金', created_at: '2026-09-12', content_hash: 'u'.repeat(64),
  definition: { name: '权益固收现金', as_of: '2026-09-12', currency: 'CNY', source: '', assets: [
    { id: 'equity', name: '权益', currency: 'CNY', role: 'growth' as const, liquidity: 'liquid' as const, rationale: '', source: '' },
    { id: 'bond', name: '固收', currency: 'CNY', role: 'rates' as const, liquidity: 'liquid' as const, rationale: '', source: '' },
    { id: 'cash', name: '现金', currency: 'CNY', role: 'liquidity' as const, liquidity: 'liquid' as const, rationale: '', source: '' },
  ] },
}
const productMap = {
  id: 'map-partial', name: '已配置权益产品', created_at: '2026-09-12', content_hash: 'p'.repeat(64),
  definition: { name: '已配置权益产品', strategic_universe_id: scopeUniverse.id, universe_snapshot_id: 'products', alloc_name: '股债分类', as_of: '2026-09-12', valid_until: '2027-09-12', assignments: [{ strategic_asset_id: 'equity', proxy_asset_id: 'equity', rationale: '' }] },
  implementation_status: 'incomplete' as const, implementation_gaps: ['bond', 'cash'],
}
function renderScope(mapping = '') {
  return render(<MemoryRouter initialEntries={[`/pre-investment/saa/policy?strategic_universe=${scopeUniverse.id}&mandate=mandate-1${mapping ? `&mapping=${mapping}` : ''}`]}><StrategicAllocationWorkspace /></MemoryRouter>)
}

it('unifies both research paths and clears incompatible assumptions and product links on scope change', async () => {
  install({ [`${root}/catalog`]: () => response({ ...strategicCatalog, strategic_universes: [scopeUniverse], implementation_maps: [productMap] }) })
  const user = userEvent.setup(); renderPolicy(); await loadCma(user)
  await user.click(screen.getByRole('button', { name: '1. 研究范围' }))
  const scope = screen.getByLabelText('研究范围')
  expect(scope).toHaveValue('allocation:股债分类')
  expect(screen.queryByLabelText('或选择独立战略范围')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('已保存的大类配置')).not.toBeInTheDocument()
  await user.selectOptions(scope, `strategic:${scopeUniverse.id}`)
  expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue('')
  expect(screen.getByRole('button', { name: '3. 政策比较' })).toBeDisabled()
  expect(screen.getByText('大类：权益、固收、现金')).toBeInTheDocument()
  expect(screen.queryByLabelText('使用已保存的产品配置')).not.toBeInTheDocument()
  expect(screen.queryByText(/待关联|尚未关联实际投资产品/)).not.toBeInTheDocument()
  await user.selectOptions(scope, 'allocation:股债分类')
  await user.selectOptions(scope, `strategic:${scopeUniverse.id}`)
  expect(screen.getByLabelText('选择已确认 LTCMA')).toHaveValue('')
})

it.each(['', productMap.id])('does not require products for SAA, including an existing mapping reference %s', async mapping => {
  install({ [`${root}/catalog`]: () => response({ ...strategicCatalog, strategic_universes: [scopeUniverse], implementation_maps: [productMap] }) })
  renderScope(mapping)
  await screen.findByText('大类：权益、固收、现金')
  expect(screen.getByText(/SAA 确定长期大类权重，无需配置实际投资产品/)).toBeVisible()
  expect(screen.getByRole('button', { name: '选择已确认 LTCMA' })).toBeEnabled()
  expect(screen.queryByLabelText('使用已保存的产品配置')).not.toBeInTheDocument()
  expect(screen.queryByRole('link', { name: '前往此范围配置实际产品' })).not.toBeInTheDocument()
  expect(screen.queryByText(/待关联|尚未关联实际投资产品/)).not.toBeInTheDocument()
})

it('disables policy comparison beside a clear reason until changed constraints pass the frontier check', async () => {
  let feasible = false
  const fetch = install({ [`${root}/policy/frontier`]: () => response({
    mode: 'single', execution: policyPreview.execution, additional_checks: {},
    target_check: { status: feasible ? 'feasible' : 'infeasible', reason: feasible ? null : 'target_outside' },
    views: [{ id: 'cma-1', name: '测试 CMA', target_return: .0772, volatility_cap: .095, cash_floor: .1,
      reference: { complete: true, points: [] }, configured: { complete: true, points: [] } }],
  }) })
  const user = userEvent.setup(); renderPolicy(); await loadCma(user)
  const button = screen.getByRole('button', { name: '比较符合目标的政策候选' })
  await waitFor(() => expect(button).toHaveAccessibleDescription(/无法同时满足年收益至少 7.72%/))
  expect(button).toBeDisabled()
  await user.click(button)
  expect(fetch.mock.calls.some(([url]) => String(url) === `${root}/policy/preview`)).toBe(false)
  feasible = true
  fireEvent.change(screen.getByLabelText('权益最高权重'), { target: { value: '90' } })
  expect(button).toBeDisabled()
  await waitFor(() => expect(button).toBeEnabled())
  await user.click(button)
  await screen.findByRole('button', { name: '复核此候选' })
})
