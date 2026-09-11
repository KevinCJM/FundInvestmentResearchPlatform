import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import TacticalAllocationWorkspace from './TacticalAllocationWorkspace'
import * as ResearchContext from '../app/ResearchContext'
import { readAllocationJourney, updateAllocationJourney } from '../app/allocationJourney'
import { taaBaseline, taaCatalog, taaExecution, taaPreview, taaPreflight } from '../test/tacticalAllocationFixtures'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="taa-chart">扣费净值对照图</div> }))
function ok(body: unknown) { return { ok: true, status: 200, json: async () => body } as Response }
function setup(handler?: (path: string, body: any) => Promise<Response> | Response | undefined, entry = '/pre-investment/taa') {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input); const body = init?.body ? JSON.parse(String(init.body)) : null
    const handled = handler?.(path, body); if (handled) return handled
    if (path.endsWith('/catalog')) return ok(taaCatalog)
    if (path.endsWith('/baselines/SAA-1')) return ok(taaBaseline)
    if (path.endsWith('/preflight')) return ok(taaPreflight)
    if (path.endsWith('/preview')) return ok({ ...taaPreview, request: body })
    throw new Error(`Unexpected request ${path}`)
  })
  vi.stubGlobal('fetch', fetchMock)
  const tree = () => <MemoryRouter initialEntries={[entry]}><Routes><Route path="/pre-investment/taa" element={<TacticalAllocationWorkspace />} /><Route path="/pre-investment/product-allocation-timing/construction" element={<p>产品配置已打开</p>} /></Routes></MemoryRouter>
  const view = render(tree())
  const rawUser = userEvent.setup()
  const user = new Proxy(rawUser, { get(target, key: keyof typeof rawUser) { const method = target[key]; return typeof method === 'function' ? (...args: unknown[]) => act(async () => { await (method as (...values: unknown[]) => Promise<unknown>)(...args) }) : method } })
  return { user, fetchMock, view, rerender: () => view.rerender(tree()) }
}
async function calculate(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByText('本次准备怎么配？')
  await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
  await screen.findByRole('region', { name: 'SAA 与战术方案对照' })
}
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks(); sessionStorage.clear(); localStorage.clear() })

describe('TacticalAllocationWorkspace', () => {
  it('显示实际有效信号期数，零信号不能搜索且固定假设必须明确选择', async () => {
    const { user, fetchMock } = setup((path, body) => path.endsWith('/preflight') ? ok({ ...taaPreflight,
      can_calculate: !body.search,
      training: { ...taaPreflight.training, eligible: false, train_signal_observations: 0, validation_signal_observations: 50 },
      guidance: [{ code: 'no-signals', action: 'fixed_comparison', message: '各档强度没有形成可用比较。' }],
    }) : undefined)
    await screen.findByText(/有效趋势信号：训练 0 \/ 540 期，验证 50 \/ 260 期/)
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preview'))).toBe(false)
    await user.click(screen.getByRole('button', { name: '改为固定假设比较' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
    await screen.findByRole('region', { name: 'SAA 与战术方案对照' })
    const sent = JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/preview'))?.[1]?.body))
    expect(sent.search).toBe(false)
  })

  it('旧冻结结果不使用新版预检的有效信号期数冒充历史证据', async () => {
    setup(path => path.endsWith('/decisions/OLD') ? ok({ id: 'OLD', name: '旧口径', created_at: '2026-09-10', preview: taaPreview, scenarios: [] })
      : path.endsWith('/preflight') ? ok({ ...taaPreflight, training: { ...taaPreflight.training, train_signal_observations: 411, validation_signal_observations: 230 } }) : undefined,
    '/pre-investment/taa?decision=OLD')
    await screen.findByText('此结果未记录有效信号期数；重新计算会采用当前趋势口径。')
    expect(screen.queryByText(/有效趋势信号：训练 411/)).not.toBeInTheDocument()
  })

  it('修改 TAA 决策日期不会冒充上游范围研究日', async () => {
    updateAllocationJourney({ universeId: taaBaseline.universe_snapshot_id!, allocationName: taaBaseline.alloc_name, baselineId: taaBaseline.id, researchDate: '2026-09-10' })
    const { user } = setup()
    await screen.findByText('本次准备怎么配？')
    await user.click(screen.getByRole('button', { name: '修改日期与费用' }))
    await user.clear(screen.getByLabelText(/决策日期/))
    await user.type(screen.getByLabelText(/决策日期/), '2026-09-09')
    expect(screen.getByLabelText(/决策日期/)).toHaveValue('2026-09-09')
    expect(readAllocationJourney().researchDate).toBe('2026-09-10')
  })

  it('初始 PIT 日期补齐不会取消正在读取的历史版本，冲突版本只读且禁止交接', async () => {
    const platform = vi.spyOn(ResearchContext, 'useResearchDay').mockReturnValue(null)
    let resolveDecision!: (response: Response) => void
    const pending = new Promise<Response>(resolve => { resolveDecision = resolve })
    const { fetchMock, rerender } = setup(path => path.endsWith('/decisions/HYDRATING') ? pending : undefined,
      '/pre-investment/taa?decision=HYDRATING')
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/decisions/HYDRATING'))).toBe(true))
    platform.mockReturnValue('2014-12-31')
    rerender()
    expect(screen.queryByText('平台数据口径已变化。草稿输入和情景假设仍保留，请检查日期并重新计算。')).not.toBeInTheDocument()
    await act(async () => { resolveDecision(ok({ id: 'HYDRATING', name: '延迟到达的历史研究', created_at: '2026-09-10', preview: taaPreview, scenarios: [] })) })
    await screen.findByRole('button', { name: '当前研究版本已保存' })
    expect(screen.getByRole('region', { name: '平台 PIT 日期冲突' })).toHaveTextContent('2026-09-10 晚于当前 PIT 截止 2014-12-31')
    expect(screen.getByRole('button', { name: '重新计算研究' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
    expect(fetchMock.mock.calls.filter(([path]) => String(path).endsWith('/decisions/HYDRATING'))).toHaveLength(1)
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preflight'))).toBe(false)
  })

  it('已有研究输入后真正切换平台 PIT 仍使预览失效并保留草稿', async () => {
    const platform = vi.spyOn(ResearchContext, 'useResearchDay').mockReturnValue(null)
    const { user, rerender } = setup()
    await calculate(user)
    platform.mockReturnValue('2014-12-31')
    rerender()
    await screen.findByText('平台数据口径已变化。草稿输入和情景假设仍保留，请检查日期并重新计算。')
    expect(screen.getByRole('region', { name: '平台 PIT 日期冲突' })).toHaveTextContent('2026-09-10 晚于当前 PIT 截止 2014-12-31')
    expect(screen.queryByRole('region', { name: 'SAA 与战术方案对照' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
  })

  it('切到更早平台 PIT 后保留旧草稿，但禁止以冲突日期继续计算', async () => {
    const first = setup()
    await screen.findByText('本次准备怎么配？')
    first.view.unmount()
    vi.spyOn(ResearchContext, 'useResearchDay').mockReturnValue('2014-12-31')
    const second = setup()
    await screen.findByRole('region', { name: '平台 PIT 日期冲突' })
    expect(screen.getByRole('region', { name: '平台 PIT 日期冲突' })).toHaveTextContent('2026-09-10 晚于当前 PIT 截止 2014-12-31')
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: '按当前 PIT 调整日期' })).not.toBeInTheDocument()
    expect(second.fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preflight'))).toBe(false)
  })

  it('平台 PIT 与旧日期存在共同覆盖时，明确点击后才调整训练验证日期', async () => {
    const first = setup()
    await screen.findByText('本次准备怎么配？')
    first.view.unmount()
    vi.spyOn(ResearchContext, 'useResearchDay').mockReturnValue('2025-03-31')
    const second = setup()
    await screen.findByRole('button', { name: '按当前 PIT 调整日期' })
    expect(second.fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preflight'))).toBe(false)
    await second.user.click(screen.getByRole('button', { name: '按当前 PIT 调整日期' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    const request = JSON.parse(String(second.fetchMock.mock.calls.find(([path]) => String(path).endsWith('/preflight'))?.[1]?.body))
    expect(request.as_of).toBe('2025-03-31')
    expect(request.end_date).toBe('2025-03-31')
    expect(request.start_date < request.train_end_date && request.train_end_date < request.end_date).toBe(true)
  })

  it('平台 PIT 冲突时仍能读取保存版本，保持历史口径并阻止交接', async () => {
    vi.spyOn(ResearchContext, 'useResearchDay').mockReturnValue('2014-12-31')
    setup(path => path.endsWith('/decisions/D-READ') ? ok({ id: 'D-READ', name: '只读历史研究', created_at: '2026-09-10', preview: taaPreview, scenarios: [] }) : undefined, '/pre-investment/taa?decision=D-READ')
    await screen.findByRole('button', { name: '当前研究版本已保存' })
    expect(screen.getByRole('region', { name: '平台 PIT 日期冲突' })).toHaveTextContent('已保存版本仅供查看')
    expect(screen.getByRole('button', { name: '重新计算研究' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
  })

  it('首屏先显示日期费用与训练资格，显式改为固定比较才解除搜索门禁', async () => {
    const { user, fetchMock } = setup((path, body) => path.endsWith('/preflight') ? ok({ ...taaPreflight, can_calculate: !body.search, training: { ...taaPreflight.training, eligible: false, unavailable_count: 120, earliest_available_date: '2026-09-10' }, guidance: [{ code: 'late-labels', message: '训练收益在训练截止后才可得。固定假设比较不代表已完成选优。', action: 'fixed_comparison' }] }) : undefined)
    await screen.findByText(/训练截止时尚不可得 120 条/)
    expect(screen.getByRole('region', { name: '本次研究条件与资格' })).toHaveTextContent('10 基点（0.10%）')
    expect(screen.getByRole('region', { name: '本次研究条件与资格' })).toHaveTextContent('SAA 方案日期')
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preview'))).toBe(false)
    await user.click(screen.getByRole('button', { name: '改为固定假设比较' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
    await screen.findByRole('region', { name: 'SAA 与战术方案对照' })
    const body = JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/preview'))?.[1]?.body))
    expect(body.search).toBe(false)
    expect(body.train_end_date).toBe(JSON.parse(String(fetchMock.mock.calls.find(([path]) => String(path).endsWith('/preflight'))?.[1]?.body)).train_end_date)
  })

  it('数量级跳变预检直接指出资产日期并阻止计算', async () => {
    const { fetchMock } = setup(path => path.endsWith('/preflight') ? ok({ ...taaPreflight, can_calculate: false, quality: { status: 'blocked', issues: [{ asset_id: 'bond', date: '2026-06-02', value: -.99, code: 'scale-break', message: '净值约缩至原来的百分之一。' }] }, guidance: [{ code: 'quality', action: 'review_data', message: '请检查产品数据，不能用于配置判断。' }] }) : undefined)
    await screen.findByText(/债券 · 2026-06-02/)
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    expect(screen.getByRole('link', { name: '返回大类检查产品' })).toHaveAttribute('href', '/pre-investment/saa/asset-classes?universe=UNIVERSE-1')
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/preview'))).toBe(false)
  })

  it('离开后恢复本基准的输入草稿，计算结果不写入浏览器缓存', async () => {
    const first = setup()
    await screen.findByText('本次准备怎么配？')
    await first.user.click(screen.getByRole('radio', { name: /研究员观点/ }))
    await first.user.clear(screen.getByRole('spinbutton', { name: /权益偏离/ })); await first.user.type(screen.getByRole('spinbutton', { name: /权益偏离/ }), '5')
    await first.user.clear(screen.getByRole('spinbutton', { name: /债券偏离/ })); await first.user.type(screen.getByRole('spinbutton', { name: /债券偏离/ }), '-5')
    first.view.unmount()
    setup()
    await screen.findByText(/已恢复本次研究草稿/)
    expect(screen.getByRole('spinbutton', { name: /权益偏离/ })).toHaveValue('5')
    expect(screen.getByRole('spinbutton', { name: /债券偏离/ })).toHaveValue('-5')
    const stored = JSON.parse(localStorage.getItem('allocation-draft:v1:taa:SAA-1')!)
    expect(stored).not.toHaveProperty('preview')
    expect(stored).not.toHaveProperty('result')
    expect(screen.queryByTestId('taa-chart')).not.toBeInTheDocument()
  })

  it('保存期间继续修改结论，迟到的保存结果不能覆盖新草稿', async () => {
    let resolve!: (value: Response) => void
    const { user } = setup(path => path.endsWith('/decisions') ? new Promise<Response>(done => { resolve = done }) : undefined)
    await calculate(user); await user.click(screen.getByRole('tab', { name: '版本与审计' }))
    await user.click(screen.getByRole('button', { name: '保存研究版本' }))
    await user.type(screen.getByLabelText(/研究结论与复核计划/), '新增判断：先观察。')
    await act(async () => { resolve(ok({ id: 'D-STALE', name: '此前判断', created_at: '2026-09-11', preview: taaPreview, scenarios: [] })); await Promise.resolve() })
    expect(screen.getByLabelText(/研究结论与复核计划/)).toHaveValue('新增判断：先观察。')
    expect(screen.getByRole('button', { name: '保存研究版本' })).toBeEnabled()
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
  })

  it('预检格式缺失时显示可重试错误并保持计算关闭', async () => {
    const { user } = setup(path => path.endsWith('/preflight') ? ok({ can_calculate: true }) : undefined)
    await screen.findByText(/研究条件检查结果不完整/)
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '重试条件检查' }))
    await screen.findByText(/研究条件检查结果不完整/)
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
  })

  it('多个具名情景随版本保存恢复，历史默认范围有效，收益差显示百分点', async () => {
    let saved: any
    const handler = (path: string, body: any) => {
      const result = (scenario: any) => ({ name: scenario.name, kind: scenario.kind, baseline_return: -.06, taa_return: -.065, excess_return: -.005, contributions: [], warnings: [], execution: taaExecution })
      if (path.endsWith('/scenarios')) return ok(result(body.scenario))
      if (path.endsWith('/decisions')) { saved = { id: 'D-SCENARIOS', name: body.name, created_at: '2026-09-11', note: body.note, preview: { ...taaPreview, request: body.request }, scenarios: body.scenarios.map((scenario: any) => ({ scenario, result: result(scenario) })) }; return ok(saved) }
      if (path.endsWith('/decisions/D-SCENARIOS')) return ok(saved)
    }
    const first = setup(handler)
    await calculate(first.user); await first.user.click(screen.getByRole('tab', { name: '情景模拟' }))
    await first.user.clear(screen.getByRole('spinbutton', { name: '权益假设涨跌（%）' })); await first.user.type(screen.getByRole('spinbutton', { name: '权益假设涨跌（%）' }), '-10')
    await first.user.click(screen.getByRole('button', { name: '计算情景影响' }))
    await screen.findByRole('region', { name: '情景模拟结果' })
    expect(screen.getByRole('region', { name: '情景实验对照' })).toHaveTextContent('-0.50 个百分点')
    await first.user.selectOptions(screen.getByLabelText('情景方式'), 'historical')
    expect(screen.getByLabelText('历史情景结束')).toHaveValue('2026-09-10')
    expect(screen.getByLabelText('历史情景开始')).toHaveValue('2023-01-03')
    await first.user.click(screen.getByRole('button', { name: '计算情景影响' }))
    await waitFor(() => expect(screen.getByRole('region', { name: '情景实验对照' })).toHaveTextContent('已保留的情景 · 2/12'))
    await first.user.click(screen.getByRole('tab', { name: '版本与审计' }))
    await first.user.type(screen.getByLabelText(/研究结论与复核计划/), '暂不采纳，季度末复核。')
    await first.user.click(screen.getByRole('button', { name: '保存研究版本' }))
    await screen.findByRole('button', { name: '当前研究版本已保存' })
    expect(saved.scenarios).toHaveLength(2)
    expect(saved.scenarios[0].scenario.shocks.equity).toBe(-.1)
    first.view.unmount()
    const second = setup(handler, '/pre-investment/taa?decision=D-SCENARIOS')
    await screen.findByRole('button', { name: '当前研究版本已保存' })
    expect(screen.getByLabelText(/研究结论与复核计划/)).toHaveValue('暂不采纳，季度末复核。')
    await second.user.click(screen.getByRole('tab', { name: '情景模拟' }))
    expect(screen.getByRole('region', { name: '情景实验对照' })).toHaveTextContent('已保留的情景 · 2/12')
    await second.user.click(screen.getByRole('button', { name: '移除情景 历史区间回放' }))
    expect(screen.getByRole('region', { name: '情景实验对照' })).toHaveTextContent('已保留的情景 · 1/12')
    expect(saved.scenarios).toHaveLength(2)
  })

  it('从真实SAA版本开始，无JSON表单，区分训练、验证与PIT边界', async () => {
    const { user, fetchMock } = setup()
    await calculate(user)
    expect(screen.getByRole('heading', { name: '战术资产配置' })).toBeInTheDocument()
    expect(screen.getAllByText('1.30%').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByText('+5.00 个百分点')).toBeInTheDocument()
    expect(screen.getAllByText('未提供参考持仓').length).toBeGreaterThanOrEqual(2)
    expect(screen.getByTestId('taa-chart')).toBeInTheDocument()
    expect(screen.getByText(/保持 SAA 也是一个方案/)).toBeInTheDocument()
    const payload = fetchMock.mock.calls.find(([url]) => String(url).endsWith('/preview'))?.[1]
    expect(JSON.parse(String(payload?.body))).toMatchObject({ baseline_id: 'SAA-1', signal_mode: 'momentum', start_date: '2023-01-03', end_date: '2026-09-10' })
    await user.click(screen.getByRole('tab', { name: '版本与审计' }))
    expect(screen.getAllByText('当前修订数据尚未证明完整历史 PIT。').length).toBeGreaterThan(0)
    expect(screen.queryByRole('textbox', { name: 'SAA 基准权重' })).not.toBeInTheDocument()
  })

  it('新建基准必须显式填写权重并保存，不自动生成SAA', async () => {
    const { user, fetchMock } = setup((path, body) => path.endsWith('/catalog') ? ok({ ...taaCatalog, baselines: [] }) : path.endsWith('/baselines') ? ok({ ...taaBaseline, ...body, id: 'SAA-1' }) : undefined)
    await screen.findByText('先确定长期配置，再讨论偏离')
    expect(fetchMock.mock.calls.some(([, init]) => init?.method === 'POST')).toBe(false)
    await user.click(screen.getByRole('button', { name: '新建研究基准' }))
    await user.selectOptions(screen.getByLabelText('资产分类方案'), '股债分类')
    expect(screen.getByRole('button', { name: '保存并使用此基准' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '以等权起步' }))
    await user.click(screen.getByRole('button', { name: '保存并使用此基准' }))
    await screen.findByText('本次准备怎么配？')
    const call = fetchMock.mock.calls.find(([path, init]) => String(path).endsWith('/baselines') && init?.method === 'POST')
    expect(JSON.parse(String(call?.[1]?.body)).weights).toEqual({ equity: .5, bond: .5 })
  })

  it('负偏离按百分点输入，资金不守恒时阻止提交', async () => {
    const { user, fetchMock } = setup()
    await screen.findByText('本次准备怎么配？')
    await user.click(screen.getByRole('radio', { name: /研究员观点/ }))
    const equity = screen.getByRole('spinbutton', { name: /权益偏离/ }); const bond = screen.getByRole('spinbutton', { name: /债券偏离/ })
    await user.clear(equity); await user.type(equity, '5')
    expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeDisabled()
    await user.clear(bond); await user.type(bond, '-5')
    expect(bond).toHaveValue('-5')
    await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
    await screen.findByRole('region', { name: 'SAA 与战术方案对照' })
    const call = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/preview'))
    expect(JSON.parse(String(call?.[1]?.body)).manual_tilts).toEqual({ equity: .05, bond: -.05 })
  })

  it('参数修改立即失效旧结果，失败请求不保留旧数值', async () => {
    let calls = 0
    const { user } = setup((path, body) => path.endsWith('/preview') ? ++calls === 1 ? ok({ ...taaPreview, request: body }) : { ok: false, status: 422, json: async () => ({ detail: { message: '缺少共同数据，无法回测。' } }) } as Response : undefined)
    await calculate(user)
    await user.clear(screen.getByRole('spinbutton', { name: /单边成本/ })); await user.type(screen.getByRole('spinbutton', { name: /单边成本/ }), '15')
    expect(screen.queryByRole('region', { name: 'SAA 与战术方案对照' })).not.toBeInTheDocument()
    await waitFor(() => expect(screen.getByRole('button', { name: '运行回测与候选比较' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '运行回测与候选比较' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('缺少共同数据')
    expect(screen.queryByTestId('taa-chart')).not.toBeInTheDocument()
  })

  it('修改条件后迟到的旧响应不能回填', async () => {
    let resolve!: (response: Response) => void
    const { user } = setup(path => path.endsWith('/preview') ? new Promise<Response>(done => { resolve = done }) : undefined)
    await screen.findByText('本次准备怎么配？')
    await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
    await user.clear(screen.getByRole('spinbutton', { name: '趋势观察窗口（交易期）' }))
    await user.type(screen.getByRole('spinbutton', { name: '趋势观察窗口（交易期）' }), '90')
    await act(async () => { resolve(ok(taaPreview)); await Promise.resolve() })
    expect(screen.queryByTestId('taa-chart')).not.toBeInTheDocument()
    expect(screen.getAllByText('待计算')).toHaveLength(4)
  })

  it('情景负冲击正确传输，编辑新假设仍保留已完成实验', async () => {
    const { user, fetchMock } = setup(path => path.endsWith('/scenarios') ? ok({ name: '自定义资产冲击', kind: 'shock', baseline_return: -.06, taa_return: -.065, excess_return: -.005, contributions: [], warnings: [], execution: taaExecution }) : undefined)
    await calculate(user); await user.click(screen.getByRole('tab', { name: '情景模拟' }))
    const input = screen.getByRole('spinbutton', { name: '权益假设涨跌（%）' })
    await user.clear(input); await user.type(input, '-10'); await user.click(screen.getByRole('button', { name: '计算情景影响' }))
    await screen.findByRole('region', { name: '情景模拟结果' })
    const call = fetchMock.mock.calls.find(([path]) => String(path).endsWith('/scenarios'))
    expect(JSON.parse(String(call?.[1]?.body)).scenario.shocks).toEqual({ equity: -.1, bond: 0 })
    await user.clear(input)
    expect(screen.getByRole('region', { name: '情景模拟结果' })).toBeInTheDocument()
  })

  it('保存后才带入产品配置，使用服务端生成的预算与产品映射', async () => {
    const transfer = { name: '研究', method: 'manual', allocation_source: { kind: 'taa', decision_id: 'D-1', baseline_id: 'SAA-1', class_weights: { equity: .65, bond: .35 } }, constituents: [{ kind: 'etf', product_id: '510300.SH', weight: 65 }] }
    const { user } = setup((path, body) => path.endsWith('/decisions') ? ok({ id: 'D-1', name: body.name, created_at: '2026-09-10', preview: { ...taaPreview, request: body.request } }) : path.endsWith('/product-allocation') ? ok(transfer) : undefined)
    await calculate(user); await user.click(screen.getByRole('tab', { name: '版本与审计' }))
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '保存研究版本' }))
    await waitFor(() => expect(screen.getByRole('button', { name: '带入产品配置' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '带入产品配置' }))
    expect(await screen.findByText('产品配置已打开')).toBeInTheDocument()
    expect(JSON.parse(sessionStorage.getItem('portfolioResearchImport')!)).toEqual(transfer)
  })

  it('从decision链接查看历史版本，过期决策不能带入下游', async () => {
    setup(path => path.endsWith('/decisions/D-OLD') ? ok({ id: 'D-OLD', name: '历史观点', created_at: '2024-01-01', preview: { ...taaPreview, recommendation: { ...taaPreview.recommendation, expires_on: '2024-02-01' } } }) : undefined, '/pre-investment/taa?decision=D-OLD')
    expect(await screen.findByText(/这个决策已到复核日期/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '当前研究版本已保存' })).toBeDisabled()
  })

  it('无固定签名NJIT证明时不显示结果', async () => {
    const { user } = setup((path, body) => path.endsWith('/preview') ? ok({ ...taaPreview, request: body, execution: { ...taaExecution, python_fallback: 1 } }) : undefined)
    await screen.findByText('本次准备怎么配？'); await waitFor(() => expect(screen.getByRole('button', { name: '计算并比较方案' })).toBeEnabled())
    await user.click(screen.getByRole('button', { name: '计算并比较方案' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('固定签名 NJIT 执行证明')
    expect(screen.queryByTestId('taa-chart')).not.toBeInTheDocument()
  })
  it('格式不完整的目录显示可操作错误，不崩溃', async () => {
    setup(path => path.endsWith('/catalog') ? ok({ items: [] }) : undefined)
    expect(await screen.findByRole('alert')).toHaveTextContent('资产配置目录格式不完整')
  })

  it('迟到的decision链接响应不能覆盖用户新选基准与编辑', async () => {
    let resolve!: (response: Response) => void
    const { user } = setup(path => path.endsWith('/decisions/D-LATE') ? new Promise<Response>(done => { resolve = done }) : undefined, '/pre-investment/taa?decision=D-LATE')
    await screen.findByRole('option', { name: /稳健长期组合/ })
    await user.selectOptions(screen.getByLabelText('SAA 基准版本'), 'SAA-1')
    await screen.findByText('本次准备怎么配？')
    const input = screen.getByRole('spinbutton', { name: '趋势观察窗口（交易期）' })
    await user.clear(input); await user.type(input, '90')
    await act(async () => { resolve(ok({ id: 'D-LATE', name: '迟到版本', created_at: '2024-01-01', preview: taaPreview })); await Promise.resolve() })
    expect(input).toHaveValue('90')
    expect(screen.queryByRole('button', { name: '当前研究版本已保存' })).not.toBeInTheDocument()
  })

  it('即使强度非零，信号回归SAA也显示维持长期配置，并阻断验证超限交接', async () => {
    const result = { ...taaPreview, recommendation: { ...taaPreview.recommendation, is_saa: true, weights: { equity: .6, bond: .4 }, tilts: { equity: 0, bond: 0 } }, candidates: taaPreview.candidates.map(item => ({ ...item, validation_feasible: false })) }
    const { user } = setup((path, body) => path.endsWith('/preview') ? ok({ ...result, request: body }) : path.endsWith('/decisions') ? ok({ id: 'D-1', name: body.name, created_at: '2026-09-11', preview: result }) : undefined)
    await calculate(user)
    expect(screen.getByRole('heading', { name: '维持长期配置' })).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: '版本与审计' }))
    await user.click(screen.getByRole('button', { name: '保存研究版本' }))
    await screen.findByRole('button', { name: '当前研究版本已保存' })
    expect(screen.getByRole('button', { name: '带入产品配置' })).toBeDisabled()
    expect(screen.getByText('所选候选在独立验证区间超出约束，请先复核。')).toBeInTheDocument()
  })

})
