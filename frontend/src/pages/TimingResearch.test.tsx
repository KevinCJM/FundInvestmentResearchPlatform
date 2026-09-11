import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import TimingResearch from './TimingResearch'
import { timingCatalogFixture, timingExecutionFixture, timingRunFixture, timingTrainingCatalogFixture } from '../components/timing-research/timingFixtures'
import { editableTimingDefinition, timingApi } from '../services/timingResearch'
import { timingTrainingError } from '../components/timing-research/TimingTrainingEditor'
import { basketValidation } from '../components/timing-research/TimingStudyContext'
import TimingTrainingAudit from '../components/timing-research/TimingTrainingAudit'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="timing-chart" /> }))
let requests: Array<{ path: string; body: any; method?: string }>
let failPrepare: boolean
let invalidAudit: boolean
let releaseRun: (() => void) | null
let delayed: boolean
let runResult: typeof timingRunFixture
let catalogResult: typeof timingCatalogFixture
const response = (value: unknown, status = 200) => ({ ok: status < 400, status, json: async () => value })
beforeEach(() => {
  requests = []; failPrepare = false; invalidAudit = false; delayed = false; releaseRun = null
  runResult = structuredClone(timingRunFixture)
  catalogResult = structuredClone(timingCatalogFixture)
  vi.stubGlobal('fetch', vi.fn(async (input: string, init?: RequestInit) => {
    const url = new URL(String(input), 'http://localhost'), path = url.pathname
    const body = init?.body ? JSON.parse(String(init.body)) : undefined
    requests.push({ path, body, method: init?.method })
    if (path.endsWith('/catalog')) return response(catalogResult)
    if (path.endsWith('/definitions')) return response(body ? { ...body, id: 'definition-1', revision: 1 } : { items: [] })
    if (path.endsWith('/definitions/definition-1')) return response({ ...body, id: 'definition-1', revision: 2 })
    if (path.endsWith('/prepare')) return failPrepare ? response({ detail: { message: '买入条件存在循环连接。' } }, 422) : response({ compile_token: 'prepared-token', definition_hash: 'testhash', execution: timingExecutionFixture })
    if (path.endsWith('/runs') && body) {
      if (delayed) await new Promise<void>(resolve => { releaseRun = resolve })
      return response({ id: 'job-1', status: 'completed', progress: 1, total: 1, run_id: 'timing-run-test' })
    }
    if (path.endsWith('/runs')) return response({ items: [{ id: 'timing-run-test', name: '趋势历史', created_at: '2026-09-10' }, { id: 'timing-run-2', name: '反转历史', created_at: '2026-09-09' }] })
    if (path.includes('/products/')) return response({ ...timingRunFixture.products[0], product_id: '510500.SH', execution: timingExecutionFixture })
    if (path.includes('/runs/')) return response({ ...runResult, ...(invalidAudit ? { execution: undefined } : {}) })
    if (path.endsWith('/compare')) return response({ items: [{ run_id: 'timing-run-test', name: '趋势历史', products: timingRunFixture.products }] })
    if (path.endsWith('/releases')) return response(body ? { id: 'release-1', run_id: body.run_id, usage: 'research_only' } : { items: [{ id: 'existing-release', run_id: 'timing-run-test', name: '已审核趋势版本', usage: 'research_only', products: [{ product_id: '510300.SH' }] }] })
    if (path.endsWith('/bindings')) return response({ id: 'binding-1' })
    if (path.includes('/instruments/search')) return response({ items: [{ code: '510500.SH', ts_code: '510500.SH', name: '示例 ETF', instrument_type: 'etf' }] })
    throw new Error(`Unexpected request ${path}`)
  }))
})
afterEach(() => { vi.unstubAllGlobals() })
const open = (path = '/', mode?: 'research' | 'application') => render(<MemoryRouter initialEntries={[path]}><TimingResearch mode={mode} /></MemoryRouter>)

describe('择时研究工作流', () => {
  it('冻结选择展示服务端选中参数而不是草稿默认值', () => {
    render(<TimingTrainingAudit definition={timingCatalogFixture.templates[0].definition} audit={{ mode: 'global', freeze_date: '2024-01-01', fit_end_date: '2023-12-29', min_trades: 5, embargo_bars: 0, selection: [{ state: '0', action_id: 'trend:2', action_label: '趋势候选 · 第二组', sample_count: 20, utility: .02 }], candidates: [{ id: 'trend:2', label: '趋势候选 · 第二组', parameters: [{ node: 'signal', parameter: 'threshold', value: 1.05 }], states: [] }], warnings: [] }} />)
    fireEvent.click(screen.getByText('训练选择与冻结审计'))
    fireEvent.click(screen.getByText('已选参数'))
    expect(screen.getByText('买入信号 · threshold')).toBeInTheDocument()
    expect(screen.getByText('1.05')).toBeInTheDocument()
  })
  it('ETF 改编说明、训练参数和独立篮子可编辑并随请求冻结', async () => {
    catalogResult = structuredClone(timingTrainingCatalogFixture)
    open(); await screen.findAllByDisplayValue('ETF 弱月选择改编')
    expect(screen.getByText(/ETF 改编说明 · A2074/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '运行研究' })).toBeDisabled()
    expect(screen.getByLabelText('市场参考篮子代码')).toHaveValue('')
    expect(screen.getByText(/算法需要市场参考篮子/)).toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('市场参考篮子代码'), { target: { value: '510300.sh, 510500.SH' } })
    fireEvent.click(screen.getByText('编辑训练规则与候选动作'))
    fireEvent.change(screen.getByLabelText('选择方式'), { target: { value: 'state' } })
    fireEvent.click(screen.getByRole('button', { name: '添加状态条件' }))
    fireEvent.change(screen.getByLabelText('状态条件 1'), { target: { value: 'signal.condition' } })
    fireEvent.change(screen.getByLabelText('动作 1 名称'), { target: { value: '趋势规则对照' } })
    fireEvent.click(screen.getByText(/参数搜索空间 · 1/))
    fireEvent.change(screen.getByLabelText('搜索 1 方案 2 参数 1 值'), { target: { value: '1.05' } })
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    const request = requests.find(item => item.path.endsWith('/runs') && item.body)!.body
    expect(request.context_baskets).toEqual({ market: ['510300.SH', '510500.SH'], category: [] })
    expect(request.definition.training).toMatchObject({ mode: 'state', state_refs: ['signal.condition'], actions: [{ id: 'trend', label: '趋势规则对照', entry: 'signal.condition' }, { id: 'cash', label: '空仓', entry: null }] })
    expect(request.definition.training.search_space[0].choices[1][0].value).toBe(1.05)
    expect(request.definition.adaptation.source_experiments).toEqual(['A2074'])
    expect(timingTrainingCatalogFixture.templates[0].definition.training!.search_space[0].choices[1][0].value).toBe(1.02)
    fireEvent.change(screen.getByLabelText('市场参考篮子代码'), { target: { value: '510300.SH, 159919.SZ' } })
    expect(screen.getByText(/配置已修改/)).toBeInTheDocument()
  })
  it('训练可禁用搜索与自动选择，历史定义不增添不存在的字段', async () => {
    catalogResult = structuredClone(timingTrainingCatalogFixture)
    open(); await screen.findAllByDisplayValue('ETF 弱月选择改编')
    fireEvent.click(screen.getByText('编辑训练规则与候选动作'))
    fireEvent.click(screen.getByText(/参数搜索空间 · 1/))
    fireEvent.click(screen.getByRole('button', { name: '禁用参数搜索' }))
    fireEvent.click(screen.getByRole('button', { name: '保存算法' }))
    await screen.findByText(/算法已保存/)
    expect(requests.find(item => item.path.endsWith('/definitions') && item.body)!.body.training.search_space).toEqual([])
    fireEvent.click(screen.getByRole('checkbox', { name: '用训练期选择方案' }))
    fireEvent.click(screen.getByRole('button', { name: '保存算法' }))
    await waitFor(() => expect(requests.some(item => item.method === 'PUT')).toBe(true))
    expect(requests.find(item => item.method === 'PUT')!.body).not.toHaveProperty('training')
    expect(editableTimingDefinition(timingCatalogFixture.templates[0].definition)).not.toHaveProperty('adaptation')
  })
  it('历史训练、篮子与审计恢复；编辑不会改变冻结快照', async () => {
    catalogResult = structuredClone(timingTrainingCatalogFixture)
    runResult.definition_snapshot = structuredClone(catalogResult.templates[0].definition)
    runResult.request_snapshot.definition = runResult.definition_snapshot
    runResult.request_snapshot.context_baskets = { market: ['510300.SH', '510500.SH'], category: [] }
    runResult.products[0].training = { mode: 'month', freeze_date: '2024-01-01', fit_end_date: '2023-12-29', embargo_bars: 1, min_trades: 5, selection: [{ state: '1', action_id: null, action_label: '样本不足，不交易', sample_count: 2, utility: null }], candidates: [{ id: 'trend', label: '趋势候选', states: [{ state: '1', sample_count: 2, utility: null, win_rate: .5, mean_return: .02, stop_rate: 0 }] }], warnings: ['训练样本不足，保持空仓。'] }
    open(); await screen.findAllByDisplayValue('ETF 弱月选择改编')
    fireEvent.click(screen.getByRole('tab', { name: '历史与对比' }))
    fireEvent.click(screen.getAllByRole('button', { name: '查看' })[0])
    await screen.findByRole('region', { name: '择时研究结果' })
    expect(screen.queryByText(/配置已修改/)).not.toBeInTheDocument()
    expect(screen.getByLabelText('市场参考篮子代码')).toHaveValue('510300.SH, 510500.SH')
    fireEvent.click(screen.getByText('训练选择与冻结审计'))
    expect(screen.getByText('样本不足，不交易')).toBeInTheDocument()
    expect(screen.getByText(/冻结日 2024-01-01，训练截止 2023-12-29/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('tab', { name: '3. 算法步骤' }))
    fireEvent.click(screen.getByText('编辑训练规则与候选动作'))
    fireEvent.change(screen.getByLabelText('最少已完成信号数'), { target: { value: '12' } })
    expect(runResult.definition_snapshot.training!.min_trades).toBe(5)
    fireEvent.click(screen.getByRole('tab', { name: '4. 研究结果' }))
    expect(screen.getByText(/配置已修改/)).toBeInTheDocument()
  })
  it('篮子端口不能连接单产品序列，节点上限使用目录', async () => {
    catalogResult = structuredClone(timingTrainingCatalogFixture)
    catalogResult.limits = { steps: 4 }
    open(); await screen.findAllByDisplayValue('ETF 弱月选择改编')
    fireEvent.click(screen.getByRole('button', { name: /篮子均价/ }))
    const input = screen.getByLabelText('篮子均价 篮子输入')
    expect(within(input).getAllByRole('option').map(option => option.getAttribute('value'))).toEqual(['', 'basket.value'])
    expect(screen.getByRole('button', { name: '添加步骤' })).toBeDisabled()
  })
  it('参数空间和缺失引用校验，不把无效动作改成空仓', () => {
    const definition = editableTimingDefinition(timingTrainingCatalogFixture.templates[0].definition)
    definition.training!.search_space = [{ label: '过多方案', choices: Array.from({ length: 109 }, () => [{ node: 'signal', parameter: 'threshold', value: 1 }]) }]
    expect(timingTrainingError(definition, timingTrainingCatalogFixture)).toContain('最多 108')
    definition.training!.search_space = []
    definition.nodes = definition.nodes.filter(node => node.id !== 'signal')
    expect(timingTrainingError(definition, timingTrainingCatalogFixture)).toContain('连接有效的条件')
    expect(definition.training!.actions[0].entry).toBe('signal.condition')
    expect(basketValidation(definition, { market: '510300.SH,510300.SH', category: '' })).toContain('重复代码')
    expect(basketValidation(definition, { market: 'broken', category: '' })).toContain('代码格式')
  })
  it('从可编辑模板准备运行，默认展示样本外，并提供年月和交易解释', async () => {
    open()
    await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    const sent = requests.find(item => item.path.endsWith('/runs') && item.body)!.body
    expect(sent.compile_token).toBe('prepared-token')
    expect(sent.targets).toEqual([{ kind: 'etf', product_id: '510300.SH' }])
    expect(sent.definition.nodes[1].inputs.input).toBe('price.value')
    expect(screen.getByRole('button', { name: '样本外' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByText('4.00%')).toBeInTheDocument()
    const priceChart = screen.getByRole('group', { name: '买卖点图表' })
    expect(within(priceChart).getByText('B')).toBeInTheDocument()
    expect(within(priceChart).getByText('S')).toBeInTheDocument()
    expect(priceChart).toHaveTextContent('成交点全部保留')
    fireEvent.click(screen.getByRole('tab', { name: '年度 / 月度' }))
    expect(screen.getByRole('table', { name: '年度表现' })).toHaveTextContent('2024')
    fireEvent.change(screen.getByLabelText('统计频率'), { target: { value: 'monthly' } })
    expect(screen.getByRole('table', { name: '月度表现' })).toHaveTextContent('2024-01')
    fireEvent.click(screen.getByRole('tab', { name: '逐笔交易' }))
    expect(screen.getByRole('table', { name: '逐笔交易' })).toHaveTextContent('退出条件触发')
  })
  it('编辑模板真实步骤并保存版本，不修改内置模板', async () => {
    open(); const user = userEvent.setup()
    await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: /买入信号/ }))
    fireEvent.change(screen.getByLabelText('阈值'), { target: { value: '2' } })
    fireEvent.change(screen.getByLabelText('算法名称'), { target: { value: '我的趋势算法' } })
    await act(async () => { await user.click(screen.getByRole('button', { name: '保存算法' })) })
    await screen.findByRole('status')
    expect(requests.find(item => item.path.endsWith('/definitions') && item.body)!.body.nodes[1].parameters.threshold).toBe(2)
    expect(timingCatalogFixture.templates[0].definition.nodes[1].parameters.threshold).toBe(1)
    await act(async () => { await user.click(screen.getByRole('button', { name: '保存算法' })) })
    await waitFor(() => expect(requests.some(item => item.method === 'PUT' && item.body.revision === 1)).toBe(true))
  })
  it('从空白添加步骤并显式提示未连接，删除会清理真实依赖', async () => {
    open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: /收盘价格/ }))
    fireEvent.click(screen.getByRole('button', { name: '移除步骤及关联连线' }))
    expect(screen.getByText('待连接：比较序列')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '从空白开始' }))
    expect(screen.getByRole('button', { name: '运行研究' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '添加步骤' }))
    expect(screen.getByLabelText('步骤名称')).toHaveValue('行情字段')
  })
  it('草稿变化使旧结果失效，异步完成不覆盖当前配置', async () => {
    delayed = true; open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await waitFor(() => expect(releaseRun).not.toBeNull())
    fireEvent.change(screen.getByLabelText('算法名称'), { target: { value: '新的研究假设' } })
    releaseRun!()
    await screen.findByRole('region', { name: '择时研究结果' })
    expect(screen.getByLabelText('算法名称')).toHaveValue('新的研究假设')
    expect(screen.getByText(/配置已修改/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '保存研究版本' })).toBeDisabled()
  })
  it('校验错误可恢复，不展示捏造结果', async () => {
    failPrepare = true; open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('循环连接')
    expect(screen.queryByRole('region', { name: '择时研究结果' })).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '运行研究' })).toBeEnabled()
  })
  it('拒绝缺少计算证明的历史结果', async () => {
    invalidAudit = true
    await expect(timingApi.getRun('timing-run-test')).rejects.toThrow('执行证明')
  })
  it('产品详情带入代码，场外基金入口明确不可运行', async () => {
    open('/?product_id=000001.OF&kind=fund')
    await screen.findByDisplayValue('均线趋势研究')
    expect(screen.getByText(/目前支持场内 ETF/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '运行研究' })).toBeDisabled()
  })
  it('产品切换按需读取步骤结果，避免批量加载所有曲线', async () => {
    runResult.products.push({ product_id: '510500.SH', status: 'ok', detail_loaded: false, summary: timingRunFixture.products[0].summary })
    open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    expect(requests.some(item => item.path.includes('/products/'))).toBe(false)
    fireEvent.change(screen.getByLabelText('查看产品'), { target: { value: '510500.SH' } })
    await waitFor(() => expect(requests.some(item => item.path.endsWith('/products/510500.SH'))).toBe(true))
    await waitFor(() => expect(screen.queryByText(/正在读取该产品/)).not.toBeInTheDocument())
  })
  it('研究引用锁定运行并绑定投前，不直接提交组合', async () => {
    open('/', 'application'); await screen.findByText('已审核趋势版本')
    fireEvent.click(screen.getByRole('tab', { name: '3. 算法步骤' }))
    await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    fireEvent.change(screen.getByLabelText('研究引用说明'), { target: { value: '宽基 ETF 对照' } })
    fireEvent.click(screen.getByRole('button', { name: '引用到投前研究' }))
    await screen.findByText(/已引用到投前研究/)
    expect(requests.find(item => item.path.endsWith('/bindings'))!.body).toEqual({ release_id: 'release-1', context: 'pre_investment', note: '宽基 ETF 对照' })
  })
  it('投前入口优先展示已有版本，直接引用不会重复发布或运行', async () => {
    open('/', 'application')
    await screen.findByText('已审核趋势版本')
    expect(screen.getByRole('tab', { name: '已有研究版本' })).toHaveAttribute('aria-selected', 'true')
    expect(screen.queryByRole('button', { name: '运行研究' })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText('本次引用说明'), { target: { value: '投前备选研究' } })
    fireEvent.click(screen.getByRole('button', { name: '引用此版本到投前' }))
    await screen.findByText(/已引用“已审核趋势版本”/)
    expect(requests.find(item => item.path.endsWith('/bindings'))!.body).toEqual({ release_id: 'existing-release', context: 'pre_investment', note: '投前备选研究' })
    expect(requests.some(item => item.path.endsWith('/releases') && item.method === 'POST')).toBe(false)
    expect(requests.some(item => item.path.endsWith('/runs') && item.method === 'POST')).toBe(false)
  })
  it('研究入口按需打开已有版本并查看冻结运行', async () => {
    runResult.request_snapshot.context_baskets = {}
    open(); await screen.findByDisplayValue('均线趋势研究')
    expect(requests.some(item => item.path.endsWith('/releases'))).toBe(false)
    fireEvent.click(screen.getByRole('tab', { name: '已有研究版本' }))
    await screen.findByText('已审核趋势版本')
    fireEvent.click(screen.getByRole('button', { name: '查看此版本结果' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    expect(screen.queryByText(/配置已修改/)).not.toBeInTheDocument()
  })
  it('零交易样本不能根据平直净值判断稳定', async () => {
    runResult.products[0].summary!.out_of_sample = { ...runResult.products[0].summary!.out_of_sample, trade_count: 0 }
    open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    expect(screen.getByText(/没有已完成的有效交易，不据此判定稳定/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '全区间' }))
    expect(screen.queryByText(/没有已完成的有效交易，不据此判定稳定/)).not.toBeInTheDocument()
  })
  it('按所选样本展示服务端信号覆盖，缺失集中度不显示为零', async () => {
    const diagnostic = { raw_signal_count: 0, known_condition_count: 80, active_month_count: 0, total_month_count: 12, top_month_signal_share: null, positive_month_fraction: .25, positive_excess_month_fraction: .5 }
    runResult.products[0].diagnostics = { all: { ...diagnostic, raw_signal_count: 24, active_month_count: 8, top_month_signal_share: .5 }, out_of_sample: diagnostic }
    open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('button', { name: '运行研究' }))
    await screen.findByRole('region', { name: '择时研究结果' })
    fireEvent.click(screen.getByText('信号覆盖与月度稳定性'))
    const details = screen.getByText('信号覆盖与月度稳定性').closest('details')!
    expect(within(details).getByText('—')).toBeInTheDocument()
    expect(details).toHaveTextContent('含不完整首尾月，非综合评分')
    fireEvent.click(screen.getByRole('button', { name: '全区间' }))
    expect(within(details).getByText('24')).toBeInTheDocument()
    expect(within(details).getByText('50.00%')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '样本内' }))
    expect(screen.queryByText('信号覆盖与月度稳定性')).not.toBeInTheDocument()
  })
  it('历史研究支持选择、同口径比较与明确缺失显示', async () => {
    open(); await screen.findByDisplayValue('均线趋势研究')
    fireEvent.click(screen.getByRole('tab', { name: '历史与对比' }))
    fireEvent.click(screen.getByLabelText('对比 趋势历史 timing-run-test'))
    fireEvent.click(screen.getByLabelText('对比 反转历史 timing-run-2'))
    fireEvent.click(screen.getByRole('button', { name: '比较选中研究' }))
    expect(await screen.findByRole('table', { name: '研究对比' })).toHaveTextContent('4.00%')
    fireEvent.click(screen.getAllByRole('button', { name: '查看' })[0])
    const result = await screen.findByRole('region', { name: '择时研究结果' })
    fireEvent.click(within(result).getByRole('button', { name: '样本内' }))
    expect(within(result).getByText('—')).toBeInTheDocument()
    expect(screen.queryByText(/配置已修改/)).not.toBeInTheDocument()
  })
})
