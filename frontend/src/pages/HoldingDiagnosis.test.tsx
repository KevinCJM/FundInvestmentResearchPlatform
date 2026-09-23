import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useNavigate } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import HoldingDiagnosis from './HoldingDiagnosis'
import { diagnosePortfolioRun, downloadPortfolioExport, getPortfolioRun, getResearchTarget, listPortfolioIndicators, listPortfolioRuns, runPortfolioScenario } from '../services/portfolioResearch'

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/portfolioResearch', () => ({
  getResearchTarget: vi.fn(), getPortfolioRun: vi.fn(), diagnosePortfolioRun: vi.fn(),
  listPortfolioRuns: vi.fn(), listPortfolioIndicators: vi.fn(), downloadPortfolioExport: vi.fn(), runPortfolioScenario: vi.fn(),
}))

describe('HoldingDiagnosis', () => {
  beforeEach(() => {
    vi.mocked(getResearchTarget).mockResolvedValue({ id: 'target-1', name: '稳健组合', kind: 'portfolio', revision: 2, definition: {} })
    vi.mocked(getPortfolioRun).mockResolvedValue({ id: 'run-1', target_id: 'target-1', name: '稳健组合', nav: [{ date: '2024-01-01', value: 1 }], drawdown: [{ date: '2024-01-01', value: 0 }], metrics: [], weights: [], contributions: [], warnings: [], regime_conditioning: {
      binding: { run_id: 'regime-run-1', publication_id: 'regime-publication-1', definition_id: 'definition-1', definition_revision: 3 },
      coverage: { periods: 20, classified_periods: 18, classified_ratio: .9 },
      conditional_performance: { 稳健组合: [{ state_id: 'bull', state_label: '牛市', observations: 10, return_observations: 9, annualized_return: .12 }] },
      period_states: [],
    } })
    vi.mocked(listPortfolioIndicators).mockResolvedValue([{
      id: 'portfolio-volatility', name: '组合波动率', revision: 2, context_kind: 'portfolio',
      source: 'custom', read_only: false, created_at: '', updated_at: '',
      description: '组合收益率标准差', expression: 'std(portfolio_returns, 1)',
      result_kind: 'scalar', output_contract: 'scalar', display_format: 'percent',
      precision: 2, unit: '%', direction: 'lower_better', annual_risk_free_rate_percent: 0,
    }])
    vi.mocked(diagnosePortfolioRun)
      .mockResolvedValueOnce({ summary: [{ name: '年化收益', value: .1, unit: 'percent' }], components: [], contributions: [{ product_id: '510300.SH', name: '沪深300ETF', contribution: .05, risk_contribution: .4 }], concentration: [{ name: 'HHI', value: .5 }], covariance: null, correlation: null, weight_path: [], warnings: ['[INSUFFICIENT_SAMPLE] 样本窗口较短'] })
      .mockResolvedValueOnce({ summary: [{ name: '年化收益', value: .1, unit: 'percent' }], components: [], contributions: [], concentration: [], covariance: null, correlation: null, weight_path: [], custom_indicators: [{ name: '组合波动率', value: null, status: 'warning', warnings: ['[INSUFFICIENT_COMMON_SAMPLE] 共同样本不足'] }], warnings: [] })
    vi.mocked(runPortfolioScenario).mockResolvedValue({ name: '历史压力区间', metrics: [{ name: '区间收益', value: -.08, unit: 'percent' }], warnings: [] })
  })

  it('以不可变快照加载诊断、情景和导出能力', async () => {
    render(<MemoryRouter initialEntries={['/holding-diagnosis?target=target-1&run=run-1']}><HoldingDiagnosis /></MemoryRouter>)
    expect(await screen.findByText('持仓诊断：稳健组合')).toBeInTheDocument()
    expect(screen.getByText(/样本不足：样本窗口较短/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_SAMPLE/)).not.toBeInTheDocument()
    expect(diagnosePortfolioRun).toHaveBeenCalledWith('run-1', [])
    expect(screen.getByText('收益贡献')).toBeInTheDocument()
    expect(screen.getByText('历史情景条件表现')).toBeInTheDocument()
    expect(screen.getByText('90.00% · 18/20 期')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
    await waitFor(() => expect(runPortfolioScenario).toHaveBeenCalledWith('run-1', expect.objectContaining({ name: '历史压力区间' })))
    expect(await screen.findByText(/区间收益/)).toBeInTheDocument()
    fireEvent.change(screen.getByRole('combobox', { name: 'CSV 导出表' }), { target: { value: 'correlation' } })
    fireEvent.click(screen.getByRole('button', { name: '导出 CSV' }))
    await waitFor(() => expect(downloadPortfolioExport).toHaveBeenCalledWith('run-1', 'csv', {
      table: 'correlation',
      scenario_start: '2020-01-01',
      scenario_end: '2020-03-31',
    }))
    fireEvent.click(screen.getByRole('button', { name: /选择组合指标/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: /组合波动率/ }))
    fireEvent.click(screen.getByRole('button', { name: '计算选中指标' }))
    await waitFor(() => expect(diagnosePortfolioRun).toHaveBeenLastCalledWith('run-1', ['portfolio-volatility']))
    expect(await screen.findByText(/共同样本不足/)).toBeInTheDocument()
    expect(screen.queryByText(/INSUFFICIENT_COMMON_SAMPLE/)).not.toBeInTheDocument()
  })
})

describe('HoldingDiagnosis AI 助手接入', () => {
  interface AgentCapture { sessions: Array<Record<string, any>>; messages: Array<Record<string, any>> }
  const REAL_RUN_ID = `run-${'a'.repeat(32)}`

  /** Only the shared agent endpoints are needed; business calls stay module-mocked. */
  function agentFetch(capture: AgentCapture) {
    let sessionCount = 0
    let lastRun: Record<string, any> | null = null
    return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input), 'http://localhost').pathname
      if (path === '/api/agent/meta') return { ok: true, json: async () => ({ configured: true, model: 'fixture-model' }) }
      if (path === '/api/agent/sessions' && init?.method === 'POST') {
        sessionCount += 1
        const body = JSON.parse(String(init?.body)); capture.sessions.push(body)
        return { ok: true, json: async () => ({ session_id: `session-${sessionCount}`, session_revision: 0, page_context: body.page_context }) }
      }
      if (path.endsWith('/messages')) {
        const body = JSON.parse(String(init?.body)); capture.messages.push(body)
        lastRun = { run_id: `run-${capture.messages.length}`, session_id: `session-${sessionCount}`, message_id: body.message_id, session_revision: 1, run_revision: 1, status: 'completed', phase: 'thinking',
          response: { session_id: `session-${sessionCount}`, session_revision: 1, reply: { text: `第 ${capture.messages.length} 次回复` } } }
        return { ok: true, status: 202, json: async () => lastRun }
      }
      if (path.endsWith('/events')) return { ok: true, json: async () => ({ items: [], has_more: false, last_seq: 0, next_event_seq: 1 }) }
      if (path.includes('/runs/')) return { ok: true, json: async () => lastRun }
      return { ok: false, status: 503, json: async () => ({ detail: 'offline fixture' }) }
    })
  }

  async function openAndSend(text: string) {
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: '发送消息' })).toBeEnabled())
    fireEvent.change(screen.getByRole('textbox', { name: '发送消息' }), { target: { value: text } })
    await waitFor(() => expect(screen.getByRole('textbox', { name: '发送消息' })).toHaveValue(text))
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
  }

  const baseRun = () => ({
    id: REAL_RUN_ID, target_id: 'target-1', name: '稳健组合',
    nav: [{ date: '2024-01-01', value: 1 }], drawdown: [{ date: '2024-01-01', value: 0 }],
    metrics: [], weights: [], contributions: [], warnings: [],
    correlation: null, covariance: null, regime_conditioning: null,
    requested_as_of: '2020-12-31', effective_as_of: '2020-12-30', target_revision: 4,
  })

  beforeEach(() => {
    vi.clearAllMocks()
    sessionStorage.clear()
    window.localStorage.clear()
    vi.mocked(getResearchTarget).mockResolvedValue({ id: 'target-1', name: '稳健组合', kind: 'portfolio', revision: 2, definition: {} } as any)
    vi.mocked(getPortfolioRun).mockResolvedValue(baseRun() as any)
    vi.mocked(listPortfolioIndicators).mockResolvedValue([{
      id: 'portfolio-volatility', name: '组合波动率', revision: 2, context_kind: 'portfolio',
      source: 'custom', read_only: false, created_at: '', updated_at: '',
      description: '组合收益率标准差', expression: 'std(portfolio_returns, 1)',
      result_kind: 'scalar', output_contract: 'scalar', display_format: 'percent',
      precision: 2, unit: '%', direction: 'lower_better', annual_risk_free_rate_percent: 0,
    }] as any)
    vi.mocked(diagnosePortfolioRun).mockResolvedValue({ summary: [{ name: '年化收益', value: .1, unit: 'percent' }], components: [], contributions: [], concentration: [], covariance: null, correlation: null, weight_path: [], warnings: [] } as any)
    vi.mocked(runPortfolioScenario).mockResolvedValue({ name: '历史压力区间', metrics: [{ name: '区间收益', value: -.08, unit: 'percent' }], warnings: [] } as any)
  })

  it('冻结真实运行 ID、锁定指标和已运行情景，且不携带权重路径或矩阵数组', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', agentFetch(capture))
    vi.mocked(getPortfolioRun).mockResolvedValue({ ...baseRun(),
      nav: [{ date: '2024-01-01', value: 987654.321 }],
      weights: [{ date: '2024-01-01', weights: { '510300.SH': 1 } }],
      correlation: { labels: ['a'], values: [[987654.321]] },
    } as any)
    render(<MemoryRouter initialEntries={[`/holding-diagnosis?target=target-1&run=${REAL_RUN_ID}`]}><HoldingDiagnosis /></MemoryRouter>)
    expect(await screen.findByText('持仓诊断：稳健组合')).toBeInTheDocument()

    const user = userEvent.setup()
    fireEvent.change(screen.getByLabelText('开始日'), { target: { value: '2020-01-01' } })
    fireEvent.change(screen.getByLabelText('结束日'), { target: { value: '2020-03-31' } })
    await user.click(screen.getByRole('button', { name: '运行情景' }))
    await waitFor(() => expect(runPortfolioScenario).toHaveBeenCalledWith(REAL_RUN_ID, expect.objectContaining({ start_date: '2020-01-01', end_date: '2020-03-31' })))

    await openAndSend('这个快照在不同情景下的口径是什么？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.sessions).toEqual([{ page_context: {
      page: 'holding-diagnosis', page_instance_id: expect.stringMatching(/^holding-diagnosis:[0-9a-f]{16}$/),
      context_revision: expect.any(Number), view_state: 'inherit',
      calculation: { context_kind: 'portfolio', run_id: REAL_RUN_ID },
    } }])
    expect(capture.messages[0].page_snapshot).toMatchObject({
      version: 1,
      snapshot_id: expect.stringMatching(/^snap-[0-9a-f]{32}$/),
      captured_at: expect.any(String),
      page: 'holding-diagnosis',
      sections: {
        request: { run_id: REAL_RUN_ID, indicators: [], scenario: { start_date: '2020-01-01', end_date: '2020-03-31' } },
        results: {
          source: 'unverified_client_display',
          displayed_source: 'portfolio-runs + diagnose + scenario',
          refs: { run_id: REAL_RUN_ID, requested_as_of: '2020-12-31', effective_as_of: '2020-12-30', target_revision: 4, component_count: 0, displayed_metrics: 1, scenario: 'explicit_page_request' },
          note: expect.any(String),
        },
      },
    })
    // 权重路径、净值与矩阵数组完全不进入快照。
    const snapshotText = JSON.stringify(capture.messages[0])
    expect(snapshotText).not.toContain('987654')
    expect(snapshotText).not.toContain('weight_path')
    expect(snapshotText).not.toContain('correlation')
  })

  it('未运行情景时不冻结情景参数，锁定的组合指标带各自版本与周期', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', agentFetch(capture))
    vi.mocked(diagnosePortfolioRun).mockResolvedValue({ summary: [], components: [], warnings: [], custom_indicators: [{ name: '组合波动率', value: .2, indicator_id: 'portfolio-volatility', indicator_revision: 7, parameters: { window: 20 } }] } as any)
    render(<MemoryRouter initialEntries={[`/holding-diagnosis?target=target-1&run=${REAL_RUN_ID}`]}><HoldingDiagnosis /></MemoryRouter>)
    expect(await screen.findByText('持仓诊断：稳健组合')).toBeInTheDocument()

    const user = userEvent.setup()
    fireEvent.click(screen.getByRole('button', { name: /选择组合指标/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: /组合波动率/ }))
    await user.click(screen.getByRole('button', { name: '计算选中指标' }))
    await waitFor(() => expect(diagnosePortfolioRun).toHaveBeenLastCalledWith(REAL_RUN_ID, ['portfolio-volatility']))

    await openAndSend('解释锁定指标的口径')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    const request = capture.messages[0].page_snapshot.sections.request
    expect(request.scenario).toBeNull()
    expect(request.indicators).toEqual([{ indicator_id: 'portfolio-volatility', indicator_revision: 7, period: 'snapshot' }])
    expect(capture.messages[0].page_snapshot.sections.results.refs.metrics.resolved_indicators[0].parameters).toEqual({ window: 20 })
  })

  it('没有真实运行快照时保留原构建入口，助手入口真实禁用且不提交占位运行', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.mocked(listPortfolioRuns).mockResolvedValue({ items: [] } as any)
    vi.stubGlobal('fetch', agentFetch(capture))
    render(<MemoryRouter initialEntries={['/holding-diagnosis?target=target-1']}><HoldingDiagnosis /></MemoryRouter>)
    expect(await screen.findByText('该研究对象尚未生成组合运行快照。')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '去构建组合' })).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: '打开 AI 助手' })).toHaveLength(1)

    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: '发送消息' })).toBeEnabled())
    expect(screen.getByRole('status')).toHaveTextContent('尚未加载真实的不可变组合快照')
    fireEvent.change(screen.getByRole('textbox', { name: '发送消息' }), { target: { value: '可以帮我分析吗？' } })
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
    expect(capture.sessions).toHaveLength(0)
    expect(capture.messages).toHaveLength(0)
  })

  it.each([false, true])('情景失败保留 error，重试成功后恢复 ready（已有结果：%s）', async hadResult => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', agentFetch(capture))
    const result = { name: '历史压力区间', metrics: [{ name: '区间收益', value: -.08 }], warnings: [] }
    vi.mocked(runPortfolioScenario).mockReset()
    if (hadResult) vi.mocked(runPortfolioScenario).mockResolvedValueOnce(result)
    vi.mocked(runPortfolioScenario).mockRejectedValueOnce(new Error('情景服务不可用')).mockResolvedValueOnce(result)
    render(<MemoryRouter initialEntries={[`/holding-diagnosis?run=${REAL_RUN_ID}`]}><HoldingDiagnosis /></MemoryRouter>)
    await screen.findByText('持仓诊断：稳健组合')
    if (hadResult) {
      fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
      await screen.findByText(/区间收益/)
    }
    fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
    await screen.findByText('情景服务不可用')
    await openAndSend('解释情景失败')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.messages[0].page_snapshot.sections.results.refs.scenario_result.status).toBe('error')
    fireEvent.click(screen.getByRole('button', { name: '关闭 AI 助手' }))
    fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
    await waitFor(() => expect(screen.queryByText('情景服务不可用')).not.toBeInTheDocument())
    await screen.findByText(/区间收益/)
    await openAndSend('解释重试后的情景')
    await waitFor(() => expect(capture.messages).toHaveLength(2))
    expect(capture.messages[1].page_snapshot.sections.results.refs.scenario_result.status).toBe('ready')
  })

  it('换到缺失快照时不能借旧运行发送，旧情景及指标迟到回包不污染新对象', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', agentFetch(capture))
    const otherRun = `run-${'b'.repeat(32)}`
    let finishScenario!: (value: any) => void
    vi.mocked(runPortfolioScenario).mockImplementation(() => new Promise(resolve => { finishScenario = resolve }))
    vi.mocked(getPortfolioRun).mockImplementation(async id => { if (id === otherRun) throw new Error('未找到指定组合运行快照'); return baseRun() as any })
    function Page() { const navigate = useNavigate(); return <><button onClick={() => navigate(`?run=${otherRun}`)}>切换快照</button><HoldingDiagnosis /></> }
    render(<MemoryRouter initialEntries={[`/holding-diagnosis?run=${REAL_RUN_ID}`]}><Page /></MemoryRouter>)
    await screen.findByText('持仓诊断：稳健组合')
    fireEvent.click(screen.getByRole('button', { name: '运行情景' }))
    await waitFor(() => expect(runPortfolioScenario).toHaveBeenCalled())
    fireEvent.click(screen.getByRole('button', { name: '切换快照' }))
    await screen.findByText('未找到指定组合运行快照')
    await act(async () => finishScenario({ name: '旧情景', metrics: [], warnings: [] }))
    await openAndSend('分析新的快照')
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
    expect(screen.queryByText('旧情景')).not.toBeInTheDocument()
    expect(screen.queryByText('持仓诊断：稳健组合')).not.toBeInTheDocument()
    expect(capture.sessions).toHaveLength(0)
  })
})
