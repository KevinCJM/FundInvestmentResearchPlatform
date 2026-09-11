import { act, fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import HistoricalRegimeCenter from './HistoricalRegimeCenter'
import type { HistoricalRegimeDefinition, HistoricalRegimeRun } from '../services/historicalRegimes'

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: { series?: Array<{ name?: string }> } }) => <div data-testid="regime-chart">{option.series?.map((series) => series.name).join('、')}</div>,
}))
vi.mock('./HistoricalRegimeWorkbench', () => ({
  default: ({ initialDefinition }: { initialDefinition?: { name?: string } }) => <div data-testid="mock-v2-workbench">{initialDefinition?.name || '空白 V2'}</div>,
}))

const definition: HistoricalRegimeDefinition = {
  id: 'DEF-1',
  revision: 1,
  name: '沪深300牛熊震荡',
  description: '单边因果识别',
  template_id: 'bull-bear-causal',
  target: { kind: 'index', series_id: '000300.SH', name: '沪深300', frequency: 'daily', source_api: 'index_daily', ts_code: '000300.SH', field: 'close', availability_mode: 'point_in_time' },
  features: { transform: 'log', filter: 'ema', window: 20, slope_window: 5, volatility_window: 20 },
  algorithm: { family: 'causal_filter', parameters: { bull_enter: 0.0015, bull_exit: 0.0002, bear_enter: -0.0015, bear_exit: -0.0002, confirmation: 3, min_duration: 5 } },
  states: [
    { id: 'bull', label: '牛市', color: '#16a34a' },
    { id: 'sideways', label: '震荡市', color: '#64748b' },
    { id: 'bear', label: '熊市', color: '#dc2626' },
  ],
  validation: { walk_forward: true, folds: 4, stability_perturbation: 0.1 },
  usage_intent: 'taa',
}

const fixedExecution = {
  backend: 'numba_njit_fixed_signature',
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { historical_regime_kernel: ['fixed'] },
}

const run: HistoricalRegimeRun = {
  id: 'RUN-1',
  definition_id: 'DEF-1',
  definition_revision: 1,
  name: '沪深300牛熊震荡',
  mode: 'realtime',
  created_at: '2026-09-03T09:00:00Z',
  immutable: true,
  states: definition.states,
  series: [
    { date: '2024-01-02', value: 100, filtered_value: 99, state_id: 'sideways', state_label: '震荡市', probabilities: { bull: 0.2, sideways: 0.7, bear: 0.1 }, confidence: 0.7, features: { slope: 0 }, reasons: ['趋势斜率位于滞回区间'], recognized_at: '2024-01-02' },
    { date: '2024-02-01', value: 105, filtered_value: 102, state_id: 'bull', state_label: '牛市', probabilities: { bull: 0.8, sideways: 0.15, bear: 0.05 }, confidence: 0.8, features: { slope: 0.002 }, reasons: ['趋势斜率连续三期高于进入阈值'], data_available_at: '2024-02-01', recognized_at: '2024-02-05', effective_date: null, executable: false },
  ],
  segments: [
    { state_id: 'sideways', state_label: '震荡市', start_date: '2024-01-02', end_date: '2024-01-31', recognized_at: '2024-01-02', duration_observations: 20, return: 0.01, volatility: 0.12, max_drawdown: -0.03, confidence: 0.7, reasons: ['趋势斜率位于滞回区间'] },
    { state_id: 'bull', state_label: '牛市', start_date: '2024-02-01', end_date: '2024-02-29', recognized_at: '2024-02-05', duration_observations: 20, return: 0.05, volatility: 0.14, max_drawdown: -0.02, confidence: 0.8, reasons: ['趋势斜率连续三期高于进入阈值'] },
  ],
  conditional_stats: [{ state_id: 'bull', state_label: '牛市', observations: 20, annualized_return: 0.18, volatility: 0.14, max_drawdown: -0.02, sharpe: 1.1, win_rate: 0.62 }],
  transition: { states: ['sideways', 'bull'], counts: [[10, 1], [1, 8]], probabilities: [[0.91, 0.09], [0.11, 0.89]] },
  causality: { classification: 'causal', is_causal: true, uses_future_data: false, repaints: false, realtime_eligible: true, publish_eligible_usages: ['research_display', 'product_research', 'formal_backtest', 'taa'], blockers: [], warnings: [] },
  stability: { agreement_rate: 0.91, boundary_shift_mean: 2.1, realtime_monitoring: { prefix_revision_rate: 0.04, label_flip_rate: 0.12 } },
  walk_forward: { status: 'passed', state_agreement: 0.82 },
  diagnostics: [],
  calculation_audits: [{ source_kind: 'historical_regime_algorithm', ...fixedExecution }],
  publications: [],
}

const retrospectiveRun: HistoricalRegimeRun = {
  ...run,
  id: 'RUN-2',
  mode: 'retrospective',
  causality: { classification: 'non_causal', is_causal: false, uses_future_data: true, repaints: true, realtime_eligible: false, publish_eligible_usages: ['research_display', 'product_research'], blockers: ['双边滤波使用未来样本'], warnings: [] },
}

const template = (id: string, name: string, nextDefinition: HistoricalRegimeDefinition) => ({ id, name, description: name + '研究模板', definition: { ...nextDefinition, id: undefined, revision: undefined, name, template_id: id } })

const meta = {
  schema_version: '1.0',
  modes: [{ id: 'realtime', label: '实时识别' }, { id: 'retrospective', label: '事后识别' }],
  data_sources: [{ id: 'inline', label: '粘贴/上传数据' }, { id: 'index', label: '指数行情' }, { id: 'relative', label: '两序列相对强弱' }, { id: 'indicator', label: '指标中心版本' }],
  features: [{ id: 'ema', label: '单边指数平滑', causal: true }, { id: 'zero_phase', label: '零相位双边滤波', causal: false, repaints: true }],
  algorithm_families: [{ id: 'causal_filter', label: '趋势滤波 + 滞回确认', supports_realtime: true }, { id: 'markov', label: 'Markov 状态切换', supports_realtime: true }, { id: 'change_point', label: '结构突变检测', supports_realtime: true }, { id: 'ensemble', label: '候选算法集成', supports_realtime: true }],
  templates: [
    template('bull-bear-causal', '沪深300牛熊震荡', definition),
    template('merrill-clock', '美林时钟', { ...definition, target: { kind: 'inline', rows: [], availability_mode: 'point_in_time' }, algorithm: { family: 'merrill_clock', parameters: { confirmation: 2 } } }),
    template('size-rotation', '大盘小盘轮动', { ...definition, target: { kind: 'relative', numerator: { ts_code: '000852.SH' }, denominator: { ts_code: '000300.SH' }, transform: 'log_ratio' }, algorithm: { family: 'relative_strength', parameters: { upper: 0.001, lower: -0.001 } } }),
    template('growth-value-rotation', '成长价值轮动', { ...definition, target: { kind: 'relative', numerator: { ts_code: '000919.SH' }, denominator: { ts_code: '000918.SH' }, transform: 'log_ratio' }, algorithm: { family: 'relative_strength', parameters: { upper: 0.001, lower: -0.001 } } }),
  ],
  application_targets: [{ id: 'research_display', label: '研究展示' }, { id: 'taa', label: '战术资产配置' }],
  formula_language: { id: 'causal_expression', allowlist_version: 'typed-njit-causal-1', operators: ['+', '-', '*', '/'], functions: ['log', 'abs', 'difference', 'cumulative_sum'], variable_rule: '变量必须来自当前数据源。', njit_required: true, python_fallback: 0 },
  indicator_catalog: [{ id: 'indicator-trend', revision: 3, name: '趋势强度', description: 'typed 指标', source: 'custom', dsl_version: '2.2.0', compiled_plan_id: 'plan-meta', applicable_product_kinds: ['etf', 'fund'], periods: ['1W', '1M'], execution_backend: 'numba_njit_fixed_signature', njit_required: true }],
  indicator_periods: [{ value: '1W', label: '近 1 周' }, { value: '1M', label: '近 1 月' }],
}

const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body } as Response)

function makeFetch(options?: { runResult?: HistoricalRegimeRun; failRun?: boolean; failFormula?: boolean }) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const path = String(input)
    if (path.endsWith('/meta')) return ok(meta)
    if (path.endsWith('/definitions') && (!init?.method || init.method === 'GET')) return ok({ items: [definition] })
    if (path.endsWith('/runs') && (!init?.method || init.method === 'GET')) return ok({ items: [{ ...run, series: undefined, series_included: false, series_detail_endpoint: `/api/historical-regimes/runs/${run.id}` }] })
    if (path.endsWith(`/runs/${run.id}`) && (!init?.method || init.method === 'GET')) return ok(run)
    if (path.endsWith('/formulas/prepare') && init?.method === 'POST') {
      if (options?.failFormula) return { ok: false, status: 422, json: async () => ({ detail: { code: 'INVALID_FORMULA_FUNCTION', message: '公式使用了未允许的函数，请从白名单中选择。', field: 'features.formula' } }) } as Response
      const payload = JSON.parse(String(init.body)) as { definition?: HistoricalRegimeDefinition }
      const requiresFormula = Boolean(payload.definition?.features?.formula)
      return ok({ required: requiresFormula, compile_token: requiresFormula ? 'prepared-formula-token' : null, request_time_compilation: 0 })
    }
    if (path.endsWith('/run') && init?.method === 'POST') {
      if (options?.failRun) return { ok: false, status: 422, json: async () => ({ detail: { message: '确认期不能超过有效样本。' } }) } as Response
      return ok(options?.runResult ?? { ...run, id: 'RUN-3' })
    }
    if (path.includes('/publish')) return ok({ run_id: 'RUN-2', publication: { id: 'PUB-1', usage: 'research_display', published_at: '2026-09-03', definition_revision: 1, run_id: 'RUN-2' }, publications: [{ id: 'PUB-1', usage: 'research_display', published_at: '2026-09-03', definition_revision: 1, run_id: 'RUN-2' }] })
    if (path.includes('/copy-to-v2?revision=1') && init?.method === 'POST') return ok({ source_v1: { id: 'DEF-1', revision: 1 }, definition: { schema_version: '2.0', id: 'REGIME-V2-1', revision: 1, name: '沪深300牛熊震荡 · V2 图谱', description: '', graph: { nodes: [], edges: [], outputs: {} }, states: definition.states, evaluation_targets: [], validation: {}, usage_intent: 'taa' }, inference: { valid: true, errors: [], warnings: [] }, warnings: [] })
    if (path.endsWith('/definitions') && init?.method === 'POST') return ok({ ...definition, id: 'DEF-2' })
    if (path.includes('/definitions/') && init?.method === 'PUT') return ok(definition)
    throw new Error('Unexpected request: ' + path + ' ' + (init?.method || 'GET'))
  })
}

describe('HistoricalRegimeCenter', () => {
  afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

  it('加载真实元数据并展示四类研究模板与历史区间', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    render(<HistoricalRegimeCenter />)

    expect(await screen.findByRole('heading', { name: '历史情景识别' })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /沪深300牛熊震荡/ })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /美林时钟/ })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /大盘小盘轮动/ })).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /成长价值轮动/ })).toBeInTheDocument()
    expect(screen.getByTestId('regime-chart')).toHaveTextContent('原始序列')
    expect(screen.getByRole('table', { name: '历史情景区间明细' })).toHaveTextContent('当时识别日')
    expect(screen.getByText('各状态概率')).toBeInTheDocument()
    expect(screen.getByText('待下一交易日生效 / 不可执行')).toBeInTheDocument()
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith(`/runs/${run.id}`))).toBe(true)
    expect(screen.queryByText(/INTERACTIVE PROTOTYPE/i)).not.toBeInTheDocument()
  })

  it('把当前经典定义通过真实接口复制并直接载入 V2 工作台', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await user.click(screen.getByRole('button', { name: '复制当前 V1 为 V2 图谱' }))
    expect(await screen.findByTestId('mock-v2-workbench')).toHaveTextContent('沪深300牛熊震荡 · V2 图谱')
    const copyCall = fetchMock.mock.calls.find(([path]) => String(path).includes('/definitions/DEF-1/copy-to-v2?revision=1'))
    expect(copyCall?.[1]?.method).toBe('POST')
    expect(copyCall?.[1]?.body).toBeUndefined()
  })

  it('导出完整不可变运行 JSON，文件名包含运行 ID', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const createObjectURL = vi.fn(() => 'blob:historical-regime')
    const revokeObjectURL = vi.fn()
    vi.stubGlobal('URL', { createObjectURL, revokeObjectURL })
    let downloadName = ''
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function captureDownload(this: HTMLAnchorElement) { downloadName = this.download })
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('button', { name: '导出完整 JSON' })) })
    expect(createObjectURL).toHaveBeenCalledWith(expect.any(Blob))
    expect(downloadName).toContain('RUN-1')
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:historical-regime')
  })

  it('参数变化使结果过期，重新运行发送完整定义并刷新不可变结果', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('button', { name: /模型/ })) })
    act(() => { fireEvent.change(screen.getByLabelText('算法参数 进入牛市阈值'), { target: { value: '0.002' } }) })
    expect(screen.getByText('配置已变化 · 结果过期')).toBeInTheDocument()

    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    expect(await screen.findByText(/识别运行已完成/)).toBeInTheDocument()
    expect(screen.getAllByText(/RUN-3/).length).toBeGreaterThan(0)
    const call = fetchMock.mock.calls.find(([path, init]) => String(path).endsWith('/run') && init?.method === 'POST')
    expect(call).toBeDefined()
    const body = JSON.parse(String(call?.[1]?.body))
    expect(body.mode).toBe('realtime')
    expect(body.definition.target.ts_code).toBe('000300.SH')
    expect(body.definition.algorithm.parameters.bull_enter).toBe(0.002)
  })

  it('选择指标中心精确版本、产品与周期后提交独立 indicator target', async () => {
    const fetchMock = makeFetch()
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await user.selectOptions(screen.getByLabelText('数据结构'), 'indicator')
    expect(screen.getByLabelText('指标中心版本')).toHaveValue('indicator-trend')
    expect(screen.getByText(/趋势强度 R3/)).toBeInTheDocument()
    expect(screen.getByText(/legacy\/Python 或未预热版本会被后端拒绝/)).toBeInTheDocument()
    await user.clear(screen.getByLabelText('指标产品编号'))
    await user.type(screen.getByLabelText('指标产品编号'), '510300.SH')
    await user.selectOptions(screen.getByLabelText('指标评价周期'), '1M')
    await user.click(screen.getByRole('button', { name: '运行历史识别' }))

    const runCall = fetchMock.mock.calls.find(([path, init]) => String(path).endsWith('/run') && init?.method === 'POST')
    const body = JSON.parse(String(runCall?.[1]?.body))
    expect(body.definition.target).toMatchObject({
      kind: 'indicator',
      indicator_id: 'indicator-trend',
      indicator_revision: 3,
      product_kind: 'etf',
      product_id: '510300.SH',
      period: '1M',
      availability_mode: 'point_in_time',
    })
  })

  it('把公式或指标的 AST DAG 与固定签名 NJIT 计划可视化展示', async () => {
    const auditedRun: HistoricalRegimeRun = {
      ...run,
      calculation_audits: [
        {
          source_kind: 'indicator',
          typed_ast: {
            nodes: [
              { id: 0, kind: 'variable', label: 'returns', inputs: [], inferred_type: { display: 'series<time>' } },
              { id: 1, kind: 'call', operator_id: 'mean', inputs: [0], inferred_type: { display: 'scalar' } },
            ],
            edges: [{ source: 0, target: 1 }],
            root: 1,
          },
          dag: { nodes: [], edges: [], roots: { result: 1 } },
          plan: { ...fixedExecution, compiled_plan_id: 'plan-run-1', compile_status: 'compiled', compiled_signatures: ['(Array(float64, 1, C),)'], kernel_version: '2.2.0', engine_version: 'typed-numba', njit_required: true, python_operator_calls: 0 },
        },
        {
          source_kind: 'historical_regime_algorithm',
          family: 'causal_filter',
          compiled_plan_id: 'historical-regime:v1:causal_filter',
          compile_status: 'compiled',
          kernel_version: 'historical-regime-kernels-1.0.0',
          engine_version: 'numba-njit-fixed-signature-1.0.0',
          njit_required: true,
          ...fixedExecution,
          python_operator_calls: 0,
          kernels: [{ kernel_id: 'trend_state', compile_status: 'compiled', compiled_signatures: ['Tuple(...)(float64[:])'], kernel_fingerprint: 'abc123', python_fallback: 0 }],
        },
      ],
    }
    vi.stubGlobal('fetch', makeFetch({ runResult: auditedRun }))
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await user.click(screen.getByRole('button', { name: '运行历史识别' }))
    await user.click(screen.getByRole('tab', { name: '计算审计' }))
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('typed AST → DAG → NJIT 固定签名')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('plan-run-1')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('1 · root')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('0 → 1')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('(Array(float64, 1, C),)')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('Python fallback 0')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('Regime algorithm')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('固定签名 NJIT 内核链')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('trend_state')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('abc123')
    expect(screen.getByLabelText('计算计划审计')).toHaveTextContent('已预热')
  })

  it('自定义因果公式展示白名单帮助，并随草稿试算请求提交', async () => {
    const fetchMock = makeFetch({ runResult: { ...run, id: 'RUN-FORMULA', definition_id: null, definition_revision: null, definition_source: 'inline_trial' } })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('button', { name: /指标 \/ 特征/ })) })
    expect(screen.getByText('白名单 typed-njit-causal-1')).toBeInTheDocument()
    expect(screen.getByText(/typed AST → DAG → NJIT/)).toBeInTheDocument()
    act(() => { fireEvent.change(screen.getByLabelText('自定义因果公式'), { target: { value: 'difference(log(value), 20)' } }) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })

    const runCall = fetchMock.mock.calls.find(([path, init]) => String(path).endsWith('/run') && init?.method === 'POST')
    const body = JSON.parse(String(runCall?.[1]?.body))
    expect(body.definition.features.formula).toBe('difference(log(value), 20)')
  })

  it('非法公式只展示中文业务错误，不暴露内部错误码', async () => {
    vi.stubGlobal('fetch', makeFetch({ failFormula: true }))
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })
    await act(async () => { await user.click(screen.getByRole('button', { name: /指标 \/ 特征/ })) })
    act(() => { fireEvent.change(screen.getByLabelText('自定义因果公式'), { target: { value: 'future(value)' } }) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })

    expect(await screen.findByRole('alert')).toHaveTextContent('公式使用了未允许的函数')
    expect(screen.queryByText('INVALID_FORMULA_FUNCTION')).not.toBeInTheDocument()
  })

  it('事后非因果结果阻止正式回测发布，但允许研究展示', async () => {
    const fetchMock = makeFetch({ runResult: retrospectiveRun })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('radio', { name: '事后划分' })) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    expect((await screen.findAllByText(/含未来信息/)).length).toBeGreaterThan(0)
    await act(async () => { await user.click(screen.getByRole('button', { name: /发布/ })) })

    await act(async () => { await user.click(screen.getByRole('radio', { name: /正式回测/ })) })
    expect(screen.getByRole('button', { name: '发布到正式回测' })).toBeDisabled()
    expect(screen.getByText(/双边滤波使用未来样本/)).toBeInTheDocument()

    await act(async () => { await user.click(screen.getByRole('radio', { name: /研究展示/ })) })
    const publishButton = screen.getByRole('button', { name: '发布到研究展示' })
    expect(publishButton).toBeEnabled()
    await act(async () => { await user.click(publishButton) })
    expect(await screen.findByText(/下游将锁定本次版本/)).toBeInTheDocument()
    const publishCall = fetchMock.mock.calls.find(([path]) => String(path).includes('/publish'))
    expect(JSON.parse(String(publishCall?.[1]?.body))).toEqual({ usage: 'research_display' })
  })

  it('后端运行错误展示中文业务说明且不暴露内部错误码', async () => {
    vi.stubGlobal('fetch', makeFetch({ failRun: true }))
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    expect(await screen.findByRole('alert')).toHaveTextContent('确认期不能超过有效样本')
    expect(screen.queryByText('REQUEST_VALIDATION_ERROR')).not.toBeInTheDocument()
  })

  it('历史情景结果缺少完整执行证明链时失败关闭', async () => {
    vi.stubGlobal('fetch', makeFetch({
      runResult: { ...run, id: 'RUN-NO-AUDIT', calculation_audit: null, calculation_audits: [] },
    }))
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    expect(await screen.findByRole('alert')).toHaveTextContent('未提供完整的数值执行证明链')
  })

  it('宏观模板允许粘贴 CSV 并校验后写入原始观测', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('radio', { name: /美林时钟/ })) })
    act(() => { fireEvent.change(screen.getByLabelText('原始数据内容'), { target: { value: 'date,growth,inflation,available_at,vintage\n2024-01-31,0.2,-0.1,2024-02-15,initial\n2024-02-29,0.3,0.1,2024-03-15,revised' } }) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '解析并应用' })) })
    expect(screen.getByText('已解析 2 条观测')).toBeInTheDocument()
    expect(screen.getByText(/配置或模式已经变化|当前模板草稿/)).toBeInTheDocument()
  })

  it('模板草稿可以试算，但发布前必须保存定义并重新运行', async () => {
    vi.stubGlobal('fetch', makeFetch({ runResult: { ...run, id: 'RUN-DRAFT', definition_id: null, definition_revision: null, definition_source: 'inline_trial', causality: { ...run.causality, publish_eligible_usages: [] } } }))
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('radio', { name: /美林时钟/ })) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    await act(async () => { await user.click(screen.getByRole('button', { name: /发布/ })) })

    expect(screen.getByText(/当前是模板草稿试算/)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '发布到研究展示' })).toBeDisabled()
  })

  it('候选算法集成校验成员配置，并把合法数组提交给后端', async () => {
    const fetchMock = makeFetch({ runResult: { ...run, id: 'RUN-ENSEMBLE', definition_id: null, definition_revision: null, definition_source: 'inline_trial' } })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })

    await act(async () => { await user.click(screen.getByRole('button', { name: /模型/ })) })
    await act(async () => { await user.click(screen.getByRole('radio', { name: /候选算法集成/ })) })
    const editor = screen.getByLabelText('候选算法与权重 JSON')
    expect((editor as HTMLTextAreaElement).value).toContain('causal_filter')

    act(() => { fireEvent.change(editor, { target: { value: '[{"family":"ensemble","weight":-1},{"family":"causal_filter","weight":1}]' } }) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '校验并应用成员' })) })
    expect(screen.getByRole('alert')).toHaveTextContent(/不能嵌套集成模型|有效算法族和正权重/)
    expect(fetchMock.mock.calls.some(([path]) => String(path).endsWith('/run'))).toBe(false)

    const validMembers = [
      { family: 'causal_filter', weight: 0.7, parameters: { confirmation: 3 } },
      { family: 'change_point', weight: 0.3, parameters: { window: 12, threshold: 1.2 } },
    ]
    act(() => { fireEvent.change(editor, { target: { value: JSON.stringify(validMembers) } }) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '校验并应用成员' })) })
    await act(async () => { await user.click(screen.getByRole('button', { name: '运行历史识别' })) })
    const runCall = fetchMock.mock.calls.find(([path, init]) => String(path).endsWith('/run') && init?.method === 'POST')
    const body = JSON.parse(String(runCall?.[1]?.body))
    expect(body.definition.algorithm.parameters.members).toEqual(validMembers)
    expect(body.definition.algorithm.parameters.consensus_threshold).toBe(0.6)
  })

  it('稳定性页以中文展示实时修订率与标签翻转率', async () => {
    vi.stubGlobal('fetch', makeFetch())
    const user = userEvent.setup()
    render(<HistoricalRegimeCenter />)
    await screen.findByRole('heading', { name: '历史情景识别' })
    await act(async () => { await user.click(screen.getByRole('tab', { name: '稳定性 / Walk-forward' })) })
    expect(screen.getByText('实时回放修订率')).toBeInTheDocument()
    expect(screen.getByText('标签翻转率')).toBeInTheDocument()
  })
})
