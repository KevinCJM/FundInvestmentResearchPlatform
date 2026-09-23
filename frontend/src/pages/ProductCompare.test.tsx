import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes, useNavigate } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductCompare from './ProductCompare'
import { cppAotAudit } from '../test/cppAotFixture'
import { evaluateCustomIndicators, getCustomIndicatorMeta, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition, MetricPresentation } from '../services/customIndicators'
const testPit = vi.hoisted(() => ({ day: null as string | null }))
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => testPit.day, useResearchContextIdentity: () => `fixture-${testPit.day}` }))

vi.mock('echarts-for-react', () => ({ default: () => <div data-testid="chart" /> }))
vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn(),
  indicatorPeriodLabel: (period: string) => period,
  indicatorsForContext: (items: IndicatorDefinition[], context: string) => items.filter((item) => (item.context_kind ?? 'single_product') === context),
  listCustomIndicators: vi.fn(),
}))

const annualIndicator: IndicatorDefinition = {
  id: 'total-return', revision: 2, source: 'custom', read_only: false,
  name: '区间累计收益', description: '真实净值计算。', expression: '\\left(\\prod\\left(\\mathbf{r}+1\\right)\\right)-1',
  periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
}

const annualPresentation: MetricPresentation = {
  indicator_id: annualIndicator.id, revision: annualIndicator.revision,
  name: annualIndicator.name, source: annualIndicator.source,
  category: 'return', category_label: '收益', context_kind: 'single_product', catalog_status: 'current',
  display_format: 'percent', precision: 2, unit: '%', notation: 'standard', value_scale: 100,
  output_measure: 'return_decimal', direction: 'higher_better', description: annualIndicator.description,
  methodology: '区间净值收益', data_basis: '复权净值', minimum_observations: 2,
  applicable_product_kinds: ['etf', 'fund'],
}

const monthlyIndicator: IndicatorDefinition = {
  ...annualIndicator, id: 'monthly-volatility', name: '月度波动率', periods: ['1M'], display_format: 'number', unit: '', precision: 4,
}

function productResponse(id: string, name: string) {
  return {
    product_id: id, name, management: '测试管理人', status: '上市', base_info: { ts_code: id, fund_type: 'ETF' },
    metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1, exp_return: 8 },
    timeseries: [
      { date: '2026-01-02', close: 3.0 }, { date: '2026-01-05', close: 3.1 }, { date: '2026-01-06', close: 3.2 },
    ],
  }
}

function compareResponse(id: string) {
  const metrics = {
    cumulativeReturn: 6.67,
    annualizedReturn: 18.5,
    volatility: 12.3,
    maxDrawdown: -4.2,
    returnToFee: 11.12,
    totalFee: 0.6,
    sharpeRatio: 1.5,
    calmarRatio: 4.4,
  }
  const normalized_nav = [
    { date: '2026-01-02', value: 1 },
    { date: '2026-01-05', value: 1.0333 },
    { date: '2026-01-06', value: 1.0667 },
  ]
  const drawdown = normalized_nav.map((point) => ({ date: point.date, value: 0 }))
  const rolling_volatility = normalized_nav.map((point, index) => ({
    date: point.date,
    value: index < 2 ? null : 12.3,
  }))
  const range = {
    window: { start_date: '2026-01-02', end_date: '2026-01-06', observation_count: 3 },
    metrics,
    normalized_nav,
    drawdown,
    rolling_volatility,
  }
  return {
    schema_version: 1,
    product_id: id,
    ranges: { performance: range, risk: range, efficiency: range },
    execution: {
      backend: 'numba_njit_fixed_signature',
      execution_backend: 'numba_njit_fixed_signature',
      engine: 'instrument-analytics-njit-1.0.0',
      kernel_version: 'instrument-statistics-quality-3',
      kernel_coverage: '31/31',
      kernel_signatures: { product_compare_analysis_kernel: ['fixed-signature'] },
      kernel_fingerprint: 'compare-fingerprint',
      nopython: true,
      object_mode: 0,
      njit_required: true,
      python_fallback: 0,
      request_time_compilation: 0,
    },
  }
}

function productFetch(input: RequestInfo | URL) {
  const url = String(input)
  const id = url.includes('510300.SH') ? '510300.SH' : '159915.SZ'
  if (url.includes('/compare-analysis')) {
    return Promise.resolve({ ok: true, json: async () => compareResponse(id) })
  }
  return Promise.resolve({
    ok: true,
    json: async () => productResponse(id, id === '510300.SH' ? '沪深300ETF' : '创业板ETF'),
  })
}

describe('ProductCompare custom indicators', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.stubGlobal('fetch', vi.fn(productFetch))
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [annualIndicator, monthlyIndicator], total: 2 })
    vi.mocked(getCustomIndicatorMeta).mockResolvedValue({ periods: [{ value: '1M', label: '近 1 月', description: '运行周期' }, { value: '1Y', label: '近 1 年', description: '运行周期' }] } as any)
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({
      results: [
        { indicator_id: annualIndicator.id, indicator_revision: annualIndicator.revision, indicator_name: annualIndicator.name, presentation: annualPresentation, target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y', value: 0.1234, status: 'ok', warnings: [], window: { requested_as_of: null, effective_as_of: null, start_date: '2025-01-06', end_date: '2026-01-06', observation_count: 250, data_latest_date: '2026-01-06' } },
        { indicator_id: annualIndicator.id, indicator_revision: annualIndicator.revision, indicator_name: annualIndicator.name, presentation: annualPresentation, target: { kind: 'etf', product_id: '159915.SZ', name: '创业板ETF' }, period: '1Y', value: null, status: 'warning', warnings: [{ code: 'INSUFFICIENT_SAMPLE', message: '样本不足' }], window: { requested_as_of: null, effective_as_of: null, start_date: null, end_date: '2026-01-06', observation_count: 10, data_latest_date: '2026-01-06' } },
      ], summary: { total: 2, ok: 1, warning: 1, error: 0 }, cache: { hits: 0, misses: 2 },
      execution: compareResponse('510300.SH').execution,
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('批量展示动态指标行，并保留部分失败的不可计算原因', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&ids=510300.SH,159915.SZ']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/159915.SZ?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    )
    expect(await screen.findByText('12.34%')).toBeInTheDocument()
    expect(screen.getByText('不可计算')).toBeInTheDocument()
    expect(screen.getByText('样本不足')).toBeInTheDocument()
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_refs: [{ indicator_id: 'total-return', indicator_revision: 2 }],
      targets: [{ kind: 'etf', product_id: '510300.SH' }, { kind: 'etf', product_id: '159915.SZ' }],
      period: '1Y',
      as_of: undefined,
    }))
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH/compare-analysis?kind=etf',
      expect.objectContaining({ method: 'POST', signal: expect.anything() }),
    ))
    const analysisCall = vi.mocked(fetch).mock.calls.find(([input]) =>
      String(input).includes('/510300.SH/compare-analysis'),
    )
    expect(JSON.parse(String(analysisCall?.[1]?.body))).toEqual({
      ranges: {
        performance: { start_date: '2026-01-02', end_date: '2026-01-06' },
        risk: { start_date: '2026-01-02', end_date: '2026-01-06' },
        efficiency: { start_date: '2026-01-02', end_date: '2026-01-06' },
      },
      rolling_window_days: 2,
      management_fee: 0.5,
      custody_fee: 0.1,
    })
    expect(await screen.findByText(/Numba NJIT · 31\/31/)).toBeInTheDocument()
  })

  it.each([cppAotAudit, { ...cppAotAudit, execution_backend: undefined, backend: 'cpp_aot' }])('展示原生 AOT 结果，不要求 NJIT 签名', async (execution) => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      if (String(input).includes('/compare-analysis')) {
        return Promise.resolve({ ok: true, json: async () => ({ ...compareResponse('510300.SH'), execution }) })
      }
      return productFetch(input)
    }))
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&ids=510300.SH,159915.SZ']}><ProductCompare /></MemoryRouter>)
    expect(await screen.findByText('数值引擎：C++ AOT')).toBeInTheDocument()
    expect(screen.queryByText(/无效的.*执行证明/)).not.toBeInTheDocument()
  })

  it('不同指标可独立选择计算区间并按区间拆分引擎请求', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&ids=510300.SH,159915.SZ']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      indicator_refs: [{ indicator_id: 'total-return', indicator_revision: 2 }],
      targets: [{ kind: 'etf', product_id: '510300.SH' }, { kind: 'etf', product_id: '159915.SZ' }],
      period: '1Y',
    })))
    vi.mocked(evaluateCustomIndicators).mockClear()

    await user.click(screen.getByRole('button', { name: /选择比较指标/ }))
    await user.click(screen.getByRole('checkbox', { name: /月度波动率/ }))

    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      indicator_refs: [{ indicator_id: 'total-return', indicator_revision: 2 }, { indicator_id: 'monthly-volatility', indicator_revision: 2 }],
      period: '1Y',
    })))
    await user.selectOptions(await screen.findByLabelText('月度波动率计算区间'), '1M')

    await waitFor(() => {
      expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
        indicator_refs: [{ indicator_id: 'total-return', indicator_revision: 2 }],
        period: '1Y',
      }))
      expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
        indicator_refs: [{ indicator_id: 'monthly-volatility', indicator_revision: 2 }],
        period: '1M',
      }))
    })
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1Y')
    expect(screen.getByLabelText('月度波动率计算区间')).toHaveValue('1M')
  })

  it('支持从资产大类按每个产品自身类别加载混合对比', async () => {
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      const id = url.includes('024011.OF') ? '024011.OF' : '159393.SZ'
      if (url.includes('/compare-analysis')) {
        return Promise.resolve({ ok: true, json: async () => compareResponse(id) })
      }
      return Promise.resolve({ ok: true, json: async () => productResponse(id, id) })
    }))

    render(<MemoryRouter initialEntries={['/product-compare?ids=159393.SZ,024011.OF&kinds=etf,fund']}><ProductCompare /></MemoryRouter>)

    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/159393.SZ?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/024011.OF?kind=fund',
      expect.objectContaining({ signal: expect.anything() }),
    )
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith(expect.objectContaining({
      targets: [
        { kind: 'etf', product_id: '159393.SZ' },
        { kind: 'fund', product_id: '024011.OF' },
      ],
    })))
    expect(listCustomIndicators).toHaveBeenCalledWith({ contextKind: 'single_product' })
  })

  it('从手动构建大类进入后通过返回按钮回到原来源页', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter
        initialEntries={[
          '/manual-construction',
          {
            pathname: '/product-compare',
            search: '?kind=etf&ids=510300.SH,159915.SZ',
            state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
          },
        ]}
        initialIndex={1}
      >
        <Routes>
          <Route path="/manual-construction" element={<div>手动构建大类来源页</div>} />
          <Route path="/product-compare" element={<ProductCompare />} />
        </Routes>
      </MemoryRouter>,
    )

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /返回手动构建大类/ }))
    })
    expect(await screen.findByText('手动构建大类来源页')).toBeInTheDocument()
  })
})

describe('ProductCompare AI 助手接入', () => {
  interface AgentCapture { sessions: Array<Record<string, any>>; messages: Array<Record<string, any>> }

  const SENTINEL = 987654.321

  /** Business calls plus the shared agent endpoints; the compare curves carry a sentinel value. */
  function compareFetch(ids: string[], capture: AgentCapture, gate?: Promise<void>) {
    let sessionCount = 0
    let lastRun: Record<string, any> | null = null
    return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      const path = new URL(url, 'http://localhost').pathname
      if (path === '/api/agent/meta') return { ok: true, json: async () => ({ configured: true, model: 'fixture-model' }) }
      if (path === '/api/agent/sessions' && init?.method === 'POST') {
        sessionCount += 1
        const body = JSON.parse(String(init?.body)); capture.sessions.push(body)
        return { ok: true, json: async () => ({ session_id: `session-${sessionCount}`, session_revision: 0, page_context: body.page_context }) }
      }
      if (path.endsWith('/messages')) {
        const body = JSON.parse(String(init?.body)); capture.messages.push(body)
        if (gate && capture.messages.length === 1) await gate
        lastRun = { run_id: `run-${capture.messages.length}`, session_id: `session-${sessionCount}`, message_id: body.message_id, session_revision: 1, run_revision: 1, status: 'completed', phase: 'thinking',
          response: { session_id: `session-${sessionCount}`, session_revision: 1, reply: { text: `第 ${capture.messages.length} 次回复` } } }
        return { ok: true, status: 202, json: async () => lastRun }
      }
      if (path.endsWith('/events')) return { ok: true, json: async () => ({ items: [], has_more: false, last_seq: 0, next_event_seq: 1 }) }
      if (path.includes('/runs/')) return { ok: true, json: async () => lastRun }
      if (url.includes('/compare-analysis')) {
        const response = compareResponse(ids.find(id => url.includes(id)) ?? ids[0] ?? '510300.SH')
        response.ranges.performance.normalized_nav = [{ date: '2026-01-02', value: SENTINEL }, { date: '2026-01-06', value: SENTINEL }]
        return { ok: true, json: async () => response }
      }
      const id = ids.find(candidate => url.includes(candidate)) ?? ids[0] ?? '510300.SH'
      return { ok: true, json: async () => productResponse(id, id) }
    })
  }

  async function openPanel() {
    fireEvent.click(screen.getByRole('button', { name: '打开 AI 助手' }))
    await waitFor(() => expect(screen.getByRole('textbox', { name: '发送消息' })).toBeEnabled())
    return screen.getByRole('textbox', { name: '发送消息' })
  }

  /** Sync change+click: the panel can re-render while the model metadata resolves. */
  async function sendMessage(text: string) {
    fireEvent.change(screen.getByRole('textbox', { name: '发送消息' }), { target: { value: text } })
    await waitFor(() => expect(screen.getByRole('textbox', { name: '发送消息' })).toHaveValue(text))
    fireEvent.click(screen.getByRole('button', { name: '发送' }))
  }

  beforeEach(() => {
    vi.clearAllMocks()
    testPit.day = null
    sessionStorage.clear()
    window.localStorage.clear()
    vi.stubGlobal('EventSource', undefined)
  })

  it('比较保留全局研究日，矩阵显式截止日独立；指标回包不改变运行上下文', async () => {
    testPit.day = '2019-12-31'
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', compareFetch(['510300.SH'], capture))
    let finish!: (value: any) => void
    vi.mocked(evaluateCustomIndicators).mockImplementation(() => new Promise(resolve => { finish = resolve }))
    render(<MemoryRouter initialEntries={['/product-compare?ids=510300.SH']}><ProductCompare /></MemoryRouter>)
    await screen.findByText('基础信息对比')
    fireEvent.change(screen.getByLabelText('截止日'), { target: { value: '2020-12-31' } })
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenLastCalledWith(expect.objectContaining({ as_of: '2020-12-31' })))
    await openPanel(); await sendMessage('比较与指标日期各是什么？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.messages[0].page_snapshot.sections.request).toMatchObject({ as_of: '2019-12-31', metrics_as_of: '2020-12-31' })
    const frozen = capture.messages[0].page_context
    await act(async () => finish({ results: [{ indicator_id: annualIndicator.id, indicator_revision: 2, target: { kind: 'etf', product_id: '510300.SH' }, value: 0, status: 'ok', window: { effective_as_of: '2020-12-30' } }] }))
    await sendMessage('保留当前口径')
    await waitFor(() => expect(capture.messages).toHaveLength(2))
    expect(capture.messages[1].page_context).toEqual(frozen)
    expect(capture.messages[1].page_snapshot.sections.results.refs.metrics).toMatchObject({ status: 'ready', frozen_request: { requests: [{ as_of: '2020-12-31', indicator_refs: [{ indicator_id: annualIndicator.id, indicator_revision: 2 }] }] } })
    expect(screen.getAllByTestId('chart').length).toBeGreaterThan(0)
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({ results: [] } as any)
  })

  it('发送时冻结混合类型、三个独立区间、滚动窗口与实际费用，且不携带净值曲线数组', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    const ids = ['510300.SH', '024011.OF']
    vi.stubGlobal('fetch', compareFetch(ids, capture))
    render(<MemoryRouter initialEntries={['/product-compare?ids=510300.SH,024011.OF&kinds=etf,fund']}><ProductCompare /></MemoryRouter>)
    await screen.findByText('基础信息对比')
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('/compare-analysis'))).toBe(true))
    expect(screen.getAllByRole('button', { name: '打开 AI 助手' })).toHaveLength(1)

    await openPanel()
    await sendMessage('这两只产品在不同区间下的表现差异是什么？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.messages[0].page_snapshot).toMatchObject({
      version: 1,
      snapshot_id: expect.stringMatching(/^snap-[0-9a-f]{32}$/),
      captured_at: expect.any(String),
      page: 'product-compare',
      sections: {
        request: {
          targets: [
            { kind: 'etf', product_id: '510300.SH', management_fee: 0.5, custody_fee: 0.1 },
            { kind: 'fund', product_id: '024011.OF', management_fee: 0.5, custody_fee: 0.1 },
          ],
          ranges: {
            performance: { start_date: '2026-01-02', end_date: '2026-01-06' },
            risk: { start_date: '2026-01-02', end_date: '2026-01-06' },
            efficiency: { start_date: '2026-01-02', end_date: '2026-01-06' },
          },
          rolling_window_days: 2,
          indicators: [{ indicator_id: 'total-return', indicator_revision: 2, period: '1Y' }],
          as_of: null,
          source: 'actual',
        },
        results: {
          source: 'unverified_client_display',
          displayed_source: 'instruments.products + compare-analysis',
          refs: { compared_products: 2, ranges_resolved: true, indicator_results: expect.any(Number), demo: false },
          note: expect.any(String),
        },
      },
    })
    expect(capture.messages[0].page_context).toMatchObject({
      page: 'product-compare',
      view_state: 'inherit',
      calculation: { context_kind: 'single_product', period: '1Y', as_of: null,
        targets: [{ kind: 'etf', product_id: '510300.SH' }, { kind: 'fund', product_id: '024011.OF' }] },
    })
    expect(capture.messages[0].page_context.page_instance_id).toMatch(/^product-compare:[0-9a-f]{16}$/)
    // 曲线数组与其中的数值完全不进入快照。
    const snapshotText = JSON.stringify(capture.messages[0])
    expect(snapshotText).not.toContain('987654')
    expect(snapshotText).not.toContain('normalized_nav')
    expect(capture.messages[0].page_snapshot.sections.results.refs).not.toHaveProperty('normalized_nav')
  })

  it('demo 预览只登记虚拟来源并在面板中说明，不冒充真实产品计算', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', compareFetch(['DEMO50'], capture))
    render(<MemoryRouter initialEntries={['/product-compare?kind=etf&preview=demo']}><ProductCompare /></MemoryRouter>)
    await screen.findByText('基础信息对比')

    await openPanel()
    expect(screen.getByRole('status')).toHaveTextContent('当前页面是虚拟示例数据')
    await sendMessage('这个页面的数据可靠吗？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    const request = capture.messages[0].page_snapshot.sections.request
    expect(request.source).toBe('demo')
    expect(request.targets.length).toBeGreaterThan(0)
    expect(request.targets[0]).toMatchObject({ kind: 'etf', product_id: 'DEMO50' })
    expect(capture.messages[0].page_snapshot.sections.results.refs.demo).toBe(true)
  })

  it('没有可比较产品时不提交页面快照，聊天仍可用', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', compareFetch([], capture))
    render(<MemoryRouter initialEntries={['/product-compare']}><ProductCompare /></MemoryRouter>)
    await screen.findByText('未能加载任何产品详情。')

    await openPanel()
    await sendMessage('没有产品时能做什么？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.messages[0]).not.toHaveProperty('page_snapshot')
    expect(capture.messages[0].page_context.calculation.targets).toEqual([])
  })

  it('切换对比对象后进入独立实例，旧实例的迟到回复不能进入新实例', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    let release: (() => void) | undefined
    const gate = new Promise<void>(resolve => { release = resolve })
    vi.stubGlobal('fetch', compareFetch(['510300.SH', '159915.SZ'], capture, gate))
    function SwitchTargets() {
      const navigate = useNavigate()
      return <button type="button" onClick={() => navigate('/product-compare?ids=159915.SZ&kind=etf')}>切换对比对象</button>
    }
    render(
      <MemoryRouter initialEntries={['/product-compare?ids=510300.SH&kind=etf']}>
        <ProductCompare />
        <SwitchTargets />
      </MemoryRouter>,
    )
    await screen.findByText('基础信息对比')

    await openPanel()
    await sendMessage('第一组产品的问题')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    const firstInstance = capture.messages[0].page_context.page_instance_id

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '切换对比对象' }))
    await waitFor(() => expect(screen.queryByText('第一组产品的问题')).not.toBeInTheDocument())
    await act(async () => { release?.() })
    expect(screen.queryByText('第 1 次回复')).not.toBeInTheDocument()

    await openPanel()
    await sendMessage('第二组产品的问题')
    await waitFor(() => expect(capture.messages).toHaveLength(2))
    expect(capture.messages[1].page_context.page_instance_id).not.toBe(firstInstance)
    expect(capture.messages[1].page_snapshot.sections.request.targets)
      .toEqual([{ kind: 'etf', product_id: '159915.SZ', management_fee: 0.5, custody_fee: 0.1 }])
  })
})
