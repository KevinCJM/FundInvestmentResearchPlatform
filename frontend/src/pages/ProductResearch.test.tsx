import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductResearch from './ProductResearch'
import { evaluateCustomIndicators, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition } from '../services/customIndicators'
vi.mock('../app/ResearchContext', () => ({ useResearchDay: () => null, useResearchContextIdentity: () => 'fixture-pit-off' }))

vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn().mockResolvedValue({ periods: [{ value: '1Y', label: '近 1 年', description: '运行周期' }] }),
  indicatorPeriodLabel: (period: string) => period,
  listCustomIndicators: vi.fn().mockResolvedValue({ items: [], total: 0 }),
}))

const fixedExecution = {
  execution_backend: 'numba_njit_fixed_signature',
  nopython: true,
  object_mode: 0,
  python_fallback: 0,
  request_time_compilation: 0,
  kernel_signatures: { product_summary_kernel: ['fixed'] },
}

const minimalPayload = (overrides: Record<string, unknown> = {}) => ({
  items: [{ ts_code: '510300.SH', name: '沪深300ETF', list_date: '2012-05-28' }],
  page: 1, page_size: 10, total: 1,
  summary: { universe_total: 1, filtered_total: 1, active_count: 1, active_rate: 1 },
  available_filters: { fund_type: [], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] },
  sort_by: 'issue_amount', sort_dir: 'desc',
  execution: fixedExecution,
  ...overrides,
})

function CurrentLocation() {
  const location = useLocation()
  return <output data-testid="location">{location.pathname}{location.search}</output>
}

describe('ProductResearch', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [], total: 0 })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{ ts_code: '510300.SH', name: '沪深300ETF', type: 'ETF', fund_type: '股票型', invest_type: '宽基', qdii_type: '非QDII', market: '上交所', status: '上市', management: '华泰柏瑞', custodian: '中国银行', issue_amount: 100, snapshot_values: { current_size: 12345.67, return_1y: 0.0821 }, snapshot_value_dates: { current_size: '2026-08-28', return_1y: '2026-08-31' }, m_fee: 0.5, c_fee: 0.1, list_date: '2012-05-28' }],
        page: 1, page_size: 10, total: 1,
        summary: { universe_total: 1, filtered_total: 1, active_count: 1, active_rate: 1, recent_listings_12m: 0, avg_m_fee: 0.5, avg_c_fee: 0.1, total_issue_amount: 100, median_issue_amount: 100, unique_managements: 1 },
        snapshot_metric_fields: [
          { field: 'current_size', label: '当前规模', data_type: 'number', source: 'instrument_metrics_snapshot', metric_source: 'system_derived', metric_source_label: '系统衍生指标', metric_type: 'scale', metric_type_label: '规模指标', unit: 'project_normalized_wan', description: 'ETF 总份额 × 同期单位净值', available: true },
          { field: 'return_1y', label: '累计收益率（1Y）', data_type: 'number', source: 'instrument_metrics_snapshot', metric_source: 'built_in', metric_source_label: '内置指标', metric_type: 'return', metric_type_label: '收益型指标', unit: 'ratio', description: '指标中心预计算', available: true },
        ],
        snapshot: { status: 'ready', as_of: '2026-08-31' },
        available_filters: { fund_type: [], type: [], invest_type: [], qdii_type: [{ value: '非QDII', label: '非QDII', count: 1 }], market: [], status: [], management: [], custodian: [] }, sort_by: 'issue_amount', sort_dir: 'desc',
        execution: fixedExecution,
      }),
    }))
  })

  it('指标视图只批量计算当前分页产品并按协议格式化结果', async () => {
    const indicator: IndicatorDefinition = {
      id: 'page-return', revision: 1, source: 'custom', read_only: false, name: '当前页收益',
      description: '当前分页真实净值收益', expression: 'product(returns + 1) - 1', periods: ['1Y'],
      unit: '%', display_format: 'percent', precision: 2, direction: 'higher_better',
      annual_risk_free_rate_percent: 1.5, context_kind: 'single_product',
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [indicator], total: 1 })
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({
      results: [{
        indicator_id: indicator.id, indicator_revision: 1, indicator_name: indicator.name,
        target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y',
        value: 0.0183, status: 'ok', warnings: [],
        window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: '2025-01-06', end_date: '2026-01-06', observation_count: 250, data_latest_date: '2026-01-06' },
      }],
      summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 },
      execution: fixedExecution,
    } as any)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(screen.getByText('占比 100%')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '指标分析' }))
    expect(await screen.findByText('1.83%')).toBeInTheDocument()
    expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_refs: [{ indicator_id: 'page-return', indicator_revision: 1 }],
      targets: [{ kind: 'etf', product_id: '510300.SH' }],
      period: '1Y',
      as_of: undefined,
    })
    expect(screen.getByLabelText('当前页收益计算区间')).toHaveValue('1Y')
    expect(screen.queryByLabelText('计算周期')).not.toBeInTheDocument()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('按 Tushare 字段语义展示基金类型和投资类型筛选', async () => {
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(screen.getByRole('button', { name: '投资类型 全部' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '基金类型 全部' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'QDII 属性 全部' })).toBeInTheDocument()
    expect(screen.getByText('非QDII')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /机构类型/ })).not.toBeInTheDocument()
  })

  it('默认每页展示 10 个产品并允许调整每页数量', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('page_size=10'))).toBe(true)

    await user.selectOptions(screen.getByLabelText('产品研究每页产品数量'), '20')
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('page_size=20'))).toBe(true))
  })

  it('提示产品名称可进入单产品研究并提供明确链接语义', async () => {
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(screen.getByText('提示：点击产品名称可进入单产品研究页面')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: '进入沪深300ETF的单产品研究页面' })).toHaveAttribute(
      'href',
      '/product-research/products/510300.SH?kind=etf',
    )
  })

  it('按来源、类型和名称逐级选择快照指标', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(screen.queryByRole('columnheader', { name: /当前规模/ })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /展示快照指标/ }))
    await user.selectOptions(screen.getByLabelText('快照指标来源'), 'system_derived')
    expect(screen.getByRole('option', { name: '规模指标' })).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('快照指标类型'), 'scale')
    await user.selectOptions(screen.getByLabelText('快照指标名称'), 'current_size')

    expect(await screen.findByRole('columnheader', { name: /当前规模/ })).toBeInTheDocument()
    expect(screen.getByText('1.23 亿')).toBeInTheDocument()
    expect(screen.getByText('快照截至 2026-08-28')).toBeInTheDocument()
    expect(screen.getByText('发行披露口径（非当前 AUM）')).toBeInTheDocument()
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('snapshot_metric=current_size'))).toBe(true)
    expect(screen.queryByRole('button', { name: /在指标中心分析/ })).not.toBeInTheDocument()
  })

  it('从 URL 初始化场外基金、筛选和成立日排序', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{ ts_code: '000001.OF', name: '示例场外基金', type: '混合型', fund_type: '混合型', invest_type: '主动型', market: '场外', status: '存续', management: '示例基金公司', custodian: '示例托管行', issue_amount: 100, m_fee: 0.5, c_fee: 0.1, found_date: '2001-01-01' }],
        page: 1, page_size: 10, total: 1,
        summary: { universe_total: 1, filtered_total: 1, active_count: 1, active_rate: 1, recent_listings_12m: 0, avg_m_fee: 0.5, avg_c_fee: 0.1, total_issue_amount: 100, median_issue_amount: 100, unique_managements: 1 },
        available_filters: { fund_type: [{ value: '混合型', label: '混合型', count: 1 }], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] }, sort_by: 'found_date', sort_dir: 'desc',
        execution: fixedExecution,
      }),
    }))

    render(
      <MemoryRouter initialEntries={['/research?kind=fund&fund_type=%E6%B7%B7%E5%90%88%E5%9E%8B&sort_by=found_date&sort_dir=desc']}>
        <ProductResearch /><CurrentLocation />
      </MemoryRouter>,
    )

    expect(await screen.findByText('示例场外基金')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '场外公募基金' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: /成立日/ })).toBeInTheDocument()
    await waitFor(() => {
      const calledUrl = String(vi.mocked(fetch).mock.calls[0][0])
      expect(calledUrl).toContain('kind=fund')
      expect(calledUrl).toContain('fund_type=%E6%B7%B7%E5%90%88%E5%9E%8B')
      expect(calledUrl).toContain('sort_by=found_date')
      expect(calledUrl).toContain('sort_dir=desc')
    })
  })

  it('添加快照指标条件并使用指定比较运算写入查询 URL', async () => {
    const responsePayload = {
      items: [{ ts_code: '510300.SH', name: '沪深300ETF', list_date: '2012-05-28' }],
      page: 1, page_size: 10, total: 1,
      summary: { universe_total: 1, filtered_total: 1, active_count: 1, active_rate: 1 },
      available_filters: { fund_type: [], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] },
      condition_fields: [
        { field: 'list_date', label: '上市日期', data_type: 'date', unit_label: null, input_scale: 1, source: 'fund_basic', available: true },
        { field: 'return_1y', label: '近1年收益率', data_type: 'number', unit_label: '%', input_scale: 100, source: 'instrument_metrics_snapshot', available: true },
      ],
      condition_operators: [
        { value: 'gte', label: '大于等于', symbol: '≥' },
        { value: 'gt', label: '大于', symbol: '>' },
      ],
      snapshot: { status: 'ready', as_of: '2026-08-31' },
      sort_by: 'issue_amount', sort_dir: 'desc',
      execution: fixedExecution,
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => responsePayload }))
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /><CurrentLocation /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    await user.selectOptions(screen.getByLabelText('筛选字段'), 'return_1y')
    await user.selectOptions(screen.getByLabelText('比较方式'), 'gt')
    await user.type(screen.getByLabelText('筛选值'), '10')
    await user.click(screen.getByRole('button', { name: '添加条件' }))

    expect(await screen.findByText('近1年收益率 > 10%')).toBeInTheDocument()
    await waitFor(() => {
      const urls = vi.mocked(fetch).mock.calls.map((call) => String(call[0]))
      expect(urls.some((url) => url.includes('condition=return_1y%7Cgt%7C10'))).toBe(true)
    })
    expect(screen.getByTestId('location')).toHaveTextContent('condition=return_1y%7Cgt%7C10')
  })

  it('支持本页全选和当前筛选结果全选，并限制批量分析动作', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [
          { ts_code: '510001.SH', name: '产品一', list_date: '2020-01-01' },
          { ts_code: '510002.SH', name: '产品二', list_date: '2021-01-01' },
        ],
        page: 1, page_size: 10, total: 12,
        summary: { universe_total: 12, filtered_total: 12, active_count: 12, active_rate: 1 },
        available_filters: { fund_type: [], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] },
        condition_fields: [{ field: 'list_date', label: '上市日期', data_type: 'date', source: 'fund_basic', available: true }],
        condition_operators: [], snapshot: { status: 'ready' }, sort_by: 'issue_amount', sort_dir: 'desc',
        execution: fixedExecution,
      }),
    }))
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('产品一')
    await user.click(screen.getByRole('checkbox', { name: '本页全选' }))
    expect(screen.getByRole('checkbox', { name: '选择 产品一' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: '选择 产品二' })).toBeChecked()
    expect(screen.getByText('已选 2')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '全选 12 条' }))
    expect(screen.getByText('已选 12')).toBeInTheDocument()
    expect(screen.getByText('已选择全部符合筛选条件的产品')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /产品对比/ })).toBeDisabled()
    expect(screen.getByText('全选筛选结果是逻辑选择；请取消全选后手动选择最多 10 个产品进行对比。')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: '选择 产品一' }))
    expect(screen.getByText('已选 11')).toBeInTheDocument()
    expect(screen.getByText('已选择全部符合筛选条件的产品，排除 1 个')).toBeInTheDocument()
  })

  it('加载失败时给出重试入口，点击后重新请求', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 500, json: async () => ({ detail: '后端暂时不可用' }) })
      .mockResolvedValue({ ok: true, json: async () => minimalPayload() })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    expect(await screen.findByRole('alert')).toHaveTextContent('后端暂时不可用')
    await user.click(screen.getByRole('button', { name: '重试' }))

    expect(await screen.findByText('沪深300ETF')).toBeInTheDocument()
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  })

  it('筛选无结果时说明原因并给出清空筛选的下一步', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => minimalPayload({ items: [], total: 0, summary: { universe_total: 8, filtered_total: 0 } }),
    }))
    render(<MemoryRouter initialEntries={['/research?q=%E6%97%A0%E6%AD%A4%E4%BA%A7%E5%93%81']}><ProductResearch /></MemoryRouter>)

    expect(await screen.findByText('没有符合当前筛选条件的ETF')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '清空筛选条件' })).toBeInTheDocument()
  })

  it('表头声明排序方向，并支持直接跳转页码', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => minimalPayload({ total: 25 }) })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    expect(screen.getByRole('columnheader', { name: /发行规模/ })).toHaveAttribute('aria-sort', 'descending')
    await user.click(screen.getByRole('button', { name: '按发行规模排序' }))
    expect(screen.getByRole('columnheader', { name: /发行规模/ })).toHaveAttribute('aria-sort', 'ascending')

    const jump = screen.getByLabelText('跳转到页码')
    await user.clear(jump)
    await user.type(jump, '3{Enter}')
    await waitFor(() => expect(fetchMock.mock.calls.some((call) => String(call[0]).includes('page=3'))).toBe(true))
  })
})

describe('ProductResearch AI 助手接入', () => {
  interface AgentCapture { sessions: Array<Record<string, any>>; messages: Array<Record<string, any>> }

  /** One router for the page's own business calls and the shared agent endpoints. */
  function researchFetch(payload: unknown, capture: AgentCapture, gate?: Promise<void>) {
    let sessionCount = 0
    let lastRun: Record<string, any> | null = null
    return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const path = new URL(String(input), 'http://localhost').pathname
      if (path === '/api/instruments/products') return { ok: true, json: async () => payload }
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
      return { ok: true, json: async () => ({ session_id: `session-${sessionCount}`, session_revision: 1, next_event_seq: 1, events: [], messages: [], active_run: lastRun }) }
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
    sessionStorage.clear()
    vi.stubGlobal('EventSource', undefined)
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [], total: 0 })
  })

  it('发送时冻结当前页面请求：当前分析批次、PIT 研究日与目录显示引用，且不含原始数值或数组', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    vi.stubGlobal('fetch', researchFetch(minimalPayload({
      pit: { as_of: '2026-08-31', snapshot_is_hindsight: true, warnings: ['快照按全部已下载数据计算'] },
    }), capture))
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    expect(screen.getAllByRole('button', { name: '打开 AI 助手' })).toHaveLength(1)
    // 面板关闭时不得发起任何助手请求。
    expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('/api/agent'))).toBe(false)

    await openPanel()
    await sendMessage('这个列表的口径是什么？')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.sessions).toEqual([{ page_context: {
      page: 'product-research', page_instance_id: 'product-research:etf', context_revision: 1, view_state: 'inherit',
      calculation: { context_kind: 'single_product', targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1Y', as_of: null },
    } }])
    expect(capture.messages[0].page_snapshot).toMatchObject({
      version: 1,
      snapshot_id: expect.stringMatching(/^snap-[0-9a-f]{32}$/),
      captured_at: expect.any(String),
      page: 'product-research',
      sections: {
        request: {
          kind: 'etf', q: '', fund_type: [], fund_category: [], invest_type: [], market: [], status: [],
          management: [], custodian: [], qdii_type: [], page: 1, page_size: 10, sort_by: 'issue_amount', sort_dir: 'desc',
          conditions: [], snapshot_metrics: [], as_of: null,
          targets: [{ kind: 'etf', product_id: '510300.SH' }], batch_offset: 0,
          visible_count: 1, selected_count: 0, selection_mode: 'current_page', excluded_ids: [], indicators: [], view_mode: 'overview',
        },
        results: {
          source: 'unverified_client_display',
          displayed_source: 'instruments.products + evaluate-custom-indicators',
          refs: { page_items: 1, total: 1, metric_results: 0, view_mode: 'basic', pit: { as_of: '2026-08-31', snapshot_is_hindsight: true } },
          note: expect.any(String),
        },
      },
    })
    // 页面结果引用只登记来源与计数：没有原始快照值，也没有完整数组。
    const snapshotText = JSON.stringify(capture.messages[0])
    expect(snapshotText).not.toContain('987654')
    expect(snapshotText).not.toContain('snapshot_values')
    expect(capture.messages[0].page_snapshot.sections.request).not.toHaveProperty('snapshot_values')
    expect(capture.messages[0].page_snapshot.sections.results.refs).not.toHaveProperty('values')
  })

  it('区分当前页、已选与全选声明，并冻结筛选、条件、分页和选择修订', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    const items = Array.from({ length: 12 }, (_, index) => ({
      ts_code: `51000${index}.SH`, name: `产品${index + 1}`, list_date: '2020-01-01',
    }))
    vi.stubGlobal('fetch', researchFetch(minimalPayload({
      items, total: 42, page_size: 20,
      pit: { as_of: null, snapshot_is_hindsight: false },
      condition_fields: [{ field: 'return_1y', label: '近1年收益率', data_type: 'number', unit_label: '%', input_scale: 1, source: 'instrument_metrics_snapshot', available: true }],
    }), capture))
    render(
      <MemoryRouter initialEntries={['/research?kind=etf&fund_type=%E8%82%A1%E7%A5%A8%E5%9E%8B&condition=return_1y%7Cgte%7C10&page_size=20']}>
        <ProductResearch />
      </MemoryRouter>,
    )
    await screen.findByText('产品1')

    await openPanel()
    await sendMessage('先解释一下当前页')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    const first = capture.messages[0].page_snapshot.sections.request
    // 当前分页 12 项，助手只冻结前 10 项这个明确批次；选择数量不冒充已计算。
    expect(first.targets).toHaveLength(10)
    expect(first.visible_count).toBe(12)
    expect(first.selection_mode).toBe('current_page')
    expect(first.page_size).toBe(20)
    expect(first.fund_type).toEqual(['股票型'])
    expect(first.conditions).toEqual([{ field: 'return_1y', operator: 'gte', value: '10' }])

    const user = userEvent.setup()
    await user.click(screen.getByRole('checkbox', { name: '选择 产品1' }))
    await user.click(screen.getByRole('checkbox', { name: '选择 产品2' }))
    await sendMessage('再看已选的两只')
    await waitFor(() => expect(capture.messages).toHaveLength(2))
    expect(capture.messages[1].page_snapshot.sections.request.selection_mode).toBe('selected')
    expect(capture.messages[1].page_snapshot.sections.request.selected_count).toBe(2)
    expect(capture.messages[1].page_snapshot.sections.request.excluded_ids).toEqual([])

    await user.click(screen.getByRole('button', { name: '全选 42 条' }))
    await user.click(screen.getByRole('checkbox', { name: '选择 产品1' }))
    await sendMessage('全选后排除一只')
    await waitFor(() => expect(capture.messages).toHaveLength(3))
    const allMatching = capture.messages[2].page_snapshot.sections.request
    expect(allMatching.selection_mode).toBe('all_matching')
    expect(allMatching.selected_count).toBe(41)
    expect(allMatching.excluded_ids).toEqual(['510000.SH'])
    expect(allMatching.targets).toHaveLength(10)
    // 选择变化递增上下文修订，页面对象本身不换实例。
    expect(capture.messages[2].page_context.context_revision).toBeGreaterThan(capture.messages[1].page_context.context_revision)
    expect(capture.messages[2].page_context.page_instance_id).toBe('product-research:etf')
  })

  it('指标视图冻结所选指标及其锁定版本和各自周期，不发原始快照值', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    const indicator: IndicatorDefinition = {
      id: 'page-return', revision: 3, source: 'custom', read_only: false, name: '当前页收益',
      description: '当前分页真实净值收益', expression: 'product(returns + 1) - 1', periods: ['1Y'],
      unit: '%', display_format: 'percent', precision: 2, direction: 'higher_better',
      annual_risk_free_rate_percent: 1.5, context_kind: 'single_product',
      created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
    }
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [indicator], total: 1 })
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({ results: [], summary: { total: 0, ok: 0, warning: 0, error: 0 }, cache: { hits: 0, misses: 0 }, execution: fixedExecution } as any)
    vi.stubGlobal('fetch', researchFetch(minimalPayload({ pit: { as_of: null, snapshot_is_hindsight: false } }), capture))
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)
    await screen.findByText('沪深300ETF')
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '指标分析' }))
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalled())

    await openPanel()
    await sendMessage('用当前指标解释一下这一列')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    const request = capture.messages[0].page_snapshot.sections.request
    expect(request.view_mode).toBe('metrics')
    expect(request.indicators).toEqual([{ indicator_id: 'page-return', indicator_revision: 3, period: '1Y' }])
    expect(JSON.stringify(capture.messages[0])).not.toContain('0.0183')
  })

  it('切换产品类型后进入独立对话实例，旧实例的迟到回复不能进入新实例', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    let release: (() => void) | undefined
    const gate = new Promise<void>(resolve => { release = resolve })
    vi.stubGlobal('fetch', researchFetch(minimalPayload(), capture, gate))
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)
    await screen.findByText('沪深300ETF')

    await openPanel()
    await sendMessage('ETF 的问题')
    await waitFor(() => expect(capture.messages).toHaveLength(1))

    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: '场外公募基金' }))
    await waitFor(() => expect(vi.mocked(fetch).mock.calls.some((call) => String(call[0]).includes('kind=fund'))).toBe(true))
    expect(screen.queryByText('ETF 的问题')).not.toBeInTheDocument()

    await act(async () => { release?.() })
    expect(screen.queryByText('第 1 次回复')).not.toBeInTheDocument()

    await openPanel()
    expect(screen.queryByText('ETF 的问题')).not.toBeInTheDocument()
    await sendMessage('基金的问题')
    await waitFor(() => expect(capture.messages).toHaveLength(2))
    expect(capture.messages[1].page_context.page_instance_id).toBe('product-research:fund')
    expect(capture.messages[1].page_snapshot.sections.request.kind).toBe('fund')
  })

  it('新产品类型加载中禁用发送，已取消列表的迟到JSON不能把旧产品混入新批次', async () => {
    const capture: AgentCapture = { sessions: [], messages: [] }
    const fallback = researchFetch(minimalPayload(), capture)
    let oldList!: (value: any) => void, newList!: (value: any) => void
    const oldResponse = new Promise(resolve => { oldList = resolve })
    const newResponse = new Promise(resolve => { newList = resolve })
    vi.stubGlobal('fetch', vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.startsWith('/api/instruments/products?')) return Promise.resolve({ ok: true, json: () => url.includes('kind=fund') ? newResponse : oldResponse })
      return fallback(input, init)
    }))
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)
    fireEvent.click(screen.getByRole('button', { name: '场外公募基金' }))
    await openPanel()
    fireEvent.change(screen.getByRole('textbox', { name: '发送消息' }), { target: { value: '新产品' } })
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled()
    await act(async () => newList(minimalPayload({ items: [{ ts_code: '000001.OF', name: '新基金' }] })))
    await screen.findByText('新基金')
    await act(async () => oldList(minimalPayload()))
    expect(screen.queryByText('沪深300ETF')).not.toBeInTheDocument()
    await sendMessage('只研究当前基金')
    await waitFor(() => expect(capture.messages).toHaveLength(1))
    expect(capture.messages[0].page_snapshot.sections.request.targets).toEqual([{ kind: 'fund', product_id: '000001.OF' }])
  })
})
