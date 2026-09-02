import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, useLocation } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductResearch from './ProductResearch'
import { evaluateCustomIndicators, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition } from '../services/customIndicators'

vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn().mockResolvedValue({ periods: [{ value: '1Y', label: '近 1 年', description: '运行周期' }] }),
  indicatorPeriodLabel: (period: string) => period,
  listCustomIndicators: vi.fn().mockResolvedValue({ items: [], total: 0 }),
}))

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
        items: [{ ts_code: '510300.SH', name: '沪深300ETF', type: 'ETF', fund_type: '股票型', invest_type: '宽基', market: '上交所', status: '上市', management: '华泰柏瑞', custodian: '中国银行', issue_amount: 100, m_fee: 0.5, c_fee: 0.1, list_date: '2012-05-28' }],
        page: 1, page_size: 10, total: 1,
        summary: { universe_total: 1, filtered_total: 1, active_count: 1, recent_listings_12m: 0, avg_m_fee: 0.5, avg_c_fee: 0.1, total_issue_amount: 100, median_issue_amount: 100, unique_managements: 1 },
        available_filters: { fund_type: [], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] }, sort_by: 'issue_amount', sort_dir: 'desc',
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
    } as any)
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('沪深300ETF')
    await user.click(screen.getByRole('button', { name: '切换到研究指标视图' }))
    expect(await screen.findByText('1.83%')).toBeInTheDocument()
    expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['page-return'],
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

  it('携带产品种类和所选产品标识进入指标中心', async () => {
    const user = userEvent.setup()
    await act(async () => { render(<MemoryRouter initialEntries={['/research']}><ProductResearch /><CurrentLocation /></MemoryRouter>) })

    await waitFor(() => expect(screen.getByText('沪深300ETF')).toBeInTheDocument())
    await act(async () => {
      await user.click(screen.getByRole('checkbox'))
      await user.click(screen.getByRole('button', { name: /在指标中心分析/ }))
    })

    expect(screen.getByTestId('location')).toHaveTextContent('/indicator-studio?kind=etf&ids=510300.SH')
  })

  it('从 URL 初始化场外基金、筛选和成立日排序', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{ ts_code: '000001.OF', name: '示例场外基金', type: '混合型', fund_type: '混合型', invest_type: '主动型', market: '场外', status: '存续', management: '示例基金公司', custodian: '示例托管行', issue_amount: 100, m_fee: 0.5, c_fee: 0.1, found_date: '2001-01-01' }],
        page: 1, page_size: 10, total: 1,
        summary: { universe_total: 1, filtered_total: 1, active_count: 1, recent_listings_12m: 0, avg_m_fee: 0.5, avg_c_fee: 0.1, total_issue_amount: 100, median_issue_amount: 100, unique_managements: 1 },
        available_filters: { fund_type: [{ value: '混合型', label: '混合型', count: 1 }], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] }, sort_by: 'found_date', sort_dir: 'desc',
      }),
    }))

    render(
      <MemoryRouter initialEntries={['/research?kind=fund&fund_type=%E6%B7%B7%E5%90%88%E5%9E%8B&sort_by=found_date&sort_dir=desc']}>
        <ProductResearch /><CurrentLocation />
      </MemoryRouter>,
    )

    expect(await screen.findByText('示例场外基金')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '场外公募基金' })).toHaveClass('text-emerald-600')
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
      summary: { universe_total: 1, filtered_total: 1, active_count: 1 },
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
        summary: { universe_total: 12, filtered_total: 12, active_count: 12 },
        available_filters: { fund_type: [], type: [], invest_type: [], market: [], status: [], management: [], custodian: [] },
        condition_fields: [{ field: 'list_date', label: '上市日期', data_type: 'date', source: 'fund_basic', available: true }],
        condition_operators: [], snapshot: { status: 'ready' }, sort_by: 'issue_amount', sort_dir: 'desc',
      }),
    }))
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/research']}><ProductResearch /></MemoryRouter>)

    await screen.findByText('产品一')
    await user.click(screen.getByRole('button', { name: '本页全选' }))
    expect(screen.getByRole('checkbox', { name: '选择 产品一' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: '选择 产品二' })).toBeChecked()
    expect(screen.getByText('已选 2')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '全选 12 条' }))
    expect(screen.getByText('已选 12')).toBeInTheDocument()
    expect(screen.getByText('已选择全部符合筛选条件的产品')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /产品对比/ })).toBeDisabled()
    expect(screen.getByText('全选筛选结果是逻辑选择；请取消全选后手动选择最多 10 个产品进行分析。')).toBeInTheDocument()

    await user.click(screen.getByRole('checkbox', { name: '选择 产品一' }))
    expect(screen.getByText('已选 11')).toBeInTheDocument()
    expect(screen.getByText('已选择全部符合筛选条件的产品，排除 1 个')).toBeInTheDocument()
  })
})
