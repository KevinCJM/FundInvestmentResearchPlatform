import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductDetail from './ProductDetail'
import { evaluateCustomIndicators, getCustomIndicatorMeta, listCustomIndicators } from '../services/customIndicators'
import type { IndicatorDefinition } from '../services/customIndicators'

vi.mock('echarts-for-react', () => ({
  default: ({ option }: {
    option?: {
      grid?: unknown | Array<{ left?: string }>;
      xAxis?: { type?: string; name?: string } | Array<{ type?: string; name?: string }>;
      yAxis?: { type?: string } | Array<{ type?: string }>;
      series?: Array<{ name?: string; data?: unknown[] }>;
    };
  }) => {
    const xAxis = Array.isArray(option?.xAxis) ? option.xAxis[0] : option?.xAxis
    const yAxis = Array.isArray(option?.yAxis) ? option.yAxis[0] : option?.yAxis
    const terminalXAxis = Array.isArray(option?.xAxis) ? option.xAxis[1] : undefined
    const grids = Array.isArray(option?.grid) ? option.grid : option?.grid ? [option.grid] : []
    const terminalHistogram = (option?.series ?? []).find((series) => series.name === '期末净值直方图')
    const medianPath = (option?.series ?? []).find((series) => series.name === '中位路径')
    const terminalHistogramTotal = (terminalHistogram?.data ?? []).reduce((sum, item) => {
      const count = Array.isArray(item) ? Number(item[0]) : 0
      return sum + (Number.isFinite(count) ? count : 0)
    }, 0)
    return <div
      data-testid="chart"
      data-series={(option?.series ?? []).map((series) => series.name).filter(Boolean).join(',')}
      data-x-axis-type={xAxis?.type}
      data-y-axis-type={yAxis?.type}
      data-grid-count={grids.length}
      data-terminal-grid-left={(grids[1] as { left?: string } | undefined)?.left}
      data-terminal-axis-name={terminalXAxis?.name}
      data-terminal-histogram-total={terminalHistogramTotal}
      data-simulation-start={medianPath ? String(Number(medianPath.data?.[0])) : undefined}
    />
  },
}))
vi.mock('../services/customIndicators', () => ({
  evaluateCustomIndicators: vi.fn(),
  getCustomIndicatorMeta: vi.fn(),
  indicatorPeriodLabel: (period: string) => period,
  indicatorsForContext: (items: IndicatorDefinition[], context: string) => items.filter((item) => (item.context_kind ?? 'single_product') === context),
  listCustomIndicators: vi.fn(),
}))

const percentIndicator: IndicatorDefinition = {
  id: 'total-return', revision: 2, source: 'custom', read_only: false,
  name: '区间累计收益', description: '基于真实净值的累计收益。', expression: '\\left(\\prod\\left(\\mathbf{r}+1\\right)\\right)-1',
  periods: ['1Y'], unit: '%', display_format: 'percent', precision: 2,
  direction: 'higher_better', annual_risk_free_rate_percent: 1.5,
  created_at: '2026-01-01T00:00:00Z', updated_at: '2026-01-01T00:00:00Z',
}

const portfolioIndicator: IndicatorDefinition = {
  ...percentIndicator,
  id: 'portfolio-volatility',
  name: '组合波动率',
  context_kind: 'portfolio',
}

describe('ProductDetail custom indicators', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    window.localStorage.clear()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        product_id: '510300.SH', name: '沪深300ETF', management: '测试管理人', status: '上市',
        base_info: { ts_code: '510300.SH', fund_type: 'ETF', list_date: '2012-05-28', delist_date: '2026-12-31' },
        metrics: {
          issue_amount: 100,
          current_size: 123456.78,
          current_size_as_of: '2026-06-30',
          current_size_source: 'instrument_metrics_snapshot',
          current_share: 100000,
          current_unit_nav: 1.2345678,
          m_fee: 0.5,
          c_fee: 0.1,
        },
        timeseries: Array.from({ length: 25 }, (_, index) => {
          const close = 3 + index * 0.002 + ((index % 5) - 2) * 0.005
          return {
            date: new Date(Date.UTC(2026, 0, 2 + index)).toISOString().slice(0, 10),
            open: close - 0.002,
            high: close + 0.01,
            low: close - 0.01,
            close,
            volume: 1000 + index * 10,
          }
        }),
      }),
    }))
    vi.mocked(listCustomIndicators).mockResolvedValue({ items: [percentIndicator, portfolioIndicator], total: 2 })
    vi.mocked(getCustomIndicatorMeta).mockResolvedValue({ periods: [{ value: '1M', label: '近 1 月', description: '运行周期' }, { value: '1Y', label: '近 1 年', description: '运行周期' }] } as any)
    vi.mocked(evaluateCustomIndicators).mockResolvedValue({
      results: [{
        indicator_id: percentIndicator.id, indicator_revision: percentIndicator.revision, indicator_name: percentIndicator.name,
        target: { kind: 'etf', product_id: '510300.SH', name: '沪深300ETF' }, period: '1Y', value: 0.1234, status: 'ok', warnings: [],
        window: { requested_as_of: null, effective_as_of: '2026-01-06', start_date: '2025-01-06', end_date: '2026-01-06', observation_count: 250, data_latest_date: '2026-01-06' },
      }], summary: { total: 1, ok: 1, warning: 0, error: 0 }, cache: { hits: 0, misses: 1 },
    })
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('展示已保存指标的当前值、百分比格式并可进入指标中心', async () => {
    const user = userEvent.setup()
    render(<MemoryRouter initialEntries={['/product/510300.SH?kind=etf']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByText('自定义研究指标')).toBeInTheDocument()
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(
      '/api/instruments/products/510300.SH?kind=etf',
      expect.objectContaining({ signal: expect.anything() }),
    ))
    expect(await screen.findByText('12.34%')).toBeInTheDocument()
    expect(screen.getByLabelText('上市日期：2012-05-28')).toBeInTheDocument()
    expect(screen.getByLabelText('退市日期：2026-12-31')).toBeInTheDocument()
    expect(screen.getByText('当前规模')).toBeInTheDocument()
    expect(screen.getByText('12.35 亿')).toBeInTheDocument()
    expect(screen.getByText('快照截至 2026-06-30 · 100,000 万份 × 1.23 元/份')).toBeInTheDocument()
    expect(screen.getByLabelText('统计区间')).toHaveValue('ALL')
    expect(screen.getByRole('heading', { name: '未来虚拟净值模拟' })).toBeInTheDocument()
    expect(screen.getByText(/所有路径统一从虚拟净值 1\.0000 出发/)).toBeInTheDocument()
    expect(screen.getByLabelText('模拟未来区间')).toHaveValue('252')
    expect(screen.getByLabelText('模拟路径数')).toHaveValue('500')
    expect(screen.getByLabelText('Bootstrap 平均区块长度')).toHaveValue('20')
    expect(screen.getByLabelText('目标期末收益率')).toHaveValue(5)
    expect(screen.getByRole('radio', { name: '参数化蒙特卡洛' })).toBeChecked()
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).not.toBeChecked()
    const combinedChart = screen.getByTestId('monte-carlo-combined-chart')
    const combinedChartRenderer = combinedChart.querySelector('[data-testid="chart"]')
    expect(combinedChart).toHaveAccessibleName('参数化蒙特卡洛（偏度/峰度校准）：路径与期末净值概率分布组合图')
    expect(screen.getByText(/历史对数收益偏度/)).toBeInTheDocument()
    expect(screen.getByText(/四矩校准已匹配|四矩近似校准/)).toBeInTheDocument()
    expect(combinedChartRenderer).toHaveAttribute('data-grid-count', '2')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-grid-left', '83%')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-axis-name', '路径数')
    expect(combinedChartRenderer).toHaveAttribute('data-terminal-histogram-total', '500')
    expect(combinedChartRenderer).toHaveAttribute('data-simulation-start', '1')
    expect(combinedChartRenderer).toHaveAttribute('data-series', expect.stringContaining('期末净值直方图'))
    expect(combinedChartRenderer).toHaveAttribute('data-series', expect.stringContaining('期末净值概率密度'))
    expect(screen.getByText(/共计 500 条/)).toBeInTheDocument()
    expect(screen.getByText('双模型结果对比')).toBeInTheDocument()
    expect(screen.getByRole('table', { name: '参数化蒙特卡洛与区块 Bootstrap 模拟结果对比' })).toBeInTheDocument()
    expect(screen.getByText('95% CVaR（预期短缺）')).toBeInTheDocument()
    expect(screen.getAllByText('平均最大回撤')).toHaveLength(2)
    expect(screen.getByText('达到 5% 概率')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    await waitFor(() => expect(combinedChartRenderer).toHaveAttribute('data-terminal-histogram-total', '200'))
    expect(screen.getByText(/共计 200 条/)).toBeInTheDocument()
    await user.click(screen.getByRole('radio', { name: '区块 Bootstrap' }))
    expect(screen.getByRole('radio', { name: '区块 Bootstrap' })).toBeChecked()
    expect(combinedChart).toHaveAccessibleName('历史区块 Bootstrap：路径与期末净值概率分布组合图')
    expect(screen.getByText(/平均区块长度 20 个交易日/)).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: '正态 Q-Q 图' })).toBeInTheDocument()
    expect(screen.getByText('查看关键分位点数据')).toBeInTheDocument()
    expect(screen.getByTestId('distribution-diagnostics-grid')).toHaveClass('lg:grid-cols-2')
    const diagnosticCharts = screen.getAllByTestId('chart')
    const boxPlotChart = diagnosticCharts.find((chart) => chart.dataset.series?.includes('箱形图'))
    const normalQqChart = diagnosticCharts.find((chart) => chart.dataset.series?.includes('正态参考线'))
    expect(boxPlotChart).toHaveAttribute('data-x-axis-type', 'value')
    expect(boxPlotChart).toHaveAttribute('data-y-axis-type', 'category')
    expect(normalQqChart).toHaveAttribute('data-x-axis-type', 'value')
    expect(normalQqChart).toHaveAttribute('data-y-axis-type', 'value')
    expect(screen.queryByText('组合波动率')).not.toBeInTheDocument()
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['total-return'], targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1Y',
      as_of: undefined,
    }))
    expect(screen.getByText(/1Y · 2025-01-06 至 2026-01-06 · 250 个观察值/)).toBeInTheDocument()
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1Y')
    await user.selectOptions(screen.getByLabelText('区间累计收益计算区间'), '1M')
    await waitFor(() => expect(evaluateCustomIndicators).toHaveBeenCalledWith({
      indicator_ids: ['total-return'], targets: [{ kind: 'etf', product_id: '510300.SH' }], period: '1M',
      as_of: undefined,
    }))
    expect(screen.getByLabelText('区间累计收益计算区间')).toHaveValue('1M')
    expect(screen.getByRole('link', { name: '在指标中心分析' })).toHaveAttribute('href', '/indicator-studio?kind=etf&ids=510300.SH')

    await user.click(screen.getByRole('button', { name: '移除指标 区间累计收益' }))
    expect(screen.queryByText('12.34%')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /选择研究指标/ })).toHaveTextContent('已选 0/8')
    await waitFor(() => expect(JSON.parse(window.localStorage.getItem('indicator-display:v2:product-detail:single_product') ?? '{}')).toEqual({
      indicatorIds: [],
      periodsByIndicator: {},
    }))

    await user.selectOptions(screen.getByLabelText('模拟未来区间'), '21')
    await user.selectOptions(screen.getByLabelText('模拟路径数'), '200')
    await user.selectOptions(screen.getByLabelText('Bootstrap 平均区块长度'), '10')
    await user.clear(screen.getByLabelText('目标期末收益率'))
    await user.type(screen.getByLabelText('目标期末收益率'), '8')
    await user.click(screen.getByRole('button', { name: '重新模拟' }))
    expect(screen.getByText('200 条虚拟路径中的样本比例')).toBeInTheDocument()
    expect(screen.getByText('达到 8% 概率')).toBeInTheDocument()

    await user.selectOptions(screen.getByLabelText('统计区间'), '1M')
    expect(screen.getByText(/要求产品完整覆盖所选区间/)).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: '未来虚拟净值模拟' })).not.toBeInTheDocument()
  })

  it('场外公募基金展示成立日期，并仅在披露时展示到期日期', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        product_id: '000001.OF', name: '示例场外基金', management: '测试管理人', status: '存续',
        base_info: { ts_code: '000001.OF', found_date: '20010101', due_date: '2030-12-31' },
        metrics: { issue_amount: 100, m_fee: 0.5, c_fee: 0.1 },
        timeseries: [
          { date: '2026-01-05', open: 1, high: 1, low: 1, close: 1, volume: 0 },
          { date: '2026-01-06', open: 1.01, high: 1.01, low: 1.01, close: 1.01, volume: 0 },
        ],
      }),
    }))

    render(<MemoryRouter initialEntries={['/product/000001.OF?kind=fund']}><Routes><Route path="/product/:productId" element={<ProductDetail />} /></Routes></MemoryRouter>)

    expect(await screen.findByLabelText('成立日期：2001-01-01')).toBeInTheDocument()
    expect(screen.getByLabelText('到期日期：2030-12-31')).toBeInTheDocument()
    expect(screen.queryByText(/上市日期/)).not.toBeInTheDocument()
  })

  it('从手动构建大类进入后返回原来源页', async () => {
    const user = userEvent.setup()
    render(
      <MemoryRouter
        initialEntries={[
          '/manual-construction',
          {
            pathname: '/product/510300.SH',
            search: '?kind=etf',
            state: { returnTo: '/manual-construction', returnLabel: '返回手动构建大类' },
          },
        ]}
        initialIndex={1}
      >
        <Routes>
          <Route path="/manual-construction" element={<div>手动构建大类来源页</div>} />
          <Route path="/product/:productId" element={<ProductDetail />} />
        </Routes>
      </MemoryRouter>,
    )

    await act(async () => {
      await user.click(screen.getByRole('button', { name: /返回手动构建大类/ }))
    })
    expect(await screen.findByText('手动构建大类来源页')).toBeInTheDocument()
  })
})
