import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductTrendChart from './ProductTrendChart'
import { evaluateTimeSeriesIndicators, type IndicatorDefinition, type TimeSeriesIndicatorResult } from '../../services/customIndicators'
import { fetchProductPriceSeries, type ProductPriceSeries } from '../../services/productAnalysis'

vi.mock('echarts-for-react', () => ({
  default: ({ option }: { option: any }) => <div
    data-testid="chart"
    data-grid-count={option.grid.length}
    data-axis={JSON.stringify(Object.fromEntries(option.series.map((item: any) => [item.name, item.yAxisIndex ?? 0])))}
    data-mark-area={JSON.stringify(option.series[0]?.markArea?.data ?? [])}
  />,
}))
vi.mock('../../services/customIndicators', async original => ({ ...await original<object>(), evaluateTimeSeriesIndicators: vi.fn() }))
vi.mock('../../services/productAnalysis', async original => ({ ...await original<object>(), fetchProductPriceSeries: vi.fn() }))

const dates = ['2026-01-02', '2026-01-05', '2026-01-06']
const priceSeries = (basis: ProductPriceSeries['basis']): ProductPriceSeries => ({
  product_id: '510300.SH', kind: 'etf', basis,
  label: basis === 'raw_kline' ? '不复权 K 线（原始行情）' : basis === 'adjusted_kline' ? '后复权 K 线' : '复权净值走势',
  available: true, reason: null, warnings: [],
  points: dates.map((date, index) => basis === 'adjusted_nav'
    ? { date, open: null, high: null, low: null, close: 3 + index * 0.1, volume: null }
    : { date, open: 3 + index * 0.1, high: 3.2 + index * 0.1, low: 2.9 + index * 0.1, close: 3.1 + index * 0.1, volume: 100 + index }),
  bases: [
    { id: 'adjusted_nav', label: '复权净值走势', description: '复权净值折线', available: true, reason: null },
    { id: 'adjusted_kline', label: '后复权 K 线', description: '原始开高低收乘以复权因子', available: false, reason: '该产品缺少复权因子数据' },
    { id: 'raw_kline', label: '不复权 K 线（原始行情）', description: '交易所原始开高低收', available: true, reason: null },
  ],
  execution: {} as ProductPriceSeries['execution'],
})

const indicator: IndicatorDefinition = {
  id: 'test-ma', revision: 3, source: 'custom', read_only: false, created_at: '', updated_at: '',
  name: '可调均线', description: '', expression: 'rolling_mean(market_close, window_1)', unit: '', display_format: 'number', precision: 4,
  direction: 'neutral', annual_risk_free_rate_percent: 0, result_kind: 'time_series', output_contract: 'series_bundle',
  parameter_contract_version: '1.0', axis_anchor: 'market_close',
  parameter_schema: [{ id: 'window_1', label: '窗口期数', type: 'integer', default: 20, minimum: 1, maximum: 500, step: 1 }],
  applicable_product_kinds: ['etf'],
} as IndicatorDefinition

const result = (window: number): TimeSeriesIndicatorResult => ({
  indicator_id: indicator.id, indicator_revision: 3, indicator_name: indicator.name, result_kind: 'time_series',
  target: { kind: 'etf', product_id: '510300.SH', name: 'ETF' }, period: 'ALL', parameters: { window_1: window },
  axis_anchor: 'market_close', history_policy: 'lookback', status: 'ok', warnings: [], dates,
  presentation: {} as TimeSeriesIndicatorResult['presentation'],
  window: { requested_as_of: null, effective_as_of: dates[2], start_date: dates[0], end_date: dates[2], observation_count: 3, data_latest_date: dates[2] },
  channels: [{ id: 'ma', label: '均线', values: [1, 2, 3], unit: '元', display_format: 'number', precision: 4, null_count: 0,
    output_measure: 'raw_market_price', semantic_dimension: 'raw_market_price', price_basis: 'raw_market' }],
})
const response = (window: number) => ({ results: [result(window)], summary: { total: 1, ok: 1, warning: 0, unavailable: 0, error: 0 }, cache: { hits: 0, misses: 1 }, execution: {} })

const renderChart = () => render(<MemoryRouter><ProductTrendChart
  productId="510300.SH" productKind="etf" indicators={[indicator]}
  regimeStateId="" onDefinition={() => undefined} studioHref="/settings/indicators-models"
/></MemoryRouter>)

const addIndicator = async () => {
  fireEvent.click(screen.getByRole('button', { name: /选择时序指标/ }))
  fireEvent.click(await screen.findByRole('checkbox', { name: /可调均线/ }))
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(fetchProductPriceSeries).mockImplementation(async (_id, _kind, basis) => priceSeries(basis))
  vi.mocked(evaluateTimeSeriesIndicators).mockResolvedValue(response(20) as never)
})
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('走势图价格口径', () => {
  it('默认读不复权行情，换口径重新取数而不是在前端换算', async () => {
    renderChart()

    await screen.findByTestId('chart')
    expect(fetchProductPriceSeries).toHaveBeenLastCalledWith('510300.SH', 'etf', 'raw_kline', expect.anything())
    fireEvent.click(screen.getByRole('radio', { name: '复权净值走势' }))
    await waitFor(() => expect(fetchProductPriceSeries).toHaveBeenLastCalledWith('510300.SH', 'etf', 'adjusted_nav', expect.anything()))
  })

  it('缺少复权因子的口径不可选，并说明原因', async () => {
    renderChart()

    const option = await screen.findByRole('radio', { name: '后复权 K 线' })
    expect(option).toBeDisabled()
    expect(option.closest('label')).toHaveAttribute('title', '该产品缺少复权因子数据')
  })

  it('口径没有数据时说清楚是哪一步缺东西，而不是画一张空图', async () => {
    vi.mocked(fetchProductPriceSeries).mockResolvedValue({
      ...priceSeries('adjusted_kline'), available: false, points: [], reason: '该产品缺少复权因子数据。',
    })
    renderChart()

    expect(await screen.findByText('这个口径暂时没有数据')).toBeInTheDocument()
    expect(screen.getByText('该产品缺少复权因子数据。')).toBeInTheDocument()
    expect(screen.queryByTestId('chart')).not.toBeInTheDocument()
  })
})

describe('叠加时序指标', () => {
  it('默认请求锁定版本，修改参数重算但不改变定义', async () => {
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    await waitFor(() => expect(evaluateTimeSeriesIndicators).toHaveBeenLastCalledWith(expect.objectContaining({
      indicator_instances: [{ indicator_id: indicator.id, indicator_revision: 3 }],
    })))
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    expect(evaluateTimeSeriesIndicators).toHaveBeenCalledTimes(1)
    vi.mocked(evaluateTimeSeriesIndicators).mockResolvedValue(response(60) as never)
    fireEvent.click(screen.getByText('应用参数'))
    await waitFor(() => expect(evaluateTimeSeriesIndicators).toHaveBeenLastCalledWith(expect.objectContaining({
      indicator_instances: [{ indicator_id: indicator.id, indicator_revision: 3, parameters: { window_1: 60 } }],
    })))
    expect(await screen.findByText(/实际计算参数: 窗口期数=60/)).toBeInTheDocument()
    expect(indicator.parameter_schema![0].default).toBe(20)
  })

  it('迟到的默认值结果不会覆盖新参数结果', async () => {
    let stale!: (value: ReturnType<typeof response>) => void
    vi.mocked(evaluateTimeSeriesIndicators)
      .mockReturnValueOnce(new Promise(resolve => { stale = resolve as typeof stale }) as never)
      .mockResolvedValueOnce(response(60) as never)
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    fireEvent.change(await screen.findByLabelText('窗口期数'), { target: { value: '60' } })
    fireEvent.click(screen.getByText('应用参数'))
    expect(await screen.findByText(/实际计算参数: 窗口期数=60/)).toBeInTheDocument()
    await act(async () => { stale(response(20)) })
    expect(screen.getByText(/实际计算参数: 窗口期数=60/)).toBeInTheDocument()
  })

  it('能否与价格共轴由通道口径决定，换成复权净值后退回子图', async () => {
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    const placement = await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    // A raw-market moving average belongs on a raw-market price axis.
    await waitFor(() => expect(placement).toHaveValue('native'))
    expect(within(placement).getByRole('option', { name: '同轴同图' })).not.toBeDisabled()
    await waitFor(() => expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 成交量: 1, 均线: 0 }))

    fireEvent.click(screen.getByRole('radio', { name: '复权净值走势' }))

    await waitFor(() => expect(placement).toHaveValue('panel'))
    expect(within(placement).getByRole('option', { name: '同轴同图' })).toBeDisabled()
    expect(screen.getByText('口径与主图不同，不能与价格共轴；请用右轴或独立子图。')).toBeInTheDocument()
    // Price grid plus one panel; a NAV path has no volume grid.
    expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '2')
    expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 均线: 1 })
  })

  it('复权口径指标只能和后复权 K 线共轴，不能和未复权行情共轴', async () => {
    const adjustedBases = priceSeries('raw_kline').bases.map(item => ({ ...item, available: true, reason: null }))
    vi.mocked(fetchProductPriceSeries).mockImplementation(async (_id, _kind, basis) => (
      { ...priceSeries(basis), bases: adjustedBases }
    ))
    const adjusted = result(20)
    adjusted.channels = [{ ...adjusted.channels[0], output_measure: 'adjusted_market_price',
      semantic_dimension: 'adjusted_market_price', price_basis: 'adjusted_market' }]
    vi.mocked(evaluateTimeSeriesIndicators).mockResolvedValue({ ...response(20), results: [adjusted] } as never)
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    const placement = await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    // Default basis is the unadjusted candle: an adjusted series must not share it.
    await waitFor(() => expect(placement).toHaveValue('panel'))
    expect(within(placement).getByRole('option', { name: '同轴同图' })).toBeDisabled()

    fireEvent.click(screen.getByRole('radio', { name: '后复权 K 线' }))

    await waitFor(() => expect(within(placement).getByRole('option', { name: '同轴同图' })).not.toBeDisabled())
    fireEvent.change(placement, { target: { value: 'native' } })
    await waitFor(() => expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 成交量: 1, 均线: 0 }))
  })
})
