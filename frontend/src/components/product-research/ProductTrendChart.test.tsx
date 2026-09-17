import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import ProductTrendChart from './ProductTrendChart'
import {
  evaluateTimeSeriesIndicators,
  type EvaluateTimeSeriesIndicatorsRequest,
  type IndicatorDefinition,
  type TimeSeriesChannelResult,
  type TimeSeriesIndicatorResult,
} from '../../services/customIndicators'
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

/** Same shape, different measurement: it can never share an axis with the price. */
const volatility = {
  ...indicator, id: 'test-vol', name: '滚动波动率',
  parameter_schema: [{ id: 'window_1', label: '窗口期数', type: 'integer', default: 30, minimum: 1, maximum: 500, step: 1 }],
} as IndicatorDefinition

const channelFor = (indicatorId: string): TimeSeriesChannelResult => (indicatorId === volatility.id
  ? { id: 'value', label: '波动率', values: [0.1, 0.2, 0.3], unit: '%', display_format: 'percent', precision: 2,
    null_count: 0, output_measure: 'return_decimal', semantic_dimension: 'return_decimal', price_basis: 'adjusted_nav' }
  : { id: 'ma', label: '均线', values: [1, 2, 3], unit: '元', display_format: 'number', precision: 4, null_count: 0,
    output_measure: 'raw_market_price', semantic_dimension: 'raw_market_price', price_basis: 'raw_market' })

const result = (
  window: number, instanceKey: string | null = 'ov1', indicatorId = indicator.id,
): TimeSeriesIndicatorResult => ({
  instance_key: instanceKey, indicator_id: indicatorId, indicator_revision: 3, indicator_name: indicator.name, result_kind: 'time_series',
  target: { kind: 'etf', product_id: '510300.SH', name: 'ETF' }, period: 'ALL', parameters: { window_1: window },
  axis_anchor: 'market_close', history_policy: 'lookback', status: 'ok', warnings: [], dates,
  presentation: {} as TimeSeriesIndicatorResult['presentation'],
  window: { requested_as_of: null, effective_as_of: dates[2], start_date: dates[0], end_date: dates[2], observation_count: 3, data_latest_date: dates[2] },
  channels: [channelFor(indicatorId)],
})

/** Mirrors the contract: one result per instance, echoing that instance's key. */
const evaluated = (request: EvaluateTimeSeriesIndicatorsRequest) => ({
  results: request.indicator_instances.map((instance) => result(
    Number(instance.parameters?.window_1 ?? 20),
    instance.instance_key ?? null,
    instance.indicator_id ?? indicator.id,
  )),
  summary: { total: request.indicator_instances.length, ok: request.indicator_instances.length, warning: 0, unavailable: 0, error: 0 },
  cache: { hits: 0, misses: request.indicator_instances.length },
  execution: {},
})
const lastRequest = (): EvaluateTimeSeriesIndicatorsRequest => {
  const calls = vi.mocked(evaluateTimeSeriesIndicators).mock.calls
  return calls[calls.length - 1][0]
}

const renderChart = (catalog: IndicatorDefinition[] = [indicator]) => render(<MemoryRouter><ProductTrendChart
  productId="510300.SH" productKind="etf" indicators={catalog}
  regimeStateId="" onDefinition={() => undefined} studioHref="/settings/indicators-models"
/></MemoryRouter>)

const openSelector = () => {
  const trigger = screen.getByRole('button', { name: /选择时序指标/ })
  if (trigger.getAttribute('aria-expanded') !== 'true') fireEvent.click(trigger)
}
const addIndicator = async (name: RegExp = /可调均线/) => {
  openSelector()
  fireEvent.click(await screen.findByRole('checkbox', { name }))
}

/** Pick an option by its visible label, the way the user does. */
const choosePlacement = (select: HTMLElement, option: string | RegExp) => {
  const target = within(select).getByRole('option', { name: option }) as HTMLOptionElement
  fireEvent.change(select, { target: { value: target.value } })
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(fetchProductPriceSeries).mockImplementation(async (_id, _kind, basis) => priceSeries(basis))
  vi.mocked(evaluateTimeSeriesIndicators).mockImplementation(async (request) => evaluated(request) as never)
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

  it('读取中、读取失败、口径无数据各自是不同的提示，并配对应姿势的吉祥物', async () => {
    let release: (value: ProductPriceSeries) => void = () => undefined
    vi.mocked(fetchProductPriceSeries).mockReturnValue(new Promise<ProductPriceSeries>((resolve) => { release = resolve }))
    renderChart()

    expect(await screen.findByText('正在读取走势数据…', { selector: 'p.font-semibold' })).toBeInTheDocument()
    expect(document.querySelector('img[src*="mascot-working"]')).not.toBeNull()
    expect(screen.queryByText('走势数据没能读出来')).not.toBeInTheDocument()

    await act(async () => { release(priceSeries('raw_kline')) })
    await screen.findByTestId('chart')
    // 图画出来之后这块区域不再有吉祥物（准则 15.2 第 4 条）。
    expect(document.querySelector('img[src*="mascot-"]')).toBeNull()
  })

  it('读取失败时显示失败姿势与原始错误，而不是当作没有数据', async () => {
    vi.mocked(fetchProductPriceSeries).mockRejectedValue(new Error('价格序列接口 500。'))
    renderChart()

    expect(await screen.findByText('走势数据没能读出来')).toBeInTheDocument()
    expect(screen.getByText('价格序列接口 500。')).toBeInTheDocument()
    expect(document.querySelector('img[src*="mascot-error"]')).not.toBeNull()
    expect(screen.queryByTestId('chart')).not.toBeInTheDocument()
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
      indicator_instances: [{ instance_key: expect.any(String), indicator_id: indicator.id, indicator_revision: 3 }],
    })))
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    expect(evaluateTimeSeriesIndicators).toHaveBeenCalledTimes(1)
    // 改了还没应用：画出来的仍是服务端那一组，所以这里报 20 而不是输入框里的 60。
    expect(screen.getByText(/实际计算参数: 窗口期数=20/)).toBeInTheDocument()
    fireEvent.click(screen.getByText('应用参数'))
    await waitFor(() => expect(evaluateTimeSeriesIndicators).toHaveBeenLastCalledWith(expect.objectContaining({
      indicator_instances: [{
        instance_key: expect.any(String), indicator_id: indicator.id, indicator_revision: 3, parameters: { window_1: 60 },
      }],
    })))
    // 算完之后输入框就是实际值，不再把同一个数再印一遍。
    await waitFor(() => expect(screen.queryByText(/实际计算参数/)).toBeNull())
    expect(indicator.parameter_schema![0].default).toBe(20)
  })

  it('迟到的默认值结果不会覆盖新参数结果', async () => {
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let calls = 0
    vi.mocked(evaluateTimeSeriesIndicators).mockImplementation(async (request) => {
      const payload = evaluated(request)
      if ((calls += 1) === 1) await held
      return payload as never
    })
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    fireEvent.change(await screen.findByLabelText('窗口期数'), { target: { value: '60' } })
    fireEvent.click(screen.getByText('应用参数'))
    // 迟到的默认值结果若被采纳，这里就会冒出「实际计算参数: 窗口期数=20」。
    await waitFor(() => expect(screen.queryByText(/实际计算参数/)).toBeNull())
    await act(async () => { release() })
    expect(screen.queryByText(/实际计算参数/)).toBeNull()
  })

  it('能否与价格共轴由通道口径决定，换成复权净值后退回子图', async () => {
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    const placement = await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    // A raw-market moving average belongs on a raw-market price axis.
    await waitFor(() => expect(placement).toHaveDisplayValue('同轴同图'))
    expect(within(placement).getByRole('option', { name: /^同轴同图/ })).not.toBeDisabled()
    await waitFor(() => expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 成交量: 1, 均线: 0 }))

    fireEvent.click(screen.getByRole('radio', { name: '复权净值走势' }))

    await waitFor(() => expect(placement).toHaveDisplayValue('独立子图'))
    // 不能共轴的理由写在那个选项自己身上，不再额外占一行说明。
    const native = within(placement).getByRole('option', { name: /^同轴同图/ })
    expect(native).toBeDisabled()
    expect(native).toHaveTextContent('口径与主图不同，不可选')
    // Price grid plus one panel; a NAV path has no volume grid.
    expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '2')
    expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 均线: 1 })
  })

  it('同一指标可以配多条，参数各自独立，结果按实例键各认各的', async () => {
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()
    await screen.findByRole('combobox', { name: '可调均线的显示位置' })

    fireEvent.click(screen.getByRole('button', { name: '为 可调均线 再加一条' }))

    await waitFor(() => expect(lastRequest().indicator_instances).toHaveLength(2))
    const [first, second] = lastRequest().indicator_instances
    expect(first.instance_key).not.toBe(second.instance_key)
    expect(first.indicator_id).toBe(second.indicator_id)

    // 刚复制出来参数还一样，行与图例都靠序号区分，不会出现两个同名的东西。
    expect(await screen.findByText('可调均线 (窗口期数=20) #2')).toBeInTheDocument()

    const windows = await screen.findAllByLabelText('窗口期数')
    expect(windows).toHaveLength(2)
    fireEvent.change(windows[1], { target: { value: '60' } })
    // 只有改过参数的那条会长出“应用参数”，改完就剩这一个。
    fireEvent.click(screen.getByText('应用参数'))

    await waitFor(() => expect(lastRequest().indicator_instances.map((item) => item.parameters))
      .toEqual([undefined, { window_1: 60 }]))
    // 两条线在列表和图例里都用实际参数区分，不再是两个同名的「均线」。
    expect(await screen.findByText('可调均线 (窗口期数=60)')).toBeInTheDocument()
    expect(screen.getByText('可调均线 (窗口期数=20)')).toBeInTheDocument()
    const axes = JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')
    expect(Object.keys(axes)).toEqual(['价格', '成交量', '可调均线 (窗口期数=20)', '可调均线 (窗口期数=60)'])
  })

  it('第二条可以各占一个子图，也可以并进第一条的子图', async () => {
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()
    await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    fireEvent.click(screen.getByRole('button', { name: '为 可调均线 再加一条' }))
    const windows = await screen.findAllByLabelText('窗口期数')
    fireEvent.change(windows[1], { target: { value: '60' } })
    fireEvent.click(screen.getByText('应用参数'))
    await screen.findByText('可调均线 (窗口期数=60)')

    const [firstPlace, secondPlace] = screen.getAllByRole('combobox', { name: /的显示位置$/ })
    choosePlacement(firstPlace, '独立子图')
    choosePlacement(secondPlace, '独立子图')
    // 价格 + 成交量 + 两个子图。
    await waitFor(() => expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '4'))

    choosePlacement(secondPlace, '并入「可调均线 (窗口期数=20)」')

    await waitFor(() => expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '3'))
    const axes = JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')
    expect(axes['可调均线 (窗口期数=60)']).toBe(axes['可调均线 (窗口期数=20)'])
    expect(screen.getByText('与「可调均线 (窗口期数=20)」共用一个子图，同一条纵轴。')).toBeInTheDocument()
  })

  it('口径不同的两个指标不给并图选项，删掉宿主后跟随的那条退回自己的子图', async () => {
    renderChart([indicator, volatility])
    await screen.findByTestId('chart')
    await addIndicator()
    await addIndicator(/滚动波动率/)
    const maPlace = await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    const volPlace = await screen.findByRole('combobox', { name: '滚动波动率的显示位置' })
    choosePlacement(maPlace, '独立子图')

    // 一个是元、一个是百分比，合到一条纵轴上读出来的就不是同一个数。
    await waitFor(() => expect(within(volPlace).queryByRole('option', { name: /并入/ })).toBeNull())
    expect(within(maPlace).queryByRole('option', { name: /并入/ })).toBeNull()
    expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '4')

    fireEvent.click(screen.getByRole('button', { name: '移除指标 可调均线' }))
    await waitFor(() => expect(screen.getByTestId('chart')).toHaveAttribute('data-grid-count', '3'))
  })

  it('条数不设上限，加到第 9 条仍然能继续加，也不挡住选择器', async () => {
    renderChart([indicator, volatility])
    await screen.findByTestId('chart')
    await addIndicator()
    await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    for (let index = 0; index < 8; index += 1) {
      fireEvent.click(screen.getAllByRole('button', { name: /再加一条$/ })[0])
    }

    await waitFor(() => expect(screen.getAllByRole('button', { name: /再加一条$/ })).toHaveLength(9))
    expect(screen.getByText(/叠加时序指标/)).toHaveTextContent('9 条')
    expect(screen.getAllByRole('button', { name: /再加一条$/ })[0]).toBeEnabled()
    openSelector()
    expect(await screen.findByRole('checkbox', { name: /滚动波动率/ })).toBeEnabled()
  })

  it('复权口径指标只能和后复权 K 线共轴，不能和未复权行情共轴', async () => {
    const adjustedBases = priceSeries('raw_kline').bases.map(item => ({ ...item, available: true, reason: null }))
    vi.mocked(fetchProductPriceSeries).mockImplementation(async (_id, _kind, basis) => (
      { ...priceSeries(basis), bases: adjustedBases }
    ))
    vi.mocked(evaluateTimeSeriesIndicators).mockImplementation(async (request) => {
      const payload = evaluated(request)
      return {
        ...payload,
        results: payload.results.map((item: TimeSeriesIndicatorResult) => ({
          ...item,
          channels: item.channels.map((channel) => ({ ...channel, output_measure: 'adjusted_market_price',
            semantic_dimension: 'adjusted_market_price', price_basis: 'adjusted_market' })),
        })),
      } as never
    })
    renderChart()
    await screen.findByTestId('chart')
    await addIndicator()

    const placement = await screen.findByRole('combobox', { name: '可调均线的显示位置' })
    // Default basis is the unadjusted candle: an adjusted series must not share it.
    await waitFor(() => expect(placement).toHaveDisplayValue('独立子图'))
    expect(within(placement).getByRole('option', { name: /^同轴同图/ })).toBeDisabled()

    fireEvent.click(screen.getByRole('radio', { name: '后复权 K 线' }))

    await waitFor(() => expect(within(placement).getByRole('option', { name: /^同轴同图/ })).not.toBeDisabled())
    choosePlacement(placement, /^同轴同图/)
    await waitFor(() => expect(JSON.parse(screen.getByTestId('chart').dataset.axis ?? '{}')).toEqual({ 价格: 0, 成交量: 1, 均线: 0 }))
  })
})
