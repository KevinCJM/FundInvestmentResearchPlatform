import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, it, vi } from 'vitest'
import RegimeNodeSeriesChart from './RegimeNodeSeriesChart'
import type { RegimeSeriesPage } from '../../services/regimeGraph'

const { resizeRenderer } = vi.hoisted(() => ({ resizeRenderer: vi.fn() }))
vi.mock('echarts-for-react', async () => {
  const { useEffect } = await import('react')
  return { default: function Chart({ option, onEvents, onChartReady }: any) {
    useEffect(() => { onChartReady?.({ resize: resizeRenderer, isDisposed: () => false }) }, [])
    return <div><div data-testid="chart-option">{JSON.stringify(option)}</div><button onClick={() => onEvents.datazoom({}, { getOption: () => ({ dataZoom: [{ startValue: 1, endValue: 2 }] }) })}>缩放后两日</button></div>
  } }
})
const page: RegimeSeriesPage = { run_id: 'job', node_id: 'price', port: 'value', value_type: 'series<float64>', offset: 0, limit: 5000, total: 3, items: [{ date: '2020-01-01', value: 100 }, { date: '2020-01-02', value: 120 }, { date: '2020-01-03', value: 90 }] }
const audit = { execution_backend: 'numba_njit_fixed_signature', nopython: true, object_mode: 0, python_fallback: 0, request_time_compilation: 0, kernel_signatures: { binary: ['fixed'] } }
const data = (base: number) => ({ run_id: 'job', node_id: 'price', port: 'value', base_index: base, base_date: page.items[base].date, base_value: page.items[base].value, values: base ? [100 / 120, 1, 0.75] : [1, 1.2, 0.9], change_pct: base ? [-100 / 6, 0, -25] : [0, 20, -10], execution: audit })
const ok = (body: unknown) => ({ ok: true, json: async () => body } as Response)
const option = () => JSON.parse(screen.getByTestId('chart-option').textContent || '{}')
afterEach(() => { vi.unstubAllGlobals(); vi.restoreAllMocks() })

it('容器改变尺寸时同步调整真实渲染器，卸载后断开观察', () => {
  let notify!: ResizeObserverCallback
  const disconnect = vi.fn()
  vi.stubGlobal('ResizeObserver', class {
    constructor(callback: ResizeObserverCallback) { notify = callback }
    observe() {}
    disconnect = disconnect
  })
  const { unmount } = render(<RegimeNodeSeriesChart page={page} />)
  resizeRenderer.mockClear()
  act(() => notify([{ contentRect: { width: 720, height: 640 } } as ResizeObserverEntry], {} as ResizeObserver))
  expect(resizeRenderer).toHaveBeenLastCalledWith({ width: 720, height: 640, silent: true })
  expect(option().grid[0].height).toBe(546)
  act(() => notify([{ contentRect: { width: 720, height: 840 } } as ResizeObserverEntry], {} as ResizeObserver))
  expect(resizeRenderer).toHaveBeenLastCalledWith({ width: 720, height: 840, silent: true })
  expect(option().dataZoom[0]).toMatchObject({ startValue: 0, endValue: 2 })
  unmount()
  expect(disconnect).toHaveBeenCalledTimes(1)
})

it('开关按可见首日归一化，缩放后重设基准，关闭恢复原值且保留区间', async () => {
  const fetch = vi.fn(async (url: string) => ok(data(Number(new URL(url, 'http://localhost').searchParams.get('base_index')))))
  vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={page} />)
  expect(option().series[0].data).toEqual([100, 120, 90])
  const toggle = screen.getByRole('switch', { name: '区间归一化（首日＝1）' })
  await userEvent.click(toggle)
  await screen.findByText(/基准日 2020-01-01/)
  expect(option().series[0].data).toEqual([1, 1.2, 0.9])
  await userEvent.click(screen.getByText('缩放后两日'))
  await screen.findByText(/基准日 2020-01-02/)
  expect(screen.getByRole('status')).toHaveTextContent('涨跌幅 -25%')
  expect(option().series[0].data).toEqual([100 / 120, 1, 0.75])
  expect(fetch.mock.calls[1][0]).toContain('base_index=1')
  await userEvent.click(toggle)
  expect(option().series[0].data).toEqual([100, 120, 90])
  expect(option().dataZoom[0].startValue).toBe(1)
  expect(page.items.map(row => row.value)).toEqual([100, 120, 90])
})

it('快速切换区间时忽略旧响应，关闭开关取消等待', async () => {
  let oldResolve: (response: Response) => void = () => undefined
  const fetch = vi.fn((url: string) => url.includes('base_index=0') ? new Promise<Response>(resolve => { oldResolve = resolve }) : Promise.resolve(ok(data(1))))
  vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={page} />)
  await userEvent.click(screen.getByRole('switch'))
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1))
  await userEvent.click(screen.getByText('缩放后两日'))
  await screen.findByText(/基准日 2020-01-02/)
  oldResolve(ok(data(0)))
  await waitFor(() => expect(option().series[0].data).toEqual([100 / 120, 1, 0.75]))
  await userEvent.click(screen.getByRole('switch'))
  expect(screen.queryByRole('status')).not.toBeInTheDocument()
})

it('首日无效时显示原因，不冒充归一化结果，调整区间后可恢复', async () => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => url.includes('base_index=0') ? ({ ok: false, status: 422, json: async () => ({ detail: { message: '区间首日数值须大于 0 且非缺失，请调整区间起点。' } }) } as Response) : ok(data(1))))
  render(<RegimeNodeSeriesChart page={page} />)
  await userEvent.click(screen.getByRole('switch'))
  await screen.findByText(/区间首日数值须大于 0/)
  expect(option().series[0].data).toEqual([null, null, null])
  fireEvent.click(screen.getByText('缩放后两日'))
  await screen.findByText(/基准日 2020-01-02/)
})

it('市场状态枚举不能按净值归一化', () => {
  const fetch = vi.fn(); vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={{ ...page, value_type: 'state_codes<int64>' }} />)
  expect(screen.getByRole('switch')).toBeDisabled()
  expect(fetch).not.toHaveBeenCalled()
})

const upstream = { node_id: 'source', node_label: '沪深300', port: 'value', port_label: '数值序列', value_type: 'series<float64>', distance: 1, plottable: true }
const upstreamPage: RegimeSeriesPage = { ...page, node_id: 'source', items: [{ date: '2020-01-01', value: 200 }, { date: '2020-01-03', value: 180 }] , total: 2 }
const withUpstream = { ...page, node_label: '卡尔曼滤波', upstream_outputs: [upstream] }
const selectUpstream = async () => {
  await userEvent.click(screen.getByText('叠加已计算节点'))
  await userEvent.click(screen.getByRole('checkbox', { name: /沪深300/ }))
}

it('同一任务的平行分支自动叠加，支持同轴、双轴和子图并共用归一化日期', async () => {
  const sibling = { ...upstream, node_id: 'ema', node_label: '单边 EMA', distance: null, relationship: 'other' as const }
  const fetch = vi.fn(async (url: string) => {
    const node = new URL(url, 'http://localhost').searchParams.get('node_id')!
    if (url.includes('/series?')) return ok({ ...page, node_id: node, items: [{ date: '2020-01-01', value: 100 }, { date: '2020-01-02', value: 110 }, { date: '2020-01-03', value: 100 }] })
    return ok({ ...data(0), node_id: node, values: node === 'ema' ? [1, 1.1, 1] : [1, 1.2, .9] })
  })
  vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={{ ...page, node_label: '递归平滑', upstream_outputs: [upstream], overlay_outputs: [upstream, sibling], comparison_targets: [{ node_id: 'ema', port: 'value' }] }} />)
  await waitFor(() => expect(option().series[1].data).toEqual([100, 110, 100]))
  expect(option().grid).toHaveLength(2)
  const placement = screen.getByRole('combobox', { name: /单边 EMA.*展示位置/ })
  await userEvent.selectOptions(placement, 'left')
  expect(option().series[1]).toMatchObject({ xAxisIndex: 0, yAxisIndex: 0 })
  await userEvent.selectOptions(placement, 'right')
  expect(option().series[1].yAxisIndex).toBe(1)
  await userEvent.click(screen.getByRole('switch'))
  await waitFor(() => expect(option().series[1].data).toEqual([1, 1.1, 1]))
  expect(fetch.mock.calls.every(([url]) => url.includes('/preview-runs/job/'))).toBe(true)
  expect(fetch.mock.calls.filter(([url]) => url.includes('/series?'))).toHaveLength(1)
  expect(page.items[1].value).toBe(120)
})

it('可选上游默认子图，按日期对齐缺失，切换左右轴不重复读取，共用缩放', async () => {
  const fetch = vi.fn(async () => ok(upstreamPage)); vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={withUpstream} />)
  expect(fetch).not.toHaveBeenCalled()
  await selectUpstream()
  await waitFor(() => expect(option().series[1].data).toEqual([['2020-01-01', 200], ['2020-01-03', 180]]))
  expect(option().grid).toHaveLength(2)
  expect(option().series[1]).toMatchObject({ xAxisIndex: 1, yAxisIndex: 1, connectNulls: false })
  expect(option().dataZoom[0].xAxisIndex).toEqual([0, 1])
  expect(option().axisPointer.link).toEqual([{ xAxisIndex: 'all' }])
  await userEvent.click(screen.getByText('缩放后两日'))
  const placement = screen.getByRole('combobox', { name: /沪深300.*展示位置/ })
  await userEvent.selectOptions(placement, 'left')
  expect(option().grid).toHaveLength(1)
  expect(option().series[1]).toMatchObject({ xAxisIndex: 0, yAxisIndex: 0 })
  await userEvent.selectOptions(placement, 'right')
  expect(option().yAxis).toHaveLength(2)
  expect(option().series[1].yAxisIndex).toBe(1)
  expect(option().dataZoom[0].startValue).toBe(1)
  expect(fetch).toHaveBeenCalledTimes(1)
  await userEvent.click(screen.getByRole('button', { name: /移除沪深300/ }))
  expect(option().series).toHaveLength(1)
  await userEvent.click(screen.getByRole('checkbox', { name: /沪深300/ }))
  expect(option().series[1].data).toEqual([['2020-01-01', 200], ['2020-01-03', 180]])
  expect(fetch).toHaveBeenCalledTimes(1)
})

it('上游按同一日期分别归一化，缺失基准不借用次日，原始对比表不变', async () => {
  const fetch = vi.fn(async (url: string) => {
    if (url.includes('/series?')) return ok(upstreamPage)
    if (url.includes('node_id=source')) return ok({ ...data(0), node_id: 'source', values: [1, 0.9], change_pct: [0, -10], base_value: 200 })
    return ok(data(Number(new URL(url, 'http://localhost').searchParams.get('base_index'))))
  }); vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={withUpstream} />)
  await selectUpstream()
  await waitFor(() => expect(option().series[1].data).toEqual([['2020-01-01', 200], ['2020-01-03', 180]]))
  await userEvent.click(screen.getByRole('switch'))
  await waitFor(() => expect(option().series[1].data).toEqual([['2020-01-01', 1], ['2020-01-03', 0.9]]))
  await userEvent.click(screen.getByText('缩放后两日'))
  await screen.findByText(/沪深300.*区间首日没有观测/)
  expect(option().series[1].data).toEqual([['2020-01-01', null], ['2020-01-03', null]])
  expect(option().series[0].data).toEqual([100 / 120, 1, 0.75])
  await userEvent.click(screen.getByText(/查看对比数据/))
  expect(screen.getByRole('table', { name: '节点对比数据' })).toHaveTextContent('180')
  expect(fetch.mock.calls.filter(([url]) => url.includes('node_id=source') && url.includes('normalized-chart'))).toHaveLength(1)
})

it('加载上游失败可重试，移除后晚到响应不重新添加曲线', async () => {
  let resolve: (response: Response) => void = () => undefined
  const fetch = vi.fn().mockResolvedValueOnce({ ok: false, status: 500, json: async () => ({ detail: { message: '读取失败' } }) }).mockImplementation(() => new Promise<Response>(done => { resolve = done }))
  vi.stubGlobal('fetch', fetch)
  render(<RegimeNodeSeriesChart page={withUpstream} />)
  await selectUpstream()
  await screen.findByText('读取失败')
  await userEvent.click(screen.getByRole('button', { name: '重试' }))
  await waitFor(() => expect(fetch).toHaveBeenCalledTimes(2))
  await userEvent.click(screen.getByRole('button', { name: /移除沪深300/ }))
  resolve(ok(upstreamPage))
  await waitFor(() => expect(option().series).toHaveLength(1))
})

it('不同频率包含上游中间日期，缩放后添加曲线保持日期范围', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => ok(page)))
  render(<RegimeNodeSeriesChart page={{ ...upstreamPage, node_id: 'filtered', upstream_outputs: [{ ...upstream, node_id: 'price' }] }} />)
  await selectUpstream()
  await waitFor(() => expect(option().xAxis[0].data).toEqual(page.items.map(item => item.date)))
  expect(option().series[0].data).toEqual([['2020-01-01', 200], ['2020-01-03', 180]])
  expect(option().series[1].data).toEqual([100, 120, 90])
})

it('全屏与返回保留上游、同轴设置、归一化和缩放，不重复请求，Esc 不关闭外层预览', async () => {
  Object.defineProperty(HTMLDialogElement.prototype, 'showModal', { configurable: true, value() { this.open = true } })
  Object.defineProperty(HTMLDialogElement.prototype, 'close', { configurable: true, value() { this.open = false } })
  const fetch = vi.fn(async (url: string) => {
    if (url.includes('/series?')) return ok({ ...page, node_id: 'source' })
    return ok({ ...data(1), node_id: new URL(url, 'http://localhost').searchParams.get('node_id') })
  })
  vi.stubGlobal('fetch', fetch)
  const outerKey = vi.fn()
  document.addEventListener('keydown', outerKey)
  try {
    render(<RegimeNodeSeriesChart page={withUpstream} />)
    await selectUpstream()
    await waitFor(() => expect(option().series).toHaveLength(2))
    await userEvent.selectOptions(screen.getByRole('combobox'), 'left')
    await userEvent.click(screen.getByText('缩放后两日'))
    await userEvent.click(screen.getByRole('switch'))
    await screen.findByText(/沪深300.*基准日 2020-01-02/)
    const before = option(), requests = fetch.mock.calls.length
    const overflow = document.body.style.overflow
    await userEvent.click(screen.getByRole('button', { name: '全屏展开' }))
    expect(screen.getByRole('dialog', { name: '节点图表全屏' })).toBeInTheDocument()
    expect(document.body.style.overflow).toBe('hidden')
    expect(option().series).toEqual(before.series)
    expect(option().dataZoom).toEqual(before.dataZoom)
    outerKey.mockClear()
    fireEvent.keyDown(screen.getByRole('button', { name: '退出全屏' }), { key: 'Escape' })
    expect(outerKey).not.toHaveBeenCalled()
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: '全屏展开' })).toHaveFocus()
    expect(screen.getByRole('combobox')).toHaveValue('left')
    expect(screen.getByRole('switch')).toHaveAttribute('aria-checked', 'true')
    expect(option().dataZoom).toEqual(before.dataZoom)
    expect(fetch).toHaveBeenCalledTimes(requests)
    expect(document.body.style.overflow).toBe(overflow)
  } finally {
    document.removeEventListener('keydown', outerKey)
    Reflect.deleteProperty(HTMLDialogElement.prototype, 'showModal')
    Reflect.deleteProperty(HTMLDialogElement.prototype, 'close')
  }
})
