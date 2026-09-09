import { useState } from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { IndicatorDefinition, IndicatorDraft, SeriesParameterDefinition, TimeSeriesIndicatorResult } from '../../services/customIndicators'
import { evaluateTimeSeriesIndicators } from '../../services/customIndicators'
import * as parameterApi from '../../services/indicatorParameters'
import IndicatorParameterInputs from './IndicatorParameterInputs'
import IndicatorParameterEditor from './IndicatorParameterEditor'
import TimeSeriesIndicatorPanel from './TimeSeriesIndicatorPanel'

vi.mock('echarts-for-react', () => ({ default: ({ option }: { option: unknown }) => <div data-testid="chart">{JSON.stringify(option)}</div> }))
vi.mock('../../services/customIndicators', async original => ({ ...await original<object>(), evaluateTimeSeriesIndicators: vi.fn() }))
vi.mock('../../services/indicatorParameters', async original => ({ ...await original<object>(), inspectIndicatorParameters: vi.fn(), bindIndicatorParameter: vi.fn() }))

const schema: SeriesParameterDefinition[] = [{ id: 'window_1', label: '窗口期数', type: 'integer', default: 20, minimum: 1, maximum: 500, step: 1 }]
const indicator: IndicatorDefinition = {
  id: 'test-ma', revision: 3, source: 'custom', read_only: false, created_at: '', updated_at: '',
  name: '可调均线', description: '', expression: 'rolling_mean(market_close, window_1)', unit: '', display_format: 'number', precision: 4,
  direction: 'neutral', annual_risk_free_rate_percent: 0, result_kind: 'time_series', output_contract: 'series_bundle',
  parameter_contract_version: '1.0', parameter_schema: schema, axis_anchor: 'market_close',
  series_outputs: [{ id: 'ma', label: '均线', expression: 'rolling_mean(market_close, window_1)', unit: '', display_format: 'number', precision: 4, output_measure: 'auto' }],
}
const fixedDraft: IndicatorDraft = { ...indicator, parameter_contract_version: null, parameter_schema: [], expression: 'rolling_mean(market_close, 20)',
  series_outputs: indicator.series_outputs!.map(output => ({ ...output, expression: 'rolling_mean(market_close, 20)' })) }
const candidate: parameterApi.ParameterCandidate = { id: 'ma:1:window:abc', output_id: 'ma', output_label: '均线', operator_id: 'rolling_mean', argument: 'window', label: '窗口期数', value: 20, parameter_id: null, type: 'integer', minimum: 1, maximum: 20000, step: 1 }
const result = (value: number): TimeSeriesIndicatorResult => ({
  indicator_id: indicator.id, indicator_revision: 3, indicator_name: indicator.name, result_kind: 'time_series',
  target: { kind: 'etf', product_id: '510300.SH', name: 'ETF' }, period: '1Y', parameters: { window_1: value },
  axis_anchor: 'market_close', history_policy: 'lookback', lookback_observations: value, minimum_observations: value,
  status: 'ok', warnings: [], dates: ['2025-01-01'],
  presentation: { indicator_id: indicator.id, revision: 3, name: indicator.name, source: 'custom', category: 'technical', category_label: '技术指标', context_kind: 'single_product', catalog_status: 'current', display_format: 'number', precision: 4, unit: '', notation: 'standard', value_scale: 1, output_measure: 'raw_market_price', direction: 'neutral', description: '', methodology: '', data_basis: '', minimum_observations: value, applicable_product_kinds: ['etf'] },
  window: { requested_as_of: null, effective_as_of: '2025-01-01', start_date: '2025-01-01', end_date: '2025-01-01', observation_count: 1, data_latest_date: '2025-01-01' },
  channels: [{ id: 'ma', label: '均线', values: [value], unit: '', display_format: 'number', precision: 4, null_count: 0, output_measure: 'raw_market_price' }],
})
const response = (value: number) => ({ results: [result(value)], summary: { total: 1, ok: 1, warning: 0, unavailable: 0, error: 0 }, cache: { hits: 0, misses: 1 }, execution: {} })
beforeEach(() => { vi.clearAllMocks() })
afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('运行参数输入', () => {
  it('默认显示默认值；编辑不计算，应用只发送覆盖值，恢复默认发送空覆盖', () => {
    const apply = vi.fn()
    render(<IndicatorParameterInputs schema={schema} values={{}} onApply={apply} />)
    expect(screen.getByLabelText('窗口期数')).toHaveValue(20)
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    expect(apply).not.toHaveBeenCalled()
    fireEvent.click(screen.getByText('应用参数'))
    expect(apply).toHaveBeenLastCalledWith({ window_1: 60 })
    fireEvent.click(screen.getByText('恢复默认'))
    expect(apply).toHaveBeenLastCalledWith({})
    expect(schema[0].default).toBe(20)
  })
  it.each(['', '0', '501', '2.5'])('非法值 %s 不触发计算', value => {
    const apply = vi.fn()
    render(<IndicatorParameterInputs schema={schema} values={{}} onApply={apply} />)
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value } })
    fireEvent.click(screen.getByText('应用参数'))
    expect(apply).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toBeInTheDocument()
  })
  it('切换版本后读取新版本默认值，不沿用之前编辑的输入', () => {
    const { rerender } = render(<IndicatorParameterInputs schema={schema} values={{}} onApply={vi.fn()} />)
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    rerender(<IndicatorParameterInputs schema={[{ ...schema[0], default: 30 }]} values={{}} onApply={vi.fn()} />)
    expect(screen.getByLabelText('窗口期数')).toHaveValue(30)
  })
})

describe('参数定义交互', () => {
  it('识别后开放参数并保留默认值，设置需显式应用', async () => {
    vi.mocked(parameterApi.inspectIndicatorParameters).mockResolvedValue({ contract_version: '1.0', candidates: [candidate] })
    vi.mocked(parameterApi.bindIndicatorParameter).mockResolvedValue({ contract_version: '1.0', candidates: [{ ...candidate, parameter_id: 'window_1' }], definition: indicator })
    const patch = vi.fn()
    function Editor() {
      const [draft, setDraft] = useState(fixedDraft)
      return <IndicatorParameterEditor draft={draft} onPatch={value => { patch(value); setDraft(current => ({ ...current, ...value })) }} />
    }
    render(<Editor />)
    fireEvent.click(screen.getByText('识别可调输入'))
    fireEvent.click(await screen.findByText('开放为参数'))
    expect(await screen.findByLabelText('默认值')).toHaveValue(20)
    fireEvent.change(screen.getByLabelText('默认值'), { target: { value: '60' } })
    expect(patch).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByText('应用参数设置'))
    expect(patch).toHaveBeenLastCalledWith({ parameter_schema: [{ ...schema[0], default: 60, description: '' }] })
  })
  it('公式变化后忽略旧的识别结果', async () => {
    let resolve!: (value: parameterApi.ParameterInspection) => void
    vi.mocked(parameterApi.inspectIndicatorParameters).mockReturnValue(new Promise(done => { resolve = done }))
    const { rerender } = render(<IndicatorParameterEditor draft={fixedDraft} onPatch={vi.fn()} />)
    fireEvent.click(screen.getByText('识别可调输入'))
    rerender(<IndicatorParameterEditor draft={{ ...fixedDraft, series_outputs: [] }} onPatch={vi.fn()} />)
    await act(async () => { resolve({ contract_version: '1.0', candidates: [candidate] }) })
    expect(screen.queryByText('开放为参数')).not.toBeInTheDocument()
  })
})

describe('使用页面参数隔离', () => {
  it('默认请求锁定版本，修改参数重算但不改变定义', async () => {
    vi.mocked(evaluateTimeSeriesIndicators).mockResolvedValue(response(20) as never)
    render(<TimeSeriesIndicatorPanel indicators={[indicator]} productId="510300.SH" productKind="etf" periods={['1Y']} />)
    fireEvent.change(screen.getByLabelText('选择一个时序指标'), { target: { value: indicator.id } })
    await screen.findByTestId('chart')
    expect(evaluateTimeSeriesIndicators).toHaveBeenLastCalledWith(expect.objectContaining({ indicator_instances: [{ indicator_id: indicator.id, indicator_revision: 3, parameters: {} }] }))
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    expect(evaluateTimeSeriesIndicators).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByText('应用参数'))
    await waitFor(() => expect(evaluateTimeSeriesIndicators).toHaveBeenCalledTimes(2))
    expect(evaluateTimeSeriesIndicators).toHaveBeenLastCalledWith(expect.objectContaining({ indicator_instances: [{ indicator_id: indicator.id, indicator_revision: 3, parameters: { window_1: 60 } }] }))
    expect(indicator.parameter_schema![0].default).toBe(20)
  })
  it('迟到的默认值结果不会覆盖新参数结果', async () => {
    let oldResolve!: (value: ReturnType<typeof response>) => void
    vi.mocked(evaluateTimeSeriesIndicators).mockReturnValueOnce(new Promise(resolve => { oldResolve = resolve as typeof oldResolve }))
      .mockResolvedValueOnce(response(60) as never)
    render(<TimeSeriesIndicatorPanel indicators={[indicator]} productId="510300.SH" productKind="etf" periods={['1Y']} />)
    fireEvent.change(screen.getByLabelText('选择一个时序指标'), { target: { value: indicator.id } })
    fireEvent.change(screen.getByLabelText('窗口期数'), { target: { value: '60' } })
    fireEvent.click(screen.getByText('应用参数'))
    await screen.findByText(/实际计算参数: 窗口期数=60/)
    await act(async () => { oldResolve(response(20)) })
    expect(screen.getByText(/实际计算参数: 窗口期数=60/)).toBeInTheDocument()
  })
})
