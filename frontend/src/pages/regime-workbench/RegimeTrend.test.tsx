import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { adaptRegimeOverview, adaptRegimeResult } from './regimeResultAdapter'
import { resultFixture } from './regimeResultFixtures'
import { buildRegimePeakOption, buildRegimeTimelineOption, buildRegimeTrendOption } from './RegimeTimelineChart'
import RegimeEvidencePanel from './RegimeEvidencePanel'
import { ParameterInput } from './RegimeNodeInspector'

vi.mock('echarts-for-react', () => ({ default: () => <div /> }))

function trendResult() {
  const { overview, rows } = resultFixture('trend', 600)
  const mutableRows = rows as Array<(typeof rows)[number] & { features?: Record<string, number | null> }>
  mutableRows[0] = { ...mutableRows[0], features: { index_value: 3800, filtered_index: 3790, distance: -0.5,
    slope: 0.2, efficiency: 0.6, phase: 1, pending_count: 1, risk: 1 } }
  mutableRows[1] = { ...mutableRows[1], features: { index_value: 3801, filtered_index: null } }
  return adaptRegimeResult(adaptRegimeOverview(overview, 'trend', 'preview'), mutableRows)
}

describe('滤波牛熊震荡结果', () => {
  it('震荡证据显示合并边界、实际振幅、方向效率，并保留原始波段说明', () => {
    const result = trendResult()
    result.overview.mode = 'retrospective'
    result.points[0].raw.features = { phase_start_index: 0, phase_end_index: 2, phase_return: .02,
      sideways_start_index: 0, sideways_end_index: 5, sideways_range: .035,
      sideways_efficiency: .12, sideways_swing_count: 4 }
    render(<RegimeEvidencePanel result={result} interval={result.intervals[0]} />)
    const evidence = screen.getByLabelText('峰谷震荡依据')
    expect(evidence).toHaveTextContent('合并波段数：4')
    expect(evidence).toHaveTextContent('3.5%')
    expect(evidence).toHaveTextContent('0.1200')
    expect(evidence).toHaveTextContent(result.points[5].observation_date)
    expect(evidence).toHaveTextContent('尾部未完成区间不参与合并')
  })
  it('峰谷连线与保留拐点独立显示，边界和缺失不连线，证据标注完整峰谷日期', () => {
    const result = trendResult()
    result.overview.mode = 'retrospective'
    result.points[0].raw.features = { index_value: 3800, pivot: -1, boundary_line: 3800, phase_start_index: 0, phase_end_index: 2, phase_return: 0.1 }
    result.points[1].raw.features = { index_value: 3900, pivot: 0, boundary_line: 3990 }
    result.points[2].raw.features = { index_value: 4180, pivot: 1, boundary_line: 4180 }
    const option = buildRegimePeakOption(result)
    expect(option?.series).toMatchObject([
      { name: '定界指数' }, { name: '峰谷连线', connectNulls: false, data: expect.arrayContaining([[3, null]]) },
      { name: '保留峰值', data: [[2, 4180]] }, { name: '保留谷值', data: [[0, 3800]] },
    ])
    render(<RegimeEvidencePanel result={result} interval={result.intervals[0]} />)
    expect(screen.getByLabelText('峰谷定界依据')).toHaveTextContent('10%')
    expect(screen.getByText(/不提供交易生效日/)).toBeInTheDocument()
  })
  it('识别指数单独绘图，不与可替换主对照混用；预热保留断点', () => {
    const result = trendResult()
    const main = buildRegimeTimelineOption(result)
    expect(main.series).toMatchObject([{ data: expect.arrayContaining([[0, 1000]]) }])
    const trend = buildRegimeTrendOption(result)
    expect(trend?.series).toMatchObject([
      { name: '识别指数', data: expect.arrayContaining([[0, 3800], [1, 3801]]) },
      { name: '趋势滤波线', connectNulls: false, data: expect.arrayContaining([[0, 3790], [1, null]]) },
    ])
    const { overview, rows } = resultFixture('legacy', 600)
    expect(buildRegimeTrendOption(adaptRegimeResult(adaptRegimeOverview(overview, 'legacy', 'preview'), rows))).toBeNull()
  })

  it('牛市回调仍显示原始指数风险，事后结果注明合并前证据', () => {
    const result = trendResult()
    result.overview.mode = 'retrospective'
    render(<RegimeEvidencePanel result={result} interval={result.intervals[0]} />)
    expect(screen.getByText(/趋势内部阶段：牛市回调/)).toBeInTheDocument()
    expect(screen.getByText(/已触发回撤或急跌警报/)).toBeInTheDocument()
    expect(screen.getByText(/合并前的趋势证据/)).toBeInTheDocument()
    expect(screen.getByText('-0.5000')).toBeInTheDocument()
  })

  it('新增连续确认与幅度参数可编辑并保留数值类型和边界', () => {
    const confirmation = vi.fn()
    const band = vi.fn()
    render(<>
      <ParameterInput name="confirmation" schema={{ type: 'integer', title: '连续确认期数', minimum: 1, maximum: 252, default: 3 }} value={3} onChange={confirmation} />
      <ParameterInput name="band" schema={{ type: 'number', title: '价格偏离门槛', minimum: 0.00000001, maximum: 100, default: 1 }} value={1} onChange={band} />
    </>)
    const count = screen.getByRole('spinbutton', { name: '连续确认期数' })
    expect(count).toHaveAttribute('min', '1')
    expect(count).toHaveAttribute('step', '1')
    fireEvent.change(count, { target: { value: '5' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: '价格偏离门槛' }), { target: { value: '1.25' } })
    expect(confirmation).toHaveBeenLastCalledWith(5)
    expect(band).toHaveBeenLastCalledWith(1.25)
  })
})
