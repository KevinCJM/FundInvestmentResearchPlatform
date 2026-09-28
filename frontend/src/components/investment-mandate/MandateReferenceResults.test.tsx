import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import MandateReferenceResults from './MandateReferenceResults'
import type { MandateAssessment } from '../../services/strategicAllocation'

vi.mock('echarts-for-react', () => ({ default: ({ option }: any) => <output data-testid="frontier-chart">{JSON.stringify(option)}</output> }))

const arc = (count: number, from: number, to: number) => Array.from({ length: count }, (_, index) => {
  const share = index / (count - 1)
  return { node_id: index, status: 'optimal_to_tolerance', volatility: from + share * (to - from), expected_return: .03 + share * .09 }
})

const assessment = (reach: Record<string, unknown> | null): MandateAssessment => ({
  risk_decision: { selected_max_level: 4, authorized_max_level: 4, selected_volatility_cap: .095,
    applied_boundaries: [.015, .03, .06, .095, .13] },
  reference_diagnosis: { status: 'validated', reference_frontier: arc(20, .004, .138), constrained_frontier: arc(20, .004, .126),
    risk_boundaries: [.015, .03, .06, .095, .13], candidates: [], limitations: [], blockers: [],
    search_seed: 42, validation_seed: 104729, paths: 2000, reachability: reach },
  blockers: [],
} as unknown as MandateAssessment)

const reachable = { volatility_cap: .095, max_return_under_cap: .092, target_return: .07, binding: 'none',
  min_volatility_for_target: .04, required_risk_level: 3 }
const blocked = { volatility_cap: .095, max_return_under_cap: .082, target_return: .11, binding: 'volatility_cap',
  min_volatility_for_target: .118, required_risk_level: 5 }

const chart = () => JSON.parse(screen.getByTestId('frontier-chart').textContent!)
const marks = (option: any) => option.series[0].markLine.data.map((item: any) => item.name)
const named = (option: any, text: RegExp) => option.series[0].markLine.data.find((item: any) => text.test(item.name))

describe('前沿图与本页填的目标和约束', () => {
  it('把授权上限、目标收益和授权外区域画进图里，而不是只写在图下面', () => {
    render(<MandateReferenceResults value={assessment(reachable)} />)
    const option = chart()

    // 上限竖线和目标横线各自带上数值，读图不用回到文字段落里找。
    expect(named(option, /^授权上限 C4 9\.50%$/).xAxis).toBeCloseTo(9.5)
    expect(named(option, /^目标 7\.00%$/).yAxis).toBeCloseTo(7)
    // 超出授权的一段压成灰底，从上限一直盖到数据右端。
    expect(option.series[0].markArea.data[0][0].xAxis).toBeCloseTo(9.5)
    expect(option.series[0].markArea.data[0][1].xAxis).toBeCloseTo(13.8)
  })

  it('用户选的那条等级线换成强调样式，其余四条退为次要，不另画一条叠标签', () => {
    render(<MandateReferenceResults value={assessment(reachable)} />)
    const lines = chart().series[0].markLine.data

    expect(lines.map((item: any) => item.name)).toEqual(['C1', 'C2', 'C3', '授权上限 C4 9.50%', 'C5', '目标 7.00%'])
    expect(lines.filter((item: any) => /^C[1-5]$/.test(item.name)).every((item: any) => item.lineStyle.width === 1)).toBe(true)
    expect(lines.find((item: any) => /^授权上限/.test(item.name)).lineStyle.width).toBe(2)
  })

  it('授权内可选的那一段单独成序列并进图例，上限之外不画', () => {
    render(<MandateReferenceResults value={assessment(reachable)} />)
    const option = chart()

    expect(option.legend.data).toContain('授权内可选段')
    const segment = option.series.find((item: any) => item.name === '授权内可选段')
    expect(segment.data.filter(Boolean).every((pair: number[]) => pair[0] <= 9.5 + 1e-9)).toBe(true)
    expect(segment.data.filter((pair: unknown) => pair === null).length).toBeGreaterThan(0)
  })

  it('目标够得到时标成功色，够不到时标出缺口点和所需等级', () => {
    const { unmount } = render(<MandateReferenceResults value={assessment(reachable)} />)
    expect(chart().series[1].markPoint.data.map((item: any) => item.name)).toEqual(['上限下可达 9.20%'])
    unmount()

    render(<MandateReferenceResults value={assessment(blocked)} />)
    const points = chart().series[1].markPoint.data
    expect(points.map((item: any) => item.name)).toEqual(['上限下可达 8.20%', '达标需 11.80%（C5）'])
    // 缺口点落在目标收益那条横线上，横坐标是达标所需的最小波动。
    expect(points[1].coord[0]).toBeCloseTo(11.8)
    expect(points[1].coord[1]).toBeCloseTo(11)
  })

  it('没有可比口径的收益下限时不画目标横线', () => {
    // 资金目标与相对目标的"所需收益"不是纵轴的算术年化收益，画上去就是口径错误。
    render(<MandateReferenceResults value={assessment({ volatility_cap: .095, max_return_under_cap: .092, target_return: null, binding: 'none', min_volatility_for_target: null, required_risk_level: null })} />)
    const option = chart()

    expect(named(option, /^目标 /)).toBeUndefined()
    expect(named(option, /^授权上限 /)).toBeDefined()
  })

  it('上限未定时不画上限线、遮罩和可选段', () => {
    const pending = assessment(null)
    ;(pending.risk_decision as any).selected_volatility_cap = null
    render(<MandateReferenceResults value={pending} />)
    const option = chart()

    expect(marks(option)).toEqual(['C1', 'C2', 'C3', 'C4', 'C5'])
    expect(option.series[0].markArea).toBeUndefined()
    expect(option.series).toHaveLength(2)
    expect(option.legend.data).not.toContain('授权内可选段')
  })
})

it('displays a compound requirement curve in the objective diagnosis without a false flat target', () => {
  const value = assessment({ ...reachable, target_return: null })
  value.reference_diagnosis!.return_requirements = { arithmetic_floor: null, compound_floor: .04, volatility_cap: .095, status: 'resolved', benchmark_return: null, target_excess_return: null }
  value.reference_diagnosis!.target_curve = [{ volatility: 0, expected_return: .04 }, { volatility: .095, expected_return: .0443 }]
  render(<MandateReferenceResults value={value} />)
  expect(chart().series.find((s: any) => s.name === '同口径收益要求').data).toEqual([[0, 4], [9.5, 4.43]])
  expect(named(chart(), /^目标 /)).toBeUndefined()
  expect(screen.getByText(/授权波动范围内有参考点/)).toBeInTheDocument()
})
