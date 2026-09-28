import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import ScopeFrontierChart, { type ScopeFrontierPoint } from './ScopeFrontierChart'

vi.mock('../../i18n/runtime', async importOriginal => ({ ...await importOriginal<object>(),
  useI18n: () => ({ s: (key: string) => key, locale: 'zh-CN' }),
}))
vi.mock('echarts-for-react', async () => {
  const { forwardRef } = await import('react')
  return { default: forwardRef<unknown, { option: object }>(({ option }, _ref) => <output data-testid="scope-chart-options">{JSON.stringify(option)}</output>) }
})
const point = (volatility: number, expected_return: number): ScopeFrontierPoint => ({ volatility, expected_return, weights: {}, status: 'optimal_to_tolerance' })
const option = () => JSON.parse(screen.getByTestId('scope-chart-options').textContent!)

describe('ScopeFrontierChart', () => {
  it('overlays the frozen scale and original constrained frontier with distinct strokes, shared axes and table rows', () => {
    render(<ScopeFrontierChart points={[point(.02, .01)]} targetReturn={.06} volatilityCap={.1}
      referencePoints={[point(.01, -.02), { ...point(.1, .1), status: 'iteration_limit' }, point(.4, .3)]}
      constrainedPoints={[point(.02, .01), point(.3, .2)]} />)
    const value = option()
    expect(value.series[3]).toMatchObject({ name: 'scopeFeasibility.chart.reference', connectNulls: false,
      lineStyle: { type: 'dashed' }, data: [[1, -2], null, [40, 30]] })
    expect(value.series[4]).toMatchObject({ name: 'scopeFeasibility.chart.constrained', lineStyle: { type: 'dotted' }, data: [[2, 1], [30, 20]] })
    expect(value.xAxis.max).toBeGreaterThan(40)
    expect(value.yAxis.max).toBeGreaterThan(30)
    expect(value.yAxis.min).toBeLessThan(-2)
    expect(value.series[1].data).toEqual([[10, 6]])
    expect(screen.getByRole('img')).toHaveAccessibleName(/scopeFeasibility.chart.reference/)
    fireEvent.click(screen.getByText('scopeFeasibility.chart.table', { selector: 'summary' }))
    expect(screen.getByRole('table')).toHaveTextContent('scopeFeasibility.chart.reference 3')
    expect(screen.getByRole('table')).toHaveTextContent('scopeFeasibility.chart.constrained 2')
    expect(screen.getByRole('table')).toHaveTextContent('40.00%')
  })

  it('keeps the whole frontier, goal boundary, shaded region and actual candidate coordinates', () => {
    render(<ScopeFrontierChart points={[point(.02, .01), point(.08, .04), point(.2, .1)]}
      targetReturn={.06} volatilityCap={.1} candidate={{ volatility: .075, expected_return: .035, weights: { '<script>': 1 } }} />)
    const value = option()
    expect(value.series[0].data).toEqual([[2, 1], [8, 4], [20, 10]])
    expect(value.series[1].markLine.data).toEqual([{ yAxis: 6 }, { xAxis: 10 }])
    expect(value.series[1].markArea.data).toEqual([[{ xAxis: 0, yAxis: 6 }, { xAxis: 10, yAxis: value.yAxis.max }]])
    expect(value.series[1]).toMatchObject({ symbol: 'diamond', data: [[10, 6]] })
    expect(value.series[2].data[0][0]).toBeCloseTo(7.5)
    expect(value.series[2].data[0][1]).toBeCloseTo(3.5)
    expect(value.tooltip).toMatchObject({ renderMode: 'richText', confine: true })
    expect(screen.getByRole('img')).toHaveAccessibleName('scopeFeasibility.chart.return scopeFeasibility.chart.description')
    expect(screen.getByText('scopeFeasibility.chart.meaning')).toBeInTheDocument()
    fireEvent.click(screen.getByText('scopeFeasibility.chart.table', { selector: 'summary' }))
    expect(screen.getByRole('table', { name: 'scopeFeasibility.chart.table' })).toBeInTheDocument()
    expect(screen.getByRole('table')).toHaveTextContent('7.50%')
    expect(document.querySelector('script')).toBeNull()
  })

  it('expands both axes to keep remote targets and actual candidates visible', () => {
    const { rerender } = render(<ScopeFrontierChart points={[point(.03, -.06), point(.1, -.02)]} targetReturn={.8} volatilityCap={.7} />)
    expect(option().xAxis.max).toBeGreaterThan(70)
    expect(option().yAxis.max).toBeGreaterThan(80)
    expect(option().yAxis.min).toBeLessThan(-6)
    rerender(<ScopeFrontierChart points={[point(.03, -.06)]} targetReturn={-.4} volatilityCap={.1}
      candidate={{ volatility: .9, expected_return: .9, weights: {} }} />)
    expect(option().yAxis.min).toBeLessThan(-40)
    expect(option().yAxis.max).toBeGreaterThan(90)
    expect(option().xAxis.max).toBeGreaterThan(90)
    expect(option().series[1].markArea.data[0][0].yAxis).toBe(-40)
  })

  it('keeps failed or invalid observations as gaps without declaring reachability', () => {
    render(<ScopeFrontierChart points={[point(.01, .01), { ...point(.02, .02), status: 'infeasible_certified' },
      point(.03, .03), point(Number.NaN, .04), point(-.01, .03), { ...point(.01, .01), volatility: null }]}
      targetReturn={.09} volatilityCap={.1} />)
    expect(option().series[0].data).toEqual([[1, 1], null, [3, 3], null, null, null])
    expect(option().series[0].connectNulls).toBe(false)
    expect(screen.getByText('scopeFeasibility.chart.gaps')).toBeInTheDocument()
    fireEvent.click(screen.getByText('scopeFeasibility.chart.table', { selector: 'summary' }))
    expect(screen.getAllByText('scopeFeasibility.chart.unresolved')).toHaveLength(4)
  })

  it('draws a visible point and useful axes for a single cash asset', () => {
    render(<ScopeFrontierChart points={[point(0, 0)]} targetReturn={0} volatilityCap={0} />)
    expect(option().series[0]).toMatchObject({ data: [[0, 0]], showSymbol: true, symbolSize: 5 })
    expect(option().xAxis.max).toBeGreaterThan(0)
    expect(option().yAxis.max).toBeGreaterThan(0)
    expect(option().series[1].data).toEqual([[0, 0]])
  })

  it('omits the intersection and target region when either comparable boundary is missing', () => {
    const { rerender } = render(<ScopeFrontierChart points={[point(.05, .02)]} targetReturn={null} volatilityCap={.1} />)
    expect(option().series[1].markLine.data).toEqual([{ xAxis: 10 }])
    expect(option().series[1].markArea).toBeUndefined()
    expect(option().series[1].data).toEqual([])
    expect(screen.getByText('scopeFeasibility.chart.noTarget')).toBeInTheDocument()
    rerender(<ScopeFrontierChart points={[]} targetReturn={.03} volatilityCap={null} />)
    expect(option().series[1].markLine.data).toEqual([{ yAxis: 3 }])
    expect(option().series[1].data).toEqual([])
    expect(option().series[2].data).toEqual([])
    expect(option().series[1].markArea).toBeUndefined()
  })

  it('toggles each curve and the whole goal overlay without changing axes, data or table; restores all', () => {
    const props = { points: [point(.1, .04)], referencePoints: [point(.2, .1)], constrainedPoints: [point(.15, .07)],
      targetReturn: .06, volatilityCap: .1, candidate: { volatility: .08, expected_return: .05, weights: {} } }
    const { rerender } = render(<ScopeFrontierChart {...props} />)
    const before = option()
    expect(screen.getAllByRole('checkbox')).toHaveLength(5)
    fireEvent.click(screen.getByRole('checkbox', { name: 'scopeFeasibility.chart.frontier' }))
    expect(option().legend.selected['scopeFeasibility.chart.frontier']).toBe(false)
    expect(option().legend.selected['frontierLegend.target']).toBe(true)
    fireEvent.click(screen.getByRole('checkbox', { name: 'frontierLegend.target' }))
    expect(option().legend.selected['frontierLegend.target']).toBe(false)
    expect(option().series[0]).not.toHaveProperty('markLine')
    expect(option().series[1]).toMatchObject({ name: 'frontierLegend.target', markLine: before.series[1].markLine, markArea: before.series[1].markArea })
    rerender(<ScopeFrontierChart {...props} targetReturn={.08} />)
    expect(screen.getByRole('checkbox', { name: 'frontierLegend.target' })).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: 'frontierLegend.hideAll' }))
    expect(Object.values(option().legend.selected).every(value => value === false)).toBe(true)
    expect(screen.getByText('frontierLegend.allHidden')).toHaveAttribute('role', 'status')
    expect(option().xAxis).toEqual(before.xAxis)
    expect(option().yAxis).toEqual(before.yAxis)
    expect(option().series[0].data).toEqual(before.series[0].data)
    fireEvent.click(screen.getByText('scopeFeasibility.chart.table', { selector: 'summary' }))
    expect(screen.getByRole('table')).toHaveTextContent('scopeFeasibility.chart.reference')
    fireEvent.click(screen.getByRole('button', { name: 'frontierLegend.showAll' }))
    expect(screen.getAllByRole('checkbox').every(box => (box as HTMLInputElement).checked)).toBe(true)
    expect(screen.queryByText('frontierLegend.allHidden')).not.toBeInTheDocument()
    rerender(<ScopeFrontierChart points={[]} targetReturn={null} volatilityCap={null} referencePoints={props.referencePoints} />)
    expect(screen.getAllByRole('checkbox')).toHaveLength(1)
    expect(screen.queryByRole('checkbox', { name: 'frontierLegend.target' })).not.toBeInTheDocument()
  })
})


it('draws the funding compound hurdle, shading and intersection on the supplied compound coordinates', () => {
  render(<ScopeFrontierChart compound points={[point(.02, .03), point(.06, .065)]}
    referencePoints={[point(.03, .05)]} constrainedPoints={[point(.04, .045)]}
    targetReturn={.0409} volatilityCap={.06} />)
  const value = option()
  expect(value.series[1].markLine.data).toEqual([{ yAxis: 4.09 }, { xAxis: 6 }])
  expect(value.series[1].data).toEqual([[6, 4.09]])
  expect(value.series[1].markArea.data[0][0]).toEqual({ xAxis: 0, yAxis: 4.09 })
  expect(value.series[0].data).toEqual([[2, 3], [6, 6.5]])
  expect(value.series[3].data).toEqual([[3, 5]])
  expect(value.series[4].data).toEqual([[4, 4.5]])
  expect(screen.getByText('scopeFeasibility.chart.compound.meaning')).toBeInTheDocument()
  expect(screen.getByRole('img')).toHaveAccessibleName(/compound.description/)
})

it('compares mixed arithmetic and compound goals on one arithmetic curve without a false flat target', () => {
  render(<ScopeFrontierChart points={[point(.02, .03), point(.06, .065)]}
    targetReturn={.04} volatilityCap={.06}
    targetCurve={[{ volatility: 0, expected_return: .04 }, { volatility: .06, expected_return: .0418 }]} />)
  const value = option()
  expect(value.series[1].data).toEqual([])
  expect(value.series[1].markLine.data).toEqual([{ xAxis: 6 }])
  expect(value.series[1].markArea).toBeUndefined()
  expect(value.series[2]).toMatchObject({ name: 'frontierLegend.target', data: [[0, 4], [6, 4.18]] })
  expect(screen.getByText('policyFrontier.returnCurveHelp')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('checkbox', { name: 'frontierLegend.target' }))
  expect(option().legend.selected['frontierLegend.target']).toBe(false)
  expect(option().yAxis).toEqual(value.yAxis)
})
