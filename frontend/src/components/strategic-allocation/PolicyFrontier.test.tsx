import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PolicyFrontier from './PolicyFrontier'
import { getPolicyFrontier, type PolicyFrontierResult, type PolicyRequest, type PolicyPreview } from '../../services/strategicAllocation'

vi.mock('../../services/strategicAllocation', async importOriginal => ({ ...await importOriginal<object>(), getPolicyFrontier: vi.fn() }))
vi.mock('echarts-for-react', async () => {
  const { forwardRef } = await import('react')
  return { default: forwardRef((_props: any, _ref) => <output data-testid="frontier-options">{JSON.stringify(_props.option)}</output>) }
})
const request = { mandate_id: 'goal', cma_id: 'cma', constraints: {}, group_limits: [], seed: 42, candidate_count: 2000, uncertainty_penalty: 1 } as PolicyRequest
const point = (volatility: number, expected_return: number) => ({ volatility, expected_return, status: 'optimal_to_tolerance', weights: {} })
function evidence(): PolicyFrontierResult {
  return { mode: 'single', basis: 'test', research_only: true, execution: {} as any,
    additional_checks: { benchmark: false, funding: false, all_models: false },
    views: [{ id: 'cma', name: '当前 CMA', cma_hash: 'hash-cma', moment_basis: 'source_model', constraint_error: null, cash_floor: .1, target_return: .0772, volatility_cap: .095, limits: {}, groups: [],
      reference: { status: 'optimal_to_tolerance', complete: true, max_return: .0417, points: [point(0, 0), point(.226, .0417)] },
      configured: { status: 'optimal_to_tolerance', complete: true, max_return: .0376, points: [point(0, 0), point(.2034, .0376)] } }] }
}
const options = () => JSON.parse(screen.getByTestId('frontier-options').textContent!)
beforeEach(() => { vi.clearAllMocks(); vi.mocked(getPolicyFrontier).mockResolvedValue(evidence()) })

describe('SAA target/frontier diagnostics', () => {
  it('keeps both curves and the target for an unreachable study without policy candidates', async () => {
    render(<PolicyFrontier request={request} clock="2019-12-31" disabled={false} result={null} />)
    await screen.findByTestId('frontier-options')
    expect(screen.getByText(/收益上限也只有 3.76%/)).toBeInTheDocument()
    expect(options().series[2].data[0][0]).toBeCloseTo(9.5)
    expect(options().series[2].data[0][1]).toBeCloseTo(7.72)
    expect(options().series[1].data).toHaveLength(2)
    expect(options().yAxis.max).toBeGreaterThan(7.72)
    expect(options().series[2].markArea.data[0][1].xAxis).toBeCloseTo(9.5)
  })
  it('shows feasible candidates on the same axes and never grants adoption', async () => {
    const data = evidence(); data.views[0].target_return = .02
    data.views[0].configured.points.splice(1, 0, point(.08, .025))
    vi.mocked(getPolicyFrontier).mockResolvedValue(data)
    const result = { request, candidates: [{ name: '低风险方案', metrics: { expected_return: .025, volatility: .08 } }] } as PolicyPreview
    render(<PolicyFrontier request={request} clock={null} disabled={false} result={result} />)
    await screen.findByText(/存在进入目标区域的已验证点/)
    expect(options().series[3].data[0].value).toEqual([8, 2.5])
    expect(screen.getByText(/是否可采纳，还需比较配置候选/)).toBeInTheDocument()
  })
  it('does not join failed points or hide the target when constraints conflict', async () => {
    const data = evidence()
    data.views[0].configured = { ...data.views[0].configured, status: 'infeasible_certified', complete: false,
      max_return: null, points: [{ volatility: null, expected_return: null, weights: {}, status: 'infeasible_certified' }] }
    vi.mocked(getPolicyFrontier).mockResolvedValue(data)
    render(<PolicyFrontier request={request} clock={null} disabled={false} result={null} />)
    await screen.findByText(/约束相互冲突/)
    expect(options().series[1].connectNulls).toBe(false)
    expect(options().series[1].data).toEqual([null])
    expect(options().series[2].data).toHaveLength(1)
  })
  it('immediately removes stale diagrams and ignores late responses on clock changes', async () => {
    let finish: (value: PolicyFrontierResult) => void = () => {}
    vi.mocked(getPolicyFrontier).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    const props = { request, disabled: false, result: null }
    const rendered = render(<PolicyFrontier {...props} clock="2019-12-31" />)
    await waitFor(() => expect(getPolicyFrontier).toHaveBeenCalledTimes(1))
    rendered.rerender(<PolicyFrontier {...props} clock={undefined} disabled />)
    await act(async () => finish(evidence()))
    expect(screen.queryByTestId('frontier-options')).not.toBeInTheDocument()
    expect(screen.getByText(/请先完成目标/)).toBeInTheDocument()
  })
  it('allows retry and separates each original model in common mode', async () => {
    vi.mocked(getPolicyFrontier).mockRejectedValueOnce(new Error('服务尚未就绪'))
    const data = evidence(); data.mode = 'compatible_all_models'; data.additional_checks.all_models = true
    data.views.push({ ...data.views[0], id: 'second', name: '第二个模型', target_return: .06 })
    vi.mocked(getPolicyFrontier).mockResolvedValueOnce(data)
    render(<PolicyFrontier request={request} clock={null} disabled={false} result={null} />)
    await screen.findByText('服务尚未就绪')
    fireEvent.click(screen.getByRole('button', { name: '重新计算前沿' }))
    await screen.findByRole('checkbox', { name: '1. 当前 CMA' })
    fireEvent.click(screen.getByRole('checkbox', { name: '目标与约束' }))
    expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
    expect(options().series.filter((series: any) => series.type === 'line')).toHaveLength(4)
    expect(screen.getByRole('checkbox', { name: '2. 第二个模型' })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: '目标与约束' })).not.toBeChecked()
    expect(screen.getByText(/各条前沿都进入目标区/)).toBeInTheDocument()
  })

  it('keeps goal layers together and each candidate independent without recalculating or changing the conclusion', async () => {
    const result = { request, candidates: [
      { id: 'minimum-risk', name: '低风险方案', metrics: { expected_return: .025, volatility: .08 } },
      { id: 'maximum-return', name: '高收益方案', metrics: { expected_return: .04, volatility: .2 } },
    ] } as PolicyPreview
    const { rerender } = render(<PolicyFrontier request={request} clock={null} disabled={false} result={result} />)
    await screen.findByTestId('frontier-options')
    const before = options()
    expect(screen.getAllByRole('checkbox')).toHaveLength(5)
    fireEvent.click(screen.getByRole('checkbox', { name: '目标与约束' }))
    expect(options().legend.selected['目标与约束']).toBe(false)
    expect(options().series[1]).not.toHaveProperty('markArea')
    expect(options().series[2]).toMatchObject({ markLine: before.series[2].markLine, markArea: before.series[2].markArea })
    fireEvent.click(screen.getByRole('checkbox', { name: '候选 1 · 低风险方案' }))
    expect(options().legend.selected['候选 1 · 低风险方案']).toBe(false)
    expect(options().legend.selected['候选 2 · 高收益方案']).toBe(true)
    rerender(<PolicyFrontier request={{ ...request }} clock={null} disabled={false} result={{ ...result, candidates: [...result.candidates].reverse() }} />)
    expect(screen.getByRole('checkbox', { name: '候选 2 · 低风险方案' })).not.toBeChecked()
    fireEvent.click(screen.getByRole('button', { name: '全部隐藏' }))
    expect(Object.values(options().legend.selected).every(value => value === false)).toBe(true)
    expect(options().xAxis).toEqual(before.xAxis)
    expect(options().yAxis).toEqual(before.yAxis)
    expect(screen.getByText(/收益上限也只有 3.76%/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: '全部显示' }))
    expect(screen.getAllByRole('checkbox').filter(box => box.getAttribute('aria-label')).every(box => (box as HTMLInputElement).checked)).toBe(true)
    expect(getPolicyFrontier).toHaveBeenCalledTimes(1)
  })
})

function commonStudy() {
  const data = evidence(); data.mode = 'compatible_all_models'; data.additional_checks.all_models = true
  data.views.push({ ...structuredClone(data.views[0]), id: 'second', cma_hash: 'hash-second', name: '当前 CMA' })
  const refs = data.views.map(v => ({ cma_id: v.id, content_hash: v.cma_hash, weight: null }))
  const commonRequest = { ...request, mode: 'compatible_all_models', cma_id: null, cma_refs: refs } as PolicyRequest
  const rows = data.views.map((v, i) => ({ cma_id: v.id, cma_hash: v.cma_hash, name: v.name,
    metrics: { expected_return: .03 + i * .01, volatility: .06 + i * .02 }, within_limits: i === 0, violations: i ? ['资金检查失败'] : [] }))
  const result = { request: commonRequest, multi_cma: { refs }, candidates: [], unavailable_candidates: [{ id: 'compatible', name: '共同组合',
    metrics: { expected_return: .99, volatility: .99 }, cross_model_results: rows, available: false }] } as unknown as PolicyPreview
  return { data, commonRequest, result }
}

it('overlays named source frontiers and the same rejected portfolio per source, not summary metrics', async () => {
  const { data, commonRequest, result } = commonStudy()
  vi.mocked(getPolicyFrontier).mockResolvedValue(data)
  const { rerender } = render(<PolicyFrontier request={commonRequest} clock={null} disabled={false} result={result} />)
  await screen.findByTestId('frontier-options')
  expect(options().series.filter((s: any) => s.id?.startsWith('candidate:')).map((s: any) => s.data[0].value)).toEqual([[6, 3], [8, 4]])
  expect(screen.getByRole('table', { name: '共同组合在各 LTCMA 下的结果' })).toHaveTextContent('未通过')
  expect(options().legend.selected['1. 当前 CMA · 无额外约束参考']).toBe(false)
  const before = options()
  fireEvent.click(screen.getByRole('checkbox', { name: '1. 当前 CMA' }))
  fireEvent.click(screen.getByRole('checkbox', { name: '显示无额外约束参考线' }))
  expect(options().legend.selected['1. 当前 CMA']).toBe(false)
  expect(options().legend.selected['2. 当前 CMA']).toBe(true)
  expect(options().legend.selected['1. 当前 CMA · 无额外约束参考']).toBe(true)
  expect(options().xAxis).toEqual(before.xAxis)
  const stale = structuredClone(result); const failed = stale.unavailable_candidates![0]
  if (failed.id === 'compatible') failed.cross_model_results![0].cma_hash = 'wrong-version'
  rerender(<PolicyFrontier request={commonRequest} clock={null} disabled={false} result={stale} />)
  expect(options().series.filter((s: any) => s.id?.startsWith('candidate:'))).toHaveLength(0)
  expect(screen.queryByRole('table')).not.toBeInTheDocument()
  expect(getPolicyFrontier).toHaveBeenCalledTimes(1)
})

it('labels the fused frontier and hides source comparison and reference curves by default', async () => {
  const data = evidence(); data.mode = 'parameter_average'; data.views[0].moment_basis = 'parameter_average'
  vi.mocked(getPolicyFrontier).mockResolvedValue(data)
  render(<PolicyFrontier request={{ ...request, mode: 'parameter_average' }} clock={null} disabled={false} result={null} />)
  await screen.findByTestId('frontier-options')
  expect(screen.getByRole('checkbox', { name: '融合参数前沿' })).toBeChecked()
  expect(screen.queryByRole('combobox')).not.toBeInTheDocument()
  expect(screen.getByText(/不是对曲线取平均/)).toBeInTheDocument()
  expect(options().legend.selected['大类资产前沿']).toBe(false)
  fireEvent.click(screen.getByRole('checkbox', { name: '显示无额外约束参考线' }))
  expect(options().legend.selected['大类资产前沿']).toBe(true)
  expect(getPolicyFrontier).toHaveBeenCalledTimes(1)
})

it('removes the old common curves and projections immediately when switching to parameter fusion', async () => {
  const { data, commonRequest, result } = commonStudy()
  vi.mocked(getPolicyFrontier).mockResolvedValueOnce(data)
  const { rerender } = render(<PolicyFrontier request={commonRequest} clock={null} disabled={false} result={result} />)
  await screen.findByTestId('frontier-options')
  let finish: (data: PolicyFrontierResult) => void = () => {}
  vi.mocked(getPolicyFrontier).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const fusedRequest = { ...commonRequest, mode: 'parameter_average' } as PolicyRequest
  rerender(<PolicyFrontier request={fusedRequest} clock={null} disabled={false} result={result} />)
  expect(screen.queryByTestId('frontier-options')).not.toBeInTheDocument()
  await waitFor(() => expect(getPolicyFrontier).toHaveBeenCalledTimes(2))
  await act(async () => finish({ ...evidence(), mode: 'parameter_average' }))
  expect(options().series.filter((s: any) => s.id?.startsWith('configured:'))).toHaveLength(1)
  expect(options().series.filter((s: any) => s.id?.startsWith('candidate:'))).toHaveLength(0)
})

it('gates comparison on current continuous evidence, never on grid samples or hidden legend entries', async () => {
  const data = evidence(); data.target_check = { status: 'infeasible', reason: 'target_outside' }
  vi.mocked(getPolicyFrontier).mockResolvedValueOnce(data)
  const controls = (gate: { disabled: boolean; reason: string }) => <><button disabled={gate.disabled}>比较测试</button><p>{gate.reason}</p></>
  const { rerender } = render(<PolicyFrontier request={request} clock={null} disabled={false} result={null}>{controls}</PolicyFrontier>)
  expect(screen.getByRole('button', { name: '比较测试' })).toBeDisabled()
  await screen.findByText(/无法同时满足年收益至少 7.72%/)
  expect(screen.getByRole('button', { name: '比较测试' })).toBeDisabled()
  fireEvent.click(screen.getByRole('button', { name: '全部隐藏' }))
  expect(screen.getByRole('button', { name: '比较测试' })).toBeDisabled()
  let finish!: (v: PolicyFrontierResult) => void
  vi.mocked(getPolicyFrontier).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  rerender(<PolicyFrontier request={{ ...request, constraints: { cash: { min_weight: 0, max_weight: 1, max_abs_tilt: .1 } } }} clock={null} disabled={false} result={null}>{controls}</PolicyFrontier>)
  expect(screen.queryByText(/无法同时满足年收益至少/)).not.toBeInTheDocument()
  expect(screen.getByText(/正在核验当前配置/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: '比较测试' })).toBeDisabled()
  await waitFor(() => expect(finish).toBeTypeOf('function'))
  await act(async () => finish({ ...data, target_check: { status: 'feasible', reason: null } }))
  // The continuous solve may find a feasible point between plot samples.
  expect(screen.getByRole('button', { name: '比较测试' })).toBeEnabled()
})

it('keeps comparison available for unresolved or failed diagnostics and ignores late prior evidence', async () => {
  let stale!: (v: PolicyFrontierResult) => void
  vi.mocked(getPolicyFrontier).mockImplementationOnce(() => new Promise(resolve => { stale = resolve }))
  const controls = (gate: { disabled: boolean; reason: string }) => <><button disabled={gate.disabled}>比较测试</button><p>{gate.reason}</p></>
  const { rerender } = render(<PolicyFrontier request={request} clock="2019-12-31" disabled={false} result={null}>{controls}</PolicyFrontier>)
  await waitFor(() => expect(stale).toBeTypeOf('function'))
  const unresolved = { ...evidence(), target_check: { status: 'undetermined', reason: null } } as PolicyFrontierResult
  vi.mocked(getPolicyFrontier).mockResolvedValueOnce(unresolved)
  rerender(<PolicyFrontier request={request} clock="2020-12-31" disabled={false} result={null}>{controls}</PolicyFrontier>)
  await screen.findByText(/前沿尚未完成可达性验证/)
  await act(async () => stale({ ...evidence(), target_check: { status: 'infeasible', reason: 'target_outside' } }))
  expect(screen.getByRole('button', { name: '比较测试' })).toBeEnabled()
  vi.mocked(getPolicyFrontier).mockRejectedValueOnce(new Error('网络错误'))
  rerender(<PolicyFrontier request={request} clock="2021-12-31" disabled={false} result={null}>{controls}</PolicyFrontier>)
  await screen.findByText('网络错误')
  expect(screen.getByRole('button', { name: '比较测试' })).toBeEnabled()
})

it('plots compound requirements at each risk and keeps the curve in the goal legend group', async () => {
  const data = evidence(); const view = data.views[0]
  view.target_return = null
  view.return_requirements = { arithmetic_floor: null, compound_floor: .04, volatility_cap: .095, status: 'resolved', benchmark_return: null, target_excess_return: null }
  view.target_curve = [{ volatility: 0, expected_return: .04 }, { volatility: .095, expected_return: .0443 }, { volatility: .2, expected_return: .0584 }]
  vi.mocked(getPolicyFrontier).mockResolvedValue(data)
  render(<PolicyFrontier request={request} clock={null} disabled={false} result={null} />)
  await screen.findByTestId('frontier-options')
  expect(options().series.find((s: any) => s.id === 'required:cma').data).toEqual([[0, 4], [9.5, 4.43], [20, 5.84]])
  expect(options().series[2].data).toEqual([])
  expect(options().series[2].markArea).toBeUndefined()
  expect(screen.queryByText(/目标点是收益下限与风险上限的交点/)).not.toBeInTheDocument()
  const before = options().yAxis
  fireEvent.click(screen.getByRole('checkbox', { name: '目标与约束' }))
  expect(options().legend.selected['目标与约束']).toBe(false)
  expect(options().yAxis).toEqual(before)
  expect(getPolicyFrontier).toHaveBeenCalledTimes(1)
})

it('uses a different absolute benchmark hurdle under each original model', async () => {
  const { data, commonRequest } = commonStudy()
  data.views.forEach((view, i) => {
    view.target_return = i ? .08 : .05
    view.target_curve = [{ volatility: 0, expected_return: view.target_return }, { volatility: .2, expected_return: view.target_return }]
  })
  vi.mocked(getPolicyFrontier).mockResolvedValue(data)
  render(<PolicyFrontier request={commonRequest} clock={null} disabled={false} result={null} />)
  await screen.findByTestId('frontier-options')
  expect(options().series.filter((s: any) => s.id?.startsWith('required:')).map((s: any) => s.data[0][1])).toEqual([5, 8])
  expect(options().series.find((s: any) => s.type === 'scatter').data).toEqual([])
})
