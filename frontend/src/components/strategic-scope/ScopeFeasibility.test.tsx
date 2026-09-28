import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import ScopeFeasibility from './ScopeFeasibility'
import { getScopeFeasibility, type ScopeFeasibilityInput, type ScopeFeasibilityResult } from '../../services/strategicScope'

const context = vi.hoisted(() => ({ identity: 'release-one' }))
vi.mock('../../app/ResearchContext', () => ({ useResearchContextIdentity: () => context.identity }))
vi.mock('../../services/strategicScope', async original => ({ ...await original<typeof import('../../services/strategicScope')>(), getScopeFeasibility: vi.fn() }))
vi.mock('./ScopeFrontierChart', () => ({ default: (props: unknown) => <div data-testid="scope-chart">{JSON.stringify(props)}</div> }))
const input: ScopeFeasibilityInput = { mandate_id: 'goal-one', as_of: '2019-12-31', product_version_ids: ['pool-one'] }
const result: ScopeFeasibilityResult = {
  status: 'feasible', reason_code: 'VERIFIED', research_only: true, reasons: [{ code: 'VERIFIED', message: '已找到符合当前边界的历史配置。' }],
  mandate: { id: 'goal-one', content_hash: 'hash', target_return: .08, volatility_cap: .1, min_cash_weight: .1 },
  sample: { requested_start: '2014-12-31', requested_end: '2019-12-31', actual_start: '2015-01-05', actual_end: '2019-12-31', observations: 1200, common_days: 1201, excluded_return_periods: 0, missing_trading_days: 0 },
  frontier: { status: 'optimal_to_tolerance', complete: true, constraints_applied: true, points: [{ volatility: .09, expected_return: .085, weights: { a: 1 }, status: 'optimal_to_tolerance' }] },
  target_check: { status: 'feasible', target_return: .08, volatility_cap: .1, max_return_under_cap: .085, candidate: { volatility: .09, expected_return: .085, weights: { a: 1 } } },
  reference_comparison: { status: 'available', risk_scale_ref: { id: 'original-scale', content_hash: 'frozen' },
    name: '目标原始标尺', as_of: '2019-12-31', currency: 'CNY', sample_start: '2010-01-04', sample_end: '2019-12-30',
    points: [{ volatility: .15, expected_return: .1, status: 'optimal_to_tolerance' }],
    constrained_points: [{ volatility: .12, expected_return: .09, status: 'optimal_to_tolerance' }] },
}
beforeEach(() => { context.identity = 'release-one'; vi.clearAllMocks(); vi.mocked(getScopeFeasibility).mockResolvedValue(result) })

it('uses the selected scope and goal, shows the chart and reruns after changing the history window', async () => {
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  expect(await screen.findByText('历史测算可达')).toBeInTheDocument()
  expect(screen.getByRole('option', { name: '近 5 年' })).toBeInTheDocument()
  expect(screen.getByText('共同样本：2015-01-05 至 2019-12-31，1200 个日收益观测。')).toBeInTheDocument()
  expect(getScopeFeasibility).toHaveBeenCalledWith({ ...input, window: { kind: '5Y' } }, expect.any(AbortSignal))
  expect(screen.getByTestId('scope-chart')).toHaveTextContent('"targetReturn":0.08')
  expect(screen.getByTestId('scope-chart')).toHaveTextContent('"referencePoints":[{"volatility":0.15')
  expect(screen.getByTestId('scope-chart')).toHaveTextContent('"constrainedPoints":[{"volatility":0.12')
  expect(screen.getByText(/原始参考：目标原始标尺/)).toBeInTheDocument()
  expect(screen.getByText('原始参考样本：2010-01-04 至 2019-12-30。')).toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('历史测算窗口'), { target: { value: '2Y' } })
  expect(screen.queryByText('历史测算可达')).not.toBeInTheDocument()
  await waitFor(() => expect(getScopeFeasibility).toHaveBeenLastCalledWith({ ...input, window: { kind: '2Y' } }, expect.any(AbortSignal)))
})

it('centres the working mascot in the reserved chart area while measuring', async () => {
  const view = render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  const waiting = await screen.findByRole('status')
  expect(waiting).toHaveTextContent('正在测算')
  // 装饰性形象：alt="" + aria-hidden，只能按资产查询。
  expect(view.container.querySelector('img[src*="mascot-working"]')).toBeInTheDocument()
  // 8.1 的整面板例外：预留图表高度居中，不铺灰底骨架屏。
  expect(waiting.className).toContain('min-h-72')
  expect(waiting.className).toContain('place-items-center')
  expect(view.container.querySelector('.animate-pulse')).toBeNull()
  await screen.findByText('历史测算可达')
  expect(view.container.querySelector('img[src*="mascot-working"]')).not.toBeInTheDocument()
})

it('invalidates immediately and discards late results when the selection changes', async () => {
  let finish!: (value: ScopeFeasibilityResult) => void
  vi.mocked(getScopeFeasibility).mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
  const view = render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await waitFor(() => expect(getScopeFeasibility).toHaveBeenCalledTimes(1))
  const oldSignal = vi.mocked(getScopeFeasibility).mock.calls[0][1]!
  view.rerender(<ScopeFeasibility input={{ ...input, mandate_id: 'goal-two' }} clock="2019-12-31" />)
  expect(oldSignal.aborted).toBe(true)
  await act(async () => finish({ ...result, status: 'infeasible' }))
  expect(screen.queryByText('历史测算不可达')).not.toBeInTheDocument()
  await screen.findByText('历史测算可达')
  context.identity = 'release-two'
  view.rerender(<ScopeFeasibility input={{ ...input, mandate_id: 'goal-two' }} clock="2019-12-31" />)
  expect(screen.queryByTestId('scope-chart')).not.toBeInTheDocument()
  view.rerender(<ScopeFeasibility input={input} clock={undefined} />)
  expect(screen.queryByTestId('scope-chart')).not.toBeInTheDocument()
  expect(screen.getByText('正在确认研究日，暂不计算。')).toBeInTheDocument()
})

it('does not infer infeasibility from failed or absent frontier points and allows retry after a failure', async () => {
  vi.mocked(getScopeFeasibility).mockRejectedValueOnce(new Error('样本读取失败')).mockResolvedValueOnce({
    ...result, status: 'undetermined', reasons: [{ code: 'SHORT', message: '共同样本不足。' }], frontier: null, sample: null, target_check: null, reference_comparison: { status: 'unavailable' },
  })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  expect(await screen.findByRole('alert')).toHaveTextContent('样本读取失败')
  fireEvent.click(screen.getByRole('button', { name: '重新计算' }))
  await screen.findByText('暂无法判断')
  expect(screen.queryByText('历史测算不可达')).not.toBeInTheDocument()
  expect(screen.queryByTestId('scope-chart')).not.toBeInTheDocument()
})

it('keeps an unavailable reference separate from reachability and labels a reference-only chart', async () => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, reference_comparison: { status: 'incompatible' } })
  const view = render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText('历史测算可达')
  expect(screen.getByText(/原始参考的币种、收益风险口径或研究日/)).toBeInTheDocument()
  expect(screen.getByTestId('scope-chart')).not.toHaveTextContent('referencePoints')
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined', frontier: null, sample: null, target_check: null })
  view.rerender(<ScopeFeasibility input={{ ...input, as_of: '2020-01-01' }} clock="2020-01-01" />)
  await screen.findByText('暂无法判断')
  expect(screen.getByText(/当前范围尚无可用前沿/)).toBeInTheDocument()
  expect(screen.getByTestId('scope-chart')).toHaveTextContent('"points":[]')
  expect(screen.queryByText('历史测算可达')).not.toBeInTheDocument()
})

it('reports the periods dropped when proxy trading calendars differ', async () => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result,
    sample: { ...result.sample!, observations: 1113, common_days: 1181, excluded_return_periods: 67, missing_trading_days: 39 } })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  expect(await screen.findByText(/1113 个日收益观测/)).toBeInTheDocument()
  expect(screen.getByText(/39 个交易日不在共同样本内，另排除 67 个跨日收益期/)).toBeInTheDocument()
})

it.each([
  ['SCOPE_FUNDING_CHECK_REQUIRED', '资金路径成功率待验证'],
  ['SCOPE_PRODUCT_LIQUIDITY_REQUIRED', '产品流动性待验证'],
])('separates a passed historical check from the pending %s gate', async (code, label) => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined', reason_code: code,
    mandate: { ...result.mandate, target_return: null, funding_requirement: { required_return: .0772, status: 'solved', basis: 'annual_effective_gross_of_model_fee' } }, reasons: [{ code, message: '仍需后续验证。' }] })
  const view = render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  expect(await screen.findByText('历史风险与权重约束初筛通过')).toBeInTheDocument()
  if (code === 'SCOPE_PRODUCT_LIQUIDITY_REQUIRED') expect(screen.getByText(label)).toBeInTheDocument()
  else expect(screen.queryByText(label)).not.toBeInTheDocument()
  expect(screen.queryByText('暂无法判断')).not.toBeInTheDocument()
  expect(screen.queryByText('历史测算可达')).not.toBeInTheDocument()
  const chart = JSON.parse(screen.getByTestId('scope-chart').textContent!)
  expect(chart.points).toEqual(result.frontier!.points)
  expect(chart.referencePoints).toEqual(result.reference_comparison!.status === 'available' ? result.reference_comparison!.points : [])
  expect(chart.constrainedPoints).toEqual(result.reference_comparison!.status === 'available' ? result.reference_comparison!.constrained_points : [])
  expect(chart.targetReturn).toBeNull()
  expect(screen.getByText('资金计划所需年化收益率')).toBeInTheDocument()
  expect(screen.getByText('7.72%')).toBeInTheDocument()
  expect(screen.getByText(/沿用 01 资金计算口径/)).toBeInTheDocument()
  if (code === 'SCOPE_FUNDING_CHECK_REQUIRED') expect(screen.getByText(/下一步在 LTCMA／SAA 中/)).toBeInTheDocument()
  else expect(screen.queryByText(/下一步在 LTCMA／SAA 中/)).not.toBeInTheDocument()
  view.rerender(<ScopeFeasibility input={{ ...input, mandate_id: 'goal-two' }} clock="2019-12-31" />)
  expect(screen.queryByText('历史风险与权重约束初筛通过')).not.toBeInTheDocument()
  expect(screen.queryByText(label)).not.toBeInTheDocument()
  expect(screen.queryByTestId('scope-chart')).not.toBeInTheDocument()
})

it.each([
  ['solved', 0, '0.00%'],
  ['solved', -1e-16, '0.00%'],
  ['above_search_bound', null, '所需收益超出计算范围'],
  ['at_lower_bound', -.99, '搜索下界已满足，未求得精确最低值'],
] as const)('preserves the funding return solver status %s instead of inventing a target', async (status, required_return, text) => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined', reason_code: 'SCOPE_FUNDING_CHECK_REQUIRED',
    mandate: { ...result.mandate, target_return: null, funding_requirement: { status, required_return, basis: 'annual_effective_gross_of_model_fee' } } })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText(text)
  expect(JSON.parse(screen.getByTestId('scope-chart').textContent!).targetReturn).toBeNull()
  expect(screen.queryByText('-99.00%')).not.toBeInTheDocument()
})

it('does not declare a passed historical check merely because a frontier was computed', async () => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined',
    reason_code: 'SCOPE_SOLVER_UNRESOLVED', target_check: { ...result.target_check!, status: 'undetermined', candidate: null },
    reasons: [{ code: 'SCOPE_SOLVER_UNRESOLVED', message: '数值求解尚未完成验证。' }], additional_checks: { funding: true, benchmark: false } })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText('暂无法判断')
  expect(screen.getByTestId('scope-chart')).toBeInTheDocument()
  expect(screen.queryByText('历史风险与权重约束初筛通过')).not.toBeInTheDocument()
  expect(screen.queryByText('资金路径成功率待验证')).not.toBeInTheDocument()
})

it('retains the arithmetic return target when an absolute objective also requires funding protection', async () => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined', reason_code: 'SCOPE_FUNDING_CHECK_REQUIRED',
    mandate: { ...result.mandate, funding_requirement: { required_return: .0772, status: 'solved', basis: 'annual_effective_gross_of_model_fee' } } })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText('历史收益与风险初筛通过')
  expect(screen.queryByText('资金路径成功率待验证')).not.toBeInTheDocument()
  expect(screen.getByText('8.00%')).toBeInTheDocument()
  expect(screen.queryByText('7.72%')).not.toBeInTheDocument()
  expect(JSON.parse(screen.getByTestId('scope-chart').textContent!).targetReturn).toBe(.08)
})

it('explains failed goals and shows an unconstrained scope curve only with its limitation', async () => {
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'infeasible', frontier: { ...result.frontier!, constraints_applied: false }, reasons: [{ code: 'CASH', message: '缺少目标要求的现金资产。' }] })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText('历史测算不可达')
  expect(screen.getByText(/仍可保存当前范围/)).toBeInTheDocument()
  expect(screen.getByText(/尚未应用完整目标约束/)).toBeInTheDocument()
})


it.each(['passed', 'no_candidate'] as const)('shows the funding hurdle and the actual compound screening verdict: %s', async status => {
  const comparison = { basis: 'annual_compound_median_gross_of_model_fee' as const, status,
    target_return: .0409, points: [{ ...result.frontier!.points[0], expected_return: .065 }],
    reference_points: [{ volatility: .1, expected_return: .08, status: 'optimal_to_tolerance' }],
    constrained_points: [{ volatility: .09, expected_return: .07, status: 'optimal_to_tolerance' }],
    candidate: { volatility: .06, expected_return: .065, weights: { a: 1 } }, probability_validated: false as const }
  vi.mocked(getScopeFeasibility).mockResolvedValueOnce({ ...result, status: 'undetermined',
    reason_code: 'SCOPE_FUNDING_CHECK_REQUIRED', reasons: [{ code: 'SCOPE_FUNDING_CHECK_REQUIRED', message: '旧的待验证说明' }],
    mandate: { ...result.mandate, target_return: null, funding_requirement: { required_return: .0409, status: 'solved', basis: 'annual_effective_gross_of_model_fee' } },
    funding_comparison: comparison })
  render(<ScopeFeasibility input={input} clock="2019-12-31" />)
  await screen.findByText(status === 'passed' ? '收益门槛与风险约束初筛通过' : '尚未找到达到收益门槛的组合')
  expect(screen.queryByText('资金路径成功率待验证')).not.toBeInTheDocument()
  expect(screen.queryByText('旧的待验证说明')).not.toBeInTheDocument()
  expect(screen.getByText('4.09%')).toBeInTheDocument()
  const chart = JSON.parse(screen.getByTestId('scope-chart').textContent!)
  expect(chart).toMatchObject({ compound: true, targetReturn: .0409, points: comparison.points,
    referencePoints: comparison.reference_points, constrainedPoints: comparison.constrained_points, candidate: comparison.candidate })
  expect(screen.getByText(/下一步在 LTCMA／SAA 中/)).toBeInTheDocument()
})
