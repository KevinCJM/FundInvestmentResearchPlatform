import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { FrontierGridControls, FrontierGridResults, defaultFrontierGrid, frontierGridIssue, type FrontierGridResult } from './FrontierGrid'

export const gridFixture: FrontierGridResult = {
  requested_points: 2, attempted_points: 2, successful_points: 1, failed_points: 1,
  unattempted_points: 0, duplicate_targets: 0, duplicate_solutions: 0, added_candidates: 1,
  max_iterations: 300, risk_solver: 'active_set_qp', optimality_scope: 'convex_quadratic_kkt',
  endpoints: [{ kind: 'minimum_risk', status: 'converged', iterations: 2 }, { kind: 'maximum_return', status: 'converged', iterations: 3 }],
  points: [
    { target_index: 0, target: .05, value: [.08, .05], weights: [.6, .4], status: 'converged', iterations: 4, optimality_residual: 0, constraint_violation: 0, duplicate_of: null, candidate_index: 100, on_frontier: true },
    { target_index: 1, target: .15, value: [null, null], weights: [null, null], status: 'infeasible_target', iterations: 0, optimality_residual: null, constraint_violation: null, duplicate_of: null, candidate_index: null, on_frontier: false },
  ], curve: [],
}

describe('整条前沿目标网格', () => {
  it('将目标点数与单点迭代预算分别输入', () => {
    const onChange = vi.fn()
    render(<FrontierGridControls value={{ ...defaultFrontierGrid, enabled: true }} quantized={false} busy={false} onChange={onChange} />)
    expect(screen.getByLabelText('前沿目标点数')).toHaveValue('20')
    expect(screen.getByLabelText('单点最大迭代次数')).toHaveValue('300')
    fireEvent.change(screen.getByLabelText('前沿目标点数'), { target: { value: '200' } })
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ point_count: 200, max_iterations: 300 }))
  })

  it('禁止静默把散点取整规则当成连续前沿的约束', async () => {
    const onChange = vi.fn()
    render(<FrontierGridControls value={{ ...defaultFrontierGrid, enabled: true }} quantized busy={false} onChange={onChange} />)
    expect(screen.getByRole('status')).toHaveTextContent('不是同一离散可行域')
    await userEvent.click(screen.getByRole('checkbox', { name: /我确认网格采用连续权重/ }))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ accept_continuous_weights: true }))
  })

  it('空值、小数与越界输入不能触发求解', () => {
    for (const point_count of [NaN, 0, 20.5, 201]) {
      expect(frontierGridIssue({ ...defaultFrontierGrid, enabled: true, point_count }, false)).not.toBe('')
    }
    expect(frontierGridIssue({ ...defaultFrontierGrid, enabled: true, max_iterations: NaN }, false)).not.toBe('')
    expect(frontierGridIssue(defaultFrontierGrid, true)).toBe('')
  })

  it('保留失败目标和真实权重，失败点不能采用', async () => {
    const onAdopt = vi.fn()
    render(<FrontierGridResults result={gridFixture} assetNames={['股', '债']} riskLabel="年化风险" returnLabel="年化收益" onAdopt={onAdopt} />)
    expect(screen.getByRole('status')).toHaveTextContent('目标 2 个 · 成功 1 个 · 失败 1 个')
    await userEvent.click(screen.getByText('逐目标状态、权重与采用（2 项）'))
    expect(screen.getByRole('table', { name: '前沿目标求解明细' })).toHaveTextContent('目标不可行')
    expect(screen.getByRole('button', { name: '采用前沿目标 2' })).toBeDisabled()
    await userEvent.click(screen.getByRole('button', { name: '采用前沿目标 1' }))
    expect(onAdopt).toHaveBeenCalledWith(gridFixture.points[0])
  })

  it('端点未完成时提示复核预算，不显示成功假象', () => {
    render(<FrontierGridResults result={{ ...gridFixture, successful_points: 0, failed_points: 0, unattempted_points: 2, endpoints: [{ kind: 'minimum_risk', status: 'max_iterations', iterations: 1 }] }} assetNames={['股', '债']} riskLabel="风险" returnLabel="收益" onAdopt={() => {}} />)
    expect(screen.getByText(/前沿端点未完成/)).toHaveTextContent('增加单点迭代预算')
  })
})
