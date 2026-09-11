import { fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import AutoPlanSummary from './AutoPlanSummary'
import { emptyDefinition, type AutoIncrementalPlan } from '../../services/etl'

afterEach(() => { vi.restoreAllMocks() })
const plan: AutoIncrementalPlan = { plan_id: 'plan', snapshot: 'active', cutoff_date: '2026-09-09', lookback_trade_days: 5,
  ready: false, errors: [{ code: 'EMPTY', message: '空表没有日期基线' }],
  steps: [{ id: 'bad', name: '行业行情', strategy: 'blocked', latest_date: null, start_date: '', end_date: '', message: '未安排下载' }],
  exclusion_proposal: { definition: emptyDefinition(), rebuild_dependencies: true, excluded: [
    { id: 'bad', name: '行业行情', reason: 'blocked' }, { id: 'child', name: '覆盖快照', reason: 'dependency' },
  ] } }

it('拦截步骤不伪装成刷新，排除和重排需要明确确认', () => {
  const change = vi.fn(), confirm = vi.spyOn(window, 'confirm').mockReturnValue(false)
  render(<AutoPlanSummary plan={plan} onExclude={change} />)
  expect(screen.getByText('不可自动增量')).toBeVisible()
  expect(screen.getByText(/覆盖快照（真实依赖受影响）/)).toBeVisible()
  fireEvent.click(screen.getByRole('button', { name: '仅保留可增量步骤' }))
  expect(change).not.toHaveBeenCalled()
  confirm.mockReturnValue(true)
  fireEvent.click(screen.getByRole('button', { name: '仅保留可增量步骤' }))
  expect(change).toHaveBeenCalledWith(plan.exclusion_proposal?.definition)
})

it('区分最新公告与查询覆盖，不把重复接收显示为新增', () => {
  render(<AutoPlanSummary plan={{ ...plan, ready: true, errors: [], exclusion_proposal: null,
    steps: [{ id: 'holdings', name: '基金持仓', strategy: 'incremental', latest_date: '2026-08-31',
      start_date: '20260909', end_date: '20260909', message: '已经覆盖', query_dates: [],
      new_query_days: 0, revision_query_days: 0, reused_query_days: 10, coverage_through: '2026-09-09',
      last_checked_at: '2026-09-10T01:00:00Z', request_estimate: { minimum: 0, page_ceiling: 0, note: '不含重试' } }] }} />)
  fireEvent.click(screen.getByText('数据集计划 · 1 个步骤'))
  expect(screen.getByText(/已有最新：2026-08-31/)).toBeVisible()
  expect(screen.getByText('2026-09-09')).toBeVisible()
  expect(screen.getByText(/区间已覆盖，无需重复查询/)).toBeVisible()
})
