import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import EtlRunHistory from './EtlRunHistory'
import type { EtlRun } from '../../services/etl'

const now = Date.parse('2026-09-07T06:00:00Z')
const run = (): EtlRun => ({ run_id: 'run', name: '全数据同步', status: 'RUNNING', created_at: new Date(now).toISOString(), updated_at: '', attempt: 1, published: false,
  steps: [{ id: 'nav', name: '公募基金持仓', kind: 'task', status: 'RUNNING', started_at: new Date(now - 120000).toISOString(), heartbeat_at: new Date(now).toISOString(),
    progress: { phase: '下载分片', message: '正在下载公告日期', completed: 120, total: 800, batches: 140, received_rows: 5000, activity_at: new Date(now).toISOString(), logs: [{ at: new Date(now).toISOString(), message: '日期进度 120/800' }] } }] })
const props = { busy: false, onResume: vi.fn(), onCancel: vi.fn(), onReuse: vi.fn() }
const currentStep = () => within(screen.getByRole('list', { name: '当前工作步骤' }))
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })

it('采集完成但数值冲突时在任务摘要和步骤中清楚显示隔离状态', () => {
  const value = run()
  value.status = 'SUCCEEDED'
  value.steps[0].status = 'SUCCEEDED'
  value.steps[0].output = { data_quality: { status: 'CONFLICTED', conflicting_keys: 2, publishable: false } }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/下载完成不代表数据质量通过或已发布/)).toBeVisible()
  fireEvent.click(screen.getByText('查看全部步骤（共 1 个，已完成 1 个）'))
  expect(screen.getByText('采集完成 · 数据待核验')).toBeVisible()
  expect(screen.getByText(/已隔离 2 个持仓冲突/)).toBeVisible()
})

it('分页一致性复核显示独立的页进度，不冒充新增披露或节点完成', () => {
  const value = run()
  value.steps[0].progress = { ...value.steps[0].progress!, phase: '分页一致性复核',
    message: '公告日 20260829 页面复核 3/8。', completed: 3, total: 8, unit: '页' }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByText('分页一致性复核 · 37%')).toBeVisible()
  expect(currentStep().getByText('本阶段已处理 3 / 8 页')).toBeVisible()
  expect(currentStep().getByText(/百分比仅代表当前阶段，不代表整个节点/)).toBeVisible()
  expect(currentStep().getByText(/接收数量包含历史重叠、分页和复核/)).toBeVisible()
})

it('失败隔离后顶部优先展示仍运行的独立节点而非之前失败节点', () => {
  const value = run()
  value.steps = [{ id:'failed', name:'指数权重', kind:'task', status:'FAILED', error:'分页失败' },
    { id:'blocked', name:'依赖权重', kind:'task', status:'SKIPPED', error:'必需依赖失败' }, ...value.steps]
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByText('3. 公募基金持仓')).toBeVisible()
  expect(currentStep().queryByText('1. 指数权重')).not.toBeInTheDocument()
  expect(screen.getByRole('status')).toHaveTextContent('已有 1 个节点失败，其他无数据依赖的步骤仍在继续')
  fireEvent.click(screen.getByText('查看全部步骤（共 3 个，已完成 0 个）'))
  expect(within(screen.getByRole('list', { name:'全部工作步骤' })).getByText('分页失败')).toBeVisible()
})

it('全部步骤保留第25步及原顺序，两处进度同步并随执行切换', () => {
  const value = run(), active = value.steps[0]
  value.steps = Array.from({ length: 31 }, (_, n) => n === 24
    ? { ...active, id: 'members', name: '指数成分与权重' }
    : { id: `step-${n + 1}`, name: `步骤${n + 1}`, kind: 'task', status: n < 24 ? 'SUCCEEDED' : 'PENDING' })
  const view = render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.queryByText(/查看其余步骤/)).not.toBeInTheDocument()
  const summary = screen.getByText('查看全部步骤（共 31 个，已完成 24 个）')
  expect(summary.closest('details')).not.toHaveAttribute('open')
  expect(currentStep().getByText('25. 指数成分与权重')).toBeVisible()
  fireEvent.click(summary)
  const all = screen.getByRole('list', { name: '全部工作步骤' })
  expect(all.children).toHaveLength(31)
  expect(all.children[23]).toHaveTextContent('24. 步骤24')
  expect(all.children[24]).toHaveTextContent('25. 指数成分与权重')
  expect(all.children[25]).toHaveTextContent('26. 步骤26')
  expect(screen.getAllByText('25. 指数成分与权重')).toHaveLength(2)
  expect(within(all).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '15')
  value.steps[24].progress = { ...active.progress!, completed: 240 }
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByRole('progressbar')).toHaveAttribute('aria-valuenow', '30')
  expect(within(all).getByRole('progressbar')).toHaveAttribute('aria-valuenow', '30')
  value.steps[24].status = 'SUCCEEDED'
  value.steps[25].status = 'RUNNING'
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByText('26. 步骤26')).toBeVisible()
  expect(within(all.children[24] as HTMLElement).getByText('已完成')).toBeVisible()
  expect(all.children[25].firstElementChild).toHaveTextContent('正在执行')
  expect(all.children).toHaveLength(31)
})

it('只有一个执行步骤时也提供全部步骤入口，空任务不显示入口', () => {
  const value = run(), view = render(<EtlRunHistory {...props} runs={[value]} />)
  fireEvent.click(screen.getByText('查看全部步骤（共 1 个，已完成 0 个）'))
  expect(screen.getByRole('list', { name: '全部工作步骤' }).children).toHaveLength(1)
  expect(screen.getAllByRole('progressbar')).toHaveLength(2)
  value.steps = []
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.queryByText(/查看全部步骤/)).not.toBeInTheDocument()
})

it('恢复链只有当前卡片，历史默认折叠，取消只指向当前运行', async () => {
  const old = { ...run(), run_id: 'old', status: 'FAILED' as const }
  const current = { ...run(), run_id: 'current', recovered_from: 'old', name: '全数据同步（恢复）（恢复）',
    history: { root_run_id: 'old', display_name: '全数据同步', resume_count: 2,
      records: [{ run_id: 'old', status: 'FAILED', attempt: 1, failed_step: '成分与权重', error: '旧数据库锁超时' }] } }
  render(<EtlRunHistory {...props} runs={[current, old]} focusedRun="old" />)
  expect(screen.getAllByTestId('etl-task-card')).toHaveLength(1)
  expect(screen.getByTestId('etl-task-card')).toHaveAttribute('open')
  expect(screen.queryByText(/（恢复）（恢复）/)).not.toBeInTheDocument()
  const history = screen.getByText(/历史记录 · 已续跑/).closest('details')!
  expect(history).not.toHaveAttribute('open')
  fireEvent.click(within(history).getByText(/历史记录/))
  expect(within(history).getByText('旧数据库锁超时')).toBeInTheDocument()
  expect(within(history).queryByRole('button')).not.toBeInTheDocument()
  await act(async () => fireEvent.click(screen.getByRole('button', { name: '取消运行' })))
  expect(props.onCancel).toHaveBeenLastCalledWith('current')
})

it('独立同名任务和仍在运行的祖先不被隐藏', () => {
  const ancestor = { ...run(), run_id: 'ancestor' }
  const current = { ...run(), run_id: 'current', history: { root_run_id: 'ancestor', display_name: '全数据同步', resume_count: 1,
    records: [{ run_id: 'ancestor', status: 'RUNNING', attempt: 1 }] } }
  render(<EtlRunHistory {...props} runs={[current, ancestor, { ...run(), run_id: 'independent' }]} />)
  expect(screen.getAllByTestId('etl-task-card')).toHaveLength(3)
})

it('恢复中显示当前校验时间，不展示旧等待消息为当前状态', () => {
  const value = run(); value.status = 'FAILED'; value.steps[0].status = 'FAILED'
  value.message = '旧任务等待继续执行。'; value.error = '旧失败'
  value.recovery = { can_resume:false, artifact_check_pending:false, blockers:[], job:{ id:'job',source_run_id:'run',target_run_id:'next',status:'RUNNING',phase:'校验与迁移',message:'已核验250个查询',created_at:'',updated_at:'2026-09-09T09:55:00Z',logs:[] } }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/已核验250个查询/)).toBeInTheDocument()
  expect(screen.queryByText('旧任务等待继续执行。')).not.toBeInTheDocument()
  expect(screen.queryByText('旧失败')).not.toBeInTheDocument()
  expect(screen.getByText(/已完成 0 \/ 1 个步骤/)).toHaveTextContent(new Date('2026-09-09T09:55:00Z').toLocaleString('zh-CN'))
})

it('独立执行器说明 API 重启不停止下载，同时保留取消操作', () => {
  const value = run()
  value.execution = { mode: 'independent', state: 'CONNECTED', message: '已连接独立任务执行器；API 服务重启不会停止下载或后续步骤。' }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/API 服务重启不会停止下载/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /取消/ })).toBeEnabled()
  expect(currentStep().getByRole('progressbar')).toBeInTheDocument()
})

it('旧工作进程有新进度时展示活动信息，但不宣称调度恢复', () => {
  vi.useFakeTimers(); vi.setSystemTime(now)
  const value = run()
  value.status = 'INTERRUPTED'
  value.steps[0].status = 'INTERRUPTED'
  value.steps[0].worker_only = true
  value.execution = { mode: 'legacy', state: 'WORKER_OBSERVED', message: '仅恢复进度展示，不能自动推进后续步骤。' }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByText(/后台有下载进度（调度中断）/)).toBeVisible()
  expect(screen.getByText(/不能自动推进后续步骤/)).toBeInTheDocument()
  expect(currentStep().getByRole('progressbar')).toBeInTheDocument()
  expect(currentStep().getByText(/工作进程进度更新/)).toBeInTheDocument()
})

it('恢复运行明确区分旧结果复用和当前下载', () => {
  const value = run()
  value.recovered_from = 'original-run'
  value.message = '等待继续执行。'
  value.steps.unshift({ id: 'info', name: '基金信息', kind: 'task', status: 'SUCCEEDED', imported_from: { run_id: 'original-run', step_id: 'info', execution_fingerprint: 'old-code' } })
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/原任务 original-run 保留审计记录/)).toBeInTheDocument()
  expect(screen.getByText('已核验复用原任务结果，未重复下载')).toBeInTheDocument()
  expect(currentStep().getAllByRole('progressbar')).toHaveLength(1)
  expect(screen.queryByText('等待继续执行。')).not.toBeInTheDocument()
})

it('跨日风险在完成或刷新后仍可见，展示时点来源而不宣称 PIT 通过', () => {
  const value = run()
  value.status = 'SUCCEEDED'; value.steps = []
  value.collection_timing = { timezone: 'Asia/Shanghai', first_date: '2026-09-07', last_date: '2026-09-08', cross_date: true,
    warnings: [{ code: 'ETL_CROSS_DATE_COLLECTION', message: '采集记录已跨日期，下载内容的时点可能不一致。' }],
    boundary: '同日采集也不代表已通过 PIT 校验。', scope: '不覆盖增量基线的全部历史批次。',
    windows: [{ run_id: 'old', step_id: 'nav', name: '净值', attempt: 1, first_at: '2026-09-07T15:59:00Z', last_at: '2026-09-07T16:01:00Z', basis: 'execution_window' }] }
  const view = render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByRole('alert')).toHaveTextContent('采集记录已跨日期')
  fireEvent.click(screen.getByText('查看采集批次时间与续跑记录'))
  expect(screen.getByRole('table')).toHaveTextContent('执行时间范围（估计）')
  expect(screen.getByRole('table')).toHaveTextContent('old')
  expect(screen.getByRole('table')).toHaveTextContent('23:59:00')
  view.unmount()
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByRole('alert')).toHaveTextContent('时点可能不一致')
})

it('展示真实阶段百分比、累计接收、耗时和可展开日志', () => {
  vi.useFakeTimers(); vi.setSystemTime(now)
  render(<EtlRunHistory {...props} runs={[run()]} />)
  expect(currentStep().getByRole('progressbar')).toHaveAttribute('aria-valuenow', '15')
  expect(currentStep().getByText(/累计接收 5,000 行/)).toBeInTheDocument()
  expect(currentStep().getByText('已耗时 2 分 0 秒')).toBeInTheDocument()
  act(() => vi.advanceTimersByTime(1000))
  expect(currentStep().getByText('已耗时 2 分 1 秒')).toBeInTheDocument()
  fireEvent.click(currentStep().getByText('最近日志（1 条，已脱敏）'))
  expect(currentStep().getByRole('list', { name: '最近执行日志' })).toHaveTextContent('120/800')
})

it('未知总量、非法计数和旧执行器不伪造百分比', () => {
  const value = run(); delete value.steps[0].progress
  const view = render(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  expect(currentStep().getByText(/当前执行器尚未上报/)).toBeInTheDocument()
  value.steps[0].progress = { phase: '下载', message: '下载中', completed: 11, total: 10 }
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(currentStep().getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
})

it('合并阶段清除下载百分比，连接中断保留旧结果并明确提示', () => {
  const value = run(), view = render(<EtlRunHistory {...props} runs={[value]} />)
  value.steps[0].progress = { phase: '合并与校验', message: '本地合并', completed: null, total: null }
  view.rerender(<EtlRunHistory {...props} connected={false} runs={[value]} />)
  expect(currentStep().getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  expect(screen.getByRole('alert')).toHaveTextContent('以下为上次收到的进度')
})

it('无进度提示不误判卡死，失败或取消后保留日志而不显示活动进度条', () => {
  vi.useFakeTimers(); vi.setSystemTime(now)
  const value = run(), view = render(<EtlRunHistory {...props} runs={[value]} />)
  act(() => vi.advanceTimersByTime(61000))
  expect(currentStep().getByText(/不能据此判定卡死/)).toBeInTheDocument()
  value.status = 'CANCELLED'; value.steps[0].status = 'CANCELLED'; value.steps[0].finished_at = new Date(now).toISOString()
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
  expect(currentStep().getByText('最后执行进度')).toBeInTheDocument()
  expect(currentStep().getByText('已耗时 2 分 0 秒')).toBeInTheDocument()
})
