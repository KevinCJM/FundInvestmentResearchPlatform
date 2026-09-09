import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import EtlRunHistory from './EtlRunHistory'
import type { EtlRun } from '../../services/etl'

const now = Date.parse('2026-09-07T06:00:00Z')
const run = (): EtlRun => ({ run_id: 'run', name: '全数据同步', status: 'RUNNING', created_at: new Date(now).toISOString(), updated_at: '', attempt: 1, published: false,
  steps: [{ id: 'nav', name: '公募基金持仓', kind: 'task', status: 'RUNNING', started_at: new Date(now - 120000).toISOString(), heartbeat_at: new Date(now).toISOString(),
    progress: { phase: '下载分片', message: '正在下载公告日期', completed: 120, total: 800, batches: 140, received_rows: 5000, activity_at: new Date(now).toISOString(), logs: [{ at: new Date(now).toISOString(), message: '日期进度 120/800' }] } }] })
const props = { busy: false, onResume: vi.fn(), onCancel: vi.fn(), onReuse: vi.fn() }
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })

it('独立执行器说明 API 重启不停止下载，同时保留取消操作', () => {
  const value = run()
  value.execution = { mode: 'independent', state: 'CONNECTED', message: '已连接独立任务执行器；API 服务重启不会停止下载或后续步骤。' }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/API 服务重启不会停止下载/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: /取消/ })).toBeEnabled()
  expect(screen.getByRole('progressbar')).toBeInTheDocument()
})

it('旧工作进程有新进度时展示活动信息，但不宣称调度恢复', () => {
  vi.useFakeTimers(); vi.setSystemTime(now)
  const value = run()
  value.status = 'INTERRUPTED'
  value.steps[0].status = 'INTERRUPTED'
  value.steps[0].worker_only = true
  value.execution = { mode: 'legacy', state: 'WORKER_OBSERVED', message: '仅恢复进度展示，不能自动推进后续步骤。' }
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getAllByText(/后台有下载进度（调度中断）/)).toHaveLength(2)
  expect(screen.getByText(/不能自动推进后续步骤/)).toBeInTheDocument()
  expect(screen.getByRole('progressbar')).toBeInTheDocument()
  expect(screen.getByText(/工作进程进度更新/)).toBeInTheDocument()
})

it('恢复运行明确区分旧结果复用和当前下载', () => {
  const value = run()
  value.recovered_from = 'original-run'
  value.message = '等待继续执行。'
  value.steps.unshift({ id: 'info', name: '基金信息', kind: 'task', status: 'SUCCEEDED', imported_from: { run_id: 'original-run', step_id: 'info', execution_fingerprint: 'old-code' } })
  render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByText(/原任务 original-run 保留审计记录/)).toBeInTheDocument()
  expect(screen.getByText('已核验复用原任务结果，未重复下载')).toBeInTheDocument()
  expect(screen.getAllByRole('progressbar')).toHaveLength(1)
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
  expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '15')
  expect(screen.getByText(/累计接收 5,000 行/)).toBeInTheDocument()
  expect(screen.getByText('已耗时 2 分 0 秒')).toBeInTheDocument()
  act(() => vi.advanceTimersByTime(1000))
  expect(screen.getByText('已耗时 2 分 1 秒')).toBeInTheDocument()
  fireEvent.click(screen.getByText('最近日志（1 条，已脱敏）'))
  expect(screen.getByRole('list', { name: '最近执行日志' })).toHaveTextContent('120/800')
})

it('未知总量、非法计数和旧执行器不伪造百分比', () => {
  const value = run(); delete value.steps[0].progress
  const view = render(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  expect(screen.getByText(/当前执行器尚未上报/)).toBeInTheDocument()
  value.steps[0].progress = { phase: '下载', message: '下载中', completed: 11, total: 10 }
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
})

it('合并阶段清除下载百分比，连接中断保留旧结果并明确提示', () => {
  const value = run(), view = render(<EtlRunHistory {...props} runs={[value]} />)
  value.steps[0].progress = { phase: '合并与校验', message: '本地合并', completed: null, total: null }
  view.rerender(<EtlRunHistory {...props} connected={false} runs={[value]} />)
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  expect(screen.getByRole('alert')).toHaveTextContent('以下为上次收到的进度')
})

it('无进度提示不误判卡死，失败或取消后保留日志而不显示活动进度条', () => {
  vi.useFakeTimers(); vi.setSystemTime(now)
  const value = run(), view = render(<EtlRunHistory {...props} runs={[value]} />)
  act(() => vi.advanceTimersByTime(61000))
  expect(screen.getByText(/不能据此判定卡死/)).toBeInTheDocument()
  value.status = 'CANCELLED'; value.steps[0].status = 'CANCELLED'; value.steps[0].finished_at = new Date(now).toISOString()
  view.rerender(<EtlRunHistory {...props} runs={[value]} />)
  expect(screen.queryByRole('progressbar')).not.toBeInTheDocument()
  expect(screen.getByText('最后执行进度')).toBeInTheDocument()
  expect(screen.getByText('已耗时 2 分 0 秒')).toBeInTheDocument()
})
