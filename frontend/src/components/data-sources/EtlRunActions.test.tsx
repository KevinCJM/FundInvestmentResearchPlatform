import { act, fireEvent, render, screen } from '@testing-library/react'
import { expect, it, vi } from 'vitest'
import EtlRunActions from './EtlRunActions'
import type { EtlRun } from '../../services/etl'

const run: EtlRun = { run_id:'r', name:'持仓下载', status:'INTERRUPTED', created_at:'', updated_at:'', attempt:1, published:false, steps:[] }
const props = { run, busy:false, onCancel:vi.fn(), onReuse:vi.fn() }

it('页内确认、加载反馈、重复点击抑制和成功提示', async () => {
  let resolve!: () => void
  const resume = vi.fn(() => new Promise<void>(r => { resolve=r }))
  render(<EtlRunActions {...props} onResume={resume} />)
  fireEvent.click(screen.getByText('恢复下载'))
  expect(resume).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('确认继续'))
  expect(screen.getByRole('status')).toHaveTextContent('正在核验执行版本')
  expect(screen.getByText('正在恢复…')).toBeDisabled()
  fireEvent.click(screen.getByText('正在恢复…'))
  expect(resume).toHaveBeenCalledTimes(1)
  await act(async () => resolve())
  expect(screen.getByRole('status')).toHaveTextContent('恢复请求已接受')
})

it('后端拒绝或网络失败在原按钮处明确展示，允许检查后再试', async () => {
  render(<EtlRunActions {...props} onResume={vi.fn().mockRejectedValue(new Error('执行程序已更新，旧任务不能原地续跑。'))} />)
  fireEvent.click(screen.getByText('恢复下载')); fireEvent.click(screen.getByText('确认继续'))
  expect(await screen.findByRole('alert')).toHaveTextContent('不能原地续跑')
  expect(screen.getByText('恢复下载')).toBeEnabled()
})

it('已有持锁或版本提示不预先禁用恢复，点击确认后交给后端核验', async () => {
  const resume=vi.fn()
  render(<EtlRunActions {...props} run={{...run,recovery:{can_resume:false,artifact_check_pending:false,blockers:[{code:'LOCK',message:'旧进程仍持锁'},{code:'CODE',message:'执行版本变化'}]}}} onResume={resume} />)
  expect(screen.getByText('恢复下载')).toBeEnabled()
  expect(screen.getByText('旧进程仍持锁')).toBeInTheDocument()
  expect(screen.getByText('执行版本变化')).toBeInTheDocument()
  expect(resume).not.toHaveBeenCalled()
  fireEvent.click(screen.getByText('恢复下载'))
  await act(async () => fireEvent.click(screen.getByText('确认继续')))
  expect(resume).toHaveBeenCalledWith('r')
})

it('跨日和版本提示不禁用按钮，保留页内确认', async () => {
  const resume = vi.fn()
  const recovery = { can_resume: true, artifact_check_pending: true, blockers: [], warnings: [{ code: 'ETL_CROSS_DATE_RESUME', message: '跨日期续跑提醒：PIT 可能不一致；允许继续。' }] }
  const view = render(<EtlRunActions {...props} run={{ ...run, recovery }} onResume={resume} />)
  expect(screen.getByRole('alert')).toHaveTextContent('PIT 可能不一致')
  expect(screen.getByText('恢复下载')).toBeEnabled()
  fireEvent.click(screen.getByText('恢复下载'))
  expect(screen.getByRole('group', { name: '确认恢复任务' })).toHaveTextContent('不会改变原下载区间')
  expect(resume).not.toHaveBeenCalled()
  await act(async () => fireEvent.click(screen.getByText('确认继续')))
  expect(resume).toHaveBeenCalledTimes(1)
  expect(resume).toHaveBeenCalledWith('r')
  view.rerender(<EtlRunActions {...props} run={{ ...run, recovery: { ...recovery, can_resume: false, blockers: [{ code: 'CODE', message: '执行版本变化' }] } }} onResume={resume} />)
  expect(screen.getByText('恢复下载')).toBeEnabled()
  expect(screen.getByText('执行版本变化')).toBeInTheDocument()
})

it('刷新后展示持久化校验进度，禁止重复提交，失败后恢复可操作', () => {
  const job = { id:'j', source_run_id:'r', target_run_id:'new', status:'RUNNING' as const, phase:'校验与迁移', message:'正在核验大文件', created_at:'',updated_at:'',logs:[] }
  const recovery = {can_resume:false,artifact_check_pending:false,blockers:[],job}
  const view = render(<EtlRunActions {...props} run={{...run,recovery}} onResume={vi.fn()} />)
  expect(screen.getByRole('progressbar')).not.toHaveAttribute('aria-valuenow')
  expect(screen.getByRole('status')).toHaveTextContent('正在核验大文件')
  expect(screen.getByText('正在恢复…')).toBeDisabled()
  view.rerender(<EtlRunActions {...props} run={{...run,recovery:{...recovery,job:{...job,status:'FAILED',message:'文件校验和变化'}}}} onResume={vi.fn()} />)
  expect(screen.getByRole('alert')).toHaveTextContent('恢复下载失败')
  expect(screen.getByRole('alert')).toHaveTextContent('文件校验和变化')
  expect(screen.getByText('恢复下载')).toBeEnabled()
})

it('已迁移的旧记录引导后续进度，不再把旧失败表现为当前阻断', () => {
  const show = vi.fn()
  render(<EtlRunActions {...props} onResume={vi.fn()} onShowRun={show} run={{...run,recovery:{can_resume:false,artifact_check_pending:false,blockers:[{code:'CODE',message:'过期阻断'}],successor:{run_id:'new',status:'RUNNING',name:'恢复'}}}} />)
  expect(screen.queryByText('恢复下载')).not.toBeInTheDocument()
  expect(screen.queryByText('过期阻断')).not.toBeInTheDocument()
  fireEvent.click(screen.getByText('查看后续任务进度'))
  expect(show).toHaveBeenCalledWith('new')
})

it('暂不继续不发请求，取消操作失败也就地展示', async () => {
  const resume=vi.fn(), cancel=vi.fn().mockRejectedValue(new Error('无法提交取消请求'))
  const view=render(<EtlRunActions {...props} onResume={resume} />)
  fireEvent.click(screen.getByText('恢复下载')); fireEvent.click(screen.getByText('暂不继续'))
  expect(screen.queryByText('确认继续')).not.toBeInTheDocument()
  expect(resume).not.toHaveBeenCalled()
  view.rerender(<EtlRunActions {...props} run={{...run,status:'RUNNING'}} onCancel={cancel} onResume={resume} />)
  fireEvent.click(screen.getByText('取消运行'))
  expect(await screen.findByRole('alert')).toHaveTextContent('无法提交取消请求')
})
