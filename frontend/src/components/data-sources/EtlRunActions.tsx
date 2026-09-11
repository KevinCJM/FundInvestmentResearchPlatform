import { useRef, useState } from 'react'
import type { EtlRun } from '../../services/etl'
import { buttonClass } from './EditorFields'

export default function EtlRunActions({ run, busy, onResume, onCancel, onReuse, onShowRun }: {
  run: EtlRun; busy: boolean; onResume: (id: string) => Promise<void> | void
  onCancel: (id: string) => Promise<void> | void; onReuse: (run: EtlRun) => void
  onShowRun?: (id: string) => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [pending, setPending] = useState<'resume' | 'cancel' | null>(null)
  const [feedback, setFeedback] = useState<{ error: boolean; message: string } | null>(null)
  const submitting = useRef(false)
  const canRecover = ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(run.status)
  const job = run.recovery?.job
  const recovering = job?.status === 'QUEUED' || job?.status === 'RUNNING'
  const successor = run.recovery?.successor
  const execute = async (action: 'resume' | 'cancel') => {
    if (submitting.current) return
    submitting.current = true; setPending(action); setConfirming(false); setFeedback(null)
    try {
      await (action === 'resume' ? onResume(run.run_id) : onCancel(run.run_id))
      setFeedback({ error: false, message: action === 'resume' ? '恢复请求已接受，请查看恢复校验或后续任务进度；尚不代表全部下载完成。' : '取消请求已提交，正在等待安全停止。' })
    } catch (reason) {
      setFeedback({ error: true, message: reason instanceof Error ? reason.message : '操作未完成，请检查任务状态后再试。' })
    } finally { submitting.current = false; setPending(null) }
  }
  if (successor && !recovering) return <section aria-label={`${run.name}运行操作`} className="mt-4 space-y-3 rounded-lg bg-blue-50 p-3 text-sm text-blue-900">
    <p>本记录是原任务，已有恢复后的后续任务。原失败记录保留，不代表当前下载失败。</p>
    <button type="button" className={buttonClass} onClick={() => onShowRun?.(successor.run_id)}>查看后续任务进度</button>
  </section>
  return <section aria-label={`${run.name}运行操作`} className="mt-4 space-y-3">
    {canRecover && run.recovery?.blockers.length ? <details className="text-xs text-slate-500"><summary className="cursor-pointer">恢复检查说明（点击恢复后重新核验）</summary>{run.recovery.blockers.map(item => <p key={item.code} className="mt-2">{item.message}</p>)}</details> : null}
    {canRecover && run.recovery?.warnings?.map(item => <p key={item.code} role="alert" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{item.message}</p>)}
    {pending ? <p role="status" className="rounded-lg bg-indigo-50 p-3 text-sm text-indigo-800">{pending === 'resume' ? '正在核验执行版本、下载锁和已完成文件；大文件校验可能耗时，请勿重复点击。' : '正在提交取消请求…'}</p> : null}
    {job ? <div className={`space-y-2 rounded-lg p-3 text-sm ${job.status === 'FAILED' || job.status === 'INTERRUPTED' ? 'bg-rose-50 text-rose-800' : 'bg-indigo-50 text-indigo-900'}`}>
      <p role={job.status === 'FAILED' || job.status === 'INTERRUPTED' ? 'alert' : 'status'}>{job.status === 'FAILED' || job.status === 'INTERRUPTED' ? '恢复下载失败：' : ''}{job.phase} · {job.message}</p>
      {recovering ? <><div role="progressbar" aria-label="恢复校验进行中" className="h-2 overflow-hidden rounded bg-indigo-100"><div className="h-full w-1/3 animate-pulse rounded bg-indigo-500" /></div><p className="text-xs">校验完成后自动继续下载；可刷新页面，独立恢复任务不会因此停止。总量未知，不估算百分比。</p></> : null}
      {job.logs.length ? <details><summary className="cursor-pointer text-xs">恢复校验日志</summary><ol className="mt-2 max-h-40 overflow-auto text-xs">{job.logs.map((entry, index) => <li key={index}>{new Date(entry.at).toLocaleTimeString('zh-CN')} · {entry.message}</li>)}</ol></details> : null}
    </div> : null}
    {feedback ? <p role={feedback.error ? 'alert' : 'status'} className={`whitespace-pre-line rounded-lg p-3 text-sm ${feedback.error ? 'bg-rose-50 text-rose-800' : 'bg-emerald-50 text-emerald-900'}`}>{feedback.message}</p> : null}
    {confirming && canRecover && !recovering ? <div role="group" aria-label="确认恢复任务" className="rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-sm">
      <p>保留已成功步骤，只继续未完成步骤；可能访问数据源并消耗配额。确认继续吗？</p>
      <p className="mt-2">点击确认后才检查文件和版本。同版本从断点继续；版本变化时，先校验并复用旧数据到新任务。无法安全复用则明确提示失败，不删除旧文件。</p>
      <p className="mt-2 text-amber-900">如本次续跑跨采集日期，新旧下载内容的时点可能不一致。可接受风险后继续；此操作不会将数据标记为 PIT 一致，也不会改变原下载区间。</p>
      <div className="mt-2 flex flex-wrap gap-2"><button type="button" className={buttonClass} disabled={busy || pending !== null} onClick={() => void execute('resume')}>确认继续</button><button type="button" className={buttonClass} onClick={() => setConfirming(false)}>暂不继续</button></div>
    </div> : null}
    <div className="flex flex-wrap gap-2">
      {canRecover ? <button type="button" className={buttonClass} disabled={busy || pending !== null || recovering} onClick={() => { setConfirming(true); setFeedback(null) }}>{pending === 'resume' || recovering ? '正在恢复…' : '恢复下载'}</button> : null}
      {run.status === 'RUNNING' ? <button type="button" className={buttonClass} disabled={busy || pending !== null || run.cancel_requested} onClick={() => void execute('cancel')}>{pending === 'cancel' ? '正在提交取消…' : '取消运行'}</button> : <button type="button" className={buttonClass} disabled={busy || pending !== null} onClick={() => onReuse(run)}>复制到编排器</button>}
    </div>
    {canRecover ? <p className="text-xs text-slate-500">“恢复下载”会核验并复用原数据；“复制到编排器”仅复制流程，不是断点续跑。</p> : null}
  </section>
}
