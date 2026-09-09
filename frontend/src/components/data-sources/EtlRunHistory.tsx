import { useEffect, useState } from 'react'
import type { EtlRun } from '../../services/etl'
import { stateLabels, stepLabels } from '../../services/etl'
import { runModeLabels } from '../../services/etl'
import AutoPlanSummary from './AutoPlanSummary'
import EtlStepProgress from './EtlStepProgress'
import EtlRunActions from './EtlRunActions'
import EtlCollectionTiming from './EtlCollectionTiming'

export default function EtlRunHistory({ runs, busy, connected = true, onResume, onCancel, onReuse, focusedRun, onShowRun }: {
  runs: EtlRun[]; busy: boolean; connected?: boolean; onResume: (id: string) => Promise<void> | void; onCancel: (id: string) => Promise<void> | void; onReuse: (run: EtlRun) => void
  focusedRun?: string; onShowRun?: (id: string) => void
}) {
  const [now, setNow] = useState(Date.now)
  useEffect(() => {
    if (!focusedRun) return
    const element = document.getElementById(`etl-run-${focusedRun}`) as HTMLDetailsElement | null
    if (element) { element.open = true; element.scrollIntoView?.({ behavior: 'smooth', block: 'start' }); element.querySelector('summary')?.focus() }
  }, [focusedRun, runs.some(run => run.run_id === focusedRun)])
  const running = runs.some(run => run.status === 'RUNNING' || run.execution?.state === 'WORKER_OBSERVED')
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [running])
  return <section className="space-y-4" aria-label="ETL 运行记录"><h2 className="text-lg font-bold">运行记录与恢复</h2>
    {!connected ? <p role="alert" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">状态连接暂时中断，以下为上次收到的进度，不代表当前执行状态；正在自动重连。</p> : null}
    {!runs.length ? <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">还没有 ETL 运行记录。可以从快速下载开始，或先保存一条流程。</p> : null}
    {runs.map((run, index) => <details id={`etl-run-${run.run_id}`} key={run.run_id} open={index === 0 || run.status === 'RUNNING' || run.run_id === focusedRun || ['QUEUED', 'RUNNING'].includes(run.recovery?.job?.status ?? '')} className="min-w-0 scroll-mt-24 rounded-xl border border-slate-200 bg-white p-4">
      <summary className="cursor-pointer text-sm font-semibold">{run.name} · {run.recovery?.successor ? '已转入后续任务（原记录）' : run.execution?.state === 'WORKER_OBSERVED' ? '后台有下载进度（调度中断）' : stateLabels[run.status] ?? run.status} · {new Date(run.created_at).toLocaleString('zh-CN')}</summary>
      {run.execution ? <p className={`mt-3 rounded-lg p-3 text-sm ${run.execution.state === 'CONNECTED' ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-900'}`}>{run.execution.message}</p> : null}
      <p className="mt-3 text-xs text-slate-500">运行第 {run.attempt} 次尝试{run.options ? ` · ${runModeLabels[run.options.mode]}` : ''} · 候选未发布{run.cancel_requested && run.status === 'RUNNING' ? ' · 已请求取消，正在等待安全停止' : ''}</p>
      {run.recovered_from ? <p className="mt-2 break-all rounded-lg bg-blue-50 p-3 text-xs text-blue-800">恢复任务：已完成数据经校验后复用，仅执行未完成步骤。原任务 {run.recovered_from} 保留审计记录；不代表全部数据由当前版本重新计算。</p> : null}
      <EtlCollectionTiming run={run} />
      {run.auto_plan ? <details className="mt-3"><summary className="cursor-pointer text-xs text-indigo-700">查看本次自动增量区间与快照依据</summary><AutoPlanSummary plan={run.auto_plan} /></details> : null}
      {run.error ? <p role="alert" className="mt-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{run.error}</p> : null}
      {run.message && !(run.recovered_from && run.status === 'RUNNING') ? <p className="mt-3 text-sm text-slate-600">{run.message}</p> : null}
      <ol className="mt-4 space-y-2">{run.steps.map((s, n) => <li key={s.id} className={`rounded-lg border p-3 ${s.status === 'FAILED' ? 'border-rose-200 bg-rose-50' : s.status === 'RUNNING' ? 'border-indigo-300 bg-indigo-50' : 'border-slate-100 bg-slate-50'}`}>
        <div className="flex flex-wrap justify-between gap-2 text-sm"><strong>{n + 1}. {s.name}</strong><span>{s.worker_only ? '后台有下载进度（调度中断）' : stateLabels[s.status] ?? s.status}</span></div>
        {s.imported_from ? <p className="mt-1 text-xs text-blue-700">已核验复用原任务结果，未重复下载</p> : null}
        <p className="mt-1 text-xs text-slate-500">{stepLabels[s.kind]}{s.rows !== undefined ? ` · ${s.rows} 行` : ''}{s.pages !== undefined ? ` · ${s.pages} 页` : ''}{s.started_at && s.finished_at ? ` · ${Math.max(0, (Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000).toFixed(1)} 秒` : ''}</p>
        {s.error ? <p className="mt-2 text-xs text-rose-800">{s.error}</p> : null}
        <EtlStepProgress step={s} now={now} />
        {s.output ? <details className="mt-2"><summary className="cursor-pointer text-xs text-indigo-700">查看本步结果与依据</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-white p-3 text-xs">{JSON.stringify(s.output, null, 2)}</pre></details> : null}
      </li>)}</ol>
      <EtlRunActions run={run} busy={busy || !connected} onResume={onResume} onCancel={onCancel} onReuse={onReuse} onShowRun={onShowRun} />
    </details>)}
  </section>
}
