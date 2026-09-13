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
  const visibleRuns = runs.filter(run => run.status === 'RUNNING' || !runs.some(current => current.history?.records.some(entry => entry.run_id === run.run_id)))
  const focusedId = visibleRuns.find(run => run.run_id === focusedRun || run.history?.records.some(entry => entry.run_id === focusedRun))?.run_id
  useEffect(() => {
    if (!focusedId) return
    const element = document.getElementById(`etl-run-${focusedId}`) as HTMLDetailsElement | null
    if (element) { element.open = true; element.scrollIntoView?.({ behavior: 'smooth', block: 'start' }); element.querySelector('summary')?.focus() }
  }, [focusedId])
  const running = runs.some(run => run.status === 'RUNNING' || run.execution?.state === 'WORKER_OBSERVED')
  useEffect(() => {
    if (!running) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [running])
  const renderStep = (s: EtlRun['steps'][number], n: number) => <li key={s.id} className={`min-w-0 rounded-lg border p-3 ${s.status === 'FAILED' ? 'border-rose-200 bg-rose-50' : s.status === 'RUNNING' ? 'border-accent-300 bg-accent-50' : 'border-slate-100 bg-slate-50'}`}>
    <div className="flex flex-wrap justify-between gap-2 text-sm"><strong>{n + 1}. {s.name}</strong><span>{s.worker_only ? '后台有下载进度（调度中断）' : s.output?.data_quality?.status === 'CONFLICTED' ? '采集完成 · 数据待核验' : stateLabels[s.status] ?? s.status}</span></div>
    {s.imported_from ? <p className="mt-1 text-xs text-accent-700">已核验复用原任务结果，未重复下载</p> : null}
    <p className="mt-1 text-xs text-slate-600">{stepLabels[s.kind]}{s.rows !== undefined ? ` · ${s.rows} 行` : ''}{s.pages !== undefined ? ` · ${s.pages} 页` : ''}{s.started_at && s.finished_at ? ` · ${Math.max(0, (Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000).toFixed(1)} 秒` : ''}</p>
    {s.error ? <p className="mt-2 break-words text-xs text-rose-800">{s.error}</p> : null}
    {s.output?.data_quality?.status === 'CONFLICTED' ? <p role="alert" className="mt-2 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">已隔离 {s.output.data_quality.conflicting_keys} 个持仓冲突，全部原始值已保留。冲突数值留空，未选取任意一条；不能作为标准研究数据或发布。详情见本步结果与依据。</p> : null}
    <EtlStepProgress step={s} now={now} />
    {s.output ? <details className="mt-2"><summary className="cursor-pointer text-xs text-accent-700">查看本步结果与依据</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-white p-3 text-xs">{JSON.stringify(s.output, null, 2)}</pre></details> : null}
  </li>
  return <section className="space-y-4" aria-label="ETL 运行记录"><h2 className="text-lg font-bold">下载任务</h2>
    <p className="text-xs text-slate-600">每个任务显示当前进度；之前的失败与续跑记录收在任务内，不会重复列出。</p>
    {!connected ? <p role="alert" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-800">状态连接暂时中断，以下为上次收到的进度，不代表当前执行状态；正在自动重连。</p> : null}
    {!runs.length ? <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">还没有 ETL 运行记录。可以从快速下载开始，或先保存一条流程。</p> : null}
    {visibleRuns.map((run, index) => {
      const title = run.history?.display_name ?? (run.recovered_from ? run.name.replace(/(?:\s*（恢复）)+$/, '') : run.name)
      const recovering = ['QUEUED', 'RUNNING'].includes(run.recovery?.job?.status ?? '')
      const updatedAt = recovering ? run.recovery?.job?.updated_at || run.updated_at || run.created_at : run.updated_at || run.created_at
      const activeIndex = run.steps.findIndex(s => s.status === 'RUNNING' || s.worker_only)
      const currentIndex = recovering ? -1 : activeIndex >= 0 ? activeIndex : run.steps.findIndex(s => ['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(s.status))
      const failedCount = run.steps.filter(s => s.status === 'FAILED').length
      const complete = run.steps.filter(s => s.status === 'SUCCEEDED').length
      const conflicts = run.steps.filter(s => s.output?.data_quality?.status === 'CONFLICTED')
      return <details id={`etl-run-${run.run_id}`} data-testid="etl-task-card" key={run.history?.root_run_id ? `${run.history.root_run_id}-${index}` : run.run_id} open={index === 0 || run.status === 'RUNNING' || run.run_id === focusedId || recovering} className="min-w-0 scroll-mt-24 rounded-xl border border-slate-200 bg-white p-4">
      <summary className="cursor-pointer break-words text-sm font-semibold focus-visible:outline-accent-600">
        <span>{title}</span><span className={`ml-2 inline-block rounded-full px-2 py-1 text-xs ${recovering || run.status === 'RUNNING' ? 'bg-accent-50 text-accent-700' : run.status === 'FAILED' ? 'bg-rose-50 text-rose-700' : 'bg-slate-100 text-slate-600'}`}>{recovering ? '正在校验并续跑' : run.execution?.state === 'WORKER_OBSERVED' ? '后台有下载进度（调度中断）' : stateLabels[run.status] ?? run.status}</span>
        <span className="mt-2 block text-xs font-normal text-slate-600">已完成 {complete} / {run.steps.length} 个步骤 · 最近更新 {new Date(updatedAt).toLocaleString('zh-CN')}</span>
      </summary>
      <EtlRunActions key={run.run_id} run={{ ...run, name: title }} busy={busy || !connected} onResume={onResume} onCancel={onCancel} onReuse={onReuse} onShowRun={onShowRun} />
      {conflicts.length > 0 ? <p role="alert" className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{conflicts.map(s => s.name).join('、')}：采集已完成，但供应商数据有冲突，原始证据已隔离保存。下载完成不代表数据质量通过或已发布。</p> : null}
      {run.execution ? <p className="mt-2 text-xs text-slate-600">{run.execution.message}</p> : null}
      <p className="mt-3 text-xs text-slate-600">{run.options ? `${runModeLabels[run.options.mode]} · ` : ''}候选未发布{run.cancel_requested && run.status === 'RUNNING' ? ' · 已请求取消，正在等待安全停止' : ''}</p>
      {run.history && (run.history.records.length > 0 || run.history.resume_count > 0 || run.history.lineage_warning) ? <details className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs">
        <summary className="cursor-pointer text-slate-600 focus-visible:outline-accent-600">历史记录 · 已续跑 {run.history.resume_count} 次，保留 {run.history.records.length} 条原记录</summary>
        <p className="mt-2 text-slate-600">下面是之前的执行记录，不代表当前失败。已完成数据经校验后复用；采集时点和原执行版本不变。</p>
        {run.history.lineage_warning ? <p role="alert" className="mt-2 text-amber-800">{run.history.lineage_warning}</p> : null}
        <ol aria-label="历次执行记录" className="mt-2 max-h-64 space-y-3 overflow-y-auto">
          {run.history.records.map(entry => <li key={entry.run_id} className="break-words border-t border-slate-200 pt-2">
            <p>{entry.created_at ? new Date(entry.created_at).toLocaleString('zh-CN') : '时间未知'} · 当时状态：{stateLabels[entry.status] ?? entry.status}{entry.failed_step ? ` · ${entry.failed_step}` : ''}</p>
            {entry.error ? <p className="mt-1 text-slate-600">{entry.error}</p> : null}
            <p className="mt-1 break-all text-slate-600">记录编号：{entry.run_id}</p>
          </li>)}
        </ol>
      </details> : run.recovered_from ? <p className="mt-2 break-all text-xs text-slate-600">原任务 {run.recovered_from} 保留审计记录；已完成数据经校验后复用。</p> : null}
      {currentIndex >= 0 ? <ol aria-label="当前工作步骤" className="mt-4">{renderStep(run.steps[currentIndex], currentIndex)}</ol> : null}
      {run.status === 'RUNNING' && failedCount > 0 ? <p role="status" className="mt-3 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">已有 {failedCount} 个节点失败，其他无数据依赖的步骤仍在继续。失败与阻断原因可在“查看全部步骤”中展开；本次数据尚未发布。</p> : null}
      <EtlCollectionTiming run={run} />
      {run.auto_plan ? <details className="mt-3"><summary className="cursor-pointer text-xs text-accent-700">查看本次自动增量区间与快照依据</summary><AutoPlanSummary plan={run.auto_plan} /></details> : null}
      {run.error && !recovering ? <p role="alert" className="mt-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{run.error}</p> : null}
      {run.message && !recovering && !(run.recovered_from && run.status === 'RUNNING') ? <p className="mt-3 text-sm text-slate-600">{run.message}</p> : null}
      {run.steps.length > 0 ? <details className="mt-4"><summary className="cursor-pointer text-xs text-accent-700">查看全部步骤（共 {run.steps.length} 个，已完成 {complete} 个）</summary><ol aria-label="全部工作步骤" className="mt-3 space-y-2">{run.steps.map(renderStep)}</ol></details> : null}
    </details>})}
  </section>
}
