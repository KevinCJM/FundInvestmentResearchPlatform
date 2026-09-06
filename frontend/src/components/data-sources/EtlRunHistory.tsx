import type { EtlRun } from '../../services/etl'
import { stateLabels, stepLabels } from '../../services/etl'
import { buttonClass } from './EditorFields'

export default function EtlRunHistory({ runs, busy, onResume, onCancel, onReuse }: {
  runs: EtlRun[]; busy: boolean; onResume: (id: string) => void; onCancel: (id: string) => void; onReuse: (run: EtlRun) => void
}) {
  return <section className="space-y-4" aria-label="ETL 运行记录"><h2 className="text-lg font-bold">运行记录与恢复</h2>
    {!runs.length ? <p className="rounded-xl bg-slate-50 p-5 text-sm text-slate-600">还没有 ETL 运行记录。可以从快速下载开始，或先保存一条流程。</p> : null}
    {runs.map((run, index) => <details key={run.run_id} open={index === 0 || run.status === 'RUNNING'} className="min-w-0 rounded-xl border border-slate-200 bg-white p-4">
      <summary className="cursor-pointer text-sm font-semibold">{run.name} · {stateLabels[run.status] ?? run.status} · {new Date(run.created_at).toLocaleString('zh-CN')}</summary>
      <p className="mt-3 text-xs text-slate-500">运行第 {run.attempt} 次尝试 · 候选未发布{run.cancel_requested && run.status === 'RUNNING' ? ' · 已请求取消，正在等待安全停止' : ''}</p>
      {run.error ? <p role="alert" className="mt-3 rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{run.error}</p> : null}
      {run.message ? <p className="mt-3 text-sm text-slate-600">{run.message}</p> : null}
      <ol className="mt-4 space-y-2">{run.steps.map((s, n) => <li key={s.id} className={`rounded-lg border p-3 ${s.status === 'FAILED' ? 'border-rose-200 bg-rose-50' : s.status === 'RUNNING' ? 'border-indigo-300 bg-indigo-50' : 'border-slate-100 bg-slate-50'}`}>
        <div className="flex flex-wrap justify-between gap-2 text-sm"><strong>{n + 1}. {s.name}</strong><span>{stateLabels[s.status] ?? s.status}</span></div>
        <p className="mt-1 text-xs text-slate-500">{stepLabels[s.kind]}{s.rows !== undefined ? ` · ${s.rows} 行` : ''}{s.pages !== undefined ? ` · ${s.pages} 页` : ''}{s.started_at && s.finished_at ? ` · ${Math.max(0, (Date.parse(s.finished_at) - Date.parse(s.started_at)) / 1000).toFixed(1)} 秒` : ''}</p>
        {s.error ? <p className="mt-2 text-xs text-rose-800">{s.error}</p> : null}
        {s.output ? <details className="mt-2"><summary className="cursor-pointer text-xs text-indigo-700">查看本步结果与依据</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-all rounded-lg bg-white p-3 text-xs">{JSON.stringify(s.output, null, 2)}</pre></details> : null}
      </li>)}</ol>
      <div className="mt-4 flex flex-wrap gap-2">{['FAILED', 'INTERRUPTED', 'CANCELLED'].includes(run.status) ? <button type="button" className={buttonClass} disabled={busy} onClick={() => onResume(run.run_id)}>继续未完成步骤</button> : null}{run.status === 'RUNNING' ? <button type="button" className={buttonClass} disabled={busy || run.cancel_requested} onClick={() => onCancel(run.run_id)}>取消运行</button> : <button type="button" className={buttonClass} onClick={() => onReuse(run)}>复制到编排器</button>}</div>
    </details>)}
  </section>
}
