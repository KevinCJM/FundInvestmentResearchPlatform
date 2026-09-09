import type { EtlRun } from '../../services/etl'

type Step = EtlRun['steps'][number]
const duration = (seconds: number) => seconds >= 3600 ? `${Math.floor(seconds / 3600)} 小时 ${Math.floor(seconds % 3600 / 60)} 分` : seconds >= 60 ? `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒` : `${Math.floor(seconds)} 秒`
const age = (date: string | undefined, now: number) => date && Number.isFinite(Date.parse(date)) ? Math.max(0, (now - Date.parse(date)) / 1000) : null

export default function EtlStepProgress({ step, now }: { step: Step; now: number }) {
  const active = step.status === 'RUNNING' || step.worker_only === true, p = step.progress
  if (!active && !p) return null
  const elapsed = age(step.started_at, active ? now : Date.parse(step.finished_at || '') || now)
  const quiet = age(p?.activity_at || step.started_at, now)
  const heartbeat = age(step.heartbeat_at, now)
  const count = p?.completed, total = p?.total
  const known = Number.isFinite(count) && Number.isFinite(total) && total! > 0 && count! >= 0 && count! <= total!
  const percent = known ? Math.floor(count! / total! * 100) : null
  return <section aria-label={`${step.name}实时进度`} className="mt-3 min-w-0 space-y-2 rounded-lg border border-indigo-100 bg-white/80 p-3 text-xs text-slate-700">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <span className="font-medium">{active ? p?.phase || '正在执行' : '最后执行进度'}{active && percent !== null ? ` · ${percent}%` : ''}</span>
      {elapsed !== null ? <span>已耗时 {duration(elapsed)}</span> : null}
    </div>
    {active ? <div role="progressbar" aria-label={`${step.name}当前阶段进度`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent ?? undefined} aria-valuetext={percent === null ? '总量尚未确定，正在执行' : `当前阶段 ${percent}%`} className="h-2 overflow-hidden rounded-full bg-indigo-100">
      <div className={`h-full rounded-full bg-indigo-500 ${percent === null ? 'w-1/3 motion-safe:animate-pulse' : 'transition-[width]'}`} style={percent === null ? undefined : { width: `${percent}%` }} />
    </div> : null}
    <p className="break-words">{p?.message || '当前执行器尚未上报详细进度；页面会持续查询，日志不可见不代表任务停止。'}</p>
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-slate-500">
      {known ? <span>本阶段已处理 {count!.toLocaleString('zh-CN')} / {total!.toLocaleString('zh-CN')} {p?.unit || '项'}</span> : active ? <span>总量尚未确定</span> : null}
      {p?.batches !== undefined ? <span>已接收 {p.batches.toLocaleString('zh-CN')} 批</span> : null}
      {p?.received_rows !== undefined ? <span>累计接收 {p.received_rows.toLocaleString('zh-CN')} 行（含重复/重试批次）</span> : null}
      {active && step.worker_only && quiet !== null ? <span>工作进程进度更新：{duration(quiet)}前</span> : active && heartbeat !== null ? <span>调度心跳：{duration(heartbeat)}前</span> : null}
    </div>
    {active && percent !== null ? <p className="text-slate-500">百分比仅代表当前分片阶段，不代表整个节点；之后仍可能合并、校验文件。</p> : null}
    {active && quiet !== null && quiet >= 60 ? <p className="rounded bg-amber-50 p-2 text-amber-800">已有 {duration(quiet)} 未收到新进度。可能在等待接口、限频或处理本地文件，不能据此判定卡死。</p> : null}
    {p?.logs?.length ? <details><summary className="cursor-pointer text-indigo-700">最近日志（{p.logs.length} 条，已脱敏）</summary>
      <ol aria-label="最近执行日志" className="mt-2 max-h-48 space-y-1 overflow-y-auto rounded bg-slate-900 p-3 font-mono text-slate-100">{p.logs.map((line, i) => <li key={`${line.at}-${i}`} className="whitespace-pre-wrap break-all"><time className="mr-2 text-slate-400">{new Date(line.at).toLocaleTimeString('zh-CN')}</time>{line.message}</li>)}</ol>
    </details> : null}
  </section>
}
