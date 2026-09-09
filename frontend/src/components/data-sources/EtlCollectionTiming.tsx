import type { EtlRun } from '../../services/etl'

const time = (value: string | null, timezone: string) => value
  ? new Date(value).toLocaleString('zh-CN', { timeZone: timezone, hour12: false }) : '未知'
const basisLabels = { batch_receipts: '批次接收时间', execution_window: '执行时间范围（估计）', unknown: '采集时点未知' }

export default function EtlCollectionTiming({ run }: { run: EtlRun }) {
  const timing = run.collection_timing
  if (!timing || !timing.windows.length && !run.resume_events?.length) return null
  return <section aria-label="采集时点与 PIT 风险" className="mt-3 space-y-2 rounded-lg border border-slate-200 p-3 text-sm">
    <h3 className="font-semibold">采集时点与 PIT 风险</h3>
    {timing.warnings.map(warning => <p key={warning.code} role="alert" className="rounded-lg bg-amber-50 p-3 text-amber-900">{warning.message}</p>)}
    <p>采集记录日期：{timing.first_date ? `${timing.first_date} 至 ${timing.last_date}` : '未知'}（北京时间）</p>
    <p className="text-xs text-slate-600">{timing.boundary}</p>
    <details>
      <summary className="cursor-pointer text-xs text-indigo-700">查看采集批次时间与续跑记录</summary>
      <p className="my-2 text-xs text-slate-500">{timing.scope} 以下按节点和尝试汇总，不是逐条数据的可得时点。</p>
      <div className="overflow-x-auto"><table className="w-full text-left text-xs" aria-label="节点采集时间范围">
        <thead><tr>{['节点 / 尝试', '时间范围（北京时间）', '时间依据 / 来源任务'].map(label => <th key={label} className="p-2">{label}</th>)}</tr></thead>
        <tbody>{timing.windows.map(window => <tr key={`${window.run_id}-${window.step_id}-${window.attempt}`} className="border-t">
          <td className="p-2">{window.name} · 第 {window.attempt} 次</td>
          <td className="p-2">{time(window.first_at, timing.timezone)}<br />{time(window.last_at, timing.timezone)}</td>
          <td className="break-all p-2">{basisLabels[window.basis]}<br />{window.run_id}</td>
        </tr>)}</tbody>
      </table></div>
      {run.resume_events?.map(event => <p key={`${event.attempt}-${event.at}`} className="mt-2 text-xs text-slate-600">
        第 {event.attempt} 次续跑 · {time(event.at, timing.timezone)}
        {event.warnings.map(warning => <span key={warning.code} className="mt-1 block text-amber-900">{warning.message}</span>)}
      </p>)}
    </details>
  </section>
}
