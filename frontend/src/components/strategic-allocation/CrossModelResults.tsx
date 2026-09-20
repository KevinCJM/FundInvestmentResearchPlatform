import { useI18n } from '../../i18n/runtime'
import type { CrossModelResult } from '../../services/strategicAllocation'
import { percentText } from '../risk-models/ResearchUI'
import { Badge } from '../ui'

export default function CrossModelResults({ rows, common = false }: { rows: CrossModelResult[]; common?: boolean }) {
  const { s } = useI18n()
  return <div className="min-w-0 space-y-3">
    <p className="text-sm leading-6 text-slate-600">{s(common ? 'multiCma.commonHint' : 'multiCma.crossHint')}</p>
    <p className="text-xs text-slate-600 sm:hidden">{s('multiCma.scroll')}</p>
    <div className="overflow-x-auto"><table aria-label={s('multiCma.crossTitle')} className="w-full min-w-[540px] text-sm">
      <thead><tr>{['source', common ? 'requiredColumn' : 'weight', 'return', 'risk', 'status'].map((key, index) => <th scope="col" key={key} className={`p-2 ${index > 0 && index < 4 ? 'text-right' : 'text-left'}`}>{s(`multiCma.${key}`)}</th>)}</tr></thead>
      <tbody>{rows.map(row => <tr key={row.cma_id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{row.name}</th>
        <td className="p-2 text-right">{common ? s('multiCma.required') : percentText(row.weight)}</td>
        {[row.metrics.expected_return, row.metrics.volatility].map((value, index) => <td key={index} className="p-2 text-right tabular-nums">{percentText(value)}</td>)}
        <td className="p-2"><Badge tone={row.within_limits ? 'success' : 'warning'}>{s(`multiCma.${row.within_limits ? 'pass' : 'fail'}`)}</Badge></td></tr>)}</tbody>
    </table></div>
    <div className="divide-y divide-slate-200">{rows.filter(row => row.violations.length || row.goal_check || row.benchmark_check || row.expected_tracking_error != null).map(row => <div className="space-y-1 py-2 text-sm" key={row.cma_id}>
      <p className="font-medium">{row.name} · {s('multiCma.details')}</p>{row.violations.map((reason, index) => <p className="text-amber-800" key={index}>{reason}</p>)}
      {row.goal_check && <p>{s('multiCma.goal')}：{s(`multiCma.${row.goal_check.within_limits ? 'pass' : 'fail'}`)}</p>}
      {row.benchmark_check && <p>{s('multiCma.trackingError')}：{percentText(row.benchmark_check.tracking_error)} / {percentText(row.benchmark_check.max_tracking_error)}</p>}
      {row.expected_tracking_error != null && <p>{s('multiCma.tacticalTrackingError')}：{percentText(row.expected_tracking_error)}</p>}
    </div>)}</div>
  </div>
}
