import type { InstitutionalDiagnostics } from '../../services/institutionalContext'
import { amountText } from './model'
export default function InstitutionalResults({ value }: { value: InstitutionalDiagnostics }) {
  const balance = value.balance_sheet
  return <section className="space-y-3 border-t border-slate-200 pt-4" aria-label="机构经济状况诊断">
    <h3 className="text-base font-semibold">经济状况与核验边界</h3>
    {balance ? <><p className="text-sm text-slate-600">研究快照：{balance.as_of} · {balance.currency}。来源：{balance.source}</p><dl className="grid gap-3 sm:grid-cols-3">{[['资产合计', balance.total_assets], ['扣除已确认负债后净资产', balance.net_assets_after_confirmed_liabilities], ['单独披露的未缴承诺', balance.uncalled_commitments]].map(([label, amount]) => <div key={String(label)}><dt className="text-sm text-slate-600">{label}</dt><dd className="text-sm tabular-nums">{amount === null ? '未提供完整数据' : amountText(amount as number)}</dd></div>)}</dl></> : <p className="text-sm text-slate-600">未提供经济快照，未计算资产负债汇总。</p>}
    <p className="text-sm text-slate-600">已计算：输入快照汇总。未建模：税务、监管、币种对冲、杠杆与特殊流动性；不存在机器合规或独立审批通过。</p>
    <div className="grid gap-2 sm:grid-cols-2">{value.review_blockers.map(reason => <p key={reason} className="text-sm text-amber-800">{reason}</p>)}</div>
    {!value.review_blockers.length && <p className="text-sm text-slate-700">研究日的人工证据已填写；当前应用仍会重新检查有效期和证据。</p>}
  </section>
}
