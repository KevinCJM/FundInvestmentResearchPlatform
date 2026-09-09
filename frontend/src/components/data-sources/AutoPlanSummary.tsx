import type { AutoIncrementalPlan } from '../../services/etl'

const dateText = (value: string) => /^\d{8}$/.test(value) ? `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6)}` : value

export default function AutoPlanSummary({ plan }: { plan: AutoIncrementalPlan }) {
  return <section aria-label="自动增量计划" className="min-w-0 space-y-3 rounded-xl border border-indigo-200 bg-indigo-50 p-4 text-sm">
    <h3 className="font-bold">本次实际下载计划</h3>
    <p className="break-all text-xs">基线快照：{plan.snapshot}</p>
    <p className="text-xs leading-6">请求截止：{plan.cutoff_date}（上海时区昨日，不代表上游已全部披露）。重取最近 {plan.lookback_trade_days} 个交易日以覆盖迟报和修订；写入私有候选，不覆盖正式快照。</p>
    <div className="grid gap-2 lg:grid-cols-2">{plan.steps.map(step => <article key={step.id} className="min-w-0 rounded-lg bg-white p-3">
      <h4 className="font-semibold">{step.name}</h4>
      <p className="mt-1 text-xs">{step.strategy === 'incremental' ? `已有最新：${step.latest_date ?? '未知'}；请求区间：${dateText(step.start_date)} 至 ${dateText(step.end_date)}` : step.strategy === 'derive' ? '本地派生，无网络下载' : '目录、日历或披露表刷新'}</p>
      <p className="mt-2 text-xs leading-5 text-slate-600">{step.message}</p>
    </article>)}</div>
    {plan.errors.map((error, i) => <p key={i} className="text-rose-800">{error.message}</p>)}
  </section>
}
