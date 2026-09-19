import { percentText } from '../risk-models/ResearchUI'
import type { FundingSummary, PolicyCandidate } from '../../services/strategicAllocation'
import { amountText } from './model'

export function FundingOverview({ value }: { value: FundingSummary }) {
  return <div className="space-y-3" aria-label="资金测算结果">
    <dl className="grid gap-4 sm:grid-cols-3">
      <div><dt className="text-xs text-slate-600">可投资本金（{value.currency}）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.investable_capital)}</dd></div>
      <div><dt className="text-xs text-slate-600">所需固定年复合收益（扣费前）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{value.root_status === 'solved' ? percentText(value.required_effective_return) : value.root_status === 'at_lower_bound' ? '不高于−99%搜索下界' : value.root_status === 'not_applicable' ? '未定义资金成功条件' : '超出500%搜索上界'}</dd></div>
      <div><dt className="text-xs text-slate-600">前{value.liquidity_months}月组合内流动性底线</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.required_liquid_capital)} / {percentText(value.required_liquid_weight)}</dd></div>
    </dl>
    <p className="text-xs leading-5 text-slate-600">期末名义目标 {amountText(value.nominal_terminal_target)}；期间投入 {amountText(value.total_contributions)}、必要支付 {amountText(value.total_withdrawals)}。所需收益是为覆盖已填费用而要求的扣费前固定复合收益，不会自动变成CMA算术预期收益。流动性底线不从本金再扣一次。</p>
    {value.liquidity_payment_buffer !== undefined && <p className="text-sm leading-6 text-slate-700" aria-label="近期支付缓冲">按零收益、压力投入计算，覆盖前{value.liquidity_months}月必要支付后，组合内本金缓冲为 {amountText(value.liquidity_payment_buffer)} {value.currency}（{percentText(value.liquidity_payment_buffer_ratio)}）。{Number(value.liquidity_shortfall_capital) > 0 ? `仍有支付缺口 ${amountText(value.liquidity_shortfall_capital)} ${value.currency}。` : ''}这是支付覆盖测算，不是可承受回撤上限。</p>}
    <details className="border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm">逐月投入与支付计划</summary>
      <div className="mt-3 overflow-x-auto"><table aria-label="逐月现金流" className="w-full text-sm"><caption className="sr-only">逐月现金流</caption><thead><tr><th scope="col" className="p-2 text-left">月份</th><th scope="col" className="p-2 text-right">投入</th><th scope="col" className="p-2 text-right">必要支付</th></tr></thead><tbody>{value.monthly_cashflows.map(row => <tr key={row.month} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">第{row.month}月</th><td className="p-2 text-right tabular-nums">{amountText(row.contribution)}</td><td className="p-2 text-right tabular-nums">{amountText(row.withdrawal)}</td></tr>)}</tbody></table></div>
    </details>
  </div>
}

export function GoalCandidateSummary({ candidate }: { candidate: PolicyCandidate }) {
  const goal = candidate.goal_check
  return <div className="space-y-2 text-sm leading-6">
    {goal && <>
      <p className={goal.within_limits ? 'text-slate-800' : 'text-amber-800'}>目标成功概率 {percentText(goal.central.success_probability)}；95%模拟采样区间 {percentText(goal.central.probability_lower)} 至 {percentText(goal.central.probability_upper)}。{goal.within_limits ? '区间下界达到' : '区间下界未达到'} {percentText(goal.threshold)} 门槛。</p>
      <p className="text-xs text-slate-600">这里的“通过”仅限所选模型和CMA假设，不代表实际收益保证。当前TAA预算不承诺维持整段资金目标概率。</p>
    </>}
    {candidate.benchmark_check && <p className="text-slate-700">相对 {candidate.benchmark_check.name}：预期年超额 {percentText(candidate.benchmark_check.expected_excess_return)}；主动风险 {percentText(candidate.benchmark_check.tracking_error)} / 上限 {percentText(candidate.benchmark_check.max_tracking_error)}。</p>}
  </div>
}
