import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Field, inputClass, percentText } from '../risk-models/ResearchUI'
import type { MandateAssessment, FundingSummary, PolicyCandidate } from '../../services/strategicAllocation'
import { amountText } from './model'

export function FundingOverview({ value }: { value: FundingSummary }) {
  return <div className="space-y-3" aria-label="资金测算结果">
    <dl className="grid gap-4 sm:grid-cols-3">
      <div><dt className="text-xs text-slate-600">可投资本金（{value.currency}）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.investable_capital)}</dd></div>
      <div><dt className="text-xs text-slate-600">所需固定年复合收益（扣费前）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{value.root_status === 'solved' ? percentText(value.required_effective_return) : value.root_status === 'at_lower_bound' ? '不高于−99%搜索下界' : '超出500%搜索上界'}</dd></div>
      <div><dt className="text-xs text-slate-600">前{value.liquidity_months}月组合内流动性底线</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.required_liquid_capital)} / {percentText(value.required_liquid_weight)}</dd></div>
    </dl>
    <p className="text-xs leading-5 text-slate-600">期末名义目标 {amountText(value.nominal_terminal_target)}；期间投入 {amountText(value.total_contributions)}、必要支付 {amountText(value.total_withdrawals)}。所需收益是为覆盖已填费用而要求的扣费前固定复合收益，不会自动变成CMA算术预期收益。流动性底线不从本金再扣一次。</p>
    {value.liquidity_payment_buffer !== undefined && <p className="text-sm leading-6 text-slate-700" aria-label="近期支付缓冲">按零收益、压力投入计算，覆盖前{value.liquidity_months}月必要支付后，组合内本金缓冲为 {amountText(value.liquidity_payment_buffer)} {value.currency}（{percentText(value.liquidity_payment_buffer_ratio)}）。{Number(value.liquidity_shortfall_capital) > 0 ? `仍有支付缺口 ${amountText(value.liquidity_shortfall_capital)} ${value.currency}。` : ''}这是支付覆盖测算，不是可承受回撤上限。</p>}
    <details className="border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm">逐月投入与支付计划</summary>
      <div className="mt-3 overflow-x-auto"><table aria-label="逐月现金流" className="w-full text-sm"><thead><tr><th scope="col" className="p-2 text-left">月份</th><th scope="col" className="p-2 text-right">投入</th><th scope="col" className="p-2 text-right">必要支付</th></tr></thead><tbody>{value.monthly_cashflows.map(row => <tr key={row.month} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">第{row.month}月</th><td className="p-2 text-right tabular-nums">{amountText(row.contribution)}</td><td className="p-2 text-right tabular-nums">{amountText(row.withdrawal)}</td></tr>)}</tbody></table></div>
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

export default function MandateResults({ value }: { value: MandateAssessment }) {
  const [candidateId, setCandidateId] = useState('')
  const candidate = value.candidates.find(c => c.id === candidateId) ?? value.candidates[0]
  const goal = candidate?.goal_check
  const central = goal?.central
  const fan = central?.annual_fan ?? []
  const currency = value.definition.currency
  return <div className="space-y-5" aria-label="目标可行性诊断结果">
    <p role="status" className={`rounded-lg p-3 text-sm leading-6 ${value.status === 'needs_revision' ? 'bg-amber-50 text-amber-900' : 'bg-slate-50 text-slate-800'}`}>{value.status === 'inputs_only'
      ? '只完成输入与资金测算：尚未使用CMA，不提供成功概率。可先保存输入版本，再建立长期假设。'
      : value.status === 'needs_revision' ? '当前设置需要复核：以下列出实际冲突或未达门槛的结果；不会自动放宽目标。'
      : '已完成当前CMA下的代表组合诊断。通过仅针对本次模型假设，不代表资金目标必然实现。'}</p>
    {value.blockers.map((reason, i) => <p key={i} className="text-sm leading-6 text-amber-800">{reason}</p>)}
    {value.funding && <FundingOverview value={value.funding} />}
    {value.cma && <p className="text-xs leading-5 text-slate-600">CMA：{value.cma.name} · 研究日 {value.cma.as_of} · 版本 {value.cma.id}。</p>}
    {!!value.candidates.length && <>
      <div className="overflow-x-auto"><table aria-label="目标诊断候选对照" className="w-full min-w-[540px] text-sm"><thead><tr>{['候选', '预期年收益', '预期年波动', ...(value.funding ? ['成功概率', '采样区间下界', '目标检查'] : ['约束检查'])].map((label, i) => <th key={label} scope="col" className={`p-2 ${i ? 'text-right' : 'text-left'}`}>{label}</th>)}</tr></thead><tbody>{value.candidates.map(c => <tr key={c.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{c.name}</th><td className="p-2 text-right tabular-nums">{percentText(c.metrics.expected_return)}</td><td className="p-2 text-right tabular-nums">{percentText(c.metrics.volatility)}</td>{value.funding && <><td className="p-2 text-right tabular-nums">{percentText(c.goal_check?.central.success_probability)}</td><td className="p-2 text-right tabular-nums">{percentText(c.goal_check?.central.probability_lower)}</td></>}<td className="p-2 text-right">{value.funding ? !c.goal_check ? '缺少诊断' : c.goal_check.within_limits ? '本次达标' : '未达标' : '通过硬约束'}</td></tr>)}</tbody></table></div>
      <Field label="查看哪个候选的详细诊断？"><select className={inputClass} value={candidate?.id ?? ''} onChange={e => setCandidateId(e.target.value)}>{value.candidates.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}</select></Field>
      {candidate && <><GoalCandidateSummary candidate={candidate} /><p className="text-xs leading-5 text-slate-600">大类权重：{Object.entries(candidate.weights).map(([name, weight]) => `${name} ${percentText(weight)}`).join('；')}。此处仅比较，不自动采用权重。</p></>}
    </>}
    {central && goal && <>
      <dl className="grid gap-4 border-t border-slate-200 pt-4 sm:grid-cols-2">
        <div><dt className="text-xs text-slate-600">期末余额中位数（{currency}）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(central.terminal_median)}</dd><p className="text-xs text-slate-600">5%至95%分位：{amountText(central.terminal_p05)} 至 {amountText(central.terminal_p95)}</p></div>
        <div><dt className="text-xs text-slate-600">期间支付不足概率</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(central.payment_failure_probability)}</dd><p className="text-xs text-slate-600">发生一次不足即记失败，后续投入不能抹除。</p></div>
        <div><dt className="text-xs text-slate-600">全路径平均期末缺口 / 未支付金额（{currency}）</dt><dd className="mt-1 text-base font-semibold tabular-nums">{amountText(central.expected_terminal_shortfall)} / {amountText(central.expected_unpaid_payments)}</dd></div>
        <div><dt className="text-xs text-slate-600">95%分位市场回撤 / 超过预警线概率</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(central.market_drawdown_p95)} / {percentText(central.drawdown_alert_probability)}</dd><p className="text-xs text-slate-600">按单位化市场增长计算，不把提款误计成投资亏损。</p></div>
      </dl>
      {!!fan.length && <div aria-label="资金余额模拟分位数曲线"><ReactECharts style={{ height: 300, width: '100%' }} notMerge option={{
        animation: false, grid: { left: 75, right: 20, top: 45, bottom: 40 }, legend: { data: ['5%分位', '中位数', '95%分位'] },
        tooltip: { trigger: 'axis', confine: true }, xAxis: { type: 'category', data: fan.map(row => `第${row.year}年`) },
        yAxis: { type: 'value', name: currency, axisLabel: { formatter: (n: number) => n.toLocaleString('zh-CN', { notation: 'compact' }) } },
        series: (['p05', 'median', 'p95'] as const).map((key, index) => ({ name: ['5%分位', '中位数', '95%分位'][index], type: 'line', smooth: false, showSymbol: false, data: fan.map(row => row[key]) })),
      }} /><p className="text-xs leading-5 text-slate-600">曲线包含计划投入和支付，是未来资金余额的模拟分位数，不是历史净值或可交易路径。</p></div>}
      {!!fan.length && <details className="border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm font-medium">逐年资金余额分位数</summary>
        <p className="mt-2 text-xs leading-5 text-slate-600">与上图使用同一份模拟结果，金额单位为{currency}；三条分位数不是同一条投资路径。</p>
        <div className="mt-3 overflow-x-auto"><table aria-label="逐年资金余额分位数" className="w-full text-sm"><thead><tr><th scope="col" className="p-2 text-left">年末</th><th scope="col" className="p-2 text-right">5%分位</th><th scope="col" className="p-2 text-right">中位数</th><th scope="col" className="p-2 text-right">95%分位</th></tr></thead><tbody>{fan.map(row => <tr key={row.year} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">第{row.year}年</th><td className="p-2 text-right tabular-nums">{amountText(row.p05)}</td><td className="p-2 text-right tabular-nums">{amountText(row.median)}</td><td className="p-2 text-right tabular-nums">{amountText(row.p95)}</td></tr>)}</tbody></table></div>
      </details>}
      <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">目标未达成时，可以比较哪些调整？</summary><div className="mt-3 space-y-3 text-sm leading-6">
        <p>保持当前组合和目标，在相同模拟路径下，达到目标概率点估计所需本金约 {amountText(central.required_initial_capital)} {currency}，较当前可投资本金需增加 {amountText(central.additional_initial_capital)} {currency}。这不包含采样区间下界保证，追加本金后应重新完整诊断。</p>
        {central.capital_gate_status === 'solved' ? <p aria-label="按采纳口径测算本金">按相同路径的95%采样区间下界达到 {percentText(goal.threshold)} 测算，所需可投资本金为 {amountText(central.gate_required_initial_capital)} {currency}，需增加 {amountText(central.gate_additional_initial_capital)} {currency}；对应成功概率 {percentText(central.gate_success_probability)}，区间下界 {percentText(central.gate_probability_lower)}。这是固定当前组合的对比；修改本金后须重新计算流动性预算和政策候选，不会自动采纳。</p>
          : central.capital_gate_status === 'insufficient_paths' ? <p className="text-amber-800" role="status">当前路径数下，即使全部模拟成功，采样区间下界也达不到门槛，因此不提供该口径的所需本金。可增加路径数重新诊断；这不能消除模型误差。</p>
          : <p className="text-xs text-slate-600">此历史诊断未计算采纳口径的所需本金；复制为新研究后可重新测算。</p>}
        <p>可投资本金增加10%：成功概率 {percentText(central.success_with_10pct_more_capital)}；期末目标降低10%：成功概率 {percentText(central.success_with_10pct_lower_target)}。期间支付不变；这些只是对比，不会改写输入。</p>
        <p>保守敏感性：{goal.conservative ? `采用较低的CMA均值，并只收到${percentText(goal.stress_contribution_ratio)}的计划投入时，成功概率为${percentText(goal.conservative.success_probability)}。` : '折扣后收益超出该模型的定义域，本次无法提供保守情景结果。'}该情景不带发生概率。</p>
      </div></details>
    </>}
    <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">模型、随机设置与证据限制</summary><div className="mt-3 space-y-2">
      {value.funding_model && <p className="text-xs leading-5 text-slate-600">模型 {value.funding_model.version}；路径 {value.funding_model.paths}；随机种子 {value.funding_model.seed}。不同候选和敏感性共用随机数。</p>}
      {value.warnings.map((text, i) => <p key={i} className="text-xs leading-5 text-slate-600">{text}</p>)}
    </div></details>
  </div>
}
