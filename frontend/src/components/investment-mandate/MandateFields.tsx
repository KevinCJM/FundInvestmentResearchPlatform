import { Button } from '../ui'
import { Field, inputClass, NumberInput } from '../risk-models/ResearchUI'
import { percentInputValue as percent, type MandateDefinition, type FundingPlan, type StrategicCatalog, type ObjectiveKind } from '../../services/strategicAllocation'
import { defaultFunding, modelMonthLabel, objectiveLabels } from './model'

type Props = { value: MandateDefinition; onChange: (patch: Partial<MandateDefinition>) => void }

export function GoalFields({ value, onChange, catalog, cutoff }: Props & { catalog: StrategicCatalog | null; cutoff: string }) {
  const kind = value.objective_kind ?? 'absolute_return'
  const plan = value.funding_plan
  const benchmark = value.benchmark
  const funding = (patch: Partial<FundingPlan>) => onChange({ funding_plan: { ...plan!, ...patch } })
  return <div className="space-y-5">
    <Field label="这笔资金以什么为成功标准？"><select className={inputClass} value={kind} onChange={e => {
      const objective = e.target.value as ObjectiveKind
      onChange({ objective_kind: objective, target_return: objective === 'absolute_return' ? NaN : 0,
        funding_plan: objective === 'funding_goal' ? defaultFunding() : null,
        benchmark: objective === 'benchmark_relative' ? { name: '', alloc_name: '', weights: {}, target_excess_return: NaN, max_tracking_error: NaN } : null })
    }}>{Object.entries(objectiveLabels).map(([id, label]) => <option key={id} value={id}>{label}</option>)}</select></Field>
    <div className="grid gap-4 sm:grid-cols-2">
      <Field label="目标名称"><input className={inputClass} maxLength={120} value={value.name} onChange={e => onChange({ name: e.target.value })} /></Field>
      <Field label="计价币种" hint="金额与CMA须使用同一币种，不自动换汇。"><select className={inputClass} value={value.currency} onChange={e => onChange({ currency: e.target.value })}>{['CNY', 'USD', 'HKD', 'EUR', 'CAD'].map(c => <option key={c}>{c}</option>)}</select></Field>
      <Field label="目标研究日"><input className={inputClass} type="date" max={cutoff} value={value.as_of} onChange={e => onChange({ as_of: e.target.value })} /></Field>
      <Field label="政策复核日期" hint="历史研究可保留，到期政策不能直接用于当前应用。"><input className={inputClass} type="date" min={value.as_of} value={value.review_date} onChange={e => onChange({ review_date: e.target.value })} /></Field>
      <Field label="投资期限（年）"><NumberInput className={inputClass} value={value.horizon_years} onValueChange={n => onChange({ horizon_years: n })} /></Field>
      {kind === 'absolute_return' && <Field label="最低预期年收益（%）" hint="CMA年度算术总收益约束，不是保底、CAGR或业绩承诺。"><NumberInput className={inputClass} value={percent(value.target_return)} onValueChange={n => onChange({ target_return: n / 100 })} /></Field>}
    </div>
    {plan && <>
      <div className="grid gap-4 border-t border-slate-200 pt-4 sm:grid-cols-2">
        <Field label={`总资金（${value.currency}）`}><NumberInput className={inputClass} value={plan.total_capital} onValueChange={n => funding({ total_capital: n })} /></Field>
        <Field label={`组合外储备（${value.currency}）`} hint="从总资金扣除一次，不参与本组合收益或支付；组合内流动性预算另在下一步设置。"><NumberInput className={inputClass} value={plan.outside_reserve} onValueChange={n => funding({ outside_reserve: n })} /></Field>
        <Field label={`期末目标金额（${value.currency}）`} hint="期间支付全部完成后，组合内仍需保有的余额，不含组合外储备。"><NumberInput className={inputClass} value={plan.terminal_target} onValueChange={n => funding({ terminal_target: n })} /></Field>
        <Field label="目标成功概率门槛（%）" hint="研究参数，可修改；不是收益保证。采纳使用模拟区间下界，而非单一点估计。"><NumberInput className={inputClass} value={percent(plan.required_probability)} onValueChange={n => funding({ required_probability: n / 100 })} /></Field>
        <Field label="金额口径"><select className={inputClass} value={plan.amount_basis} onChange={e => funding({ amount_basis: e.target.value as FundingPlan['amount_basis'] })}><option value="nominal">未来名义金额</option><option value="real">按研究日购买力</option></select></Field>
        {plan.amount_basis === 'real' && <Field label="年通胀假设（%）" hint="用于期末目标和现金流金额的购买力换算。"><NumberInput className={inputClass} value={percent(plan.inflation)} onValueChange={n => funding({ inflation: n / 100 })} /></Field>}
        <Field label="年费用扣减（%）" hint="默认0只表示未计费，请按实际成本填写；模型不推算税法，税费支出可列为现金流。"><NumberInput className={inputClass} value={percent(plan.annual_fee)} onValueChange={n => funding({ annual_fee: n / 100 })} /></Field>
      </div>
      <div className="space-y-4 border-t border-slate-200 pt-4">
        <h3 className="text-base font-semibold">期间投入与必要支付</h3>
        <p className="text-xs leading-5 text-slate-600">每年按12期等长月测算，模型期末先投入后支付；日期便于核对计划，不做逐日计息或同月结算排序。单次现金流选择相同的开始月、结束月。</p>
        {!plan.flows.length && <p className="text-sm text-slate-600">当前没有期间现金流，按初始本金和期末目标测算。</p>}
        {plan.flows.map((flow, index) => <fieldset key={index} className="min-w-0 space-y-3 border-b border-slate-200 pb-4"><legend className="text-sm font-medium">现金流 {index + 1}</legend>
          <div className="grid gap-3 sm:grid-cols-3">
            <Field label={`现金流 ${index + 1}名称`}><input className={inputClass} value={flow.name} maxLength={120} onChange={e => funding({ flows: plan.flows.map((f, i) => i === index ? { ...f, name: e.target.value } : f) })} /></Field>
            <Field label={`现金流 ${index + 1}方向`}><select className={inputClass} value={flow.kind} onChange={e => funding({ flows: plan.flows.map((f, i) => i === index ? { ...f, kind: e.target.value as typeof f.kind } : f) })}><option value="contribution">投入组合</option><option value="withdrawal">必要支付</option></select></Field>
            <Field label={`现金流 ${index + 1}金额（${value.currency}）`}><NumberInput className={inputClass} value={flow.amount} onValueChange={n => funding({ flows: plan.flows.map((f, i) => i === index ? { ...f, amount: n } : f) })} /></Field>
            {(['first_month', 'last_month'] as const).map(key => <Field key={key} label={`现金流 ${index + 1}${key === 'first_month' ? '开始月' : '结束月'}`}><select className={inputClass} value={flow[key]} onChange={e => funding({ flows: plan.flows.map((f, i) => i === index ? { ...f, [key]: Number(e.target.value) } : f) })}>{Array.from({ length: Number.isInteger(value.horizon_years) && value.horizon_years >= 1 && value.horizon_years <= 30 ? value.horizon_years * 12 : 0 }, (_, month) => <option key={month} value={month + 1}>{modelMonthLabel(value.as_of, month + 1)}</option>)}</select></Field>)}
            <Field label={`现金流 ${index + 1}频率`}><select className={inputClass} value={flow.every_months} onChange={e => funding({ flows: plan.flows.map((f, i) => i === index ? { ...f, every_months: Number(e.target.value) as 1 | 3 | 12 } : f) })}><option value={1}>每月</option><option value={3}>每季度</option><option value={12}>每年</option></select></Field>
          </div>
          <Button onClick={() => funding({ flows: plan.flows.filter((_, i) => i !== index) })}>移除现金流 {index + 1}</Button>
        </fieldset>)}
        <Button disabled={plan.flows.length >= 24} onClick={() => funding({ flows: [...plan.flows, { name: '', kind: 'withdrawal', amount: NaN, first_month: 1, last_month: 1, every_months: 1 }] })}>添加投入或支付</Button>
      </div>
    </>}
    {benchmark && <div className="space-y-4 border-t border-slate-200 pt-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label="基准名称"><input className={inputClass} value={benchmark.name} onChange={e => onChange({ benchmark: { ...benchmark, name: e.target.value } })} /></Field>
        <Field label="基准大类方案"><select className={inputClass} value={benchmark.alloc_name} onChange={e => {
          const scope = catalog?.allocations.find(a => a.alloc_name === e.target.value)
          onChange({ benchmark: { ...benchmark, alloc_name: e.target.value, weights: Object.fromEntries((scope?.assets ?? []).map(a => [a.id, NaN])) } })
        }}><option value="">选择已有大类方案</option>{catalog?.allocations.map(a => <option key={a.alloc_name} value={a.alloc_name}>{a.alloc_name}</option>)}</select></Field>
        <Field label="目标年超额收益（百分点）"><NumberInput className={inputClass} value={percent(benchmark.target_excess_return)} onValueChange={n => onChange({ benchmark: { ...benchmark, target_excess_return: n / 100 } })} /></Field>
        <Field label="相对基准主动风险上限（%）" hint="以本页基准为参照；不是TAA相对SAA的偏离预算。"><NumberInput className={inputClass} value={percent(benchmark.max_tracking_error)} onValueChange={n => onChange({ benchmark: { ...benchmark, max_tracking_error: n / 100 } })} /></Field>
        {Object.entries(benchmark.weights).map(([name, weight]) => <Field key={name} label={`${name}基准权重（%）`}><NumberInput className={inputClass} value={percent(weight)} onValueChange={n => onChange({ benchmark: { ...benchmark, weights: { ...benchmark.weights, [name]: n / 100 } } })} /></Field>)}
      </div>
      {!catalog?.allocations.length && <p className="text-sm text-amber-800">尚无可用大类方案，须先构建大类后才能定义可计算的基准。</p>}
    </div>}
  </div>
}

export function BoundaryFields({ value, onChange }: Props) {
  const plan = value.funding_plan
  const funding = (patch: Partial<FundingPlan>) => onChange({ funding_plan: { ...plan!, ...patch } })
  return <div className="space-y-5">
    <p className="text-sm leading-6 text-slate-600">先填写不可突破的预算。波动上限不等于最大亏损保证；风险偏好和预警线放在下面单独设置。</p>
    <div className="grid gap-4 sm:grid-cols-2">
      <Field label="最高预期年波动（%）"><NumberInput className={inputClass} value={percent(value.max_volatility)} onValueChange={n => onChange({ max_volatility: n / 100 })} /></Field>
      <Field label="最低流动性资产占比（%）" hint="实际使用此值与现金流测算底线中较高者；不是组合外储备。"><NumberInput className={inputClass} value={percent(value.min_liquid_weight)} onValueChange={n => onChange({ min_liquid_weight: n / 100 })} /></Field>
      <Field label="最高非流动性资产占比（%）"><NumberInput className={inputClass} value={percent(value.max_illiquid_weight)} onValueChange={n => onChange({ max_illiquid_weight: n / 100 })} /></Field>
      <Field label="TAA 主动风险上限（%）" hint="0表示只保持SAA；大于0才允许战术偏离，后续研究不可放宽。"><NumberInput className={inputClass} value={percent(value.max_tracking_error)} onValueChange={n => onChange({ max_tracking_error: n / 100 })} /></Field>
      {plan && <><Field label="流动性保障窗口（月）"><NumberInput className={inputClass} value={plan.liquidity_months} onValueChange={n => funding({ liquidity_months: n })} /></Field>
        <Field label="压力下可收到的投入比例（%）" hint="只用于压力敏感性及流动性预算；不代表该情景发生概率。"><NumberInput className={inputClass} value={percent(plan.contribution_stress_ratio)} onValueChange={n => funding({ contribution_stress_ratio: n / 100 })} /></Field></>}
    </div>
    <Field label="风险与流动性边界的依据" hint="说明必要支付、可以承受的损失和资金不可用时间，而不只是填写“高风险/低风险”。"><textarea className={inputClass} rows={3} value={value.boundary_reason ?? ''} maxLength={2000} onChange={e => onChange({ boundary_reason: e.target.value })} /></Field>
    <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">效用偏好、预警与复核规则</summary>
      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <Field label="风险厌恶系数" hint="默认5仅为模型偏好起点，不是风险测评分数；它影响效用候选，不改变硬边界。"><NumberInput className={inputClass} value={value.risk_aversion} onValueChange={n => onChange({ risk_aversion: n })} /></Field>
        {plan && <Field label="市场回撤预警线（%）" hint="诊断超线概率，不作为本金保护或交易止损保证。"><NumberInput className={inputClass} value={percent(plan.drawdown_alert)} onValueChange={n => funding({ drawdown_alert: n / 100 })} /></Field>}
        <Field label="政策再平衡方式"><select className={inputClass} value={value.rebalance_policy} onChange={e => onChange({ rebalance_policy: e.target.value as MandateDefinition['rebalance_policy'] })}><option value="monthly">每月复核</option><option value="quarterly">每季复核</option><option value="annually">每年复核</option><option value="threshold">触及阈值时复核</option></select></Field>
      </div>
      <div className="mt-4"><Field label="再平衡触发与恢复规则" hint="这是研究政策约定，不会自动交易。"><textarea className={inputClass} rows={2} value={value.rebalance_note} onChange={e => onChange({ rebalance_note: e.target.value })} /></Field></div>
    </details>
    <Field label="资金用途与其他说明"><textarea className={inputClass} rows={2} value={value.note} onChange={e => onChange({ note: e.target.value })} /></Field>
  </div>
}
