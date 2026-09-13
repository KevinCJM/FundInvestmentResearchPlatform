import type { FundingPlan, MandateDefinition, MandateStudyRequest, ObjectiveKind } from '../../services/strategicAllocation'

export const objectiveLabels: Record<ObjectiveKind, string> = {
  absolute_return: '绝对收益目标', funding_goal: '期末金额与期间支付', benchmark_relative: '相对基准目标',
}
export const defaultFunding = (): FundingPlan => ({
  total_capital: NaN, outside_reserve: 0, terminal_target: NaN, amount_basis: 'nominal',
  inflation: 0, annual_fee: 0, required_probability: .8, liquidity_months: 12,
  contribution_stress_ratio: .5, drawdown_alert: .2, flows: [],
})
export const newMandate = (day: string): MandateDefinition => ({
  name: '', as_of: day, review_date: new Date(Date.parse(day) + 180 * 86400000).toISOString().slice(0, 10),
  currency: 'CNY', horizon_years: 10, target_return: NaN, max_volatility: NaN,
  min_liquid_weight: 0, max_illiquid_weight: 0, max_tracking_error: 0, risk_aversion: 5,
  rebalance_policy: 'quarterly', rebalance_note: '', note: '', boundary_reason: '',
  objective_kind: 'absolute_return', funding_plan: null, benchmark: null, asset_limits: {}, group_limits: [],
})
const finite = (value: number, low: number, high: number) => Number.isFinite(value) && value >= low && value <= high
const integer = (value: number, low: number, high: number) => Number.isInteger(value) && finite(value, low, high)

/** Input validation only; funding, probabilities and risk are calculated by the API. */
export function mandateIssues(value: MandateDefinition, cutoff: string): [string, string] {
  const kind = value.objective_kind ?? 'absolute_return'
  let fact = ''
  if (!value.name.trim()) fact = '请填写目标名称。'
  else if (!value.as_of || !value.review_date || value.as_of > cutoff || value.review_date <= value.as_of) fact = '研究日不能超过知识截止日，复核日须晚于研究日。'
  else if (!integer(value.horizon_years, 1, 30)) fact = '投资期限须为1至30年的整数。'
  else if (kind === 'absolute_return' && !finite(value.target_return, -.5, 1)) fact = '请填写最低预期年收益，范围为−50%至100%。'
  else if (kind === 'funding_goal') {
    const f = value.funding_plan
    if (!f || !finite(f.total_capital, .01, 1e12) || !finite(f.outside_reserve, 0, f.total_capital - .01) || !finite(f.terminal_target, 0, 1e13)) fact = '请填写总资金和期末目标；组合外储备须小于总资金。'
    else if (f.terminal_target === 0 && !f.flows.some(flow => flow.kind === 'withdrawal')) fact = '期末目标为0时须至少定义一笔必要支付，不能将空目标评为成功。'
    else if (!finite(f.required_probability, .5, .99) || !finite(f.inflation, -.05, .2) || !finite(f.annual_fee, 0, .1)) fact = '请核对成功概率（50%至99%）、通胀（−5%至20%）及费用（0%至10%）。'
    else if (f.flows.some(flow => !flow.name.trim() || !finite(flow.amount, .01, 1e12) || !integer(flow.first_month, 1, value.horizon_years * 12) || !integer(flow.last_month, flow.first_month, value.horizon_years * 12))) fact = '现金流须有名称、正金额及期限内的开始/结束月；单次支付设为同一月。'
  } else if (kind === 'benchmark_relative') {
    const b = value.benchmark
    if (!b?.name.trim() || !b.alloc_name || !Object.keys(b.weights).length || Object.values(b.weights).some(w => !finite(w, 0, 1)) || Math.abs(Object.values(b.weights).reduce((sum, w) => sum + w, 0) - 1) > 1e-8) fact = '请选择基准大类方案、填写名称和合计100%的基准权重。'
    else if (!finite(b.target_excess_return, -.5, 1) || !finite(b.max_tracking_error, 0, 1)) fact = '请填写基准超额目标及相对基准的主动风险上限。'
  }
  let boundary = ''
  if (!finite(value.max_volatility, .000001, 2)) boundary = '请明确最高预期年波动，不能以默认值代替风险承受判断。'
  else if (!finite(value.max_tracking_error, 0, 1)) boundary = '请填写TAA主动风险预算；0表示不允许战术偏离。'
  else if (!finite(value.min_liquid_weight, 0, 1) || !finite(value.max_illiquid_weight, 0, 1)) boundary = '流动性比例须在0%至100%之间。'
  else if ((value.boundary_reason ?? '').trim().length < 5) boundary = '请说明风险及流动性边界的依据，至少5个字符。'
  else if (!finite(value.risk_aversion, .000001, 1000)) boundary = '风险厌恶系数须为正数；它只是效用偏好，不是风险测评分数。'
  else if (value.rebalance_policy === 'threshold' && !value.rebalance_note.trim()) boundary = '请说明阈值触发及恢复条件。'
  else if (value.funding_plan && (!integer(value.funding_plan.liquidity_months, 1, Math.min(36, value.horizon_years * 12)) || !finite(value.funding_plan.contribution_stress_ratio, 0, 1) || !finite(value.funding_plan.drawdown_alert, .000001, 1))) boundary = '请核对流动性窗口、压力投入比例和回撤预警线。'
  else if (Object.values(value.asset_limits ?? {}).some(x => !finite(x.min_weight, 0, 1) || !finite(x.max_weight, x.min_weight, 1) || !finite(x.max_abs_tilt, 0, 1))) boundary = '请核对资产上下限，最低权重不能大于最高权重。'
  else if ((value.group_limits ?? []).some(g => !g.id.trim() || !g.assets.length || new Set(g.assets).size !== g.assets.length || !finite(g.lo, 0, 1) || !finite(g.hi, g.lo, 1))
    || new Set((value.group_limits ?? []).map(g => g.id)).size !== (value.group_limits ?? []).length) boundary = '联合约束须有唯一名称和不重复成员，比例在0%至100%之间且最低不高于最高。'
  else if ((Object.keys(value.asset_limits ?? {}).length || (value.group_limits ?? []).length) && !value.allocation_scope) boundary = '资产和联合约束须绑定所属大类方案，不能仅按资产同名复用授权。'
  else if (value.benchmark && value.allocation_scope && value.benchmark.alloc_name !== value.allocation_scope) boundary = '基准与授权边界须使用同一个大类方案。'
  return [fact, boundary]
}
export function studyIssue(value: MandateStudyRequest): string {
  return !integer(value.simulation_paths, 500, 10000) || !integer(value.seed, 0, 2 ** 32 - 1) || !finite(value.uncertainty_penalty, 0, 5)
    ? '模拟路径须为500至10000，种子须为非负整数，不确定性惩罚须在0至5之间。' : ''
}
export function modelMonthLabel(day: string, month: number): string {
  const start = new Date(`${day}T00:00:00Z`)
  if (!Number.isFinite(start.getTime())) return `第${month}月`
  const lastDay = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + month + 1, 0)).getUTCDate()
  const date = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + month, Math.min(start.getUTCDate(), lastDay)))
  return `第${month}月（${date.toISOString().slice(0, 10)}）`
}
export const amountText = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('zh-CN', { maximumFractionDigits: 2 }) : '—'
