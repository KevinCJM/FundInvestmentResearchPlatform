import { institutionalIssue } from '../../services/institutionalContext'
import { type CashBudget, type FundingPlan, type MandateDefinition, type MandateStudyRequest, type ObjectiveKind } from '../../services/strategicAllocation'

/** Marks a benchmark the researcher weighted by hand, so reloading an objective keeps it. */
export const CUSTOM_BENCHMARK = '自定义加权基准'

export const objectiveLabels: Record<ObjectiveKind, string> = {
  absolute_return: '绝对收益目标', funding_goal: '期末金额与期间支付', benchmark_relative: '相对基准目标',
}

export const defaultFunding = (): FundingPlan => ({
  total_capital: NaN, outside_reserve: 0, terminal_target: NaN, amount_basis: 'nominal',
  inflation: 0, annual_fee: 0, required_probability: .8, liquidity_months: 12,
  contribution_stress_ratio: .5, drawdown_alert: .2, flows: [],
})

/** Legacy 1.0 constructor retained for existing saved-workflow compatibility. */
export const newMandate = (day: string): MandateDefinition => ({
  name: '', as_of: day, review_date: new Date(Date.parse(day) + 180 * 86400000).toISOString().slice(0, 10),
  currency: 'CNY', horizon_years: 10, target_return: NaN, target_excess_return: 0, min_cash_weight: 0,
  max_volatility: NaN, min_liquid_weight: 0, max_illiquid_weight: 0, max_tracking_error: 0, risk_aversion: 5,
  rebalance_policy: 'quarterly', rebalance_note: '', note: '', boundary_reason: '',
  objective_kind: 'absolute_return', funding_plan: null, benchmark: null, stated_benchmark: '', asset_limits: {}, group_limits: [],
})

export const newCashBudget = (day: string): CashBudget => ({
  total_capital: NaN, outside_reserve: 0, balance_as_of: day, source: 'investment_objectives_cash_plan',
  amount_basis: 'nominal', inflation: 0, annual_fee: 0, flows: [],
})

/** New compact objective contract: downstream-only settings stay neutral and hidden. */
export const newBoundaryMandate = (day: string): MandateDefinition => ({
  schema_version: '2.0', name: '', as_of: day, review_date: null, currency: 'CNY', horizon_years: 10,
  target_return: NaN, target_excess_return: 0, min_cash_weight: 0, max_volatility: null,
  min_liquid_weight: 0, max_illiquid_weight: 1, max_tracking_error: 1, risk_aversion: 5,
  rebalance_policy: 'quarterly', rebalance_note: '', note: '', boundary_reason: '',
  objective_kind: 'absolute_return', funding_plan: null, cash_budget: null, funding_target: null,
  cash_protection: null, boundary_policy: null, benchmark: null, stated_benchmark: '', institutional_context: null,
  strategic_universe_id: null, allocation_scope: null, asset_limits: {}, group_limits: [],
  risk_authorization: { mode: 'manual_level', source: 'risk_scale_selection', risk_scale_ref: null,
    authorized_max_level: null, selected_max_level: null },
})

/** Convert editable legacy/new drafts into the compact objective contract without discarding funding facts. */
export function compactMandateDefinition(input: Partial<MandateDefinition> | null | undefined, fallbackDay: string): MandateDefinition {
  if (!input) return newBoundaryMandate(fallbackDay)
  const day = typeof input.as_of === 'string' && input.as_of ? input.as_of : fallbackDay
  const base = newBoundaryMandate(day)
  const kind = input.objective_kind ?? 'absolute_return'
  const oldLevel = input.risk_authorization?.selected_max_level ?? input.risk_authorization?.authorized_max_level ?? null
  const oldRef = input.risk_authorization?.risk_scale_ref ?? null
  const oldCash = input.cash_budget
  const oldPlan = input.funding_plan
  const migratedCash: CashBudget | null = oldCash ? { ...newCashBudget(day), ...oldCash, balance_as_of: day }
    : oldPlan ? { ...newCashBudget(day), total_capital: oldPlan.total_capital, outside_reserve: oldPlan.outside_reserve,
      amount_basis: oldPlan.amount_basis, inflation: oldPlan.inflation, annual_fee: oldPlan.annual_fee,
      flows: oldPlan.flows, source: 'legacy_funding_plan' } : null
  return {
    ...base,
    name: input.name ?? '', as_of: day, review_date: input.review_date ?? null,
    currency: input.currency ?? base.currency, horizon_years: input.horizon_years ?? base.horizon_years,
    objective_kind: kind,
    target_return: kind === 'absolute_return' ? input.target_return ?? NaN : 0,
    target_excess_return: kind === 'benchmark_relative' ? input.target_excess_return ?? input.benchmark?.target_excess_return ?? NaN : 0,
    min_cash_weight: input.min_cash_weight ?? input.boundary_policy?.cash_reserve_weight ?? 0,
    cash_budget: migratedCash ?? (kind === 'funding_goal' ? newCashBudget(day) : null),
    funding_target: kind === 'funding_goal' ? input.funding_target ?? (oldPlan ? { amount: oldPlan.terminal_target, amount_basis: oldPlan.amount_basis } : { amount: NaN, amount_basis: 'nominal' }) : null,
    cash_protection: kind !== 'funding_goal' && migratedCash ? input.cash_protection ?? null : null,
    risk_authorization: { mode: 'manual_level', source: 'risk_scale_selection', risk_scale_ref: oldRef,
      authorized_max_level: oldRef ? oldLevel : null, selected_max_level: oldRef ? oldLevel : null },
    // Auto-frozen benchmarks are re-derived server-side; only hand-weighted ones are carried back.
    benchmark: kind === 'benchmark_relative' && input.benchmark?.name === CUSTOM_BENCHMARK ? input.benchmark : null,
    stated_benchmark: kind === 'benchmark_relative' ? input.stated_benchmark ?? '' : '',
  }
}

const finite = (value: unknown, low: number, high: number): value is number => typeof value === 'number' && Number.isFinite(value) && value >= low && value <= high
const integer = (value: unknown, low: number, high: number): value is number => Number.isInteger(value) && finite(value, low, high)

function legacyMandateStepIssues(value: MandateDefinition, cutoff: string): [string, string, string] {
  const kind = value.objective_kind ?? 'absolute_return', plan = value.funding_plan
  let task = '', goal = '', boundary = ''
  if (!value.name.trim()) task = '请填写目标名称。'
  else if (!value.as_of || !value.review_date || value.as_of > cutoff || value.review_date <= value.as_of) task = '研究日不能超过知识截止日，复核日须晚于研究日。'
  else if (!integer(value.horizon_years, 1, 30)) task = '投资期限须为1至30年的整数。'
  else if (!/^[A-Z]{3}$/.test(value.currency)) task = '请明确计价币种。'
  if (kind === 'absolute_return' && !finite(value.target_return, -.5, 1)) goal = '请填写最低预期年收益，范围为−50%至100%。'
  else if (kind === 'funding_goal' && (!plan || !finite(plan.total_capital, .01, 1e12) || !finite(plan.terminal_target, 0, 1e13))) goal = '请填写总资金和期末目标。'
  else if (kind === 'benchmark_relative') {
    const benchmark = value.benchmark
    if (!benchmark?.name.trim() || !benchmark.alloc_name || !Object.keys(benchmark.weights).length
      || Object.values(benchmark.weights).some(weight => !finite(weight, 0, 1))
      || Math.abs(Object.values(benchmark.weights).reduce((sum, weight) => sum + weight, 0) - 1) > 1e-8) goal = '请选择基准大类方案、填写名称和合计100%的基准权重。'
  }
  if (!finite(value.max_volatility, .000001, 2)) boundary = '请明确最高预期年波动。'
  else if (!finite(value.max_tracking_error, 0, 1)) boundary = '请填写TAA主动风险预算。'
  else if (!finite(value.min_liquid_weight, 0, 1) || !finite(value.max_illiquid_weight, 0, 1)) boundary = '流动性比例须在0%至100%之间。'
  else if ((value.boundary_reason ?? '').trim().length < 5) boundary = '请说明风险及流动性边界的依据，至少5个字符。'
  else if (!finite(value.risk_aversion, .000001, 1000)) boundary = '风险厌恶系数须为正数。'
  return [task || institutionalIssue(value.institutional_context, value.as_of, value.currency), goal, boundary]
}

/** Only validate choices the human actually makes on the compact objective page. */
export function mandateStepIssues(value: MandateDefinition, cutoff: string): [string, string, string] {
  if (value.schema_version !== '2.0') return legacyMandateStepIssues(value, cutoff)
  const kind = value.objective_kind ?? 'absolute_return'
  let basic = '', goal = '', riskCash = ''
  if (!value.name.trim()) basic = '请填写目标名称。'
  else if (!value.as_of || value.as_of > cutoff) basic = '目标研究日不能超过当前研究口径。'
  else if (value.review_date && value.review_date <= value.as_of) basic = '复核日期如填写，须晚于目标研究日。'
  else if (!integer(value.horizon_years, 1, 30)) basic = '投资期限须为1至30年的整数。'

  if (kind === 'absolute_return' && !finite(value.target_return, -.5, 1)) goal = '请填写最低预期年收益，范围为−50%至100%。'
  else if (kind === 'benchmark_relative' && !finite(value.target_excess_return, -.5, 1)) {
    goal = '请填写相对所选风险等级参考组合的目标年超额收益。'
  }

  const risk = value.risk_authorization
  if (!risk?.risk_scale_ref) riskCash = '请选择当前研究日可用的风险等级配置。'
  else if (!integer(risk.authorized_max_level, 1, 5) || !integer(risk.selected_max_level, 1, 5)
    || risk.authorized_max_level !== risk.selected_max_level) riskCash = '请选择本次最大可承受风险等级 C1–C5。'
  else if (!finite(value.min_cash_weight, 0, 1)) riskCash = '最低现金占比须在0%至100%之间。'
  else if (kind === 'funding_goal' && !finite(value.funding_target?.amount, 0, 1e13)) riskCash = '请填写期末目标金额。'
  else if (value.benchmark && Math.abs(Object.values(value.benchmark.weights).reduce((sum, w) => sum + w, 0) - 1) > 1e-8) {
    riskCash = '自定义基准的大类权重合计须为100%。'
  }
  else if (value.cash_budget) {
    const cash = value.cash_budget
    if (!finite(cash.total_capital, .01, 1e12)) riskCash = '请填写组合总资金。'
    else if (!finite(cash.outside_reserve, 0, 1e12) || cash.outside_reserve >= cash.total_capital) riskCash = '组合外储备须小于总资金。'
    else if (!finite(cash.inflation, -.05, .20)) riskCash = '年通胀假设须在−5%至20%之间。'
    else if (!finite(cash.annual_fee, 0, .10)) riskCash = '额外年费用须在0%至10%之间。'
    else if (cash.balance_as_of !== value.as_of) riskCash = '现金预算日期须与目标研究日一致。'
    else if (cash.flows.some(flow => !flow.name.trim() || !finite(flow.amount, .01, 1e12)
      || !integer(flow.first_month, 1, value.horizon_years * 12)
      || !integer(flow.last_month, flow.first_month, value.horizon_years * 12)
      || ![1, 3, 12].includes(flow.every_months))) riskCash = '现金流须有名称、正金额及期限内的开始/结束月份。'
  }
  return [basic, goal, riskCash]
}

/** Existing callers use the two-section validation contract. */
export function mandateIssues(value: MandateDefinition, cutoff: string): [string, string] {
  const [basic, goal, boundary] = mandateStepIssues(value, cutoff)
  return [basic || goal, boundary]
}

/** Hidden model settings remain fixed and validated, but are no longer normal-user fields. */
export function studyIssue(value: MandateStudyRequest): string {
  if (!integer(value.simulation_paths, 500, 10000) || !integer(value.seed, 0, 2 ** 32 - 1)
    || !finite(value.uncertainty_penalty, 0, 5)) return '内部诊断参数无效，请新建目标后重试。'
  if (value.definition.schema_version === '2.0' && (!integer(value.validation_seed, 0, 2 ** 32 - 1)
    || value.validation_seed === value.seed)) return '内部独立验证参数无效，请新建目标后重试。'
  return ''
}

/** 冻结参考组合的收益够不够填写的目标；null 表示无法比较。 */
export const levelReaches = (reference: number | null | undefined, target: number | null) =>
  target == null || typeof reference !== 'number' || !Number.isFinite(reference) ? null : reference >= target

/** 最后一次真正发生的支付月：开始月 + k×频率，不越过投资期限。 */
export const lastPaymentMonth = (first: number, everyMonths: number, horizonMonths: number) =>
  first + Math.max(0, Math.floor((horizonMonths - first) / everyMonths)) * everyMonths

export function modelMonthLabel(day: string, month: number): string {
  const start = new Date(`${day}T00:00:00Z`)
  if (!Number.isFinite(start.getTime())) return `第${month}月`
  const lastDay = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + month + 1, 0)).getUTCDate()
  const date = new Date(Date.UTC(start.getUTCFullYear(), start.getUTCMonth() + month, Math.min(start.getUTCDate(), lastDay)))
  return `第${month}月（${date.toISOString().slice(0, 10)}）`
}

export const amountText = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value)
  ? value.toLocaleString('zh-CN', { maximumFractionDigits: 2 }) : '—'
