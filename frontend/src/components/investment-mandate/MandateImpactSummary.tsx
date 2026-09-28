import { percentText } from '../risk-models/ResearchUI'
import type { FundingSummary, MandateDefinition, ReturnRequirements } from '../../services/strategicAllocation'
import type { RiskDecision } from '../../services/mandateTypes'
import { useMandateText } from './text'
import { fundingReturnText } from './model'

type Row = { key: string; value: string; effect: string; pending: boolean }

/**
 * The constraint contract this objective hands to SAA and TAA, visible while it is
 * still being filled. Values are echoed from the draft or from the server decision;
 * nothing here is computed as new business truth.
 */
export default function MandateImpactSummary({ value, decision, effectiveCash, funding, returns, saved = false }: {
  value: MandateDefinition; decision?: RiskDecision; effectiveCash?: number | null; funding?: FundingSummary | null
  returns?: ReturnRequirements
  saved?: boolean
}) {
  const { t } = useMandateText()
  const cap = decision?.selected_volatility_cap ?? returns?.volatility_cap ?? value.max_volatility
  const level = decision?.selected_max_level ?? value.risk_authorization?.selected_max_level
  const kind = value.objective_kind ?? 'absolute_return'
  const cash = typeof effectiveCash === 'number' ? effectiveCash : value.min_cash_weight
  const expiry = [value.review_date, value.risk_reference_valid_until].filter(Boolean).sort()[0]
  const pending = t('pendingValue')

  const floor = value.target_return_basis === 'annual_compound' ? null : value.target_return
  const cashflowNotice = funding && kind !== 'funding_goal' ? t('compoundComparisonHelp') : ''
  const goal = kind === 'absolute_return'
    ? Number.isFinite(value.target_return) ? percentText(floor ?? value.target_return) : pending
    : kind === 'benchmark_relative'
      ? Number.isFinite(value.target_excess_return) ? t('excessOf', { value: percentText(value.target_excess_return) }) : pending
      : Number.isFinite(value.funding_target?.amount) ? `${value.funding_target!.amount.toLocaleString('zh-CN')} ${value.currency}` : pending

  const rows: Row[] = [
    { key: 'impactVolatility', value: cap == null ? pending : `${percentText(cap)}${level ? `（C${level}）` : ''}`,
      effect: t('impactVolatilityEffect'), pending: cap == null },
    { key: kind === 'funding_goal' ? 'impactFundingTarget' : kind === 'benchmark_relative' ? 'impactExcess' : value.target_return_basis === 'annual_compound' ? 'requiredCompound' : 'impactReturn',
      value: goal, effect: t(value.target_return_basis === 'annual_compound' ? 'compoundTargetHint' : kind === 'funding_goal' ? 'impactFundingTargetEffect' : kind === 'benchmark_relative' ? 'impactExcessEffect' : 'impactReturnEffect'),
      pending: goal === pending },
    { key: 'impactCash', value: percentText(cash), effect: t('impactCashEffect'), pending: !Number.isFinite(cash) },
    { key: 'impactExpiry', value: expiry ? t('until', { date: expiry }) : t('noExpiry'), effect: t('impactExpiryEffect'), pending: false },
    { key: 'impactCurrency', value: value.currency, effect: t('impactCurrencyEffect'), pending: false },
    { key: 'impactHorizon', value: t('horizonYears', { count: value.horizon_years }), effect: t('impactHorizonEffect'), pending: false },
  ]
  if (kind === 'funding_goal') rows.splice(2, 0, {
    key: 'impactFundingReturn', value: fundingReturnText(funding, saved ? undefined : t('fundingReturnPending')),
    effect: t('fundingReturnBasis'), pending: !funding,
  })

  return <section aria-label={t('impactSummary')} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
    <h2 className="text-sm font-semibold text-slate-900">{t('impactSummary')}</h2>
    <p className="mt-1 text-xs leading-5 text-slate-600">{t('impactSummaryHint')}</p>
    <dl className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{rows.map(row =>
      <div key={row.key} className="min-w-0">
        <dt className="text-xs text-slate-600">{t(row.key)}</dt>
        <dd className={`mt-0.5 text-sm font-semibold tabular-nums ${row.pending ? 'text-slate-600' : 'text-slate-900'}`}>{row.value}</dd>
        <dd className="mt-0.5 text-xs leading-5 text-slate-600">{row.effect}</dd>
      </div>)}
    </dl>
    {cashflowNotice && <p role="status" className={`mt-3 text-sm leading-6 text-slate-700`}>{cashflowNotice}</p>}
    {cashflowNotice && <p className="mt-1 text-xs leading-5 text-slate-600">{t('cashflowReturnBasis')}</p>}
  </section>
}
