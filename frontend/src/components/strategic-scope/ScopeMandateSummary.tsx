import { systemText } from '../../i18n/runtime'
import { Link } from 'react-router-dom'
import { mandateEffectiveCash, type MandateVersion } from '../../services/strategicAllocation'
import { percentText, sectionClass } from '../risk-models/ResearchUI'
import { useMandateText } from '../investment-mandate/text'
import { fundingReturnText } from '../investment-mandate/model'

const present = (value: number | null | undefined) => typeof value === 'number' && Number.isFinite(value)
const percentage = (value: number | null | undefined) => present(value) ? percentText(value!) : systemText('preInvestment.scopeMandateSummary.pendingConfirmation')


/** Read the selected saved objective; this panel never recalculates its constraints. */
export default function ScopeMandateSummary({ mandate, loading, blockedReason, backHref, onRetry, researchDay }: {
  mandate: MandateVersion | null; loading: boolean; blockedReason: string; backHref: string
  onRetry?: () => void; researchDay: string | null | undefined
}) {
  const rebalanceLabels = { monthly: systemText('preInvestment.scopeMandateSummary.monthly'), quarterly: systemText('preInvestment.scopeMandateSummary.quarterly'), annually: systemText('preInvestment.scopeMandateSummary.annually'), threshold: systemText('preInvestment.scopeMandateSummary.whenTheSpecifiedThresholdIsReached') }
  const { t } = useMandateText()
  if (!mandate) return <section aria-label={systemText('preInvestment.scopeMandateSummary.currentInvestmentObjectivesAndConstraints')} className={`${sectionClass} shadow-sm`}>
    <h2 className="text-sm font-semibold text-slate-900">{systemText('preInvestment.scopeMandateSummary.currentInvestmentObjectivesAndConstraints')}</h2>
    <p role="status" className="mt-2 text-sm text-slate-600">{blockedReason}</p>
    {!loading && <div className="mt-2 flex flex-wrap gap-4">
      <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={backHref}>{systemText('preInvestment.scopeMandateSummary.returnToObjectiveSelection')}</Link>
      {onRetry && <button type="button" className="min-h-10 text-sm font-semibold text-accent-800 underline" onClick={onRetry}>{systemText('preInvestment.scopeMandateSummary.retryLoadingObjective')}</button>}
    </div>}
  </section>

  const d = mandate.definition
  const kind = d.objective_kind ?? 'absolute_return'
  const decision = mandate.assessment?.risk_decision
  const level = d.risk_authorization?.selected_max_level ?? decision?.selected_max_level
  const effectiveCash = mandateEffectiveCash(mandate)
  const budget = d.cash_budget ?? d.funding_plan
  const target = d.funding_target ?? (d.funding_plan ? { amount: d.funding_plan.terminal_target, amount_basis: d.funding_plan.amount_basis } : null)
  const money = (amount: number | null | undefined) => present(amount) ? `${amount!.toLocaleString('zh-CN', { maximumFractionDigits: 2 })} ${d.currency}` : systemText('preInvestment.scopeMandateSummary.notSet')
  const goalLabel = kind === 'funding_goal' ? systemText('preInvestment.scopeMandateSummary.terminalFundingTarget') : kind === 'benchmark_relative' ? systemText('preInvestment.scopeMandateSummary.targetAnnualExcessReturn') : d.target_return_basis === 'annual_compound' ? t('requiredCompound') : systemText('preInvestment.scopeMandateSummary.expectedAnnualReturn')
  const goal = kind === 'funding_goal' ? `${money(target?.amount)}${target?.amount_basis === 'real' ? systemText('preInvestment.scopeMandateSummary.currentPurchasingPower') : ''}`
    : percentage(kind === 'benchmark_relative' ? d.benchmark?.target_excess_return ?? d.target_excess_return : d.target_return)
  const funding = mandate.assessment?.funding ?? mandate.funding_summary
  const fundingReturn = fundingReturnText(funding)
  const facts: Array<[string, string]> = [
    [systemText('preInvestment.scopeMandateSummary.objectiveType'), t(kind)], [systemText('preInvestment.scopeMandateSummary.objectiveResearchDate'), d.as_of], [systemText('preInvestment.scopeMandateSummary.objectiveReviewDate'), d.review_date || systemText('preInvestment.scopeMandateSummary.notSet')],
    [systemText('preInvestment.scopeMandateSummary.minimumLiquidAssetAllocation'), percentage(d.min_liquid_weight)], [systemText('preInvestment.scopeMandateSummary.maximumRestrictedLiquidityAllocation'), percentage(d.max_illiquid_weight)],
    [systemText('preInvestment.scopeMandateSummary.portfolioRebalancing'), rebalanceLabels[d.rebalance_policy]],
  ]
  if (kind === 'absolute_return' && present(d.effective_target_return)) facts.push([systemText('preInvestment.scopeMandateSummary.returnFloorForSubsequentScreening'), percentage(d.effective_target_return)])
  if (present(effectiveCash)) facts.push([systemText('preInvestment.scopeMandateSummary.enteredCashFloor'), percentage(d.min_cash_weight)])
  if (kind === 'benchmark_relative') facts.push([systemText('preInvestment.scopeMandateSummary.comparisonBenchmark'), d.benchmark?.name || d.stated_benchmark || decision?.scale_name || systemText('preInvestment.scopeMandateSummary.notSet')], [systemText('preInvestment.scopeMandateSummary.trackingErrorLimit'), percentage(d.benchmark?.max_tracking_error ?? d.max_tracking_error)])
  if (budget) facts.push([systemText('preInvestment.scopeMandateSummary.totalFunding'), money(budget.total_capital)], [systemText('preInvestment.scopeMandateSummary.reservesOutsideThePortfolio'), money(budget.outside_reserve)], [systemText('preInvestment.scopeMandateSummary.fundingBasis'), budget.amount_basis === 'real' ? systemText('preInvestment.scopeMandateSummary.currentPurchasingPower2') : systemText('preInvestment.scopeMandateSummary.nominalAmount')])
  if (d.cash_protection) facts.push([systemText('preInvestment.scopeMandateSummary.fundingProtectionRequirements'), d.cash_protection.mode === 'payments_only' ? systemText('preInvestment.scopeMandateSummary.coverInterimPayments') : systemText('preInvestment.scopeMandateSummary.coverInterimPaymentsAndRetainAtLeast', { p0: money(d.cash_protection.terminal_floor?.amount), p1: d.cash_protection.terminal_floor?.amount_basis === 'real' ? systemText('preInvestment.scopeMandateSummary.currentPurchasingPower') : '' })])
  const assetLimitCount = Object.keys(d.asset_limits ?? {}).length
  const groupLimitCount = d.group_limits?.length ?? 0

  return <section aria-label={systemText('preInvestment.scopeMandateSummary.currentInvestmentObjectivesAndConstraints')} className={`${sectionClass} min-w-0 shadow-sm`}>
    <div className="flex flex-wrap items-end justify-between gap-x-8 gap-y-3">
      <div className="min-w-0">
        <p className="text-xs font-medium text-slate-600">{systemText('preInvestment.scopeMandateSummary.currentInvestmentObjectivesAndConstraints')}</p>
        <h2 className="break-words text-base font-semibold text-slate-900">{mandate.name}</h2>
        <p className="text-sm text-slate-600 tabular-nums">{d.horizon_years} {" " + systemText('preInvestment.scopeMandateSummary.years') + " "}{d.currency}</p>
      </div>
      <dl className="flex min-w-0 flex-wrap gap-x-8 gap-y-3">
        {[[goalLabel, goal], ...(kind === 'funding_goal' ? [[systemText('preInvestment.scopeMandateSummary.fundingReturn'), fundingReturn]] : []), [systemText('preInvestment.scopeMandateSummary.riskLimit'), d.max_volatility == null ? systemText('preInvestment.scopeMandateSummary.pendingConfirmation') : systemText('preInvestment.scopeMandateSummary.annualVolatility', { p0: level ? `C${level} · ` : '', p1: percentage(d.max_volatility) })],
          [present(effectiveCash) ? systemText('preInvestment.scopeMandateSummary.cashFloor') : systemText('preInvestment.scopeMandateSummary.enteredCashFloor'), percentage(present(effectiveCash) ? effectiveCash : d.min_cash_weight)]].map(([label, value]) => <div key={label} className="min-w-0">
          <dt className="text-xs text-slate-600">{label}</dt><dd className="mt-1 break-words text-sm font-semibold text-slate-900 tabular-nums">{value}</dd>
        </div>)}
      </dl>
    </div>
    {kind === 'funding_goal' && <p className="mt-3 text-sm leading-6 text-slate-600">{systemText('preInvestment.scopeMandateSummary.fundingReturnHelp')}</p>}
    {blockedReason && <p role="status" className="mt-2 text-sm text-amber-800">{blockedReason}</p>}
    <details key={mandate.id} className="mt-2 border-t border-slate-200 pt-2">
      <summary className="min-h-10 cursor-pointer text-sm font-medium leading-10 text-slate-700">{systemText('preInvestment.scopeMandateSummary.viewObjectiveAndConstraintDetails')}</summary>
      <div className="space-y-4 pb-1">
        <dl className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{facts.map(([label, value]) => <div key={label} className="min-w-0">
          <dt className="text-xs text-slate-600">{label}</dt><dd className="mt-1 break-words text-sm text-slate-900 tabular-nums">{value}</dd>
        </div>)}</dl>
        {budget && budget.flows.length > 0 && <div><h3 className="text-sm font-medium text-slate-900">{systemText('preInvestment.scopeMandateSummary.fundingArrangements')}</h3><ul className="mt-2 space-y-2 text-sm text-slate-700">{budget.flows.map((flow, index) => <li key={index} className="break-words tabular-nums">
          {flow.name || (flow.kind === 'withdrawal' ? systemText('preInvestment.scopeMandateSummary.plannedPayments') : systemText('preInvestment.scopeMandateSummary.additionalContribution'))}：{flow.kind === 'withdrawal' ? systemText('preInvestment.scopeMandateSummary.payment') : systemText('preInvestment.scopeMandateSummary.contribution')} {money(flow.amount)}{systemText('preInvestment.scopeMandateSummary.fromMonth') + " "}{flow.first_month} {" " + systemText('preInvestment.scopeMandateSummary.to') + " "}{flow.last_month} {" " + systemText('preInvestment.scopeMandateSummary.every') + " "}{flow.every_months} {" " + systemText('preInvestment.scopeMandateSummary.monthS')}</li>)}</ul></div>}
        {(assetLimitCount > 0 || groupLimitCount > 0) && <p className="text-sm text-slate-700">{systemText('preInvestment.scopeMandateSummary.thereAreAlso') + " "}{assetLimitCount} {" " + systemText('preInvestment.scopeMandateSummary.assetAllocationLimitsAnd')}{groupLimitCount} {" " + systemText('preInvestment.scopeMandateSummary.groupLimitsViewThemInTheFull')}</p>}
        {d.note && <p className="whitespace-pre-wrap break-words text-sm text-slate-700">{systemText('preInvestment.scopeMandateSummary.notes')}{d.note}</p>}
        {d.rebalance_note && <p className="whitespace-pre-wrap break-words text-sm text-slate-700">{systemText('preInvestment.scopeMandateSummary.rebalancingNotes')}{d.rebalance_note}</p>}
        {typeof researchDay === 'string' && d.as_of !== researchDay && <p className="text-sm text-amber-800">{systemText('preInvestment.scopeMandateSummary.objectiveResearchDate') + " "}{d.as_of} {" " + systemText('preInvestment.scopeMandateSummary.andCurrentResearchDate') + " "}{researchDay} {" " + systemText('preInvestment.scopeMandateSummary.differ')}</p>}
        <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={`/pre-investment/objectives/new?view=${encodeURIComponent(mandate.id)}`}>{systemText('preInvestment.scopeMandateSummary.viewFullInvestmentObjective')}</Link>
      </div>
    </details>
  </section>
}
