import { percentText } from '../risk-models/ResearchUI'
import type { FundingSummary, PolicyCandidate } from '../../services/strategicAllocation'
import { amountText } from './model'
import { useI18n, systemText } from '../../i18n/runtime'

export function FundingOverview({ value }: { value: FundingSummary }) {
  useI18n()
  return <div className="space-y-3" aria-label={systemText('preInvestment.mandateResults.fundingCalculationResults')}>
    <dl className="grid gap-4 sm:grid-cols-3">
      <div><dt className="text-xs text-slate-600">{systemText('preInvestment.mandateResults.investablePrincipal')}{value.currency}）</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.investable_capital)}</dd></div>
      <div><dt className="text-xs text-slate-600">{systemText('preInvestment.mandateResults.requiredConstantAnnualCompoundReturnBeforeFees')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{value.root_status === 'solved' ? percentText(value.required_effective_return) : value.root_status === 'at_lower_bound' ? systemText('preInvestment.mandateResults.atOrBelowThe99SearchFloor') : value.root_status === 'not_applicable' ? systemText('preInvestment.mandateResults.fundingSuccessConditionsNotDefined') : systemText('preInvestment.mandateResults.aboveThe500SearchCeiling')}</dd></div>
      <div><dt className="text-xs text-slate-600">{systemText('preInvestment.mandateResults.first')}{value.liquidity_months}{systemText('preInvestment.mandateResults.monthsInPortfolioLiquidityFloor')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{amountText(value.required_liquid_capital)} / {percentText(value.required_liquid_weight)}</dd></div>
    </dl>
    <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.mandateResults.nominalTerminalTarget') + " "}{amountText(value.nominal_terminal_target)}{systemText('preInvestment.mandateResults.contributionsDuringThePeriod') + " "}{amountText(value.total_contributions)}{systemText('preInvestment.mandateResults.requiredPayments') + " "}{amountText(value.total_withdrawals)}{systemText('preInvestment.mandateResults.requiredReturnIsTheConstantPreFee')}</p>
    {value.liquidity_payment_buffer !== undefined && <p className="text-sm leading-6 text-slate-700" aria-label={systemText('preInvestment.mandateResults.nearTermPaymentBuffer')}>{systemText('preInvestment.mandateResults.assumingZeroReturnsAndStressedContributionsAfter')}{value.liquidity_months}{systemText('preInvestment.mandateResults.monthsTheRemainingInPortfolioPrincipalBuffer') + " "}{amountText(value.liquidity_payment_buffer)} {value.currency}（{percentText(value.liquidity_payment_buffer_ratio)}）。{Number(value.liquidity_shortfall_capital) > 0 ? systemText('preInvestment.mandateResults.aPaymentShortfallRemains', { p0: amountText(value.liquidity_shortfall_capital), p1: value.currency }) : ''}{systemText('preInvestment.mandateResults.thisMeasuresPaymentCoverageNotMaximumTolerable')}</p>}
    <details className="border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm">{systemText('preInvestment.mandateResults.monthlyContributionsAndPaymentPlan')}</summary>
      <div className="mt-3 overflow-x-auto"><table aria-label={systemText('preInvestment.mandateResults.monthlyCashFlows')} className="w-full text-sm"><caption className="sr-only">{systemText('preInvestment.mandateResults.monthlyCashFlows')}</caption><thead><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.mandateResults.month')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.mandateResults.contribution')}</th><th scope="col" className="p-2 text-right">{systemText('preInvestment.mandateResults.requiredPayment')}</th></tr></thead><tbody>{value.monthly_cashflows.map(row => <tr key={row.month} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">{systemText('preInvestment.mandateResults.month2')}{row.month}{systemText('preInvestment.mandateResults.monthS')}</th><td className="p-2 text-right tabular-nums">{amountText(row.contribution)}</td><td className="p-2 text-right tabular-nums">{amountText(row.withdrawal)}</td></tr>)}</tbody></table></div>
    </details>
  </div>
}

export function GoalCandidateSummary({ candidate }: { candidate: PolicyCandidate }) {
  const { s } = useI18n()
  const goal = candidate.goal_check
  return <div className="space-y-2 text-sm leading-6">
    {candidate.return_check && <p className={candidate.return_check.within_limits ? 'text-slate-800' : 'text-amber-800'}>
      {s(candidate.return_check.within_limits ? 'mandate.candidateReturnPass' : 'mandate.candidateReturnFail', {
        actual: percentText(candidate.return_check.arithmetic_return), required: percentText(candidate.return_check.required_arithmetic_return) })}
    </p>}
    {goal && <>
      <p className={goal.within_limits ? 'text-slate-800' : 'text-amber-800'}>{systemText('preInvestment.mandateResults.targetSuccessProbability') + " "}{percentText(goal.central.success_probability)}{systemText('preInvestment.mandateResults.95SimulationSamplingInterval') + " "}{percentText(goal.central.probability_lower)} {" " + systemText('preInvestment.mandateResults.to') + " "}{percentText(goal.central.probability_upper)}。{goal.within_limits ? systemText('preInvestment.mandateResults.theIntervalSLowerBoundMeetsThe') : systemText('preInvestment.mandateResults.theIntervalSLowerBoundDoesNot')} {percentText(goal.threshold)} {" " + systemText('preInvestment.mandateResults.threshold')}</p>
      <p className="text-xs text-slate-600">{systemText('preInvestment.mandateResults.passAppliesOnlyToTheSelectedModel')}</p>
      <p className="text-xs leading-5 text-slate-600">{s('ltcma.fundingDistributionHint')}</p>
    </>}
    {candidate.benchmark_check && <p className="text-slate-700">{systemText('preInvestment.mandateResults.relativeTo') + " "}{candidate.benchmark_check.name}{systemText('preInvestment.mandateResults.expectedAnnualExcessReturn') + " "}{percentText(candidate.benchmark_check.expected_excess_return)}{systemText('preInvestment.mandateResults.activeRisk') + " "}{percentText(candidate.benchmark_check.tracking_error)} {" " + systemText('preInvestment.mandateResults.limit') + " "}{percentText(candidate.benchmark_check.max_tracking_error)}。</p>}
  </div>
}
