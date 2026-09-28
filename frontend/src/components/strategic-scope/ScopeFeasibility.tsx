import { useEffect, useState } from 'react'
import { useI18n } from '../../i18n/runtime'
import { useResearchContextIdentity } from '../../app/ResearchContext'
import { getScopeFeasibility, type ScopeFeasibilityInput, type ScopeFeasibilityResult, type ScopeHistoryWindow } from '../../services/strategicScope'
import { Badge, Button, ErrorPanel, LoadingPanel } from '../ui'
import { Field, inputClass, percentText } from '../risk-models/ResearchUI'
import ScopeFrontierChart from './ScopeFrontierChart'

/** Read-only early check. Saving the scope never publishes these historical estimates as a CMA. */
export default function ScopeFeasibility({ input, blockedReason = '', clock }: {
  input: ScopeFeasibilityInput | null; blockedReason?: string; clock: string | null | undefined
}) {
  const { s } = useI18n()
  const researchIdentity = useResearchContextIdentity()
  const t = (key: string, values?: Record<string, string | number>) => s(`scopeFeasibility.${key}`, values)
  const [window, setWindow] = useState<ScopeHistoryWindow>('5Y')
  const [retry, setRetry] = useState(0)
  const [response, setResponse] = useState<{ key: string; result: ScopeFeasibilityResult } | null>(null)
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null)
  const query = input ? JSON.stringify({ ...input, window: { kind: window } }) : ''
  const key = JSON.stringify([query, clock, researchIdentity, retry])
  const blocked = blockedReason || (clock === undefined ? t('clockPending') : !input ? t('inputsPending') : '')
  useEffect(() => {
    if (blocked || !query) return
    const controller = new AbortController()
    const timer = setTimeout(() => {
      getScopeFeasibility(JSON.parse(query), controller.signal)
        .then(result => { if (!controller.signal.aborted) setResponse({ key, result }) })
        .catch(error => { if (!controller.signal.aborted) setFailure({ key, message: error instanceof Error ? error.message : '' }) })
    }, 500)
    return () => { clearTimeout(timer); controller.abort() }
  }, [key, blocked])
  // A changed scope, goal, date or window must hide old claims before the next request completes.
  const data = !blocked && response?.key === key ? response.result : null
  const error = !blocked && failure?.key === key ? failure.message || t('failed') : ''
  const loading = !blocked && !data && !error
  const frontier = data?.frontier
  const hasPoints = frontier?.points.some(p => p.status === 'optimal_to_tolerance' && p.expected_return != null && p.volatility != null)
  const comparison = data?.reference_comparison
  const reference = comparison?.status === 'available' ? comparison : null
  const hasReference = reference?.points.some(p => p.status === 'optimal_to_tolerance' && p.expected_return != null && p.volatility != null)
  // A solved frontier alone is not a passed check. Use the backend's separate
  // historical verdict, and retain the outstanding funding/liquidity gate.
  const pendingCheck = data?.status === 'undetermined' && data.target_check?.status === 'feasible'
    ? ({ SCOPE_FUNDING_CHECK_REQUIRED: 'fundingPending', SCOPE_PRODUCT_LIQUIDITY_REQUIRED: 'liquidityPending' } as Record<string, string>)[data.reason_code ?? '']
    : undefined
  const fundingRequirement = data?.mandate.target_return == null ? data?.mandate.funding_requirement : null
  const fundingComparison = fundingRequirement ? data?.funding_comparison : null
  const fundingStatus = fundingComparison?.status === 'passed' ? 'fundingPassed'
    : fundingComparison?.status === 'no_candidate' ? 'fundingNotReached' : null
  const requiredReturn = fundingRequirement?.required_return
  // Match step 01's display precision; a root-solver residual is not -0.00%.
  const displayedRequiredReturn = requiredReturn != null && Math.abs(requiredReturn) < 5e-5 ? 0 : requiredReturn
  const historicalStatus = data?.mandate.target_return == null ? 'riskPassed' : 'historicalPassed'
  return <section className="min-w-0 space-y-4 border-t border-slate-200 pt-5" aria-label={t('title')} aria-busy={loading}>
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="space-y-2"><h2 className="text-lg font-semibold text-slate-900">{t('title')}</h2>
        <p className="text-sm leading-6 text-slate-600">{t('intro')}</p></div>
      <Field label={t('window')}><select className={inputClass} value={window} onChange={event => setWindow(event.target.value as ScopeHistoryWindow)}>
        {(['1Y', '2Y', '3Y', '5Y', '10Y'] as const).map(kind => <option key={kind} value={kind}>{t('years', { years: kind.slice(0, -1) })}</option>)}
        <option value="common_since_inception">{t('commonWindow')}</option>
      </select></Field>
    </div>
    {blocked ? <p role="status" className="text-sm text-slate-600">{blockedReason ? t('blocked') : blocked}</p> : error ? <ErrorPanel message={`${t('failed')} ${error}`} action={<Button onClick={() => setRetry(n => n + 1)}>{t('retry')}</Button>} className="min-h-72" /> : loading ? <LoadingPanel text={t('loading')} className="min-h-72" /> : data && <>
      <div role="status" className="space-y-2" aria-live="polite">
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone={fundingStatus ? fundingStatus === 'fundingPassed' ? 'success' : 'warning' : data.status === 'feasible' || pendingCheck ? 'success' : 'warning'}>{t(fundingStatus ?? (pendingCheck ? historicalStatus : data.status))}</Badge>
          {pendingCheck === 'liquidityPending' && <Badge tone="warning">{t(pendingCheck)}</Badge>}
        </div>
        {data.reasons.filter(reason => reason.code !== 'SCOPE_FUNDING_CHECK_REQUIRED').map((reason, index) => <p key={`${reason.code}:${index}`} className={`text-sm leading-6 ${data.status === 'infeasible' ? 'text-amber-900' : 'text-slate-700'}`}>{reason.message}</p>)}
        {fundingStatus && <p className="text-sm leading-6 text-slate-700">{t(`${fundingStatus}Help`)}</p>}
        {pendingCheck === 'fundingPending' && <p className="text-sm leading-6 text-slate-600">{t('fundingNext')}</p>}
      </div>
      <dl className="grid gap-3 sm:grid-cols-3">
        <div><dt className="text-xs text-slate-600">{t(fundingRequirement ? 'fundingReturn' : 'target')}</dt>
          <dd className="mt-1 font-semibold tabular-nums">{fundingRequirement
            ? fundingRequirement.status === 'solved' && fundingRequirement.required_return != null
              ? percentText(displayedRequiredReturn) : t(`fundingReturnStatus.${fundingRequirement.status}`)
            : data.mandate.target_return == null ? t('separateTarget') : percentText(data.mandate.target_return)}</dd>
          {fundingRequirement && <dd className="mt-1 text-xs leading-5 text-slate-600">{t('fundingReturnBasis')}</dd>}
        </div>
        <div><dt className="text-xs text-slate-600">{t('risk')}</dt><dd className="mt-1 font-semibold tabular-nums">{percentText(data.mandate.volatility_cap)}</dd></div>
        <div><dt className="text-xs text-slate-600">{t('cash')}</dt><dd className="mt-1 font-semibold tabular-nums">{percentText(data.mandate.min_cash_weight)}</dd></div>
      </dl>
      {!fundingComparison && data.mandate.return_requirements?.compound_floor != null && <p className="text-sm leading-6 text-slate-600">
        {s('mandate.requiredCompound')}: {percentText(data.mandate.return_requirements.compound_floor)}. {s('policyFrontier.returnCurveHelp')}</p>}
      {hasPoints && frontier?.constraints_applied && <p className="text-sm leading-6 text-slate-600">{t('currentScope')}</p>}
      {frontier && !frontier.constraints_applied && <p className="text-sm leading-6 text-amber-900">{t('scopeOnly')}</p>}
      {(hasPoints || hasReference) && <ScopeFrontierChart points={fundingComparison?.points ?? frontier?.points ?? []}
        referencePoints={fundingComparison?.reference_points ?? reference?.points}
        constrainedPoints={fundingComparison?.constrained_points ?? reference?.constrained_points}
        compound={Boolean(fundingComparison)}
        targetCurve={!fundingComparison && data.mandate.return_requirements?.compound_floor != null ? data.mandate.target_curve : undefined}
        targetReturn={fundingComparison ? fundingComparison.target_return : data.mandate.target_return}
        volatilityCap={data.mandate.volatility_cap} candidate={fundingComparison ? fundingComparison.candidate : data.target_check?.candidate} />}
      {reference ? <div className="space-y-1 text-xs leading-5 text-slate-600">
        <p>{t('reference.source', { name: reference.name, date: reference.as_of, currency: reference.currency })}</p>
        {reference.sample_start && reference.sample_end && <p>{t('reference.sample', { start: reference.sample_start, end: reference.sample_end })}</p>}
        <p>{t('reference.meaning')}</p>
        {!reference.constrained_points.length && <p>{t('reference.noConstrained')}</p>}
        {!hasPoints && hasReference && <p className="text-amber-800">{t('reference.only')}</p>}
      </div> : <p className="text-xs leading-5 text-slate-600">{t(`reference.${comparison?.status ?? 'unavailable'}`)}</p>}
      {frontier && !frontier.complete && <p className="text-xs leading-5 text-amber-800">{t('partial')}</p>}
      {!fundingComparison && data.target_check?.max_return_under_cap != null && <p className="text-sm text-slate-700 tabular-nums">{t('maximum', { value: percentText(data.target_check.max_return_under_cap) })}</p>}
      {data.status === 'infeasible' && <p className="text-sm leading-6 text-amber-900">{t('revise')}</p>}
      {(data.additional_checks?.funding || data.additional_checks?.benchmark) && <p className="text-xs leading-5 text-slate-600">{t('additional')}</p>}
      {data.sample && <p className="text-xs leading-5 text-slate-600 tabular-nums">{t('sample', { start: data.sample.actual_start, end: data.sample.actual_end, count: data.sample.observations })}</p>}
      {data.sample && (data.sample.excluded_return_periods > 0 || data.sample.missing_trading_days > 0) && <p className="text-xs leading-5 text-amber-800 tabular-nums">{t('alignment', { dropped: data.sample.excluded_return_periods, missing: data.sample.missing_trading_days })}</p>}
    </>}
    <p className="text-xs leading-5 text-slate-600">{t(fundingComparison ? 'compoundBasis' : 'basis')}</p>
  </section>
}
