import { researchMessage } from '../../i18n/researchMessages'
import { systemText, useI18n } from '../../i18n/runtime'
import type { TaaBaseline, TaaPreflight, TaaPreview, TaaPreviewRequest } from '../../services/tacticalAllocation'
import { buttonClass, sectionClass } from '../risk-models/ResearchUI'
import { Link } from 'react-router-dom'
import { allocationJourneyPath } from '../../app/allocationJourney'
import { DataTable } from '../ui'

export default function TaaResearchContext({ baseline, request, preview, preflight, checking, error, contextIssue, onChange, onEdit, onRetry }: {
  baseline: TaaBaseline; request: TaaPreviewRequest; preview: TaaPreview | null; preflight: TaaPreflight | null
  checking: boolean; error: string; contextIssue?: string; onRetry: () => void; onChange: (patch: Partial<TaaPreviewRequest>) => void; onEdit: () => void
}) {
  useI18n()
  const names = Object.fromEntries(baseline.assets.map(asset => [asset.id, asset.name]))
  const signalTraining = preview ? preview.data.training : preflight?.training
  const alignment = !checking && !error ? (preview ? preview.data.alignment : preflight?.alignment) : null
  const blocked = preflight?.quality.status === 'blocked'
  const researchProxies = baseline.strategic_universe_id && (baseline.implementation_status === 'incomplete' || baseline.assets.some(asset => asset.research_proxy))
  const reasons = [...new Set([...(preflight?.training.reasons ?? []), ...(preflight?.pit.reasons ?? [])])]
  return <section className={`${sectionClass} space-y-3`} aria-label={systemText('preInvestment.taaResearchContext.researchConditionsAndEligibility')}>
    <div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-sm font-semibold">{systemText('preInvestment.taaResearchContext.researchConditions')}</h2><button type="button" className="min-h-9 text-sm font-medium text-accent-800 underline" onClick={onEdit}>{systemText('preInvestment.taaResearchContext.editDatesAndCosts')}</button></div>
    <p className="text-sm leading-6 text-slate-700">{systemText('preInvestment.taaResearchContext.assetClassResearch')}{baseline.assets.map(asset => asset.name).join('、')}。{researchProxies ? systemText('preInvestment.taaResearchContext.usesIndicesFundProxiesAndCashReturn') : systemText('preInvestment.taaResearchContext.usesSavedSaaAssetClassReturnsTo')}</p>
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-xs sm:grid-cols-3 lg:grid-cols-6">
      {[
        [systemText('preInvestment.taaResearchContext.saaPolicyDate'), baseline.as_of.slice(0, 10)],
        [systemText('preInvestment.taaResearchContext.marketDataThrough'), preview?.data.end_date ?? preflight?.coverage.end_date ?? systemText('preInvestment.taaResearchContext.checking')],
        [systemText('preInvestment.taaResearchContext.researchDateKnowledgeCutoff'), request.as_of],
        [systemText('preInvestment.taaResearchContext.trainingPeriod'), `${request.start_date} — ${request.train_end_date}`],
        [systemText('preInvestment.taaResearchContext.independentValidation'), systemText('preInvestment.taaResearchContext.afterThrough', { p0: request.train_end_date, p1: request.end_date })],
        [systemText('preInvestment.taaResearchContext.oneWayCosts'), systemText('preInvestment.taaResearchContext.basisPoints', { p0: request.transaction_cost_bps, p1: (request.transaction_cost_bps / 100).toFixed(2) })],
      ].map(([label, value]) => <div key={label} className="min-w-0"><dt className="text-slate-600">{label}</dt><dd className="mt-1 break-words font-medium leading-5">{value}</dd></div>)}
    </dl>
    <div role="status" className={`rounded-lg px-3 py-2 text-xs leading-5 ${blocked || error ? 'bg-rose-50 text-rose-900' : preflight?.can_calculate ? 'bg-accent-50 text-accent-900' : 'bg-amber-50 text-amber-900'}`}>
      {contextIssue ? systemText('preInvestment.taaResearchContext.theResearchDateConflictsWithPlatformPit') : checking ? systemText('preInvestment.taaResearchContext.checkingDataQualityCommonCoverageAndTraining') : error || (blocked ? systemText('preInvestment.taaResearchContext.dataQualityNeedsAttentionCurrentResultsCannot') : !preflight ? systemText('preInvestment.taaResearchContext.awaitingResearchPreflightChecks') : preflight.can_calculate ? systemText('preInvestment.taaResearchContext.trainingPeriodsValidationPeriodsHistoricalPitNot', { p0: request.search ? systemText('preInvestment.taaResearchContext.trainingPeriodStrengthSearchAvailable') : systemText('preInvestment.taaResearchContext.fixedHypothesisComparisonAvailableNoStrengthSelection'), p1: preflight.training.train_observations, p2: preflight.training.validation_observations }) : systemText('preInvestment.taaResearchContext.currentConditionsPreventCalculationResolveTheItems'))}
    </div>
    {alignment && <div className="space-y-2 text-xs leading-5 text-slate-600">
      <p>{systemText('preInvestment.taaCalendar.summary', { dates: alignment.common_observations, periods: alignment.return_periods, excluded: alignment.non_common_dates, spans: alignment.multi_observation_periods, days: alignment.max_calendar_days })}</p>
      <p>{systemText('preInvestment.taaCalendar.convention')}</p>
      <details>
        <summary className="min-h-10 cursor-pointer py-2 font-medium text-slate-700">{systemText('preInvestment.taaCalendar.details')}</summary>
        <p className="mb-3">{systemText('preInvestment.taaCalendar.unknownCalendar')}</p>
        <DataTable caption={systemText('preInvestment.taaCalendar.details')} rows={alignment.sources} rowKey={row => row.series_id} minWidth="36rem" empty={systemText('preInvestment.taaCalendar.cashOnly')} columns={[
          { header: systemText('preInvestment.taaCalendar.source'), cell: row => row.name },
          { header: systemText('preInvestment.taaCalendar.coverage'), cell: row => `${row.start_date} — ${row.end_date}` },
          { header: systemText('preInvestment.taaCalendar.absent'), cell: row => row.not_observed_dates, numeric: true },
          { header: systemText('preInvestment.taaCalendar.examples'), cell: row => row.date_examples.join(', ') || '—' },
        ]} />
      </details>
    </div>}
    {signalTraining?.train_signal_observations != null && <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.taaResearchContext.validTrendSignalsTraining') + " "}{signalTraining.train_signal_observations} / {signalTraining.train_observations} {" " + systemText('preInvestment.taaResearchContext.periodsValidation') + " "}{signalTraining.validation_signal_observations} / {signalTraining.validation_observations} {" " + systemText('preInvestment.taaResearchContext.periodsSaaIsRetainedWithoutSignalsData')}</p>}
    {preview && request.signal_mode === 'momentum' && !preview.data.training && <p className="text-xs text-slate-600">{systemText('preInvestment.taaResearchContext.thisResultHasNoValidSignalPeriod')}</p>}
    {error && <button type="button" className={buttonClass} onClick={onRetry}>{systemText('preInvestment.taaResearchContext.retryConditionChecks')}</button>}
    {error && researchProxies && <Link className={buttonClass} to={`/pre-investment/product-pool/new?scope=strategic&strategic_universe=${encodeURIComponent(baseline.strategic_universe_id!)}`}>{systemText('preInvestment.taaResearchContext.checkResearchScopeAndProxies')}</Link>}
    {preflight && (!preflight.can_calculate || preflight.training.unavailable_count > 0) && <div className="space-y-2 text-xs leading-5">
      {preflight.training.train_signal_observations === 0 && <p className="text-amber-900">{systemText('preInvestment.taaResearchContext.noValidTrendSignalsInTrainingSo')}</p>}
      {preflight.quality.issues.map((issue, index) => <p key={`${issue.asset_id}-${issue.date}-${index}`} className="text-rose-900">{names[issue.asset_id] ?? issue.asset_id} · {issue.date}：{researchMessage(issue.message)}</p>)}
      {preflight.training.unavailable_count > 0 && <p className="text-amber-900">{systemText('preInvestment.taaResearchContext.notAvailableByTheTrainingCutoff') + " "}{preflight.training.unavailable_count} {" " + systemText('preInvestment.taaResearchContext.availabilityTimeUnknown') + " "}{preflight.training.unknown_count} {" " + systemText('preInvestment.taaResearchContext.items')}{preflight.training.earliest_available_date ? systemText('preInvestment.taaResearchContext.earliestAvailableDate', { p0: preflight.training.earliest_available_date }) : ''}</p>}
      {preflight.guidance.map(item => <div key={item.code} className="flex flex-wrap items-center gap-2"><p className="min-w-0 flex-1 basis-64">{researchMessage(item.message)}</p>{item.action === 'review_data' ? <Link className={buttonClass} to={allocationJourneyPath('classes')}>{systemText('preInvestment.taaResearchContext.returnToAssetClassesToCheckProducts')}</Link> : <button type="button" className={buttonClass} onClick={() => onChange(item.action === 'fixed_comparison' ? { search: false } : item.patch ?? preflight.dates)}>{item.action === 'fixed_comparison' ? systemText('preInvestment.taaResearchContext.switchToFixedHypothesisComparison') : systemText('preInvestment.taaResearchContext.useSuggestedDates')}</button>}</div>)}
    </div>}
    <details className="text-xs leading-5 text-slate-600"><summary className="cursor-pointer">{systemText('preInvestment.taaResearchContext.dateMeaningsInheritedScopeAndEvidenceLimits')}</summary><div className="mt-2 space-y-1"><p>{systemText('preInvestment.taaResearchContext.inheritsSaaWeightsAssetScopeAndConstraints')}</p><p>{systemText('preInvestment.taaResearchContext.theResearchDateLimitsWhatCanBe')}</p>{reasons.map(reason => <p key={reason}>{researchMessage(reason)}</p>)}</div></details>
  </section>
}
