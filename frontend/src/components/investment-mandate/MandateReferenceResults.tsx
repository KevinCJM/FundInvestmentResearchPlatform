import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { Badge } from '../ui'
import { percentText } from '../risk-models/ResearchUI'
import type { MandateAssessment } from '../../services/strategicAllocation'
import { useMandateText } from './text'

const usable = (point: { expected_return: number | null; volatility: number | null; status?: string; solver_status?: string }) =>
  (point.status ?? point.solver_status) === 'optimal_to_tolerance'
  && typeof point.expected_return === 'number' && Number.isFinite(point.expected_return)
  && typeof point.volatility === 'number' && Number.isFinite(point.volatility)

export default function MandateReferenceResults({ value }: { value: MandateAssessment }) {
  const { t } = useMandateText()
  const risk = value.risk_decision, result = value.reference_diagnosis
  const reference = result?.reference_frontier ?? [], constrained = result?.constrained_frontier ?? []
  const boundaries = result?.risk_boundaries ?? risk?.applied_boundaries ?? []
  const hasChart = reference.some(usable) && constrained.some(usable)
  const option = useMemo(() => ({
    animation: false,
    aria: { enabled: true, description: t('frontierChartDescription') },
    grid: { left: 16, right: 24, bottom: 48, top: 34, containLabel: true },
    tooltip: { trigger: 'item', valueFormatter: (number: number) => `${number.toFixed(2)}%` },
    legend: { data: [t('referenceFrontier'), t('constrainedFrontier')] },
    xAxis: { type: 'value', name: t('volatilityAxis'), nameLocation: 'middle', nameGap: 32, axisLabel: { formatter: '{value}%' }, scale: true },
    yAxis: { type: 'value', name: t('returnAxis'), axisLabel: { formatter: '{value}%' }, scale: true },
    series: [
      { name: t('referenceFrontier'), type: 'line', showSymbol: false, lineStyle: { type: 'dashed', width: 2 },
        data: reference.filter(usable).map(point => [point.volatility! * 100, point.expected_return! * 100]),
        markLine: { silent: true, symbol: 'none', label: { formatter: '{b}', fontSize: 12 },
          data: boundaries.map((cap, index) => ({ name: `C${index + 1}`, xAxis: cap * 100 })) } },
      { name: t('constrainedFrontier'), type: 'line', showSymbol: true, symbolSize: 5, lineStyle: { type: 'solid', width: 3 },
        data: constrained.filter(usable).map(point => [point.volatility! * 100, point.expected_return! * 100]) },
    ],
  }), [reference, constrained, boundaries, t])
  if (!risk) return null
  const validated = result?.status === 'validated'
  const selectedLevel = risk.selected_max_level ?? risk.authorized_max_level
  const cash = result?.cash_constraint, reach = result?.reachability
  return <section className="space-y-5" aria-label={t('referenceDiagnosis')}>
    <div className="flex flex-wrap items-center gap-2"><h3 className="text-lg font-semibold">{t('objectiveFeasibility')}</h3>
      <Badge tone={validated ? 'success' : result ? 'warning' : 'neutral'}>{validated ? t('feasible') : result ? t('needsRevision') : t('notCalculated')}</Badge>
    </div>
    <p className="text-sm leading-6 text-slate-600">{t('referenceScopeSimple')}</p>
    <dl className="grid gap-4 sm:grid-cols-3">
      <div><dt className="text-xs text-slate-600">{t('maxRiskLevel')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{selectedLevel ? `C${selectedLevel}` : '—'}</dd></div>
      <div><dt className="text-xs text-slate-600">{t('minimumTestedLevel')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{result?.minimum_tested_feasible_level ? `C${result.minimum_tested_feasible_level}` : t('notValidated')}</dd></div>
      <div><dt className="text-xs text-slate-600">{t('selectedCap')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(risk.selected_volatility_cap ?? risk.authorized_volatility_cap)}</dd></div>
    </dl>
    {result?.blockers.map((text, index) => <p key={index} className="text-sm leading-6 text-amber-800">{text}</p>)}

    {reach && <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <h4 className="text-sm font-semibold">{t('reachability')}</h4>
      <p className="mt-1 text-sm leading-6 text-slate-800">{t('reachableUnderCap', {
        cap: percentText(reach.volatility_cap), reachable: percentText(reach.max_return_under_cap) })}</p>
      {reach.binding === 'volatility_cap' && <p className="mt-2 text-sm leading-6 text-amber-800">{t('bindingVolatilityCap', {
        target: percentText(reach.target_return), volatility: percentText(reach.min_volatility_for_target),
        level: reach.required_risk_level ? `C${reach.required_risk_level}` : t('outsideScale') })}</p>}
      {reach.binding === 'unreachable_at_any_level' && <p className="mt-2 text-sm leading-6 text-amber-800">{t('bindingUnreachable', { target: percentText(reach.target_return) })}</p>}
      {reach.binding === 'none' && reach.target_return != null && <p className="mt-2 text-sm leading-6 text-slate-700">{t('bindingNone', { target: percentText(reach.target_return) })}</p>}
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('reachabilityBasis')}</p>
    </div>}

    {cash && <div className="border-t border-slate-200 pt-4"><h4 className="text-base font-semibold">{t('effectiveCashConstraint')}</h4>
      <dl className="mt-3 grid gap-4 sm:grid-cols-3"><div><dt className="text-xs text-slate-600">{t('requestedCash')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.requested_min_cash_weight)}</dd></div>
        <div><dt className="text-xs text-slate-600">{t('cashflowDerived')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.cashflow_derived_weight)}</dd></div>
        <div><dt className="text-xs text-slate-600">{t('effectiveCash')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.effective_min_cash_weight)}</dd></div></dl>
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('cashConstraintRule')}</p>
    </div>}

    <div className="border-t border-slate-200 pt-4"><div className="mb-3"><h4 className="text-base font-semibold">{t('twoFrontiers')}</h4><p className="mt-1 text-xs leading-5 text-slate-600">{t('fixedBandsHint')}</p></div>
      {hasChart ? <><ReactECharts option={option} notMerge style={{ height: 320, width: '100%' }} /><div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-600"><span>{t('referenceLineMeaning')}</span><span>{t('constrainedLineMeaning')}</span></div></>
        : <p className="text-sm text-slate-600">{t('frontierUnavailable')}</p>}
    </div>

    {result && <details className="border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('advancedEvidence')}</summary><div className="mt-3 space-y-3">
      <p className="text-xs leading-5 text-slate-600">{t('searchValidationEvidence', { search: result.search_seed, validation: result.validation_seed, paths: result.paths })}</p>
      <div className="overflow-x-auto"><table className="w-full min-w-[480px] text-sm" aria-label={t('referenceCandidates')}><caption className="sr-only">{t('referenceCandidates')}</caption><thead><tr>{['candidate', 'referenceReturn', 'volatility', 'level'].map(key => <th key={key} scope="col" className="p-2 text-right first:text-left">{t(key)}</th>)}</tr></thead><tbody>{result.candidates.map(candidate => <tr key={candidate.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">{candidate.id}</th><td className="p-2 text-right tabular-nums">{percentText(candidate.expected_return)}</td><td className="p-2 text-right tabular-nums">{percentText(candidate.volatility)}</td><td className="p-2 text-right">{candidate.risk_level ? `C${candidate.risk_level}` : '—'}</td></tr>)}</tbody></table></div>
      {result.limitations.map((text, index) => <p key={index} className="text-xs leading-5 text-slate-600">{text}</p>)}
    </div></details>}
  </section>
}
