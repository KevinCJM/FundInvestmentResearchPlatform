import { systemText, useI18n } from '../../i18n/runtime'
import { useState } from 'react'
import type { TaaDatedSignal, TaaPreviewRequest, TaaSignalComponent } from '../../services/tacticalAllocation'
import { Button } from '../ui'
import { Field, inputClass, NumberInput, sectionClass } from '../risk-models/ResearchUI'

export const scheduledPolicy = { mode: 'scheduled', cost_basis: 'half_turnover', decision_frequency: 'monthly', execution_frequency: 'daily', execution_lag: 1, min_holding_periods: 0, deviation_threshold: 0 } as const



export function policySignalIssue(request: TaaPreviewRequest, assets: string[]): string {
  const policy = request.decision_policy
  if (policy && (![policy.execution_lag, policy.min_holding_periods, policy.deviation_threshold].every(Number.isFinite)
    || !Number.isInteger(policy.execution_lag) || policy.execution_lag < 1 || policy.execution_lag > 250
    || !Number.isInteger(policy.min_holding_periods) || policy.min_holding_periods < 0 || policy.min_holding_periods > 2500
    || policy.deviation_threshold < 0 || policy.deviation_threshold > 1)) return systemText('preInvestment.taaPolicySignals.enterValidExecutionLagMinimumHoldingPeriod')
  if (request.signal_mode !== 'composite') return ''
  const components = request.signal_components ?? []
  if (!Array.isArray(components) || !components.length) return systemText('preInvestment.taaPolicySignals.addAtLeastOneSignalComponentWith')
  if (components.some(c => !c || typeof c.source !== 'string' || typeof c.methodology !== 'string' || !Array.isArray(c.observations))) return systemText('preInvestment.taaPolicySignals.theSignalDefinitionIsIncompleteAddThe')
  if (components.some(c => !Number.isFinite(c.weight) || c.weight < 0 || c.weight > 1)
    || Math.abs(components.reduce((sum, c) => sum + c.weight, 0) - 1) > 1e-8) return systemText('preInvestment.taaPolicySignals.signalWeightsMustBeCompleteAndTotal')
  if (components.some(c => !c.source.trim() || !c.methodology.trim())) return systemText('preInvestment.taaPolicySignals.enterASourceAndResearchNormalizationMethod')
  for (const c of components) {
    if (!Number.isInteger(c.lookback) || c.lookback < 2 || c.lookback > 1000 || !Number.isInteger(c.max_age_days) || c.max_age_days < 1 || c.max_age_days > 3650) return systemText('preInvestment.taaPolicySignals.enterAValidObservationWindowAndSignal')
    if (c.kind === 'momentum') continue
    if (!c.observations.length) return systemText('preInvestment.taaPolicySignals.externalSignalsRequireDatedResearchScoresNav')
    if (c.observations.some(r => !r || !r.values || !r.observed_on || !r.available_on || !r.expires_on
      || r.observed_on > r.available_on || r.available_on > r.expires_on || Object.keys(r.values).length !== assets.length
      || assets.some(a => typeof r.values[a] !== 'number' || !Number.isFinite(r.values[a]) || Math.abs(r.values[a]) > 1))) return systemText('preInvestment.taaPolicySignals.externalSignalsMustCoverTheFullAsset')
  }
  return ''
}

function DatedInput({ component, update }: { component: TaaSignalComponent; update: (patch: Partial<TaaSignalComponent>) => void }) {
  useI18n()
  const [raw, setRaw] = useState(() => component.observations.length ? JSON.stringify(component.observations, null, 2) : '')
  const [error, setError] = useState('')
  return <Field label={systemText('preInvestment.taaPolicySignals.datedScoresJsonArray')} hint={systemText('preInvestment.taaPolicySignals.eachRowContainsObservedOnAvailableOn')}>
    <textarea aria-label={systemText('preInvestment.taaPolicySignals.datedScoresJsonArray')} className={`${inputClass} font-mono placeholder:text-slate-600 placeholder:opacity-100`} rows={5} value={raw} onChange={event => {
      setRaw(event.target.value)
      try {
        const rows: unknown = JSON.parse(event.target.value)
        if (!Array.isArray(rows)) throw new Error(systemText('preInvestment.taaPolicySignals.aJsonArrayIsRequired'))
        update({ observations: rows as TaaDatedSignal[] }); setError('')
      } catch { update({ observations: [] }); setError(systemText('preInvestment.taaPolicySignals.jsonIsIncompleteCompleteItBeforeCalculating')) }
    }} />
    {error && <p role="alert" className="text-sm text-rose-800">{error}</p>}
  </Field>
}

export function TaaPolicySignals({ request, assets, update }: { request: TaaPreviewRequest; assets: string[]; update: (patch: Partial<TaaPreviewRequest>) => void }) {
  const frequencies = [['daily', systemText('preInvestment.taaPolicySignals.daily')], ['weekly', systemText('preInvestment.taaPolicySignals.weekly')], ['monthly', systemText('preInvestment.taaPolicySignals.monthly')], ['quarterly', systemText('preInvestment.taaPolicySignals.quarterly')]] as const
  const kinds = [['momentum', systemText('preInvestment.taaPolicySignals.causalMomentumWindow')], ['value', systemText('preInvestment.taaPolicySignals.valueResearchInput')], ['carry', systemText('preInvestment.taaPolicySignals.carryResearchInput')], ['macro', systemText('preInvestment.taaPolicySignals.macroResearchInput')], ['risk_sentiment', systemText('preInvestment.taaPolicySignals.riskSentimentResearchInput')]] as const
  useI18n()
  const policy = request.decision_policy
  const components = request.signal_components ?? []
  const change = (index: number, patch: Partial<TaaSignalComponent>) => update({ signal_components: components.map((c, i) => i === index ? { ...c, ...patch } : c) })
  return <section className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.taaPolicySignals.decisionAndExecutionRules')}>
    <div><h2 className="text-lg font-semibold">{systemText('preInvestment.taaPolicySignals.whenToDecideAndWhenToAdjust')}</h2><p className="mt-1 text-sm text-slate-600">{systemText('preInvestment.taaPolicySignals.decisionsUpdateTargetsSimulatedTradesOccurOnly')}</p></div>
    <Field label={systemText('preInvestment.taaPolicySignals.rebalancingConvention')}><select className={inputClass} value={policy ? 'scheduled' : 'daily_target'} onChange={e => update({ decision_policy: e.target.value === 'scheduled' ? { ...scheduledPolicy } : null })}>
      <option value="scheduled">{systemText('preInvestment.taaPolicySignals.separateDecisionAndExecution')}</option><option value="daily_target">{systemText('preInvestment.taaPolicySignals.originalConventionRestoreDailyTargets')}</option>
    </select></Field>
    {!policy && <p className="text-sm text-slate-600">{systemText('preInvestment.taaPolicySignals.retainsTheOriginalDailyTargetAndCost')}</p>}
    {policy && <><div className="grid gap-4 sm:grid-cols-2">
      <Field label={systemText('preInvestment.taaPolicySignals.decisionFrequency')}><select className={inputClass} value={policy.decision_frequency} onChange={e => update({ decision_policy: { ...policy, decision_frequency: e.target.value as typeof policy.decision_frequency } })}>{frequencies.map(([v, text]) => <option key={v} value={v}>{text}</option>)}</select></Field>
      <Field label={systemText('preInvestment.taaPolicySignals.executionOpportunities')}><select className={inputClass} value={policy.execution_frequency} onChange={e => update({ decision_policy: { ...policy, execution_frequency: e.target.value as typeof policy.execution_frequency } })}>{frequencies.map(([v, text]) => <option key={v} value={v}>{text}</option>)}</select></Field>
    </div><p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.taaPolicySignals.weeklyMonthlyAndQuarterlySchedulesUseThe')}</p>
      <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">{systemText('preInvestment.taaPolicySignals.executionLagThresholdsAndActualHoldingsDates')}</summary><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label={systemText('preInvestment.taaPolicySignals.costMeasurementConvention')}><select className={inputClass} value={policy.cost_basis ?? 'half_turnover'} onChange={e => update({ decision_policy: { ...policy, cost_basis: e.target.value as typeof policy.cost_basis } })}><option value="half_turnover">{systemText('preInvestment.taaPolicySignals.originalHalfTheAbsoluteWeightChange')}</option><option value="gross_traded_weight">{systemText('preInvestment.taaPolicySignals.twoSidedTurnoverBuyPlusSellWeights')}</option></select></Field>
        <Field label={systemText('preInvestment.taaPolicySignals.executionLagCommonObservationPeriods')}><NumberInput className={inputClass} min={1} max={250} value={policy.execution_lag} onValueChange={v => update({ decision_policy: { ...policy, execution_lag: v } })} /></Field>
        <Field label={systemText('preInvestment.taaPolicySignals.minimumHoldingPeriodCommonObservationPeriods')}><NumberInput className={inputClass} min={0} max={2500} value={policy.min_holding_periods} onValueChange={v => update({ decision_policy: { ...policy, min_holding_periods: v } })} /></Field>
        <Field label={systemText('preInvestment.taaPolicySignals.perAssetDeviationThresholdPercentagePoints')} hint={systemText('preInvestment.taaPolicySignals.adjustWhenAnyAssetReachesTheThreshold')}><NumberInput className={inputClass} min={0} max={100} aria-label={systemText('preInvestment.taaPolicySignals.perAssetDeviationThresholdPercentagePoints')} value={policy.deviation_threshold * 100} onValueChange={v => update({ decision_policy: { ...policy, deviation_threshold: v / 100 } })} /></Field>
        <Field label={systemText('preInvestment.taaPolicySignals.actualHoldingsSnapshotDate')} hint={systemText('preInvestment.taaPolicySignals.enterWeightsUnderReferenceHoldingsBelowThe')}><input className={inputClass} type="date" value={request.current_weights_as_of ?? ''} onChange={e => update({ current_weights_as_of: e.target.value || null })} /></Field>
        <Field label={systemText('preInvestment.taaPolicySignals.mostRecentActualExecutionDate')} hint={systemText('preInvestment.taaPolicySignals.requiredWhenAMinimumHoldingPeriodIs')}><input className={inputClass} type="date" value={request.last_execution_date ?? ''} onChange={e => update({ last_execution_date: e.target.value || null })} /></Field>
      </div></details><p role="status" className="text-sm text-amber-800">{systemText('preInvestment.taaPolicySignals.withoutActualHoldingsOnTheResearchDate')}</p></>}
    {request.signal_mode === 'composite' && <div className="space-y-4 border-t border-slate-100 pt-4"><h3 className="text-lg font-semibold">{systemText('preInvestment.taaPolicySignals.combineSignalsWithSources')}</h3><p className="break-words text-sm text-slate-600">{systemText('preInvestment.taaPolicySignals.fullAssetAxis')}{assets.join('、')}{systemText('preInvestment.taaPolicySignals.aMissingUnavailableOrExpiredNonzeroWeight')}</p>
      <div className="divide-y divide-slate-100">{components.map((c, i) => <div key={c.id} className="space-y-3 py-4">
        <div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-sm font-semibold">{systemText('preInvestment.taaPolicySignals.signal') + " "}{i + 1}</h4><Button tone="secondary" onClick={() => update({ signal_components: components.filter((_, k) => k !== i) })}>{systemText('preInvestment.taaPolicySignals.removeThisSignal')}</Button></div>
        <div className="grid gap-3 sm:grid-cols-2"><Field label={systemText('preInvestment.taaPolicySignals.signalSourceType', { p0: i + 1 })}><select className={inputClass} value={c.kind} onChange={e => change(i, { kind: e.target.value as TaaSignalComponent['kind'], observations: [], source: '', methodology: '' })}>{kinds.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></Field>
          <Field label={systemText('preInvestment.taaPolicySignals.signalWeight', { p0: i + 1 })}><NumberInput className={inputClass} min={0} max={100} value={c.weight * 100} onValueChange={v => change(i, { weight: v / 100 })} /></Field>
          {c.kind === 'momentum' && <Field label={systemText('preInvestment.taaPolicySignals.signalMomentumWindow', { p0: i + 1 })}><NumberInput className={inputClass} value={c.lookback} min={2} max={1000} onValueChange={v => change(i, { lookback: v })} /></Field>}
          <Field label={systemText('preInvestment.taaPolicySignals.signalMaximumObservationAgeDays', { p0: i + 1 })}><NumberInput className={inputClass} value={c.max_age_days} min={1} max={3650} onValueChange={v => change(i, { max_age_days: v })} /></Field>
          <Field label={systemText('preInvestment.taaPolicySignals.signalResearchSource', { p0: i + 1 })}><input className={inputClass} value={c.source} onChange={e => change(i, { source: e.target.value })} /></Field>
          <Field label={systemText('preInvestment.taaPolicySignals.signalNormalizationMethodAndRationale', { p0: i + 1 })}><input className={inputClass} value={c.methodology} onChange={e => change(i, { methodology: e.target.value })} /></Field>
        </div>{c.kind !== 'momentum' && <DatedInput key={c.kind} component={c} update={patch => change(i, patch)} />}
      </div>)}</div>
      <Button tone="secondary" disabled={components.length >= 8} onClick={() => update({ signal_components: [...components, { id: `signal-${Date.now()}`, kind: 'momentum', weight: components.length ? 0 : 1, source: systemText('preInvestment.taaPolicySignals.frozenNavWindowForThisStudy'), methodology: systemText('preInvestment.taaPolicySignals.centerWindowReturnsAndNormalizeByMaximum'), unit: 'standardized_score_minus1_plus1', lookback: 60, max_age_days: 31, observations: [] }] })}>{systemText('preInvestment.taaPolicySignals.addSignalComponent')}</Button>
      {components.length >= 8 && <p className="text-xs text-slate-600">{systemText('preInvestment.taaPolicySignals.upTo8ComponentsCanBeCombined')}</p>}
    </div>}
  </section>
}
