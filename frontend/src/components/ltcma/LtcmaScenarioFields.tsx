import { useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Field } from '../risk-models/ResearchUI'
import { Skeleton } from '../ui'
import { isScenarioCma, type ScenarioCmaRequest } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import type { LtcmaOptions, ScenarioHistoryOption, ScenarioReference } from '../../services/ltcma'
import { historicalScenarioPath, LtcmaProxyFields } from './LtcmaStatisticsFields'
import { control, linkClass, useLtcmaText } from './shared'
import LtcmaScenarioSelect from './LtcmaScenarioSelect'

const sameReference = (left?: ScenarioReference | null, right?: ScenarioReference | null) => Boolean(left && right
  && left.run_id === right.run_id && left.publication_id === right.publication_id && left.content_hash === right.content_hash)

export function matchingRealtime(options: LtcmaOptions, model: ScenarioCmaRequest) {
  return (options.scenario_options?.realtime_runs ?? []).filter(item => sameReference(item.reference, model.historical_reference))
}

/** Eligibility is supplied by the server; names never establish a scenario binding. */
export function scenarioSelectionIssue(value: CmaDraft, options: LtcmaOptions | null, text: (key: string) => string): string | null {
  const model = value.model
  if (model?.method === 'historical_regime_occupancy') {
    const selected = options?.regime_runs.find(item => item.id === model.run_ref.id && item.content_hash === model.run_ref.content_hash)
    if (!selected) return text('scenarioChooseHistory')
    if (selected.available === false || !selected.as_of || selected.as_of > value.as_of)
      return selected.reasons?.map(item => item.message).join('；') || text('scenarioHistoryUnavailable')
    return null
  }
  if (!isScenarioCma(model)) return null
  if (!options?.scenario_options) return text('scenarioOptionsUnavailable')
  const history = options.scenario_options?.historical_references.find(item => item.id === model.run_ref.id
    && item.content_hash === model.run_ref.content_hash && (model.historical_reference
      ? sameReference(item.reference, model.historical_reference) : !item.reference))
  if (!history) return text('scenarioChooseHistory')
  if (!history.available) return history.reasons.map(item => item.message).join('；') || text('scenarioHistoryUnavailable')
  if (model.method === 'conditional_scenario') {
    const realtime = matchingRealtime(options, model).find(item => item.id === model.realtime_ref.id && item.content_hash === model.realtime_ref.content_hash)
    if (!realtime) return text('scenarioRealtimeMissing')
    if (!realtime.available) return realtime.reasons.map(item => item.message).join('；') || text('scenarioRealtimeUnavailable')
  }
  return null
}

export default function LtcmaScenarioFields({ value, options, onChange, sourceLabels, onLabels, mandateHorizonDays, optionsReady = true }: {
  value: CmaDraft; options: LtcmaOptions; onChange: (value: CmaDraft) => void
  sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void; mandateHorizonDays?: number; optionsReady?: boolean
}) {
  const navigate = useNavigate()
  const { t } = useLtcmaText(), model = value.model
  const history = options.scenario_options?.historical_references ?? []
  const available = history.filter(item => item.available)
  const selectHistory = (item?: ScenarioHistoryOption) => {
    if (!isScenarioCma(model) || !optionsReady) return
    onChange({ ...value, model: { ...model, run_ref: { id: item?.id ?? '', content_hash: item?.content_hash ?? '' },
      historical_reference: item?.reference ?? undefined,
      ...(model.method === 'conditional_scenario' ? { realtime_ref: { id: '', content_hash: '' } } : {}) } })
  }
  useEffect(() => {
    if (!isScenarioCma(model) || !optionsReady) return
    if (model.method === 'conditional_scenario' && !model.realtime_ref.id) {
      const eligible = matchingRealtime(options, model).filter(item => item.available)
      if (eligible.length === 1) onChange({ ...value, model: { ...model, realtime_ref: { id: eligible[0].id, content_hash: eligible[0].content_hash } } })
    }
  }, [model, options, optionsReady])
  if (!isScenarioCma(model)) return null
  if (!optionsReady) return <div role="status" className="space-y-3"><p className="text-sm text-slate-600">{t('scenarioLoading')}</p><Skeleton className="h-11 w-full" /></div>
  const current = history.find(item => item.id === model.run_ref.id && item.content_hash === model.run_ref.content_hash
    && (model.historical_reference ? sameReference(item.reference, model.historical_reference) : !item.reference))
  const realtime = matchingRealtime(options, model), validRealtime = realtime.filter(item => item.available)
  const selectedRealtime = model.method === 'conditional_scenario' ? realtime.find(item => item.id === model.realtime_ref.id && item.content_hash === model.realtime_ref.content_hash) : undefined
  const keyOf = (item: ScenarioHistoryOption) => `${item.id}:${item.reference?.publication_id ?? ''}`
  return <section aria-label={t('scenarioInputs')} className="space-y-4 border-t border-slate-200 pt-4">
    <div className="grid gap-3 md:grid-cols-2">
      <Field label={t('scenarioHistory')} hint={t('scenarioHistoryHint')}>
        <LtcmaScenarioSelect label={t('scenarioHistory')} value={current ? keyOf(current) : ''}
          choices={history.map(item => ({ value: keyOf(item), name: item.name, available: item.available, reasons: item.reasons.map(reason => reason.message) }))}
          onChange={key => selectHistory(history.find(item => keyOf(item) === key))}
          action={{ name: t('configureHistoricalScenario'), run: () => navigate(historicalScenarioPath) }} />
      </Field>
      {model.method === 'conditional_scenario' && <Field label={t('conditionalHorizon')} hint={`${t('tradingDays', { days: model.horizon_days })} · ${t('conditionalHorizonHint')}`}>
        <select aria-label={t('conditionalHorizon')} className={`${control} tabular-nums`} value={model.horizon_days} onChange={event => onChange({ ...value, model: { ...model, horizon_days: Number(event.target.value) } })}>
          {[21, 63, 126, 252, 756, 1260, 2520, model.horizon_days, ...(mandateHorizonDays && mandateHorizonDays <= 2520 ? [mandateHorizonDays] : [])].filter((day, index, all) => all.indexOf(day) === index).sort((a, b) => a - b).map(days => <option key={days} value={days}>{days % 252 === 0 ? t('forecastYears', { years: days / 252 }) : days % 21 === 0 ? t('forecastMonths', { months: days / 21 }) : t('tradingDays', { days })}{days === mandateHorizonDays ? ` · ${t('mandateHorizon')}` : ''}</option>)}
        </select>
      </Field>}
    </div>
    {!available.length && <div role="status" className="space-y-2 text-sm leading-6 text-amber-800"><p>{t(history.length ? 'scenarioFiltered' : 'scenarioMissing')}</p>
      {[...new Set(history.flatMap(item => item.reasons.map(reason => reason.message)))].map(message => <p key={message}>{message}</p>)}
    </div>}
    {current && !current.available && <p role="status" className="text-sm text-amber-800">{current.reasons.map(item => item.message).join('；')}</p>}
    {model.method === 'conditional_scenario' && <>
      <Field label={t('scenarioRealtime')} hint={t('scenarioRealtimeHint')}>
        {validRealtime.length === 1 && selectedRealtime?.available ? <p className="flex min-h-11 items-center text-sm font-medium">{selectedRealtime.name}</p>
          : <LtcmaScenarioSelect label={t('scenarioRealtime')} value={selectedRealtime?.id ?? ''}
              choices={realtime.map(item => ({ value: item.id, name: item.name, available: item.available, reasons: item.reasons.map(reason => reason.message) }))}
              onChange={id => {
                const chosen = realtime.find(item => item.id === id)
                onChange({ ...value, model: { ...model, realtime_ref: { id: chosen?.id ?? '', content_hash: chosen?.content_hash ?? '' } } })
              }} />}
      </Field>
      {!validRealtime.length && <div role="status" className="space-y-2 text-sm leading-6 text-amber-800"><p>{t('scenarioRealtimeMissing')}</p>{[...new Set(realtime.flatMap(item => item.reasons.map(reason => reason.message)))].map(message => <p key={message}>{message}</p>)}</div>}
      {mandateHorizonDays && mandateHorizonDays > 2520 && <p role="status" className="text-sm text-amber-800">{t('scenarioMandateTooLong')}</p>}
      <p className="text-sm leading-6 text-slate-600">{t('scenarioConditionalResearchOnly')}</p>
    </>}
    <Link className={linkClass} to="/settings/scenario-algorithms">{t('openScenarioCenter')}</Link>
    <p className="text-sm leading-6 text-slate-600">{t('scenarioAutomatic')}</p>
    <LtcmaProxyFields value={value} onChange={onChange} sourceLabels={sourceLabels} onLabels={onLabels} />
  </section>
}
