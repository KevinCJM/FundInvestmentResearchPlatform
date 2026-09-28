import { useId } from 'react'
import { useNavigate } from 'react-router-dom'
import { Field } from '../risk-models/ResearchUI'
import { CategoryEditor } from '../risk-scales/CategoryEditor'
import { isStatisticalCma, type CmaWindow } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import type { LtcmaOptions } from '../../services/ltcma'
import { cmaMethodText, control, RateInput, useLtcmaText } from './shared'
import LtcmaNiwStrengths from './LtcmaNiwStrengths'
import LtcmaScenarioSelect from './LtcmaScenarioSelect'
import { priorReason } from '../../services/cmaCompatibility'

export const historicalScenarioPath = '/settings/scenario-algorithms?center=market-state&stage=historical'

export function LtcmaHistoryWindow({ value, onChange }: { value: CmaDraft; onChange: (value: CmaDraft) => void }) {
  const { t } = useLtcmaText(), model = value.model
  if (!isStatisticalCma(model)) return null
  const window = model.window ?? { kind: '5Y' as const }
  const changeWindow = (next: CmaWindow) => onChange({ ...value, model: { ...model, window: next } })
  return <div className="space-y-3"><Field label={t('window')} hint={t('windowHint')}><select aria-label={t('window')} className={control} value={window.kind ?? '5Y'} onChange={event => changeWindow({ kind: event.target.value as CmaWindow['kind'] })}>
    {['1Y', '2Y', '3Y', '5Y', '10Y', 'common_since_inception', 'custom'].map(kind => <option key={kind} value={kind}>{kind.endsWith('Y') ? t('windowYears', { years: Number(kind.slice(0, -1)) }) : t(kind)}</option>)}
  </select></Field>
    {window.kind === 'custom' && <div className="grid gap-3 sm:grid-cols-2"><Field label={t('start')}><input type="date" className={control} max={value.as_of} value={window.start_date ?? ''} onChange={event => changeWindow({ ...window, start_date: event.target.value })} /></Field><Field label={t('end')}><input type="date" className={control} max={value.as_of} value={window.end_date ?? ''} onChange={event => changeWindow({ ...window, end_date: event.target.value })} /></Field></div>}
  </div>
}

export function LtcmaProxyFields({ value, onChange, sourceLabels, onLabels }: {
  value: CmaDraft; onChange: (value: CmaDraft) => void; sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void
}) {
  const { t } = useLtcmaText(), model = value.model
  if (!value.strategic_universe_id || !isStatisticalCma(model) || !model.proxy_inputs) return null
  return <details open={model.proxy_inputs.assets.some(asset => asset.asset_type === 'cash' ? asset.cash_return == null : !asset.components.length || Math.abs(asset.components.reduce((sum, c) => sum + c.weight, 0) - 1) > 1e-8)} className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-base font-semibold">{t('proxies')}</summary><p className="text-sm leading-6 text-slate-600">{t('proxyHint')}</p>
    <CategoryEditor value={model.proxy_inputs} fixedAssets cashEligibleIds={value.assets.filter(a => a.role === 'liquidity' && a.liquidity === 'liquid').map(a => a.id)} sourceLabels={sourceLabels}
      onChange={(proxy_inputs, labels) => { onChange({ ...value, model: { ...model, proxy_inputs } }); if (labels) onLabels(labels) }} />
  </details>
}

export default function LtcmaStatisticsFields({ value, options, onChange, sourceLabels, onLabels, optionsReady = true, optionsError }: {
  value: CmaDraft; options: LtcmaOptions; onChange: (value: CmaDraft) => void
  sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void
  optionsReady?: boolean; optionsError?: string
}) {
  const navigate = useNavigate()
  const { t } = useLtcmaText(), model = value.model
  const helpId = useId()
  if (!isStatisticalCma(model)) return null
  const universe = options.strategic_universes.find(item => item.id === value.strategic_universe_id)?.definition
  const choices = options.assumptions.map(item => ({ item, reason: priorReason(item, value, universe) }))
  const priors = choices.filter(choice => !choice.reason).map(choice => choice.item)
  const selectedReason = model.method === 'bayesian_niw' && model.prior_ref.id
    ? choices.find(choice => choice.item.id === model.prior_ref.id)?.reason ?? (priors.some(item => item.id === model.prior_ref.id) ? null : 'priorUnavailable') : null
  const runs = options.regime_runs
  const availableRuns = runs.filter(item => item.available !== false && item.as_of && item.as_of <= value.as_of)
  const run = model.method === 'historical_regime_occupancy' ? availableRuns.find(item => item.id === model.run_ref.id) : undefined
  return <section className="space-y-5">
    <LtcmaHistoryWindow value={value} onChange={onChange} />
    <p className="text-sm leading-6 text-slate-600">{t('sampleHint')}</p>
    <LtcmaProxyFields value={value} onChange={onChange} sourceLabels={sourceLabels} onLabels={onLabels} />
    {!optionsReady && !optionsError && <p role="status" className="text-sm text-slate-600">{t('methodOptionsLoading')}</p>}
    {optionsReady && model.method === 'bayesian_niw' && <div className="space-y-4 border-t border-slate-200 pt-4">
      <Field label={t('prior')}><select aria-label={t('prior')} className={control} value={model.prior_ref.id} onChange={event => {
        const prior = priors.find(item => item.id === event.target.value)
        onChange({ ...value, model: { ...model, prior_ref: { id: prior?.id ?? '', content_hash: prior?.content_hash ?? '' }, prior_mode: 'recenter', data_reuse_acknowledged: false } })
      }}><option value="">{t('choose')}</option>{choices.map(({ item, reason }) => <option key={item.id} value={item.id} disabled={Boolean(reason)}>{item.name} · {cmaMethodText(t, item.method, item.history?.window.kind)} · {item.as_of}{reason ? ` · ${t(reason)}` : ''}</option>)}</select></Field>
      {!priors.length && <p className="text-sm text-amber-800">{t(choices.length ? 'noMatchingPrior' : 'noPrior')}</p>}
      {selectedReason && <p role="status" className="text-sm text-amber-800">{t(selectedReason)}</p>}
      {choices.some(choice => choice.reason) && <details className="border-l-2 border-slate-200 pl-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('priorExclusions')}</summary>
        <div className="divide-y divide-slate-200">{choices.filter(choice => choice.reason).map(({ item, reason }) => <p key={item.id} className="py-2 text-sm leading-6 text-slate-600">{item.name} · {cmaMethodText(t, item.method, item.history?.window.kind)}：{t(reason!)}</p>)}</div>
      </details>}
      <div><Field label={t('priorMode')}><select aria-label={t('priorMode')} aria-describedby={`${helpId}-mode`} className={control} value={model.prior_mode ?? 'recenter'} onChange={event => onChange({ ...value, model: { ...model,
        prior_mode: event.target.value as 'recenter' | 'continue', data_reuse_acknowledged: false,
        mean_prior_observations: null, covariance_prior_observations: null } })}>
        <option value="recenter">{t('recenter')}</option><option value="continue" disabled={priors.find(item => item.id === model.prior_ref.id)?.method !== 'bayesian_niw'}>{t('continuePrior')}</option>
      </select></Field><p id={`${helpId}-mode`} className="mt-1 text-sm leading-6 text-slate-600">{t('priorModeHint')}</p></div>
      {model.prior_mode !== 'continue' && <LtcmaNiwStrengths value={value} onChange={onChange} />}
      <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input type="checkbox" className="mt-1.5" checked={model.data_reuse_acknowledged ?? false} onChange={event => onChange({ ...value, model: { ...model, data_reuse_acknowledged: event.target.checked } })} />{t('overlap')}</label>
      <p className="text-sm leading-6 text-slate-600">{t('niwReuseHint')}</p>
    </div>}
    {optionsReady && model.method === 'historical_regime_occupancy' && <div className="space-y-4 border-t border-slate-200 pt-4">
      <Field label={t('regimeRun')}><LtcmaScenarioSelect label={t('regimeRun')} value={model.run_ref.id}
        choices={runs.map(item => ({ value: item.id, name: `${item.name}${item.as_of ? ` · ${item.as_of}` : ''}`,
          available: availableRuns.includes(item), reasons: item.reasons?.map(reason => reason.message) ?? [t('scenarioHistoryUnavailable')] }))}
        onChange={id => {
          const chosen = availableRuns.find(item => item.id === id)
          onChange({ ...value, model: { ...model, run_ref: { id: chosen?.id ?? '', content_hash: chosen?.content_hash ?? '' }, probabilities: null, probability_reason: '' } })
        }} action={{ name: t('configureHistoricalScenario'), run: () => navigate(historicalScenarioPath) }} /></Field>
      {!availableRuns.length && <p className="text-sm text-amber-800">{t(runs.length ? 'scenarioFiltered' : 'noRegime')}</p>}
      <p className="text-sm leading-6 text-slate-600">{t('regimeHint')}</p>
      <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" disabled={!run} checked={model.probabilities != null} onChange={event => onChange({ ...value, model: { ...model,
        probabilities: event.target.checked && run ? Object.fromEntries(run.states.map(state => [state.id, NaN])) : null, probability_reason: '' } })} />{t('overrideProbabilities')}</label>
      {model.probabilities != null && <><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{run?.states.map(state => <Field key={state.id} label={`${state.label ?? state.id} · ${t('probability')}`}><RateInput value={model.probabilities?.[state.id]} onChange={probability => onChange({ ...value, model: { ...model, probabilities: { ...model.probabilities, [state.id]: probability } } })} /></Field>)}</div>
        <Field label={t('probabilityReason')}><textarea className={control} value={model.probability_reason ?? ''} onChange={event => onChange({ ...value, model: { ...model, probability_reason: event.target.value } })} /></Field></>}
    </div>}
  </section>
}
