import { Link } from 'react-router-dom'
import { Field, NumberInput } from '../risk-models/ResearchUI'
import { ReferenceEditor } from '../risk-scales/ReferenceEditor'
import { isStatisticalCma, type CmaWindow } from '../../services/cmaModelTypes'
import type { CmaDraft } from '../../services/strategicAllocation'
import type { LtcmaOptions } from '../../services/ltcma'
import { control, linkClass, RateInput, useLtcmaText } from './shared'

export default function LtcmaStatisticsFields({ value, options, onChange, sourceLabels, onLabels }: {
  value: CmaDraft; options: LtcmaOptions; onChange: (value: CmaDraft) => void
  sourceLabels: Record<string, string>; onLabels: (value: Record<string, string>) => void
}) {
  const { t } = useLtcmaText(), model = value.model
  if (!isStatisticalCma(model)) return null
  const window = model.window ?? { kind: '5Y' as const }
  const changeWindow = (next: CmaWindow) => onChange({ ...value, model: { ...model, window: next } })
  const priors = options.assumptions.filter(item => item.schema_version === '2.0' && !item.retired
    && item.moment_semantics === 'annualized_periodic_arithmetic' && item.currency === value.currency
    && item.horizon_years === value.horizon_years && item.as_of <= value.as_of
    && (item.strategic_universe_id ?? null) === (value.strategic_universe_id ?? null)
    && (item.alloc_name ?? null) === (value.alloc_name ?? null)
    && item.asset_ids.join('\u0000') === value.assets.map(asset => asset.id).join('\u0000'))
  const runs = options.regime_runs.filter(item => item.as_of && item.as_of <= value.as_of && item.frequency === 'daily')
  const run = model.method === 'historical_regime_occupancy' ? runs.find(item => item.id === model.run_ref.id) : undefined
  return <section className="space-y-5">
    <div className="space-y-3"><Field label={t('window')} hint={t('horizonHint')}><select aria-label={t('window')} className={control} value={window.kind ?? '5Y'} onChange={event => changeWindow({ kind: event.target.value as CmaWindow['kind'] })}>
      {['1Y', '2Y', '3Y', '5Y', '10Y', 'common_since_inception', 'custom'].map(kind => <option key={kind} value={kind}>{kind.endsWith('Y') ? t('horizonValue', { years: Number(kind.slice(0, -1)) }) : t(kind)}</option>)}
    </select></Field>
      {window.kind === 'custom' && <div className="grid gap-3 sm:grid-cols-2"><Field label={t('start')}><input type="date" className={control} max={value.as_of} value={window.start_date ?? ''} onChange={event => changeWindow({ ...window, start_date: event.target.value })} /></Field><Field label={t('end')}><input type="date" className={control} max={value.as_of} value={window.end_date ?? ''} onChange={event => changeWindow({ ...window, end_date: event.target.value })} /></Field></div>}
      <p className="text-sm leading-6 text-slate-600">{t('sampleHint')}</p>
    </div>
    {value.strategic_universe_id && model.proxy_inputs && <details open className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-base font-semibold">{t('proxies')}</summary><p className="text-sm leading-6 text-slate-600">{t('proxyHint')}</p>
      <ReferenceEditor value={model.proxy_inputs} fixedAssets cashEligibleIds={value.assets.filter(a => a.role === 'liquidity' && a.liquidity === 'liquid').map(a => a.id)} sourceLabels={sourceLabels}
        onChange={(proxy_inputs, labels) => { onChange({ ...value, model: { ...model, proxy_inputs } }); if (labels) onLabels(labels) }} />
    </details>}
    {model.method === 'bayesian_niw' && <div className="space-y-4 border-t border-slate-200 pt-4">
      <Field label={t('prior')}><select aria-label={t('prior')} className={control} value={model.prior_ref.id} onChange={event => {
        const prior = priors.find(item => item.id === event.target.value)
        onChange({ ...value, model: { ...model, prior_ref: { id: prior?.id ?? '', content_hash: prior?.content_hash ?? '' }, prior_mode: 'recenter', data_reuse_acknowledged: false } })
      }}><option value="">{t('choose')}</option>{priors.map(item => <option key={item.id} value={item.id}>{item.name} · {item.as_of}</option>)}</select></Field>
      {!priors.length && <p className="text-sm text-amber-800">{t('noPrior')}</p>}
      <Field label={t('priorMode')}><select aria-label={t('priorMode')} className={control} value={model.prior_mode ?? 'recenter'} onChange={event => onChange({ ...value, model: { ...model, prior_mode: event.target.value as 'recenter' | 'continue', data_reuse_acknowledged: false } })}>
        <option value="recenter">{t('recenter')}</option><option value="continue" disabled={priors.find(item => item.id === model.prior_ref.id)?.method !== 'bayesian_niw'}>{t('continuePrior')}</option>
      </select></Field><p className="text-sm leading-6 text-slate-600">{t('priorHint')}</p>
      {model.prior_mode !== 'continue' && <div className="grid gap-3 sm:grid-cols-2"><Field label={t('meanStrength')}><NumberInput className={control} value={model.mean_prior_observations ?? NaN} onValueChange={mean_prior_observations => onChange({ ...value, model: { ...model, mean_prior_observations } })} /></Field>
        <Field label={t('riskStrength')}><NumberInput className={control} value={model.covariance_prior_observations ?? NaN} onValueChange={covariance_prior_observations => onChange({ ...value, model: { ...model, covariance_prior_observations } })} /></Field></div>}
      <label className="flex min-h-10 items-start gap-2 text-sm leading-6"><input type="checkbox" className="mt-1.5" checked={model.data_reuse_acknowledged ?? false} onChange={event => onChange({ ...value, model: { ...model, data_reuse_acknowledged: event.target.checked } })} />{t('overlap')}</label>
    </div>}
    {model.method === 'historical_regime_occupancy' && <div className="space-y-4 border-t border-slate-200 pt-4">
      <Field label={t('regimeRun')}><select aria-label={t('regimeRun')} className={control} value={model.run_ref.id} onChange={event => {
        const chosen = runs.find(item => item.id === event.target.value)
        onChange({ ...value, model: { ...model, run_ref: { id: chosen?.id ?? '', content_hash: chosen?.content_hash ?? '' }, probabilities: null, probability_reason: '' } })
      }}><option value="">{t('choose')}</option>{runs.map(item => <option key={item.id} value={item.id}>{item.name} · {item.as_of}</option>)}</select></Field>
      {!runs.length && <p className="text-sm text-amber-800">{t('noRegime')}</p>}<Link className={linkClass} to="/settings/scenario-algorithms">{t('openRegime')}</Link>
      <p className="text-sm leading-6 text-slate-600">{t('regimeHint')}</p>
      <label className="flex min-h-10 items-center gap-2 text-sm"><input type="checkbox" disabled={!run} checked={model.probabilities != null} onChange={event => onChange({ ...value, model: { ...model,
        probabilities: event.target.checked && run ? Object.fromEntries(run.states.map(state => [state.id, NaN])) : null, probability_reason: '' } })} />{t('overrideProbabilities')}</label>
      {model.probabilities != null && <><div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{run?.states.map(state => <Field key={state.id} label={`${state.label ?? state.id} · ${t('probability')}`}><RateInput value={model.probabilities?.[state.id]} onChange={probability => onChange({ ...value, model: { ...model, probabilities: { ...model.probabilities, [state.id]: probability } } })} /></Field>)}</div>
        <Field label={t('probabilityReason')}><textarea className={control} value={model.probability_reason ?? ''} onChange={event => onChange({ ...value, model: { ...model, probability_reason: event.target.value } })} /></Field></>}
    </div>}
    {model.method !== 'bayesian_niw' && <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('advanced')}</summary><Field label={t('shrinkage')}><RateInput value={model.shrinkage ?? 0} onChange={shrinkage => onChange({ ...value, model: { ...model, shrinkage } })} /></Field></details>}
  </section>
}
