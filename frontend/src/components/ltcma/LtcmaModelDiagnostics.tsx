import { percentText } from '../risk-models/ResearchUI'
import { useLtcmaText } from './shared'

const object = (value: unknown): Record<string, unknown> => value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const array = (value: unknown): unknown[] => Array.isArray(value) ? value : []
const number = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(3) : '—'
const rate = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? percentText(value) : '—'

export default function LtcmaModelDiagnostics({ audit, assets, names }: { audit: Record<string, unknown>; assets: string[]; names: Map<string, string> }) {
  const { t } = useLtcmaText()
  const sample = object(object(audit.evidence).sample_horizon)
  const transition = object(audit.transition_diagnostics)
  const states = array(audit.state_ids).filter((v): v is string => typeof v === 'string')
  const labels = object(object(audit.regime).state_labels)
  return <div className="min-w-0 space-y-4">
    {Object.keys(sample).length > 0 && <section className="space-y-2 border-t border-slate-200 pt-4" aria-label={t('sampleHorizon')}>
      <h3 className="text-sm font-semibold">{t('sampleHorizon')}</h3>
      <p className="text-sm tabular-nums">{t('sampleHorizonSummary', { sample: number(sample.observation_years), horizon: typeof sample.forecast_years === 'number' ? sample.forecast_years : '—' })}</p>
      {sample.scope === 'incremental_evidence_batch' && <p className="text-xs leading-5 text-slate-600">{t('incrementalEvidence')}</p>}
      <p className="text-xs leading-5 text-slate-600">{t('sampleHorizonHint')}</p>
    </section>}
    {audit.diagnostics_version === 'bl-diagnostics/1.0' && <section className="min-w-0 space-y-3 border-t border-slate-200 pt-4" aria-label={t('blDiagnostics')}>
      <h3 className="text-sm font-semibold">{t('blDiagnostics')}</h3>
      <dl className="grid gap-3 sm:grid-cols-3">{[['marketExcess',rate(audit.implied_market_excess_return)],['marketVolatility',rate(audit.implied_market_volatility)],['marketSharpe',number(audit.implied_market_sharpe)]].map(([key,value]) => <div key={key}><dt className="text-xs text-slate-600">{t(key)}</dt><dd className="mt-1 text-sm tabular-nums">{value}</dd></div>)}</dl>
      <p className="text-xs leading-5 text-slate-600">{t('blDiagnosticsHint')}</p>
      {array(audit.view_precision_ratio).length > 0 && <div className="overflow-x-auto"><table className="w-full text-sm" aria-label={t('viewInfluence')}>
        <caption className="py-2 text-left text-sm font-medium">{t('viewInfluence')}</caption>
        <thead><tr><th scope="col" className="p-2 text-left">{t('blView')}</th><th scope="col" className="min-w-28 p-2 text-right">{t('viewVarianceRatio')}</th>{assets.map(id => <th key={id} scope="col" className="min-w-28 p-2 text-right">{names.get(id) ?? id}</th>)}</tr></thead>
        <tbody>{array(audit.view_precision_ratio).map((ratio,index) => <tr key={index} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{t('viewTitle',{index:index+1})}</th><td className="p-2 text-right tabular-nums">{number(ratio)}</td>{assets.map((id,i) => <td key={id} className="p-2 text-right tabular-nums">{rate(array(array(audit.view_return_contributions)[index])[i])}</td>)}</tr>)}</tbody>
      </table></div>}
    </section>}
    {Object.keys(transition).length > 0 && <section className="min-w-0 space-y-3 border-t border-slate-200 pt-4" aria-label={t('transitionDiagnostics')}>
      <h3 className="text-sm font-semibold">{t('transitionDiagnostics')}</h3>
      <p className="text-xs leading-5 text-slate-600">{t('transitionHint')}</p>
      {transition.stationary_status !== 'unique_irreducible_stationary' && <p className="text-xs text-amber-800">{t('stationaryUnavailable')}</p>}
      <div className="overflow-x-auto"><table className="w-full text-sm" aria-label={t('transitionDiagnostics')}>
        <thead><tr><th scope="col" className="p-2 text-left">{t('transitionState')}</th>{states.map(id => <th key={id} scope="col" className="min-w-24 p-2 text-right">{typeof labels[id] === 'string' ? String(labels[id]) : id}</th>)}<th scope="col" className="min-w-28 p-2 text-right">{t('stateDuration')}</th><th scope="col" className="min-w-24 p-2 text-right">{t('stationaryProbability')}</th></tr></thead>
        <tbody>{states.map((id,i) => <tr key={id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{typeof labels[id] === 'string' ? String(labels[id]) : id}</th>{states.map((other,j) => <td key={other} className="p-2 text-right tabular-nums">{rate(array(array(transition.matrix)[i])[j])}</td>)}<td className="p-2 text-right tabular-nums">{number(array(transition.markov_duration_observations)[i])}</td><td className="p-2 text-right tabular-nums">{rate(array(transition.stationary_probabilities)[i])}</td></tr>)}</tbody>
      </table></div>
    </section>}
  </div>
}
