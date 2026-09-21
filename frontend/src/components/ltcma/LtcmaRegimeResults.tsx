import { percentText } from '../risk-models/ResearchUI'
import { useLtcmaText } from './shared'

const record = (value: unknown): Record<string, unknown> => value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const items = (value: unknown): unknown[] => Array.isArray(value) ? value : []
const numeric = (value: unknown): number | undefined => typeof value === 'number' && Number.isFinite(value) ? value : undefined

/** Display frozen state evidence; never infer probabilities from labels or missing values. */
export default function LtcmaRegimeResults({ audit }: { audit: Record<string, unknown> }) {
  const { t } = useLtcmaText()
  const states = items(audit.state_ids)
  const counts = items(audit.counts), detected = items(audit.detected_probabilities), applied = items(audit.applied_probabilities)
  const regime = record(audit.regime), labels = record(regime.state_labels)
  const estimated = new Set(items(audit.estimated_states)), unestimated = new Set(items(audit.unestimated_states))
  if (!states.length || states.some(id => typeof id !== 'string')) return <p role="status" className="text-sm text-amber-800">{t('regimeEvidenceMissing')}</p>
  return <section className="min-w-0 space-y-3 border-t border-slate-200 pt-4" aria-label={t('stateEvidence')}>
    <h2 className="text-lg font-semibold">{t('stateEvidence')}</h2>
    <p className="text-sm leading-6 text-slate-600">{t('stateProbabilityHint')}</p>
    <p className="text-xs leading-5 text-slate-600 sm:hidden">{t('tableScrollHint')}</p>
    <div className="overflow-x-auto"><table className="w-full min-w-[560px] text-sm" aria-label={t('stateEvidence')}>
      <thead><tr>{['stateName', 'observations', 'detectedProbability', 'appliedProbability', 'stateEstimate'].map(key => <th key={key} scope="col" className="p-3 text-right text-xs text-slate-600 first:text-left">{t(key)}</th>)}</tr></thead>
      <tbody>{states.map((entry, index) => {
        const id = String(entry), name = typeof labels[id] === 'string' ? String(labels[id]) : id
        return <tr key={id} className="border-b border-slate-200">
          <th scope="row" className="p-3 text-left font-medium">{name}</th>
          <td className="p-3 text-right tabular-nums">{numeric(counts[index]) ?? '—'}</td>
          <td className="p-3 text-right tabular-nums">{percentText(numeric(detected[index]))}</td>
          <td className="p-3 text-right tabular-nums">{percentText(numeric(applied[index]))}</td>
          <td className="p-3 text-right">{t(unestimated.has(id) ? 'stateUnestimated' : estimated.has(id) ? 'stateEstimated' : 'unavailable')}</td>
        </tr>
      })}</tbody>
    </table></div>
    <p className="text-sm text-slate-600">{t('excludedStateObservations', { count: numeric(regime.unknown_observations) ?? '—' })}</p>
    {typeof audit.probability_reason === 'string' && audit.probability_reason && <p className="break-words text-sm leading-6 text-slate-700">{t('probabilityReason')}：{audit.probability_reason}</p>}
  </section>
}
