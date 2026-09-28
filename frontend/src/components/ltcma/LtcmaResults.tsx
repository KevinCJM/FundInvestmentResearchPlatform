import { type CmaPreview } from '../../services/strategicAllocation'
import { percentText } from '../risk-models/ResearchUI'
import { useLtcmaText } from './shared'
import LtcmaRegimeResults from './LtcmaRegimeResults'
import LtcmaModelDiagnostics from './LtcmaModelDiagnostics'
import LtcmaScenarioResults from './LtcmaScenarioResults'
import { isScenarioCma } from '../../services/cmaModelTypes'

const object = (value: unknown): Record<string, unknown> => value != null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

export default function LtcmaResults({ value }: { value: CmaPreview }) {
  const { t } = useLtcmaText()
  const scenario = isScenarioCma(value.definition.model)
  const showUncertainty = value.definition.model?.method !== 'conditional_scenario'
  const effective = value.effective_assumptions ?? value.definition
  const audit = object(value.model_result?.model_audit), evidence = object(audit.evidence)
  const names = new Map(value.source_snapshot.assets.map(asset => [asset.id, asset.name || asset.id]))
  const ids = effective.assets.map(asset => asset.id)
  const standardErrors = Array.isArray(audit.mean_standard_error) ? audit.mean_standard_error : null
  const estimatedUncertainty = ['historical_statistics', 'bayesian_niw'].includes(value.definition.model?.method ?? '')
    || (value.definition.model?.method === 'long_term_scenario' && audit.uncertainty_status === 'estimated_fixed_definition_block_bootstrap')
  const evidenceDetails = Object.keys(evidence).length > 0 && <dl className="grid gap-3 sm:grid-cols-3">
    {[['start', evidence.actual_start], ['end', evidence.actual_end], ['observations', evidence.observations]].map(([key, entry]) => <div key={String(key)}><dt className="text-xs text-slate-600">{t(String(key))}</dt><dd className="mt-1 text-sm tabular-nums">{typeof entry === 'string' || finite(entry) ? String(entry) : '—'}</dd></div>)}
  </dl>
  const diagnostics = <>
    <LtcmaModelDiagnostics audit={audit} assets={ids} names={names} />
    <section className="space-y-2 border-t border-slate-200 pt-4" aria-label={t('limitations')}><h2 className="text-lg font-semibold">{t('limitations')}</h2>{[...new Set(value.warnings)].map((warning, index) => <p key={index} className="text-sm leading-6 text-slate-700">{warning}</p>)}</section>
  </>
  const matrix = (label: string, data: Array<Array<number | null>> | null | undefined, covariance = false) => <div className="overflow-x-auto">
    <table className="w-full text-sm" aria-label={label}><caption className="py-3 text-left font-medium">{label}</caption>
      <thead><tr><th scope="col" className="p-2 text-left text-xs text-slate-600">{t('asset')}</th>{ids.map(id => <th scope="col" key={id} className="min-w-24 p-2 text-right text-xs text-slate-600">{names.get(id) ?? id}</th>)}</tr></thead>
      <tbody>{ids.map((id, i) => <tr key={id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-medium">{names.get(id) ?? id}</th>{ids.map((other, j) => {
        const n = data?.[i]?.[j]
        const cashPair = !covariance && (effective.assets[i].annual_volatility === 0 || effective.assets[j].annual_volatility === 0)
        return <td key={other} className="p-2 text-right tabular-nums">{finite(n) && !cashPair ? n.toFixed(covariance ? 6 : 3) : '—'}</td>
      })}</tr>)}</tbody>
    </table>
  </div>
  return <div className="min-w-0 space-y-5">
    <section className="space-y-3" aria-label={t('results')}><h2 className="text-lg font-semibold">{t('results')}</h2><p className="text-sm leading-6 text-slate-600">{t('resultHint')}</p>
      <p className="text-sm text-slate-700">{effective.currency} · {effective.as_of}{effective.moment_semantics ? ` · ${t(effective.moment_semantics)}` : ''}</p>
      {value.definition.model?.method === 'conditional_scenario' && <p className="text-sm text-slate-600">{t('scenarioAnnualHint')}</p>}
      <p className="text-xs leading-5 text-slate-600 sm:hidden">{t('tableScrollHint')}</p>
      <div className="overflow-x-auto"><table className="w-full min-w-[560px] text-sm" aria-label={t('results')}><thead><tr>
        {['asset', 'return', 'volatility', ...(showUncertainty ? ['uncertainty'] : []), ...(standardErrors ? ['meanStandardError'] : [])].map(key => <th key={key} scope="col" className="p-3 text-right text-xs text-slate-600 first:text-left">{t(key)}</th>)}</tr></thead>
        <tbody>{effective.assets.map((asset, index) => <tr key={asset.id} className="border-b border-slate-200"><th scope="row" className="p-3 text-left font-medium">{names.get(asset.id) ?? asset.id}</th>
          <td className="p-3 text-right tabular-nums">{percentText(asset.annual_return)}</td><td className="p-3 text-right tabular-nums">{percentText(asset.annual_volatility)}</td>
          {showUncertainty && <td className="p-3 text-right tabular-nums">{asset.mean_uncertainty === 0 && !estimatedUncertainty ? t('unknownUncertainty') : percentText(asset.mean_uncertainty)}</td>}
          {standardErrors && <td className="p-3 text-right tabular-nums">{finite(standardErrors[index]) ? percentText(standardErrors[index]) : '—'}</td>}
        </tr>)}</tbody></table></div>
      {standardErrors && <p className="text-xs leading-5 text-slate-600">{t('meanStandardErrorHint')}</p>}
      {!scenario && matrix(t('correlation'), effective.correlation)}
    </section>
    {scenario && <LtcmaScenarioResults value={value} />}
    {!scenario && evidenceDetails && <details open className="space-y-3 border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('quality')}</summary>{evidenceDetails}</details>}
    {value.definition.model?.method === 'historical_regime_occupancy' && <LtcmaRegimeResults audit={audit} />}
    {!scenario && diagnostics}
    <details className="border-t border-slate-200 pt-3"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('details')}</summary>
      {!showUncertainty && <p className="mb-3 text-sm text-slate-600">{t('uncertainty')}：{t('unknownUncertainty')}</p>}
      {scenario && <div className="space-y-4">{evidenceDetails}{diagnostics}{matrix(t('correlation'), effective.correlation)}</div>}
      {matrix(t('covariance'), value.covariance, true)}
      <pre className="mt-3 whitespace-pre-wrap break-words text-xs leading-5 text-slate-600">{JSON.stringify({ definition: value.definition, model: audit, semantics: value.semantics, execution: value.execution }, null, 2)}</pre>
    </details>
  </div>
}
