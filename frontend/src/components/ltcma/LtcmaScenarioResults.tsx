import { DataTable, type TableColumn } from '../ui'
import { percentText } from '../risk-models/ResearchUI'
import type { CmaPreview } from '../../services/strategicAllocation'
import { ltcmaSaaIssue } from '../../services/ltcma'
import { useLtcmaText } from './shared'

const record = (value: unknown): Record<string, unknown> => value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {}
const numberAt = (value: unknown, index: number) => Array.isArray(value) && typeof value[index] === 'number' && Number.isFinite(value[index]) ? value[index] as number : undefined
const rate = (value: unknown, index: number) => { const n = numberAt(value, index); return n == null ? '—' : percentText(n) }

export default function LtcmaScenarioResults({ value }: { value: CmaPreview }) {
  const { t } = useLtcmaText(), audit = record(value.model_result?.model_audit)
  const probabilities = record(audit.scenario_probabilities), horizon = record(audit.horizon_distribution)
  const conditional = value.definition.model?.method === 'conditional_scenario'
  const states = Array.isArray(probabilities.state_ids) ? probabilities.state_ids.filter((state): state is string => typeof state === 'string') : []
  const labels = Array.isArray(probabilities.state_labels) ? probabilities.state_labels : []
  const columns: TableColumn<number>[] = [{ header: t('scenarioState'), cell: index => typeof labels[index] === 'string' ? labels[index] : states[index] }]
  const keys = conditional ? ['current', 'endpoint', 'average'] : ['historical', 'applied']
  const label: Record<string, string> = { historical: 'scenarioHistorical', applied: 'scenarioApplied', current: 'scenarioCurrent', endpoint: 'scenarioEndpoint', average: 'scenarioAverage' }
  keys.forEach(key => columns.push({ header: t(label[key]), numeric: true, cell: index => rate(probabilities[key], index) }))
  const saaIssue = ltcmaSaaIssue(value)
  const assets = value.effective_assumptions?.assets ?? value.definition.assets
  const names = new Map(value.source_snapshot.assets.map(asset => [asset.id, asset.name || asset.id]))
  const quantiles = Array.isArray(horizon.quantiles) ? horizon.quantiles : []
  return <section className="min-w-0 space-y-4 border-t border-slate-200 pt-4" aria-label={t('scenarioInputs')}>
    {saaIssue && <p role="status" className="text-sm leading-6 text-amber-800">{saaIssue}</p>}
    {states.length > 0 && <>
      <h2 className="text-lg font-semibold">{t('scenarioProbabilities')}</h2>
      <DataTable caption={t('scenarioProbabilities')} rows={states.map((_, index) => index)} rowKey={index => states[index]} columns={columns} empty={t('scenarioMissing')} />
      <p className="text-xs leading-5 text-slate-600">{t(conditional ? 'scenarioProbabilityHint' : 'scenarioLongProbabilityHint')}</p>
    </>}
    {conditional && Object.keys(horizon).length > 0 && <>
      <h2 className="text-lg font-semibold">{t('scenarioHorizonResults')}{typeof horizon.horizon_days === 'number' ? ` · ${t('tradingDays', { days: horizon.horizon_days })}` : ''}</h2>
      <p className="text-sm leading-6 text-slate-600">{t('scenarioHorizonResultHint')}</p>
      <DataTable caption={t('scenarioHorizonResults')} rows={assets.map((_, index) => index)} rowKey={index => assets[index].id} minWidth="720px" empty={t('selectScopeFirst')} columns={[
        { header: t('asset'), cell: index => names.get(assets[index].id) ?? assets[index].id },
        { header: t('scenarioCumulativeMean'), numeric: true, cell: index => rate(horizon.expected_returns, index) },
        { header: t('scenarioCumulativeMedian'), numeric: true, cell: index => rate(quantiles[1], index) },
        { header: t('scenarioP05'), numeric: true, cell: index => rate(quantiles[0], index) },
        { header: t('scenarioP95'), numeric: true, cell: index => rate(quantiles[2], index) },
        { header: t('scenarioLossProbability'), numeric: true, cell: index => rate(horizon.loss_probabilities, index) },
      ]} />
    </>}
  </section>
}
