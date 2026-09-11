import { useEffect, useId, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { useI18n } from '../../i18n/runtime'
import { evaluateTimeSeriesIndicators, type IndicatorDefinition, type ProductKind, type TimeSeriesIndicatorResult } from '../../services/customIndicators'
import IndicatorParameterInputs from './IndicatorParameterInputs'

interface Props {
  indicators: IndicatorDefinition[]
  productId: string
  productKind: ProductKind
  periods: string[]
  asOf?: string
}
const field = 'mt-1 min-h-11 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-sky-300'

function SeriesCalculation({ indicator, productId, productKind, period, asOf }: {
  indicator: IndicatorDefinition; productId: string; productKind: ProductKind; period: string; asOf?: string
}) {
  const { s } = useI18n()
  const [parameters, setParameters] = useState<Record<string, number>>({})
  const [record, setRecord] = useState<{ key: string; result?: TimeSeriesIndicatorResult; error?: string } | null>(null)
  const requestKey = JSON.stringify([indicator.id, indicator.revision, parameters, productId, productKind, period, asOf])
  useEffect(() => {
    let active = true
    evaluateTimeSeriesIndicators({
      indicator_instances: [{ indicator_id: indicator.id, indicator_revision: indicator.revision, parameters }],
      target: { kind: productKind, product_id: productId }, period, as_of: asOf || undefined, max_points: 5000,
    }).then(response => {
      if (active) setRecord({ key: requestKey, result: response.results[0] })
    }).catch(failure => {
      if (active) setRecord({ key: requestKey, error: failure instanceof Error ? failure.message : s('indicatorParameters.requestError') })
    })
    return () => { active = false }
  }, [requestKey])
  const current = record?.key === requestKey ? record : null
  const result = current?.result
  const schema = indicator.parameter_contract_version === '1.0' ? indicator.parameter_schema ?? [] : []
  const groups = new Map<string, TimeSeriesIndicatorResult['channels']>()
  for (const channel of result?.channels ?? []) {
    const key = JSON.stringify([channel.semantic_dimension, channel.price_basis, channel.unit, channel.display_format])
    groups.set(key, [...(groups.get(key) ?? []), channel])
  }
  return <div className="mt-4 min-w-0">
    <p className="text-sm font-semibold text-slate-800">{indicator.name} · v{indicator.revision}</p>
    {schema.length ? <IndicatorParameterInputs schema={schema} values={parameters} onApply={setParameters} /> : <p className="mt-2 text-xs text-slate-500">{s('indicatorParameters.fixedHint')}</p>}
    {!current && <p role="status" className="mt-4 text-sm text-slate-500">{s('indicatorParameters.calculating')}</p>}
    {current?.error && <p role="alert" className="mt-4 rounded-lg bg-rose-50 p-3 text-sm text-rose-700">{current.error}</p>}
    {result && <div className="mt-4">
      <p className="break-words text-xs text-slate-600">{s('indicatorParameters.actual')}: {Object.entries(result.parameters).map(([id, value]) => `${schema.find(item => item.id === id)?.label ?? id}=${value}`).join(' · ') || '—'} · {result.window.start_date ?? '—'} – {result.window.end_date ?? '—'}</p>
      {result.warnings.map((warning, index) => <p key={`${warning.code}-${index}`} role="status" className="mt-2 text-xs text-amber-800">{warning.message}</p>)}
      {Array.from(groups.entries()).map(([group, channels]) => <ReactECharts key={group} notMerge lazyUpdate style={{ height: 340 }} aria-label={`${indicator.name} ${channels.map(channel => channel.label).join(' / ')}`} option={{
        animation: false, tooltip: { trigger: 'axis' },
        legend: { type: 'scroll', top: 8, data: channels.map(channel => channel.label) },
        grid: { top: 50, left: 65, right: 25, bottom: 60, containLabel: true },
        xAxis: { type: 'category', boundaryGap: false, data: result.dates },
        yAxis: { type: 'value', scale: true, name: channels[0]?.unit || '', axisLabel: { formatter: channels[0]?.display_format === 'percent' ? '{value}%' : '{value}' } },
        dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 8, height: 18 }],
        series: channels.map(channel => ({ name: channel.label, type: 'line', showSymbol: false, connectNulls: false,
          data: channel.display_format === 'percent' ? channel.values.map(value => value === null ? null : value * 100) : channel.values })),
      }} />)}
    </div>}
  </div>
}

export default function TimeSeriesIndicatorPanel({ indicators, productId, productKind, periods, asOf }: Props) {
  const { s } = useI18n()
  const id = useId()
  const [selectedId, setSelectedId] = useState('')
  const [period, setPeriod] = useState('1Y')
  const available = indicators.filter(item => item.result_kind === 'time_series')
  const selected = available.find(item => item.id === selectedId)
  const actualPeriod = periods.includes(period) ? period : periods[0] || '1Y'
  if (!available.length) return null
  return <section className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm" aria-labelledby={`${id}-title`}>
    <h2 id={`${id}-title`} className="font-semibold text-slate-900">{s('indicatorParameters.customSeries')}</h2>
    <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
      <label className="text-sm text-slate-700">{s('indicatorParameters.choose')}<select className={field} value={selected?.id ?? ''} onChange={event => setSelectedId(event.target.value)}><option value="">{s('indicatorParameters.choose')}</option>{available.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.revision}</option>)}</select></label>
      <label className="text-sm text-slate-700">{s('indicatorParameters.period')}<select className={field} value={actualPeriod} onChange={event => setPeriod(event.target.value)}>{(periods.length ? periods : ['1Y']).map(value => <option key={value} value={value}>{value}</option>)}</select></label>
    </div>
    {selected && <SeriesCalculation key={`${selected.id}@${selected.revision}:${productKind}:${productId}`} indicator={selected} productId={productId} productKind={productKind} period={actualPeriod} asOf={asOf} />}
  </section>
}
