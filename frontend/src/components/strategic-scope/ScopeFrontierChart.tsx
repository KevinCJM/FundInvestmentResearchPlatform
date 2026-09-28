import { useEffect, useRef, useState } from 'react'
import FrontierChart from '../strategic-allocation/FrontierChart'
import { DataTable } from '../ui'
import FrontierLegend, { type FrontierLegendItem } from '../strategic-allocation/FrontierLegend'
import { formatNumber, useI18n } from '../../i18n/runtime'

export interface ScopeFrontierPoint {
  volatility: number | null
  expected_return: number | null
  weights?: Record<string, number>
  status: string
}

export interface ScopeFrontierChartProps {
  points: ScopeFrontierPoint[]
  referencePoints?: ScopeFrontierPoint[]
  constrainedPoints?: ScopeFrontierPoint[]
  targetReturn: number | null
  volatilityCap: number | null
  compound?: boolean
  targetCurve?: Array<{ volatility: number; expected_return: number | null }>
  candidate?: { volatility: number; expected_return: number; weights: Record<string, number> } | null
}

const finite = (value: number | null): value is number => value !== null && Number.isFinite(value)
const usable = (point: ScopeFrontierPoint) => point.status === 'optimal_to_tolerance'
  && finite(point.volatility) && point.volatility >= 0 && finite(point.expected_return)

export default function ScopeFrontierChart({ points, referencePoints = [], constrainedPoints = [], targetReturn, volatilityCap, candidate, compound = false, targetCurve = [] }: ScopeFrontierChartProps) {
  const { s, locale } = useI18n()
  const t = (key: string) => s(`scopeFeasibility.chart.${compound && ['return', 'meaning', 'noTarget', 'targetLine', 'candidate'].includes(key) ? `compound.${key}` : key}`)
  const percent = (value: number | null) => formatNumber(value, { style: 'percent', minimumFractionDigits: 2, maximumFractionDigits: 2 }, locale)
  const tokens = useRef<HTMLDivElement>(null)
  const [paint, setPaint] = useState<string[]>([])
  const [selected, setSelected] = useState<Record<string, boolean>>({})
  useEffect(() => { setPaint(Array.from(tokens.current?.children ?? []).map(node => getComputedStyle(node).color)) }, [])

  const target = finite(targetReturn) ? targetReturn : null
  const cap = finite(volatilityCap) && volatilityCap >= 0 ? volatilityCap : null
  const computed = candidate && finite(candidate.volatility) && candidate.volatility >= 0 && finite(candidate.expected_return) ? candidate : null
  const allPoints = [...points, ...referencePoints, ...constrainedPoints]
  const validPoints = allPoints.filter(usable)
  const ys = [...targetCurve.filter(p => p.expected_return != null).map(p => p.expected_return!), ...validPoints.map(point => point.expected_return!), ...(computed ? [computed.expected_return] : []), target ?? 0, 0]
  const xs = [...validPoints.map(point => point.volatility!), ...(computed ? [computed.volatility] : []), cap ?? 0, 0]
  const yLow = Math.min(...ys) * 100, yHigh = Math.max(...ys) * 100
  const yPadding = Math.max((yHigh - yLow) * .15, .5)
  const yMax = yHigh + yPadding
  const hasTarget = target !== null && cap !== null && targetCurve.length === 0
  const comparisons = [
    { id: 'reference', points: referencePoints, label: t('reference'), color: paint[1], line: 'dashed', symbolClass: 'w-6 border-t-2 border-dashed border-slate-600' },
    { id: 'constrained', points: constrainedPoints, label: t('constrained'), color: paint[6], line: 'dotted', symbolClass: 'w-6 border-t-2 border-dotted border-amber-700' },
  ].filter(group => group.points.length)
  const items: FrontierLegendItem[] = [
    ...(points.length ? [{ id: 'frontier', name: t('frontier'), symbolClass: 'w-6 border-t-2 border-accent-600' }] : []),
    ...comparisons.map(group => ({ id: group.id, name: group.label, symbolClass: group.symbolClass })),
    ...(target !== null || cap !== null ? [{ id: 'target', name: s('frontierLegend.target'), symbolClass: 'h-2.5 w-2.5 rotate-45 bg-slate-800',
      detail: [target === null ? '' : `${t('targetLine')} ${percent(target)}`, cap === null ? '' : `${t('capLine')} ${percent(cap)}`].filter(Boolean).join(' · ') }] : []),
    ...(computed ? [{ id: 'candidate', name: t('candidate'), symbolClass: 'h-2.5 w-2.5 rounded-full bg-emerald-700' }] : []),
  ]
  const description = [t('return'), s(`scopeFeasibility.chart.${compound ? 'compound.' : ''}description`, { target: percent(target), cap: percent(cap) }), ...comparisons.map(group => group.label)].join(' ')
  const option = {
    animation: false,
    legend: { show: false, selected: Object.fromEntries(items.map(item => [item.name, selected[item.id] !== false])) },
    aria: { enabled: true, description },
    grid: { left: 8, right: 20, top: 20, bottom: 48, containLabel: true },
    textStyle: { color: paint[1], fontSize: 12 },
    tooltip: {
      trigger: 'item', renderMode: 'richText', confine: true,
      formatter: (item: { seriesName: string; value?: number[] }) => Array.isArray(item.value)
        ? `${item.seriesName}\n${t('volatility')}: ${percent(item.value[0] / 100)}\n${t('return')}: ${percent(item.value[1] / 100)}` : '',
    },
    xAxis: { type: 'value', min: 0, max: Math.max(Math.max(...xs) * 110, 1),
      name: t('volatility'), nameLocation: 'middle', nameGap: 30, splitNumber: 4,
      nameTextStyle: { color: paint[1], fontSize: 12 },
      axisLabel: { color: paint[1], fontSize: 12, formatter: (value: number) => `${Number(value.toFixed(2))}%` },
      splitLine: { lineStyle: { color: paint[5] } } },
    yAxis: { type: 'value', min: yLow < 0 ? yLow - yPadding : 0, max: yMax, splitNumber: 4,
      axisLabel: { color: paint[1], fontSize: 12, formatter: (value: number) => `${Number(value.toFixed(2))}%` },
      splitLine: { lineStyle: { color: paint[5] } } },
    series: [
      { name: t('frontier'), type: 'line', showSymbol: true, symbolSize: 5, connectNulls: false,
        itemStyle: { color: paint[0] }, lineStyle: { color: paint[0], width: 3 },
        data: points.map(point => usable(point) ? [point.volatility! * 100, point.expected_return! * 100] : null) },
      { name: s('frontierLegend.target'), type: 'scatter', symbol: 'diamond', symbolSize: 16, itemStyle: { color: paint[2] },
        data: hasTarget ? [[cap! * 100, target! * 100]] : [],
        markLine: { symbol: 'none', silent: true, label: { show: false },
          lineStyle: { color: paint[2], width: 2, type: 'dashed' },
          data: [...(target === null || targetCurve.length ? [] : [{ yAxis: target * 100 }]), ...(cap === null ? [] : [{ xAxis: cap * 100 }])] },
        markArea: hasTarget ? { silent: true, itemStyle: { color: paint[3] },
          data: [[{ xAxis: 0, yAxis: target! * 100 }, { xAxis: cap! * 100, yAxis: yMax }]] } : undefined },
      ...(targetCurve.length ? [{ name: s('frontierLegend.target'), type: 'line', showSymbol: false,
        lineStyle: { color: paint[2], width: 2, type: 'dashed' },
        data: targetCurve.map(p => p.expected_return == null ? null : [p.volatility * 100, p.expected_return * 100]) }] : []),
      { name: t('candidate'), type: 'scatter' , symbol: 'circle', symbolSize: 11, itemStyle: { color: paint[4] },
        data: computed ? [[computed.volatility * 100, computed.expected_return * 100]] : [] },
      ...comparisons.map(group => ({ name: group.label, type: 'line', showSymbol: true, symbolSize: 4,
        connectNulls: false, itemStyle: { color: group.color }, lineStyle: { color: group.color, width: 2, type: group.line },
        data: group.points.map(point => usable(point) ? [point.volatility! * 100, point.expected_return! * 100] : null) })),
    ],
  }
  const rows = [{ points, label: t('frontier') }, ...comparisons].flatMap(group => group.points.map((point, index) => ({
    label: `${group.label} ${index + 1}`, volatility: usable(point) ? point.volatility : null,
    expected_return: usable(point) ? point.expected_return : null, status: usable(point) ? t('computed') : t('unresolved') })))
  if (computed) rows.push({ label: t('candidate'), volatility: computed.volatility, expected_return: computed.expected_return, status: t('computed') })
  if (hasTarget) rows.push({ label: t('targetPoint'), volatility: cap, expected_return: target, status: t('boundary') })

  return <div className="min-w-0 space-y-3">
    <div ref={tokens} aria-hidden="true" className="hidden"><span className="text-accent-600" /><span className="text-slate-600" /><span className="text-slate-800" /><span className="text-emerald-100" /><span className="text-emerald-700" /><span className="text-slate-200" /><span className="text-amber-700" /></div>
    <p className="text-xs text-slate-600">{t('return')}</p>
    <FrontierChart option={option} height={340} description={description} testId="scope-frontier-chart" />
    <FrontierLegend items={items} selected={selected} onChange={setSelected} />
    <p className="text-xs leading-5 text-slate-600">{targetCurve.length ? s('policyFrontier.returnCurveHelp') : t(hasTarget ? 'meaning' : 'noTarget')}</p>
    {allPoints.some(point => !usable(point)) && <p className="text-xs leading-5 text-amber-800">{t('gaps')}</p>}
    <details className="border-t border-slate-200 pt-3">
      <summary className="min-h-10 cursor-pointer text-sm font-medium text-slate-800">{t('table')}</summary>
      <DataTable caption={t('table')} rows={rows} rowKey={(_, index) => String(index)} minWidth="420px" empty={t('empty')}
        columns={[{ header: t('point'), cell: row => row.label },
          { header: t('volatility'), numeric: true, cell: row => percent(row.volatility) },
          { header: t('return'), numeric: true, cell: row => percent(row.expected_return) },
          { header: t('status'), cell: row => row.status }]} />
    </details>
  </div>
}
