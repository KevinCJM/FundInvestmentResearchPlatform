import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import * as echarts from 'echarts'
import { Badge, Button, EmptyState } from '../ui'
import { metadata, textValue, type PreviewResponse, type ScaleResult } from '../../services/riskScales'
import { pct, Problems, useRiskText } from './shared'

export function RiskScaleResults({ preview, compact = false, quality }: { preview: PreviewResponse; compact?: boolean; quality?: Record<string, unknown> }) {
  const { t, locale } = useRiskText()
  const result = preview.result
  const sourceQuality = quality ?? metadata(result.parameter_evidence?.data_quality)
  const [selected, setSelected] = useState('C1'), [themeReady, setThemeReady] = useState(false)
  const tokens = useRef<HTMLDivElement>(null)
  const chartContainer = useRef<HTMLDivElement>(null)
  const chartInstance = useRef<echarts.ECharts | null>(null)
  useEffect(() => {
    const colors = Array.from(tokens.current?.children ?? []).map(element => getComputedStyle(element).color)
    if (colors.length === 3) { echarts.registerTheme('risk-scales', { color: [colors[0]], textStyle: { color: colors[1], fontSize: 12 }, line: { lineStyle: { width: 2 }, symbolSize: 5 }, valueAxis: { axisLabel: { color: colors[1], fontSize: 12 }, nameTextStyle: { color: colors[1], fontSize: 12 }, splitLine: { lineStyle: { color: colors[2] } } }, tooltip: { textStyle: { fontSize: 12, color: colors[1] } } }); setThemeReady(true) }
  }, [])
  const pointOk = (point: ScaleResult['frontier'][number]) => point.status === 'optimal_to_tolerance' && point.volatility != null && Number.isFinite(point.volatility) && point.expected_return != null && Number.isFinite(point.expected_return)
  const active = result.levels.find(level => level.level_code === selected)
  const hasFrontier = result.frontier.some(pointOk)
  useEffect(() => {
    const container = chartContainer.current
    if (!container || compact || !themeReady || !hasFrontier) return
    let frame = 0
    const resize = () => {
      if (!chartInstance.current) return
      window.cancelAnimationFrame(frame)
      frame = window.requestAnimationFrame(() => {
        const instance = chartInstance.current
        if (instance && !instance.isDisposed() && container.clientWidth > 0) {
          instance.resize({ width: container.clientWidth })
        }
      })
    }
    // Observe the real content box: sidebar changes, reloads and zoom need not
    // trigger the chart library's window-size sensor at the right moment.
    const observer = typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(resize)
    observer?.observe(container)
    window.addEventListener('resize', resize)
    resize()
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', resize)
      window.cancelAnimationFrame(frame)
    }
  }, [compact, themeReady, hasFrontier])
  const option = useMemo(() => ({
    animation: false, aria: { enabled: true }, grid: { left: 12, right: 22, bottom: 46, top: 24, containLabel: true },
    xAxis: { type: 'value', name: t('volatilityAxis'), nameLocation: 'middle', nameGap: 30, axisLabel: { formatter: '{value}%' }, scale: true },
    yAxis: { type: 'value', axisLabel: { formatter: '{value}%' }, scale: true },
    tooltip: { trigger: 'item', valueFormatter: (value: number) => `${value.toFixed(2)}%` },
    series: [
      { name: t('frontier'), type: 'line', smooth: false, connectNulls: false, data: result.frontier.map(point => pointOk(point) ? { value: [point.volatility! * 100, point.expected_return! * 100], nodeId: point.node_id } : null), markLine: { silent: true, symbol: 'none', label: { fontSize: 12, formatter: '{b}' }, data: result.levels.slice(0, 4).map(level => ({ name: level.level_code, xAxis: level.upper_bound * 100 })) } },
      { name: selected, type: 'line', lineStyle: { width: 5 }, silent: true, smooth: false, connectNulls: false, symbol: 'none', data: result.frontier.map(point => pointOk(point) && active && point.volatility! <= active.upper_bound && (active.lower_inclusive ? point.volatility! >= active.lower_bound : point.volatility! > active.lower_bound) ? [point.volatility! * 100, point.expected_return! * 100] : null) },
      { name: t('representative'), type: 'scatter', symbol: 'diamond', data: result.levels.filter(level => level.representative_node_id != null && level.expected_return.value != null && level.volatility.value != null && Number.isFinite(level.expected_return.value) && Number.isFinite(level.volatility.value)).map(level => ({ name: level.level_code, level: level.level_code, value: [level.volatility.value! * 100, level.expected_return.value! * 100], symbolSize: level.level_code === selected ? 17 : 10, label: { show: true, formatter: '{b}', position: 'top', fontSize: 12 } })) },
    ],
  }), [result, selected, locale])
  const selectPoint = (event: any) => {
    if (event.data?.level) { setSelected(event.data.level); return }
    // Highlight a backend representative only; classification is never reimplemented here.
    const level = result.levels.find(item => item.representative_node_id === event.data?.nodeId)
    if (level) setSelected(level.level_code)
  }
  return <section className="min-w-0 space-y-3" aria-label={t('results')}>
    <div ref={tokens} aria-hidden="true" className="hidden"><span className="text-accent-600" /><span className="text-slate-600" /><span className="text-slate-200" /></div>
    <div className="flex flex-wrap items-center gap-2"><Badge>{t(`algorithm.${result.algorithm_id}`)}</Badge><Badge tone={preview.publication_eligibility.eligible ? 'success' : 'warning'}>{t(preview.publication_eligibility.eligible ? 'previewReady' : 'notPublishable')}</Badge></div>
    <p className="text-xs text-slate-600">{t('researchDate')}: {preview.request_echo.definition.research_as_of}</p>
    <p className="text-xs text-slate-600">{t('dataWindow', { start: String(sourceQuality.actual_start ?? t('unavailable')), end: String(sourceQuality.data_as_of ?? t('unavailable')), observations: typeof sourceQuality.observations === 'number' ? sourceQuality.observations : t('unavailable') })}</p>
    {!compact && (result.frontier.some(pointOk) ? themeReady && <div ref={chartContainer} className="min-w-0 w-full" data-testid="risk-frontier-chart" role="img" aria-label={t('chartDescription')}><p className="text-xs text-slate-600">{t('returnAxis')}</p><ReactECharts theme="risk-scales" option={option} notMerge style={{ height: 280, width: '100%' }} onChartReady={instance => { chartInstance.current = instance; if (chartContainer.current) instance.resize({ width: chartContainer.current.clientWidth }) }} onEvents={{ click: selectPoint }} /></div> : <EmptyState mascot={false} title={t('noFrontier')} hint={t('noFrontierHint')} />)}
    <p className="text-sm text-slate-600">{t('capMeaning')}</p><p className="text-xs text-slate-600">{t('representativeMeaning')}</p>
    <p className="text-xs text-slate-600">{t(result.diagnostics?.representative_portfolio_rebalance === 'monthly' ? 'portraitMonthlyRebalance' : 'portraitRebalanceUnknown')}</p>
    <p className="text-xs text-slate-600 sm:hidden">{t('horizontalTableHint')}</p>
    <div className="overflow-x-auto" tabIndex={0} role="region" aria-label={t('horizontalTableHint')}><table className="w-full text-sm" aria-label={t('levels')}><caption className="sr-only">{t('tableHelp')}</caption><thead><tr>{['level', 'riskInterval', 'cap', 'expectedReturn', 'volatility', 'calibration'].map(key => <th scope="col" key={key} className={`px-2 py-1.5 ${['riskInterval', 'cap', 'expectedReturn', 'volatility'].includes(key) ? 'text-right' : 'text-left'}`}>{t(key)}</th>)}</tr></thead><tbody>{result.levels.map(level => <tr key={level.level_code} className={`border-b border-slate-200 ${selected === level.level_code ? 'bg-accent-50' : ''}`} onClick={() => setSelected(level.level_code)}><th scope="row" className="px-2 py-1.5 text-left"><Button aria-pressed={selected === level.level_code} onClick={() => setSelected(level.level_code)}>{level.level_code}</Button></th><td className="whitespace-nowrap px-2 py-1.5 text-right tabular-nums">{level.lower_inclusive ? '[' : '('}{pct(level.lower_bound)}, {pct(level.upper_bound)}]</td><td className="px-2 py-1.5 text-right tabular-nums">{pct(level.authorized_volatility_cap)}</td><td className="px-2 py-1.5 text-right tabular-nums">{level.representative_node_id == null ? t('unavailable') : pct(level.expected_return.value)}</td><td className="px-2 py-1.5 text-right tabular-nums">{level.representative_node_id == null ? t('unavailable') : pct(level.volatility.value)}</td><td className="px-2 py-1.5 text-xs">{t(`calibration.${level.calibration_status}`)}</td></tr>)}</tbody></table></div>
    <Problems items={preview.publication_eligibility.blockers} /><Problems items={preview.warnings} />
    {result.fallback_reason && <p className="text-sm text-amber-900">{t('fallback', { reason: result.fallback_reason })}</p>}
    {!compact && <details><summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">{t('portraitDetails', { level: selected })}</summary>{active?.representative_node_id == null ? <p className="text-sm text-slate-600">{t('noRepresentative')}</p> : <div className="space-y-3"><dl className="grid gap-3 text-sm sm:grid-cols-2">{result.ordered_asset_ids.map((asset, index) => <div key={asset}><dt>{textValue(metadata(result.diagnostics?.asset_names)[asset]) || asset}</dt><dd className="tabular-nums">{pct(active.representative_weights?.[index])}</dd></div>)}</dl><p className="text-sm">{t('historicalEs')}: {pct(active.historical_es.value)} · {t(`metricUnit.${active.historical_es.unit ?? 'decimal'}`)} · {active.historical_es.reason || t(active.historical_es.status === 'available' ? 'available' : 'unavailable')}</p><p className="text-sm">{t('historicalMdd')}: {pct(active.historical_mdd.value)} · {t(`metricUnit.${active.historical_mdd.unit ?? 'decimal'}`)} · {active.historical_mdd.reason || t(active.historical_mdd.status === 'available' ? 'available' : 'unavailable')}</p><p className="text-xs text-slate-600">{t('representativeMeaning')}</p></div>}</details>}
  </section>
}
