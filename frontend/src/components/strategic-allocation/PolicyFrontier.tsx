import { researchMessage } from '../../i18n/researchMessages'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import FrontierChart from './FrontierChart'
import * as echarts from 'echarts'
import { Badge, Button, DataTable, ErrorPanel, LoadingPanel } from '../ui'
import { percentText, sectionClass } from '../risk-models/ResearchUI'
import { getPolicyFrontier, type PolicyFrontierResult, type PolicyRequest, type PolicyPreview } from '../../services/strategicAllocation'
import { useI18n, systemText } from '../../i18n/runtime'
import FrontierLegend, { type FrontierLegendItem } from './FrontierLegend'

import { frontierCandidatePoints, MODEL_COLOR_CLASSES, MODEL_SYMBOLS, type FrontierView } from './policyFrontierPresentation'

let themeRegistered = false

export default function PolicyFrontier({ request, disabled, clock, result, children }: {
  request: PolicyRequest; disabled: boolean; clock: string | null | undefined; result: PolicyPreview | null
  children?: (gate: { disabled: boolean; reason: string }) => ReactNode
}) {
  const { s } = useI18n()
  // Risk-budget scoring and mean uncertainty do not change this asset-return frontier.
  const query = JSON.stringify({ ...request, risk_budget: null, uncertainty_set: 'box',
    uncertainty_confidence: null, uncertainty_approximation_acknowledged: false })
  const key = JSON.stringify([query, clock])
  const [response, setResponse] = useState<{ key: string; data: PolicyFrontierResult } | null>(null)
  const [failure, setFailure] = useState<{ key: string; message: string } | null>(null)
  const [retry, setRetry] = useState(0)
  const [showReference, setShowReference] = useState(false)
  const [selected, setSelected] = useState<Record<string, boolean>>({})
  const tokens = useRef<HTMLDivElement>(null)
  const [paint, setPaint] = useState<string[]>([])
  useEffect(() => {
    const colors = Array.from(tokens.current?.children ?? []).map(node => getComputedStyle(node).color)
    if (!themeRegistered && colors.length === 5 + MODEL_COLOR_CLASSES.length) {
      echarts.registerTheme('saa-frontier', { color: [colors[1], colors[0], colors[2], colors[4]],
        textStyle: { color: colors[1], fontSize: 12 }, valueAxis: { axisLabel: { color: colors[1], fontSize: 12 } } })
      themeRegistered = true
    }
    setPaint(colors)
  }, [])
  useEffect(() => {
    if (disabled) return
    const controller = new AbortController()
    setFailure(null)
    const timer = setTimeout(() => {
      getPolicyFrontier(JSON.parse(query), controller.signal).then(data => {
        if (!controller.signal.aborted) setResponse({ key, data })
      }).catch(error => {
        if (!controller.signal.aborted) setFailure({ key, message: error instanceof Error ? error.message : systemText('preInvestment.policyFrontier.unableToCalculateTheFrontierPleaseRetry') })
      })
    }, 300)
    return () => { clearTimeout(timer); controller.abort() }
  }, [key, disabled, retry])
  const data = !disabled && response?.key === key ? response.data : null
  const error = failure?.key === key ? failure.message : ''
  const views = data?.views ?? [], view = views[0]
  const common = data?.mode === 'compatible_all_models', fused = data?.mode === 'parameter_average'
  const target = view?.target_return, cap = view?.volatility_cap
  const compound = view?.return_requirements?.compound_floor
  const hasCurves = views.some(v => v.target_curve?.some(p => p.expected_return != null))
  const curved = views.some(v => v.return_requirements?.compound_floor != null)
  const curvePoints = views.flatMap(v => v.target_curve ?? []).filter(p => p.expected_return != null)
  const blocked = data?.target_check?.status === 'infeasible'
  const pending = !disabled && !error && !data
  const reason = disabled ? '' : pending ? s('policyFrontier.checkingTarget')
    : blocked ? data?.target_check?.reason === 'constraint_conflict' ? s('policyFrontier.blockedBounds')
      : s(common ? 'policyFrontier.blockedCommon' : target == null ? 'policyFrontier.blockedRisk' : 'policyFrontier.blockedTarget', { target: percentText(target), cap: percentText(cap) })
    : error || data?.target_check?.status !== 'feasible' ? s('policyFrontier.targetUndetermined') : ''
  const modelIds = views.map(v => v.id).sort()
  const color = (v: FrontierView) => common ? paint[5 + modelIds.indexOf(v.id)] : paint[0]
  const modelName = (v: FrontierView) => `${views.indexOf(v) + 1}. ${v.name}`
  const configuredName = (v: FrontierView) => common ? modelName(v) : fused ? s('policyFrontier.fused') : systemText('preInvestment.policyFrontier.currentConstrainedFrontier')
  const referenceName = (v: FrontierView) => common ? s('policyFrontier.modelReference', { model: modelName(v) }) : systemText('preInvestment.policyFrontier.assetClassFrontier')
  const configuredId = (v: FrontierView) => `configured:${v.id}`
  const referenceId = (v: FrontierView) => `reference:${v.id}`
  const validPoints = views.flatMap(v => [...v.reference.points, ...v.configured.points]).filter(p => p.status === 'optimal_to_tolerance')
  const candidatePoints = frontierCandidatePoints(data, result, request)
  const candidateItems = candidatePoints.map((point, index) => ({ id: point.id,
    name: common ? s('policyFrontier.commonPoint', { model: modelName(views.find(v => v.id === point.modelId)!) })
      : s('frontierLegend.candidate', { index: index + 1, name: researchMessage(point.candidateName) }),
    symbolClass: 'h-2.5 w-2.5 rounded-full bg-current',
    symbolColor: common ? color(views.find(v => v.id === point.modelId)!) : paint[4],
  }))
  const items: FrontierLegendItem[] = view ? [
    ...views.map(v => ({ id: configuredId(v), name: configuredName(v), symbolClass: 'w-6 border-t-2 border-current', symbolColor: color(v) })),
    ...(showReference ? views.map(v => ({ id: referenceId(v), name: referenceName(v),
      symbolClass: 'w-6 border-t-2 border-dashed border-current', symbolColor: common ? color(v) : paint[1] })) : []),
    ...(target != null || cap != null ? [{ id: 'target', name: s('frontierLegend.target'), symbolClass: 'h-2.5 w-2.5 rotate-45 bg-rose-700',
      detail: [target == null ? '' : `${s('scopeFeasibility.chart.targetLine')} ${percentText(target)}`,
        cap == null ? '' : `${s('scopeFeasibility.chart.capLine')} ${percentText(cap)}`].filter(Boolean).join(' · ') }] : []),
    ...candidateItems,
  ] : []
  const ys = [...curvePoints.map(p => p.expected_return!), ...validPoints.map(p => p.expected_return!), ...candidatePoints.map(p => p.expected_return), target ?? 0, 0]
  const xs = [...validPoints.map(p => p.volatility!), ...candidatePoints.map(p => p.volatility), cap ?? 0, 0]
  const yLow = Math.min(...ys) * 100, yHigh = Math.max(...ys) * 100
  const yPadding = Math.max((yHigh - yLow) * .18, .5)
  const xmax = Math.ceil(Math.max(Math.max(...xs) * 110, 1))
  const ymax = Math.ceil(yHigh + yPadding)
  const description = systemText('preInvestment.policyFrontier.theHorizontalAxisIsAnnualVolatilityThe', { p0: percentText(target), p1: percentText(cap) })
  const option = {
    animation: false,
    legend: { show: false, selected: { ...Object.fromEntries(items.map(item => [item.name, selected[item.id] !== false])),
      ...Object.fromEntries(views.map(v => [referenceName(v), showReference && selected[referenceId(v)] !== false])) } },
    aria: { enabled: true, description },
    grid: { left: 12, right: 24, top: 24, bottom: 52, containLabel: true },
    textStyle: { color: paint[1], fontSize: 12 },
    tooltip: { trigger: 'item', formatter: (p: { seriesName: string; name?: string; value?: number[]; data?: { weights?: Record<string, number> } }) => {
      if (!Array.isArray(p.value)) return ''
      // ECharts rich text avoids interpreting stored model/asset names as HTML.
      return systemText('preInvestment.policyFrontier.annualVolatilityExpectedAnnualReturn', { p0: p.name || p.seriesName, p1: p.value[0].toFixed(2), p2: p.value[1].toFixed(2) })
    }, renderMode: 'richText', confine: true },
    xAxis: { type: 'value', min: 0, max: xmax, name: systemText('preInvestment.policyFrontier.annualVolatilityRisk'), nameLocation: 'middle', nameGap: 32,
      axisLabel: { formatter: (v: number) => `${Number(v.toFixed(2))}%` }, nameTextStyle: { color: paint[1], fontSize: 12 } },
    yAxis: { type: 'value', min: yLow - (yLow < 0 ? yPadding : 0), max: ymax,
      axisLabel: { formatter: (v: number) => `${Number(v.toFixed(2))}%` } },
    series: view ? [
      ...views.flatMap(v => [
        { id: referenceId(v), name: referenceName(v), type: 'line', symbolSize: 3, connectNulls: false,
          itemStyle: { color: common ? color(v) : paint[1] }, lineStyle: { type: 'dashed', width: 1 },
          data: v.reference.points.map(p => p.status === 'optimal_to_tolerance' ? [p.volatility! * 100, p.expected_return! * 100] : null) },
        { id: configuredId(v), name: configuredName(v), type: 'line', symbolSize: 5, connectNulls: false,
          symbol: MODEL_SYMBOLS[modelIds.indexOf(v.id) % MODEL_SYMBOLS.length],
          lineStyle: { width: 2 }, itemStyle: { color: color(v) },
          data: v.configured.points.map(p => p.status === 'optimal_to_tolerance' ? [p.volatility! * 100, p.expected_return! * 100] : null) },
      ]),
      { name: s('frontierLegend.target'), type: 'scatter', symbol: 'diamond', symbolSize: 15,
        itemStyle: { color: paint[2] },
        data: curved || common ? [] : target == null || cap == null ? [] : [[cap * 100, target * 100]],
        label: { show: true, position: 'top', formatter: systemText('preInvestment.policyFrontier.target'), color: paint[2], fontSize: 12 },
        markLine: { symbol: 'none', silent: true, label: { show: false }, lineStyle: { color: paint[2], type: 'dashed' },
          data: [...(cap == null ? [] : [{ xAxis: cap * 100 }]), ...(target == null || hasCurves ? [] : [{ yAxis: target * 100 }])] },
        markArea: target == null || cap == null || curved || common ? undefined : { silent: true, itemStyle: { color: paint[3], opacity: .4 },
          data: [[{ xAxis: 0, yAxis: target * 100 }, { xAxis: cap * 100, yAxis: ymax }]] } },
      ...views.filter(v => v.target_curve?.some(p => p.expected_return != null)).map(v => ({
        id: `required:${v.id}`, name: s('frontierLegend.target'), type: 'line', showSymbol: false,
        lineStyle: { type: 'dashed', width: 2 }, itemStyle: { color: common ? color(v) : paint[2] },
        data: v.target_curve!.map(p => p.expected_return == null ? null : [p.volatility * 100, p.expected_return * 100]),
      })),
      ...candidatePoints.map((candidate, index) => ({ id: candidate.id, name: candidateItems[index].name, type: 'scatter', symbolSize: 13,
        symbol: 'circle', itemStyle: { color: candidateItems[index].symbolColor, borderColor: paint[1], borderWidth: 1 },
        data: [{ name: candidateItems[index].name, value: [candidate.volatility * 100, candidate.expected_return * 100] }] })),
    ] : [],
  }
  function conclusion(v: FrontierView) {
    const upper = v.configured.max_return
    const impossible = v.return_requirements?.compound_floor == null && v.target_return != null && upper != null && upper < v.target_return - 1e-8
    const intersects = v.configured.points.some(p => p.status === 'optimal_to_tolerance' && v.volatility_cap != null && p.volatility! <= v.volatility_cap + 1e-10 && (p.return_check ? p.return_check.within_limits : v.target_return == null || p.expected_return! >= v.target_return - 1e-10))
    return v.constraint_error || (v.configured.status === 'infeasible_certified' ? systemText('preInvestment.policyFrontier.weightCashOrOtherConstraintsConflictSo')
      : impossible ? systemText('preInvestment.policyFrontier.theTargetAnnualReturnIsUnderCurrent', { p0: percentText(v.target_return), p1: percentText(upper) })
      : intersects ? systemText('preInvestment.policyFrontier.verifiedPointsOnTheConstrainedFrontierEnter') : systemText('preInvestment.policyFrontier.noVerifiedFrontierPointEntersTheTarget'))
  }
  return <><section className={`${sectionClass} min-w-0 space-y-4`} aria-label={systemText('preInvestment.policyFrontier.objectivesAndEfficientFrontier')}>
    <div ref={tokens} aria-hidden="true" className="hidden"><span className="text-accent-600" /><span className="text-slate-600" /><span className="text-rose-700" /><span className="text-emerald-100" /><span className="text-emerald-700" />{MODEL_COLOR_CLASSES.map(token => <span key={token} className={token} />)}</div>
    <h2 className="text-lg font-semibold">{systemText('preInvestment.policyFrontier.objectivesAndEfficientFrontier')}</h2>
    <p className="text-sm leading-6 text-slate-600">{s(common ? 'policyFrontier.commonHelp' : fused ? 'policyFrontier.fusedHelp' : 'policyFrontier.singleHelp')}</p>
    {disabled ? <p className="text-sm text-slate-600">{systemText('preInvestment.policyFrontier.completeObjectiveAndLtcmaSelectionThenCheck')}</p> : error ? <ErrorPanel message={error} action={<Button onClick={() => setRetry(x => x + 1)}>{systemText('preInvestment.policyFrontier.recalculateFrontier')}</Button>} className="min-h-[350px]" /> : !view ? <LoadingPanel text={systemText('preInvestment.policyFrontier.calculatingTheEfficientFrontierForThisConfiguration')} className="min-h-[350px]" /> : <>
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
        <div><dt className="text-xs text-slate-600">{compound != null ? s('policyFrontier.compoundRequirement') : systemText('preInvestment.policyFrontier.minimumTargetAnnualReturn')}</dt><dd className="mt-1 font-semibold tabular-nums">{compound != null ? percentText(compound) : target == null ? systemText('preInvestment.policyFrontier.verifiedSeparatelyAgainstObjectiveConditions') : percentText(target)}</dd></div>
        <div><dt className="text-xs text-slate-600">{systemText('preInvestment.policyFrontier.maximumAnnualVolatility')}</dt><dd className="mt-1 font-semibold tabular-nums">{percentText(cap)}</dd></div>
        <div><dt className="text-xs text-slate-600">{systemText('preInvestment.policyFrontier.minimumCashRetained')}</dt><dd className="mt-1 font-semibold tabular-nums">{percentText(view.cash_floor)}</dd></div>
      </dl>
      {curved && <p className="text-sm leading-6 text-slate-600">{s('policyFrontier.returnCurveHelp')}</p>}
      {view.return_requirements?.benchmark_return != null && <p className="text-sm leading-6 text-slate-600">{s('policyFrontier.benchmarkRequirement')}</p>}
      <p className="text-xs text-slate-600">{systemText('preInvestment.policyFrontier.expectedAnnualReturn')}</p>
      <FrontierChart key={key} option={option} theme="saa-frontier" height={350} description={description} testId="saa-frontier-chart" />
      <label className="flex min-h-10 cursor-pointer items-center gap-2 text-sm text-slate-700">
        <input type="checkbox" className="accent-accent-600" checked={showReference} onChange={e => setShowReference(e.target.checked)} />
        {s('policyFrontier.showReference')}
      </label>
      {showReference && <p className="text-xs text-slate-600">{s('policyFrontier.referenceHelp')}</p>}
      <FrontierLegend items={items} selected={selected} onChange={setSelected} />
      {common && <p className="text-sm leading-6 text-slate-700">{s('policyFrontier.commonCaution')}</p>}
      {common && candidatePoints.length > 0 && <DataTable caption={s('policyFrontier.pointTable')} rows={candidatePoints} rowKey={p => p.id}
        minWidth="480px" empty={s('policyFrontier.noPoints')} columns={[
          { header: s('multiCma.source'), cell: p => modelName(views.find(v => v.id === p.modelId)!) },
          { header: s('multiCma.return'), numeric: true, cell: p => percentText(p.expected_return) },
          { header: s('multiCma.risk'), numeric: true, cell: p => percentText(p.volatility) },
          { header: s('multiCma.status'), cell: p => <Badge tone={p.withinLimits ? 'success' : 'warning'}>{s(`multiCma.${p.withinLimits ? 'pass' : 'fail'}`)}</Badge> },
        ]} />}
      {common && candidatePoints.length === 0 && <p className="text-xs text-slate-600">{s('policyFrontier.noPoints')}</p>}
      <div role="status" className="space-y-2 border-t border-slate-200 pt-3 text-sm leading-6 text-slate-700">
        {views.map(v => <p key={v.id}>{common && <strong>{modelName(v)}： </strong>}{conclusion(v)}</p>)}
      </div>
      {views.filter(v => !v.configured.complete || (showReference && !v.reference.complete)).map(v => <p key={v.id} className="text-xs text-amber-800">
        {common && `${modelName(v)}：`}{systemText('preInvestment.policyFrontier.someFrontierPointsCouldNotBeSolved')}
      </p>)}
      {(data!.additional_checks.benchmark || data!.additional_checks.funding) && <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.policyFrontier.theChartComparesAnnualReturnAndVolatility')}</p>}
      {!curved && !common && <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.policyFrontier.theChartUsesTheSelectedLtcmaAssumptions')}</p>}
    </>}
  </section>{children?.({ disabled: disabled || pending || blocked, reason })}</>
}
