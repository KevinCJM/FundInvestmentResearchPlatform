import { useEffect, useMemo, useRef, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { Badge } from '../ui'
import { percentText } from '../risk-models/ResearchUI'
import type { MandateAssessment } from '../../services/strategicAllocation'
import { useMandateText } from './text'

const usable = (point: { expected_return: number | null; volatility: number | null; status?: string; solver_status?: string }) =>
  (point.status ?? point.solver_status) === 'optimal_to_tolerance'
  && typeof point.expected_return === 'number' && Number.isFinite(point.expected_return)
  && typeof point.volatility === 'number' && Number.isFinite(point.volatility)

const xy = (point: { expected_return: number | null; volatility: number | null }) => [point.volatility! * 100, point.expected_return! * 100]

export default function MandateReferenceResults({ value }: { value: MandateAssessment }) {
  const { t } = useMandateText()
  const risk = value.risk_decision, result = value.reference_diagnosis
  const reference = result?.reference_frontier ?? [], constrained = result?.constrained_frontier ?? []
  const boundaries = result?.risk_boundaries ?? risk?.applied_boundaries ?? []
  const reach = result?.reachability
  const hasChart = reference.some(usable) && constrained.some(usable)
  // 本页填的目标和上限就是这张图的两条参考线；只写在图下面的文字里，读者还得自己把数字映射回坐标。
  const level = risk?.selected_max_level ?? risk?.authorized_max_level ?? null
  const cap = reach?.volatility_cap ?? risk?.selected_volatility_cap ?? null
  // 后端把相对目标换成同模型绝对收益；复利目标则按各风险位置换算。
  const compoundCurve = result?.return_requirements?.compound_floor != null ? result.target_curve ?? [] : []
  const target = reach?.target_return ?? null
  const capText = percentText(cap), targetText = percentText(target)

  // 图表配色从 Tailwind 令牌读出，不在源码里再添十六进制字面量（准则 10.1）。
  const tokens = useRef<HTMLDivElement>(null)
  const [paint, setPaint] = useState<string[]>([])
  useEffect(() => { setPaint(Array.from(tokens.current?.children ?? []).map(node => getComputedStyle(node).color)) }, [])
  const tone = (index: number) => paint[index] || undefined
  const accent = tone(0), muted = tone(1), mask = tone(2), reached = tone(3), missed = tone(4)

  const capMeaning = cap == null ? '' : t('capLineMeaning', { value: capText })
  const targetMeaning = target == null ? '' : t('targetLineMeaning', { value: targetText })
  const option = useMemo(() => {
    const edge = Math.max(...[...reference, ...constrained].filter(usable).map(point => point.volatility!), ...boundaries, cap ?? 0) * 100
    const emphasis = { lineStyle: { color: accent, width: 2, type: 'solid' }, label: { color: accent } }
    // 上限通常就落在某一条等级线上。那条线换成强调样式即可，另画一条只会让两个标签叠在一起。
    const atCap = (bound: number) => cap != null && Math.abs(bound - cap) < 1e-12
    const lines: Record<string, unknown>[] = boundaries.map((bound, index) => atCap(bound)
      ? { name: t('capLine', { level: `C${index + 1}`, value: capText }), xAxis: bound * 100, ...emphasis }
      : { name: `C${index + 1}`, xAxis: bound * 100, lineStyle: { color: mask, width: 1, type: 'dashed' }, label: { color: muted } })
    if (cap != null && !boundaries.some(atCap)) lines.push({ ...emphasis,
      name: t('capLine', { level: level ? `C${level}` : '', value: capText }), xAxis: cap * 100 })
    if (target != null && !compoundCurve.length) lines.push({ name: t('targetLine', { value: targetText }),
      yAxis: target * 100, lineStyle: { color: accent, width: 2, type: 'dashed' }, label: { color: accent, position: 'insideStartTop' } })

    const points: Record<string, unknown>[] = []
    if (cap != null && reach?.max_return_under_cap != null) points.push({
      name: t('reachableMark', { value: percentText(reach.max_return_under_cap) }), coord: [cap * 100, reach.max_return_under_cap * 100],
      itemStyle: { color: reach.binding === 'none' ? reached : missed } })
    if (target != null && reach?.binding === 'volatility_cap' && reach.min_volatility_for_target != null) points.push({
      name: t('shortfallMark', { value: percentText(reach.min_volatility_for_target),
        level: reach.required_risk_level ? `C${reach.required_risk_level}` : t('outsideScale') }),
      coord: [reach.min_volatility_for_target * 100, target * 100], itemStyle: { color: missed } })

    return {
      animation: false,
      aria: { enabled: true, description: [t('frontierChartDescription'), capMeaning, targetMeaning].filter(Boolean).join(' ') },
      grid: { left: 16, right: 24, bottom: 48, top: 34, containLabel: true },
      tooltip: { trigger: 'item', valueFormatter: (number: number) => `${number.toFixed(2)}%` },
      legend: { data: [t('referenceFrontier'), t('constrainedFrontier'), ...(cap == null ? [] : [t('selectableSegment')]), ...(compoundCurve.length ? [t('requiredReturnCurve')] : [])] },
      xAxis: { type: 'value', name: t('volatilityAxis'), nameLocation: 'middle', nameGap: 32, axisLabel: { formatter: '{value}%' }, scale: true },
      yAxis: { type: 'value', name: t('returnAxis'), axisLabel: { formatter: '{value}%' }, scale: true },
      series: [
        { name: t('referenceFrontier'), type: 'line', showSymbol: false, lineStyle: { type: 'dashed', width: 2 },
          data: reference.filter(usable).map(xy),
          markLine: { silent: true, symbol: 'none', label: { formatter: '{b}', fontSize: 12 }, data: lines },
          // 授权之外的那段前沿本次选不了，压成灰底，别和可选区抢同样的视觉权重。
          markArea: cap == null ? undefined : { silent: true, itemStyle: { color: mask, opacity: .45 },
            label: { show: true, position: 'insideTop', color: muted, fontSize: 12, formatter: t('beyondAuthorization') },
            data: [[{ xAxis: cap * 100 }, { xAxis: edge }]] } },
        { name: t('constrainedFrontier'), type: 'line', showSymbol: true, symbolSize: 5, lineStyle: { type: 'solid', width: 3 },
          data: constrained.filter(usable).map(xy),
          markPoint: points.length ? { silent: true, symbol: 'circle', symbolSize: 10,
            label: { show: true, position: 'top', fontSize: 12, color: muted, formatter: '{b}' }, data: points } : undefined },
        ...(compoundCurve.length ? [{ name: t('requiredReturnCurve'), type: 'line', showSymbol: false,
          itemStyle: { color: missed }, lineStyle: { type: 'dashed', width: 2 },
          data: compoundCurve.map(point => point.expected_return == null ? null : xy(point)) }] : []),
        // 授权内可选的那一段单独成序列并进图例，这样加粗有名字，不会被读成"绿线换了颜色"。
        ...(cap == null ? [] : [{ name: t('selectableSegment'), type: 'line', silent: true, showSymbol: false,
          itemStyle: { color: accent }, lineStyle: { color: accent, width: 7, opacity: .35 },
          data: constrained.map(point => usable(point) && point.volatility! <= cap ? xy(point) : null) }]),
      ],
    }
  }, [reference, constrained, compoundCurve, boundaries, reach, cap, target, level, capText, targetText, capMeaning, targetMeaning, accent, muted, mask, reached, missed, t])
  if (!risk) return null
  const validated = result?.status === 'validated'
  const selectedLevel = risk.selected_max_level ?? risk.authorized_max_level
  const cash = result?.cash_constraint
  return <section className="space-y-5" aria-label={t('referenceDiagnosis')}>
    <div className="flex flex-wrap items-center gap-2"><h3 className="text-lg font-semibold">{t('objectiveFeasibility')}</h3>
      <Badge tone={validated ? 'success' : result ? 'warning' : 'neutral'}>{validated ? t('feasible') : result ? t('needsRevision') : t('notCalculated')}</Badge>
    </div>
    <p className="text-sm leading-6 text-slate-600">{t('referenceScopeSimple')}</p>
    <dl className="grid gap-4 sm:grid-cols-3">
      <div><dt className="text-xs text-slate-600">{t('maxRiskLevel')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{selectedLevel ? `C${selectedLevel}` : '—'}</dd></div>
      <div><dt className="text-xs text-slate-600">{t('minimumTestedLevel')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{result?.minimum_tested_feasible_level ? `C${result.minimum_tested_feasible_level}` : t('notValidated')}</dd></div>
      <div><dt className="text-xs text-slate-600">{t('selectedCap')}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(risk.selected_volatility_cap ?? risk.authorized_volatility_cap)}</dd></div>
    </dl>
    {result?.blockers.map((text, index) => <p key={index} className="text-sm leading-6 text-amber-800">{text}</p>)}

    {reach && <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <h4 className="text-sm font-semibold">{t('reachability')}</h4>
      <p className="mt-1 text-sm leading-6 text-slate-800">{t('reachableUnderCap', {
        cap: percentText(reach.volatility_cap), reachable: percentText(reach.max_return_under_cap) })}</p>
      {compoundCurve.length > 0 && <p className="mt-2 text-sm leading-6 text-slate-700">{t(reach.binding === 'none' ? 'compoundReachable' : 'compoundNotReached')}</p>}
      {!compoundCurve.length && reach.binding === 'volatility_cap' && <p className="mt-2 text-sm leading-6 text-amber-800">{t('bindingVolatilityCap', {
        target: percentText(reach.target_return), volatility: percentText(reach.min_volatility_for_target),
        level: reach.required_risk_level ? `C${reach.required_risk_level}` : t('outsideScale') })}</p>}
      {!compoundCurve.length && reach.binding === 'unreachable_at_any_level' && <p className="mt-2 text-sm leading-6 text-amber-800">{t('bindingUnreachable', { target: percentText(reach.target_return) })}</p>}
      {reach.binding === 'none' && reach.target_return != null && <p className="mt-2 text-sm leading-6 text-slate-700">{t('bindingNone', { target: percentText(reach.target_return) })}</p>}
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('reachabilityBasis')}</p>
    </div>}

    {cash && <div className="border-t border-slate-200 pt-4"><h4 className="text-base font-semibold">{t('effectiveCashConstraint')}</h4>
      <dl className="mt-3 grid gap-4 sm:grid-cols-3"><div><dt className="text-xs text-slate-600">{t('requestedCash')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.requested_min_cash_weight)}</dd></div>
        <div><dt className="text-xs text-slate-600">{t('cashflowDerived')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.cashflow_derived_weight)}</dd></div>
        <div><dt className="text-xs text-slate-600">{t('effectiveCash')}</dt><dd className="mt-1 text-base font-semibold tabular-nums">{percentText(cash.effective_min_cash_weight)}</dd></div></dl>
      <p className="mt-2 text-xs leading-5 text-slate-600">{t('cashConstraintRule')}</p>
    </div>}

    <div className="border-t border-slate-200 pt-4"><div className="mb-3"><h4 className="text-base font-semibold">{t('twoFrontiers')}</h4><p className="mt-1 text-xs leading-5 text-slate-600">{t('fixedBandsHint')}</p></div>
      <div ref={tokens} aria-hidden="true" className="hidden"><span className="text-accent-600" /><span className="text-slate-600" /><span className="text-slate-200" /><span className="text-emerald-600" /><span className="text-rose-700" /></div>
      {compoundCurve.length > 0 && <p className="mb-3 text-xs leading-5 text-slate-600">{t('requirementsHelp')}</p>}
      {hasChart ? <><ReactECharts option={option} notMerge style={{ height: 320, width: '100%' }} /><div className="mt-2 flex flex-wrap gap-4 text-xs text-slate-600"><span>{t('referenceLineMeaning')}</span><span>{t('constrainedLineMeaning')}</span>{capMeaning && <span>{capMeaning}</span>}{targetMeaning && <span>{targetMeaning}</span>}</div></>
        : <p className="text-sm text-slate-600">{t('frontierUnavailable')}</p>}
    </div>

    {result && <details className="border-t border-slate-200 pt-4"><summary className="min-h-10 cursor-pointer text-sm font-medium">{t('advancedEvidence')}</summary><div className="mt-3 space-y-3">
      <p className="text-xs leading-5 text-slate-600">{t('searchValidationEvidence', { search: result.search_seed, validation: result.validation_seed, paths: result.paths })}</p>
      <div className="overflow-x-auto"><table className="w-full min-w-[480px] text-sm" aria-label={t('referenceCandidates')}><caption className="sr-only">{t('referenceCandidates')}</caption><thead><tr>{['candidate', 'referenceReturn', 'volatility', 'level'].map(key => <th key={key} scope="col" className="p-2 text-right first:text-left">{t(key)}</th>)}</tr></thead><tbody>{result.candidates.map(candidate => <tr key={candidate.id} className="border-b border-slate-200"><th scope="row" className="p-2 text-left font-normal">{candidate.id}</th><td className="p-2 text-right tabular-nums">{percentText(candidate.expected_return)}</td><td className="p-2 text-right tabular-nums">{percentText(candidate.volatility)}</td><td className="p-2 text-right">{candidate.risk_level ? `C${candidate.risk_level}` : '—'}</td></tr>)}</tbody></table></div>
      {result.limitations.map((text, index) => <p key={index} className="text-xs leading-5 text-slate-600">{text}</p>)}
    </div></details>}
  </section>
}
