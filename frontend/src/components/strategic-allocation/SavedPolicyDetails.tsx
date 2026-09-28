import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import * as echarts from 'echarts'
import { useI18n, systemText } from '../../i18n/runtime'
import type { StrategicBaseline } from '../../services/strategicAllocation'
import { actionClass, Badge, Card, DataTable } from '../ui'
import { percentText } from '../risk-models/ResearchUI'
import { GoalCandidateSummary } from '../investment-mandate/MandateResults'
import CompatibilityResults from './CompatibilityResults'
import CrossModelResults from './CrossModelResults'
import { MeanUncertaintySummary } from './MeanUncertaintyFields'
import { MODEL_COLOR_CLASSES } from './policyFrontierPresentation'

const linkClass = 'inline-flex min-h-10 items-center break-words text-sm font-medium text-accent-700 underline'
const summaryClass = 'min-h-10 cursor-pointer py-2 text-sm font-semibold text-slate-700'

/** Presentation reads only the selected frozen baseline; it never resolves newer source versions. */
export default function SavedPolicyDetails({ baseline, newPolicyHref }: { baseline: StrategicBaseline; newPolicyHref: string }) {
  const { s } = useI18n()
  const policy = baseline.policy, selection = policy?.selection, mandate = policy?.mandate
  const common = policy?.mode === 'compatible_all_models'
  const modeLabel = s(common ? 'multiCma.common' : policy?.mode === 'parameter_average' ? 'multiCma.average' : 'saaCenter.single')
  const sources: Array<{ cma_id: string; name: string; as_of?: string; weight?: number | null }> = policy?.multi_cma?.sources ?? (policy?.cma_id ? [{ cma_id: policy.cma_id,
    name: policy.assumptions?.name || s('saaDetail.cmaVersion'), as_of: policy.assumptions?.as_of, weight: 1 }] : [])
  const total = baseline.assets.length ? baseline.assets.reduce((sum, asset) => sum + asset.base_weight, 0) : null
  const cash = mandate?.effective_cash_reserve_weight ?? mandate?.min_cash_weight
  const goalKind = mandate?.objective_kind ?? 'absolute_return'
  const targetAmount = mandate?.funding_target?.amount ?? mandate?.funding_plan?.terminal_target
  const targetLabel = systemText(goalKind === 'funding_goal' ? 'preInvestment.scopeMandateSummary.terminalFundingTarget'
    : goalKind === 'benchmark_relative' ? 'preInvestment.scopeMandateSummary.targetAnnualExcessReturn'
    : mandate?.target_return_basis === 'annual_compound' ? 'saaDetail.compoundTarget' : 'saaDetail.returnTarget')
  const targetValue = goalKind === 'funding_goal' ? (targetAmount == null ? s('saaDetail.notRecorded') : `${targetAmount.toLocaleString()} ${mandate?.currency ?? ''}`)
    : percentText(goalKind === 'benchmark_relative' ? mandate?.benchmark?.target_excess_return ?? mandate?.target_excess_return : mandate?.target_return)
  return <>
    <header className="space-y-2">
      <div className="flex flex-wrap items-center gap-3"><p className="text-sm font-medium text-slate-600">{s('saaDetail.title')}</p><Badge>{s('saaDetail.saved')}</Badge></div>
      <h1 className="break-words text-2xl font-bold text-slate-900 sm:text-3xl">{baseline.name}</h1>
      <p className="text-sm leading-6 text-slate-600 tabular-nums">{s('saaCenter.researchDate')} {baseline.as_of}{mandate?.currency ? ` · ${mandate.currency}` : ''}</p>
      <p className="text-xs leading-5 text-slate-600">{s('saaDetail.frozenHint')}</p>
    </header>

    <Card className="min-w-0">
      <div className="grid gap-6 xl:grid-cols-2">
        <section className="min-w-0" aria-label={s('saaDetail.weights')}>
          <div className="flex flex-wrap items-baseline justify-between gap-2"><h2 className="text-lg font-semibold">{s('saaDetail.weights')}</h2>
            <p className="text-sm text-slate-600 tabular-nums">{s('saaDetail.total')} <strong className="text-slate-900">{percentText(total)}</strong></p></div>
          {baseline.assets.length ? <div className="mt-3 grid items-center gap-4 sm:grid-cols-[180px_minmax(0,1fr)]">
            <WeightChart assets={baseline.assets} />
            <dl className="min-w-0 divide-y divide-slate-200">{baseline.assets.map((asset, index) => <div key={asset.id} className="flex items-center justify-between gap-3 py-3">
              <dt className="flex min-w-0 items-start gap-2 text-sm font-medium"><span aria-hidden="true" className={`${MODEL_COLOR_CLASSES[index % MODEL_COLOR_CLASSES.length]} mt-1 inline-block h-3 w-3 shrink-0 rounded-sm bg-current`} /><span className="break-words">{asset.name}</span></dt>
              <dd className="shrink-0 text-lg font-semibold tabular-nums text-slate-900">{percentText(asset.base_weight)}</dd>
            </div>)}</dl>
          </div> : <p className="mt-4 text-sm text-slate-600">{s('saaDetail.noWeights')}</p>}
        </section>
        <section aria-label={s('saaDetail.outcome')} className="min-w-0 border-t border-slate-200 pt-5 xl:border-l xl:border-t-0 xl:pl-6 xl:pt-0">
          <h2 className="text-lg font-semibold">{s('saaDetail.outcome')}</h2>
          <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-4">
            {[[s(common ? 'saaDetail.minimumReturn' : 'multiCma.return'), percentText(selection?.metrics.expected_return)],
              [s(common ? 'saaDetail.maximumRisk' : 'multiCma.risk'), percentText(selection?.metrics.volatility)]].map(([label, value]) => <div key={label} className="min-w-0">
              <dt className="text-xs leading-5 text-slate-600">{label}</dt><dd className="mt-1 text-2xl font-bold tabular-nums text-slate-900">{value}</dd>
            </div>)}
          </dl>
          <p className="mt-3 text-xs leading-5 text-slate-600">{s(!selection ? 'saaDetail.noMetrics' : common ? 'saaDetail.commonMetricsHint' : 'saaDetail.metricsHint')}</p>
          <dl className="mt-4 grid gap-x-4 gap-y-3 border-t border-slate-200 pt-4 sm:grid-cols-3">
            <div className="min-w-0 sm:col-span-3"><dt className="text-xs text-slate-600">{s('saaCenter.mandate')}</dt><dd>{policy?.mandate_id
              ? <Link className={linkClass} to={`/pre-investment/objectives/new?view=${encodeURIComponent(policy.mandate_id)}`}>{mandate?.name || s('saaDetail.viewObjective')}</Link>
              : <span className="text-sm text-slate-600">{s('saaDetail.notRecorded')}</span>}</dd></div>
            <div className="min-w-0"><dt className="text-xs text-slate-600">{targetLabel}</dt><dd className="mt-1 break-words text-sm font-semibold tabular-nums">{targetValue}</dd></div>
            <div><dt className="text-xs text-slate-600">{s('saaDetail.riskCap')}</dt><dd className="mt-1 text-sm font-semibold tabular-nums">{percentText(mandate?.max_volatility)}</dd></div>
            <div><dt className="text-xs text-slate-600">{s('saaDetail.cashFloor')}</dt><dd className="mt-1 text-sm font-semibold tabular-nums">{percentText(cash)}</dd></div>
          </dl>
        </section>
      </div>
    </Card>

    {policy && <Card className="min-w-0 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-lg font-semibold">{s('saaDetail.researchBasis')}</h2><Badge>{modeLabel}</Badge></div>
      <p className="text-sm leading-6 text-slate-600">{s(common ? 'saaDetail.commonHint' : policy.mode === 'parameter_average' ? 'saaDetail.averageHint' : 'saaDetail.singleHint')}</p>
      {sources.length > 0 && <section aria-label={s(common ? 'multiCma.commonFrozen' : 'multiCma.frozen')}><DataTable caption={s('saaDetail.sources')} rows={sources} rowKey={source => source.cma_id} empty={s('saaDetail.notRecorded')} minWidth="400px" columns={[
        { header: 'LTCMA', cell: source => <Link className={`${linkClass} leading-6`} to={`/pre-investment/ltcma/${encodeURIComponent(source.cma_id)}`}>{source.name}</Link> },
        { header: s('saaCenter.researchDate'), nowrap: true, cell: source => source.as_of || s('saaDetail.notRecorded') },
        { header: s(common ? 'multiCma.requiredColumn' : 'multiCma.weight'), numeric: !common, cell: source => common ? s('multiCma.required') : percentText(source.weight) },
      ]} /></section>}
      <section className="border-t border-slate-200 pt-4" aria-label={s('saaDetail.reason')}>
        <h3 className="text-sm font-semibold">{s('saaDetail.reason')}</h3>
        <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6 text-slate-700">{policy.reason || s('saaDetail.notRecorded')}</p>
      </section>
      {selection?.cross_model_results && selection.cross_model_results.length > 0 && <details className="border-t border-slate-200 pt-2">
        <summary className={summaryClass}>{s('saaDetail.modelChecks')}</summary><div className="pb-2 pt-3"><CrossModelResults common={common} rows={selection.cross_model_results} /></div>
      </details>}
      {selection?.goal_check && <details className="border-t border-slate-200 pt-2"><summary className={summaryClass}>{s('saaDetail.fundingChecks')}</summary><GoalCandidateSummary candidate={selection} /></details>}
      {(policy.compatibility || policy.uncertainty_model) && <details className="border-t border-slate-200 pt-2">
        <summary className={summaryClass}>{s('saaDetail.technicalDetails')}</summary><div className="space-y-5 pb-2 pt-3">
          {policy.compatibility && <CompatibilityResults evidence={policy.compatibility} />}
          {policy.uncertainty_model && <MeanUncertaintySummary value={policy.uncertainty_model} />}
        </div>
      </details>}
      <details className="border-t border-slate-200 pt-2"><summary className={summaryClass}>{systemText('preInvestment.strategicAllocationWorkspace.objectiveAndAssumptionVersions')}</summary>
        <dl className="grid gap-3 py-3 text-xs sm:grid-cols-2"><div><dt className="text-slate-600">{s('saaCenter.mandate')}</dt><dd className="mt-1 break-all tabular-nums">{policy.mandate_id}</dd></div>
          <div><dt className="text-slate-600">LTCMA</dt><dd className="mt-1 break-all tabular-nums">{sources.map(source => source.cma_id).join(' · ') || s('saaDetail.notRecorded')}</dd></div></dl>
      </details>
    </Card>}

    <footer className="flex flex-wrap gap-3 border-t border-slate-200 pt-4">
      <Link className={actionClass('primary', 'max-w-full !whitespace-normal py-2 text-center motion-reduce:transition-none')} to={`/pre-investment/taa?baseline=${encodeURIComponent(baseline.id)}`}>{systemText('preInvestment.strategicAllocationWorkspace.returnToTaaResearchForThisPolicy')}</Link>
      <Link className={actionClass('secondary', 'max-w-full !whitespace-normal py-2 text-center motion-reduce:transition-none')} to={`/pre-investment/product-allocation-timing?source=${encodeURIComponent(baseline.id)}`}>{s('implementation.handoff')}</Link>
      <Link className={actionClass('secondary', 'max-w-full !whitespace-normal py-2 text-center motion-reduce:transition-none')} to={newPolicyHref}>{systemText('preInvestment.strategicAllocationWorkspace.createANewPolicyWithThisObjective')}</Link>
    </footer>
  </>
}

function WeightChart({ assets }: { assets: StrategicBaseline['assets'] }) {
  const { s } = useI18n()
  const tokens = useRef<HTMLDivElement>(null), container = useRef<HTMLDivElement>(null), chart = useRef<ReactECharts>(null)
  const [ready, setReady] = useState(false)
  useEffect(() => {
    const colors = Array.from(tokens.current?.children ?? []).map(node => getComputedStyle(node).color)
    echarts.registerTheme('saa-weights', { color: colors, backgroundColor: 'transparent' })
    setReady(true)
  }, [])
  useEffect(() => {
    if (!container.current || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => chart.current?.getEchartsInstance().resize())
    observer.observe(container.current)
    return () => observer.disconnect()
  }, [ready])
  const valid = assets.every(asset => Number.isFinite(asset.base_weight) && asset.base_weight >= 0) && assets.some(asset => asset.base_weight > 0)
  return <div ref={container} className="min-w-0">
    <div ref={tokens} className="hidden" aria-hidden="true">{MODEL_COLOR_CLASSES.map(token => <span key={token} className={token} />)}</div>
    {ready && valid && <div role="img" aria-label={s('saaDetail.chartDescription')} data-testid="saa-weight-chart">
      <ReactECharts ref={chart} theme="saa-weights" notMerge opts={{ renderer: 'svg' }} style={{ height: 180, width: '100%' }} option={{
        animation: false, tooltip: { show: false },
        series: [{ type: 'pie', radius: ['62%', '88%'], silent: true, label: { show: false },
          itemStyle: { borderWidth: 2, borderColor: 'white' },
          data: assets.map(asset => ({ name: asset.name, value: asset.base_weight })) }],
      }} />
    </div>}
  </div>
}
