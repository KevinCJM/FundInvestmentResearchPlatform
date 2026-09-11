import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import {
  diagnosePortfolioRun,
  downloadPortfolioExport,
  getPortfolioRun,
  getResearchTarget,
  listPortfolioIndicators,
  listPortfolioRuns,
  PortfolioDiagnosis,
  PortfolioExportTable,
  PortfolioMetric,
  PortfolioIndicatorDefinition,
  PortfolioRun,
  ResearchTarget,
  runPortfolioScenario,
} from '../services/portfolioResearch'
import { MetricSelector, MetricValue } from '../components/metrics/MetricDisplay'
import { useMetricDisplayPreference } from '../components/metrics/useMetricDisplayPreference'
import type { MetricPresentation } from '../services/customIndicators'
import { humanizeIndicatorMessage } from '../utils/indicatorDiagnostics'
import { RegimeConditioningPanel } from '../components/HistoricalRegimeBacktest'
import { PortfolioRiskSection } from '../components/risk-models/PublishedRiskPanel'

const percent = (value: number | null | undefined) => value === null || value === undefined || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(2)}%`
const metricPresentation = (metric: PortfolioMetric): MetricPresentation => metric.presentation ?? {
  indicator_id: metric.metric_id ?? metric.name, revision: 1, name: metric.name, source: 'built_in',
  category: 'portfolio_summary', category_label: '组合指标', context_kind: 'portfolio', catalog_status: 'current',
  display_format: metric.unit === 'percent' ? 'percent' : 'number', precision: 3, unit: metric.unit === 'percent' ? '%' : metric.unit ?? '',
  notation: 'standard', value_scale: metric.unit === 'percent' ? 100 : 1, output_measure: 'dimensionless', direction: metric.direction ?? 'higher_better',
  description: '', methodology: '', data_basis: '锁定运行快照', minimum_observations: 1, applicable_product_kinds: ['portfolio'],
}

function MatrixTable({ title, matrix }: { title: string; matrix?: { labels: string[]; values: number[][] } | null }) {
  if (!matrix?.labels.length) return <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">{title}</h2><p className="mt-2 text-sm text-slate-500">该快照没有足够的共同历史样本，无法计算矩阵。</p></section>
  const option = { tooltip: { position: 'top' }, xAxis: { type: 'category', data: matrix.labels, axisLabel: { rotate: 35 } }, yAxis: { type: 'category', data: matrix.labels }, visualMap: { min: Math.min(...matrix.values.flat()), max: Math.max(...matrix.values.flat()), calculable: true, orient: 'horizontal', left: 'center', bottom: 0 }, series: [{ type: 'heatmap', data: matrix.values.flatMap((row, y) => row.map((value, x) => [x, y, value])), label: { show: true, formatter: (params: { data: [number, number, number] }) => Number(params.data[2]).toFixed(2) } }] }
  return <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">{title}</h2><ReactECharts option={option} style={{ height: 340 }} notMerge lazyUpdate aria-label={`${title}热力图`} /><div className="overflow-x-auto"><table className="mt-2 w-full text-xs"><thead><tr><th className="p-2 text-left">产品</th>{matrix.labels.map((label) => <th className="p-2 text-right" key={label}>{label}</th>)}</tr></thead><tbody>{matrix.values.map((row, index) => <tr key={matrix.labels[index]} className="border-t"><th className="p-2 text-left">{matrix.labels[index]}</th>{row.map((value, valueIndex) => <td className="p-2 text-right" key={`${index}-${valueIndex}`}>{value.toFixed(4)}</td>)}</tr>)}</tbody></table></div></section>
}

export default function HoldingDiagnosis() {
  const [params] = useSearchParams()
  const targetId = params.get('target')
  const suppliedRunId = params.get('run')
  const [target, setTarget] = useState<ResearchTarget | null>(null)
  const [run, setRun] = useState<PortfolioRun | null>(null)
  const [diagnosis, setDiagnosis] = useState<PortfolioDiagnosis | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [exporting, setExporting] = useState<'csv' | 'zip' | null>(null)
  const [exportTable, setExportTable] = useState<PortfolioExportTable>('summary')
  const [scenario, setScenario] = useState({ name: '历史压力区间', start_date: '2020-01-01', end_date: '2020-03-31' })
  const [scenarioResult, setScenarioResult] = useState<{ name: string; metrics: PortfolioMetric[]; warnings?: string[]; start_date: string; end_date: string } | null>(null)
  const [scenarioLoading, setScenarioLoading] = useState(false)
  const [portfolioIndicators, setPortfolioIndicators] = useState<PortfolioIndicatorDefinition[]>([])
  const [indicatorLoading, setIndicatorLoading] = useState(false)
  const [indicatorError, setIndicatorError] = useState('')
  const [metricPreference, setMetricPreference] = useMetricDisplayPreference(
    'holding-diagnosis',
    'portfolio',
    [],
    'snapshot',
    portfolioIndicators.map((indicator) => indicator.id),
  )

  useEffect(() => {
    let active = true
    async function load() {
      if (!targetId && !suppliedRunId) { if (active) { setLoading(false); setError('请选择一个已保存的组合研究对象。') }; return }
      setLoading(true); setError('')
      try {
        const resolvedTarget = targetId ? await getResearchTarget(targetId) : null
        const runId = suppliedRunId ?? (await listPortfolioRuns()).items.find((item) => item.target_id === resolvedTarget?.id)?.id
        if (!runId) throw new Error('该研究对象尚未生成组合运行快照。')
        const [snapshot, detail] = await Promise.all([getPortfolioRun(runId), diagnosePortfolioRun(runId, [])])
        let indicators: PortfolioIndicatorDefinition[] = []
        try { indicators = await listPortfolioIndicators() } catch (caught: any) { if (active) setIndicatorError(humanizeIndicatorMessage(caught?.message, '组合指标目录加载失败')) }
        if (active) { setTarget(resolvedTarget); setRun(snapshot); setDiagnosis(detail); setPortfolioIndicators(indicators) }
      } catch (caught: any) { if (active) setError(humanizeIndicatorMessage(caught?.message, '持仓诊断加载失败')) } finally { if (active) setLoading(false) }
    }
    load()
    return () => { active = false }
  }, [suppliedRunId, targetId])

  const navOption = useMemo(() => ({ tooltip: { trigger: 'axis' }, legend: { data: ['组合净值', '回撤'] }, xAxis: { type: 'category', data: run?.nav.map((point) => point.date) ?? [] }, yAxis: [{ type: 'value', name: '净值' }, { type: 'value', name: '回撤', axisLabel: { formatter: '{value}%' } }], series: [{ name: '组合净值', type: 'line', smooth: true, data: run?.nav.map((point) => point.value) ?? [] }, { name: '回撤', type: 'line', yAxisIndex: 1, areaStyle: {}, data: run?.drawdown.map((point) => point.value * 100) ?? [] }] }), [run])
  const weightOption = useMemo(() => ({ tooltip: { trigger: 'axis' }, legend: { type: 'scroll' }, xAxis: { type: 'category', data: (diagnosis?.weight_path ?? run?.weights ?? []).map((point) => point.date) }, yAxis: { type: 'value', axisLabel: { formatter: (value: number) => `${(value * 100).toFixed(0)}%` } }, series: Object.keys((diagnosis?.weight_path ?? run?.weights ?? [])[0]?.weights ?? {}).map((key) => ({ name: key, type: 'line', stack: 'weights', areaStyle: {}, data: (diagnosis?.weight_path ?? run?.weights ?? []).map((point) => point.weights[key]) })) }), [diagnosis, run])
  const metrics = diagnosis && Array.isArray(diagnosis.summary) ? diagnosis.summary : []

  async function handleExport(format: 'csv' | 'zip') {
    if (!run) return
    setExporting(format); setError('')
    try {
      const includeScenario = Boolean(scenarioResult)
      const options = {
        ...(exportTable !== 'summary' ? { table: exportTable } : {}),
        ...(includeScenario ? { scenario_start: scenarioResult!.start_date, scenario_end: scenarioResult!.end_date } : {}),
      }
      if (Object.keys(options).length) await downloadPortfolioExport(run.id, format, options)
      else await downloadPortfolioExport(run.id, format)
    } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, '导出失败')) } finally { setExporting(null) }
  }
  async function handleScenario() { if (!run) return; setScenarioLoading(true); setError(''); try { setScenarioResult({ ...(await runPortfolioScenario(run.id, scenario)), start_date: scenario.start_date, end_date: scenario.end_date }) } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, '情景分析失败')) } finally { setScenarioLoading(false) } }
  async function refreshPortfolioIndicators() {
    if (!run || !metricPreference.indicatorIds.length) return
    setIndicatorLoading(true); setIndicatorError('')
    try { setDiagnosis(await diagnosePortfolioRun(run.id, metricPreference.indicatorIds)) } catch (caught: any) { setIndicatorError(humanizeIndicatorMessage(caught?.message, '组合指标计算失败')) } finally { setIndicatorLoading(false) }
  }

  if (loading) return <div className="mx-auto max-w-7xl p-6" role="status">正在加载不可变组合快照…</div>
  if (error && !run) return <div className="mx-auto max-w-3xl p-6"><div role="alert" className="rounded-xl border border-rose-300 bg-rose-50 p-4 text-rose-800">{error}</div><Link to="/pre-investment/product-allocation-timing/construction" className="mt-4 inline-block text-emerald-700 underline">去构建组合</Link></div>
  if (!run || !diagnosis) return null

  return <div className="mx-auto max-w-7xl space-y-5 p-4 sm:p-6" aria-busy={scenarioLoading || exporting !== null}>
    <header className="rounded-2xl bg-slate-900 px-5 py-6 text-white"><p className="text-sm text-emerald-300">不可变运行快照 · {target ? `研究对象 ${target.revision} 版` : '直接快照查看'}</p><h1 className="mt-1 text-2xl font-semibold">持仓诊断：{target?.name ?? run.name}</h1><p className="mt-2 text-sm text-slate-300">快照 ID：{run.id}。后续配置变更不会改写本次诊断。</p></header>
    {error && <div role="alert" className="rounded border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800">{error}</div>}
    <div className="flex flex-wrap items-end gap-2"><label className="text-xs text-slate-600">单表 CSV<select aria-label="CSV 导出表" value={exportTable} onChange={(event) => setExportTable(event.target.value as PortfolioExportTable)} className="mt-1 block rounded border border-slate-300 bg-white px-2 py-2 text-sm"><option value="summary">汇总</option><option value="components">成分</option><option value="daily-contributions">日收益贡献</option><option value="weight-path">权重路径</option><option value="covariance">协方差矩阵</option><option value="correlation">相关矩阵</option><option value="risk-contributions">风险贡献</option>{scenarioResult && <><option value="scenario-summary">情景汇总</option><option value="scenario-series">情景序列</option></>}</select></label><button type="button" onClick={() => handleExport('csv')} disabled={exporting !== null} className="rounded border border-slate-300 bg-white px-3 py-2 text-sm">{exporting === 'csv' ? '导出中…' : '导出 CSV'}</button><button type="button" onClick={() => handleExport('zip')} disabled={exporting !== null} className="rounded border border-slate-300 bg-white px-3 py-2 text-sm">{exporting === 'zip' ? '导出中…' : '导出 ZIP'}</button><span className="text-xs text-slate-500">ZIP 包含全部表{scenarioResult ? '及当前情景结果' : ''}</span><Link to="/pre-investment/product-allocation-timing/construction" className="rounded bg-emerald-700 px-3 py-2 text-sm text-white">新建组合研究</Link></div>
    <section className="grid grid-cols-2 gap-3 md:grid-cols-4">{metrics.map((metric) => <div key={metric.name} className="rounded-xl border border-slate-200 bg-white p-4"><p className="text-xs text-slate-500">{metric.name}</p><p className="mt-1 text-lg font-semibold"><MetricValue value={metric.value} presentation={metricPresentation(metric)} /></p></div>)}</section>
    <RegimeConditioningPanel result={run.regime_conditioning} />
    <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">组合净值与回撤</h2><ReactECharts option={navOption} style={{ height: 360 }} notMerge lazyUpdate aria-label="组合净值与回撤图" /><table className="sr-only"><caption>组合净值与回撤数据表</caption><tbody>{run.nav.map((point, index) => <tr key={point.date}><td>{point.date}</td><td>{point.value}</td><td>{run.drawdown[index]?.value ?? ''}</td></tr>)}</tbody></table></section>
    <section className="grid gap-5 lg:grid-cols-2"><section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">持仓与收益贡献</h2><div className="mt-3 overflow-x-auto"><table className="w-full text-sm"><thead><tr><th className="p-2 text-left">产品</th><th className="p-2 text-right">收益贡献</th><th className="p-2 text-right">风险贡献</th></tr></thead><tbody>{(diagnosis.contributions ?? run.contributions ?? []).map((row) => <tr className="border-t" key={row.product_id}><td className="p-2">{row.name}</td><td className="p-2 text-right">{percent(row.contribution)}</td><td className="p-2 text-right">{percent(row.risk_contribution)}</td></tr>)}</tbody></table></div></section><section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">集中度与风险贡献</h2><ul className="mt-3 space-y-2 text-sm">{(diagnosis.concentration ?? []).map((item) => <li key={item.name} className="flex justify-between border-b pb-2"><span>{item.name}</span><b><MetricValue value={item.value} presentation={metricPresentation(item)} /></b></li>)}{(diagnosis.risk_contributions ?? []).map((item) => <li key={item.product_id} className="flex justify-between border-b pb-2"><span>{item.name}</span><b>{percent(item.risk_contribution ?? item.contribution)}</b></li>)}</ul></section></section>
    <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">权重路径与调仓</h2><ReactECharts option={weightOption} style={{ height: 320 }} notMerge lazyUpdate aria-label="组合权重路径图" /><div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr><th className="p-2 text-left">调仓日</th><th className="p-2 text-right">换手率</th><th className="p-2 text-left">说明</th></tr></thead><tbody>{(diagnosis.rebalances ?? run.rebalances ?? []).map((item) => <tr className="border-t" key={item.date}><td className="p-2">{item.date}</td><td className="p-2 text-right">{percent(item.turnover)}</td><td className="p-2">{item.message ?? '—'}</td></tr>)}</tbody></table></div></section>
    <section className="grid gap-5 lg:grid-cols-2"><MatrixTable title="相关性矩阵" matrix={diagnosis.correlation} /><MatrixTable title="协方差矩阵" matrix={diagnosis.covariance} /></section>
    <section className="rounded-xl border border-slate-200 bg-white p-4" aria-labelledby="portfolio-indicator-title"><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 id="portfolio-indicator-title" className="font-semibold">组合研究指标</h2><p className="mt-1 text-sm text-slate-500">仅展示组合指标目录；数值、单位和不可计算状态使用锁定指标版本的展示规则。</p></div>{portfolioIndicators.length > 0 && <MetricSelector indicators={portfolioIndicators} selectedIds={metricPreference.indicatorIds} onChange={(indicatorIds) => setMetricPreference((current) => ({ ...current, indicatorIds }))} maxSelected={10} label="选择组合指标" />}</div>{portfolioIndicators.length ? <button type="button" onClick={refreshPortfolioIndicators} disabled={!metricPreference.indicatorIds.length || indicatorLoading} className="mt-3 min-h-11 rounded bg-emerald-700 px-3 text-sm text-white disabled:cursor-not-allowed disabled:bg-slate-400">{indicatorLoading ? '正在计算…' : '计算选中指标'}</button> : <p className="mt-3 text-sm text-slate-500">当前没有已保存的组合指标。可在指标中心创建组合指标后返回此处。</p>}{indicatorError && <p role="alert" className="mt-3 rounded border border-rose-300 bg-rose-50 p-2 text-sm text-rose-800">{indicatorError}</p>}{diagnosis.custom_indicators?.length ? <div className="mt-3 grid grid-cols-1 gap-3 md:grid-cols-2">{diagnosis.custom_indicators.map((metric, index) => <div key={`${metric.name}-${index}`} className={`rounded border p-3 ${metric.status === 'error' ? 'border-rose-300 bg-rose-50' : metric.status === 'warning' ? 'border-amber-300 bg-amber-50' : 'border-slate-200 bg-slate-50'}`}><p className="text-xs text-slate-600">{metric.name}</p><b><MetricValue value={metric.value} presentation={metricPresentation(metric)} /></b>{metric.warnings?.length ? <ul className="mt-2 list-disc pl-5 text-xs text-amber-900">{metric.warnings.map((warning) => <li key={warning}>{humanizeIndicatorMessage(warning)}</li>)}</ul> : null}</div>)}</div> : metricPreference.indicatorIds.length ? <p className="mt-3 text-sm text-slate-500" aria-live="polite">已选指标尚无结果。</p> : null}</section>
    <section className="rounded-xl border border-slate-200 bg-white p-4"><h2 className="font-semibold">历史情景</h2><div className="mt-3 grid gap-3 md:grid-cols-4"><label className="text-sm">名称<input value={scenario.name} onChange={(event) => setScenario({ ...scenario, name: event.target.value })} className="mt-1 w-full rounded border border-slate-300 p-2" /></label><label className="text-sm">开始日<input type="date" value={scenario.start_date} onChange={(event) => setScenario({ ...scenario, start_date: event.target.value })} className="mt-1 w-full rounded border border-slate-300 p-2" /></label><label className="text-sm">结束日<input type="date" value={scenario.end_date} onChange={(event) => setScenario({ ...scenario, end_date: event.target.value })} className="mt-1 w-full rounded border border-slate-300 p-2" /></label><button type="button" onClick={handleScenario} disabled={scenarioLoading} className="self-end rounded bg-slate-900 px-4 py-2 text-sm text-white">{scenarioLoading ? '计算中…' : '运行情景'}</button></div>{scenarioResult && <div className="mt-3 rounded bg-slate-50 p-3 text-sm"><b>{scenarioResult.name}</b><ul className="mt-2 flex flex-wrap gap-4">{scenarioResult.metrics.map((metric) => <li key={metric.name}>{metric.name}：<MetricValue value={metric.value} presentation={metricPresentation(metric)} /></li>)}</ul>{scenarioResult.warnings?.map((warning) => <p className="mt-1 text-amber-800" key={warning}>{humanizeIndicatorMessage(warning)}</p>)}</div>}</section>
    <PortfolioRiskSection key={run.id} portfolioRunId={run.id} context="使用本次不可变研究快照的期末持仓。读取已发布敏感度，不重新拟合，也不改变真实持仓。" />
    {diagnosis.warnings.length > 0 && <section className="rounded-xl border border-amber-300 bg-amber-50 p-4"><h2 className="font-semibold text-amber-900">数据与运行警告</h2><ul className="mt-2 list-disc pl-5 text-sm text-amber-900">{diagnosis.warnings.map((warning) => <li key={warning}>{humanizeIndicatorMessage(warning)}</li>)}</ul></section>}
  </div>
}
