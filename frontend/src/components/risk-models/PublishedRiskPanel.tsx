import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import ReactECharts from 'echarts-for-react'
import type { EChartsOption } from 'echarts'
import {
  getRiskRun, riskPortfolios, riskReleases, runRiskImpact, scenarioReleases,
  type PortfolioChoice, type RiskImpact, type RiskRelease, type RiskRun, type ScenarioRelease,
} from '../../services/riskModels'
import RiskRunView from './RiskRunView'
import { buttonClass, Empty, Feedback, Field, frequencyLabels, inputClass, numberText, percentText, primaryClass, sectionClass, statusLabels, today } from './ResearchUI'

function incompatibility(exposure: RiskRelease | undefined, scenario: ScenarioRelease | undefined): string {
  if (!exposure || !scenario) return ''
  if (exposure.status !== 'active') return `风险成果${statusLabels[exposure.status] ?? '不可用'}。`
  if (scenario.status !== 'active') return `情景${statusLabels[scenario.status] ?? '不可用'}。`
  if (exposure.method === 'cashflow' ? scenario.horizon !== 1 : exposure.frequency !== scenario.frequency) return '周期不匹配。现金流估值只支持单期；回归暴露必须与情景频率相同。'
  const missing = scenario.factors.filter(factor => !exposure.inputs.some(input => input.id === factor.id && input.contract_hash === factor.contract_hash))
  return missing.length ? `这份风险模型未覆盖相同单位和来源的因子：${missing.map(item => item.name).join('、')}。` : ''
}

export function RiskImpactResult({ result }: { result: RiskImpact }) {
  const option = useMemo<EChartsOption>(() => ({
    aria: { enabled: true, description: '已发布情景下，以1为起点的模型解释影响路径，不含基线预期收益。' },
    tooltip: { trigger: 'axis' }, grid: { left: 55, right: 20, top: 20, bottom: 36 },
    xAxis: { type: 'category', data: [0, ...result.path.map(item => item.step)], name: '期' },
    yAxis: { type: 'value', scale: true },
    series: [{ name: '模型影响路径', type: 'line', showSymbol: result.path.length < 12, data: [1, ...result.path.map(item => item.nav)] }],
  }), [result])
  return <section className={`${sectionClass} space-y-5`} aria-label="本次压测结果">
    <div><h3 className="font-semibold">{result.name}</h3><p className="mt-1 text-xs text-slate-500">研究日期 {result.as_of} · {frequencyLabels[result.frequency]} · {result.transient ? '本次计算结果未写入磁盘' : '旧版历史结果 · 已保存'}</p></div>
    <div className="grid gap-5 sm:grid-cols-3"><div><p className="text-xs text-slate-500">模型解释的情景影响</p><p className={`mt-1 text-2xl font-semibold tabular-nums ${result.summary.terminal_return < 0 ? 'text-rose-700' : 'text-teal-800'}`}>{percentText(result.summary.terminal_return)}</p></div><div><p className="text-xs text-slate-500">所填假设本金的损益</p><p className="mt-1 text-xl font-semibold tabular-nums">{numberText(result.summary.pnl_amount, 2)} 元</p></div><div><p className="text-xs text-slate-500">{result.path.length > 1 ? '设定路径内最大回撤' : '冲击后相对价值'}</p><p className="mt-1 text-xl font-semibold tabular-nums">{result.path.length > 1 ? percentText(result.summary.max_drawdown) : numberText(result.summary.terminal_nav, 4)}</p></div></div>
    {result.path.length > 1 && <div><p className="mb-2 text-xs text-slate-500">以 1 为起点，仅展示冲击解释部分，未加入基线收益。</p><ReactECharts option={option} style={{ height: 250 }} notMerge /></div>}
    <div className="grid gap-5 lg:grid-cols-2"><div className="overflow-auto"><table className="w-full text-sm"><caption className="mb-2 text-left font-medium">哪些产品贡献了损益</caption><thead className="text-xs text-slate-500"><tr><th className="p-2 text-left">产品</th><th className="p-2 text-right">起始权重</th><th className="p-2 text-right">终值贡献</th></tr></thead><tbody className="divide-y divide-slate-100">{result.by_asset.map(item => <tr key={item.key}><td className="p-2">{item.name}</td><td className="p-2 text-right tabular-nums">{percentText(item.weight)}</td><td className="p-2 text-right tabular-nums">{percentText(item.contribution)}</td></tr>)}</tbody></table></div><div className="overflow-auto"><table className="w-full text-sm"><caption className="mb-2 text-left font-medium">哪些风险来源贡献了损益</caption><thead className="text-xs text-slate-500"><tr><th className="p-2 text-left">风险来源</th><th className="p-2 text-right">终值贡献</th></tr></thead><tbody className="divide-y divide-slate-100">{result.by_factor.map(item => <tr key={item.id}><td className="p-2">{item.name}</td><td className="p-2 text-right tabular-nums">{percentText(item.contribution)}</td></tr>)}</tbody></table></div></div>
    <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium text-slate-700">路径数据、限制与来源</summary><div className="mt-3 space-y-2 text-xs leading-5 text-slate-500">{result.limitations.map(item => <p key={item}>{item}</p>)}{result.target.holdings_date && <p>使用的期末持仓：{result.target.holdings_date}；未重新运行组合策略。</p>}<p>资产贡献和因子贡献均已由后端与组合终值对账；未将单期百分比直接相加。</p><p className="break-all">情景：{result.request.scenario_release_id}</p><p className="break-all">风险模型成果：{result.request.exposure_release_id}</p><p className="break-all">结果：{result.id}</p></div><div className="mt-3 max-h-64 overflow-auto"><table className="w-full text-xs"><thead className="text-slate-500"><tr><th className="p-2 text-left">期数</th><th className="p-2 text-right">当期影响</th><th className="p-2 text-right">相对价值</th><th className="p-2 text-right">回撤</th></tr></thead><tbody>{result.path.map(item => <tr key={item.step}><td className="p-2">{item.step}</td><td className="p-2 text-right">{percentText(item.return)}</td><td className="p-2 text-right">{numberText(item.nav, 5)}</td><td className="p-2 text-right">{percentText(item.drawdown)}</td></tr>)}</tbody></table></div></details>
  </section>
}

export default function PublishedRiskPanel({ productKey, productName, portfolioRunId, portfolioOnly = false }: { productKey?: string; productName?: string; portfolioRunId?: string; portfolioOnly?: boolean }) {
  const [params] = useSearchParams()
  const [mode, setMode] = useState<'product' | 'portfolio_run'>(portfolioRunId || portfolioOnly ? 'portfolio_run' : 'product')
  const [asOf, setAsOf] = useState(today)
  const [exposures, setExposures] = useState<RiskRelease[]>([])
  const [scenarios, setScenarios] = useState<ScenarioRelease[]>([])
  const [portfolios, setPortfolios] = useState<PortfolioChoice[]>([])
  const [exposureId, setExposureId] = useState(params.get('exposure_release_id') ?? '')
  const [scenarioId, setScenarioId] = useState(params.get('scenario_release_id') ?? '')
  const [selectedProduct, setSelectedProduct] = useState(productKey ?? params.get('product_key') ?? '')
  const [selectedPortfolio, setSelectedPortfolio] = useState(portfolioRunId ?? params.get('portfolio_run_id') ?? '')
  const [exposureRun, setExposureRun] = useState<RiskRun | null>(null)
  const [result, setResult] = useState<RiskImpact | null>(null)
  const [compareEnabled, setCompareEnabled] = useState(false)
  const [comparisonPortfolio, setComparisonPortfolio] = useState('')
  const [comparison, setComparison] = useState<RiskImpact | null>(null)
  const [notional, setNotional] = useState('1000000')
  const [policy, setPolicy] = useState<'buy_and_hold' | 'constant_weights_zero_cost'>('buy_and_hold')
  const [acknowledged, setAcknowledged] = useState(false)
  const [loading, setLoading] = useState(true)
  const [readingRun, setReadingRun] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const alive = useRef(true); const current = useRef('')
  const exposure = exposures.find(item => item.id === exposureId)
  const scenario = scenarios.find(item => item.id === scenarioId)
  const mismatch = incompatibility(exposure, scenario)
  const comparing = mode === 'portfolio_run' && compareEnabled
  const comparisonMissing = comparing && (!comparisonPortfolio || comparisonPortfolio === selectedPortfolio)
  const signature = JSON.stringify([productKey, portfolioRunId, mode, asOf, exposureId, scenarioId, selectedProduct, selectedPortfolio, notional, policy, comparing, comparisonPortfolio])
  current.current = signature
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  useEffect(() => { setResult(null); setComparison(null); setAcknowledged(false); setError('') }, [signature])
  useEffect(() => { if (productKey) { setSelectedProduct(productKey); setMode('product') } if (portfolioRunId) { setSelectedPortfolio(portfolioRunId); setMode('portfolio_run') } }, [productKey, portfolioRunId])
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setExposures([]); setScenarios([]); setExposureRun(null)
    Promise.all([riskReleases('product', { as_of: asOf, ...(productKey ? { product_key: productKey } : {}) }, controller.signal), scenarioReleases(asOf, controller.signal)])
      .then(([nextExposures, nextScenarios]) => { if (!controller.signal.aborted) { setExposures(nextExposures); setScenarios(nextScenarios); setExposureId(old => old || nextExposures.find(item => item.status === 'active')?.id || '') } })
      .catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '已发布成果读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [asOf, productKey, refresh])
  useEffect(() => {
    if (mode !== 'portfolio_run' || (portfolioRunId && !comparing)) return
    const controller = new AbortController()
    riskPortfolios(controller.signal).then(items => { if (!controller.signal.aborted) setPortfolios(items) }).catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '组合快照读取失败。') })
    return () => controller.abort()
  }, [mode, portfolioRunId, comparing, refresh])
  useEffect(() => {
    const controller = new AbortController(); setExposureRun(null)
    if (!exposure) { setReadingRun(false); return () => controller.abort() }
    setReadingRun(true)
    getRiskRun('product', exposure.run_id, controller.signal).then(run => { if (!controller.signal.aborted) setExposureRun(run) }).catch(caught => { if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : '敏感度成果读取失败。') }).finally(() => { if (!controller.signal.aborted) setReadingRun(false) })
    return () => controller.abort()
  }, [exposure?.run_id])
  async function calculate() {
    if (!exposure || !scenario || mismatch || !acknowledged || loading || busy || comparisonMissing) return
    const requested = signature; setBusy(true); setError(''); setResult(null); setComparison(null)
    try {
      if (!Number.isFinite(Number(notional)) || Number(notional) <= 0 || (mode === 'product' ? !selectedProduct : !selectedPortfolio)) throw new Error('请明确选择研究对象，并填写正数的假设本金。')
      const shared = { scenario_release_id: scenario.id, exposure_release_id: exposure.id, as_of: asOf, holding_policy: policy, hold_other_factors_constant: true as const, notional: Number(notional), usage: 'research' as const }
      const primary = runRiskImpact({ ...shared, target: mode === 'product' ? { kind: 'product', product_key: selectedProduct } : { kind: 'portfolio_run', portfolio_run_id: selectedPortfolio } })
      const other = comparing ? runRiskImpact({ ...shared, target: { kind: 'portfolio_run', portfolio_run_id: comparisonPortfolio } }) : Promise.resolve(null)
      const [value, compared] = await Promise.all([primary, other])
      if (alive.current && current.current === requested) { setResult(value); setComparison(compared) }
    } catch (caught) { if (alive.current && current.current === requested) setError(caught instanceof Error ? caught.message : '情景压测未完成。') }
    finally { if (alive.current) setBusy(false) }
  }
  const riskLink = `/settings/risk-models${productKey ? `?${new URLSearchParams({ product_key: productKey, product_name: productName ?? productKey })}` : ''}`
  const displayRun = exposureRun && mode === 'product' && selectedProduct ? { ...exposureRun, rows: exposureRun.rows.filter(row => row.target_id === selectedProduct) } : exposureRun
  return <div className="min-w-0 space-y-4" data-testid="published-risk-panel"><section className={`${sectionClass} space-y-4`}>
    <div><h3 className="text-lg font-semibold">{productKey ? '这只产品怕什么？' : '使用已发布成果做压测'}</h3><p className="mt-1 text-sm leading-6 text-slate-500">这里只读取已发布敏感度与情景。需要更改算法或重新训练，请前往风险模型中心。</p></div>
    <Feedback error={error} />
    <fieldset disabled={busy} className="min-w-0 space-y-4"><div className="grid gap-4 sm:grid-cols-2"><Field label="风险研究日期" hint="与历史表现区间独立；不能使用该日以后才发布的模型。"><input className={inputClass} type="date" max={today()} value={asOf} onChange={event => setAsOf(event.target.value)} /></Field>{!productKey && !portfolioRunId && !portfolioOnly && <Field label="测试什么对象"><select className={inputClass} value={mode} onChange={event => setMode(event.target.value as typeof mode)}><option value="product">单个产品</option><option value="portfolio_run">已保存的产品组合</option></select></Field>}</div>
      {loading ? <p role="status" className="py-3 text-sm text-slate-500">正在读取本地已发布成果，不会重新计算敏感度…</p> : !exposures.length ? <Empty title={productKey ? '这只产品尚无已发布风险模型' : '还没有已发布风险模型'}><p>先在风险模型中心选择对象、计算并验证，然后发布成果。</p><Link className={`${primaryClass} mt-3`} to={riskLink}>到风险模型中心研究</Link></Empty> : <>
        <Field label="选择已发布的敏感度成果"><select className={inputClass} value={exposureId} onChange={event => setExposureId(event.target.value)}><option value="">请选择风险成果</option>{exposures.map(item => <option key={item.id} value={item.id}>{item.name} · {frequencyLabels[item.frequency]} · {item.as_of} · {statusLabels[item.status] ?? item.status}</option>)}</select></Field>
        {exposureId && !exposure && <p role="status" className="text-sm text-amber-800">指定风险成果在当前对象或日期下不可用，请重新选择；没有自动替换版本。</p>}
        {scenarioId && !scenario && <p role="status" className="text-sm text-amber-800">指定情景不在当前目录中，请重新选择；没有自动替换版本。</p>}
        {exposure && <p className={`rounded-lg p-3 text-xs leading-5 ${exposure.status === 'active' ? 'bg-slate-50 text-slate-600' : 'bg-amber-50 text-amber-900'}`}>状态：{statusLabels[exposure.status] ?? exposure.status}。生效 {exposure.effective_at.slice(0, 10)}，到期 {exposure.expires_at.slice(0, 10)}。只引用这一份结果，不会自动替换成另一个模型。<Link className="ml-2 underline" to={riskLink}>查看或重新研究</Link></p>}
        {displayRun && <RiskRunView run={displayRun} published />}
        {mode === 'product' && !productKey && <Field label="选择产品"><select className={inputClass} value={selectedProduct} onChange={event => setSelectedProduct(event.target.value)}><option value="">请选择这份成果覆盖的产品</option>{exposure?.targets.map(item => <option key={item.key} value={item.key}>{item.name} · {item.product_id}</option>)}</select></Field>}
        {mode === 'portfolio_run' && (portfolioRunId ? <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">使用当前不可变组合快照的期末持仓。缺少任一非零持仓的暴露都会阻断，不会删掉资产重新分配权重。</p> : <Field label="选择已保存的产品组合快照" hint="这里测试的是明确保存的产品组合，不会按大类名称猜测基金。"><select className={inputClass} value={selectedPortfolio} onChange={event => setSelectedPortfolio(event.target.value)}><option value="">请选择一个不可变组合运行</option>{portfolios.map(item => <option key={item.id} value={item.id}>{item.name} · 持仓截至 {item.as_of}</option>)}</select>{!portfolios.length && <Link className="mt-2 block text-xs text-teal-800 underline" to="/pre-investment/product-allocation-timing/construction">先去保存产品组合研究</Link>}</Field>)}
        {mode === 'portfolio_run' && <div className="space-y-3"><label className="flex min-h-11 items-center gap-2 text-sm text-slate-700"><input type="checkbox" checked={compareEnabled} onChange={event => setCompareEnabled(event.target.checked)} />对比另一个已保存组合</label>{comparing && <Field label="选择对照组合快照" hint="两个方案使用相同情景、敏感度、本金和持有规则。不会修改任一方案。"><select className={inputClass} value={comparisonPortfolio} onChange={event => setComparisonPortfolio(event.target.value)}><option value="">请选择不同的组合快照</option>{portfolios.filter(item => item.id !== selectedPortfolio).map(item => <option key={item.id} value={item.id}>{item.name} · 持仓截至 {item.as_of}</option>)}</select></Field>}</div>}
        <div className="border-t border-slate-100 pt-4"><Field label="选择已发布情景"><select className={inputClass} value={scenarioId} onChange={event => setScenarioId(event.target.value)}><option value="">请选择要测试的情景</option>{scenarios.map(item => <option key={item.id} value={item.id}>{item.name} · {frequencyLabels[item.frequency]} · {item.horizon} 期 · {statusLabels[item.status] ?? item.status}</option>)}</select></Field>{!scenarios.length && <p className="mt-2 text-sm text-amber-800">还没有已发布情景。<Link className="ml-2 underline" to="/settings/scenario-algorithms?center=simulation">去构建并发布情景</Link></p>}</div>
        {mismatch && <p role="status" className="rounded-lg bg-amber-50 p-3 text-sm text-amber-900">{mismatch}</p>}
        <div className="grid gap-4 sm:grid-cols-2"><Field label="假设本金（人民币元）" hint="仅用于换算损益金额，不读取或改动真实账户资金。"><input type="number" min="1" className={inputClass} value={notional} onChange={event => setNotional(event.target.value)} /></Field><Field label="冲击期间怎么持有"><select className={inputClass} value={policy} onChange={event => setPolicy(event.target.value as typeof policy)}><option value="buy_and_hold">不调仓，权重自然变化</option><option value="constant_weights_zero_cost">每期恢复原权重（假设零交易成本）</option></select></Field></div>
        <label className="flex items-start gap-2 text-sm leading-6 text-slate-600"><input className="mt-1" type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />未设置冲击的已建模因子保持不变；模型外的风险并不等于零。结果仅表示本次假设下的模型解释部分。</label>
        <div className="flex flex-wrap gap-3"><button type="button" className={primaryClass} disabled={busy || loading || readingRun || !exposureRun || !exposure || !scenario || Boolean(mismatch) || !acknowledged || comparisonMissing || (mode === 'product' ? !selectedProduct : !selectedPortfolio)} onClick={() => void calculate()}>{busy ? '正在应用已发布成果…' : comparing ? '计算并对比' : '计算情景影响'}</button></div>
      </>}
    </fieldset>
    <button type="button" className={buttonClass} disabled={busy || loading} onClick={() => { setError(''); setRefresh(old => old + 1) }}>刷新成果目录</button>
  </section>{result && (comparison ? <RiskImpactComparison primary={result} comparison={comparison} /> : <RiskImpactResult result={result} />)}</div>
}

export function RiskImpactComparison({ primary, comparison }: { primary: RiskImpact; comparison: RiskImpact }) {
  return <section className={`${sectionClass} space-y-4`} aria-label="组合情景对比结果">
    <h3 className="font-semibold">相同情景下，哪个方案受影响更大？</h3>
    <p className="text-sm text-slate-600">使用同一情景版本与风险成果；两份对比结果都只存在于当前页面。持仓日期如下，不把历史快照称为实时持仓。</p>
    <div className="overflow-x-auto"><table className="w-full min-w-[420px] text-sm"><thead className="text-left text-slate-600"><tr><th className="p-2">比较项目</th><th className="p-2">当前方案</th><th className="p-2">对照方案</th></tr></thead><tbody className="divide-y divide-slate-100">
      <tr><th className="p-2 text-left font-medium">组合</th>{[primary, comparison].map(value => <td className="p-2" key={value.id}>{value.target.name}</td>)}</tr>
      <tr><th className="p-2 text-left font-medium">持仓截至</th>{[primary, comparison].map(value => <td className="p-2" key={value.id}>{value.target.holdings_date ?? '未提供'}</td>)}</tr>
      <tr><th className="p-2 text-left font-medium">情景影响</th>{[primary, comparison].map(value => <td className="p-2 font-semibold tabular-nums" key={value.id}>{percentText(value.summary.terminal_return)}</td>)}</tr>
      <tr><th className="p-2 text-left font-medium">假设本金损益</th>{[primary, comparison].map(value => <td className="p-2 tabular-nums" key={value.id}>{numberText(value.summary.pnl_amount, 2)} 元</td>)}</tr>
    </tbody></table></div>
  </section>
}

export function PortfolioRiskSection({ portfolioRunId, context }: { portfolioRunId?: string; context?: string }) {
  const [open, setOpen] = useState(false)
  return <details className="mt-5 min-w-0 rounded-xl border border-slate-200 bg-white p-4" onToggle={event => setOpen(event.currentTarget.open)}><summary className="cursor-pointer text-sm font-semibold text-slate-800">使用已发布模型与情景做组合压测</summary>{context && <p className="mt-3 text-sm leading-6 text-slate-500">{context}</p>}{open && <div className="mt-4"><PublishedRiskPanel portfolioRunId={portfolioRunId} portfolioOnly /></div>}</details>
}
