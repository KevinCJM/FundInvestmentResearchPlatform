import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  createResearchTarget,
  PortfolioConstituent,
  PortfolioInstrument,
  PortfolioMethod,
  PortfolioRun,
  PortfolioRunRequest,
  runPortfolio,
} from '../services/portfolioResearch'
import { MetricValue } from '../components/metrics/MetricDisplay'
import type { MetricPresentation } from '../services/customIndicators'
import { humanizeIndicatorMessage } from '../utils/indicatorDiagnostics'
import { evaluateNumericControls, type NumericControlResult } from '../services/businessNumeric'
import {
  getInvestableUniverse,
  investableUniverseEligibleCount,
  searchInvestableUniverseProducts,
  type InvestableUniverseSnapshot,
} from '../services/productPools'
import {
  HistoricalRegimeBacktestSelector,
  RegimeConditioningPanel,
} from '../components/HistoricalRegimeBacktest'
import type { HistoricalRegimeBacktestReference } from '../services/portfolioRegime'
import { PortfolioRiskSection } from '../components/risk-models/PublishedRiskPanel'
import { readAllocationDraft, readAllocationJourney, updateAllocationJourney, writeAllocationDraft } from '../app/allocationJourney'

const METHODS: Array<{ value: PortfolioMethod; label: string; help: string }> = [
  { value: 'equal_weight', label: '等权', help: '在选中产品间平均分配资金。' },
  { value: 'manual', label: '手工权重', help: '按研究假设直接设置各产品权重。' },
  { value: 'risk_budget', label: '风险预算', help: '按目标风险贡献求解资金权重。' },
  { value: 'target_optimization', label: '目标优化', help: '以历史窗口求解最大夏普、最小波动或目标收益。' },
]

const percent = (value: number | null | undefined, digits = 2) =>
  value === null || value === undefined || !Number.isFinite(value) ? '—' : `${(value * 100).toFixed(digits)}%`

const resultMetricPresentation = (metric: PortfolioRun['metrics'][number]): MetricPresentation => metric.presentation ?? {
  indicator_id: metric.metric_id ?? metric.name, revision: 1, name: metric.name, source: 'built_in', category: 'portfolio_summary', category_label: '组合汇总', context_kind: 'portfolio', catalog_status: 'current',
  display_format: metric.unit === 'percent' ? 'percent' : 'number', precision: 3, unit: metric.unit === 'percent' ? '%' : metric.unit ?? '', notation: 'standard', value_scale: metric.unit === 'percent' ? 100 : 1,
  output_measure: 'dimensionless', direction: metric.direction ?? 'higher_better', description: '', methodology: '', data_basis: '锁定运行快照', minimum_observations: 1, applicable_product_kinds: ['portfolio'],
}

type PortfolioDraft = {
  name: string; universeId: string; constituents: PortfolioConstituent[]; method: PortfolioMethod;
  minWeight: number; maxWeight: number; windowMode: 'all' | 'rolling'; observations: number;
  rebalance: PortfolioRunRequest['rebalance']['frequency']; benchmark: PortfolioInstrument | null;
  objective: NonNullable<PortfolioRunRequest['objective']>; targetReturn: number;
  allocationSource?: PortfolioRunRequest['allocation_source']; historicalRegime: HistoricalRegimeBacktestReference | null;
}

export default function PortfolioConstruction() {
  const [searchParams] = useSearchParams()
  return <PortfolioConstructionEditor key={searchParams.toString()} />
}

function PortfolioConstructionEditor() {
  const [searchParams] = useSearchParams()
  const [initialImport] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem('portfolioResearchImport') ?? 'null') } catch { return null }
  })
  const draftScope = `products:${searchParams.get('universe') ?? initialImport?.universe_snapshot_id ?? readAllocationJourney().universeId ?? 'local'}:${searchParams.get('decision') ?? initialImport?.allocation_source?.decision_id ?? 'manual'}`
  const [draft] = useState(() => readAllocationDraft<PortfolioDraft>(draftScope))
  const [inputsChanged, setInputsChanged] = useState(false)
  const [universeId, setUniverseId] = useState(() => searchParams.get('universe') ?? initialImport?.universe_snapshot_id ?? draft?.universeId ?? readAllocationJourney().universeId ?? '')
  const [universe, setUniverse] = useState<InvestableUniverseSnapshot | null>(null)
  const [universeLoading, setUniverseLoading] = useState(false)
  const [step, setStep] = useState(1)
  const [name, setName] = useState(draft?.name ?? '未命名组合研究')
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [candidates, setCandidates] = useState<PortfolioInstrument[]>([])
  const [constituents, setConstituents] = useState<PortfolioConstituent[]>(draft?.constituents ?? [])
  const [method, setMethod] = useState<PortfolioMethod>(draft?.method ?? 'equal_weight')
  const [minWeight, setMinWeight] = useState(draft?.minWeight ?? 0)
  const [maxWeight, setMaxWeight] = useState(draft?.maxWeight ?? 100)
  const [windowMode, setWindowMode] = useState<'all' | 'rolling'>(draft?.windowMode ?? 'all')
  const [observations, setObservations] = useState(draft?.observations ?? 252)
  const [rebalance, setRebalance] = useState<PortfolioRunRequest['rebalance']['frequency']>(draft?.rebalance ?? 'monthly')
  const [benchmark, setBenchmark] = useState<PortfolioInstrument | null>(draft?.benchmark ?? null)
  const [objective, setObjective] = useState<NonNullable<PortfolioRunRequest['objective']>>(draft?.objective ?? 'max_sharpe')
  const [targetReturn, setTargetReturn] = useState(draft?.targetReturn ?? 8)
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<PortfolioRun | null>(null)
  const [historicalRegime, setHistoricalRegime] = useState<HistoricalRegimeBacktestReference | null>(draft?.historicalRegime ?? null)
  const [numericControls, setNumericControls] = useState<Record<string, NumericControlResult>>({})
  const [numericControlError, setNumericControlError] = useState('')
  const [selectedAssetClassId, setSelectedAssetClassId] = useState('')
  const [allocationSource, setAllocationSource] = useState<PortfolioRunRequest['allocation_source']>(draft?.allocationSource)

  useEffect(() => {
    const raw = sessionStorage.getItem('portfolioResearchImport')
    if (!raw) return
    sessionStorage.removeItem('portfolioResearchImport')
    try {
      const imported = JSON.parse(raw) as {
        name?: string
        method?: PortfolioMethod
        universe_snapshot_id?: string
        constituents?: PortfolioConstituent[]
        allocation_source?: PortfolioRunRequest['allocation_source']
      }
      if (imported.name) setName(imported.name)
      if (imported.allocation_source) { setAllocationSource(imported.allocation_source); setMethod('manual'); updateAllocationJourney({ taaRunId: imported.allocation_source.decision_id, universeId: imported.universe_snapshot_id }) }
      if (imported.universe_snapshot_id) setUniverseId(imported.universe_snapshot_id)
      if (Array.isArray(imported.constituents)) setConstituents(imported.constituents)
      if (!imported.allocation_source && imported.method && METHODS.some((item) => item.value === imported.method)) setMethod(imported.method)
    } catch {
      setError('导入的组合配置无法读取，请重新从来源页面导入。')
    }
  }, [])

  const inputState: PortfolioDraft = { name, universeId, constituents, method, minWeight, maxWeight, windowMode, observations, rebalance, benchmark, objective, targetReturn, allocationSource, historicalRegime }
  const inputKey = JSON.stringify(inputState)
  const latestInput = useRef(inputKey)
  latestInput.current = inputKey
  useEffect(() => {
    if (universeId && constituents.length) writeAllocationDraft(draftScope, inputState)
    setResult(null)
    setInputsChanged(true)
  }, [inputKey, draftScope])

  useEffect(() => {
    if (!universeId) {
      setUniverse(null)
      return
    }
    let active = true
    setUniverseLoading(true)
    getInvestableUniverse(universeId)
      .then((snapshot) => { if (active) setUniverse(snapshot) })
      .catch((caught) => {
        if (active) {
          setUniverse(null)
          setError(caught instanceof Error ? caught.message : '无法加载可投资域快照。')
        }
      })
      .finally(() => { if (active) setUniverseLoading(false) })
    return () => { active = false }
  }, [universeId])

  useEffect(() => {
    const controller = new AbortController()
    const trimmed = query.trim()
    if (!trimmed || !universeId || !universe) {
      setCandidates([])
      return () => controller.abort()
    }
    setSearching(true)
    setSearchError('')
    searchInvestableUniverseProducts(universeId, {
      query: trimmed,
      eligibleOnly: true,
      pageSize: 20,
      signal: controller.signal,
    })
      .then((response) => setCandidates(response.items.map((item) => ({
        product_id: item.product_id,
        kind: item.kind,
        name: item.name,
        code: item.code,
      }))))
      .catch((caught) => {
        if ((caught as DOMException)?.name !== 'AbortError') {
          setSearchError(caught instanceof Error ? caught.message : '产品搜索失败')
        }
      })
      .finally(() => { if (!controller.signal.aborted) setSearching(false) })
    return () => controller.abort()
  }, [query, universe, universeId])

  const assetClasses = useMemo(() => {
    const items = new Map<string, string>()
    constituents.forEach((item) => {
      if (item.asset_class_id && item.asset_class_name) {
        items.set(item.asset_class_id, item.asset_class_name)
      }
    })
    return [...items.entries()].map(([id, name]) => ({ id, name }))
  }, [constituents])

  useEffect(() => {
    if (assetClasses.some((item) => item.id === selectedAssetClassId)) return
    setSelectedAssetClassId(assetClasses[0]?.id ?? '')
  }, [assetClasses, selectedAssetClassId])

  useEffect(() => {
    const controller = new AbortController()
    setNumericControls({})
    setNumericControlError('')
    evaluateNumericControls([
      { key: 'portfolio-weights', values: constituents.map((item) => item.weight ?? 0), target: 100, tolerance: 0.01 },
      { key: 'portfolio-risk-budgets', values: constituents.map((item) => item.risk_budget ?? 0), target: 100, tolerance: 0.01 },
      ...Object.entries(allocationSource?.class_weights ?? {}).map(([name, weight]) => ({ key: `class-budget:${name}`, values: constituents.filter(item => item.asset_class_name === name).map(item => item.weight ?? 0), target: weight * 100, tolerance: 0.01 })),
    ], controller.signal)
      .then((response) => setNumericControls(Object.fromEntries(response.items.map((item) => [item.key, item]))))
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setNumericControlError('权重校验暂不可用，请稍后重试。')
      })
    return () => controller.abort()
  }, [constituents, allocationSource])

  const weightControl = numericControls['portfolio-weights']
  const budgetControl = numericControls['portfolio-risk-budgets']
  const classBudgetsValid = !allocationSource || (method === 'manual' && Object.keys(allocationSource.class_weights).every(name => numericControls[`class-budget:${name}`]?.within_tolerance === true))
  const readyToRun = classBudgetsValid && Boolean(universe) && constituents.length >= 2 && minWeight <= maxWeight &&
    constituents.every((item) => item.asset_class_id && item.asset_class_name) &&
    (method !== 'manual' || weightControl?.within_tolerance === true) &&
    (method !== 'risk_budget' || budgetControl?.within_tolerance === true)

  function addInstrument(item: PortfolioInstrument) {
    const assetClass = assetClasses.find((entry) => entry.id === selectedAssetClassId)
    if (!assetClass) {
      setError('请先从大类构建页面导入至少一个资产大类。')
      return
    }
    setConstituents((current) => current.some((entry) => entry.product_id === item.product_id && entry.kind === item.kind)
      ? current
      : [...current, {
        ...item,
        weight: 0,
        risk_budget: 0,
        asset_class_id: assetClass.id,
        asset_class_name: assetClass.name,
      }])
  }
  function updateNumber(index: number, field: 'weight' | 'risk_budget', raw: string) {
    const value = Math.max(0, Number(raw) || 0)
    setConstituents((current) => current.map((item, currentIndex) => currentIndex === index ? { ...item, [field]: value } : item))
  }
  function buildRequest(): PortfolioRunRequest {
    return {
      name: name.trim() || '未命名组合研究',
      universe_snapshot_id: universe?.id ?? (universeId || null),
      allocation_source: allocationSource,
      constituents,
      method,
      constraints: { min_weight: minWeight / 100, max_weight: maxWeight / 100 },
      window: windowMode === 'all' ? { mode: 'all' } : { mode: 'rolling', observations },
      rebalance: { frequency: rebalance, transaction_cost_bps: 0 },
      benchmark: benchmark ? { kind: benchmark.kind, product_id: benchmark.product_id, name: benchmark.name } : null,
      objective: method === 'target_optimization' ? objective : null,
      target_return: method === 'target_optimization' && objective === 'target_return' ? targetReturn / 100 : null,
    }
  }
  async function handleRun() {
    if (!readyToRun) { setError('请先锁定可投资域、从大类构建导入产品，并完成当前权重或风险预算校验。'); return }
    const submittedInput = inputKey
    setRunning(true); setError('')
    try {
      const target = await createResearchTarget({ name: name.trim() || '未命名组合研究', kind: 'portfolio', definition: buildRequest() })
      const run = await runPortfolio(target.id, { historical_regime: historicalRegime })
      if (latestInput.current !== submittedInput) { setError('研究输入已变化，请重新运行当前方案。'); return }
      setResult({ ...run, target_id: target.id })
      setInputsChanged(false)
      setStep(4)
    } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, '组合运行失败')) } finally { setRunning(false) }
  }
  async function handleSave() {
    if (!result) return
    setSaving(true); setError('')
    try {
      const target = await createResearchTarget({ name: name.trim() || result.name, kind: 'portfolio', definition: buildRequest() })
      setResult({ ...result, target_id: target.id })
    } catch (caught: any) { setError(humanizeIndicatorMessage(caught?.message, '保存研究对象失败')) } finally { setSaving(false) }
  }

  return <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" aria-busy={running || saving || universeLoading}>
    <header className="rounded-xl bg-slate-900 px-5 py-6 text-white sm:px-7">
      <p className="text-sm text-emerald-300">工作区共享 · 真实历史数据</p><h1 className="mt-1 text-2xl font-semibold">产品组合构建</h1>
      <p className="mt-2 max-w-3xl text-sm text-slate-200">把大类资金预算分配到具体产品，检查类内权重后运行历史研究。</p>
    </header>
    {universe ? <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">可投资域：<b>{universe.name}</b> · {investableUniverseEligibleCount(universe)} 只可用产品 · 研究日期 {universe.research_date}</div> : <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"><span>{universeLoading ? '正在读取可投资域…' : '尚未从产品池和大类构建进入，不能直接从全市场配置产品。'}</span><Link to="/pre-investment/product-pool" className="rounded-lg bg-amber-800 px-3 py-2 font-medium text-white">选择产品池版本</Link></div>}
    <ol className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4" aria-label="组合构建步骤">
      {['选择产品', '权重方法', '约束与回测', '运行与保存'].map((label, index) => <li key={label}><button type="button" onClick={() => { setStep(index + 1); document.getElementById(`portfolio-step-${index + 1}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }} className={`w-full rounded-lg border px-3 py-2 text-left ${step === index + 1 ? 'border-emerald-500 bg-emerald-50 font-semibold text-emerald-800' : 'border-slate-200 bg-white text-slate-600'}`}>{index + 1}. {label}</button></li>)}
    </ol>
    {error && <div role="alert" className="rounded-lg border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800">{error}</div>}
    {allocationSource && <aside className="rounded-xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950">
      <p className="font-semibold">已承接 TAA 大类预算</p>
      {allocationSource.expires_on && <p className="mt-1 text-xs">方案有效至 {allocationSource.expires_on}；运行时仍须通过有效期及风险检查。</p>}
      <div className="mt-2 grid gap-2 sm:grid-cols-3">{Object.entries(allocationSource.class_weights).map(([name, weight]) => { const control = numericControls[`class-budget:${name}`]; return <div key={name} className="rounded-lg bg-white p-2"><b>{name}</b> · 预算 {(weight * 100).toFixed(2)}%<p className={control?.within_tolerance ? 'text-emerald-700' : 'text-amber-800'}>{control ? `已分配 ${control.total.toFixed(2)}% · ${control.within_tolerance ? '符合预算' : '请调整类内权重'}` : '正在检查预算…'}</p></div> })}</div>
      <p className="mt-2">只调整类内产品，保持各类合计预算。此页回放当前目标组合，费用为 0；动态调仓及扣费表现请回 TAA 查看。</p>
      <Link className="mt-2 inline-block underline" to={`/pre-investment/taa?decision=${encodeURIComponent(allocationSource.decision_id)}`}>返回 TAA 查看或重新研究</Link>
    </aside>}
    {(draft || inputsChanged) && !result && <p className="text-xs text-slate-600">输入自动保留；修改后请重新运行，历史结果不会沿用。</p>}
    <section id="portfolio-step-1" className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
      <div className="space-y-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <div><label className="block text-sm font-medium" htmlFor="portfolio-name">研究名称</label><input id="portfolio-name" value={name} onChange={(event) => setName(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></div>
        <div className="space-y-2"><div className="grid gap-3 sm:grid-cols-[1fr_180px]"><label className="block text-sm font-medium" htmlFor="portfolio-search">从可投资域搜索产品<input id="portfolio-search" disabled={!universe || !assetClasses.length} value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入代码或名称" className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 disabled:bg-slate-100" /></label><label className="block text-sm font-medium">加入大类<select aria-label="候选产品所属大类" disabled={!assetClasses.length} value={selectedAssetClassId} onChange={(event) => setSelectedAssetClassId(event.target.value)} className="mt-1 w-full rounded-xl border border-slate-300 bg-white px-3 py-2 disabled:bg-slate-100"><option value="">请选择</option>{assetClasses.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}</select></label></div>
          {!assetClasses.length && <p className="rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">请先在<Link to={universe ? `/pre-investment/saa/asset-classes?universe=${encodeURIComponent(universe.id)}` : '/pre-investment/product-pool'} className="mx-1 font-medium underline">大类构建</Link>中建立大类并导入产品。</p>}
          <p aria-live="polite" className="text-xs text-slate-600">{searching ? '正在检索可投资域产品…' : searchError || (query && !candidates.length ? '没有匹配产品。' : '')}</p>
          <ul className="divide-y rounded-lg border border-slate-200" aria-label="产品搜索结果">{candidates.map((item) => <li key={`${item.kind}-${item.product_id}`} className="flex items-center justify-between gap-3 px-3 py-2 text-sm"><span><b>{item.name}</b> <span className="text-slate-600">{item.code ?? item.product_id} · {item.kind === 'etf' ? 'ETF' : '公募基金'}</span></span><button type="button" disabled={!selectedAssetClassId} onClick={() => addInstrument(item)} className="rounded-lg border border-emerald-600 px-2 py-1 text-emerald-700 disabled:border-slate-300 disabled:text-slate-600">加入</button></li>)}</ul>
        </div>
        <div><h2 className="text-base font-semibold">已选产品（至少 2 只）</h2>{!constituents.length ? <p className="mt-2 text-sm text-slate-600">搜索并加入产品后开始构建。</p> : <ul className="mt-2 space-y-2">{constituents.map((item, index) => <li key={`${item.kind}-${item.product_id}`} className="grid grid-cols-[1fr_auto] gap-2 rounded-lg border border-slate-200 p-2"><span className="text-sm"><b>{item.name}</b><br /><span className="text-xs text-slate-600">{item.code ?? item.product_id} · {item.asset_class_name || '未指定大类'}</span></span><button type="button" onClick={() => setConstituents((current) => current.filter((_, currentIndex) => currentIndex !== index))} className="text-sm text-rose-700 underline">移除</button>{method === 'manual' && <label className="col-span-2 text-sm">权重 (%)<input aria-label={`${item.name} 权重`} type="number" min="0" max="100" value={item.weight ?? 0} onChange={(event) => updateNumber(index, 'weight', event.target.value)} className="ml-2 w-24 rounded-lg border border-slate-300 px-2 py-1" /></label>}{method === 'risk_budget' && <label className="col-span-2 text-sm">风险预算 (%)<input aria-label={`${item.name} 风险预算`} type="number" min="0" max="100" value={item.risk_budget ?? 0} onChange={(event) => updateNumber(index, 'risk_budget', event.target.value)} className="ml-2 w-24 rounded-lg border border-slate-300 px-2 py-1" /></label>}</li>)}</ul>}
          {(method === 'manual' || method === 'risk_budget') && (() => { const control = method === 'manual' ? weightControl : budgetControl; return <p className={`mt-2 text-sm ${control?.within_tolerance ? 'text-emerald-700' : 'text-amber-700'}`}>{method === 'manual' ? '权重' : '风险预算'}合计：{control ? `${control.total.toFixed(2)}%（需为 100%）` : numericControlError || '正在校验…'}</p> })()}</div>
      </div>
      <div className="space-y-5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <fieldset id="portfolio-step-2"><legend className="text-base font-semibold">权重方法</legend><div className="mt-2 grid gap-2">{METHODS.map((item) => <label key={item.value} className={`cursor-pointer rounded-lg border p-3 ${method === item.value ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200'}`}><input type="radio" name="method" disabled={Boolean(allocationSource) && item.value !== 'manual'} checked={method === item.value} onChange={() => setMethod(item.value)} /> <b className="ml-1">{item.label}</b><span className="block pl-5 text-xs text-slate-600">{item.help}</span></label>)}</div></fieldset>
        {method === 'target_optimization' && <div className="space-y-2"><label className="block text-sm">优化目标<select value={objective} onChange={(event) => setObjective(event.target.value as typeof objective)} className="ml-2 rounded-lg border border-slate-300 p-1"><option value="max_sharpe">最大夏普</option><option value="min_volatility">最小波动</option><option value="target_return">目标收益</option></select></label>{objective === 'target_return' && <label className="block text-sm">目标年化收益 (%)<input type="number" value={targetReturn} onChange={(event) => setTargetReturn(Number(event.target.value))} className="ml-2 w-24 rounded-lg border border-slate-300 p-1" /></label>}</div>}
        {allocationSource && <p className="text-xs text-slate-600">当前承接 TAA 预算，使用手工权重做类内调整。</p>}
        <fieldset id="portfolio-step-3" className="space-y-2"><legend className="text-base font-semibold">约束、窗口与调仓</legend><div className="grid grid-cols-2 gap-3 text-sm"><label>最小权重 (%)<input type="number" value={minWeight} onChange={(event) => setMinWeight(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label><label>最大权重 (%)<input type="number" value={maxWeight} onChange={(event) => setMaxWeight(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label><label>样本窗口<select value={windowMode} onChange={(event) => setWindowMode(event.target.value as 'all' | 'rolling')} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="all">全历史</option><option value="rolling">滚动观察值</option></select></label><label>调仓<select value={rebalance} onChange={(event) => setRebalance(event.target.value as typeof rebalance)} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="fixed">固定权重</option><option value="weekly">周度</option><option value="monthly">月度</option><option value="yearly">年度</option></select></label>{windowMode === 'rolling' && <label>观察值<input type="number" min="2" value={observations} onChange={(event) => setObservations(Number(event.target.value))} className="mt-1 w-full rounded-lg border border-slate-300 p-2" /></label>}</div><p className="text-xs text-slate-600">当前组合研究固定交易成本为 0，仅输出未扣费的毛收益。</p></fieldset>
        <HistoricalRegimeBacktestSelector
          value={historicalRegime}
          disabled={running}
          onChange={(value) => {
            setHistoricalRegime(value)
            setResult(null)
          }}
        />
        <div><label className="block text-sm font-medium" htmlFor="benchmark">基准（可选，按代码或名称搜索后选择）</label><select id="benchmark" value={benchmark ? `${benchmark.kind}:${benchmark.product_id}` : ''} onChange={(event) => { const picked = candidates.find((item) => `${item.kind}:${item.product_id}` === event.target.value) ?? null; setBenchmark(picked) }} className="mt-1 w-full rounded-lg border border-slate-300 p-2"><option value="">不设基准</option>{candidates.map((item) => <option key={`${item.kind}:${item.product_id}`} value={`${item.kind}:${item.product_id}`}>{item.name}</option>)}</select></div>
        {!classBudgetsValid && <p role="status" className="text-sm text-amber-800">请先让各大类的产品权重合计符合上方 TAA 预算。</p>}
        <button id="portfolio-step-4" type="button" disabled={!readyToRun || running} onClick={handleRun} className="w-full rounded-lg bg-emerald-700 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400">{running ? '正在使用真实历史数据运行…' : '运行组合研究'}</button>
      </div>
    </section>
    {result && <section className="rounded-xl border border-emerald-200 bg-white p-4 shadow-sm sm:p-6" aria-live="polite"><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">运行结果</h2><p className="text-sm text-slate-600">运行编号：{result.id}</p></div>{result.target_id ? <Link to={`/post-investment/research-diagnosis?target=${encodeURIComponent(result.target_id)}&run=${encodeURIComponent(result.id)}`} className="rounded-lg bg-slate-900 px-3 py-2 text-sm text-white">进入持仓诊断</Link> : <button type="button" onClick={handleSave} disabled={saving} className="rounded-lg bg-slate-900 px-3 py-2 text-sm text-white">{saving ? '保存中…' : '保存为研究对象'}</button>}</div><div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">{result.metrics.slice(0, 8).map((metric) => <div key={metric.name} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">{metric.name}</p><p className="mt-1 font-semibold"><MetricValue value={metric.value} presentation={resultMetricPresentation(metric)} /></p></div>)}</div>{result.warnings.length > 0 && <ul className="mt-4 list-disc rounded-lg bg-amber-50 px-6 py-3 text-sm text-amber-900">{result.warnings.map((warning) => <li key={warning}>{humanizeIndicatorMessage(warning)}</li>)}</ul>}<div className="mt-5"><RegimeConditioningPanel result={result.regime_conditioning} /></div></section>}
    {result?.id && <PortfolioRiskSection key={result.id} portfolioRunId={result.id} context="测试上方已保存运行的期末持仓。继续编辑组合不会改写这份快照；新方案请先重新运行，再做压测。" />}
  </div>
}
