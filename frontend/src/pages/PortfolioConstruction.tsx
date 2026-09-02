import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  createResearchTarget,
  PortfolioConstituent,
  PortfolioInstrument,
  PortfolioMethod,
  PortfolioRun,
  PortfolioRunRequest,
  runPortfolio,
  searchPortfolioInstruments,
} from '../services/portfolioResearch'
import { MetricValue } from '../components/metrics/MetricDisplay'
import type { MetricPresentation } from '../services/customIndicators'
import { humanizeIndicatorMessage } from '../utils/indicatorDiagnostics'

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

export default function PortfolioConstruction() {
  const [step, setStep] = useState(1)
  const [name, setName] = useState('未命名组合研究')
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const [candidates, setCandidates] = useState<PortfolioInstrument[]>([])
  const [constituents, setConstituents] = useState<PortfolioConstituent[]>([])
  const [method, setMethod] = useState<PortfolioMethod>('equal_weight')
  const [minWeight, setMinWeight] = useState(0)
  const [maxWeight, setMaxWeight] = useState(100)
  const [windowMode, setWindowMode] = useState<'all' | 'rolling'>('all')
  const [observations, setObservations] = useState(252)
  const [rebalance, setRebalance] = useState<PortfolioRunRequest['rebalance']['frequency']>('monthly')
  const [benchmark, setBenchmark] = useState<PortfolioInstrument | null>(null)
  const [objective, setObjective] = useState<NonNullable<PortfolioRunRequest['objective']>>('max_sharpe')
  const [targetReturn, setTargetReturn] = useState(8)
  const [running, setRunning] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<PortfolioRun | null>(null)

  useEffect(() => {
    const raw = sessionStorage.getItem('portfolioResearchImport')
    if (!raw) return
    sessionStorage.removeItem('portfolioResearchImport')
    try {
      const imported = JSON.parse(raw) as { name?: string; method?: PortfolioMethod; constituents?: PortfolioConstituent[] }
      if (imported.name) setName(imported.name)
      if (Array.isArray(imported.constituents)) setConstituents(imported.constituents)
      if (imported.method && METHODS.some((item) => item.value === imported.method)) setMethod(imported.method)
    } catch {
      setError('导入的组合配置无法读取，请重新从来源页面导入。')
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    const trimmed = query.trim()
    if (!trimmed) { setCandidates([]); return () => controller.abort() }
    setSearching(true); setSearchError('')
    searchPortfolioInstruments(trimmed, controller.signal)
      .then((response) => setCandidates(response.items ?? []))
      .catch((caught) => { if (caught.name !== 'AbortError') setSearchError(caught.message ?? '产品搜索失败') })
      .finally(() => setSearching(false))
    return () => controller.abort()
  }, [query])

  const totalWeight = useMemo(() => constituents.reduce((sum, item) => sum + (item.weight ?? 0), 0), [constituents])
  const totalBudget = useMemo(() => constituents.reduce((sum, item) => sum + (item.risk_budget ?? 0), 0), [constituents])
  const readyToRun = constituents.length >= 2 && minWeight <= maxWeight &&
    (method !== 'manual' || Math.abs(totalWeight - 100) < 0.01) &&
    (method !== 'risk_budget' || Math.abs(totalBudget - 100) < 0.01)

  function addInstrument(item: PortfolioInstrument) {
    setConstituents((current) => current.some((entry) => entry.product_id === item.product_id && entry.kind === item.kind)
      ? current
      : [...current, { ...item, weight: 0, risk_budget: 0 }])
  }
  function updateNumber(index: number, field: 'weight' | 'risk_budget', raw: string) {
    const value = Math.max(0, Number(raw) || 0)
    setConstituents((current) => current.map((item, currentIndex) => currentIndex === index ? { ...item, [field]: value } : item))
  }
  function buildRequest(): PortfolioRunRequest {
    return {
      name: name.trim() || '未命名组合研究', constituents, method,
      constraints: { min_weight: minWeight / 100, max_weight: maxWeight / 100 },
      window: windowMode === 'all' ? { mode: 'all' } : { mode: 'rolling', observations },
      rebalance: { frequency: rebalance, transaction_cost_bps: 0 },
      benchmark: benchmark ? { kind: benchmark.kind, product_id: benchmark.product_id, name: benchmark.name } : null,
      objective: method === 'target_optimization' ? objective : null,
      target_return: method === 'target_optimization' && objective === 'target_return' ? targetReturn / 100 : null,
    }
  }
  async function handleRun() {
    if (!readyToRun) { setError('请至少选择两只产品，并完成当前权重或风险预算校验。'); return }
    setRunning(true); setError('')
    try {
      const target = await createResearchTarget({ name: name.trim() || '未命名组合研究', kind: 'portfolio', definition: buildRequest() })
      const run = await runPortfolio(target.id)
      setResult({ ...run, target_id: target.id })
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

  return <div className="mx-auto max-w-7xl space-y-6 p-4 sm:p-6" aria-busy={running || saving}>
    <header className="rounded-2xl bg-slate-900 px-5 py-6 text-white sm:px-7">
      <p className="text-sm text-emerald-300">工作区共享 · 真实历史数据</p><h1 className="mt-1 text-2xl font-semibold">产品组合构建</h1>
      <p className="mt-2 max-w-3xl text-sm text-slate-300">从真实 ETF 与公募基金构造可复现的组合运行；运行时会保存研究对象并生成不可变快照，供持仓诊断复用。</p>
    </header>
    <ol className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-4" aria-label="组合构建步骤">
      {['选择产品', '权重方法', '约束与回测', '运行与保存'].map((label, index) => <li key={label} className={`rounded-lg border px-3 py-2 ${step === index + 1 ? 'border-emerald-500 bg-emerald-50 font-semibold text-emerald-800' : 'border-slate-200 bg-white text-slate-600'}`}>{index + 1}. {label}</li>)}
    </ol>
    {error && <div role="alert" className="rounded-lg border border-rose-300 bg-rose-50 p-3 text-sm text-rose-800">{error}</div>}
    <section className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
      <div className="space-y-5 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <div><label className="block text-sm font-medium" htmlFor="portfolio-name">研究名称</label><input id="portfolio-name" value={name} onChange={(event) => setName(event.target.value)} className="mt-1 w-full rounded border border-slate-300 px-3 py-2" /></div>
        <div><label className="block text-sm font-medium" htmlFor="portfolio-search">搜索 ETF 或基金</label><input id="portfolio-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="输入代码或名称" className="mt-1 w-full rounded border border-slate-300 px-3 py-2" />
          <p aria-live="polite" className="mt-1 text-xs text-slate-500">{searching ? '正在检索真实产品…' : searchError || (query && !candidates.length ? '没有匹配产品。' : '')}</p>
          <ul className="mt-2 divide-y rounded border border-slate-200" aria-label="产品搜索结果">{candidates.map((item) => <li key={`${item.kind}-${item.product_id}`} className="flex items-center justify-between gap-3 px-3 py-2 text-sm"><span><b>{item.name}</b> <span className="text-slate-500">{item.code ?? item.product_id} · {item.kind === 'etf' ? 'ETF' : '公募基金'}</span></span><button type="button" onClick={() => addInstrument(item)} className="rounded border border-emerald-600 px-2 py-1 text-emerald-700">加入</button></li>)}</ul>
        </div>
        <div><h2 className="text-base font-semibold">已选产品（至少 2 只）</h2>{!constituents.length ? <p className="mt-2 text-sm text-slate-500">搜索并加入产品后开始构建。</p> : <ul className="mt-2 space-y-2">{constituents.map((item, index) => <li key={`${item.kind}-${item.product_id}`} className="grid grid-cols-[1fr_auto] gap-2 rounded border border-slate-200 p-2"><span className="text-sm"><b>{item.name}</b><br /><span className="text-xs text-slate-500">{item.code ?? item.product_id}</span></span><button type="button" onClick={() => setConstituents((current) => current.filter((_, currentIndex) => currentIndex !== index))} className="text-sm text-rose-700 underline">移除</button>{method === 'manual' && <label className="col-span-2 text-sm">权重 (%)<input aria-label={`${item.name} 权重`} type="number" min="0" max="100" value={item.weight ?? 0} onChange={(event) => updateNumber(index, 'weight', event.target.value)} className="ml-2 w-24 rounded border border-slate-300 px-2 py-1" /></label>}{method === 'risk_budget' && <label className="col-span-2 text-sm">风险预算 (%)<input aria-label={`${item.name} 风险预算`} type="number" min="0" max="100" value={item.risk_budget ?? 0} onChange={(event) => updateNumber(index, 'risk_budget', event.target.value)} className="ml-2 w-24 rounded border border-slate-300 px-2 py-1" /></label>}</li>)}</ul>}
          {(method === 'manual' || method === 'risk_budget') && <p className={`mt-2 text-sm ${Math.abs((method === 'manual' ? totalWeight : totalBudget) - 100) < .01 ? 'text-emerald-700' : 'text-amber-700'}`}>{method === 'manual' ? '权重' : '风险预算'}合计：{(method === 'manual' ? totalWeight : totalBudget).toFixed(2)}%（需为 100%）</p>}</div>
      </div>
      <div className="space-y-5 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-6">
        <fieldset><legend className="text-base font-semibold">权重方法</legend><div className="mt-2 grid gap-2">{METHODS.map((item) => <label key={item.value} className={`cursor-pointer rounded border p-3 ${method === item.value ? 'border-emerald-500 bg-emerald-50' : 'border-slate-200'}`}><input type="radio" name="method" checked={method === item.value} onChange={() => setMethod(item.value)} /> <b className="ml-1">{item.label}</b><span className="block pl-5 text-xs text-slate-600">{item.help}</span></label>)}</div></fieldset>
        {method === 'target_optimization' && <div className="space-y-2"><label className="block text-sm">优化目标<select value={objective} onChange={(event) => setObjective(event.target.value as typeof objective)} className="ml-2 rounded border border-slate-300 p-1"><option value="max_sharpe">最大夏普</option><option value="min_volatility">最小波动</option><option value="target_return">目标收益</option></select></label>{objective === 'target_return' && <label className="block text-sm">目标年化收益 (%)<input type="number" value={targetReturn} onChange={(event) => setTargetReturn(Number(event.target.value))} className="ml-2 w-24 rounded border border-slate-300 p-1" /></label>}</div>}
        <fieldset className="space-y-2"><legend className="text-base font-semibold">约束、窗口与调仓</legend><div className="grid grid-cols-2 gap-3 text-sm"><label>最小权重 (%)<input type="number" value={minWeight} onChange={(event) => setMinWeight(Number(event.target.value))} className="mt-1 w-full rounded border border-slate-300 p-2" /></label><label>最大权重 (%)<input type="number" value={maxWeight} onChange={(event) => setMaxWeight(Number(event.target.value))} className="mt-1 w-full rounded border border-slate-300 p-2" /></label><label>样本窗口<select value={windowMode} onChange={(event) => setWindowMode(event.target.value as 'all' | 'rolling')} className="mt-1 w-full rounded border border-slate-300 p-2"><option value="all">全历史</option><option value="rolling">滚动观察值</option></select></label><label>调仓<select value={rebalance} onChange={(event) => setRebalance(event.target.value as typeof rebalance)} className="mt-1 w-full rounded border border-slate-300 p-2"><option value="fixed">固定权重</option><option value="weekly">周度</option><option value="monthly">月度</option><option value="yearly">年度</option></select></label>{windowMode === 'rolling' && <label>观察值<input type="number" min="2" value={observations} onChange={(event) => setObservations(Number(event.target.value))} className="mt-1 w-full rounded border border-slate-300 p-2" /></label>}</div><p className="text-xs text-slate-500">当前组合研究固定交易成本为 0，仅输出未扣费的毛收益。</p></fieldset>
        <div><label className="block text-sm font-medium" htmlFor="benchmark">基准（可选，按代码或名称搜索后选择）</label><select id="benchmark" value={benchmark ? `${benchmark.kind}:${benchmark.product_id}` : ''} onChange={(event) => { const picked = candidates.find((item) => `${item.kind}:${item.product_id}` === event.target.value) ?? null; setBenchmark(picked) }} className="mt-1 w-full rounded border border-slate-300 p-2"><option value="">不设基准</option>{candidates.map((item) => <option key={`${item.kind}:${item.product_id}`} value={`${item.kind}:${item.product_id}`}>{item.name}</option>)}</select></div>
        <button type="button" disabled={!readyToRun || running} onClick={handleRun} className="w-full rounded bg-emerald-700 px-4 py-2 font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-400">{running ? '正在使用真实历史数据运行…' : '运行组合研究'}</button>
      </div>
    </section>
    {result && <section className="rounded-2xl border border-emerald-200 bg-white p-4 shadow-sm sm:p-6" aria-live="polite"><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">运行结果</h2><p className="text-sm text-slate-600">运行编号：{result.id}</p></div>{result.target_id ? <Link to={`/holding-diagnosis?target=${encodeURIComponent(result.target_id)}&run=${encodeURIComponent(result.id)}`} className="rounded bg-slate-900 px-3 py-2 text-sm text-white">进入持仓诊断</Link> : <button type="button" onClick={handleSave} disabled={saving} className="rounded bg-slate-900 px-3 py-2 text-sm text-white">{saving ? '保存中…' : '保存为研究对象'}</button>}</div><div className="mt-4 grid grid-cols-2 gap-3 md:grid-cols-4">{result.metrics.slice(0, 8).map((metric) => <div key={metric.name} className="rounded bg-slate-50 p-3"><p className="text-xs text-slate-500">{metric.name}</p><p className="mt-1 font-semibold"><MetricValue value={metric.value} presentation={resultMetricPresentation(metric)} /></p></div>)}</div>{result.warnings.length > 0 && <ul className="mt-4 list-disc rounded bg-amber-50 px-6 py-3 text-sm text-amber-900">{result.warnings.map((warning) => <li key={warning}>{humanizeIndicatorMessage(warning)}</li>)}</ul>}</section>}
  </div>
}
