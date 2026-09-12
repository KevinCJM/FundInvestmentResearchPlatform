import { useEffect, useMemo, useRef, useState } from 'react'
import { EmptyState } from '../components/ui'
import { useSearchParams } from 'react-router-dom'
import { searchInstruments, type InstrumentSearchItem } from '../services/customIndicators'
import { editableTimingDefinition, timingApi, timingNumber, timingPercent, type SavedTimingDefinition, type TimingCatalog, type TimingComparison, type TimingDefinition, type TimingJob, type TimingRequest, type TimingRun, type TimingRunSummary } from '../services/timingResearch'
import TimingRuleEditor, { timingField } from '../components/timing-research/TimingRuleEditor'
import TimingResults from '../components/timing-research/TimingResults'
import TimingReleaseLibrary from '../components/timing-research/TimingReleaseLibrary'
import TimingTrainingEditor, { timingTrainingError } from '../components/timing-research/TimingTrainingEditor'
import { TimingAdaptationNote, TimingBasketEditor, basketsFromDraft, basketValidation, type BasketDraft } from '../components/timing-research/TimingStudyContext'

const button = 'min-h-10 rounded-xl border border-slate-300 bg-white px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-40'
const primary = 'min-h-10 rounded-lg bg-accent-600 px-4 py-2 text-sm font-semibold text-white hover:bg-accent-700 disabled:cursor-not-allowed disabled:opacity-40'
const emptyDefinition = (): TimingDefinition => ({ name: '我的择时算法', description: '', nodes: [], entry: '', exit: null, execution: { take_profit: .15, stop_loss: .15, max_holding_bars: 15, cooldown_bars: 0, fee_bps: 3, slippage_bps: 2 } })
type Mode = 'library' | 'research' | 'application'
type Product = { product_id: string; name: string }
const errorText = (error: unknown) => error instanceof Error ? error.message : '操作未完成，请稍后重试。'
const currentDefinition = (definition: TimingDefinition) => editableTimingDefinition(definition)
function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const aborted = () => { window.clearTimeout(timer); reject(new DOMException('Aborted', 'AbortError')) }
    const timer = window.setTimeout(() => { signal.removeEventListener('abort', aborted); resolve() }, milliseconds)
    if (signal.aborted) aborted(); else signal.addEventListener('abort', aborted, { once: true })
  })
}

export default function TimingResearch({ mode = 'research' }: { mode?: Mode }) {
  const [params] = useSearchParams()
  const [catalog, setCatalog] = useState<TimingCatalog | null>(null)
  const [definitions, setDefinitions] = useState<SavedTimingDefinition[]>([])
  const [definition, setDefinition] = useState<TimingDefinition>(emptyDefinition)
  const [saved, setSaved] = useState<SavedTimingDefinition | undefined>()
  const [templateId, setTemplateId] = useState('')
  const [targets, setTargets] = useState<Product[]>(() => (params.get('product_id') || params.get('ids') || '510300.SH').split(',').filter(Boolean).slice(0, 12).map(product_id => ({ product_id, name: product_id })))
  const [search, setSearch] = useState('')
  const [products, setProducts] = useState<InstrumentSearchItem[]>([])
  const [searchError, setSearchError] = useState('')
  const [startDate, setStartDate] = useState('2020-01-01')
  const [endDate, setEndDate] = useState(params.get('as_of') || '2026-09-03')
  const [holdout, setHoldout] = useState('2024-01-01')
  const [priceBasis, setPriceBasis] = useState<TimingRequest['price_basis']>('hfq')
  const [basketDraft, setBasketDraft] = useState<BasketDraft>({ market: '', category: '' })
  const [run, setRun] = useState<TimingRun | null>(null)
  const [runKey, setRunKey] = useState('')
  const [job, setJob] = useState<TimingJob | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [loadAttempt, setLoadAttempt] = useState(0)
  const [tab, setTab] = useState<'editor' | 'results' | 'history' | 'releases'>(mode === 'application' ? 'releases' : 'editor')
  const [settingsOpen, setSettingsOpen] = useState(true)
  const [history, setHistory] = useState<TimingRunSummary[]>([])
  const [comparisonIds, setComparisonIds] = useState<string[]>([])
  const [comparison, setComparison] = useState<TimingComparison | null>(null)
  const [note, setNote] = useState('')
  const activeOperation = useRef<AbortController | null>(null)
  const mounted = useRef(true)
  const version = useRef(0)
  const study = useMemo(() => ({ definition, targets: targets.map(target => ({ kind: 'etf' as const, product_id: target.product_id })), start_date: startDate, end_date: endDate, holdout_start: holdout, walk_forward_splits: 3, price_basis: priceBasis, context_baskets: basketsFromDraft(basketDraft) }), [definition, targets, startDate, endDate, holdout, priceBasis, basketDraft])
  const studyKey = JSON.stringify(study)
  const stale = !!run && runKey !== studyKey
  const unsupportedKind = !!params.get('kind') && params.get('kind') !== 'etf'
  const trainingError = catalog ? timingTrainingError(definition, catalog) : ''
  const validationError = unsupportedKind ? '目前支持场内 ETF。场外基金需要独立的申赎执行规则。' : !targets.length ? '请先添加一个 ETF。' : !startDate || !endDate || startDate >= endDate ? '研究结束日必须晚于开始日。' : !holdout || holdout <= startDate || holdout > endDate ? '样本外开始日应晚于研究开始日，且不晚于结束日。' : !definition.name.trim() ? '请填写算法名称。' : !definition.nodes.length || !definition.entry ? '请添加计算步骤，并选择买入条件。' : trainingError || basketValidation(definition, basketDraft)
  const title = ({ library: '择时算法中心', research: '产品择时研究', application: '产品配置与择时' })[mode]
  const change = (next: TimingDefinition) => { version.current += 1; setDefinition(next); setNotice(''); setError('') }
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; activeOperation.current?.abort() }
  }, [])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    void Promise.all([timingApi.catalog(controller.signal), timingApi.definitions(controller.signal), timingApi.runs(controller.signal)]).then(([nextCatalog, nextDefinitions, nextHistory]) => {
      if (controller.signal.aborted) return
      setCatalog(nextCatalog); setDefinitions(nextDefinitions.items); setHistory(nextHistory.items)
      if (nextCatalog.templates[0]) { setDefinition(currentDefinition(nextCatalog.templates[0].definition)); setTemplateId(nextCatalog.templates[0].id) }
    }).catch(reason => { if (!controller.signal.aborted) setError(errorText(reason)) })
    return () => controller.abort()
  }, [loadAttempt])
  useEffect(() => {
    const controller = new AbortController()
    setProducts([]); setSearchError('')
    if (!search.trim()) return () => controller.abort()
    const timer = window.setTimeout(() => {
      void searchInstruments({ kind: 'etf', query: search, pageSize: 8, signal: controller.signal }).then(response => {
        if (!controller.signal.aborted) setProducts(response.items)
      }).catch(reason => { if (!controller.signal.aborted) setSearchError(errorText(reason)) })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [search])
  const addTarget = (productId: string, name = productId) => {
    if (targets.length >= 12 || targets.some(target => target.product_id === productId)) return
    version.current += 1; setTargets(previous => [...previous, { product_id: productId, name }]); setSearch('')
  }
  const applyTemplate = (id: string) => {
    const template = catalog?.templates.find(item => item.id === id)
    if (template) { change(currentDefinition(template.definition)); setTemplateId(id); setSaved(undefined); setTab('editor') }
  }
  const save = async () => {
    setBusy('save'); setError(''); setNotice('')
    const captured = definition, capturedVersion = version.current
    try {
      const next = await timingApi.save(captured, saved)
      if (!mounted.current) return
      setDefinitions(previous => [next, ...previous.filter(item => item.id !== next.id)])
      if (version.current === capturedVersion) setSaved(next)
      setNotice('算法已保存。以后修改会产生新版本，历史研究保持原样。')
    } catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  const execute = async () => {
    if (validationError) { setError(validationError); return }
    activeOperation.current?.abort()
    const controller = new AbortController(); activeOperation.current = controller
    const snapshot = study, snapshotKey = studyKey
    setBusy('run'); setJob(null); setError(''); setNotice('')
    try {
      const prepared = await timingApi.prepare(snapshot.definition, controller.signal)
      let next = await timingApi.run({ ...snapshot, compile_token: prepared.compile_token }, controller.signal)
      if (controller.signal.aborted) return
      setJob(next)
      while (next.status === 'queued' || next.status === 'running') {
        await delay(1000, controller.signal)
        next = await timingApi.job(next.id, controller.signal)
        if (controller.signal.aborted) return
        setJob(next)
      }
      if (next.status === 'failed') throw new Error(next.error || '研究计算未完成，请检查数据后重试。')
      if (!next.run_id) throw new Error('任务没有返回研究结果，请刷新历史记录。')
      const result = await timingApi.getRun(next.run_id, controller.signal)
      if (controller.signal.aborted) return
      setRun(result); setRunKey(snapshotKey); setTab('results'); setJob(null); setSettingsOpen(false)
      setHistory(previous => [{ id: result.id, name: result.name, created_at: result.created_at }, ...previous.filter(item => item.id !== result.id)])
    } catch (reason) { if (!controller.signal.aborted) setError(errorText(reason)) }
    finally { if (!controller.signal.aborted && mounted.current) { setBusy(null); setJob(null) } }
  }
  const loadRun = async (id: string) => {
    activeOperation.current?.abort()
    const controller = new AbortController(); activeOperation.current = controller
    const capturedVersion = version.current
    setBusy('load'); setError('')
    try {
      const result = await timingApi.getRun(id, controller.signal)
      if (controller.signal.aborted) return
      if (capturedVersion !== version.current) { setNotice('读取期间草稿已修改，请再次选择历史研究。'); return }
      const request = result.request_snapshot
      const next = currentDefinition(result.definition_snapshot)
      setDefinition(next); setSaved(undefined); setTemplateId(''); setTargets(request.targets.map(target => ({ product_id: target.product_id, name: target.product_id })))
      setStartDate(request.start_date); setEndDate(request.end_date); setHoldout(request.holdout_start); setPriceBasis(request.price_basis)
      const baskets = { market: request.context_baskets?.market || [], category: request.context_baskets?.category || [] }
      setBasketDraft({ market: baskets.market.join(', '), category: baskets.category.join(', ') })
      setRun(result); setRunKey(JSON.stringify({ definition: next, targets: request.targets, start_date: request.start_date, end_date: request.end_date, holdout_start: request.holdout_start, walk_forward_splits: 3, price_basis: request.price_basis, context_baskets: baskets })); setTab('results'); setSettingsOpen(false)
    } catch (reason) { if (!controller.signal.aborted) setError(errorText(reason)) }
    finally { if (!controller.signal.aborted && mounted.current) setBusy(null) }
  }
  const compare = async () => {
    setBusy('compare'); setError('')
    try { const result = await timingApi.compare(comparisonIds); if (mounted.current) setComparison(result) }
    catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  const publish = async (bind = false) => {
    if (!run || stale) return
    setBusy('release'); setError(''); setNotice('')
    try {
      const release = await timingApi.release(run.id, note)
      if (bind) await timingApi.bind(release.id, 'pre_investment', note)
      if (mounted.current) setNotice(bind ? '已引用到投前研究，未改变组合权重或生成交易。' : '研究版本已保存，可以在投前引用此固定结果。')
    } catch (reason) { if (mounted.current) setError(errorText(reason)) }
    finally { if (mounted.current) setBusy(null) }
  }
  return <main className="mx-auto w-full min-w-0 max-w-[1600px] space-y-5 p-3 sm:p-5 lg:p-6">
    <header className="flex flex-wrap items-start justify-between gap-4"><div className="min-w-0"><h1 className="text-2xl font-semibold tracking-tight text-slate-950">{title}</h1><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">选择 ETF，组合计算与买卖规则，用真实历史数据检验效果。</p>{mode === 'application' && <p className="mt-1 text-xs leading-5 text-slate-600">研究版本引用不自动交易、不改变组合权重；改变大类配置比例时进入 TAA 决策。</p>}</div><span className="rounded-full bg-accent-50 px-3 py-1 text-xs font-medium text-accent-700">日频 ETF · 研究用途</span></header>
    {error && <div role="alert" className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"><p>{error}</p>{!catalog && <button type="button" className="min-h-9 px-2 underline" onClick={() => setLoadAttempt(value => value + 1)}>重新加载</button>}</div>}
    {notice && <p role="status" className="rounded-xl bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    {!catalog && !error && <p role="status" className="p-8 text-sm text-slate-600">正在读取算法目录…</p>}
    {catalog && tab !== 'releases' && <button type="button" aria-expanded={settingsOpen} aria-controls="timing-study-settings" className="flex w-full items-center justify-between gap-3 rounded-xl border border-slate-200 bg-white p-3 text-left lg:hidden" onClick={() => setSettingsOpen(value => !value)}><span className="min-w-0"><strong className="block text-sm text-slate-800">研究设置 · {targets.length} 个 ETF</strong><span className="text-xs text-slate-600">{startDate} 至 {endDate}</span></span><span className="shrink-0 text-xs text-accent-700">{settingsOpen ? '收起' : '展开修改'}</span></button>}
    {catalog && <div className={`grid min-w-0 items-start gap-5 ${tab === 'releases' ? '' : 'lg:grid-cols-[280px_minmax(0,1fr)]'}`}>
      {tab !== 'releases' && <aside id="timing-study-settings" className={`${settingsOpen ? 'block' : 'hidden'} min-w-0 space-y-5 rounded-xl border border-slate-200 bg-white p-4 lg:block`}>
        <section className="space-y-3"><h2 className="text-sm font-semibold text-slate-900">1. 选择研究对象</h2><div className="flex flex-wrap gap-2">{targets.map(target => <span key={target.product_id} className="inline-flex max-w-full items-center gap-1 rounded-lg bg-slate-100 py-1 pl-2 text-xs text-slate-700"><span className="truncate" title={target.name}>{target.name}</span><button type="button" aria-label={`移除 ${target.product_id}`} className="min-h-8 min-w-8 rounded-lg hover:bg-slate-200" onClick={() => { version.current += 1; setTargets(targets.filter(item => item.product_id !== target.product_id)) }}>×</button></span>)}</div>
          <label className="block text-xs text-slate-600">搜索 ETF 名称或代码<input className={timingField} value={search} onChange={event => setSearch(event.target.value)} placeholder="例如 510300" disabled={targets.length >= 12} /></label>
          {!!products.length && <ul aria-label="ETF 搜索结果" className="max-h-56 overflow-auto rounded-lg border border-slate-200">{products.map(item => { const code = item.code || item.ts_code || ''; return <li key={code}><button type="button" className="min-h-11 w-full px-3 py-2 text-left text-xs hover:bg-accent-50 disabled:opacity-40" disabled={!code || targets.some(target => target.product_id === code)} onClick={() => addTarget(code, item.name || code)}>{item.name || code}<span className="ml-2 text-slate-600">{code}</span></button></li> })}</ul>}
          {searchError && <p className="text-xs text-amber-700">{searchError}</p>}
          {/^[0-9]{6}\.(SH|SZ)$/i.test(search.trim()) && !targets.some(target => target.product_id === search.trim().toUpperCase()) && <button type="button" className="min-h-9 text-xs text-accent-700" onClick={() => addTarget(search.trim().toUpperCase())}>添加代码 {search.trim().toUpperCase()}</button>}
          <p className="text-xs text-slate-600">最多 12 个 ETF，分别检验；不合成为投资组合。</p>
          <label className="block text-xs text-slate-600">研究开始日<input type="date" className={timingField} value={startDate} onChange={event => { version.current += 1; setStartDate(event.target.value) }} /></label><label className="block text-xs text-slate-600">研究结束日<input type="date" className={timingField} value={endDate} onChange={event => { version.current += 1; setEndDate(event.target.value) }} /></label><label className="block text-xs text-slate-600">样本外开始日<input type="date" className={timingField} value={holdout} onChange={event => { version.current += 1; setHoldout(event.target.value) }} /></label><p className="text-xs leading-5 text-slate-600">在前段构建规则，查看后段表现。反复查看后调整参数，会消耗这段样本的独立性。</p>
        </section>
        <section className="space-y-3 border-t border-slate-100 pt-4"><h2 className="text-sm font-semibold text-slate-900">2. 选择起点</h2><label className="block text-xs text-slate-600">内置模板<select className={timingField} value={templateId} onChange={event => applyTemplate(event.target.value)}><option value="">自建算法</option>{catalog.templates.map(template => <option key={template.id} value={template.id}>{template.label}</option>)}</select></label>{templateId && <p className="text-xs leading-5 text-slate-600">{catalog.templates.find(template => template.id === templateId)?.description}</p>}
          <label className="block text-xs text-slate-600">已保存算法<select className={timingField} value={saved?.id || ''} onChange={event => { const item = definitions.find(value => value.id === event.target.value); if (item) { change(currentDefinition(item)); setSaved(item); setTemplateId(''); setTab('editor') } }}><option value="">选择已保存算法</option>{definitions.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.revision}</option>)}</select></label><button type="button" className="min-h-9 text-xs font-medium text-accent-700" onClick={() => { change(emptyDefinition()); setSaved(undefined); setTemplateId(''); setTab('editor') }}>从空白开始</button>
        </section>
        <TimingBasketEditor value={basketDraft} definition={definition} onChange={next => { version.current += 1; setBasketDraft(next) }} />
        <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer py-1 text-sm font-medium text-slate-700">交易与价格设置</summary><p className="mt-2 text-xs leading-5 text-slate-600">下一交易日开盘执行；买入当天不卖出。同日同时触及止盈止损时，按止损处理。</p><div className="mt-3 space-y-3"><label className="block text-xs text-slate-600">价格口径<select className={timingField} value={priceBasis} onChange={event => { version.current += 1; setPriceBasis(event.target.value as TimingRequest['price_basis']) }}><option value="hfq">后复权（研究收益）</option><option value="qfq">前复权（截止日锚定）</option><option value="raw">原始交易价格</option></select></label>{([
          ['take_profit', '止盈幅度（%）', 100], ['stop_loss', '止损幅度（%）', 100], ['max_holding_bars', '最长持有交易日', 1], ['cooldown_bars', '退出后冷却交易日', 1], ['fee_bps', '单边手续费（基点）', 1], ['slippage_bps', '单边滑点（基点）', 1],
        ] as const).map(([key, label, scale]) => <label key={key} className="block text-xs text-slate-600">{label}<input type="number" className={timingField} min={key === 'max_holding_bars' ? 1 : 0} step={key.includes('bars') ? 1 : 'any'} value={definition.execution[key] * scale} onChange={event => change({ ...definition, execution: { ...definition.execution, [key]: Number(event.target.value) / scale } })} /></label>)}<p className="text-xs text-slate-600">1 基点 = 0.01%；费用与滑点双边计入。</p></div></details>
      </aside>}
      <div className="min-w-0 space-y-4">
        <div className="min-w-0 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
          {tab !== 'releases' && <div className="grid grid-cols-2 items-end gap-3 sm:flex sm:flex-wrap"><label className="col-span-2 min-w-0 flex-1 text-xs text-slate-600">算法名称<input className={timingField} value={definition.name} maxLength={100} onChange={event => change({ ...definition, name: event.target.value })} /></label><button type="button" className={button} disabled={!!busy || !definition.nodes.length} onClick={save}>{busy === 'save' ? '保存中…' : '保存算法'}</button><button type="button" className={primary} disabled={!!busy || !!validationError} onClick={execute}>{busy === 'run' ? job ? `研究中 ${job.progress}/${job.total}` : '准备计算…' : '运行研究'}</button></div>}
          {tab !== 'releases' && !!validationError && <p className="mt-2 text-xs text-amber-700">{validationError}</p>}
          {busy === 'run' && <div role="status" className="mt-4 rounded-lg bg-accent-50 p-3 text-sm text-accent-700">{job ? `正在研究第 ${Math.min(job.progress + 1, job.total)} / ${job.total} 个产品…` : '正在校验计算步骤并准备执行…'}<p className="mt-1 text-xs">可继续调整草稿；本次结果会保留启动时的配置。</p></div>}
          <div role="tablist" aria-label="择时工作区" className="mt-5 flex flex-wrap gap-4 border-b border-slate-200">{(mode === 'application' ? ['releases', 'editor', 'results', 'history'] as const : ['editor', 'results', 'history', 'releases'] as const).map(value => <button type="button" key={value} role="tab" aria-selected={tab === value} aria-controls="timing-workspace-panel" onClick={() => setTab(value)} className={`min-h-11 border-b-2 text-sm ${tab === value ? 'border-accent-600 font-semibold text-accent-700' : 'border-transparent text-slate-600'}`}>{({ editor: '3. 算法步骤', results: '4. 研究结果', history: '历史与对比', releases: '已有研究版本' })[value]}</button>)}</div>
          <div id="timing-workspace-panel" role="tabpanel" className="mt-5 min-w-0">
            {tab === 'releases' && <TimingReleaseLibrary onView={loadRun} disabled={!!busy} />}
            {tab === 'editor' && <div className="space-y-4"><TimingAdaptationNote adaptation={definition.adaptation} /><details><summary className="cursor-pointer text-xs text-slate-600">算法说明</summary><textarea className={timingField} rows={2} maxLength={2000} value={definition.description} onChange={event => change({ ...definition, description: event.target.value })} placeholder="记录你的研究假设和适用范围" /></details><TimingTrainingEditor definition={definition} catalog={catalog} onChange={change} /><TimingRuleEditor definition={definition} catalog={catalog} onChange={change} /></div>}
            {tab === 'results' && (run ? <>{stale && <p role="status" className="mb-4 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">配置已修改。下方为上次运行结果，请重新运行后再保存研究版本。</p>}<TimingResults key={run.id} run={run} /><div className="mt-5 space-y-3 border-t border-slate-200 pt-4"><label className="block text-xs text-slate-600">研究引用说明<input className={timingField} value={note} onChange={event => setNote(event.target.value)} placeholder="例如：仅用于宽基 ETF 的研究对照" /></label><div className="flex flex-wrap gap-2"><button type="button" className={button} disabled={!!busy || stale} onClick={() => publish()}>保存研究版本</button><button type="button" className={button} disabled={!!busy || stale} onClick={() => publish(true)}>引用到投前研究</button></div><p className="text-xs text-slate-600">研究引用锁定本次结果和适用范围，不自动生成交易。</p></div></> : <EmptyState title="还没有研究结果" hint="检查左侧产品与日期，调整算法后点击“运行研究”。" />)}
            {tab === 'history' && <div className="space-y-4"><div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-600">选择历史结果查看，或勾选 2–4 次研究比较。</p><button type="button" className={button} disabled={!!busy || comparisonIds.length < 2} onClick={compare}>比较选中研究</button></div>{!history.length && <p className="rounded-xl bg-slate-50 p-6 text-sm text-slate-600">运行研究后，历史记录会保存在这里。</p>}<div className="space-y-2">{history.map(item => <div key={item.id} className="flex items-center gap-3 rounded-xl border border-slate-200 p-3"><input type="checkbox" aria-label={`对比 ${item.name} ${item.id}`} className="h-4 w-4 shrink-0" checked={comparisonIds.includes(item.id)} disabled={!comparisonIds.includes(item.id) && comparisonIds.length >= 4} onChange={event => setComparisonIds(previous => event.target.checked ? [...previous, item.id] : previous.filter(id => id !== item.id))} /><button type="button" className="min-w-0 flex-1 text-left" disabled={!!busy} onClick={() => loadRun(item.id)}><strong className="block truncate text-sm text-slate-800">{item.name}</strong><span className="text-xs text-slate-600">{item.created_at?.replace('T', ' ').slice(0, 19)}</span></button><button type="button" className="min-h-9 shrink-0 px-2 text-xs text-accent-700" disabled={!!busy} onClick={() => loadRun(item.id)}>查看</button></div>)}</div>{comparison && <div className="overflow-auto rounded-lg border border-slate-200"><table aria-label="研究对比" className="w-full text-left text-xs [&_td]:whitespace-nowrap [&_td]:border-t [&_td]:p-3 [&_th]:whitespace-nowrap [&_th]:bg-slate-50 [&_th]:p-3"><thead><tr><th scope="col">研究</th><th scope="col">产品</th><th scope="col">样本外收益</th><th scope="col">最大回撤</th><th scope="col">已平仓交易</th></tr></thead><tbody>{comparison.items.flatMap(item => item.products.map(product => <tr key={`${item.run_id}:${product.product_id}`}><td>{item.name}</td><td>{product.product_id}</td><td>{timingPercent(product.summary?.out_of_sample.total_return)}</td><td>{timingPercent(product.summary?.out_of_sample.max_drawdown)}</td><td>{timingNumber(product.summary?.out_of_sample.trade_count, 0)}</td></tr>))}</tbody></table></div>}</div>}
          </div>
        </div>
      </div>
    </div>}
  </main>
}
