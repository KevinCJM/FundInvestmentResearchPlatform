import { useEffect, useMemo, useReducer, useRef, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { fetchSourceCatalog, type SourceCatalog } from '../../services/dataSources'
import { blankStep, cancelEtl, deleteEtlWorkflow, downloadStep, emptyDefinition, listEtlRuns, listEtlWorkflows, quickPlan, recoverEtl, getEtlRun, runEtl, saveEtlWorkflow, stepLabels, validateEtl, type EtlDefinition, type EtlRun, type EtlValidation, type EtlWorkflow, type EtlRunOptions } from '../../services/etl'
import { buttonClass, inputClass, primaryClass, validateEditor } from './EditorFields'
import EtlDownloadFields from './EtlDownloadFields'
import EtlWorkflowEditor from './EtlWorkflowEditor'
import EtlTemplatePicker from './EtlTemplatePicker'
import EtlRunHistory from './EtlRunHistory'
import EtlRunOptionsEditor from './EtlRunOptionsEditor'
import AutoIncrementalWorkspace from './AutoIncrementalWorkspace'
import AutoPlanSummary from './AutoPlanSummary'
import { runModeLabels } from '../../services/etl'

import { createTimelineReducer } from '../computation-graph/history'
import { asGraph, etlPositions } from './etlGraphAdapter'

const draftReducer = createTimelineReducer<EtlDefinition>()

export default function DataDownloadWorkspace() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [catalog, setCatalog] = useState<SourceCatalog | null>(null)
  const [workflows, setWorkflows] = useState<EtlWorkflow[]>([])
  const [runs, setRuns] = useState<EtlRun[]>([])
  const [focusedRun, setFocusedRun] = useState('')
  const recoveryRequests = useRef(new Map<string, string>())
  const seenRecoveries = useRef(new Set<string>())
  const runGeneration = useRef(0)
  const [view, setView] = useState<'auto' | 'quick' | 'workflow' | 'runs'>(() => {
    const saved = searchParams.get('downloadView')
    return saved === 'auto' || saved === 'workflow' || saved === 'runs' ? saved : 'quick'
  })
  const [sourceId, setSourceId] = useState(searchParams.get('source') ?? '')
  const [selected, setSelected] = useState<string[]>([])
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'full' | 'incremental'>('incremental')
  const [runOptions, setRunOptions] = useState<EtlRunOptions>({ mode: 'incremental', parameters: {} })
  const [history, setHistory] = useState(true)
  const [downloadParams, setDownloadParams] = useState<Record<string, Record<string, unknown>>>({})
  const [snapshot, setSnapshot] = useState(false)
  const [emptyAllowed, setEmptyAllowed] = useState<Record<string, boolean>>({})
  const [timeline, dispatchDraft] = useReducer(draftReducer, { past: [], present: emptyDefinition(), future: [] })
  const draft = timeline.present
  const setDraft = (definition: EtlDefinition) => dispatchDraft({ type: 'reset', definition })
  const [saved, setSaved] = useState<EtlWorkflow | null>(null)
  const [validation, setValidation] = useState<EtlValidation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [retry, setRetry] = useState(0)
  const [statusReady, setStatusReady] = useState(false)
  const form = useRef<HTMLFormElement>(null)
  const submitting = useRef(false)
  const pending = useRef<{ key: string; id: string } | null>(null)
  const dirty = saved ? JSON.stringify(draft) !== JSON.stringify(saved.definition) : draft.steps.length > 0
  useEffect(() => {
    setSearchParams(previous => {
      if (previous.get('downloadView') === view) return previous
      const next = new URLSearchParams(previous); next.set('downloadView', view); return next
    }, { replace: true })
  }, [view, setSearchParams])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    Promise.all([fetchSourceCatalog(controller.signal), listEtlWorkflows()]).then(([c, w]) => { if (!controller.signal.aborted) { setCatalog(c); setWorkflows(w) } }).catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '无法加载数据源与流程。') })
    return () => controller.abort()
  }, [retry])
  useEffect(() => {
    let cancelled = false
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      const generation = runGeneration.current
      try {
        const value = await listEtlRuns()
        if (focusedRun && !value.some(run => run.run_id === focusedRun)) value.push(await getEtlRun(focusedRun))
        if (!cancelled && generation === runGeneration.current) { setRuns(value); setStatusReady(true) }
      }
      catch { if (!cancelled) setStatusReady(false) }
      if (!cancelled) timer = setTimeout(poll, 3000)
    }
    void poll()
    return () => { cancelled = true; clearTimeout(timer) }
  }, [retry, focusedRun])
  useEffect(() => {
    for (const run of runs) {
      const job = run.recovery?.job
      if (job && recoveryRequests.current.get(run.run_id) === job.id) recoveryRequests.current.delete(run.run_id)
      if (job?.status === 'SUCCEEDED' && !seenRecoveries.current.has(job.id)) {
        seenRecoveries.current.add(job.id); setFocusedRun(job.target_run_id)
      }
    }
  }, [runs])
  useEffect(() => {
    if (!dirty) return
    const warn = (e: BeforeUnloadEvent) => { e.preventDefault(); e.returnValue = '' }
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [dirty])
  const source = catalog?.sources.find(s => s.config.id === sourceId) ?? catalog?.sources[0]
  const choices = useMemo(() => catalog?.interfaces.filter(i => i.config.source_id === source?.config.id) ?? [], [catalog, source])
  const selectedRecords = useMemo(() => choices.filter(i => selected.includes(i.config.id)), [choices, selected])
  const plan = useMemo(() => {
    const value = quickPlan(selectedRecords, mode, downloadParams, history)
    value.steps = value.steps.map(s => s.kind === 'download' ? { ...s, allow_empty: emptyAllowed[s.interface_id ?? ''] ?? false } : s)
    if (catalog) value.steps = value.steps.map(s => s.kind === 'resolve' ? { ...s, name: `取值 · ${catalog.targets.tables.find(t => t.table_id === s.table_id)?.label ?? s.table_id}` } : s)
    const results = value.steps.filter(s => s.kind === 'resolve')
    if (snapshot && ['master.instrument', 'market.nav_daily'].every(t => results.some(s => s.table_id === t))) {
      value.steps.push({ ...blankStep('snapshot'), inputs: results.filter(s => ['master.instrument', 'market.nav_daily', 'market.quote_daily', 'master.trading_calendar'].includes(s.table_id ?? '')).map(s => s.id) })
    }
    return value
  }, [catalog, selectedRecords, mode, downloadParams, history, snapshot, emptyAllowed])
  const canSnapshot = ['master.instrument', 'market.nav_daily'].every(table => selectedRecords.some(i => i.config.mappings.some(m => m.enabled && m.target_table === table)))
  const executing = runs.some(r => r.status === 'RUNNING' || ['QUEUED', 'RUNNING'].includes(r.recovery?.job?.status ?? ''))
  const changeDraft = (value: EtlDefinition) => { dispatchDraft({ type: 'edit', definition: value }); setValidation(null); setNotice('') }
  const perform = async (operation: () => Promise<void>) => {
    if (submitting.current) return
    submitting.current = true; setBusy(true); setError(''); setNotice('')
    try { await operation() } catch (reason) { setError(reason instanceof Error ? reason.message : '操作未完成。') }
    finally { submitting.current = false; setBusy(false) }
  }
  const runAction = async (id: string, operation: (id: string) => Promise<EtlRun>) => {
    if (submitting.current) throw new Error('已有操作正在提交，请等待当前请求结束。')
    submitting.current = true; setBusy(true); runGeneration.current++
    try {
      const result = await operation(id)
      runGeneration.current++
      setRuns(current => current.map(run => run.run_id === id ? result : run))
    } catch (reason) {
      if (reason instanceof TypeError) throw new Error('无法连接后端，请求是否被接受尚未确认。请先等待状态刷新，避免重复提交。')
      throw reason
    } finally { submitting.current = false; setBusy(false) }
  }
  const recover = async (id: string) => {
    if (submitting.current) throw new Error('已有操作正在提交，请等待。')
    submitting.current = true; setBusy(true); runGeneration.current++
    try {
      const requestId = recoveryRequests.current.get(id) ?? crypto.randomUUID()
      recoveryRequests.current.set(id, requestId)
      const job = await recoverEtl(id, requestId)
      runGeneration.current++
      recoveryRequests.current.delete(id)
      setRuns(value => value.map(run => run.run_id === id ? { ...run, recovery: { can_resume: false, artifact_check_pending: true, blockers: [], ...run.recovery, job } } : run))
      if (job.status === 'SUCCEEDED') setFocusedRun(job.target_run_id)
    } catch (reason) {
      // Keep the request ID after an uncertain response, so retries cannot fork downloads.
      throw reason instanceof Error ? reason : new Error('恢复请求未完成，请等待状态刷新后重试。')
    } finally { submitting.current = false; setBusy(false) }
  }
  const launch = (definition: EtlDefinition) => void perform(async () => {
    if ((view !== 'workflow' || runOptions.mode !== 'auto_incremental') && !validateEditor(form.current)) return
    const options = view === 'quick' ? { mode, parameters: {} } : runOptions
    if (options.mode === 'auto_incremental' && !validation?.auto_plan?.ready) {
      setError('请先点击“分析自动增量区间”，查看快照和实际下载范围。'); return
    }
    const checked = await validateEtl(definition, options)
    setValidation(checked)
    if (!checked.valid) return
    if (options.mode === 'auto_incremental' && checked.auto_plan?.plan_id !== validation?.auto_plan?.plan_id) {
      setNotice('数据快照或运行范围已变化，请查看更新后的计划再确认。'); return
    }
    if (!window.confirm(`运行“${definition.name}”？\n本次模式：${runModeLabels[options.mode]}。\n${definition.steps.length} 个步骤，按校验后的依赖顺序执行。\n${checked.auto_plan ? `基线：${checked.auto_plan.snapshot}；请求截止：${checked.auto_plan.cutoff_date}。\n` : ''}下载将消耗来源配额；结果为候选，不覆盖正式研究数据。`)) return
    const key = JSON.stringify({ definition, options })
    if (pending.current?.key !== key) pending.current = { key, id: crypto.randomUUID() }
    const result = checked.auto_plan
      ? await runEtl(definition, pending.current!.id, options, checked.auto_plan.plan_id)
      : await runEtl(definition, pending.current!.id, options)
    pending.current = null
    setRuns(current => [result, ...current.filter(r => r.run_id !== result.run_id)])
    setView('runs'); setNotice('任务已提交。关闭页面不会取消下载；再次打开可查看进度。')
  })
  const save = (copy: boolean) => void perform(async () => {
    if (!validateEditor(form.current)) return
    const identifier = copy || !saved ? 'flow_' + crypto.randomUUID().replace(/-/g, '') : saved.id
    const graph = asGraph(draft)
    const stored = { ...graph, canvas: { version: 1 as const, ...graph.canvas, positions: etlPositions(graph) } }
    const result = await saveEtlWorkflow(identifier, stored, copy ? 0 : saved?.revision ?? 0)
    setSaved(result); setDraft(result.definition); setWorkflows(await listEtlWorkflows()); setNotice('流程已保存，可反复执行；保存不会启动下载。')
  })
  if (!catalog) return <section className="rounded-xl border border-slate-200 bg-white p-5">{error ? <p role="alert">{error}<button className={buttonClass} onClick={() => setRetry(v => v + 1)}>重新加载</button></p> : <p role="status">正在读取数据源与 ETL 流程…</p>}</section>
  return <form ref={form} noValidate onSubmit={e => e.preventDefault()} className="min-w-0 space-y-5">
    <nav className="flex flex-wrap gap-2" aria-label="下载工作方式">{(['auto', 'quick', 'workflow', 'runs'] as const).map(key => <button type="button" key={key} aria-pressed={view === key} className={view === key ? primaryClass : buttonClass} onClick={() => { setView(key); setValidation(null); setError('') }}>{({ auto: '自动增量（无需日期）', quick: '按数据源下载', workflow: 'ETL 任务编排', runs: '运行记录与恢复' })[key]}</button>)}</nav>
    {view === 'auto' ? <AutoIncrementalWorkspace catalog={catalog} canRun={!busy && statusReady && !executing && catalog.editing_enabled} onStarted={result => { setRuns(value => [result, ...value.filter(r => r.run_id !== result.run_id)]); setView('runs'); setNotice('自动增量已提交，运行记录保留了实际区间和基线快照。') }} /> : null}
    {validation?.auto_plan ? <AutoPlanSummary plan={validation.auto_plan} /> : null}
    {error ? <p role="alert" className="rounded-xl bg-rose-50 p-4 text-sm text-rose-800">{error}</p> : null}
    {notice ? <p role="status" className="rounded-xl bg-emerald-50 p-4 text-sm text-emerald-900">{notice}</p> : null}
    {!statusReady ? <p role="status" className="text-sm text-amber-900">尚未取得可靠任务状态，暂不允许启动。<button type="button" className={buttonClass} onClick={() => setRetry(v => v + 1)}>重新检查</button></p> : null}
    {!catalog.editing_enabled ? <p className="text-sm text-amber-900">当前环境只读，不能保存或执行流程。</p> : null}
    {view === 'quick' ? <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
      <h2 className="text-lg font-bold">选择数据源与下载内容</h2>
      <div className="grid gap-3 sm:grid-cols-2"><label className="text-sm font-semibold">下载数据源<select aria-label="下载数据源" className={inputClass} value={source?.config.id ?? ''} onChange={e => { setSourceId(e.target.value); setSelected([]); setQuery(''); setValidation(null); setSnapshot(false) }}>{catalog.sources.map(s => <option key={s.config.id} value={s.config.id}>{s.config.name}{s.config.enabled ? '' : '（已停用）'}</option>)}</select></label><label className="text-sm font-semibold">搜索可下载数据<input className={inputClass} value={query} onChange={e => setQuery(e.target.value)} placeholder="例如：行情、净值、接口名称" /></label></div>
      <p className="text-xs leading-6 text-slate-500">只显示 {source?.config.name} 已配置的接口。{source?.credential_required ? source?.credential_configured ? '凭据已保存。' : '尚未配置有效凭据。' : '此来源无需认证。'}<Link className="ml-1 text-indigo-700 underline" to={`/settings/source-center?source=${source?.config.id}`}>管理来源、凭据与接口</Link></p>
      <fieldset disabled={busy || !source?.config.enabled} className="space-y-3" aria-label="可下载数据">
        {[...catalog.targets.categories, { category_id: 'other', label: '其他接口' }].map(category => {
          const items = choices.filter(i => {
            const first = catalog.targets.tables.find(t => i.config.mappings.some(m => m.enabled && m.target_table === t.table_id))
            const text = [i.config.name, i.config.api_name, ...i.config.source_fields.map(f => `${f.name} ${f.description}`)].join(' ').toLowerCase()
            return (first?.category_id ?? 'other') === category.category_id && text.includes(query.trim().toLowerCase())
          })
          return items.length ? <section key={category.category_id} className="rounded-lg border border-slate-200 p-3"><h3 className="text-sm font-semibold">{category.label} · {items.length}</h3><div className="mt-2 grid gap-2 sm:grid-cols-2">{items.map(i => <label key={i.config.id} className={`flex min-w-0 items-start gap-2 rounded-lg p-3 text-sm ${selected.includes(i.config.id) ? 'bg-indigo-50' : 'bg-slate-50'}`}><input aria-label={i.config.name} className="mt-1" type="checkbox" checked={selected.includes(i.config.id)} disabled={!i.config.enabled || !i.validation?.ready} onChange={e => { setSelected(current => e.target.checked ? [...current, i.config.id] : current.filter(id => id !== i.config.id)); setValidation(null) }} /><span className="min-w-0"><strong>{i.config.name}</strong><code className="mt-1 block break-all text-xs text-slate-500">{i.config.api_name || i.config.id}</code>{!i.config.enabled || !i.validation?.ready ? <span className="text-xs text-amber-800">{!i.config.enabled ? '未启用' : '映射待完善'}</span> : null}</span></label>)}</div></section> : null
        })}
        {!choices.length ? <p className="text-sm text-slate-500">该来源还没有接口，请先配置支持的数据。</p> : null}
      </fieldset>
      <div className="flex flex-wrap items-center gap-3"><label className="text-sm font-semibold">下载模式<select aria-label="下载模式" className={inputClass} value={mode} onChange={e => { setMode(e.target.value as typeof mode); setValidation(null) }}><option value="incremental">增量补充</option><option value="full">全量重新请求所选范围</option></select></label><span className="text-xs text-slate-500">已选 {selectedRecords.length} 项；全量不表示删除整个仓库或自动请求全市场。</span></div>
      {selectedRecords.map(record => <details key={record.config.id} open className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-bold">{record.config.name} · 下载范围</summary><div className="mt-3"><EtlDownloadFields step={{ ...downloadStep(record, 'inherit', downloadParams[record.config.id]), allow_empty: emptyAllowed[record.config.id] ?? false }} catalog={catalog} chooseInterface={false} onChange={s => { setEmptyAllowed(p => ({ ...p, [record.config.id]: s.allow_empty })); setDownloadParams(p => ({ ...p, [record.config.id]: s.params })); setValidation(null) }} /></div></details>)}
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={history} onChange={e => setHistory(e.target.checked)} />取值时包含启动前的历史候选</label>
      <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={snapshot && canSnapshot} disabled={!canSnapshot} onChange={e => setSnapshot(e.target.checked)} />最后计算指标快照</label>
      {!canSnapshot ? <p className="text-xs text-slate-500">指标快照需要产品信息和基金净值。跨来源组合或不同计算位置，请使用 ETL 编排。</p> : null}
      {plan.steps.length ? <div className="rounded-lg bg-slate-50 p-3"><p className="text-sm font-semibold">执行清单</p><ol className="mt-2 space-y-1 text-xs text-slate-600">{plan.steps.map((s, i) => <li key={s.id}>{i + 1}. {stepLabels[s.kind]}：{s.name}</li>)}</ol></div> : null}
      <div className="flex flex-wrap gap-2"><button type="button" className={primaryClass} disabled={busy || !statusReady || executing || !catalog.editing_enabled || !source?.config.enabled || !selectedRecords.length} onClick={() => launch(plan)}>确认并开始下载</button><button type="button" className={buttonClass} disabled={!selectedRecords.length} onClick={() => { if (dirty && !window.confirm('替换当前尚未保存的流程草稿？')) return; setDraft(plan); setRunOptions({ mode, parameters: {} }); setSaved(null); setView('workflow'); setValidation(null) }}>转为 ETL 流程编辑</button></div>
    </section> : null}
    {view === 'workflow' ? <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-4 sm:p-5">
      <div className="flex flex-wrap items-end gap-3"><label className="min-w-0 flex-1 text-sm font-semibold">已保存流程<select aria-label="已保存流程" className={inputClass} value={saved?.id ?? ''} onChange={e => { if (dirty && !window.confirm('放弃当前未保存修改并切换流程？')) return; const item = workflows.find(w => w.id === e.target.value); setSaved(item ?? null); setDraft(item?.definition ?? emptyDefinition()); setRunOptions({ mode: 'incremental', parameters: {} }); setValidation(null) }}><option value="">新流程草稿</option>{workflows.map(w => <option key={w.id} value={w.id}>{w.definition.name} · v{w.revision}</option>)}</select></label><button type="button" className={buttonClass} onClick={() => { if (!dirty || window.confirm('放弃当前草稿并新建流程？')) { setDraft(emptyDefinition()); setSaved(null); setValidation(null) } }}>新建流程</button></div>
      <EtlRunOptionsEditor definition={draft} value={runOptions} onChange={value => { setRunOptions(value); setValidation(null) }} disabled={busy || executing || !catalog.editing_enabled} />
      <EtlTemplatePicker disabled={busy || !catalog.editing_enabled} onChoose={definition => {
        if (dirty && !window.confirm('替换当前未保存草稿？')) return
        setDraft(definition); setSaved(null); setRunOptions({ mode: 'incremental', parameters: {} }); setValidation(null)
      }} />
      <EtlWorkflowEditor key={saved?.id ?? 'draft'} definition={draft} catalog={catalog} onChange={changeDraft} readOnly={busy || !catalog.editing_enabled} canUndo={timeline.past.length > 0} canRedo={timeline.future.length > 0} onUndo={() => { dispatchDraft({ type: 'undo' }); setValidation(null) }} onRedo={() => { dispatchDraft({ type: 'redo' }); setValidation(null) }} />
      <div className="flex flex-wrap gap-2"><button type="button" className={buttonClass} disabled={busy || !draft.steps.length} onClick={() => void perform(async () => setValidation(await validateEtl(draft, runOptions)))}>{runOptions.mode === 'auto_incremental' ? '分析自动增量区间' : '校验流程'}</button><button type="button" className={buttonClass} disabled={busy || !catalog.editing_enabled || !draft.steps.length} onClick={() => save(false)}>保存流程</button>{saved ? <><button type="button" className={buttonClass} disabled={busy || !catalog.editing_enabled} onClick={() => save(true)}>另存为新流程</button><button type="button" className={buttonClass} disabled={busy || !catalog.editing_enabled} onClick={() => { if (window.confirm('删除此流程？历史运行结果会保留。')) void perform(async () => { await deleteEtlWorkflow(saved.id, saved.revision); setWorkflows(await listEtlWorkflows()); setSaved(null); setDraft(emptyDefinition()) }) }}>删除流程</button></> : null}<button type="button" className={primaryClass} disabled={busy || !statusReady || executing || !catalog.editing_enabled || !draft.steps.length} onClick={() => launch(draft)}>确认并运行流程</button></div>
    </section> : null}
    {validation ? <section role={validation.valid ? 'status' : 'alert'} className={`rounded-xl p-4 text-sm ${validation.valid ? 'bg-emerald-50 text-emerald-900' : 'bg-rose-50 text-rose-800'}`}><strong>{validation.valid ? '流程校验通过；不代表已下载或已发布。' : '流程需要调整'}</strong>{validation.errors.map((e, i) => <p key={i} className="mt-2">{e.message}</p>)}</section> : null}
    {view === 'runs' ? <EtlRunHistory runs={runs} focusedRun={focusedRun} onShowRun={setFocusedRun} connected={statusReady} busy={busy || !catalog.editing_enabled} onCancel={id => runAction(id, cancelEtl)} onResume={recover} onReuse={run => void perform(async () => {
      if (dirty && !window.confirm('替换当前未保存的流程草稿？')) return
      const response = await fetch(`/api/data-sources/etl/runs/${run.run_id}`)
      if (!response.ok) throw new Error('无法读取历史流程。')
      const detail = await response.json() as EtlRun
      if (!detail.definition) throw new Error('历史运行缺少流程定义。')
      setDraft(detail.template_definition ?? detail.definition); setRunOptions(detail.options ?? { mode: 'incremental', parameters: {} }); setSaved(null); setView('workflow'); setValidation(null)
    })} /> : null}
  </form>
}
