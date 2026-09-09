import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { listHistoricalRegimeDefinitions, type HistoricalRegimeDefinition } from '../services/historicalRegimes'
import {
  copyHistoricalRegimeDefinitionToV2,
  getRegimeGraphTemplates,
  listRegimeFormalRuns,
  listRegimeGraphDefinitions,
  type RegimeFormalRun,
  type RegimeGraphDefinition,
  type RegimeGraphTemplate,
} from '../services/regimeGraph'

const workbenchPath = '/settings/scenario-algorithms/workbench'

function readableError(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback
}

function definitionHref(definitionId: string, revision: number) {
  const query = new URLSearchParams({ definition: definitionId, revision: String(revision) })
  return `${workbenchPath}?${query.toString()}`
}

function templateHref(template: RegimeGraphTemplate) {
  const query = new URLSearchParams({ template: template.id })
  if (template.default_mode === 'retrospective') query.set('mode', 'retrospective')
  return `${workbenchPath}?${query.toString()}`
}

function formatTime(value?: string) {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function runUsages(run: RegimeFormalRun) {
  const publications = (run.publications || []).map((item) => item.usage)
  const bindings = (run.application_bindings || []).flatMap((binding) => {
    const value = binding.usage ?? binding.target ?? binding.application ?? binding.consumer
    return typeof value === 'string' ? [value] : []
  })
  return [...new Set([...publications, ...bindings])]
}

function usageLabel(usage: string) {
  return ({ research_display: '研究展示', product_research: '产品研究', formal_backtest: '组合回测', taa: '战术配置' } as Record<string, string>)[usage] || usage
}

function DefinitionCard({ definition, runs }: { definition: RegimeGraphDefinition; runs: RegimeFormalRun[] }) {
  const [revisionText, setRevisionText] = useState(String(definition.revision || 1))
  const parsedRevision = Number(revisionText)
  const revisionValid = Number.isInteger(parsedRevision) && parsedRevision >= 1
  const latestRun = runs[0]
  const usages = [...new Set(runs.flatMap(runUsages))]
  return <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
    <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2"><h3 className="text-base font-bold text-slate-950">{definition.name}</h3><span className="rounded-full bg-indigo-50 px-2 py-1 text-[10px] font-bold text-indigo-700">V2 · r{definition.revision || 1}</span></div>
        <p className="mt-1 text-xs leading-5 text-slate-600">{definition.description || '尚未填写研究说明。'}</p>
      </div>
      <div className="flex shrink-0 items-end gap-2">
        <label className="text-[10px] font-bold text-slate-600">精确 revision<input aria-label={`${definition.name} 精确 revision`} type="number" min={1} value={revisionText} onChange={(event) => setRevisionText(event.target.value)} onBlur={() => { if (!revisionValid) setRevisionText(String(definition.revision || 1)) }} className="mt-1 block min-h-9 w-20 rounded-lg border border-slate-300 px-2 text-xs font-normal" /></label>
        <Link aria-disabled={!revisionValid} to={revisionValid ? definitionHref(definition.id || '', parsedRevision) : '#'} className={`inline-flex min-h-9 items-center rounded-lg bg-slate-950 px-3 text-xs font-bold text-white ${revisionValid ? '' : 'pointer-events-none opacity-40'}`}>打开精确版本</Link>
      </div>
    </div>
    <dl className="mt-4 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
      <div className="rounded-xl bg-slate-50 p-2"><dt className="text-[10px] font-bold text-slate-500">计算节点</dt><dd className="mt-1 font-bold text-slate-900">{definition.graph.nodes.length}</dd></div>
      <div className="rounded-xl bg-slate-50 p-2"><dt className="text-[10px] font-bold text-slate-500">状态语义</dt><dd className="mt-1 font-bold text-slate-900">{definition.states.length}</dd></div>
      <div className="rounded-xl bg-slate-50 p-2"><dt className="text-[10px] font-bold text-slate-500">正式运行</dt><dd className="mt-1 font-bold text-slate-900">{runs.length}</dd></div>
      <div className="rounded-xl bg-slate-50 p-2"><dt className="text-[10px] font-bold text-slate-500">最近运行</dt><dd className="mt-1 truncate font-bold text-slate-900">{latestRun ? formatTime(latestRun.created_at) : '尚无'}</dd></div>
    </dl>
    <div className="mt-3 flex flex-wrap items-center gap-1.5 text-[10px]"><span className="font-bold text-slate-500">下游应用</span>{usages.length ? usages.map((usage) => <span key={usage} className="rounded-full bg-emerald-50 px-2 py-1 font-bold text-emerald-800">{usageLabel(usage)}</span>) : <span className="text-slate-400">尚无已发布绑定</span>}<span className="ml-auto text-slate-400">更新 {formatTime(definition.updated_at || definition.created_at)}</span></div>
  </article>
}

export default function HistoricalRegimeDirectory() {
  const navigate = useNavigate()
  const [definitions, setDefinitions] = useState<RegimeGraphDefinition[]>([])
  const [templates, setTemplates] = useState<RegimeGraphTemplate[]>([])
  const [runs, setRuns] = useState<RegimeFormalRun[]>([])
  const [legacyDefinitions, setLegacyDefinitions] = useState<HistoricalRegimeDefinition[]>([])
  const [loading, setLoading] = useState(true)
  const [copying, setCopying] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setLoading(true)
    void Promise.allSettled([
      listRegimeGraphDefinitions(controller.signal),
      getRegimeGraphTemplates(controller.signal),
      listRegimeFormalRuns(undefined, controller.signal),
      listHistoricalRegimeDefinitions(),
    ]).then((results) => {
      if (!active) return
      if (results[0].status === 'fulfilled') setDefinitions(results[0].value)
      if (results[1].status === 'fulfilled') setTemplates(results[1].value)
      if (results[2].status === 'fulfilled') setRuns(results[2].value)
      if (results[3].status === 'fulfilled') setLegacyDefinitions(results[3].value)
      const failures = results.filter((result): result is PromiseRejectedResult => result.status === 'rejected')
      setError(failures.length ? failures.map((result) => readableError(result.reason, '目录数据加载失败。')).join('；') : '')
    }).finally(() => { if (active) setLoading(false) })
    return () => { active = false; controller.abort() }
  }, [])

  const runsByDefinition = useMemo(() => {
    const grouped = new Map<string, RegimeFormalRun[]>()
    runs.forEach((run) => {
      if (!run.definition_id) return
      const current = grouped.get(run.definition_id) || []
      current.push(run)
      grouped.set(run.definition_id, current)
    })
    grouped.forEach((items) => items.sort((left, right) => String(right.created_at).localeCompare(String(left.created_at))))
    return grouped
  }, [runs])

  const copyLegacy = async (definition: HistoricalRegimeDefinition) => {
    if (!definition.id || !definition.revision) return
    const key = `${definition.id}:${definition.revision}`
    setCopying(key); setError(''); setNotice('')
    try {
      const result = await copyHistoricalRegimeDefinitionToV2(definition.id, definition.revision)
      if (!result.definition.id || !result.definition.revision) throw new Error('迁移接口未返回已保存的 V2 精确版本。')
      setNotice(`“${definition.name}”已复制为 V2 定义，正在打开 r${result.definition.revision}。`)
      navigate(definitionHref(result.definition.id, result.definition.revision))
    } catch (reason) {
      setError(readableError(reason, '旧版定义迁移失败。'))
    } finally {
      setCopying('')
    }
  }

  return <div className="space-y-5" data-testid="historical-regime-directory">
    <section className="overflow-hidden rounded-3xl bg-slate-950 p-5 text-white shadow-lg sm:p-7">
      <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
        <div className="max-w-3xl"><p className="text-[10px] font-bold uppercase tracking-[0.2em] text-indigo-300">Historical regime graph directory</p><h2 className="mt-2 text-2xl font-black sm:text-3xl">历史情景识别</h2><p className="mt-3 text-sm leading-6 text-slate-300">识别牛熊、波动或宏观状态，检查不同情景下的市场与产品表现。从模板开始，逐步调整为自己的研究规则。</p><Link to="/settings/research-data-lab" className="mt-3 inline-flex min-h-9 items-center text-xs font-semibold text-indigo-200 underline">研究数据实验室</Link></div>
        <div className="grid gap-2 sm:grid-cols-3 lg:w-[450px]"><a href="#regime-templates" className="inline-flex min-h-11 items-center justify-center rounded-xl bg-indigo-500 px-4 text-sm font-bold text-white">从模板开始</a><a href="#regime-saved" className="inline-flex min-h-11 items-center justify-center rounded-xl border border-white/20 px-4 text-sm font-bold text-white">继续已有研究</a><Link to={workbenchPath} className="inline-flex min-h-11 items-center justify-center rounded-xl border border-white/20 px-4 text-sm font-bold text-white">自由构建</Link></div>
      </div>
    </section>

    {error ? <p role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900">{error}</p> : null}
    {notice ? <p role="status" className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-3 text-sm text-indigo-950">{notice}</p> : null}
    {loading ? <p role="status" className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-sm font-semibold text-slate-600">正在读取 V2 定义、模板、运行摘要与旧版迁移目录…</p> : null}

    <section id="regime-templates" tabIndex={-1} aria-labelledby="templates-heading" className="scroll-mt-5 rounded-2xl border border-slate-200 bg-white p-4 shadow-sm focus:outline-indigo-500 sm:p-5">
      <div className="flex items-end justify-between gap-3"><div><h2 id="templates-heading" className="text-lg font-black text-slate-950">选择研究模板</h2><p className="mt-1 text-xs leading-5 text-slate-500">先选要识别的市场状态，再设置数据与参数、运行看结果。模板中的规则都可以查看和修改，也可以随时进入自由画板。</p></div><span className="shrink-0 text-xs font-bold text-slate-500">{templates.length} 个</span></div>
      {(['realtime', 'retrospective'] as const).map(mode => {
        const items = templates.filter(template => (template.default_mode ?? 'realtime') === mode)
        if (!items.length) return null
        const label = mode === 'realtime' ? '实时识别' : '事后识别'
        return <section key={mode} aria-label={label} className="mt-5">
          <h3 className="text-sm font-bold text-slate-900">{label}</h3>
          <p className="mt-1 text-xs text-slate-500">{mode === 'realtime' ? '使用当时可得数据；也可用于历史回放。' : '用于解释历史区间；需要之后的数据，不能用作当时的交易信号。'}</p>
          <div className="mt-3 grid gap-3 md:grid-cols-2 xl:grid-cols-3">{items.map(template => <article key={template.id} className="flex min-h-40 flex-col rounded-xl border border-slate-200 p-4"><div className="flex flex-wrap gap-1">{(template.tags || []).slice(0, 3).map((tag) => <span key={tag} className="rounded bg-slate-100 px-1.5 py-0.5 text-[9px] font-bold text-slate-600">{tag}</span>)}</div><h3 className="mt-2 text-sm font-bold text-slate-950">{template.name}</h3><p className="mt-1 flex-1 text-xs leading-5 text-slate-500">{template.description || '从这套识别规则开始，按研究需要调整。'}</p><Link to={templateHref(template)} aria-label={`使用模板：${template.name}`} className="mt-3 inline-flex min-h-9 items-center justify-center rounded-lg border border-indigo-300 px-3 text-xs font-bold text-indigo-700">使用此模板</Link></article>)}</div>
        </section>
      })}
      {!loading && !templates.length ? <p className="mt-4 rounded-xl bg-slate-50 p-4 text-xs text-slate-500">当前没有可用模板。可以<Link to={workbenchPath} className="font-bold text-indigo-700 underline">从空白画板自由构建</Link>，或继续已有研究。</p> : null}
    </section>

    <section id="regime-saved" tabIndex={-1} aria-labelledby="saved-v2-heading" className="scroll-mt-5 focus:outline-indigo-500">
      <div className="mb-3 flex items-end justify-between gap-3"><div><h2 id="saved-v2-heading" className="text-lg font-black text-slate-950">已有识别方案</h2><p className="mt-1 text-xs text-slate-500">打开保存的版本继续研究，已有结果和发布引用保留原版本。</p></div><span className="text-xs font-bold text-slate-500">{definitions.length} 个方案</span></div>
      <div className="grid gap-3 2xl:grid-cols-2">{definitions.map((definition) => definition.id ? <DefinitionCard key={definition.id} definition={definition} runs={runsByDefinition.get(definition.id) || []} /> : null)}</div>
      {!loading && !definitions.length ? <div className="rounded-2xl border border-dashed border-slate-300 bg-white p-8 text-center"><h3 className="text-sm font-bold text-slate-900">还没有识别方案</h3><p className="mt-2 text-xs text-slate-500">选择上方模板开始第一次研究，保存后可在这里继续。</p><a href="#regime-templates" className="mt-4 inline-flex min-h-10 items-center rounded-lg bg-slate-950 px-4 text-xs font-bold text-white">选择第一个模板</a></div> : null}
    </section>

    <details className="rounded-2xl border border-amber-200 bg-amber-50/50 p-4 shadow-sm sm:p-5">
      <summary className="cursor-pointer text-sm font-bold text-amber-950">V1 只读迁移区 · {legacyDefinitions.length} 个旧定义</summary>
      <p className="mt-2 text-xs leading-5 text-amber-900">旧版固定“单源 → 特征 → 模型”定义不再直接编辑。复制会创建新的 V2 定义并保留原始 V1 版本；无法无损转换时服务端会拒绝并说明原因。</p>
      <div className="mt-4 space-y-2">{legacyDefinitions.map((definition) => {
        const key = `${definition.id}:${definition.revision}`
        return <article key={key} className="flex flex-col gap-3 rounded-xl border border-amber-200 bg-white p-3 sm:flex-row sm:items-center sm:justify-between"><div className="min-w-0"><div className="flex flex-wrap items-center gap-2"><h3 className="text-sm font-bold text-slate-950">{definition.name}</h3><span className="rounded bg-amber-100 px-2 py-0.5 text-[10px] font-bold text-amber-900">V1 · r{definition.revision || 1}</span>{definition.status ? <span className="text-[10px] font-bold text-slate-500">{definition.status}</span> : null}</div><p className="mt-1 text-xs text-slate-500">{definition.target?.name || definition.target?.series_id || '未命名数据源'} · {definition.algorithm?.family || '未知模型'}</p></div><button type="button" disabled={!definition.id || !definition.revision || copying === key} onClick={() => void copyLegacy(definition)} className="min-h-9 shrink-0 rounded-lg bg-amber-900 px-3 text-xs font-bold text-white disabled:opacity-40">{copying === key ? '迁移中…' : '复制为 V2 图谱'}</button></article>
      })}{!loading && !legacyDefinitions.length ? <p className="rounded-xl bg-white p-4 text-xs text-slate-500">没有需要迁移的 V1 定义。</p> : null}</div>
    </details>
  </div>
}
