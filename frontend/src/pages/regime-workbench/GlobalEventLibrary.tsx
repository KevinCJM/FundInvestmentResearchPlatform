import { useEffect, useRef, useState } from 'react'
import { createRegimeGraphDefinition, instantiateRegimeTemplate, listRegimeGraphDefinitions, type ManualHistoricalEvent, type RegimeGraphDefinition } from '../../services/regimeGraph'
import { eventCategories, listLibraryEvents, listEventPacks, getEventPack, resolveLibraryEvents, importLibraryDefinition, getLibraryEventHistory, type EventSelection, type LibraryEvent, type EventPack, type EventPage } from '../../services/eventLibrary'
import EventLibraryEditor from './EventLibraryEditor'

const control = 'min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-2 text-sm'
const identity = (e: EventSelection) => `${e.event_id}:${e.revision}:${e.window_id}`
const message = (error: unknown) => error instanceof Error ? error.message : '操作未完成，请重试。'
export default function GlobalEventLibrary({ onSelect, onCancel }: { onSelect?: (events: ManualHistoricalEvent[]) => void; onCancel?: () => void }) {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('')
  const [region, setRegion] = useState('')
  const [verification, setVerification] = useState('')
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [archived, setArchived] = useState(false)
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<EventPage>({ items: [], total: 0, offset: 0, limit: 20 })
  const [packs, setPacks] = useState<EventPack[]>([])
  const [selected, setSelected] = useState<EventSelection[]>([])
  const [selectionLabels, setSelectionLabels] = useState<Record<string, string>>({})
  const [detail, setDetail] = useState<LibraryEvent | null>(null)
  const [history, setHistory] = useState<LibraryEvent[]>([])
  const [editing, setEditing] = useState<LibraryEvent | 'new' | null>(null)
  const [definitions, setDefinitions] = useState<RegimeGraphDefinition[]>([])
  const [importId, setImportId] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [createdId, setCreatedId] = useState('')
  const [refresh, setRefresh] = useState(0)
  const action = useRef<AbortController | null>(null)
  const detailGeneration = useRef(0)
  useEffect(() => () => { action.current?.abort(); detailGeneration.current++ }, [])
  useEffect(() => { setOffset(0) }, [query, category, region, verification, start, end, archived])
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    const timer = window.setTimeout(() => {
      void listLibraryEvents({ query, category, region, verification, start, end, archived, offset, limit: 20 }, controller.signal)
        .then(next => { if (!controller.signal.aborted) { setPage(next); setError('') } })
        .catch(reason => { if (!controller.signal.aborted) setError(message(reason)) })
        .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    }, 180)
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [query, category, region, verification, start, end, archived, offset, refresh])
  useEffect(() => {
    const controller = new AbortController()
    void listEventPacks(controller.signal).then(data => { if (!controller.signal.aborted) setPacks(data.items) }).catch(reason => { if (!controller.signal.aborted) setError(message(reason)) })
    return () => controller.abort()
  }, [refresh])
  const add = (incoming: EventSelection[], labels: Record<string, string> = {}) => {
    setSelectionLabels(current => ({ ...current, ...labels }))
    setSelected(current => {
      const merged = Array.from(new Map([...current, ...incoming].map(e => [identity(e), e])).values())
      if (merged.length > 100) { setError('一次最多选择100个事件窗口，请减少选择。'); return current }
      return merged
    })
  }
  const toggle = (event: LibraryEvent, windowId: string) => {
    const selection = { event_id: event.id, revision: event.revision, window_id: windowId }
    const key = identity(selection)
    if (selected.some(e => identity(e) === key)) setSelected(items => items.filter(e => identity(e) !== key))
    else add([selection], { [key]: event.name + ' · ' + event.windows.find(w => w.id === windowId)!.label })
  }
  const chosen = (event: LibraryEvent, windowId: string) => selected.some(e => e.event_id === event.id && e.revision === event.revision && e.window_id === windowId)
  const work = async (fn: (signal: AbortSignal) => Promise<void>) => {
    action.current?.abort(); const controller = new AbortController(); action.current = controller
    setBusy(true); setError(''); setNotice('')
    try { await fn(controller.signal) } catch (reason) { if (!controller.signal.aborted) setError(message(reason)) }
    finally { if (!controller.signal.aborted) setBusy(false) }
  }
  const show = (event: LibraryEvent) => {
    const generation = ++detailGeneration.current
    setDetail(event); setHistory([]); setEditing(null)
    void getLibraryEventHistory(event.id).then(data => { if (generation === detailGeneration.current) setHistory(data.items) }).catch(reason => { if (generation === detailGeneration.current) setError(message(reason)) })
  }
  const finish = () => void work(async signal => {
    const response = await resolveLibraryEvents(selected, signal)
    if (signal.aborted) return
    if (onSelect) { onSelect(response.events); return }
    const definition = await instantiateRegimeTemplate('manual-historical-events-v1', signal)
    definition.name = '我的历史事件研究'
    const node = definition.graph.nodes.find(n => n.type === 'annotation.manual_events')
    if (!node) throw new Error('人工历史事件模板不可用。')
    node.parameters = { ...node.parameters, events: response.events }
    const saved = await createRegimeGraphDefinition(definition, signal)
    if (!signal.aborted) { setCreatedId(saved.id || ''); setNotice('已创建事后情景。观察序列默认沪深300，可进入算法定义更换。') }
  })
  return <section aria-label={onSelect ? '选择库中历史事件' : '全球历史事件库'} className="min-w-0 space-y-4 rounded-2xl border border-slate-200 bg-white p-3 sm:p-5">
    <header className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="text-lg font-semibold">全球历史事件库</h2><p className="mt-1 text-xs leading-5 text-slate-500">先选事件，再选研究窗口。加入情景时锁定版本；库更新不会改变旧结果。</p></div><div className="flex gap-2"><button type="button" disabled={busy} onClick={() => setEditing('new')} className="min-h-10 rounded-lg border border-violet-200 px-3 text-sm text-violet-700">新建事件</button>{onCancel && <button type="button" onClick={() => { action.current?.abort(); onCancel() }} className="min-h-10 rounded-lg border px-3 text-sm">取消选择</button>}</div></header>
    {error && <div role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}<button onClick={() => setRefresh(v => v + 1)} className="ml-3 min-h-9 underline">重新读取</button></div>}
    {notice && <p role="status" className="rounded-lg bg-emerald-50 p-3 text-sm text-emerald-800">{notice}</p>}
    {createdId && <a href={`/settings/scenario-algorithms/workbench?definition=${encodeURIComponent(createdId)}&mode=retrospective`} className="inline-flex min-h-10 items-center text-sm font-semibold text-violet-700 underline">打开新建的事后情景</a>}
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><label className="text-xs font-semibold sm:col-span-2">搜索事件<input aria-label="事件库搜索" type="search" value={query} onChange={e => setQuery(e.target.value)} placeholder="名称、英文名或说明" className={'mt-1 ' + control} /></label><label className="text-xs font-semibold">类别<select aria-label="事件库类别" value={category} onChange={e => setCategory(e.target.value)} className={'mt-1 ' + control}><option value="">全部类别</option>{Object.entries(eventCategories).map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label><label className="text-xs font-semibold">核验状态<select aria-label="事件核验状态" value={verification} onChange={e => setVerification(e.target.value)} className={'mt-1 ' + control}><option value="">全部状态</option><option value="verified">已人工核验</option><option value="unreviewed">待核验</option></select></label></div>
    <details className="rounded-lg border p-3"><summary className="cursor-pointer text-xs font-semibold">地区、时间与归档筛选</summary><div className="mt-3 grid gap-3 sm:grid-cols-3"><label className="text-xs">地区<input aria-label="事件库地区" value={region} onChange={e => setRegion(e.target.value)} className={control} /></label><label className="text-xs">研究窗口开始<input type="date" value={start} onChange={e => setStart(e.target.value)} className={control} /></label><label className="text-xs">研究窗口结束<input type="date" value={end} onChange={e => setEnd(e.target.value)} className={control} /></label></div><label className="mt-2 flex min-h-9 items-center gap-2 text-xs"><input type="checkbox" checked={archived} onChange={e => setArchived(e.target.checked)} />包含归档事件（只能查阅，不可新加入）</label></details>
    <div className="flex flex-wrap gap-2" aria-label="事件包">{packs.filter(p => p.count > 0).map(pack => <button type="button" key={pack.id} disabled={busy} onClick={() => void work(async signal => { const data = await getEventPack(pack.id, signal); if (!signal.aborted) add(data.selections, data.labels) })} className="min-h-9 rounded-full border border-slate-200 bg-slate-50 px-3 text-xs">{pack.name} · {pack.count}</button>)}</div>
    <div className="grid min-w-0 gap-4 lg:grid-cols-2">
      <div className="min-w-0"><div className="mb-2 text-xs text-slate-500">{loading ? '正在读取…' : `共 ${page.total} 个事件`} · 列表按主要研究窗口倒序</div>
        {loading ? <p role="status" className="p-5 text-sm">正在读取事件库…</p> : !page.items.length ? <p className="rounded-lg border border-dashed p-5 text-sm text-slate-500">没有匹配事件。可新建事件，或从已保存的人工情景导入。</p> : <div className="max-h-[65vh] space-y-2 overflow-y-auto" aria-label="事件列表">{page.items.map(event => <article key={event.id} className={'rounded-xl border p-3 ' + (detail?.id === event.id ? 'border-violet-400 bg-violet-50/40' : 'border-slate-200')}><button type="button" onClick={() => show(event)} className="min-h-10 w-full break-words text-left text-sm font-semibold">{event.name} <span className="text-xs font-normal text-slate-500">v{event.revision}</span></button><p className="text-xs text-slate-500">{event.windows[0].start_date} — {event.windows[0].end_date}</p><div className="mt-2 flex flex-wrap gap-1 text-[11px]"><span className={'rounded-full px-2 py-1 ' + (event.verification === 'verified' ? 'bg-emerald-50 text-emerald-800' : 'bg-amber-50 text-amber-800')}>{event.verification === 'verified' ? '已核验' : '待核验'}</span>{event.archived && <span className="px-2 py-1">已归档</span>}{event.categories.map(c => <span key={c} className="rounded-full bg-slate-100 px-2 py-1">{eventCategories[c] || c}</span>)}</div><button type="button" disabled={event.archived || busy} aria-pressed={chosen(event, event.windows[0].id)} onClick={() => toggle(event, event.windows[0].id)} className="mt-2 min-h-9 text-xs font-semibold text-violet-700 disabled:opacity-40">{chosen(event, event.windows[0].id) ? '已选择 · 移除主要窗口' : '选择主要窗口'}</button></article>)}</div>}
        <div className="mt-3 flex items-center justify-between gap-2 text-xs"><button type="button" disabled={!offset || loading} onClick={() => setOffset(n => Math.max(0, n - 20))} className="min-h-10 rounded-lg border px-3 disabled:opacity-40">上一页</button><span>第 {Math.floor(offset / 20) + 1} 页</span><button type="button" disabled={offset + 20 >= page.total || loading} onClick={() => setOffset(n => n + 20)} className="min-h-10 rounded-lg border px-3 disabled:opacity-40">下一页</button></div>
      </div>
      <div className="min-w-0">{editing ? <EventLibraryEditor key={editing === 'new' ? 'new' : editing.id + ':' + editing.revision} current={editing === 'new' ? undefined : editing} onCancel={() => setEditing(null)} onSaved={event => { setEditing(null); show(event); setRefresh(n => n + 1); setNotice('事件已保存为新修订，既有情景不变。') }} /> : detail ? <article aria-label="事件详情" className="space-y-4 rounded-xl border bg-slate-50 p-4">
        <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="break-words font-semibold">{detail.name}</h3><button type="button" disabled={busy} onClick={() => setEditing(detail)} className="min-h-10 text-xs font-semibold text-violet-700">编辑这个版本</button></div>
        <label className="block text-xs">历史修订<select aria-label="事件历史修订" value={detail.revision} onChange={e => { const version = history.find(v => v.revision === Number(e.target.value)); if (version) setDetail(version) }} className={'mt-1 ' + control}>{(history.length ? history : [detail]).map(v => <option key={v.revision} value={v.revision}>v{v.revision} · {v.updated_at.slice(0, 10)}</option>)}</select></label>
        <p className="whitespace-pre-wrap break-words text-xs leading-6">{detail.description || '暂无背景说明。'}</p><dl className="grid gap-3 text-xs sm:grid-cols-2"><div><dt className="text-slate-500">事实日期</dt><dd>{detail.fact_start || '尚未核实'} — {detail.fact_end || '结束未知'}</dd></div><div><dt className="text-slate-500">信息公开时间</dt><dd className="break-words">{detail.known_at || '未知（不能据此构造实时信号）'}</dd></div><div><dt className="text-slate-500">地区</dt><dd>{detail.regions.join('、') || '未分类'}</dd></div><div><dt className="text-slate-500">来源状态</dt><dd>{detail.verification === 'verified' ? '已人工核验' : '待核验，不是官方标准区间'}</dd></div></dl>
        <section><h4 className="text-sm font-semibold">研究窗口</h4><div className="mt-2 space-y-2">{detail.windows.map(w => <div key={w.id} className="rounded-lg bg-white p-3"><p className="text-xs font-semibold">{w.label}</p><p className="mt-1 text-xs">{w.start_date} — {w.end_date}</p><p className="mt-2 break-words text-xs leading-5 text-slate-500">{w.rationale}</p><button type="button" disabled={busy || detail.archived} aria-pressed={chosen(detail, w.id)} onClick={() => toggle(detail, w.id)} className="mt-2 min-h-9 text-xs font-semibold text-violet-700">{chosen(detail, w.id) ? '已选择 · 移除此窗口' : '选择此窗口'}</button></div>)}</div></section>
        <section><h4 className="text-sm font-semibold">参考来源</h4>{detail.sources.length ? detail.sources.map((s, i) => <a key={i} href={/^https?:\/\//.test(s.url) ? s.url : undefined} target="_blank" rel="noopener noreferrer" className="mt-2 block break-words text-xs text-violet-700 underline">{s.title}</a>) : <p className="mt-2 text-xs text-amber-800">尚无核实的来源链接。原人工说明仅作为待核验线索。</p>}</section>
      </article> : <div className="rounded-xl border border-dashed p-6 text-sm text-slate-500">点击事件名称查看事实、来源与可选研究窗口。</div>}</div>
    </div>
    {selected.length > 0 && <details className="rounded-xl border p-3"><summary className="cursor-pointer text-xs font-semibold">查看已选窗口（可逐项移除）</summary><div className="mt-2 max-h-48 space-y-2 overflow-auto">{selected.map(item => <div key={identity(item)} className="flex items-start justify-between gap-2 text-xs"><span className="break-words">{selectionLabels[identity(item)] || item.event_id} · v{item.revision}</span><button type="button" disabled={busy} onClick={() => setSelected(items => items.filter(e => identity(e) !== identity(item)))} className="min-h-9 shrink-0 text-rose-700">移除</button></div>)}</div></details>}
    <div className="sticky bottom-0 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-violet-200 bg-white p-3 shadow-sm" role="region" aria-label="已选事件"><div className="text-sm">已选 <strong>{selected.length}</strong> 个窗口<span className="ml-2 text-xs text-slate-500">每个事件独立保留，可重叠</span></div><div className="flex gap-2"><button type="button" disabled={busy || !selected.length} onClick={() => setSelected([])} className="min-h-10 rounded-lg border px-3 text-xs disabled:opacity-40">清空选择</button><button type="button" disabled={busy || !selected.length} onClick={finish} className="min-h-10 rounded-lg bg-violet-600 px-3 text-sm font-semibold text-white disabled:opacity-40">{busy ? '处理中…' : onSelect ? '加入当前情景' : '创建事后情景'}</button></div></div>
    {!onSelect && <details className="rounded-lg border p-3"><summary className="cursor-pointer text-xs font-semibold" onClick={() => { if (!definitions.length) void listRegimeGraphDefinitions().then(items => setDefinitions(items.filter(d => d.graph.nodes.some(n => n.type === 'annotation.manual_events')))).catch(reason => setError(message(reason))) }}>导入已有人工事件情景</summary><p className="my-3 text-xs leading-5 text-slate-500">复制事件为待核验库条目；不更改原情景，重复导入不会覆盖已维护的事件。</p><div className="flex flex-wrap gap-2"><select aria-label="待导入事件情景" value={importId} onChange={e => setImportId(e.target.value)} className={control + ' sm:flex-1'}><option value="">选择已保存情景</option>{definitions.map(d => <option key={d.id} value={d.id}>{d.name} · v{d.revision}</option>)}</select><button type="button" disabled={!importId || busy} onClick={() => void work(async signal => { const d = definitions.find(d => d.id === importId)!; const result = await importLibraryDefinition(importId, d.revision!); if (!signal.aborted) { setNotice(`导入 ${result.imported} 项，跳过已有 ${result.skipped} 项；待人工核验。`); setRefresh(n => n + 1) } })} className="min-h-10 rounded-lg border px-3 text-xs disabled:opacity-40">导入事件</button></div></details>}
  </section>
}
