import { useState } from 'react'
import { eventCategories, eventDraft, saveLibraryEvent, type LibraryEvent, type EventDraft } from '../../services/eventLibrary'

const input = 'mt-1 min-h-10 w-full min-w-0 rounded-xl border border-slate-300 bg-white px-2 py-2 text-sm font-normal'
export default function EventLibraryEditor({ current, onSaved, onCancel }: {
  current?: LibraryEvent; onSaved: (event: LibraryEvent) => void; onCancel: () => void
}) {
  const [draft, setDraft] = useState(() => eventDraft(current))
  const [regionsText, setRegionsText] = useState(draft.regions.join(', '))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const patch = (change: Partial<EventDraft>) => setDraft(value => ({ ...value, ...change }))
  const save = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setError('')
    try { onSaved(await saveLibraryEvent({ ...draft, regions: Array.from(new Set(regionsText.split(/[,，]/).map(s => s.trim()).filter(Boolean))) }, current)) } catch (reason) { setError(reason instanceof Error ? reason.message : '保存失败。') } finally { setBusy(false) }
  }
  return <form onSubmit={event => void save(event)} aria-label="历史事件编辑" className="space-y-4 rounded-xl border border-slate-200 bg-white p-4">
    <h3 className="font-semibold">{current ? `修改事件 · 当前v${current.revision}` : '新建历史事件'}</h3>
    <p className="text-xs leading-5 text-slate-600">事件事实与市场研究窗口分开记录。保存会生成新修订，不改变已有情景。</p>
    {error && <p role="alert" className="rounded-lg bg-rose-50 p-3 text-sm text-rose-800">{error}</p>}
    <fieldset disabled={busy} className="space-y-4 disabled:opacity-60">
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="text-xs font-semibold">事件名称<input aria-label="库事件名称" required maxLength={100} value={draft.name} onChange={e => patch({ name: e.target.value })} className={input} /></label>
        <label className="text-xs font-semibold">英文名称（可选）<input value={draft.name_en} maxLength={150} onChange={e => patch({ name_en: e.target.value })} className={input} /></label>
      </div>
      <fieldset><legend className="mb-2 text-xs font-semibold">事件类别（可多选）</legend><div className="flex flex-wrap gap-2">{Object.entries(eventCategories).map(([id, label]) => <label key={id} className="inline-flex min-h-9 items-center gap-2 rounded-lg border px-2 text-xs"><input type="checkbox" checked={draft.categories.includes(id)} onChange={e => patch({ categories: e.target.checked ? [...draft.categories, id] : draft.categories.filter(v => v !== id) })} />{label}</label>)}</div></fieldset>
      <label className="block text-xs font-semibold">地区（逗号分隔）<input value={regionsText} onChange={e => setRegionsText(e.target.value)} className={input} /></label>
      <label className="block text-xs font-semibold">背景说明<textarea rows={3} maxLength={2000} value={draft.description} onChange={e => patch({ description: e.target.value })} className={input} /></label>
      <details className="rounded-lg border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-semibold">事实日期与公开时间（不知道可留空）</summary>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <label className="text-xs">事实开始日期<input type="date" value={draft.fact_start || ''} onChange={e => patch({ fact_start: e.target.value || null })} className={input} /></label>
          <label className="text-xs">事实结束日期<input type="date" disabled={draft.status === 'ongoing'} value={draft.fact_end || ''} onChange={e => patch({ fact_end: e.target.value || null })} className={input} /></label>
          <label className="text-xs">事件状态<select value={draft.status} onChange={e => patch({ status: e.target.value as EventDraft['status'], ...(e.target.value === 'ongoing' ? { fact_end: null } : {}) })} className={input}><option value="unknown">尚未确认</option><option value="ongoing">进行中</option><option value="closed">已结束</option></select></label>
          <label className="text-xs">事实日期精度<select value={draft.date_precision} onChange={e => patch({ date_precision: e.target.value as EventDraft['date_precision'] })} className={input}><option value="unknown">未知</option><option value="day">日</option><option value="month">月</option><option value="year">年</option></select></label>
          <label className="text-xs sm:col-span-2">信息可得时间（含时区，可选）<input placeholder="2025-06-13T08:00:00+08:00" value={draft.known_at || ''} onChange={e => patch({ known_at: e.target.value || null })} className={input} /></label>
        </div>
      </details>
      <section className="space-y-3" aria-label="研究窗口设置"><div className="flex items-center justify-between gap-2"><h4 className="text-sm font-semibold">市场研究窗口</h4><button type="button" disabled={draft.windows.length >= 20} onClick={() => patch({ windows: [...draft.windows, { id: 'window_' + Date.now().toString(36), label: '', start_date: '', end_date: '', rationale: '' }] })} className="min-h-10 text-xs font-semibold text-accent-700">添加研究窗口</button></div>
        {draft.windows.map((window, i) => <div key={window.id} className="space-y-2 rounded-lg border bg-slate-50 p-3"><label className="block text-xs">窗口名称<input required aria-label={`窗口${i + 1}名称`} maxLength={100} value={window.label} onChange={e => patch({ windows: draft.windows.map((w, j) => j === i ? { ...w, label: e.target.value } : w) })} className={input} /></label>
          <div className="grid gap-2 sm:grid-cols-2">{(['start_date', 'end_date'] as const).map(key => <label key={key} className="text-xs">{key === 'start_date' ? '开始日期' : '结束/研究截至日期'}<input required type="date" aria-label={`窗口${i + 1}${key === 'start_date' ? '开始' : '结束'}`} value={window[key]} onChange={e => patch({ windows: draft.windows.map((w, j) => j === i ? { ...w, [key]: e.target.value } : w) })} className={input} /></label>)}</div>
          <label className="block text-xs">为什么选择这段区间？<textarea required aria-label={`窗口${i + 1}理由`} maxLength={1000} value={window.rationale} onChange={e => patch({ windows: draft.windows.map((w, j) => j === i ? { ...w, rationale: e.target.value } : w) })} className={input} /></label>
          {draft.windows.length > 1 && <button type="button" onClick={() => patch({ windows: draft.windows.filter((_, j) => j !== i) })} className="min-h-9 text-xs text-rose-700">移除窗口</button>}
        </div>)}
      </section>
      <section className="space-y-2"><div className="flex items-center justify-between"><h4 className="text-sm font-semibold">参考来源</h4><button type="button" disabled={draft.sources.length >= 20} onClick={() => patch({ sources: [...draft.sources, { title: '', url: '', published_at: null }] })} className="min-h-10 text-xs text-accent-700">添加来源</button></div>
        {draft.sources.map((source, i) => <div key={i} className="space-y-2 rounded-lg border p-3"><label className="block text-xs">机构/文献标题<input required value={source.title} maxLength={200} onChange={e => patch({ sources: draft.sources.map((s, j) => j === i ? { ...s, title: e.target.value } : s) })} className={input} /></label><label className="block text-xs">来源链接<input type="url" required placeholder="https://" maxLength={2000} value={source.url} onChange={e => patch({ sources: draft.sources.map((s, j) => j === i ? { ...s, url: e.target.value } : s) })} className={input} /></label><button type="button" onClick={() => patch({ sources: draft.sources.filter((_, j) => j !== i) })} className="min-h-9 text-xs text-rose-700">移除来源</button></div>)}
        <label className="flex min-h-10 items-center gap-2 text-xs"><input type="checkbox" checked={draft.verification === 'verified'} onChange={e => patch({ verification: e.target.checked ? 'verified' : 'unreviewed' })} />已人工核对来源和区间理由（未填来源不能标记）</label>
      </section>
      <div className="flex flex-wrap items-center gap-3"><label className="inline-flex min-h-10 items-center gap-2 text-xs">颜色<input aria-label="库事件颜色" type="color" value={draft.color} onChange={e => patch({ color: e.target.value })} /></label>{current && <label className="inline-flex min-h-10 items-center gap-2 text-xs"><input type="checkbox" checked={draft.archived} onChange={e => patch({ archived: e.target.checked })} />归档（保留旧引用）</label>}</div>
    </fieldset>
    <div className="flex gap-3"><button type="submit" disabled={busy} className="min-h-11 rounded-lg bg-accent-600 px-4 text-sm font-semibold text-white disabled:opacity-50">{busy ? '保存中…' : '保存事件'}</button><button type="button" disabled={busy} onClick={onCancel} className="min-h-11 rounded-lg border px-4 text-sm">取消编辑</button></div>
  </form>
}
