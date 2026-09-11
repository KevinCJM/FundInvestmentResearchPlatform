import { useState } from 'react'
import GlobalEventLibrary from './GlobalEventLibrary'
import type { ManualHistoricalEvent } from '../../services/regimeGraph'

const EVENT_COLORS = ['#7c3aed', '#dc2626', '#2563eb', '#d97706', '#0f766e', '#db2777', '#4f46e5', '#059669']

function eventsFrom(value: unknown): ManualHistoricalEvent[] {
  if (!Array.isArray(value)) return []
  return value.flatMap(item => item && typeof item === 'object' && !Array.isArray(item) ? [{
    id: String((item as Record<string, unknown>).id ?? ''),
    label: String((item as Record<string, unknown>).label ?? ''),
    start_date: String((item as Record<string, unknown>).start_date ?? ''),
    end_date: String((item as Record<string, unknown>).end_date ?? ''),
    color: String((item as Record<string, unknown>).color ?? '#7c3aed'),
    description: String((item as Record<string, unknown>).description ?? ''),
    ...((item as ManualHistoricalEvent).library_reference ? { library_reference: (item as ManualHistoricalEvent).library_reference } : {}),
  }] : [])
}

function newEvent(index: number): ManualHistoricalEvent {
  const uuid = typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID().replace(/-/g, '')
    : `${Date.now()}_${Math.random().toString(36).slice(2)}`
  return { id: `event_${uuid}`, label: `历史事件 ${index + 1}`, start_date: '', end_date: '', color: EVENT_COLORS[index % EVENT_COLORS.length], description: '' }
}

export default function RegimeManualEventEditor({ value, onChange }: { value: unknown; onChange: (events: ManualHistoricalEvent[]) => void }) {
  const [selecting, setSelecting] = useState(false)
  const [selectionError, setSelectionError] = useState('')
  const events = eventsFrom(value)
  const patch = (index: number, change: Partial<ManualHistoricalEvent>) => onChange(events.map((event, position) => position === index ? { ...event, ...change } : event))
  if (selecting) return <GlobalEventLibrary onCancel={() => setSelecting(false)} onSelect={incoming => {
    const merged = Array.from(new Map([...events, ...incoming].map(e => [e.id, e])).values())
    if (merged.length > 100) { setSelectionError('加入后超过100个事件，请减少选择。'); setSelecting(false); return }
    onChange(merged); setSelectionError(''); setSelecting(false)
  }} />
  return <section aria-label="人工历史事件区间" className="space-y-3 rounded-xl border border-violet-200 bg-violet-50/40 p-3">
    {selectionError && <p role="alert" className="text-xs text-rose-700">{selectionError}</p>}
    <button type="button" onClick={() => setSelecting(true)} className="min-h-10 rounded-lg border border-violet-300 bg-white px-3 text-sm font-semibold text-violet-700">从事件库选择</button>
    <div className="flex flex-wrap items-start justify-between gap-2">
      <div><h4 className="text-sm font-bold text-slate-900">人工历史事件区间</h4><p className="mt-1 text-xs leading-5 text-slate-600">每个事件独立定义开始和结束日期，<strong>允许重叠</strong>。事件不是互斥市场状态，也不是实时交易信号。</p></div>
      <button type="button" disabled={events.length >= 100} onClick={() => onChange([...events, newEvent(events.length)])} className="min-h-10 rounded-lg bg-violet-600 px-3 text-xs font-bold text-white disabled:opacity-40">添加事件</button>
    </div>
    {!events.length ? <div className="rounded-lg border border-dashed border-violet-200 bg-white p-4 text-center text-xs text-slate-500">尚未定义事件。点击“添加事件”，填写事件名称和日期区间。</div> : null}
    <div className="space-y-3">{events.map((event, index) => {
      const badRange = Boolean(event.start_date && event.end_date && event.start_date > event.end_date)
      return <article key={event.id || index} className="rounded-xl border border-slate-200 bg-white p-3" aria-label={`历史事件 ${index + 1}`}>
        <div className="flex items-center justify-between gap-2"><p className="text-xs font-bold text-slate-700">事件 {index + 1}</p><button type="button" onClick={() => onChange(events.filter((_, position) => position !== index))} className="min-h-9 px-2 text-xs font-bold text-rose-600">删除</button></div>
        {event.library_reference && <div className="my-2 rounded-lg bg-violet-50 p-2 text-xs text-violet-800">事件库 v{event.library_reference.revision} · 已锁定窗口；库更新不改变本项。<button type="button" onClick={() => { const { library_reference: _ref, ...local } = event; onChange(events.map((e, i) => i === index ? { ...local, id: newEvent(index).id } : e)) }} className="block min-h-9 underline">转为人工副本后编辑</button></div>}
        <div className="mt-2 grid gap-3 sm:grid-cols-2">
          <label className="text-xs font-semibold text-slate-600">事件名称<input aria-label={`事件${index + 1}名称`} readOnly={Boolean(event.library_reference)} maxLength={100} value={event.label} onChange={e => patch(index, { label: e.target.value })} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" /></label>
          <label className="text-xs font-semibold text-slate-600">颜色<span className="mt-1 flex min-h-10 items-center gap-2 rounded-lg border border-slate-300 px-2"><input aria-label={`事件${index + 1}颜色`} type="color" disabled={Boolean(event.library_reference)} value={/^#[0-9a-f]{6}$/i.test(event.color) ? event.color : '#7c3aed'} onChange={e => patch(index, { color: e.target.value })} className="h-7 w-10 cursor-pointer border-0 bg-transparent p-0" /><span className="font-mono font-normal text-slate-500">{event.color}</span></span></label>
          <label className="text-xs font-semibold text-slate-600">开始日期<input aria-label={`事件${index + 1}开始日期`} type="date" readOnly={Boolean(event.library_reference)} value={event.start_date} onChange={e => patch(index, { start_date: e.target.value })} className={`mt-1 min-h-10 w-full rounded-lg border px-2 font-normal ${badRange ? 'border-rose-400' : 'border-slate-300'}`} /></label>
          <label className="text-xs font-semibold text-slate-600">结束日期<input aria-label={`事件${index + 1}结束日期`} type="date" readOnly={Boolean(event.library_reference)} value={event.end_date} onChange={e => patch(index, { end_date: e.target.value })} className={`mt-1 min-h-10 w-full rounded-lg border px-2 font-normal ${badRange ? 'border-rose-400' : 'border-slate-300'}`} /></label>
        </div>
        {badRange ? <p role="alert" className="mt-2 text-xs font-semibold text-rose-700">开始日期不能晚于结束日期。</p> : null}
        <label className="mt-3 block text-xs font-semibold text-slate-600">说明（可选）<textarea aria-label={`事件${index + 1}说明`} readOnly={Boolean(event.library_reference)} maxLength={500} rows={2} value={event.description || ''} onChange={e => patch(index, { description: e.target.value })} className="mt-1 w-full rounded-lg border border-slate-300 px-2 py-2 font-normal" /></label>
      </article>
    })}</div>
    <p className="text-[11px] leading-5 text-slate-500">日期采用闭区间；边界可以不是交易日，运行时按观察序列中实际落入区间的观测点统计。多个事件重叠时不会互相覆盖或自动合并。</p>
  </section>
}
