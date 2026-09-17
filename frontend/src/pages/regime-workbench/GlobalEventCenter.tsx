import { useEffect, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import HistoricalRegimeWorkbench from '../HistoricalRegimeWorkbench'
import GlobalEventLibrary from './GlobalEventLibrary'

export default function GlobalEventCenter() {
  const [params, setParams] = useSearchParams()
  const manual = params.get('event_view') === 'manual'
  const [visited, setVisited] = useState({ library: !manual, manual })
  useEffect(() => {
    const key = manual ? 'manual' : 'library'
    setVisited(previous => previous[key] ? previous : { ...previous, [key]: true })
  }, [manual])
  const choose = (next: boolean) => {
    setVisited(previous => ({ ...previous, [next ? 'manual' : 'library']: true }))
    setParams(previous => {
      const query = new URLSearchParams(previous)
      query.set('center', 'events'); query.set('event_view', next ? 'manual' : 'library')
      return query
    })
  }
  return <div className="min-w-0 space-y-4">
    <nav aria-label="全球历史事件库工作区" className="flex flex-wrap gap-2">
      {([[false, '事件库'], [true, '人工历史事件']] as const).map(([value, label]) => <button key={label} type="button" aria-pressed={manual === value} onClick={() => choose(value)} className={`min-h-10 rounded-lg px-4 text-sm font-semibold ${manual === value ? 'bg-accent-600 text-white' : 'border border-slate-200 bg-white text-slate-700'}`}>{label}</button>)}
    </nav>
    <section hidden={manual} aria-label="事件库管理">{(visited.library || !manual) && <GlobalEventLibrary />}</section>
    <section hidden={!manual} aria-label="人工历史事件工作区">{(visited.manual || manual) && <HistoricalRegimeWorkbench workspace="events" />}</section>
  </div>
}
