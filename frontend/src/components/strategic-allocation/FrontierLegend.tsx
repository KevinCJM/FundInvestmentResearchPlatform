import { Button } from '../ui'
import { useI18n } from '../../i18n/runtime'

export interface FrontierLegendItem {
  id: string
  name: string
  symbolClass: string
  symbolColor?: string
  detail?: string
}

export default function FrontierLegend({ items, selected, onChange }: {
  items: FrontierLegendItem[]
  selected: Record<string, boolean>
  onChange: (selected: Record<string, boolean>) => void
}) {
  const { s } = useI18n()
  return <fieldset className="min-w-0 space-y-2">
    <legend className="text-sm font-medium text-slate-800">{s('frontierLegend.title')}</legend>
    <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
      <p className="text-xs leading-5 text-slate-600">{s('frontierLegend.help')}</p>
      <Button onClick={() => onChange({})}>{s('frontierLegend.showAll')}</Button>
      <Button onClick={() => onChange(Object.fromEntries(items.map(item => [item.id, false])))}>{s('frontierLegend.hideAll')}</Button>
    </div>
    <div className="flex flex-wrap gap-2">
      {items.map(item => <label key={item.id} className={`inline-flex min-h-10 max-w-full cursor-pointer items-center gap-2 rounded-lg border px-3 py-2 text-xs leading-5 text-slate-600 hover:bg-slate-50 ${selected[item.id] === false ? 'border-dashed border-slate-300' : 'border-slate-200 bg-slate-50'}`}>
        <input type="checkbox" className="shrink-0 accent-accent-600" aria-label={item.name} checked={selected[item.id] !== false}
          onChange={event => onChange({ ...selected, [item.id]: event.target.checked })} />
        <i aria-hidden="true" style={item.symbolColor ? { color: item.symbolColor } : undefined} className={`shrink-0 ${item.symbolClass}`} />
        <span className={`min-w-0 break-words ${selected[item.id] === false ? 'line-through' : ''}`}>{item.name}{item.detail && `：${item.detail}`}</span>
      </label>)}
    </div>
    {items.length > 0 && items.every(item => selected[item.id] === false) && <p role="status" className="text-xs leading-5 text-slate-600">{s('frontierLegend.allHidden')}</p>}
  </fieldset>
}
