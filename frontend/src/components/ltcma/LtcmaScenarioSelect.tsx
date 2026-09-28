import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { CheckIcon, ChevronDownIcon } from '@heroicons/react/24/outline'
import { control, useLtcmaText } from './shared'

const MENU_LAYER = 180
export interface ScenarioChoice { value: string; name: string; available: boolean; reasons: string[] }

/** Native disabled options cannot reliably expose hover/focus explanations. */
export default function LtcmaScenarioSelect({ label, value, choices, onChange, action }: {
  label: string; value: string; choices: ScenarioChoice[]; onChange: (value: string) => void
  action?: { name: string; run: () => void }
}) {
  const { t } = useLtcmaText(), id = useId()
  const root = useRef<HTMLDivElement>(null), trigger = useRef<HTMLButtonElement>(null), menu = useRef<HTMLDivElement>(null)
  const explanation = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false), [inspected, setInspected] = useState<string | null>(null)
  const current = choices.find(item => item.value === value), detail = choices.find(item => item.value === inspected)
  const reasons = detail && !detail.available ? detail.reasons.length ? detail.reasons : [t('scenarioHistoryUnavailable')] : []
  const close = () => { setOpen(false); setInspected(null); trigger.current?.focus() }
  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      if (!trigger.current || !menu.current) return
      const box = trigger.current.getBoundingClientRect()
      const below = window.innerHeight - box.bottom - 12, above = box.top - 12
      const up = below < 260 && above > below
      Object.assign(menu.current.style, { left: `${box.left}px`, width: `${box.width}px`,
        maxHeight: `${Math.max(80, up ? above : below)}px`,
        top: up ? 'auto' : `${box.bottom + 4}px`, bottom: up ? `${window.innerHeight - box.top + 4}px` : 'auto' })
    }
    const outside = (event: Event) => { if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false) }
    const scroll = (event: Event) => { if (event.target instanceof Node && !menu.current?.contains(event.target)) place() }
    place()
    const first = menu.current?.querySelector<HTMLButtonElement>('[aria-checked="true"]') ?? menu.current?.querySelector<HTMLButtonElement>('button')
    first?.focus({ preventScroll: true })
    document.addEventListener('pointerdown', outside)
    document.addEventListener('scroll', scroll, true)
    window.addEventListener('resize', place)
    return () => { document.removeEventListener('pointerdown', outside); document.removeEventListener('scroll', scroll, true); window.removeEventListener('resize', place) }
  }, [open])
  useLayoutEffect(() => {
    if (!open || !reasons.length) return
    const place = () => {
      const row = menu.current?.querySelector<HTMLButtonElement>('[aria-describedby]')
      const panel = explanation.current
      if (!row || !panel) return
      const box = row.getBoundingClientRect(), gap = 8, padding = 12
      const width = Math.min(340, window.innerWidth - padding * 2)
      const right = window.innerWidth - box.right >= width + gap + padding
      const left = !right && box.left >= width + gap + padding
      panel.style.width = `${width}px`
      const height = Math.min(160, panel.scrollHeight)
      const below = window.innerHeight - box.bottom - padding >= height + gap
      const top = right || left ? box.top : below ? box.bottom + gap : box.top - height - gap
      Object.assign(panel.style, {
        left: `${right ? box.right + gap : left ? box.left - width - gap : Math.max(padding, Math.min(box.left, window.innerWidth - width - padding))}px`,
        top: `${Math.max(padding, Math.min(top, window.innerHeight - height - padding))}px`,
      })
    }
    place()
    window.addEventListener('resize', place)
    document.addEventListener('scroll', place, true)
    return () => { window.removeEventListener('resize', place); document.removeEventListener('scroll', place, true) }
  }, [open, inspected, reasons.join('\n')])
  const navigate = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); close(); return }
    if (!open || !menu.current?.contains(event.target as Node) || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const buttons = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('button') ?? [])
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + buttons.length) % buttons.length
    buttons[next]?.focus()
  }
  const pick = (item: ScenarioChoice) => { if (!item.available) { setInspected(item.value); return }; onChange(item.value); close() }
  return <div ref={root} className="min-w-0" onKeyDown={navigate}
    onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false) }}>
    <button ref={trigger} type="button" value={value} aria-label={label} aria-haspopup="menu" aria-expanded={open}
      aria-controls={open ? `${id}-menu` : undefined} className={`${control} flex items-center justify-between gap-2 text-left`}
      onClick={() => { setInspected(null); setOpen(previous => !previous) }} onKeyDown={event => {
        if (!open && ['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); setOpen(true) }
      }}>
      <span className="min-w-0 break-words">{current?.name ?? t('choose')}</span><ChevronDownIcon className="h-4 w-4 shrink-0" aria-hidden="true" />
    </button>
    {open && <div ref={menu} id={`${id}-menu`} style={{ zIndex: MENU_LAYER }} className="fixed flex flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl">
      <div role="menu" aria-label={label} className="min-h-11 overflow-y-auto overscroll-contain p-1">
        {[{ value: '', name: t('choose'), available: true, reasons: [] }, ...choices].map(item => <button key={item.value} type="button"
          role="menuitemradio" tabIndex={-1} aria-checked={value === item.value} aria-disabled={!item.available} value={item.value}
          aria-describedby={!item.available && inspected === item.value ? `${id}-reason` : undefined}
          className={`flex min-h-11 w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm focus-visible:ring-2 focus-visible:ring-accent-500 ${item.available
            ? `font-medium text-slate-900 hover:bg-slate-100 ${value === item.value ? 'bg-accent-50' : ''}`
            : 'cursor-not-allowed bg-slate-100 font-normal text-slate-600 hover:bg-slate-200'}`}
          onMouseEnter={() => setInspected(item.value)} onFocus={() => setInspected(item.value)} onClick={() => pick(item)}>
          <CheckIcon className={`h-4 w-4 shrink-0 ${value === item.value ? item.available ? 'text-accent-700' : 'text-slate-600' : 'invisible'}`} aria-hidden="true" />
          <span className="min-w-0 break-words">{item.name}{item.available ? '' : ` · ${t('unavailable')}`}</span>
        </button>)}
        {action && <button type="button" role="menuitem" tabIndex={-1} className="min-h-11 w-full rounded-lg border-t border-slate-200 px-3 py-2 text-left text-sm font-medium text-accent-700 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-accent-500"
          onMouseEnter={() => setInspected(null)} onFocus={() => setInspected(null)} onClick={() => { close(); action.run() }}>{action.name}</button>}
      </div>
    </div>}
    {open && reasons.length > 0 && <div ref={explanation} id={`${id}-reason`} role="tooltip" tabIndex={0}
      style={{ zIndex: MENU_LAYER + 1 }} className="fixed max-h-40 overflow-y-auto rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm leading-6 text-amber-900 shadow-lg">
      {reasons.map((reason, index) => <p key={index}>{reason}</p>)}
    </div>}
  </div>
}
