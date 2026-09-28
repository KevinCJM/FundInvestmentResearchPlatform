import { useId, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { CheckIcon, ChevronDownIcon } from '@heroicons/react/24/outline'
import type { CmaMethodId, LtcmaCapabilities } from '../../services/ltcma'
import LtcmaMethodHelp from './LtcmaMethodHelp'
import { control, useLtcmaText } from './shared'

const MENU_LAYER = 180

export default function LtcmaMethodSelect({ value, methods, describedBy, onChange }: {
  value: CmaMethodId; methods: LtcmaCapabilities['methods']; describedBy: string; onChange: (method: CmaMethodId) => void
}) {
  const { t } = useLtcmaText(), id = useId()
  const root = useRef<HTMLDivElement>(null), trigger = useRef<HTMLButtonElement>(null), menu = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [helpMethod, setHelpMethod] = useState<CmaMethodId | null>(null)
  const close = () => { setHelpMethod(null); setOpen(false); trigger.current?.focus() }
  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      if (!trigger.current || !menu.current) return
      const box = trigger.current.getBoundingClientRect()
      const below = window.innerHeight - box.bottom - 12, above = box.top - 12
      const upwards = below < 240 && above > below
      Object.assign(menu.current.style, {
        left: `${box.left}px`, width: `${box.width}px`, maxHeight: `${Math.max(44, upwards ? above : below)}px`,
        top: upwards ? 'auto' : `${box.bottom + 4}px`, bottom: upwards ? `${window.innerHeight - box.top + 4}px` : 'auto',
      })
    }
    const outside = (event: Event) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false)
    }
    const scroll = (event: Event) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) place()
    }
    setHelpMethod(null)
    place()
    const selected = menu.current?.querySelector<HTMLButtonElement>('[aria-checked="true"]:not(:disabled)')
      ?? menu.current?.querySelector<HTMLButtonElement>('[role="menuitemradio"]:not(:disabled)')
    selected?.focus({ preventScroll: true })
    window.addEventListener('resize', place)
    document.addEventListener('pointerdown', outside)
    document.addEventListener('scroll', scroll, true)
    return () => {
      window.removeEventListener('resize', place)
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('scroll', scroll, true)
    }
  }, [open])
  const navigate = (event: KeyboardEvent<HTMLDivElement>) => {
    // The worked example has its own keyboard navigation and Escape handling.
    if (!menu.current?.contains(event.target as Node)) return
    if (event.key === 'Escape') { event.preventDefault(); close(); return }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    event.preventDefault()
    const items = Array.from(menu.current.querySelectorAll<HTMLButtonElement>('button[role^="menuitem"]:not(:disabled)'))
    const index = items.indexOf(document.activeElement as HTMLButtonElement)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length
    items[next]?.focus()
  }
  return <div ref={root} className="min-w-0 self-start" onKeyDown={navigate}
    onBlur={event => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOpen(false) }}>
    <label id={`${id}-label`} htmlFor={id} className="mb-1 block text-sm font-medium text-slate-700">{t('method')}</label>
    <button ref={trigger} id={id} type="button" value={value} aria-label={t('method')} aria-describedby={describedBy}
      aria-haspopup="menu" aria-expanded={open} aria-controls={open ? `${id}-menu` : undefined}
      className={`${control} flex items-center justify-between gap-2 text-left`}
      onClick={() => setOpen(previous => !previous)} onKeyDown={event => {
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp') { event.preventDefault(); setOpen(true) }
      }}>
      <span>{t(value)}</span><ChevronDownIcon className="h-4 w-4 shrink-0" aria-hidden="true" />
    </button>
    {open && <div ref={menu} id={`${id}-menu`} role="menu" aria-labelledby={`${id}-label`}
      style={{ zIndex: MENU_LAYER }} className="fixed overflow-y-auto overscroll-contain rounded-xl border border-slate-200 bg-white p-1 shadow-xl">
      {methods.map(item => <div key={item.id} role="none" className={`flex items-center rounded-lg ${value === item.id ? 'bg-accent-50' : ''}`}>
        <button type="button" role="menuitemradio" aria-checked={value === item.id} value={item.id} tabIndex={-1} disabled={!item.available}
          title={item.available ? undefined : item.reason ?? t('unavailable')}
          className="flex min-h-11 min-w-0 items-center gap-2 rounded-lg px-2 text-left text-sm text-slate-700 hover:bg-slate-100 focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed"
          onClick={() => { onChange(item.id); close() }}>
          <CheckIcon className={`h-4 w-4 shrink-0 ${value === item.id ? 'text-accent-700' : 'invisible'}`} aria-hidden="true" />
          <span>{t(item.id)}{item.available ? '' : ` · ${t('unavailable')}`}</span>
        </button>
        <LtcmaMethodHelp method={item.id} open={helpMethod === item.id} containerRef={root} onOpen={() => setHelpMethod(item.id)} onDismiss={() => setHelpMethod(current => current === item.id ? null : current)} />
      </div>)}
    </div>}
  </div>
}
