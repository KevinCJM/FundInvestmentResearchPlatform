import { useEffect, useRef, type ReactNode } from 'react'

export default function RegimeWorkbenchDrawer({ title, side = 'right', wide = false, onClose, closeDisabled = false, children }: {
  title: string
  side?: 'left' | 'right'
  wide?: boolean
  closeDisabled?: boolean
  onClose: () => void
  children: ReactNode
}) {
  const panel = useRef<HTMLDivElement>(null)
  const close = useRef(onClose)
  close.current = onClose
  useEffect(() => {
    const trigger = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const focusable = () => [...(panel.current?.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), a[href], [tabindex="0"]') ?? [])].filter((element) => !element.closest('[hidden]') && !element.closest('details:not([open]) > :not(summary)'))
    focusable()[0]?.focus()
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); close.current(); return }
      if (event.key !== 'Tab') return
      const items = focusable()
      const first = items[0], last = items[items.length - 1]
      if (!first) { event.preventDefault(); return }
      if (event.shiftKey && (document.activeElement === first || !panel.current?.contains(document.activeElement))) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || !panel.current?.contains(document.activeElement))) { event.preventDefault(); first.focus() }
    }
    document.addEventListener('keydown', onKey)
    return () => { document.removeEventListener('keydown', onKey); trigger?.focus() }
  }, [title])
  return <div className="fixed inset-0 z-[110] flex min-w-0" data-testid="regime-workbench-drawer">
    <button type="button" tabIndex={-1} aria-label="关闭浮层遮罩" disabled={closeDisabled} className="absolute inset-0 bg-slate-950/15" onClick={onClose} />
    <div ref={panel} role="dialog" aria-modal="true" aria-label={title} className={`relative flex h-full min-w-0 max-w-full flex-col border-slate-200 bg-slate-50 shadow-2xl ${side === 'left' ? 'mr-auto border-r' : 'ml-auto border-l'} ${wide ? 'w-[900px]' : 'w-[420px]'}`}>
      <div className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-200 bg-white px-4 py-3"><h2 className="text-sm font-bold text-slate-950">{title}</h2><button type="button" aria-label={`关闭${title}`} disabled={closeDisabled} onClick={onClose} className="min-h-9 rounded-lg border border-slate-200 px-3 text-xs font-bold text-slate-600">关闭</button></div>
      <div className="min-h-0 min-w-0 flex-1 space-y-4 overflow-auto p-3">{children}</div>
    </div>
  </div>
}
