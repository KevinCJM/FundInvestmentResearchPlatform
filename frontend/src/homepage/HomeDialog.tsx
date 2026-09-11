import { useEffect, useId, useRef, type ReactNode } from 'react'
import { XMarkIcon } from '@heroicons/react/24/outline'
import { useI18n } from '../i18n/runtime'

/** Native modal supplies focus containment, inert background, and Escape handling. */
export function HomeDialog({ title, children, onClose }: { title: string; children: ReactNode; onClose: () => void }) {
  const dialog = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const { s } = useI18n()
  useEffect(() => {
    const element = dialog.current!
    const previousFocus = document.activeElement
    const previousOverflow = document.body.style.overflow
    element.showModal()
    element.querySelector<HTMLElement>('[data-home-autofocus]')?.focus()
    document.body.style.overflow = 'hidden'
    return () => {
      element.close()
      document.body.style.overflow = previousOverflow
      if (previousFocus instanceof HTMLElement && previousFocus.isConnected) previousFocus.focus()
    }
  }, [])
  return <dialog ref={dialog} className="home-dialog" aria-labelledby={titleId} onCancel={event => { event.preventDefault(); onClose() }} onKeyDown={event => {
    // Search inputs otherwise consume the first Escape to clear their text.
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose(); return }
    // Keep Tab cycling inside the modal instead of moving into browser chrome.
    if (event.key !== 'Tab') return
    const items = Array.from(event.currentTarget.querySelectorAll<HTMLElement>('a[href], button, input, select, textarea, [tabindex]'))
      .filter(item => item.tabIndex >= 0 && !item.matches(':disabled') && item.getClientRects().length > 0)
    const first = items[0], last = items[items.length - 1]
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus() }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus() }
  }}>
    <div className="home-dialog-heading"><h2 id={titleId}>{title}</h2><button type="button" className="home-icon-button" aria-label={s('landing.close')} onClick={onClose}><XMarkIcon aria-hidden="true" /></button></div>
    {children}
  </dialog>
}
