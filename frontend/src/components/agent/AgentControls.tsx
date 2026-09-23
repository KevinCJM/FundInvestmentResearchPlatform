import { useEffect, useId, useRef, useState, type ButtonHTMLAttributes } from 'react'
import { createPortal } from 'react-dom'
const AGENT_TOOLTIP_LAYER = 71
export type Feedback = { text: string; error?: boolean; saved?: boolean }

export function IconButton({ label, hint = label, children, className = '', placement = 'top', primary = false, ...props }: ButtonHTMLAttributes<HTMLButtonElement> & { label: string; hint?: string; placement?: 'top' | 'bottom'; primary?: boolean }) {
  const [anchor, setAnchor] = useState<DOMRect | null>(null)
  const triggerRef = useRef<HTMLSpanElement>(null)
  const showHint = !!anchor
  const show = () => setAnchor(triggerRef.current?.getBoundingClientRect() || null)
  const hide = () => setAnchor(null)
  const hintId = useId()
  useEffect(() => {
    if (!showHint) return
    const dismiss = (event: KeyboardEvent) => { if (event.key === 'Escape') { event.preventDefault(); hide() } }
    document.addEventListener('keydown', dismiss, true)
    document.addEventListener('scroll', hide, true)
    window.addEventListener('resize', hide)
    return () => { document.removeEventListener('keydown', dismiss, true); document.removeEventListener('scroll', hide, true); window.removeEventListener('resize', hide) }
  }, [showHint])
  return <span ref={triggerRef} className={`agent-icon-control relative inline-flex shrink-0 ${className}`} onPointerEnter={event => { if (event.pointerType !== 'touch') show() }} onPointerLeave={event => { if (!(event.relatedTarget instanceof Element && event.relatedTarget.id === hintId)) hide() }}>
    <button {...props} type={props.type || 'button'} aria-label={label} aria-describedby={showHint ? hintId : undefined} onFocus={show} onBlur={hide} onClick={event => { hide(); props.onClick?.(event) }} className={`inline-flex h-10 w-10 items-center justify-center rounded-lg focus-visible:ring-2 focus-visible:ring-accent-500 disabled:cursor-not-allowed ${primary ? 'bg-accent-600 text-white hover:bg-accent-700 disabled:bg-slate-100 disabled:text-slate-600' : 'text-slate-600 hover:bg-slate-100 disabled:opacity-50'}`}>
      {children}
    </button>
    {anchor && createPortal(<span id={hintId} role="tooltip" onPointerEnter={show} onPointerLeave={event => { if (!(event.relatedTarget instanceof Node && triggerRef.current?.contains(event.relatedTarget))) hide() }} style={{ zIndex: AGENT_TOOLTIP_LAYER, right: Math.max(8, window.innerWidth - anchor.right), ...(placement === 'bottom' ? { top: anchor.bottom } : { bottom: Math.max(8, window.innerHeight - anchor.top) }) }} className="agent-tooltip fixed w-max max-w-[min(220px,calc(100vw-16px))] rounded-lg bg-slate-900 px-2 py-1 text-xs font-normal text-white">{hint}</span>, document.body)}
  </span>
}

export function ActionFeedback({ value }: { value?: Feedback }) {
  return value ? <p role={value.error ? 'alert' : 'status'} className={`mt-2 text-xs ${value.error ? 'text-rose-700' : 'text-emerald-800'}`}>{value.text}</p> : null
}
