import { useId, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

export default function RegimeHelpTip({ text, label = '查看说明', dark = false }: { text: string; label?: string; dark?: boolean }) {
  const tooltipId = useId()
  const trigger = useRef<HTMLSpanElement>(null)
  const tooltip = useRef<HTMLSpanElement>(null)
  const [open, setOpen] = useState(false)
  const hideTimer = useRef<number>()
  const show = () => { window.clearTimeout(hideTimer.current); setOpen(true) }
  const hideSoon = () => { window.clearTimeout(hideTimer.current); hideTimer.current = window.setTimeout(() => setOpen(false), 120) }
  useLayoutEffect(() => () => window.clearTimeout(hideTimer.current), [])

  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      if (!trigger.current || !tooltip.current) return
      const anchor = trigger.current.getBoundingClientRect()
      const box = tooltip.current.getBoundingClientRect()
      const width = document.documentElement.clientWidth || window.innerWidth
      const height = window.innerHeight
      const left = Math.max(8, Math.min(anchor.left + anchor.width / 2 - box.width / 2, width - box.width - 8))
      const top = anchor.top >= box.height + 15 ? anchor.top - box.height - 7 : anchor.bottom + 7
      Object.assign(tooltip.current.style, { left: `${left}px`, top: `${Math.max(8, Math.min(top, height - box.height - 8))}px`, visibility: 'visible' })
    }
    const dismiss = () => { window.clearTimeout(hideTimer.current); setOpen(false) }
    const onScroll = (event: Event) => { if (!(event.target instanceof Node) || !tooltip.current?.contains(event.target)) dismiss() }
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); dismiss() }
    }
    place()
    window.addEventListener('resize', place)
    // The anchor may scroll behind a drawer header; do not leave a detached hint floating.
    document.addEventListener('scroll', onScroll, true)
    document.addEventListener('keydown', onKey, true)
    return () => {
      window.removeEventListener('resize', place)
      document.removeEventListener('scroll', onScroll, true)
      document.removeEventListener('keydown', onKey, true)
    }
  }, [open, text])

  return <>
    <span
      ref={trigger} tabIndex={0} role="button" aria-label={label} aria-describedby={open ? tooltipId : undefined}
      onMouseEnter={show} onMouseLeave={hideSoon}
      onFocus={show} onBlur={() => setOpen(false)}
      onClick={event => { event.preventDefault(); event.stopPropagation(); show() }}
      onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); event.stopPropagation(); show() } }}
      className={`ml-1 inline-grid h-4 w-4 cursor-help place-items-center rounded-full border align-middle text-xs font-black leading-none outline-none focus-visible:ring-2 focus-visible:ring-accent-500 ${dark ? 'border-slate-500 text-slate-600' : 'border-slate-300 bg-white text-slate-600'}`}
    >?</span>
    {open && createPortal(<span
      ref={tooltip} id={tooltipId} role="tooltip" onMouseEnter={show} onMouseLeave={hideSoon}
      style={{ left: 0, top: 0, visibility: 'hidden' }}
      className="fixed z-[200] max-h-[calc(100vh-16px)] w-[min(16rem,calc(100vw-16px))] overflow-auto whitespace-normal break-words rounded-lg bg-slate-950 px-3 py-2 text-left text-xs font-normal leading-5 text-white shadow-xl"
    >{text}</span>, document.body)}
  </>
}
