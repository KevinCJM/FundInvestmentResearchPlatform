import { useId, useLayoutEffect, useRef, type RefObject } from 'react'
import { createPortal } from 'react-dom'
import { QuestionMarkCircleIcon, XMarkIcon } from '@heroicons/react/24/outline'
import { Button } from '../ui'
import type { CmaMethodId } from '../../services/ltcma'
import { useLtcmaText } from './shared'

// Match the existing help layer, above the sticky research navigation.
const HELP_LAYER = 200

export default function LtcmaMethodHelp({ method, open, onOpen, onDismiss, containerRef }: {
  method: CmaMethodId; open: boolean; onOpen: () => void; onDismiss: () => void; containerRef: RefObject<HTMLDivElement>
}) {
  const { t, locale } = useLtcmaText()
  const id = useId(), titleId = `${id}-title`
  const trigger = useRef<HTMLButtonElement>(null), panel = useRef<HTMLDivElement>(null)
  const timer = useRef<number>(), focusOnOpen = useRef(false)
  const show = () => { window.clearTimeout(timer.current); onOpen() }
  const dismiss = () => { window.clearTimeout(timer.current); onDismiss() }
  const close = () => { trigger.current?.focus({ preventScroll: true }); dismiss() }
  const hideSoon = () => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => {
      if (document.activeElement !== trigger.current && !panel.current?.contains(document.activeElement)) onDismiss()
    }, 180)
  }
  const read = () => {
    show()
    if (panel.current) panel.current.focus({ preventScroll: true })
    else focusOnOpen.current = true
  }
  useLayoutEffect(() => () => window.clearTimeout(timer.current), [])
  useLayoutEffect(() => {
    if (!open) return
    const place = () => {
      if (!trigger.current || !panel.current) return
      const anchor = trigger.current.getBoundingClientRect(), box = panel.current.getBoundingClientRect()
      const width = document.documentElement.clientWidth || window.innerWidth
      const left = Math.max(8, Math.min(anchor.right - box.width, width - box.width - 8))
      const top = Math.max(8, Math.min(anchor.bottom + 6, window.innerHeight - box.height - 8))
      Object.assign(panel.current.style, { left: `${left}px`, top: `${top}px`, visibility: 'visible' })
    }
    const outside = (event: Event) => {
      if (event.target instanceof Node && !panel.current?.contains(event.target) && !trigger.current?.contains(event.target)) dismiss()
    }
    const scroll = (event: Event) => {
      if (event.target instanceof Node && panel.current?.contains(event.target)) return
      if (document.activeElement === trigger.current || panel.current?.contains(document.activeElement)) place()
      else dismiss()
    }
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close() }
    }
    place()
    if (focusOnOpen.current) { panel.current?.focus({ preventScroll: true }); focusOnOpen.current = false }
    window.addEventListener('resize', place)
    document.addEventListener('pointerdown', outside)
    document.addEventListener('scroll', scroll, true)
    document.addEventListener('keydown', key, true)
    return () => {
      window.removeEventListener('resize', place)
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('scroll', scroll, true)
      document.removeEventListener('keydown', key, true)
    }
  }, [open, locale])

  return <>
    <button ref={trigger} type="button" role="menuitem" tabIndex={-1} aria-label={t('methodExplanationOpen', { method: t(method) })} aria-haspopup="dialog" aria-expanded={open}
      aria-controls={open ? id : undefined} onMouseEnter={show} onMouseLeave={hideSoon} onFocus={show} onClick={read}
      // Viewport clamping can cover the trigger before pointerup; open on press as well as keyboard activation.
      onPointerDown={event => { if (event.button === 0) { event.preventDefault(); read() } }}
      onBlur={event => { if (!panel.current?.contains(event.relatedTarget as Node | null)) dismiss() }}
      className="inline-flex min-h-11 w-10 shrink-0 items-center justify-center rounded-lg text-slate-600 hover:bg-slate-100 hover:text-accent-700 focus-visible:ring-2 focus-visible:ring-accent-500">
      <QuestionMarkCircleIcon className="h-5 w-5" aria-hidden="true" />
    </button>
    {open && createPortal(<div ref={panel} id={id} role="dialog" aria-labelledby={titleId} tabIndex={0}
      onMouseEnter={show} onMouseLeave={hideSoon}
      onBlur={event => { if (!panel.current?.contains(event.relatedTarget as Node | null) && event.relatedTarget !== trigger.current) dismiss() }}
      style={{ zIndex: HELP_LAYER, left: 0, top: 0, visibility: 'hidden' }}
      className="fixed max-h-[calc(100dvh-16px)] w-[min(56rem,calc(100vw-16px))] overflow-auto overscroll-contain rounded-xl border border-slate-200 bg-white text-sm leading-6 tabular-nums text-slate-700 shadow-xl focus-visible:ring-2 focus-visible:ring-accent-500">
      <div className="sticky top-0 flex items-start justify-between gap-3 bg-white px-4 py-2">
        <h3 id={titleId} className="pt-2 font-semibold text-slate-900">{t('methodGuide', { method: t(method) })}</h3>
        <Button className="shrink-0" aria-label={t('methodExplanationClose')} onClick={close}><XMarkIcon className="h-4 w-4" aria-hidden="true" /></Button>
      </div>
      <div className="px-4 pb-4">
        <p className="mt-2"><strong className="font-medium text-slate-900">{t('methodInputs')}：</strong>{t(`methodHelp.${method}`)}</p>
        {method === 'long_term_scenario' && <p className="mt-2">{t('longTermExample.idea')}</p>}
        <p className="mt-2 text-xs text-slate-600">{t(method === 'long_term_scenario' ? 'longTermExample.exampleOnly' : 'methodExampleOnly')}</p>
        <ol className="mt-3 list-decimal space-y-3 pl-5 marker:font-semibold">
          {method === 'long_term_scenario' ? ['reference', 'blend', 'check', 'apply', 'combine'].map(step => <li key={step} className="pl-1">
            <p className="font-semibold text-slate-900">{t(`longTermExample.${step}.title`)}</p>
            <p>{t(`longTermExample.${step}.body`)}</p>
            {step === 'blend' && <p className="mt-1 font-medium text-slate-900">{t('longTermExample.formula')}</p>}
            {step === 'check' && <p className="mt-1 whitespace-pre-line">{t('longTermExample.rounds')}</p>}
          </li>) : [
            { title: 'methodCalculation', body: `methodHow.${method}` },
            { title: 'methodExampleTitle', body: `methodExample.${method}` },
            { title: 'methodInterpretation', body: `methodResult.${method}` },
          ].map(step => <li key={step.title} className="pl-1">
            <p className="font-semibold text-slate-900">{t(step.title)}</p>
            <p className="whitespace-pre-line">{t(step.body)}</p>
          </li>)}
        </ol>
        {method === 'long_term_scenario' && <div className="mt-4 space-y-2 border-t border-slate-200 pt-3 text-xs text-slate-600">
          <p>{t('longTermExample.fallback')}</p>
          <p>{t('longTermExample.limit')}</p>
        </div>}
      </div>
    </div>, containerRef.current ?? document.body)}
  </>
}
