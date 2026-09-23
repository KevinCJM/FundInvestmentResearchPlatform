import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { ArrowDownIcon, ArrowUpIcon, ArrowsPointingInIcon, ArrowsPointingOutIcon, CheckIcon, ClipboardDocumentIcon, InformationCircleIcon, PencilIcon, PencilSquareIcon, StopIcon, XMarkIcon } from '@heroicons/react/24/outline'
import Mascot from '../Mascot'
import { Button } from '../ui'
import { fetchAgentMeta, type AgentEvent, type AgentMeta, type AgentPageContext } from '../../services/agent'
import AgentMessageContent from './AgentMessageContent'
import AgentMemory from './AgentMemory'
import useAgentConversation, { agentSessionStorageKey, running, type AgentConversationState, type AgentConversationOptions } from './useAgentConversation'
import { IconButton, ActionFeedback, type Feedback } from './AgentControls'
import { useI18n } from '../../i18n/runtime'
import './AgentPanel.css'

export type AgentArtifactContext = {
  event: AgentEvent; eventKey: string; isCurrent: boolean; chat: AgentConversationState
  busy: boolean; close: (returnFocus?: boolean) => void
}

type Props = {
  pageContext: AgentPageContext
  busy?: boolean
  conversationOptions?: AgentConversationOptions
  renderStatus?: (chat: AgentConversationState) => ReactNode
  renderWelcome?: (context: { chat: AgentConversationState; fillInput: (text: string) => void }) => ReactNode
} & ({
  hasArtifacts: (event: AgentEvent, chat: AgentConversationState) => boolean
  renderArtifacts: (context: AgentArtifactContext) => ReactNode
} | { hasArtifacts?: never; renderArtifacts?: never })

const AGENT_FLOATING_LAYER = 70
// This is the existing server resume command, not a translated UI label.
const RESUME_MESSAGE = '继续'
const TOOL_LABELS: Record<string, string> = {
  'metrics.lookup': 'agent.tool.lookup', 'metrics.infer': 'agent.tool.infer', 'context.read': 'agent.tool.read',
  'page.read': 'agent.tool.read',
  'products.search': 'agent.tool.search', 'metrics.availability': 'agent.tool.availability', 'metrics.validate': 'agent.tool.validate',
  'metrics.preview': 'agent.tool.preview', 'metrics.draft_save': 'agent.tool.draft', 'metrics.rolling_draft': 'agent.tool.rolling',
  'products.eval': 'agent.tool.eval', 'products.series': 'agent.tool.series', 'products.plans': 'agent.tool.plans',
  'portfolios.context': 'agent.tool.portfolioContext', 'portfolios.eval': 'agent.tool.portfolioEval',
}
const PAGE_LABELS: Record<AgentPageContext['page'], string> = {
  'indicator-studio': 'agent.page.indicator', 'product-detail': 'agent.page.research', 'product-research': 'agent.page.research',
  'product-compare': 'agent.page.compare', 'holding-diagnosis': 'agent.page.holding', 'evaluation-plan': 'agent.page.evaluation',
}
function Activity({ events, status }: { events: AgentEvent[]; status?: string }) {
  const { s } = useI18n()
  if (!events.length) return status ? <p role="status" className="py-2 text-sm text-slate-600">{status}</p> : null
  return <div className="my-2">
    {status && <p role="status" className="text-sm text-slate-600">{status}</p>}
    <details>
      <summary className="flex min-h-10 cursor-pointer items-center text-xs font-semibold text-slate-600">{s('agent.activityHistory', { count: events.length })}</summary>
      <ol className="divide-y divide-slate-100 text-xs text-slate-600">
        {events.map(event => <li key={event.seq || event.id} className="flex flex-wrap justify-between gap-2 py-2">
          <span>{s(TOOL_LABELS[String(event.data?.tool)] || 'agent.tool.unknown')}</span>
          <span>{event.type === 'tool.started' ? s('agent.activityStarted') : event.data?.status === 'ok' ? s('agent.activityCompleted') : event.data?.status === 'not_executed' ? s('agent.activitySkipped') : s('agent.activityIncomplete')}{typeof event.data?.duration_ms === 'number' ? s('agent.duration', { duration: event.data.duration_ms }) : ''}</span>
        </li>)}
      </ol>
    </details>
  </div>
}

export default function AgentPanel(props: Props) {
  const [conversation, setConversation] = useState(0)
  // Remounting disposes the old subscription, cursors and all turn-local UI state.
  return <AgentConversation key={`${agentSessionStorageKey(props.pageContext)}:${conversation}`} {...props} initiallyOpen={conversation > 0} onNewConversation={() => setConversation(value => value + 1)} />
}

function AgentConversation({ pageContext, busy: externalBusy = false, conversationOptions, renderStatus, renderWelcome, hasArtifacts, renderArtifacts, initiallyOpen, onNewConversation }: Props & { initiallyOpen: boolean; onNewConversation: () => void }) {
  const { s } = useI18n()
  const panelId = useId()
  const [open, setOpen] = useState(initiallyOpen)
  const [expanded, setExpanded] = useState(false)
  const [contextOpen, setContextOpen] = useState(false)
  const [meta, setMeta] = useState<AgentMeta | null>(null)
  const [text, setText] = useState('')
  const [editing, setEditing] = useState<{ id: string; text: string; draft: string } | null>(null)
  const [feedback, setFeedback] = useState<Record<string, Feedback>>({})
  const [unread, setUnread] = useState(false)
  const [viewportHeight, setViewportHeight] = useState<number>()
  const chat = useAgentConversation(pageContext, open, conversationOptions)
  const { session, messages: events, error, setError } = chat
  const busy = chat.busy || externalBusy || chat.restoring
  const sendBlocked = chat.sending || chat.restoring || externalBusy || !meta?.configured || !!chat.run?.execution_blocked_by || !!session?.execution_blocked_by
  const focusNewInput = initiallyOpen && !!meta?.configured
  const thinking = chat.sending || running(chat.run)
  const launcherRef = useRef<HTMLButtonElement>(null)
  const titleRef = useRef<HTMLHeadingElement>(null)
  const messagesRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const followingRef = useRef(true)
  const anchorRef = useRef<{ id: string; top: number } | null>(null)
  const activity = chat.activity
  const activeActivity = activity.filter(event => event.run_id === chat.run?.run_id)
  const latestTool = activeActivity[activeActivity.length - 1]
  const status = chat.run?.status === 'stopping' ? s('agent.phase.stopping')
    : chat.sending && !running(chat.run) ? s('agent.phase.sending')
    : chat.run?.status === 'queued' ? s('agent.phase.queued')
    : chat.run?.phase === 'compacting' ? s('agent.phase.compacting')
    : chat.run?.phase === 'waiting_retry' ? s('agent.phase.retry')
    : chat.run?.phase === 'recovery' ? s('agent.phase.recovery')
    : chat.run?.phase === 'tool' ? s('agent.phase.tool', { tool: s(TOOL_LABELS[String(latestTool?.data?.tool)] || 'agent.tool.generic') }) : s('agent.phase.thinking')
  const hasActiveReply = running(chat.run) && events.some(event => event.speaker === 'assistant' && event.run_id === chat.run?.run_id && (event.text || hasArtifacts?.(event, chat)))
  const targets = (pageContext.calculation.targets || []) as Array<{ name?: string; product_id: string }>
  const setActionFeedback = (key: string, value: Feedback) => setFeedback(previous => ({ ...previous, [key]: { ...previous[key], error: false, ...value } }))

  useEffect(() => {
    const viewport = window.visualViewport
    if (!viewport) return
    const resize = () => setViewportHeight(viewport.height)
    resize(); viewport.addEventListener('resize', resize)
    return () => viewport.removeEventListener('resize', resize)
  }, [])
  useLayoutEffect(() => {
    const input = inputRef.current
    if (!input || !open) return
    input.style.height = 'auto'
    input.style.height = `${Math.min(input.scrollHeight, 144)}px`
  }, [text, open])
  useLayoutEffect(() => {
    if (!open) return
    const area = messagesRef.current
    if (!area) return
    if (anchorRef.current) {
      const anchor = Array.from(area.querySelectorAll<HTMLElement>('[data-message-id]')).find(item => item.dataset.messageId === anchorRef.current?.id)
      if (anchor) area.scrollTop += anchor.getBoundingClientRect().top - anchorRef.current.top
      if (!chat.loadingEarlier) anchorRef.current = null
      return
    }
    if (followingRef.current) { area.scrollTop = area.scrollHeight; setUnread(false) }
    else if (events.length) setUnread(true)
  }, [events, thinking, error, open, expanded, chat.loadingEarlier, activity, status])

  const close = (returnFocus = true) => {
    setOpen(false)
    if (returnFocus) requestAnimationFrame(() => launcherRef.current?.focus())
  }
  useEffect(() => {
    if (!open) return
    let active = true
    void fetchAgentMeta().then(value => { if (active) setMeta(value) }).catch(reason => { if (active) setError(reason instanceof Error ? reason.message : s('agent.metaFailed')) })
    return () => { active = false }
  }, [open])
  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !event.defaultPrevented) { event.preventDefault(); close() }
    }
    document.addEventListener('keydown', onKeyDown)
    const frame = requestAnimationFrame(() => (focusNewInput ? inputRef.current : titleRef.current)?.focus())
    return () => { cancelAnimationFrame(frame); document.removeEventListener('keydown', onKeyDown) }
  }, [open, focusNewInput])

  const send = () => {
    const content = text.trim()
    if (!content || sendBlocked) return
    if (editing) {
      const target = { id: editing.id, text: editing.text }
      followingRef.current = true
      void chat.edit(target, content).then(sent => {
        if (!sent) return
        // Fence the accepted submission: only the edit and text this request still owns are cleared.
        setEditing(current => current?.id === target.id ? null : current)
        setText(current => current.trim() === content ? '' : current)
      })
      return
    }
    setText(''); followingRef.current = true; void chat.send(content)
  }
  const retryFailed = async () => {
    const pending = chat.failed
    if (!pending || chat.sending || chat.restoring) return
    if (!await chat.retry()) return
    if (!pending.editOf) return
    setEditing(null)
    setText(current => current.trim() === pending.text.trim() ? '' : current)
  }
  const startEdit = (event: AgentEvent) => {
    if (!event.id || !event.text) return
    setEditing({ id: event.id, text: event.text, draft: text })
    setText(event.text)
    requestAnimationFrame(() => inputRef.current?.focus())
  }
  const cancelEdit = () => {
    if (!editing) return
    if (chat.failed?.editOf === editing.id) chat.clearFailed()
    setText(editing.draft); setEditing(null)
  }
  const clearContext = () => {
    if (busy || chat.restoring) return
    try { sessionStorage.removeItem(agentSessionStorageKey(pageContext)) }
    catch { setError(s('agent.clearContextFailed')); return }
    onNewConversation()
  }
  const loadEarlier = () => {
    const area = messagesRef.current
    const first = area && Array.from(area.querySelectorAll<HTMLElement>('[data-message-id]')).find(item => item.getBoundingClientRect().bottom > area.getBoundingClientRect().top)
    if (first) anchorRef.current = { id: first.dataset.messageId!, top: first.getBoundingClientRect().top }
    followingRef.current = false; void chat.loadEarlier()
  }
  const copyMessage = async (content: string, key: string) => {
    try { await navigator.clipboard.writeText(content); setActionFeedback(key, { text: s('agent.messageCopied') }) }
    catch { setActionFeedback(key, { text: s('agent.messageCopyFailed'), error: true }) }
  }
  const lastUserMessageId = events.reduce<string | undefined>((found, event) => event.speaker === 'user' && event.text && event.id ? event.id : found, undefined)
  const artifacts = chat.run?.response?.artifacts
  // A validated draft or a preview is a real result; an invalid draft is never shown and stays replaceable.
  const producedArtifacts = artifacts?.draft?.valid === true || artifacts?.preview != null
  const editableTurn = !busy && !chat.restoring && !running(chat.run) && chat.run?.status === 'cancelled'
    && chat.run.message_id === lastUserMessageId && !producedArtifacts
  const canEdit = (event: AgentEvent) => editableTurn && event.speaker === 'user' && event.id === lastUserMessageId && editing?.id !== event.id

  return createPortal(<div style={{ zIndex: AGENT_FLOATING_LAYER, ...(viewportHeight ? { '--agent-viewport-height': `${viewportHeight}px` } : {}) } as CSSProperties} className="agent-floating pointer-events-none fixed bottom-3 right-3 max-w-[calc(100vw-24px)] sm:bottom-5 sm:right-5 sm:max-w-[calc(100vw-40px)]">
    <div id={panelId} hidden={!open} role="dialog" aria-modal="false" aria-labelledby={`${panelId}-title`} className={`agent-dialog pointer-events-auto ${open ? 'flex' : 'hidden'} ${expanded ? 'agent-expanded' : ''} max-w-full flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-xl`}>
      <header className="flex shrink-0 items-center gap-1 border-b border-slate-200 px-3 py-2">
        {open && <Mascot state="welcome" className="h-8 w-8 object-contain" />}
        <h2 ref={titleRef} tabIndex={-1} id={`${panelId}-title`} className="min-w-0 flex-1 text-lg font-semibold text-slate-900">{s('agent.title')}</h2>
        <IconButton label={s('agent.clearContext')} hint={busy ? s('agent.clearContextBusy') : chat.restoring ? s('agent.loading') : s('agent.clearContextHint')} placement="bottom" disabled={busy || chat.restoring} onClick={clearContext}><PencilSquareIcon className="h-5 w-5" aria-hidden="true" /></IconButton>
        <IconButton label={s('agent.contextAndModel')} placement="bottom" aria-expanded={contextOpen} aria-controls={`${panelId}-context`} onClick={() => { setContextOpen(value => !value); if (!contextOpen) { followingRef.current = false; requestAnimationFrame(() => { if (messagesRef.current) messagesRef.current.scrollTop = 0 }) } }}><InformationCircleIcon className="h-5 w-5" aria-hidden="true" /></IconButton>
        <IconButton className="hidden lg:inline-flex" label={expanded ? s('agent.collapseWidth') : s('agent.expandReading')} placement="bottom" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? <ArrowsPointingInIcon className="h-5 w-5" aria-hidden="true" /> : <ArrowsPointingOutIcon className="h-5 w-5" aria-hidden="true" />}</IconButton>
        <IconButton label={s('agent.closeAssistant')} placement="bottom" onClick={() => close()}><XMarkIcon className="h-5 w-5" aria-hidden="true" /></IconButton>
      </header>
      <div ref={messagesRef} onScroll={() => { const area = messagesRef.current; if (area) { followingRef.current = area.scrollHeight - area.scrollTop - area.clientHeight < 48; if (followingRef.current) setUnread(false) } }} className="min-h-0 flex-1 space-y-3 overflow-y-auto overscroll-contain p-3">
        <section id={`${panelId}-context`} aria-label={s('agent.contextRegion')} hidden={!contextOpen} className="space-y-2 border-b border-slate-200 pb-3 text-xs text-slate-600">
          <h3 className="font-semibold">{s('agent.contextRegion')}</h3>
          <p className="break-words">{meta ? (meta.configured ? s('agent.configuredModel', { model: meta.model || s('agent.selectedApi') }) : s('agent.configureModel')) : s('agent.readingMeta')}</p>
          <p>{s(PAGE_LABELS[pageContext.page])} · {targets.length ? s('agent.targetCount', { count: targets.length }) : s('agent.noProductRequired')}</p>
          <p>{s('agent.period', { period: String(pageContext.calculation.period || s('agent.pageSettings')) })}</p><p>{s('agent.asOf', { date: pageContext.view_state === 'unknown' ? s('agent.unknownPointInTime') : pageContext.calculation.as_of ? String(pageContext.calculation.as_of) : pageContext.view_state === 'off' ? s('agent.unrestricted') : s('agent.pageBasis') })}</p>{targets.map(target => <p key={target.product_id}>{target.name || target.product_id}</p>)}
        </section>
        {chat.hasEarlier && <Button disabled={chat.loadingEarlier} onClick={loadEarlier}>{chat.loadingEarlier ? s('agent.loading') : s('agent.earlierMessages')}</Button>}
        {session?.legacy_history_incomplete && <p className="text-xs text-slate-600">{s('agent.incompleteHistory')}</p>}
        {events.length === 0 && <div className="py-3"><p className="text-base font-semibold text-slate-900">{s('agent.welcome')}</p>{renderWelcome?.({ chat, fillInput: value => { setText(value); inputRef.current?.focus() } })}</div>}
        <div role="log" aria-label={s('agent.conversationLog')} aria-live="polite" aria-relevant="additions text" className="space-y-4 break-words">
          {events.map((event, index) => {
            const isCurrent = !!event.run_id && event.run_id === chat.run?.run_id
            if (!event.text && !hasArtifacts?.(event, chat)) return null
            const eventKey = `${event.speaker}-${event.id || event.seq || index}`
            const records = activity.filter(item => item.run_id === event.run_id)
            return <article key={eventKey} data-message-id={eventKey} data-run-id={event.run_id} aria-label={event.speaker === 'user' ? s('agent.you') : 'AI'} className={event.speaker === 'user' ? 'agent-bubble ml-auto min-w-0 max-w-[96%] text-sm leading-6 text-accent-900 sm:max-w-[90%]' : 'agent-reply min-w-0 text-sm leading-6 text-slate-700'}>
              <div data-message-bubble className={event.speaker === 'user' ? 'rounded-xl border border-accent-100 bg-accent-50 px-3 py-2' : 'rounded-xl border border-slate-200 bg-slate-50 px-3 py-2'}>
                {event.speaker === 'assistant' && <Activity events={records} status={isCurrent && running(chat.run) ? status : undefined} />}
                {event.text && (event.speaker === 'assistant' ? <AgentMessageContent text={event.text} /> : <p className="whitespace-pre-wrap">{event.text}</p>)}
                {event.queued && <div className="mt-1 text-xs text-slate-600"><p>{s('agent.queueWaiting')}</p><Button onClick={chat.cancelQueued}>{s('agent.cancelQueued')}</Button></div>}
                {event.failed && <p className="mt-1 text-xs text-rose-700">{s('agent.messagePreserved')}</p>}
                {renderArtifacts?.({ event, eventKey, isCurrent, chat, busy, close })}
              </div>
              {(canEdit(event) || !!event.text) && <div className="mt-1 flex items-center justify-end gap-1">
                {canEdit(event) && <IconButton className="agent-reply-copy" label={s('agent.editMessage')} onClick={() => startEdit(event)}><PencilIcon className="h-5 w-5" aria-hidden="true" /></IconButton>}
                {!!event.text && <IconButton className={`agent-reply-copy ${feedback[eventKey] && !feedback[eventKey].error ? 'agent-copy-confirmed' : ''}`} label={s('agent.copyMessage')} hint={feedback[eventKey] && !feedback[eventKey].error ? s('agent.messageCopied') : s('agent.copyMessage')} onClick={() => void copyMessage(event.text!, eventKey)}>{feedback[eventKey] && !feedback[eventKey].error ? <CheckIcon className="h-5 w-5" aria-hidden="true" /> : <ClipboardDocumentIcon className="h-5 w-5" aria-hidden="true" />}</IconButton>}
              </div>}
              {!!event.text && <ActionFeedback value={feedback[eventKey]} />}
            </article>
          })}
          {thinking && !hasActiveReply && <Activity events={activeActivity} status={status} />}
        </div>
        {renderStatus?.(chat)}
        {error && <p role="alert" className="rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
        <AgentMemory chat={chat} busy={busy} />
        {chat.restoring && <p role="status" className="text-sm text-slate-600">{s('agent.loading')}</p>}
        {chat.failed && <Button disabled={chat.sending || chat.restoring} onClick={() => void retryFailed()}>{s('agent.retryMessage')}</Button>}
        {chat.disconnected && <p role="status" className="text-sm text-slate-600">{s('agent.disconnected')}<Button onClick={chat.refresh}>{s('agent.reconnect')}</Button></p>}
        {chat.run?.status === 'paused' && <Button disabled={busy} onClick={() => void chat.send(RESUME_MESSAGE)}>{s('agent.continue')}</Button>}
        {(chat.run?.status === 'interrupted' || chat.run?.status === 'failed') && <Button disabled={busy} onClick={() => void chat.send(RESUME_MESSAGE)}>{s('agent.resume')}</Button>}
        {(chat.run?.execution_blocked_by || session?.execution_blocked_by) && <p role="alert" className="text-sm text-amber-900">{s('agent.blocked')}</p>}
        {chat.contextChanged && <p className="text-xs text-slate-600">{s('agent.contextChanged')}</p>}
        {meta?.configured === false && <div className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-900"><p>{s('agent.noModel')}</p><a href="/settings/llm-api" className="mt-1 inline-flex min-h-10 items-center rounded-lg font-semibold underline underline-offset-4 focus-visible:ring-2 focus-visible:ring-accent-500">{s('agent.openSettings')}</a></div>}
      </div>
      {unread && <div className="mx-auto mb-1 shrink-0"><IconButton label={s('agent.newMessages')} onClick={() => { followingRef.current = true; if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight; setUnread(false) }}><ArrowDownIcon className="h-5 w-5" aria-hidden="true" /></IconButton></div>}
      <form className="agent-composer shrink-0 border-t border-slate-200 bg-white p-2" onSubmit={event => { event.preventDefault(); send() }}>
        <label className="sr-only" htmlFor={`${panelId}-message`}>{s('agent.messageLabel')}</label>
        <div className="rounded-lg border border-slate-300 bg-white focus-within:ring-2 focus-within:ring-accent-500">
          {editing && <div className="flex min-h-10 items-center justify-between gap-2 border-b border-slate-200 px-2 text-xs font-semibold text-slate-700">
            <span>{s('agent.editingMessage')}</span>
            <Button className="min-h-10 px-2" disabled={chat.sending} onClick={cancelEdit}>{s('agent.cancelEdit')}</Button>
          </div>}
          <textarea ref={inputRef} id={`${panelId}-message`} rows={1} maxLength={4000} value={text} onChange={event => setText(event.target.value)} onKeyDown={event => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && !event.nativeEvent.isComposing && event.keyCode !== 229) { event.preventDefault(); send() } }} placeholder={s('agent.placeholder')} className="block w-full resize-none rounded-lg border-0 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-600 placeholder:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-500" disabled={!meta?.configured} />
          <div className="flex flex-wrap items-center justify-between gap-1 px-1 pb-1">
            {text.length >= 3600 && <span className="text-xs tabular-nums text-slate-600">{text.length}/4000</span>}
            <div className="ml-auto flex items-center gap-1">
              {running(chat.run) && <IconButton label={chat.run?.status === 'stopping' ? s('agent.stopping') : s('agent.stop')} disabled={chat.run?.status === 'stopping'} onClick={() => void chat.stop()}><StopIcon className="h-5 w-5" aria-hidden="true" /></IconButton>}
              {running(chat.run) && !!text.trim() ? <Button className="px-2" type="submit" tone="primary" disabled={sendBlocked}>{s('agent.stopAndSend')}</Button>
                : !running(chat.run) && <IconButton label={s('agent.send')} type="submit" primary disabled={sendBlocked || !text.trim()}><ArrowUpIcon className="h-5 w-5" aria-hidden="true" /></IconButton>}
            </div>
          </div>
        </div>
      </form>
    </div>
    {!open && <button ref={launcherRef} type="button" onClick={() => setOpen(true)} aria-label={s('agent.openAssistant')} aria-haspopup="dialog" aria-expanded={false} aria-controls={panelId} className="pointer-events-auto flex shrink-0 flex-col items-center rounded-lg p-1 focus-visible:ring-2 focus-visible:ring-accent-500 focus-visible:ring-offset-2"><Mascot state="welcome" className="h-16 w-16 object-contain sm:h-20 sm:w-20" /><span className="rounded-full border border-slate-200 bg-white px-2 py-1 text-xs font-semibold text-slate-700 shadow-sm">{s('agent.title')}</span></button>}
  </div>, document.body)
}
