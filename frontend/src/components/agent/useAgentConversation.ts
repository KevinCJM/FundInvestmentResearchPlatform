import { agentSessionStorageKey, agentContextKey as contextKey } from '../../services/agentContext'
import { freezePageSnapshot } from '../../services/agentPageEvidence'
export { agentSessionStorageKey } from '../../services/agentContext'
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  AgentApiError, agentEventUrl, cancelAgentRun, createAgentSession, fetchAgentEvents,
  fetchAgentRun, fetchAgentSession, fetchEarlierMessages, invalidateAgentContext, sendAgentMessage,
  type AgentDraft, type AgentEvent, type AgentPageContext, type AgentRun, type AgentSession, type AgentPreview, type AgentPreviewReference,
  type PageEvidenceSnapshot,
} from '../../services/agent'

export type ChatMessage = AgentEvent & { failed?: boolean; queued?: boolean }
type Pending = { id: string; text: string; context: AgentPageContext; snapshot?: PageEvidenceSnapshot; revision?: number; editOf?: string; resumeFrom?: string | null }
type EditTarget = { id: string; text: string }
type RunBinding = { run: AgentRun; frozen: AgentPageContext; key: string; adoptedKey?: string }
type ConversationCore = { session: Omit<AgentSession, 'active_run' | 'execution_blocked_by'> | null; binding: RunBinding | null; unboundBlock: string | null }
type ConversationInput =
  | { kind: 'session'; snapshot: AgentSession }
  | { kind: 'message'; run: AgentRun; context: AgentPageContext }
  | { kind: 'run'; run: AgentRun }
  | { kind: 'phase'; sessionId: string; runId: string; phase: string; status?: AgentRun['status'] }
const emptyCore = (): ConversationCore => ({ session: null, binding: null, unboundBlock: null })
export const messageId = () => globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`
export const running = (run?: AgentRun | null) => !!run && ['queued', 'running', 'stopping'].includes(run.status)
export type AgentConversationOptions = {
  enabled?: boolean
  cancelOnUnmount?: boolean
  loadPreview?: (sessionId: string, previewId: string) => Promise<AgentPreview>
  adoptPreviewContext?: (frozen: AgentPageContext, reference: AgentPreviewReference) => AgentPageContext | null
  /** Called once per outgoing message; the returned copy travels with that message and is never re-read later. */
  capturePageSnapshot?: () => PageEvidenceSnapshot | null
}

export default function useAgentConversation(pageContext: AgentPageContext, open: boolean, options: AgentConversationOptions = {}) {
  const optionsRef = useRef(options)
  optionsRef.current = options
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const [core, setCore] = useState<ConversationCore>(emptyCore)
  const coreRef = useRef(core)
  const publishCore = useCallback((next: ConversationCore) => { coreRef.current = next; setCore(next) }, [])
  const run = core.binding?.run || null
  const session: AgentSession | null = core.session ? { ...core.session, active_run: run,
    execution_blocked_by: core.binding ? run?.execution_blocked_by || null : core.unboundBlock } : null
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [activity, setActivity] = useState<AgentEvent[]>([])
  const [draft, setDraft] = useState<AgentDraft | null>(null)
  const [previewReference, setPreviewReference] = useState<AgentPreviewReference | null>(null)
  const [preview, setPreview] = useState<AgentPreview | null>(null)
  const [previewError, setPreviewError] = useState('')
  const [loadingPreview, setLoadingPreview] = useState(false)
  const [error, setError] = useState('')
  const [disconnected, setDisconnected] = useState(false)
  const [sending, setSending] = useState(false)
  const [waitingToSend, setWaitingToSend] = useState(false)
  const [olderCursor, setOlderCursor] = useState<number | null>(null)
  const [loadingEarlier, setLoadingEarlier] = useState(false)
  const [failed, setFailed] = useState<Pending | null>(null)
  const [reload, setReload] = useState(0)
  const [restoring, setRestoring] = useState(() => {
    try { return !!sessionStorage.getItem(agentSessionStorageKey(pageContext)) } catch { return false }
  })
  // A failed/unfinished restore is not evidence that the stored session is absent.
  const restorationPending = useRef(restoring)
  const contextRef = useRef(pageContext), messagesRef = useRef(messages)
  useEffect(() => () => {
    // Closing the floating panel does not unmount it. Changing its research object does.
    queueMicrotask(() => {
      const active = coreRef.current.binding?.run, current = coreRef.current.session
      if (!mounted.current && optionsRef.current.cancelOnUnmount && current && active && running(active)) {
        void cancelAgentRun(current.session_id, active.run_id, messageId()).catch(() => undefined)
      }
    })
  }, [])
  messagesRef.current = messages
  const cursor = useRef(0), invalidated = useRef(''), posting = useRef(false)
  const queued = useRef<Pending | null>(null)
  const queuedRead = useRef<Pending | null>(null)
  const editAttempt = useRef<Pending | null>(null)
  const replacedRuns = useRef(new Set<string>())
  const restoredActivity = useRef({ sessionId: '', seq: 0 })
  const previewReferenceRef = useRef(previewReference)
  previewReferenceRef.current = previewReference
  const captureSnapshot = useCallback(() => {
    // One immutable copy per message: later page edits or arriving results cannot rewrite it.
    const snapshot = optionsRef.current.capturePageSnapshot?.()
    return snapshot ? freezePageSnapshot(snapshot) : undefined
  }, [])
  const matchesContext = useCallback((context: AgentPageContext) => {
    const key = contextKey(context)
    const { binding, session: current } = coreRef.current
    const frozen = binding?.frozen || current?.page_context
    if (!frozen) return false
    if ((binding?.key || contextKey(frozen)) === key || binding?.adoptedKey === key) return true
    const reference = previewReferenceRef.current
    // A historical preview cannot grant a new run permission to adopt changed controls.
    if (!binding || !reference?.preview_id || reference.run_id !== binding.run.run_id) return false
    const adopted = optionsRef.current.adoptPreviewContext?.(frozen, reference)
    return !!adopted && key === contextKey(adopted)
  }, [])
  contextRef.current = pageContext
  const storageKey = agentSessionStorageKey(pageContext)
  const mergeActivity = useCallback((incoming: AgentEvent[]) => {
    const publicEvents: AgentEvent[] = []
    for (const event of incoming) {
      if (!['tool.started', 'tool.completed'].includes(event.type || '') || !Number.isSafeInteger(event.seq) || !event.seq || event.seq < 1 || !event.run_id) continue
      const data = event.data || {}
      if (typeof data.tool !== 'string' || !data.tool) continue
      publicEvents.push({ seq: event.seq, type: event.type, run_id: event.run_id, id: event.id || `activity-${event.seq}`, data: {
        tool: data.tool,
        ...(typeof data.status === 'string' ? { status: data.status } : {}),
        ...(typeof data.duration_ms === 'number' && Number.isFinite(data.duration_ms) && data.duration_ms >= 0 ? { duration_ms: data.duration_ms } : {}),
      } })
    }
    if (!publicEvents.length) return
    setActivity(previous => Array.from(new Map([...previous, ...publicEvents].map(event => [event.seq, event])).values())
      .sort((a, b) => a.seq! - b.seq!).slice(-100))
  }, [])
  const merge = useCallback((incoming: ChatMessage[]) => {
    setMessages(previous => {
      // A replacement event names the message it supersedes: live updates, snapshots and reconnects
      // all drop that message and the stopped turn around it, even when the cache still holds them.
      const superseded = new Set(replacedRuns.current), replacedIds = new Set<string>()
      for (const event of incoming) {
        if (event.speaker !== 'user' || !event.edit_of) continue
        replacedIds.add(event.edit_of)
        // The superseded message may only be inside this batch, e.g. a snapshot from a client that
        // cached the stopped turn; its run is still what must be suppressed.
        const replaced = previous.find(item => item.speaker === 'user' && item.id === event.edit_of)
          || incoming.find(item => item.speaker === 'user' && item.id === event.edit_of)
        if (replaced?.run_id) superseded.add(replaced.run_id)
      }
      replacedRuns.current = superseded
      const result = previous.filter(item => !(item.id && replacedIds.has(item.id)) && !(item.run_id && superseded.has(item.run_id)))
      for (const event of incoming) {
        if (!event.text && !event.artifacts) continue
        if (event.run_id && superseded.has(event.run_id)) continue
        const key = event.id || (event.speaker === 'user' ? event.message_id : event.run_id ? `${event.run_id}-reply` : undefined) || `event-${event.seq}`
        const found = result.findIndex(item => item.id === key && item.speaker === event.speaker)
        const next = { ...event, id: key, failed: false, queued: false }
        if (found >= 0) result[found] = { ...result[found], ...next, artifacts: { ...result[found].artifacts, ...next.artifacts } }
        else result.push(next)
      }
      return result.sort((a, b) => (a.seq ?? Number.MAX_SAFE_INTEGER) - (b.seq ?? Number.MAX_SAFE_INTEGER))
    })
  }, [])
  const acceptInput = useCallback((input: ConversationInput) => {
    if (!mounted.current) return false
    const before = coreRef.current
    let head = before.session, binding = before.binding, unboundBlock = before.unboundBlock
    const snapshot = input.kind === 'session' ? input.snapshot : null
    if (snapshot) {
      if ((snapshot.page_context && agentSessionStorageKey(snapshot.page_context) !== storageKey)
        || (head && (head.session_id !== snapshot.session_id || head.session_revision > snapshot.session_revision))) return false
      const { active_run: _active, execution_blocked_by: block, ...metadata } = snapshot
      head = metadata
      if (!binding) unboundBlock = block || null
    }
    if (!head) return false
    let value: AgentRun | null = null
    let frozen: AgentPageContext | undefined
    if (input.kind === 'phase') {
      if (!binding || head.session_id !== input.sessionId || binding.run.run_id !== input.runId || !running(binding.run)) return false
      value = { ...binding.run, phase: input.phase, status: input.status || binding.run.status }
    } else if (input.kind === 'session') {
      value = input.snapshot.active_run || null
      frozen = input.snapshot.page_context || contextRef.current
    } else {
      value = input.run
      if (input.kind === 'message') frozen = input.context
    }
    const previous = binding?.run, changingRun = !!value && previous?.run_id !== value.run_id
    const accepted = !!value && value.session_id === head.session_id && (changingRun
      ? !!frozen && agentSessionStorageKey(frozen) === storageKey && (!previous || value.session_revision > previous.session_revision)
      : !!previous && previous.run_revision <= value.run_revision && (running(previous) || !running(value))
        && !(previous.status === 'stopping' && ['queued', 'running'].includes(value.status)))
    if (accepted && value) {
      if (changingRun) {
        const key = contextKey(frozen!)
        binding = { run: value, frozen: JSON.parse(key) as AgentPageContext, key }
      } else binding = { ...binding!, run: value }
      unboundBlock = null
      if (input.kind === 'message' && head.session_revision <= value.session_revision) {
        head = { ...head, page_context: input.context, session_revision: value.session_revision }
      }
      if (input.kind !== 'phase' && value.response && value.response.session_revision > head.session_revision) {
        head = { ...head, session_revision: value.response.session_revision }
      }
    } else if (!snapshot) return false
    // All imperative readers and the next render observe the same accepted core object.
    publishCore({ session: head, binding, unboundBlock })
    if (!before.session) setActivity([])
    if (snapshot && (accepted || !binding)) {
      setDraft(snapshot.draft || null); setPreviewReference(snapshot.preview || null)
    }
    if (!accepted || !value) return true
    if (!changingRun && running(previous) && !running(value)) setReload(n => n + 1)
    // Ordered phase events cannot replay a full reply or its artifacts.
    if (input.kind === 'phase') return true
    if (value.artifacts && Object.keys(value.artifacts).length) merge([{ id: `${value.run_id}-reply`, run_id: value.run_id, speaker: 'assistant',
      ...(snapshot ? { seq: (snapshot.next_event_seq || 1) - 1 } : {}), artifacts: value.artifacts }])
    if (value.response) {
      const text = value.response.reply?.text
      if (text) merge([{ id: `${value.run_id}-reply`, run_id: value.run_id, speaker: 'assistant', text, artifacts: value.response.artifacts }])
      if (matchesContext(contextRef.current)) {
        setDraft(value.response.draft || null)
        if ('preview' in value.response) setPreviewReference(value.response.preview || null)
      }
    }
    return true
  }, [merge, matchesContext, publishCore, storageKey])
  const acceptSession = useCallback((snapshot: AgentSession) => acceptInput({ kind: 'session', snapshot }), [acceptInput])
  const ownsRun = useCallback((sessionId: string, runId: string) => mounted.current
    && coreRef.current.session?.session_id === sessionId && coreRef.current.binding?.run.run_id === runId, [])

  const post = useCallback(async (message: Pending, onFailure?: () => void): Promise<AgentRun | null> => {
    if (!mounted.current || posting.current || restorationPending.current || optionsRef.current.enabled === false) return null
    posting.current = true; setSending(true); setError(''); setFailed(null)
    if (queued.current === message) {
      queued.current = null; setWaitingToSend(false)
      setMessages(previous => previous.map(item => item.id === message.id ? { ...item, queued: false } : item))
    }
    let current = coreRef.current.session
    const restoredRun = () => coreRef.current.binding?.run?.session_id === current?.session_id && coreRef.current.binding?.run?.message_id === message.id ? coreRef.current.binding?.run : null
    try {
      if (!current) {
        current = await createAgentSession(message.context)
        if (!mounted.current) return null
        acceptSession(current)
        try { sessionStorage.setItem(storageKey, current.session_id) } catch { /* Storage is optional. */ }
      }
      // Revision is an optimistic lock, not part of the server's message identity.
      // Freeze the resume parent (including its absence) across every retry.
      message.revision = current.session_revision
      if (message.resumeFrom === undefined) {
        const last = coreRef.current.binding?.run
        message.resumeFrom = !message.editOf && last && !running(last) && last.status !== 'completed' ? last.run_id : null
      }
      const response = await sendAgentMessage(current.session_id, {
        message_id: message.id, expected_session_revision: message.revision, text: message.text, page_context: message.context,
        ...(message.snapshot ? { page_snapshot: message.snapshot } : {}),
        // A replaced turn is never the parent of its replacement.
        ...(message.editOf ? { edit_of_message_id: message.editOf }
          : message.resumeFrom ? { resume_from_run_id: message.resumeFrom } : {}),
      })
      if (!mounted.current) {
        if (optionsRef.current.cancelOnUnmount && running(response)) {
          void cancelAgentRun(current.session_id, response.run_id, messageId()).catch(() => undefined)
        }
        return null
      }
      // Bind an accepted edit before merging its reply can retire the replaced run.
      if (message.editOf) setMessages(items => items.map(item => item.id === message.id ? { ...item, run_id: response.run_id } : item))
      if (acceptInput({ kind: 'message', run: response, context: message.context })) setReload(n => n + 1)
      // A restored receipt is authoritative even if the original POST arrives later.
      return restoredRun() || response
    } catch (reason) {
      if (!mounted.current) return null
      const restored = restoredRun()
      if (restored) return restored
      setError(reason instanceof Error ? reason.message : message.editOf ? '消息修改未保存，请重试。' : '消息发送失败，请重试。')
      setFailed(message)
      if (reason instanceof AgentApiError && reason.code === 'REVISION_CONFLICT') setReload(n => n + 1)
      setMessages(previous => previous.map(item => item.id === message.id ? { ...item, failed: true, queued: false } : item))
      onFailure?.()
      return null
    } finally { posting.current = false; if (mounted.current) setSending(false) }
  }, [acceptInput, acceptSession, storageKey])

  const submitEdit = useCallback(async (target: EditTarget, text: string, reuse?: Pending): Promise<AgentRun | null> => {
    const content = text.trim(), active = coreRef.current.binding?.run
    if (!content || content.length > 4000 || !target.id || !coreRef.current.session || !active || posting.current || queued.current
      || running(active) || active.status !== 'cancelled') {
      setError('这条消息已不能再修改，请刷新后重试。')
      return null
    }
    // One attempt keeps one identity: pressing send again after a failure replays the same request
    // instead of duplicating an accepted one or aiming at a message that is no longer last.
    const attempt = reuse ?? (editAttempt.current?.editOf === target.id && editAttempt.current.text === content ? editAttempt.current : undefined)
    const message: Pending = attempt ?? { id: messageId(), text: content, context: structuredClone(contextRef.current), snapshot: captureSnapshot(), editOf: target.id }
    editAttempt.current = message
    // The replaced turn leaves both the transcript and the composer's target, so a failed
    // request puts the user message and the stopped turn's notice back before it reports.
    const removed = messagesRef.current.filter(item => item.speaker === 'assistant' && !!item.run_id && item.run_id === active.run_id)
    setMessages(previous => previous.flatMap(item => item.speaker === 'user' && item.id === target.id
      ? [{ ...item, id: message.id, text: content, failed: false, queued: false }]
      : item.speaker === 'assistant' && !!item.run_id && item.run_id === active.run_id ? [] : [item]))
    const accepted = await post(message, () => setMessages(previous => [...previous.map(item => item.id === message.id
      ? { ...item, id: target.id, text: target.text, failed: false, queued: false } : item), ...removed]
      .sort((a, b) => (a.seq ?? Number.MAX_SAFE_INTEGER) - (b.seq ?? Number.MAX_SAFE_INTEGER))))
    if (!accepted) return null
    // The replaced turn is closed for this client too, and the optimistic message belongs to the new
    // run so a late event or a stale snapshot of the old turn cannot bring it back.
    editAttempt.current = null
    if (active.run_id) replacedRuns.current.add(active.run_id)
    setMessages(previous => previous.map(item => item.id === message.id ? { ...item, run_id: accepted.run_id } : item))
    return accepted
  }, [captureSnapshot, post])

  const stop = useCallback(async () => {
    const current = coreRef.current.session, active = coreRef.current.binding?.run
    if (!current || !active || !running(active)) return
    try {
      const value = await cancelAgentRun(current.session_id, active.run_id, messageId())
      if (!ownsRun(current.session_id, active.run_id)) return
      acceptInput({ kind: 'run', run: value }); setReload(n => n + 1)
    } catch (reason) {
      if (ownsRun(current.session_id, active.run_id) && running(coreRef.current.binding?.run)) setError(reason instanceof Error ? reason.message : '停止请求未送达，请重试。')
    }
  }, [acceptInput, ownsRun])

  const send = useCallback(async (text: string, retry?: Pending): Promise<AgentRun | null> => {
    if (restorationPending.current) return null
    const message = retry || { id: messageId(), text: text.trim(), context: structuredClone(contextRef.current), snapshot: captureSnapshot() }
    if (!message.text || posting.current || queued.current) return null
    merge([{ id: message.id, speaker: 'user', text: message.text }])
    if (running(coreRef.current.binding?.run) && !retry) {
      queued.current = message
      setWaitingToSend(true)
      setMessages(previous => previous.map(item => item.id === message.id ? { ...item, queued: true } : item))
      await stop()
      return null
    }
    return post(message)
  }, [captureSnapshot, merge, post, stop])

  useEffect(() => {
    if (!queued.current || running(run) || queuedRead.current === queued.current) return
    // Keep the queue locked until post takes over, including duplicate terminal snapshots.
    const message = queued.current; queuedRead.current = message
    void (async () => {
      try {
        const snapshot = coreRef.current.session ? await fetchAgentSession(coreRef.current.session.session_id) : null
        if (!mounted.current || queued.current !== message) return
        if (snapshot) acceptSession(snapshot)
        if (!await post(message) && mounted.current && queued.current === message) throw new Error('消息尚未发送，请稍后重试。')
      } catch (reason) {
        if (!mounted.current || queued.current !== message) return
        setError(reason instanceof Error ? reason.message : '无法恢复会话。'); setFailed(message)
        setMessages(previous => previous.map(item => item.id === message.id ? { ...item, queued: false, failed: true } : item))
      } finally {
        if (queued.current === message) { queued.current = null; if (mounted.current) setWaitingToSend(false) }
        if (queuedRead.current === message) queuedRead.current = null
      }
    })()
  }, [run, post, acceptSession])

  useEffect(() => {
    if (!open || optionsRef.current.enabled === false) return
    restorationPending.current = !coreRef.current.session
    setRestoring(true)
    let active = true, source: EventSource | null = null, timer: ReturnType<typeof setTimeout> | undefined
    let polling = false, snapshotSeq = 0, failures = 0
    const currentSession = () => coreRef.current.session?.session_id
    const applyEvent = (event: AgentEvent) => {
      if (!active || !event.seq || event.seq <= cursor.current) return
      cursor.current = event.seq
      mergeActivity([event])
      if (event.type === 'user.message' || event.type === 'assistant.message') merge([event])
      if (event.seq <= snapshotSeq || event.run_id !== coreRef.current.binding?.run?.run_id) return
      const data = event.data || {}
      if (event.type === 'draft.updated' || event.type === 'preview.updated') {
        const key = event.type === 'draft.updated' ? 'draft' : 'preview'
        merge([{ id: `${event.run_id}-reply`, run_id: event.run_id, seq: event.seq, speaker: 'assistant', artifacts: { [key]: data[key] } }])
      }
      if (event.type === 'tool.started' && coreRef.current.binding?.run && running(coreRef.current.binding?.run)) {
        acceptInput({ kind: 'phase', sessionId: coreRef.current.binding.run.session_id, runId: coreRef.current.binding.run.run_id, phase: 'tool' })
      }
      if (event.type === 'draft.updated' && matchesContext(contextRef.current)) setDraft(data.draft as AgentDraft)
      if (event.type === 'preview.updated' && matchesContext(contextRef.current)) setPreviewReference(data.preview as AgentPreviewReference | null)
      if (event.type === 'run.phase' || event.type === 'run.started' || event.type === 'run.recovering') {
        const previous = coreRef.current.binding?.run
        if (previous && running(previous)) acceptInput({ kind: 'phase', sessionId: previous.session_id, runId: previous.run_id,
          phase: String(data.phase || 'thinking'), status: (data.status || previous.status) as AgentRun['status'] })
      }
    }
    const pull = async () => {
      const sid = currentSession(); if (!sid) return
      let more = true
      while (active && more) {
        const page = await fetchAgentEvents(sid, cursor.current)
        if (!active) return
        for (const event of page.items) applyEvent(event)
        more = page.has_more
      }
      if (active && coreRef.current.binding?.run) {
        const value = await fetchAgentRun(sid, coreRef.current.binding?.run.run_id)
        if (active) acceptInput({ kind: 'run', run: value })
      }
    }
    const poll = async () => {
      if (!active) return
      try { await pull(); if (active) { setDisconnected(false); failures = 0 } }
      catch { if (active) { setDisconnected(true); failures += 1 } }
      if (active && (running(coreRef.current.binding?.run) || coreRef.current.binding?.run?.execution_blocked_by || failures)) timer = setTimeout(() => void poll(), Math.min(15000, 1000 * 2 ** Math.min(failures, 4)))
    }
    const fallback = () => {
      source?.close()
      if (!polling && active) { polling = true; setDisconnected(true); void poll() }
    }
    void (async () => {
      let foundSession = false
      try {
        let sid = currentSession()
        if (!sid) { try { sid = sessionStorage.getItem(storageKey) || undefined } catch { /* optional */ } }
        if (!sid) { restorationPending.current = false; return }
        const snapshot = await fetchAgentSession(sid)
        if (!active) return
        foundSession = true
        if (coreRef.current.session?.session_id === sid && snapshot.session_revision < coreRef.current.session.session_revision) return
        if (snapshot.page_context && agentSessionStorageKey(snapshot.page_context) !== storageKey) {
          try { sessionStorage.removeItem(storageKey) } catch { /* optional */ }
          restorationPending.current = false
          return
        }
        if (!acceptSession(snapshot)) return
        restorationPending.current = false
        snapshotSeq = (snapshot.next_event_seq || 1) - 1
        mergeActivity((snapshot.events || []).filter(event => (event.seq || 0) > snapshotSeq - 200))
        if (snapshot.messages && snapshotSeq > 0 && (restoredActivity.current.sessionId !== sid || restoredActivity.current.seq < snapshotSeq)) {
          // Fetch only the recent event window; activity never advances the message/SSE cursor.
          void fetchAgentEvents(sid, Math.max(0, snapshotSeq - 200)).then(page => {
            if (!active || currentSession() !== sid) return
            mergeActivity(page.items.filter(event => (event.seq || 0) <= snapshotSeq))
            restoredActivity.current = { sessionId: sid, seq: snapshotSeq }
          }).catch(() => { /* Optional activity history must not block conversation recovery. */ })
        }
        if (snapshot.messages) {
          merge(snapshot.messages)
          cursor.current = Math.max(cursor.current, snapshotSeq)
          setOlderCursor(snapshot.older_message_cursor || null)
        }
        await pull()
        if (!active) return
        setDisconnected(false)
        if (!running(coreRef.current.binding?.run) && !coreRef.current.binding?.run?.execution_blocked_by) return
        if (typeof EventSource === 'undefined') { fallback(); return }
        source = new EventSource(agentEventUrl(sid, cursor.current))
        source.addEventListener('agent', raw => {
          try {
            const event = JSON.parse((raw as MessageEvent).data) as AgentEvent
            if (event.seq && event.seq > cursor.current + 1) { fallback(); return }
            applyEvent(event)
            if (['run.completed', 'run.paused', 'run.cancelled', 'run.failed'].includes(event.type || '')) void pull().catch(fallback)
          } catch { fallback() }
        })
        source.onopen = () => { if (active) setDisconnected(false) }
        source.onerror = fallback
      } catch (reason) {
        if (!active) return
        if (!foundSession && reason instanceof AgentApiError && reason.status === 404) {
          restorationPending.current = false
          try { sessionStorage.removeItem(storageKey) } catch { /* optional */ }
          publishCore(emptyCore()); cursor.current = 0; setActivity([])
          setDraft(null); setPreviewReference(null); setPreview(null)
        } else { setDisconnected(true); timer = setTimeout(() => setReload(n => n + 1), 2000) }
      } finally { if (active) setRestoring(false) }
    })()
    return () => { active = false; source?.close(); if (timer) clearTimeout(timer) }
  }, [open, options.enabled, storageKey, reload, session?.session_id, acceptInput, acceptSession, merge, mergeActivity, matchesContext, publishCore])

  const currentKey = contextKey(pageContext)
  useEffect(() => {
    const current = coreRef.current, binding = current.binding
    if (binding && binding.key !== currentKey && binding.adoptedKey !== currentKey && matchesContext(pageContext)) {
      publishCore({ ...current, binding: { ...binding, adoptedKey: currentKey } })
    }
  }, [currentKey, run?.run_id, previewReference?.preview_id, pageContext, matchesContext, publishCore])
  const previewId = previewReference?.preview_id
  const previewMatchesDraft = !!previewReference && previewReference.definition_hash === draft?.definition_hash
  const hasLegacyPreview = !!previewReference && !previewId
  useEffect(() => {
    let active = true
    if (preview && preview.preview_id === previewId && previewMatchesDraft && matchesContext(pageContext)) return
    setPreview(null); setPreviewError(''); setLoadingPreview(false)
    if (hasLegacyPreview) { setPreviewError('这次旧试算未保存完整结果，请让助手重新试算。'); return }
    const loadPreview = optionsRef.current.loadPreview
    if (!loadPreview || !previewId || !session?.session_id || !previewMatchesDraft || !matchesContext(pageContext)) return
    setLoadingPreview(true)
    void loadPreview(session.session_id, previewId).then(value => {
      if (active && matchesContext(contextRef.current)) setPreview(value)
    }).catch(reason => {
      if (active) setPreviewError(reason instanceof Error ? reason.message : '试算结果加载失败，请重新连接或重试。')
    }).finally(() => { if (active) setLoadingPreview(false) })
    return () => { active = false }
  }, [previewId, session?.session_id, previewMatchesDraft, hasLegacyPreview, currentKey, reload])
  useEffect(() => {
    const current = coreRef.current.session, active = coreRef.current.binding?.run
    if (!current || !active || !running(active) || matchesContext(pageContext) || invalidated.current === `${active.run_id}:${currentKey}`) return
    invalidated.current = `${active.run_id}:${currentKey}`
    setDraft(previous => previous ? { ...previous, stale: true } : null)
    // The editor's indicator revision may decrease when switching indicators.
    // Invalidation advances relative to this run's frozen context, not that revision.
    const invalidContext = { ...pageContext, context_revision: Math.max(pageContext.context_revision, (current.page_context?.context_revision || 0) + 1) }
    void invalidateAgentContext(current.session_id, active.run_id, messageId(), invalidContext).then(value => {
      if (ownsRun(current.session_id, active.run_id)) acceptInput({ kind: 'run', run: value })
    }).catch(reason => {
      if (ownsRun(current.session_id, active.run_id) && running(coreRef.current.binding?.run) && currentKey === contextKey(contextRef.current)) {
        setError(reason instanceof Error ? reason.message : '研究口径变更未同步，请停止后重试。')
      }
    })
  }, [currentKey, run?.run_id, pageContext, acceptInput, ownsRun])

  const cancelQueued = () => {
    const message = queued.current
    if (!message) return
    queued.current = null; setWaitingToSend(false); setFailed(message)
    setMessages(previous => previous.map(item => item.id === message.id ? { ...item, queued: false, failed: true } : item))
  }
  const loadEarlier = async () => {
    if (!coreRef.current.session || !olderCursor || loadingEarlier) return
    setLoadingEarlier(true)
    try {
      const page = await fetchEarlierMessages(coreRef.current.session.session_id, olderCursor)
      merge(page.items); setOlderCursor(page.older_cursor)
    } catch (reason) { setError(reason instanceof Error ? reason.message : '更早消息加载失败。') }
    finally { setLoadingEarlier(false) }
  }
  const retry = async (): Promise<AgentRun | null> => {
    if (!failed) return null
    if (!failed.editOf) return await send(failed.text, failed)
    // Restoration may already have replaced the old message with this accepted edit.
    // Replay its receipt without treating it as a new edit of the now-missing target.
    if (coreRef.current.binding?.run?.message_id === failed.id) {
      const accepted = await post(failed)
      if (accepted) editAttempt.current = null
      return accepted
    }
    const target = messagesRef.current.find(item => item.speaker === 'user' && item.id === failed.editOf)
    if (!target?.id) { setError('找不到要重新发送的消息，请刷新后重试。'); return null }
    return submitEdit({ id: target.id, text: target.text || '' }, failed.text, failed)
  }
  return { session, run, messages, activity, draft, preview, previewError, loadingPreview, error, setError, disconnected, restoring: restoring || restorationPending.current, sending: sending || waitingToSend, failed, cancelQueued,
    // Leaving an abandoned edit drops its failed receipt and its cached request identity.
    clearFailed: () => { editAttempt.current = null; setFailed(null) },
    contextChanged: !!(core.binding || core.session?.page_context) && !matchesContext(pageContext),
    loadEarlier, hasEarlier: !!olderCursor, loadingEarlier,
    busy: sending || waitingToSend || running(run) || !!run?.execution_blocked_by || !!session?.execution_blocked_by,
    send, stop, edit: submitEdit, retry, refresh: () => setReload(n => n + 1), acceptSession }
}

export type AgentConversationState = ReturnType<typeof useAgentConversation>
