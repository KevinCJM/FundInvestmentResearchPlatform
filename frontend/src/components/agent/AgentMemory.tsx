import { useEffect, useState } from 'react'
import { Button } from '../ui'
import { useI18n } from '../../i18n/runtime'
import { decideAgentMemory, fetchAgentMemory, fetchAgentSession, revokeAgentMemory, type AgentMemorySource, type AgentSessionReceipt } from '../../services/agent'
import type { AgentConversationState } from './useAgentConversation'

/** Separate human approval for preferences; never invokes indicator saving. */
export default function AgentMemory({ chat, busy }: { chat: AgentConversationState; busy: boolean }) {
  const { s } = useI18n()
  const [items, setItems] = useState<AgentMemorySource[]>([])
  const [pending, setPending] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [legacyUnavailable, setLegacyUnavailable] = useState(false)
  const [reload, setReload] = useState(0)
  const session = chat.session
  const failure = actionError || error
  useEffect(() => { setActionError('') }, [session?.session_id])
  const memoryAvailable = !!session && ('memory_proposals' in session || 'memory_sources' in session)
  useEffect(() => {
    let active = true
    if (!session || !memoryAvailable) { setItems([]); return }
    setLoading(true)
    void fetchAgentMemory(session.session_id).then(value => { if (active) { setItems(value.items || []); setLegacyUnavailable(!!value.legacy_unavailable); setError('') } })
      .catch(reason => { if (active) setError(reason instanceof Error ? reason.message : s('agent.memory.loadFailed')) })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [session?.session_id, session?.session_revision, memoryAvailable, reload])
  const proposals = (session?.memory_proposals || []).filter(item => item.status === 'pending')
  const used = session?.memory_sources || []
  if (!session || (!proposals.length && !items.length && !used.length && !failure && !legacyUnavailable)) return null
  const act = async (operation?: () => Promise<AgentSessionReceipt>) => {
    if (pending || busy || loading) return
    setPending(true); setActionError('')
    try {
      const receipt = await operation?.()
      if (receipt) chat.acceptSessionRevision(receipt.session_id, receipt.session_revision)
      // Reconcile proposals before reloading records, including after a lost decision response.
      chat.acceptSession(await fetchAgentSession(session.session_id))
      chat.refresh(); setReload(value => value + 1)
    }
    catch (reason) { setActionError(reason instanceof Error ? reason.message : s('agent.memory.actionFailed')) }
    finally { setPending(false) }
  }
  return <details className="my-2 text-sm text-slate-700">
    <summary className="flex min-h-10 cursor-pointer items-center text-xs font-semibold">{s('agent.memory.title', { count: proposals.length })}</summary>
    <p className="text-xs text-slate-600">{s('agent.memory.precedence')}</p>
    {legacyUnavailable && <p className="text-xs text-slate-600">{s('agent.memory.legacyUnavailable')}</p>}
    {busy && <p className="text-xs text-slate-600">{s('agent.memory.busy')}</p>}
    <div className="divide-y divide-slate-200">
      {proposals.map(proposal => {
        const replacing = items.find(item => item.key === proposal.key && item.object_id === (proposal.object_id || 'scope'))
        return <div key={proposal.proposal_id} className="py-2">
          <p className="whitespace-pre-wrap break-words">{proposal.summary}</p>
          <p className="text-xs text-slate-600">{s('agent.memory.userSource')}</p>
          {replacing && <p className="text-xs text-slate-600">{s('agent.memory.replacing')}{replacing.text}</p>}
          <div className="flex flex-wrap gap-2">
            <Button disabled={busy || pending || loading || !!failure} onClick={() => void act(() => decideAgentMemory(session.session_id, proposal.proposal_id, 'accept', replacing))}>
              {s(replacing ? 'agent.memory.replace' : 'agent.memory.accept')}</Button>
            <Button disabled={busy || pending || loading || !!failure} onClick={() => void act(() => decideAgentMemory(session.session_id, proposal.proposal_id, 'reject'))}>{s('agent.memory.reject')}</Button>
          </div>
        </div>
      })}
      {items.map(item => <div key={item.memory_id} className="py-2">
        <p className="whitespace-pre-wrap break-words">{item.text}</p>
        <p className="text-xs text-slate-600">{s(used.some(source => source.memory_id === item.memory_id) ? 'agent.memory.used' : 'agent.memory.available')}</p>
        <p className="break-words text-xs text-slate-600">{s('agent.memory.confirmedAt', { date: item.accepted_at })}</p>
        <details className="text-xs text-slate-600"><summary className="flex min-h-10 cursor-pointer items-center">{s('agent.memory.sourceDetails')}</summary><p className="break-all">{item.source_session_id} / {item.source_message_id || item.memory_id}</p></details>
        <Button disabled={busy || pending || loading || !!failure} onClick={() => void act(() => revokeAgentMemory(session.session_id, item, crypto.randomUUID()))}>{s('agent.memory.revoke')}</Button>
      </div>)}
    </div>
    {failure && <p role="alert" className="text-sm text-rose-700">{failure}<Button disabled={pending || busy || loading} onClick={() => void act()}>{s('agent.memory.retry')}</Button></p>}
    {loading && <p role="status" className="text-xs text-slate-600">{s('agent.loading')}</p>}
    {pending && <p role="status" className="text-xs text-slate-600">{s('agent.memory.saving')}</p>}
  </details>
}
