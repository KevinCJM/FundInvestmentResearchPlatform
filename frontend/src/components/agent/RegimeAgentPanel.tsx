import { useEffect, useRef, useState } from 'react'
import AgentPanel, { type AgentArtifactContext } from './AgentPanel'
import { ActionFeedback, type Feedback } from './AgentControls'
import { Button } from '../ui'
import { useI18n } from '../../i18n/runtime'
import { useResearchContextIdentity, useResearchDay } from '../../app/ResearchContext'
import { inferRegimeGraph, definitionForRequest, type RegimeGraphDefinition, type RegimeMode } from '../../services/regimeGraph'
import { mergeRegimeAgentDraft, regimeAgentSnapshot } from '../../services/regimeAgent'
import { newPageSnapshotId } from '../../services/agentPageEvidence'
import { agentContextKey } from '../../services/agentContext'
import type { AgentPageContext } from '../../services/agent'

type Props = {
  definition?: RegimeGraphDefinition
  mode: RegimeMode
  asOf?: string
  busy?: boolean
  selectedNodeId?: string
  onApply: (definition: RegimeGraphDefinition) => void
}

export default function RegimeAgentPanel(props: Props) {
  const { s } = useI18n()
  const day = useResearchDay()
  const researchIdentity = useResearchContextIdentity()
  const asOf = props.asOf ?? day ?? ''
  const signature = JSON.stringify([props.definition ? definitionForRequest(props.definition) : null, props.mode, asOf, researchIdentity, !!props.busy])
  const revision = useRef({ signature, value: 0, token: newPageSnapshotId() })
  if (revision.current.signature !== signature) revision.current = { signature, value: revision.current.value + 1, token: newPageSnapshotId() }
  const pageContext: AgentPageContext = { page: 'regime-workbench', page_instance_id: props.definition ? 'regime-editor' : 'regime-library',
    context_revision: revision.current.value, view_state: day === undefined ? 'unknown' : 'inherit',
    calculation: { context_kind: 'regime_graph', editor_token: revision.current.token, mode: props.mode, as_of: asOf || null } }
  return <AgentPanel pageContext={pageContext} busy={props.busy}
    conversationOptions={{ capturePageSnapshot: () => regimeAgentSnapshot(props.definition, props.mode, asOf, !!props.busy, props.selectedNodeId) }}
    hasArtifacts={event => event.artifacts?.draft?.artifact_kind === 'regime_graph'}
    renderWelcome={({ fillInput }) => <div className="space-y-3 pt-2">
      <p className="text-sm leading-6 text-slate-600">{s('agent.regime.help')}</p>
      <div className="flex flex-wrap gap-2">{['design', ...(props.definition ? ['explain', 'improve'] : [])].map(key =>
        <Button key={key} onClick={() => fillInput(s(`agent.regime.${key}Prompt`))}>{s(`agent.regime.${key}`)}</Button>)}</div>
    </div>}
    renderArtifacts={context => <RegimeDraft {...context} {...props} signature={signature} pageContext={pageContext} />} />
}

function RegimeDraft({ event, chat, busy, definition, mode, signature, pageContext, onApply }: Props & AgentArtifactContext & { signature: string; pageContext: AgentPageContext }) {
  const { s } = useI18n()
  const [feedback, setFeedback] = useState<Feedback>()
  const [applying, setApplying] = useState(false)
  const [applied, setApplied] = useState(false)
  const current = useRef(signature)
  current.current = signature
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const draft = event.artifacts?.draft
  if (draft?.artifact_kind !== 'regime_graph') return null
  const outdated = draft.stale || chat.draft?.stale || chat.draft?.definition_hash !== draft.definition_hash
    || (!chat.session?.page_context || agentContextKey(chat.session.page_context) !== agentContextKey(pageContext))
  const nodes = (draft.definition.graph as RegimeGraphDefinition['graph'])?.nodes || []
  const apply = async () => {
    const started = current.current
    setApplying(true); setFeedback(undefined)
    try {
      const next = mergeRegimeAgentDraft(definition, draft.definition)
      const result = await inferRegimeGraph(next, undefined, mode)
      if (!mounted.current) return
      if (current.current !== started) throw new Error(s('agent.regime.changed'))
      if (!result.valid || (mode === 'realtime' && !result.temporal_capability?.realtime_supported)) {
        throw new Error(result.errors.map(item => item.message).join('；') || s('agent.regime.invalid'))
      }
      onApply(next); setApplied(true); setFeedback({ text: s('agent.regime.applied') })
    } catch (error) {
      if (mounted.current) setFeedback({ text: error instanceof Error ? error.message : s('agent.regime.invalid'), error: true })
    } finally { if (mounted.current) setApplying(false) }
  }
  return <section className="mt-3 space-y-3 rounded-xl border border-slate-200 bg-white p-3" aria-label={s('agent.regime.proposal')}>
    <div><p className="font-semibold text-slate-900">{String(draft.definition.name || '')}</p>
      <p className="mt-1 text-xs text-slate-600">{s('agent.regime.summary', { count: nodes.length })}</p></div>
    <details><summary className="min-h-10 cursor-pointer py-2 text-sm font-semibold text-slate-700">{s('agent.regime.steps')}</summary>
      <ol className="space-y-2 text-xs leading-5 text-slate-600">{nodes.map(node => <li key={node.id} className="break-words">
        <span className="font-semibold">{node.label || node.id}</span> · {node.type}
        {Object.keys(node.parameters || {}).length > 0 && <p>{Object.entries(node.parameters || {}).map(([key, value]) => `${key}: ${String(value)}`).join(' · ')}</p>}
      </li>)}</ol>
    </details>
    {!draft.valid && <p role="alert" className="text-sm text-rose-700">{draft.diagnostics?.map(item => item.message).filter(Boolean).join('；') || s('agent.regime.invalid')}</p>}
    {outdated && !applied && <p className="text-xs text-amber-800">{s('agent.regime.changed')}</p>}
    <Button disabled={busy || applying || applied || outdated || !draft.valid} onClick={() => void apply()}>{s(applying ? 'agent.regime.checking' : definition ? 'agent.regime.apply' : 'agent.regime.open')}</Button>
    <ActionFeedback value={feedback} />
  </section>
}
