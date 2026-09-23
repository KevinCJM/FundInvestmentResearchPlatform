import { useEffect, useId, useRef, useState, type ReactNode } from 'react'
import { CheckIcon, ClipboardDocumentIcon } from '@heroicons/react/24/outline'
import { Button, Badge } from '../ui'
import { IconButton, ActionFeedback, type Feedback } from './AgentControls'
import AgentPanel, { type AgentArtifactContext } from './AgentPanel'
import type { AgentDraft, AgentEvent, AgentPageContext, AgentPreview, AgentPreviewReference, PageEvidenceSnapshot } from '../../services/agent'
import { agentSessionStorageKey } from '../../services/agentContext'
import { commitAgentDraft, fetchAgentPreview, previewAgentCommit } from '../../services/indicatorAgent'
import { AgentApiError } from '../../services/agentClient'
import { messageId, type AgentConversationState, type AgentConversationOptions } from './useAgentConversation'
import type { SeriesOutputDefinition, SeriesParameterDefinition } from '../../services/customIndicators'
import { useI18n } from '../../i18n/runtime'

type Props = {
  pageContext: AgentPageContext
  draft: Record<string, unknown>
  /** Host-owned freeze of the displayed page; captured per outgoing message. */
  capturePageSnapshot?: () => PageEvidenceSnapshot | null
  onApplyDraft?: (draft: Record<string, unknown>) => void
  onCommitted?: () => void
  onPreview?: (preview: AgentPreview | null) => void
  onViewPreview?: (preview: AgentPreview) => void | boolean
}

type CommitAttempt = { request: Record<string, unknown>; result?: Record<string, unknown> }

function turnArtifacts(event: AgentEvent, chat: AgentConversationState) {
  const isCurrent = !!event.run_id && event.run_id === chat.run?.run_id
  const candidate = event.artifacts?.draft
  return { draft: candidate?.valid === true && !candidate.stale && !(isCurrent && chat.draft?.stale) ? candidate : null, preview: event.artifacts?.preview }
}

const hasArtifacts = (event: AgentEvent, chat: AgentConversationState) => {
  const { draft, preview } = turnArtifacts(event, chat)
  return !!draft || !!preview
}

const conversationOptions: AgentConversationOptions = {
  loadPreview: fetchAgentPreview,
  adoptPreviewContext: (frozen, reference) => frozen.calculation.context_kind === 'single_product' ? {
    ...frozen, calculation: { ...frozen.calculation,
      targets: [{ kind: reference.target.kind, product_id: reference.target.product_id }],
      period: reference.period, as_of: reference.as_of || null,
    },
  } : null,
}

function IndicatorSessionStatus({ chat, onPreview, onCommitted }: { chat: AgentConversationState; onPreview?: Props['onPreview']; onCommitted?: Props['onCommitted'] }) {
  const { s } = useI18n()
  useEffect(() => { onPreview?.(chat.preview) }, [chat.preview, onPreview])
  useEffect(() => { if (chat.session?.saved_commit) onCommitted?.() }, [chat.session?.saved_commit, onCommitted])
  return chat.previewError && !chat.messages.some(event => event.run_id === chat.run?.run_id && event.artifacts?.preview)
    ? <div role="alert" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900"><p>{chat.previewError}</p><Button onClick={chat.refresh}>{s('agent.reloadPreview')}</Button></div> : null
}

export default function IndicatorAgentPanel(props: Props) {
  return <IndicatorIntegration key={agentSessionStorageKey(props.pageContext)} {...props} />
}

function IndicatorIntegration(props: Props) {
  const { s } = useI18n()
  const [operation, setOperation] = useState('')
  // Own attempts above the floating panel: closing it cannot turn a retry into a new save.
  const commits = useRef(new Map<string, CommitAttempt>())
  const notifiedCommits = useRef(new Set<string>())
  const notifyCommitted = (key: string) => {
    if (!props.onCommitted || notifiedCommits.current.has(key)) return
    notifiedCommits.current.add(key)
    props.onCommitted()
  }
  const editorDraft = props.draft
  const editorFormula = editorDraft.result_kind === 'time_series' && Array.isArray(editorDraft.series_outputs)
    ? (editorDraft.series_outputs as SeriesOutputDefinition[]).map(output => `${output.label || output.id} = ${output.expression}`).join('\n') : String(editorDraft.expression || '')
  const explainPrompt = editorFormula && editorFormula.length < 3800 ? s('agent.explainPrompt', { formula: editorFormula }) : s('agent.explainEmptyPrompt')
  const options = props.capturePageSnapshot
    ? { ...conversationOptions, capturePageSnapshot: props.capturePageSnapshot }
    : conversationOptions
  return <AgentPanel pageContext={props.pageContext} busy={!!operation} conversationOptions={options}
    renderStatus={chat => <IndicatorSessionStatus chat={chat} onPreview={props.onPreview}
      onCommitted={() => notifyCommitted(`${chat.session?.session_id}:${chat.session?.saved_commit?.definition_hash}`)} />} hasArtifacts={hasArtifacts}
    renderWelcome={({ chat, fillInput }) => <><p className="mt-2 text-sm leading-6 text-slate-600">{s('agent.welcomeHelp')}</p>{props.pageContext.page === 'indicator-studio' && <div className="mt-4 flex flex-wrap gap-2">{[[s('agent.designIndicator'), s('agent.designPrompt')], [s('agent.explainFormula'), explainPrompt], [s('agent.findProduct'), chat.draft?.valid && !chat.draft.stale ? s('agent.previewPrompt') : s('agent.definePreviewPrompt')]].map(([label, prompt]) => <Button key={String(label)} onClick={() => fillInput(String(prompt))}>{label}</Button>)}</div>}</>}
    renderArtifacts={context => <IndicatorArtifacts {...props} {...context} commits={commits.current} operation={operation} setOperation={setOperation}
      onCommitted={() => notifyCommitted(`${context.chat.session?.session_id}:${context.chat.draft?.definition_hash}`)} />} />
}

function DraftFormula({ definition, children }: { definition: AgentDraft['definition']; children?: ReactNode }) {
  const { s } = useI18n()
  const [section, setSection] = useState<'formula' | 'parameters' | null>(null)
  const sectionId = useId()
  const [originalLines, setOriginalLines] = useState(false)
  const [copyFeedback, setCopyFeedback] = useState<Feedback>()
  const series = definition.result_kind === 'time_series' && Array.isArray(definition.series_outputs)
    ? definition.series_outputs as SeriesOutputDefinition[] : null
  const outputs = series || [{ id: 'scalar', label: '', expression: String(definition.expression || '') }]
  const parameters = Array.isArray(definition.parameter_schema) ? definition.parameter_schema as SeriesParameterDefinition[] : []
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(outputs.map(output => series ? `${output.label || output.id} = ${output.expression}` : output.expression).join('\n\n'))
      setCopyFeedback({ text: s('agent.formulaCopied') })
    } catch { setCopyFeedback({ text: s('agent.formulaCopyFailed'), error: true }) }
  }
  return <div className="mt-2">
    <div className="flex flex-wrap items-center gap-1">
      <button type="button" aria-label={s('agent.fullFormula', { count: outputs.length })} aria-expanded={section === 'formula'} aria-controls={`${sectionId}-formula`} onClick={() => setSection(value => value === 'formula' ? null : 'formula')} className="min-h-10 rounded-lg px-2 text-xs font-semibold text-accent-700 hover:bg-accent-50">{s('agent.formulaShort')}<span aria-hidden="true"> {section === 'formula' ? '▴' : '▾'}</span></button>
      {parameters.length > 0 && <button type="button" aria-label={s('agent.parameters', { count: parameters.length })} aria-expanded={section === 'parameters'} aria-controls={`${sectionId}-parameters`} onClick={() => setSection(value => value === 'parameters' ? null : 'parameters')} className="min-h-10 rounded-lg px-2 text-xs font-semibold text-slate-600 hover:bg-slate-100">{s('agent.parametersShort')}<span aria-hidden="true"> {section === 'parameters' ? '▴' : '▾'}</span></button>}
      <div className="ml-auto flex flex-wrap gap-1">{children}</div>
    </div>
    <div id={`${sectionId}-formula`} hidden={section !== 'formula'}>
    <div className="flex flex-wrap items-center justify-between gap-2">
      <label className="flex min-h-10 items-center gap-2 text-xs text-slate-600"><input type="checkbox" checked={originalLines} onChange={event => setOriginalLines(event.target.checked)} />{s('agent.originalLines')}</label>
      <IconButton label={s('agent.copyFormula')} hint={copyFeedback && !copyFeedback.error ? s('agent.formulaCopied') : s('agent.copyFormula')} onClick={() => void copy()} className={copyFeedback && !copyFeedback.error ? 'agent-copy-confirmed' : ''}>{copyFeedback && !copyFeedback.error ? <CheckIcon className="h-5 w-5" aria-hidden="true" /> : <ClipboardDocumentIcon className="h-5 w-5" aria-hidden="true" />}</IconButton>
    </div>
    {outputs.map(output => <pre key={output.id} tabIndex={0} aria-label={s('agent.formulaLabel', { name: output.label || output.id })} className={`mb-3 max-w-full overflow-x-auto rounded-lg bg-slate-950 p-3 text-xs leading-6 text-slate-200 focus-visible:ring-2 focus-visible:ring-accent-500 ${originalLines ? 'whitespace-pre' : 'whitespace-pre-wrap break-words [overflow-wrap:anywhere]'}`}>{output.expression ? (series ? `${output.label || output.id} = ${output.expression}` : output.expression) : s('agent.formulaMissing')}</pre>)}
    <ActionFeedback value={copyFeedback} />
    </div>
    <dl id={`${sectionId}-parameters`} hidden={section !== 'parameters'} className="space-y-2 pt-2 text-xs text-slate-600">{parameters.map(parameter => <div key={parameter.id}><dt className="font-semibold">{parameter.label || parameter.id}</dt><dd className="tabular-nums">{s('agent.parameterDetails', { defaultValue: parameter.default, minComparison: s(parameter.exclusive_minimum ? 'agent.greaterThan' : 'agent.atLeast'), minimum: parameter.minimum, maxComparison: s(parameter.exclusive_maximum ? 'agent.lessThan' : 'agent.atMost'), maximum: parameter.maximum, step: parameter.step })}</dd></div>)}</dl>
  </div>
}


function IndicatorArtifacts({ pageContext, onApplyDraft, onCommitted, onPreview, onViewPreview, event, eventKey, isCurrent, chat, busy, close, operation, setOperation, commits }: Props & AgentArtifactContext & { operation: string; setOperation: (value: string) => void; commits: Map<string, CommitAttempt> }) {
  const { s } = useI18n()
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  const [feedback, setFeedback] = useState<Record<string, Feedback>>({})
  const setActionFeedback = (key: string, value: Feedback) => setFeedback(previous => ({ ...previous, [key]: { ...previous[key], error: false, ...value } }))
  const session = chat.session
  const currentDraft = chat.draft?.valid === true && !chat.draft.stale ? chat.draft : null
  const { draft: turnDraft, preview } = turnArtifacts(event, chat)
  const canSave = isCurrent && turnDraft && currentDraft?.definition_hash === turnDraft.definition_hash && currentDraft?.draft_revision === turnDraft.draft_revision
  const draftKey = `${eventKey}:${turnDraft?.definition_hash}:${turnDraft?.draft_revision}`
  const commitKey = `${session?.session_id}:${draftKey}`
  const savedRevision = turnDraft?.definition_hash && session?.saved_commit?.definition_hash === turnDraft.definition_hash
    ? session.saved_commit.revision : null
  const saved = !!feedback[draftKey]?.saved || !!commits.get(commitKey)?.result || savedRevision !== null
  useEffect(() => {
    if (savedRevision !== null) setFeedback(previous => ({ ...previous,
      [draftKey]: { text: s('agent.savedRevision', { revision: String(savedRevision) }), saved: true } }))
  }, [draftKey, savedRevision, s])
  const previewKey = `${eventKey}:${preview?.preview_id}`
  const series = turnDraft?.definition.result_kind === 'time_series' && Array.isArray(turnDraft.definition.series_outputs) ? turnDraft.definition.series_outputs as SeriesOutputDefinition[] : []
  const result = chat.preview?.preview_id === preview?.preview_id ? chat.preview?.result.results[0] : undefined
  const applyDraft = (draft: AgentDraft, key: string) => {
    if (!onApplyDraft || busy) return
    try { onApplyDraft(draft.definition); setActionFeedback(key, { text: saved ? s('agent.draftAppliedSaved') : s('agent.draftApplied') }) }
    catch (reason) { setActionFeedback(key, { text: reason instanceof Error ? reason.message : s('agent.draftApplyFailed'), error: true }) }
  }
  const commit = async (draft: AgentDraft, key: string) => {
    if (!session || busy || !currentDraft || currentDraft.definition_hash !== draft.definition_hash || currentDraft.draft_revision !== draft.draft_revision || saved) return
    setOperation(key)
    try {
      let attempt = commits.get(commitKey)
      if (!attempt) {
        const target = (pageContext.calculation as { targets?: Array<{ kind: string; product_id: string }> }).targets?.[0]
        const preview = await previewAgentCommit(session.session_id, { draft_revision: draft.draft_revision, definition: draft.definition, ...(target ? { target } : {}), page_context: pageContext })
        if (!mounted.current) return
        const confirmationId = String(preview.confirmation_id || '')
        if (!confirmationId) throw new Error(s('agent.confirmationMissing'))
        attempt = { request: { request_id: messageId(), confirmation_id: confirmationId, definition_hash: preview.definition_hash || draft.definition_hash, draft_revision: draft.draft_revision, confirmed: true } }
        commits.set(commitKey, attempt)
      }
      const result = await commitAgentDraft(session.session_id, attempt.request)
      attempt.result = result
      if (!mounted.current) return
      setActionFeedback(key, { text: result.revision ? s('agent.savedRevision', { revision: String(result.revision) }) : s('agent.indicatorSaved'), saved: true })
      chat.refresh(); onCommitted?.()
    } catch (reason) {
      // A stale confirmation explicitly proves no write occurred. An uncertain response does not.
      if (reason instanceof AgentApiError && reason.code === 'AGENT_CONFIRMATION_STALE') commits.delete(commitKey)
      setActionFeedback(key, { text: reason instanceof Error ? reason.message : s('agent.commitFailed'), error: true })
    }
    finally { setOperation('') }
  }
  const viewPreview = async (reference: AgentPreviewReference, key: string) => {
    if (!session || !reference.preview_id || busy) return
    setOperation(key)
    try {
      const value = await fetchAgentPreview(session.session_id, reference.preview_id, true)
      if (!mounted.current) return
      if (onViewPreview) {
        if (onViewPreview(value) === false) throw new Error(s('agent.previewChanged'))
        close(false)
      } else { onPreview?.(value); close() }
    } catch (reason) { setActionFeedback(key, { text: reason instanceof Error ? reason.message : s('agent.previewReadFailed'), error: true }) }
    finally { setOperation('') }
  }
  return <>
              {turnDraft && <section aria-label={s('agent.draftRegion')} className="mt-3 min-w-0 rounded-xl border border-slate-200 p-3">
                <div className="flex flex-wrap items-start justify-between gap-2"><h3 className="text-base font-semibold text-slate-900">{String(turnDraft.definition.name || s('agent.draftTitle'))}</h3><Badge tone="success">{s('agent.validated')}</Badge></div>
                <p className="mt-1 text-xs text-slate-600">{series.length ? s('agent.seriesCount', { count: series.length }) : s('agent.scalar')} · {saved ? s('agent.saved') : canSave ? s('agent.unsavedDraft') : s('agent.historicalDraft')}</p>
                {series.length > 0 && <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1">{series.map(output => <p key={output.id} className="text-sm font-semibold text-slate-700">{output.label || output.id}<span className="ml-2 text-xs font-normal text-slate-600">{output.unit || ''}</span></p>)}</div>}
                <DraftFormula definition={turnDraft.definition}>
                  <Button className="px-2" aria-label={s('agent.applyDraft')} disabled={busy || !onApplyDraft} onClick={() => applyDraft(turnDraft, draftKey)}>{s('agent.applyShort')}</Button>
                  {canSave && <Button className="px-2" tone="primary" aria-label={operation === draftKey ? s('agent.saving') : saved ? s('agent.saved') : s('agent.confirmSave')} disabled={busy || saved} onClick={() => void commit(turnDraft, draftKey)}>{operation === draftKey ? s('agent.saving') : saved ? s('agent.saved') : s('agent.saveShort')}</Button>}
                </DraftFormula>
                {busy && <p className="mt-2 text-xs text-slate-600">{s('agent.draftBusy')}</p>}
                {!onApplyDraft && <p className="mt-2 text-xs text-slate-600">{s('agent.noEditor')}</p>}
                {!canSave && <p className="mt-2 text-xs text-slate-600">{s('agent.historicalDraftHelp')}</p>}
                <ActionFeedback value={feedback[draftKey]} />
              </section>}
              {preview && <section aria-label={s('agent.previewRegion')} className="mt-3 space-y-2 rounded-xl border border-slate-200 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-sm font-semibold text-slate-900">{s('agent.previewTitle')}</h3>{preview.preview_id && <Button className="px-2" aria-label={operation === previewKey ? s('agent.reading') : s('agent.viewPreview')} disabled={busy || (isCurrent && chat.loadingPreview)} onClick={() => void viewPreview(preview, previewKey)}>{operation === previewKey ? s('agent.reading') : s('agent.previewShort')}</Button>}</div><p className="text-xs text-slate-600">{s('agent.previewConditions', { product: preview.target.name ? s('agent.productWithCode', { name: preview.target.name, code: preview.target.product_id }) : preview.target.product_id, period: preview.period, date: preview.as_of || s('agent.unrestricted') })}</p>
                {result && <><p className="text-xs text-slate-600">{s('agent.calculationStatus', { status: ({ ok: s('agent.calculable'), warning: s('agent.calculationWarning'), unavailable: s('agent.unavailable'), error: s('agent.calculationFailed') })[result.status] || s('agent.resultDetails') })}</p>{Object.entries(result.parameters || {}).map(([key, value]) => <p key={key} className="text-xs tabular-nums text-slate-600">{s('agent.parameterValue', { label: chat.preview?.definition.parameter_schema?.find(parameter => parameter.id === key)?.label || key, value })}</p>)}</>}
                {!preview.preview_id && <p className="text-xs text-amber-900">{s('agent.legacyPreview')}</p>}
                {busy && <p className="text-xs text-slate-600">{s('agent.previewBusy')}</p>}
                <ActionFeedback value={feedback[previewKey]} />
                {isCurrent && chat.previewError && !feedback[previewKey]?.error && <div role="alert" className="text-xs text-amber-900"><p>{chat.previewError}</p><Button onClick={chat.refresh}>{s('agent.reloadPreview')}</Button></div>}
              </section>}
              {isCurrent && chat.loadingPreview && <p role="status" className="text-sm text-slate-600">{s('agent.loadingPreview')}</p>}
  </>
}
