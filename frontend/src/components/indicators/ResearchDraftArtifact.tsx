import { useEffect, useRef, useState } from 'react'
import { Badge, Button } from '../ui'
import { ActionFeedback, type Feedback } from '../ActionControls'
import DefinitionDetails from './DefinitionDetails'
import type { IndicatorDraft } from '../../services/customIndicators'
import type { IndicatorPreview, IndicatorPreviewReference } from '../../services/researchContracts'
import type { ArtifactBinding, PortableAgentElement } from '../../integrations/portable-agent/contract'
import { authoringUrl, registerContext, researchRequest, ResearchRequestError, type ContextCapture } from '../../integrations/portable-agent/client'
import { useI18n } from '../../i18n/runtime'

type Draft = { definition: IndicatorDraft; definition_hash: string; draft_revision: number; valid: boolean; stale?: boolean }
type Saved = { indicator_id: string; indicator_revision: number; definition_hash: string; status: string }
type Authoring = { id: string; revision: number; current_revision: number; draft: Draft | null; current_draft: Omit<Draft, 'definition'>; preview?: IndicatorPreviewReference; saved: Saved[] }
export type PublicationAttempt = { body: Record<string, unknown>; receipt?: Saved }
const canonical = (value: unknown): string => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item)
export type DraftActions = {
  capture: () => ContextCapture
  onApplyDraft?: (definition: Record<string, unknown>) => void
  onCommitted?: () => void
  onPreview?: (preview: IndicatorPreview | null) => void
  onViewPreview?: (preview: IndicatorPreview) => void | boolean
}

export default function ResearchDraftArtifact({ binding, agent, actions, attempts, notified }: {
  binding: ArtifactBinding; agent: PortableAgentElement; actions: DraftActions; attempts: Map<string, PublicationAttempt>; notified: Set<string>
}) {
  const { s } = useI18n()
  const aid = String(binding.artifact.data.authoring_id)
  const revision = Number(binding.artifact.data.revision)
  const [state, setState] = useState<Authoring | null>(null)
  const [preview, setPreview] = useState<IndicatorPreview | null>(null)
  const [feedback, setFeedback] = useState<Feedback>()
  const [busy, setBusy] = useState(false)
  const [retry, setRetry] = useState(0)
  const latest = useRef({ binding, actions }); latest.current = { binding, actions }
  const alive = useRef(true)
  const adopted = useRef('')
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const notify = (key: string) => { if (!notified.has(key)) { notified.add(key); latest.current.actions.onCommitted?.() } }
  const key = `${aid}:${state?.draft?.definition_hash || ''}`
  const saved = state?.saved.find(value => value.definition_hash === state.draft?.definition_hash)
  const canSave = binding.isCurrent && state?.current_draft?.valid && state.current_draft.definition_hash === state.draft?.definition_hash && state.current_draft.draft_revision === state.draft?.draft_revision
  const blocked = busy || !!binding.busy

  useEffect(() => {
    let live = true
    void researchRequest<Authoring>(`${authoringUrl(aid)}?revision=${revision}`).then(async value => {
      if (!live) return
      setState(value)
      const receipt = value.saved.find(item => item.definition_hash === value.draft?.definition_hash)
      if (receipt) { notify(`${aid}:${receipt.definition_hash}`); setFeedback({ text: s('agent.saved'), saved: true }) }
      const previewId = String(binding.artifact.data.preview_id || '')
      if (previewId) {
        const result = await researchRequest<IndicatorPreview>(`${authoringUrl(aid)}/previews/${encodeURIComponent(previewId)}`)
        if (live) setPreview(result)
      }
    }).catch(error => { if (live) setFeedback({ text: error.message, error: true }) })
    return () => { live = false }
  }, [aid, revision, retry])

  useEffect(() => {
    if (!preview || adopted.current === preview.preview_id || !binding.isCurrent || preview.run_id !== binding.runId || !actions.onPreview) return
    let live = true
    const originalRevision = binding.bindingRevision
    void agent.adoptContext({ receiptRef: binding.artifact.id, runId: binding.runId, expectedContextRevision: originalRevision }).then(accepted => {
      if (accepted && live) { adopted.current = preview.preview_id; latest.current.actions.onPreview?.(preview) }
    }).catch(error => { if (live) setFeedback({ text: error.message, error: true }) })
    return () => { live = false }
  }, [preview?.preview_id, binding.isCurrent, binding.runId, binding.bindingRevision])

  async function save() {
    if (!state?.draft || !canSave || blocked || saved) return
    let release = () => {}
    try {
      release = agent.beginHostAction(); setBusy(true)
      let attempt = attempts.get(key)
      if (!attempt) {
        const current = await researchRequest<Authoring>(authoringUrl(aid))
        if (!alive.current || current.draft?.definition_hash !== state.draft.definition_hash) throw new Error(s('agent.confirmationMissing'))
        const capture = latest.current.actions.capture()
        const grant = await registerContext({ ...capture, authoring_id: aid })
        const frozen = await researchRequest<{ id: string; definition: IndicatorDraft; definition_hash: string; impact: {
          action: string; name: string; same_name_exists: boolean; calculation: Record<string, unknown>; target?: { product_id: string }
        } }>(`${authoringUrl(aid)}/confirmations`, { method: 'POST', body: JSON.stringify({ context_ref: grant.ref, expected_revision: current.current_revision, definition_hash: state.draft.definition_hash }) })
        if (!alive.current) return
        if (!frozen.id || frozen.impact?.action !== 'create' || frozen.definition_hash !== state.draft.definition_hash || canonical(frozen.definition) !== canonical(state.draft.definition)) throw new Error(s('agent.confirmationMissing'))
        // A memory proposal and a business save remain independent human decisions.
        void agent.proposeMemory(frozen.id).catch(error => { if (alive.current) setFeedback({ text: error.message, error: true }) })
        const definition = frozen.definition
        const formula = definition.result_kind === 'time_series' ? (definition.series_outputs || []).map(output => `${output.label || output.id} = ${output.expression}`).join('\n') : definition.expression
        if (!window.confirm(s('agent.saveImpact', { name: frozen.impact.name, formula,
          target: frozen.impact.target?.product_id || s('agent.saveWithoutTarget'), period: String(frozen.impact.calculation.period || s('agent.unrestricted')),
          asOf: String(frozen.impact.calculation.as_of || s('agent.unrestricted')), impact: frozen.impact.same_name_exists ? s('agent.saveNameConflict') : '',
        }))) return
        attempt = { body: { context_ref: grant.ref, expected_revision: current.current_revision, definition_hash: frozen.definition_hash,
          confirmation_id: frozen.id, request_id: crypto.randomUUID(), confirmed: true } }
        attempts.set(key, attempt)
      }
      const receipt = await researchRequest<Saved>(`${authoringUrl(aid)}/commits`, { method: 'POST', body: JSON.stringify(attempt.body) })
      attempt.receipt = receipt
      if (alive.current) { setState(value => value ? { ...value, saved: [...value.saved, receipt] } : value); setFeedback({ text: s('agent.saved'), saved: true }); notify(key) }
    } catch (error) {
      if (error instanceof ResearchRequestError && error.code === 'CONFIRMATION_STALE') attempts.delete(key)
      if (alive.current) setFeedback({ text: error instanceof Error ? error.message : s('agent.commitFailed'), error: true })
    } finally { if (alive.current) setBusy(false); release() }
  }

  if (!state) return <div role="status" className="py-2 text-sm text-slate-600">{feedback?.text || s('agent.loadingPreview')}<Button onClick={() => setRetry(value => value + 1)}>{s('agent.reloadPreview')}</Button></div>
  return <div className="text-left font-sans text-sm text-slate-900">
    {state.draft?.valid && (!state.draft.stale || binding.artifact.data.historical === true) && <section aria-label={s('agent.draftRegion')} className="min-w-0 rounded-xl border border-slate-200 bg-white p-3">
      <div className="flex items-start justify-between gap-2"><h3 className="text-base font-semibold">{state.draft.definition.name}</h3><Badge tone={binding.artifact.data.historical ? 'neutral' : 'success'}>{s(binding.artifact.data.historical ? 'agent.historicalDraft' : 'agent.validated')}</Badge></div>
      <p className="mt-1 text-xs text-slate-600">{saved ? s('agent.saved') : canSave ? s('agent.unsavedDraft') : s('agent.historicalDraft')}</p>
      <DefinitionDetails definition={state.draft.definition as unknown as Record<string, unknown>}>
        <Button disabled={blocked || !actions.onApplyDraft} onClick={() => { try { actions.onApplyDraft?.(structuredClone(state.draft!.definition) as unknown as Record<string, unknown>); setFeedback({ text: s('agent.draftApplied') }) } catch (error) { setFeedback({ text: String(error), error: true }) } }}>{s('agent.applyShort')}</Button>
        {canSave && <Button tone="primary" disabled={blocked || !!saved} onClick={() => void save()}>{saved ? s('agent.saved') : busy ? s('agent.saving') : s('agent.saveShort')}</Button>}
      </DefinitionDetails>
    </section>}
    {preview && <section aria-label={s('agent.previewRegion')} className="mt-2 rounded-xl border border-slate-200 bg-white p-3">
      <p className="text-xs text-slate-600">{s('agent.previewConditions', { product: preview.target.name || preview.target.product_id, period: preview.period, date: preview.as_of || s('agent.unrestricted') })}</p>
      <Button disabled={blocked || !actions.onViewPreview} onClick={() => { if (actions.onViewPreview?.(preview) !== false) agent.close() }}>{s('agent.viewPreview')}</Button>
    </section>}
    <ActionFeedback value={feedback} />
  </div>
}
