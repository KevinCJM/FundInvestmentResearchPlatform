import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { AgentEvent, AgentPageContext, AgentPreview, AgentRun } from '../../services/agent'
import type { IndicatorDraft } from '../../services/customIndicators'
import { agentContextKey } from '../../services/agentContext'
import { fetchAgentPreview } from '../../services/indicatorAgent'
import type { AgentConversationState } from './useAgentConversation'

export function adoptIndicatorPreviewContext(frozen: AgentPageContext, run: AgentRun, artifacts: AgentEvent['artifacts']) {
  const reference = artifacts?.preview
  // Only a receipt owned by this run may authorize programmatic control updates.
  if (frozen.calculation.context_kind !== 'single_product' || !reference?.preview_id || reference.run_id !== run.run_id) return null
  return { ...frozen, calculation: { ...frozen.calculation,
    targets: [{ kind: reference.target.kind, product_id: reference.target.product_id }],
    period: reference.period, as_of: reference.as_of || null,
  } }
}

export const emptyIndicatorPreview = { preview: null as AgentPreview | null, previewError: '', loadingPreview: false }
export type IndicatorPreviewState = typeof emptyIndicatorPreview

export function useIndicatorPreview(chat: AgentConversationState, pageContext: AgentPageContext): IndicatorPreviewState {
  const [preview, setPreview] = useState<AgentPreview | null>(null)
  const [previewError, setPreviewError] = useState('')
  const [loadingPreview, setLoadingPreview] = useState(false)
  const contextRef = useRef(pageContext)
  contextRef.current = pageContext
  const { previewReference, session, matchesContext, refreshRevision } = chat
  const previewId = previewReference?.preview_id
  const previewMatchesDraft = !!previewReference && previewReference.definition_hash === chat.draft?.definition_hash
  const hasLegacyPreview = !!previewReference && !previewId
  const currentKey = agentContextKey(pageContext)
  useEffect(() => {
    let active = true
    if (preview && preview.preview_id === previewId && previewMatchesDraft && matchesContext(pageContext)) return
    setPreview(null); setPreviewError(''); setLoadingPreview(false)
    if (hasLegacyPreview) { setPreviewError('这次旧试算未保存完整结果，请让助手重新试算。'); return }
    if (!previewId || !session?.session_id || !previewMatchesDraft || !matchesContext(pageContext)) return
    setLoadingPreview(true)
    void fetchAgentPreview(session.session_id, previewId).then(value => {
      if (active && matchesContext(contextRef.current)) setPreview(value)
    }).catch(reason => {
      if (active) setPreviewError(reason instanceof Error ? reason.message : '试算结果加载失败，请重新连接或重试。')
    }).finally(() => { if (active) setLoadingPreview(false) })
    return () => { active = false }
  }, [previewId, session?.session_id, previewMatchesDraft, hasLegacyPreview, currentKey, refreshRevision, matchesContext])
  return useMemo(() => ({ preview, previewError, loadingPreview }), [preview, previewError, loadingPreview])
}

/** Indicator-workbench ownership: clearing a result must not discard its active definition. */
export function useAdoptedIndicatorPreview(draft: IndicatorDraft, selectedId: string | null, onAdopt: (value: AgentPreview) => void) {
  const [preview, setPreview] = useState<AgentPreview | null>(null)
  const [definition, setDefinition] = useState<IndicatorDraft | null>(null)
  const adoptedId = useRef('')
  const adoptRef = useRef(onAdopt)
  adoptRef.current = onAdopt
  const receive = useCallback((value: AgentPreview | null, explicit = false) => {
    if (!explicit && value && adoptedId.current === value.preview_id) return
    setPreview(value)
    if (!value) return
    adoptedId.current = value.preview_id
    setDefinition(value.definition)
    adoptRef.current(value)
  }, [])
  useEffect(() => { setDefinition(null); setPreview(null) }, [draft, selectedId])
  return { preview, definition, setPreview, receive }
}
