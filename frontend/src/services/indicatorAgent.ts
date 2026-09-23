import { agentRequest as request } from './agentClient'
import type { AgentPreview } from './agent'
import type { IndicatorDraft } from './customIndicators'

export type AgentCommitPreview = {
  session_id: string
  session_revision: number
  confirmation_id: string
  definition_hash: string
  draft_revision: number
  definition: IndicatorDraft
  preview_status: 'valid'
  impact: { action: 'create'; name: string; context_kind: 'single_product'; target: { product_id: string; name?: string } | null; name_conflict_indicator_id: string | null }
}

export const fetchAgentPreview = (id: string, previewId: string, historical = false) => request<AgentPreview>(`/api/agent/sessions/${encodeURIComponent(id)}/previews/${encodeURIComponent(previewId)}${historical ? '?historical=true' : ''}`)

export const previewAgentCommit = (sessionId: string, payload: Record<string, unknown>) => request<AgentCommitPreview>(`/api/agent/sessions/${encodeURIComponent(sessionId)}/commit-preview`, {
  method: 'POST',
  body: JSON.stringify(payload),
})

export const commitAgentDraft = (sessionId: string, payload: Record<string, unknown>) => request<Record<string, unknown>>(`/api/agent/sessions/${encodeURIComponent(sessionId)}/commit`, {
  method: 'POST',
  body: JSON.stringify(payload),
})
