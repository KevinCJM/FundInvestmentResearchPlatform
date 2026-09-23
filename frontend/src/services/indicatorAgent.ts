import { agentRequest as request } from './agentClient'
import type { AgentPreview } from './agent'

export const fetchAgentPreview = (id: string, previewId: string, historical = false) => request<AgentPreview>(`/api/agent/sessions/${encodeURIComponent(id)}/previews/${encodeURIComponent(previewId)}${historical ? '?historical=true' : ''}`)

export const previewAgentCommit = (sessionId: string, payload: Record<string, unknown>) => request<Record<string, unknown>>(`/api/agent/sessions/${encodeURIComponent(sessionId)}/commit-preview`, {
  method: 'POST',
  body: JSON.stringify(payload),
})

export const commitAgentDraft = (sessionId: string, payload: Record<string, unknown>) => request<Record<string, unknown>>(`/api/agent/sessions/${encodeURIComponent(sessionId)}/commit`, {
  method: 'POST',
  body: JSON.stringify(payload),
})
