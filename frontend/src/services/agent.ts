import { agentRequest as request } from './agentClient'
export { AgentApiError } from './agentClient'

export type AgentPage = 'indicator-studio' | 'product-detail' | 'product-research' | 'product-compare' | 'holding-diagnosis' | 'evaluation-plan'

import type { EvaluationResult, IndicatorDraft, TimeSeriesIndicatorResult } from './customIndicators'

export interface AgentPreviewReference {
  preview_id?: string
  run_id?: string
  definition_hash: string
  target: { kind: string; product_id: string; name?: string }
  period: string
  as_of?: string | null
  result_kind: 'scalar' | 'time_series'
  created_at?: string
}

export type AgentPreview = AgentPreviewReference & {
  preview_id: string
  session_id: string
  definition: IndicatorDraft
  expires_at: string
  /** Server-frozen provenance of the run that produced this preview, when available. */
  context_hash?: string
  data_generation?: string | null
  effective_context?: { as_of?: string | null; run_mode?: string | null; data_release_id?: string | null } | null
} & ({ result_kind: 'scalar'; result: { results: EvaluationResult[] } }
  | { result_kind: 'time_series'; result: { results: TimeSeriesIndicatorResult[] } })

/**
 * Versioned, curated copy of what the page displayed when the user sent a message.
 * Transport-only: the server binds it to that run and never treats it as authorization.
 */
export interface PageEvidenceSnapshot {
  version: 1
  snapshot_id: string
  captured_at?: string
  page: AgentPage
  sections: Record<string, unknown>
}

export interface AgentPageContext {
  page: AgentPage
  page_instance_id: string
  context_revision: number
  view_state: 'unknown' | 'inherit' | 'explicit' | 'off'
  calculation: Record<string, unknown>
}

export interface AgentMeta {
  configured: boolean
  model?: string | null
  provider?: string | null
  catalog_version?: string
  scopes?: Array<{ id: string; label: string; pages: AgentPage[]; tools: string[] }>
  limits?: Record<string, number | null>
}

export interface AgentDraft {
  draft_revision: number
  valid: boolean
  stale?: boolean
  definition_hash?: string | null
  definition: Record<string, unknown>
  diagnostics?: Array<{ code?: string; message?: string }>
  compile_token?: string | null
}

export interface AgentEvent {
  seq?: number
  type?: string
  run_id?: string
  message_id?: string
  edit_of?: string
  data?: Record<string, unknown>
  id?: string
  speaker?: 'user' | 'assistant' | 'tool' | 'system'
  text?: string
  summary?: string
  created_at?: string
  tool?: string
  artifacts?: { draft?: AgentDraft | null; preview?: AgentPreviewReference | null }
}

export interface AgentSession {
  memory_proposals?: AgentMemoryProposal[]
  memory_sources?: AgentMemorySource[]
  task_state?: Record<string, unknown>
  session_id: string
  session_revision: number
  scope?: string
  state?: string
  events?: AgentEvent[]
  draft?: AgentDraft | null
  saved_commit?: { definition_hash: string; indicator_id: string; revision: number } | null
  preview?: AgentPreviewReference | null
  pending_decision?: Record<string, unknown> | null
  page_context?: AgentPageContext
  next_event_seq?: number
  active_run?: AgentRun | null
  execution_blocked_by?: string | null
  messages?: AgentEvent[]
  older_message_cursor?: number | null
  legacy_history_incomplete?: boolean
}

export interface AgentTurnResponse {
  session_id: string
  session_revision: number
  assistant_text?: string
  message?: string
  text?: string
  draft?: AgentDraft | null
  events?: AgentEvent[]
  pending_decision?: Record<string, unknown> | null
  usage?: Record<string, number>
  confirmation_id?: string | null
  preview?: AgentPreviewReference | null
  reply?: { role?: string; text?: string }
  tool_trace?: Array<Record<string, unknown>>
  stop_reason?: string | null
  artifacts?: AgentEvent['artifacts']
}

export type RunStatus = 'queued' | 'running' | 'stopping' | 'paused' | 'completed' | 'cancelled' | 'failed' | 'interrupted'
export interface AgentRun {
  run_id: string; session_id: string; message_id: string; session_revision: number
  status: RunStatus; phase: string; stop_reason?: string | null
  run_revision: number; response?: AgentTurnResponse | null
  execution_blocked_by?: string | null
  error?: { code: string; message: string }
  artifacts?: AgentEvent['artifacts']
}
export interface AgentEventPage { items: AgentEvent[]; has_more: boolean; last_seq: number; next_event_seq: number }
const sessionPath = (id: string) => `/api/agent/sessions/${encodeURIComponent(id)}`
export const fetchAgentSession = (id: string) => request<AgentSession>(sessionPath(id))
export const fetchAgentEvents = (id: string, after: number) => request<AgentEventPage>(`${sessionPath(id)}/events?after_seq=${after}&limit=200`)
export const fetchEarlierMessages = (id: string, before: number) => request<{ items: AgentEvent[]; older_cursor: number | null }>(`${sessionPath(id)}/events?kind=messages&before_seq=${before}&limit=200`)
export const agentEventUrl = (id: string, after: number) => `${sessionPath(id)}/events?after_seq=${after}&stream=1`
export const fetchAgentRun = (id: string, runId: string) => request<AgentRun>(`${sessionPath(id)}/runs/${encodeURIComponent(runId)}`)
export const cancelAgentRun = (id: string, runId: string, requestId: string) => request<AgentRun>(`${sessionPath(id)}/runs/${encodeURIComponent(runId)}/cancel`, { method: 'POST', body: JSON.stringify({ request_id: requestId }) })
export const invalidateAgentContext = (id: string, runId: string, requestId: string, pageContext: AgentPageContext) => request<AgentRun>(`${sessionPath(id)}/runs/${encodeURIComponent(runId)}/invalidate-context`, { method: 'POST', body: JSON.stringify({ request_id: requestId, page_context: pageContext }) })

export const fetchAgentMeta = () => request<AgentMeta>('/api/agent/meta')

export const createAgentSession = (page_context: AgentPageContext) => request<AgentSession>('/api/agent/sessions', {
  method: 'POST',
  body: JSON.stringify({ page_context }),
})

export const sendAgentMessage = (sessionId: string, payload: {
  message_id: string
  expected_session_revision: number
  text?: string
  page_context: AgentPageContext
  page_snapshot?: PageEvidenceSnapshot
  resume_from_run_id?: string
  edit_of_message_id?: string
}) => request<AgentRun>(`/api/agent/sessions/${encodeURIComponent(sessionId)}/messages?response_mode=async`, {
  method: 'POST',
  body: JSON.stringify(payload),
  headers: payload.page_context.view_state === 'off' ? { 'x-pit-off': '1' } : undefined,
})

export interface AgentMemoryProposal {
  proposal_id: string; status: 'pending' | 'accepted' | 'rejected'; summary: string
  source_message_id?: string; key?: string; object_id?: string
}
export interface AgentMemorySource {
  memory_id: string; version: number; scope: string; object_id: string; key: string; text: string
  source_session_id: string; source_message_id?: string; accepted_at: string
}
export const fetchAgentMemory = (id: string) => request<{ items: AgentMemorySource[]; legacy_unavailable?: boolean }>(`${sessionPath(id)}/memory`)
export const decideAgentMemory = (id: string, proposal: string, decision: 'accept' | 'reject', replace?: AgentMemorySource) =>
  request(`${sessionPath(id)}/memory`, { method: 'POST', body: JSON.stringify({ proposal_id: proposal, decision,
    ...(replace ? { replace_memory_id: replace.memory_id, expected_version: replace.version } : {}) }) })
export const revokeAgentMemory = (id: string, memory: AgentMemorySource, requestId: string) =>
  request(`${sessionPath(id)}/memory/revoke`, { method: 'POST', body: JSON.stringify({ request_id: requestId,
    memory_id: memory.memory_id, expected_version: memory.version }) })
