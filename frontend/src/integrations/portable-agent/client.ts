import { pitOverrideHeaders } from '../../services/pitOverride'
import type { PageEvidenceSnapshot, ResearchPageContext } from '../../services/researchContracts'

export type ContextCapture = { page_context: ResearchPageContext; page_snapshot?: PageEvidenceSnapshot | null; authoring_id?: string; capability_id?: string; intent_parameters?: Record<string, unknown> }
export type ContextGrant = { ref: string; hash: string; scope_key: string; workspace: string; authoring_id?: string | null; [key: string]: unknown }
export type AgentRelease = { source_commit: string; protocol_major: number; widget_version: string; manifest_sha256: string }
export type Bootstrap = { endpoint: string; module_url: string; app: string; principal_id: string; protocol_major: number; context: ContextGrant; token: string;
  required_capabilities: string[]; expected_release: AgentRelease | null }

export class ResearchRequestError extends Error {
  constructor(message: string, public code?: string, public status?: number) { super(message) }
}

export async function researchRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { credentials: 'same-origin', ...init, headers: {
    'Content-Type': 'application/json', ...pitOverrideHeaders(), ...init.headers,
  } })
  const result = await response.json().catch(() => null)
  if (!response.ok) {
    const error = result?.error || result?.detail
    throw new ResearchRequestError(error?.message || `请求失败（HTTP ${response.status}）`, error?.code, response.status)
  }
  if (!result || typeof result !== 'object') throw new ResearchRequestError('服务返回了无效响应。')
  return result as T
}

export async function registerContext(capture: ContextCapture): Promise<ContextGrant> {
  try {
    return await researchRequest<ContextGrant>('/api/integrations/portable-agent/contexts', { method: 'POST', body: JSON.stringify(capture) })
  } catch (error) {
    if (!(error instanceof ResearchRequestError) || error.status !== 401) throw error
    // Only an explicitly enabled loopback deployment can establish this local owner session.
    await researchRequest('/api/integrations/portable-agent/local-session', { method: 'POST', headers: { 'X-Portable-Local': '1' } })
    return researchRequest<ContextGrant>('/api/integrations/portable-agent/contexts', { method: 'POST', body: JSON.stringify(capture) })
  }
}

export const bootstrap = (context: ContextGrant) => researchRequest<Bootstrap>('/api/integrations/portable-agent/bootstrap', {
  method: 'POST', body: JSON.stringify({ context_ref: context.ref }),
})

/** Only business conditions, never message/session state. */
export function captureIdentity(capture: ContextCapture) {
  const { page, page_instance_id, view_state, calculation } = capture.page_context
  return { page, page_instance_id, view_state, calculation }
}

export const authoringUrl = (id: string) => `/api/custom-indicators/authorings/${encodeURIComponent(id)}`
