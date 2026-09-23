export class AgentApiError extends Error {
  constructor(message: string, public code?: string, public status?: number, public runId?: string) { super(message) }
}
export async function agentRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  if (!headers.has('Content-Type')) headers.set('Content-Type', 'application/json')
  const response = await fetch(path, { ...init, headers })
  const body = await response.json().catch(() => null)
  if (!response.ok) {
    const detail = body?.detail
    throw new AgentApiError(typeof detail === 'string' ? detail : detail?.message || `请求失败（HTTP ${response.status}）`, detail?.code, response.status, detail?.run_id)
  }
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    throw new AgentApiError('服务返回了无效响应，请重试。', 'AGENT_RESPONSE_INVALID', response.status)
  }
  return body as T
}
