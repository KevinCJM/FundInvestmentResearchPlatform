import { agentRequest as request } from './agentClient'

export type ReasoningEffort = 'default' | 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'
export interface LlmSettings {
  configured: boolean
  provider: string
  model: string
  base_url: string
  api_key_masked: string | null
  timeout_seconds?: number
  context_window_tokens?: number
  reasoning_effort?: ReasoningEffort
  enabled?: boolean
  updated_at?: string | null
  profiles: LlmProfile[]
  active_profile_id: string | null
}

export interface LlmProfile {
  id: string; name: string; provider: string; model: string; base_url: string
  configured: boolean; api_key_masked: string | null
  reasoning_effort?: ReasoningEffort; timeout_seconds?: number; context_window_tokens?: number
}


export const fetchLlmSettings = () => request<LlmSettings>('/api/settings/llm')

export const saveLlmProfile = (id: string | null, payload: { name: string; provider: string; model: string; base_url: string; api_key?: string; reasoning_effort?: ReasoningEffort; timeout_seconds?: number; context_window_tokens?: number }) => request<LlmSettings>(id ? `/api/settings/llm/profiles/${encodeURIComponent(id)}` : '/api/settings/llm/profiles', {
  method: id ? 'PUT' : 'POST',
  body: JSON.stringify(payload),
})
export const activateLlmProfile = (profile_id: string | null) => request<LlmSettings>('/api/settings/llm/active', { method: 'PUT', body: JSON.stringify({ profile_id }) })
