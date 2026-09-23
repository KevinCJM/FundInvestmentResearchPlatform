import type { AgentPageContext } from './agent'

export const agentSessionStorageKey = (context: AgentPageContext) => {
  const scope = `${context.page}:${String(context.calculation.context_kind)}`
  // Keep the shipped indicator-page session key; new hosts include their object identity.
  return context.page === 'indicator-studio' && context.page_instance_id === context.page
    ? `indicator-agent-session:${scope}` : `agent-session:${scope}:${encodeURIComponent(context.page_instance_id)}`
}
export const agentContextKey = (context: AgentPageContext) => {
  const normalize = (value: unknown): unknown => Array.isArray(value) ? value.map(normalize)
    : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, normalize(item)])) : value
  const calculation = context.calculation.context_kind === 'single_product'
    ? { targets: [], period: '1Y', as_of: null, ...context.calculation } : context.calculation
  return JSON.stringify(normalize({ ...context, calculation }))
}
