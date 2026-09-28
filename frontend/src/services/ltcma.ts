import { systemText } from '../i18n/runtime'
import { checkedCma, cmaRequest, getCma, previewCma, strategicRequest,
  type CmaDefinition, type CmaPreview, type CmaVersion, type StrategicCatalog } from './strategicAllocation'
import type { CmaDraftView, CmaDraftWrite, CmaListItem, CmaListResponse, CmaMethodId, CmaSampleRequest, CmaSampleSummary, CmaVersionRef } from './ltcmaContract.generated'
export type { CmaDraftView, CmaDraftWrite, CmaListItem, CmaListResponse, CmaMethodId } from './ltcmaContract.generated'

export interface LtcmaCapabilities {
  methods: Array<{ id: CmaMethodId; name: string; available: boolean; reason: string | null }>
  maximum_assets: number; maximum_scenarios: number; historical_frequency: string; historical_currency: string
}
export interface LtcmaOptions {
  allocations: StrategicCatalog['allocations']
  strategic_universes: NonNullable<StrategicCatalog['strategic_universes']>
  assumptions: CmaListItem[]
  existing_names: string[]
  regime_runs: Array<{ id: string; name: string; content_hash: string; as_of: string; frequency: string; states: Array<{ id: string; label?: string }>; available?: boolean; reasons?: Array<{ code: string; message: string }> }>
  scenario_options?: {
    historical_references: ScenarioHistoryOption[]
    realtime_runs: ScenarioRealtimeOption[]
    default_historical_id: string | null
  }
}
export interface ScenarioReference { run_id: string; publication_id: string; content_hash: string }
interface ScenarioOption {
  id: string; name: string; content_hash: string; available: boolean
  reference: ScenarioReference | null; reasons: Array<{ code: string; message: string }>
}
export interface ScenarioHistoryOption extends ScenarioOption {
  frequency: string; states: Array<{ id: string; label?: string }>
}
export interface ScenarioRealtimeOption extends ScenarioOption { as_of: string }
export interface LtcmaView { version: CmaVersion; retired: boolean }
export type LtcmaOptionSection = 'priors' | 'regimes' | 'scenarios'
const idPath = (id: string) => encodeURIComponent(id)
export const ltcma = {
  list: async (params: { q?: string; method?: string; include_retired?: boolean; offset?: number; limit?: number } = {}, signal?: AbortSignal) => {
    const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]))
    const value = await strategicRequest<CmaListResponse>(`/cma?${query}`, undefined, signal)
    if (!Array.isArray(value.items) || !Number.isInteger(value.total)) throw new Error(systemText('preInvestment.ltcma.invalidLtcmaListResponse'))
    return value
  },
  capabilities: (signal?: AbortSignal) => strategicRequest<LtcmaCapabilities>('/cma/capabilities', undefined, signal),
  options: (signal?: AbortSignal, asOf?: string) => strategicRequest<LtcmaOptions>(`/cma/study-options?${new URLSearchParams({ section: 'base', ...(asOf ? { as_of: asOf } : {}) })}`, undefined, signal),
  methodOptions: (section: LtcmaOptionSection, asOf: string, signal?: AbortSignal, selectedPriorId?: string) =>
    strategicRequest<Partial<Pick<LtcmaOptions, 'assumptions' | 'regime_runs' | 'scenario_options'>>>(`/cma/study-options?${new URLSearchParams({ section, as_of: asOf,
      ...(section === 'priors' && selectedPriorId ? { selected_prior_id: selectedPriorId } : {}) })}`, undefined, signal),
  get: getCma,
  view: async (id: string, signal?: AbortSignal): Promise<LtcmaView> => {
    const value = await strategicRequest<LtcmaView>(`/cma/${idPath(id)}/view`, undefined, signal)
    return { ...value, version: checkedCma(value.version) }
  },
  preview: previewCma,
  sample: (body: CmaSampleRequest, signal?: AbortSignal) => strategicRequest<CmaSampleSummary>('/cma/sample', body, signal),
  publish: async (definition: CmaDefinition, previewHash: string, operationKey: string, copiedFromId?: string | null, signal?: AbortSignal) =>
    checkedCma(await strategicRequest<CmaVersion>('/cma', { request: cmaRequest(definition), preview_hash: previewHash,
      confirm: true, idempotency_key: operationKey, copied_from_id: copiedFromId ?? null }, signal)),
  update: async (reference: CmaVersionRef, definition: CmaDefinition, previewHash: string, operationKey: string, signal?: AbortSignal) =>
    checkedCma(await strategicRequest<CmaVersion>(`/cma/${idPath(reference.id)}`, {
      request: cmaRequest(definition), preview_hash: previewHash, confirm: true,
      idempotency_key: operationKey, expected_content_hash: reference.content_hash,
    }, signal, 'PATCH')),
  retire: (value: CmaListItem | CmaVersion, reason: string, signal?: AbortSignal) => strategicRequest<{ id: string; retired: true }>(
    `/cma/${idPath(value.id)}/retire`, { confirm: true, content_hash: value.content_hash, reason }, signal),
  drafts: (signal?: AbortSignal) => strategicRequest<{ items: CmaDraftView[] }>('/cma/drafts', undefined, signal),
  draft: (id: string, signal?: AbortSignal) => strategicRequest<CmaDraftView>(`/cma/drafts/${idPath(id)}`, undefined, signal),
  saveDraft: (body: CmaDraftWrite, id?: string, signal?: AbortSignal) => strategicRequest<CmaDraftView>(
    id ? `/cma/drafts/${idPath(id)}` : '/cma/drafts', body, signal, id ? 'PATCH' : 'POST'),
  deleteDraft: (draft: CmaDraftView, signal?: AbortSignal) => strategicRequest<{ deleted: true }>(
    `/cma/drafts/${idPath(draft.id)}`, { expected_revision: draft.revision }, signal, 'DELETE'),
}

export function ltcmaSaaPath(value: CmaVersion): string {
  const query = new URLSearchParams({ cma: value.id })
  if (value.definition.strategic_universe_id) query.set('strategic_universe', value.definition.strategic_universe_id)
  else if (value.definition.alloc_name) query.set('alloc', value.definition.alloc_name)
  return `/pre-investment/saa/policy?${query}`
}

/** A frozen research result is not itself evidence that it can drive SAA. */
export function ltcmaSaaIssue(value: CmaPreview): string | null {
  const method = value.definition.model?.method
  const validation = value.model_result?.model_audit.model_validation
  const status = validation && typeof validation === 'object' && !Array.isArray(validation) ? validation as Record<string, unknown> : {}
  if (method === 'conditional_scenario') return typeof status.reason === 'string' && status.reason.trim()
    ? status.reason : systemText('preInvestment.ltcma.conditionalScenariosAreResearchOnlyForecastingAnd')
  if (status.downstream_eligible === false || method === 'long_term_scenario' && status.downstream_eligible !== true) return typeof status.reason === 'string' && status.reason.trim()
    ? status.reason : systemText('preInvestment.ltcma.thisScenarioStudyIsNotEligibleFor')
  return null
}
