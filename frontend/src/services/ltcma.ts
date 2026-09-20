import { checkedCma, cmaRequest, getCma, previewCma, strategicRequest,
  type CmaDefinition, type CmaVersion, type StrategicCatalog } from './strategicAllocation'
import type { CmaDraftView, CmaDraftWrite, CmaListItem, CmaListResponse, CmaMethodId } from './ltcmaContract.generated'
export type { CmaDraftView, CmaDraftWrite, CmaListItem, CmaListResponse, CmaMethodId } from './ltcmaContract.generated'

export interface LtcmaCapabilities {
  methods: Array<{ id: CmaMethodId; name: string; available: boolean; reason: string | null }>
  maximum_assets: number; maximum_scenarios: number; historical_frequency: string; historical_currency: string
}
export interface LtcmaOptions {
  allocations: StrategicCatalog['allocations']
  strategic_universes: NonNullable<StrategicCatalog['strategic_universes']>
  assumptions: CmaListItem[]
  regime_runs: Array<{ id: string; name: string; content_hash: string; as_of: string; frequency: string; states: Array<{ id: string; label?: string }> }>
}
export interface LtcmaView { version: CmaVersion; retired: boolean }
const idPath = (id: string) => encodeURIComponent(id)
export const ltcma = {
  list: async (params: { q?: string; method?: string; include_retired?: boolean; offset?: number; limit?: number } = {}, signal?: AbortSignal) => {
    const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]))
    const value = await strategicRequest<CmaListResponse>(`/cma?${query}`, undefined, signal)
    if (!Array.isArray(value.items) || !Number.isInteger(value.total)) throw new Error('LTCMA 列表响应无效。')
    return value
  },
  capabilities: (signal?: AbortSignal) => strategicRequest<LtcmaCapabilities>('/cma/capabilities', undefined, signal),
  options: (signal?: AbortSignal) => strategicRequest<LtcmaOptions>('/cma/study-options', undefined, signal),
  get: getCma,
  view: async (id: string, signal?: AbortSignal): Promise<LtcmaView> => {
    const value = await strategicRequest<LtcmaView>(`/cma/${idPath(id)}/view`, undefined, signal)
    return { ...value, version: checkedCma(value.version) }
  },
  preview: previewCma,
  publish: async (definition: CmaDefinition, previewHash: string, operationKey: string, copiedFromId?: string | null, signal?: AbortSignal) =>
    checkedCma(await strategicRequest<CmaVersion>('/cma', { request: cmaRequest(definition), preview_hash: previewHash,
      confirm: true, idempotency_key: operationKey, copied_from_id: copiedFromId ?? null }, signal)),
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
