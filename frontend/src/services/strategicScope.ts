import { strategicRequest, type EconomicRole } from './strategicAllocation'

export interface StrategicAsset {
  id: string; name: string; currency: string; role: EconomicRole
  liquidity: 'liquid' | 'illiquid'; rationale: string; source: string
}
export interface UniverseDefinition {
  name: string; as_of: string; currency: string; source: string; assets: StrategicAsset[]
}
export interface UniversePreview {
  definition: UniverseDefinition; preview_hash: string; implementation_status: 'unmapped'
  implementation_gaps: string[]; research_only: true
}
export interface UniverseVersion extends UniversePreview { id: string; name: string; content_hash: string; created_at: string }
export interface MappingDefinition {
  name: string; strategic_universe_id: string; universe_snapshot_id: string; alloc_name: string
  as_of: string; valid_until: string
  assignments: Array<{ strategic_asset_id: string; proxy_asset_id: string; rationale: string }>
}
export interface MappingPreview {
  definition: MappingDefinition; preview_hash: string; implementation_status: 'complete' | 'incomplete'
  implementation_gaps: string[]
  coverage: Array<{ strategic_asset_id: string; proxy_asset_id: string | null; status: 'mapped' | 'missing_products' }>
}
export interface MappingVersion extends MappingPreview { id: string; name: string; content_hash: string; created_at: string }
export const previewUniverse = (body: UniverseDefinition, signal?: AbortSignal) => strategicRequest<UniversePreview>('/universes/preview', body, signal)
export const confirmUniverse = (body: UniverseDefinition, hash: string, signal?: AbortSignal) => strategicRequest<UniverseVersion>('/universes/confirm', { request: body, preview_hash: hash }, signal)
export const getStrategicUniverse = (id: string, signal?: AbortSignal) => strategicRequest<UniverseVersion>(`/universes/${encodeURIComponent(id)}`, undefined, signal)
export const previewImplementationMap = (body: MappingDefinition, signal?: AbortSignal) => strategicRequest<MappingPreview>('/implementation-maps/preview', body, signal)
export const confirmImplementationMap = (body: MappingDefinition, hash: string, signal?: AbortSignal) => strategicRequest<MappingVersion>('/implementation-maps/confirm', { request: body, preview_hash: hash }, signal)
export const getImplementationMap = (id: string, signal?: AbortSignal) => strategicRequest<MappingVersion>(`/implementation-maps/${encodeURIComponent(id)}`, undefined, signal)
export const economicRoles: Array<[EconomicRole, string]> = [['growth', '增长参与'], ['rates', '利率防御'], ['inflation', '通胀分散'], ['credit', '信用收益'], ['liquidity', '现金与流动性储备'], ['diversifier', '其他分散用途']]
