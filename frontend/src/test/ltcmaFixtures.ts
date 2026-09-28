// Offline fixtures; never imported by production components.
import type { CmaListItem, LtcmaCapabilities, LtcmaOptions } from '../services/ltcma'
import type { CmaDefinition, CmaVersion } from '../services/strategicAllocation'
import { cmaDefinition, cmaVersion, strategicCatalog } from './strategicAllocationFixtures'

export const ltcmaDefinition: CmaDefinition = { ...cmaDefinition, schema_version: '2.0', implementation_mapping_id: null,
  moment_semantics: 'annualized_periodic_arithmetic', fee_basis: 'source_embedded_no_additional_fee', fx_hedging_basis: 'same_currency_no_conversion' }
export const ltcmaVersion: CmaVersion = { ...cmaVersion, definition: ltcmaDefinition }
export const ltcmaItem: CmaListItem = { id: ltcmaVersion.id, name: ltcmaVersion.name, content_hash: ltcmaVersion.content_hash,
  created_at: ltcmaVersion.created_at, method: 'manual', as_of: ltcmaDefinition.as_of, currency: 'CNY',
  return_basis: ltcmaDefinition.return_basis, fee_basis: ltcmaDefinition.fee_basis, fx_hedging_basis: ltcmaDefinition.fx_hedging_basis,
  retired: false, schema_version: '2.0', moment_semantics: 'annualized_periodic_arithmetic', alloc_name: ltcmaDefinition.alloc_name,
  strategic_universe_id: null, implementation_mapping_id: null, asset_ids: ltcmaDefinition.assets.map(asset => asset.id), scope_name: ltcmaDefinition.alloc_name,
  version: { lineage_id: ltcmaVersion.id, number: 1, status: 'current', latest_id: ltcmaVersion.id, latest_number: 1 }, upstream: [], usable: { status: 'ready', reasons: [] } }
export const ltcmaCapabilities: LtcmaCapabilities = {
  methods: (['manual', 'historical_statistics', 'black_litterman', 'bayesian_niw', 'scenario_mixture', 'historical_regime_occupancy', 'long_term_scenario', 'conditional_scenario'] as const).map(id => ({ id, name: id, available: true, reason: null })),
  maximum_assets: 30, maximum_scenarios: 60, historical_frequency: 'daily', historical_currency: 'CNY',
}
export const ltcmaOptions: LtcmaOptions = { allocations: strategicCatalog.allocations, strategic_universes: [], assumptions: [ltcmaItem], regime_runs: [], existing_names: [] }
