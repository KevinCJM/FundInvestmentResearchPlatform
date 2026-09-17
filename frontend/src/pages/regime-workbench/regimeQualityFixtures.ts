// Offline fixture, never imported by product components.
import type { RegimeQualityPreview } from '../../services/regimeDiagnostics'
import { reliabilityDefinition } from './regimeReliabilityFixtures'
export const qualityDefinition = { ...reliabilityDefinition, id: 'history', revision: 3, default_mode: 'retrospective' as const, study: { purpose: 'historical_reference' as const, family: 'custom' as const } }
export function qualityPreviewFixture(): RegimeQualityPreview {
  return { preview_hash: 'q'.repeat(64), request: { definition_id: 'history', revision: 3, mode: 'retrospective', as_of: null, policy: {} }, report: {
    schema_version: '1.0', kind: 'historical_reference_quality', status: 'diagnostic_only',
    sample: { input: 100, classified: 80, unknown: 20, coverage: 0.8, head_unknown: 5, tail_unknown: 15, first_date: '2020-01-01', last_date: '2024-12-31' },
    segments: { total: 4, transitions: 3, per_state: [{ state_id: 'expansion', observations: 80, segments: 4, independent_complete_episodes: 3, conditional_estimation_status: 'ready', min_length: 10, median_length: 20, max_length: 30, mean_length: 20, price_return_samples: 0, mean_price_return: null, price_return_reason: 'no_verified_price_source' }] },
    price_returns: { status: 'unavailable', reason: 'no_verified_price_source', semantics: 'simple_endpoint_return' },
    horizon_profile: { method: 'empirical_complete_episode_duration', observation_frequency: 'monthly', calendar_span_days: 1827, classified_coverage: 0.8, transitions: 3, transitions_per_year: 0.6, censoring: 'open_head_tail_and_unknown_bounded_episodes_excluded', interpretation: 'measured_persistence_not_user_declared_horizon', per_state: [{ state_id: 'expansion', independent_complete_episodes: 3, duration_observations_p25: 15, duration_observations_median: 20, duration_observations_p75: 25, duration_calendar_days_p25: 450, duration_calendar_days_median: 600, duration_calendar_days_p75: 750, classified_occupancy: 1 }] },
    conditional_estimation: { status: 'ready', ready_states: ['expansion'], fallback_states: [], minimum_state_episodes: 3, fallback_policy: 'shrink_or_base_ltcma_for_insufficient_states' },
    stability: { status: 'completed', seed_status: 'not_applicable', variants: [{ kind: 'parameter', status: 'completed', changes: [{ node_id: 'trend', parameter: 'threshold', before: 0.1, after: 0.11 }], agreement: 0.9, classification_coverage: 0.8, comparable_observations: 80, boundary_distance: 2, boundary_distance_unit: 'observation_steps', graph_hash: 'variant-hash' }] },
    lineage: { definition_hash: 'exact-hash', snapshot: 'snapshot-1' }, warnings: [], execution: {},
  } }
}
