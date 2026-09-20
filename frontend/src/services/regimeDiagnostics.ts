// Contracts from docs/regimes/validation.md.
export interface RegimeStabilityPolicy {
  enabled?: boolean; max_variants?: number; perturbation?: number
  parameters?: boolean; windows?: boolean; seeds?: boolean; truncation?: boolean
}
export interface RegimeBootstrapPolicy {
  enabled?: boolean; replicates?: number; block_length?: number; confidence_level?: number
  minimum_blocks?: number; minimum_cycles?: number; minimum_valid_replicates?: number
}
export interface RegimeStabilityResult {
  status: string; reason?: string | null; seed_status?: string
  limits?: Record<string, unknown>
  variants?: Array<{
    kind: 'parameter' | 'window' | 'seed' | 'truncation'; status: string
    changes: Array<{ node_id: string; parameter: string; before: unknown; after: unknown }>
    graph_hash?: string; reason?: string | null; agreement?: number | null
    comparable_observations?: number | null; classification_coverage?: number | null
    boundary_distance?: number | null; boundary_distance_unit?: 'observation_steps'
    [key: string]: unknown
  }>
}
export interface RegimeConfidenceInterval {
  status: string; reason?: string | null; method?: string; scope?: string; conditional_on?: string
  confidence_level?: number; block_length?: number; replicates?: number; seed?: number
  samples?: number; full_blocks?: number; complete_cycles?: number
  metrics?: Record<string, {
    estimate: number | null; lower: number | null; upper: number | null
    valid_replicates: number; reason: string | null; unit: 'fraction' | 'brier_score'
  }>
}
export interface RegimeProbabilityEvidence {
  selected_probability: number | null; top_probability: number | null; second_probability: number | null
  margin: number | null; entropy: number | null; entropy_unit: 'nats'
}
export interface RegimeQualityRequest {
  definition_id: string; revision: number; mode: 'retrospective'; as_of?: string | null
  policy: { stability?: RegimeStabilityPolicy; include_price_returns?: boolean; minimum_state_episodes_for_estimation?: number }
}
export interface RegimeHorizonProfile {
  method: 'empirical_complete_episode_duration'; observation_frequency: string
  calendar_boundary?: 'observation_inclusive_next_observation_exclusive'
  calendar_span_days: number | null; classified_coverage: number | null
  transitions: number; transitions_per_year: number | null
  censoring: 'open_head_tail_and_unknown_bounded_episodes_excluded'
  per_state: Array<{
    state_id: string; independent_complete_episodes: number
    duration_observations_p25: number | null; duration_observations_median: number | null; duration_observations_p75: number | null
    duration_calendar_days_p25: number | null; duration_calendar_days_median: number | null; duration_calendar_days_p75: number | null
    classified_occupancy: number | null
  }>
  interpretation: 'measured_persistence_not_user_declared_horizon'
}
export interface RegimeQualityReport {
  schema_version: '1.0'; kind: 'historical_reference_quality'
  status: 'diagnostic_only' | 'insufficient_evidence'
  sample: { input: number; classified: number; unknown: number; coverage: number | null; head_unknown: number; tail_unknown: number; first_date: string | null; last_date: string | null }
  segments: { total: number; transitions: number; per_state: Array<{
    state_id: string; observations: number; segments: number; independent_complete_episodes: number
    conditional_estimation_status: 'ready' | 'insufficient_evidence'
    min_length: number | null; median_length: number | null; max_length: number | null; mean_length: number | null
    price_return_samples: number; mean_price_return: number | null; price_return_reason: string | null
  }> }
  price_returns: { status: 'available' | 'unavailable' | 'disabled'; reason: string | null; semantics: string | Record<string, unknown> | null }
  horizon_profile?: RegimeHorizonProfile
  conditional_estimation: { status: 'ready' | 'partially_ready' | 'insufficient_evidence'; ready_states: string[]; fallback_states: string[]; minimum_state_episodes: number; fallback_policy: string; scope?: 'episode_sample_sufficiency_only'; ltcma_estimated?: false }
  stability: RegimeStabilityResult; lineage: Record<string, unknown>; warnings: string[]; execution: Record<string, unknown>
}
export interface RegimeQualityPreview { preview_hash: string; request: RegimeQualityRequest; report: RegimeQualityReport }
export interface SavedRegimeQuality extends RegimeQualityPreview { id: string; created_at: string; immutable: true; content_hash: string }
export interface RegimeQualityCatalogItem { id: string; created_at: string; definition_id: string; revision: number; status: RegimeQualityReport['status']; conditional_estimation?: RegimeQualityReport['conditional_estimation'] | null }
