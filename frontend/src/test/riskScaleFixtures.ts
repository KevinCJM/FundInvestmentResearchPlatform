// Synthetic UI test data only. Never imported by a production page or service.
import type { Capabilities, PreviewResponse, ReferenceVersion, RiskScaleDefinition, VersionView } from '../services/riskScales'

export const riskReference: ReferenceVersion = {
  id: 'reference-fixture', content_hash: 'a'.repeat(64), created_at: '2026-09-17', artifact_type: 'reference_inputs', immutable: true,
  preview_hash: 'd'.repeat(64), definition: {
    name: 'Fixture reference assets', currency: 'CNY', as_of: '2026-09-17', calendar: 'SSE', frequency: 'daily', periods_per_year: 252,
    return_basis: 'selected_index_and_adjusted_product_total_return', fee_basis: 'source_embedded_no_additional_fee', fx_basis: 'same_currency_no_conversion',
    assets: [
      { id: 'cash', name: '现金', asset_type: 'cash', rationale: '', cash_return: .01, components: [], rebalance: null },
      { id: 'equity', name: '权益', asset_type: 'market', rationale: '', cash_return: null, rebalance: 'daily', components: [{ kind: 'index', series_id: 'index:index_daily:000300.SH', field: 'close', weight: 1 }] },
    ],
  }, ordered_asset_ids: ['cash', 'equity'], quality: { intersection_start: '2020-01-02', intersection_end: '2025-12-31', observations: 1500, periods_per_year: 252 }, warnings: [],
  moments: { annual_returns: [.01, .06], annual_volatilities: [0, .12], covariance: [[0, 0], [0, .0144]] }, provenance: {},
} as ReferenceVersion

export const riskDefinition: RiskScaleDefinition = {
  name: 'Fixture reference scale', description: 'Synthetic unit-test fixture', scheme_id: 'scheme-fixture', base_currency: 'CNY',
  risk_basis_id: 'annualized-periodic-volatility-v1', research_as_of: '2026-09-17', review_due_at: '2027-01-01', purpose: 'Synthetic test research',
  reference_input_ref: { id: riskReference.id, content_hash: riskReference.content_hash },
  constraint_profile: { asset_limits: {}, group_limits: [], allow_short: false, allow_leverage: false, gross_exposure: 1 }, segmentation: { algorithm_id: 'frontier_shape_dp_v2' },
}
export const riskCapabilities: Capabilities = { ready: true, execution: {}, algorithms: ['frontier_shape_dp_v2', 'equal_volatility_v1', 'equal_arclength_v1', 'equal_return_v1', 'manual_volatility_bands_v1'].map(id => ({ id, available: true })), limits: { primary_points: 101, stability_points: 200 }, reference: {}, templates: [], trust_mode: 'single_local_trusted_workspace' }
const available = (value: number) => ({ value, status: 'available', unit: 'annual_decimal' })
const unavailable = { value: null, status: 'unavailable', reason: 'No test history', unit: 'decimal' }
export const riskPreview: PreviewResponse = {
  request_echo: { definition: riskDefinition }, resolved_refs: { reference_inputs: riskDefinition.reference_input_ref }, data_fingerprints: {}, mathematical_status: 'verified', publication_eligibility: { eligible: true, blockers: [] }, default_eligibility: { eligible: true, blockers: [] }, warnings: [{ code: 'TEST_RESEARCH_WARNING', message: 'Synthetic test warning' }], limitations: ['Synthetic test data only'], execution_audit: { python_fallback: 0 }, preview_hash: 'b'.repeat(64),
  result: { ordered_asset_ids: ['cash', 'equity'], algorithm_id: 'frontier_shape_dp_v2', algorithm_version: 'test-only', applied_boundaries: [.02, .04, .06, .08, .10], diagnostics: { asset_names: { cash: '现金', equity: '权益' } }, stability: { status: 'stable' }, parameter_evidence: { method_identity: { id: 'historical_common_intersection' }, data_quality: riskReference.quality }, frontier: [
    { node_id: 0, volatility: .01, expected_return: .015, weights: [1, 0], status: 'optimal_to_tolerance' },
    { node_id: 1, volatility: .03, expected_return: .025, weights: [.8, .2], status: 'optimal_to_tolerance' },
    { node_id: 2, volatility: .05, expected_return: .035, weights: [.6, .4], status: 'optimal_to_tolerance' },
    { node_id: 3, volatility: .07, expected_return: .045, weights: [.4, .6], status: 'optimal_to_tolerance' },
    { node_id: 4, volatility: .09, expected_return: .055, weights: [.2, .8], status: 'optimal_to_tolerance' },
  ], levels: [1, 2, 3, 4, 5].map((level, index) => ({ level_code: `C${level}` as 'C1', lower_bound: index * .02, upper_bound: level * .02, lower_inclusive: index === 0, authorized_volatility_cap: level * .02, calibration_status: 'calibrated', representative_node_id: index, representative_weights: [1 - index * .2, index * .2], expected_return: available(.015 + index * .01), volatility: available(.01 + index * .02), historical_es: unavailable, historical_mdd: unavailable })) },
}
export const riskVersion: VersionView = { id: 'risk-fixture', name: riskDefinition.name, artifact_type: 'risk_scale', content_hash: 'c'.repeat(64), scheme_id: riskDefinition.scheme_id, version_number: 1, created_at: '2026-01-01', immutable: true, preview: riskPreview, current_eligibility: { eligible: true, blockers: [] }, retired: false, review_due_at: '2027-01-01', review_status: 'scheduled' }
