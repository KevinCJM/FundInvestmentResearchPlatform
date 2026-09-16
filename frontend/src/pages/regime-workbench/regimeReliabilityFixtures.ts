// Offline test data only. Never imported by product components.
import type { HistoricalReference, RegimeGraphDefinition, RegimeReliabilityPreview } from '../../services/regimeGraph'
export const reliabilityReference: HistoricalReference = {
  run_id: 'reference-run', publication_id: 'reference-publication', content_hash: 'a'.repeat(64),
  definition_id: 'history', definition_revision: 3, name: '离线参考夹具', frequency: 'monthly',
  states: [{ id: 'expansion', label: '扩张', color: '#22c55e' }, { id: 'contraction', label: '收缩', color: '#ef4444' }],
  as_of: '2024-12-31', created_at: '2025-01-01',
  series_summary: { first_observation_date: '2019-01-01', last_observation_date: '2024-12-31', row_count: 72 },
}
export const reliabilityDefinition: RegimeGraphDefinition = {
  schema_version: '2.0', id: 'recognition', revision: 2, name: '离线实时模型', description: '仅用于测试',
  default_mode: 'realtime', study: { purpose: 'realtime_recognition', family: 'custom', reference: { run_id: reliabilityReference.run_id, publication_id: reliabilityReference.publication_id, content_hash: reliabilityReference.content_hash } },
  graph: { nodes: [{ id: 'source', type: 'source.inline', parameters: { name: '离线观察数据' }, inputs: {} }], outputs: { state: { node_id: 'source', port: 'state' } } },
  states: reliabilityReference.states, evaluation_targets: [], validation: {}, usage_intent: 'research_display',
}
export function reliabilityPreviewFixture(): RegimeReliabilityPreview {
  const classification = { confusion: [[8, 1, 1], [2, 7, 1]], rows: ['expansion', 'contraction'], columns: ['expansion', 'contraction', 'abstained'], per_state: [
    { state_id: 'expansion', support: 10, precision: 0.8, recall: 0.8, f1: 0.8, iou: 0.67 },
    { state_id: 'contraction', support: 10, precision: 0.875, recall: 0.7, f1: 0.778, iou: 0.636 },
  ], accuracy: 0.75, balanced_accuracy: 0.75, macro_f1: 0.789, accepted_coverage: 0.9, accepted_error: 0.167 }
  const metric = { samples: 18, brier: 0.35, logloss: 0.5, ece: 0.07, reason: null, bins: [{ index: 0, samples: 8, mean_confidence: 0.6, match_rate: 0.625 }, { index: 1, samples: 10, mean_confidence: 0.8, match_rate: 0.8 }] }
  return {
    preview_hash: 'b'.repeat(64), request: { definition_id: 'recognition', revision: 2, reference: reliabilityDefinition.study!.reference!, policy: { calibration_end: '2020-12-31' } },
    report: {
      schema_version: '1.0', status: 'retrospective_only', states: reliabilityReference.states,
      warnings: ['Reference agreement is not ground truth.'],
      sample: { input: 22, matched: 21, unknown_reference: 2, prediction_abstentions: 2, missing_prediction: 1, invalid_prediction_labels: 0, excluded_dates: { reference_after_cutoff: 3, prediction_without_reference: 1 }, blocks: { calibration: { start: '2019-01-01', end: '2020-12-31', samples: 10, per_class: { expansion: 5, contraction: 5 }, complete_segments: 2, sufficient: false }, holdout: { start: '2021-01-01', end: '2024-12-31', samples: 10, per_class: { expansion: 5, contraction: 5 }, complete_segments: 2, sufficient: false } } },
      classification,
      intervals: { reference_segments: 5, predicted_segments: 6, matches: 4, misses: 1, false_events: 2, complete_reference_segments: 3, equal_reference_segment_iou: 0.61 },
      transitions: { reference_events: 4, predicted_events: 5, matches: 3, misses: 1, false_events: 2, delay_median: 1, delay_p90: 2, delay_unit: 'observation_steps', tolerance: 3 },
      verification: { status: 'partially_verified', scope: 'holdout', purpose: 'recognition_state_evidence', verified_states: ['expansion'], fallback_states: ['contraction'], probability_improves_class_base: true, recognition_ready: true, production_eligible: false, reasons: ['some_states_have_insufficient_evidence'], policy: { minimum_state_episodes: 3, minimum_state_predictions: 5, minimum_state_precision: 0.65, confidence_floor: 0.6 }, unverified_state_policy: 'do_not_authorize_unverified_states', states: [
        { state_id: 'expansion', status: 'verified', reference_observations: 5, independent_complete_episodes: 3, accepted_predictions: 6, matches: 5, precision: 0.833, recall: 1, reasons: [] },
        { state_id: 'contraction', status: 'insufficient_evidence', reference_observations: 5, independent_complete_episodes: 1, accepted_predictions: 2, matches: 2, precision: 1, recall: 0.4, reasons: ['insufficient_independent_state_episodes', 'insufficient_accepted_state_predictions'] },
      ] },
      probability: { raw_type: 'deterministic_state', blocks: { holdout: { raw: metric, calibrated: metric, class_base: metric, classification } } },
      calibration: { method: 'class_frequency', evidence_type: 'class_average', fitted: true, reason: null, parameters: {}, calibration_end: '2020-12-31', validation_end: null, test_end: '2024-12-31', label_known_at: '2025-01-01', deployment_eligible: false, reasons: ['reference_labels_unavailable_at_calibration_end', 'model_and_reference_selection_history_unverified'], available_from: '2026-09-14', expires_on: '2026-12-13' },
      lineage: { definition_id: 'recognition', reference_content_hash: reliabilityReference.content_hash },
      stability: { status: 'not_executed', reason: 'not run' }, confidence_interval: { status: 'unavailable', reason: 'not implemented' },
      points: [{ observation_date: '2024-12-31', reference_state: null, predicted_state: 'expansion', block: 'holdout', raw_probabilities: { expansion: 1, contraction: 0 }, calibrated_probabilities: { expansion: 0.8, contraction: 0.2 }, calibrated_confidence: 0.8, decision_status: 'diagnostic_only' }],
    },
  }
}
