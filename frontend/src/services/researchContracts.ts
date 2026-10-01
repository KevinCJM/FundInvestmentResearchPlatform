import type { EvaluationResult, IndicatorDraft, TimeSeriesIndicatorResult } from './customIndicators'

export type ResearchPage = 'platform-agent' | 'indicator-studio' | 'product-detail' | 'product-research' | 'product-compare' | 'holding-diagnosis' | 'evaluation-plan' | 'scenario-algorithms' | 'historical-regimes' | 'published-scenarios' | 'global-events'

export interface IndicatorPreviewReference {
  preview_id?: string
  run_id?: string
  definition_hash: string
  target: { kind: string; product_id: string; name?: string }
  period: string
  as_of?: string | null
  result_kind: 'scalar' | 'time_series'
  created_at?: string
}

export type IndicatorPreview = IndicatorPreviewReference & {
  preview_id: string
  authoring_id: string
  definition: IndicatorDraft
  expires_at?: string
  /** Server-frozen provenance of the run that produced this preview, when available. */
  context_hash?: string
  data_generation?: string | null
  effective_context?: { as_of?: string | null; run_mode?: string | null; data_release_id?: string | null } | null
} & ({ result_kind: 'scalar'; result: { results: EvaluationResult[] } }
  | { result_kind: 'time_series'; result: { results: TimeSeriesIndicatorResult[] } })

/**
 * Versioned, curated copy of what the page displayed when the user sent a message.
 * Transport-only: the server binds it to that run and never treats it as authorization.
 */
export interface PageEvidenceSnapshot {
  version: 1
  snapshot_id: string
  captured_at?: string
  page: ResearchPage
  sections: Record<string, unknown>
}

export interface ResearchPageContext {
  page: ResearchPage
  page_instance_id: string
  context_revision: number
  view_state: 'unknown' | 'inherit' | 'explicit' | 'off'
  calculation: Record<string, unknown>
}
