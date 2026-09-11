import type { HistoricalRegimeRun, RegimePublication } from './historicalRegimes'

type Application = 'taa' | 'formal_backtest'
type PublicationCandidate = Pick<HistoricalRegimeRun,
  'id' | 'schema_version' | 'definition_id' | 'definition_revision'
  | 'immutable' | 'mode' | 'content_hash' | 'definition_snapshot_hash'
  | 'causality' | 'governance' | 'publications'
>

const snapshotHash = /^[a-f0-9]{64}$/

/** Display eligibility only; the server revalidates the immutable snapshot on use. */
export function eligibleRegimePublications(
  run: PublicationCandidate,
  application: Application,
): RegimePublication[] {
  const isV2 = run.schema_version === '2.0'
  const isV1 = run.schema_version === undefined || run.schema_version === '1.0'
  if (!isV2 && !isV1) return []
  // The portfolio conditioning contract supports V2 only; TAA also accepts legacy runs.
  if (application === 'formal_backtest' && !isV2) return []
  if (
    !run.id || !run.definition_id
    || !Number.isInteger(run.definition_revision) || (run.definition_revision ?? 0) < 1
    || run.immutable !== true || run.mode !== 'realtime'
    || !run.content_hash
    || run.causality?.is_causal !== true
    || run.causality?.uses_future_data !== false
    || run.causality?.repaints !== false
    || run.causality?.realtime_eligible !== true
  ) return []
  if (isV2 && (
    !snapshotHash.test(run.content_hash)
    || !snapshotHash.test(run.definition_snapshot_hash ?? '')
    || run.governance?.formal_gate_passed !== true
  )) return []

  const gate = isV2 ? 'comprehensive_formal_gate_passed' : 'causality_passed'
  return (run.publications ?? []).filter((publication) => (
    Boolean(publication.id)
    && (publication.usage === 'formal_backtest' || (application === 'taa' && publication.usage === 'taa'))
    && publication.run_id === run.id
    && publication.definition_revision === run.definition_revision
    && publication.run_content_hash === run.content_hash
    && publication.gate === gate
    && (!isV2 || run.governance?.publish_eligible_usages?.includes(publication.usage) === true)
  ))
}

export function isRegimeRunEligibleForTaa(run: PublicationCandidate): boolean {
  return eligibleRegimePublications(run, 'taa').length > 0
}
