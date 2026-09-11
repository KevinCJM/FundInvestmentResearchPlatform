import type { HistoricalRegimeRun, RegimePublication } from './historicalRegimes'

export const latestProductResearchPublication = (run: HistoricalRegimeRun): RegimePublication | undefined => (
  [...(run.publications || [])].filter(item => item.usage === 'product_research' && item.run_id === run.id)
    .sort((a, b) => b.published_at.localeCompare(a.published_at))[0]
)

/** One current result per algorithm revision/mode, without changing an active reference. */
export function researchVersionChoices(runs: HistoricalRegimeRun[], selectedId = '') {
  const versions = new Map<string, HistoricalRegimeRun>()
  const eligible = runs.filter(run => run.immutable && latestProductResearchPublication(run))
    .sort((a, b) => (latestProductResearchPublication(b)?.published_at || '').localeCompare(latestProductResearchPublication(a)?.published_at || ''))
  for (const run of eligible) {
    const key = run.definition_id && run.definition_revision
      ? `${run.definition_id}:${run.definition_revision}:${run.mode}` : run.id
    const current = versions.get(key)
    if (!current || run.id === selectedId) versions.set(key, run)
  }
  return [...versions.values()]
}
