import type { PolicyFrontierResult, PolicyPreview, PolicyRequest } from '../../services/strategicAllocation'

// Model identity is a chart category, not an action/status color. Up to 20 models.
export const MODEL_COLOR_CLASSES = [
  'text-accent-700', 'text-emerald-700', 'text-amber-700', 'text-violet-700', 'text-cyan-700',
  'text-rose-700', 'text-slate-700', 'text-orange-700', 'text-lime-700', 'text-fuchsia-700',
  'text-accent-950', 'text-emerald-950', 'text-amber-950', 'text-violet-950', 'text-cyan-950',
  'text-rose-950', 'text-slate-950', 'text-orange-950', 'text-lime-950', 'text-fuchsia-950',
]
export const MODEL_SYMBOLS = ['circle', 'rect', 'triangle', 'diamond', 'pin']
export type FrontierView = PolicyFrontierResult['views'][number]
export type CandidatePoint = { id: string; candidateName: string; modelId?: string;
  expected_return: number; volatility: number; withinLimits?: boolean }

/** Use frozen per-source evaluations, never the common candidate's worst-per-metric summary. */
export function frontierCandidatePoints(data: PolicyFrontierResult | null, result: PolicyPreview | null, request: PolicyRequest): CandidatePoint[] {
  if (!data || !result || result.request?.mandate_id !== request.mandate_id ||
      (result.request.mode ?? 'single') !== (request.mode ?? 'single')) return []
  if (data.mode !== (request.mode ?? 'single')) return []
  if (data.mode === 'single' && result.request.cma_id !== request.cma_id) return []
  if (data.mode === 'parameter_average' && result.multi_cma?.content_hash !== data.views[0]?.cma_hash) return []
  if (data.mode !== 'compatible_all_models') {
    return result.candidates.filter(c => Number.isFinite(c.metrics.expected_return) && Number.isFinite(c.metrics.volatility))
      .map(c => ({ id: `candidate:${c.id}`, candidateName: c.name, ...c.metrics }))
  }
  const refs = result.multi_cma?.refs
  if (!refs || refs.length !== data.views.length || !data.views.every(view => refs.some(ref =>
    ref.cma_id === view.id && ref.content_hash === view.cma_hash))) return []
  return [...result.candidates, ...(result.unavailable_candidates ?? [])].flatMap(candidate => {
    if (candidate.id !== 'compatible') return []
    const rows = candidate.cross_model_results
    // Incomplete evidence is not a partial all-model result, even if one row is valid.
    if (!rows || rows.length !== data.views.length || !data.views.every(view => rows.some(row =>
      row.cma_id === view.id && row.cma_hash === view.cma_hash &&
      Number.isFinite(row.metrics.expected_return) && Number.isFinite(row.metrics.volatility)))) return []
    return data.views.map(view => {
      const row = rows.find(row => row.cma_id === view.id)!
      return { id: `candidate:${candidate.id}:${view.id}`, candidateName: candidate.name, modelId: view.id,
        expected_return: row.metrics.expected_return, volatility: row.metrics.volatility,
        withinLimits: row.within_limits }
    })
  })
}
