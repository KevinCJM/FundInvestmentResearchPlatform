import { useI18n } from '../../i18n/runtime'
import type { CompatibilityEvidence } from '../../services/strategicAllocation'
import { percentText } from '../risk-models/ResearchUI'
import CrossModelResults from './CrossModelResults'

export default function CompatibilityResults({ evidence }: { evidence: CompatibilityEvidence }) {
  const { s } = useI18n()
  const solver = evidence.joint_solver
  return <section className="min-w-0 space-y-3" aria-label={s('multiCma.jointResult')}>
    <h3 className="font-semibold">{s('multiCma.jointResult')}</h3>
    <p role="status" className="text-sm text-slate-800">{s(`multiCma.solver.${solver.status}`)}</p>
    <p className="text-sm text-slate-600">{s(`multiCma.${evidence.objective}`)}</p>
    {solver.objective_gap != null && <p className="text-sm tabular-nums">{s('multiCma.objectiveGap', { gap: solver.objective_gap.toExponential(2) })}</p>}
    {solver.phase_one_lower_bound != null && <p className="text-sm tabular-nums">{s('multiCma.infeasibleBound', { bound: solver.phase_one_lower_bound.toExponential(4) })}</p>}
    {evidence.regret_basis === 'approximate_regret' && <p className="text-sm text-amber-800">{s('multiCma.approximateRegret')}</p>}
    <details className="min-w-0 border-t border-slate-200 pt-3"><summary className="cursor-pointer text-sm font-medium">{s('multiCma.anchors')}</summary>
      <p className="mt-3 text-sm leading-6 text-slate-600">{s('multiCma.anchorsHint')}</p>
      {evidence.anchors.map(anchor => <div key={anchor.cma_id} className="min-w-0 space-y-2 border-b border-slate-200 py-4">
        <h4 className="text-sm font-semibold">{anchor.name}</h4>
        <p className="text-sm">{s(`multiCma.solver.${anchor.solver.status}`)}</p>
        {anchor.weights && <p className="text-sm tabular-nums">{Object.entries(anchor.weights).map(([name, weight]) => `${name} ${percentText(weight)}`).join(' · ')}</p>}
        {anchor.reference_value != null && <p className="text-sm tabular-nums">{s('multiCma.anchorRange', { value: percentText(anchor.reference_value), upper: percentText(anchor.reference_upper_bound) })}</p>}
        {anchor.cross_model_results.length > 0 && <CrossModelResults common rows={anchor.cross_model_results} />}
      </div>)}
    </details>
    {evidence.limitations.map(message => <p key={message} className="text-xs leading-5 text-slate-600">{message}</p>)}
  </section>
}
