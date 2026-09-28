import { systemText, useI18n } from '../../i18n/runtime'
import { Button } from '../ui'
import { Field, NumberInput, inputClass, percentText } from '../risk-models/ResearchUI'

export interface FrontierGridSettings {
  enabled: boolean
  point_count: number
  max_iterations: number
  accept_continuous_weights: boolean
}
export const defaultFrontierGrid: FrontierGridSettings = {
  enabled: false, point_count: 20, max_iterations: 300, accept_continuous_weights: false,
}
export interface FrontierGridPoint {
  target_index: number; target: number | null; value: [number | null, number | null]
  weights: Array<number | null>; status: string; iterations: number
  optimality_residual: number | null; constraint_violation: number | null
  duplicate_of: number | null; candidate_index: number | null; on_frontier: boolean
  adoption?: { weight_domain: 'discrete'; step: number; status: string; target_met: boolean
    candidate_index: number | null; duplicate_of?: number | null; weights: Array<number | null>; value: [number | null, number | null] }
}
export interface FrontierGridResult {
  requested_points: number; attempted_points: number; successful_points: number; failed_points: number
  unattempted_points: number; duplicate_targets: number; duplicate_solutions: number; added_candidates: number
  adoption_weight_domain?: 'discrete' | 'continuous'
  adoption_duplicate_solutions?: number
  max_iterations: number; risk_solver: string; optimality_scope: string
  points: FrontierGridPoint[]; curve: Array<FrontierGridPoint | null>
  endpoints: Array<{ kind: string; status: string; iterations: number }>
}
const statusText: Record<string, string> = {
  get feasible() { return systemText('preInvestment.frontierGrid.availableForAdoption') }, get infeasible() { return systemText('preInvestment.frontierGrid.infeasibleAtTheSelectedPrecision') }, get search_budget() { return systemText('preInvestment.frontierGrid.discreteSearchBudgetExhausted') }, get grid_failed() { return systemText('preInvestment.frontierGrid.continuousTargetUnsolved') },
  get converged() { return systemText('preInvestment.frontierGrid.converged') }, get max_iterations() { return systemText('preInvestment.frontierGrid.iterationLimitReached') }, get infeasible_target() { return systemText('preInvestment.frontierGrid.targetInfeasible') },
  get numerical_failure() { return systemText('preInvestment.frontierGrid.numericalSolverFailed') }, get line_search_failed() { return systemText('preInvestment.frontierGrid.lineSearchDidNotConverge') }, get range_unresolved() { return systemText('preInvestment.frontierGrid.endpointNotConvergedNotStarted') },
}
export function frontierGridIssue(value: FrontierGridSettings, quantized: boolean): string {
  if (!value.enabled) return ''
  if (!Number.isInteger(value.point_count) || value.point_count < 2 || value.point_count > 200) return systemText('preInvestment.frontierGrid.frontierTargetCountMustBeAnInteger')
  if (!Number.isInteger(value.max_iterations) || value.max_iterations < 1 || value.max_iterations > 1000) return systemText('preInvestment.frontierGrid.maximumIterationsPerPointMustBeAn')
  if (quantized && !value.accept_continuous_weights) return systemText('preInvestment.frontierGrid.confirmThatTheContinuousTheoreticalCurveAnd')
  return ''
}

export function FrontierGridControls({ value, quantized, busy, onChange }: {
  value: FrontierGridSettings; quantized: boolean; busy: boolean
  onChange: (next: FrontierGridSettings) => void
}) {
  useI18n()
  const issue = frontierGridIssue(value, quantized)
  return <div className="space-y-4" aria-label={systemText('preInvestment.frontierGrid.fullFrontierSolverSettings')}>
    <label className="flex min-h-10 items-center gap-2 text-sm font-medium">
      <input type="checkbox" checked={value.enabled} disabled={busy}
        onChange={event => onChange({ ...value, enabled: event.target.checked })} />{systemText('preInvestment.frontierGrid.densifyTheFullFrontierWithATarget')}</label>
    <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.frontierGrid.withinCurrentAssetsAndIndividualGroupBounds')}</p>
    {value.enabled && <>
      <div className="grid gap-4 sm:grid-cols-2">
        <Field label={systemText('preInvestment.frontierGrid.frontierTargetCount')} hint={systemText('preInvestment.frontierGrid.2200ReturnTargetsMorePointsDensify')}>
          <NumberInput aria-label={systemText('preInvestment.frontierGrid.frontierTargetCount')} className={inputClass} value={value.point_count} min={2} max={200} disabled={busy}
            onValueChange={point_count => onChange({ ...value, point_count })} />
        </Field>
        <Field label={systemText('preInvestment.frontierGrid.maximumIterationsPerPoint')} hint={systemText('preInvestment.frontierGrid.separateBudgetForEachOptimizationProblemDefaults')}>
          <NumberInput aria-label={systemText('preInvestment.frontierGrid.maximumIterationsPerPoint')} className={inputClass} value={value.max_iterations} min={1} max={1000} disabled={busy}
            onValueChange={max_iterations => onChange({ ...value, max_iterations })} />
        </Field>
      </div>
      <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.frontierGrid.volatilityBasedRiskSolvesMinimumRiskPortfolios')}</p>
      {quantized && <label className="flex items-start gap-2 text-sm leading-6">
        <input className="mt-1" type="checkbox" disabled={busy} checked={value.accept_continuous_weights}
          onChange={event => onChange({ ...value, accept_continuous_weights: event.target.checked })} />
        {systemText('preInvestment.frontierGrid.iConfirmThatTheGridUsesContinuous')}</label>}
      {issue && <p role="status" className="text-sm text-amber-800">{issue}</p>}
    </>}
  </div>
}

export function FrontierGridResults({ result, assetNames, riskLabel, returnLabel, onAdopt }: {
  result: FrontierGridResult; assetNames: string[]; riskLabel: string; returnLabel: string
  onAdopt: (point: FrontierGridPoint) => void
}) {
  useI18n()
  const discrete = result.adoption_weight_domain === 'discrete'
  const unresolved = result.endpoints.filter(endpoint => endpoint.status !== 'converged')
  return <section className="mt-4 space-y-3" aria-label={systemText('preInvestment.frontierGrid.frontierSolutionsByTarget')}>
    <h3 className="text-base font-semibold">{systemText('preInvestment.frontierGrid.fullFrontierSolverResults')}</h3>
    <p role="status" className="text-sm leading-6">{systemText('preInvestment.frontierGrid.target') + " "}{result.requested_points} {" " + systemText('preInvestment.frontierGrid.pointsSuccessful') + " "}{result.successful_points} {" " + systemText('preInvestment.frontierGrid.pointsFailed') + " "}{result.failed_points} {" " + systemText('preInvestment.frontierGrid.pointsNotStarted') + " "}{result.unattempted_points} {" " + systemText('preInvestment.frontierGrid.pointsDuplicateTargets') + " "}{result.duplicate_targets} {" " + systemText('preInvestment.frontierGrid.pointsDuplicateSolutions') + " "}{result.duplicate_solutions} {" " + systemText('preInvestment.frontierGrid.points')}{discrete ? systemText('preInvestment.frontierGrid.duplicatePortfoliosAtTheSelectedPrecision', { p0: result.adoption_duplicate_solutions ?? 0 }) : ''}</p>
    {unresolved.length > 0 && <p className="text-sm text-amber-800">{systemText('preInvestment.frontierGrid.frontierEndpointsIncomplete')}{unresolved.map(endpoint => `${endpoint.kind === 'minimum_risk' ? systemText('preInvestment.frontierGrid.minimumRisk') : systemText('preInvestment.frontierGrid.maximumReturn')}：${statusText[endpoint.status] ?? endpoint.status}`).join('；')}{systemText('preInvestment.frontierGrid.increaseThePerPointIterationBudgetOr')}</p>}
    <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.frontierGrid.theCurveConnectsActualSolutionsInReturn')}{result.optimality_scope === 'local_numerical_stationarity' ? systemText('preInvestment.frontierGrid.onlyLocalNumericalConvergenceCanBeReported') : systemText('preInvestment.frontierGrid.usesConvexQuadraticProgrammingWithNumericalKkt')}{discrete ? systemText('preInvestment.frontierGrid.theContinuousTheoreticalFrontierIsReferenceOnly') : systemText('preInvestment.frontierGrid.bothTheGridAndAdoptablePortfoliosUse')}</p>
    <details>
      <summary className="min-h-10 cursor-pointer py-2 text-sm font-medium">{systemText('preInvestment.frontierGrid.perTargetStatusWeightsAndAdoption')}{result.points.length} {" " + systemText('preInvestment.frontierGrid.items')}</summary>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm" aria-label={systemText('preInvestment.frontierGrid.frontierTargetSolverDetails')}>
          <thead><tr>
            <th scope="col" className="p-2 text-left">{systemText('preInvestment.frontierGrid.target')}</th>
            <th scope="col" className="p-2 text-right">{systemText('preInvestment.frontierGrid.target')}{returnLabel}</th>
            <th scope="col" className="p-2 text-right">{systemText('preInvestment.frontierGrid.actual')}{returnLabel}</th>
            <th scope="col" className="p-2 text-right">{riskLabel}</th>
            <th scope="col" className="p-2 text-right">{systemText('preInvestment.frontierGrid.iterations')}</th>
            <th scope="col" className="p-2 text-left">{systemText('preInvestment.frontierGrid.status')}</th>
            <th scope="col" className="p-2 text-left">{discrete ? systemText('preInvestment.frontierGrid.continuousReferenceWeights') : systemText('preInvestment.frontierGrid.weight')}</th>
            {discrete && <th scope="col" className="p-2 text-left">{systemText('preInvestment.frontierGrid.adoptablePortfolioAtSelectedPrecision')}</th>}
            <th scope="col" className="p-2 text-left">{systemText('preInvestment.frontierGrid.actions')}</th>
          </tr></thead>
          <tbody>{result.points.map(point => <tr key={point.target_index} className="border-b border-slate-200">
            <th scope="row" className="p-2 text-left font-medium">{point.target_index + 1}</th>
            <td className="p-2 text-right tabular-nums">{percentText(point.target)}</td>
            <td className="p-2 text-right tabular-nums">{percentText(point.value[1])}</td>
            <td className="p-2 text-right tabular-nums">{percentText(point.value[0])}</td>
            <td className="p-2 text-right tabular-nums">{point.iterations}</td>
            <td className="p-2 text-xs leading-5">{statusText[point.status] ?? point.status}{point.duplicate_of != null ? systemText('preInvestment.frontierGrid.duplicatesTargetSolution', { p0: point.duplicate_of + 1 }) : ''}{point.status === 'converged' && !point.on_frontier ? systemText('preInvestment.frontierGrid.notOnTheFinalFrontier') : ''}</td>
            <td className="p-2 text-xs leading-5">{assetNames.map((name, index) => <span className="block whitespace-nowrap" key={name}>{name}：{percentText(point.weights[index])}</span>)}</td>
            {discrete && <td className="p-2 text-xs leading-5">
              {point.adoption ? <>
                <span className="block">{statusText[point.adoption.status] ?? point.adoption.status}{point.adoption.status === 'feasible' && !point.adoption.target_met ? systemText('preInvestment.frontierGrid.originalReturnTargetNotMet') : ''}</span>
                {point.adoption.status === 'feasible' && <>
                  {assetNames.map((name, index) => <span className="block whitespace-nowrap" key={name}>{name}：{percentText(point.adoption!.weights[index])}</span>)}
                  <span className="block">{returnLabel} {percentText(point.adoption.value[1])} · {riskLabel} {percentText(point.adoption.value[0])}</span>
                </>}
              </> : systemText('preInvestment.frontierGrid.noPortfolioSatisfiesTheSelectedPrecisionYet')}
            </td>}
            <td className="p-2"><Button disabled={point.status !== 'converged' || point.candidate_index == null || (discrete && point.adoption?.status !== 'feasible')}
              aria-label={systemText('preInvestment.frontierGrid.adoptFrontierTarget', { p0: point.target_index + 1 })} onClick={() => onAdopt(point.adoption ? { ...point, weights: point.adoption.weights, value: point.adoption.value } : point)}>{systemText('preInvestment.frontierGrid.adoptWeights')}</Button></td>
          </tr>)}</tbody>
        </table>
      </div>
    </details>
  </section>
}
