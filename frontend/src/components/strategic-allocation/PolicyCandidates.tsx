import { researchMessage } from '../../i18n/researchMessages'
import { useId } from 'react'
import RiskBudgetEditor, { riskBudgetError } from './RiskBudgetEditor'
import MeanUncertaintyFields, { MeanUncertaintySummary } from './MeanUncertaintyFields'
import { GoalCandidateSummary } from '../investment-mandate/MandateResults'
import { Button } from '../ui'
import { Field, inputClass, NumberInput, percentText, sectionClass } from '../risk-models/ResearchUI'
import { percentInputValue, type PolicyCandidate, type PolicyPreview, type PolicyRequest, type RiskAdjustedBasis } from '../../services/strategicAllocation'
import RegimeHelpTip from '../../pages/regime-workbench/RegimeHelpTip'
import CompatibilityResults from './CompatibilityResults'
import CrossModelResults from './CrossModelResults'
import { useI18n, systemText } from '../../i18n/runtime'

const METHOD_HELP: Partial<Record<PolicyCandidate['id'], string>> = {
  'minimum-risk': 'MinimumRisk', 'nominal-utility': 'NominalUtility', 'robust-utility': 'RobustUtility', 'maximum-return': 'MaximumReturn',
  'maximum-sharpe': 'MaximumSharpe', 'minimum-drawdown': 'MinimumDrawdown', 'risk-budget': 'RiskBudget',
}

function riskFreeSource({ risk_free: source }: RiskAdjustedBasis) {
  const key = 'preInvestment.policyCandidates.riskFree'
  return source.source === 'scope_cash' ? systemText(`${key}ScopeCash`, { p0: source.asset_name ?? '' })
    : source.source === 'risk_scale_cash' ? systemText(`${key}RiskScaleCash`, { p0: source.risk_scale?.name ?? '', p1: source.asset_name ?? '' })
      : systemText(`${key}DefaultZero`)
}

/** Plain-language meaning of each selection rule; Sharpe and drawdown also state their frozen basis. */
function methodHelp(id: PolicyCandidate['id'], basis?: RiskAdjustedBasis) {
  const suffix = METHOD_HELP[id]
  if (!suffix) return ''
  const key = `preInvestment.policyCandidates.methodHelp${suffix}`
  if (id === 'maximum-sharpe') return basis ? systemText(key, { p0: percentText(basis.risk_free.rate), p1: riskFreeSource(basis) }) : ''
  if (id === 'minimum-drawdown') return basis ? systemText(key, { p0: basis.drawdown.paths, p1: basis.drawdown.months }) : ''
  return systemText(key)
}

/** `settings` edits bounds and starts a comparison; `results` is the separate comparison page. */
export default function PolicyCandidates({ view, value, assets, assetLabels = {}, result, busy, compareDisabled, compareReason = '', meanCovarianceAvailable = false, onChange, onCompare, onSelect, onBack, onShowResults }: {
  view: 'settings' | 'results'
  value: PolicyRequest; assets: string[]; assetLabels?: Record<string, string>; result: PolicyPreview | null; busy: boolean; compareDisabled: boolean; meanCovarianceAvailable?: boolean
  compareReason?: string
  onChange: (value: PolicyRequest) => void; onCompare: () => void; onSelect: (candidate: PolicyCandidate) => void
  onBack?: () => void; onShowResults?: () => void
}) {
  const { s } = useI18n()
  const compareReasonId = useId()
  const assetName = (id: string) => assetLabels[id] || id
  const common = value.mode === 'compatible_all_models'
  const budgetIssue = riskBudgetError(assets, value.risk_budget)
  const ellipseIssue = value.uncertainty_set === 'ellipsoidal' && (!meanCovarianceAvailable || !value.uncertainty_confidence || Boolean(value.mode && value.mode !== 'single'))
  const adjusted = result?.candidates.some(c => c.risk_adjusted) ?? false
  const metricHeaders = [systemText('preInvestment.policyCandidates.expectedAnnualReturn'), systemText('preInvestment.policyCandidates.expectedAnnualVolatility'), systemText('preInvestment.policyCandidates.conservativeAnnualReturn'),
    ...(adjusted ? [systemText('preInvestment.policyCandidates.sharpeRatio'), systemText('preInvestment.policyCandidates.simulatedMaxDrawdown')] : [])]
  return <div className="space-y-5">
    {view === 'settings' && <section className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.policyCandidates.policyAssetAllocationBounds')}>
      <h2 className="text-lg font-semibold">{systemText('preInvestment.policyCandidates.howMuchCanBeAllocated')}</h2>
      <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.policyCandidates.setMinimumAndMaximumWeightsForEach')}</p>
      <div className="overflow-x-auto"><table aria-label={systemText('preInvestment.policyCandidates.policyAssetConstraints')} className="w-full min-w-[460px] text-sm"><thead><tr><th scope="col" className="p-2 text-left">{systemText('preInvestment.policyCandidates.assetClass')}</th>{[systemText('preInvestment.policyCandidates.minimumWeight'), systemText('preInvestment.policyCandidates.maximumWeight')].map(label => <th scope="col" className="p-2 text-right" key={label}>{label}</th>)}</tr></thead><tbody>{assets.map(asset => <tr key={asset} className="border-t border-slate-200"><th scope="row" className="p-2 text-left font-medium">{assetName(asset)}</th>{(['min_weight', 'max_weight'] as const).map((field, i) => <td className="p-2" key={field}><NumberInput aria-label={`${assetName(asset)}${[systemText('preInvestment.policyCandidates.minimumWeight2'), systemText('preInvestment.policyCandidates.maximumWeight2')][i]}`} className={`${inputClass} text-right tabular-nums`} value={percentInputValue(value.constraints[asset]?.[field])} onValueChange={number => onChange({ ...value, constraints: { ...value.constraints, [asset]: { ...value.constraints[asset], [field]: number / 100 } } })} /></td>)}</tr>)}</tbody></table></div>
      {common && <Field label={s('multiCma.objective')}><select className={inputClass} value={value.compatibility_objective ?? 'minimax_regret'} onChange={event => onChange({ ...value, compatibility_objective: event.target.value as 'minimax_regret' | 'maximin_return' })}>
        <option value="minimax_regret">{s('multiCma.minimax_regret')}</option><option value="maximin_return">{s('multiCma.maximin_return')}</option>
      </select></Field>}
      <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">{systemText('preInvestment.policyCandidates.jointConstraintsAndCandidateSearchSettings')}</summary>
        <div className="mt-4 space-y-4">
          {value.group_limits.map((group, index) => <fieldset className="min-w-0 space-y-3 border-b border-slate-200 pb-4" key={index}><legend className="text-sm font-medium">{systemText('preInvestment.policyCandidates.jointConstraints') + " "}{index + 1}</legend>
            <div className="grid gap-3 sm:grid-cols-3"><Field label={systemText('preInvestment.policyCandidates.jointConstraintName', { p0: index + 1 })}><input className={inputClass} value={group.id} onChange={e => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, id: e.target.value } : g) })} /></Field>
              {(['lo', 'hi'] as const).map(field => <Field key={field} label={systemText('preInvestment.policyCandidates.jointConstraintCoefficient', { p0: index + 1, p1: field === 'lo' ? systemText('preInvestment.policyCandidates.minimum') : systemText('preInvestment.policyCandidates.maximum') })}><NumberInput className={inputClass} value={percentInputValue(group[field])} onValueChange={number => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, [field]: number / 100 } : g) })} /></Field>)}</div>
            <div className="flex flex-wrap gap-4">{assets.map(asset => <label className="flex items-center gap-2 text-sm" key={asset}><input type="checkbox" checked={group.assets.includes(asset)} onChange={e => onChange({ ...value, group_limits: value.group_limits.map((g, i) => i === index ? { ...g, assets: e.target.checked ? [...g.assets, asset] : g.assets.filter(name => name !== asset) } : g) })} />{assetName(asset)}</label>)}</div>
            <Button onClick={() => onChange({ ...value, group_limits: value.group_limits.filter((_, i) => i !== index) })}>{systemText('preInvestment.policyCandidates.removeThisJointConstraint')}</Button>
          </fieldset>)}
          <Button disabled={value.group_limits.length >= 28} onClick={() => onChange({ ...value, group_limits: [...value.group_limits, { id: systemText('preInvestment.policyCandidates.jointConstraint', { p0: value.group_limits.length + 1 }), assets: [], lo: 0, hi: 1 }] })}>{systemText('preInvestment.policyCandidates.addJointConstraint')}</Button>
          <MeanUncertaintyFields value={value} available={meanCovarianceAvailable} onChange={onChange} />
          <div className="grid gap-3 sm:grid-cols-2">
            {!common && <Field label={systemText('preInvestment.policyCandidates.randomCandidateCount')}><NumberInput className={inputClass} value={value.candidate_count} min={200} max={5000} onValueChange={number => onChange({ ...value, candidate_count: number })} /></Field>}
            <Field label={systemText('preInvestment.policyCandidates.randomSeed')}><NumberInput className={inputClass} value={value.seed} min={0} onValueChange={number => onChange({ ...value, seed: number })} /></Field>
          </div>
        </div>
      </details>
      {!common && <RiskBudgetEditor assets={assets} assetLabels={assetLabels} value={value.risk_budget} onChange={risk_budget => onChange({ ...value, risk_budget })} />}
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <Button tone="primary" className="shrink-0 disabled:border-slate-200 disabled:bg-slate-100 disabled:text-slate-600 disabled:opacity-100" aria-describedby={compareReason ? compareReasonId : undefined} disabled={busy || Boolean(budgetIssue) || ellipseIssue || compareDisabled || !assets.length || (value.mode && value.mode !== 'single' ? !value.cma_refs?.length : !value.cma_id) || !value.mandate_id} onClick={onCompare}>{busy ? systemText('preInvestment.policyCandidates.comparing') : systemText('preInvestment.policyCandidates.comparePolicyCandidatesAgainstObjectives')}</Button>
        {result && onShowResults && <Button className="shrink-0" onClick={onShowResults}>{systemText('preInvestment.policyCandidates.viewComparison')}</Button>}
        {compareReason && <p id={compareReasonId} role="status" className={`min-w-0 text-sm leading-6 ${compareDisabled ? 'text-amber-800' : 'text-slate-600'}`}>{compareReason}</p>}
      </div>
    </section>}
    {view === 'results' && result && <section className={`${sectionClass} space-y-4`} aria-label={systemText('preInvestment.policyCandidates.policyCandidateResults')}>
      <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="text-lg font-semibold">{systemText('preInvestment.policyCandidates.compareFirstThenChoose')}</h2>{onBack && <Button onClick={onBack}>{systemText('preInvestment.policyCandidates.backToSettings')}</Button>}</div>
      {result.uncertainty_model && <MeanUncertaintySummary value={result.uncertainty_model} />}
      {result.multi_cma && <p className="text-sm leading-6 text-slate-600">{common ? s('multiCma.commonHint') : `${s('multiCma.hint')} ${s('multiCma.uncertainty')}`}</p>}
      {result.compatibility && <CompatibilityResults evidence={result.compatibility} />}
      {result.candidates.some(c => c.goal_check || c.benchmark_check) && <div className="space-y-3" aria-label={systemText('preInvestment.policyCandidates.investmentObjectiveChecks')}>{result.candidates.map(c => <div className="border-b border-slate-200 pb-3" key={c.id}><h3 className="text-sm font-medium">{c.name}</h3><GoalCandidateSummary candidate={c} /></div>)}</div>}
      {result.unavailable_candidates?.map(c => <div key={c.id} className="space-y-3"><p role="status" className="text-sm text-amber-800">{c.name}{systemText('preInvestment.policyCandidates.unavailable')}{c.unavailable_reason}</p>{c.id === 'compatible' && <CrossModelResults common rows={c.cross_model_results ?? []} />}</div>)}
      {(!common || result.candidates.length > 0) && <p className="text-sm leading-6 text-slate-600">{common ? s('multiCma.summaryBasis') : result.uncertainty_model ? systemText('preInvestment.policyCandidates.theseUseTheSameFeasibleCandidateSet') : systemText('preInvestment.policyCandidates.theseUseTheSameFeasibleCandidateSet2')}</p>}
      {(!common || result.candidates.length > 0) && <div className="overflow-x-auto"><table aria-label={systemText('preInvestment.policyCandidates.longTermPolicyCandidateComparison')} className="w-full min-w-[600px] text-sm"><thead><tr>{[systemText('preInvestment.policyCandidates.candidateMethod'), ...metricHeaders, ...assets, systemText('preInvestment.policyCandidates.next')].map(label => <th key={label} scope="col" className={`whitespace-nowrap p-3 ${label === systemText('preInvestment.policyCandidates.candidateMethod') || label === systemText('preInvestment.policyCandidates.next') ? 'text-left' : 'text-right'}`}>{assetName(label)}</th>)}</tr></thead><tbody>{result.candidates.map(candidate => <tr className="border-t border-slate-200" key={candidate.id}><th scope="row" className="whitespace-nowrap p-3 text-left font-medium">{researchMessage(candidate.name)}{methodHelp(candidate.id, result.risk_adjusted_basis) && <RegimeHelpTip label={systemText('preInvestment.policyCandidates.methodHelpLabel', { p0: researchMessage(candidate.name) })} text={methodHelp(candidate.id, result.risk_adjusted_basis)} />}</th>{[candidate.metrics.expected_return, candidate.metrics.volatility, candidate.metrics.conservative_return].map((number, i) => <td key={i} className="whitespace-nowrap p-3 text-right tabular-nums">{percentText(number)}</td>)}{adjusted && <><td className="whitespace-nowrap p-3 text-right tabular-nums">{candidate.risk_adjusted?.sharpe_ratio == null ? '—' : candidate.risk_adjusted.sharpe_ratio.toFixed(2)}</td><td className="whitespace-nowrap p-3 text-right tabular-nums">{percentText(candidate.risk_adjusted?.mean_max_drawdown)}</td></>}{assets.map(asset => <td key={asset} className="whitespace-nowrap p-3 text-right tabular-nums">{percentText(candidate.weights[asset])}</td>)}<td className="p-3"><Button disabled={busy || candidate.available === false || common && candidate.all_models_pass !== true} onClick={() => onSelect(candidate)}>{systemText('preInvestment.policyCandidates.reviewThisCandidate')}</Button></td></tr>)}</tbody></table></div>}
      {result.multi_cma && result.candidates.length > 0 && <section className="min-w-0 space-y-4" aria-label={s('multiCma.crossTitle')}><h3 className="font-semibold">{s('multiCma.crossTitle')}</h3>{result.candidates.map(candidate => <div key={candidate.id} className="min-w-0 space-y-2 border-b border-slate-200 pb-4"><h4 className="text-sm font-medium">{researchMessage(candidate.name)}</h4><CrossModelResults common={common} rows={candidate.cross_model_results ?? []} /></div>)}</section>}
      <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">{systemText('preInvestment.policyCandidates.riskContributionsAndEvidenceLimitations')}</summary><div className="mt-3 space-y-3"><p className="text-sm text-slate-600">{systemText('preInvestment.policyCandidates.riskContributionsRetainTheirSignsNegativeValues')}</p>{result.candidates.map(candidate => <p className="text-sm leading-6 text-slate-600" key={candidate.id}>{researchMessage(candidate.name)}：{assets.map(asset => `${assetName(asset)} ${percentText(candidate.risk_contributions[asset])}`).join('；')}{candidate.risk_budget_distance !== undefined && systemText('preInvestment.policyCandidates.squaredRiskBudgetDistance', { p0: candidate.risk_budget_distance.toPrecision(5) })}</p>)}{result.warnings.map((warning, index) => <p key={index} className="text-xs leading-5 text-slate-600">{warning}</p>)}</div></details>
    </section>}
  </div>
}
