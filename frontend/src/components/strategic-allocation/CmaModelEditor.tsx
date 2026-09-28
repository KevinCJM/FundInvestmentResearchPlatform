import { systemText, useI18n } from '../../i18n/runtime'
import { Button, SectionHeader } from '../ui'
import BlackLittermanViews from './BlackLittermanViews'
import { Field, inputClass, NumberInput, percentText } from '../risk-models/ResearchUI'
import { percentInputValue } from '../../services/strategicAllocation'
import { cmaModelInputError, type CmaModelContext, type CmaModelRequest, type CmaScenario } from '../../services/cmaModelTypes'

const input = `${inputClass} !rounded-lg placeholder:text-slate-600 placeholder:opacity-100 focus-visible:ring-2 focus-visible:ring-accent-500`
const numeric = `${input} tabular-nums`
const button = 'active:translate-y-px motion-reduce:transform-none motion-reduce:transition-none'
const blankMatrix = (count: number) => Array.from({ length: count }, () => Array<number>(count).fill(NaN))

export interface CmaModelEditorProps {
  context: Pick<CmaModelContext, 'asset_ids' | 'as_of' | 'currency'>
  /** null means the existing manual CMA form; incomplete new draft numbers are NaN. */
  value: CmaModelRequest | null
  onChange: (value: CmaModelRequest | null) => void
  onPreview?: (value: CmaModelRequest) => void
  busy?: boolean
  readOnly?: boolean
  disabledReason?: string
  error?: string
  onCopy?: () => void
  hideMethodChoice?: boolean
  assetLabels?: Record<string, string>
}

function CovarianceEditor({ label, axis, value, onChange, assetLabels = {} }: {
  assetLabels?: Record<string, string>
  label: string; axis: string[]; value: number[][] | null | undefined; onChange: (matrix: number[][]) => void
}) {
  useI18n()
  function edit(row: number, column: number, number: number) {
    onChange(axis.map((_, i) => axis.map((__, j) => (i === row && j === column) || (i === column && j === row) ? number : value?.[i]?.[j] ?? NaN)))
  }
  return <details className="min-w-0 space-y-3">
    <summary className="cursor-pointer py-2 text-sm font-semibold text-slate-900">{label}</summary>
    <p className="text-xs leading-5 text-slate-600">{systemText('preInvestment.cmaModelEditor.unitSquaredAnnualDecimalReturnsForExample')}</p>
    <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">{axis.flatMap((asset, i) => axis.slice(i).map((other, offset) => <Field key={`${i}-${offset}`} label={`${label}：${assetLabels[asset] ?? asset} / ${assetLabels[other] ?? other}`}>
      <NumberInput className={numeric} value={value?.[i]?.[i + offset] ?? NaN} onValueChange={number => edit(i, i + offset, number)} />
    </Field>))}</div>
  </details>
}

/** Controlled editor only: no fetch, persistence, inferred market data, or model calculations. */
export default function CmaModelEditor({ context, value, onChange, onPreview, busy = false, readOnly = false, disabledReason, error, onCopy, hideMethodChoice = false, assetLabels = {} }: CmaModelEditorProps) {
  useI18n()
  const sameContext = !value || value.as_of === context.as_of && value.currency === context.currency && JSON.stringify(value.asset_ids) === JSON.stringify(context.asset_ids)
  const reason = disabledReason || (readOnly ? systemText('preInvestment.cmaModelEditor.thisIsASavedVersionWithRead') : !sameContext ? systemText('preInvestment.cmaModelEditor.theAssetScopeDateOrCurrencyChanged') : '')
  const validation = value ? cmaModelInputError(value) : null
  const changeMethod = (method: string) => {
    if (method === 'manual') { onChange(null); return }
    const common = { ...context, asset_ids: [...context.asset_ids], source: '', return_basis: 'annual_arithmetic_total_return' as const }
    onChange(method === 'black_litterman'
      ? { ...common, method, covariance: blankMatrix(context.asset_ids.length), risk_covariance_basis: 'input_covariance', market_weights: Object.fromEntries(context.asset_ids.map(a => [a, NaN])), market_weight_source: '', delta: NaN, tau: 0.05, risk_free_rate: NaN, views: [] }
      : { ...common, method: 'scenario_mixture', risk_mode: 'shared', shared_covariance: blankMatrix(context.asset_ids.length), scenarios: [] })
  }
  function patchScenario(index: number, patch: Partial<CmaScenario>) {
    if (value?.method === 'scenario_mixture') onChange({ ...value, scenarios: value.scenarios.map((s, i) => i === index ? { ...s, ...patch } : s) })
  }
  return <section aria-label={systemText('preInvestment.cmaModelEditor.cmaGenerationMethod')} aria-busy={busy} className="min-w-0 space-y-4 break-words text-slate-900">
    {!hideMethodChoice && <SectionHeader title={systemText('preInvestment.cmaModelEditor.howToGenerateLongTermAssumptions')} description={systemText('preInvestment.cmaModelEditor.usesTheSameAssetScopeAndAnnual')} />}
    {!hideMethodChoice && <Field label={systemText('preInvestment.cmaModelEditor.expectedReturnGenerationMethod')}><select className={input} disabled={busy || readOnly || !!disabledReason || !context.asset_ids.length} value={value?.method ?? 'manual'} onChange={event => changeMethod(event.target.value)}>
      <option value="manual">{systemText('preInvestment.cmaModelEditor.manualInputs')}</option><option value="black_litterman">{systemText('preInvestment.cmaModelEditor.marketBaselinePlusViews')}</option><option value="scenario_mixture">{systemText('preInvestment.cmaModelEditor.multipleScenarios')}</option>
    </select></Field>}
    {!context.asset_ids.length && <p role="status" className="text-sm text-slate-600">{systemText('preInvestment.cmaModelEditor.assetScopeNotLoadedSelectOrConfirm')}</p>}
    {reason && <p role="status" className="text-sm leading-6 text-slate-600">{reason}</p>}
    {readOnly && onCopy && <Button className={button} onClick={onCopy} disabled={busy}>{systemText('preInvestment.cmaModelEditor.copyAsNewResearch')}</Button>}
    {busy && <div role="status" className="space-y-2 text-sm text-slate-600"><div aria-hidden="true" className="h-3 w-2/3 rounded-lg bg-slate-200" /><div aria-hidden="true" className="h-3 w-1/2 rounded-lg bg-slate-200" />{systemText('preInvestment.cmaModelEditor.checkingModelInputsAndCalculatingPreview')}</div>}
    {!value && <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.cmaModelEditor.continueEnteringReturnsVolatilityAndCorrelationsIn')}</p>}
    {value && <fieldset disabled={busy || !!reason} className="min-w-0 space-y-4">
      <legend className="sr-only">{systemText('preInvestment.cmaModelEditor.modelInputs')}</legend>
      <p className="break-words text-sm text-slate-600">{value.as_of} · {value.currency} {" " + systemText('preInvestment.cmaModelEditor.annualArithmeticTotalReturn')}</p>
      {!hideMethodChoice && <Field label={systemText('preInvestment.cmaModelEditor.modelAndRiskEvidence')}><textarea className={input} rows={2} maxLength={2000} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field>}
      {value.method === 'black_litterman' ? <>
        <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.cmaModelEditor.provideMarketWeightsExplicitlyWithoutViewsThe')}</p>
        <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">{value.asset_ids.map(asset => <Field key={asset} label={systemText('preInvestment.cmaModelEditor.marketWeight', { p0: assetLabels[asset] ?? asset })}><NumberInput className={numeric} value={percentInputValue(value.market_weights[asset])} onValueChange={n => onChange({ ...value, market_weights: { ...value.market_weights, [asset]: n / 100 } })} /></Field>)}</div>
        <p className="text-sm tabular-nums text-slate-600">{systemText('preInvestment.cmaModelEditor.totalMarketWeight')}{percentText(Object.values(value.market_weights).reduce((sum, n) => sum + n, 0))}</p>
        <Field label={systemText('preInvestment.cmaModelEditor.marketWeightSource')}><input className={input} maxLength={2000} value={value.market_weight_source} onChange={e => onChange({ ...value, market_weight_source: e.target.value })} /></Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label={systemText('preInvestment.cmaModelEditor.marketRiskAversion')} hint={systemText('preInvestment.cmaModelEditor.finitePositiveNumberDimensionless')}><NumberInput aria-label={systemText('preInvestment.cmaModelEditor.marketRiskAversion')} className={numeric} value={value.delta} onValueChange={delta => onChange({ ...value, delta })} /></Field>
          <Field label={systemText('preInvestment.cmaModelEditor.annualRiskFreeReturn')}><NumberInput className={numeric} value={percentInputValue(value.risk_free_rate)} onValueChange={n => onChange({ ...value, risk_free_rate: n / 100 })} /></Field>
        </div>
        <CovarianceEditor assetLabels={assetLabels} label={systemText('preInvestment.cmaModelEditor.assetRiskCovariance')} axis={value.asset_ids} value={value.covariance} onChange={covariance => onChange({ ...value, covariance })} />
        <details><summary className="cursor-pointer py-2 text-sm font-semibold">{systemText('preInvestment.cmaModelEditor.advancedSettings')}</summary><Field label={systemText('preInvestment.cmaModelEditor.priorMeanUncertaintyCoefficient')} hint={systemText('preInvestment.cmaModelEditor.finitePositiveNumberScalesOnlyPriorMean')}><NumberInput className={numeric} value={value.tau} onValueChange={tau => onChange({ ...value, tau })} /></Field></details>
        <BlackLittermanViews assetLabels={assetLabels} value={value} onChange={onChange} />
      </> : value.method === 'scenario_mixture' ? <>
        <p className="text-sm leading-6 text-slate-600">{systemText('preInvestment.cmaModelEditor.onlyScenariosWithExplicitProbabilitiesCanBe')}</p>
        <Field label={systemText('preInvestment.cmaModelEditor.scenarioRiskConvention')}><select className={input} value={value.risk_mode} onChange={e => onChange({ ...value, risk_mode: e.target.value as 'shared' | 'scenario_specific', shared_covariance: e.target.value === 'shared' ? blankMatrix(value.asset_ids.length) : null, scenarios: value.scenarios.map(s => ({ ...s, covariance: e.target.value === 'shared' ? null : blankMatrix(value.asset_ids.length) })) })}><option value="shared">{systemText('preInvestment.cmaModelEditor.explicitlyShareOneRiskMatrix')}</option><option value="scenario_specific">{systemText('preInvestment.cmaModelEditor.provideARiskMatrixPerScenario')}</option></select></Field>
        <p className="text-xs text-slate-600">{systemText('preInvestment.cmaModelEditor.afterChangingTheRiskConventionEnterThe')}</p>
        {value.risk_mode === 'shared' && <CovarianceEditor assetLabels={assetLabels} label={systemText('preInvestment.cmaModelEditor.sharedRiskCovariance')} axis={value.asset_ids} value={value.shared_covariance} onChange={shared_covariance => onChange({ ...value, shared_covariance })} />}
        <p role="status" className="text-sm tabular-nums text-slate-600">{systemText('preInvestment.cmaModelEditor.totalScenarioProbability')}{value.scenarios.length ? percentText(value.scenarios.reduce((sum, s) => sum + s.probability, 0)) : systemText('preInvestment.cmaModelEditor.notProvided')}</p>
        {!value.scenarios.length && <p className="text-sm text-slate-600">{systemText('preInvestment.cmaModelEditor.noScenariosYetAddScenariosAndEnter')}</p>}
        <div className="space-y-4 divide-y divide-slate-200">{value.scenarios.map((scenario, index) => <fieldset key={index} className="min-w-0 space-y-3 pt-4">
          <legend className="text-sm font-semibold">{systemText('preInvestment.cmaModelEditor.scenario') + " "}{index + 1}</legend>
          <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Field label={systemText('preInvestment.cmaModelEditor.scenarioName', { p0: index + 1 })}><input className={input} maxLength={120} value={scenario.id} onChange={e => patchScenario(index, { id: e.target.value })} /></Field>
            <Field label={systemText('preInvestment.cmaModelEditor.scenarioProbability', { p0: index + 1 })}><NumberInput className={numeric} value={percentInputValue(scenario.probability)} onValueChange={n => patchScenario(index, { probability: n / 100 })} /></Field>
            <Field label={systemText('preInvestment.cmaModelEditor.scenarioRationale', { p0: index + 1 })}><input className={input} maxLength={2000} value={scenario.source} onChange={e => patchScenario(index, { source: e.target.value })} /></Field>
            {value.asset_ids.map(asset => <Field key={asset} label={systemText('preInvestment.cmaModelEditor.scenarioAnnualReturn', { p0: index + 1, p1: assetLabels[asset] ?? asset })}><NumberInput className={numeric} value={percentInputValue(scenario.annual_returns[asset])} onValueChange={n => patchScenario(index, { annual_returns: { ...scenario.annual_returns, [asset]: n / 100 } })} /></Field>)}
          </div>
          {value.risk_mode === 'scenario_specific' && <CovarianceEditor assetLabels={assetLabels} label={systemText('preInvestment.cmaModelEditor.scenarioRiskCovariance', { p0: index + 1 })} axis={value.asset_ids} value={scenario.covariance} onChange={covariance => patchScenario(index, { covariance })} />}
          <Button className={button} onClick={() => onChange({ ...value, scenarios: value.scenarios.filter((_, i) => i !== index) })}>{systemText('preInvestment.cmaModelEditor.removeScenario') + " "}{index + 1}</Button>
        </fieldset>)}</div>
        <Button className={button} disabled={value.scenarios.length >= 60} onClick={() => onChange({ ...value, scenarios: [...value.scenarios, { id: '', probability: NaN, annual_returns: Object.fromEntries(value.asset_ids.map(a => [a, NaN])), covariance: value.risk_mode === 'shared' ? null : blankMatrix(value.asset_ids.length), source: '' }] })}>{systemText('preInvestment.cmaModelEditor.addScenario')}</Button>
        {value.scenarios.length >= 60 && <p className="text-xs text-slate-600">{systemText('preInvestment.cmaModelEditor.theLimitOf60ScenariosHasBeen')}</p>}
      </> : null}
    </fieldset>}
    {value && validation && !readOnly && <p role="status" className="text-sm leading-6 text-amber-900">{validation}</p>}
    {error && <p role="alert" className="text-sm leading-6 text-rose-700">{error}</p>}
    {value && onPreview && <Button tone="primary" className={button} disabled={busy || !!reason || !!validation} onClick={() => { if (!busy && !reason && !cmaModelInputError(value)) onPreview(value) }}>{error ? systemText('preInvestment.cmaModelEditor.retryModelPreview') : systemText('preInvestment.cmaModelEditor.previewModelAssumptions')}</Button>}
  </section>
}
