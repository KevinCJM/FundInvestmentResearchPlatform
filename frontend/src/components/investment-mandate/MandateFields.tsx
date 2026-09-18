import { useEffect, useState } from 'react'
import { Field, inputClass, NumberInput, percentText } from '../risk-models/ResearchUI'
import { metadata, riskScales, textValue, type ReferenceAsset, type VersionView } from '../../services/riskScales'
import { percentInputValue as percent, type MandateDefinition, type MandateFundingEcho, type ObjectiveKind } from '../../services/strategicAllocation'
import { CUSTOM_BENCHMARK, lastPaymentMonth, levelReaches, newCashBudget } from './model'
import { useMandateText } from './text'

type Props = { value: MandateDefinition; onChange: (patch: Partial<MandateDefinition>) => void }

/** A confirmed reference version carries its definition untyped; read just the asset list. */
const referenceAssets = (definition: unknown): ReferenceAsset[] => {
  const assets = (definition as { assets?: unknown } | null)?.assets
  return Array.isArray(assets) ? assets as ReferenceAsset[] : []
}

export function TaskFields({ value, onChange, cutoff, pitLocked, pitLabel }: Props & {
  cutoff: string; pitLocked: boolean; pitLabel: string
}) {
  const { t } = useMandateText()
  const changeHorizon = (years: number) => onChange({ horizon_years: years,
    ...(value.cash_budget && Number.isInteger(years) && years >= 1 && years <= 30 ? {
      cash_budget: { ...value.cash_budget, flows: value.cash_budget.flows.map(flow => flow.last_month === flow.first_month
        ? flow : { ...flow, last_month: lastPaymentMonth(flow.first_month, flow.every_months, years * 12) }) },
    } : {}) })
  return <div className="space-y-5">
    <div className="grid gap-4 sm:grid-cols-2">
      <Field label={t('name')}><input className={inputClass} value={value.name} maxLength={120} onChange={e => onChange({ name: e.target.value })} /></Field>
      <Field label={t('researchDate')} hint={pitLocked ? t('pitDateLocked') : t('pitDateManual')}>
        <input className={inputClass} type="date" value={value.as_of} max={cutoff} disabled={pitLocked} onChange={e => onChange({ as_of: e.target.value })} />
      </Field>
      <Field label={t('horizon')} hint={t('horizonHint')}><NumberInput aria-label={t('horizon')} className={inputClass} value={value.horizon_years} min={1} max={30} onValueChange={changeHorizon} /></Field>
      <Field label={t('reviewDate')} hint={t('reviewDateOptional')}><input className={inputClass} type="date" min={value.as_of} value={value.review_date ?? ''} onChange={e => onChange({ review_date: e.target.value || null })} /></Field>
    </div>
    <p className="text-xs leading-5 text-slate-600">{pitLabel}</p>
  </div>
}

export function GoalFields({ value, onChange, version }: Props & { version: VersionView | null }) {
  const { t } = useMandateText()
  const [basis, setBasis] = useState<'annual' | 'total'>('annual')
  const kind = value.objective_kind ?? 'absolute_return'
  // Same number, two ways to say it. Only the annual value is stored; the conversion is geometric.
  const years = Number.isInteger(value.horizon_years) && value.horizon_years >= 1 ? value.horizon_years : 1
  const totalReturn = Number.isFinite(value.target_return) ? (1 + value.target_return) ** years - 1 : NaN
  const setTotal = (number: number) => onChange({ target_return: (1 + number / 100) ** (1 / years) - 1 })
  const levels = version?.preview.result.levels ?? []
  const selectedLevel = value.risk_authorization?.selected_max_level ?? null
  const portrait = selectedLevel ? levels[selectedLevel - 1] : undefined
  const changeKind = (objective: ObjectiveKind) => {
    // Budgets are shared facts; changing a success criterion must not erase them.
    const cash = value.cash_budget ?? (objective === 'funding_goal' ? newCashBudget(value.as_of) : null)
    // A relative objective is measured against the level's representative portfolio; a
    // level without one cannot carry it, so the choice goes back to the researcher.
    const keepLevel = objective !== 'benchmark_relative' || !portrait || portrait.representative_node_id != null
    onChange({
      objective_kind: objective,
      ...keepLevel ? {} : { risk_authorization: { ...value.risk_authorization!, authorized_max_level: null, selected_max_level: null } },
      target_return: objective === 'absolute_return' ? NaN : 0,
      target_excess_return: objective === 'benchmark_relative' ? NaN : 0,
      cash_budget: cash,
      funding_target: objective === 'funding_goal' ? { amount: NaN, amount_basis: 'nominal' } : null,
      cash_protection: objective === 'funding_goal' ? null : value.cash_protection ?? null,
      benchmark: null,
      stated_benchmark: '',
    })
  }
  // A relative objective measures against the frozen C-level portfolio or against a
  // weighting stated on the same asset axis; both are weight vectors on that axis.
  const assetIds = version?.preview.result.ordered_asset_ids ?? []
  const assetNames = metadata(version?.preview.result.diagnostics?.asset_names)
  const custom = value.benchmark?.source === 'risk_scale_reference' && value.benchmark.name === CUSTOM_BENCHMARK
  // 大类不是一个抽象名字，它就是参考输入里一篮子指数或产品；给它加权之前先让人看见里头是什么。
  const [composition, setComposition] = useState<ReferenceAsset[]>([])
  const referenceId = version?.preview.request_echo.definition.reference_input_ref?.id ?? null
  useEffect(() => {
    if (!custom || !referenceId) return
    const controller = new AbortController()
    riskScales.reference(referenceId, controller.signal)
      .then(result => { if (!controller.signal.aborted) setComposition(referenceAssets(result.definition)) })
      .catch(() => { /* 成分只是解释性信息，取不到就仍然按大类名称加权 */ })
    return () => controller.abort()
  }, [custom, referenceId])
  const madeOf = (id: string) => {
    const asset = composition.find(item => item.id === id)
    if (!asset) return ''
    if (asset.asset_type === 'cash') return t('benchmarkCashBasis', { rate: percentText(asset.cash_return ?? 0) })
    const items = asset.components.map(part => `${part.series_id.split(':').pop() || part.series_id} ${percentText(part.weight)}`)
    return items.length ? t('benchmarkComponents', { items: items.join(' · ') }) : ''
  }
  const referenceWeights = portrait?.representative_weights ?? null
  const writeBenchmark = (weights: Record<string, number> | null) => onChange({ benchmark: weights ? {
    name: CUSTOM_BENCHMARK, alloc_name: 'risk-scale-reference', weights,
    target_excess_return: Number.isFinite(value.target_excess_return) ? value.target_excess_return : 0,
    max_tracking_error: 1, source: 'risk_scale_reference' } : null })
  // 参考组合权重直接来自求解器，带 1e-47 量级残差；按展示精度取整后把余额并回最大的一项，合计仍是 100%。
  const seeded = () => {
    if (!referenceWeights) return null
    const weights = referenceWeights.map(weight => Math.round(weight * 10000) / 10000)
    const top = weights.indexOf(Math.max(...weights))
    weights[top] = Number((1 - weights.reduce((sum, weight, index) => index === top ? sum : sum + weight, 0)).toFixed(6))
    return weights
  }
  const startCustom = () => {
    const seed = seeded()
    writeBenchmark(Object.fromEntries(assetIds.map((id, index) => [id, seed?.[index] ?? (index === 0 ? 1 : 0)])))
  }
  const weightTotal = custom ? Object.values(value.benchmark!.weights).reduce((sum, weight) => sum + weight, 0) : 1
  return <div className="space-y-5">
    <Field label={t('objectiveType')} hint={t('objectiveTypeHint')}><select className={inputClass} value={kind} onChange={e => changeKind(e.target.value as ObjectiveKind)}>
      {(['absolute_return', 'funding_goal', 'benchmark_relative'] as const).map(id => <option key={id} value={id}>{t(id)}</option>)}
    </select></Field>
    {kind === 'absolute_return' && <><div className="grid gap-4 sm:grid-cols-2">
      <Field label={basis === 'annual' ? t('targetReturn') : t('targetTotalReturn', { years })} hint={t('returnHint')}>
        <NumberInput className={inputClass} value={percent(basis === 'annual' ? value.target_return : totalReturn)}
          onValueChange={n => basis === 'annual' ? onChange({ target_return: n / 100 }) : setTotal(n)} /></Field>
      <Field label={t('returnBasis')} hint={t('returnBasisHint')}><select className={inputClass} value={basis} onChange={e => setBasis(e.target.value as 'annual' | 'total')}>
        <option value="annual">{t('annualBasis')}</option><option value="total">{t('totalBasis')}</option></select></Field>
    </div>
    {Number.isFinite(value.target_return) && <p className="text-xs leading-5 text-slate-600">{t('returnConversion',
      { annual: percentText(value.target_return), years, total: percentText(totalReturn) })}</p>}
</>}
    {kind === 'funding_goal' && <p className="text-sm leading-6 text-slate-600">{t('fundingGoalHint')}</p>}
    {kind === 'benchmark_relative' && <>
      <Field label={t('excessReturn')} hint={t('referenceBenchmarkHint')}><NumberInput className={inputClass} value={percent(value.target_excess_return)}
        onValueChange={n => onChange({ target_excess_return: n / 100,
          ...value.benchmark ? { benchmark: { ...value.benchmark, target_excess_return: n / 100 } } : {} })} /></Field>
      <Field label={t('statedBenchmark')} hint={t('statedBenchmarkHint')}>
        <input className={inputClass} maxLength={500} value={value.stated_benchmark ?? ''}
          onChange={e => onChange({ stated_benchmark: e.target.value })} /></Field>
      <fieldset className="min-w-0">
        <legend className="text-sm font-medium text-slate-800">{t('benchmarkSource')}</legend>
        <p className="mt-1 text-xs leading-5 text-slate-600">{t('benchmarkSourceHint')}</p>
        {!version ? <p className="mt-3 text-sm text-slate-600">{t('chooseScaleFirst')}</p> : <>
          <select className={`${inputClass} mt-3`} value={custom ? 'custom' : 'reference'}
            onChange={event => event.target.value === 'custom' ? startCustom() : writeBenchmark(null)}>
            <option value="reference">{selectedLevel ? t('benchmarkFromLevel', { level: selectedLevel }) : t('benchmarkFromScale')}</option>
            <option value="custom">{t('benchmarkCustom')}</option>
          </select>
          {custom && <><ul className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{assetIds.map(id =>
            <li key={id}><Field label={textValue(assetNames[id]) || id}><NumberInput className={inputClass} value={percent(value.benchmark!.weights[id])}
              min={0} max={100} onValueChange={number => writeBenchmark({ ...value.benchmark!.weights, [id]: number / 100 })} /></Field>
              {madeOf(id) && <p className="mt-1 break-words text-xs leading-5 text-slate-600">{madeOf(id)}</p>}</li>)}
          </ul>
          <p className={`mt-2 text-sm leading-6 ${Math.abs(weightTotal - 1) > 1e-8 ? 'text-amber-800' : 'text-slate-700'}`}>
            {t('benchmarkWeightTotal', { total: percentText(weightTotal) })}</p></>}
        </>}
        {(value.stated_benchmark ?? '').trim() && <p role="status" className="mt-3 text-sm leading-6 text-amber-800">{t('statedBenchmarkProxy')}</p>}
      </fieldset>
    </>}
  </div>
}

/**
 * The last thing on the page: what annual return this whole configuration demands, and
 * whether the chosen level's frozen reference portfolio delivers it. Cash flows move the
 * demand, so the number is read from the server's funding kernel, never computed here.
 */
export function ReturnCheck({ value, version, funding, pending }: {
  value: MandateDefinition; version: VersionView | null; funding: MandateFundingEcho | null; pending: boolean
}) {
  const { t } = useMandateText()
  const levels = version?.preview.result.levels ?? []
  const selectedLevel = value.risk_authorization?.selected_max_level ?? null
  const portrait = selectedLevel ? levels[selectedLevel - 1] : undefined
  const kind = value.objective_kind ?? 'absolute_return'
  const stated = kind === 'absolute_return' && Number.isFinite(value.target_return) ? value.target_return : null
  const solved = funding?.funding?.cashflow_required_return
  const cashflow = typeof solved === 'number' && Number.isFinite(solved) ? solved : null
  // An absolute target is floored by the cash flows (the server takes the larger); the
  // other objectives state no arithmetic return, so the flows alone say what is needed.
  // A ledger with no echo back yet cannot be read as "no cash flows"; that would quote a
  // target the money does not actually support.
  const waiting = Boolean(value.cash_budget) && !funding
  const solution = waiting ? null : kind === 'absolute_return' ? funding?.effective_target_return ?? stated : cashflow
  // 二分求解出的 0% 带 1e-16 量级残差，显示成 −0.00% 没有意义；低于展示精度就按 0 读。
  const required = solution != null && Math.abs(solution) < 5e-5 ? 0 : solution
  const raised = stated != null && required != null && required > stated + 1e-10
  const sufficient = required == null ? -1 : levels.findIndex(level => levelReaches(level.expected_return.value, required) === true)
  const reaches = levelReaches(portrait?.expected_return.value, required)
  const verdict = !portrait ? t('returnCheckNeedsLevel')
    : reaches === true ? t('levelReachesRequired', { level: selectedLevel ?? '', reference: percentText(portrait.expected_return.value) })
      : reaches === false ? sufficient >= 0
        ? t('targetNeedsHigherLevel', { target: percentText(required), level: selectedLevel ?? '',
            reference: percentText(portrait.expected_return.value), suggested: sufficient + 1 })
        : t('targetAboveAllLevels', { target: percentText(required) })
        : ''
  return <section aria-label={t('returnCheck')} className="rounded-xl border border-slate-200 bg-slate-50 p-4">
    <h4 className="text-sm font-semibold text-slate-900">{t('returnCheck')}</h4>
    {funding?.funding?.cashflow_required_return_status === 'above_search_bound'
      ? <p role="status" className="mt-2 text-sm leading-6 text-amber-800">{t('cashflowReturnUnreachable')}</p>
      : required == null
        ? <p className="mt-2 text-sm leading-6 text-slate-600">{t(pending || waiting ? 'returnCheckPending' : 'returnCheckNotApplicable')}</p>
        : <>
          <p className={`mt-2 text-sm leading-6 ${raised ? 'text-amber-800' : 'text-slate-900'}`}>
            {raised ? t('returnRaisedByCashflow', { required: percentText(required), stated: percentText(stated) })
              : t(cashflow == null ? 'requiredReturnFromTarget' : 'requiredReturnFromCashflow', { required: percentText(required) })}</p>
          {verdict && <p role="status" className={`mt-1 text-sm leading-6 ${reaches === true ? 'text-slate-700' : 'text-amber-800'}`}>{verdict}</p>}
          {cashflow != null && <p className="mt-1 text-xs leading-5 text-slate-600">{t('cashflowReturnBasis')}</p>}
        </>}
  </section>
}
