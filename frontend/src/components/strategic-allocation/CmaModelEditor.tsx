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
}

function CovarianceEditor({ label, axis, value, onChange }: {
  label: string; axis: string[]; value: number[][] | null | undefined; onChange: (matrix: number[][]) => void
}) {
  function edit(row: number, column: number, number: number) {
    onChange(axis.map((_, i) => axis.map((__, j) => (i === row && j === column) || (i === column && j === row) ? number : value?.[i]?.[j] ?? NaN)))
  }
  return <details className="min-w-0 space-y-3">
    <summary className="cursor-pointer py-2 text-sm font-semibold text-slate-900">{label}</summary>
    <p className="text-xs leading-5 text-slate-600">单位：年化小数收益的平方。例如20%波动对应对角值0.04。编辑上三角会同步对称位置；缺项保持空白，矩阵有效性由服务端核验。</p>
    <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">{axis.flatMap((asset, i) => axis.slice(i).map((other, offset) => <Field key={`${i}-${offset}`} label={`${label}：${asset} / ${other}`}>
      <NumberInput className={numeric} value={value?.[i]?.[i + offset] ?? NaN} onValueChange={number => edit(i, i + offset, number)} />
    </Field>))}</div>
  </details>
}

/** Controlled editor only: no fetch, persistence, inferred market data, or model calculations. */
export default function CmaModelEditor({ context, value, onChange, onPreview, busy = false, readOnly = false, disabledReason, error, onCopy, hideMethodChoice = false }: CmaModelEditorProps) {
  const sameContext = !value || value.as_of === context.as_of && value.currency === context.currency && JSON.stringify(value.asset_ids) === JSON.stringify(context.asset_ids)
  const reason = disabledReason || (readOnly ? '这是已保存版本，模型输入只读；复制为新研究后编辑。' : !sameContext ? '资产范围、日期或币种已变化；请重新选择方法建立新研究。' : '')
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
  return <section aria-label="CMA生成方法" aria-busy={busy} className="min-w-0 space-y-4 break-words text-slate-900">
    {!hideMethodChoice && <SectionHeader title="长期假设的生成方法" description="使用相同资产范围与年化算术总收益口径；模型预览不会保存版本。" />}
    {!hideMethodChoice && <Field label="预期生成方法"><select className={input} disabled={busy || readOnly || !!disabledReason || !context.asset_ids.length} value={value?.method ?? 'manual'} onChange={event => changeMethod(event.target.value)}>
      <option value="manual">直接填写</option><option value="black_litterman">市场基准＋观点</option><option value="scenario_mixture">多情景假设</option>
    </select></Field>}
    {!context.asset_ids.length && <p role="status" className="text-sm text-slate-600">尚未载入资产范围，请先选择或确认战略资产。</p>}
    {reason && <p role="status" className="text-sm leading-6 text-slate-600">{reason}</p>}
    {readOnly && onCopy && <Button className={button} onClick={onCopy} disabled={busy}>复制为新研究</Button>}
    {busy && <div role="status" className="space-y-2 text-sm text-slate-600"><div aria-hidden="true" className="h-3 w-2/3 rounded-lg bg-slate-200" /><div aria-hidden="true" className="h-3 w-1/2 rounded-lg bg-slate-200" />正在核验模型输入并计算预览…</div>}
    {!value && <p className="text-sm leading-6 text-slate-600">继续使用原长期假设表单填写收益、波动和相关性。</p>}
    {value && <fieldset disabled={busy || !!reason} className="min-w-0 space-y-4">
      <legend className="sr-only">模型输入</legend>
      <p className="break-words text-sm text-slate-600">{value.as_of} · {value.currency} · 年化算术总收益</p>
      {!hideMethodChoice && <Field label="模型与风险依据"><textarea className={input} rows={2} maxLength={2000} value={value.source} onChange={e => onChange({ ...value, source: e.target.value })} /></Field>}
      {value.method === 'black_litterman' ? <>
        <p className="text-sm leading-6 text-slate-600">市场权重必须由你明确提供。没有观点时，结果等于市场均衡先验加无风险收益；资产风险沿用下方输入协方差。</p>
        <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">{value.asset_ids.map(asset => <Field key={asset} label={`${asset}市场权重（%）`}><NumberInput className={numeric} value={percentInputValue(value.market_weights[asset])} onValueChange={n => onChange({ ...value, market_weights: { ...value.market_weights, [asset]: n / 100 } })} /></Field>)}</div>
        <p className="text-sm tabular-nums text-slate-600">市场权重合计：{percentText(Object.values(value.market_weights).reduce((sum, n) => sum + n, 0))}</p>
        <Field label="市场权重来源"><input className={input} maxLength={2000} value={value.market_weight_source} onChange={e => onChange({ ...value, market_weight_source: e.target.value })} /></Field>
        <div className="grid gap-3 sm:grid-cols-2">
          <Field label="市场风险厌恶系数 δ" hint="有限正数；无单位。"><NumberInput aria-label="市场风险厌恶系数 δ" className={numeric} value={value.delta} onValueChange={delta => onChange({ ...value, delta })} /></Field>
          <Field label="年化无风险收益（%）"><NumberInput className={numeric} value={percentInputValue(value.risk_free_rate)} onValueChange={n => onChange({ ...value, risk_free_rate: n / 100 })} /></Field>
        </div>
        <CovarianceEditor label="资产风险协方差" axis={value.asset_ids} value={value.covariance} onChange={covariance => onChange({ ...value, covariance })} />
        <details><summary className="cursor-pointer py-2 text-sm font-semibold">高级设置</summary><Field label="先验均值不确定性系数 τ" hint="有限正数；仅缩放均值先验协方差，不改变资产风险。"><NumberInput className={numeric} value={value.tau} onValueChange={tau => onChange({ ...value, tau })} /></Field></details>
        <BlackLittermanViews value={value} onChange={onChange} />
      </> : value.method === 'scenario_mixture' ? <>
        <p className="text-sm leading-6 text-slate-600">只有明确概率的情景才能混合。结果包含情景之间的均值差异风险，是单期矩匹配，不是尾部风险预测。</p>
        <Field label="情景风险口径"><select className={input} value={value.risk_mode} onChange={e => onChange({ ...value, risk_mode: e.target.value as 'shared' | 'scenario_specific', shared_covariance: e.target.value === 'shared' ? blankMatrix(value.asset_ids.length) : null, scenarios: value.scenarios.map(s => ({ ...s, covariance: e.target.value === 'shared' ? null : blankMatrix(value.asset_ids.length) })) })}><option value="shared">明确共用一个风险矩阵</option><option value="scenario_specific">逐情景提供风险矩阵</option></select></Field>
        <p className="text-xs text-slate-600">切换风险口径后，请重新填写对应矩阵，原矩阵不会被自动复制。</p>
        {value.risk_mode === 'shared' && <CovarianceEditor label="共用风险协方差" axis={value.asset_ids} value={value.shared_covariance} onChange={shared_covariance => onChange({ ...value, shared_covariance })} />}
        <p role="status" className="text-sm tabular-nums text-slate-600">情景概率合计：{value.scenarios.length ? percentText(value.scenarios.reduce((sum, s) => sum + s.probability, 0)) : '尚未提供'}</p>
        {!value.scenarios.length && <p className="text-sm text-slate-600">尚无情景，请添加情景并填写概率、每项资产收益及风险依据。</p>}
        <div className="space-y-4 divide-y divide-slate-200">{value.scenarios.map((scenario, index) => <fieldset key={index} className="min-w-0 space-y-3 pt-4">
          <legend className="text-sm font-semibold">情景 {index + 1}</legend>
          <div className="grid min-w-0 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            <Field label={`情景${index + 1}名称`}><input className={input} maxLength={120} value={scenario.id} onChange={e => patchScenario(index, { id: e.target.value })} /></Field>
            <Field label={`情景${index + 1}概率（%）`}><NumberInput className={numeric} value={percentInputValue(scenario.probability)} onValueChange={n => patchScenario(index, { probability: n / 100 })} /></Field>
            <Field label={`情景${index + 1}依据`}><input className={input} maxLength={2000} value={scenario.source} onChange={e => patchScenario(index, { source: e.target.value })} /></Field>
            {value.asset_ids.map(asset => <Field key={asset} label={`情景${index + 1}${asset}年化收益（%）`}><NumberInput className={numeric} value={percentInputValue(scenario.annual_returns[asset])} onValueChange={n => patchScenario(index, { annual_returns: { ...scenario.annual_returns, [asset]: n / 100 } })} /></Field>)}
          </div>
          {value.risk_mode === 'scenario_specific' && <CovarianceEditor label={`情景${index + 1}风险协方差`} axis={value.asset_ids} value={scenario.covariance} onChange={covariance => patchScenario(index, { covariance })} />}
          <Button className={button} onClick={() => onChange({ ...value, scenarios: value.scenarios.filter((_, i) => i !== index) })}>移除情景 {index + 1}</Button>
        </fieldset>)}</div>
        <Button className={button} disabled={value.scenarios.length >= 60} onClick={() => onChange({ ...value, scenarios: [...value.scenarios, { id: '', probability: NaN, annual_returns: Object.fromEntries(value.asset_ids.map(a => [a, NaN])), covariance: value.risk_mode === 'shared' ? null : blankMatrix(value.asset_ids.length), source: '' }] })}>添加情景</Button>
        {value.scenarios.length >= 60 && <p className="text-xs text-slate-600">已达到60个情景上限。</p>}
      </> : null}
    </fieldset>}
    {value && validation && !readOnly && <p role="status" className="text-sm leading-6 text-amber-900">{validation}</p>}
    {error && <p role="alert" className="text-sm leading-6 text-rose-700">{error}</p>}
    {value && onPreview && <Button tone="primary" className={button} disabled={busy || !!reason || !!validation} onClick={() => { if (!busy && !reason && !cmaModelInputError(value)) onPreview(value) }}>{error ? '重试模型预览' : '预览模型假设'}</Button>}
  </section>
}
