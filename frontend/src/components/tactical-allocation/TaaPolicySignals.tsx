import { useState } from 'react'
import type { TaaDatedSignal, TaaPreviewRequest, TaaSignalComponent } from '../../services/tacticalAllocation'
import { Button } from '../ui'
import { Field, inputClass, NumberInput, sectionClass } from '../risk-models/ResearchUI'

export const scheduledPolicy = { mode: 'scheduled', cost_basis: 'half_turnover', decision_frequency: 'monthly', execution_frequency: 'daily', execution_lag: 1, min_holding_periods: 0, deviation_threshold: 0 } as const
const frequencies = [['daily', '每日'], ['weekly', '每周'], ['monthly', '每月'], ['quarterly', '每季']] as const
const kinds = [['momentum', '因果动量窗口'], ['value', '价值研究输入'], ['carry', '持有收益研究输入'], ['macro', '宏观研究输入'], ['risk_sentiment', '风险情绪研究输入']] as const

export function policySignalIssue(request: TaaPreviewRequest, assets: string[]): string {
  const policy = request.decision_policy
  if (policy && (![policy.execution_lag, policy.min_holding_periods, policy.deviation_threshold].every(Number.isFinite)
    || !Number.isInteger(policy.execution_lag) || policy.execution_lag < 1 || policy.execution_lag > 250
    || !Number.isInteger(policy.min_holding_periods) || policy.min_holding_periods < 0 || policy.min_holding_periods > 2500
    || policy.deviation_threshold < 0 || policy.deviation_threshold > 1)) return '请填写有效的执行滞后、最小持有期和权重阈值。'
  if (request.signal_mode !== 'composite') return ''
  const components = request.signal_components ?? []
  if (!Array.isArray(components) || !components.length) return '请添加至少一个有来源的信号分量。'
  if (components.some(c => !c || typeof c.source !== 'string' || typeof c.methodology !== 'string' || !Array.isArray(c.observations))) return '信号定义不完整，请重新添加信号分量。'
  if (components.some(c => !Number.isFinite(c.weight) || c.weight < 0 || c.weight > 1)
    || Math.abs(components.reduce((sum, c) => sum + c.weight, 0) - 1) > 1e-8) return '各信号权重须完整且合计为 100%，缺失分量不会重新分配权重。'
  if (components.some(c => !c.source.trim() || !c.methodology.trim())) return '请填写每个信号的来源和研究标准化方法。'
  for (const c of components) {
    if (!Number.isInteger(c.lookback) || c.lookback < 2 || c.lookback > 1000 || !Number.isInteger(c.max_age_days) || c.max_age_days < 1 || c.max_age_days > 3650) return '请填写有效的观察窗口和信号有效期。'
    if (c.kind === 'momentum') continue
    if (!c.observations.length) return '外部信号需要日期化的研究标准值，不能用净值替代。'
    if (c.observations.some(r => !r || !r.values || !r.observed_on || !r.available_on || !r.expires_on
      || r.observed_on > r.available_on || r.available_on > r.expires_on || Object.keys(r.values).length !== assets.length
      || assets.some(a => typeof r.values[a] !== 'number' || !Number.isFinite(r.values[a]) || Math.abs(r.values[a]) > 1))) return '外部信号须完整覆盖资产轴，标准值在 [-1, 1]，观察日 ≤ 可得日 ≤ 到期日。'
  }
  return ''
}

function DatedInput({ component, update }: { component: TaaSignalComponent; update: (patch: Partial<TaaSignalComponent>) => void }) {
  const [raw, setRaw] = useState(() => component.observations.length ? JSON.stringify(component.observations, null, 2) : '')
  const [error, setError] = useState('')
  return <Field label="日期化标准值（JSON 数组）" hint={'每行包含 observed_on、available_on、expires_on、values；values 以完整资产 ID 为键，数值范围 [-1, 1]，0 为中性。'}>
    <textarea aria-label="日期化标准值（JSON 数组）" className={`${inputClass} font-mono placeholder:text-slate-600 placeholder:opacity-100`} rows={5} value={raw} onChange={event => {
      setRaw(event.target.value)
      try {
        const rows: unknown = JSON.parse(event.target.value)
        if (!Array.isArray(rows)) throw new Error('需要 JSON 数组。')
        update({ observations: rows as TaaDatedSignal[] }); setError('')
      } catch { update({ observations: [] }); setError('JSON 尚不完整，补全后才能计算。') }
    }} />
    {error && <p role="alert" className="text-sm text-rose-800">{error}</p>}
  </Field>
}

export function TaaPolicySignals({ request, assets, update }: { request: TaaPreviewRequest; assets: string[]; update: (patch: Partial<TaaPreviewRequest>) => void }) {
  const policy = request.decision_policy
  const components = request.signal_components ?? []
  const change = (index: number, patch: Partial<TaaSignalComponent>) => update({ signal_components: components.map((c, i) => i === index ? { ...c, ...patch } : c) })
  return <section className={`${sectionClass} space-y-4`} aria-label="决策与执行规则">
    <div><h2 className="text-lg font-semibold">何时判断、何时调整？</h2><p className="mt-1 text-sm text-slate-600">决策更新目标；执行机会与滞后满足后才模拟交易。未交易时持仓随收益自然变化。</p></div>
    <Field label="调仓口径"><select className={inputClass} value={policy ? 'scheduled' : 'daily_target'} onChange={e => update({ decision_policy: e.target.value === 'scheduled' ? { ...scheduledPolicy } : null })}>
      <option value="scheduled">分开决策与执行</option><option value="daily_target">原研究口径：每日恢复目标</option>
    </select></Field>
    {!policy && <p className="text-sm text-slate-600">保留原日频目标与费用算法。旧版本按保存时口径阅读，复制为新研究后可更换。</p>}
    {policy && <><div className="grid gap-4 sm:grid-cols-2">
      <Field label="决策频率"><select className={inputClass} value={policy.decision_frequency} onChange={e => update({ decision_policy: { ...policy, decision_frequency: e.target.value as typeof policy.decision_frequency } })}>{frequencies.map(([v, text]) => <option key={v} value={v}>{text}</option>)}</select></Field>
      <Field label="执行机会"><select className={inputClass} value={policy.execution_frequency} onChange={e => update({ decision_policy: { ...policy, execution_frequency: e.target.value as typeof policy.execution_frequency } })}>{frequencies.map(([v, text]) => <option key={v} value={v}>{text}</option>)}</select></Field>
    </div><p className="text-xs leading-5 text-slate-600">周、月、季取周期首个已知共同观察点；共同净值日期不是交易所或基金申赎日历。费用分母独立设置；默认保持原半数绝对权重变化口径。</p>
      <details className="border-t border-slate-100 pt-3"><summary className="cursor-pointer text-sm font-medium">执行滞后、阈值与实际持仓时点</summary><div className="mt-4 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Field label="费用计量口径"><select className={inputClass} value={policy.cost_basis ?? 'half_turnover'} onChange={e => update({ decision_policy: { ...policy, cost_basis: e.target.value as typeof policy.cost_basis } })}><option value="half_turnover">原口径：半数绝对权重变化</option><option value="gross_traded_weight">双边成交：买入加卖出权重</option></select></Field>
        <Field label="执行滞后（共同观察期）"><NumberInput className={inputClass} min={1} max={250} value={policy.execution_lag} onValueChange={v => update({ decision_policy: { ...policy, execution_lag: v } })} /></Field>
        <Field label="最小持有期（共同观察期）"><NumberInput className={inputClass} min={0} max={2500} value={policy.min_holding_periods} onValueChange={v => update({ decision_policy: { ...policy, min_holding_periods: v } })} /></Field>
        <Field label="单资产偏离阈值（百分点）" hint="任一资产达到阈值（含等号）才调整；0 表示有差异就可调整。"><NumberInput className={inputClass} min={0} max={100} aria-label="单资产偏离阈值（百分点）" value={policy.deviation_threshold * 100} onValueChange={v => update({ decision_policy: { ...policy, deviation_threshold: v / 100 } })} /></Field>
        <Field label="实际持仓快照日" hint="权重在下方参考持仓中填写；时点必须等于研究日。"><input className={inputClass} type="date" value={request.current_weights_as_of ?? ''} onChange={e => update({ current_weights_as_of: e.target.value || null })} /></Field>
        <Field label="实际最近执行日" hint="启用最小持有期时必须填写。"><input className={inputClass} type="date" value={request.last_execution_date ?? ''} onChange={e => update({ last_execution_date: e.target.value || null })} /></Field>
      </div></details><p role="status" className="text-sm text-amber-800">缺少研究日实际持仓时，只能研究目标，不能判断阈值触发或交接交易；这里不会执行真实交易。</p></>}
    {request.signal_mode === 'composite' && <div className="space-y-4 border-t border-slate-100 pt-4"><h3 className="text-lg font-semibold">组合有来源的信号</h3><p className="break-words text-sm text-slate-600">完整资产轴：{assets.join('、')}。任一非零权重分量缺失、尚不可得或过期，则组合无效；不向其他信号转移权重。</p>
      <div className="divide-y divide-slate-100">{components.map((c, i) => <div key={c.id} className="space-y-3 py-4">
        <div className="flex flex-wrap items-center justify-between gap-2"><h4 className="text-sm font-semibold">信号 {i + 1}</h4><Button tone="secondary" onClick={() => update({ signal_components: components.filter((_, k) => k !== i) })}>移除此信号</Button></div>
        <div className="grid gap-3 sm:grid-cols-2"><Field label={`信号 ${i + 1} 来源类型`}><select className={inputClass} value={c.kind} onChange={e => change(i, { kind: e.target.value as TaaSignalComponent['kind'], observations: [], source: '', methodology: '' })}>{kinds.map(([v, t]) => <option key={v} value={v}>{t}</option>)}</select></Field>
          <Field label={`信号 ${i + 1} 权重（%）`}><NumberInput className={inputClass} min={0} max={100} value={c.weight * 100} onValueChange={v => change(i, { weight: v / 100 })} /></Field>
          {c.kind === 'momentum' && <Field label={`信号 ${i + 1} 动量窗口`}><NumberInput className={inputClass} value={c.lookback} min={2} max={1000} onValueChange={v => change(i, { lookback: v })} /></Field>}
          <Field label={`信号 ${i + 1} 最长观察年龄（天）`}><NumberInput className={inputClass} value={c.max_age_days} min={1} max={3650} onValueChange={v => change(i, { max_age_days: v })} /></Field>
          <Field label={`信号 ${i + 1} 研究来源`}><input className={inputClass} value={c.source} onChange={e => change(i, { source: e.target.value })} /></Field>
          <Field label={`信号 ${i + 1} 标准化方法与依据`}><input className={inputClass} value={c.methodology} onChange={e => change(i, { methodology: e.target.value })} /></Field>
        </div>{c.kind !== 'momentum' && <DatedInput component={c} update={patch => change(i, patch)} />}
      </div>)}</div>
      <Button tone="secondary" disabled={components.length >= 8} onClick={() => update({ signal_components: [...components, { id: `signal-${Date.now()}`, kind: 'momentum', weight: components.length ? 0 : 1, source: '本次冻结净值窗口', methodology: '窗口收益中心化后按最大绝对偏差标准化；相同收益视为中性', unit: 'standardized_score_minus1_plus1', lookback: 60, max_age_days: 31, observations: [] }] })}>添加信号分量</Button>
      {components.length >= 8 && <p className="text-xs text-slate-600">最多组合 8 个分量。</p>}
    </div>}
  </section>
}
