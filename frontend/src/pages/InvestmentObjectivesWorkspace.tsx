import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useAllocationDraft } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button } from '../components/ui'
import { Empty, Feedback, Field, inputClass, NumberInput, percentText, sectionClass, today } from '../components/risk-models/ResearchUI'
import { getStrategicCatalog, saveMandate, percentInputValue, type MandateDefinition, type MandateVersion } from '../services/strategicAllocation'

const defaultDefinition = (day: string): MandateDefinition => ({
  name: '', as_of: day, review_date: new Date(Date.parse(day) + 180 * 86400000).toISOString().slice(0, 10),
  currency: 'CNY', horizon_years: 10, target_return: NaN, max_volatility: .15,
  min_liquid_weight: 0, max_illiquid_weight: 0, max_tracking_error: .1, risk_aversion: 5,
  rebalance_policy: 'quarterly', rebalance_note: '', note: '',
})
const inputPercent = percentInputValue

export default function InvestmentObjectivesWorkspace() {
  const platformDay = useResearchDay()
  const [draft, setDraft] = useAllocationDraft<MandateDefinition>('strategic-mandate:editor', () => defaultDefinition(platformDay ?? today()))
  const [versions, setVersions] = useState<MandateVersion[]>([])
  const [selected, setSelected] = useState<MandateVersion | null>(null)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [reload, setReload] = useState(0)
  const generation = useRef(0)
  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getStrategicCatalog(controller.signal).then(value => { if (!controller.signal.aborted) setVersions(value.mandates) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '目标版本读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  useEffect(() => () => { generation.current += 1 }, [])

  function update(patch: Partial<MandateDefinition>) {
    generation.current += 1
    setBusy(false); setSelected(null); setError(''); setDraft(value => ({ ...value, ...patch }))
  }
  const invalid = !draft.name.trim() || !draft.as_of || !draft.review_date || draft.as_of > today()
    || draft.review_date <= draft.as_of || Boolean(platformDay && draft.as_of > platformDay)
    || ![draft.horizon_years, draft.target_return, draft.max_volatility, draft.min_liquid_weight,
      draft.max_illiquid_weight, draft.max_tracking_error, draft.risk_aversion].every(Number.isFinite)
    || draft.max_volatility <= 0 || draft.max_tracking_error <= 0
    || (draft.rebalance_policy === 'threshold' && !draft.rebalance_note.trim())
  async function save() {
    const token = ++generation.current
    setBusy(true); setError('')
    try {
      const version = await saveMandate(draft)
      if (generation.current !== token) return
      setSelected(version); setVersions(items => [version, ...items])
    } catch (reason) {
      if (generation.current === token) setError(reason instanceof Error ? reason.message : '目标版本保存失败。')
    } finally { if (generation.current === token) setBusy(false) }
  }

  return <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6">
    <header><h1 className="text-2xl font-semibold text-slate-900">投资目标与边界</h1>
      <p className="mt-2 text-sm leading-6 text-slate-600">先明确资金的用途、期限和可承担的风险，再决定配什么。这里保存的是研究约束，不是收益承诺或外部审批。</p></header>
    <Feedback error={error} />
    <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(240px,1fr)]">
      <section className={`${sectionClass} space-y-5`} aria-label="投资目标编辑">
        <h2 className="text-lg font-semibold">这笔资金要实现什么？</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="目标名称"><input className={inputClass} value={draft.name} maxLength={120} onChange={e => update({ name: e.target.value })} placeholder="例如：长期稳健配置" /></Field>
          <Field label="计价币种" hint="后续长期假设须采用相同币种；不会自动进行汇率换算。"><select className={inputClass} value={draft.currency} onChange={e => update({ currency: e.target.value })}>{['CNY', 'USD', 'HKD', 'EUR', 'CAD'].map(value => <option key={value}>{value}</option>)}</select></Field>
          <Field label="目标研究日"><input type="date" className={inputClass} max={platformDay ?? today()} value={draft.as_of} onChange={e => update({ as_of: e.target.value })} /></Field>
          <Field label="政策复核日期" hint="到期后不能直接沿用，须重新研究并保存新版本。"><input type="date" className={inputClass} min={draft.as_of} value={draft.review_date} onChange={e => update({ review_date: e.target.value })} /></Field>
          <Field label="投资期限（年）"><NumberInput className={inputClass} value={draft.horizon_years} min={1} max={30} onValueChange={value => update({ horizon_years: value })} /></Field>
          <Field label="最低预期年收益（%）" hint="年化算术总收益假设约束，不是保底收益或 CAGR。"><NumberInput className={inputClass} value={inputPercent(draft.target_return)} onValueChange={value => update({ target_return: value / 100 })} /></Field>
          <Field label="最高预期年波动（%）"><NumberInput className={inputClass} value={inputPercent(draft.max_volatility)} min={0} onValueChange={value => update({ max_volatility: value / 100 })} /></Field>
          <Field label="TAA 主动风险上限（%）" hint="相对政策组合的跟踪误差上限，战术研究不能擅自扩大。"><NumberInput className={inputClass} value={inputPercent(draft.max_tracking_error)} min={0} onValueChange={value => update({ max_tracking_error: value / 100 })} /></Field>
          <Field label="最低流动性资产占比（%）" hint="指经研究员确认可变现的资产，不等于现金余额。"><NumberInput className={inputClass} value={inputPercent(draft.min_liquid_weight)} min={0} max={100} onValueChange={value => update({ min_liquid_weight: value / 100 })} /></Field>
          <Field label="最高非流动性资产占比（%）"><NumberInput className={inputClass} value={inputPercent(draft.max_illiquid_weight)} min={0} max={100} onValueChange={value => update({ max_illiquid_weight: value / 100 })} /></Field>
        </div>
        <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">效用偏好与再平衡政策</summary>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label="风险厌恶系数" hint="越大越重视方差惩罚，不自动代表风险测评结果。"><NumberInput className={inputClass} value={draft.risk_aversion} min={0} onValueChange={value => update({ risk_aversion: value })} /></Field>
            <Field label="政策再平衡方式"><select className={inputClass} value={draft.rebalance_policy} onChange={e => update({ rebalance_policy: e.target.value as MandateDefinition['rebalance_policy'] })}><option value="monthly">每月复核</option><option value="quarterly">每季复核</option><option value="annually">每年复核</option><option value="threshold">触及阈值时复核</option></select></Field>
          </div>
          <Field label="再平衡触发与恢复规则" hint="记录政策约定，不自动执行调仓。"><textarea className={inputClass} value={draft.rebalance_note} rows={2} onChange={e => update({ rebalance_note: e.target.value })} /></Field>
        </details>
        <Field label="资金用途与其他说明"><textarea className={inputClass} rows={3} value={draft.note} onChange={e => update({ note: e.target.value })} /></Field>
        <div className="flex flex-wrap items-center gap-3"><Button tone="primary" disabled={busy || invalid || Boolean(selected)} onClick={() => void save()}>{busy ? '正在保存…' : selected ? '已锁定此目标版本' : '保存新目标版本'}</Button>
          {invalid && <p className="text-sm text-slate-600">请填写完整名称、日期及数值，确认目标收益与风险边界。</p>}</div>
      </section>
      <aside className="space-y-4" aria-label="目标版本与下一步">
        {selected ? <section className={`${sectionClass} space-y-3`}><h2 className="text-lg font-semibold">下一步：研究长期配置</h2><p className="text-sm text-slate-600">{selected.name} · {selected.definition.currency} · {selected.definition.horizon_years} 年</p><p className="text-sm text-slate-600">预期波动上限 {percentText(selected.definition.max_volatility)}；TAA 主动风险上限 {percentText(selected.definition.max_tracking_error)}。</p>
          <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={`/pre-investment/saa/policy?mandate=${encodeURIComponent(selected.id)}`}>使用此目标进入长期配置 →</Link><p className="text-xs leading-5 text-slate-600">修改输入会形成新版本，不覆盖已保存目标。</p></section>
          : <Empty title="先确定目标，再比较权重"><p>保存目标后，选择已有大类、填写长期假设，比较符合这些边界的政策候选。</p></Empty>}
        <section className={`${sectionClass} space-y-3`}><h2 className="text-lg font-semibold">已保存的目标</h2>
          {loading ? <p role="status" className="text-sm text-slate-600">正在读取版本…</p> : versions.length ? <div className="max-h-96 space-y-2 overflow-auto">{versions.map(version => <button key={version.id} type="button" className="block min-h-11 w-full rounded-lg border border-slate-200 px-3 py-2 text-left text-sm hover:border-accent-600 focus-visible:outline focus-visible:outline-accent-600" onClick={() => { generation.current += 1; setBusy(false); setDraft(version.definition); setSelected(version); setError('') }}><span className="block font-medium">{version.name}</span><span className="text-xs text-slate-600">{version.definition.currency} · {version.definition.horizon_years} 年 · {version.created_at.slice(0, 10)}</span></button>)}</div> : <p className="text-sm text-slate-600">尚无目标版本。</p>}
          <Button disabled={loading} onClick={() => { setError(''); setReload(value => value + 1) }}>重新读取版本</Button>
        </section>
      </aside>
    </div>
  </div>
}
