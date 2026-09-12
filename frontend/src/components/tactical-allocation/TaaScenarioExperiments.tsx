import { useEffect, useRef, useState } from 'react'
import { useAllocationDraft } from '../../app/allocationJourney'
import { simulateTaaScenario, type TaaPreview, type TaaScenario, type TaaScenarioExperiment } from '../../services/tacticalAllocation'
import { buttonClass, Empty, Feedback, Field, inputClass, NumberInput, percentText, primaryClass, sectionClass } from '../risk-models/ResearchUI'
import { PortfolioRiskSection } from '../risk-models/PublishedRiskPanel'
import { TaaScenarioView } from './TaaResults'

const points = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(2)} 个百分点`
export default function TaaScenarioExperiments({ preview, baselineId, experiments, onChange, onBacktest }: {
  preview: TaaPreview | null; baselineId: string; experiments: TaaScenarioExperiment[]
  onChange: (value: TaaScenarioExperiment[]) => void; onBacktest: () => void
}) {
  const [draft, setDraft] = useAllocationDraft<TaaScenario>(`taa-scenario:${baselineId}`, { kind: 'shock', name: '自定义资产冲击', shocks: {} })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState<string | null>(null)
  const sequence = useRef(0)
  useEffect(() => { sequence.current += 1; setBusy(false); setError(''); return () => { sequence.current += 1 } }, [preview?.preview_hash])
  const names = Object.fromEntries((preview?.baseline.assets ?? []).map(asset => [asset.id, asset.name]))
  const range = preview ? { start: preview.data.start_date, end: [preview.data.end_date, preview.request.as_of].sort()[0] } : null
  const invalidHistory = draft.kind === 'historical' && (!range || !draft.start_date || !draft.end_date || draft.start_date < range.start || draft.end_date > range.end || draft.start_date >= draft.end_date)
  const active = experiments.find(item => item.scenario.name === selected)
  function update(patch: Partial<TaaScenario>) { sequence.current += 1; setBusy(false); setError(''); setDraft(previous => ({ ...previous, ...patch })) }
  function chooseKind(kind: TaaScenario['kind']) {
    update(kind === 'historical' ? { kind, name: '历史区间回放', start_date: range?.start, end_date: range?.end } : { kind, name: '自定义资产冲击', shocks: Object.fromEntries(Object.keys(names).map(id => [id, 0])) })
  }
  async function run(scenario: TaaScenario) {
    if (!preview || busy) return
    const normalized = { ...scenario, name: scenario.name.trim(), ...(scenario.kind === 'shock' ? { shocks: Object.fromEntries(Object.keys(names).map(id => [id, scenario.shocks?.[id] ?? 0])) } : {}) }
    if (!normalized.name) { setError('请给情景取一个名称。'); return }
    if (experiments.length >= 12 && !experiments.some(item => item.scenario.name === normalized.name)) { setError('每份研究最多保留 12 个情景，请先移除不需要的实验。'); return }
    const token = ++sequence.current; setBusy(true); setError('')
    try {
      const result = await simulateTaaScenario({ preview_request: preview.request, candidate_id: preview.selected_id, scenario: normalized })
      if (token !== sequence.current) return
      onChange([...experiments.filter(item => item.scenario.name !== normalized.name), { scenario: normalized, result }]); setSelected(normalized.name)
    } catch (failure) { if (token === sequence.current) setError(failure instanceof Error ? failure.message : '情景模拟未完成。') }
    finally { if (token === sequence.current) setBusy(false) }
  }
  return <div className="min-w-0 space-y-4">
    <section className={`${sectionClass} space-y-4`}><h2 className="text-lg font-semibold">如果市场不按预期走，会怎样？</h2><p className="text-sm leading-6 text-slate-600">把多个具名实验放在一起比较。完成的实验随研究版本保存；改动配置后保留假设，重新计算。</p><Feedback error={error} />
      {!preview ? <Empty title="先计算一个可比较方案"><p>完成回测与候选选择，再对锁定结果模拟情景。已有假设仍保留。</p><button type="button" className={`${buttonClass} mt-3`} onClick={onBacktest}>去回测与选优</button></Empty> : <>
        <div className="grid gap-4 sm:grid-cols-2"><Field label="情景方式"><select className={inputClass} value={draft.kind} onChange={event => chooseKind(event.target.value as TaaScenario['kind'])}><option value="shock">自定义资产冲击</option><option value="historical">历史区间回放</option></select></Field><Field label="情景名称"><input className={inputClass} value={draft.name} onChange={event => update({ name: event.target.value })} /></Field></div>
        {draft.kind === 'shock' ? <><div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{preview.baseline.assets.map(asset => <Field key={asset.id} label={`${asset.name}假设涨跌（%）`}><NumberInput className={inputClass} value={draft.shocks?.[asset.id] ?? 0} onValueChange={value => update({ shocks: { ...draft.shocks, [asset.id]: value } })} /></Field>)}</div><p className="text-xs text-slate-600">0% 表示明确假设该资产不变。假设不含发生概率，不能作为预测。</p></> : <><p className="text-xs text-slate-600">可用历史范围：{range?.start} — {range?.end}；结束日不能晚于研究日。</p><div className="grid gap-4 sm:grid-cols-2"><Field label="历史情景开始"><input type="date" className={inputClass} min={range?.start} max={draft.end_date} value={draft.start_date ?? ''} onChange={event => update({ start_date: event.target.value })} /></Field><Field label="历史情景结束"><input type="date" className={inputClass} min={draft.start_date} max={range?.end} value={draft.end_date ?? ''} onChange={event => update({ end_date: event.target.value })} /></Field></div>{invalidHistory && <p role="status" className="text-sm text-amber-900">历史日期超出当前可用范围，请调整后计算。<button type="button" className="ml-2 min-h-9 underline" onClick={() => update({ start_date: range?.start, end_date: range?.end })}>使用当前可用范围</button></p>}</>}
        <button type="button" className={primaryClass} disabled={busy || invalidHistory || !draft.name.trim() || (draft.kind === 'shock' && Object.values(draft.shocks ?? {}).some(value => !Number.isFinite(value)))} onClick={() => void run({ ...draft, ...(draft.kind === 'shock' ? { shocks: Object.fromEntries(Object.entries(draft.shocks ?? {}).map(([id, value]) => [id, value / 100])) } : {}) })}>{busy ? '正在模拟…' : experiments.some(item => item.scenario.name === draft.name.trim()) ? '重新计算同名情景' : '计算情景影响'}</button>
      </>}
    </section>
    <section className={`${sectionClass} space-y-3`} aria-label="情景实验对照"><h2 className="text-base font-semibold">已保留的情景 · {experiments.length}/12</h2>{!experiments.length ? <p className="text-sm text-slate-600">完成第一个实验后，它会出现在这里。新增实验不会覆盖其他名称的结果。</p> : <div className="grid gap-3 lg:grid-cols-2">{experiments.map(item => <article key={item.scenario.name} className="min-w-0 rounded-lg border border-slate-200 p-3"><div className="flex flex-wrap items-start justify-between gap-2"><h3 className="min-w-0 break-words text-sm font-semibold">{item.scenario.name}</h3><span className="text-xs text-slate-600">{item.result ? '随当前版本保存' : '条件已更改 · 待重算'}</span></div><p className="mt-1 text-xs leading-5 text-slate-600">{item.scenario.kind === 'historical' ? `${item.scenario.start_date} — ${item.scenario.end_date}` : Object.entries(item.scenario.shocks ?? {}).map(([id, value]) => `${names[id] ?? id} ${percentText(value)}`).join('；')}</p>{item.result && <dl className="mt-3 grid grid-cols-3 gap-2 text-xs"><div><dt>SAA</dt><dd className="mt-1 font-semibold">{percentText(item.result.baseline_return)}</dd></div><div><dt>TAA</dt><dd className="mt-1 font-semibold">{percentText(item.result.taa_return)}</dd></div><div><dt>收益差</dt><dd className="mt-1 font-semibold">{points(item.result.excess_return)}</dd></div></dl>}<div className="mt-3 flex flex-wrap gap-2">{item.result ? <button type="button" className={buttonClass} onClick={() => setSelected(item.scenario.name)}>查看明细</button> : <button type="button" className={buttonClass} disabled={!preview || busy} onClick={() => void run(item.scenario)}>重算情景</button>}<button type="button" className={buttonClass} disabled={busy} aria-label={`移除情景 ${item.scenario.name}`} onClick={() => { onChange(experiments.filter(other => other !== item)); if (selected === item.scenario.name) setSelected(null) }}>移除</button></div></article>)}</div>}</section>
    {active?.result && <TaaScenarioView result={active.result} names={names} />}
    {preview && <details className={sectionClass}><summary className="cursor-pointer text-sm font-medium">进阶：使用已发布风险模型做宏观压力测试</summary><div className="mt-4"><PortfolioRiskSection context="选择已保存的产品组合，用已发布模型与情景做压测。大类假设冲击与产品模型保留各自口径。" /></div></details>}
  </div>
}
