import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { readAllocationDraft, useAllocationDraft, updateAllocationJourney, allocationJourneyPath } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button } from '../components/ui'
import { Feedback, Field, inputClass, NumberInput, percentText, sectionClass, today } from '../components/risk-models/ResearchUI'
import { GoalFields, BoundaryFields } from '../components/investment-mandate/MandateFields'
import InstitutionalFields from '../components/investment-mandate/InstitutionalFields'
import InstitutionalResults from '../components/investment-mandate/InstitutionalResults'
import AssetAuthorizations from '../components/investment-mandate/AssetAuthorizations'
import MandateResults from '../components/investment-mandate/MandateResults'
import { amountText, mandateIssues, newMandate, objectiveLabels, studyIssue } from '../components/investment-mandate/model'
import { confirmMandate, getMandate, getStrategicCatalog, previewMandate,
  type MandateAssessment, type MandateDefinition, type MandateStudyRequest, type MandateVersion, type StrategicCatalog } from '../services/strategicAllocation'

const steps = ['资金与成功标准', '风险与限制', '量化诊断', '核对与确认']
const statusLabel = (status?: string) => status === 'diagnosed' ? '已诊断' : status === 'needs_revision' ? '需要复核' : '仅保存输入'

export default function InvestmentObjectivesWorkspace() {
  const platformDay = useResearchDay()
  const clockIssue = platformDay === undefined ? '平台知识截止日尚未确认，请先恢复 PIT 设置；草稿保留，暂不能诊断或保存。' : ''
  const cutoff = platformDay && platformDay < today() ? platformDay : today()
  const [draft, setDraft] = useAllocationDraft<MandateStudyRequest>('mandate-study:editor', () => ({
    definition: { ...newMandate(cutoff), ...(readAllocationDraft<MandateDefinition>('strategic-mandate:editor') ?? {}) },
    cma_id: null, simulation_paths: 2000, seed: 42, uncertainty_penalty: 1,
  }))
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [selected, setSelected] = useState<MandateVersion | null>(null)
  const [preview, setPreview] = useState<MandateAssessment | null>(null)
  const [step, setStep] = useState(0)
  const [acknowledged, setAcknowledged] = useState(false)
  const [busy, setBusy] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [reload, setReload] = useState(0)
  const generation = useRef(0)
  const operation = useRef<AbortController | null>(null)
  const heading = useRef<HTMLHeadingElement>(null)
  const previousClock = useRef(platformDay)
  const definition = draft.definition
  const issues = mandateIssues(definition, cutoff)
  const numericIssue = studyIssue(draft)
  const matchingCmas = catalog?.assumptions.filter(cma => cma.currency === definition.currency && cma.as_of === definition.as_of
    && cma.horizon_years === definition.horizon_years && (!definition.benchmark || cma.alloc_name === definition.benchmark.alloc_name)
    && (!definition.allocation_scope || cma.alloc_name === definition.allocation_scope)
    && (!definition.strategic_universe_id || cma.strategic_universe_id === definition.strategic_universe_id)) ?? []
  const cmaIssue = !selected && draft.cma_id && catalog && !matchingCmas.some(cma => cma.id === draft.cma_id)
    ? '已选CMA不在当前日期、币种、期限及大类范围内，请重新选择或明确改为仅资金测算。' : ''
  const activeIssue = clockIssue || (step === 0 ? issues[0] : step === 1 ? issues[1] : issues.find(Boolean) || numericIssue || cmaIssue)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getStrategicCatalog(controller.signal).then(result => { if (!controller.signal.aborted) setCatalog(result) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => () => { generation.current += 1; operation.current?.abort() }, [])
  useEffect(() => {
    if (platformDay !== previousClock.current) {
      previousClock.current = platformDay
      generation.current += 1; operation.current?.abort(); setBusy(false); setAcknowledged(false)
      if (selected) {
        setNotice('知识截止日已变化；当前只读版本仍显示保存时的诊断，不代表新截止日下可用。复制为新研究后须重新核对。')
      } else {
        setPreview(null)
        setNotice('知识截止日已变化，请重新核对输入并运行诊断。')
      }
    }
  }, [platformDay, selected])

  function invalidate() {
    generation.current += 1; operation.current?.abort(); setBusy(false)
    setPreview(null); setSelected(null); setAcknowledged(false); setError(''); setNotice('输入已更新，请重新运行诊断。')
  }
  function update(patch: Partial<MandateDefinition>) {
    invalidate()
    setDraft(current => ({ ...current, definition: { ...current.definition, ...patch },
      cma_id: ['as_of', 'currency', 'horizon_years', 'benchmark', 'allocation_scope', 'strategic_universe_id'].some(key => key in patch) ? null : current.cma_id }))
  }
  function updateStudy(patch: Partial<Omit<MandateStudyRequest, 'definition'>>) {
    invalidate(); setDraft(current => ({ ...current, ...patch }))
  }
  async function run<T,>(work: (signal: AbortSignal) => Promise<T>, consume: (value: T) => void) {
    operation.current?.abort()
    const controller = new AbortController(); operation.current = controller
    const token = ++generation.current
    setBusy(true); setError(''); setNotice('')
    try { const result = await work(controller.signal); if (generation.current === token && !controller.signal.aborted) consume(result) }
    catch (reason) { if (generation.current === token && !controller.signal.aborted) setError(reason instanceof Error ? reason.message : '研究操作失败，请重试。') }
    finally { if (generation.current === token) setBusy(false) }
  }
  function diagnose() {
    if (clockIssue || issues.some(Boolean) || numericIssue || cmaIssue || selected) return
    void run(signal => previewMandate(draft, signal), result => { setPreview(result); setAcknowledged(false) })
  }
  function save() {
    if (clockIssue || !preview || !acknowledged || selected || issues.some(Boolean) || numericIssue || cmaIssue) return
    void run(signal => confirmMandate(draft, preview.preview_hash, signal), version => {
      setSelected(version); setPreview(version.assessment ?? null)
      updateAllocationJourney({ mandateId: version.id })
      setCatalog(current => current && ({ ...current, mandates: [version, ...current.mandates] }))
      setNotice('已保存不可变目标版本；诊断状态与模型依据一并保留。')
    })
  }
  function readVersion(id: string) {
    void run(signal => getMandate(id, signal), version => {
      const assessment = version.assessment?.preview_hash ? version.assessment : null
      setDraft(assessment?.request ?? { definition: version.definition, cma_id: null,
        simulation_paths: 2000, seed: 42, uncertainty_penalty: 1 })
      updateAllocationJourney({ mandateId: version.id })
      setSelected(version); setPreview(assessment); setAcknowledged(false); setStep(assessment ? 3 : 0)
      setNotice('正在查看已保存版本，输入只读；复制为新研究后才能修改。')
    })
  }

  return <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6">
    <header><h1 className="text-2xl font-semibold text-slate-900">投资目标与边界</h1><p className="mt-2 text-sm leading-6 text-slate-600">先明确成功标准和支付需求，再判断目标能否实现。资金测算、市场假设和当前应用资格分别核验。</p></header>
    <nav aria-label="投资目标步骤" className="grid grid-cols-2 gap-2 border-b border-slate-200 pb-4 sm:grid-cols-4">{steps.map((label, index) => <button key={label} type="button" aria-current={index === step ? 'step' : undefined}
      disabled={(!selected && ((index > 0 && Boolean(issues[0])) || (index > 1 && Boolean(issues[1])))) || (index === 3 && !preview)} onClick={() => setStep(index)}
      className={`min-h-11 rounded-lg p-3 text-left text-sm disabled:cursor-not-allowed disabled:opacity-50 ${index === step ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{index + 1}. {label}</button>)}</nav>
    <Feedback error={error || clockIssue} notice={notice} />
    {selected && <div className="flex flex-wrap items-center gap-3 rounded-lg bg-slate-50 p-3"><p className="text-sm text-slate-700">只读版本：{selected.name} · {statusLabel(selected.assessment?.status ?? selected.assessment_status)}</p>
      <Button onClick={() => { invalidate(); setStep(0); setNotice('已复制输入，原版本未改变；请核对研究日期后重新诊断。') }}>复制为新研究</Button>
      <Button onClick={() => { invalidate(); setDraft({ definition: newMandate(cutoff), cma_id: null, simulation_paths: 2000, seed: 42, uncertainty_penalty: 1 }); setStep(0); setNotice('') }}>建立新目标</Button>
      <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={allocationJourneyPath('pool')}>下一步：确定投资范围 →</Link>
      <Link className="inline-flex min-h-10 items-center text-sm font-semibold text-accent-800 underline" to={`/pre-investment/saa/policy?mandate=${encodeURIComponent(selected.id)}`}>使用此目标进入长期配置 →</Link>
    </div>}
    <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold text-slate-900">{steps[step]}</h2>
    <div className="grid items-start gap-5 xl:grid-cols-[minmax(0,3fr)_minmax(230px,1fr)]">
      <section className={`${sectionClass} min-w-0 space-y-5`} aria-label="投资目标编辑">
        {step < 2 && <fieldset disabled={Boolean(selected)} className="min-w-0 space-y-5">
          {step === 0 ? <><InstitutionalFields value={definition} onChange={update} /><GoalFields value={definition} onChange={update} catalog={catalog} cutoff={cutoff} /></>
            : <><BoundaryFields value={definition} onChange={update} /><AssetAuthorizations value={definition} catalog={catalog} onChange={update} /></>}
        </fieldset>}
        {step === 2 && <>
          <Field label="用于诊断的CMA版本" hint="只列出同研究日、币种及投资期限的版本，不用未来假设评估过去的目标。"><select className={inputClass} value={draft.cma_id ?? ''} disabled={Boolean(selected) || loading} onChange={e => updateStudy({ cma_id: e.target.value || null })}><option value="">暂不使用CMA，只做输入与资金测算</option>{matchingCmas.map(c => <option key={c.id} value={c.id}>{c.name} · {c.alloc_name} · {c.as_of}</option>)}</select></Field>
          {!matchingCmas.length && <p className="text-sm leading-6 text-slate-600">暂无匹配CMA。可以先运行资金测算并保存“输入版本”，再进入长期配置建立CMA；以后返回此页继续诊断，不会用示例数据补概率。</p>}
          <details className="border-t border-slate-200 pt-4"><summary className="cursor-pointer text-sm font-medium">模拟与不确定性设置</summary><fieldset disabled={Boolean(selected)} className="mt-3 grid min-w-0 gap-4 sm:grid-cols-3">
            <Field label="模拟路径数"><NumberInput className={inputClass} value={draft.simulation_paths} onValueChange={n => updateStudy({ simulation_paths: n })} /></Field>
            <Field label="模拟随机种子"><NumberInput className={inputClass} value={draft.seed} onValueChange={n => updateStudy({ seed: n })} /></Field>
            <Field label="均值不确定性惩罚倍数"><NumberInput className={inputClass} value={draft.uncertainty_penalty} onValueChange={n => updateStudy({ uncertainty_penalty: n })} /></Field>
          </fieldset></details>
          <Button tone="primary" disabled={busy || Boolean(clockIssue) || Boolean(selected) || issues.some(Boolean) || Boolean(numericIssue) || Boolean(cmaIssue)} onClick={diagnose}>{busy ? '正在测算资金与目标…' : '运行目标诊断'}</Button>
          {busy && <p role="status" className="text-sm text-slate-600">正在按当前输入计算；尚无完成结果，不显示预测数值。</p>}
          {preview ? <MandateResults key={preview.preview_hash} value={preview} /> : <p className="text-sm text-slate-600">运行后在这里查看资金要求、约束冲突和可用CMA下的量化诊断。</p>}
        </>}
        {preview?.institutional_diagnostics && step >= 2 && <InstitutionalResults value={preview.institutional_diagnostics} />}
        {step === 3 && preview && <>
          <h3 className="text-base font-semibold">确认的是目标与边界，不是收益承诺</h3>
          <dl className="grid gap-4 sm:grid-cols-2"><div><dt className="text-xs text-slate-600">成功标准</dt><dd className="mt-1 text-sm">{objectiveLabels[definition.objective_kind ?? 'absolute_return']} · {definition.horizon_years}年</dd></div>
            <div><dt className="text-xs text-slate-600">硬风险边界</dt><dd className="mt-1 text-sm tabular-nums">波动 ≤ {percentText(definition.max_volatility)}；TAA ≤ {percentText(definition.max_tracking_error)}</dd></div>
            <div><dt className="text-xs text-slate-600">诊断状态</dt><dd className="mt-1 text-sm">{statusLabel(preview.status)} · {preview.cma?.name ?? '未使用CMA'}</dd></div>
            <div><dt className="text-xs text-slate-600">研究 / 复核日期</dt><dd className="mt-1 text-sm tabular-nums">{definition.as_of} / {definition.review_date}</dd></div></dl>
          {preview.funding && <p className="text-sm leading-6 text-slate-700">可投资本金 {amountText(preview.funding.investable_capital)} {definition.currency}，期末名义目标 {amountText(preview.funding.nominal_terminal_target)}，所需扣费前年复合收益 {percentText(preview.funding.required_effective_return)}。这不是市场预期收益。</p>}
          <p className="text-sm leading-6 text-slate-700">边界依据：{definition.boundary_reason}</p>
          <p className="text-xs leading-5 text-slate-600">模型偏好：风险厌恶系数 {definition.risk_aversion}；复核方式 {({ monthly: '每月', quarterly: '每季', annually: '每年', threshold: '触及阈值' })[definition.rebalance_policy]}。只记录复核约定，不自动交易。</p>
          {preview.status !== 'diagnosed' && <p className="text-sm leading-6 text-amber-800">此次保存仅固定输入及未通过/未诊断的状态。SAA采纳仍须实际计算并满足目标，不能凭“已保存”绕过检查。</p>}
          {definition.review_date <= today() && <p className="text-sm text-amber-800">这是已过复核日的历史目标，可以保存研究，但不能直接用于当前应用。</p>}
          {!selected && <label className="flex min-h-11 items-start gap-2 text-sm leading-6"><input className="mt-1.5" type="checkbox" checked={acknowledged} onChange={e => setAcknowledged(e.target.checked)} />我已核对输入、诊断状态和模型限制；此确认不是外部审批或收益保证。</label>}
          <Button tone="primary" disabled={busy || Boolean(clockIssue) || !acknowledged || Boolean(selected) || issues.some(Boolean) || Boolean(numericIssue) || Boolean(cmaIssue)} onClick={save}>{busy ? '正在复算并保存…' : selected ? '已锁定此目标版本' : '保存新目标版本'}</Button>
          {!selected && !acknowledged && <p className="text-xs text-slate-600">核对后勾选确认，服务端将重新计算并检查输入是否变化。</p>}
        </>}
        {!selected && activeIssue && <p role="status" className="text-sm text-amber-800">{activeIssue}</p>}
        <div className="flex flex-wrap justify-between gap-3 border-t border-slate-200 pt-4">
          <Button disabled={step === 0} onClick={() => setStep(s => s - 1)}>上一步</Button>
          {step < 3 && <Button tone="primary" disabled={(!selected && Boolean(activeIssue)) || (step === 2 && !preview) || busy} onClick={() => setStep(s => s + 1)}>下一步：{steps[step + 1]}</Button>}
        </div>
      </section>
      <aside className="min-w-0 space-y-4" aria-label="目标版本与下一步">
        <div className={`${sectionClass} space-y-3`}><h3 className="text-base font-semibold">当前研究</h3><p className="text-sm text-slate-700">{definition.name || '尚未命名'} · {definition.currency}</p><p className="text-xs leading-5 text-slate-600">填写目标与风险 → 运行诊断 → 核对后保存。未选择CMA时仍可先保存输入，再建立长期假设。</p><Link className="inline-flex min-h-10 items-center text-sm text-accent-800 underline" to="/pre-investment/saa/asset-classes">查看或构建大类</Link></div>
        <details className={`${sectionClass} space-y-3`}><summary className="cursor-pointer text-sm font-medium">已保存的目标（{catalog?.mandates.length ?? 0}）</summary>
          {loading ? <p role="status" className="text-sm text-slate-600">正在读取目标与CMA目录…</p> : catalog?.mandates.length ? <div className="divide-y divide-slate-200">{catalog.mandates.map(v => <button key={v.id} type="button" className="block min-h-11 w-full py-3 text-left text-sm" disabled={busy} onClick={() => readVersion(v.id)}><span className="block font-medium text-slate-800">{v.name}</span><span className="text-xs text-slate-600">{v.definition.currency} · {v.definition.horizon_years} 年 · {statusLabel(v.assessment?.status ?? v.assessment_status)}</span></button>)}</div> : <p className="text-sm text-slate-600">尚无版本；完成诊断后可保存输入和结果。</p>}
          <Button disabled={loading} onClick={() => setReload(n => n + 1)}>重新读取目录</Button>
        </details>
        {!catalog && !loading && <p className="text-sm text-amber-800">目录不可用时仍可填写目标；重新读取后再选择CMA，不能将读取失败当成不存在历史版本。</p>}
      </aside>
    </div>
  </div>
}
