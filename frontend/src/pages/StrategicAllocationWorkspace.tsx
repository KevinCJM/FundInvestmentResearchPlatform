import { useEffect, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { allocationJourneyPath, readAllocationJourney, updateAllocationJourney, useAllocationDraft } from '../app/allocationJourney'
import { useResearchDay } from '../app/ResearchContext'
import { Button } from '../components/ui'
import { Empty, Feedback, Field, inputClass, percentText, sectionClass, today } from '../components/risk-models/ResearchUI'
import AssumptionEditor from '../components/strategic-allocation/AssumptionEditor'
import PolicyCandidates from '../components/strategic-allocation/PolicyCandidates'
import { GoalCandidateSummary } from '../components/investment-mandate/MandateResults'
import { amountText, objectiveLabels } from '../components/investment-mandate/model'
import {
  completeCma, getCma, getStrategicCatalog, previewCma, previewPolicy, publishCma, publishPolicy, riskReference,
  type CmaDraft, type CmaPreview, type CmaVersion, type PolicyCandidate, type PolicyPreview,
  type PolicyRequest, type RiskReferenceRequest, type StrategicCatalog,
} from '../services/strategicAllocation'
import type { TaaBaseline } from '../services/tacticalAllocation'

interface Draft {
  mandateId: string; allocationName: string; assumptions: CmaDraft | null
  settings: Omit<PolicyRequest, 'mandate_id' | 'cma_id'>; reference: RiskReferenceRequest | null
  policyName: string; reason: string
}
const steps = ['研究范围', '长期假设', '政策比较', '确认与交接']
const historicalLabPath = (allocationName: string) => {
  const journey = readAllocationJourney()
  const query = new URLSearchParams()
  if (allocationName) query.set('alloc', allocationName)
  if (journey.universeId) query.set('universe', journey.universeId)
  return `/pre-investment/saa/allocation-lab${query.size ? `?${query}` : ''}`
}

export default function StrategicAllocationWorkspace() {
  const [params] = useSearchParams()
  const allocationName = params.get('alloc') ?? readAllocationJourney().allocationName ?? ''
  const mandateId = params.get('mandate') ?? ''
  return <StrategicEditor key={`${allocationName}:${mandateId}`} initialAllocation={allocationName} initialMandate={mandateId} />
}

function StrategicEditor({ initialAllocation, initialMandate }: { initialAllocation: string; initialMandate: string }) {
  const navigate = useNavigate()
  const platformDay = useResearchDay()
  const [editor, setEditor] = useAllocationDraft<Draft>(`strategic-policy:${initialAllocation}:${initialMandate}`, () => ({
    mandateId: initialMandate, allocationName: initialAllocation, assumptions: null, reference: null,
    settings: { constraints: {}, group_limits: [], uncertainty_penalty: 1, candidate_count: 2000, seed: 42 }, policyName: '', reason: '',
  }))
  const [catalog, setCatalog] = useState<StrategicCatalog | null>(null)
  const [reload, setReload] = useState(0)
  const [loading, setLoading] = useState(true)
  const [step, setStep] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [cmaPreview, setCmaPreview] = useState<CmaPreview | null>(null)
  const [cmaVersion, setCmaVersion] = useState<CmaVersion | null>(null)
  const [policyPreview, setPolicyPreview] = useState<PolicyPreview | null>(null)
  const [candidate, setCandidate] = useState<PolicyCandidate | null>(null)
  const [savedPolicy, setSavedPolicy] = useState<TaaBaseline | null>(null)
  const generation = useRef(0)
  const previousClock = useRef(platformDay)
  const heading = useRef<HTMLHeadingElement>(null)
  const mandate = catalog?.mandates.find(value => value.id === editor.mandateId)
  const allocation = catalog?.allocations.find(value => value.alloc_name === editor.allocationName)
  const draft = editor.assumptions
  const policyRequest: PolicyRequest = { ...editor.settings, mandate_id: editor.mandateId, cma_id: cmaVersion?.id ?? '' }
  const dateIssue = draft && ((platformDay && draft.as_of > platformDay) || draft.as_of > today()) ? '假设研究日晚于平台知识截止或今天，请调整研究日；草稿会保留。' : ''

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    getStrategicCatalog(controller.signal).then(value => { if (!controller.signal.aborted) setCatalog(value) })
      .catch(reason => { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '长期配置目录读取失败。') })
      .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    return () => controller.abort()
  }, [reload])
  useEffect(() => () => { generation.current += 1 }, [])
  useEffect(() => { heading.current?.focus() }, [step])
  useEffect(() => {
    if (previousClock.current === platformDay) return
    previousClock.current = platformDay
    generation.current += 1
    setBusy(false)
    setError('')
    if (savedPolicy) {
      setNotice('当前展示已保存的历史政策；知识截止日已变化，不代表该政策在新截止日下可用。')
      return
    }
    setPolicyPreview(null)
    setCandidate(null)
    if (!cmaVersion) setCmaPreview(null)
    setStep(current => current > 2 ? 2 : current)
    setNotice('知识截止日已变化，未保存的政策候选已失效，请核对日期后重新比较。')
  }, [platformDay, savedPolicy, cmaVersion])
  useEffect(() => {
    if (!allocation || !mandate || editor.assumptions) return
    const day = mandate.definition.as_of
    const names = allocation.assets.map(asset => asset.id)
    const assumptions: CmaDraft = {
      name: `${allocation.alloc_name} · 长期假设`.slice(0, 120), alloc_name: allocation.alloc_name,
      as_of: day, currency: mandate.definition.currency, horizon_years: mandate.definition.horizon_years,
      return_basis: 'annual_arithmetic_total_return', source: '', basis_confirmed: false,
      assets: names.map(id => ({ id, role: '', liquidity: '', rationale: '', annual_return: NaN, annual_volatility: NaN, mean_uncertainty: NaN })),
      correlation: names.map((_, i) => names.map((__, j) => i === j ? 1 : NaN)), risk_origin: 'manual', risk_reference: null, risk_reference_hash: null,
    }
    setEditor(current => ({ ...current, assumptions, policyName: `${allocation.alloc_name} · 长期政策`.slice(0, 120),
      reference: { alloc_name: allocation.alloc_name, as_of: day, start_date: allocation.coverage?.start_date ?? '', end_date: [day, allocation.coverage?.end_date].filter(Boolean).sort()[0]!, shrinkage: .1, periods_per_year: 252 },
      settings: { ...current.settings, constraints: Object.fromEntries(names.map(id => [id, { min_weight: mandate.definition.asset_limits?.[id]?.min_weight ?? 0, max_weight: mandate.definition.asset_limits?.[id]?.max_weight ?? 1, max_abs_tilt: mandate.definition.max_tracking_error === 0 ? 0 : mandate.definition.asset_limits?.[id]?.max_abs_tilt ?? .1 }])), group_limits: [] } }))
  }, [allocation, mandate, editor.assumptions, platformDay])

  function invalidate(assumptions = false) {
    generation.current += 1
    setBusy(false); setError(''); setNotice(''); setPolicyPreview(null); setCandidate(null); setSavedPolicy(null)
    if (assumptions) { setCmaPreview(null); setCmaVersion(null) }
  }
  function changeDraft(value: CmaDraft) {
    invalidate(true)
    setEditor(current => ({ ...current, assumptions: value }))
  }
  function changeScope(patch: Partial<Pick<Draft, 'mandateId' | 'allocationName'>>) {
    invalidate(true)
    setEditor(current => ({ ...current, ...patch, assumptions: null, reference: null }))
  }
  async function run<T,>(work: () => Promise<T>, consume: (result: T) => void) {
    const token = ++generation.current
    setBusy(true); setError(''); setNotice('')
    try { const result = await work(); if (generation.current === token) consume(result) }
    catch (reason) { if (generation.current === token) setError(reason instanceof Error ? reason.message : '计算失败，请检查输入。') }
    finally { if (generation.current === token) setBusy(false) }
  }
  function loadRisk() {
    if (!draft || !editor.reference || dateIssue) return
    const request = { ...editor.reference, alloc_name: draft.alloc_name, as_of: draft.as_of }
    void run(() => riskReference(request), value => {
      if (value.assets.join('\u0000') !== draft.assets.map(asset => asset.id).join('\u0000')) throw new Error('风险参考资产轴与当前分类不同，已停止回填。')
      changeDraft({ ...draft, assets: draft.assets.map((asset, i) => ({ ...asset, annual_volatility: value.volatility[i] })),
        correlation: value.correlation, risk_origin: 'historical_reference', risk_reference: request, risk_reference_hash: value.preview_hash })
      setNotice(`已读取 ${value.observations} 个共同收益观察期；只回填风险，预期收益由你另行设定。`)
    })
  }
  function validateAssumptions() {
    if (!draft || !completeCma(draft) || dateIssue) return
    void run(() => previewCma(draft), value => setCmaPreview(value))
  }
  function confirmAssumptions() {
    if (!draft || !completeCma(draft) || !cmaPreview || dateIssue) return
    void run(() => publishCma(draft, cmaPreview.preview_hash), value => {
      setCmaVersion(value); setStep(2)
      setCatalog(current => current && ({ ...current, assumptions: [{ id: value.id, name: value.name,
        alloc_name: value.definition.alloc_name, as_of: value.definition.as_of, currency: value.definition.currency,
        horizon_years: value.definition.horizon_years }, ...current.assumptions] }))
      setNotice('长期假设已保存为不可变版本。接下来比较政策权重。')
    })
  }
  function loadAssumptions(id: string) {
    if (!id) return
    invalidate(true)
    void run(() => getCma(id), value => {
      if (value.id !== id || value.definition.alloc_name !== editor.allocationName || value.definition.currency !== mandate?.definition.currency || value.definition.horizon_years !== mandate?.definition.horizon_years) throw new Error('该假设与当前目标或分类不一致，请选择相同范围、币种和期限的版本。')
      setEditor(current => ({ ...current, assumptions: value.definition }))
      setCmaVersion(value); setCmaPreview(value); setStep(2)
      setNotice('已引用保存的假设；政策比较时服务端会重新检查来源是否变化。')
    })
  }
  function adopt() {
    if (dateIssue || !candidate || !policyPreview || candidate.goal_check?.within_limits === false || !editor.reason.trim() || !editor.policyName.trim()) return
    void run(() => publishPolicy(policyRequest, policyPreview.preview_hash, candidate.id, editor.policyName, editor.reason), value => {
      setSavedPolicy(value)
      updateAllocationJourney({ allocationName: value.alloc_name, universeId: value.universe_snapshot_id ?? undefined, baselineId: value.id, taaRunId: undefined })
      setCatalog(current => current && ({ ...current, policies: [value, ...current.policies] }))
      setNotice('长期政策已确认。目标、假设、产品映射和预算已锁定；这是研究采纳，不是外部审批。')
    })
  }

  return <div className="mx-auto max-w-6xl space-y-5 p-4 sm:p-6">
    <header><h1 className="text-2xl font-semibold text-slate-900">长期政策配置</h1><p className="mt-2 text-sm leading-6 text-slate-600">从投资目标与前瞻假设出发，选定长期资金比例，再研究是否需要战术偏离。历史有效前沿属于独立实验工具，不替代前瞻 CMA。</p><Link className="mt-2 inline-flex min-h-10 items-center text-sm font-medium text-accent-800 underline" to={historicalLabPath(editor.allocationName)}>打开历史有效前沿与策略回测</Link></header>
    <nav aria-label="长期配置步骤" className="grid gap-2 border-b border-slate-200 pb-4 sm:grid-cols-4">{steps.map((label, i) => <button key={label} type="button" aria-current={step === i ? 'step' : undefined} disabled={busy || (i > 0 && (!draft || !mandate)) || (i > 1 && !cmaVersion) || (i > 2 && !candidate)} onClick={() => setStep(i)} className={`min-h-11 rounded-lg px-3 py-2 text-left text-sm focus-visible:outline focus-visible:outline-accent-600 disabled:cursor-not-allowed disabled:opacity-50 ${step === i ? 'bg-accent-50 font-semibold text-accent-900' : 'text-slate-600 hover:bg-slate-50'}`}>{i + 1}. {label}</button>)}</nav>
    <h2 ref={heading} tabIndex={-1} className="text-lg font-semibold outline-none">{steps[step]}</h2>
    <Feedback error={error || dateIssue} notice={notice} />
    {busy && <p role="status" className="text-sm text-slate-600">正在核验当前输入…</p>}
    {loading ? <p role="status" className="text-sm text-slate-600">正在读取目标与分类…</p> : !catalog ? <Empty title="目录暂时不可用"><Button onClick={() => setReload(value => value + 1)}>重试读取</Button></Empty> : <>
      {step === 0 && <section className={`${sectionClass} space-y-5`} aria-label="目标与大类来源">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="投资目标版本"><select className={inputClass} value={editor.mandateId} onChange={e => changeScope({ mandateId: e.target.value })}><option value="">选择已保存目标</option>{catalog.mandates.map(value => <option key={value.id} value={value.id}>{value.name} · {value.definition.currency} · {value.definition.horizon_years} 年</option>)}</select></Field>
          <Field label="已保存的大类配置"><select className={inputClass} value={editor.allocationName} onChange={e => changeScope({ allocationName: e.target.value })}><option value="">选择大类与代理产品</option>{catalog.allocations.map(value => <option key={value.alloc_name} value={value.alloc_name}>{value.alloc_name}</option>)}</select></Field>
        </div>
        <div className="flex flex-wrap gap-4 text-sm"><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/objectives">建立或复核投资目标</Link><Link className="inline-flex min-h-10 items-center text-accent-800 underline" to="/pre-investment/saa/asset-classes">构建与检查大类</Link></div>
        {mandate && <p className="rounded-lg bg-slate-50 p-3 text-sm leading-6 text-slate-700">目标：{objectiveLabels[mandate.definition.objective_kind ?? 'absolute_return']}{mandate.definition.funding_plan ? `，期末保有${amountText(mandate.definition.funding_plan.terminal_target)} ${mandate.definition.currency}` : mandate.definition.benchmark ? `，相对${mandate.definition.benchmark.name}预期超额至少${percentText(mandate.definition.benchmark.target_excess_return)}` : `，预期年收益至少${percentText(mandate.definition.target_return)}`}；预期波动不超过 {percentText(mandate.definition.max_volatility)}，流动性资产至少 {percentText(mandate.definition.min_liquid_weight)}；政策于 {mandate.definition.review_date} 复核。</p>}
        {allocation && <p className="text-sm leading-6 text-slate-600">大类：{allocation.assets.map(asset => asset.name).join('、')}。后续保留原产品映射，不自动从名称推断经济角色。</p>}
        <Button tone="primary" disabled={!draft || !mandate || !allocation} onClick={() => setStep(1)}>填写长期假设</Button>
        <Field label="或者使用已保存的长期假设"><select className={inputClass} value={cmaVersion?.id ?? ''} disabled={!mandate || !allocation || busy} onChange={e => loadAssumptions(e.target.value)}><option value="">选择假设版本</option>{catalog.assumptions.filter(value => value.alloc_name === editor.allocationName).map(value => <option key={value.id} value={value.id}>{value.name} · {value.as_of} · {value.currency}</option>)}</select></Field>
      </section>}
      {step === 1 && draft && editor.reference && <>
        <AssumptionEditor value={draft} onChange={changeDraft} reference={editor.reference} onReferenceChange={reference => { invalidate(); setEditor(current => ({ ...current, reference })) }} onLoadReference={loadRisk} busy={busy} />
        <section className={`${sectionClass} space-y-3`}><div className="flex flex-wrap gap-3"><Button tone="primary" disabled={busy || !completeCma(draft) || Boolean(dateIssue)} onClick={validateAssumptions}>验证长期假设</Button><Button disabled={busy || !cmaPreview || Boolean(dateIssue)} onClick={confirmAssumptions}>确认保存假设版本</Button></div>
          {!completeCma(draft) && <p className="text-sm text-slate-600">请填写全部预期数值、相关性、分类理由与来源，并确认口径。</p>}
          {cmaPreview && <><p role="status" className="text-sm text-accent-800">资产轴和风险矩阵已通过校验；确认保存前不会写入研究库。</p><details><summary className="cursor-pointer text-sm text-slate-700">查看假设依据与限制</summary>{cmaPreview.warnings.map((warning, index) => <p className="mt-2 text-xs leading-5 text-slate-600" key={index}>{warning}</p>)}</details></>}
        </section>
      </>}
      {step === 2 && draft && cmaVersion && <PolicyCandidates value={policyRequest} assets={draft.assets.map(asset => asset.id)} result={policyPreview} busy={busy} compareDisabled={Boolean(dateIssue)}
        onChange={value => { invalidate(); const { mandate_id: _mandate, cma_id: _cma, ...settings } = value; setEditor(current => ({ ...current, settings })) }}
        onCompare={() => { if (!dateIssue) void run(() => previewPolicy(policyRequest), value => { setPolicyPreview(value); setCandidate(null) }) }}
        onSelect={value => { setCandidate(value); setStep(3) }} />}
      {step === 3 && candidate && policyPreview && <section className={`${sectionClass} space-y-5`} aria-label="政策采纳确认">
        <h2 className="text-lg font-semibold">这份长期政策是否符合你的判断？</h2><p className="text-sm leading-6 text-slate-600">{candidate.name} · 预期年收益 {percentText(candidate.metrics.expected_return)} · 预期年波动 {percentText(candidate.metrics.volatility)}。这些是冻结假设下的计算，不是业绩预测保证。</p>
        <GoalCandidateSummary candidate={candidate} />
        {policyPreview.current_application_eligible === false && <p className="text-sm text-amber-800">此政策按历史研究日评价，可以保存研究，已到复核日则不能直接用于当前产品应用。</p>}
        <dl className="grid gap-3 sm:grid-cols-3">{Object.entries(candidate.weights).map(([asset, weight]) => <div key={asset} className="rounded-lg bg-slate-50 p-3"><dt className="text-sm text-slate-600">{asset}</dt><dd className="mt-1 text-lg font-semibold tabular-nums">{percentText(weight)}</dd></div>)}</dl>
        <Field label="政策版本名称"><input className={inputClass} maxLength={120} value={editor.policyName} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, policyName: e.target.value })) }} /></Field>
        <Field label="采纳理由与复核关注点" hint="记录为何采用、哪些假设变化会触发复核。"><textarea rows={3} className={inputClass} value={editor.reason} disabled={busy} onChange={e => { setSavedPolicy(null); setEditor(current => ({ ...current, reason: e.target.value })) }} /></Field>
        <p className="text-xs leading-5 text-slate-600">确认后生成不可变 SAA 基线，TAA 继承目标、资产映射、权重及约束。政策再平衡约定与 TAA 日频研究回测并非同一执行规则。</p>
        <div className="flex flex-wrap gap-3"><Button tone="primary" disabled={busy || Boolean(dateIssue) || Boolean(savedPolicy) || candidate.goal_check?.within_limits === false || editor.reason.trim().length < 5 || !editor.policyName.trim()} onClick={adopt}>{savedPolicy ? '长期政策已确认' : '确认采用此长期政策'}</Button>
          {savedPolicy && <Button tone="primary" onClick={() => navigate(allocationJourneyPath('taa', { ...readAllocationJourney(), baselineId: savedPolicy.id, taaRunId: undefined }))}>进入 TAA，研究是否需要偏离 →</Button>}</div>
      </section>}
      <details className={`${sectionClass} text-sm`}><summary className="cursor-pointer font-medium">已保存政策与历史研究工具</summary><div className="mt-3 space-y-2">{catalog.policies.length ? catalog.policies.map(value => <Link key={value.id} className="flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/taa?baseline=${encodeURIComponent(value.id)}`}>{value.name} · {value.as_of}</Link>) : <p className="text-slate-600">尚无已确认政策。</p>}<Link className="inline-flex min-h-10 items-center text-accent-800 underline" to={`/pre-investment/saa/allocation-lab?alloc=${encodeURIComponent(editor.allocationName)}`}>历史有效前沿、风险预算与策略回测</Link><p className="text-xs leading-5 text-slate-600">历史实验保留原功能；不能将历史最优权重直接当成长期前瞻政策。</p></div></details>
    </>}
  </div>
}
