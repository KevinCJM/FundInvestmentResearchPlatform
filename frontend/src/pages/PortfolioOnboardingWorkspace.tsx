import { useMemo, useState, type FormEvent } from 'react'
import StaticDemoBanner from '../components/StaticDemoBanner'
import {
  accountingPolicyVersions,
  actualPortfolioDemoData,
  approvedResearchPlans,
  externalAccounts,
  managerEntities,
  portfolioManagerAssignments,
} from '../app/actualPortfolioDemoData'

type OnboardingMode = 'new' | 'adjustment'

interface PreviewRecord {
  id: string
  mode: OnboardingMode
  title: string
  detail: string
}

export default function PortfolioOnboardingWorkspace() {
  const [mode, setMode] = useState<OnboardingMode>('new')
  const [records, setRecords] = useState<PreviewRecord[]>([])
  const [notice, setNotice] = useState('')
  const [newDraft, setNewDraft] = useState({
    planId: approvedResearchPlans[0].planId,
    name: '',
    externalProductCode: '',
    managerId: managerEntities[0].entityId,
    custodianName: '示例托管银行',
    baseCurrency: 'CNY',
    activationDate: '2026-09-15',
    accountingPolicyVersion: accountingPolicyVersions[0],
    executionAccountId: 'EXEC-FM-DEMO-SSE',
    settlementAccountId: '',
    portfolioManagerId: 'PM-001',
  })
  const [adjustmentDraft, setAdjustmentDraft] = useState({
    portfolioId: actualPortfolioDemoData[0].portfolioId,
    targetVersion: 'TARGET-R19.0',
    effectiveDate: '2026-09-15',
    reason: '',
  })

  const selectedPlan = useMemo(() => approvedResearchPlans.find((plan) => plan.planId === newDraft.planId) ?? approvedResearchPlans[0], [newDraft.planId])
  const selectedPortfolio = useMemo(() => actualPortfolioDemoData.find((portfolio) => portfolio.portfolioId === adjustmentDraft.portfolioId) ?? actualPortfolioDemoData[0], [adjustmentDraft.portfolioId])

  const submitNewPortfolio = (event: FormEvent) => {
    event.preventDefault()
    if (!newDraft.name.trim() || !newDraft.externalProductCode.trim() || !newDraft.settlementAccountId.trim()) return
    const sequence = records.filter((record) => record.mode === 'new').length + 1
    const previewId = `PF-DEMO-NEW-${String(sequence).padStart(3, '0')}`
    setRecords((current) => [...current, {
      id: previewId,
      mode: 'new',
      title: `${newDraft.name.trim()} · ${previewId}`,
      detail: `${selectedPlan.name} ${selectedPlan.version} → 待启用真实组合主数据`,
    }])
    setNotice(`已生成 ${previewId} 的当前会话登记预览；未保存、未依法设立、未开户。`)
    setNewDraft((current) => ({ ...current, name: '', externalProductCode: '' }))
  }

  const submitAdjustment = (event: FormEvent) => {
    event.preventDefault()
    if (!adjustmentDraft.targetVersion.trim() || !adjustmentDraft.reason.trim()) return
    setRecords((current) => [...current, {
      id: `${selectedPortfolio.portfolioId}-${adjustmentDraft.targetVersion}`,
      mode: 'adjustment',
      title: `${selectedPortfolio.name} · ${adjustmentDraft.targetVersion}`,
      detail: `沿用原 portfolio_id ${selectedPortfolio.portfolioId}，仅登记新的目标版本关系`,
    }])
    setNotice(`已生成 ${selectedPortfolio.portfolioId} 的存量组合调整预览；未修改真实持仓、账套或交易。`)
    setAdjustmentDraft((current) => ({ ...current, reason: '' }))
  }

  return <div className="space-y-5" data-testid="portfolio-onboarding-workspace">
    <StaticDemoBanner />
    <header className="rounded-2xl bg-gradient-to-r from-amber-800 via-orange-700 to-slate-900 p-6 text-white shadow-sm"><p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-100">Approved research → actual portfolio master</p><h2 className="mt-2 text-2xl font-bold">组合落地与启用</h2><p className="mt-2 max-w-4xl text-sm leading-6 text-amber-50/90">把平台外已批准研究方案登记为真实组合主数据，或为存量真实组合关联新的目标版本；后续投中、会计与投后统一引用同一个 portfolio_id。</p></header>

    <aside className="border-l-2 border-amber-300 bg-amber-50/60 px-4 py-4" role="note" aria-label="组合落地边界说明（非交互）"><p className="text-xs font-semibold uppercase tracking-[0.16em] text-amber-700/70">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">本页只登记平台主数据</h3><p className="mt-2 text-sm leading-6 text-slate-700">本页不完成基金法律设立、不代表监管备案、不办理证券、资金或托管账户开户，不生成订单，也不执行下单；正式状态只能依据平台外有效文件和授权结果登记。</p></aside>

    <nav className="grid gap-2 rounded-xl border border-slate-200 bg-white p-2 shadow-sm sm:grid-cols-2" aria-label="组合落地模式">
      <button type="button" aria-pressed={mode === 'new'} onClick={() => { setMode('new'); setNotice('') }} className={`min-h-11 rounded-lg px-4 py-3 text-left text-sm font-semibold ${mode === 'new' ? 'bg-amber-800 text-white' : 'text-slate-600 hover:bg-slate-100'}`}><span className="block">研究方案落地</span><span className={`mt-1 block text-xs font-normal ${mode === 'new' ? 'text-amber-100' : 'text-slate-400'}`}>登记一个新的真实组合身份</span></button>
      <button type="button" aria-pressed={mode === 'adjustment'} onClick={() => { setMode('adjustment'); setNotice('') }} className={`min-h-11 rounded-lg px-4 py-3 text-left text-sm font-semibold ${mode === 'adjustment' ? 'bg-amber-800 text-white' : 'text-slate-600 hover:bg-slate-100'}`}><span className="block">存量组合调整</span><span className={`mt-1 block text-xs font-normal ${mode === 'adjustment' ? 'text-amber-100' : 'text-slate-400'}`}>沿用 portfolio_id，登记新目标版本</span></button>
    </nav>

    {mode === 'new' ? <form onSubmit={submitNewPortfolio} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div><h3 className="font-semibold text-slate-900">研究方案落地登记</h3><p className="mt-1 text-sm text-slate-500">仅可选择已记录平台外批准结果的研究方案。</p></div><div className="mt-5 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      <label className="text-sm text-slate-600">已批准研究方案<select aria-label="已批准研究方案" value={newDraft.planId} onChange={(event) => setNewDraft((current) => ({ ...current, planId: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{approvedResearchPlans.map((plan) => <option key={plan.planId} value={plan.planId}>{plan.name} · {plan.version}</option>)}</select></label>
      <label className="text-sm text-slate-600">真实组合名称<input aria-label="真实组合名称" value={newDraft.name} onChange={(event) => setNewDraft((current) => ({ ...current, name: event.target.value }))} placeholder="输入平台登记名称" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">外部产品或委托代码<input aria-label="外部产品或委托代码" value={newDraft.externalProductCode} onChange={(event) => setNewDraft((current) => ({ ...current, externalProductCode: event.target.value }))} placeholder="以外部有效文件为准" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">管理人主体<select aria-label="管理人主体" value={newDraft.managerId} onChange={(event) => setNewDraft((current) => ({ ...current, managerId: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{managerEntities.map((manager) => <option key={manager.entityId} value={manager.entityId}>{manager.name} · {manager.entityId}</option>)}</select></label>
      <label className="text-sm text-slate-600">托管人<input aria-label="托管人" value={newDraft.custodianName} onChange={(event) => setNewDraft((current) => ({ ...current, custodianName: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">记账本位币<select aria-label="记账本位币" value={newDraft.baseCurrency} onChange={(event) => setNewDraft((current) => ({ ...current, baseCurrency: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2"><option>CNY</option><option>USD</option><option>HKD</option></select></label>
      <label className="text-sm text-slate-600">计划启用日<input aria-label="计划启用日" type="date" value={newDraft.activationDate} onChange={(event) => setNewDraft((current) => ({ ...current, activationDate: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">会计政策版本<select aria-label="会计政策版本" value={newDraft.accountingPolicyVersion} onChange={(event) => setNewDraft((current) => ({ ...current, accountingPolicyVersion: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{accountingPolicyVersions.map((version) => <option key={version}>{version}</option>)}</select></label>
      <label className="text-sm text-slate-600">共享执行通道<select aria-label="共享执行通道" value={newDraft.executionAccountId} onChange={(event) => setNewDraft((current) => ({ ...current, executionAccountId: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{externalAccounts.filter((account) => account.kind === '执行通道账户').map((account) => <option key={account.accountId} value={account.accountId}>{account.name} · {account.accountId}</option>)}</select></label>
      <label className="text-sm text-slate-600">基金专用资金账户<input aria-label="基金专用资金账户" value={newDraft.settlementAccountId} onChange={(event) => setNewDraft((current) => ({ ...current, settlementAccountId: event.target.value }))} placeholder="填写外部已开户账户编号" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">首任基金经理<select aria-label="首任基金经理" value={newDraft.portfolioManagerId} onChange={(event) => setNewDraft((current) => ({ ...current, portfolioManagerId: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{Array.from(new Map(portfolioManagerAssignments.map((assignment) => [assignment.portfolioManagerId, assignment])).values()).map((assignment) => <option key={assignment.portfolioManagerId} value={assignment.portfolioManagerId}>{assignment.portfolioManagerName} · {assignment.portfolioManagerId}</option>)}</select></label>
      <div className="rounded-xl bg-slate-50 p-4 text-sm"><p className="text-xs text-slate-500">自动带入的版本关系</p><p className="mt-2 font-semibold text-slate-900">批准 {selectedPlan.approvalRecordId}</p><p className="mt-1 text-slate-600">初始目标 {selectedPlan.targetVersion}</p></div>
    </div><p className="mt-4 text-xs leading-5 text-slate-500">执行通道允许多组合共享；基金专用资金账户必须与新组合的核算主体一致。页面只登记外部已开户结果，不办理开户。</p><button type="submit" disabled={!newDraft.name.trim() || !newDraft.externalProductCode.trim() || !newDraft.settlementAccountId.trim()} className="mt-5 min-h-11 rounded-lg bg-amber-800 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">生成落地登记预览（演示）</button></form> : null}

    {mode === 'adjustment' ? <form onSubmit={submitAdjustment} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><div><h3 className="font-semibold text-slate-900">存量组合目标版本调整</h3><p className="mt-1 text-sm text-slate-500">调整研究关系不会新建组合，也不会覆盖历史目标版本。</p></div><div className="mt-5 grid gap-4 md:grid-cols-2">
      <label className="text-sm text-slate-600">存量真实组合<select aria-label="存量真实组合" value={adjustmentDraft.portfolioId} onChange={(event) => setAdjustmentDraft((current) => ({ ...current, portfolioId: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{actualPortfolioDemoData.map((portfolio) => <option key={portfolio.portfolioId} value={portfolio.portfolioId}>{portfolio.name} · {portfolio.portfolioId}</option>)}</select></label>
      <label className="text-sm text-slate-600">新目标组合版本<input aria-label="新目标组合版本" value={adjustmentDraft.targetVersion} onChange={(event) => setAdjustmentDraft((current) => ({ ...current, targetVersion: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">拟生效日<input aria-label="拟生效日" type="date" value={adjustmentDraft.effectiveDate} onChange={(event) => setAdjustmentDraft((current) => ({ ...current, effectiveDate: event.target.value }))} className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
      <label className="text-sm text-slate-600">调整原因<input aria-label="调整原因" value={adjustmentDraft.reason} onChange={(event) => setAdjustmentDraft((current) => ({ ...current, reason: event.target.value }))} placeholder="记录外部批准或研究变更依据" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
    </div><section className="mt-5 rounded-xl border border-slate-200 bg-slate-50 p-4" aria-label="存量组合不可变身份"><p className="text-xs font-semibold uppercase tracking-wide text-slate-500">保持不变</p><div className="mt-3 grid gap-3 text-sm sm:grid-cols-3"><p><span className="block text-xs text-slate-500">portfolio_id</span><strong>{selectedPortfolio.portfolioId}</strong></p><p><span className="block text-xs text-slate-500">会计主体</span><strong>{selectedPortfolio.accountingEntityId}</strong></p><p><span className="block text-xs text-slate-500">主账套</span><strong>{selectedPortfolio.primaryLedgerId}</strong></p></div></section><button type="submit" disabled={!adjustmentDraft.targetVersion.trim() || !adjustmentDraft.reason.trim()} className="mt-5 min-h-11 rounded-lg bg-amber-800 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">生成存量组合调整预览（演示）</button></form> : null}

    {notice ? <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-950" aria-live="polite">{notice}</p> : null}
    {records.length ? <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">当前会话预览记录</h3><ul className="mt-4 divide-y divide-slate-100">{records.map((record) => <li key={record.id} className="py-3"><div className="flex flex-wrap items-center gap-2"><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${record.mode === 'new' ? 'bg-cyan-100 text-cyan-900' : 'bg-violet-100 text-violet-900'}`}>{record.mode === 'new' ? '新组合落地' : '存量组合调整'}</span><strong className="text-sm text-slate-900">{record.title}</strong></div><p className="mt-2 text-sm text-slate-600">{record.detail}</p></li>)}</ul></section> : null}
  </div>
}
