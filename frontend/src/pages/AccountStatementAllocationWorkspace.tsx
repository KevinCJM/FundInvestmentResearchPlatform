import { useEffect, useMemo, useState } from 'react'
import { actualPortfolioDemoData, externalAccounts, getPortfolioManagerAssignments, getSettlementAccountForPortfolio } from '../app/actualPortfolioDemoData'
import StaticDemoBanner from '../components/StaticDemoBanner'
import { evaluateNumericControls, type NumericControlResult } from '../services/businessNumeric'

type ReconciliationStatus = '全部' | '已匹配' | '待认领'

const channelAllocations = [
  { allocationId: 'SET-ALLOC-001', portfolioId: 'PF-DEMO-01', assignmentId: 'PMA-PF01-002', amount: 4_032_000, settlementAccountId: 'CASH-PF01-CNY', status: '已匹配' },
  { allocationId: 'SET-ALLOC-002', portfolioId: 'PF-DEMO-03', assignmentId: 'PMA-PF03-001', amount: 2_688_000, settlementAccountId: 'CASH-PF03-CNY', status: '已匹配' },
]

const cashStatements = [
  { sourceId: 'BANK-PF01-20260902-088', accountId: 'CASH-PF01-CNY', valueDate: '2026-09-02', amount: -4_032_000, reference: 'SET-ALLOC-001', portfolioId: 'PF-DEMO-01', status: '已匹配' },
  { sourceId: 'BANK-PF03-20260902-041', accountId: 'CASH-PF03-CNY', valueDate: '2026-09-02', amount: -2_688_000, reference: 'SET-ALLOC-002', portfolioId: 'PF-DEMO-03', status: '已匹配' },
  { sourceId: 'BANK-PF01-20260902-093', accountId: 'CASH-PF01-CNY', valueDate: '2026-09-02', amount: -1_200, reference: '银行手续费待取得回单', portfolioId: 'PF-DEMO-01', status: '待认领' },
] as const

export default function AccountStatementAllocationWorkspace() {
  const [status, setStatus] = useState<ReconciliationStatus>('全部')
  const [allocationControl, setAllocationControl] = useState<NumericControlResult | null>(null)
  const [controlError, setControlError] = useState('')
  const visibleStatements = useMemo(() => cashStatements.filter((item) => status === '全部' || item.status === status), [status])
  const sourceTotal = 6_720_000

  useEffect(() => {
    const controller = new AbortController()
    setAllocationControl(null)
    setControlError('')
    evaluateNumericControls([{
      key: 'statement-allocation',
      values: channelAllocations.map((item) => item.amount),
      target: sourceTotal,
      tolerance: 0.005,
    }], controller.signal)
      .then((response) => setAllocationControl(response.items[0] ?? null))
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setControlError('NJIT 分配校验暂不可用')
      })
    return () => controller.abort()
  }, [])

  return <div className="space-y-5" data-testid="account-statement-allocation-workspace">
    <StaticDemoBanner />
    <header className="rounded-xl bg-gradient-to-r from-accent-950 via-accent-900 to-slate-900 p-6 text-white shadow-sm"><p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-100/80">External account records → allocations → settlement accounts → booking</p><h2 className="mt-2 text-2xl font-bold">外部账户流水、拆分与结算匹配</h2><p className="mt-2 max-w-5xl text-sm leading-6 text-accent-50/85">先保留外部账户原始记录，再按成交分配关系拆到真实组合；共享执行通道的回单与各基金专用资金账户流水分别核对。</p></header>

    <section className="rounded-xl border border-accent-200 bg-white p-5 shadow-sm">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between"><div><p className="text-xs font-semibold tracking-wide text-accent-600">共享执行通道回单</p><h3 className="mt-1 font-semibold text-slate-950">EXEC-FM-DEMO-CIBM · 银行间成交结算通知</h3><p className="mt-1 text-sm text-slate-600">来源记录 TRADE-CONF-20260901-027 · 总额 ¥{sourceTotal.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}</p></div><p className={`rounded-lg px-3 py-2 text-sm font-semibold ${allocationControl?.within_tolerance ? 'bg-emerald-50 text-emerald-800' : allocationControl ? 'bg-rose-50 text-rose-800' : 'bg-amber-50 text-amber-800'}`}>{allocationControl ? `分配尾差 ¥${allocationControl.difference.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}` : controlError || 'NJIT 分配校验中…'}</p></div>
      <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[1080px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['分配编号', '真实组合', '基金经理任职 / 投资单元', '分配金额', '组合专用资金账户', '账户归属校验', '状态'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{channelAllocations.map((allocation) => {
        const portfolio = actualPortfolioDemoData.find((item) => item.portfolioId === allocation.portfolioId)!
        const assignment = getPortfolioManagerAssignments(allocation.portfolioId).find((item) => item.assignmentId === allocation.assignmentId)!
        const settlementAccount = getSettlementAccountForPortfolio(allocation.portfolioId)
        return <tr key={allocation.allocationId} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-xs text-slate-600">{allocation.allocationId}</td><td className="px-4 py-3 font-semibold text-slate-900">{portfolio.name}<br /><span className="font-mono text-xs font-normal text-slate-600">{portfolio.portfolioId}</span></td><td className="px-4 py-3 text-slate-700">{assignment.portfolioManagerName} · {assignment.role}<br /><span className="font-mono text-xs text-slate-600">{assignment.assignmentId} / {assignment.sleeveId}</span></td><td className="px-4 py-3 tabular-nums">¥{allocation.amount.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{allocation.settlementAccountId}</td><td className="px-4 py-3 text-slate-600">{settlementAccount?.legalOwnerEntityId === portfolio.accountingEntityId ? '核算主体一致' : '不一致，硬阻断'}</td><td className="px-4 py-3 font-semibold text-emerald-700">{allocation.status}</td></tr>
      })}</tbody></table></div>
    </section>

    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 sm:flex-row sm:items-end sm:justify-between"><div><h3 className="font-semibold text-slate-900">基金专用资金账户流水</h3><p className="mt-1 text-sm text-slate-600">资金流水按账户原样留存；待认领项目进入悬账队列，不自动归入费用或某位基金经理。</p></div><label className="text-sm text-slate-600">匹配状态<select aria-label="筛选账户流水状态" value={status} onChange={(event) => setStatus(event.target.value as ReconciliationStatus)} className="mt-1 block min-w-40 rounded-xl border border-slate-300 bg-white px-3 py-2"><option>全部</option><option>已匹配</option><option>待认领</option></select></label></div>
      <div className="overflow-x-auto"><table className="w-full min-w-[1040px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['来源流水号', '资金账户', '账户归属', '价值日', '金额', '关联分配/摘要', '匹配状态'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{visibleStatements.map((statement) => {
        const account = externalAccounts.find((item) => item.accountId === statement.accountId)!
        return <tr key={statement.sourceId} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-xs text-slate-600">{statement.sourceId}</td><td className="px-4 py-3 font-mono text-xs font-semibold text-slate-800">{statement.accountId}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{account.legalOwnerEntityId}</td><td className="px-4 py-3 text-slate-600">{statement.valueDate}</td><td className="px-4 py-3 tabular-nums">¥{statement.amount.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}</td><td className="px-4 py-3 text-slate-600">{statement.reference}</td><td className={`px-4 py-3 font-semibold ${statement.status === '已匹配' ? 'text-emerald-700' : 'text-amber-700'}`}>{statement.status}</td></tr>
      })}</tbody></table></div>
    </section>

    <aside className="border-l-2 border-accent-300 bg-accent-50/50 px-4 py-4" role="note" aria-label="账户拆分规则说明（非交互）"><p className="text-xs font-semibold tracking-wide text-accent-700">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">共享的是执行通道，不是基金财产</h3><p className="mt-2 text-sm leading-6 text-slate-700">共享通道回单可以按组合拆分；每只基金的实际现金付款仍需与其专用资金账户逐笔核对。只有分配数量、金额、组合、经理任职、核算主体和账户关系全部有效，才能生成组合级 Booking。</p></aside>
  </div>
}
