import { useEffect, useState } from 'react'
import {
  actualPortfolioDemoData,
  externalAccounts,
  getPortfolioManagerAssignments,
  getSettlementAccountForPortfolio,
} from '../app/actualPortfolioDemoData'
import StaticDemoBanner from '../components/StaticDemoBanner'
import { evaluateTradeAllocation, type TradeAllocationResponse } from '../services/businessNumeric'

const executionAccount = externalAccounts.find((account) => account.accountId === 'EXEC-FM-DEMO-SSE')!
const filledQuantity = 1_000_000
const averagePrice = 4.2

const initialAllocations = [
  { portfolioId: 'PF-DEMO-01', assignmentId: 'PMA-PF01-002', plannedQuantity: 600_000, allocatedQuantity: 600_000 },
  { portfolioId: 'PF-DEMO-03', assignmentId: 'PMA-PF03-001', plannedQuantity: 400_000, allocatedQuantity: 400_000 },
]

export default function TradeAllocationWorkspace() {
  const [quantities, setQuantities] = useState<Record<string, string>>(() => Object.fromEntries(initialAllocations.map((item) => [item.portfolioId, String(item.allocatedQuantity)])))
  const [allocationResult, setAllocationResult] = useState<TradeAllocationResponse | null>(null)
  const [allocationError, setAllocationError] = useState('')

  useEffect(() => {
    const controller = new AbortController()
    setAllocationResult(null)
    setAllocationError('')
    evaluateTradeAllocation({
      source_quantity: filledQuantity,
      unit_price: averagePrice,
      allocations: initialAllocations.map((item) => ({
        key: item.portfolioId,
        quantity: Number(quantities[item.portfolioId]) || 0,
      })),
      tolerance: 0.000001,
    }, controller.signal)
      .then(setAllocationResult)
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setAllocationError('NJIT 成交分配校验暂不可用')
      })
    return () => controller.abort()
  }, [quantities])

  return <div className="space-y-5" data-testid="trade-allocation-workspace">
    <StaticDemoBanner />
    <header className="rounded-xl bg-gradient-to-r from-amber-900 via-orange-800 to-slate-900 p-6 text-white shadow-sm">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-amber-100/80">Portfolio instructions → aggregate order → fills → allocations</p>
      <h2 className="mt-2 text-2xl font-bold">汇总订单与公平交易分配</h2>
      <p className="mt-2 max-w-5xl text-sm leading-6 text-amber-50">多个组合及基金经理的独立指令可通过同一执行通道汇总成交；成交后必须按预分配规则拆回组合、投资单元和专用结算账户。</p>
    </header>

    <section className="grid gap-4 lg:grid-cols-[1.1fr_1fr]">
      <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-xs font-semibold tracking-wide text-slate-600">共享执行通道</p>
        <h3 className="mt-2 font-semibold text-slate-950">{executionAccount.name}</h3>
        <dl className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
          <div><dt className="text-slate-600">账户编号</dt><dd className="mt-1 font-mono font-semibold text-slate-900">{executionAccount.accountId}</dd></div>
          <div><dt className="text-slate-600">法定归属</dt><dd className="mt-1 font-mono font-semibold text-slate-900">{executionAccount.legalOwnerEntityId}</dd></div>
          <div><dt className="text-slate-600">共享规则</dt><dd className="mt-1 font-semibold text-accent-700">{executionAccount.sharingPolicy}</dd></div>
          <div><dt className="text-slate-600">业务边界</dt><dd className="mt-1 text-slate-700">只汇总执行，不承载共享基金资产</dd></div>
        </dl>
      </article>
      <article className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-xs font-semibold tracking-wide text-slate-600">成交汇总</p>
        <h3 className="mt-2 font-semibold text-slate-950">ORD-DEMO-20260901-001 · 510300.SH 买入</h3>
        <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
          <div><dt className="text-slate-600">成交数量</dt><dd className="mt-1 text-xl font-bold tabular-nums text-slate-950">{filledQuantity.toLocaleString('zh-CN')}</dd></div>
          <div><dt className="text-slate-600">成交均价</dt><dd className="mt-1 text-xl font-bold tabular-nums text-slate-950">¥{averagePrice.toFixed(4)}</dd></div>
          <div><dt className="text-slate-600">成交金额</dt><dd className="mt-1 font-semibold tabular-nums text-slate-900">{allocationResult ? `¥${allocationResult.source_amount.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}` : allocationError || 'NJIT 计算中…'}</dd></div>
          <div><dt className="text-slate-600">分配规则版本</dt><dd className="mt-1 font-mono text-xs font-semibold text-slate-900">FAIR-ALLOC-v3</dd></div>
        </dl>
      </article>
    </section>

    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 px-5 py-4"><h3 className="font-semibold text-slate-900">成交分配明细</h3><p className="mt-1 text-sm text-slate-600">示例数量可修改；刷新页面后恢复，不生成订单、成交或真实 Booking。</p></div>
      <div className="overflow-x-auto"><table className="w-full min-w-[1180px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['分配编号', '真实组合', '基金经理任职', '内部投资单元', '事前指令数量', '成交分配数量', '分配金额', '专用资金结算账户'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{initialAllocations.map((allocation, index) => {
        const portfolio = actualPortfolioDemoData.find((item) => item.portfolioId === allocation.portfolioId)!
        const assignment = getPortfolioManagerAssignments(allocation.portfolioId).find((item) => item.assignmentId === allocation.assignmentId)!
        const settlementAccount = getSettlementAccountForPortfolio(allocation.portfolioId)
        const computed = allocationResult?.allocations.find((item) => item.key === allocation.portfolioId)
        return <tr key={allocation.portfolioId} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-xs text-slate-600">ALLOC-DEMO-{String(index + 1).padStart(3, '0')}</td><td className="px-4 py-3 font-semibold text-slate-900">{portfolio.name}<br /><span className="font-mono text-xs font-normal text-slate-600">{portfolio.portfolioId}</span></td><td className="px-4 py-3 text-slate-700">{assignment.portfolioManagerName} · {assignment.role}<br /><span className="font-mono text-xs text-slate-600">{assignment.assignmentId}</span></td><td className="px-4 py-3 font-mono text-xs text-slate-600">{assignment.sleeveId}</td><td className="px-4 py-3 tabular-nums">{allocation.plannedQuantity.toLocaleString('zh-CN')}</td><td className="px-4 py-3"><input aria-label={`${portfolio.name}成交分配数量`} inputMode="numeric" value={quantities[allocation.portfolioId]} onChange={(event) => setQuantities((current) => ({ ...current, [allocation.portfolioId]: event.target.value }))} className="w-36 rounded-lg border border-slate-300 px-3 py-2 tabular-nums" /></td><td className="px-4 py-3 tabular-nums">{computed ? `¥${computed.amount.toLocaleString('zh-CN', { minimumFractionDigits: 2 })}` : '—'}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{settlementAccount?.accountId ?? '缺失'}</td></tr>
      })}</tbody></table></div>
      <div className={`flex flex-col gap-2 border-t px-5 py-4 text-sm font-semibold sm:flex-row sm:items-center sm:justify-between ${allocationResult?.balanced ? 'border-emerald-200 bg-emerald-50 text-emerald-900' : allocationResult ? 'border-rose-200 bg-rose-50 text-rose-900' : 'border-amber-200 bg-amber-50 text-amber-900'}`}>{allocationResult ? <><span>成交 {allocationResult.source_quantity.toLocaleString('zh-CN')} − 已分配 {allocationResult.allocated_total.toLocaleString('zh-CN')} = 尾差 {allocationResult.residual.toLocaleString('zh-CN')}</span><span>{allocationResult.balanced ? '分配校验通过，可进入确认与交收匹配' : '硬阻断：不得形成组合 Booking'}</span></> : <span>{allocationError || 'NJIT 成交分配校验中…'}</span>}</div>
    </section>

    <aside className="border-l-2 border-amber-300 bg-amber-50/60 px-4 py-4" role="note" aria-label="公平交易规则说明（非交互）"><p className="text-xs font-semibold tracking-wide text-amber-700">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">分配不能在成交后随意挑选组合</h3><p className="mt-2 text-sm leading-6 text-slate-700">参与组合、目标数量和例外规则原则上应在下单前冻结；部分成交按生效规则比例分配，任何偏离都要记录原因、审批人和版本。每条分配只能指向一个真实组合、一个有效基金经理任职关系和该组合可用的结算账户。</p></aside>
  </div>
}
