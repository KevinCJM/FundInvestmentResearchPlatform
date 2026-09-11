import { useEffect, useMemo, useState } from 'react'
import { useActualPortfolio } from '../app/ActualPortfolioContext'
import {
  getAccountRelationshipsForPortfolio,
  getExternalAccount,
  getPortfolioManagerAssignments,
  getSettlementAccountForPortfolio,
} from '../app/actualPortfolioDemoData'
import StaticDemoBanner from '../components/StaticDemoBanner'
import { evaluateLedgerSummary, type LedgerSummaryResponse } from '../services/businessNumeric'

type BookTreatment = '表内确认' | '表外披露/备查' | '表内确认＋表外披露' | '不入该账' | '待规则判断'
type PerformanceFlow = '内部投资交易' | '外部资金流' | '非现金事项'
type AccountingEntityType = '组合/基金账' | '管理人公司账'
type EventScope = '组合事项' | '跨主体事项' | '管理人单独事项'

interface BookingRecord {
  id: string
  tradeDate: string
  accountingDate?: string
  settlementDate: string
  sourceAccountId: string
  orderId: string
  executionId: string
  allocationId: string
  portfolio: string
  manager: string
  portfolioManagerAssignmentId: string
  sleeveId: string
  eventScope: EventScope
  eventType: string
  instrument: string
  amount: string
  portfolioTreatment: BookTreatment
  managerTreatment: BookTreatment
  performanceFlow: PerformanceFlow
  status: string
}

interface JournalLine {
  accountCode: string
  accountName: string
  accountClass: string
  debit: number
  credit: number
}

interface JournalVoucher {
  id: string
  sourceId: string
  entity: AccountingEntityType
  entityId: string
  accountingDate: string
  eventPhase: string
  policyVersion: string
  status: string
  lines: JournalLine[]
}

const initialRecords: BookingRecord[] = [
  { id: 'BK-240831-001', tradeDate: '2026-08-31', settlementDate: '2026-09-01', sourceAccountId: 'EXEC-FM-DEMO-SSE', orderId: 'ORD-240831-001', executionId: 'FILL-240831-001', allocationId: 'ALLOC-240831-001-A', portfolio: 'PF-DEMO-01', manager: 'FM-DEMO', portfolioManagerAssignmentId: 'PMA-PF01-001', sleeveId: 'SLV-PF01-CORE', eventScope: '组合事项', eventType: 'ETF 买入', instrument: '510300.SH', amount: '490,008.00', portfolioTreatment: '表内确认', managerTreatment: '不入该账', performanceFlow: '内部投资交易', status: '已生成两张待复核凭证' },
  { id: 'BK-240831-002', tradeDate: '2026-08-31', settlementDate: '2026-09-02', sourceAccountId: 'CASH-PF01-CNY', orderId: 'ORD-240831-002', executionId: 'CONF-240831-002', allocationId: 'ALLOC-240831-002-A', portfolio: 'PF-DEMO-01', manager: 'FM-DEMO', portfolioManagerAssignmentId: 'PMA-PF01-002', sleeveId: 'SLV-PF01-TACTICAL', eventScope: '组合事项', eventType: '底层基金赎回', instrument: '000012.OF', amount: '3,500,000.00', portfolioTreatment: '表内确认', managerTreatment: '不入该账', performanceFlow: '内部投资交易', status: '待交收' },
  { id: 'BK-240831-003', tradeDate: '2026-08-31', settlementDate: '2026-08-31', sourceAccountId: 'CASH-PF01-CNY', orderId: '不适用', executionId: 'BANK-240831-003', allocationId: 'CASH-ALLOC-240831-003', portfolio: 'PF-DEMO-01', manager: 'FM-DEMO', portfolioManagerAssignmentId: '不适用', sleeveId: '全组合', eventScope: '组合事项', eventType: '客户资金投入', instrument: 'CASH.CNY', amount: '5,000,000.00', portfolioTreatment: '表内确认', managerTreatment: '不入该账', performanceFlow: '外部资金流', status: '待份额登记核对' },
  { id: 'BK-240831-004', tradeDate: '2026-08-31', settlementDate: '2026-08-31', sourceAccountId: 'CASH-PF01-CNY', orderId: '不适用', executionId: 'FEE-ACCRUAL-240831', allocationId: 'FEE-ALLOC-240831-004', portfolio: 'PF-DEMO-01', manager: 'FM-DEMO', portfolioManagerAssignmentId: '不适用', sleeveId: '全组合', eventScope: '跨主体事项', eventType: '管理费计提', instrument: 'MGMT_FEE', amount: '86,420.00', portfolioTreatment: '表内确认', managerTreatment: '表内确认', performanceFlow: '非现金事项', status: '双主体凭证待复核' },
  { id: 'BK-240831-005', tradeDate: '2026-08-31', settlementDate: '2026-09-01', sourceAccountId: 'EXEC-FM-DEMO-SSE', orderId: 'ORD-240831-005', executionId: 'FILL-240831-005', allocationId: 'ALLOC-240831-005-A', portfolio: 'PF-DEMO-01', manager: 'FM-DEMO', portfolioManagerAssignmentId: 'PMA-PF01-002', sleeveId: 'SLV-PF01-TACTICAL', eventScope: '组合事项', eventType: '股指期货开仓', instrument: 'IF2609.CFE', amount: '1,280,000.00', portfolioTreatment: '表内确认＋表外披露', managerTreatment: '不入该账', performanceFlow: '内部投资交易', status: '待确认计量规则' },
]

const vouchers: JournalVoucher[] = [
  {
    id: 'JV-FUND-0001', sourceId: 'BK-240831-001', entity: '组合/基金账', entityId: 'AE-PF-DEMO-01', accountingDate: '2026-08-31', eventPhase: 'ETF 买入｜交易日确认', policyVersion: 'FUND-GAAP-DEMO-v1', status: '借贷平衡 · 待复核',
    lines: [
      { accountCode: '1105', accountName: '交易性基金投资—成本', accountClass: '资产类', debit: 490000, credit: 0 },
      { accountCode: '6111', accountName: '投资收益—交易费用', accountClass: '损益类（费用方向）', debit: 8, credit: 0 },
      { accountCode: '3003', accountName: '证券清算款', accountClass: '共同类', debit: 0, credit: 490008 },
    ],
  },
  {
    id: 'JV-FUND-0002', sourceId: 'BK-240831-001', entity: '组合/基金账', entityId: 'AE-PF-DEMO-01', accountingDate: '2026-09-01', eventPhase: 'ETF 买入｜交收日付款', policyVersion: 'FUND-GAAP-DEMO-v1', status: '借贷平衡 · 待复核',
    lines: [
      { accountCode: '3003', accountName: '证券清算款', accountClass: '共同类', debit: 490008, credit: 0 },
      { accountCode: '1002', accountName: '银行存款', accountClass: '资产类', debit: 0, credit: 490008 },
    ],
  },
  {
    id: 'JV-FUND-0003', sourceId: 'BK-240831-004', entity: '组合/基金账', entityId: 'AE-PF-DEMO-01', accountingDate: '2026-08-31', eventPhase: '管理费｜权责发生制计提', policyVersion: 'FUND-GAAP-DEMO-v1', status: '借贷平衡 · 待复核',
    lines: [
      { accountCode: '6403', accountName: '管理人报酬', accountClass: '损益类（费用）', debit: 86420, credit: 0 },
      { accountCode: '2206', accountName: '应付管理人报酬', accountClass: '负债类', debit: 0, credit: 86420 },
    ],
  },
  {
    id: 'JV-MGR-0001', sourceId: 'BK-240831-004', entity: '管理人公司账', entityId: 'FM-DEMO', accountingDate: '2026-08-31', eventPhase: '管理费｜公司收入计提', policyVersion: 'CORP-GAAP-DEMO-v1', status: '借贷平衡 · 待复核',
    lines: [
      { accountCode: '1122', accountName: '应收管理费（示例映射）', accountClass: '资产类', debit: 86420, credit: 0 },
      { accountCode: '6001', accountName: '管理费收入（示例映射）', accountClass: '收入类', debit: 0, credit: 86420 },
    ],
  },
]

const sourceColumns = [
  'source_record_id', 'source_account_id', 'actual_portfolio_id', 'portfolio_manager_assignment_id', 'sleeve_id', 'manager_entity_id',
  'order_id', 'execution_id', 'allocation_id', 'settlement_account_id', 'event_scope', 'event_type', 'instrument_code',
  'trade_date', 'accounting_date', 'confirmation_date', 'settlement_date', 'value_date', 'currency', 'quantity', 'price',
  'gross_amount', 'transaction_cost', 'portfolio_book_treatment', 'manager_book_treatment',
  'performance_cash_flow_type', 'counterparty_entity_id',
]

const journalColumns = [
  'voucher_id', 'accounting_entity_id', 'accounting_date', 'policy_version', 'account_code',
  'line_description', 'debit_amount', 'credit_amount', 'posting_status', 'reversal_voucher_id',
]

const tabs = ['业务录入', '会计规则', '复式凭证', '试算平衡', '双账勾稽', '校验异常']
const eventScopeOptions: EventScope[] = ['组合事项', '跨主体事项', '管理人单独事项']

const accountingRules = [
  ['Booking 与凭证分层', 'Booking 保存原始业务事实；会计规则按核算主体、确认时点和计量口径，把一个事件拆成一张或多张复式凭证。'],
  ['每个主体独立平衡', '每只组合/基金单独建账，管理人公司另设账套。每张凭证在本主体内借方合计必须等于贷方合计，禁止跨账套配平。'],
  ['交易日与交收日分开', '证券成交在交易日确认投资和清算款，实际收付款在交收日冲销清算款；不能把成交确认和现金交割混成一笔。'],
  ['权责发生制计提', '管理费、托管费、利息等按会计政策逐日或按期计提，不以是否已收到或支付现金作为唯一确认条件。'],
  ['确认分类由规则判断', '用户登记业务事实，不得任意把应确认事项改成表外。规则输出表内确认、表外披露/备查或仅运营跟踪；衍生工具公允价值和保证金仍需按政策表内确认。'],
  ['过账不可直接删除', '已过账凭证进入总账和明细账；发现错误应红字冲销或反向凭证后重新入账，并保留来源、审批和版本链。'],
]

const normalBalanceRules = [
  ['资产', '借方增加，贷方减少'],
  ['负债', '贷方增加，借方减少'],
  ['基金净资产', '贷方增加，借方减少'],
  ['公司所有者权益', '贷方增加，借方减少'],
  ['收入 / 利得', '通常记贷方'],
  ['费用 / 损失', '通常记借方'],
]

const currency = (value: number) => `¥${value.toLocaleString('zh-CN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`

const classifyEvent = (eventType: string): { eventScope: EventScope; portfolioTreatment: BookTreatment; managerTreatment: BookTreatment; performanceFlow: PerformanceFlow } => {
  if (eventType === '管理费计提') {
    return { eventScope: '跨主体事项', portfolioTreatment: '表内确认', managerTreatment: '表内确认', performanceFlow: '非现金事项' }
  }
  if (eventType === '管理人自有资金跟投') {
    return { eventScope: '跨主体事项', portfolioTreatment: '表内确认', managerTreatment: '表内确认', performanceFlow: '外部资金流' }
  }
  if (eventType === '管理人公司费用') {
    return { eventScope: '管理人单独事项', portfolioTreatment: '不入该账', managerTreatment: '表内确认', performanceFlow: '非现金事项' }
  }
  if (eventType === '衍生工具交易') {
    return { eventScope: '组合事项', portfolioTreatment: '表内确认＋表外披露', managerTreatment: '不入该账', performanceFlow: '内部投资交易' }
  }
  if (eventType === '客户资金投入' || eventType === '客户资金转出') {
    return { eventScope: '组合事项', portfolioTreatment: '表内确认', managerTreatment: '不入该账', performanceFlow: '外部资金流' }
  }
  return { eventScope: '组合事项', portfolioTreatment: '表内确认', managerTreatment: '不入该账', performanceFlow: '内部投资交易' }
}

export default function AccountingBookingWorkspace() {
  const { portfolios, selectedPortfolio, selectedPortfolioId, selectPortfolio } = useActualPortfolio()
  const [activeTab, setActiveTab] = useState(tabs[0])
  const [selectedFile, setSelectedFile] = useState('')
  const [records, setRecords] = useState(initialRecords)
  const [notice, setNotice] = useState('')
  const [checkDraft, setCheckDraft] = useState({ debit: '100000', credit: '99800' })
  const [ledgerSummary, setLedgerSummary] = useState<LedgerSummaryResponse | null>(null)
  const [ledgerError, setLedgerError] = useState('')
  const [draft, setDraft] = useState({
    tradeDate: '2026-09-01', accountingDate: '2026-09-01', settlementDate: '2026-09-01', sourceAccountId: 'EXEC-FM-DEMO-SSE', orderId: 'ORD-DEMO-NEW', executionId: 'FILL-DEMO-NEW', allocationId: 'ALLOC-DEMO-NEW', portfolioManagerAssignmentId: 'PMA-PF01-001', sleeveId: 'SLV-PF01-CORE', eventType: 'ETF 买入', instrument: '', amount: '', ...classifyEvent('ETF 买入'),
  })
  const availableAccountRelationships = getAccountRelationshipsForPortfolio(selectedPortfolioId)
  const availableAssignments = getPortfolioManagerAssignments(selectedPortfolioId)
  const selectedAssignment = availableAssignments.find((assignment) => assignment.assignmentId === draft.portfolioManagerAssignmentId) ?? availableAssignments[0]
  const settlementAccount = getSettlementAccountForPortfolio(selectedPortfolioId)

  const visibleRecords = useMemo(
    () => records.filter((record) => record.portfolio === selectedPortfolioId || (record.eventScope === '管理人单独事项' && record.manager === selectedPortfolio.managerId)),
    [records, selectedPortfolio.managerId, selectedPortfolioId],
  )
  const visibleVouchers = useMemo(
    () => vouchers.filter((voucher) => voucher.entity === '组合/基金账' ? voucher.entityId === selectedPortfolio.accountingEntityId : voucher.entityId === selectedPortfolio.managerId),
    [selectedPortfolio.accountingEntityId, selectedPortfolio.managerId],
  )

  useEffect(() => {
    const controller = new AbortController()
    setLedgerSummary(null)
    setLedgerError('')
    evaluateLedgerSummary({
      entities: ['组合/基金账', '管理人公司账'],
      vouchers: visibleVouchers.map((voucher) => ({
        id: voucher.id,
        entity: voucher.entity,
        lines: voucher.lines.map((line) => ({ debit: line.debit, credit: line.credit })),
      })),
      pending_flags: visibleRecords.map((record) => record.portfolioTreatment === '待规则判断' || record.managerTreatment === '待规则判断' || record.status.includes('待确认')),
      trial_debit: Number(checkDraft.debit) || 0,
      trial_credit: Number(checkDraft.credit) || 0,
      tolerance: 0.005,
    }, controller.signal)
      .then(setLedgerSummary)
      .catch((reason) => {
        if ((reason as DOMException)?.name !== 'AbortError') setLedgerError('NJIT 借贷校验暂不可用')
      })
    return () => controller.abort()
  }, [checkDraft.credit, checkDraft.debit, visibleRecords, visibleVouchers])

  const voucherTotalsById = useMemo(
    () => new Map((ledgerSummary?.vouchers ?? []).map((item) => [item.id, item])),
    [ledgerSummary],
  )
  const metrics = ledgerSummary?.metrics ?? null
  const entityTotals = ledgerSummary?.entities ?? []

  const updateDraft = (key: keyof typeof draft, value: string) => setDraft((current) => ({ ...current, [key]: value }))
  const selectEventType = (eventType: string) => setDraft((current) => ({ ...current, eventType, ...classifyEvent(eventType) }))
  const selectBookingPortfolio = (portfolioId: string) => {
    selectPortfolio(portfolioId)
    const relationship = getAccountRelationshipsForPortfolio(portfolioId)[0]
    const assignment = getPortfolioManagerAssignments(portfolioId)[0]
    setDraft((current) => ({ ...current, sourceAccountId: relationship?.accountId ?? '', portfolioManagerAssignmentId: assignment?.assignmentId ?? '不适用', sleeveId: assignment?.sleeveId ?? '全组合' }))
  }
  const addBooking = () => {
    if (!draft.instrument || !draft.amount) return
    setRecords((current) => [...current, {
      id: `BK-DEMO-${String(current.length + 1).padStart(3, '0')}`,
      ...draft,
      portfolio: draft.eventScope === '管理人单独事项' ? '不适用' : selectedPortfolio.portfolioId,
      manager: selectedPortfolio.managerId,
      status: '仅登记业务事实 · 未生成凭证',
    }])
    setDraft((current) => ({ ...current, instrument: '', amount: '' }))
    setNotice('已加入当前会话的 Booking 业务事实；未生成凭证、未复核、未过账。')
  }
  const removeBooking = (id: string) => {
    setRecords((current) => current.filter((record) => record.id !== id))
    setNotice('已从当前会话移除未过账的演示业务事实；未修改任何真实账簿。')
  }
  return (
    <div className="space-y-5" data-testid="accounting-booking-workspace">
      <StaticDemoBanner />
      <header className="rounded-2xl bg-gradient-to-r from-cyan-950 via-cyan-900 to-teal-800 p-6 text-white shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-100/80">Source event → journal voucher → ledger → statements</p>
        <h2 className="mt-2 text-2xl font-bold">组合 Booking 与复式记账</h2>
        <p className="mt-2 max-w-5xl text-sm leading-6 text-cyan-50/85">本页承接已经完成来源账户、真实组合和基金经理拆分的业务事实。Booking 不等于会计凭证，系统按基金与管理人公司两个独立核算主体分别生成借贷分录。</p>
      </header>

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['业务事实', metrics ? `${metrics.source_event_count} 笔` : '—', '可拆分为多主体、多时点凭证'],
          ['待复核凭证', metrics ? `${metrics.voucher_count} 张` : '—', '示例凭证尚未正式过账'],
          ['借贷平衡', metrics ? `${metrics.balanced_count} / ${metrics.voucher_count}` : '—', '逐张、逐主体独立校验'],
          ['待分类 / 待确认', metrics ? `${metrics.pending_count} 笔` : ledgerError || '计算中…', '硬阻断凭证生成或过账'],
        ].map(([label, value, hint]) => <article key={label} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-sm text-slate-500">{label}</p><p className="mt-2 text-2xl font-bold text-slate-950">{value}</p><p className="mt-1 text-xs text-slate-400">{hint}</p></article>)}
      </section>

      <nav className="flex gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1" aria-label="Booking 工作台页面标签">
        {tabs.map((tab) => <button type="button" key={tab} onClick={() => setActiveTab(tab)} className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-semibold ${activeTab === tab ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>{tab}</button>)}
      </nav>

      {activeTab === '业务录入' ? (
        <div className="grid gap-5 xl:grid-cols-2">
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="font-semibold text-slate-900">原始业务 Booking</h3>
            <p className="mt-1 text-sm text-slate-500">这里只登记可追溯的经济业务事实；会计科目、借贷方向和金额由已生效规则生成并由会计复核。</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <label className="text-sm text-slate-600">交易日期<input aria-label="交易日期" type="date" value={draft.tradeDate} onChange={(event) => updateDraft('tradeDate', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">会计日期<input aria-label="会计日期" type="date" value={draft.accountingDate} onChange={(event) => updateDraft('accountingDate', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">交收日期<input aria-label="交收日期" type="date" value={draft.settlementDate} onChange={(event) => updateDraft('settlementDate', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">真实组合（来自组合中心）<select aria-label="Booking 真实组合" value={selectedPortfolioId} onChange={(event) => selectBookingPortfolio(event.target.value)} disabled={draft.eventScope === '管理人单独事项'} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 disabled:bg-slate-100">{portfolios.map((portfolio) => <option key={portfolio.portfolioId} value={portfolio.portfolioId}>{portfolio.name} · {portfolio.portfolioId}</option>)}</select></label>
              <label className="text-sm text-slate-600">来源账户<select aria-label="Booking 来源账户" value={draft.sourceAccountId} onChange={(event) => updateDraft('sourceAccountId', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{availableAccountRelationships.map((relationship) => { const account = getExternalAccount(relationship.accountId); return account ? <option key={relationship.relationshipId} value={account.accountId}>{account.kind} · {account.accountId}</option> : null })}</select></label>
              <label className="text-sm text-slate-600">基金经理任职关系<select aria-label="Booking 基金经理任职关系" value={selectedAssignment?.assignmentId ?? '不适用'} onChange={(event) => { const assignment = availableAssignments.find((item) => item.assignmentId === event.target.value); setDraft((current) => ({ ...current, portfolioManagerAssignmentId: assignment?.assignmentId ?? '不适用', sleeveId: assignment?.sleeveId ?? '全组合' })) }} disabled={draft.eventScope === '管理人单独事项'} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 disabled:bg-slate-100">{availableAssignments.map((assignment) => <option key={assignment.assignmentId} value={assignment.assignmentId}>{assignment.portfolioManagerName} · {assignment.role}</option>)}</select></label>
              <label className="text-sm text-slate-600">内部投资单元<input aria-label="Booking 内部投资单元" value={selectedAssignment?.sleeveId ?? '全组合'} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 text-slate-600" /></label>
              <label className="text-sm text-slate-600">基金专用资金账户<input aria-label="Booking 资金结算账户" value={settlementAccount?.accountId ?? '缺失'} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 font-mono text-slate-600" /></label>
              <label className="text-sm text-slate-600">管理人主体与主账簿<input aria-label="管理人主体与主账簿" value={`${selectedPortfolio.managerName} · ${selectedPortfolio.primaryLedgerId}`} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 text-slate-600" /></label>
              <label className="text-sm text-slate-600">订单编号<input aria-label="Booking 订单编号" value={draft.orderId} onChange={(event) => updateDraft('orderId', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">成交/外部确认编号<input aria-label="Booking 成交编号" value={draft.executionId} onChange={(event) => updateDraft('executionId', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600 sm:col-span-2">分配编号<input aria-label="Booking 分配编号" value={draft.allocationId} onChange={(event) => updateDraft('allocationId', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">事项范围<select aria-label="事项范围" value={draft.eventScope} onChange={(event) => updateDraft('eventScope', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2">{eventScopeOptions.map((option) => <option key={option}>{option}</option>)}</select></label>
              <label className="text-sm text-slate-600">业务类型<select aria-label="业务类型" value={draft.eventType} onChange={(event) => selectEventType(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2"><option>ETF 买入</option><option>ETF 卖出</option><option>底层基金申购</option><option>底层基金赎回</option><option>客户资金投入</option><option>客户资金转出</option><option>管理费计提</option><option>管理人自有资金跟投</option><option>衍生工具交易</option><option>管理人公司费用</option></select></label>
              <label className="text-sm text-slate-600">产品或事项代码<input aria-label="产品或事项代码" value={draft.instrument} onChange={(event) => updateDraft('instrument', event.target.value)} placeholder="例如 510300.SH" className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">本币金额<input aria-label="本币金额" inputMode="decimal" value={draft.amount} onChange={(event) => updateDraft('amount', event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">组合/基金账规则预判<input aria-label="组合或基金账规则预判" value={draft.portfolioTreatment} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 text-slate-600" /></label>
              <label className="text-sm text-slate-600">管理人账规则预判<input aria-label="管理人账规则预判" value={draft.managerTreatment} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 text-slate-600" /></label>
              <label className="text-sm text-slate-600 sm:col-span-2">组合绩效现金流规则预判<input aria-label="组合绩效现金流规则预判" value={draft.performanceFlow} readOnly className="mt-1 w-full rounded-lg border border-slate-200 bg-slate-100 px-3 py-2 text-slate-600" /></label>
            </div>
            <p className="mt-3 text-xs leading-5 text-slate-500">只有来源账户、订单/成交、分配编号、真实组合和核算主体关系完整的记录才能制证；分类仍是规则预判而非最终会计结论。</p>
            <button type="button" onClick={addBooking} disabled={!draft.instrument || !draft.amount} className="mt-4 min-h-11 rounded-lg bg-cyan-800 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">登记业务事实（演示）</button>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="font-semibold text-slate-900">Excel / CSV 导入</h3>
            <p className="mt-1 text-sm text-slate-500">本期只展示字段契约与文件选择反馈，不读取、不上传文件，也不模拟凭证过账。</p>
            <label className="mt-4 block rounded-xl border-2 border-dashed border-cyan-200 bg-cyan-50 p-6 text-center text-sm text-cyan-900">
              选择 Booking 文件
              <input aria-label="选择 Booking 数据文件" type="file" accept=".csv,.xls,.xlsx" className="mt-3 block w-full text-xs" onChange={(event) => setSelectedFile(event.target.files?.[0]?.name ?? '')} />
            </label>
            <p className="mt-3 min-h-6 text-sm font-medium text-cyan-800" aria-live="polite">{selectedFile ? `已选择：${selectedFile}（未解析、未上传）` : '尚未选择文件'}</p>
            <h4 className="mt-5 text-sm font-semibold text-slate-800">业务事实字段</h4>
            <div className="mt-2 flex flex-wrap gap-2">{sourceColumns.map((column) => <code key={column} className="rounded bg-slate-100 px-2 py-1 text-xs text-slate-700">{column}</code>)}</div>
            <h4 className="mt-5 text-sm font-semibold text-slate-800">存量凭证明细字段</h4>
            <div className="mt-2 flex flex-wrap gap-2">{journalColumns.map((column) => <code key={column} className="rounded bg-amber-50 px-2 py-1 text-xs text-amber-900">{column}</code>)}</div>
          </section>
        </div>
      ) : null}

      {activeTab === '会计规则' ? (
        <section className="border-y border-slate-200 bg-slate-50/60 px-4 py-5 sm:px-5" aria-label="会计规则说明（非交互）">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-slate-400">非交互说明</p>
          <h3 className="mt-1 text-lg font-semibold text-slate-800">会计规则与口径</h3>
          <p className="mt-1 text-sm text-slate-500">以下内容用于解释当前静态演示采用的记账边界，不是可点击的规则配置项。</p>
          <dl className="mt-5 divide-y divide-slate-200 border-y border-slate-200">
            {accountingRules.map(([title, description]) => <div key={title} className="grid gap-1 py-3 sm:grid-cols-[180px_1fr] sm:gap-5"><dt className="text-sm font-semibold text-slate-700">{title}</dt><dd className="text-sm leading-6 text-slate-600">{description}</dd></div>)}
          </dl>
          <div className="mt-6">
            <h4 className="font-semibold text-slate-800">账户正常余额与增减方向</h4>
            <p className="mt-1 text-sm text-slate-500">借、贷是记账方向，不直接等同于增加或减少；必须结合账户性质判断。</p>
            <dl className="mt-3 divide-y divide-slate-200 text-sm">{normalBalanceRules.map(([accountClass, rule]) => <div key={accountClass} className="grid grid-cols-[140px_1fr] gap-3 py-2.5"><dt className="font-medium text-slate-700">{accountClass}</dt><dd className="text-slate-600">{rule}</dd></div>)}</dl>
          </div>
          <div className="mt-6 border-t border-slate-200 pt-4 text-sm">
            <p className="grid gap-1 py-1 sm:grid-cols-[180px_1fr]"><span className="font-semibold text-slate-700">组合/基金账</span><span className="font-mono font-semibold text-slate-800">资产 = 负债 + 净资产</span></p>
            <p className="grid gap-1 py-1 sm:grid-cols-[180px_1fr]"><span className="font-semibold text-slate-700">管理人公司账</span><span className="font-mono font-semibold text-slate-800">资产 = 负债 + 所有者权益</span></p>
          </div>
        </section>
      ) : null}

      {activeTab === '复式凭证' ? (
        <section className="space-y-4">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">凭证与分录预览</h3><p className="mt-1 text-sm leading-6 text-slate-500">以下为规则生成的静态凭证示例。每张凭证至少两行、单行不能同时有借贷金额、逐张借贷相等；当前均未复核、未过账。</p></div>
          {visibleVouchers.map((voucher) => {
            const totals = voucherTotalsById.get(voucher.id)
            return (
              <article key={voucher.id} className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
                <div className="flex flex-col gap-3 border-b border-slate-100 px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
                  <div><div className="flex flex-wrap items-center gap-2"><h4 className="font-bold text-slate-950">{voucher.id}</h4><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${voucher.entity === '组合/基金账' ? 'bg-cyan-100 text-cyan-900' : 'bg-violet-100 text-violet-900'}`}>{voucher.entity}</span><span className="rounded-full bg-amber-100 px-2.5 py-1 text-xs font-semibold text-amber-900">{voucher.status}</span></div><p className="mt-2 text-sm text-slate-500">来源 {voucher.sourceId} · {voucher.entityId} · {voucher.eventPhase}</p></div>
                  <div className="text-sm text-slate-500"><p>会计日期：{voucher.accountingDate}</p><p>政策版本：{voucher.policyVersion}</p></div>
                </div>
                <div className="overflow-x-auto"><table className="w-full min-w-[760px] text-sm"><thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500"><tr><th className="px-4 py-3">科目编码</th><th className="px-4 py-3">会计科目</th><th className="px-4 py-3">账户类别</th><th className="px-4 py-3 text-right">借方</th><th className="px-4 py-3 text-right">贷方</th></tr></thead><tbody>{voucher.lines.map((line, index) => <tr key={`${voucher.id}-${index}`} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-slate-500">{line.accountCode}</td><td className="px-4 py-3 font-medium text-slate-900">{line.accountName}</td><td className="px-4 py-3 text-slate-500">{line.accountClass}</td><td className="px-4 py-3 text-right tabular-nums">{line.debit ? currency(line.debit) : '—'}</td><td className="px-4 py-3 text-right tabular-nums">{line.credit ? currency(line.credit) : '—'}</td></tr>)}</tbody><tfoot className="border-t-2 border-slate-200 bg-slate-50 font-semibold text-slate-900"><tr><td className="px-4 py-3" colSpan={3}>凭证合计</td><td className="px-4 py-3 text-right tabular-nums">借方合计 {totals ? currency(totals.debit) : '—'}</td><td className="px-4 py-3 text-right tabular-nums">贷方合计 {totals ? currency(totals.credit) : '—'}</td></tr></tfoot></table></div>
                <div className={`px-5 py-3 text-sm font-semibold ${totals?.balanced ? 'bg-emerald-50 text-emerald-800' : totals ? 'bg-rose-50 text-rose-800' : 'bg-amber-50 text-amber-800'}`}>{totals ? <>借贷差额：{currency(totals.difference)} · {totals.balanced ? '凭证级校验通过，可进入会计复核' : '硬阻断，不得复核或过账'}</> : ledgerError || 'NJIT 借贷校验中…'}</div>
              </article>
            )
          })}
          {visibleVouchers.length === 0 ? <p className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">当前真实组合没有凭证示例。</p> : null}
        </section>
      ) : null}

      {activeTab === '试算平衡' ? (
        <section className="space-y-5">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">逐主体试算与资产负债等式</h3><p className="mt-1 text-sm leading-6 text-slate-500">借贷平衡是凭证和总账的机械校验，资产负债等式是期末余额校验；两者都通过仍不代表确认、计量和估值一定正确。</p></div>
          <div className="grid gap-4 lg:grid-cols-2">{entityTotals.map((item) => <article key={item.entity} className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-sm font-semibold text-slate-500">{item.entity}｜本期凭证发生额</p><div className="mt-4 grid grid-cols-2 gap-3"><div className="rounded-xl bg-slate-50 p-4"><p className="text-xs text-slate-500">借方发生额</p><p className="mt-1 text-lg font-bold tabular-nums text-slate-950">{currency(item.debit)}</p></div><div className="rounded-xl bg-slate-50 p-4"><p className="text-xs text-slate-500">贷方发生额</p><p className="mt-1 text-lg font-bold tabular-nums text-slate-950">{currency(item.credit)}</p></div></div><p className="mt-3 text-sm font-semibold text-emerald-700">试算差额 {currency(item.difference)} · {item.voucher_count} 张凭证</p></article>)}</div>
          <div className="grid gap-4 lg:grid-cols-2">
            <article className="rounded-2xl border border-cyan-200 bg-cyan-50 p-5"><p className="text-sm font-semibold text-cyan-900">组合/基金账期末余额（静态截面，单位：万元）</p><p className="mt-4 text-xl font-bold text-cyan-950">12,648 = 118 + 12,530</p><p className="mt-2 text-sm text-cyan-900/80">资产 = 负债 + 净资产 · 等式差额 0.00</p></article>
            <article className="rounded-2xl border border-violet-200 bg-violet-50 p-5"><p className="text-sm font-semibold text-violet-900">管理人公司账期末余额（静态截面，单位：万元）</p><p className="mt-4 text-xl font-bold text-violet-950">13,480 = 2,310 + 11,170</p><p className="mt-2 text-sm text-violet-900/80">资产 = 负债 + 所有者权益 · 等式差额 0.00</p></article>
          </div>
        </section>
      ) : null}

      {activeTab === '双账勾稽' ? (
        <section className="space-y-5">
          <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">跨主体业务勾稽</h3><p className="mt-1 text-sm leading-6 text-slate-500">双账勾稽只核对同一经济业务的来源、期间、对手方和金额；基金账的一条借方不能抵销管理人账的一条贷方。</p></div>
          <div className="grid gap-4 lg:grid-cols-[1fr_auto_1fr] lg:items-stretch">
            <article className="rounded-2xl border border-cyan-200 bg-cyan-50 p-5"><p className="text-xs font-semibold uppercase tracking-wide text-cyan-700">{selectedPortfolio.portfolioId} · JV-FUND-0003</p><h4 className="mt-2 font-bold text-cyan-950">基金承担管理费</h4><p className="mt-4 text-sm text-cyan-950">借：6403 管理人报酬　{currency(86420)}</p><p className="mt-2 text-sm text-cyan-950">贷：2206 应付管理人报酬　{currency(86420)}</p><p className="mt-4 text-xs font-semibold text-emerald-700">本主体借贷平衡</p></article>
            <div className="flex items-center justify-center text-2xl font-bold text-slate-300">↔</div>
            <article className="rounded-2xl border border-violet-200 bg-violet-50 p-5"><p className="text-xs font-semibold uppercase tracking-wide text-violet-700">{selectedPortfolio.managerId} · JV-MGR-0001</p><h4 className="mt-2 font-bold text-violet-950">管理人确认管理费收入</h4><p className="mt-4 text-sm text-violet-950">借：1122 应收管理费　{currency(86420)}</p><p className="mt-2 text-sm text-violet-950">贷：6001 管理费收入　{currency(86420)}</p><p className="mt-4 text-xs font-semibold text-emerald-700">本主体借贷平衡</p></article>
          </div>
          <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">来源 BK-240831-004、对手主体、会计期间与金额一致；两张凭证分别复核和过账。</div>
        </section>
      ) : null}

      {activeTab === '校验异常' ? (
        <section className="space-y-5">
          <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
            <h3 className="font-semibold text-slate-900">凭证平衡检查（静态交互演示）</h3>
            <p className="mt-1 text-sm text-slate-500">修改借贷合计可验证硬阻断逻辑；本演示不生成、不复核、不提交凭证。</p>
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              <label className="text-sm text-slate-600">借方合计<input aria-label="校验借方合计" inputMode="decimal" value={checkDraft.debit} onChange={(event) => setCheckDraft((current) => ({ ...current, debit: event.target.value }))} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
              <label className="text-sm text-slate-600">贷方合计<input aria-label="校验贷方合计" inputMode="decimal" value={checkDraft.credit} onChange={(event) => setCheckDraft((current) => ({ ...current, credit: event.target.value }))} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
            </div>
            <p className={`mt-4 rounded-xl px-4 py-3 text-sm font-semibold ${ledgerSummary?.trial.balanced ? 'bg-emerald-50 text-emerald-800' : ledgerSummary ? 'bg-rose-50 text-rose-800' : 'bg-amber-50 text-amber-800'}`} aria-live="polite">{ledgerSummary ? <>差额 {currency(ledgerSummary.trial.difference)} · {ledgerSummary.trial.balanced ? '凭证平衡，可进入会计复核' : '凭证不平，硬阻断复核与过账'}</> : ledgerError || 'NJIT 借贷校验中…'}</p>
          </article>
          <aside className="border-l-2 border-amber-300 bg-amber-50/50 px-4 py-4" role="note" aria-label="阻断规则说明（非交互）">
            <p className="text-xs font-semibold uppercase tracking-[0.16em] text-amber-700/70">非交互说明</p>
            <h3 className="mt-1 font-semibold text-slate-800">阻断规则说明</h3>
            <dl className="mt-3 divide-y divide-amber-200/70">{[
            ['凭证硬阻断', '核算主体或政策版本缺失、账期关闭、科目停用、单行同时记借贷、金额非正、凭证借贷不平时不得过账。'],
            ['业务与结算核对', '来源号重复、交易日与交收日事件缺失、证券/资金/份额与托管或银行不一致时进入悬账和差错队列。'],
            ['表外与主体边界', '表外备查不进入总账试算；不得用表外规避表内确认，也不得用组合账和管理人账跨主体抵销。'],
            ['估值与净资产', '价格、汇率、利息费用应计、份额登记或估值政策缺失时，不得冻结净资产和生成正式报表。'],
            ['冲销与审计轨迹', '已过账凭证不得物理删除；调整必须关联原凭证、原因、操作者、复核者、时间和新版本。'],
            ['绩效数据隔离', 'TWR 使用估值与外部现金流，XIRR 使用带日期外部现金流和期末价值；内部投资交易不得误标为外部现金流。'],
          ].map(([title, body]) => <div key={title} className="grid gap-1 py-3 sm:grid-cols-[160px_1fr] sm:gap-5"><dt className="text-sm font-semibold text-slate-700">{title}</dt><dd className="text-sm leading-6 text-slate-600">{body}</dd></div>)}</dl>
          </aside>
        </section>
      ) : null}

      <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="border-b border-slate-100 px-5 py-4"><h3 className="font-semibold text-slate-900">组合 Booking 业务事实（示例）</h3><p className="mt-1 text-sm text-slate-500">以下记录已经完成来源账户和组合分配；它们用于制证、重建持仓与现金，但本身不是总账。</p></div>
        <div className="overflow-x-auto"><table className="w-full min-w-[1840px] text-sm"><thead className="bg-slate-50 text-left text-xs uppercase tracking-wide text-slate-500"><tr>{['来源号', '来源账户/分配', '交易/会计/交收日', '真实组合/管理人主体', '基金经理任职/投资单元', '事项范围', '业务类型', '产品或事项', '金额', '组合/基金账', '管理人账', '绩效性质', '会计状态', '操作'].map((heading) => <th key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{visibleRecords.map((record) => <tr key={record.id} className="border-t border-slate-100"><td className="px-4 py-3 font-medium text-slate-900">{record.id}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{record.sourceAccountId}<br /><span className="text-slate-400">{record.allocationId}</span></td><td className="px-4 py-3 text-slate-600"><span title="交易日期">{record.tradeDate}</span><br /><span className="text-xs text-slate-400" title="会计日期">{record.accountingDate ?? record.tradeDate}</span><br /><span className="text-xs text-slate-400" title="交收日期">{record.settlementDate}</span></td><td className="px-4 py-3 font-mono text-xs text-slate-600">{record.portfolio}<br /><span className="text-slate-400">{record.manager}</span></td><td className="px-4 py-3 font-mono text-xs text-slate-600">{record.portfolioManagerAssignmentId}<br /><span className="text-slate-400">{record.sleeveId}</span></td><td className="px-4 py-3">{record.eventScope}</td><td className="px-4 py-3">{record.eventType}</td><td className="px-4 py-3">{record.instrument}</td><td className="px-4 py-3 tabular-nums">{record.amount}</td><td className="px-4 py-3">{record.portfolioTreatment}</td><td className="px-4 py-3">{record.managerTreatment}</td><td className="px-4 py-3">{record.performanceFlow}</td><td className="px-4 py-3"><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700">{record.status}</span></td><td className="px-4 py-3"><button type="button" onClick={() => removeBooking(record.id)} className="min-h-11 rounded-lg px-3 text-sm font-semibold text-rose-700 hover:bg-rose-50">移除演示记录</button></td></tr>)}</tbody></table></div>
        {visibleRecords.length === 0 ? <p className="border-t border-slate-100 px-5 py-8 text-center text-sm text-slate-500">当前真实组合没有 Booking 示例记录。</p> : null}
      </section>

      {notice ? <p className="rounded-xl border border-cyan-200 bg-cyan-50 px-4 py-3 text-sm text-cyan-900" aria-live="polite">{notice}</p> : null}
    </div>
  )
}
