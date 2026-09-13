import { useMemo, useState } from 'react'
import { useActualPortfolio } from '../app/ActualPortfolioContext'
import StaticDemoBanner from '../components/StaticDemoBanner'
import {
  actualPortfolioDemoData,
  actualPortfolioStatuses,
  externalAccounts,
  getAccountRelationshipsForPortfolio,
  getExternalAccount,
  getPortfolioManagerAssignments,
  portfolioAccountRelationships,
  portfolioSleeves,
  type ActualPortfolio,
  type ActualPortfolioStatus,
} from '../app/actualPortfolioDemoData'

export type PortfolioCenterView = 'register' | 'master' | 'relationships' | 'responsibilities' | 'versions' | 'lifecycle'

const viewCopy: Record<PortfolioCenterView, { eyebrow: string; title: string; description: string }> = {
  register: { eyebrow: 'Actual portfolio register', title: '真实组合登记簿', description: '统一检索和选择已经在平台登记的真实基金或投资组合，不与研究组合、运行快照或展示方案混用。' },
  master: { eyebrow: 'Portfolio master data', title: '组合主数据', description: '查看真实组合的稳定身份、责任主体、币种、基准和外部业务标识。' },
  relationships: { eyebrow: 'Account master and relationships', title: '账户主档与组合关系', description: '把账户作为独立主数据，区分可共享执行通道与单一核算主体专用的资金、证券和 TA 账户。' },
  responsibilities: { eyebrow: 'Portfolio manager assignments', title: '基金经理与投资单元', description: '按生效期维护组合、内部投资单元与基金经理任职关系，保留历史决策责任边界。' },
  versions: { eyebrow: 'Research-to-live lineage', title: '研究方案与目标版本', description: '追溯已批准研究方案如何落地为真实组合，并保留后续目标组合版本的变更关系。' },
  lifecycle: { eyebrow: 'Portfolio lifecycle', title: '组合生命周期与状态', description: '记录筹备、启用、运行、暂停、清算及终止状态，不改写外部法律事实。' },
}

const statusStyle: Record<ActualPortfolioStatus, string> = {
  筹备中: 'bg-accent-100 text-accent-800',
  待启用: 'bg-amber-100 text-amber-900',
  运行中: 'bg-emerald-100 text-emerald-800',
  暂停: 'bg-orange-100 text-orange-900',
  清算中: 'bg-rose-100 text-rose-800',
  已终止: 'bg-slate-200 text-slate-700',
}

function PortfolioPicker({ value, onChange }: { value: string; onChange: (portfolioId: string) => void }) {
  return (
    <label className="text-sm font-medium text-slate-200">
      真实组合
      <select aria-label="选择真实组合" value={value} onChange={(event) => onChange(event.target.value)} className="mt-1 block w-full min-w-0 rounded-xl border border-slate-300 bg-white px-3 py-2 text-slate-900 sm:min-w-72">
        {actualPortfolioDemoData.map((portfolio) => <option key={portfolio.portfolioId} value={portfolio.portfolioId}>{portfolio.name} · {portfolio.portfolioId}</option>)}
      </select>
    </label>
  )
}

function DefinitionList({ rows }: { rows: Array<[string, string]> }) {
  return <dl className="divide-y divide-slate-100">{rows.map(([label, value]) => <div key={label} className="grid gap-1 py-3 sm:grid-cols-[180px_1fr]"><dt className="text-sm text-slate-600">{label}</dt><dd className="break-words text-sm font-medium text-slate-900">{value}</dd></div>)}</dl>
}

function RegisterView({ selectedId, onSelect }: { selectedId: string; onSelect: (portfolioId: string) => void }) {
  const [keyword, setKeyword] = useState('')
  const [status, setStatus] = useState<'全部状态' | ActualPortfolioStatus>('全部状态')
  const portfolios = useMemo(() => actualPortfolioDemoData.filter((portfolio) => (
    (status === '全部状态' || portfolio.status === status)
    && `${portfolio.portfolioId}${portfolio.name}${portfolio.externalProductCode}`.toLowerCase().includes(keyword.trim().toLowerCase())
  )), [keyword, status])

  return <div className="space-y-5">
    <section className="grid gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-[220px_minmax(0,1fr)]">
      <label className="text-sm text-slate-600">组合状态<select aria-label="筛选组合状态" value={status} onChange={(event) => setStatus(event.target.value as typeof status)} className="mt-1 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2"><option>全部状态</option>{actualPortfolioStatuses.map((item) => <option key={item}>{item}</option>)}</select></label>
      <label className="text-sm text-slate-600">搜索真实组合<input aria-label="搜索真实组合" value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="输入名称、portfolio_id 或外部代码" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
    </section>
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm" aria-label="真实组合列表">
      <div className="overflow-x-auto"><table className="w-full min-w-[920px] text-sm"><thead className="bg-slate-50 text-left text-xs font-semibold text-slate-600"><tr>{['portfolio_id', '组合名称', '类型', '状态', '管理人', '启用日', '主账套', '操作'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{portfolios.map((portfolio) => <tr key={portfolio.portfolioId} className={selectedId === portfolio.portfolioId ? 'border-t border-accent-200 bg-accent-50/60' : 'border-t border-slate-100'}><td className="px-4 py-3 font-mono text-xs font-semibold text-slate-800">{portfolio.portfolioId}</td><td className="px-4 py-3 font-semibold text-slate-900">{portfolio.name}</td><td className="px-4 py-3 text-slate-600">{portfolio.portfolioType}</td><td className="px-4 py-3"><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${statusStyle[portfolio.status]}`}>{portfolio.status}</span></td><td className="px-4 py-3 text-slate-600">{portfolio.managerName}</td><td className="px-4 py-3 text-slate-600">{portfolio.activationDate}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{portfolio.primaryLedgerId}</td><td className="px-4 py-3"><button type="button" onClick={() => onSelect(portfolio.portfolioId)} className="min-h-11 rounded-lg px-3 text-sm font-semibold text-accent-800 hover:bg-accent-100 focus:outline-none focus:ring-2 focus:ring-accent-500">设为当前组合</button></td></tr>)}</tbody></table></div>
      {portfolios.length === 0 ? <p className="p-8 text-center text-sm text-slate-600">没有符合条件的示例真实组合。</p> : null}
    </section>
  </div>
}

function MasterView({ portfolio }: { portfolio: ActualPortfolio }) {
  return <div className="grid gap-5 xl:grid-cols-2">
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">身份与状态</h3><DefinitionList rows={[
      ['不可变 portfolio_id', portfolio.portfolioId], ['组合名称', portfolio.name], ['组合类型', portfolio.portfolioType], ['生命周期状态', portfolio.status], ['外部产品或委托代码', portfolio.externalProductCode], ['成立日 / 启用日', `${portfolio.inceptionDate} / ${portfolio.activationDate}`],
    ]} /></section>
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">责任主体与投资口径</h3><DefinitionList rows={[
      ['管理人主体', `${portfolio.managerName} · ${portfolio.managerId}`], ['托管主体', `${portfolio.custodianName} · ${portfolio.custodianEntityId}`], ['记账本位币', portfolio.baseCurrency], ['业绩比较基准', portfolio.benchmark], ['来源批准记录', portfolio.approvalRecordId],
    ]} /></section>
  </div>
}

function RelationshipsView({ portfolio }: { portfolio: ActualPortfolio }) {
  const relationships = getAccountRelationshipsForPortfolio(portfolio.portfolioId)
  return <div className="space-y-5">
    <section className="grid gap-4 md:grid-cols-3">{[
      ['会计主体', portfolio.accountingEntityId], ['主核算账套', portfolio.primaryLedgerId], ['会计政策版本', portfolio.accountingPolicyVersion],
    ].map(([label, value]) => <article key={label} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-sm text-slate-600">{label}</p><p className="mt-2 break-all font-mono text-sm font-bold text-slate-950">{value}</p></article>)}</section>
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"><div className="border-b border-slate-100 px-5 py-4"><h3 className="font-semibold text-slate-900">账户关系清单</h3><p className="mt-1 text-sm text-slate-600">账户主档独立于组合；组合通过有生效期的关系使用账户，历史关系不随当前配置被覆盖。</p></div><div className="overflow-x-auto"><table className="w-full min-w-[1120px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['账户编号', '账户类型', '机构与法定归属', '共享规则', '组合用途', '虚拟子账号', '生效期', '状态'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{relationships.map((relationship) => {
      const account = getExternalAccount(relationship.accountId)
      if (!account) return null
      return <tr key={relationship.relationshipId} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-xs font-semibold text-slate-800">{account.accountId}</td><td className="px-4 py-3 font-medium text-slate-900">{account.kind}</td><td className="px-4 py-3 text-slate-600">{account.institution}<br /><span className="font-mono text-xs text-slate-600">归属 {account.legalOwnerEntityId}</span></td><td className="px-4 py-3"><span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${account.sharingPolicy === '允许多组合共享执行' ? 'bg-accent-100 text-accent-800' : 'bg-emerald-100 text-emerald-800'}`}>{account.sharingPolicy}</span></td><td className="px-4 py-3 text-slate-600">{relationship.purpose}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{relationship.virtualSubAccountId ?? '—'}</td><td className="px-4 py-3 text-slate-600">{relationship.effectiveFrom}<br /><span className="text-xs text-slate-600">至 {relationship.effectiveTo ?? '持续有效'}</span></td><td className="px-4 py-3">{relationship.status}</td></tr>
    })}</tbody></table></div></section>
    <section className="overflow-hidden rounded-xl border border-accent-200 bg-white shadow-sm"><div className="border-b border-accent-100 px-5 py-4"><h3 className="font-semibold text-slate-900">共享执行通道覆盖范围</h3><p className="mt-1 text-sm text-slate-600">这里只展示集中交易关系；资金结算账户和托管证券账户仍按基金核算主体隔离。</p></div><div className="overflow-x-auto"><table className="w-full min-w-[820px] text-sm"><thead className="bg-accent-50/60 text-left text-xs text-slate-600"><tr>{['执行通道', '机构', '已关联组合', '分配要求'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{externalAccounts.filter((account) => account.kind === '执行通道账户' && portfolioAccountRelationships.some((relationship) => relationship.accountId === account.accountId && relationship.portfolioId === portfolio.portfolioId)).map((account) => {
      const relatedPortfolioNames = portfolioAccountRelationships.filter((relationship) => relationship.accountId === account.accountId).map((relationship) => actualPortfolioDemoData.find((item) => item.portfolioId === relationship.portfolioId)?.name ?? relationship.portfolioId)
      return <tr key={account.accountId} className="border-t border-accent-100"><td className="px-4 py-3 font-mono text-xs font-semibold text-accent-900">{account.accountId}</td><td className="px-4 py-3 text-slate-600">{account.institution}</td><td className="px-4 py-3 text-slate-700">{relatedPortfolioNames.join('、')}</td><td className="px-4 py-3 text-slate-600">成交后必须按预分配规则拆到组合，不形成共享基金账</td></tr>
    })}</tbody></table></div></section>
  </div>
}

function ResponsibilitiesView({ portfolio }: { portfolio: ActualPortfolio }) {
  const assignments = getPortfolioManagerAssignments(portfolio.portfolioId)
  const sleeves = portfolioSleeves.filter((sleeve) => sleeve.portfolioId === portfolio.portfolioId)
  return <div className="space-y-5">
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"><div className="border-b border-slate-100 px-5 py-4"><h3 className="font-semibold text-slate-900">基金经理任职关系</h3><p className="mt-1 text-sm text-slate-600">基金经理是决策责任和分析维度，不是会计主体；任职变更按有效期新增记录。</p></div><div className="overflow-x-auto"><table className="w-full min-w-[980px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['任职编号', '基金经理', '角色', '内部投资单元', '管理人主体', '生效期', '状态'].map((heading) => <th scope="col" key={heading} className="px-4 py-3">{heading}</th>)}</tr></thead><tbody>{assignments.map((assignment) => <tr key={assignment.assignmentId} className="border-t border-slate-100"><td className="px-4 py-3 font-mono text-xs text-slate-600">{assignment.assignmentId}</td><td className="px-4 py-3 font-semibold text-slate-900">{assignment.portfolioManagerName}<br /><span className="font-mono text-xs font-normal text-slate-600">{assignment.portfolioManagerId}</span></td><td className="px-4 py-3">{assignment.role}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{assignment.sleeveId}</td><td className="px-4 py-3 font-mono text-xs text-slate-600">{assignment.managerEntityId}</td><td className="px-4 py-3 text-slate-600">{assignment.effectiveFrom}<br /><span className="text-xs text-slate-600">至 {assignment.effectiveTo ?? '持续有效'}</span></td><td className="px-4 py-3">{assignment.status}</td></tr>)}</tbody></table></div></section>
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"><div className="border-b border-slate-100 px-5 py-4"><h3 className="font-semibold text-slate-900">内部投资单元</h3><p className="mt-1 text-sm text-slate-600">Sleeve 用于指令、持仓和绩效拆分；除非依法成为独立核算主体，否则不单独生成法定财务报表。</p></div><div className="grid gap-3 p-5 md:grid-cols-2">{sleeves.map((sleeve) => <article key={sleeve.sleeveId} className="border-l-2 border-accent-300 bg-slate-50 px-4 py-3"><p className="font-mono text-xs text-slate-600">{sleeve.sleeveId}</p><h4 className="mt-1 font-semibold text-slate-900">{sleeve.name}</h4><p className="mt-2 text-sm text-slate-600">会计处理：{sleeve.accountingTreatment}</p></article>)}</div></section>
  </div>
}

function VersionsView({ portfolio }: { portfolio: ActualPortfolio }) {
  const lineage = [
    ['研究方案', `${portfolio.sourceResearchPlanId} · ${portfolio.sourceResearchPlanVersion}`],
    ['外部批准记录', portfolio.approvalRecordId],
    ['真实组合', `${portfolio.portfolioId} · ${portfolio.name}`],
    ['当前目标版本', portfolio.currentTargetVersion],
  ]
  return <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">版本追溯链</h3><p className="mt-1 text-sm leading-6 text-slate-600">一个研究方案可落地为多个真实组合；同一真实组合后续可依次采用多个目标版本，但 portfolio_id 不随目标调整而改变。</p><ol className="mt-5 grid gap-3 lg:grid-cols-4">{lineage.map(([label, value], index) => <li key={label} className="relative rounded-xl border border-slate-200 bg-slate-50 p-4"><p className="text-xs font-semibold text-accent-700">{String(index + 1).padStart(2, '0')} · {label}</p><p className="mt-2 break-words text-sm font-semibold text-slate-900">{value}</p>{index < lineage.length - 1 ? <span className="absolute -right-3 top-1/2 hidden -translate-y-1/2 text-slate-600 lg:block" aria-hidden="true">→</span> : null}</li>)}</ol></section>
}

function LifecycleView({ portfolio }: { portfolio: ActualPortfolio }) {
  return <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex flex-wrap items-center justify-between gap-3"><div><h3 className="font-semibold text-slate-900">生命周期事件</h3><p className="mt-1 text-sm text-slate-600">状态来自平台外有效文件或授权结果，平台只登记、校验和留痕。</p></div><span className={`rounded-full px-3 py-1 text-xs font-semibold ${statusStyle[portfolio.status]}`}>当前：{portfolio.status}</span></div><ol className="mt-5 border-l-2 border-slate-200 pl-5">{portfolio.lifecycle.map((item) => <li key={`${item.date}-${item.status}`} className="relative pb-6 last:pb-0"><span className="absolute -left-[27px] top-1.5 h-3 w-3 rounded-full border-2 border-white bg-accent-700 ring-2 ring-slate-200" aria-hidden="true" /><div className="flex flex-wrap items-center gap-2"><time className="text-xs font-semibold text-slate-600">{item.date}</time><span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${statusStyle[item.status]}`}>{item.status}</span></div><p className="mt-2 text-sm font-medium text-slate-900">{item.event}</p><p className="mt-1 font-mono text-xs text-slate-600">来源：{item.source}</p></li>)}</ol></section>
}

export default function PortfolioCenterWorkspace({ view }: { view: PortfolioCenterView }) {
  const { selectedPortfolioId: selectedId, selectPortfolio: setSelectedId } = useActualPortfolio()
  const portfolio = actualPortfolioDemoData.find((item) => item.portfolioId === selectedId) ?? actualPortfolioDemoData[0]
  const copy = viewCopy[view]

  return <div className="space-y-5" data-testid={`portfolio-center-${view}`}>
    <StaticDemoBanner />
    <header className="rounded-xl bg-gradient-to-r from-accent-950 via-slate-900 to-accent-900 p-6 text-white shadow-sm"><div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between"><div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-200">{copy.eyebrow}</p><h2 className="mt-2 text-2xl font-bold">{copy.title}</h2><p className="mt-2 max-w-4xl text-sm leading-6 text-slate-200">{copy.description}</p></div>{view !== 'register' ? <PortfolioPicker value={selectedId} onChange={setSelectedId} /> : null}</div></header>
    {view === 'register' ? <RegisterView selectedId={selectedId} onSelect={setSelectedId} /> : null}
    {view === 'master' ? <MasterView portfolio={portfolio} /> : null}
    {view === 'relationships' ? <RelationshipsView portfolio={portfolio} /> : null}
    {view === 'responsibilities' ? <ResponsibilitiesView portfolio={portfolio} /> : null}
    {view === 'versions' ? <VersionsView portfolio={portfolio} /> : null}
    {view === 'lifecycle' ? <LifecycleView portfolio={portfolio} /> : null}
    <aside className="border-l-2 border-slate-300 bg-slate-50/70 px-4 py-4" role="note" aria-label="真实组合边界说明（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">组合中心的职责边界</h3><p className="mt-2 text-sm leading-6 text-slate-600">组合中心保存真实组合身份及其账户、账套和版本关系；不保存研究组合运行结果，不替代组合方案展示，也不完成基金法律设立、账户开户或交易执行。</p></aside>
  </div>
}
