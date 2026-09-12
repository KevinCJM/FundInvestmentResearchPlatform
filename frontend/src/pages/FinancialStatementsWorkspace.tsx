import { useState } from 'react'
import { useActualPortfolio } from '../app/ActualPortfolioContext'
import StaticDemoBanner from '../components/StaticDemoBanner'

interface StatementRow {
  item: string
  current: string
  prior: string
  source: string
  strong?: boolean
}

type StatementScope = '组合/基金账' | '管理人公司账'

const portfolioStatements: Record<string, { title: string; subtitle: string; rows: StatementRow[] }> = {
  '资产负债表': {
    title: '资产负债表',
    subtitle: '时点报表｜资产 = 负债 + 净资产',
    rows: [
      { item: '银行存款与结算备付金', current: '1,218', prior: '1,046', source: '银行存款 / 结算备付金' },
      { item: '交易性金融资产', current: '11,268', prior: '10,814', source: '股票、债券、基金投资' },
      { item: '应收项目及其他资产', current: '162', prior: '134', source: '应收清算款 / 应收利息等' },
      { item: '资产合计', current: '12,648', prior: '11,994', source: '资产类科目汇总', strong: true },
      { item: '负债合计', current: '118', prior: '214', source: '应付清算款 / 费用应计等', strong: true },
      { item: '净资产', current: '12,530', prior: '11,780', source: '实收资金 / 未分配利润', strong: true },
    ],
  },
  '利润表': {
    title: '利润表',
    subtitle: '期间报表｜投资收益、公允价值变动、利息收入与费用',
    rows: [
      { item: '利息收入', current: '46', prior: '39', source: '利息收入明细账' },
      { item: '投资收益', current: '210', prior: '164', source: '处置价差 / 股息红利等' },
      { item: '公允价值变动损益', current: '68', prior: '-21', source: '估值调整分录' },
      { item: '管理、托管及其他费用', current: '-42', prior: '-37', source: '费用应计明细账' },
      { item: '本期利润', current: '282', prior: '145', source: '损益类科目结转', strong: true },
    ],
  },
  '净资产变动表': {
    title: '净资产变动表',
    subtitle: '期间报表｜经营结果、份额交易和利润分配共同解释净资产变化',
    rows: [
      { item: '期初净资产', current: '11,780', prior: '11,220', source: '上期关账余额' },
      { item: '本期经营活动产生的净资产变动', current: '282', prior: '145', source: '本期利润' },
      { item: '本期份额交易产生的净资产变动', current: '468', prior: '415', source: '持有人申购 / 赎回' },
      { item: '本期利润分配', current: '0', prior: '0', source: '利润分配明细账' },
      { item: '期末净资产', current: '12,530', prior: '11,780', source: '期初 + 各类变动', strong: true },
    ],
  },
}

const managerStatements: Record<string, { title: string; subtitle: string; rows: StatementRow[] }> = {
  '资产负债表': {
    title: '管理人资产负债表',
    subtitle: '公司时点报表｜不包含受托管理的基金财产',
    rows: [
      { item: '货币资金', current: '8,620', prior: '7,980', source: '管理人公司银行账户' },
      { item: '应收管理费', current: '286', prior: '251', source: '管理费应收明细账' },
      { item: '自有资金投资', current: '1,240', prior: '1,116', source: '管理人自有资金投资账' },
      { item: '资产合计', current: '13,480', prior: '12,704', source: '公司资产类科目汇总', strong: true },
      { item: '负债合计', current: '2,310', prior: '2,184', source: '公司负债类科目汇总', strong: true },
      { item: '所有者权益', current: '11,170', prior: '10,520', source: '实收资本 / 留存收益', strong: true },
    ],
  },
  '利润表': {
    title: '管理人利润表',
    subtitle: '公司期间报表｜管理费收入与管理人自身成本费用',
    rows: [
      { item: '管理费收入', current: '1,286', prior: '1,142', source: '公司总账及 Booking 来源号勾稽' },
      { item: '投资收益', current: '82', prior: '64', source: '自有资金投资明细账' },
      { item: '营业成本及管理费用', current: '-716', prior: '-682', source: '公司费用明细账' },
      { item: '税费及其他', current: '-128', prior: '-116', source: '税费及其他损益科目' },
      { item: '净利润', current: '524', prior: '408', source: '公司损益类科目结转', strong: true },
    ],
  },
  '现金流量表': {
    title: '管理人现金流量表',
    subtitle: '公司期间报表｜经营、投资与筹资活动现金流',
    rows: [
      { item: '经营活动现金流量净额', current: '612', prior: '486', source: '管理费收款及公司运营支出' },
      { item: '投资活动现金流量净额', current: '-124', prior: '-86', source: '管理人自有资金投资' },
      { item: '筹资活动现金流量净额', current: '0', prior: '0', source: '公司筹资业务' },
      { item: '现金及现金等价物净增加额', current: '488', prior: '400', source: '三类现金流合计', strong: true },
    ],
  },
  '所有者权益变动表': {
    title: '管理人所有者权益变动表',
    subtitle: '公司期间报表｜资本、利润和分配共同解释权益变化',
    rows: [
      { item: '期初所有者权益', current: '10,520', prior: '10,112', source: '上期公司关账余额' },
      { item: '本期净利润', current: '524', prior: '408', source: '管理人利润表' },
      { item: '其他权益变动', current: '126', prior: '0', source: '公司权益类明细账' },
      { item: '期末所有者权益', current: '11,170', prior: '10,520', source: '期初 + 本期变动', strong: true },
    ],
  },
}

const scopeOptions: StatementScope[] = ['组合/基金账', '管理人公司账']

export default function FinancialStatementsWorkspace() {
  const { selectedPortfolio } = useActualPortfolio()
  const [scope, setScope] = useState<StatementScope>('组合/基金账')
  const statements = scope === '组合/基金账' ? portfolioStatements : managerStatements
  const statementTabs = [...Object.keys(statements), '附注与勾稽']
  const [activeTab, setActiveTab] = useState(statementTabs[0])
  const [notice, setNotice] = useState('')
  const statement = statements[activeTab]
  const selectScope = (nextScope: StatementScope) => {
    setScope(nextScope)
    const nextStatements = nextScope === '组合/基金账' ? portfolioStatements : managerStatements
    setActiveTab(Object.keys(nextStatements)[0])
    setNotice('')
  }

  return (
    <div className="space-y-3" data-testid="financial-statements-workspace">
      <StaticDemoBanner compact />
      <header>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h1 className="text-lg font-bold text-slate-950 sm:text-2xl">双主体财务报表</h1>
          <button type="button" onClick={() => setNotice('已刷新静态报表预览；未生成、保存或发布正式财务报表。')} className="min-h-10 rounded-lg border border-slate-300 bg-white px-2 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">生成报表预览</button>
        </div>
        <p className="mt-1 text-sm leading-6 text-slate-600">组合与管理人分别核算，通过来源号勾稽。</p>
      </header>
      {notice ? <p className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-900" aria-live="polite">{notice}</p> : null}

      <section className="rounded-xl border border-slate-200 bg-white p-2 shadow-sm" aria-label="报表核算主体">
        <div className="grid grid-cols-2 gap-2">
          {scopeOptions.map((option) => <button type="button" key={option} onClick={() => selectScope(option)} aria-pressed={scope === option} className={`min-h-11 rounded-lg px-2 py-2 text-sm font-semibold transition ${scope === option ? 'bg-accent-900 text-white shadow-sm' : 'text-slate-600 hover:bg-slate-100'}`}>{option}</button>)}
        </div>
      </section>

      <nav className="flex flex-wrap gap-1 rounded-xl border border-slate-200 bg-white p-1" aria-label="财务报表页面标签">
        {statementTabs.map((tab) => <button type="button" key={tab} onClick={() => setActiveTab(tab)} aria-pressed={activeTab === tab} className={`min-h-10 whitespace-nowrap rounded-lg px-3 py-2 text-sm font-semibold ${activeTab === tab ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>{tab}</button>)}
      </nav>

      {statement ? (
        <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
          <div className="border-b border-slate-100 px-3 py-3 sm:px-5"><h2 id="statement-title" className="text-lg font-bold text-slate-950">{statement.title}</h2><p className="mt-1 text-xs leading-5 text-slate-600">2026 年 8 月 · 单位：万元 · 示例数据</p><p className="mt-1 text-xs leading-5 text-slate-600 sm:hidden">左右滑动查看完整报表</p></div>
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-labelledby="statement-title"><table className="w-full min-w-[560px] table-fixed text-sm sm:min-w-[720px]"><caption className="sr-only">{statement.subtitle}</caption><colgroup><col className="w-32 sm:w-60" /><col className="w-20 sm:w-28" /><col className="w-20 sm:w-28" /><col /></colgroup><thead className="bg-slate-50 text-left text-xs tracking-wide text-slate-600"><tr><th scope="col" className="sticky left-0 z-10 bg-slate-50 px-3 py-3">报表项目</th><th scope="col" className="px-3 py-3 text-right">本期</th><th scope="col" className="px-3 py-3 text-right">上期</th><th scope="col" className="px-3 py-3">总账来源</th></tr></thead><tbody>{statement.rows.map((row) => <tr key={row.item} className={`border-t border-slate-100 ${row.strong ? 'bg-accent-50 font-semibold text-slate-950' : 'bg-white text-slate-700'}`}><th scope="row" className={`sticky left-0 z-10 break-words px-3 py-3 text-left ${row.strong ? 'bg-accent-50 font-semibold' : 'bg-white font-normal'}`}>{row.item}</th><td className="px-3 py-3 text-right tabular-nums">{row.current}</td><td className="px-3 py-3 text-right tabular-nums">{row.prior}</td><td className="px-3 py-3 text-slate-600">{row.source}</td></tr>)}</tbody></table></div>
        </section>
      ) : (
        <section className="grid gap-4 lg:grid-cols-2">
          {scope === '组合/基金账' ? <article className="rounded-xl border border-emerald-200 bg-emerald-50 p-5"><h3 className="font-semibold text-emerald-950">组合/基金账核心勾稽</h3><ul className="mt-3 space-y-2 text-sm leading-6 text-emerald-900"><li>资产 12,648 = 负债 118 + 净资产 12,530</li><li>期末净资产 12,530 = 期初 11,780 + 本期利润 282 + 份额交易 468</li><li>利润表本期利润 282 = 净资产变动表经营活动变动 282</li></ul></article> : <article className="rounded-xl border border-emerald-200 bg-emerald-50 p-5"><h3 className="font-semibold text-emerald-950">管理人公司账核心勾稽</h3><ul className="mt-3 space-y-2 text-sm leading-6 text-emerald-900"><li>资产 13,480 = 负债 2,310 + 所有者权益 11,170</li><li>期末所有者权益 11,170 = 期初 10,520 + 净利润 524 + 其他变动 126</li><li>管理费应收与各组合应付管理人报酬按来源号核对</li></ul></article>}
          <aside className="border-l-2 border-slate-300 bg-slate-50/70 px-4 py-4" role="note" aria-label="报表范围说明（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">报表范围说明</h3><p className="mt-2 text-sm leading-6 text-slate-600">{scope === '组合/基金账' ? '资产管理产品展示资产负债表、利润表、净资产变动表和附注；现金流量表按适用情形提供。每个基金或组合分别出表。' : '管理人公司按企业会计口径形成公司级报表。受托管理的基金财产不作为管理人固有资产；是否需要合并某个结构化主体应另行判断。'}</p></aside>
          <section className="border-y border-slate-200 py-4 lg:col-span-2" aria-label="报表生成链路（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">流程说明 · 非交互</p><h3 className="mt-1 font-semibold text-slate-800">报表生成链路</h3><ol className="mt-3 flex flex-col gap-2 text-sm text-slate-600 sm:flex-row sm:flex-wrap sm:items-center">{(scope === '组合/基金账' ? ['Booking 事件', '复式凭证', '总账与明细账', '试算与估值', '组合账关账', '基金报表与绩效'] : ['Booking 事件', '公司凭证映射', '公司总账回执', '公司试算平衡', '管理人账关账', '公司财务报表']).map((step, index) => <li key={step} className="flex items-center gap-2"><span className="font-mono text-xs text-slate-600">{String(index + 1).padStart(2, '0')}</span><span>{step}</span>{index < 5 ? <span className="hidden text-slate-600 sm:inline" aria-hidden="true">→</span> : null}</li>)}</ol></section>
          <aside className="border-l-2 border-slate-300 bg-slate-50/70 px-4 py-4 lg:col-span-2" role="note" aria-label="两套报表关联说明（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">两套报表如何关联</h3><p className="mt-2 text-sm leading-6 text-slate-600">管理费、自有资金跟投等事项通过 Booking 来源号建立双边关系；系统展示金额、日期和对手主体差异，但不会把两套报表直接相加，也不会用跨主体抵销掩盖差异。</p></aside>
        </section>
      )}

      <details className="rounded-xl border border-slate-200 bg-white px-3 py-3">
        <summary className="cursor-pointer text-sm font-semibold text-slate-700">报表期间、账簿与核算边界</summary>
        <dl className="mt-3 grid gap-3 sm:grid-cols-3">
          {[
            ['报表期间', '2026 年 8 月', '示例月度期间'],
            ['核算主体', scope === '组合/基金账' ? selectedPortfolio.accountingEntityId : selectedPortfolio.managerId, scope === '组合/基金账' ? `${selectedPortfolio.name} · 独立核算主体` : `${selectedPortfolio.managerName} · 公司级核算主体`],
            ['账簿版本', scope === '组合/基金账' ? selectedPortfolio.primaryLedgerId : 'CORP-LEDGER-DEMO', '未完成正式复核'],
          ].map(([label, value, hint]) => <div key={label} className="min-w-0"><dt className="text-xs text-slate-600">{label}</dt><dd className="mt-1 break-words text-sm font-semibold text-slate-950">{value}</dd><dd className="mt-1 text-xs text-slate-600">{hint}</dd></div>)}
        </dl>
        <aside className="mt-4 border-t border-slate-200 pt-3" role="note" aria-label="基金经理报表边界说明（非交互）"><h3 className="font-semibold text-slate-800">基金经理和 Sleeve 不是当然的报表主体</h3><p className="mt-2 text-sm leading-6 text-slate-600">成交可以按基金经理任职关系和内部投资单元拆分，用于责任分析、辅助核算和绩效归因；法定财务报表仍按基金或其他独立核算主体生成，不能因为内部拆分而重复确认资产、负债和损益。</p><p className="mt-2 text-sm leading-6 text-slate-600">组合/基金账与管理人公司账分别关账、分别出表。两套报表通过 Booking 来源号勾稽，但不能直接混加或相互抵销。</p></aside>
      </details>
    </div>
  )
}
