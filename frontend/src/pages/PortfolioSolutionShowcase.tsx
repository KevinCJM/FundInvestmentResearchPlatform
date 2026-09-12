import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import StaticDemoBanner from '../components/StaticDemoBanner'
import {
  algorithmRules,
  disclosureItems,
  performanceSeries,
  periodPerformance,
  portfolioSolutions,
  scenarioResults,
} from '../app/portfolioSolutionDemoData'

type ShowcaseView = 'profile' | 'performance' | 'backtest' | 'scenarios' | 'rules' | 'publishing'

const viewLabels: Record<ShowcaseView, { title: string; description: string }> = {
  profile: { title: '组合画像与适用范围', description: '用统一结构说明组合要解决什么问题、适合谁、承担什么风险，以及当前展示版本的配置特征。' },
  performance: { title: '收益与风险表现', description: '以同区间、同频率、同口径对比组合与基准，并同时呈现收益、风险和回撤。' },
  backtest: { title: '历史回测模拟', description: '展示冻结算法版本在 PIT 数据、交易成本和真实申赎规则下的历史模拟结果。' },
  scenarios: { title: '周期与情景模拟', description: '观察组合在经济周期、历史危机及自定义冲击下的损失、相对表现和恢复时间。' },
  rules: { title: '配置与算法说明', description: '解释组合从战略中枢、市场状态、产品选择到再平衡的规则链和版本依赖。' },
  publishing: { title: '披露与展示版本', description: '冻结展示口径、风险揭示和版本信息，形成可审阅、可归档的对外展示快照。' },
}

function MetricCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return <article className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 text-xl font-bold text-slate-950">{value}</p><p className="mt-1 text-xs text-slate-600">{hint}</p></article>
}

function PerformanceLineChart() {
  const toPoints = (field: 'portfolio' | 'benchmark') => performanceSeries.map((point, index) => {
    const x = 50 + index * 132
    const y = 220 - ((point[field] - 95) / 45) * 170
    return `${x},${y}`
  }).join(' ')

  return (
    <figure className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <figcaption className="font-semibold text-slate-900">组合与基准累计净值</figcaption>
        <div className="flex gap-3 text-xs"><span className="text-accent-700">● 组合</span><span className="text-slate-600">● 基准</span></div>
      </div>
      <svg className="mt-4 h-auto w-full" viewBox="0 0 760 260" role="img" aria-label="组合与基准累计净值示例折线图">
        {[70, 120, 170, 220].map((y) => <line key={y} x1="45" y1={y} x2="720" y2={y} stroke="#e2e8f0" strokeWidth="1" />)}
        <polyline points={toPoints('benchmark')} fill="none" stroke="#94a3b8" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
        <polyline points={toPoints('portfolio')} fill="none" stroke="#a21caf" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        {performanceSeries.map((point, index) => <text key={point.label} x={50 + index * 132} y="248" textAnchor="middle" fontSize="12" fill="#64748b">{point.label}</text>)}
        <line x1="446" y1="42" x2="446" y2="224" stroke="#0f172a" strokeDasharray="5 5" />
        <text x="454" y="57" fontSize="11" fill="#475569">实盘跟踪起点（示例）</text>
      </svg>
      <p className="mt-2 text-xs leading-5 text-slate-600">虚线左侧为历史回测，右侧为实盘跟踪示例。两段数据必须分别标识，不拼接为未经说明的连续实盘业绩。</p>
    </figure>
  )
}

function ProfileView({ solution }: { solution: (typeof portfolioSolutions)[number] }) {
  return <div className="space-y-5">
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard label="年化收益" value={solution.annualizedReturn} hint="费后模拟 + 实盘分段口径" />
      <MetricCard label="年化波动" value={solution.volatility} hint="月度收益年化" />
      <MetricCard label="最大回撤" value={solution.maxDrawdown} hint="展示期内峰谷回撤" />
      <MetricCard label="夏普比率" value={solution.sharpe} hint="无风险利率 2.0%（示例）" />
    </section>
    <div className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
      <PerformanceLineChart />
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h3 className="font-semibold text-slate-900">目标配置中枢</h3>
        <div className="mt-5 space-y-4">{solution.allocation.map((item) => <div key={item.label}><div className="flex justify-between text-sm"><span className="text-slate-600">{item.label}</span><span className="font-semibold text-slate-900">{item.value}%</span></div><div className="mt-1 h-2 rounded-full bg-slate-100"><div className={`h-2 rounded-full ${item.color}`} style={{ width: `${item.value}%` }} /></div></div>)}</div>
      </section>
    </div>
    <section className="border-y border-slate-200 bg-slate-50/60 px-4 py-4" aria-label="组合适用性说明（非交互）">
      <p className="text-xs font-semibold tracking-wide text-slate-600">组合说明 · 非交互</p>
      <dl className="mt-2 divide-y divide-slate-200">{[
        ['适用目标', '追求中长期资产稳健增值，能接受阶段性净值波动。'],
        ['不适用情形', '短期刚性资金用途、无法承受本金波动或需要保本承诺。'],
        ['主要风险', '市场风险、模型失效、产品风格漂移、流动性和申赎时滞。'],
      ].map(([title, text]) => <div key={title} className="grid gap-1 py-3 sm:grid-cols-[140px_1fr] sm:gap-5"><dt className="text-sm font-semibold text-slate-700">{title}</dt><dd className="text-sm leading-6 text-slate-600">{text}</dd></div>)}</dl>
    </section>
  </div>
}

function PerformanceView({ solution }: { solution: (typeof portfolioSolutions)[number] }) {
  const [period, setPeriod] = useState('成立以来')
  return <div className="space-y-5">
    <div className="flex flex-wrap gap-2 rounded-xl border border-slate-200 bg-white p-2" aria-label="表现区间">{['近1年', '近3年', '成立以来'].map((item) => <button key={item} type="button" onClick={() => setPeriod(item)} className={`rounded-lg px-4 py-2 text-sm font-semibold ${period === item ? 'bg-accent-700 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>{item}</button>)}</div>
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="区间年化收益" value={solution.annualizedReturn} hint={`${period} · 示例`} /><MetricCard label="相对基准" value="+1.54%" hint="同口径年化超额" /><MetricCard label="最大回撤" value={solution.maxDrawdown} hint="组合 / 基准 -9.16%" /><MetricCard label="下行波动" value="4.83%" hint="仅统计负收益月份" /></section>
    <PerformanceLineChart />
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm"><table className="w-full min-w-[640px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['区间', '组合收益', '基准收益', '相对收益'].map((item) => <th scope="col" key={item} className="px-4 py-3">{item}</th>)}</tr></thead><tbody>{periodPerformance.map((row) => <tr key={row[0]} className="border-t border-slate-100">{row.map((cell) => <td key={cell} className="px-4 py-3 text-slate-700">{cell}</td>)}</tr>)}</tbody></table></section>
  </div>
}

function BacktestView() {
  return <div className="space-y-5">
    <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="样本内年化" value="9.12%" hint="2016-01 至 2021-12" /><MetricCard label="样本外年化" value="7.68%" hint="2022-01 至 2025-12" /><MetricCard label="成本后衰减" value="-0.74%" hint="年化收益差" /><MetricCard label="参数稳定区间" value="78%" hint="通过敏感性组合占比" /></section>
    <PerformanceLineChart />
    <div className="grid gap-5 xl:grid-cols-2">
      <section className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">冻结回测口径</h3><dl className="mt-4 grid gap-3 text-sm">{[
        ['数据快照', 'PIT-2026-08-31'], ['算法版本', 'PORTFOLIO-V3.2'], ['产品池版本', 'POOL-FOF-R12'], ['再平衡', '月度检查 + 3% 偏离阈值'], ['交易成本', 'ETF 8bps；场外基金按申赎费率'], ['信号生效', 'T 日收盘计算，最早 T+1 生效'],
      ].map(([term, value]) => <div key={term} className="grid grid-cols-[100px_1fr] gap-3"><dt className="text-slate-600">{term}</dt><dd className="font-medium text-slate-800">{value}</dd></div>)}</dl></section>
      <section className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">稳健性检查</h3><ul className="mt-4 space-y-3 text-sm">{['Walk-forward：通过 8 / 10 个滚动窗口', '参数扰动：核心结论在 ±20% 参数变化下保持', '容量压力：1 亿至 5 亿规模下成本可控', '替代产品：主产品缺失时仍保留大类风险特征'].map((item) => <li key={item} className="flex gap-3 rounded-lg bg-emerald-50 px-3 py-2 text-emerald-900"><span aria-hidden="true">✓</span><span>{item}</span></li>)}</ul></section>
    </div>
  </div>
}

function ScenarioView() {
  const [selectedId, setSelectedId] = useState(scenarioResults[0].id)
  const selected = scenarioResults.find((item) => item.id === selectedId) ?? scenarioResults[0]
  return <div className="space-y-5">
    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">{scenarioResults.map((scenario) => <button type="button" key={scenario.id} onClick={() => setSelectedId(scenario.id)} className={`rounded-xl border p-4 text-left transition ${selectedId === scenario.id ? 'border-accent-400 bg-accent-50 ring-2 ring-accent-100' : 'border-slate-200 bg-white hover:border-accent-200'}`}><span className="text-sm font-semibold text-slate-900">{scenario.name}</span><span className="mt-1 block text-xs leading-5 text-slate-600">{scenario.description}</span></button>)}</section>
    <section className="rounded-xl bg-slate-950 p-6 text-white"><div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between"><div><p className="text-xs font-semibold uppercase tracking-wider text-accent-300">Selected scenario</p><h3 className="mt-2 text-2xl font-bold">{selected.name}</h3><p className="mt-2 text-sm text-slate-300">{selected.description}</p></div><span className="rounded-full bg-white/10 px-3 py-1 text-xs">{selected.confidence}</span></div><div className="mt-6 grid gap-4 sm:grid-cols-3">{[['组合模拟', selected.portfolio], ['基准模拟', selected.benchmark], ['恢复时间', selected.recovery]].map(([label, value]) => <div key={label} className="rounded-xl bg-white/[0.07] p-4"><p className="text-xs text-slate-400">{label}</p><p className="mt-1 text-xl font-bold">{value}</p></div>)}</div></section>
    <section className="grid gap-5 xl:grid-cols-2"><article className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">周期矩阵</h3><div className="mt-4 grid grid-cols-2 gap-3">{[['衰退', '-4.8%'], ['滞胀', '-3.6%'], ['复苏', '+9.3%'], ['扩张', '+7.1%']].map(([label, value]) => <div key={label} className="rounded-xl bg-slate-50 p-4"><p className="text-sm text-slate-600">{label}</p><p className="mt-1 font-bold text-slate-900">{value}</p></div>)}</div></article><aside className="border-l-2 border-slate-300 bg-slate-50/70 px-4 py-4" role="note" aria-label="情景口径说明（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">非交互说明</p><h3 className="mt-1 font-semibold text-slate-800">口径边界</h3><ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-6 text-slate-600"><li>情景由“设置 / 情景算法中心”的冻结版本计算。</li><li>历史情景、因子冲击和周期模型必须区分展示。</li><li>结果是条件模拟，不是概率预测，也不是收益承诺。</li></ul></aside></section>
  </div>
}

function RulesView({ solution }: { solution: (typeof portfolioSolutions)[number] }) {
  return <div className="space-y-5">
    <section className="border-y border-slate-200 bg-slate-50/60 px-4 py-4" aria-label="组合规则链说明（非交互）"><p className="text-xs font-semibold tracking-wide text-slate-600">规则链说明 · 非交互</p><ol className="mt-3 flex flex-col gap-2 text-sm text-slate-700 lg:flex-row lg:flex-wrap lg:items-center">{['目标与约束', 'SAA 配置中枢', 'TAA 状态偏离', '产品选择与替代', '再平衡与风控'].map((item, index) => <li key={item} className="flex items-center gap-2"><span className="font-mono text-xs text-accent-700">0{index + 1}</span><span className="font-medium">{item}</span>{index < 4 ? <span className="hidden text-slate-600 lg:inline" aria-hidden="true">→</span> : null}</li>)}</ol></section>
    <div className="grid gap-5 xl:grid-cols-[0.7fr_1.3fr]"><section className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">资产配置边界</h3><div className="mt-4 space-y-4">{solution.allocation.map((item) => <div key={item.label}><div className="flex justify-between text-sm"><span>{item.label}</span><span className="font-semibold">中枢 {item.value}% · 区间 ±5%</span></div><div className="mt-1 h-2 rounded-full bg-slate-100"><div className={`h-2 rounded-full ${item.color}`} style={{ width: `${item.value}%` }} /></div></div>)}</div></section><section className="overflow-hidden rounded-xl border border-slate-200 bg-white"><table className="w-full min-w-[720px] text-sm"><thead className="bg-slate-50 text-left text-xs text-slate-600"><tr>{['模块', '方法', '运行频率', '版本'].map((item) => <th scope="col" key={item} className="px-4 py-3">{item}</th>)}</tr></thead><tbody>{algorithmRules.map((row) => <tr key={row[0]} className="border-t border-slate-100">{row.map((cell) => <td key={cell} className="px-4 py-3 text-slate-700">{cell}</td>)}</tr>)}</tbody></table></section></div>
  </div>
}

function PublishingView() {
  const [notice, setNotice] = useState('')
  return <div className="space-y-5">
    <section className="grid gap-4 md:grid-cols-3"><MetricCard label="展示版本" value="V3.2" hint="不可变快照" /><MetricCard label="数据截止日" value="2026-08-31" hint="PIT 快照锁定" /><MetricCard label="准备度" value="6 / 6" hint="示例披露项已齐备" /></section>
    <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]"><section className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">展示前披露检查</h3><ul className="mt-4 space-y-3">{disclosureItems.map((item) => <li key={item} className="flex gap-3 rounded-xl bg-emerald-50 px-4 py-3 text-sm leading-6 text-emerald-950"><span aria-hidden="true" className="font-bold">✓</span><span>{item}</span></li>)}</ul></section><section className="rounded-xl border border-slate-200 bg-white p-5"><h3 className="font-semibold text-slate-900">版本链</h3><dl className="mt-4 space-y-3 text-sm">{[['研究方案', 'PF-RESEARCH-R18'], ['产品池', 'POOL-FOF-R12'], ['回测档案', 'BT-STRICT-20260831'], ['情景算法', 'SCENARIO-R4'], ['指标方案', 'METRIC-R9'], ['展示快照', 'SHOWCASE-V3.2']].map(([term, value]) => <div key={term} className="rounded-lg bg-slate-50 p-3"><dt className="text-xs text-slate-500">{term}</dt><dd className="mt-1 font-mono text-xs font-semibold text-slate-800">{value}</dd></div>)}</dl><button type="button" onClick={() => setNotice('已生成当前会话内的静态预览；未发布、未保存。')} className="mt-5 w-full rounded-lg bg-accent-700 px-4 py-2 text-sm font-semibold text-white hover:bg-accent-600">生成静态预览</button>{notice ? <p className="mt-3 rounded-lg bg-accent-50 p-3 text-sm text-accent-800" aria-live="polite">{notice}</p> : null}</section></div>
  </div>
}

export default function PortfolioSolutionShowcase({ view }: { view: ShowcaseView }) {
  const [searchParams, setSearchParams] = useSearchParams()
  const solutionId = searchParams.get('solution') ?? portfolioSolutions[0].id
  const solution = useMemo(() => portfolioSolutions.find((item) => item.id === solutionId) ?? portfolioSolutions[0], [solutionId])
  const page = viewLabels[view]
  const updateSolution = (id: string) => setSearchParams(id === portfolioSolutions[0].id ? {} : { solution: id })

  return (
    <div className="space-y-5" data-testid={`portfolio-solution-${view}`}>
      <StaticDemoBanner />
      <header className="rounded-xl bg-gradient-to-r from-accent-900 via-accent-800 to-accent-800 p-6 text-white shadow-sm">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between"><div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-200">Published solution prototype</p><h2 className="mt-2 text-2xl font-bold">{page.title}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-white/80">{page.description}</p></div><label className="text-xs font-semibold text-accent-100">展示组合<select value={solution.id} onChange={(event) => updateSolution(event.target.value)} className="mt-1 block w-full min-w-64 rounded-lg border border-white/20 bg-white px-3 py-2 text-sm text-slate-900">{portfolioSolutions.map((item) => <option key={item.id} value={item.id}>{item.name} · {item.version}</option>)}</select></label></div>
      </header>

      <section className="rounded-xl border border-accent-200 bg-accent-50 p-5"><div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between"><div><div className="flex flex-wrap gap-2 text-xs font-semibold"><span className="rounded-full bg-accent-200 px-2.5 py-1 text-accent-900">{solution.riskLevel}</span><span className="rounded-full bg-white px-2.5 py-1 text-slate-600">{solution.horizon}</span><span className="rounded-full bg-white px-2.5 py-1 text-slate-600">{solution.status}</span></div><h3 className="mt-3 text-xl font-bold text-slate-950">{solution.name} · {solution.version}</h3><p className="mt-2 max-w-3xl text-sm leading-6 text-slate-600">{solution.subtitle}</p></div><Link to="/portfolio-solutions/catalog" className="shrink-0 text-sm font-semibold text-accent-800">返回方案目录 →</Link></div><p className="mt-4 border-t border-accent-200 pt-3 text-xs leading-5 text-slate-600"><span className="font-semibold">业绩比较基准：</span>{solution.benchmark}</p></section>

      {view === 'profile' ? <ProfileView solution={solution} /> : null}
      {view === 'performance' ? <PerformanceView solution={solution} /> : null}
      {view === 'backtest' ? <BacktestView /> : null}
      {view === 'scenarios' ? <ScenarioView /> : null}
      {view === 'rules' ? <RulesView solution={solution} /> : null}
      {view === 'publishing' ? <PublishingView /> : null}

      <footer className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-950"><span className="font-bold">重要说明：</span>页面数值均为预置示例。历史回测、情景模拟和目标描述不代表未来表现，不构成投资建议或收益承诺；正式对外展示前需由适用的合规流程确认。</footer>
    </div>
  )
}
