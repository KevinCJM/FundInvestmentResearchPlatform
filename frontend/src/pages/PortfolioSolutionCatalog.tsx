import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import StaticDemoBanner from '../components/StaticDemoBanner'
import { portfolioSolutions } from '../app/portfolioSolutionDemoData'

export default function PortfolioSolutionCatalog() {
  const [risk, setRisk] = useState('全部风险等级')
  const [keyword, setKeyword] = useState('')
  const filtered = useMemo(() => portfolioSolutions.filter((solution) => (
    (risk === '全部风险等级' || solution.riskLevel === risk)
    && `${solution.name}${solution.subtitle}`.includes(keyword.trim())
  )), [keyword, risk])

  return (
    <div className="space-y-5">
      <StaticDemoBanner />
      <header className="rounded-xl bg-gradient-to-r from-accent-800 to-accent-700 p-6 text-white shadow-sm">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-accent-100">Solution shelf</p>
        <h2 className="mt-2 text-2xl font-bold">组合方案目录</h2>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-accent-50/90">这里只陈列已建立展示版本的组合方案。研究草稿、临时回测和未冻结参数不会直接进入对外页面。</p>
      </header>

      <section className="grid gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:grid-cols-[220px_minmax(0,1fr)]">
        <label className="text-sm text-slate-600">风险等级
          <select value={risk} onChange={(event) => setRisk(event.target.value)} className="mt-1 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2">
            {['全部风险等级', '中低风险', '中风险', '中高风险'].map((option) => <option key={option}>{option}</option>)}
          </select>
        </label>
        <label className="text-sm text-slate-600">搜索组合方案
          <input value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="输入组合名称或定位" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" />
        </label>
      </section>

      <section className="grid gap-5 xl:grid-cols-2" aria-label="组合方案列表">
        {filtered.map((solution) => (
          <article key={solution.id} className="flex flex-col rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <div className="flex flex-wrap gap-2 text-xs font-semibold">
                  <span className="rounded-full bg-accent-100 px-2.5 py-1 text-accent-800">{solution.riskLevel}</span>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-600">{solution.status}</span>
                </div>
                <h3 className="mt-3 text-xl font-bold text-slate-950">{solution.name}</h3>
                <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">{solution.subtitle}</p>
              </div>
              <span className="rounded-lg border border-slate-200 px-2.5 py-1 text-xs font-semibold text-slate-600">{solution.version}</span>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ['年化收益', solution.annualizedReturn],
                ['年化波动', solution.volatility],
                ['最大回撤', solution.maxDrawdown],
                ['夏普比率', solution.sharpe],
              ].map(([label, value]) => <div key={label} className="rounded-xl bg-slate-50 p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 font-bold text-slate-900">{value}</p></div>)}
            </div>

            <div className="mt-5">
              {/* 分段之间留一条白线：相邻段的边界不能只靠色差，色差一变就连成一段。 */}
              <div role="img" aria-label={`大类配置：${solution.allocation.map((item) => `${item.label} ${item.value}%`).join('、')}`} className="flex h-2 overflow-hidden rounded-full bg-slate-100">
                {solution.allocation.map((item) => <span key={item.label} className={`${item.color} ring-1 ring-inset ring-white`} style={{ width: `${item.value}%` }} />)}
              </div>
              {/* 图例带色块：没有色块时条形图上的颜色读者无从对应。 */}
              <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-600">{solution.allocation.map((item) => <li key={item.label} className="flex items-center gap-1.5"><span aria-hidden="true" className={`h-2 w-2 shrink-0 rounded-full ${item.color}`} />{item.label} {item.value}%</li>)}</ul>
            </div>

            <div className="mt-5 flex flex-col gap-3 border-t border-slate-100 pt-4 sm:flex-row sm:items-center sm:justify-between">
              <p className="text-xs text-slate-600">展示数据截止：{solution.effectiveDate} · 预置示例</p>
              <Link to={`/portfolio-solutions/profile?solution=${solution.id}`} className="rounded-lg bg-accent-700 px-4 py-2 text-center text-sm font-semibold text-white hover:bg-accent-600 focus:outline-none focus:ring-2 focus:ring-accent-500">查看标准化展示 →</Link>
            </div>
          </article>
        ))}
      </section>
      {filtered.length === 0 ? <p className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-600">没有符合条件的示例组合。</p> : null}
    </div>
  )
}
