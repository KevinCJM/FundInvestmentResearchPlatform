import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import StaticDemoBanner from '../components/StaticDemoBanner'
import { getPrototypeConfig, type PrototypeConfig } from '../app/prototypeRegistry'
import { getPrototypeBlueprint, type PrototypeBlueprint } from '../app/prototypeBlueprintRegistry'

const familyStyles: Record<PrototypeConfig['family'], string> = {
  product: 'from-accent-700 to-accent-600',
  allocation: 'from-accent-700 to-accent-600',
  execution: 'from-orange-800 to-orange-700',
  accounting: 'from-accent-800 to-accent-700',
  post: 'from-emerald-700 to-accent-600',
  feedback: 'from-rose-800 to-pink-700',
  settings: 'from-slate-800 to-slate-600',
}

function SampleChart({ config }: { config: PrototypeConfig }) {
  const maxValue = Math.max(...config.chartPoints.map((point) => point.value), 1)
  return (
    <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <h3 className="font-semibold text-slate-900">{config.chartTitle}</h3>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-600">示例数据</span>
      </div>
      <div className="mt-6 flex h-52 items-end gap-3" aria-label={config.chartTitle}>
        {config.chartPoints.map((point) => (
          <div key={point.label} className="flex min-w-0 flex-1 flex-col items-center justify-end gap-2">
            <span className="text-xs font-semibold text-slate-600">{point.value}</span>
            <div className={`w-full max-w-16 rounded-t-lg bg-gradient-to-t ${familyStyles[config.family]}`} style={{ height: `${Math.max(12, (point.value / maxValue) * 150)}px` }} />
            <span className="truncate text-xs text-slate-600">{point.label}</span>
          </div>
        ))}
      </div>
    </section>
  )
}

function GuidancePanel({ config }: { config: PrototypeConfig }) {
  return (
    <aside className="border-l-2 border-slate-300 bg-slate-50/70 px-4 py-4" role="note" aria-label="研究口径说明（非交互）">
      <p className="text-xs font-semibold tracking-wide text-slate-600">非交互说明</p>
      <h3 className="mt-1 font-semibold text-slate-800">研究口径说明</h3>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-6 text-slate-600">
        {config.guidance.map((item) => <li key={item}>{item}</li>)}
      </ul>
    </aside>
  )
}

function BlueprintPanel({ blueprint }: { blueprint: PrototypeBlueprint }) {
  const boundaries = [
    { label: '本节点负责', value: blueprint.boundary.owns },
    { label: '复用公共或上游能力', value: blueprint.boundary.reuses },
    { label: '明确不负责', value: blueprint.boundary.excludes },
  ]
  return (
    <details data-testid="non-interactive-blueprint" className="group overflow-hidden rounded-xl border border-dashed border-slate-300 bg-slate-50/60">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-4 py-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent-500 [&::-webkit-details-marker]:hidden">
        <div>
          <h3 className="text-sm font-semibold text-slate-700">页面职责说明（非功能区）</h3>
          <p className="mt-0.5 text-xs text-slate-600">仅用于解释页面边界，不可执行业务操作</p>
        </div>
        <span className="shrink-0 text-xs font-medium text-slate-600 group-open:hidden">展开查看 ↓</span>
        <span className="hidden shrink-0 text-xs font-medium text-slate-600 group-open:inline">收起说明 ↑</span>
      </summary>

      <div className="border-t border-slate-200 px-4 py-5 sm:px-5" aria-label="页面职责说明内容">
        <dl className="divide-y divide-slate-200 border-y border-slate-200">
          {boundaries.map((item) => <div key={item.label} className="grid gap-1 py-3 sm:grid-cols-[180px_1fr] sm:gap-5"><dt className="text-sm font-semibold text-slate-700">{item.label}</dt><dd className="text-sm leading-6 text-slate-600">{item.value}</dd></div>)}
        </dl>

        <div className="mt-6 grid gap-6 xl:grid-cols-2">
          <section aria-labelledby="blueprint-steps-title">
            <h4 id="blueprint-steps-title" className="text-sm font-semibold text-slate-800">节点工作步骤</h4>
            <ol className="mt-3 list-decimal space-y-2 pl-5 text-sm leading-6 text-slate-600">{blueprint.steps.map((step) => <li key={step}>{step}</li>)}</ol>
          </section>
          <section aria-labelledby="blueprint-outputs-title">
            <h4 id="blueprint-outputs-title" className="text-sm font-semibold text-slate-800">形成的研究产出</h4>
            <ul className="mt-3 list-disc space-y-2 pl-5 text-sm leading-6 text-slate-600">{blueprint.outputs.map((output) => <li key={output}>{output}</li>)}</ul>
          </section>
        </div>

        <div className="mt-6 divide-y divide-slate-200 border-t border-slate-200">{blueprint.capabilities.map((group) => <section key={group.title} className="grid gap-2 py-4 sm:grid-cols-[180px_1fr] sm:gap-5"><h4 className="text-sm font-semibold text-slate-800">{group.title}</h4><ul className="list-disc space-y-1 pl-5 text-sm leading-6 text-slate-600">{group.items.map((item) => <li key={item}>{item}</li>)}</ul></section>)}</div>
      </div>
    </details>
  )
}

function RecordsTable({ config, rows, onRemove }: { config: PrototypeConfig; rows: string[][]; onRemove: (index: number) => void }) {
  return (
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[720px] text-sm">
          <thead className="bg-slate-50 text-left text-xs font-semibold text-slate-600"><tr>{config.columns.map((column) => <th scope="col" key={column} className="px-4 py-3">{column}</th>)}<th scope="col" className="px-4 py-3 text-right">演示操作</th></tr></thead>
          <tbody>{rows.map((row, rowIndex) => <tr key={`${row.join('-')}-${rowIndex}`} className="border-t border-slate-100"><>{row.map((cell, index) => <td key={`${cell}-${index}`} className="px-4 py-3 text-slate-700">{cell}</td>)}</><td className="px-4 py-3 text-right"><button type="button" onClick={() => onRemove(rowIndex)} className="rounded-lg px-2 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-50">移除示例</button></td></tr>)}</tbody>
        </table>
      </div>
      {rows.length === 0 && <p className="p-6 text-center text-sm text-slate-600">当前筛选下没有演示记录。</p>}
    </section>
  )
}

export default function PrototypeWorkspace({ pageKey }: { pageKey: string }) {
  const config = getPrototypeConfig(pageKey)
  const blueprint = getPrototypeBlueprint(pageKey)
  const tabs = config.tabs ?? ['研究概览', '参数与口径', '记录明细']
  const [activeTab, setActiveTab] = useState(tabs[0])
  const [scope, setScope] = useState(config.filterOptions[0])
  const [keyword, setKeyword] = useState('')
  const [rows, setRows] = useState(() => config.rows.map((row) => [...row]))
  const [notice, setNotice] = useState('')

  const filteredRows = useMemo(() => {
    const normalized = keyword.trim().toLowerCase()
    return normalized ? rows.filter((row) => row.some((cell) => cell.toLowerCase().includes(normalized))) : rows
  }, [keyword, rows])

  const addRow = (cells: string[]) => {
    setRows((current) => [...current, cells])
    setNotice('已加入当前会话的演示记录；未保存到系统。')
  }
  const removeRow = (index: number) => {
    const target = filteredRows[index]
    setRows((current) => current.filter((row) => row !== target))
    setNotice('已从当前页面移除一条演示记录；未修改真实数据。')
  }
  const reset = () => {
    setRows(config.rows.map((row) => [...row]))
    setKeyword('')
    setScope(config.filterOptions[0])
    setNotice('已恢复预置示例数据。')
  }

  return (
    <div className="space-y-5" data-testid={`prototype-${pageKey}`}>
      <StaticDemoBanner />
      <header className={`rounded-xl bg-gradient-to-r ${familyStyles[config.family]} p-6 text-white shadow-sm`}>
        <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
          <div><p className="text-xs font-semibold uppercase tracking-[0.2em] text-white">Interactive prototype</p><h2 className="mt-2 text-2xl font-bold">{config.title}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-white">{config.description}</p></div>
          <span className="shrink-0 rounded-full border border-white/40 px-3 py-1 text-xs font-semibold">预置示例数据</span>
        </div>
      </header>

      <BlueprintPanel blueprint={blueprint} />

      <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white p-4 shadow-sm sm:flex-row sm:items-end sm:justify-between">
        <div className="flex flex-1 flex-col gap-3 sm:flex-row">
          <label className="text-sm text-slate-600">{config.filterLabel}<select value={scope} onChange={(event) => setScope(event.target.value)} className="mt-1 block w-full rounded-xl border border-slate-300 bg-white px-3 py-2 sm:w-56">{config.filterOptions.map((option) => <option key={option}>{option}</option>)}</select></label>
          <label className="flex-1 text-sm text-slate-600">筛选演示记录<input value={keyword} onChange={(event) => setKeyword(event.target.value)} placeholder="输入关键词" className="mt-1 block w-full rounded-lg border border-slate-300 px-3 py-2" /></label>
        </div>
        <button type="button" onClick={reset} className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50">恢复示例</button>
      </div>

      <nav className="flex gap-1 overflow-x-auto rounded-xl border border-slate-200 bg-white p-1" aria-label={`${config.title}页面标签`}>
        {tabs.map((tab) => <button type="button" key={tab} onClick={() => setActiveTab(tab)} className={`whitespace-nowrap rounded-lg px-4 py-2 text-sm font-semibold ${activeTab === tab ? 'bg-slate-900 text-white' : 'text-slate-600 hover:bg-slate-100'}`}>{tab}</button>)}
      </nav>

      {activeTab === tabs[0] ? <><section className="grid gap-4 sm:grid-cols-3">{config.metrics.map((metric) => <article key={metric.label} className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><p className="text-sm text-slate-600">{metric.label}</p><p className="mt-2 text-2xl font-bold text-slate-950">{metric.value}</p><p className="mt-1 text-xs text-slate-600">{metric.hint}</p></article>)}</section><div className="grid gap-5 xl:grid-cols-[minmax(0,1.4fr)_minmax(280px,0.6fr)]"><SampleChart config={config} /><GuidancePanel config={config} /></div></> : null}

      {activeTab !== tabs[0] && activeTab !== tabs[tabs.length - 1] ? <div className="grid gap-5 lg:grid-cols-2"><GuidancePanel config={config} /><section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h3 className="font-semibold text-slate-900">{activeTab}（示例）</h3><div className="mt-4 space-y-3">{config.metrics.map((metric) => <label key={metric.label} className="block text-sm text-slate-600">{metric.label}<input value={metric.value} readOnly className="mt-1 w-full rounded-lg border border-slate-300 bg-slate-50 px-3 py-2" /></label>)}</div><p className="mt-4 text-xs text-slate-600">输入控件用于呈现未来交互结构，本期不计算、不保存。</p></section></div> : null}

      {activeTab === tabs[tabs.length - 1] ? <RecordsTable config={config} rows={filteredRows} onRemove={removeRow} /> : null}

      {config.links?.length ? <section className="rounded-xl border border-emerald-200 bg-emerald-50 p-5"><h3 className="font-semibold text-emerald-950">可复用的已有能力</h3><div className="mt-3 grid gap-3 md:grid-cols-2">{config.links.map((link) => <Link key={link.path} to={link.path} className="rounded-xl border border-emerald-200 bg-white p-4 hover:border-accent-400"><span className="font-semibold text-emerald-800">{link.label} →</span><p className="mt-1 text-sm text-slate-600">{link.description}</p></Link>)}</div></section> : null}
      {notice && <p className="rounded-xl border border-accent-200 bg-accent-50 px-4 py-3 text-sm text-accent-800" aria-live="polite">{notice}</p>}
    </div>
  )
}
