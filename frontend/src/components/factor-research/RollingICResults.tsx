import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { numberText, percentText, type FactorRun } from '../../services/factorResearch'
import { Card, Field, inputClass } from './shared'

export default function RollingICResults({ run, sample }: { run: FactorRun; sample: 'in_sample' | 'out_of_sample' }) {
  const [index, setIndex] = useState(run.factor_snapshots.length)
  const [metric, setMetric] = useState<'ic' | 'rank_ic'>('rank_ic')
  const rolling = run.rolling_diagnostics
  const names = [...run.factor_snapshots.map(factor => factor.name), '组合因子']
  const selected = Math.min(index, names.length - 1)
  if (!rolling) return <Card title="滚动 IC 观察"><p className="text-sm leading-6 text-slate-600">这是旧版运行，保留原有逐期 IC，不补造滚动结果。重新运行方案后可查看滚动均值、ICIR 和正值比例。</p></Card>
  const rows = rolling.rows.filter(row => row.sample === sample)
  const periods = new Map(run.periods.map(period => [period.date, period]))
  const label = metric === 'ic' ? 'IC' : 'RankIC'
  return <Card title="逐期 IC 与滚动统计">
    <p className="mb-4 text-sm leading-6 text-slate-600">逐期 IC：某次截面的特征与随后收益是否相关。滚动统计：最近 {rolling.window} 期的 IC 是否稳定，至少需要 {rolling.min_periods} 个有效截面。</p>
    <div className="mb-4 grid gap-3 sm:grid-cols-2"><Field label="滚动检验因子"><select className={inputClass} value={selected} onChange={event => setIndex(Number(event.target.value))}>{names.map((name, i) => <option key={i} value={i}>{name}</option>)}</select></Field><Field label="滚动相关口径"><select className={inputClass} value={metric} onChange={event => setMetric(event.target.value as 'ic' | 'rank_ic')}><option value="rank_ic">RankIC · 秩相关</option><option value="ic">IC · 线性相关</option></select></Field></div>
    {rows.length ? <>
      <ReactECharts style={{ height: 290, width: '100%' }} notMerge option={{
        tooltip: { trigger: 'axis' }, legend: { data: ['逐期 ' + label, '滚动均值'], top: 0 },
        grid: { left: 45, right: 18, top: 42, bottom: 55 },
        xAxis: { type: 'category', data: rows.map(row => row.date), boundaryGap: false }, yAxis: { type: 'value', min: -1, max: 1 },
        dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
        series: [
          { name: '逐期 ' + label, type: 'bar', data: rows.map(row => periods.get(row.signal_date)?.[metric][selected] ?? null) },
          { name: '滚动均值', type: 'line', showSymbol: false, connectNulls: false, data: rows.map(row => row[metric][selected]?.mean ?? null) },
        ],
      }} />
      <details className="mt-3"><summary className="cursor-pointer text-sm font-semibold text-slate-700">查看滚动统计明细</summary><div className="mt-3 max-h-72 overflow-auto"><table className="w-full min-w-[650px] text-left text-xs" aria-label="滚动 IC 统计"><thead><tr><th scope="col" className="p-2">标签兑现日</th><th scope="col">对应信号日</th><th scope="col">有效截面</th><th scope="col">滚动均值</th><th scope="col">ICIR（未年化）</th><th scope="col">正值比例</th></tr></thead><tbody>{rows.map(row => { const stats = row[metric][selected]; return <tr key={row.date} className="border-t border-slate-100"><td className="p-2">{row.date}</td><td>{row.signal_date}</td><td>{stats?.observations ?? 0}</td><td>{numberText(stats?.mean)}</td><td>{numberText(stats?.icir)}</td><td>{percentText(stats?.positive_rate)}</td></tr> })}</tbody></table></div></details>
    </> : <p className="rounded-lg bg-slate-50 p-4 text-sm text-slate-600">当前样本区间尚无已兑现的有效标签。</p>}
    <p className="mt-3 text-xs leading-6 text-slate-600">横轴为标签兑现日，不是信号日当时已知的信息；样本内外分开滚动，跨界标签剔除。相邻标签可能重叠，ICIR 不是独立样本显著性检验。</p>
  </Card>
}
