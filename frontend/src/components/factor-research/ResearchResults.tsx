import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { numberText, percentText, type FactorRun } from '../../services/factorResearch'
import { buttonClass, Card, secondaryClass } from './shared'
import RollingICResults from './RollingICResults'

export default function ResearchResults({ run, onPublish, onBuildReturns }: { run: FactorRun; onPublish: () => void; onBuildReturns?: () => void }) {
  const [sample, setSample] = useState<'in_sample' | 'out_of_sample'>('out_of_sample')
  const summary = run.summaries[sample]
  const performance = summary.performance
  const composite = summary.factors[summary.factors.length - 1]
  const factors = run.factor_snapshots
  const curve = {
    tooltip: { trigger: 'axis' }, legend: { data: ['因子组合（扣费）', '比较基准'], top: 0 },
    grid: { left: 45, right: 16, top: 42, bottom: 56 },
    xAxis: { type: 'category', data: run.curves.map(row => row.date), boundaryGap: false },
    yAxis: { type: 'value', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }],
    series: [{ name: '因子组合（扣费）', type: 'line', showSymbol: false, connectNulls: false, data: run.curves.map(row => row.nav), lineStyle: { color: '#4f46e5', width: 2 },
      markLine: { symbol: 'none', label: { formatter: '样本外开始' }, data: [{ xAxis: run.study_snapshot.oos_date }] } },
      { name: '比较基准', type: 'line', showSymbol: false, connectNulls: false, data: run.curves.map(row => row.benchmark_nav), lineStyle: { color: '#94a3b8', width: 1.5 } }],
  }
  return <div className="space-y-5" aria-label="因子检验结果">
    <Card title={run.name + ' · 检验结果'}>
      <div className="flex flex-wrap items-center justify-between gap-3"><p className="text-sm text-slate-600">方案 v{run.study_revision} · 数据截至 {run.as_of} · {run.latest_scores.length} 个研究产品</p><div className="flex flex-wrap gap-2">{onBuildReturns && <button className={secondaryClass} onClick={onBuildReturns}>构建因子收益率</button>}<button className={buttonClass} onClick={onPublish}>发布研究版</button></div></div>
      <div className="mt-4 flex flex-wrap gap-2">{([['out_of_sample', '样本外'], ['in_sample', '样本内']] as const).map(([id, label]) => <button className={sample === id ? buttonClass : secondaryClass} aria-pressed={sample === id} key={id} onClick={() => setSample(id)}>{label}</button>)}</div>
      <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{[
        ['扣费累计收益', percentText(performance.total_return)], ['年化收益', percentText(performance.annualized_return)],
        ['最大回撤', percentText(performance.max_drawdown)], ['累计超额（差值）', percentText(performance.excess_return)],
        ['组合 RankIC', numberText(composite.rank_ic.mean)], ['有效检验截面', numberText(composite.rank_ic.observations, 0)],
        ['年化波动', percentText(performance.annualized_volatility)], ['累计换手（双边）', numberText(performance.turnover, 2)],
      ].map(([label, value]) => <div key={label} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-500">{label}</p><p className="mt-2 text-xl font-bold tabular-nums text-slate-900">{value}</p></div>)}</div>
      <p className="mt-3 text-xs leading-5 text-slate-500">收益统计基于每日净值路径，费用已计入；累计费用比率之和 {percentText(performance.fee_sum)}。样本内检验剔除标签跨界的信号。ICIR 为逐期均值 ÷ 标准差，未年化。</p>
    </Card>
    <Card title="组合净值与比较基准">
      <p className="mb-3 text-xs text-slate-500">{run.study_snapshot.benchmark.label} · {run.study_snapshot.benchmark.code} · 图形展示完整研究区间</p>
      <ReactECharts option={curve} style={{ height: 320, width: '100%' }} notMerge />
    </Card>
    <div className="grid min-w-0 gap-5 xl:grid-cols-2">
      <Card title="因子检验"><div className="overflow-x-auto"><table className="w-full min-w-[440px] text-left text-sm" aria-label="因子检验统计"><thead className="text-xs text-slate-500"><tr><th className="p-2">因子</th><th>IC</th><th>RankIC</th><th>ICIR</th><th>正值比例</th></tr></thead><tbody>{summary.factors.map(item => <tr className="border-t border-slate-100" key={item.factor_id}><th className="p-2 font-medium">{item.name}</th><td>{numberText(item.ic.mean)}</td><td>{numberText(item.rank_ic.mean)}</td><td>{numberText(item.rank_ic.icir)}</td><td>{percentText(item.rank_ic.positive_rate)}</td></tr>)}</tbody></table></div></Card>
      <Card title="分组远期收益">
        <ReactECharts style={{ height: 235 }} notMerge option={{ grid: { left: 60, right: 15, top: 20, bottom: 28 }, tooltip: { trigger: 'axis' }, xAxis: { type: 'category', data: summary.group_returns.map((_, i) => 'Q' + (i + 1)) }, yAxis: { type: 'value' }, series: [{ type: 'bar', data: summary.group_returns.map(item => item.mean), itemStyle: { color: '#6366f1', borderRadius: [4, 4, 0, 0] } }] }} />
        <p className="text-xs leading-5 text-slate-500">Q1 为低分组，末组为高分组。展示 {run.study_snapshot.horizon} 日标签平均收益（小数）；这是因子检验，不是可直接连乘的组合净值。</p>
      </Card>
    </div>
    <RollingICResults key={run.id} run={run} sample={sample} />
    <Card title="因子相关性"><div className="overflow-x-auto"><table className="w-full min-w-[400px] text-left text-sm" aria-label="因子相关性"><thead><tr><th className="p-2">因子</th>{factors.map(f => <th key={f.id} className="p-2 font-medium">{f.name}</th>)}</tr></thead><tbody>{factors.map((factor, i) => <tr className="border-t border-slate-100" key={factor.id}><th className="p-2 font-medium">{factor.name}</th>{summary.factor_correlation[i].map((value, j) => <td key={j} className="p-2 tabular-nums">{numberText(value)}</td>)}</tr>)}</tbody></table></div><p className="mt-2 text-xs text-slate-500">所选样本区间的标准化因子面板相关性，用于识别信息重复。</p></Card>
    <Card title="最新产品得分与因子贡献">
      <div className="overflow-x-auto"><table className="w-full min-w-[640px] text-left text-sm" aria-label="最新因子得分"><thead className="text-xs text-slate-500"><tr><th className="p-2">排名 / 产品</th><th>组合分数</th>{factors.map(f => <th className="p-2" key={f.id}>{f.name}<span className="block font-normal">原始值 / 分数贡献</span></th>)}</tr></thead><tbody>{run.latest_scores.map(row => <tr key={row.code} className="border-t border-slate-100"><th className="p-2 font-medium"><span className="text-indigo-600">{numberText(row.rank, 1)}</span> · <a className="hover:underline" href={'/product-research/products/' + encodeURIComponent(row.product_id) + '?kind=' + row.kind}>{row.name}</a><span className="block text-xs font-normal text-slate-500">{row.code}{row.exclusion_reason && ' · ' + row.exclusion_reason}</span></th><td className="font-semibold">{numberText(row.score, 2)}</td>{row.factors.map(f => <td className="p-2" key={f.factor_id}><span>{numberText(f.raw_value, 4)}</span><span className="block text-xs text-indigo-600">{numberText(f.contribution, 2)}</span></td>)}</tr>)}</tbody></table></div>
      <p className="mt-3 text-xs text-slate-500">这是 {run.as_of} 的可用数据得分快照；最新分数尚没有完整远期标签。空值保持缺失。</p>
    </Card>
    <details className="rounded-xl border border-slate-200 bg-white p-4"><summary className="cursor-pointer text-sm font-semibold text-slate-800">逐期检验与数据证据</summary>
      <div className="mt-4 max-h-72 overflow-auto"><table className="w-full min-w-[560px] text-left text-xs"><thead><tr><th className="p-2">信号日</th><th>模拟入场日</th><th>标签终点</th><th>样本</th><th>组合 RankIC</th><th>配对数</th></tr></thead><tbody>{run.periods.map(row => <tr key={row.date} className="border-t border-slate-100"><td className="p-2">{row.date}</td><td>{row.entry_date || '—'}</td><td>{row.label_end || '—'}</td><td>{row.sample === 'in_sample' ? '样本内' : row.sample === 'out_of_sample' ? '样本外' : '跨界剔除'}</td><td>{numberText(row.rank_ic[row.rank_ic.length - 1])}</td><td>{row.pair_counts[row.pair_counts.length - 1]}</td></tr>)}</tbody></table></div>
      <p className="mt-3 break-all text-xs text-slate-500">运行：{run.id}<br />输入哈希：{run.input_checksum}<br />数据快照：{run.data_lineage.snapshot}<br />计算版本：{run.engine_version}</p>
      <pre className="mt-3 max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-slate-50 p-3 text-[11px]">{JSON.stringify({ quality: run.data_quality, lineage: run.data_lineage, execution: run.execution }, null, 2)}</pre>
    </details>
    <Card title="结果解释与适用范围"><ul className="list-disc space-y-2 pl-5 text-sm leading-6 text-slate-600">{run.warnings.map(warning => <li key={warning}>{warning}</li>)}</ul></Card>
  </div>
}
