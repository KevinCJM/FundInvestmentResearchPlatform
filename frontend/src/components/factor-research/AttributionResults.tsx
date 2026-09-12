import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { numberText, percentText, type AttributionRun } from '../../services/factorResearch'
import { Card, Field, inputClass, secondaryClass } from './shared'
import { downloadBlob, downloadJson } from './returnShared'
import { attributionCsv, contributionStatus, contributionText } from './attributionPresentation'

type Sample = 'all' | 'in_sample' | 'out_of_sample'
const samples: Array<[Sample, string]> = [['out_of_sample', '样本外'], ['in_sample', '样本内'], ['all', '全区间']]
const pointAxis = (value: number) => (value * 100).toLocaleString('zh-CN', { maximumFractionDigits: 3 })
const pointTooltip = (value: unknown) => contributionText(typeof value === 'number' ? value : null)

function RegressionOverview({ run }: { run: AttributionRun }) {
  const rolling = run.attribution?.mode === 'rolling'
  return <div className="overflow-x-auto">
    <table className="w-full min-w-[760px] text-left text-sm" aria-label="暴露与拟合摘要">
      <thead className="text-xs text-slate-600"><tr><th scope="col" className="p-2">产品</th><th scope="col">风格 / 因子暴露</th><th scope="col">{rolling ? '末次窗口 R²' : '样本内 R²'}</th><th scope="col">{rolling ? '前推样本外 R²' : '样本外 R²'}</th><th scope="col">年化截距</th><th scope="col">样本外残差波动</th><th scope="col">拟合 / 样本外日数</th></tr></thead>
      <tbody>{run.results.map(row => <tr key={row.code} className="border-t border-slate-100">
        <th scope="row" className="p-2 font-medium">{row.name}<span className="block text-xs text-slate-600">{row.code}{row.reason && ' · ' + row.reason}</span></th>
        <td className="py-2">{row.exposures.map(exposure => <div className="text-xs" key={exposure.factor}>{exposure.factor}：{numberText(exposure.value)}</div>)}</td>
        <td>{numberText(row.train_r2)}</td><td>{numberText(row.test_r2)}</td><td>{percentText(row.annualized_intercept)}</td><td>{percentText(row.test_residual_volatility)}</td><td>{numberText(row.train_observations, 0)} / {numberText(row.test_observations, 0)}</td>
      </tr>)}</tbody>
    </table>
    {rolling && <p className="mt-2 text-xs leading-6 text-slate-600">此摘要展示末次计划拟合的系数；下方逐日贡献使用各日实际适用的系数，而非用末次系数回填历史。</p>}
  </div>
}

export default function AttributionResults({ run }: { run: AttributionRun }) {
  const [productCode, setProductCode] = useState('')
  const [sample, setSample] = useState<Sample>('out_of_sample')
  const [page, setPage] = useState(0)
  const [error, setError] = useState('')
  const analysis = run.attribution
  const product = analysis?.products.find(item => item.code === productCode) || analysis?.products[0]
  const summary = product?.summaries[sample]
  const daily = product?.daily.filter(row => sample === 'all' || row.sample === sample) || []
  const curve = product?.curves[sample] || []
  const factorComponents = analysis?.components.filter(item => item.kind === 'factor') || []
  const monthly = product ? Object.entries(product.summaries).filter(([key]) => /^\d{4}-\d{2}$/.test(key)) : []
  const rows = daily.slice(page * 40, (page + 1) * 40)
  const chartBase = {
    animation: false, tooltip: { trigger: 'axis', valueFormatter: pointTooltip },
    legend: { type: 'scroll', top: 0 }, grid: { left: 65, right: 20, top: 48, bottom: 65 },
    dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 5 }],
    xAxis: { type: 'category', boundaryGap: false, data: curve.map(row => row.date) },
    yAxis: { type: 'value', name: '百分点', axisLabel: { formatter: pointAxis }, scale: true },
  }
  const exporting = (work: () => void) => { setError(''); try { work() } catch (reason) { setError(reason instanceof Error ? reason.message : '导出失败。') } }
  return <div className="min-w-0 space-y-5" aria-label="收益归因结果">
    <Card title={run.name + ' · 风格与收益贡献'}>
      <p className="mb-3 break-words text-xs leading-6 text-slate-600">已保存运行 · {run.request.model} · {run.request.start_date}—{run.request.end_date} · 样本外从 {run.request.oos_date} 开始。来源：{run.request.dataset_id || '本地指数代理'}。</p>
      <RegressionOverview run={run} />
      <p className="mt-3 text-xs leading-6 text-slate-600">{run.warnings.join(' ')}</p>
      {!analysis && <p className="mt-4 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">旧运行未保存逐日贡献，请重新运行。不会根据历史摘要伪造贡献或曲线。</p>}
    </Card>
    {analysis && (!product || !summary) && <p className="text-sm text-slate-600">没有可展示的产品贡献结果。</p>}
    {analysis && product && summary && <>
      <Card title="查看归因对象与区间">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="归因结果产品"><select className={inputClass} value={product.code} onChange={e => { setProductCode(e.target.value); setPage(0) }}>{analysis.products.map(item => <option value={item.code} key={item.code}>{item.name} · {item.code}</option>)}</select></Field>
          <Field label="贡献评价区间"><select className={inputClass} value={sample} onChange={e => { setSample(e.target.value as Sample); setPage(0) }}>{samples.map(([key, label]) => <option value={key} key={key}>{label}</option>)}</select></Field>
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-600">{analysis.mode === 'rolling' ? `滚动暴露：排除起始 ${analysis.warmup_days} 个交易日预热，贡献评价从 ${analysis.evaluation_start} 开始。` : '固定暴露：样本内系数是事后拟合；样本外保持这组系数不变。'} 本区间：{summary.start_date || '—'} — {summary.end_date || '—'}。</p>
        <p className="mt-2 text-xs leading-6 text-slate-600">{analysis.dependent_return === 'excess' ? '拟合使用超额收益；贡献对账时单列 RF，恢复至基金实际总收益。' : '拟合使用总收益，不单列 RF；模型截距不是风险调整 Alpha。'} 系数不等同真实持仓比例。</p>
      </Card>
      <Card title="累计贡献与实际收益对账">
        <p role="status" className={'mb-4 rounded-lg p-3 text-sm ' + (summary.status === 'complete' ? 'bg-emerald-50 text-emerald-900' : 'bg-amber-50 text-amber-900')}>
          {contributionStatus(summary.status)} · 可归因 {summary.valid_days} / {summary.days} 日。{summary.status !== 'complete' && '缺口未被跳过；不得把部分数据当作完整区间。'}
        </p>
        <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">{[
          ['实际复利收益', percentText(summary.total_return)], ['累计贡献合计', contributionText(summary.contribution_sum)],
          ['对账误差', contributionText(summary.reconciliation_error)], ['模型 R²（有效配对日）', numberText(summary.model_r2)],
        ].map(([label, value]) => <div key={label} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-2 break-words text-lg font-bold tabular-nums text-slate-900">{value}</p></div>)}</div>
        <p className="mt-3 text-xs leading-6 text-slate-600">对账闭合不代表模型有效；残差也参与对账。样本外解释度与残差波动 {percentText(summary.residual_volatility)} 需要一起看。</p>
        <div className="mt-4 overflow-x-auto"><table className="w-full min-w-[340px] text-left text-sm" aria-label="累计因子贡献"><thead><tr><th scope="col" className="p-2">组成</th><th scope="col">累计收益贡献</th></tr></thead><tbody>{analysis.components.map((item, i) => <tr key={item.id} className="border-t border-slate-100"><th scope="row" className="p-2 font-medium">{item.label}</th><td>{contributionText(summary.contributions[i])}</td></tr>)}</tbody></table></div>
      </Card>
      <Card title="风格暴露随时间变化">
        <p className="mb-3 text-xs leading-6 text-slate-600">纵轴是回归系数，不是收益或实际持仓比例。估计窗口与原始因子收益可在逐日明细和 CSV 中核对；无法估计的日期断开。</p>
        <ReactECharts notMerge style={{ height: 320, width: '100%' }} option={{ ...chartBase,
          tooltip: { trigger: 'axis' }, xAxis: { type: 'category', boundaryGap: false, data: daily.map(row => row.date) },
          yAxis: { type: 'value', name: '暴露系数', scale: true },
          series: factorComponents.map((item, i) => ({ name: item.label, type: 'line', showSymbol: false, connectNulls: false, data: daily.map(row => row.exposures[i]) })),
        }} />
      </Card>
      <Card title="各组成的累计收益贡献">
        <ReactECharts notMerge style={{ height: 310, width: '100%' }} option={{ animation: false,
          tooltip: { trigger: 'axis', valueFormatter: pointTooltip }, grid: { left: 65, right: 20, top: 40, bottom: 85 },
          xAxis: { type: 'category', data: analysis.components.map(item => item.label), axisLabel: { rotate: 30, hideOverlap: true } },
          yAxis: { type: 'value', name: '百分点', axisLabel: { formatter: pointAxis } },
          series: [{ type: 'bar', data: summary.contributions }],
        }} />
        <p className="text-xs leading-6 text-slate-600">按每日基金期初财富串联贡献，而非直接求和或分别复利各因子。负贡献如实保留。</p>
      </Card>
      <Card title="累计贡献路径与基金实际收益">
        <ReactECharts notMerge style={{ height: 350, width: '100%' }} option={{ ...chartBase,
          series: [
            ...analysis.components.map((item, i) => ({ name: item.label, type: 'line', showSymbol: false, connectNulls: false, data: curve.map(row => row.contributions[i]) })),
            { name: '实际累计收益', type: 'line', showSymbol: false, connectNulls: false, data: curve.map(row => row.total_return), lineStyle: { width: 3 } },
            { name: '贡献合计', type: 'line', showSymbol: false, connectNulls: false, data: curve.map(row => row.contribution_sum), lineStyle: { type: 'dashed' } },
          ],
        }} />
      </Card>
      <details className="min-w-0 rounded-xl border border-slate-200 bg-white p-4">
        <summary className="cursor-pointer text-sm font-semibold">逐月对账 · 各月独立计算</summary>
        <p className="mt-3 text-xs leading-6 text-slate-600">展示完整评价期的每个月，与上方区间选择无关。首尾月份可能不完整；各月贡献不能直接相加代替全区间贡献。</p>
        <div className="mt-3 max-h-80 overflow-auto"><table className="w-full min-w-[700px] text-left text-xs" aria-label="月度贡献对账"><thead><tr><th scope="col" className="p-2">月份 / 实际覆盖</th><th scope="col">有效 / 应有日数</th><th scope="col">实际收益</th>{analysis.components.map(component => <th scope="col" key={component.id}>{component.label}</th>)}<th scope="col">贡献合计</th><th scope="col">对账误差</th><th scope="col">状态</th></tr></thead><tbody>{monthly.map(([month, item]) => <tr key={month} className="border-t border-slate-100"><th scope="row" className="p-2">{month}<span className="block font-normal">{item.start_date}—{item.end_date}</span></th><td>{item.valid_days} / {item.days}</td><td>{percentText(item.total_return)}</td>{item.contributions.map((value, i) => <td className="whitespace-nowrap p-2" key={analysis.components[i].id}>{contributionText(value)}</td>)}<td>{contributionText(item.contribution_sum)}</td><td>{contributionText(item.reconciliation_error)}</td><td>{contributionStatus(item.status)}</td></tr>)}</tbody></table></div>
      </details>
      <details className="min-w-0 rounded-xl border border-slate-200 bg-white p-4">
        <summary className="cursor-pointer text-sm font-semibold">逐日暴露、收益贡献与估计证据</summary>
        <div className="mt-3 flex flex-wrap items-center gap-3"><button className={secondaryClass} disabled={page === 0} onClick={() => setPage(page - 1)}>上一页</button><span className="text-xs">第 {page + 1} 页 · 共 {daily.length} 日</span><button className={secondaryClass} disabled={(page + 1) * 40 >= daily.length} onClick={() => setPage(page + 1)}>下一页</button></div>
        <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[1000px] text-left text-xs" aria-label="逐日收益贡献"><thead><tr><th scope="col" className="p-2">收益日期 / 估计窗口</th><th scope="col">暴露系数</th><th scope="col">实际收益</th>{analysis.components.map(item => <th scope="col" key={item.id}>{item.label}</th>)}<th scope="col">贡献合计</th><th scope="col">对账误差</th><th scope="col">状态</th></tr></thead><tbody>{rows.map(row => <tr key={row.date} className="border-t border-slate-100"><th scope="row" className="p-2">{row.date}<span className="block whitespace-nowrap font-normal">{row.fit_start || '—'}—{row.fit_end || '—'}</span><span className="block font-normal">有效拟合 {numberText(row.fit_observations, 0)} 日</span></th><td>{factorComponents.map((item, i) => <div className="whitespace-nowrap" key={item.id}>{item.label}：{numberText(row.exposures[i])}<span className="block text-slate-600">因子收益 {percentText(row.factor_returns[i])}</span></div>)}</td><td>{percentText(row.actual_return)}</td>{row.contributions.map((value, i) => <td className="whitespace-nowrap p-2" key={analysis.components[i].id}>{contributionText(value)}</td>)}<td className="whitespace-nowrap">{contributionText(row.contribution_sum)}</td><td className="whitespace-nowrap">{contributionText(row.reconciliation_error)}</td><td>{row.reason || '已对账'}</td></tr>)}</tbody></table></div>
      </details>
      <Card title="导出与研究边界">
        <div className="flex flex-wrap gap-3"><button className={secondaryClass} onClick={() => exporting(() => downloadBlob(`attribution-${product.code.replace(/[^a-zA-Z0-9_.-]/g, '_')}.csv`, new Blob([attributionCsv(run, product)], { type: 'text/csv;charset=utf-8' })))}>导出逐日贡献 CSV</button><button className={secondaryClass} onClick={() => exporting(() => downloadJson(`attribution-${run.id}.json`, run))}>导出完整运行 JSON</button></div>
        <p className="mt-3 text-xs leading-6 text-slate-600">CSV 包含所选产品全区间的暴露、因子收益、贡献和状态；数值为原始小数，0.01代表1个百分点，缺失留空。</p>
        {error && <p role="alert" className="mt-3 text-sm text-rose-700">{error}</p>}
        {analysis.notes.map(note => <p key={note} className="mt-2 text-xs leading-6 text-slate-600">{note}</p>)}
      </Card>
    </>}
  </div>
}
