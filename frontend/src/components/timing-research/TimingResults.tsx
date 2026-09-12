import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { timingApi, timingNumber, timingPercent, timingReason, type TimingMetrics, type TimingPeriod, type TimingProductResult, type TimingRun } from '../../services/timingResearch'
import { timingField } from './TimingRuleEditor'
import { timingPriceOption } from './timingPriceChart'
import TimingTrainingAudit from './TimingTrainingAudit'

const tableClass = 'w-full text-left text-xs [&_th]:whitespace-nowrap [&_th]:bg-slate-50 [&_th]:px-3 [&_th]:py-3 [&_th]:font-medium [&_th]:text-slate-600 [&_td]:whitespace-nowrap [&_td]:border-t [&_td]:border-slate-100 [&_td]:px-3 [&_td]:py-3 [&_td]:text-slate-700'
function Periods({ rows, label }: { rows: TimingPeriod[]; label: string }) {
  return <div className="max-h-[420px] overflow-auto rounded-lg border border-slate-200"><table aria-label={label} className={tableClass}><thead><tr><th scope="col">时期</th><th scope="col">策略收益</th><th scope="col">买入持有</th><th scope="col">最大回撤</th><th scope="col">已平仓交易</th><th scope="col">胜率</th><th scope="col">持仓比例</th></tr></thead><tbody>{rows.map(row => <tr key={row.period}><td>{row.period}</td><td>{timingPercent(row.total_return)}</td><td>{timingPercent(row.buy_hold_return)}</td><td>{timingPercent(row.max_drawdown)}</td><td>{timingNumber(row.trade_count, 0)}</td><td>{timingPercent(row.win_rate)}</td><td>{timingPercent(row.exposure)}</td></tr>)}</tbody></table>{!rows.length && <p className="p-4 text-sm text-slate-600">此区间没有可用统计。</p>}</div>
}
function Summary({ metrics }: { metrics?: TimingMetrics }) {
  const items = [ ['策略收益', timingPercent(metrics?.total_return)], ['买入持有', timingPercent(metrics?.buy_hold_return)], ['最大回撤', timingPercent(metrics?.max_drawdown)], ['已平仓交易', timingNumber(metrics?.trade_count, 0)], ['交易胜率', timingPercent(metrics?.win_rate)], ['持仓时间比例', timingPercent(metrics?.exposure)] ]
  return <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">{items.map(([label, value]) => <div key={label} className="rounded-xl border border-slate-200 bg-white p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 text-xl font-semibold tabular-nums text-slate-900">{value}</p></div>)}</div>
}

export default function TimingResults({ run }: { run: TimingRun }) {
  const [productId, setProductId] = useState(run.products[0]?.product_id || '')
  const [loaded, setLoaded] = useState<TimingProductResult | null>(null)
  const [loadError, setLoadError] = useState('')
  const [retry, setRetry] = useState(0)
  const [tab, setTab] = useState<'overview' | 'periods' | 'trades' | 'nodes'>('overview')
  const [sample, setSample] = useState<'out_of_sample' | 'in_sample' | 'all'>('out_of_sample')
  const [period, setPeriod] = useState<'yearly' | 'monthly' | 'walk_forward'>('yearly')
  const [channelId, setChannelId] = useState('')
  const [selectedDate, setSelectedDate] = useState('')
  const [page, setPage] = useState(0)
  const [curvePage, setCurvePage] = useState(0)
  const manifestProduct = run.products.find(product => product.product_id === productId) || run.products[0]
  const product = loaded?.product_id === productId ? loaded : manifestProduct
  const needsDetail = manifestProduct?.status === 'ok' && manifestProduct.detail_loaded === false
  useEffect(() => {
    setLoaded(null); setLoadError(''); setSelectedDate(''); setChannelId(''); setPage(0); setCurvePage(0)
    if (!needsDetail) return
    const controller = new AbortController()
    void timingApi.product(run.id, productId, controller.signal).then(value => { if (!controller.signal.aborted) setLoaded(value) }).catch(error => { if (!controller.signal.aborted) setLoadError(error instanceof Error ? error.message : '读取产品结果失败。') })
    return () => controller.abort()
  }, [run.id, productId, needsDetail, retry])
  const curve = product?.curve || []
  const trades = product?.trades || []
  const channel = product?.channels?.find(item => item.id === channelId) || product?.channels?.[0]
  const selectedPoint = curve.find(point => point.date === selectedDate)
  const diagnostics = sample === 'in_sample' ? undefined : product?.diagnostics?.[sample]
  const chartOption = useMemo(() => ({
    animation: false, aria: { enabled: true }, legend: { data: ['择时策略', '买入持有'], top: 0 },
    tooltip: { trigger: 'axis', confine: true, renderMode: 'richText', valueFormatter: (value: number) => timingNumber(value, 4) },
    grid: { left: 50, right: 16, top: 40, bottom: 70 }, xAxis: { type: 'category', data: curve.map(point => point.date), axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', scale: true, name: '净值' },
    dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 10, height: 22 }],
    series: [{ name: '择时策略', type: 'line', showSymbol: false, connectNulls: false, data: curve.map(point => point.nav), itemStyle: { color: '#4f46e5' } }, { name: '买入持有', type: 'line', showSymbol: false, connectNulls: false, data: curve.map(point => point.buy_hold_nav), itemStyle: { color: '#94a3b8' }, lineStyle: { type: 'dashed' } }],
  }), [curve])
  const priceOption = useMemo(() => timingPriceOption(curve, trades), [curve, trades])
  const nodeOption = useMemo(() => ({
    animation: false, aria: { enabled: true }, tooltip: { trigger: 'axis', confine: true, renderMode: 'richText' }, grid: { left: 55, right: 20, top: 20, bottom: 55 },
    xAxis: { type: 'category', data: curve.map(point => point.date), axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', scale: true },
    dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 5, height: 20 }], series: [{ name: channel?.label, type: 'line', showSymbol: false, step: channel?.type === 'condition' ? 'end' : false, connectNulls: false, data: channel?.values, itemStyle: { color: '#4f46e5' } }],
  }), [curve, channel])
  const chartEvents = useMemo(() => ({ click: (event: { name?: string }) => { if (event.name) setSelectedDate(event.name) } }), [])
  if (!product) return <p className="p-6 text-sm text-slate-600">此研究没有产品结果。</p>
  return <section aria-label="择时研究结果" className="min-w-0 space-y-4">
    <div className="flex flex-wrap items-end justify-between gap-3"><label className="min-w-[180px] text-xs text-slate-600">查看产品<select className={timingField} value={productId} onChange={event => setProductId(event.target.value)}>{run.products.map(item => <option key={item.product_id} value={item.product_id}>{item.product_id}{item.status === 'error' ? ' · 数据不足' : ''}</option>)}</select></label><span className="text-xs text-slate-600">运行于 {run.created_at?.replace('T', ' ').slice(0, 19)}</span></div>
    {product.status === 'error' ? <p role="alert" className="rounded-xl bg-amber-50 p-4 text-sm text-amber-800">{product.error || '此产品无法完成研究，请检查数据和区间。'}</p> : <>
      <div className="flex flex-wrap gap-2" aria-label="统计样本">{(['out_of_sample', 'in_sample', 'all'] as const).map(value => <button key={value} type="button" aria-pressed={sample === value} className={`min-h-9 rounded-lg px-3 text-sm ${sample === value ? 'bg-accent-50 font-semibold text-accent-700' : 'text-slate-600 hover:bg-slate-50'}`} onClick={() => setSample(value)}>{({ out_of_sample: '样本外', in_sample: '样本内', all: '全区间' })[value]}</button>)}</div>
      <Summary metrics={product.summary?.[sample]} />
      <TimingTrainingAudit audit={product.training} definition={run.definition_snapshot} />
      {product.summary?.[sample]?.trade_count === 0 && <p role="status" className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">此样本没有已完成的有效交易，不据此判定稳定。请同时检查信号数量、预热期和未平仓头寸。</p>}
      {diagnostics && <details className="rounded-xl border border-slate-200 p-3"><summary className="cursor-pointer text-sm font-medium text-slate-700">信号覆盖与月度稳定性</summary><div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">{[
        ['候选信号数', timingNumber(diagnostics.raw_signal_count, 0)],
        ['最多信号月份占比', timingPercent(diagnostics.top_month_signal_share)],
        ['正收益月份占比', timingPercent(diagnostics.positive_month_fraction)],
      ].map(([label, value]) => <div key={label} className="rounded-lg bg-slate-50 p-3"><p className="text-xs text-slate-600">{label}</p><p className="mt-1 text-lg font-semibold tabular-nums text-slate-900">{value}</p></div>)}</div><p className="mt-3 text-xs leading-5 text-slate-600">有信号月份 {timingNumber(diagnostics.active_month_count, 0)} / {timingNumber(diagnostics.total_month_count, 0)}；含不完整首尾月，非综合评分。候选信号可能重复，不等于成交次数。</p></details>}
      <div className="flex flex-wrap gap-2 border-b border-slate-200" role="tablist" aria-label="研究结果视图">{(['overview', 'periods', 'trades', 'nodes'] as const).map(value => <button key={value} id={`timing-tab-${value}`} type="button" role="tab" aria-selected={tab === value} aria-controls="timing-result-panel" onClick={() => setTab(value)} className={`min-h-11 border-b-2 px-3 text-sm ${tab === value ? 'border-accent-600 font-semibold text-accent-700' : 'border-transparent text-slate-600'}`}>{({ overview: '净值与信号', periods: '年度 / 月度', trades: '逐笔交易', nodes: '步骤预览' })[value]}</button>)}</div>
      <div id="timing-result-panel" role="tabpanel" aria-labelledby={`timing-tab-${tab}`} className="min-w-0 space-y-4">
        {loadError ? <div role="alert" className="rounded-lg bg-rose-50 p-4 text-sm text-rose-700">{loadError}<button type="button" className="ml-3 underline" onClick={() => setRetry(value => value + 1)}>重新加载</button></div> : needsDetail && !loaded ? <p role="status" className="p-5 text-sm text-slate-600">正在读取该产品的曲线与步骤…</p> : <>
          {tab === 'overview' && <><p className="text-xs leading-5 text-slate-600">下方图表展示完整研究区间。样本外从 {run.request_snapshot.holdout_start} 开始；上方统计按所选样本切换。</p><ReactECharts option={chartOption} notMerge style={{ height: 330, width: '100%' }} onEvents={chartEvents} />
            <div role="group" aria-label="买卖点图表" className="min-w-0">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-600"><h3 className="mr-auto text-sm font-medium text-slate-800">价格与买卖点</h3><span className="inline-flex items-center gap-1.5"><span className="rounded-lg border border-rose-200 bg-rose-50 px-1.5 py-0.5 font-bold text-rose-700">B</span>买入</span><span className="inline-flex items-center gap-1.5"><span className="rounded-lg border border-emerald-200 bg-emerald-50 px-1.5 py-0.5 font-bold text-emerald-700">S</span>卖出</span></div>
              <ReactECharts option={priceOption} notMerge style={{ height: 300, width: '100%' }} onEvents={chartEvents} />
              <p className="text-xs leading-5 text-slate-600">B 在成交点下方，S 在上方。拖动滑块放大区间；密集文字自动避让，成交点全部保留。悬停查看成交详情，点击标记定位日期。</p>
            </div>
            <label className="block max-w-xs text-xs text-slate-600">查看某日信号<input type="date" className={timingField} min={curve[0]?.date} max={curve[curve.length - 1]?.date} value={selectedDate} onChange={event => setSelectedDate(event.target.value)} /></label>{selectedDate && <div className="rounded-lg bg-slate-50 p-3 text-sm text-slate-700">{selectedPoint ? `${selectedPoint.date} · 价格 ${timingNumber(selectedPoint.close, 4)} · ${selectedPoint.position > 0 ? '持仓' : '空仓'} · ${timingReason(selectedPoint.reason)}` : '该日期没有研究数据，请选择交易日。'}</div>}
            <details><summary className="cursor-pointer py-2 text-sm text-accent-700">查看图表数据表</summary><div className="overflow-auto rounded-lg border border-slate-200"><table className={tableClass} aria-label="净值与信号数据"><thead><tr><th scope="col">日期</th><th scope="col">价格</th><th scope="col">策略净值</th><th scope="col">买入持有净值</th><th scope="col">持仓</th><th scope="col">操作原因</th></tr></thead><tbody>{curve.slice(curvePage * 100, curvePage * 100 + 100).map(point => <tr key={point.date}><td>{point.date}</td><td>{timingNumber(point.close, 4)}</td><td>{timingNumber(point.nav, 4)}</td><td>{timingNumber(point.buy_hold_nav, 4)}</td><td>{point.position > 0 ? '持仓' : '空仓'}</td><td>{timingReason(point.reason)}</td></tr>)}</tbody></table></div><Pager page={curvePage} total={curve.length} onChange={setCurvePage} /></details>
            {!!product.signal_quality?.length && <details><summary className="cursor-pointer py-2 text-sm text-accent-700">查看信号有效性与持有体验</summary><p className="mb-2 text-xs text-slate-600">以下是信号后的路径诊断，不代表可成交交易收益；末尾未成熟样本不纳入评价。</p><div className="overflow-auto"><table className={tableClass} aria-label="信号质量"><thead><tr><th scope="col">观察期</th><th scope="col">候选数 / 成熟数</th><th scope="col">上涨比例</th><th scope="col">收益中位数</th><th scope="col">平均最大不利波动</th><th scope="col">平均最大有利波动</th><th scope="col">正收益持有时间</th></tr></thead><tbody>{product.signal_quality.map(row => <tr key={row.horizon}><td>{row.horizon} 日</td><td>{row.signal_count} / {row.eligible_count}</td><td>{timingPercent(row.win_rate)}</td><td>{timingPercent(row.median_return)}</td><td>{timingPercent(row.mean_max_adverse)}</td><td>{timingPercent(row.mean_max_favorable)}</td><td>{timingPercent(row.positive_close_fraction)}</td></tr>)}</tbody></table></div></details>}</>}
          {tab === 'periods' && <><label className="block max-w-xs text-xs text-slate-600">统计频率<select className={timingField} value={period} onChange={event => setPeriod(event.target.value as typeof period)}><option value="yearly">年度</option><option value="monthly">月度</option><option value="walk_forward">时间分段稳定性</option></select></label><p className="text-xs text-slate-600">按完整研究区间逐期统计；时间分段检查固定规则的表现，不会自动调参。</p><Periods rows={product[period] || []} label={period === 'yearly' ? '年度表现' : period === 'monthly' ? '月度表现' : '时间分段表现'} /></>}
          {tab === 'trades' && <><p className="text-xs text-slate-600">收盘信号于下一交易日开盘执行，买入当天不卖出；净收益包含双边费用与滑点。未平仓头寸保留在净值中。</p><div className="overflow-auto rounded-lg border border-slate-200"><table className={tableClass} aria-label="逐笔交易"><thead><tr><th scope="col">信号日</th><th scope="col">买入日</th><th scope="col">卖出日</th><th scope="col">净收益</th><th scope="col">持有日数</th><th scope="col">退出原因</th><th scope="col">最大不利波动</th><th scope="col">最大有利波动</th></tr></thead><tbody>{trades.slice(page * 100, page * 100 + 100).map((trade, index) => <tr key={`${trade.entry_date}:${index}`}><td>{trade.signal_date}</td><td>{trade.entry_date}</td><td>{trade.exit_date}</td><td>{timingPercent(trade.net_return)}</td><td>{trade.holding_days}</td><td>{timingReason(trade.reason)}</td><td>{timingPercent(trade.max_adverse_excursion)}</td><td>{timingPercent(trade.max_favorable_excursion)}</td></tr>)}</tbody></table>{!trades.length && <p className="p-4 text-sm text-slate-600">此区间没有已完成交易。请结合信号、预热期和未平仓状态判断。</p>}</div><Pager page={page} total={trades.length} onChange={setPage} /></>}
          {tab === 'nodes' && <><label className="block text-xs text-slate-600">选择计算步骤<select className={timingField} value={channel?.id || ''} onChange={event => setChannelId(event.target.value)}>{product.channels?.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}</select></label><p className="text-xs text-slate-600">{channel?.type === 'condition' ? '条件：1 表示满足，0 表示不满足，空值或 −1 表示数据不足或未知。' : '展示此运行保存的原始步骤结果；缺失值保留为空。'}</p><ReactECharts option={nodeOption} notMerge style={{ height: 320, width: '100%' }} /><details><summary className="cursor-pointer py-2 text-sm text-accent-700">查看步骤数据表</summary><div className="max-h-80 overflow-auto"><table className={tableClass} aria-label="步骤数据"><thead><tr><th scope="col">日期</th><th scope="col">{channel?.label || '数值'}</th></tr></thead><tbody>{curve.slice(curvePage * 100, curvePage * 100 + 100).map((point, index) => <tr key={point.date}><td>{point.date}</td><td>{timingNumber(channel?.values[curvePage * 100 + index], 6)}</td></tr>)}</tbody></table></div><Pager page={curvePage} total={curve.length} onChange={setCurvePage} /></details></>}
        </>}
      </div>
    </>}
    {!!product.warnings?.length && <div className="space-y-1 rounded-lg bg-amber-50 p-3 text-xs leading-5 text-amber-800">{product.warnings.map((warning, index) => <p key={index}>{warning}</p>)}</div>}
    {!!run.restrictions?.length && <details className="rounded-lg bg-slate-50 p-3"><summary className="cursor-pointer text-xs font-medium text-slate-600">研究边界与数据口径</summary><div className="mt-2 space-y-1 text-xs leading-5 text-slate-600">{run.restrictions.map((restriction, index) => <p key={index}>{restriction}</p>)}</div></details>}
  </section>
}

function Pager({ page, total, onChange }: { page: number; total: number; onChange: (page: number) => void }) {
  if (total <= 100) return null
  return <div className="mt-3 flex items-center justify-between gap-2 text-xs text-slate-600"><button type="button" className="min-h-9 rounded-lg border px-3 disabled:opacity-40" disabled={page === 0} onClick={() => onChange(page - 1)}>上一页</button><span>{page + 1} / {Math.ceil(total / 100)} 页 · {total} 条</span><button type="button" className="min-h-9 rounded-lg border px-3 disabled:opacity-40" disabled={(page + 1) * 100 >= total} onClick={() => onChange(page + 1)}>下一页</button></div>
}
