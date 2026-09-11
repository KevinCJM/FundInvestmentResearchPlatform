import { useEffect, useMemo, useState } from 'react'
import ReactECharts from 'echarts-for-react'
import type { ProductRegimeAnalysis, ProductRegimeSegment } from '../../services/productAnalysis'

const percent = (value: number | null | undefined) => value == null ? '—' : new Intl.NumberFormat('zh-CN', { style: 'percent', maximumFractionDigits: 2 }).format(value)
const cell = 'whitespace-nowrap px-4 py-3 text-right tabular-nums'
const pageSize = 12

export default function ProductScenarioPanel({ analysis, selectedStateId, selectedSegmentId, loading, onStateChange, onSegmentChange, onLocate }: {
  analysis: ProductRegimeAnalysis | null
  selectedStateId: string
  selectedSegmentId: string
  loading: boolean
  onStateChange: (id: string) => void
  onSegmentChange: (segment: ProductRegimeSegment) => void
  onLocate: (segment: ProductRegimeSegment) => void
}) {
  const [page, setPage] = useState(0)
  const segments = useMemo(() => (analysis?.segments ?? []).filter(item => !selectedStateId || item.stateId === selectedStateId), [analysis, selectedStateId])
  useEffect(() => { setPage(0) }, [analysis, selectedStateId])
  const displayed = segments.slice(page * pageSize, (page + 1) * pageSize)
  const option = useMemo(() => ({
    animation: false,
    grid: { left: 58, right: 24, top: 32, bottom: 58, containLabel: true },
    tooltip: { trigger: 'item', renderMode: 'richText', formatter: (item: { data: { segment: ProductRegimeSegment } }) => {
      const segment = item.data.segment
      return `${segment.startDate} 至 ${segment.endDate}\n${segment.observations} 个净值 / 价格观察值\n区间收益 ${percent(segment.cumulativeReturn)}`
    } },
    xAxis: { type: 'value', name: '区间长度（观察值）', nameLocation: 'middle', nameGap: 32, minInterval: 1, splitLine: { show: false } },
    yAxis: { type: 'value', name: '区间收益', axisLabel: { formatter: (value: number) => percent(value) }, splitLine: { lineStyle: { color: '#e2e8f0', type: 'dashed' } } },
    series: [{ name: '连续区间', type: 'scatter', symbolSize: 10,
      data: segments.filter(item => item.cumulativeReturn !== null).map(segment => ({ value: [segment.observations, segment.cumulativeReturn], segment, itemStyle: { color: segment.color, opacity: segment.id === selectedSegmentId ? 1 : 0.65, borderColor: segment.id === selectedSegmentId ? '#0f172a' : 'transparent', borderWidth: 2 } })),
    }],
  }), [segments, selectedSegmentId])
  if (loading) return <div role="status" className="rounded-2xl border border-slate-200 bg-white p-10 text-center text-sm text-slate-500">正在分析本产品在各情景中的表现…</div>
  if (!analysis) return <div className="rounded-2xl border border-dashed border-slate-300 bg-white px-6 py-16 text-center">
    <h2 className="text-lg font-semibold text-slate-900">选择一个已发布情景，开始比较</h2>
    <p className="mx-auto mt-2 max-w-md text-sm leading-6 text-slate-500">在上方选择情景方案，查看产品在不同市场状态中的收益与回撤，再深入某个连续区间。</p>
  </div>
  return <div className="space-y-5">
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="px-5 py-5"><h2 className="text-lg font-semibold text-slate-900">不同市场状态，产品表现如何</h2><p className="mt-1 text-sm text-slate-500">先比较状态，再查看连续区间。点击状态可联动收益统计与模拟样本。</p></div>
      <p className="px-5 pb-2 text-[11px] text-slate-400 sm:hidden">左右滑动查看完整指标</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm" aria-label="产品历史情景表现">
          <thead className="bg-slate-50 text-xs text-slate-500"><tr><th className="whitespace-nowrap px-5 py-3 text-left">市场状态</th><th className={cell}>有效收益 / 净值 / 价格点</th><th className={cell}>区间数 / 典型长度</th><th className={cell}>平均单期收益</th><th className={cell}>年化波动</th><th className={cell}>上涨占比</th><th className={cell}>区间收益中位数</th><th className={cell}>最差区间回撤</th></tr></thead>
          <tbody className="divide-y divide-slate-100">{analysis.states.map(state => <tr key={state.stateId} className={selectedStateId === state.stateId ? 'bg-sky-50/70' : ''}>
            <th scope="row" className="whitespace-nowrap px-5 py-3 text-left"><button type="button" onClick={() => onStateChange(state.stateId)} className="inline-flex items-center gap-2 whitespace-nowrap rounded-md py-1 text-left font-semibold text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500"><span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: state.color }} />{state.stateLabel}</button></th>
            <td className={cell}>{state.returnObservations} / {state.observations}</td><td className={cell}>{state.segmentCount} 段 / {state.medianSegmentObservations ?? '—'} 点</td><td className={cell}>{percent(state.meanDailyReturn)}</td><td className={cell}>{percent(state.annualizedVolatility)}</td><td className={cell}>{percent(state.winRate)}</td><td className={cell}>{percent(state.medianSegmentReturn)}<span className="mt-1 block text-[11px] text-slate-400">可计算 {state.eligibleSegmentCount}/{state.segmentCount} 段</span></td><td className={`${cell} text-rose-700`}>{percent(state.worstSegmentDrawdown)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <div className="border-t border-slate-100 px-5 py-3 text-xs leading-5 text-slate-500">单期收益按有效观察值汇总；收益与回撤逐段计算。跨情景边界收益不会归入任一状态，不拼接净值。典型长度为区间净值 / 价格观察值的中位数。</div>
    </section>
    <section className="rounded-2xl border border-slate-200 bg-white p-5">
      <h3 className="font-semibold text-slate-900">每段表现是否一致</h3><p className="mt-1 text-xs text-slate-500">每个点是一个连续区间。横轴显示长度，纵轴显示收益；点击可选中区间。</p>
      {segments.some(item => item.cumulativeReturn !== null) ? <ReactECharts option={option} style={{ height: 280 }} onEvents={{ click: (item: { data?: { segment?: ProductRegimeSegment } }) => { if (item.data?.segment) onSegmentChange(item.data.segment) } }} notMerge /> : <p className="py-12 text-center text-sm text-slate-500">没有具备有效区间收益的样本。</p>}
    </section>
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-5"><div><h3 className="font-semibold text-slate-900">连续区间明细</h3><p className="mt-1 text-xs text-slate-500">保留算法原始分段，短区间不会合并或隐藏；区间数不代表独立事件数。</p></div><span className="text-xs text-slate-500">{segments.length} 段</span></div>
      <div className="overflow-x-auto"><table className="w-full text-sm" aria-label="情景连续区间明细"><thead className="bg-slate-50 text-xs text-slate-500"><tr><th className="whitespace-nowrap px-5 py-3 text-left">日期与状态</th><th className={cell}>有效收益 / 净值 / 价格点</th><th className={cell}>区间收益</th><th className={cell}>段内最大回撤</th><th className="px-4 py-3 text-left">样本情况</th><th className={cell}>操作</th></tr></thead><tbody className="divide-y divide-slate-100">{displayed.map(segment => <tr key={segment.id} className={segment.id === selectedSegmentId ? 'bg-sky-50/70' : ''}>
        <th scope="row" className="whitespace-nowrap px-5 py-3 text-left font-medium"><button aria-label={`查看区间 ${segment.startDate} 至 ${segment.endDate}`} onClick={() => onSegmentChange(segment)} className="rounded text-slate-800 hover:text-sky-700 focus:outline-none focus:ring-2 focus:ring-sky-500">{segment.startDate} — {segment.endDate}<span className="mt-1 block text-left text-xs text-slate-500">{segment.stateLabel}{segment.windowClipped ? ' · 研究窗口内部分区间' : ''}</span></button></th>
        <td className={cell}>{segment.returnObservations} / {segment.observations}</td><td className={cell}>{percent(segment.cumulativeReturn)}</td><td className={`${cell} text-rose-700`}>{percent(segment.maxDrawdown)}</td><td className="min-w-[160px] px-4 py-3 text-xs leading-5 text-slate-500">{segment.reason || '可查看段内表现'}</td><td className={cell}><button type="button" onClick={() => onLocate(segment)} className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs text-slate-600 hover:border-sky-300 hover:text-sky-700">定位走势</button></td>
      </tr>)}</tbody></table></div>
      {segments.length === 0 && <p className="px-5 py-10 text-center text-sm text-slate-500">当前研究窗口与所选状态没有连续区间交集。</p>}
      <div className="flex items-center justify-between border-t border-slate-100 px-5 py-3 text-xs text-slate-500"><span>第 {page + 1} 页</span><div className="flex gap-2"><button type="button" disabled={page === 0} onClick={() => setPage(value => value - 1)} className="rounded-lg border border-slate-200 px-3 py-2 disabled:opacity-40">上一页</button><button type="button" disabled={(page + 1) * pageSize >= segments.length} onClick={() => setPage(value => value + 1)} className="rounded-lg border border-slate-200 px-3 py-2 disabled:opacity-40">下一页</button></div></div>
    </section>
  </div>
}
