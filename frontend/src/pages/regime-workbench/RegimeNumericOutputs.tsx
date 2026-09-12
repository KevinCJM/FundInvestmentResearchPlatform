import { useState } from 'react'
import ReactECharts from 'echarts-for-react'
import { regimeFeatureNumber, type RegimeResultData } from './regimeResultAdapter'

export default function RegimeNumericOutputs({ result }: { result: RegimeResultData }) {
  const entries = Object.entries(result.overview.numeric_channels || {})
  const [selected, setSelected] = useState('')
  if (!entries.length) return null
  const [id, channel] = entries.find(([key]) => key === selected) || entries[0]
  const scale = channel.display_format === 'percent' ? 100 : 1
  const values = result.points.map(point => { const value = regimeFeatureNumber(point, `channel:${id}`); return value === null ? null : value * scale })
  const unit = channel.display_format === 'percent' ? '%' : channel.unit || ''
  return <details className="rounded-xl border border-slate-200 bg-white p-4" aria-label="数值输出通道"><summary className="cursor-pointer text-sm font-semibold text-slate-800">数值输出 · {entries.length} 个通道</summary>
    <div className="mt-3 flex flex-wrap gap-2" role="tablist" aria-label="数值结果通道">{entries.map(([key, item]) => <button type="button" key={key} role="tab" aria-selected={key === id} onClick={() => setSelected(key)} className={`min-h-10 rounded-lg border px-3 text-xs ${key === id ? 'border-accent-400 bg-accent-50 text-accent-800' : 'border-slate-200'}`}>{item.label}</button>)}</div>
    <ReactECharts option={{ animation: false, tooltip: { trigger: 'axis' }, grid: { left: 60, right: 24, top: 40, bottom: 58 }, xAxis: { type: 'category', data: result.points.map(point => point.observation_date) }, yAxis: { type: 'value', scale: true, name: unit }, dataZoom: [{ type: 'inside' }, { type: 'slider', height: 18, bottom: 8 }], series: [{ name: channel.label, type: 'line', showSymbol: false, connectNulls: false, data: values }] }} style={{ height: 280 }} notMerge aria-label={`${channel.label}输出走势图`} />
    <p className="mb-2 text-xs text-slate-600">图表使用完整样本；下表显示最近 {Math.min(250, values.length)} 个观测。缺失值保留为空。</p>
    <div className="max-h-56 overflow-auto"><table className="w-full text-left text-xs" aria-label={`${channel.label}输出数据`}><thead><tr><th scope="col" className="p-2">日期</th><th scope="col" className="p-2">{channel.label}{unit ? `（${unit}）` : ''}</th></tr></thead><tbody>{result.points.slice(-250).map((point, i, points) => { const value = values[values.length - points.length + i]; return <tr key={point.observation_date} className="border-t border-slate-100"><td className="p-2">{point.observation_date}</td><td className="p-2">{value === null ? '—' : value.toFixed(channel.precision ?? 4)}</td></tr> })}</tbody></table></div>
  </details>
}
