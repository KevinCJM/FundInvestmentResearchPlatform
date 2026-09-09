import { useEffect, useState } from 'react'
import type { RegimeGraphNode } from '../../services/regimeGraph'
import { searchResearchProducts, type ResearchSeriesCatalogItem } from '../../services/researchSeries'
const PERIOD_LABELS: Record<string, string> = { '1W': '近一周', '1M': '近一月', '3M': '近三月', '6M': '近六月', '1Y': '近一年', '2Y': '近两年', '3Y': '近三年', '5Y': '近五年', '10Y': '近十年', '20Y': '近二十年', '30Y': '近三十年', W1: '上周', W2: '上上周', M1: '上月', M2: '上上月', Y1: '去年', Y2: '前年', YTD: '今年以来', ALL: '成立以来', SI: '成立以来' }
const periodLabel = (period: string) => PERIOD_LABELS[period] || period

export default function RegimeIndicatorSourceFields({ node, series, onPatchNode }: {
  node: RegimeGraphNode; series?: ResearchSeriesCatalogItem; onPatchNode: (patch: Partial<RegimeGraphNode>) => void
}) {
  const p = node.parameters
  const kind = String(p.product_kind || '')
  const [query, setQuery] = useState(String(p.product_id || ''))
  const [page, setPage] = useState(1)
  const [retry, setRetry] = useState(0)
  const [result, setResult] = useState<{ key: string; items: Array<{ ts_code: string; name: string }>; total: number }>()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const key = `${kind}:${query.trim()}`
  const items = result?.key === key ? result.items : []
  useEffect(() => {
    if (!kind || !p.indicator_id || !['fund', 'etf'].includes(kind)) return
    const controller = new AbortController()
    setLoading(true); setError('')
    const timer = window.setTimeout(() => {
      void searchResearchProducts(kind, query.trim(), page, controller.signal).then(response => {
        if (!controller.signal.aborted) setResult(previous => ({ key, total: response.total, items: page > 1 && previous?.key === key ? [...new Map([...previous.items, ...response.items].map(item => [item.ts_code, item])).values()] : response.items }))
      }).catch(() => { if (!controller.signal.aborted) setError('计算对象加载失败，请重试。') }).finally(() => { if (!controller.signal.aborted) setLoading(false) })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [key, kind, query, page, retry, p.indicator_id])
  const patch = (values: Record<string, unknown>) => {
    const parameters = { ...p, ...values }
    delete parameters.data_fingerprint; delete parameters.indicator_data_snapshot
    onPatchNode({ parameters })
  }
  const kinds = series?.product_kinds || (['fund', 'etf'].includes(kind) ? [kind] : [])
  const periods = series?.periods || []
  const input = 'mt-1 min-h-10 w-full min-w-0 rounded-lg border border-slate-300 bg-white px-2 text-xs disabled:opacity-50'
  return <section aria-label="指标计算对象" className="space-y-3 rounded-xl border border-slate-200 p-3">
    <h4 className="text-sm font-semibold text-slate-900">计算对象与窗口</h4>
    <p className="text-xs leading-5 text-slate-600">指标是计算方法，还需指定对谁计算。例如“年度收益率”作用于一只基金，生成它每天的年度收益率。当前支持基金和 ETF。</p>
    <label className="block text-xs text-slate-600">对象类型<select aria-label="指标对象类型" value={kind} disabled={!p.indicator_id} className={input} onChange={event => { patch({ product_kind: event.target.value, product_id: '', product_name: '' }); setQuery(''); setPage(1) }}><option value="">请选择对象类型</option>{kind && !kinds.includes(kind) && <option value={kind} disabled>{kind}（该版本不支持）</option>}{kinds.map(value => <option key={value} value={value}>{value === 'etf' ? 'ETF' : '场外基金'}</option>)}</select></label>
    <label className="block text-xs text-slate-600">搜索计算对象<input aria-label="搜索计算对象" type="search" disabled={!kind} placeholder="输入基金或 ETF 的名称、代码" value={query} className={input} onChange={event => { setQuery(event.target.value); setPage(1) }} /></label>
    <label className="block text-xs text-slate-600">计算对象<select aria-label="指标计算对象" disabled={!kind} value={String(p.product_id || '')} className={input} onChange={event => patch({ product_id: event.target.value, product_name: items.find(item => item.ts_code === event.target.value)?.name || '' })}><option value="">请选择计算对象</option>{Boolean(p.product_id) && !items.some(item => item.ts_code === p.product_id) && <option value={String(p.product_id)}>{String(p.product_name || p.product_id)}（当前绑定）</option>}{items.map(item => <option key={item.ts_code} value={item.ts_code}>{item.name} · {item.ts_code}</option>)}</select></label>
    {loading ? <p role="status" className="text-xs text-slate-500">正在搜索计算对象…</p> : error ? <p role="alert" className="text-xs text-rose-700">{error}<button type="button" className="ml-2 underline" onClick={() => setRetry(value => value + 1)}>重试</button></p> : kind && result?.key === key && <p className="text-xs text-slate-500">{result.total ? `找到 ${result.total} 个对象。` : '没有匹配的对象，请换一个名称或代码。'}</p>}
    {result?.key === key && items.length < result.total && <button type="button" disabled={loading} onClick={() => setPage(value => value + 1)} className="min-h-9 text-xs text-indigo-700">加载更多计算对象</button>}
    <label className="block text-xs text-slate-600">计算窗口<select aria-label="指标计算窗口" value={String(p.period || '')} disabled={!p.indicator_id} className={input} onChange={event => patch({ period: event.target.value })}><option value="">请选择窗口</option>{Boolean(p.period) && !periods.includes(String(p.period)) && <option value={String(p.period)}>{periodLabel(String(p.period))}（当前配置）</option>}{periods.map(period => <option key={period} value={period}>{periodLabel(period)}</option>)}</select></label>
  </section>
}
