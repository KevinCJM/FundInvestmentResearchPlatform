import { useEffect, useState, type ReactNode } from 'react'
import { listResearchSeries, listUploadedResearchSeries, type ResearchSeriesCatalogItem } from '../../services/researchSeries'
import type { RegimeGraphNode, RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeUploadSeriesEditor from './RegimeUploadSeriesEditor'
import RegimeIndicatorSourceFields from './RegimeIndicatorSourceFields'
import { regimeEnumLabel, regimeParameterLabel } from './regimeDisplay'

const SOURCE_KINDS: Record<string, string> = {
  'source.etf': 'etf', 'source.fund': 'fund', 'source.index': 'index', 'source.macro': 'macro', 'source.indicator': 'indicator', 'source.upload': 'upload',
}
const MARKET_BINDING_FIELDS = new Set(['ts_code', 'source_api', 'name', 'snapshot_id', 'snapshot_generation', 'source_file', 'file_checksum'])
const BINDING_FIELDS: Record<string, Set<string>> = {
  'source.etf': new Set([...MARKET_BINDING_FIELDS, 'adjustment_checksum']),
  'source.fund': MARKET_BINDING_FIELDS,
  'source.index': MARKET_BINDING_FIELDS,
  'source.macro': new Set([...MARKET_BINDING_FIELDS, 'dataset', 'series_id', 'code', 'date_field', 'available_at_field']),
  'source.indicator': new Set(['name', 'indicator_id', 'indicator_revision', 'data_fingerprint', 'indicator_data_snapshot']),
  'source.upload': new Set(['name', 'artifact_id', 'checksum', 'format', 'value_field', 'date_field', 'available_at_field', 'vintage_field', 'revision_field']),
  'source.inline': new Set(['name', 'rows', 'inline_rows', 'value_field']),
}
const PLACEHOLDERS: Record<string, string> = { etf: '输入 ETF 名称或代码，例如 510300', fund: '输入基金名称或代码，例如 华夏成长、000001', index: '输入名称或代码，例如 沪深300、000300.SH', macro: '输入宏观序列名称，例如 CPI、GDP', indicator: '输入指标名称，例如 年化收益率', upload: '输入已上传序列的名称' }
const PAGE_SIZE = 100

export const hasResearchSeriesPicker = (node: RegimeGraphNode) => node.type in BINDING_FIELDS
export const isEditableSourceParameter = (node: RegimeGraphNode, name: string) => !BINDING_FIELDS[node.type]?.has(name) && !(node.type === 'source.indicator' && ['product_kind', 'product_id', 'product_name', 'period'].includes(name))
export function researchSeriesFieldOptions(series?: ResearchSeriesCatalogItem) {
  return (series?.fields || []).flatMap(field => {
    const value = typeof field === 'string' ? field : field.name || field.id
    return value ? [{ value, label: typeof field === 'string' ? field : `${field.label || value}${field.available === false ? `（${field.unavailable_reason || '暂不可用'}）` : ''}`, disabled: typeof field !== 'string' && field.available === false }] : []
  })
}

function bindSource(parameters: RegimeGraphNode['parameters'], nodeType: string, binding: RegimeGraphNode['parameters']) {
  const next = { ...parameters }
  BINDING_FIELDS[nodeType]?.forEach(key => { delete next[key] })
  return Object.assign(next, binding)
}

function fieldBinding(series: ResearchSeriesCatalogItem | undefined, value: unknown) {
  const field = series?.fields?.find(item => typeof item !== 'string' && (item.name || item.id) === value)
  return typeof field === 'object' && field.available !== false ? field.binding_parameters : undefined
}

export function researchSourceParameterPatch(node: RegimeGraphNode, name: string, value: unknown, series?: ResearchSeriesCatalogItem): Partial<RegimeGraphNode> {
  let parameters = { ...node.parameters, [name]: value }
  const binding = node.type === 'source.etf' && name === 'field' ? fieldBinding(series, value) : undefined
  // A field edit must not silently replace an older frozen data generation.
  if (binding && (!node.parameters.snapshot_generation || node.parameters.snapshot_generation === binding.snapshot_generation)) {
    parameters = bindSource(parameters, node.type, binding)
  }
  return { parameters }
}

function bindingIdentity(node: RegimeGraphNode) {
  const p = node.parameters
  if (node.type === 'source.etf') return [p.ts_code || '']
  if (node.type === 'source.fund') return [p.ts_code || '', p.source_api || 'fund_nav']
  if (node.type === 'source.index') return [p.ts_code || '', p.source_api || 'index_daily']
  if (node.type === 'source.macro') return [p.dataset || '', p.ts_code || p.series_id || p.code || '']
  if (node.type === 'source.indicator') return [p.indicator_id || '', p.indicator_revision || 1]
  return [p.artifact_id || '', p.frequency || 'daily', p.availability_mode || 'point_in_time']
}

export const researchSeriesPickerKey = (node: RegimeGraphNode) => JSON.stringify([node.id, node.type, ...(node.type === 'source.upload' ? [node.parameters.artifact_id || ''] : node.type === 'source.inline' ? [] : bindingIdentity(node))])

function matchesNode(series: ResearchSeriesCatalogItem, node: RegimeGraphNode) {
  if (series.regime_node_type !== node.type) return false
  const candidate = { ...node, parameters: series.binding_parameters || {} }
  return JSON.stringify(bindingIdentity(candidate)) === JSON.stringify(bindingIdentity(node))
}

type Props = {
  node: RegimeGraphNode
  schema?: RegimeNodeSchema
  onPatchNode: (patch: Partial<RegimeGraphNode>) => void
  children: (series?: ResearchSeriesCatalogItem) => ReactNode
}
export default function RegimeResearchSeriesPicker(props: Props) {
  if (props.node.type === 'source.inline') return <p className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">此已有定义保留原始数据。若要更换数据，请添加上传时序节点。</p>
  if (props.node.type === 'source.upload') return <>{props.children()}<RegimeUploadSeriesEditor node={props.node} onPatchNode={props.onPatchNode} /><SeriesCatalogPicker {...props} children={() => null} /></>
  return <SeriesCatalogPicker {...props} />
}

function SeriesCatalogPicker({ node, schema, onPatchNode, children }: Props) {
  const p = node.parameters
  const initialQuery = String(node.type === 'source.upload' ? '' : p.ts_code || p.name || p.indicator_id || p.dataset || '')
  const [query, setQuery] = useState(initialQuery)
  const [offset, setOffset] = useState(0)
  const [retry, setRetry] = useState(0)
  const [result, setResult] = useState<{ query: string; items: ResearchSeriesCatalogItem[]; total: number } | null>(null)
  const [selected, setSelected] = useState<ResearchSeriesCatalogItem>()
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const normalizedQuery = query.trim()
  const kind = SOURCE_KINDS[node.type]
  const identity = researchSeriesPickerKey(node)
  const items = result?.query === normalizedQuery ? result.items : []
  const current = selected && matchesNode(selected, node) ? selected : items.find(item => matchesNode(item, node))

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError('')
    const timer = window.setTimeout(() => {
      void (kind === 'upload' ? listUploadedResearchSeries({ query: normalizedQuery, offset, limit: PAGE_SIZE }, controller.signal) : listResearchSeries({ kind, query: normalizedQuery, offset, limit: PAGE_SIZE }, controller.signal))
        .then(response => {
          if (controller.signal.aborted) return
          setResult(previous => ({
            query: normalizedQuery,
            items: offset && previous?.query === normalizedQuery
              ? [...new Map([...previous.items, ...response.items].map(item => [item.id, item])).values()]
              : response.items,
            total: response.total,
          }))
          const matched = response.items.find(item => matchesNode(item, node))
          if (matched) setSelected(matched)
        })
        .catch(() => { if (!controller.signal.aborted) setError('研究数据加载失败，请重试。') })
        .finally(() => { if (!controller.signal.aborted) setLoading(false) })
    }, 250)
    return () => { window.clearTimeout(timer); controller.abort() }
  }, [kind, identity, normalizedQuery, offset, retry])

  const options = items.filter(item => item.regime_node_type === node.type || (!item.regime_node_type && item.kind === kind))
  if (current && !options.some(item => item.id === current.id)) options.unshift(current)
  const selectSeries = (id: string) => {
    const series = options.find(item => item.id === id)
    if (!series?.binding_parameters || series.status !== 'available' || series.binding_supported === false || series.regime_node_type !== node.type) return
    let parameters = bindSource(p, node.type, series.binding_parameters)
    if (node.type === 'source.indicator') {
      if (series.product_kinds?.includes(String(p.product_kind))) {
        parameters.product_kind = p.product_kind; parameters.product_id = p.product_id; parameters.product_name = p.product_name
      } else { parameters.product_kind = ''; parameters.product_id = ''; delete parameters.product_name }
      if (series.periods?.includes(String(p.period))) parameters.period = p.period
    }
    const fields = researchSeriesFieldOptions(series)
    if (fields.some(field => !field.disabled && field.value === p.field)) {
      parameters.field = p.field
      const binding = node.type === 'source.etf' ? fieldBinding(series, p.field) : undefined
      if (binding) parameters = bindSource(parameters, node.type, binding)
    }
    const label = !node.label || node.label === p.name || node.label === schema?.label ? series.name : node.label
    setSelected(series)
    onPatchNode({ label, parameters })
  }
  const sourceProperties = schema?.parameter_schema?.properties || schema?.parameters || {}
  const boundName = current?.name || String(p.name || p.ts_code || p.indicator_id || p.artifact_id || p.series_id || '')

  return <>
    <section className="space-y-3 rounded-xl border border-indigo-200 bg-indigo-50/40 p-3" aria-label="研究数据选择">
      <h4 className="text-xs font-bold text-indigo-950">{node.type === 'source.indicator' ? '选择指标版本' : node.type === 'source.upload' ? '选择已上传数据' : '研究数据'}</h4>
      {boundName && <p className="break-words text-xs text-slate-700">当前绑定：{boundName}{p.ts_code && boundName !== p.ts_code ? ` · ${p.ts_code}` : ''}</p>}
      <label className="block text-xs font-semibold text-slate-700">搜索研究数据<input type="search" aria-label="搜索研究数据" value={query} placeholder={PLACEHOLDERS[kind]} onChange={event => { setQuery(event.target.value); setOffset(0) }} className="mt-1 min-h-10 w-full rounded-lg border border-indigo-200 bg-white px-2 font-normal" /></label>
      <label className="block text-xs font-semibold text-slate-700">数据序列<select aria-label="节点研究数据序列" value={current?.id || (boundName ? '__current__' : '')} onChange={event => selectSeries(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-indigo-200 bg-white px-2 font-normal">
        <option value="">{boundName ? '选择其他研究数据' : '请选择研究数据'}</option>
        {!current && boundName && <option value="__current__" disabled>{boundName}{p.ts_code && boundName !== p.ts_code ? ` · ${p.ts_code}` : ''}（当前绑定）</option>}
        {options.map(series => <option key={series.id} value={series.id} disabled={series.status !== 'available' || series.binding_supported === false || series.regime_node_type !== node.type}>{series.name}{series.code && node.type !== 'source.indicator' ? ` · ${series.code}` : ''}{node.type === 'source.upload' && series.binding_parameters?.artifact_id ? ` · ${series.coverage?.observations ?? '—'} 行 · ${String(series.binding_parameters.artifact_id).slice(-8)}` : ''}{series.indicator_version ? ` · 第 ${series.indicator_version.revision} 版` : ''}{series.binding_supported === false ? `（${series.binding_reason || '暂不支持'}）` : series.status === 'available' ? '' : '（尚未下载）'}</option>)}
      </select></label>
      {loading ? <p role="status" className="text-xs text-slate-500">正在搜索研究数据…</p> : error ? <p role="alert" className="text-xs text-rose-700">{error}<button type="button" onClick={() => setRetry(value => value + 1)} className="ml-2 underline">重试</button></p> : <p role="status" className="text-xs text-slate-500">{result?.total ? `找到 ${result.total} 条，已加载 ${items.length} 条。` : '没有匹配的研究数据，请换一个名称或代码。'}</p>}
      {!error && result?.query === normalizedQuery && items.length < result.total && <button type="button" disabled={loading} onClick={() => setOffset(items.length)} className="min-h-9 text-xs font-semibold text-indigo-700 disabled:opacity-40">加载更多结果</button>}
      <>
        <p className="text-xs leading-5 text-slate-500">{node.type === 'source.indicator' ? '选择后固定使用这个指标版本；更新版本需要重新选择。' : node.type === 'source.upload' ? '已上传数据是固定版本，更换文件不会覆盖此前的研究结果。' : '选择后自动绑定数据来源，无需重复配置代码或接口。'}</p>
        {Boolean(boundName) && <details><summary className="cursor-pointer text-xs text-slate-600">查看数据来源</summary><dl className="mt-2 space-y-2 text-xs">{[...BINDING_FIELDS[node.type]].filter(name => p[name] !== undefined && p[name] !== '').map(name => {
          const property = sourceProperties[name] || {}
          const enumIndex = property.enum?.indexOf(p[name]) ?? -1
          return <div key={name}><dt className="text-slate-500">{regimeParameterLabel(name, property)}</dt><dd className="break-all text-slate-700">{enumIndex >= 0 ? regimeEnumLabel(p[name], property, enumIndex) : String(p[name])}</dd></div>
        })}</dl></details>}
      </>
    </section>
    {node.type === 'source.indicator' && <RegimeIndicatorSourceFields key={identity} node={node} series={current} onPatchNode={onPatchNode} />}
    {node.type === 'source.etf' && <p className="text-xs leading-5 text-slate-500">可选择已下载的复权净值，或不复权、前复权、后复权市价。净值直接读取，不需要复权因子；切换字段自动匹配数据来源。复权数据仅用于事后分析。</p>}
    {node.type === 'source.fund' && <p className="text-xs leading-5 text-slate-500">基金行情以净值展示；实时分析按公告日期使用数据。复权净值仅用于事后分析，累计净值不等同于分红再投资收益。</p>}
    {node.type === 'source.etf'
      ? <fieldset disabled={!current} className="min-w-0 space-y-3">{children(current)}</fieldset>
      : children(current)}
  </>
}
