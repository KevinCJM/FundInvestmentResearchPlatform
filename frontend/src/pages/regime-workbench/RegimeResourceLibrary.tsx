import { useState } from 'react'
import type { RegimeNodeSchema } from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'

export default function RegimeResourceLibrary({ schemas, onAdd, onOpenDataLab, busy = false }: {
  schemas: RegimeNodeSchema[]; onAdd: (schema: RegimeNodeSchema) => void; onOpenDataLab: () => void; busy?: boolean
}) {
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState('all')
  const resources = schemas.filter(schema => !schema.authoring_hidden && !['source.inline', 'source.indicator', 'source.relative'].includes(schema.id))
  const kind = (schema: RegimeNodeSchema) => schema.id === 'source.constant' ? 'constant' : schema.category
  const operatorCategories = [...new Map(resources.filter(schema => !['source', 'constant', 'indicator_calculation'].includes(kind(schema))).map(schema => [schema.category, schema.category_label || schema.category])).entries()]
  const normalized = query.trim().toLowerCase()
  const visible = resources.filter(schema => (category === 'all' || kind(schema) === category || category === `granularity:${schema.granularity?.kind}` || (category === 'operators' && !['source', 'constant', 'indicator_calculation'].includes(kind(schema)))) && (!normalized || `${schema.label} ${schema.description || ''} ${schema.tags?.join(' ') || ''}`.toLowerCase().includes(normalized)))
  return <aside className="min-w-0 rounded-2xl border border-slate-200 bg-white p-3 shadow-sm" aria-label="历史情景节点资源库">
    <div className="flex items-center justify-between gap-2"><div><h3 className="text-sm font-bold text-slate-950">节点资源库 <RegimeHelpTip label="节点资源库说明" text="基础算子只做一件事；组合模板会展开为真实步骤；耦合内核保留必要的递推或联合求解。选择模板时可一次撤销。" /></h3><p className="mt-1 text-[10px] text-slate-500">选择数据、计算步骤和市场状态规则</p></div><button type="button" onClick={onOpenDataLab} className="min-h-9 shrink-0 rounded-lg bg-indigo-50 px-2 text-[11px] font-bold text-indigo-700">数据实验室</button></div>
    <label className="mt-3 block text-xs font-semibold text-slate-700">节点类型<select aria-label="节点类型" value={category} onChange={event => setCategory(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 bg-white px-2 font-normal">
      <option value="all">全部类型</option><option value="source">数据源</option><option value="constant">常量</option><option value="indicator_calculation">指标计算</option><option value="granularity:primitive">基础算子</option><option value="granularity:composite">组合模板与算法</option><option value="granularity:coupled">耦合内核</option><option value="operators">全部算子</option>
      <optgroup label="按用途选择算子">{operatorCategories.map(([id, label]) => <option key={id} value={id}>{label}</option>)}</optgroup>
    </select></label>
    <label className="mt-3 block text-xs font-semibold text-slate-700">搜索节点<input type="search" aria-label="搜索计算节点" value={query} onChange={event => setQuery(event.target.value)} className="mt-1 min-h-10 w-full rounded-lg border border-slate-300 px-2 font-normal" placeholder="指数、均线、布林带、滤波…" /></label>
    {category === 'indicator_calculation' && <p className="mt-2 text-xs leading-5 text-slate-500">来自指标中心的内置和自定义版本。添加后连接所需数据；多输出指标可分别连接下游。</p>}
    <p className="mt-3 text-[11px] text-slate-500">{visible.length} 个可选节点</p>
    <div className="mt-2 max-h-[520px] space-y-2 overflow-y-auto pr-1">{visible.map(schema => {
      const available = schema.available !== false && (!schema.status || schema.status === 'available')
      const expand = schema.granularity?.expandable === true
      const action = expand ? '添加并展开' : '添加'
      return <article key={schema.id} className={`rounded-xl border p-3 ${available ? 'border-slate-200' : 'border-amber-200 bg-amber-50/60'}`}>
        <div className="flex items-start justify-between gap-2"><div className="min-w-0"><p className="break-words text-xs font-bold text-slate-900">{schema.label} <RegimeHelpTip label={`${schema.label}说明`} text={schema.description || schema.label} /></p><p className="mt-1 line-clamp-2 text-[10px] leading-4 text-slate-500">{schema.description}</p></div><button type="button" aria-label={`${action}${schema.label}${schema.indicator_reference ? ` 第${schema.indicator_reference.revision}版` : ''}`} disabled={!available || busy} onClick={() => onAdd(schema)} className="min-h-9 shrink-0 rounded-lg bg-indigo-600 px-2 text-xs font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">{available ? action : '不可用'}</button></div>
        <div className="mt-2 flex flex-wrap gap-1 text-[10px] text-slate-500"><span>{schema.granularity?.label || (schema.id === 'source.constant' ? '常量' : schema.category_label || '计算节点')}</span>{schema.indicator_reference && <><span>· 第 {schema.indicator_reference.revision} 版</span><span>· {schema.outputs.length} 个输出</span>{schema.indicator_reference.result_kind === 'scalar' && <span>· 滚动计算</span>}</>}</div>
        {expand && <p className="mt-2 text-xs leading-5 text-indigo-800">{schema.granularity?.steps?.join(' → ')}。展开后可逐步修改。</p>}
        {schema.granularity?.kind === 'composite' && !expand && <p className="mt-2 text-xs leading-5 text-amber-800">{schema.granularity.reason}</p>}
        {!available && <p role="note" className="mt-2 text-xs leading-5 text-amber-800">{schema.unavailable_reason || '当前节点暂不可用'}</p>}
      </article>
    })}{!visible.length && <p className="rounded-xl border border-dashed border-slate-300 p-4 text-center text-xs text-slate-500">没有匹配节点，请调整类型或搜索词。</p>}</div>
  </aside>
}
