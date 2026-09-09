import { useState } from 'react'
import type { RegimeGraphDefinition, RegimeGraphTemplate, RegimeNodeSchema } from '../../services/regimeGraph'

export default function RegimeDefinitionLibrary({ definitions, templates, schemas, selectedId, selectedTemplate, loading, busy, onDefinition, onTemplate, onLegacy }: {
  definitions: RegimeGraphDefinition[]; templates: RegimeGraphTemplate[]; schemas: RegimeNodeSchema[]; selectedId?: string; selectedTemplate: string
  loading: boolean; busy: boolean; onDefinition: (definition: RegimeGraphDefinition) => void; onTemplate: (id: string) => void; onLegacy: () => void
}) {
  const [query, setQuery] = useState('')
  const [source, setSource] = useState('all')
  const [mode, setMode] = useState('all')
  const matches = (name: string, description?: string) => `${name} ${description || ''}`.toLowerCase().includes(query.trim().toLowerCase())
  const builtins = source === 'custom' ? [] : templates.filter(item => matches(item.name, item.description) && (mode === 'all' || (item.default_mode || 'realtime') === mode))
  const definitionMode = (item: RegimeGraphDefinition) => item.graph.nodes.some(node => { const schema = schemas.find(entry => (entry.id || entry.type) === node.type); return schema?.causal === false || schema?.repaints === true || schema?.supports_realtime === false }) ? 'retrospective' : 'realtime'
  const custom = source === 'built_in' ? [] : definitions.filter(item => matches(item.name, item.description) && (mode === 'all' || definitionMode(item) === mode))
  const field = 'mt-1 block min-h-11 w-full rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-100'
  return <section aria-label="情景算法库" className="min-w-0 rounded-2xl border border-slate-200 bg-white shadow-sm">
    <div className="border-b border-slate-100 p-4">
      <div className="flex items-center justify-between"><h2 className="font-semibold text-slate-900">情景算法库</h2><span className="rounded-full bg-slate-100 px-2 py-0.5 text-xs text-slate-500">{builtins.length + custom.length}</span></div>
      <p className="mt-1 text-xs leading-5 text-slate-500">选择后在右侧编辑。内置算法保存为工作区副本。</p>
      <label className="mt-3 block text-xs font-semibold text-slate-600">搜索算法<input type="search" aria-label="搜索情景算法" placeholder="名称或说明" value={query} onChange={event => setQuery(event.target.value)} className={field} /></label>
      <div className="mt-3 grid grid-cols-2 gap-2">
        <label className="text-xs font-semibold text-slate-600">来源<select aria-label="情景算法来源" value={source} onChange={event => setSource(event.target.value)} className={field}><option value="all">全部来源</option><option value="built_in">内置算法</option><option value="custom">工作区算法</option></select></label>
        <label className="text-xs font-semibold text-slate-600">识别方式<select aria-label="算法库识别方式" value={mode} onChange={event => setMode(event.target.value)} className={field}><option value="all">全部方式</option><option value="realtime">实时识别</option><option value="retrospective">事后识别</option></select></label>
      </div>
    </div>
    <div className="max-h-[65vh] space-y-5 overflow-y-auto p-3">
      {loading ? <p role="status" className="p-3 text-sm text-slate-500">正在读取情景算法库…</p> : <>
        {source !== 'custom' && <section><h3 className="px-2 text-xs font-semibold text-slate-400">内置算法</h3><div className="mt-2 space-y-1">{builtins.map(item => <button key={item.id} type="button" disabled={busy} aria-label={`选择算法：${item.name}`} aria-pressed={!selectedId && selectedTemplate === item.id} onClick={() => onTemplate(item.id)} className={`w-full rounded-xl px-3 py-3 text-left transition disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-violet-300 ${!selectedId && selectedTemplate === item.id ? 'bg-violet-100 text-violet-900' : 'text-slate-700 hover:bg-slate-50'}`}><span className="block text-sm font-semibold">{item.name}</span><span className="mt-1 flex gap-1 text-[11px]"><span className="rounded-full bg-slate-100 px-2 py-0.5">内置</span><span className={`rounded-full px-2 py-0.5 ${item.default_mode === 'retrospective' ? 'bg-amber-50 text-amber-800' : 'bg-emerald-50 text-emerald-800'}`}>{item.default_mode === 'retrospective' ? '事后识别' : '实时识别'}</span></span><span className="mt-1 block line-clamp-2 text-xs leading-5 text-slate-500">{item.description}</span></button>)}</div></section>}
        {source !== 'built_in' && <section><h3 className="px-2 text-xs font-semibold text-slate-400">我的工作区算法</h3><div className="mt-2 space-y-1">{custom.map(item => <button key={item.id} type="button" disabled={busy} aria-label={`选择算法：${item.name}`} aria-pressed={selectedId === item.id} onClick={() => onDefinition(item)} className={`w-full rounded-xl px-3 py-3 text-left disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-violet-300 ${selectedId === item.id ? 'bg-violet-100 text-violet-900' : 'text-slate-700 hover:bg-slate-50'}`}><span className="flex items-center justify-between gap-2 text-sm font-semibold"><span>{item.name}</span><span className="shrink-0 text-xs text-violet-700">v{item.revision}</span></span><span className="mt-1 block line-clamp-2 text-xs leading-5 text-slate-500">{item.description || '枚举时序算法'}</span></button>)}</div>{!custom.length && <p className="px-2 py-3 text-xs text-slate-400">暂无匹配的工作区算法。</p>}</section>}
        {!builtins.length && !custom.length && <p className="rounded-xl border border-dashed border-slate-200 p-4 text-sm text-slate-500">当前筛选条件下没有算法。</p>}
      </>}
    </div>
    <div className="border-t border-slate-100 p-3"><button type="button" onClick={onLegacy} className="min-h-10 text-xs font-semibold text-slate-500 hover:text-violet-700">历史版本与旧版迁移目录</button></div>
  </section>
}
