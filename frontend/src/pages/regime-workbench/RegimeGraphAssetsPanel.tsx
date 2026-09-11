import { useEffect, useState } from 'react'
import {
  createRegimeGraphAsset,
  definitionForRequest,
  instantiateRegimeGraphAsset,
  listRegimeGraphAssets,
  updateRegimeGraphAsset,
  type RegimeGraphAsset,
  type RegimeGraphAssetKind,
  type RegimeGraphDefinition,
} from '../../services/regimeGraph'
import RegimeHelpTip from './RegimeHelpTip'

function message(reason: unknown, fallback: string) {
  return reason instanceof Error ? reason.message : fallback
}

function selectedGraph(definition: RegimeGraphDefinition, selectedNodeIds: string[]): RegimeGraphDefinition['graph'] {
  const selected = new Set(selectedNodeIds)
  const nodes = definition.graph.nodes.filter((node) => selected.has(node.id)).map((node) => ({
    ...node,
    inputs: Object.fromEntries(Object.entries(node.inputs || {}).filter(([, input]) => selected.has(input.node_id))),
  }))
  const edges = nodes.flatMap((node) => Object.entries(node.inputs).map(([port, source]) => ({ source: { ...source }, target: { node_id: node.id, port } })))
  return { nodes, edges, outputs: {} }
}

export default function RegimeGraphAssetsPanel({ definition, selectedNodeIds, valid, onLoadDefinition, onInsertGraph, onError, onNotice }: {
  definition: RegimeGraphDefinition
  selectedNodeIds: string[]
  valid: boolean
  onLoadDefinition: (definition: RegimeGraphDefinition) => void
  onInsertGraph: (graph: RegimeGraphDefinition['graph']) => void
  onError: (value: string) => void
  onNotice: (value: string) => void
}) {
  const [assets, setAssets] = useState<RegimeGraphAsset[]>([])
  const [kind, setKind] = useState<RegimeGraphAssetKind>('template')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [selectedAssetId, setSelectedAssetId] = useState('')
  const [busy, setBusy] = useState(false)

  const refresh = async () => setAssets(await listRegimeGraphAssets())
  useEffect(() => { const controller = new AbortController(); void listRegimeGraphAssets(undefined, controller.signal).then(setAssets).catch((reason) => { if (!controller.signal.aborted) onError(message(reason, '用户图谱资产加载失败。')) }); return () => controller.abort() }, [onError])
  const selectedAsset = assets.find((asset) => asset.id === selectedAssetId)
  const payload = () => kind === 'template'
    ? { name: name.trim(), description, definition: definitionForRequest(definition) }
    : { name: name.trim(), description, graph: selectedGraph(definition, selectedNodeIds) }

  const save = async (update: boolean) => {
    if (!name.trim()) { onError('请填写模板或子图名称。'); return }
    if (kind === 'template' && !valid) { onError('完整模板只能保存通过图谱检查的定义。'); return }
    if (kind === 'subgraph' && !selectedNodeIds.length) { onError('请先在画布中选择至少一个节点。'); return }
    setBusy(true); onError(''); onNotice('')
    try {
      const saved = update && selectedAsset
        ? await updateRegimeGraphAsset(selectedAsset.id, selectedAsset.revision, payload())
        : await createRegimeGraphAsset(kind, payload())
      await refresh()
      setSelectedAssetId(saved.id); setName(saved.name); setDescription(saved.description || '')
      onNotice(`${saved.kind === 'template' ? '用户模板' : '用户子图'}已保存 · r${saved.revision}。`)
    } catch (reason) { onError(message(reason, '用户图谱资产保存失败。')) } finally { setBusy(false) }
  }

  const instantiate = async (asset: RegimeGraphAsset) => {
    setBusy(true); onError(''); onNotice('')
    try {
      const result = await instantiateRegimeGraphAsset(asset.id, asset.revision)
      if (result.kind === 'template' && result.definition) onLoadDefinition(result.definition)
      else if (result.kind === 'subgraph' && result.graph) onInsertGraph(result.graph)
      else throw new Error('图谱资产实例化响应不完整。')
      onNotice(`${asset.name} · r${asset.revision} 已实例化。`)
    } catch (reason) { onError(message(reason, '图谱资产实例化失败。')) } finally { setBusy(false) }
  }

  return <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm" aria-label="用户模板与子图">
    <div><h3 className="text-sm font-bold text-slate-950">用户模板与可复用子图 <RegimeHelpTip label="模板与子图说明" text="完整模板保存整套计算图和状态语义；子图只保存选中节点及内部连线，适合复用一段特征或状态处理流程。两者都不绑定最终用途。" /></h3><p className="mt-1 text-[10px] text-slate-500">模板保存完整定义；子图只保存当前框选节点及其内部连线，均由服务端版本化。</p></div>
    <div className="mt-3 grid gap-2 sm:grid-cols-[130px_minmax(140px,1fr)_minmax(160px,1.5fr)]"><label className="text-[10px] font-bold text-slate-600">资产类型<RegimeHelpTip label="资产类型说明" text="完整模板可作为独立研究草稿打开；所选子图可插入任何兼容计算图。" /><select aria-label="图谱资产类型" value={kind} onChange={(event) => { setKind(event.target.value as RegimeGraphAssetKind); setSelectedAssetId('') }} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 bg-white px-2 text-[11px] font-normal"><option value="template">完整模板</option><option value="subgraph">所选子图</option></select></label><label className="text-[10px] font-bold text-slate-600">名称<RegimeHelpTip label="图谱资产名称说明" text="用于在自己的模板与子图库中识别这项资产。" /><input aria-label="图谱资产名称" value={name} onChange={(event) => setName(event.target.value)} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-[11px] font-normal" /></label><label className="text-[10px] font-bold text-slate-600">说明<RegimeHelpTip label="图谱资产说明" text="记录这套图谱适合解决的问题、主要假设或使用注意事项。" /><input aria-label="图谱资产说明" value={description} onChange={(event) => setDescription(event.target.value)} className="mt-1 min-h-9 w-full rounded-lg border border-slate-300 px-2 text-[11px] font-normal" /></label></div>
    <div className="mt-2 flex flex-wrap items-center gap-2"><button type="button" disabled={busy || !name.trim() || (kind === 'template' ? !valid : !selectedNodeIds.length)} onClick={() => void save(false)} className="min-h-9 rounded-lg bg-indigo-600 px-3 text-[11px] font-bold text-white disabled:opacity-40">另存为新{kind === 'template' ? '模板' : '子图'}</button><button type="button" disabled={busy || !selectedAsset || selectedAsset.kind !== kind || !name.trim()} onClick={() => void save(true)} className="min-h-9 rounded-lg border border-indigo-300 px-3 text-[11px] font-bold text-indigo-700 disabled:opacity-40">保存为新修订</button><span className="text-[10px] text-slate-500">画布当前选择 {selectedNodeIds.length} 个节点</span></div>
    <div className="mt-4 grid gap-2 md:grid-cols-2">{assets.map((asset) => <article key={asset.id} className={`rounded-xl border p-3 ${selectedAssetId === asset.id ? 'border-indigo-400 bg-indigo-50/40' : 'border-slate-200'}`}><div className="flex items-start justify-between gap-2"><button type="button" onClick={() => { setSelectedAssetId(asset.id); setKind(asset.kind); setName(asset.name); setDescription(asset.description || '') }} className="min-w-0 text-left"><span className="block truncate text-xs font-bold text-slate-900">{asset.name}</span><span className="mt-1 block text-[9px] font-bold text-slate-500">{asset.kind === 'template' ? '模板' : '子图'} · r{asset.revision} · {asset.registry_version || 'registry —'}</span></button><button type="button" disabled={busy} onClick={() => void instantiate(asset)} className="min-h-8 shrink-0 rounded-lg border border-slate-300 px-2 text-[10px] font-bold text-slate-700 disabled:opacity-40">{asset.kind === 'template' ? '作为草稿打开' : '插入画布'}</button></div><p className="mt-2 line-clamp-2 text-[10px] leading-4 text-slate-500">{asset.description || '无说明'}</p></article>)}{!assets.length ? <p className="rounded-xl bg-slate-50 p-4 text-xs text-slate-500">尚未保存用户模板或子图。</p> : null}</div>
  </section>
}
