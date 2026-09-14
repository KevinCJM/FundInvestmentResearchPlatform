import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { updateAllocationJourney, useAllocationDraft } from '../../app/allocationJourney'
import { useResearchDay } from '../../app/ResearchContext'
import { Button } from '../ui'
import { Field, Feedback, inputClass, today } from '../risk-models/ResearchUI'
import type { StrategicCatalog } from '../../services/strategicAllocation'
import { previewImplementationMap, confirmImplementationMap, getImplementationMap, type MappingDefinition, type MappingPreview, type MappingVersion, type UniverseVersion } from '../../services/strategicScope'
import { useScopeOperation } from './useScopeOperation'

export default function ImplementationMappingEditor({ universe, catalog, domainId, requestedMapping, onSaved }: {
  universe: UniverseVersion; catalog: StrategicCatalog; domainId: string; requestedMapping: string; onSaved: (value: MappingVersion) => void
}) {
  const clock = useResearchDay()
  const [draft, setDraft] = useAllocationDraft<MappingDefinition>(`implementation-map:${universe.id}:${domainId}`, () => ({
    name: `${universe.name}实施映射`, strategic_universe_id: universe.id, universe_snapshot_id: domainId,
    alloc_name: '', as_of: clock ?? today(), valid_until: '', assignments: [],
  }))
  const [preview, setPreview] = useState<MappingPreview | null>(null)
  const [saved, setSaved] = useState<MappingVersion | null>(null)
  const operation = useScopeOperation()
  useEffect(() => { operation.invalidate(); setPreview(null) }, [clock])
  useEffect(() => {
    if (!requestedMapping) return
    void operation.run(signal => getImplementationMap(requestedMapping, signal), result => {
      if (result.definition.strategic_universe_id !== universe.id) throw new Error('映射与所选战略范围不同。')
      setDraft(result.definition); setSaved(result); setPreview(result)
      updateAllocationJourney({ strategicUniverseId: universe.id, universeId: result.definition.universe_snapshot_id, implementationMappingId: result.id })
    })
  }, [requestedMapping, universe.id])
  const change = (patch: Partial<MappingDefinition>) => { operation.invalidate(); setPreview(null); setSaved(null); setDraft(current => ({ ...current, ...patch })) }
  const allocations = catalog.allocations.filter(a => a.universe_snapshot_id === draft.universe_snapshot_id)
  const allocation = allocations.find(a => a.alloc_name === draft.alloc_name)
  const issue = clock === undefined ? '知识截止日尚未确认，暂不能预览或确认。' : !draft.name.trim() || !draft.universe_snapshot_id || !allocation ? '先选择已锁定产品域及其真实大类方案。' : !draft.as_of || draft.as_of > (clock ?? today()) || !draft.valid_until || draft.valid_until <= draft.as_of ? '映射研究日须在知识截止之前，复核日须更晚。' : draft.assignments.some(a => a.rationale.trim().length < 3) ? '每个代理须说明匹配理由。' : ''
  return <section className="space-y-4 border-t border-slate-200 pt-5" aria-label="战略实施映射">
    <h2 className="text-lg font-semibold">匹配真实代理产品（可稍后完成）</h2>
    <p className="text-sm leading-6 text-slate-600">缺口保留在战略研究中，不填零收益、不重分配权重。完整真实映射及净值检查通过后才能进入TAA；现金产品的用途与可用性仍需人工核验。</p>
    <Feedback error={operation.error} />
    {saved && <p className="text-sm text-slate-700">只读映射：{saved.name}。<Button onClick={() => { operation.invalidate(); setSaved(null); setPreview(null) }}>复制映射为新研究</Button></p>}
    <fieldset disabled={Boolean(saved)} className="min-w-0 space-y-4">
      <div className="grid gap-4 sm:grid-cols-2"><Field label="映射名称"><input className={inputClass} value={draft.name} onChange={e => change({ name: e.target.value })} /></Field>
        <Field label="锁定的产品域"><select className={inputClass} value={draft.universe_snapshot_id} onChange={e => change({ universe_snapshot_id: e.target.value, alloc_name: '', assignments: [] })}><option value="">选择真实产品域</option>{Array.from(new Set([domainId, ...catalog.allocations.map(a => a.universe_snapshot_id)].filter((id): id is string => Boolean(id)))).map(id => <option key={id} value={id}>{id}</option>)}</select></Field>
        <Field label="实际代理大类方案"><select className={inputClass} value={draft.alloc_name} onChange={e => change({ alloc_name: e.target.value, assignments: [] })}><option value="">选择同产品域方案</option>{allocations.map(a => <option key={a.alloc_name} value={a.alloc_name}>{a.alloc_name}</option>)}</select></Field>
        <Field label="映射研究日"><input type="date" className={inputClass} value={draft.as_of} onChange={e => change({ as_of: e.target.value })} /></Field><Field label="映射复核日"><input type="date" className={inputClass} value={draft.valid_until} onChange={e => change({ valid_until: e.target.value })} /></Field></div>
      <div className="divide-y divide-slate-200">{universe.definition.assets.map(asset => {
        const row = draft.assignments.find(a => a.strategic_asset_id === asset.id)
        return <div key={asset.id} className="grid gap-3 py-4 sm:grid-cols-2"><Field label={`${asset.name}的真实代理`}><select className={inputClass} value={row?.proxy_asset_id ?? ''} onChange={e => change({ assignments: [...draft.assignments.filter(a => a.strategic_asset_id !== asset.id), ...(e.target.value ? [{ strategic_asset_id: asset.id, proxy_asset_id: e.target.value, rationale: row?.rationale ?? '' }] : [])] })}><option value="">保留产品缺口</option>{allocation?.assets.map(a => <option key={a.id} value={a.id}>{a.name}</option>)}</select></Field>{row ? <Field label={`${asset.name}匹配理由`}><input className={inputClass} value={row.rationale} onChange={e => change({ assignments: draft.assignments.map(a => a.strategic_asset_id === asset.id ? { ...a, rationale: e.target.value } : a) })} /></Field> : <p className="self-center text-sm text-amber-800">缺少产品，可继续前瞻研究。</p>}</div>
      })}</div>
    </fieldset>
    {issue && <p role="status" className="text-sm text-amber-800">{issue}</p>}
    {!allocations.length && <Link className="inline-flex min-h-10 items-center text-sm text-accent-800 underline" to="/pre-investment/product-pool">先选择产品池并构建真实大类</Link>}
    <div className="flex flex-wrap gap-3"><Button disabled={Boolean(issue) || operation.busy || Boolean(saved)} onClick={() => void operation.run(signal => previewImplementationMap(draft, signal), setPreview)}>检查映射覆盖</Button><Button tone="primary" disabled={Boolean(issue) || operation.busy || !preview || Boolean(saved)} onClick={() => preview && void operation.run(signal => confirmImplementationMap(draft, preview.preview_hash, signal), result => { setSaved(result); setPreview(result); updateAllocationJourney({ strategicUniverseId: universe.id, universeId: result.definition.universe_snapshot_id, implementationMappingId: result.id }); onSaved(result) })}>确认保存实施映射</Button></div>
    {!preview && <p className="text-xs text-slate-600">先检查覆盖与来源，再确认保存不可变映射。</p>}
    {operation.busy && <p role="status" className="text-sm text-slate-600">正在核验真实产品域与代理来源…</p>}
    {preview && <div aria-live="polite" className="space-y-2 text-sm"><p>{preview.implementation_status === 'complete' ? '已覆盖全部战略资产，当前适用性会在交接时复核。' : '存在实施缺口，暂不能进入TAA或产品应用。'}</p>{preview.coverage.map(row => <p key={row.strategic_asset_id}>{row.strategic_asset_id}：{row.proxy_asset_id ?? '缺少产品'}</p>)}</div>}
  </section>
}
